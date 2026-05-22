# -*- coding: utf-8 -*-
"""
Simulation-based rollout evaluation for LIBERO tasks.

Runs the policy in the LIBERO/robosuite environment and computes
task success rates via actual rollouts.
"""

import os
import collections
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn


LIBERO_EVAL_WARMUP_STEPS = 5


def _predict_action_ddim_core(model, batch, num_inference_steps=16, x_init=None):
    """DDIM-style accelerated inference (deterministic, eta=0).

    Differentiable through ``model`` so it can be used inside an SDFT loss for
    the student. Pass ``x_init`` to share the initial noise between two models
    (e.g. teacher and student) so their denoising chains are aligned.

    Notes:
        * Caller is responsible for wrapping in ``torch.no_grad()`` if running
          a frozen teacher.
        * Returns the predicted action chunk, shape (B, action_horizon, action_dim).
    """
    obs_cond = model.encode_obs(batch)
    B = obs_cond.shape[0]
    device = obs_cond.device
    T = model.num_diffusion_steps

    if x_init is None:
        x = torch.randn(B, model.action_horizon, model.action_dim, device=device)
    else:
        x = x_init

    step_ratio = T // num_inference_steps
    ddim_timesteps = (np.arange(0, num_inference_steps) * step_ratio).astype(np.int64)
    ddim_timesteps = np.flip(ddim_timesteps).copy()

    for i in range(len(ddim_timesteps)):
        t = int(ddim_timesteps[i])
        ts = torch.full((B,), t, device=device, dtype=torch.long)
        noise_pred = model.noise_pred_net(x, ts, obs_cond)

        alpha_cumprod_t = model.alphas_cumprod[t]

        if i + 1 < len(ddim_timesteps):
            alpha_cumprod_prev = model.alphas_cumprod[int(ddim_timesteps[i + 1])]
        else:
            alpha_cumprod_prev = torch.tensor(1.0, device=device)

        pred_x0 = (x - torch.sqrt(1.0 - alpha_cumprod_t) * noise_pred) / torch.sqrt(
            alpha_cumprod_t
        )
        pred_dir = torch.sqrt(1.0 - alpha_cumprod_prev) * noise_pred
        x = torch.sqrt(alpha_cumprod_prev) * pred_x0 + pred_dir

    return x


@torch.no_grad()
def predict_action_ddim(model, batch, num_inference_steps=16):
    """No-grad wrapper around ``_predict_action_ddim_core`` for evaluation."""
    return _predict_action_ddim_core(model, batch, num_inference_steps)


def _quat_to_axis_angle(quat_xyzw: np.ndarray) -> np.ndarray:
    import math

    w = float(np.clip(quat_xyzw[3], -1.0, 1.0))
    den = np.sqrt(1.0 - w * w)
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * math.acos(w)
    return (quat_xyzw[:3] * angle / den).astype(np.float32)


def process_env_obs(obs: dict, obs_keys: list = None) -> dict:
    processed = {}

    img = obs.get("agentview_image")
    if img is not None:
        img = img.astype(np.float32) / 255.0
        processed["agentview_image"] = np.transpose(img, (2, 0, 1))

    eye_img = obs.get("robot0_eye_in_hand_image")
    if eye_img is not None:
        eye_img = eye_img.astype(np.float32) / 255.0
        processed["eye_in_hand_image"] = np.transpose(eye_img, (2, 0, 1))

    robosuite_to_demo_key = {
        "robot0_eef_pos": "ee_pos",
        "robot0_eef_quat": "ee_ori",
        "robot0_gripper_qpos": "gripper_states",
    }

    for rs_key, demo_key in robosuite_to_demo_key.items():
        if rs_key in obs:
            if rs_key == "robot0_eef_quat":
                processed[demo_key] = _quat_to_axis_angle(obs[rs_key])
            else:
                processed[demo_key] = obs[rs_key].astype(np.float32)

    return processed


def obs_buffer_to_batch(
    obs_buffer: collections.deque,
    obs_horizon: int,
    use_eye_in_hand: bool,
    device: torch.device,
    low_dim_keys: list = None,
    task_emb: torch.Tensor = None,
) -> dict:
    if low_dim_keys is None:
        low_dim_keys = ["ee_pos", "ee_ori", "gripper_states"]

    buf_list = list(obs_buffer)
    while len(buf_list) < obs_horizon:
        buf_list.insert(0, buf_list[0])
    buf_list = buf_list[-obs_horizon:]

    batch = {}

    imgs = np.stack([o["agentview_image"] for o in buf_list])
    batch["obs_agentview_image"] = torch.from_numpy(imgs).unsqueeze(0).to(device)

    if use_eye_in_hand and "eye_in_hand_image" in buf_list[0]:
        imgs = np.stack([o["eye_in_hand_image"] for o in buf_list])
        batch["obs_eye_in_hand_image"] = torch.from_numpy(imgs).unsqueeze(0).to(device)

    for key in low_dim_keys:
        if key in buf_list[0]:
            data = np.stack([o[key] for o in buf_list])
            batch[f"obs_{key}"] = torch.from_numpy(data).unsqueeze(0).to(device)

    if task_emb is not None:
        batch["task_emb"] = task_emb.unsqueeze(0).to(device)

    return batch


def _sanitize_filename(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def _get_video_frame(obs: dict, camera_key: str):
    frame = obs.get(camera_key)
    if frame is None:
        image_keys = [
            key for key, value in obs.items()
            if isinstance(value, np.ndarray) and value.ndim == 3
        ]
        if not image_keys:
            return None
        frame = obs[image_keys[0]]

    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        if np.issubdtype(frame.dtype, np.floating) and frame.max() <= 1.0:
            frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
        else:
            frame = frame.clip(0, 255).astype(np.uint8)

    if frame.ndim == 3 and frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)

    frame = np.rot90(frame, 2)
    return frame


def _make_zero_action(env, fallback_action_dim: int = None) -> np.ndarray:
    """Build a zero action robustly across robosuite / LIBERO env versions."""
    action_spec = getattr(env, "action_spec", None)
    if action_spec is not None:
        try:
            if callable(action_spec):
                action_spec = action_spec()
            low, high = action_spec
            low = np.asarray(low, dtype=np.float32)
            return np.zeros_like(low, dtype=np.float32)
        except Exception:
            pass

    action_dim = getattr(env, "action_dim", None)
    if action_dim is not None:
        return np.zeros(int(action_dim), dtype=np.float32)

    action_space = getattr(env, "action_space", None)
    if action_space is not None:
        shape = getattr(action_space, "shape", None)
        if shape is not None:
            return np.zeros(shape, dtype=np.float32)

    if fallback_action_dim is not None:
        return np.zeros(int(fallback_action_dim), dtype=np.float32)

    raise AttributeError(
        "Could not infer action dimension for warm-up steps. "
        "Environment exposes neither action_spec, action_dim, nor action_space.shape, "
        "and no fallback_action_dim was provided."
    )


def evaluate_policy_on_task(
    model: nn.Module,
    benchmark,
    task_idx: int,
    num_episodes: int = 20,
    max_steps: int = 600,
    action_execution_horizon: int = 8,
    action_mean: np.ndarray = None,
    action_std: np.ndarray = None,
    obs_horizon: int = 2,
    image_size: tuple = (128, 128),
    use_eye_in_hand: bool = True,
    low_dim_keys: list = None,
    device: torch.device = None,
    use_ddim: bool = True,
    ddim_steps: int = 16,
    seed: int = 42,
    save_video: bool = False,
    video_dir: str = None,
    video_fps: int = 20,
    video_camera_key: str = "agentview_image",
    video_episodes_to_save: int = 3,
    task_emb: torch.Tensor = None,
) -> tuple:
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    if "NUMBA_CACHE_DIR" not in os.environ:
        cache_dir = "/tmp/numba_cache"
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["NUMBA_CACHE_DIR"] = cache_dir

    from libero.libero.envs import OffScreenRenderEnv

    if device is None:
        device = next(model.parameters()).device

    bddl_file = benchmark.get_task_bddl_file_path(task_idx)
    init_states = benchmark.get_task_init_states(task_idx)
    task_name = benchmark.get_task_names()[task_idx]

    if save_video:
        video_root = Path(video_dir) if video_dir else Path("videos")
        video_root.mkdir(parents=True, exist_ok=True)
    else:
        video_root = None

    camera_names = ["agentview"]
    if use_eye_in_hand:
        camera_names.append("robot0_eye_in_hand")
    if save_video and video_camera_key.endswith("_image"):
        requested_camera = video_camera_key[: -len("_image")]
        if requested_camera and requested_camera not in camera_names:
            camera_names.append(requested_camera)

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_names=camera_names,
        camera_heights=image_size[0],
        camera_widths=image_size[1],
    )
    print(
        f"    Environment created for task {task_idx}: {task_name}",
        flush=True,
    )
    env.seed(seed)

    model.eval()
    successes = []
    failed_videos_saved = 0

    for ep in range(num_episodes):
        print(f"    Episode {ep+1:02d}/{num_episodes}: reset", flush=True)
        video_writer = None
        video_path = None
        # Record every episode; keep only failed ones (up to video_episodes_to_save)
        should_record = save_video and failed_videos_saved < max(video_episodes_to_save, 0)
        if should_record:
            import imageio.v2 as imageio

            video_name = (
                f"task_{task_idx:02d}_{_sanitize_filename(task_name)}_ep_{ep:03d}.mp4"
            )
            video_path = video_root / video_name
            video_writer = imageio.get_writer(str(video_path), fps=video_fps)

        env.reset()
        init_state = init_states[ep % len(init_states)]
        print(f"    Episode {ep+1:02d}/{num_episodes}: set_init_state", flush=True)
        obs = env.set_init_state(init_state)
        zero_action = _make_zero_action(
            env,
            fallback_action_dim=getattr(model, "action_dim", None),
        )

        print(
            f"    Episode {ep+1:02d}/{num_episodes}: warmup {LIBERO_EVAL_WARMUP_STEPS} step(s)",
            flush=True,
        )
        warmup_done = False
        for _ in range(LIBERO_EVAL_WARMUP_STEPS):
            obs, _, warmup_done, _ = env.step(zero_action)
            if video_writer is not None:
                frame = _get_video_frame(obs, video_camera_key)
                if frame is not None:
                    video_writer.append_data(frame)
            if warmup_done:
                break

        if warmup_done:
            successes.append(False)
            print(
                f"    Episode {ep+1:02d}/{num_episodes}: FAIL (env terminated during warmup)",
                flush=True,
            )
            if video_writer is not None:
                video_writer.close()
                if video_path is not None:
                    print(f"    Saved video: {video_path}")
            continue

        print(f"    Episode {ep+1:02d}/{num_episodes}: policy rollout", flush=True)
        obs_buffer = collections.deque(maxlen=obs_horizon)
        obs_buffer.append(process_env_obs(obs))

        episode_success = False
        steps_taken = 0

        while steps_taken < max_steps and not episode_success:
            batch = obs_buffer_to_batch(
                obs_buffer=obs_buffer,
                obs_horizon=obs_horizon,
                use_eye_in_hand=use_eye_in_hand,
                device=device,
                low_dim_keys=low_dim_keys,
                task_emb=task_emb,
            )

            if use_ddim:
                pred_actions = predict_action_ddim(
                    model, batch, num_inference_steps=ddim_steps
                )
            else:
                pred_actions = model.sample_action(batch)

            pred_actions = pred_actions[0].detach().cpu().numpy()

            if action_mean is not None and action_std is not None:
                pred_actions = pred_actions * action_std + action_mean

            exec_horizon = min(
                action_execution_horizon, pred_actions.shape[0], max_steps - steps_taken
            )

            for t in range(exec_horizon):
                action = pred_actions[t]
                try:
                    obs, _, done, info = env.step(action)
                except ValueError:
                    # LIBERO masks robosuite's done=True (horizon timeout) as done=False,
                    # so the env can appear live while internally terminated.
                    steps_taken = max_steps  # force outer while to exit
                    break
                steps_taken += 1

                if video_writer is not None:
                    frame = _get_video_frame(obs, video_camera_key)
                    if frame is not None:
                        video_writer.append_data(frame)

                processed = process_env_obs(obs)
                obs_buffer.append(processed)

                success_flag = False
                if isinstance(info, dict):
                    success_flag = bool(
                        info.get("success", False)
                        or info.get("task_success", False)
                        or info.get("is_success", False)
                    )
                if done or success_flag:
                    episode_success = success_flag or done
                    break

        successes.append(bool(episode_success))
        print(
            f"    Episode {ep+1:02d}/{num_episodes}: "
            f"{'SUCCESS' if episode_success else 'FAIL'} "
            f"({steps_taken} steps)"
        )

        if video_writer is not None:
            video_writer.close()
            if episode_success:
                # Delete video for successful episodes — we only want failures
                if video_path is not None and video_path.exists():
                    video_path.unlink()
            else:
                failed_videos_saved += 1
                if video_path is not None:
                    print(f"    Saved failure video: {video_path}")

    env.close()

    success_rate = float(np.mean(successes)) if successes else 0.0
    return success_rate, successes


def evaluate_checkpoint_on_all_tasks(
    model: nn.Module,
    benchmark,
    task_indices: list,
    num_episodes: int = 20,
    max_steps: int = 600,
    action_execution_horizon: int = 8,
    action_mean: np.ndarray = None,
    action_std: np.ndarray = None,
    obs_horizon: int = 2,
    image_size: tuple = (128, 128),
    use_eye_in_hand: bool = True,
    low_dim_keys: list = None,
    device: torch.device = None,
    use_ddim: bool = True,
    ddim_steps: int = 16,
    seed: int = 42,
    save_video: bool = False,
    video_dir: str = None,
    video_fps: int = 20,
    video_camera_key: str = "agentview_image",
    video_episodes_to_save: int = 3,
    task_embeddings: dict = None,
) -> dict:
    task_names = benchmark.get_task_names()
    results = {}
    for task_idx in task_indices:
        task_name = task_names[task_idx]
        print(f"\n  Evaluating task {task_idx}: {task_name}")
        task_emb = task_embeddings.get(task_name) if task_embeddings else None
        sr, _ = evaluate_policy_on_task(
            model=model,
            benchmark=benchmark,
            task_idx=task_idx,
            num_episodes=num_episodes,
            max_steps=max_steps,
            action_execution_horizon=action_execution_horizon,
            action_mean=action_mean,
            action_std=action_std,
            obs_horizon=obs_horizon,
            image_size=image_size,
            use_eye_in_hand=use_eye_in_hand,
            low_dim_keys=low_dim_keys,
            device=device,
            use_ddim=use_ddim,
            ddim_steps=ddim_steps,
            seed=seed,
            save_video=save_video,
            video_dir=video_dir,
            video_fps=video_fps,
            video_camera_key=video_camera_key,
            video_episodes_to_save=video_episodes_to_save,
            task_emb=task_emb,
        )
        results[task_idx] = sr
        print(f"    Success Rate: {sr:.4f}")

    return results
