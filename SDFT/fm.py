# -*- coding: utf-8 -*-
"""
FM-SDFT (Flow Matching Self-Distillation Fine-Tuning) helpers.

Migrated from SDFT/MSE/sdft.py (DDPM-based) to Flow Matching.

Key differences from DDPM-SDFT:
  - Teacher produces x1 via Euler integration (not DDIM)
  - SDFT loss is computed in velocity space: |v_student - v_teacher|^2
  - No alphas_cumprod, no noise schedule needed
  - t ~ Uniform[0,1], x_t = (1-t)*x0 + t*x1 (linear interpolation)
"""

from __future__ import annotations

import os
import sys
import collections
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Ensure repo root is on sys.path so scripts.* imports resolve.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.evaluation.rollout_evaluator import (  # noqa: E402
    process_env_obs,
    obs_buffer_to_batch,
)


# ------------------------------------------------------------------
# Obs-batch utilities  (unchanged from sdft.py)
# ------------------------------------------------------------------

def clone_obs_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in batch.items()}


def stack_obs_batches(batch_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not batch_list:
        return {}
    keys = batch_list[0].keys()
    return {k: torch.cat([b[k] for b in batch_list], dim=0) for k in keys}


def subsample_obs_batches(
    batch_list: List[Dict[str, torch.Tensor]],
    max_states: int,
    rng: np.random.Generator,
) -> List[Dict[str, torch.Tensor]]:
    if len(batch_list) <= max_states:
        return batch_list
    idx = rng.choice(len(batch_list), size=max_states, replace=False)
    idx = np.sort(idx)
    return [batch_list[i] for i in idx]


# ------------------------------------------------------------------
# FM inference helpers
# ------------------------------------------------------------------

def euler_integration(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    num_steps: int,
    x_init: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Euler integration of the learned vector field.

    Differentiable through model when called outside torch.no_grad().
    Pass x_init to align teacher and student chains.

    Returns shape (B, action_horizon, action_dim).
    """
    obs_cond = model.encode_obs(batch)
    B = obs_cond.shape[0]
    device = obs_cond.device

    if x_init is None:
        x = torch.randn(B, model.action_horizon, model.action_dim, device=device)
    else:
        x = x_init

    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = torch.full((B,), i * dt, device=device)
        t_scaled = t * 1000
        v = model.vector_field_net(x, t_scaled, obs_cond)
        x = x + v * dt

    return x


# ------------------------------------------------------------------
# On-policy rollout collection (FM variant)
# ------------------------------------------------------------------

def collect_onpolicy_observations(
    model: torch.nn.Module,
    benchmark,
    task_idx: int,
    *,
    num_episodes: int,
    max_steps: int,
    action_execution_horizon: int,
    action_mean: Optional[np.ndarray],
    action_std: Optional[np.ndarray],
    obs_horizon: int,
    image_size: Tuple[int, int],
    use_eye_in_hand: bool,
    low_dim_keys: Optional[List[str]],
    device: torch.device,
    num_flow_steps: int = 10,
    max_states: int = 200,
    seed: int = 42,
    log_debug: bool = False,
) -> Tuple[List[Dict[str, torch.Tensor]], int]:
    """Run FlowPolicy in a task env; collect on-policy obs batches.

    Uses Euler integration (not DDIM) for inference.
    """
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

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=image_size[0],
        camera_widths=image_size[1],
    )
    env.seed(int(seed))

    model.eval()
    collected: List[Dict[str, torch.Tensor]] = []

    with torch.no_grad():
        for ep in range(num_episodes):
            env.reset()
            init_state = init_states[ep % len(init_states)]
            obs = env.set_init_state(init_state)

            obs_buffer = collections.deque(maxlen=obs_horizon)
            obs_buffer.append(process_env_obs(obs))

            success = False
            steps = 0

            while steps < max_steps and not success:
                if len(collected) >= max_states:
                    break

                batch = obs_buffer_to_batch(
                    obs_buffer, obs_horizon, use_eye_in_hand, device,
                    low_dim_keys=low_dim_keys,
                )
                collected.append(clone_obs_batch(batch))

                # FM inference: Euler integration
                actions = euler_integration(model, batch, num_flow_steps)
                actions = actions[0].cpu().numpy()

                if action_mean is not None and action_std is not None:
                    actions = actions * action_std + action_mean

                n_exec = min(action_execution_horizon, len(actions))
                for a_idx in range(n_exec):
                    obs, _, done, _ = env.step(actions[a_idx])
                    steps += 1
                    obs_buffer.append(process_env_obs(obs))
                    if env.check_success():
                        success = True
                        break
                    if steps >= max_steps:
                        break

                if len(collected) >= max_states:
                    break

            if len(collected) >= max_states:
                break

    env.close()

    rng = np.random.default_rng(int(seed) + 17)
    n_before = len(collected)
    collected = subsample_obs_batches(collected, max_states, rng)
    n_after = len(collected)

    if log_debug:
        print(
            f"    [fm_sdft] collected on-policy states: {n_before} "
            f"(used {n_after}, cap={max_states})",
            flush=True,
        )

    return collected, n_after


# ------------------------------------------------------------------
# FM-SDFT loss
# ------------------------------------------------------------------

def compute_fm_sdft_loss(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    sdft_batch: Dict[str, torch.Tensor],
    num_steps: int,
) -> torch.Tensor:
    """FM-SDFT distillation loss.

    D_FM = E_{t,x0} [ |v_student(x_t,t) - v_teacher(x_t,t)|^2 ]

    where x_t is built from teacher's predicted x1 (Euler) and fresh noise x0.
    Both student and teacher evaluate the velocity at the *same* (x_t, t) point.
    """
    device = next(student.parameters()).device

    # Teacher produces x1 via Euler integration (frozen)
    with torch.no_grad():
        x_init = torch.randn(
            sdft_batch[list(sdft_batch.keys())[0]].shape[0],
            student.action_horizon,
            student.action_dim,
            device=device,
        )
        x1 = euler_integration(teacher, sdft_batch, num_steps, x_init=x_init)
        obs_teacher = teacher.encode_obs(sdft_batch)

    obs_student = student.encode_obs(sdft_batch)
    B = x1.shape[0]

    # FM forward: linear interpolation
    x0 = torch.randn_like(x1)
    t = torch.rand(B, device=device)
    t_b = t.view(-1, 1, 1)
    x_t = (1 - t_b) * x0 + t_b * x1

    t_scaled = t * 1000

    with torch.no_grad():
        v_teacher = teacher.vector_field_net(x_t, t_scaled, obs_teacher)
    v_student = student.vector_field_net(x_t, t_scaled, obs_student)

    assert v_student.shape == v_teacher.shape, (
        f"Shape mismatch: v_student={v_student.shape}, v_teacher={v_teacher.shape}"
    )

    return F.mse_loss(v_student, v_teacher)
