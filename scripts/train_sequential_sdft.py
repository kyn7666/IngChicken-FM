# -*- coding: utf-8 -*-
"""
Sequential Continual Learning with Experience Replay (ER) + FM-SDFT.

Two anti-forgetting signals:
  * ER  — replay buffer from previous-task demos
  * FM-SDFT — velocity-space distillation from a frozen teacher

    L_total = L_FM(merged_batch) + sdft_weight * L_FM_SDFT(on_policy_obs)

Usage (from repo root):
  python -m scripts.train_sequential_sdft \
      --config configs/cl_object_sdft.yaml [--skip-eval] [--pretrain-ckpt PATH]
"""

import copy
import os
import math
import json
import time
import random
import argparse
from pathlib import Path
from datetime import datetime

import yaml
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from model.flow_policy import FlowPolicy, EMAModel
from scripts.datasets import (
    SingleTaskDataset,
    create_single_task_dataloader,
    compute_global_action_stats,
)
from scripts.utils_er import ReplayMemory, cycle, merge_batches, split_batch_size
from scripts.evaluation import (
    evaluate_checkpoint_on_all_tasks,
    compute_nbt,
    compute_average_sr,
    compute_average_sr_per_stage,
    save_results_json,
    save_results_csv,
    plot_performance_matrix,
    plot_forgetting_summary,
)

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from SDFT.fm import (  # noqa: E402
    collect_onpolicy_observations,
    compute_fm_sdft_loss,
    stack_obs_batches,
)

from libero.libero.benchmark import get_benchmark


# ---------------------------------------------------------------------------
# Helpers  (same pattern as train_sequential.py)
# ---------------------------------------------------------------------------

def _checkpoint_step(path: Path) -> int:
    digits = "".join(ch if ch.isdigit() else " " for ch in path.stem).split()
    return int(digits[-1]) if digits else -1


def _prepare_run_dirs(cfg: dict) -> tuple:
    log_cfg = cfg["logging"]
    exp_name = log_cfg.get("exp_name") or cfg.get("exp_name")
    if exp_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path("output") / f"{exp_name}_{timestamp}"
        ckpt_dir = run_dir / "checkpoints"
        results_dir = run_dir / "results"
        cfg["run_name"] = run_dir.name
        cfg["run_dir"] = str(run_dir.resolve())
        cfg["logging"]["checkpoint_dir"] = str(ckpt_dir.resolve())
        cfg["logging"]["results_dir"] = str(results_dir.resolve())
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "config_resolved.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    else:
        run_dir = None
        ckpt_dir = Path(log_cfg["checkpoint_dir"])
        results_dir = Path(log_cfg["results_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, ckpt_dir, results_dir


def _init_tensorboard_writer(cfg: dict, results_dir: Path):
    log_cfg = cfg["logging"]
    if not log_cfg.get("use_tensorboard", False):
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        return None
    tb_dir = log_cfg.get("tensorboard_dir")
    if tb_dir:
        tb_dir = Path(tb_dir)
    elif cfg.get("run_dir"):
        tb_dir = Path(cfg["run_dir"]) / "tensorboard"
    else:
        tb_dir = results_dir / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)
    cfg["logging"]["tensorboard_dir"] = str(tb_dir.resolve())
    return SummaryWriter(log_dir=str(tb_dir))


def _resolve_weights_path(weights_dir: str):
    if not weights_dir:
        return None
    path = Path(weights_dir).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"weights_dir does not exist: {path}")
    if path.is_file():
        return path
    for candidate in [
        path / "checkpoints" / "best_ema.pt", path / "checkpoints" / "best.pt",
        path / "best_ema.pt", path / "best.pt",
    ]:
        if candidate.exists():
            return candidate
    ema_candidates = sorted(path.rglob("*_ema.pt"), key=_checkpoint_step)
    if ema_candidates:
        return ema_candidates[-1]
    ckpt_candidates = sorted(path.rglob("*.pt"), key=_checkpoint_step)
    if ckpt_candidates:
        return ckpt_candidates[-1]
    raise FileNotFoundError(f"No checkpoint found under: {path}")


def _load_initial_weights(model: FlowPolicy, cfg: dict, device: torch.device):
    weights_path = _resolve_weights_path(cfg.get("weights_dir"))
    if weights_path is None:
        print("Training mode: scratch")
        return None
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model_state = model.state_dict()
    compatible = {k: v for k, v in state_dict.items()
                  if k in model_state and model_state[k].shape == v.shape}
    if not compatible:
        raise ValueError(f"No compatible parameters in checkpoint: {weights_path}")
    model_state.update(compatible)
    model.load_state_dict(model_state)
    print(f"Training mode: finetune from {weights_path} ({len(compatible)} tensors loaded)")
    return weights_path


def _save_checkpoint(path: Path, payload: dict, model: FlowPolicy, ema: EMAModel, use_ema: bool):
    checkpoint = dict(payload)
    checkpoint["checkpoint_kind"] = "ema" if use_ema else "raw"
    checkpoint["model_state_dict"] = ema.state_dict() if use_ema else model.state_dict()
    checkpoint["ema_state_dict"] = ema.state_dict()
    torch.save(checkpoint, path)


def verify_task_names(benchmark, benchmark_name: str):
    task_names = benchmark.get_task_names()
    n = benchmark.get_num_tasks()
    print("\n" + "=" * 70)
    print(f"Benchmark: {benchmark_name}  |  Tasks: {n}  |  Order: {benchmark.task_order_index}")
    print("=" * 70)
    for i, name in enumerate(task_names):
        print(f"  Task {i:2d}: {name}")
    print("=" * 70 + "\n")
    return task_names


def verify_data_files(data_root: str, benchmark):
    print("Verifying data files...")
    missing = []
    for i in range(benchmark.get_num_tasks()):
        demo_rel = benchmark.get_task_demonstration(i)
        demo_path = os.path.join(data_root, demo_rel)
        exists = os.path.exists(demo_path)
        print(f"  Task {i}: {demo_rel} [{'OK' if exists else 'MISSING'}]")
        if not exists:
            missing.append(demo_path)
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} demo file(s):\n"
                                + "\n".join(f"  - {p}" for p in missing))
    print("All data files verified.\n")


def _sample_sdft_minibatch(collected: list, batch_size: int, rng: random.Random,
                            device: torch.device) -> dict:
    chosen = rng.choices(collected, k=min(batch_size, len(collected)))
    stacked = stack_obs_batches(chosen)
    return {k: v.to(device) for k, v in stacked.items()}


def _collect_sdft_observations(
    *, model: FlowPolicy, benchmark, task_k: int, sdft_cfg: dict,
    eval_cfg: dict, data_cfg: dict, action_mean, action_std,
    low_dim_keys: list, device: torch.device, seed: int,
) -> list:
    if task_k <= 0:
        return []
    print(f"  [FM-SDFT] Collecting on-policy obs from {task_k} previous task(s)...")
    t0 = time.time()
    max_states_total = sdft_cfg.get("max_states", 200)
    states_per_task = max(1, max_states_total // task_k)
    all_collected = []
    for prev_task in range(task_k):
        prev_collected, _ = collect_onpolicy_observations(
            model=model, benchmark=benchmark, task_idx=prev_task,
            num_episodes=sdft_cfg.get("num_episodes", 5),
            max_steps=sdft_cfg.get("max_steps_per_episode", 600),
            action_execution_horizon=eval_cfg.get("action_execution_horizon", 8),
            action_mean=action_mean if data_cfg.get("normalize_action", True) else None,
            action_std=action_std if data_cfg.get("normalize_action", True) else None,
            obs_horizon=data_cfg["obs_horizon"],
            image_size=tuple(data_cfg.get("image_size", [128, 128])),
            use_eye_in_hand=data_cfg.get("use_eye_in_hand", True),
            low_dim_keys=low_dim_keys, device=device,
            num_flow_steps=eval_cfg.get("num_flow_steps", model.num_flow_steps),
            max_states=states_per_task,
            seed=seed + task_k * 100 + prev_task,
            log_debug=True,
        )
        all_collected.extend(prev_collected)
    print(f"  [FM-SDFT] Collected {len(all_collected)} states in {time.time() - t0:.1f}s")
    return all_collected


# ---------------------------------------------------------------------------
# Per-task training
# ---------------------------------------------------------------------------

def train_on_task(
    model: FlowPolicy,
    task_idx: int,
    task_name: str,
    demo_path: str,
    cfg: dict,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    device: torch.device,
    tb_writer=None,
    tb_global_step_offset: int = 0,
    replay_memory: ReplayMemory = None,
    teacher: FlowPolicy = None,
    sdft_collected: list = None,
    use_wandb: bool = False,
    task_embeddings: dict = None,
) -> tuple:
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    cl_cfg = cfg["continual_learning"]
    log_cfg = cfg["logging"]
    replay_cfg = cfg.get("replay", {})
    sdft_cfg = cfg.get("sdft", {})

    sdft_enabled = (
        sdft_cfg.get("enabled", False)
        and teacher is not None
        and sdft_collected is not None
        and len(sdft_collected) > 0
    )
    sdft_weight = float(sdft_cfg.get("weight", 1.0))
    sdft_num_steps = int(sdft_cfg.get("num_flow_steps", model.num_flow_steps))
    sdft_batch_size = int(sdft_cfg.get("batch_size", 16))

    epochs = cl_cfg["epochs_per_task"]
    current_batch_size = data_cfg["batch_size"]
    replay_iterator = None
    configured_steps_per_epoch = train_cfg.get("steps_per_epoch")
    use_configured_steps = configured_steps_per_epoch is not None
    steps_per_epoch = int(configured_steps_per_epoch) if use_configured_steps else None
    if use_configured_steps and steps_per_epoch <= 0:
        raise ValueError("training.steps_per_epoch must be a positive integer")

    if replay_memory is not None and replay_memory.has_samples():
        mix_ratio = float(replay_cfg.get("mix_ratio", 0.5))
        current_batch_size, replay_batch_size = split_batch_size(data_cfg["batch_size"], mix_ratio)
        replay_loader = replay_memory.build_loader(
            cfg=cfg, action_mean=action_mean, action_std=action_std, batch_size=replay_batch_size,
            task_embeddings=task_embeddings,
        )
        if replay_loader is not None:
            replay_iterator = cycle(replay_loader)
            print(f"  Replay: {replay_memory.num_samples()} samples / "
                  f"{replay_memory.num_tasks()} task(s)  "
                  f"[cur={current_batch_size}, rep={replay_batch_size}]")

    task_emb = task_embeddings.get(task_name) if task_embeddings else None
    print(f"  Loading dataset: {demo_path}")
    loader, dataset = create_single_task_dataloader(
        hdf5_path=demo_path, batch_size=current_batch_size,
        num_workers=data_cfg["num_workers"], obs_horizon=data_cfg["obs_horizon"],
        action_horizon=data_cfg["action_horizon"],
        action_mean=action_mean if data_cfg.get("normalize_action", True) else None,
        action_std=action_std if data_cfg.get("normalize_action", True) else None,
        obs_keys=data_cfg["obs_keys"], use_eye_in_hand=data_cfg.get("use_eye_in_hand", True),
        image_size=tuple(data_cfg.get("image_size", [128, 128])),
        samples_per_epoch=steps_per_epoch * current_batch_size if use_configured_steps else None,
        task_emb=task_emb,
    )
    if not use_configured_steps:
        steps_per_epoch = len(loader)
    print(f"  Dataset: {len(dataset)} samples, {len(loader)} batches/epoch")
    if use_configured_steps:
        print(f"  Steps/epoch: {steps_per_epoch} (random replacement sampling)")
    if sdft_enabled:
        print(f"  FM-SDFT: ON  (weight={sdft_weight}, steps={sdft_num_steps}, "
              f"sdft_bs={sdft_batch_size}, n_states={len(sdft_collected)})")
    else:
        print("  FM-SDFT: OFF")

    lr = train_cfg.get("_effective_lr", train_cfg["learning_rate"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=train_cfg.get("weight_decay", 1e-6)
    )

    total_steps = epochs * steps_per_epoch
    warmup_steps = train_cfg.get("lr_warmup_steps", 500)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    ema = EMAModel(model, decay=train_cfg.get("ema_decay", 0.995))
    use_amp = train_cfg.get("mixed_precision", True)
    scaler = GradScaler(enabled=use_amp)
    sdft_rng = random.Random(cfg.get("seed", 42) + task_idx)

    epoch_losses = []
    global_step = 0

    for epoch in range(epochs):
        model.train()
        if sdft_enabled:
            teacher.eval()
        batch_losses = []

        if use_configured_steps:
            batch_iter = cycle(loader)
            pbar = tqdm(range(steps_per_epoch), desc=f"  Task {task_idx} | Epoch {epoch+1}/{epochs}", leave=False)
        else:
            batch_iter = iter(loader)
            pbar = tqdm(batch_iter, desc=f"  Task {task_idx} | Epoch {epoch+1}/{epochs}", leave=False)

        for _ in pbar:
            batch = next(batch_iter) if use_configured_steps else _
            batch = {k: v.to(device) for k, v in batch.items()}
            if replay_iterator is not None:
                replay_batch = next(replay_iterator)
                replay_batch = {k: v.to(device) for k, v in replay_batch.items()}
                batch = merge_batches(batch, replay_batch)

            with autocast(enabled=use_amp):
                task_loss = model.compute_loss(batch)

                if sdft_enabled:
                    sdft_batch = _sample_sdft_minibatch(
                        sdft_collected, sdft_batch_size, sdft_rng, device
                    )
                    sdft_loss_v = compute_fm_sdft_loss(model, teacher, sdft_batch, sdft_num_steps)
                    sdft_loss = sdft_weight * sdft_loss_v
                    loss = task_loss + sdft_loss
                else:
                    sdft_loss_v = None
                    sdft_loss = None
                    loss = task_loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if train_cfg.get("gradient_clip", 0) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["gradient_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)

            loss_val = loss.item()
            batch_losses.append(loss_val)
            global_step += 1

            postfix = {"loss": f"{loss_val:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"}
            if sdft_enabled:
                postfix["sdft"] = f"{sdft_loss_v.item():.4f}"
            pbar.set_postfix(**postfix)

            if use_wandb and global_step % log_cfg.get("log_interval", 50) == 0:
                import wandb as _wandb
                log_dict = {
                    "train/loss": loss_val,
                    "train/lr": scheduler.get_last_lr()[0],
                }
                if sdft_enabled:
                    log_dict["train/sdft_loss"] = sdft_loss.item()
                    log_dict["train/sdft_loss_v"] = sdft_loss_v.item()
                _wandb.log(log_dict, step=tb_global_step_offset + global_step)
            if tb_writer and global_step % log_cfg.get("log_interval", 50) == 0:
                tb_step = tb_global_step_offset + global_step
                tb_writer.add_scalar("train/loss", loss_val, tb_step)
                tb_writer.add_scalar("train/lr", scheduler.get_last_lr()[0], tb_step)
                if sdft_enabled:
                    tb_writer.add_scalar("train/sdft_loss", sdft_loss.item(), tb_step)
                    tb_writer.add_scalar("train/sdft_loss_v", sdft_loss_v.item(), tb_step)

        avg_loss = np.mean(batch_losses)
        epoch_losses.append(avg_loss)
        print(f"  Task {task_idx} | Epoch {epoch+1:3d}/{epochs} | "
              f"loss={avg_loss:.4f} | lr={scheduler.get_last_lr()[0]:.2e}")

    return model, ema, epoch_losses, global_step, dataset


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg, skip_eval=False, pretrain_ckpt=None):
    device = torch.device(cfg.get("device", "cuda"))
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    benchmark_cfg = cfg["benchmark"]
    data_cfg = cfg["data"]
    eval_cfg = cfg["evaluation"]
    replay_cfg = cfg.get("replay", {})
    sdft_cfg = cfg.get("sdft", {})
    run_dir, ckpt_dir, results_dir = _prepare_run_dirs(cfg)
    tb_writer = _init_tensorboard_writer(cfg, results_dir)

    wandb_cfg = cfg.get("wandb", {})
    use_wandb = wandb_cfg.get("enabled", False)
    wandb_run_id = None
    if use_wandb:
        try:
            import wandb
            date_str = datetime.now().strftime("%m%d")
            run_name = f"{wandb_cfg['name']}_{date_str}"
            use_fixed_steps = cfg.get("training", {}).get("steps_per_epoch") is not None
            wandb_project = wandb_cfg["project"] + ("-step" if use_fixed_steps else "")
            wandb.init(
                entity=wandb_cfg["entity"],
                project=wandb_project,
                group=wandb_cfg.get("group"),
                name=run_name,
                tags=wandb_cfg.get("tags", []),
                config=cfg,
                resume=wandb_cfg.get("resume", "allow"),
            )
            wandb_run_id = wandb.run.id
            print(f"wandb run: {run_name}  (id={wandb_run_id})")
        except ImportError:
            print("wandb not available, disabling wandb logging")
            use_wandb = False

    replay_memory = None
    if replay_cfg.get("enabled", False):
        buffer_size = int(replay_cfg.get("buffer_size", 0))
        if buffer_size > 0:
            replay_memory = ReplayMemory(capacity=buffer_size, seed=seed)
            print(f"Replay enabled: buffer_size={buffer_size}, "
                  f"mix_ratio={replay_cfg.get('mix_ratio', 0.5):.2f}\n")
        else:
            print("Replay requested but buffer_size <= 0, disabling.\n")
    else:
        print("Replay disabled.\n")

    if sdft_cfg.get("enabled", False):
        print(f"FM-SDFT enabled: weight={sdft_cfg.get('weight', 1.0)}, "
              f"max_states={sdft_cfg.get('max_states', 200)}, "
              f"num_episodes={sdft_cfg.get('num_episodes', 5)}\n")
    else:
        print("FM-SDFT disabled.\n")

    benchmark = get_benchmark(benchmark_cfg["name"])(
        task_order_index=benchmark_cfg.get("task_order_index", 0)
    )
    n_tasks = benchmark.get_num_tasks()
    task_names = verify_task_names(benchmark, benchmark_cfg["name"])

    data_root = benchmark_cfg["data_root"]
    verify_data_files(data_root, benchmark)

    print("Computing global action normalization stats...")
    action_mean, action_std = compute_global_action_stats(data_root, benchmark)

    clip_emb_path = data_cfg.get("clip_emb_path")
    task_embeddings = None
    if clip_emb_path:
        task_embeddings = torch.load(clip_emb_path, map_location="cpu", weights_only=False)
        print(f"Loaded CLIP embeddings: {len(task_embeddings)} tasks from {clip_emb_path}")

    print("\nBuilding Flow Matching Policy...")
    model = FlowPolicy(cfg).to(device)
    init_weights_path = _load_initial_weights(model, cfg, device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")

    if pretrain_ckpt is not None:
        print(f"\nLoading pretrained weights from: {pretrain_ckpt}")
        ckpt = torch.load(pretrain_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        if "finetune_learning_rate" in cfg.get("training", {}):
            cfg["training"]["_effective_lr"] = cfg["training"]["finetune_learning_rate"]
        else:
            cfg["training"]["_effective_lr"] = cfg["training"]["learning_rate"]
    else:
        cfg["training"]["_effective_lr"] = cfg["training"]["learning_rate"]
    print()

    perf_matrix = np.full((n_tasks, n_tasks), np.nan)
    training_log = []
    tb_global_step = 0
    low_dim_keys = [k for k in data_cfg["obs_keys"] if "image" not in k]

    for task_k in range(n_tasks):
        print("\n" + "=" * 70)
        print(f"STAGE {task_k + 1}/{n_tasks}: Task {task_k}  —  {task_names[task_k]}")
        print("=" * 70)

        demo_rel = benchmark.get_task_demonstration(task_k)
        demo_path = os.path.join(data_root, demo_rel)

        # Freeze teacher + collect on-policy obs for FM-SDFT
        teacher = None
        sdft_collected = None
        n_states = 0
        if sdft_cfg.get("enabled", False) and task_k > 0:
            print(f"\n  [FM-SDFT] Freezing teacher snapshot for task {task_k}...")
            teacher = copy.deepcopy(model).to(device)
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad_(False)

            sdft_collected = _collect_sdft_observations(
                model=model, benchmark=benchmark, task_k=task_k,
                sdft_cfg=sdft_cfg, eval_cfg=eval_cfg, data_cfg=data_cfg,
                action_mean=action_mean, action_std=action_std,
                low_dim_keys=low_dim_keys, device=device, seed=seed,
            )
            n_states = len(sdft_collected)

        t_start = time.time()
        model, ema, epoch_losses, task_steps, task_dataset = train_on_task(
            model=model, task_idx=task_k, task_name=task_names[task_k],
            demo_path=demo_path, cfg=cfg, action_mean=action_mean, action_std=action_std,
            device=device, tb_writer=tb_writer, tb_global_step_offset=tb_global_step,
            replay_memory=replay_memory, teacher=teacher, sdft_collected=sdft_collected,
            use_wandb=use_wandb, task_embeddings=task_embeddings,
        )
        tb_global_step += task_steps

        del teacher
        del sdft_collected
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if replay_memory is not None:
            replay_memory.add_task(demo_path, task_dataset.index, task_name=task_names[task_k])
            print(f"  Replay buffer: {replay_memory.num_samples()} samples / "
                  f"{replay_memory.num_tasks()} task(s)")
        del task_dataset

        train_time = time.time() - t_start
        print(f"\n  Training time: {train_time:.1f}s | Final loss: {epoch_losses[-1]:.4f}")

        ckpt_path = ckpt_dir / f"after_task_{task_k:02d}.pt"
        ema_ckpt_path = ckpt_dir / f"after_task_{task_k:02d}_ema.pt"
        payload = {
            "task_idx": task_k, "task_name": task_names[task_k],
            "config": cfg, "action_mean": action_mean, "action_std": action_std,
            "epoch_losses": epoch_losses, "wandb_run_id": wandb_run_id,
        }
        _save_checkpoint(ckpt_path, payload, model, ema, use_ema=False)
        _save_checkpoint(ema_ckpt_path, payload, model, ema, use_ema=True)
        print(f"  Checkpoint: {ckpt_path}")

        stage_log = {
            "task_idx": task_k, "task_name": task_names[task_k],
            "train_time_s": train_time, "final_train_loss": float(epoch_losses[-1]),
            "sdft_n_states": int(n_states),
        }
        if replay_memory is not None:
            stage_log["replay_buffer_samples"] = replay_memory.num_samples()

        if not skip_eval:
            print(f"\n  Evaluating on tasks 0..{task_k}:")
            eval_model = ema.model
            eval_model.eval()
            t_eval = time.time()
            eval_results = evaluate_checkpoint_on_all_tasks(
                model=eval_model, benchmark=benchmark,
                task_indices=list(range(task_k + 1)),
                num_episodes=eval_cfg.get("num_episodes", 20),
                max_steps=eval_cfg.get("max_steps_per_episode", 600),
                action_execution_horizon=eval_cfg.get("action_execution_horizon", 8),
                action_mean=action_mean if data_cfg.get("normalize_action", True) else None,
                action_std=action_std if data_cfg.get("normalize_action", True) else None,
                obs_horizon=data_cfg["obs_horizon"],
                image_size=tuple(data_cfg.get("image_size", [128, 128])),
                use_eye_in_hand=data_cfg.get("use_eye_in_hand", True),
                low_dim_keys=low_dim_keys, device=device,
                use_ddim=False,  # FM uses Euler integration via sample_action()
                ddim_steps=eval_cfg.get("num_flow_steps", model.num_flow_steps),
                seed=seed,
                save_video=eval_cfg.get("save_video", False),
                video_dir=str(results_dir / "videos" / f"stage_{task_k:02d}"),
                task_embeddings=task_embeddings,
            )
            eval_time = time.time() - t_eval

            for task_j, sr in eval_results.items():
                perf_matrix[task_k, task_j] = sr
            avg_sr_stage = np.nanmean(perf_matrix[task_k, : task_k + 1])
            nbt_so_far = compute_nbt(perf_matrix[: task_k + 1, : task_k + 1])

            stage_log.update({
                "eval_time_s": eval_time, "avg_sr": float(avg_sr_stage),
                "nbt": float(nbt_so_far),
                "eval_results": {str(k): float(v) for k, v in eval_results.items()},
            })

            if use_wandb:
                import wandb as _wandb
                eval_log = {f"eval/task{j}_sr": float(sr) for j, sr in eval_results.items()}
                eval_log["eval/avg_sr"] = float(avg_sr_stage)
                eval_log["eval/nbt"] = float(nbt_so_far)
                _wandb.log(eval_log, step=tb_global_step)

            print(f"\n  --- Stage {task_k + 1} ---  "
                  f"Avg SR: {avg_sr_stage:.4f}  NBT: {nbt_so_far:.4f}")
        else:
            print("\n  [skip-eval] evaluation skipped")

        training_log.append(stage_log)
        _save_intermediate(perf_matrix, task_names, training_log, cfg, results_dir, task_k)

    # Final metrics
    run_meta = {
        "end_time": datetime.now().isoformat(), "config": cfg,
        "n_tasks": n_tasks, "task_names": task_names,
        "param_count": param_count, "training_log": training_log,
    }

    if not skip_eval:
        nbt_final = compute_nbt(perf_matrix)
        avg_sr_final = compute_average_sr(perf_matrix)
        print(f"\nFINAL: Avg SR = {avg_sr_final:.4f}  |  NBT = {nbt_final:.4f}")
        run_meta["nbt"] = float(nbt_final)
        run_meta["avg_sr_final"] = float(avg_sr_final)
        save_results_json(perf_matrix, task_names, nbt_final, avg_sr_final, cfg,
                          str(results_dir / "results.json"))
        save_results_csv(perf_matrix, task_names, str(results_dir / "perf_matrix.csv"))
        np.save(results_dir / "perf_matrix.npy", perf_matrix)
        plot_performance_matrix(perf_matrix, task_names, str(results_dir / "heatmap.png"),
                                benchmark_name=benchmark_cfg.get("name"))
        plot_forgetting_summary(perf_matrix, task_names, str(results_dir / "forgetting_summary.png"))

    with open(results_dir / "run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2, default=str)

    print(f"\nAll results saved to: {results_dir}")
    if tb_writer:
        tb_writer.flush()
        tb_writer.close()
    if use_wandb:
        import wandb as _wandb
        _wandb.finish()


def _save_intermediate(perf_matrix, task_names, training_log, cfg, results_dir, task_k):
    np.save(results_dir / "perf_matrix_intermediate.npy", perf_matrix)
    nbt = compute_nbt(perf_matrix[: task_k + 1, : task_k + 1])
    avg_sr = np.nanmean(perf_matrix[task_k, : task_k + 1])
    save_results_json(perf_matrix, task_names, nbt, avg_sr, cfg,
                      str(results_dir / "results_intermediate.json"))
    with open(results_dir / "training_log.json", "w") as f:
        json.dump({"completed_tasks": task_k + 1, "training_log": training_log},
                  f, indent=2, default=str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sequential CL with ER + FM-SDFT for Flow Matching Policy on LIBERO"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--pretrain-ckpt", type=str, default=None)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg, skip_eval=args.skip_eval, pretrain_ckpt=args.pretrain_ckpt)
