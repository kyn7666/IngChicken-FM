# -*- coding: utf-8 -*-
"""
Flow Matching Policy pretraining on LIBERO-90.

Usage (from repo root):
  python -m scripts.train_pretrain --config configs/pretrain.yaml
"""

import math
import argparse
from pathlib import Path
from datetime import datetime

import yaml
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from model.flow_policy import FlowPolicy, EMAModel
from scripts.datasets.libero_dataset import create_dataloader


def _checkpoint_step(path: Path) -> int:
    digits = "".join(ch if ch.isdigit() else " " for ch in path.stem).split()
    return int(digits[-1]) if digits else -1


def _prepare_run_dirs(cfg: dict) -> Path:
    log_cfg = cfg["logging"]
    exp_name = log_cfg.get("exp_name") or cfg.get("exp_name")

    if exp_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path("output") / f"{exp_name}_{timestamp}"
        ckpt_dir = run_dir / "checkpoints"
        cfg["run_name"] = run_dir.name
        cfg["run_dir"] = str(run_dir.resolve())
        cfg["logging"]["checkpoint_dir"] = str(ckpt_dir.resolve())
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "config_resolved.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    else:
        ckpt_dir = Path(log_cfg["checkpoint_dir"])

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


def _init_tensorboard_writer(cfg: dict, ckpt_dir: Path):
    log_cfg = cfg["logging"]
    if not log_cfg.get("use_tensorboard", False):
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("TensorBoard not available, skipping tensorboard logging")
        return None

    tb_dir = log_cfg.get("tensorboard_dir")
    if tb_dir:
        tb_dir = Path(tb_dir)
    elif cfg.get("run_dir"):
        tb_dir = Path(cfg["run_dir"]) / "tensorboard"
    else:
        tb_dir = ckpt_dir / "tensorboard"

    tb_dir.mkdir(parents=True, exist_ok=True)
    cfg["logging"]["tensorboard_dir"] = str(tb_dir.resolve())
    print(f"TensorBoard logs will be saved to: {tb_dir}")
    return SummaryWriter(log_dir=str(tb_dir))


def _resolve_weights_path(weights_dir: str) -> Path:
    if not weights_dir:
        return None
    path = Path(weights_dir).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"weights_dir does not exist: {path}")
    if path.is_file():
        return path
    for candidate in [path / "checkpoints" / "best_ema.pt", path / "best_ema.pt", path / "best.pt"]:
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
    checkpoint = torch.load(weights_path, map_location=device)
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


def train(cfg):
    device = torch.device(cfg.get("device", "cuda"))
    torch.manual_seed(cfg.get("seed", 42))
    np.random.seed(cfg.get("seed", 42))

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    log_cfg = cfg["logging"]

    ckpt_dir = _prepare_run_dirs(cfg)
    tb_writer = _init_tensorboard_writer(cfg, ckpt_dir)

    print("=" * 60)
    print("Building dataset (LIBERO-90)...")
    print("=" * 60)

    configured_steps_per_epoch = train_cfg.get("steps_per_epoch")
    if configured_steps_per_epoch is not None:
        samples_per_epoch = int(configured_steps_per_epoch) * data_cfg["batch_size"]
        print(f"Step-based training: {configured_steps_per_epoch} steps/epoch "
              f"(samples_per_epoch={samples_per_epoch})")
    else:
        samples_per_epoch = data_cfg.get("samples_per_epoch")

    loader, dataset = create_dataloader(
        data_dir=data_cfg["data_dir"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        obs_horizon=data_cfg["obs_horizon"],
        action_horizon=data_cfg["action_horizon"],
        samples_per_epoch=samples_per_epoch,
        normalize_action=data_cfg.get("normalize_action", True),
        use_eye_in_hand=data_cfg.get("use_eye_in_hand", True),
        image_size=tuple(data_cfg.get("image_size", [128, 128])),
        obs_keys=data_cfg.get("obs_keys"),
    )

    print("=" * 60)
    print("Building Flow Matching Policy...")
    print("=" * 60)

    model = FlowPolicy(cfg).to(device)
    init_weights_path = _load_initial_weights(model, cfg, device)
    ema = EMAModel(model, decay=train_cfg.get("ema_decay", 0.995))
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 1e-6),
    )

    total_steps = train_cfg["num_epochs"] * len(loader)
    warmup_steps = train_cfg.get("lr_warmup_steps", 500)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_amp = train_cfg.get("mixed_precision", True)
    scaler = GradScaler(enabled=use_amp)

    wandb_cfg = cfg.get("wandb", {})
    use_wandb = wandb_cfg.get("enabled", False)
    _wandb = None
    wandb_run_id = None
    if use_wandb:
        try:
            import wandb as _wandb
            date_str = datetime.now().strftime("%m%d")
            run_name = f"{wandb_cfg['name']}_{date_str}"
            _wandb.init(
                entity=wandb_cfg["entity"],
                project=wandb_cfg["project"],
                group=wandb_cfg.get("group"),
                name=run_name,
                tags=wandb_cfg.get("tags", []),
                config=cfg,
                resume=wandb_cfg.get("resume", "allow"),
            )
            wandb_run_id = _wandb.run.id
            print(f"wandb run: {run_name}  (id={wandb_run_id})")
        except ImportError:
            print("wandb not available, disabling wandb logging")
            use_wandb = False

    global_step = 0
    best_loss = float("inf")

    for epoch in range(train_cfg["num_epochs"]):
        model.train()
        epoch_losses = []

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{train_cfg['num_epochs']}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}

            with autocast(enabled=use_amp):
                loss = model.compute_loss(batch)

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
            epoch_losses.append(loss_val)
            global_step += 1
            pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

            if use_wandb and global_step % log_cfg.get("log_interval", 100) == 0:
                _wandb.log({
                    "train/loss": loss_val,
                    "train/lr": scheduler.get_last_lr()[0],
                }, step=global_step)
            if tb_writer and global_step % log_cfg.get("log_interval", 100) == 0:
                tb_writer.add_scalar("train/loss", loss_val, global_step)
                tb_writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1} | avg_loss={avg_loss:.4f} | lr={scheduler.get_last_lr()[0]:.2e}")
        if tb_writer:
            tb_writer.add_scalar("epoch/avg_loss", avg_loss, epoch + 1)

        checkpoint_payload = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": cfg,
            "action_mean": dataset.action_mean if hasattr(dataset, "action_mean") else None,
            "action_std": dataset.action_std if hasattr(dataset, "action_std") else None,
            "wandb_run_id": wandb_run_id,
        }

        if (epoch + 1) % log_cfg.get("save_interval", 10) == 0:
            ckpt_path = ckpt_dir / f"epoch_{epoch+1:04d}.pt"
            ema_ckpt_path = ckpt_dir / f"epoch_{epoch+1:04d}_ema.pt"
            _save_checkpoint(ckpt_path, checkpoint_payload, model, ema, False)
            _save_checkpoint(ema_ckpt_path, checkpoint_payload, model, ema, True)
            print(f"Saved checkpoint: {ckpt_path}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            _save_checkpoint(ckpt_dir / "best.pt", checkpoint_payload, model, ema, False)
            _save_checkpoint(ckpt_dir / "best_ema.pt", checkpoint_payload, model, ema, True)

    if tb_writer:
        tb_writer.flush()
        tb_writer.close()
    if use_wandb:
        _wandb.finish()

    print(f"\nTraining complete! Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    train(cfg)
