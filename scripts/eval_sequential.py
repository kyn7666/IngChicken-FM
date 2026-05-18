# -*- coding: utf-8 -*-
"""
Sequential CL evaluation — FM version.

Loads after_task_{K:02d}_ema.pt checkpoints produced by train_sequential.py,
runs rollout evaluation on tasks 0..K, updates the shared perf_matrix, and
logs results to the same wandb run as training.

Usage (from repo root):
  # Evaluate a single task checkpoint:
  python -m scripts.eval_sequential --config configs/cl_object_pt.yaml --task 0

  # Evaluate multiple specific checkpoints:
  python -m scripts.eval_sequential --config configs/cl_object_pt.yaml --task 0 --task 1

  # Evaluate all available checkpoints:
  python -m scripts.eval_sequential --config configs/cl_object_pt.yaml --all
"""

import re
import json
import argparse
from pathlib import Path
from datetime import datetime

import yaml
import numpy as np
import torch

from model.flow_policy import FlowPolicy
from scripts.datasets import compute_global_action_stats
from scripts.evaluation import (
    evaluate_checkpoint_on_all_tasks,
    compute_nbt,
    compute_average_sr,
    save_results_json,
    save_results_csv,
    plot_performance_matrix,
    plot_forgetting_summary,
)
from libero.libero.benchmark import get_benchmark


def main(cfg, task_indices: list):
    device = torch.device(cfg.get("device", "cuda"))
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    benchmark_cfg = cfg["benchmark"]
    data_cfg = cfg["data"]
    eval_cfg = cfg["evaluation"]
    log_cfg = cfg["logging"]
    wandb_cfg = cfg.get("wandb", {})

    ckpt_dir = Path(log_cfg["checkpoint_dir"])
    results_dir = Path(log_cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    benchmark = get_benchmark(benchmark_cfg["name"])(
        task_order_index=benchmark_cfg.get("task_order_index", 0)
    )
    n_tasks = benchmark.get_num_tasks()
    task_names = benchmark.get_task_names()
    data_root = benchmark_cfg["data_root"]
    low_dim_keys = [k for k in data_cfg["obs_keys"] if "image" not in k]

    print("Computing global action normalization stats...")
    action_mean, action_std = compute_global_action_stats(data_root, benchmark)

    # Load or initialise the shared perf_matrix
    inter_path = results_dir / "perf_matrix_intermediate.npy"
    if inter_path.exists():
        saved = np.load(inter_path)
        perf_matrix = np.full((n_tasks, n_tasks), np.nan)
        perf_matrix[:saved.shape[0], :saved.shape[1]] = saved
        print(f"Loaded existing perf_matrix from {inter_path}")
    else:
        perf_matrix = np.full((n_tasks, n_tasks), np.nan)

    # Init wandb — resume the same run as training using the run_id in the checkpoint
    use_wandb = wandb_cfg.get("enabled", False)
    wandb_run_id = None
    if use_wandb:
        first_ckpt = ckpt_dir / f"after_task_{task_indices[0]:02d}.pt"
        if first_ckpt.exists():
            meta = torch.load(first_ckpt, map_location="cpu", weights_only=False)
            wandb_run_id = meta.get("wandb_run_id")
        try:
            import wandb
            date_str = datetime.now().strftime("%m%d%H%M")
            run_name = f"{wandb_cfg['name']}_eval_{date_str}"
            wandb.init(
                entity=wandb_cfg["entity"],
                project=wandb_cfg["project"],
                group=wandb_cfg.get("group"),
                name=run_name,
                tags=wandb_cfg.get("tags", []),
                config=cfg,
                resume="allow" if wandb_run_id else "allow",
                id=wandb_run_id if wandb_run_id else None,
            )
            print(f"wandb run: {run_name}  (id={wandb.run.id})")
        except ImportError:
            print("wandb not available, disabling")
            use_wandb = False

    for task_k in task_indices:
        ckpt_path = ckpt_dir / f"after_task_{task_k:02d}.pt"
        if not ckpt_path.exists():
            print(f"[skip] Checkpoint not found: {ckpt_path}")
            continue

        if not np.all(np.isnan(perf_matrix[task_k, :task_k + 1])):
            print(f"[skip] Task {task_k} already evaluated")
            continue

        print(f"\n{'='*70}")
        print(f"EVAL after task {task_k}: {task_names[task_k]}")
        print(f"{'='*70}")

        ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = FlowPolicy(cfg).to(device)
        model.load_state_dict(ckpt_data["model_state_dict"], strict=True)
        model.eval()

        eval_results = evaluate_checkpoint_on_all_tasks(
            model=model, benchmark=benchmark,
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
            use_ddim=False,
            ddim_steps=eval_cfg.get("num_flow_steps", 10),
            seed=seed,
        )
        del model

        for j, sr in eval_results.items():
            perf_matrix[task_k, j] = sr

        avg_sr = float(np.nanmean(perf_matrix[task_k, :task_k + 1]))
        nbt = float(compute_nbt(perf_matrix[:task_k + 1, :task_k + 1]))
        print(f"\n  Avg SR: {avg_sr:.4f}  |  NBT: {nbt:.4f}")

        np.save(inter_path, perf_matrix)

        if use_wandb:
            import wandb as _wandb
            step = ckpt_data.get("tb_global_step", 0)
            log = {f"eval/task{j}/sr": float(sr) for j, sr in eval_results.items()}
            log["eval/avg_sr"] = avg_sr
            log["eval/nbt"] = nbt
            _wandb.log(log, step=step)

    # Save final metrics once all lower-triangle cells are filled
    tril_mask = np.tril(np.ones((n_tasks, n_tasks), dtype=bool))
    if not np.any(np.isnan(perf_matrix[tril_mask])):
        nbt_final = compute_nbt(perf_matrix)
        avg_sr_final = compute_average_sr(perf_matrix)
        print(f"\nFINAL: Avg SR = {avg_sr_final:.4f}  |  NBT = {nbt_final:.4f}")
        save_results_json(perf_matrix, task_names, nbt_final, avg_sr_final, cfg,
                          str(results_dir / "results.json"))
        save_results_csv(perf_matrix, task_names, str(results_dir / "perf_matrix.csv"))
        np.save(results_dir / "perf_matrix.npy", perf_matrix)
        plot_performance_matrix(perf_matrix, task_names, str(results_dir / "heatmap.png"),
                                benchmark_name=benchmark_cfg.get("name"))
        plot_forgetting_summary(perf_matrix, task_names, str(results_dir / "forgetting_summary.png"))

    if use_wandb:
        import wandb as _wandb
        _wandb.finish()

    print(f"\nResults saved to: {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--task", type=int, action="append", dest="tasks",
                        help="Task index to evaluate (can repeat: --task 0 --task 1)")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all available checkpoints in the checkpoint dir")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.all:
        ckpt_dir = Path(cfg["logging"]["checkpoint_dir"])
        available = []
        for p in sorted(ckpt_dir.glob("after_task_*.pt")):
            m = re.match(r"after_task_(\d+)$", p.stem)
            if m:
                available.append(int(m.group(1)))
        if not available:
            raise FileNotFoundError(f"No after_task_*.pt checkpoints found in {ckpt_dir}")
        task_indices = available
    elif args.tasks:
        task_indices = sorted(args.tasks)
    else:
        parser.error("Specify at least one --task K or use --all")

    main(cfg, task_indices)
