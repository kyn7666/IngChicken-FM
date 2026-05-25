# -*- coding: utf-8 -*-
"""
Visualize policy distribution drift via t-SNE on predicted actions.

Loads fixed observations from LIBERO-Long task 0 demos and compares
action predictions across checkpoints for PT-only vs SDFT.

Usage:
  python -m scripts.visualize_drift_tsne
"""

import os
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from torch.utils.data import DataLoader

from model.flow_policy import FlowPolicy
from scripts.datasets.libero_single_task_dataset import SingleTaskDataset
from scripts.datasets import compute_global_action_stats
from libero.libero.benchmark import get_benchmark

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
SUITE         = "libero_10"
DATA_ROOT     = "/scratch2/kyn7666/libero_data"
CLIP_EMB_PATH = "data/clip_embeddings/libero_10.pt"

PT_CKPT_DIR   = Path("checkpoints/cl_long_clip_s220")
SDFT_CKPT_DIR = Path("checkpoints/cl_long_sdft_clip_l0.1_s220")
BASE_CKPT     = Path("checkpoints/pretrain_fm_clip_1000ep/best_ema.pt")

N_OBS         = 200   # observations per checkpoint
BATCH_SIZE    = 50
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH     = "results_step/drift_tsne_long.png"

OBS_KEYS = ["agentview_image", "ee_pos", "ee_ori", "gripper_states"]

CFG_PATH = "configs/cl_long_sdft_clip_l0.1_s220.yaml"

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def load_model(ckpt_path, cfg, device):
    model = FlowPolicy(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def get_action_vectors(model, dataloader, n_samples, device):
    """Run predict_action on fixed obs, return (n_samples, action_horizon*action_dim)."""
    vecs = []
    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        actions = model.predict_action(batch)   # (B, H, D)
        vecs.append(actions.reshape(actions.shape[0], -1).cpu().numpy())
        if sum(v.shape[0] for v in vecs) >= n_samples:
            break
    return np.concatenate(vecs, axis=0)[:n_samples]


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)

    benchmark = get_benchmark(SUITE)(task_order_index=0)
    task_names = benchmark.get_task_names()

    print("Computing action normalization stats...")
    action_mean, action_std = compute_global_action_stats(DATA_ROOT, benchmark)

    task_embeddings = torch.load(CLIP_EMB_PATH, map_location="cpu", weights_only=False)
    task_emb = task_embeddings[task_names[0]]   # task 0 embedding

    # Build fixed dataset from task 0
    task_file = benchmark.get_task_demonstration(0)
    hdf5_path = os.path.join(DATA_ROOT, task_file)
    dataset = SingleTaskDataset(
        hdf5_path=hdf5_path,
        obs_horizon=cfg["data"]["obs_horizon"],
        action_horizon=cfg["data"]["action_horizon"],
        obs_keys=[k for k in OBS_KEYS],
        image_size=tuple(cfg["data"]["image_size"]),
        use_eye_in_hand=cfg["data"]["use_eye_in_hand"],
        action_mean=action_mean,
        action_std=action_std,
        task_emb=task_emb,
    )

    # Fix a random subset
    rng = np.random.default_rng(42)
    indices = rng.choice(len(dataset), size=min(N_OBS, len(dataset)), replace=False)
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    checkpoints = (
        [("Base", BASE_CKPT)] +
        [(f"T{k}", PT_CKPT_DIR / f"after_task_{k:02d}.pt") for k in range(10)] +
        [(f"T{k}", SDFT_CKPT_DIR / f"after_task_{k:02d}.pt") for k in range(10)]
    )

    all_vecs   = []
    all_labels = []   # "Base" / "PT_T0" .. "PT_T9" / "SDFT_T0" .. "SDFT_T9"

    for i, (tag, ckpt_path) in enumerate(checkpoints):
        if not Path(ckpt_path).exists():
            print(f"[skip] {ckpt_path}")
            continue
        print(f"Loading {tag}: {ckpt_path}")
        model = load_model(ckpt_path, cfg, DEVICE)
        vecs = get_action_vectors(model, loader, N_OBS, DEVICE)
        all_vecs.append(vecs)
        if i == 0:
            label = "Base"
        elif i <= 10:
            label = f"PT_T{i-1}"
        else:
            label = f"SDFT_T{i-11}"
        all_labels.extend([label] * len(vecs))
        del model
        torch.cuda.empty_cache()

    X = np.concatenate(all_vecs, axis=0)
    labels = np.array(all_labels)

    print(f"Running t-SNE on {X.shape}...")
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, perplexity=40, random_state=42, max_iter=1000)
    X_2d = tsne.fit_transform(X)

    # ── Plot ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Policy Distribution Drift: PT-only vs SDFT (LIBERO-Long, Task 0 obs)", fontsize=13)

    task_steps = list(range(10))
    cmap = cm.get_cmap("RdYlGn_r", 10)

    for ax, method, title in [
        (axes[0], "PT",   "PT-only (no SDFT)"),
        (axes[1], "SDFT", "SDFT"),
    ]:
        for k in task_steps:
            mask = labels == f"{method}_T{k}"
            if mask.sum() == 0:
                continue
            color = cmap(k / 9)
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=[color], s=15, alpha=0.6, label=f"After T{k}")

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 9))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Task index")

    plt.tight_layout()
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    plt.savefig(SAVE_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved: {SAVE_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
