# -*- coding: utf-8 -*-
"""
Extended drift visualization:
  1. Separate t-SNE per method (PT / SDFT independently, with Base)
  2. Cosine distance from base model per task (line plot)

Usage:
  python -m scripts.visualize_drift_extended
"""

import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances

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

N_OBS      = 200
BATCH_SIZE = 50
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR   = "results_step"

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
    vecs = []
    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        actions = model.predict_action(batch)   # (B, H, D)
        vecs.append(actions.reshape(actions.shape[0], -1).cpu().numpy())
        if sum(v.shape[0] for v in vecs) >= n_samples:
            break
    return np.concatenate(vecs, axis=0)[:n_samples]


def run_tsne(vecs_list, perplexity=30):
    X = np.concatenate(vecs_list, axis=0)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    return tsne.fit_transform(X)


def mean_cosine_dist(base_vecs, ckpt_vecs):
    """Per-observation cosine distance, then average."""
    dists = cosine_distances(base_vecs, ckpt_vecs)   # (N, N)
    return float(np.diag(dists).mean())              # paired distance


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
    task_emb = task_embeddings[task_names[0]]

    task_file = benchmark.get_task_demonstration(0)
    hdf5_path = os.path.join(DATA_ROOT, task_file)
    dataset = SingleTaskDataset(
        hdf5_path=hdf5_path,
        obs_horizon=cfg["data"]["obs_horizon"],
        action_horizon=cfg["data"]["action_horizon"],
        obs_keys=list(OBS_KEYS),
        image_size=tuple(cfg["data"]["image_size"]),
        use_eye_in_hand=cfg["data"]["use_eye_in_hand"],
        action_mean=action_mean,
        action_std=action_std,
        task_emb=task_emb,
    )

    rng = np.random.default_rng(42)
    indices = rng.choice(len(dataset), size=min(N_OBS, len(dataset)), replace=False)
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Collect all vectors ────────────────────
    def collect(tag, ckpt_path):
        if not Path(ckpt_path).exists():
            print(f"[skip] {ckpt_path}")
            return None
        print(f"Loading {tag}: {ckpt_path}")
        model = load_model(ckpt_path, cfg, DEVICE)
        vecs = get_action_vectors(model, loader, N_OBS, DEVICE)
        del model; torch.cuda.empty_cache()
        return vecs

    base_vecs = collect("Base", BASE_CKPT)

    pt_vecs   = []   # list of (N, D) per task, None if missing
    sdft_vecs = []
    for k in range(10):
        pt_vecs.append(collect(f"PT_T{k}",   PT_CKPT_DIR   / f"after_task_{k:02d}.pt"))
        sdft_vecs.append(collect(f"SDFT_T{k}", SDFT_CKPT_DIR / f"after_task_{k:02d}.pt"))

    os.makedirs(SAVE_DIR, exist_ok=True)
    cmap = plt.colormaps.get_cmap("RdYlGn_r").resampled(10)
    task_steps = list(range(10))

    # ══════════════════════════════════════════
    # 1. Separate t-SNE
    # ══════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Policy Distribution Drift – Separate t-SNE (LIBERO-Long, Task 0 obs)", fontsize=13)

    for ax, vecs_list_raw, method, title in [
        (axes[0], pt_vecs,   "PT",   "PT-only (no SDFT)"),
        (axes[1], sdft_vecs, "SDFT", "SDFT"),
    ]:
        available = [(k, v) for k, v in enumerate(vecs_list_raw) if v is not None]
        if not available:
            ax.set_title(f"{title} – no checkpoints")
            continue

        # t-SNE on this method's checkpoints only (no base)
        all_v = np.concatenate([v for _, v in available], axis=0)
        sizes = [v.shape[0] for _, v in available]
        X_2d = run_tsne([all_v])

        offset = 0
        for (k, _), sz in zip(available, sizes):
            color = cmap(k / 9)
            ax.scatter(X_2d[offset:offset+sz, 0], X_2d[offset:offset+sz, 1],
                       c=[color], s=15, alpha=0.6)
            offset += sz

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 9))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Task index")

    plt.tight_layout()
    save_path_tsne = os.path.join(SAVE_DIR, "drift_tsne_separate.png")
    plt.savefig(save_path_tsne, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path_tsne}")

    # ══════════════════════════════════════════
    # 2. Cosine distance from base (line plot)
    # ══════════════════════════════════════════
    if base_vecs is None:
        print("[skip cosine plot] base checkpoint missing")
        return

    pt_dists   = []
    sdft_dists = []
    x_ticks    = []

    for k in range(10):
        if pt_vecs[k] is not None:
            pt_dists.append((k, mean_cosine_dist(base_vecs, pt_vecs[k])))
        if sdft_vecs[k] is not None:
            sdft_dists.append((k, mean_cosine_dist(base_vecs, sdft_vecs[k])))

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    if pt_dists:
        xs, ys = zip(*pt_dists)
        ax2.plot(xs, ys, "o-", color="tomato",      label="PT-only",  linewidth=2, markersize=6)
    if sdft_dists:
        xs, ys = zip(*sdft_dists)
        ax2.plot(xs, ys, "s-", color="steelblue",   label="SDFT",     linewidth=2, markersize=6)

    ax2.set_title("Policy Distribution Drift from Base Model\n(LIBERO-Long, Task 0 obs)", fontsize=12)
    ax2.set_xlabel("Task index (after training on task k)", fontsize=11)
    ax2.set_ylabel("Mean cosine distance from base", fontsize=11)
    ax2.set_xticks(range(10))
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path_cos = os.path.join(SAVE_DIR, "drift_cosine_long.png")
    plt.savefig(save_path_cos, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path_cos}")


if __name__ == "__main__":
    main()
