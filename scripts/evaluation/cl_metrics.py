"""
Continual learning metrics and visualization.

Computes:
  - Performance matrix (task success rate after each training stage)
  - Negative Backward Transfer (NBT)
  - Average Success Rate
  - Heatmap and forgetting plots
"""

import json
import csv
import os
import numpy as np


def compute_nbt(perf_matrix: np.ndarray) -> float:
    """Compute Exponential Normalized Negative Backward Transfer.

    NBT_k = (1 / (K - k)) * sum_{tau=k+1}^{K} (exp((c_{k,k} - c_{k,tau}) / (c_{k,k} + epsilon)) - 1)

    where c_{k,k} = perf_matrix[k, k] (SR just after learning task k),
    c_{k,tau} = perf_matrix[tau, k] (SR on task k after training through task tau),
    K = N-1 (last task index), epsilon = 1e-8.
    The outer 1/(K-k) uses the count of valid (non-NaN) terms to handle missing entries.
    Overall NBT is the mean of NBT_k over tasks with at least one subsequent evaluation.

    NBT > 0 indicates forgetting.

    Args:
        perf_matrix: (N, N) array where perf_matrix[i, j] is the success rate
                     on task j after training through task i.
                     Upper-triangle entries (j > i) should be NaN.
    """
    N = perf_matrix.shape[0]
    if N <= 1:
        return 0.0

    epsilon = 1e-8
    nbt_sum = 0.0
    count = 0

    for k in range(N - 1):
        c_kk = perf_matrix[k, k]
        if np.isnan(c_kk):
            continue

        terms = [
            np.exp((c_kk - perf_matrix[tau, k]) / (c_kk + epsilon)) - 1
            for tau in range(k + 1, N)
            if not np.isnan(perf_matrix[tau, k])
        ]
        if terms:
            nbt_sum += sum(terms) / len(terms)
            count += 1

    return nbt_sum / max(count, 1)


def compute_forgetting_per_task(perf_matrix: np.ndarray) -> np.ndarray:
    """Compute per-task forgetting: best_past_SR - final_SR for each task."""
    N = perf_matrix.shape[0]
    forgetting = np.zeros(N)
    for j in range(N):
        valid = [
            perf_matrix[i, j] for i in range(j, N) if not np.isnan(perf_matrix[i, j])
        ]
        if len(valid) >= 2:
            forgetting[j] = max(valid) - valid[-1]
    return forgetting


def compute_average_sr(perf_matrix: np.ndarray) -> float:
    """Average success rate across all evaluated (non-NaN) entries in the final row."""
    N = perf_matrix.shape[0]
    final_row = perf_matrix[N - 1]
    valid = final_row[~np.isnan(final_row)]
    return float(np.mean(valid)) if len(valid) > 0 else 0.0


def compute_average_sr_per_stage(perf_matrix: np.ndarray) -> np.ndarray:
    """Average SR at each training stage (row-wise average of valid entries)."""
    N = perf_matrix.shape[0]
    avg_srs = np.zeros(N)
    for i in range(N):
        valid = perf_matrix[i, : i + 1]
        valid = valid[~np.isnan(valid)]
        avg_srs[i] = float(np.mean(valid)) if len(valid) > 0 else 0.0
    return avg_srs


def save_results_json(
    perf_matrix: np.ndarray,
    task_names: list,
    nbt: float,
    avg_sr: float,
    config: dict,
    save_path: str,
):
    """Save all metrics to a JSON file."""
    N = perf_matrix.shape[0]
    matrix_dict = {}
    for i in range(N):
        row = {}
        for j in range(i + 1):
            row[task_names[j]] = (
                float(perf_matrix[i, j])
                if not np.isnan(perf_matrix[i, j])
                else None
            )
        matrix_dict[f"after_task_{i}_{task_names[i]}"] = row

    results = {
        "benchmark": config.get("benchmark", {}).get("name", "unknown"),
        "task_order_index": config.get("benchmark", {}).get("task_order_index", 0),
        "task_names": task_names,
        "performance_matrix": matrix_dict,
        "nbt": nbt,
        "average_sr_final": avg_sr,
        "average_sr_per_stage": compute_average_sr_per_stage(perf_matrix).tolist(),
        "forgetting_per_task": compute_forgetting_per_task(perf_matrix).tolist(),
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {save_path}")


def save_results_csv(
    perf_matrix: np.ndarray,
    task_names: list,
    save_path: str,
):
    """Save performance matrix as CSV."""
    N = perf_matrix.shape[0]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["trained_through"] + [f"task_{j}_{task_names[j][:30]}" for j in range(N)]
        writer.writerow(header)
        for i in range(N):
            row = [f"task_{i}_{task_names[i][:30]}"]
            for j in range(N):
                if np.isnan(perf_matrix[i, j]):
                    row.append("")
                else:
                    row.append(f"{perf_matrix[i, j]:.4f}")
            writer.writerow(row)
    print(f"CSV saved to {save_path}")


def plot_performance_matrix(
    perf_matrix: np.ndarray,
    task_names: list,
    save_path: str,
    benchmark_name: str = None,
):
    """Plot heatmap of the performance matrix."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = perf_matrix.shape[0]

    mask = np.isnan(perf_matrix)
    display_matrix = np.ma.masked_where(mask, perf_matrix)
    cmap = plt.cm.get_cmap("RdYlGn").copy()
    cmap.set_bad(color="white")

    fig, ax = plt.subplots(1, 1, figsize=(10, 8), facecolor="white")
    ax.set_facecolor("white")
    im = ax.imshow(display_matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    for i in range(N):
        for j in range(N):
            if not mask[i, j]:
                val = perf_matrix[i, j]
                color = "white" if val < 0.4 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

    short_names = [n[:20] for n in task_names]
    ax.set_xticks(range(N))
    ax.set_xticklabels([f"T{j}" for j in range(N)], fontsize=8)
    ax.set_yticks(range(N))
    ax.set_yticklabels([f"After T{i}" for i in range(N)], fontsize=8)
    ax.set_xlabel("Evaluated Task", fontsize=11)
    ax.set_ylabel("Trained Through Task", fontsize=11)
    ax.set_title("Performance Matrix (Success Rate)", fontsize=13, fontweight="bold")
    if benchmark_name:
        benchmark_label = benchmark_name.replace("_", "-")
        ax.text(
            0.5,
            1.03,
            f"Dataset: {benchmark_label}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10,
            color="dimgray",
        )

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Success Rate", fontsize=10)

    for j in range(N):
        for i in range(j):
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                        fill=True, facecolor="lightgray",
                                        edgecolor="gray", linewidth=0.5, alpha=0.5))

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Heatmap saved to {save_path}")


def plot_forgetting_summary(
    perf_matrix: np.ndarray,
    task_names: list,
    save_path: str,
):
    """Plot forgetting summary: per-task forgetting and average SR over stages."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = perf_matrix.shape[0]
    forgetting = compute_forgetting_per_task(perf_matrix)
    avg_sr_stages = compute_average_sr_per_stage(perf_matrix)
    nbt = compute_nbt(perf_matrix)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Per-task forgetting bar chart
    ax = axes[0]
    colors = ["#d32f2f" if f > 0.05 else "#4caf50" for f in forgetting]
    ax.bar(range(N), forgetting, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(N))
    ax.set_xticklabels([f"T{j}" for j in range(N)], fontsize=8)
    ax.set_ylabel("Forgetting (best SR - final SR)", fontsize=10)
    ax.set_title(f"Per-Task Forgetting  |  NBT = {nbt:.3f}", fontsize=11, fontweight="bold")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_ylim(bottom=min(-0.05, forgetting.min() - 0.05))

    # (b) Average SR across training stages
    ax = axes[1]
    ax.plot(range(N), avg_sr_stages, "o-", color="#1976d2", linewidth=2, markersize=6)
    ax.set_xticks(range(N))
    ax.set_xticklabels([f"After T{i}" for i in range(N)], fontsize=8, rotation=45)
    ax.set_ylabel("Avg SR (over seen tasks)", fontsize=10)
    ax.set_title("Average Success Rate Over Training", fontsize=11, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # (c) Diagonal (learning) vs final performance per task
    ax = axes[2]
    diag = np.array([perf_matrix[i, i] for i in range(N)])
    final_row = perf_matrix[N - 1]
    x = np.arange(N)
    width = 0.35
    ax.bar(x - width / 2, diag, width, label="Just learned", color="#4caf50", edgecolor="black", linewidth=0.5)
    final_valid = np.where(np.isnan(final_row), 0, final_row)
    ax.bar(x + width / 2, final_valid, width, label="After all tasks", color="#d32f2f", edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(N))
    ax.set_xticklabels([f"T{j}" for j in range(N)], fontsize=8)
    ax.set_ylabel("Success Rate", fontsize=10)
    ax.set_title("Learning vs Retention", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.05)

    plt.suptitle("Catastrophic Forgetting Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Forgetting summary saved to {save_path}")
