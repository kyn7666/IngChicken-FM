"""
Single-task dataset for continual learning with Diffusion Policy on LIBERO.

Loads one HDF5 demo file at a time. Supports external action normalization
stats (computed globally across all tasks before training).
"""

import math
import os
import h5py
import numpy as np
import torch
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    RandomSampler,
    Subset,
    WeightedRandomSampler,
)
from typing import Dict, List, Optional, Tuple


class SingleTaskDataset(Dataset):
    """Loads demonstrations from a single LIBERO task HDF5 file."""

    def __init__(
        self,
        hdf5_path: str,
        obs_horizon: int = 2,
        action_horizon: int = 16,
        obs_keys: Optional[List[str]] = None,
        image_size: Tuple[int, int] = (128, 128),
        use_eye_in_hand: bool = True,
        action_mean: Optional[np.ndarray] = None,
        action_std: Optional[np.ndarray] = None,
        task_emb: Optional[torch.Tensor] = None,
    ):
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.image_size = image_size
        self.use_eye_in_hand = use_eye_in_hand
        self.action_mean = action_mean
        self.action_std = action_std
        self.task_emb = task_emb  # (512,) pre-computed CLIP embedding, or None

        if obs_keys is None:
            obs_keys = [
                "agentview_image",
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
            ]
        self.obs_keys = obs_keys

        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

        self.episodes = []
        self._load_task(hdf5_path)
        self._build_index()

    def _load_task(self, hdf5_path: str):
        with h5py.File(hdf5_path, "r") as f:
            if "data" not in f:
                raise ValueError(f"No 'data' group in {hdf5_path}")

            demos = sorted(
                f["data"].keys(), key=lambda x: int(x.replace("demo_", ""))
            )

            for demo_key in demos:
                demo = f["data"][demo_key]
                ep_data = {"actions": demo["actions"][:]}

                obs_dict = {}
                for key in self.obs_keys:
                    if key in demo["obs"]:
                        obs_dict[key] = demo["obs"][key][:]
                    elif key == "agentview_image" and "agentview_rgb" in demo["obs"]:
                        obs_dict[key] = demo["obs"]["agentview_rgb"][:]

                if self.use_eye_in_hand:
                    for eye_key in ["eye_in_hand_image", "eye_in_hand_rgb"]:
                        if eye_key in demo["obs"]:
                            obs_dict["eye_in_hand_image"] = demo["obs"][eye_key][:]
                            break

                ep_data["obs"] = obs_dict
                self.episodes.append(ep_data)

        print(
            f"  Loaded {len(self.episodes)} episodes, "
            f"avg len={np.mean([len(e['actions']) for e in self.episodes]):.0f}"
        )

    def _build_index(self):
        self.index = []
        for ep_idx, ep in enumerate(self.episodes):
            T = len(ep["actions"])
            max_start = T - self.action_horizon + 1
            for t in range(max(1, max_start)):
                self.index.append((ep_idx, t))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_idx, start_t = self.index[idx]
        ep = self.episodes[ep_idx]
        T = len(ep["actions"])

        obs_start = max(0, start_t - self.obs_horizon + 1)
        obs_end = start_t + 1
        act_end = min(start_t + self.action_horizon, T)

        actions = ep["actions"][start_t:act_end].astype(np.float32)
        if len(actions) < self.action_horizon:
            pad = np.repeat(actions[-1:], self.action_horizon - len(actions), axis=0)
            actions = np.concatenate([actions, pad], axis=0)

        if self.action_mean is not None and self.action_std is not None:
            actions = (actions - self.action_mean) / self.action_std

        result = {"action": torch.from_numpy(actions)}

        for key in self.obs_keys:
            if key not in ep["obs"]:
                continue
            obs_data = ep["obs"][key][obs_start:obs_end]
            if len(obs_data) < self.obs_horizon:
                pad = np.repeat(obs_data[:1], self.obs_horizon - len(obs_data), axis=0)
                obs_data = np.concatenate([pad, obs_data], axis=0)

            if obs_data.ndim == 4:  # image: (T, H, W, C)
                obs_data = obs_data.astype(np.float32) / 255.0
                obs_data = np.transpose(obs_data, (0, 3, 1, 2))  # (T, C, H, W)

            result[f"obs_{key}"] = torch.from_numpy(obs_data.astype(np.float32))

        if "eye_in_hand_image" in ep["obs"]:
            obs_data = ep["obs"]["eye_in_hand_image"][obs_start:obs_end]
            if len(obs_data) < self.obs_horizon:
                pad = np.repeat(obs_data[:1], self.obs_horizon - len(obs_data), axis=0)
                obs_data = np.concatenate([pad, obs_data], axis=0)
            obs_data = obs_data.astype(np.float32) / 255.0
            obs_data = np.transpose(obs_data, (0, 3, 1, 2))
            result["obs_eye_in_hand_image"] = torch.from_numpy(obs_data)

        if self.task_emb is not None:
            result["task_emb"] = self.task_emb

        return result


def compute_global_action_stats(
    data_root: str,
    benchmark,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute action mean/std across all tasks in the benchmark.

    Uses all demo data to ensure consistent normalization throughout
    sequential training and evaluation.
    """
    all_actions = []
    n_tasks = benchmark.get_num_tasks()

    for i in range(n_tasks):
        demo_rel = benchmark.get_task_demonstration(i)
        demo_path = os.path.join(data_root, demo_rel)
        if not os.path.exists(demo_path):
            raise FileNotFoundError(
                f"Demo file not found: {demo_path}\n"
                f"  Expected from benchmark.get_task_demonstration({i}) = '{demo_rel}'"
            )
        with h5py.File(demo_path, "r") as f:
            for demo_key in sorted(f["data"].keys()):
                all_actions.append(f["data"][demo_key]["actions"][:])

    all_actions = np.concatenate(all_actions, axis=0)
    mean = all_actions.mean(axis=0).astype(np.float32)
    std = np.clip(all_actions.std(axis=0).astype(np.float32), 1e-6, None)
    print(f"Global action stats (from {n_tasks} tasks, {len(all_actions)} steps):")
    print(f"  mean = {mean}")
    print(f"  std  = {std}")
    return mean, std


def create_single_task_dataloader(
    hdf5_path: str,
    batch_size: int = 64,
    num_workers: int = 4,
    obs_horizon: int = 2,
    action_horizon: int = 16,
    action_mean: Optional[np.ndarray] = None,
    action_std: Optional[np.ndarray] = None,
    samples_per_epoch: Optional[int] = None,
    **dataset_kwargs,
) -> Tuple[DataLoader, SingleTaskDataset]:
    dataset = SingleTaskDataset(
        hdf5_path=hdf5_path,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        action_mean=action_mean,
        action_std=action_std,
        **dataset_kwargs,
    )

    try:
        torch.zeros(1).pin_memory()
        can_pin = True
    except RuntimeError:
        can_pin = False

    sampler = None
    shuffle = True
    if samples_per_epoch is not None:
        sampler = RandomSampler(
            dataset,
            replacement=True,
            num_samples=int(samples_per_epoch),
        )
        shuffle = False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=can_pin,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )

    return loader, dataset


def _resolve_replay_sample_count(dataset_len: int, buffer_size) -> int:
    """Convert replay.buffer_size into a sample count per previous task."""
    if buffer_size is None:
        return 0
    if isinstance(buffer_size, float) and 0 < buffer_size <= 1:
        return min(dataset_len, max(1, math.ceil(dataset_len * buffer_size)))
    if isinstance(buffer_size, int) and buffer_size > 0:
        return min(dataset_len, buffer_size)
    if isinstance(buffer_size, float) and buffer_size > 1:
        return min(dataset_len, int(buffer_size))
    return 0


def create_replay_dataloader(
    current_hdf5_path: str,
    replay_hdf5_paths: List[str],
    replay_cfg: Dict,
    batch_size: int = 64,
    num_workers: int = 4,
    obs_horizon: int = 2,
    action_horizon: int = 16,
    action_mean: Optional[np.ndarray] = None,
    action_std: Optional[np.ndarray] = None,
    **dataset_kwargs,
):
    """Create a dataloader that mixes current-task data with fixed replay samples."""
    current_dataset = SingleTaskDataset(
        hdf5_path=current_hdf5_path,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        action_mean=action_mean,
        action_std=action_std,
        **dataset_kwargs,
    )

    datasets = [current_dataset]
    replay_lengths = []
    buffer_size = replay_cfg.get("buffer_size", 0)

    for replay_path in replay_hdf5_paths:
        replay_dataset = SingleTaskDataset(
            hdf5_path=replay_path,
            obs_horizon=obs_horizon,
            action_horizon=action_horizon,
            action_mean=action_mean,
            action_std=action_std,
            **dataset_kwargs,
        )
        sample_count = _resolve_replay_sample_count(len(replay_dataset), buffer_size)
        if sample_count <= 0:
            continue

        if sample_count < len(replay_dataset):
            indices = np.random.choice(
                len(replay_dataset), size=sample_count, replace=False
            )
            replay_dataset = Subset(replay_dataset, indices.tolist())

        datasets.append(replay_dataset)
        replay_lengths.append(len(replay_dataset))

    try:
        torch.zeros(1).pin_memory()
        can_pin = True
    except RuntimeError:
        can_pin = False

    if len(datasets) == 1:
        loader = DataLoader(
            current_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=can_pin,
            persistent_workers=num_workers > 0,
            drop_last=True,
        )
        return loader, current_dataset, {
            "current_samples": len(current_dataset),
            "replay_samples": 0,
            "replay_tasks": 0,
        }

    merged_dataset = ConcatDataset(datasets)
    current_len = len(current_dataset)
    replay_len = sum(replay_lengths)
    mix_ratio = float(replay_cfg.get("mix_ratio", 0.5))
    mix_ratio = min(max(mix_ratio, 0.0), 1.0)

    weights = np.empty(len(merged_dataset), dtype=np.float64)
    weights[:current_len] = (1.0 - mix_ratio) / max(current_len, 1)
    weights[current_len:] = mix_ratio / max(replay_len, 1)

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(merged_dataset),
        replacement=True,
    )

    loader = DataLoader(
        merged_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=can_pin,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )

    return loader, merged_dataset, {
        "current_samples": current_len,
        "replay_samples": replay_len,
        "replay_tasks": len(replay_lengths),
    }
