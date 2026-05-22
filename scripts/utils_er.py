"""
Utilities for experience replay in Diffusion Policy training.

This module keeps replay-specific code out of the main training and dataset
files so the baseline code path stays easy to read.
"""

import glob
import os
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def merge_batches(current_batch: dict, replay_batch: dict) -> dict:
    if set(current_batch.keys()) != set(replay_batch.keys()):
        missing_from_replay = sorted(set(current_batch.keys()) - set(replay_batch.keys()))
        missing_from_current = sorted(set(replay_batch.keys()) - set(current_batch.keys()))
        raise KeyError(
            "Replay batch keys do not match current batch keys. "
            f"Missing from replay: {missing_from_replay}; "
            f"missing from current: {missing_from_current}"
        )

    return {
        key: torch.cat([current_batch[key], replay_batch[key]], dim=0)
        for key in current_batch.keys()
    }


def can_pin_memory() -> bool:
    try:
        torch.zeros(1).pin_memory()
        return True
    except RuntimeError:
        return False


def split_batch_size(batch_size: int, mix_ratio: float) -> Tuple[int, int]:
    replay_batch_size = int(round(batch_size * mix_ratio))
    replay_batch_size = min(max(replay_batch_size, 1), batch_size - 1)
    current_batch_size = batch_size - replay_batch_size
    return current_batch_size, replay_batch_size


def list_hdf5_files(data_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(data_dir, "**/*.hdf5"), recursive=True))


def compute_action_stats_from_data_dirs(data_dirs: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    sum_actions = None
    sumsq_actions = None
    total_steps = 0

    for data_dir in data_dirs:
        hdf5_files = list_hdf5_files(data_dir)
        if not hdf5_files:
            raise FileNotFoundError(f"No HDF5 files found in {data_dir}")

        for fpath in hdf5_files:
            with h5py.File(fpath, "r") as f:
                if "data" not in f:
                    continue
                for demo_key in sorted(f["data"].keys(), key=lambda x: int(x.replace("demo_", ""))):
                    actions = f["data"][demo_key]["actions"][:].astype(np.float64)
                    if sum_actions is None:
                        action_dim = actions.shape[-1]
                        sum_actions = np.zeros(action_dim, dtype=np.float64)
                        sumsq_actions = np.zeros(action_dim, dtype=np.float64)

                    sum_actions += actions.sum(axis=0)
                    sumsq_actions += np.square(actions).sum(axis=0)
                    total_steps += actions.shape[0]

    if total_steps == 0 or sum_actions is None or sumsq_actions is None:
        raise ValueError("Could not compute action statistics from the provided data directories")

    mean = sum_actions / total_steps
    var = np.maximum(sumsq_actions / total_steps - np.square(mean), 1e-12)
    std = np.sqrt(var)
    return mean.astype(np.float32), np.clip(std.astype(np.float32), 1e-6, None)


class ReplayTaskDataset(Dataset):
    """Lazy replay dataset over a selected subset of single-task sample windows."""

    def __init__(
        self,
        hdf5_path: str,
        sample_index: List[Tuple[int, int]],
        obs_horizon: int = 2,
        action_horizon: int = 16,
        obs_keys: Optional[List[str]] = None,
        image_size: Tuple[int, int] = (128, 128),
        use_eye_in_hand: bool = True,
        action_mean: Optional[np.ndarray] = None,
        action_std: Optional[np.ndarray] = None,
        task_emb: Optional[torch.Tensor] = None,
    ):
        self.hdf5_path = hdf5_path
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.image_size = image_size
        self.use_eye_in_hand = use_eye_in_hand
        self.action_mean = action_mean
        self.action_std = action_std
        self.task_emb = task_emb
        self.sample_index = [tuple(idx) for idx in sample_index]

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

        self._file_handle = None
        self.episodes = []
        self._scan_task()

    def _scan_task(self):
        with h5py.File(self.hdf5_path, "r") as f:
            if "data" not in f:
                raise ValueError(f"No 'data' group in {self.hdf5_path}")

            demos = sorted(
                f["data"].keys(), key=lambda x: int(x.replace("demo_", ""))
            )

            for demo_key in demos:
                demo = f["data"][demo_key]
                obs_group = demo["obs"]

                obs_sources = {}
                for key in self.obs_keys:
                    if key in obs_group:
                        obs_sources[key] = key
                    elif key == "agentview_image" and "agentview_rgb" in obs_group:
                        obs_sources[key] = "agentview_rgb"

                eye_in_hand_source = None
                if self.use_eye_in_hand:
                    for eye_key in ["eye_in_hand_image", "eye_in_hand_rgb"]:
                        if eye_key in obs_group:
                            eye_in_hand_source = eye_key
                            break

                self.episodes.append(
                    {
                        "demo_key": demo_key,
                        "length": int(demo["actions"].shape[0]),
                        "obs_sources": obs_sources,
                        "eye_in_hand_source": eye_in_hand_source,
                    }
                )

    def _get_file_handle(self) -> h5py.File:
        if self._file_handle is None:
            self._file_handle = h5py.File(self.hdf5_path, "r")
        return self._file_handle

    def close(self):
        if self._file_handle is not None:
            try:
                self._file_handle.close()
            except Exception:
                pass
            self._file_handle = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file_handle"] = None
        return state

    def __del__(self):
        self.close()

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_idx, start_t = self.sample_index[idx]
        ep = self.episodes[ep_idx]
        T = ep["length"]

        file_handle = self._get_file_handle()
        demo = file_handle["data"][ep["demo_key"]]
        obs_group = demo["obs"]

        obs_start = max(0, start_t - self.obs_horizon + 1)
        obs_end = start_t + 1
        act_end = min(start_t + self.action_horizon, T)

        actions = demo["actions"][start_t:act_end].astype(np.float32)
        if len(actions) < self.action_horizon:
            pad = np.repeat(actions[-1:], self.action_horizon - len(actions), axis=0)
            actions = np.concatenate([actions, pad], axis=0)

        if self.action_mean is not None and self.action_std is not None:
            actions = (actions - self.action_mean) / self.action_std

        result = {"action": torch.from_numpy(actions)}

        for key, source_key in ep["obs_sources"].items():
            obs_data = obs_group[source_key][obs_start:obs_end]
            if len(obs_data) < self.obs_horizon:
                pad = np.repeat(obs_data[:1], self.obs_horizon - len(obs_data), axis=0)
                obs_data = np.concatenate([pad, obs_data], axis=0)

            if obs_data.ndim == 4:
                obs_data = obs_data.astype(np.float32) / 255.0
                obs_data = np.transpose(obs_data, (0, 3, 1, 2))

            result[f"obs_{key}"] = torch.from_numpy(obs_data.astype(np.float32))

        if ep["eye_in_hand_source"] is not None:
            obs_data = obs_group[ep["eye_in_hand_source"]][obs_start:obs_end]
            if len(obs_data) < self.obs_horizon:
                pad = np.repeat(obs_data[:1], self.obs_horizon - len(obs_data), axis=0)
                obs_data = np.concatenate([pad, obs_data], axis=0)
            obs_data = obs_data.astype(np.float32) / 255.0
            obs_data = np.transpose(obs_data, (0, 3, 1, 2))
            result["obs_eye_in_hand_image"] = torch.from_numpy(obs_data)

        if self.task_emb is not None:
            result["task_emb"] = self.task_emb

        return result


class ReplayMemory:
    """Bounded per-task replay memory for rehearsal-based continual learning."""

    def __init__(self, capacity: int, seed: int = 0):
        self.capacity = max(int(capacity), 0)
        self.rng = np.random.default_rng(seed)
        self.entries = []

    def has_samples(self) -> bool:
        return any(entry["sample_index"] for entry in self.entries)

    def num_tasks(self) -> int:
        return sum(1 for entry in self.entries if entry["sample_index"])

    def num_samples(self) -> int:
        return sum(len(entry["sample_index"]) for entry in self.entries)

    def add_task(self, hdf5_path: str, sample_index: List[Tuple[int, int]],
                 task_name: str = None):
        if self.capacity <= 0 or len(sample_index) == 0:
            return

        self.entries.append(
            {
                "hdf5_path": hdf5_path,
                "task_name": task_name,
                "all_index": [tuple(idx) for idx in sample_index],
                "sample_index": [],
            }
        )
        self._rebalance()

    def _rebalance(self):
        if not self.entries:
            return

        n_tasks = len(self.entries)
        base = self.capacity // n_tasks
        remainder = self.capacity % n_tasks

        for entry_idx, entry in enumerate(self.entries):
            target = base + (1 if entry_idx < remainder else 0)
            target = min(target, len(entry["all_index"]))

            if target <= 0:
                entry["sample_index"] = []
                continue

            chosen = self.rng.choice(len(entry["all_index"]), size=target, replace=False)
            chosen = np.sort(chosen)
            entry["sample_index"] = [entry["all_index"][i] for i in chosen.tolist()]

    def build_loader(
        self,
        cfg: dict,
        action_mean: np.ndarray,
        action_std: np.ndarray,
        batch_size: int,
        task_embeddings: dict = None,
    ):
        if batch_size <= 0 or not self.has_samples():
            return None

        data_cfg = cfg["data"]
        datasets = []
        for entry in self.entries:
            if not entry["sample_index"]:
                continue
            task_emb = None
            if task_embeddings and entry.get("task_name"):
                task_emb = task_embeddings.get(entry["task_name"])
            datasets.append(
                ReplayTaskDataset(
                    hdf5_path=entry["hdf5_path"],
                    sample_index=entry["sample_index"],
                    obs_horizon=data_cfg["obs_horizon"],
                    action_horizon=data_cfg["action_horizon"],
                    obs_keys=data_cfg["obs_keys"],
                    use_eye_in_hand=data_cfg.get("use_eye_in_hand", True),
                    image_size=tuple(data_cfg.get("image_size", [128, 128])),
                    action_mean=action_mean if data_cfg.get("normalize_action", True) else None,
                    action_std=action_std if data_cfg.get("normalize_action", True) else None,
                    task_emb=task_emb,
                )
            )

        if not datasets:
            return None

        replay_dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
        num_workers = data_cfg["num_workers"]
        return DataLoader(
            replay_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=can_pin_memory(),
            persistent_workers=num_workers > 0,
            drop_last=False,
        )
