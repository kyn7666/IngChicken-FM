"""
LIBERO-90 Dataset for Diffusion Policy.

Default behavior:
  Each epoch iterates over every valid (obs_horizon, action_horizon) sample
  window once in shuffled order.

Optional behavior:
  If samples_per_epoch is provided, use per-task uniform sampling instead.
"""

import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm


class LiberoUniformDataset(Dataset):
    """
    Loads all LIBERO-90 HDF5 demo files and provides uniform per-task sampling.
    """

    def __init__(
        self,
        data_dir: str,
        obs_horizon: int = 2,
        action_horizon: int = 16,
        obs_keys: Optional[List[str]] = None,
        image_size: Tuple[int, int] = (128, 128),
        use_eye_in_hand: bool = True,
        normalize_action: bool = True,
        max_episodes_per_task: Optional[int] = None,
        task_embeddings: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.image_size = image_size
        self.use_eye_in_hand = use_eye_in_hand
        self.normalize_action = normalize_action

        if obs_keys is None:
            obs_keys = [
                "agentview_image",
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
            ]
        self.obs_keys = obs_keys
        self.task_embeddings = task_embeddings  # {task_name: tensor(512)} or None

        hdf5_files = sorted(glob.glob(os.path.join(data_dir, "**/*.hdf5"), recursive=True))
        if not hdf5_files:
            raise FileNotFoundError(f"No HDF5 files found in {data_dir}")

        print(f"Found {len(hdf5_files)} HDF5 files")

        self.task_data: List[Dict] = []
        self.task_names: List[str] = []
        self._load_all_tasks(hdf5_files, max_episodes_per_task)

        if self.normalize_action:
            self._compute_action_stats()

        self._build_index()
        print(f"Dataset ready: {len(self.task_data)} tasks, {self.total_samples} total samples")

    def _load_all_tasks(self, hdf5_files: List[str], max_episodes: Optional[int]):
        for fpath in tqdm(hdf5_files, desc="Loading tasks"):
            task_name = os.path.splitext(os.path.basename(fpath))[0]

            with h5py.File(fpath, "r") as f:
                if "data" not in f:
                    print(f"  Skipping {task_name}: no 'data' group")
                    continue

                demos = sorted(f["data"].keys(), key=lambda x: int(x.replace("demo_", "")))
                if max_episodes:
                    demos = demos[:max_episodes]

                episodes = []
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
                    episodes.append(ep_data)

                if episodes:
                    self.task_data.append({"name": task_name, "episodes": episodes})
                    self.task_names.append(task_name)
                    print(
                        f"  {task_name}: {len(episodes)} episodes, "
                        f"avg len={np.mean([len(e['actions']) for e in episodes]):.0f}"
                    )

    def _compute_action_stats(self):
        all_actions = []
        for task in self.task_data:
            for ep in task["episodes"]:
                all_actions.append(ep["actions"])
        all_actions = np.concatenate(all_actions, axis=0)
        self.action_mean = all_actions.mean(axis=0).astype(np.float32)
        self.action_std = all_actions.std(axis=0).astype(np.float32)
        self.action_std = np.clip(self.action_std, 1e-6, None)
        print(f"Action stats computed: mean={self.action_mean}, std={self.action_std}")

    def _build_index(self):
        self.index = []
        for task_idx, task in enumerate(self.task_data):
            for ep_idx, ep in enumerate(task["episodes"]):
                T = len(ep["actions"])
                max_start = T - self.action_horizon + 1
                for t in range(max(0, max_start)):
                    self.index.append((task_idx, ep_idx, t))
        self.total_samples = len(self.index)

        self.task_indices: Dict[int, List[int]] = {}
        for flat_idx, (task_idx, _, _) in enumerate(self.index):
            if task_idx not in self.task_indices:
                self.task_indices[task_idx] = []
            self.task_indices[task_idx].append(flat_idx)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        task_idx, ep_idx, start_t = self.index[idx]
        ep = self.task_data[task_idx]["episodes"][ep_idx]

        T = len(ep["actions"])

        obs_start = max(0, start_t - self.obs_horizon + 1)
        obs_end = start_t + 1
        act_end = min(start_t + self.action_horizon, T)

        actions = ep["actions"][start_t:act_end].astype(np.float32)
        if len(actions) < self.action_horizon:
            pad = np.repeat(actions[-1:], self.action_horizon - len(actions), axis=0)
            actions = np.concatenate([actions, pad], axis=0)

        if self.normalize_action:
            actions = (actions - self.action_mean) / self.action_std

        result = {"action": torch.from_numpy(actions)}

        for key in self.obs_keys:
            if key not in ep["obs"]:
                continue
            obs_data = ep["obs"][key][obs_start:obs_end]
            if len(obs_data) < self.obs_horizon:
                pad = np.repeat(obs_data[:1], self.obs_horizon - len(obs_data), axis=0)
                obs_data = np.concatenate([pad, obs_data], axis=0)

            if obs_data.ndim == 4:
                obs_data = obs_data.astype(np.float32) / 255.0
                obs_data = np.transpose(obs_data, (0, 3, 1, 2))

            result[f"obs_{key}"] = torch.from_numpy(obs_data.astype(np.float32))

        if "eye_in_hand_image" in ep["obs"]:
            obs_data = ep["obs"]["eye_in_hand_image"][obs_start:obs_end]
            if len(obs_data) < self.obs_horizon:
                pad = np.repeat(obs_data[:1], self.obs_horizon - len(obs_data), axis=0)
                obs_data = np.concatenate([pad, obs_data], axis=0)
            obs_data = obs_data.astype(np.float32) / 255.0
            obs_data = np.transpose(obs_data, (0, 3, 1, 2))
            result["obs_eye_in_hand_image"] = torch.from_numpy(obs_data)

        result["task_id"] = torch.tensor(task_idx, dtype=torch.long)

        if self.task_embeddings is not None:
            task_name = self.task_data[task_idx]["name"]
            # HDF5 filenames have _demo suffix; CLIP keys don't
            lookup_name = task_name[:-5] if task_name.endswith("_demo") else task_name
            if lookup_name in self.task_embeddings:
                result["task_emb"] = self.task_embeddings[lookup_name]

        return result


class TaskUniformSampler(Sampler):
    """
    Sampler that ensures uniform distribution across tasks.
    Each iteration:
      1. Pick a task uniformly at random
      2. Pick a random sample from that task
    """

    def __init__(self, dataset: LiberoUniformDataset, num_samples: int):
        self.dataset = dataset
        self.num_samples = num_samples
        self.n_tasks = len(dataset.task_data)
        self.task_indices = dataset.task_indices

    def __iter__(self):
        for _ in range(self.num_samples):
            task_idx = np.random.randint(0, self.n_tasks)
            indices = self.task_indices[task_idx]
            sample_idx = indices[np.random.randint(0, len(indices))]
            yield sample_idx

    def __len__(self):
        return self.num_samples


def create_dataloader(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
    obs_horizon: int = 2,
    action_horizon: int = 16,
    samples_per_epoch: Optional[int] = None,
    **dataset_kwargs,
) -> Tuple[DataLoader, LiberoUniformDataset]:
    dataset = LiberoUniformDataset(
        data_dir=data_dir,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        **dataset_kwargs,
    )

    sampler = None
    shuffle = True
    if samples_per_epoch is not None:
        sampler = TaskUniformSampler(dataset, num_samples=samples_per_epoch)
        shuffle = False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return loader, dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    loader, dataset = create_dataloader(args.data_dir, batch_size=args.batch_size)

    print(f"\nDataset stats:")
    print(f"  Tasks: {len(dataset.task_data)}")
    print(f"  Total samples: {len(dataset)}")
    print(
        f"  Samples per epoch: "
        f"{len(loader.sampler) if loader.sampler is not None else len(dataset)}"
    )

    batch = next(iter(loader))
    print(f"\nBatch sample:")
    for k, v in batch.items():
        print(f"  {k}: {v.shape} ({v.dtype})")

    task_counts = torch.zeros(len(dataset.task_data))
    for i, batch in enumerate(loader):
        task_ids = batch["task_id"]
        for tid in task_ids:
            task_counts[tid] += 1
        if i >= 20:
            break
    print(f"\nTask distribution (first 20 batches):")
    print(
        f"  min={task_counts.min():.0f}, max={task_counts.max():.0f}, "
        f"std={task_counts.std():.1f}, mean={task_counts.mean():.1f}"
    )
