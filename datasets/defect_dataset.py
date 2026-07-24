"""Dataset loaders for annotated masonry point-cloud patches."""

from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


NUM_CLASSES = 2


def _load_patch(path):
    data = np.load(path)
    if data.ndim != 2 or data.shape[1] != 7:
        raise ValueError(
            f"{path} must have shape (N, 7) with XYZRGB and label columns; "
            f"found {data.shape}"
        )
    if data.shape[0] == 0:
        raise ValueError(f"{path} is empty")
    labels = data[:, 6]
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError(f"{path} contains labels other than 0 and 1")
    return data.astype(np.float32, copy=True)


def _split_files(data_root, split):
    split_dir = Path(data_root).expanduser().resolve() / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Dataset split directory not found: {split_dir}")
    files = sorted(split_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy patches found in {split_dir}")
    return files


def _class_weights(labels):
    counts = np.bincount(labels.astype(np.int64), minlength=NUM_CLASSES)
    if np.any(counts == 0):
        raise ValueError(f"Both labels must occur in the split; counts={counts.tolist()}")
    proportions = counts.astype(np.float64) / counts.sum()
    return np.power(proportions.max() / proportions, 1.0 / 3.0).astype(
        np.float32
    )


class DefectDataset(Dataset):
    """Random 0.5 m block sampler used during training and validation."""

    def __init__(
        self,
        split,
        data_root,
        num_point=4096,
        block_size=0.5,
        sample_rate=1.0,
    ):
        if split not in {"train", "val"}:
            raise ValueError("DefectDataset split must be 'train' or 'val'")
        self.split = split
        self.num_point = int(num_point)
        self.block_size = float(block_size)
        self.files = _split_files(data_root, split)
        self.points = []
        self.labels = []
        self.coord_max = []
        point_counts = []
        all_labels = []

        for path in self.files:
            data = _load_patch(path)
            data[:, :3] -= data[:, :3].mean(axis=0, keepdims=True)
            points = data[:, :6]
            labels = data[:, 6].astype(np.int64)
            self.points.append(points)
            self.labels.append(labels)
            self.coord_max.append(np.max(points[:, :3], axis=0))
            point_counts.append(labels.size)
            all_labels.append(labels)

        self.labelweights = _class_weights(np.concatenate(all_labels))
        sample_probabilities = np.asarray(point_counts) / np.sum(point_counts)
        iterations = int(np.sum(point_counts) * sample_rate / self.num_point)
        patch_indices = []
        for index, probability in enumerate(sample_probabilities):
            patch_indices.extend([index] * int(round(probability * iterations)))
        self.patch_indices = np.asarray(patch_indices, dtype=np.int64)

    def __len__(self):
        return len(self.patch_indices)

    def __getitem__(self, index):
        patch_index = self.patch_indices[index]
        points = self.points[patch_index]
        labels = self.labels[patch_index]
        point_count = points.shape[0]

        center_probabilities = None
        if self.split == "train" and np.any(labels == 1):
            center_probabilities = np.ones(point_count, dtype=np.float64)
            center_probabilities[labels == 1] *= 3.0
            center_probabilities /= center_probabilities.sum()

        point_indices = np.arange(point_count)
        for _ in range(100):
            center_index = np.random.choice(
                point_count, p=center_probabilities
            )
            center = points[center_index, :3]
            half = self.block_size / 2.0
            point_indices = np.where(
                (points[:, 0] >= center[0] - half)
                & (points[:, 0] <= center[0] + half)
                & (points[:, 1] >= center[1] - half)
                & (points[:, 1] <= center[1] + half)
            )[0]
            if point_indices.size > 1024:
                break
        else:
            center = points[np.random.choice(point_count), :3]
            point_indices = np.arange(point_count)

        selected_indices = np.random.choice(
            point_indices,
            self.num_point,
            replace=point_indices.size < self.num_point,
        )
        selected = points[selected_indices].copy()
        features = np.zeros((self.num_point, 9), dtype=np.float32)
        denominator = self.coord_max[patch_index].copy()
        denominator[np.abs(denominator) < 1e-12] = 1.0
        features[:, 6:9] = selected[:, :3] / denominator
        selected[:, 0] -= center[0]
        selected[:, 1] -= center[1]
        features[:, :6] = selected
        selected_labels = labels[selected_indices]
        sample_weights = self.labelweights[selected_labels]
        return features, selected_labels, sample_weights


class DefectDatasetWholeScene:
    """Sliding-window loader that returns every point in a test patch."""

    def __init__(
        self,
        data_root,
        split="test",
        block_points=4096,
        stride=0.5,
        block_size=0.5,
        padding=0.001,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        self.files = _split_files(data_root, split)
        self.block_points = int(block_points)
        self.stride = float(stride)
        self.block_size = float(block_size)
        self.padding = float(padding)
        self.scene_points_list = []
        self.semantic_labels_list = []
        all_labels = []

        for path in self.files:
            data = _load_patch(path)
            data[:, :3] -= data[:, :3].mean(axis=0, keepdims=True)
            self.scene_points_list.append(data[:, :6])
            labels = data[:, 6].astype(np.int64)
            self.semantic_labels_list.append(labels)
            all_labels.append(labels)
        self.labelweights = _class_weights(np.concatenate(all_labels))

    @property
    def file_list(self):
        return [path.name for path in self.files]

    def __len__(self):
        return len(self.scene_points_list)

    def __getitem__(self, index):
        points = self.scene_points_list[index]
        labels = self.semantic_labels_list[index]
        coord_min = np.min(points[:, :3], axis=0)
        coord_max = np.max(points[:, :3], axis=0)
        grid_x = int(
            np.ceil((coord_max[0] - coord_min[0] - self.block_size) / self.stride)
            + 1
        )
        grid_y = int(
            np.ceil((coord_max[1] - coord_min[1] - self.block_size) / self.stride)
            + 1
        )
        denominator = coord_max.copy()
        denominator[np.abs(denominator) < 1e-12] = 1.0

        data_blocks = []
        label_blocks = []
        weight_blocks = []
        index_blocks = []
        for index_y in range(max(grid_y, 1)):
            for index_x in range(max(grid_x, 1)):
                start_x = coord_min[0] + index_x * self.stride
                end_x = min(start_x + self.block_size, coord_max[0])
                start_x = end_x - self.block_size
                start_y = coord_min[1] + index_y * self.stride
                end_y = min(start_y + self.block_size, coord_max[1])
                start_y = end_y - self.block_size
                point_indices = np.where(
                    (points[:, 0] >= start_x - self.padding)
                    & (points[:, 0] <= end_x + self.padding)
                    & (points[:, 1] >= start_y - self.padding)
                    & (points[:, 1] <= end_y + self.padding)
                )[0]
                if point_indices.size == 0:
                    continue

                batch_count = int(np.ceil(point_indices.size / self.block_points))
                padded_size = batch_count * self.block_points
                missing = padded_size - point_indices.size
                if missing:
                    repeated = np.random.choice(
                        point_indices,
                        missing,
                        replace=missing > point_indices.size,
                    )
                    point_indices = np.concatenate([point_indices, repeated])
                np.random.shuffle(point_indices)
                block = points[point_indices].copy()
                global_xyz = block[:, :3] / denominator
                block[:, 0] -= start_x + self.block_size / 2.0
                block[:, 1] -= start_y + self.block_size / 2.0
                features = np.concatenate([block, global_xyz], axis=1)
                block_labels = labels[point_indices]

                data_blocks.append(features.reshape(-1, self.block_points, 9))
                label_blocks.append(
                    block_labels.reshape(-1, self.block_points)
                )
                weight_blocks.append(
                    self.labelweights[block_labels].reshape(
                        -1, self.block_points
                    )
                )
                index_blocks.append(
                    point_indices.reshape(-1, self.block_points)
                )

        if not data_blocks:
            raise RuntimeError(f"No spatial blocks could be formed for {self.files[index]}")
        return (
            np.concatenate(data_blocks, axis=0),
            np.concatenate(label_blocks, axis=0),
            np.concatenate(weight_blocks, axis=0),
            np.concatenate(index_blocks, axis=0),
        )
