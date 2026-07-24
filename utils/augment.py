"""Point-cloud augmentations used for the released experiment."""

import numpy as np


def rotate_z(batch_xyz):
    output = np.zeros_like(batch_xyz, dtype=np.float32)
    for batch_index in range(batch_xyz.shape[0]):
        angle = np.random.uniform() * 2 * np.pi
        cosine, sine = np.cos(angle), np.sin(angle)
        rotation = np.array(
            [[cosine, sine, 0], [-sine, cosine, 0], [0, 0, 1]],
            dtype=np.float32,
        )
        output[batch_index] = batch_xyz[batch_index] @ rotation
    return output


def jitter(batch_xyz, sigma=0.01, clip=0.05):
    noise = np.clip(
        sigma * np.random.randn(*batch_xyz.shape), -clip, clip
    )
    return batch_xyz + noise


def random_scale(batch_xyz, low=0.8, high=1.25):
    scales = np.random.uniform(low, high, batch_xyz.shape[0])
    return batch_xyz * scales[:, None, None]


def random_shift(batch_xyz, shift_range=0.1):
    shifts = np.random.uniform(
        -shift_range, shift_range, (batch_xyz.shape[0], 3)
    )
    return batch_xyz + shifts[:, None, :]
