"""Tensor utilities shared by the point-cloud models."""

import torch

from pointnet2 import pointnet2_utils


def get_dist(src, dst):
    """Return pairwise squared Euclidean distances for BxMx3 and BxNx3."""

    dist = -2 * torch.matmul(src, dst.transpose(1, 2))
    dist += torch.sum(src**2, dim=-1, keepdim=True)
    dist += torch.sum(dst**2, dim=-1).unsqueeze(1)
    return dist


def index_points(points, idx):
    """Gather BxM point indices from a BxNxC tensor."""

    return pointnet2_utils.gather_operation(
        points.transpose(1, 2).contiguous(), idx.to(torch.int32)
    ).transpose(1, 2).contiguous()


def sample_and_group(radius, k, xyz, feat, centroid, dist):
    """Gather the k nearest valid features within each squared radius."""

    if xyz.shape[1] < k:
        raise ValueError(f"Grouping requires at least {k} points")
    _, idx = torch.topk(dist, k, dim=-1, largest=False)
    gathered_dist = torch.gather(dist, -1, idx)
    valid = gathered_dist <= radius
    first = idx[:, :, :1].expand(-1, -1, k)
    idx = torch.where(valid, idx, first).to(torch.int32).contiguous()
    grouped = pointnet2_utils.grouping_operation(
        feat.transpose(1, 2).contiguous(), idx
    )
    return grouped.transpose(1, 2).transpose(-1, -2).contiguous(), idx
