"""Neural-network building blocks used by the NSA models."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet2 import pointnet2_utils

from .utils import get_dist, index_points, sample_and_group


class Conv1x1(nn.Module):
    """A 1 x 1 Conv1d followed by batch normalization and activation."""

    def __init__(self, in_channels, out_channels, act=None, bias_=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias_),
            nn.BatchNorm1d(out_channels),
        )
        self.act = nn.GELU() if act is None else act
        nn.init.xavier_normal_(self.conv[0].weight.data)

    def forward(self, x):
        x = self.conv(x.transpose(1, 2).contiguous())
        x = x.transpose(1, 2).contiguous()
        return self.act(x) if self.act is not False else x


class PositionEncoder(nn.Module):
    """Encode absolute and relative geometry around each sampled point."""

    def __init__(self, out_channel, radius, k=20):
        super().__init__()
        self.k = k
        self.xyz2feature = nn.Sequential(
            nn.Conv2d(19, out_channel // 8, kernel_size=1),
            nn.BatchNorm2d(out_channel // 8),
            nn.GELU(),
        )
        self.mlp = nn.Sequential(
            Conv1x1(out_channel // 8, out_channel // 4),
            Conv1x1(out_channel // 4, out_channel, act=False),
        )
        # Kept for checkpoint compatibility with the released model.
        self.qg = pointnet2_utils.QueryAndGroup(radius, self.k)

    @staticmethod
    def _knn_indices(xyz, k):
        if xyz.shape[1] < k:
            raise ValueError(f"Position encoding requires at least {k} points")
        distances = torch.cdist(xyz, xyz, p=2)
        return distances.argsort(dim=-1)[:, :, :k]

    def _relative_position_encoding(self, xyz):
        batch_size, point_count, _ = xyz.shape
        neighbor_idx = self._knn_indices(xyz, self.k)
        neighbor_xyz = torch.gather(
            xyz.unsqueeze(1).expand(batch_size, point_count, point_count, 3),
            2,
            neighbor_idx.unsqueeze(-1).expand(
                batch_size, point_count, self.k, 3
            ),
        )
        center_xyz = xyz.unsqueeze(2).expand(-1, -1, self.k, -1)
        relative_xyz = center_xyz - neighbor_xyz
        relative_distance = torch.linalg.norm(
            relative_xyz, dim=-1, keepdim=True
        )
        return torch.cat(
            [relative_distance, relative_xyz, center_xyz, neighbor_xyz], dim=-1
        )

    def forward(self, centroid, xyz, radius, dist):
        neighbor_xyz, _ = sample_and_group(
            radius, self.k, xyz, xyz, centroid, dist
        )
        center_xyz = centroid.unsqueeze(2).expand(-1, -1, self.k, -1)
        variation = neighbor_xyz - center_xyz
        relative_features = self._relative_position_encoding(centroid)
        features = torch.cat(
            [center_xyz, neighbor_xyz, variation, relative_features], dim=-1
        )
        features = self.xyz2feature(features.permute(0, 3, 1, 2).contiguous())
        features = torch.max(features, dim=-1)[0].transpose(1, 2)
        return self.mlp(features)


class MaskedAttention(nn.Module):
    """Masked local attention with balanced renormalization."""

    def __init__(self, in_channels, hid_channels=128):
        super().__init__()
        hid_channels = hid_channels or 1
        self.conv_q = Conv1x1(in_channels + 3, hid_channels, act=False)
        self.conv_k = Conv1x1(in_channels + 3, hid_channels, act=False)

    def forward(self, cent_feat, feat, mask):
        query = self.conv_q(cent_feat)
        key = self.conv_k(feat)
        scores = torch.bmm(query, key.transpose(1, 2))
        scores = scores.masked_fill(mask < 1e-9, -1e9)
        attention = torch.softmax(scores, dim=-1)

        attention = (
            torch.sqrt(mask + 1e-9) * torch.sqrt(attention + 1e-9) - 1e-9
        )
        attention = F.normalize(attention, p=1, dim=1)
        return F.normalize(attention, p=1, dim=-1)


def farthest_point_sample(xyz, npoint):
    """Farthest-point sampling implemented in PyTorch."""

    device = xyz.device
    batch_size, point_count, _ = xyz.shape
    centroids = torch.zeros(
        batch_size, npoint, dtype=torch.long, device=device
    )
    distance = torch.full(
        (batch_size, point_count), 1e10, dtype=xyz.dtype, device=device
    )
    farthest = torch.randint(0, point_count, (batch_size,), device=device)
    batch_indices = torch.arange(batch_size, device=device)
    for index in range(npoint):
        centroids[:, index] = farthest
        centroid = xyz[batch_indices, farthest, :].view(batch_size, 1, 3)
        current_distance = torch.sum((xyz - centroid) ** 2, dim=-1)
        update = current_distance < distance
        distance[update] = current_distance[update]
        farthest = torch.max(distance, dim=-1)[1]
    return centroids


def dilated_ball_query(dist, bandwidth, base_radius, max_radius):
    """Estimate a density-adaptive squared radius for each key point."""

    gaussian = torch.exp(-dist / (2 * bandwidth**2))
    kernel_density = torch.sum(gaussian, dim=-1, keepdim=True)
    density_score = kernel_density / (
        torch.max(kernel_density, dim=1, keepdim=True)[0] + 1e-9
    )
    return base_radius + (max_radius - base_radius) * density_score


class FeatureExtractionModule(nn.Module):
    """Novel Set Abstraction (NSA) feature-extraction module."""

    def __init__(self, in_channels, out_channels, base_radius, bottleneck=4):
        super().__init__()
        self.conv_v = Conv1x1(2 * in_channels, out_channels, act=False)
        self.mat = MaskedAttention(in_channels, in_channels // bottleneck)
        self.pos_conv = PositionEncoder(out_channels, np.sqrt(base_radius))
        self.base_radius = base_radius
        self.k = 20

    def forward(self, x, xyz, cent_num):
        _, point_count, _ = xyz.shape
        if cent_num < point_count:
            idx = farthest_point_sample(xyz, cent_num)
            centroid = index_points(xyz, idx)
            cent_feat = index_points(x, idx)
        else:
            centroid = xyz.clone()
            cent_feat = x.clone()

        dist = get_dist(centroid, xyz)
        radius = dilated_ball_query(
            dist,
            bandwidth=0.1,
            base_radius=self.base_radius,
            max_radius=self.base_radius * 2,
        )
        mask = (dist < radius).float()

        embedded_centroid = torch.cat([cent_feat, centroid], dim=-1)
        embedded_points = torch.cat([x, xyz], dim=-1)
        adjacency = self.mat(embedded_centroid, embedded_points, mask)
        smoothed = torch.bmm(adjacency, x)

        variation = smoothed - cent_feat
        output = self.conv_v(torch.cat([variation, cent_feat], dim=-1))
        output = output + self.pos_conv(centroid, xyz, radius, dist)
        return F.gelu(output), centroid


class AttentionBlock(nn.Module):
    """Attention gate used during feature propagation."""

    def __init__(self, gate_channels, skip_channels, inner_channels):
        super().__init__()
        # Attribute names are retained for compatibility with released weights.
        self.W_g = nn.Sequential(
            nn.Conv1d(gate_channels, inner_channels, 1, bias=True),
            nn.BatchNorm1d(inner_channels),
        )
        self.W_x = nn.Sequential(
            nn.Conv1d(skip_channels, inner_channels, 1, bias=True),
            nn.BatchNorm1d(inner_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv1d(inner_channels, 1, 1, bias=True),
            nn.BatchNorm1d(1),
            nn.Sigmoid(),
        )

    def forward(self, gate, skip):
        gate = self.W_g(gate.transpose(1, 2))
        skip = self.W_x(skip.transpose(1, 2))
        return self.psi(F.gelu(gate + skip)).transpose(1, 2)


class PointFeaturePropagation(nn.Module):
    """Three-neighbor interpolation followed by gated feature fusion."""

    def __init__(self, in_channel1, in_channel2, out_channel):
        super().__init__()
        in_channel = in_channel1 + in_channel2
        self.conv = nn.Sequential(
            Conv1x1(in_channel, in_channel // 2),
            Conv1x1(in_channel // 2, in_channel // 2),
            Conv1x1(in_channel // 2, out_channel),
        )
        self.att = AttentionBlock(in_channel1, in_channel2, in_channel2)

    def forward(self, xyz1, xyz2, feat1, feat2):
        distances, indices = pointnet2_utils.three_nn(xyz1, xyz2)
        reciprocal = 1.0 / (distances + 1e-8)
        weights = reciprocal / torch.sum(reciprocal, dim=-1, keepdim=True)
        interpolated = pointnet2_utils.three_interpolate(
            feat2.transpose(1, 2).contiguous(), indices, weights
        ).transpose(1, 2)

        if feat1 is not None:
            feat1 = feat1 * self.att(interpolated, feat1)
        fused = (
            torch.cat([feat1, interpolated], dim=-1)
            if feat1 is not None
            else interpolated
        )
        return self.conv(fused)
