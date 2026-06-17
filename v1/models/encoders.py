"""
models/encoders.py

Per-modality backbone encoders.
Backbones are frozen after pretraining; only LoRA + adapter layers are trained.

Modalities:
  - LiDAR  : PointPillars-style BEV encoder
  - Camera : ConvNeXt-style image encoder (6 surround cameras)
  - Radar  : Range-Doppler map CNN encoder
  - IMU    : MLP state encoder
  - GNSS   : MLP state encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from .lora import inject_lora, AdapterLayer


# ── Utility block ─────────────────────────────────────────────────────────────

class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


# ── LiDAR Encoder ─────────────────────────────────────────────────────────────

class PillarFeatureNet(nn.Module):
    """
    PointPillars-style pillar feature network.
    Input:  (N_pillars, max_points, in_channels)  +  coords (N_pillars, 4)
    Output: (B, out_channels, H/4, W/4)  BEV feature map
    """
    def __init__(self, in_channels=4, out_channels=256,
                 voxel_size=(0.2, 0.2, 8.0),
                 point_cloud_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)):
        super().__init__()
        import math
        self.out_channels = out_channels
        self.bev_h = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])
        self.bev_w = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])

        # Pillar feature extractor
        self.pfe = nn.Sequential(
            nn.Linear(in_channels + 6, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )
        # 2D backbone on BEV canvas
        self.backbone = nn.Sequential(
            ConvBNReLU(out_channels, 128, 3, 2, 1),
            ConvBNReLU(128, 128, 3, 1, 1),
            ConvBNReLU(128, 256, 3, 2, 1),
            ConvBNReLU(256, out_channels, 3, 1, 1),
        )

    def forward(self, voxels: torch.Tensor, coords: torch.Tensor,
                batch_size: int) -> torch.Tensor:
        N, P, C = voxels.shape
        non_empty = (voxels.abs().sum(-1) > 0).float().unsqueeze(-1)
        centroid  = (voxels * non_empty).sum(1) / (non_empty.sum(1) + 1e-6)
        offsets   = voxels - centroid.unsqueeze(1)
        x_aug = torch.cat([voxels, offsets, non_empty.expand(-1, -1, 2)], dim=-1)

        x = x_aug.reshape(N * P, -1)
        x = self.pfe(x)
        x = x.reshape(N, P, -1).max(dim=1).values   # (N, out_channels)

        bev = torch.zeros(batch_size, self.out_channels,
                          self.bev_h, self.bev_w,
                          device=voxels.device, dtype=voxels.dtype)
        bi  = coords[:, 0].long()
        yi  = coords[:, 2].long().clamp(0, self.bev_h - 1)
        xi  = coords[:, 3].long().clamp(0, self.bev_w - 1)
        bev[bi, :, yi, xi] = x

        return self.backbone(bev)


class LiDAREncoder(nn.Module):
    def __init__(self, cfg: dict, lora_cfg: dict, adapter_cfg: dict,
                 quantize: bool = True, n_bits: int = 8):
        super().__init__()
        out_ch = cfg.get("out_channels", 256)
        self.pillar_net = PillarFeatureNet(
            in_channels=cfg.get("in_channels", 4),
            out_channels=out_ch,
        )
        self.pool      = nn.AdaptiveAvgPool2d((8, 8))
        self.flat_proj = nn.Linear(out_ch * 64, out_ch)
        inject_lora(self.flat_proj, r=lora_cfg["ranks"].get("lidar", 8),
                    alpha=lora_cfg["alpha"], n_bits=n_bits, quantize=quantize)
        self.adapter = AdapterLayer(out_ch, adapter_cfg["bottleneck_dim"],
                                    n_bits=n_bits, quantize=quantize)

    def forward(self, voxels, coords, batch_size):
        bev = self.pillar_net(voxels, coords, batch_size)
        x   = self.pool(bev).flatten(1)
        x   = self.flat_proj(x)
        return self.adapter(x)


# ── Camera Encoder ────────────────────────────────────────────────────────────

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dw_conv = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)
        self.norm    = nn.LayerNorm(dim)
        self.pw1     = nn.Linear(dim, 4 * dim)
        self.pw2     = nn.Linear(4 * dim, dim)
        self.act     = nn.GELU()

    def forward(self, x):
        shortcut = x
        x = self.dw_conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pw2(self.act(self.pw1(x)))
        x = x.permute(0, 3, 1, 2).contiguous()
        return shortcut + x


class CameraEncoder(nn.Module):
    """Processes N_cam surround images, mean-pools across cameras."""
    def __init__(self, cfg: dict, lora_cfg: dict, adapter_cfg: dict,
                 quantize: bool = True, n_bits: int = 8):
        super().__init__()
        out_ch = cfg.get("out_channels", 256)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 96, 4, 4),
            nn.GroupNorm(1, 96),
        )
        self.stages = nn.Sequential(
            *[ConvNeXtBlock(96) for _ in range(3)],
            nn.Conv2d(96, out_ch, 1),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        inject_lora(self.stages, r=lora_cfg["ranks"].get("camera", 8),
                    alpha=lora_cfg["alpha"], n_bits=n_bits, quantize=quantize)
        self.adapter = AdapterLayer(out_ch, adapter_cfg["bottleneck_dim"],
                                    n_bits=n_bits, quantize=quantize)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, N_cam, 3, H, W)"""
        B, N, C, H, W = images.shape
        x = images.reshape(B * N, C, H, W)
        x = self.stages(self.stem(x))
        x = self.pool(x).flatten(1)          # (B*N, out_ch)
        x = x.reshape(B, N, -1).mean(1)      # mean over cameras → (B, out_ch)
        return self.adapter(x)


# ── Radar Encoder ─────────────────────────────────────────────────────────────

class RadarEncoder(nn.Module):
    """Encodes a 2-channel Range-Doppler map via a small CNN."""
    def __init__(self, cfg: dict, lora_cfg: dict, adapter_cfg: dict,
                 quantize: bool = True, n_bits: int = 8):
        super().__init__()
        out_ch = cfg.get("out_channels", 128)
        in_ch  = cfg.get("in_channels", 2)
        self.cnn = nn.Sequential(
            ConvBNReLU(in_ch, 32, 3, 2, 1),
            ConvBNReLU(32, 64, 3, 2, 1),
            ConvBNReLU(64, out_ch, 3, 2, 1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(out_ch, out_ch)
        inject_lora(self.proj, r=lora_cfg["ranks"].get("radar", 4),
                    alpha=lora_cfg["alpha"], n_bits=n_bits, quantize=quantize)
        self.adapter = AdapterLayer(out_ch, max(adapter_cfg["bottleneck_dim"] // 2, 16),
                                    n_bits=n_bits, quantize=quantize)

    def forward(self, rd_map: torch.Tensor) -> torch.Tensor:
        """rd_map: (B, 2, H, W)"""
        x = self.cnn(rd_map).flatten(1)
        return self.adapter(self.proj(x))


# ── IMU / GNSS Encoders ───────────────────────────────────────────────────────

class MLPEncoder(nn.Module):
    def __init__(self, in_dim, out_ch, hidden=128,
                 lora_r=4, lora_alpha=16,
                 quantize=True, n_bits=8, bottleneck_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, out_ch),
        )
        inject_lora(self.net, r=lora_r, alpha=lora_alpha,
                    n_bits=n_bits, quantize=quantize)
        self.adapter = AdapterLayer(out_ch, bottleneck_dim,
                                    n_bits=n_bits, quantize=quantize)

    def forward(self, x): return self.adapter(self.net(x))


class IMUEncoder(nn.Module):
    def __init__(self, cfg, lora_cfg, adapter_cfg, quantize=True, n_bits=8):
        super().__init__()
        out_ch = cfg.get("out_channels", 64)
        self.enc = MLPEncoder(
            in_dim=cfg.get("in_dim", 6), out_ch=out_ch,
            lora_r=lora_cfg["ranks"].get("imu", 4),
            lora_alpha=lora_cfg["alpha"], quantize=quantize, n_bits=n_bits,
            bottleneck_dim=max(adapter_cfg["bottleneck_dim"] // 4, 8),
        )
    def forward(self, x): return self.enc(x)


class GNSSEncoder(nn.Module):
    def __init__(self, cfg, lora_cfg, adapter_cfg, quantize=True, n_bits=8):
        super().__init__()
        out_ch = cfg.get("out_channels", 64)
        self.enc = MLPEncoder(
            in_dim=cfg.get("in_dim", 4), out_ch=out_ch,
            lora_r=lora_cfg["ranks"].get("gnss", 4),
            lora_alpha=lora_cfg["alpha"], quantize=quantize, n_bits=n_bits,
            bottleneck_dim=max(adapter_cfg["bottleneck_dim"] // 4, 8),
        )
    def forward(self, x): return self.enc(x)


# ── Output dims per modality (used by projection heads) ───────────────────────

MODALITY_DIMS = {
    "lidar":  256,
    "camera": 256,
    "radar":  128,
    "imu":    64,
    "gnss":   64,
}


def build_encoders(cfg: dict, quantize: bool = True,
                   n_bits: int = 8) -> nn.ModuleDict:
    enc_cfg  = cfg["model"]["encoders"]
    lora_cfg = cfg["model"]["lora"]
    adap_cfg = cfg["model"]["adapters"]
    return nn.ModuleDict({
        "lidar":  LiDAREncoder(enc_cfg["lidar"],  lora_cfg, adap_cfg, quantize, n_bits),
        "camera": CameraEncoder(enc_cfg["camera"], lora_cfg, adap_cfg, quantize, n_bits),
        "radar":  RadarEncoder(enc_cfg["radar"],  lora_cfg, adap_cfg, quantize, n_bits),
        "imu":    IMUEncoder(enc_cfg["imu"],    lora_cfg, adap_cfg, quantize, n_bits),
        "gnss":   GNSSEncoder(enc_cfg["gnss"],   lora_cfg, adap_cfg, quantize, n_bits),
    })
