"""
models/q_peft_dml.py

Full Q-PEFT-DML model assembly.

Forward pass:
  raw sensor inputs
      → per-modality encoders (frozen backbone + LoRA/adapter, quantized)
      → projection heads (shared latent space)
      → cross-attention fusion with gating
      → detection head (3D boxes + classes)

Also returns intermediate embeddings needed for loss computation:
  - per-modality projected embeddings (for metric + geometry loss)
  - fused embedding (for consistency loss)
  - full-precision embeddings (for geometry preservation loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from .encoders import build_encoders
from .projection import MultiModalProjector
from .fusion import ModalityFusion
from .detection_head import DetectionHead
from .lora import set_qat_enabled, freeze_backbone


class QPEFTDMLModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        model_cfg = cfg["model"]
        quant_cfg = cfg["quantization"]

        embed_dim  = model_cfg["embed_dim"]
        n_bits     = quant_cfg["bits"].get("lora", 8)
        quantize   = quant_cfg["enabled"] and quant_cfg["mode"] == "qat"

        # ── Encoders (backbone frozen; LoRA + adapter trained) ──────────────
        self.encoders = build_encoders(cfg, quantize=quantize, n_bits=n_bits)

        # ── Projection heads → shared latent space ──────────────────────────
        proj_cfg = model_cfg["projection"]
        self.projector = MultiModalProjector(
            embed_dim=embed_dim,
            hidden_dim=proj_cfg["hidden_dim"],
            normalize=proj_cfg["normalize"],
            n_bits=quant_cfg["bits"].get("projection", 8),
            quantize=quantize,
        )

        # ── Cross-attention fusion with drop-modality gating ─────────────────
        fus_cfg = model_cfg["fusion"]
        self.fusion = ModalityFusion(
            embed_dim=embed_dim,
            num_heads=fus_cfg["num_heads"],
            num_layers=fus_cfg["num_layers"],
            dropout=fus_cfg["dropout"],
            drop_modality_prob=fus_cfg["drop_modality_prob"],
            n_bits=quant_cfg["bits"].get("fusion", 8),
            quantize=quantize,
        )

        # ── 3D Detection head ────────────────────────────────────────────────
        det_cfg = model_cfg["detection"]
        self.detection_head = DetectionHead(
            embed_dim=embed_dim,
            num_classes=det_cfg["num_classes"],
            n_bits=n_bits,
            quantize=quantize,
        )

        # Freeze backbone, only train PEFT modules
        freeze_backbone(self)
        self._qat_enabled = False

    # ── QAT control ──────────────────────────────────────────────────────────

    def enable_qat(self):
        """Enable fake quantizers (called after QAT warmup steps)."""
        set_qat_enabled(self, True)
        self._qat_enabled = True

    def disable_qat(self):
        """Disable fake quantizers (full-precision mode)."""
        set_qat_enabled(self, False)
        self._qat_enabled = False

    # ── Forward ──────────────────────────────────────────────────────────────

    def encode_modalities(self, batch: dict) -> Dict[str, Optional[torch.Tensor]]:
        """
        Run per-modality encoders. Returns raw encoder features (pre-projection).
        Missing modalities (not in batch or None) → None in output dict.
        """
        features = {}
        for mod, encoder in self.encoders.items():
            data = batch.get(mod)
            if data is None:
                features[mod] = None
                continue
            try:
                if mod == "lidar":
                    features[mod] = encoder(
                        data["voxels"], data["coords"], data["batch_size"])
                elif mod == "camera":
                    features[mod] = encoder(data)         # (B, N_cam, C, H, W)
                else:
                    features[mod] = encoder(data)         # (B, in_dim) or (B, C, H, W)
            except Exception:
                features[mod] = None
        return features

    def forward(self, batch: dict,
                forced_missing: Optional[list] = None) -> dict:
        """
        batch: dict with keys 'lidar', 'camera', 'radar', 'imu', 'gnss'
               (any subset may be absent or None → treated as missing sensor)
        forced_missing: list of modality names to force-mask (eval dropout)

        Returns:
          cls_logits      : (B, num_classes)
          box_preds       : (B, 10)
          embeddings      : dict[mod → (B, D) | None]  — projected embeddings
          fused_embedding : (B, D)                      — fused latent vector
          modality_tokens : (B, N_mod, D)               — for geometry loss
          avail_mask      : (B, N_mod) bool             — True = missing
        """
        # 1. Encode each modality
        features = self.encode_modalities(batch)

        # 2. Project into shared latent space
        embeddings = self.projector(features)

        # 3. Build forced mask for eval-time sensor dropout
        forced_mask = None
        if forced_missing:
            from .fusion import MODALITY_ORDER
            B = next(e for e in embeddings.values() if e is not None).shape[0]
            device = next(e for e in embeddings.values() if e is not None).device
            forced_mask = torch.zeros(B, len(MODALITY_ORDER),
                                      dtype=torch.bool, device=device)
            for i, mod in enumerate(MODALITY_ORDER):
                if mod in forced_missing:
                    forced_mask[:, i] = True

        # 4. Cross-attention fusion
        fused, modality_tokens, avail_mask = self.fusion(
            embeddings, forced_mask=forced_mask)

        # 5. Detection
        det_out = self.detection_head(fused)

        return {
            "cls_logits":      det_out["cls_logits"],
            "box_preds":       det_out["box_preds"],
            "embeddings":      embeddings,
            "fused_embedding": fused,
            "modality_tokens": modality_tokens,
            "avail_mask":      avail_mask,
        }

    def count_trainable_params(self) -> dict:
        """Report trainable vs total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "pct_trainable": 100.0 * trainable / total,
        }
