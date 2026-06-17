"""
models/detection_head.py

3D object detection head.
Takes the fused embedding and predicts:
  - Class probabilities  : (B, num_classes)
  - 3D bounding box      : (B, 10) → [x, y, z, log_w, log_l, log_h, sin_yaw, cos_yaw, vx, vy]
"""

import torch
import torch.nn as nn
from .projection import QuantizedLinear


class DetectionHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int,
                 n_bits: int = 8, quantize: bool = True):
        super().__init__()
        self.num_classes = num_classes

        self.shared = nn.Sequential(
            QuantizedLinear(embed_dim, 512, n_bits=n_bits, quantize=quantize),
            nn.LayerNorm(512),
            nn.GELU(),
            QuantizedLinear(512, 256, n_bits=n_bits, quantize=quantize),
            nn.LayerNorm(256),
            nn.GELU(),
        )
        self.cls_head = QuantizedLinear(256, num_classes, n_bits=n_bits, quantize=quantize)
        # [x, y, z, log_w, log_l, log_h, sin_yaw, cos_yaw, vx, vy]
        self.box_head = QuantizedLinear(256, 10, n_bits=n_bits, quantize=quantize)

        # Focal loss bias init: prior P = 0.01
        import math
        bias_val = -math.log((1 - 0.01) / 0.01)
        nn.init.constant_(self.cls_head.bias, bias_val)
        nn.init.zeros_(self.box_head.bias)

    def forward(self, fused: torch.Tensor) -> dict:
        feat = self.shared(fused)
        return {
            "cls_logits": self.cls_head(feat),
            "box_preds":  self.box_head(feat),
        }

    def decode_boxes(self, box_preds: torch.Tensor,
                     anchor: torch.Tensor) -> torch.Tensor:
        import torch
        x  = box_preds[:, 0] + anchor[0]
        y  = box_preds[:, 1] + anchor[1]
        z  = box_preds[:, 2] + anchor[2]
        w  = torch.exp(box_preds[:, 3]) * anchor[3]
        l  = torch.exp(box_preds[:, 4]) * anchor[4]
        h  = torch.exp(box_preds[:, 5]) * anchor[5]
        return torch.stack([x, y, z, w, l, h,
                            box_preds[:, 6], box_preds[:, 7],
                            box_preds[:, 8], box_preds[:, 9]], dim=-1)
