"""
models/projection.py

Projection heads: map each modality's encoder output into a
normalized d-dimensional shared embedding space.

  - Outputs are L2-normalized → cosine metric
  - Quantization-aware: projection weights carry fake quantizers
  - Different input dims per modality → unified embed_dim
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .lora import FakeQuantize
from .encoders import MODALITY_DIMS


class QuantizedLinear(nn.Linear):
    """nn.Linear with QAT fake-quantized weight forward."""
    def __init__(self, in_features, out_features, bias=True,
                 n_bits=8, quantize=True):
        super().__init__(in_features, out_features, bias)
        if quantize:
            self.quant_w = FakeQuantize(n_bits=n_bits, symmetric=True,
                                        per_channel=True, ch_axis=0)
            self.quant_a = FakeQuantize(n_bits=n_bits, symmetric=True)
        else:
            self.quant_w = nn.Identity()
            self.quant_a = nn.Identity()

    def forward(self, x):
        w = self.quant_w(self.weight)
        x = self.quant_a(x)
        return F.linear(x, w, self.bias)


class ProjectionHead(nn.Module):
    """
    Two-layer MLP: in_dim → hidden_dim → out_dim, with L2 normalization.
    """
    def __init__(self, in_dim, hidden_dim, out_dim,
                 normalize=True, n_bits=8, quantize=True):
        super().__init__()
        self.normalize = normalize
        self.norm_in  = nn.LayerNorm(in_dim)
        self.fc1      = QuantizedLinear(in_dim, hidden_dim, n_bits=n_bits, quantize=quantize)
        self.act      = nn.GELU()
        self.norm_mid = nn.LayerNorm(hidden_dim)
        self.fc2      = QuantizedLinear(hidden_dim, out_dim, n_bits=n_bits, quantize=quantize)
        nn.init.trunc_normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.trunc_normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.norm_in(x)
        x = self.act(self.fc1(x))
        x = self.norm_mid(x)
        x = self.fc2(x)
        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)
        return x


class MultiModalProjector(nn.Module):
    """One ProjectionHead per modality. Missing modalities return None."""
    def __init__(self, embed_dim, hidden_dim, normalize=True,
                 n_bits=8, quantize=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = nn.ModuleDict({
            mod: ProjectionHead(
                in_dim=dim, hidden_dim=hidden_dim, out_dim=embed_dim,
                normalize=normalize, n_bits=n_bits, quantize=quantize,
            )
            for mod, dim in MODALITY_DIMS.items()
        })

    def forward(self, features: dict) -> dict:
        return {
            mod: (self.heads[mod](feat) if feat is not None else None)
            for mod, feat in features.items()
        }
