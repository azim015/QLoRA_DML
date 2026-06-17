"""
models/lora.py

LoRA (Low-Rank Adaptation) and Adapter layers with integrated
Quantization-Aware Training (QAT) via Straight-Through Estimators (STE).

Key design:
  - LoRA: W' = W_frozen + (B @ A) * (alpha / r)
  - Adapter: x' = x + adapter_up(act(adapter_down(x)))
  - Both A, B, adapter weights are passed through fake quantizers during training
  - STE allows gradients to flow through the rounding operation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ── Straight-Through Estimator Quantizer ──────────────────────────────────────

class STEQuantizer(torch.autograd.Function):
    """
    Fake quantizer with Straight-Through Estimator gradient.
    Forward:  rounds weights to n_bits integers
    Backward: passes gradients straight through (identity)
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: torch.Tensor,
                zero_point: torch.Tensor, n_bits: int) -> torch.Tensor:
        q_min = -(2 ** (n_bits - 1))
        q_max =  (2 ** (n_bits - 1)) - 1
        # Reshape scale/zero_point for broadcasting (per-channel along dim 0)
        if scale.dim() > 0 and scale.numel() > 1:
            shape = [-1] + [1] * (x.dim() - 1)
            scale      = scale.view(shape)
            zero_point = zero_point.view(shape)
        x_q = torch.clamp(torch.round(x / scale + zero_point), q_min, q_max)
        x_dq = (x_q - zero_point) * scale      # dequantize back to float
        return x_dq

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through: gradient passes unchanged, no grad for scale/zp/bits
        return grad_output, None, None, None


class FakeQuantize(nn.Module):
    """
    Learnable fake quantizer module.
    Maintains a running scale and zero_point estimated from observed activations.
    """
    def __init__(self, n_bits: int = 8, symmetric: bool = True,
                 per_channel: bool = False, ch_axis: int = 0,
                 observer: str = "minmax", percentile: float = 99.9):
        super().__init__()
        self.n_bits = n_bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.ch_axis = ch_axis
        self.observer = observer
        self.percentile = percentile
        self.enabled = True          # toggled off outside QAT warmup

        # Buffers updated by observer (not trained by gradient)
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("zero_point", torch.tensor(0.0))
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))

    @torch.no_grad()
    def _update_stats(self, x: torch.Tensor):
        """Update running min/max statistics from a batch."""
        if self.per_channel:
            # Reduce over all dims except channel axis
            dims = list(range(x.dim()))
            dims.pop(self.ch_axis)
            if self.observer == "percentile":
                x_flat = x.transpose(0, self.ch_axis).reshape(x.size(self.ch_axis), -1)
                min_v = torch.quantile(x_flat, (100 - self.percentile) / 100, dim=1)
                max_v = torch.quantile(x_flat, self.percentile / 100, dim=1)
            else:
                min_v = x.amin(dim=dims)
                max_v = x.amax(dim=dims)
        else:
            if self.observer == "percentile":
                min_v = torch.quantile(x, (100 - self.percentile) / 100)
                max_v = torch.quantile(x, self.percentile / 100)
            else:
                min_v = x.min()
                max_v = x.max()

        self.min_val = torch.minimum(self.min_val, min_v)
        self.max_val = torch.maximum(self.max_val, max_v)

        # Recompute scale / zero_point
        if self.symmetric:
            abs_max = torch.maximum(self.min_val.abs(), self.max_val.abs())
            self.scale = abs_max / (2 ** (self.n_bits - 1) - 1)
            self.zero_point = torch.zeros_like(self.scale)
        else:
            self.scale = (self.max_val - self.min_val) / (2 ** self.n_bits - 1)
            self.zero_point = -torch.round(self.min_val / self.scale)

        # Clamp scale to avoid divide-by-zero
        self.scale = self.scale.clamp(min=1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled or not self.training:
            return x
        self._update_stats(x.detach())
        return STEQuantizer.apply(x, self.scale, self.zero_point, self.n_bits)

    def extra_repr(self):
        return (f"n_bits={self.n_bits}, symmetric={self.symmetric}, "
                f"per_channel={self.per_channel}")


# ── LoRA Layer with QAT ────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """
    Wraps a frozen Linear layer with LoRA adaptation.
    W_out = W_frozen(x) + (x @ A.T @ B.T) * (alpha / r)

    Both A and B carry fake quantizers for QAT.
    """
    def __init__(self, in_features: int, out_features: int,
                 r: int = 8, alpha: int = 16, dropout: float = 0.0,
                 n_bits: int = 8, quantize: bool = True):
        super().__init__()
        self.r = r
        self.scale = alpha / r
        self.quantize = quantize

        # Frozen base weight (not registered as parameter → no grad)
        self.register_buffer(
            "weight", torch.zeros(out_features, in_features))
        self.register_buffer(
            "bias_buf", torch.zeros(out_features))
        self.has_bias = True

        # LoRA low-rank matrices (trained)
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B initialized to zero → adapter is identity at start

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # QAT fake quantizers for each LoRA matrix
        if quantize:
            self.quant_A = FakeQuantize(n_bits=n_bits, symmetric=True,
                                        per_channel=True, ch_axis=0)
            self.quant_B = FakeQuantize(n_bits=n_bits, symmetric=True,
                                        per_channel=True, ch_axis=0)
        else:
            self.quant_A = nn.Identity()
            self.quant_B = nn.Identity()

    def load_pretrained(self, weight: torch.Tensor,
                        bias: Optional[torch.Tensor] = None):
        """Load frozen pretrained weights (called once after init)."""
        self.weight.copy_(weight)
        if bias is not None:
            self.bias_buf.copy_(bias)
        else:
            self.has_bias = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Frozen base forward
        base_out = F.linear(x, self.weight,
                            self.bias_buf if self.has_bias else None)
        # LoRA path with quantized weights
        A = self.quant_A(self.lora_A)
        B = self.quant_B(self.lora_B)
        lora_out = self.dropout(x) @ A.T @ B.T * self.scale
        return base_out + lora_out

    def extra_repr(self):
        return (f"in={self.weight.shape[1]}, out={self.weight.shape[0]}, "
                f"r={self.r}, scale={self.scale:.3f}, "
                f"quantize={self.quantize}")


# ── Adapter Layer with QAT ────────────────────────────────────────────────────

class AdapterLayer(nn.Module):
    """
    Bottleneck adapter: x → down → act → up → x (residual)
    Inserted after each transformer/attention block in frozen backbone.

    down and up weights carry fake quantizers.
    """
    def __init__(self, in_dim: int, bottleneck_dim: int = 64,
                 dropout: float = 0.1, n_bits: int = 8, quantize: bool = True):
        super().__init__()
        self.down = nn.Linear(in_dim, bottleneck_dim)
        self.act  = nn.GELU()
        self.up   = nn.Linear(bottleneck_dim, in_dim)
        self.norm = nn.LayerNorm(in_dim)
        self.dropout = nn.Dropout(dropout)

        # Initialize near-zero so adapter starts as identity
        nn.init.normal_(self.down.weight, std=0.01)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

        if quantize:
            self.quant_down = FakeQuantize(n_bits=n_bits, symmetric=True)
            self.quant_up   = FakeQuantize(n_bits=n_bits, symmetric=True)
        else:
            self.quant_down = nn.Identity()
            self.quant_up   = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        # Quantize weights at each forward pass (QAT)
        down_w = self.quant_down(self.down.weight)
        up_w   = self.quant_up(self.up.weight)
        x = F.linear(x, down_w, self.down.bias)
        x = self.act(x)
        x = self.dropout(x)
        x = F.linear(x, up_w, self.up.bias)
        return residual + x


# ── Utility: inject LoRA into an existing nn.Module ───────────────────────────

def inject_lora(module: nn.Module, target_cls=nn.Linear,
                r: int = 8, alpha: int = 16, n_bits: int = 8,
                quantize: bool = True) -> nn.Module:
    """
    Recursively replace all `target_cls` layers in `module` with LoRALinear,
    copying pretrained weights and freezing them.
    Returns the modified module.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, target_cls):
            lora_layer = LoRALinear(
                in_features=child.in_features,
                out_features=child.out_features,
                r=r, alpha=alpha, n_bits=n_bits, quantize=quantize
            )
            lora_layer.load_pretrained(
                child.weight.data,
                child.bias.data if child.bias is not None else None
            )
            setattr(module, name, lora_layer)
        else:
            inject_lora(child, target_cls, r, alpha, n_bits, quantize)
    return module


def set_qat_enabled(module: nn.Module, enabled: bool):
    """Enable or disable all FakeQuantize modules in the model."""
    for m in module.modules():
        if isinstance(m, FakeQuantize):
            m.enabled = enabled


def freeze_backbone(module: nn.Module):
    """Freeze all parameters except LoRA and adapter weights."""
    for name, param in module.named_parameters():
        is_peft = any(k in name for k in
                      ["lora_A", "lora_B", "adapter", "down.", "up.",
                       "projection", "fusion", "detection_head"])
        param.requires_grad = is_peft
