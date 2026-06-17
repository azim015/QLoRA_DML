"""
models/fusion.py

Cross-attention BEV fusion with drop-modality gating.

Key features:
  - Cross-attention over available modality tokens (missing = masked)
  - Learned gating weights per modality
  - Drop-modality augmentation during training
  - Quantized Q/K/V projection weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from .projection import QuantizedLinear

MODALITY_ORDER = ["lidar", "camera", "radar", "imu", "gnss"]


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, n_bits=8, quantize=True):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.q_proj   = QuantizedLinear(embed_dim, embed_dim, n_bits=n_bits, quantize=quantize)
        self.k_proj   = QuantizedLinear(embed_dim, embed_dim, n_bits=n_bits, quantize=quantize)
        self.v_proj   = QuantizedLinear(embed_dim, embed_dim, n_bits=n_bits, quantize=quantize)
        self.out_proj = QuantizedLinear(embed_dim, embed_dim, n_bits=n_bits, quantize=quantize)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None):
        """
        query : (B, 1, D)
        key   : (B, N_mod, D)
        mask  : (B, N_mod) — True = missing
        """
        B, _, D = query.shape
        N = key.shape[1]

        def split_heads(t, seq):
            return t.reshape(B, seq, self.num_heads, self.head_dim).transpose(1, 2)

        Q = split_heads(self.q_proj(query), 1)
        K = split_heads(self.k_proj(key),   N)
        V = split_heads(self.v_proj(value),  N)

        attn = (Q @ K.transpose(-2, -1)) * self.scale   # (B, H, 1, N)

        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = torch.nan_to_num(F.softmax(attn, dim=-1), nan=0.0)
        attn = self.dropout(attn)

        out = (attn @ V).squeeze(2).transpose(1, 2).reshape(B, 1, D)
        return self.out_proj(out).squeeze(1)


class FusionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, n_bits=8, quantize=True):
        super().__init__()
        self.attn  = MultiHeadCrossAttention(embed_dim, num_heads, dropout, n_bits, quantize)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            QuantizedLinear(embed_dim, embed_dim * 4, n_bits=n_bits, quantize=quantize),
            nn.GELU(),
            nn.Dropout(dropout),
            QuantizedLinear(embed_dim * 4, embed_dim, n_bits=n_bits, quantize=quantize),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, fused, modality_tokens, mask):
        attended = self.attn(fused.unsqueeze(1), modality_tokens, modality_tokens, mask)
        fused = self.norm1(fused + self.dropout(attended))
        fused = self.norm2(fused + self.dropout(self.ff(fused)))
        return fused


class ModalityFusion(nn.Module):
    """
    Full fusion module:
      1. Stack available modality embeddings into token sequence
      2. Apply N cross-attention fusion layers
      3. Add gated residual → single fused embedding (B, embed_dim)
    """
    def __init__(self, embed_dim, num_heads=8, num_layers=2,
                 dropout=0.1, drop_modality_prob=0.3,
                 n_bits=8, quantize=True):
        super().__init__()
        self.embed_dim          = embed_dim
        self.drop_modality_prob = drop_modality_prob

        self.fusion_query        = nn.Parameter(torch.randn(1, embed_dim) * 0.02)
        self.modality_embeddings = nn.Embedding(len(MODALITY_ORDER), embed_dim)

        self.layers = nn.ModuleList([
            FusionLayer(embed_dim, num_heads, dropout, n_bits, quantize)
            for _ in range(num_layers)
        ])

        self.gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, 1),
        )
        self.output_norm = nn.LayerNorm(embed_dim)

    def _apply_dropout_augmentation(self, available):
        if not self.training:
            return available
        available = list(available)
        can_drop  = [i for i, a in enumerate(available) if a]
        for i in can_drop:
            if torch.rand(1).item() < self.drop_modality_prob:
                available[i] = False
        if not any(available):
            available[can_drop[0]] = True
        return available

    def forward(self, embeddings: Dict[str, Optional[torch.Tensor]],
                forced_mask: Optional[torch.BoolTensor] = None):
        """
        Returns: fused (B,D), modality_tokens (B,N,D), mask (B,N) True=missing
        """
        B      = next(e for e in embeddings.values() if e is not None).shape[0]
        device = next(e for e in embeddings.values() if e is not None).device

        available = [embeddings.get(m) is not None for m in MODALITY_ORDER]
        available = self._apply_dropout_augmentation(available)

        tokens    = []
        mask_list = []
        for i, (mod, avail) in enumerate(zip(MODALITY_ORDER, available)):
            type_emb = self.modality_embeddings(torch.tensor(i, device=device))
            if avail and embeddings.get(mod) is not None:
                tokens.append(embeddings[mod] + type_emb.unsqueeze(0))
                mask_list.append(torch.zeros(B, device=device, dtype=torch.bool))
            else:
                tokens.append(torch.zeros(B, self.embed_dim, device=device))
                mask_list.append(torch.ones(B, device=device, dtype=torch.bool))

        modality_tokens = torch.stack(tokens, dim=1)       # (B, N_mod, D)
        mask            = torch.stack(mask_list, dim=1)    # (B, N_mod)

        if forced_mask is not None:
            mask = mask | forced_mask.to(device)

        gate_scores  = self.gate(modality_tokens).squeeze(-1)
        gate_scores  = gate_scores.masked_fill(mask, float("-inf"))
        gate_weights = torch.nan_to_num(F.softmax(gate_scores, dim=-1), nan=0.0)

        fused = self.fusion_query.expand(B, -1)
        for layer in self.layers:
            fused = layer(fused, modality_tokens, mask)

        fused = fused + (gate_weights.unsqueeze(-1) * modality_tokens).sum(1)
        fused = self.output_norm(fused)

        return fused, modality_tokens, mask
