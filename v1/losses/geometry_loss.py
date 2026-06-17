"""
losses/geometry_loss.py  (combined losses file)

All loss functions for Q-PEFT-DML:

1. DetectionLoss   — Focal CE + IoU box + L1 orientation
2. MetricLoss      — Hard triplet loss (metric alignment)
3. ConsistencyLoss — Temporal + cross-modal consistency
4. GeometryLoss    — NEW: pairwise distance matrix alignment (full-prec vs quantized)
5. QATLoss         — KL divergence between full-prec and quantized output distributions
6. JointLoss       — Combines all the above with configurable λ weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ── 1. Detection Loss ─────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, C) raw class logits
        targets : (B,)   integer class labels
        """
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal = self.alpha * (1 - pt) ** self.gamma * ce
        return focal.mean() if self.reduction == "mean" else focal


class IoULoss(nn.Module):
    """3D axis-aligned IoU loss for bounding box regression."""
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target: (B, 6)  [x, y, z, w, l, h]
        """
        # Convert center-size to corner representation
        pred_min   = pred[:, :3] - pred[:, 3:] / 2
        pred_max   = pred[:, :3] + pred[:, 3:] / 2
        target_min = target[:, :3] - target[:, 3:] / 2
        target_max = target[:, :3] + target[:, 3:] / 2

        inter_min = torch.maximum(pred_min, target_min)
        inter_max = torch.minimum(pred_max, target_max)
        inter = (inter_max - inter_min).clamp(min=0).prod(dim=-1)

        vol_pred   = pred[:, 3:].clamp(min=1e-6).prod(dim=-1)
        vol_target = target[:, 3:].clamp(min=1e-6).prod(dim=-1)
        union = vol_pred + vol_target - inter

        iou = inter / union.clamp(min=1e-6)
        return (1 - iou).mean()


class DetectionLoss(nn.Module):
    def __init__(self, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.focal = FocalLoss(focal_alpha, focal_gamma)
        self.iou   = IoULoss()

    def forward(self, cls_logits: torch.Tensor, box_preds: torch.Tensor,
                cls_targets: torch.Tensor, box_targets: torch.Tensor) -> dict:
        """
        cls_logits  : (B, C)
        box_preds   : (B, 10)  [x,y,z,log_w,log_l,log_h,sin,cos,vx,vy]
        cls_targets : (B,)
        box_targets : (B, 10)  [x,y,z,w,l,h,sin,cos,vx,vy]
        """
        # Classification
        loss_cls = self.focal(cls_logits, cls_targets)

        # Box regression (decode log dimensions)
        pred_xyz = box_preds[:, :3]
        pred_wlh = torch.exp(box_preds[:, 3:6].clamp(-3, 3))
        pred_box6 = torch.cat([pred_xyz, pred_wlh], dim=-1)

        tgt_box6 = box_targets[:, :6]
        loss_box = self.iou(pred_box6, tgt_box6)

        # Orientation regression (sin/cos)
        loss_orient = F.l1_loss(box_preds[:, 6:8], box_targets[:, 6:8])

        # Velocity regression
        loss_vel = F.l1_loss(box_preds[:, 8:10], box_targets[:, 8:10])

        total = loss_cls + loss_box + loss_orient + 0.2 * loss_vel
        return {
            "loss_det": total,
            "loss_cls": loss_cls,
            "loss_box": loss_box,
            "loss_orient": loss_orient,
            "loss_vel": loss_vel,
        }


# ── 2. Metric (Triplet) Loss ──────────────────────────────────────────────────

class TripletLoss(nn.Module):
    """
    Batch-hard triplet loss.
    For each anchor, select the hardest positive (same class, max dist)
    and hardest negative (different class, min dist).
    """
    def __init__(self, margin: float = 0.3, mining: str = "hard"):
        super().__init__()
        self.margin  = margin
        self.mining  = mining

    def _pairwise_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute pairwise squared L2 distances. (N, N)"""
        dot = embeddings @ embeddings.T
        sq  = dot.diag().unsqueeze(1) + dot.diag().unsqueeze(0) - 2 * dot
        return sq.clamp(min=0).sqrt()

    def forward(self, embeddings: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        embeddings : (N, D) — L2-normalized
        labels     : (N,)   — integer class labels
        """
        dist = self._pairwise_distances(embeddings)            # (N, N)
        same = labels.unsqueeze(1) == labels.unsqueeze(0)      # (N, N)
        diff = ~same

        # Mask diagonal
        eye = torch.eye(len(labels), device=labels.device, dtype=torch.bool)
        same = same & ~eye

        if self.mining == "hard":
            # Hardest positive: same class, maximum distance
            pos_dist = dist.masked_fill(~same, 0).max(dim=1).values
            # Hardest negative: different class, minimum distance
            neg_dist = dist.masked_fill(~diff, float("inf")).min(dim=1).values
        else:
            # Semi-hard: negative further than positive but within margin
            pos_dist = (dist * same.float()).sum(1) / same.float().sum(1).clamp(1)
            neg_dist = dist.masked_fill(~diff, float("inf")).min(dim=1).values

        loss = F.relu(pos_dist - neg_dist + self.margin)
        valid = same.any(dim=1) & diff.any(dim=1)
        return loss[valid].mean() if valid.any() else loss.sum() * 0


# ── 3. Consistency Loss ───────────────────────────────────────────────────────

class ConsistencyLoss(nn.Module):
    """
    Temporal consistency: adjacent frame embeddings should be similar.
    Cross-modal consistency: same scene, different modalities → close.
    """
    def forward(self, emb_t: torch.Tensor,
                emb_t1: Optional[torch.Tensor] = None,
                cross_modal_pairs: Optional[list] = None) -> torch.Tensor:
        """
        emb_t   : (B, D) — current frame fused embedding
        emb_t1  : (B, D) — next frame fused embedding (if available)
        cross_modal_pairs: list of (emb_a, emb_b) tuples from same scene
        """
        total = torch.tensor(0.0, device=emb_t.device)
        count = 0

        if emb_t1 is not None:
            # L2 temporal consistency
            total = total + F.mse_loss(emb_t, emb_t1)
            count += 1

        if cross_modal_pairs:
            for emb_a, emb_b in cross_modal_pairs:
                if emb_a is not None and emb_b is not None:
                    total = total + F.mse_loss(
                        F.normalize(emb_a, p=2, dim=-1),
                        F.normalize(emb_b, p=2, dim=-1)
                    )
                    count += 1

        return total / max(count, 1)


# ── 4. Geometry Preservation Loss (NOVEL) ─────────────────────────────────────

class GeometryPreservationLoss(nn.Module):
    """
    Ensures the quantized embedding space preserves the geometric structure
    (pairwise distance relationships) of the full-precision embedding space.

    Method:
      1. Compute pairwise distance matrices for full-precision and quantized embeddings
      2. Soft-normalize via temperature scaling → "distance distributions"
      3. KL divergence between the two distributions

    This explicitly prevents quantization from collapsing or distorting the
    metric structure learned by the triplet loss.
    """
    def __init__(self, temperature: float = 0.07, num_pairs: int = 512):
        super().__init__()
        self.temperature = temperature
        self.num_pairs   = num_pairs

    def _distance_matrix(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (N, D), L2-normalized
        Returns cosine distance matrix (N, N), values in [0, 2]
        """
        # cosine similarity → cosine distance
        sim = z @ z.T                          # (N, N) — cosine sim (normalized)
        return 1 - sim                         # cosine distance ∈ [0, 2]

    def _dist_to_distribution(self, D: torch.Tensor) -> torch.Tensor:
        """
        Convert distance matrix to a probability distribution per row.
        Rows are softmax over negative distances (smaller distance = higher prob).
        Diagonal (self-distance) is masked out.
        D: (N, N)
        """
        N = D.shape[0]
        mask = torch.eye(N, device=D.device, dtype=torch.bool)
        D_masked = D.masked_fill(mask, float("inf"))
        return F.softmax(-D_masked / self.temperature, dim=-1)  # (N, N)

    def forward(self, z_fp: torch.Tensor, z_q: torch.Tensor) -> torch.Tensor:
        """
        z_fp : (B, D) — full-precision projected embeddings
        z_q  : (B, D) — quantized projected embeddings (from QAT forward)

        Returns scalar geometry preservation loss.
        """
        # Sub-sample for efficiency if batch is large
        N = z_fp.shape[0]
        if N > self.num_pairs:
            idx = torch.randperm(N, device=z_fp.device)[:self.num_pairs]
            z_fp = z_fp[idx]
            z_q  = z_q[idx]

        # Normalize (should already be normalized by projection head, but be safe)
        z_fp = F.normalize(z_fp, p=2, dim=-1)
        z_q  = F.normalize(z_q,  p=2, dim=-1)

        # Distance matrices
        D_fp = self._distance_matrix(z_fp)   # (N, N) full-precision distances
        D_q  = self._distance_matrix(z_q)    # (N, N) quantized distances

        # Convert to probability distributions
        P_fp = self._dist_to_distribution(D_fp)   # (N, N) — target
        P_q  = self._dist_to_distribution(D_q)    # (N, N) — prediction

        # KL divergence: KL(P_fp || P_q)  — preserve full-precision geometry
        # Add epsilon to avoid log(0)
        eps = 1e-8
        kl = (P_fp * (torch.log(P_fp + eps) - torch.log(P_q + eps))).sum(dim=-1)
        return kl.mean()


# ── 5. QAT Distribution Loss ──────────────────────────────────────────────────

class QATDistributionLoss(nn.Module):
    """
    KL divergence between full-precision and quantized class output distributions.
    Penalizes quantization from shifting the detection confidence distributions.
    """
    def forward(self, logits_fp: torch.Tensor,
                logits_q: torch.Tensor) -> torch.Tensor:
        """
        logits_fp : (B, C) — full-precision logits
        logits_q  : (B, C) — quantized logits
        """
        p_fp = F.softmax(logits_fp, dim=-1)
        log_p_q = F.log_softmax(logits_q, dim=-1)
        return F.kl_div(log_p_q, p_fp, reduction="batchmean")


# ── 6. Joint Loss ─────────────────────────────────────────────────────────────

class JointLoss(nn.Module):
    """
    Combines all loss components with configurable λ weights.

    L = λ_det   * L_detection
      + λ_metric * L_triplet
      + λ_cons   * L_consistency
      + λ_geo    * L_geometry       ← NEW
      + λ_qat    * L_qat_distrib    ← NEW
    """
    def __init__(self, cfg: dict):
        super().__init__()
        loss_cfg = cfg["loss"]

        self.λ_det   = loss_cfg["lambda_det"]
        self.λ_metric = loss_cfg["lambda_metric"]
        self.λ_cons  = loss_cfg["lambda_consistency"]
        self.λ_geo   = loss_cfg["lambda_geometry"]
        self.λ_qat   = loss_cfg["lambda_qat"]

        self.det_loss  = DetectionLoss(
            loss_cfg["focal_alpha"], loss_cfg["focal_gamma"])
        self.metric_loss = TripletLoss(
            loss_cfg["triplet_margin"], loss_cfg["triplet_mining"])
        self.cons_loss  = ConsistencyLoss()
        self.geo_loss   = GeometryPreservationLoss(
            loss_cfg["geometry_temperature"], loss_cfg["geometry_num_pairs"])
        self.qat_loss   = QATDistributionLoss()

    def forward(self,
                # Detection
                cls_logits: torch.Tensor, box_preds: torch.Tensor,
                cls_targets: torch.Tensor, box_targets: torch.Tensor,
                # Metric alignment
                embeddings: dict, labels: torch.Tensor,
                # Consistency
                fused_emb: torch.Tensor,
                fused_emb_prev: Optional[torch.Tensor] = None,
                # Geometry preservation (QAT-specific)
                embeddings_fp: Optional[dict] = None,
                # QAT output distribution
                logits_fp: Optional[torch.Tensor] = None,
                ) -> dict:
        """
        embeddings    : dict[mod → (B, D)]  — quantized projected embeddings
        embeddings_fp : dict[mod → (B, D)]  — full-precision embeddings (no QAT)
                        (computed by running model with QAT disabled)
        labels        : (B,) — ground truth class labels for triplet loss
        """
        losses = {}

        # 1. Detection loss
        det_out = self.det_loss(cls_logits, box_preds, cls_targets, box_targets)
        losses.update(det_out)

        # 2. Metric loss — triplet over all available embeddings concatenated
        valid_embs = [e for e in embeddings.values() if e is not None]
        if valid_embs:
            # Stack: (N_mod_avail * B, D), repeat labels per modality
            stacked = torch.cat(valid_embs, dim=0)
            rep_labels = labels.repeat(len(valid_embs))
            losses["loss_metric"] = self.metric_loss(stacked, rep_labels)
        else:
            losses["loss_metric"] = torch.tensor(0.0, device=cls_logits.device)

        # 3. Consistency loss
        cross_pairs = []
        mods = [m for m, e in embeddings.items() if e is not None]
        for i in range(len(mods)):
            for j in range(i + 1, len(mods)):
                cross_pairs.append((embeddings[mods[i]], embeddings[mods[j]]))
        losses["loss_cons"] = self.cons_loss(fused_emb, fused_emb_prev, cross_pairs)

        # 4. Geometry preservation loss (only when QAT is active + fp embeddings available)
        if embeddings_fp is not None:
            geo_losses = []
            for mod in embeddings:
                e_q  = embeddings.get(mod)
                e_fp = embeddings_fp.get(mod)
                if e_q is not None and e_fp is not None:
                    geo_losses.append(self.geo_loss(e_fp, e_q))
            losses["loss_geometry"] = (
                torch.stack(geo_losses).mean() if geo_losses
                else torch.tensor(0.0, device=cls_logits.device)
            )
        else:
            losses["loss_geometry"] = torch.tensor(0.0, device=cls_logits.device)

        # 5. QAT output distribution loss
        if logits_fp is not None:
            losses["loss_qat"] = self.qat_loss(logits_fp, cls_logits)
        else:
            losses["loss_qat"] = torch.tensor(0.0, device=cls_logits.device)

        # Weighted total
        losses["loss_total"] = (
            self.λ_det    * losses["loss_det"]
          + self.λ_metric * losses["loss_metric"]
          + self.λ_cons   * losses["loss_cons"]
          + self.λ_geo    * losses["loss_geometry"]
          + self.λ_qat    * losses["loss_qat"]
        )

        return losses
