"""
training/trainer.py

Main training loop for Q-PEFT-DML.

Key aspects:
  1. Full-precision warmup (QAT disabled for first N steps)
  2. After warmup: QAT enabled + geometry loss computed via
     dual-forward (one FP pass + one QAT pass per batch)
  3. Mixed-precision training (AMP) for speed
  4. Cosine LR scheduling with linear warmup
  5. Periodic checkpointing + metric logging
"""

import os
import time
import json
import math
import logging
from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

logger = logging.getLogger(__name__)


class CosineWarmupScheduler:
    """Linear warmup then cosine annealing."""
    def __init__(self, optimizer, warmup_steps: int, total_steps: int,
                 min_lr: float = 1e-6):
        self.optimizer    = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.min_lr       = min_lr
        self.base_lrs     = [pg["lr"] for pg in optimizer.param_groups]
        self._step        = 0

    def step(self):
        self._step += 1
        s = self._step
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if s <= self.warmup_steps:
                pg["lr"] = base_lr * s / max(self.warmup_steps, 1)
            else:
                progress = (s - self.warmup_steps) / max(
                    self.total_steps - self.warmup_steps, 1)
                pg["lr"] = self.min_lr + 0.5 * (base_lr - self.min_lr) * (
                    1 + math.cos(math.pi * progress))

    def get_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


class Trainer:
    def __init__(self, model, criterion, train_loader, val_loader,
                 cfg: dict, device: torch.device):
        self.model        = model.to(device)
        self.criterion    = criterion
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg
        self.device       = device

        train_cfg = cfg["training"]
        log_cfg   = cfg["logging"]
        quant_cfg = cfg["quantization"]

        # Optimizer (only trainable params)
        opt_cfg = train_cfg["optimizer"]
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg["weight_decay"],
            betas=opt_cfg["betas"],
        )

        # Steps per epoch
        self.steps_per_epoch = (train_cfg.get("steps_per_epoch")
                                 or len(train_loader))
        total_steps = train_cfg["epochs"] * self.steps_per_epoch
        warmup_steps = train_cfg["scheduler"]["warmup_epochs"] * self.steps_per_epoch

        self.scheduler = CosineWarmupScheduler(
            self.optimizer, warmup_steps, total_steps,
            train_cfg["scheduler"]["min_lr"])

        # AMP
        self.amp     = train_cfg.get("amp", True) and device.type == "cuda"
        self.scaler  = GradScaler() if self.amp else None

        # QAT warmup
        self.qat_warmup_steps = quant_cfg.get("qat_warmup_steps", 1000)
        self.qat_enabled      = False

        # Misc
        self.epochs         = train_cfg["epochs"]
        self.grad_clip      = train_cfg["grad_clip"]
        self.log_every      = log_cfg["log_every_n_steps"]
        self.save_every     = log_cfg["save_every_n_epochs"]
        self.checkpoint_dir = log_cfg["checkpoint_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.global_step = 0
        self.best_map    = 0.0

    def _forward_pass(self, batch: dict, qat_on: bool) -> dict:
        """Run a single forward pass with QAT on or off."""
        if qat_on:
            self.model.enable_qat()
        else:
            self.model.disable_qat()

        return self.model(batch)

    def _compute_targets(self, batch: dict) -> tuple:
        """Extract per-sample targets for detection loss."""
        # Use first label per sample as the class for the scene embedding
        # (simplified; full implementation uses anchor assignment)
        labels = batch["labels"].to(self.device)
        # For box regression, use first annotation per sample
        boxes_list = batch["boxes"]
        # Pad/stack to (B, 10)
        B = len(boxes_list)
        box_targets = torch.zeros(B, 10, device=self.device)
        cls_targets = torch.zeros(B, dtype=torch.long, device=self.device)
        for i, (boxes, lbl) in enumerate(zip(boxes_list,
                batch["labels"].split(1) if hasattr(batch["labels"], "split") else
                [batch["labels"][i:i+1] for i in range(B)])):
            if len(boxes) > 0:
                box_targets[i] = boxes[0].to(self.device)
                # cls_targets[i] = lbl[0]  (simplified: use batch labels)
        # Use batch labels directly
        cls_targets = batch["labels"][:B].to(self.device)
        return cls_targets, box_targets

    def train_step(self, batch: dict) -> dict:
        self.model.train()
        self.optimizer.zero_grad()

        # ── QAT activation schedule ───────────────────────────────────────
        if (not self.qat_enabled and
                self.global_step >= self.qat_warmup_steps and
                self.cfg["quantization"]["enabled"]):
            self.model.enable_qat()
            self.qat_enabled = True
            logger.info(f"QAT enabled at step {self.global_step}")

        cls_targets, box_targets = self._compute_targets(batch)

        # ── Dual forward: FP reference + QAT ─────────────────────────────
        embeddings_fp  = None
        logits_fp      = None

        if self.qat_enabled:
            # 1. Full-precision reference forward (no grad needed)
            with torch.no_grad():
                self.model.disable_qat()
                with (autocast() if self.amp else torch.no_grad()):
                    out_fp = self.model(batch)
                embeddings_fp = {m: e.detach() if e is not None else None
                                 for m, e in out_fp["embeddings"].items()}
                logits_fp = out_fp["cls_logits"].detach()

            # 2. QAT forward (with grad)
            self.model.enable_qat()

        with (autocast() if self.amp else contextlib_nullcontext()):
            out = self.model(batch)
            losses = self.criterion(
                cls_logits=out["cls_logits"],
                box_preds=out["box_preds"],
                cls_targets=cls_targets,
                box_targets=box_targets,
                embeddings=out["embeddings"],
                labels=cls_targets,
                fused_emb=out["fused_embedding"],
                fused_emb_prev=None,           # TODO: implement temporal buffering
                embeddings_fp=embeddings_fp,
                logits_fp=logits_fp,
            )

        loss = losses["loss_total"]

        if self.amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                self.grad_clip)
            self.optimizer.step()

        self.scheduler.step()
        self.global_step += 1

        return {k: v.item() for k, v in losses.items()}

    @torch.no_grad()
    def val_step(self, batch: dict) -> dict:
        self.model.eval()
        self.model.enable_qat() if self.qat_enabled else self.model.disable_qat()

        cls_targets, box_targets = self._compute_targets(batch)
        out = self.model(batch)
        losses = self.criterion(
            cls_logits=out["cls_logits"],
            box_preds=out["box_preds"],
            cls_targets=cls_targets,
            box_targets=box_targets,
            embeddings=out["embeddings"],
            labels=cls_targets,
            fused_emb=out["fused_embedding"],
        )
        return {k: v.item() for k, v in losses.items()}

    def train_epoch(self, epoch: int) -> dict:
        running = {}
        t0 = time.time()
        for step, batch in enumerate(self.train_loader):
            if step >= self.steps_per_epoch:
                break
            batch = {k: (v.to(self.device)
                         if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            loss_dict = self.train_step(batch)
            for k, v in loss_dict.items():
                running.setdefault(k, []).append(v)

            if self.global_step % self.log_every == 0:
                avg = {k: sum(vs) / len(vs) for k, vs in running.items()}
                lr  = self.scheduler.get_lr()[0]
                logger.info(
                    f"Epoch {epoch} step {step}/{self.steps_per_epoch} | "
                    f"LR={lr:.2e} | "
                    + " | ".join(f"{k}={v:.4f}" for k, v in avg.items())
                    + f" | {(time.time()-t0)/(step+1):.2f}s/step"
                )

        return {k: sum(vs) / len(vs) for k, vs in running.items()}

    def validate(self) -> dict:
        running = {}
        for batch in self.val_loader:
            batch = {k: (v.to(self.device)
                         if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            loss_dict = self.val_step(batch)
            for k, v in loss_dict.items():
                running.setdefault(k, []).append(v)
        return {k: sum(vs) / len(vs) for k, vs in running.items()}

    def save_checkpoint(self, epoch: int, metrics: dict, tag: str = ""):
        fname = os.path.join(self.checkpoint_dir,
                             f"checkpoint_epoch{epoch}{tag}.pth")
        torch.save({
            "epoch":        epoch,
            "global_step":  self.global_step,
            "model_state":  self.model.state_dict(),
            "optimizer":    self.optimizer.state_dict(),
            "scheduler_step": self.scheduler._step,
            "metrics":      metrics,
            "qat_enabled":  self.qat_enabled,
        }, fname)
        logger.info(f"Checkpoint saved: {fname}")
        return fname

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler._step = ckpt.get("scheduler_step", 0)
        self.global_step = ckpt.get("global_step", 0)
        self.qat_enabled = ckpt.get("qat_enabled", False)
        if self.qat_enabled:
            self.model.enable_qat()
        logger.info(f"Loaded checkpoint from {path} (epoch {ckpt['epoch']})")
        return ckpt["epoch"]

    def fit(self):
        """Full training loop."""
        logger.info("=" * 60)
        logger.info("Starting Q-PEFT-DML Training")
        param_info = self.model.count_trainable_params()
        logger.info(
            f"Params: {param_info['trainable']:,} trainable / "
            f"{param_info['total']:,} total "
            f"({param_info['pct_trainable']:.1f}%)"
        )
        logger.info(f"QAT warmup: {self.qat_warmup_steps} steps")
        logger.info("=" * 60)

        history = []
        for epoch in range(1, self.epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            # Validate
            val_metrics = self.validate()

            log_entry = {
                "epoch": epoch,
                "train": train_metrics,
                "val":   val_metrics,
                "lr":    self.scheduler.get_lr()[0],
                "qat":   self.qat_enabled,
            }
            history.append(log_entry)
            logger.info(
                f"\n{'='*60}\nEpoch {epoch}/{self.epochs} Summary\n"
                f"Train total_loss: {train_metrics.get('loss_total', 0):.4f}\n"
                f"Val   total_loss: {val_metrics.get('loss_total', 0):.4f}\n"
                f"QAT enabled: {self.qat_enabled}\n{'='*60}"
            )

            # Save checkpoint
            if epoch % self.save_every == 0:
                self.save_checkpoint(epoch, log_entry)

            # Save best model (proxy: lowest val detection loss)
            val_det = val_metrics.get("loss_det", float("inf"))
            if val_det < self.best_map:
                self.best_map = val_det
                self.save_checkpoint(epoch, log_entry, tag="_best")

        # Save final
        self.save_checkpoint(self.epochs, history[-1], tag="_final")

        # Save training history
        hist_path = os.path.join(self.cfg["logging"]["log_dir"], "history.json")
        os.makedirs(self.cfg["logging"]["log_dir"], exist_ok=True)
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)
        logger.info(f"Training history saved to {hist_path}")


# contextlib null context for Python < 3.7 compatibility
try:
    from contextlib import nullcontext as contextlib_nullcontext
except ImportError:
    from contextlib import contextmanager
    @contextmanager
    def contextlib_nullcontext():
        yield
