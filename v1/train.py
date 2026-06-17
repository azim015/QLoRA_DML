"""
train.py — Q-PEFT-DML Training Entry Point

Usage:
  python train.py --config configs/q_peft_dml_nuscenes.yaml
  python train.py --config configs/q_peft_dml_nuscenes.yaml \
                  --resume checkpoints/checkpoint_epoch8.pth
  python train.py --config configs/q_peft_dml_nuscenes.yaml \
                  --data_root /path/to/nuscenes --epochs 12
"""

import os
import argparse
import logging
import random
import yaml
import numpy as np
import torch

from models.q_peft_dml import QPEFTDMLModel
from losses.geometry_loss import JointLoss
from data.nuscenes_dataset import build_dataloaders
from training.trainer import Trainer


def setup_logging(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, "train.log")),
        ]
    )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def override_cfg(cfg: dict, args: argparse.Namespace) -> dict:
    """Apply CLI argument overrides to config."""
    if args.data_root:
        cfg["data"]["data_root"] = args.data_root
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["data"]["batch_size"] = args.batch_size
    if args.lr:
        cfg["training"]["optimizer"]["lr"] = args.lr
    if args.no_qat:
        cfg["quantization"]["enabled"] = False
    if args.bits:
        for k in cfg["quantization"]["bits"]:
            cfg["quantization"]["bits"][k] = args.bits
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Train Q-PEFT-DML on nuScenes")
    parser.add_argument("--config",    default="configs/q_peft_dml_nuscenes.yaml")
    parser.add_argument("--resume",    default=None, help="Resume from checkpoint")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--epochs",    type=int, default=None)
    parser.add_argument("--batch_size",type=int, default=None)
    parser.add_argument("--lr",        type=float, default=None)
    parser.add_argument("--no_qat",    action="store_true",
                        help="Disable QAT (FP32 baseline)")
    parser.add_argument("--bits",      type=int, default=None,
                        help="Override quantization bits (e.g. 4 or 8)")
    parser.add_argument("--device",    default="cuda" if torch.cuda.is_available()
                                                       else "cpu")
    args = parser.parse_args()

    # Load and patch config
    cfg = load_config(args.config)
    cfg = override_cfg(cfg, args)

    setup_logging(cfg["logging"]["log_dir"])
    logger = logging.getLogger("train")

    set_seed(cfg["training"]["seed"])
    device = torch.device(args.device)
    logger.info(f"Device: {device}")
    logger.info(f"Config: {args.config}")
    if cfg["quantization"]["enabled"]:
        logger.info(f"QAT: enabled (warmup {cfg['quantization']['qat_warmup_steps']} steps)")
        logger.info(f"Bits: {cfg['quantization']['bits']}")
    else:
        logger.info("QAT: DISABLED (FP32 training)")

    # ── Build components ─────────────────────────────────────────────────────
    logger.info("Building model ...")
    model = QPEFTDMLModel(cfg)

    logger.info("Building dataloaders ...")
    train_loader, val_loader = build_dataloaders(cfg)
    logger.info(f"Train: {len(train_loader.dataset):,} samples | "
                f"Val: {len(val_loader.dataset):,} samples")

    criterion = JointLoss(cfg)

    trainer = Trainer(model, criterion, train_loader, val_loader, cfg, device)

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch = 1
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume) + 1
        logger.info(f"Resuming from epoch {start_epoch}")

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer.fit()


if __name__ == "__main__":
    main()
