"""
evaluate.py — Q-PEFT-DML Evaluation Entry Point

Usage:
  python evaluate.py --config configs/q_peft_dml_nuscenes.yaml \
                     --checkpoint checkpoints/checkpoint_final.pth
  python evaluate.py --config configs/q_peft_dml_nuscenes.yaml \
                     --checkpoint checkpoints/checkpoint_final.pth \
                     --output_dir results/run1/
"""

import argparse
import logging
import yaml
import torch

from models.q_peft_dml import QPEFTDMLModel
from data.nuscenes_dataset import build_dataloaders
from evaluation.evaluator import QPEFTDMLEvaluator


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Q-PEFT-DML on nuScenes")
    parser.add_argument("--config",     default="configs/q_peft_dml_nuscenes.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="results/")
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available()
                                                        else "cpu")
    parser.add_argument("--fp32_only",  action="store_true",
                        help="Evaluate FP32 model only (no QAT)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s]: %(message)s")
    logger = logging.getLogger("evaluate")

    cfg    = load_config(args.config)
    device = torch.device(args.device)

    logger.info(f"Loading model from {args.checkpoint} ...")
    model = QPEFTDMLModel(cfg)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    if args.fp32_only:
        cfg["quantization"]["enabled"] = False
        model.disable_qat()

    _, val_loader = build_dataloaders(cfg)

    evaluator = QPEFTDMLEvaluator(model, val_loader, cfg, device)
    results   = evaluator.evaluate_all(output_dir=args.output_dir)
    evaluator.print_summary_table(results)


if __name__ == "__main__":
    main()
