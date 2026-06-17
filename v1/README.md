# Q-PEFT-DML: Quantized Parameter-Efficient Fine-Tuning Deep Metric Learning

## Project Structure

```
q_peft_dml/
├── configs/
│   └── q_peft_dml_nuscenes.yaml       # All hyperparameters
├── models/
│   ├── lora.py                         # STEQuantizer, FakeQuantize, LoRALinear, AdapterLayer
│   ├── encoders.py                     # LiDAR / Camera / Radar / IMU / GNSS encoders
│   ├── projection.py                   # QuantizedLinear, ProjectionHead, MultiModalProjector
│   ├── fusion.py                       # Cross-attention + gating fusion (ModalityFusion)
│   ├── detection_head.py               # 3D bounding box + class prediction (DetectionHead)
│   └── q_peft_dml.py                  # Full model assembly (QPEFTDMLModel)
├── losses/
│   └── geometry_loss.py               # All losses: Detection, Triplet, Consistency,
│                                       #   GeometryPreservation (NEW), QATDistribution, JointLoss
├── data/
│   └── nuscenes_dataset.py            # nuScenes multi-modal loader + collate + build_dataloaders
├── training/
│   └── trainer.py                      # CosineWarmupScheduler, Trainer (dual-forward QAT loop)
├── evaluation/
│   └── evaluator.py                    # QPEFTDMLEvaluator: mAP, latency, geometry metrics
├── utils/
│   └── __init__.py
├── train.py                            # Training entry point
└── evaluate.py                         # Evaluation entry point
```

## Setup

```bash
pip install torch torchvision nuscenes-devkit pyyaml
```

## nuScenes Data

Download from https://www.nuscenes.org/nuscenes and set `data_root` in
`configs/q_peft_dml_nuscenes.yaml`.

## Training

```bash
# Full training (QAT enabled)
python train.py --config configs/q_peft_dml_nuscenes.yaml \
                --data_root /path/to/nuscenes

# FP32 baseline (no quantization)
python train.py --config configs/q_peft_dml_nuscenes.yaml \
                --data_root /path/to/nuscenes --no_qat

# Resume from checkpoint
python train.py --config configs/q_peft_dml_nuscenes.yaml \
                --resume checkpoints/checkpoint_epoch8.pth

# INT4 experiment
python train.py --config configs/q_peft_dml_nuscenes.yaml --bits 4
```

## Evaluation

```bash
python evaluate.py --config configs/q_peft_dml_nuscenes.yaml \
                   --checkpoint checkpoints/checkpoint_final.pth \
                   --output_dir results/
```

## Key Contributions vs PEFT-DML

| Component                  | PEFT-DML | Q-PEFT-DML     |
|----------------------------|----------|----------------|
| LoRA adapters              | ✅        | ✅ (quantized)  |
| Triplet metric loss        | ✅        | ✅              |
| Temporal consistency loss  | ✅        | ✅              |
| Sensor dropout robustness  | ✅        | ✅              |
| QAT (INT8/INT4) via STE    | ❌        | ✅              |
| Geometry preservation loss | ❌        | ✅ (NEW)        |
| QAT distribution loss      | ❌        | ✅ (NEW)        |
| Dual-forward training      | ❌        | ✅              |
| Latency benchmarking       | ❌        | ✅              |

## Architecture Overview

```
Sensor inputs
    ├── LiDAR   → PillarFeatureNet → LoRA Linear → AdapterLayer ─┐
    ├── Camera  → ConvNeXtBlocks  → LoRA Linear → AdapterLayer  ├─→ ProjectionHead ──┐
    ├── Radar   → CNN             → LoRA Linear → AdapterLayer  │   (QuantizedLinear) │
    ├── IMU     → MLP             → LoRA Linear → AdapterLayer  │                     │
    └── GNSS    → MLP             → LoRA Linear → AdapterLayer ─┘                     │
                                                                                       ↓
                                                                    Shared Latent Space (L2-norm)
                                                                                       ↓
                                                                    ModalityFusion (Cross-Attention + Gating)
                                                                                       ↓
                                                                    DetectionHead → cls_logits + box_preds
```

## Loss Function

```
L = λ_det  * L_focal+IoU+orient      (detection)
  + λ_met  * L_triplet               (metric alignment)
  + λ_cons * L_consistency           (temporal + cross-modal)
  + λ_geo  * L_geometry              (NEW: pairwise distance KL: FP vs QAT)
  + λ_qat  * L_kl_distribution       (NEW: output distribution alignment)
```
