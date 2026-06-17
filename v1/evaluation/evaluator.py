"""
evaluation/evaluator.py

Q-PEFT-DML evaluation pipeline:
  1. Detection mAP per sensor-dropout scenario
  2. Latency benchmarking: FP32 vs QAT (ms/frame, FPS)
  3. Embedding geometry preservation: Pearson correlation + KL divergence
     of pairwise distance matrices between FP and QAT embeddings
  4. Parameter efficiency summary
"""

import os
import time
import json
import logging
import numpy as np
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class QPEFTDMLEvaluator:
    def __init__(self, model, val_loader, cfg: dict,
                 device: torch.device):
        self.model      = model
        self.val_loader = val_loader
        self.cfg        = cfg
        self.device     = device

    # ── Detection mAP ────────────────────────────────────────────────────────

    @torch.no_grad()
    def collect_predictions(self, missing_modalities: Optional[list] = None):
        self.model.eval()
        all_preds, all_targets = [], []

        for batch in self.val_loader:
            batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            out        = self.model(batch, forced_missing=missing_modalities)
            cls_probs  = torch.softmax(out["cls_logits"], dim=-1)
            cls_pred   = cls_probs.argmax(dim=-1)
            conf       = cls_probs.max(dim=-1).values

            for i in range(cls_pred.shape[0]):
                all_preds.append({
                    "cls":   cls_pred[i].item(),
                    "conf":  conf[i].item(),
                    "box":   out["box_preds"][i].cpu().numpy(),
                    "token": batch["sample_tokens"][i],
                })
            for i, (boxes, labels) in enumerate(
                    zip(batch["boxes"],
                        batch["labels"].split(1) if hasattr(batch["labels"], "split")
                        else [batch["labels"][i:i+1] for i in range(len(batch["boxes"]))])):
                all_targets.append({
                    "boxes":  boxes,
                    "labels": labels,
                    "token":  batch["sample_tokens"][i],
                })

        return all_preds, all_targets

    def compute_map(self, preds: list, targets: list,
                    iou_threshold: float = 0.5) -> dict:
        num_classes  = self.cfg["model"]["detection"]["num_classes"]
        per_class_ap = {}

        for cls_idx in range(num_classes):
            cls_preds = sorted(
                [(p["conf"], p["token"]) for p in preds if p["cls"] == cls_idx],
                key=lambda x: -x[0])
            n_gt = sum(1 for t in targets
                       for lbl in t["labels"] if lbl.item() == cls_idx)
            if n_gt == 0:
                per_class_ap[cls_idx] = 0.0
                continue

            gt_matched = set()
            tp, fp = [], []
            for conf, token in cls_preds:
                matched = False
                for t in targets:
                    if t["token"] != token:
                        continue
                    for j, lbl in enumerate(t["labels"]):
                        if lbl.item() == cls_idx and (token, j) not in gt_matched:
                            gt_matched.add((token, j))
                            matched = True
                            break
                    if matched:
                        break
                tp.append(1 if matched else 0)
                fp.append(0 if matched else 1)

            tp_c = np.cumsum(tp)
            fp_c = np.cumsum(fp)
            prec = tp_c / (tp_c + fp_c + 1e-6)
            rec  = tp_c / n_gt
            per_class_ap[cls_idx] = float(
                np.trapz(prec, rec) if len(rec) > 1 else 0.0)

        return {"mAP": float(np.mean(list(per_class_ap.values()))),
                "per_class_AP": per_class_ap}

    # ── Latency Benchmarking ──────────────────────────────────────────────────

    def benchmark_latency(self, n_iter=100, warmup=10, mode="qat") -> dict:
        self.model.eval()
        self.model.enable_qat() if mode == "qat" else self.model.disable_qat()

        batch = next(iter(self.val_loader))
        batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}

        with torch.no_grad():
            for _ in range(warmup):
                self.model(batch)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        latencies = []
        with torch.no_grad():
            for _ in range(n_iter):
                t0 = time.perf_counter()
                self.model(batch)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t0) * 1000)

        lat = np.array(latencies)
        return {
            "mode":    mode,
            "mean_ms": float(lat.mean()),
            "std_ms":  float(lat.std()),
            "p50_ms":  float(np.percentile(lat, 50)),
            "p95_ms":  float(np.percentile(lat, 95)),
            "p99_ms":  float(np.percentile(lat, 99)),
            "fps":     float(1000 / lat.mean()),
        }

    # ── Geometry Preservation ─────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate_geometry_preservation(self, n_batches=10) -> dict:
        self.model.eval()
        corrs = {m: [] for m in ["lidar", "camera", "radar", "imu", "gnss"]}
        kls   = {m: [] for m in corrs}

        for i, batch in enumerate(self.val_loader):
            if i >= n_batches:
                break
            batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}

            self.model.disable_qat()
            out_fp = self.model(batch)
            self.model.enable_qat()
            out_q  = self.model(batch)

            for mod in corrs:
                e_fp = out_fp["embeddings"].get(mod)
                e_q  = out_q["embeddings"].get(mod)
                if e_fp is None or e_q is None:
                    continue
                e_fp = F.normalize(e_fp, p=2, dim=-1).cpu().numpy()
                e_q  = F.normalize(e_q,  p=2, dim=-1).cpu().numpy()

                # Pairwise cosine distance matrices → flatten
                D_fp = (1 - e_fp @ e_fp.T).flatten()
                D_q  = (1 - e_q  @ e_q.T).flatten()

                corrs[mod].append(float(np.corrcoef(D_fp, D_q)[0, 1]))

                hist_fp, edges = np.histogram(D_fp, bins=50, density=True)
                hist_q,  _     = np.histogram(D_q,  bins=edges, density=True)
                eps = 1e-8
                kls[mod].append(float(np.sum(
                    hist_fp * np.log((hist_fp + eps) / (hist_q + eps)))))

        return {
            mod: {"pearson_corr": float(np.mean(corrs[mod])),
                  "kl_div":       float(np.mean(kls[mod]))}
            for mod in corrs if corrs[mod]
        }

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def evaluate_all(self, output_dir: str = "results/") -> dict:
        os.makedirs(output_dir, exist_ok=True)
        results = {}

        # 1. All sensors
        logger.info("Evaluating: all sensors ...")
        preds, targets = self.collect_predictions()
        full_metrics   = self.compute_map(preds, targets)
        results["all_sensors"] = full_metrics
        logger.info(f"  mAP (all sensors): {full_metrics['mAP']:.4f}")

        # 2. Sensor dropout scenarios
        for scenario in self.cfg["evaluation"]["dropout_scenarios"]:
            name, missing = scenario["name"], scenario["missing"]
            if not missing:
                continue
            logger.info(f"Evaluating: {name} (missing: {missing}) ...")
            preds, targets = self.collect_predictions(missing_modalities=missing)
            m = self.compute_map(preds, targets)
            results[name] = m
            ratio = m["mAP"] / max(full_metrics["mAP"], 1e-6)
            logger.info(f"  mAP: {m['mAP']:.4f} ({ratio*100:.1f}% of full-sensor)")

        # 3. Latency
        logger.info("Benchmarking latency ...")
        n_iter, warmup = (self.cfg["evaluation"]["benchmark_iterations"],
                          self.cfg["evaluation"]["benchmark_warmup"])
        lat_fp  = self.benchmark_latency(n_iter, warmup, mode="fp32")
        lat_qat = self.benchmark_latency(n_iter, warmup, mode="qat")
        results["latency"] = {"fp32": lat_fp, "qat": lat_qat}
        speedup = lat_fp["mean_ms"] / max(lat_qat["mean_ms"], 1e-3)
        logger.info(f"  FP32: {lat_fp['mean_ms']:.2f} ms | "
                    f"QAT: {lat_qat['mean_ms']:.2f} ms | Speedup: {speedup:.2f}x")

        # 4. Geometry preservation
        logger.info("Evaluating geometry preservation ...")
        geo = self.evaluate_geometry_preservation(n_batches=20)
        results["geometry"] = geo
        for mod, m in geo.items():
            logger.info(f"  {mod}: corr={m['pearson_corr']:.4f}, "
                        f"KL={m['kl_div']:.4f}")

        # 5. Parameter efficiency
        results["parameters"] = self.model.count_trainable_params()

        with open(os.path.join(output_dir, "eval_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_dir}/eval_results.json")
        return results

    def print_summary_table(self, results: dict):
        print("\n" + "=" * 70)
        print("Q-PEFT-DML EVALUATION SUMMARY")
        print("=" * 70)

        print("\n📊 Detection mAP by Sensor Configuration:")
        print(f"  {'Scenario':<28} {'mAP':>8}")
        print("  " + "-" * 38)
        for name, m in results.items():
            if isinstance(m, dict) and "mAP" in m:
                print(f"  {name:<28} {m['mAP']:>8.4f}")

        if "latency" in results:
            lat = results["latency"]
            print("\n⚡ Latency Benchmarking:")
            for mode, l in lat.items():
                print(f"  {mode.upper():<8} {l['mean_ms']:.2f} ± {l['std_ms']:.2f} ms"
                      f"  ({l['fps']:.1f} FPS)")
            speedup = lat["fp32"]["mean_ms"] / max(lat["qat"]["mean_ms"], 1e-3)
            print(f"  Speedup: {speedup:.2f}x")

        if "geometry" in results:
            print("\n📐 Embedding Geometry Preservation (FP32 vs QAT):")
            print(f"  {'Modality':<12} {'Pearson r':>10} {'KL div':>10}")
            print("  " + "-" * 35)
            for mod, m in results["geometry"].items():
                print(f"  {mod:<12} {m['pearson_corr']:>10.4f} {m['kl_div']:>10.4f}")

        if "parameters" in results:
            p = results["parameters"]
            print(f"\n💾 Params: {p['trainable']:,} trainable / "
                  f"{p['total']:,} total ({p['pct_trainable']:.1f}%)")
        print("=" * 70 + "\n")
