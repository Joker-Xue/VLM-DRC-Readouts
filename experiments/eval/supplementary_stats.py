"""Supplementary statistics quoted in Sections V-A and VI-B, from saved outputs only.

  * Section VI-B: Spearman correlation between the standard deviation of the first five
    temperature-0.5 samples and the absolute error of their mean, and the AUC of that
    standard deviation for detecting threshold errors (mean >= 0.5 versus label).
  * Section V-A: percentile bootstrap intervals (3,000 resamples within each group, seed 42)
    for Gemma-3-12B strong-versus-clean and weak-versus-clean AUC, with the CNN-prior panel,
    from the archived per-call outputs of that screen.

Usage: PYTHONPATH=. python experiments/eval/supplementary_stats.py
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "results"


def disagreement():
    for name, label in (("phase7_selfensemble", "test_N28"), ("phase7_selfensemble_traincross", "train_N28")):
        rows = json.loads((RESULTS / name / "per_region.json").read_text())
        y = np.array([r["is_hotspot"] for r in rows])
        samples = np.array([r["samples"][:5] for r in rows], dtype=float)
        sd, mean = samples.std(axis=1), samples.mean(axis=1)
        wrong = ((mean >= 0.5).astype(int) != y).astype(int)
        print(f"{label}: Spearman(SD, |mean - y|) = {spearmanr(sd, np.abs(mean - y)).correlation:.3f}; "
              f"error-detection AUC of SD = {roc_auc_score(wrong, sd):.3f}")


def gemma_intervals(reps=3000, seed=42):
    records = {}
    with open(ROOT / "data" / "evidence_cache" / "test_N28_cnnargmax_evidence.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            records[r["patch_id"]] = r
    calls = next((RESULTS / "v3_perception_gate_gemma3_12b").glob("calls_*"))
    groups = {}
    prefix = "v3_t23_withcnn_"
    for f in calls.glob(prefix + "*.json"):
        rec = records[f.stem[len(prefix):]]
        strength = rec["label_strength"]
        group = ("strong" if strength >= 0.20 else
                 "weak" if strength >= 0.01 and rec["is_hotspot"] else "clean")
        p = json.loads(f.read_text()).get("p_hotspot")
        groups.setdefault(group, []).append(float(p[0] if isinstance(p, list) else p))

    def auc(pos, neg):
        return roc_auc_score([1] * len(pos) + [0] * len(neg), list(pos) + list(neg))

    clean = np.array(groups["clean"])
    for group in ("strong", "weak"):
        pos = np.array(groups[group])
        rng = np.random.default_rng(seed)
        boot = [auc(rng.choice(pos, len(pos)), rng.choice(clean, len(clean))) for _ in range(reps)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"Gemma-3-12B {group} vs clean (with prior): AUC {auc(pos, clean):.3f}, "
              f"95% interval [{lo:.2f}, {hi:.2f}] (n={len(pos)} vs {len(clean)})")


if __name__ == "__main__":
    disagreement()
    gemma_intervals()
