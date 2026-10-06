"""T15 ablation (R1 §1.39 / R2 §2.2) — render the A1 broad view WITHOUT the CNN_prior panel.

Reuses render_views.render_agent_view() unmodified: passing cnn_map=None already skips
appending the CNN_prior panel (see render_views.py L82). This script just drives that
existing function over the exact same n=2000 test_N28 cnn-argmax stratified sample used
for Table X's "single" row, so the ablation is a clean paired comparison — no CNN
re-inference needed (we never call CNNEvidenceExtractor), pure CPU image rendering.

Usage (from paper&experiment/):
    PYTHONPATH=. python3 experiments/feature_extraction/render_no_cnn_prior.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))

from experiments.feature_extraction.render_views import (  # noqa: E402
    DATAROOT,
    render_agent_view,
)
from experiments.eval.run_arms import load_records, stratified_sample  # noqa: E402

OUT_DIR = BASE / "data" / "render_cache_no_cnn_prior"


def main() -> int:
    records = load_records("test_N28", "cnn-argmax")
    sample = stratified_sample(records, 2000, seed=42)
    print(f"rendering {len(sample)} no-CNN-prior broad views -> {OUT_DIR}")

    for i, rec in enumerate(sample, 1):
        pid = rec["patch_id"]
        out_path = OUT_DIR / pid / "drc_pattern_forensics.png"
        if out_path.exists():
            continue
        feat = np.load(DATAROOT / "feature" / f"{pid}.npy").transpose(2, 0, 1).astype(np.float32)
        x, y, radius = int(rec["x"]), int(rec["y"]), int(rec["radius"])
        render_agent_view(feat, "drc_pattern_forensics", x, y, radius, out_path, cnn_map=None)
        if i % 50 == 0 or i == len(sample):
            print(f"  [{i}/{len(sample)}] {pid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
