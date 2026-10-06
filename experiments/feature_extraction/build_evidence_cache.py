"""⑥ Pre-compute + cache CNN evidence for the official CircuitNet-N28 split.

Runs on M5 (MPS, free). The cache lets AutoDL LLM runs read structured evidence
WITHOUT re-running the CNN (offloads perception to the Mac → saves GPU). One argmax
region per patch (mirrors dataset.py region selection). Uses the official design-
separated split (train_N28 = RISCY/RISCY-FPU, test_N28 = zero-riscy) per proposal-v2.

Usage (from paper&experiment/):
    python3 experiments/feature_extraction/build_evidence_cache.py --split test_N28 --limit 50
    python3 experiments/feature_extraction/build_evidence_cache.py --split both
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
from experiments.feature_extraction.cnn_evidence import CNNEvidenceExtractor
from experiments.feature_extraction.region_ops import _box_mean_argmax, _crop_mean

CSV_DIR = BASE / ".thirdparty/CircuitNet/routability_ir_drop_prediction/files"
DATAROOT = BASE / "data/circuitnet_n28/training_set/DRC"
HOTSPOT_THR = 0.005
RADIUS = 8


def design_of(name: str) -> str:
    b = name.split("-")
    return b[1] if len(b) > 1 else "?"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test_N28", choices=["train_N28", "test_N28", "both"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--region", choices=["label-argmax", "cnn-argmax"], default="label-argmax",
                    help="region center: label-argmax (uses GT, back-compat default) or "
                         "cnn-argmax (label-free; CNN hotspot-map argmax — the v2 deploy story)")
    ap.add_argument("--out", default=str(BASE / "data/evidence_cache"))
    a = ap.parse_args(argv)
    splits = ["train_N28", "test_N28"] if a.split == "both" else [a.split]

    ex = CNNEvidenceExtractor()
    print(f"extractor ready (in_ch={ex.in_channels}, device={ex.device})")
    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        rows = [l.strip() for l in open(CSV_DIR / f"{split}.csv") if l.strip()]
        if a.limit:
            rows = rows[:a.limit]
        suffix = "_cnnargmax" if a.region == "cnn-argmax" else ""
        outf = outdir / f"{split}{suffix}_evidence.jsonl"
        n = pos = 0; t0 = time.time()
        with outf.open("w") as fo:
            for line in rows:
                fp, lp = line.split(",")
                name = Path(fp).stem
                feat = np.load(DATAROOT / fp).transpose(2, 0, 1).astype(np.float32)
                lab = np.load(DATAROOT / lp).transpose(2, 0, 1).astype(np.float32)[0]
                cnn_map = ex.hotspot_map(feat)
                if a.region == "cnn-argmax":
                    cx, cy, _ = _box_mean_argmax(cnn_map, RADIUS)   # CNN picks region (label-FREE)
                    strength = _crop_mean(lab, cx, cy, RADIUS)      # GT strength AT the cnn-chosen region
                else:
                    cx, cy, strength = _box_mean_argmax(lab, RADIUS)  # legacy GT-argmax
                ev = ex.region_evidence(feat, cx, cy, RADIUS, cnn_map=cnn_map)
                is_hot = int(strength > HOTSPOT_THR); pos += is_hot
                fo.write(json.dumps({
                    "patch_id": name, "design": design_of(name), "split": split,
                    "region_select": a.region,
                    "x": int(cx), "y": int(cy), "radius": RADIUS,
                    "is_hotspot": is_hot, "label_strength": float(strength),
                    "evidence": ev["per_agent"], "cnn_prior": ev["_cnn_prior_full"],
                }, ensure_ascii=False) + "\n")
                n += 1
                if n % 500 == 0:
                    print(f"  {split}: {n}/{len(rows)} ({n/(time.time()-t0):.0f}/s)")
        print(f"✓ {split}: {n} patches ({pos} hotspot, {100*pos/max(n,1):.1f}%) → {outf} ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
