"""Experiment A — strong-base self-ensemble (Branch-C-persist probe).

Diagnosis (2026-06-14): prompt-level debate loses in every slice; the root cause is that
the debate harness CORRUPTS each agent's perception at the source (forensics-in-A3 sees the
same full view as A1 but scores 0.535 vs A1's 0.765) via temp 0.7 + vote-machinery prompt +
channel isolation. So before any debate/RL, test the cleanest collaboration: ENSEMBLE the
strong single perceiver (A1 setup: full forensics view, clean v3-gatemirror prompt) N times
and aggregate. No protocol/orchestrator touched — just N independent VisualCaller calls with
distinct `arm` tags (so the content cache yields N fresh samples) at temperature>0.

Question: does same-model ensembling beat single A1 (ROC 0.765 / ECE 0.111)?
  - yes  -> collaboration has ensemble headroom -> pursue heterogeneous models / RL
  - ~no  -> same-model ensemble is a dead end -> need genuinely different models (B) or RL (C)

Run:
  LG_VLM_MODEL=... PYTHONPATH=. python3 -u experiments/eval/run_selfensemble.py \
    --sample 150 --seed 42 --n 5 --temp 0.5 --out experiments/results/phase7_selfensemble
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from experiments.eval.run_arms import (
    load_records, stratified_sample, strength_bucket, _region_xy, PROMPTS_DIR, ROOT,
)
from experiments.eval.metrics import compute_metrics
from experiments.agents.vlm_local import LocalVLM, VisualCaller

BUCKET_ORDER = ["clean(<0.01)", "weak[0.01,0.05)", "mid[0.05,0.20)", "strong[>=0.20]"]


def _p_from(msg: dict) -> float:
    p = msg.get("p_hotspot", msg.get("confidence"))
    try:
        return max(0.0, min(1.0, float(p)))
    except (TypeError, ValueError):
        return 0.5


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=5, help="ensemble size (samples per region)")
    ap.add_argument("--temp", type=float, default=0.5, help="sampling temperature for diversity")
    ap.add_argument("--region", choices=["label-argmax", "cnn-argmax"], default="cnn-argmax")
    ap.add_argument("--split", default="test_N28")
    ap.add_argument("--prompts-subdir", default="visual_v3_gatemirror")
    ap.add_argument("--render-dir", default=str(ROOT / "data" / "render_cache"))
    ap.add_argument("--out", default=str(ROOT / "experiments" / "results" / "phase7_selfensemble"))
    a = ap.parse_args(argv)

    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    calls_dir = out_dir / "calls"
    system = (PROMPTS_DIR / a.prompts_subdir / "baseline_single_llm.md").read_text()

    vlm = LocalVLM()
    # N callers with distinct arm tags -> distinct cache keys -> N fresh samples of the SAME
    # strong A1 call (mode="broad" = the full forensics view), all at temperature=a.temp.
    callers = [VisualCaller(vlm, a.render_dir, mode="broad", arm=f"AE{i}",
                            cache_dir=calls_dir, temperature=a.temp) for i in range(a.n)]

    records = load_records(a.split, a.region)
    sample = stratified_sample(records, a.sample, a.seed)
    n_pos = sum(int(r["is_hotspot"]) for r in sample)
    print(f"[selfensemble] n={len(sample)} hot={n_pos} N={a.n} temp={a.temp} "
          f"prompts={a.prompts_subdir} region={a.region}")

    rows = []
    for idx, rec in enumerate(sample, 1):
        pid = rec["patch_id"]
        payload = {"region": _region_xy(rec), "features": {"_patch_id": pid}}
        ps = []
        for c in callers:
            c.set_patch(pid)
            ps.append(_p_from(c(system, payload)))
        ens = sum(ps) / len(ps)
        rows.append({
            "patch_id": pid, "is_hotspot": int(rec["is_hotspot"]),
            "label_strength": float(rec["label_strength"]),
            "bucket": strength_bucket(float(rec["label_strength"])),
            "samples": ps, "ens_mean": ens, "ens_median": st.median(ps), "single0": ps[0],
        })
        if idx % 10 == 0 or idx == len(sample):
            print(f"  [{idx}/{len(sample)}] {pid} lbl={rec['label_strength']:.3f} "
                  f"hot={rec['is_hotspot']} samples={['%.2f'%x for x in ps]} ens={ens:.2f}")

    labels = [bool(r["is_hotspot"]) for r in rows]
    def metr(key):
        scores = [r[key] for r in rows]
        decisions = ["hotspot" if s >= 0.5 else "not_hotspot" for s in scores]
        return compute_metrics(method=key, decisions=decisions, scores=scores, labels=labels,
                               api_calls=[a.n]*len(rows), latencies=[0.0]*len(rows))

    lines = [f"# Experiment A — strong-base self-ensemble (N={a.n}, temp={a.temp})",
             f"- n={len(rows)} hot={n_pos} · prompts={a.prompts_subdir} · region={a.region}",
             "- reference: A1 single (v3 gate-mirror) ROC 0.765 / ECE 0.111\n",
             "| readout | ROC-AUC | PR-AUC | ECE | AURC | Flag | Hit |",
             "|---|--:|--:|--:|--:|--:|--:|"]
    for key in ("single0", "ens_median", "ens_mean"):
        m = metr(key)
        lines.append(f"| {key} | {m.roc_auc:.3f} | {m.pr_auc:.3f} | {m.ece:.3f} | "
                     f"{m.aurc:.3f} | {m.flag_rate:.3f} | {m.hit_rate:.3f} |")
    # stratified mean score by bucket (does ensemble preserve separation?)
    lines.append("\n## mean ens score by bucket")
    lines.append("| bucket | n | single0 | ens_mean |\n|---|--:|--:|--:|")
    for b in BUCKET_ORDER:
        sub = [r for r in rows if r["bucket"] == b]
        if not sub:
            continue
        s0 = sum(r["single0"] for r in sub) / len(sub)
        em = sum(r["ens_mean"] for r in sub) / len(sub)
        lines.append(f"| {b} | {len(sub)} | {s0:.3f} | {em:.3f} |")
    md = "\n".join(lines)
    (out_dir / "summary.md").write_text(md + "\n")
    (out_dir / "per_region.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print("\n" + md)
    print(f"\nwritten → {out_dir/'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
