"""Bootstrap 95% CI over an existing per_region.json (no new model calls).

Answers "is this ROC-AUC/ECE delta real or a single-run fluke" without re-running the
VLM — resamples regions with replacement (the standard non-parametric bootstrap for a
fixed evaluation set), not new stochastic model samples. Complements (does not replace)
literal seed reruns if those are ever also done.

Usage:
    PYTHONPATH=. python3 experiments/eval/bootstrap_ci.py \
        experiments/results/phase7_selfensemble/per_region.json --keys single0 ens_median ens_mean
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from experiments.eval.metrics import compute_metrics


def bootstrap_metric(rows: list[dict], key: str, n_boot: int, seed: int) -> dict:
    rng = random.Random(seed)
    n = len(rows)
    rocs, eces = [], []
    for _ in range(n_boot):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        labels = [bool(r["is_hotspot"]) for r in sample]
        scores = [r[key] for r in sample]
        if sum(labels) == 0 or sum(labels) == len(labels):
            continue  # degenerate resample, extremely rare at this class balance; skip
        decisions = ["hotspot" if s >= 0.5 else "not_hotspot" for s in scores]
        m = compute_metrics(method=key, decisions=decisions, scores=scores, labels=labels,
                            api_calls=[0] * len(sample), latencies=[0.0] * len(sample))
        rocs.append(m.roc_auc)
        eces.append(m.ece)
    rocs.sort(); eces.sort()

    def ci(vals):
        lo = vals[int(0.025 * len(vals))]
        hi = vals[int(0.975 * len(vals)) - 1]
        return lo, hi

    roc_lo, roc_hi = ci(rocs)
    ece_lo, ece_hi = ci(eces)
    return {
        "key": key, "n_boot_used": len(rocs),
        "roc_auc_median": rocs[len(rocs) // 2], "roc_auc_ci95": [roc_lo, roc_hi],
        "ece_median": eces[len(eces) // 2], "ece_ci95": [ece_lo, ece_hi],
    }


def paired_bootstrap_diff(rows: list[dict], key_a: str, key_b: str, n_boot: int, seed: int) -> dict:
    """key_b - key_a on the SAME resampled indices each draw (paired; both scores come
    from the same regions). Correct test for 'does ensembling beat single-shot' — comparing
    marginal CIs of key_a and key_b separately is the wrong (overlapping-CI) fallacy."""
    rng = random.Random(seed)
    n = len(rows)
    roc_diffs, ece_diffs = [], []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        sample = [rows[i] for i in idx]
        labels = [bool(r["is_hotspot"]) for r in sample]
        if sum(labels) == 0 or sum(labels) == len(labels):
            continue
        def m(key):
            scores = [r[key] for r in sample]
            decisions = ["hotspot" if s >= 0.5 else "not_hotspot" for s in scores]
            return compute_metrics(method=key, decisions=decisions, scores=scores, labels=labels,
                                   api_calls=[0] * len(sample), latencies=[0.0] * len(sample))
        ma, mb = m(key_a), m(key_b)
        roc_diffs.append(mb.roc_auc - ma.roc_auc)
        ece_diffs.append(mb.ece - ma.ece)
    roc_diffs.sort(); ece_diffs.sort()

    def ci(vals):
        lo = vals[int(0.025 * len(vals))]
        hi = vals[int(0.975 * len(vals)) - 1]
        p_favors_b = sum(1 for v in vals if v > 0) / len(vals)
        return lo, hi, p_favors_b

    roc_lo, roc_hi, roc_p = ci(roc_diffs)
    ece_lo, ece_hi, ece_p = ci(ece_diffs)  # for ECE, b<a (lower ECE) is the improvement
    return {
        "key_a": key_a, "key_b": key_b, "n_boot_used": len(roc_diffs),
        "roc_auc_diff_ci95": [roc_lo, roc_hi], "frac_boot_roc_b_gt_a": roc_p,
        "ece_diff_ci95": [ece_lo, ece_hi], "frac_boot_ece_b_gt_a": ece_p,
    }


def unpaired_bootstrap_diff(rows_a: list[dict], rows_b: list[dict], key: str, n_boot: int, seed: int) -> dict:
    """ROC-AUC(b) - ROC-AUC(a) where a and b are INDEPENDENT evaluation conditions (e.g.
    test_N28 vs train_N28) — each resampled from its own set on its own draw, sharing one
    rng stream. Correct test for 'is this cross-condition gap real or noise'; do NOT compare
    the two conditions' marginal CIs directly (same overlapping-CI fallacy as the paired case)."""
    rng = random.Random(seed)
    na, nb = len(rows_a), len(rows_b)
    roc_diffs = []
    for _ in range(n_boot):
        sample_a = [rows_a[rng.randrange(na)] for _ in range(na)]
        sample_b = [rows_b[rng.randrange(nb)] for _ in range(nb)]
        def roc(sample):
            labels = [bool(r["is_hotspot"]) for r in sample]
            scores = [r[key] for r in sample]
            decisions = ["hotspot" if s >= 0.5 else "not_hotspot" for s in scores]
            return compute_metrics(method=key, decisions=decisions, scores=scores, labels=labels,
                                   api_calls=[0] * len(sample), latencies=[0.0] * len(sample)).roc_auc
        la = [bool(r["is_hotspot"]) for r in sample_a]
        lb = [bool(r["is_hotspot"]) for r in sample_b]
        if sum(la) in (0, len(la)) or sum(lb) in (0, len(lb)):
            continue
        roc_diffs.append(roc(sample_b) - roc(sample_a))
    roc_diffs.sort()
    lo = roc_diffs[int(0.025 * len(roc_diffs))]
    hi = roc_diffs[int(0.975 * len(roc_diffs)) - 1]
    p_b_gt_a = sum(1 for v in roc_diffs if v > 0) / len(roc_diffs)
    return {
        "key": key, "n_boot_used": len(roc_diffs),
        "roc_auc_diff_median": roc_diffs[len(roc_diffs) // 2], "roc_auc_diff_ci95": [lo, hi],
        "frac_boot_b_gt_a": p_b_gt_a,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("per_region_json")
    ap.add_argument("--keys", nargs="+", default=["single0", "ens_median", "ens_mean"])
    ap.add_argument("--baseline-key", default="single0", help="key to diff the others against")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compare", default=None,
                    help="path to a SECOND per_region.json from an independent condition (e.g. "
                         "train_N28 vs the primary arg's test_N28) — switches to unpaired "
                         "cross-condition bootstrap over --keys instead of the default paired mode")
    a = ap.parse_args(argv)

    if a.compare:
        rows_a = json.loads(Path(a.per_region_json).read_text())
        rows_b = json.loads(Path(a.compare).read_text())
        print(f"[unpaired bootstrap] a(n={len(rows_a)})={a.per_region_json} vs b(n={len(rows_b)})={a.compare} "
              f"n_boot={a.n_boot} seed={a.seed}\n")
        print("| key | ROC-AUC diff b−a (median [95% CI]) | P(b>a) | n_boot_used |")
        print("|---|---|--:|--:|")
        results = []
        for key in a.keys:
            r = unpaired_bootstrap_diff(rows_a, rows_b, key, a.n_boot, a.seed)
            results.append(r)
            print(f"| {r['key']} | {r['roc_auc_diff_median']:+.3f} [{r['roc_auc_diff_ci95'][0]:+.3f}, "
                  f"{r['roc_auc_diff_ci95'][1]:+.3f}] | {r['frac_boot_b_gt_a']:.1%} | {r['n_boot_used']} |")
        out = Path(a.per_region_json).parent / "unpaired_bootstrap_vs_compare.json"
        out.write_text(json.dumps({"a": a.per_region_json, "b": a.compare, "seed": a.seed,
                                    "n_boot": a.n_boot, "results": results}, indent=1))
        print(f"\nwritten → {out}")
        return 0

    rows = json.loads(Path(a.per_region_json).read_text())
    print(f"[bootstrap] n_regions={len(rows)} n_boot={a.n_boot} (resampling regions with replacement)\n")
    print("| key | ROC-AUC (median [95% CI]) | ECE (median [95% CI]) | n_boot_used |")
    print("|---|---|---|--:|")
    results = []
    for key in a.keys:
        r = bootstrap_metric(rows, key, a.n_boot, a.seed)
        results.append(r)
        print(f"| {r['key']} | {r['roc_auc_median']:.3f} [{r['roc_auc_ci95'][0]:.3f}, {r['roc_auc_ci95'][1]:.3f}] "
              f"| {r['ece_median']:.3f} [{r['ece_ci95'][0]:.3f}, {r['ece_ci95'][1]:.3f}] | {r['n_boot_used']} |")

    print(f"\n## paired bootstrap diff vs {a.baseline_key} (correct test — same resample for both arms)")
    print("| vs baseline | ROC-AUC diff (median [95% CI]) | P(better) | ECE diff (median [95% CI]) | P(lower ECE) |")
    print("|---|---|--:|---|--:|")
    diffs = []
    for key in a.keys:
        if key == a.baseline_key:
            continue
        d = paired_bootstrap_diff(rows, a.baseline_key, key, a.n_boot, a.seed)
        diffs.append(d)
        roc_med = sorted([d["roc_auc_diff_ci95"][0], d["roc_auc_diff_ci95"][1]])
        print(f"| {key} | [{d['roc_auc_diff_ci95'][0]:+.3f}, {d['roc_auc_diff_ci95'][1]:+.3f}] "
              f"| {d['frac_boot_roc_b_gt_a']:.1%} | [{d['ece_diff_ci95'][0]:+.3f}, {d['ece_diff_ci95'][1]:+.3f}] "
              f"| {1 - d['frac_boot_ece_b_gt_a']:.1%} |")

    out = Path(a.per_region_json).parent / "bootstrap_ci.json"
    out.write_text(json.dumps({"marginal": results, "paired_diff_vs_" + a.baseline_key: diffs}, indent=1))
    print(f"\nwritten → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
