"""Supplementary statistics quoted in Sections V-A and VI-B, from saved outputs only.

  * Section VI-B: Spearman correlation between the standard deviation of the first five
    temperature-0.5 samples and the absolute error of their mean, and the AUC of that
    standard deviation for detecting threshold errors (mean >= 0.5 versus label).
  * Section V-A: percentile bootstrap intervals (3,000 resamples within each group, seed 42)
    for Gemma-3-12B strong-versus-clean and weak-versus-clean AUC, with the CNN-prior panel,
    from the archived per-call outputs of that screen.
  * Abstract and Section V-C: paired bootstrap of the corrected H own-history fixed readout
    against the transferred isotonic CNN on the same test candidates, and of the own-history
    fixed readout against the round-1 mean in each configuration.
  * Section V-C: exact McNemar calculations for the historical A1 decisions against A2 and A3.

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


def _load(path):
    import gzip
    if path.exists():
        return json.loads(path.read_text())
    with gzip.open(str(path) + ".gz", "rt") as fh:
        return json.load(fh)


def corrected_contrasts():
    import sys
    sys.path.insert(0, str(ROOT))
    from sklearn.isotonic import IsotonicRegression
    from experiments.eval.audited_paper_metrics import bootstrap
    from experiments.eval.run_arms import load_records, stratified_sample

    rows = _load(RESULTS / "corrected_debate_20260922_v3_n2000" / "per_region.json")
    y = np.array([r["is_hotspot"] for r in rows])

    def fixed(cfg, branch):
        return [float(np.mean([m["p_hotspot"] for m in r["configurations"][cfg]["branches"][branch]["rounds"][2]]))
                for r in rows]

    test = stratified_sample(load_records("test_N28", "cnn-argmax"), 2000, 42)
    train = stratified_sample(load_records("train_N28", "cnn-argmax"), 2000, 42)
    assert [r["patch_id"] for r in test] == [r["patch_id"] for r in rows]
    peak = lambda recs: np.array([r["cnn_prior"]["peak"] for r in recs])
    label = lambda recs: np.array([r["is_hotspot"] for r in recs])
    cnn = IsotonicRegression(out_of_bounds="clip").fit(peak(train), label(train)).predict(peak(test))
    b = bootstrap(cnn, fixed("H", "own"), y, 3000)
    print(f"H own-history fixed minus transferred CNN: ECE interval "
          f"[{b['ece_ci'][0]:+.3f}, {b['ece_ci'][1]:+.3f}], ROC interval [{b['roc_ci'][0]:+.3f}, {b['roc_ci'][1]:+.3f}]")
    for cfg in ("H", "X"):
        r1 = [r["configurations"][cfg]["r1"]["score"] for r in rows]
        b = bootstrap(r1, fixed(cfg, "own"), y, 3000)
        print(f"{cfg} own-history fixed minus round-1 mean: ROC interval "
              f"[{b['roc_ci'][0]:+.3f}, {b['roc_ci'][1]:+.3f}], ECE interval [{b['ece_ci'][0]:+.3f}, {b['ece_ci'][1]:+.3f}]")


def mcnemar():
    from scipy.stats import binomtest
    rows = json.loads((RESULTS / "phase7_arms_v3gm_n2000_v2" / "per_region.json").read_text())
    y = np.array([r["is_hotspot"] for r in rows])
    correct = {a: np.array([(r["arms"][a]["decision"] == "hotspot") == bool(t) for r, t in zip(rows, y)])
               for a in ("A1", "A2", "A3")}
    for arm in ("A3", "A2"):
        b = int(np.sum(correct["A1"] & ~correct[arm])); c = int(np.sum(~correct["A1"] & correct[arm]))
        p = binomtest(b, b + c, 0.5).pvalue
        print(f"A1 versus {arm}: discordant {b}:{c}, exact McNemar p = {p:.1e}")


if __name__ == "__main__":
    disagreement()
    gemma_intervals()
    corrected_contrasts()
    mcnemar()
