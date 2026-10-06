"""Phase 1 evaluation metrics: PR-AUC, ROC-AUC, Win-Tie, hit rate."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass
class MethodMetrics:
    method: str
    n: int
    pr_auc: float
    roc_auc: float
    ece: float            # expected calibration error (lower=better; defined single-class too)
    aurc: float           # area under risk-coverage curve (lower=better)
    risk_at_cov80: float  # error rate over the most-confident 80% (conf = |score-0.5|)
    hit_rate: float       # decision == is_hotspot
    flag_rate: float      # fraction decided 'hotspot'
    abstain_rate: float   # fraction decided 'abstain'
    total_api_calls: int
    mean_latency_s: float


def expected_calibration_error(
    scores: Sequence[float], labels: Sequence[bool], n_bins: int = 10
) -> float:
    """Equal-width binned ECE on P(hotspot) scores vs binary labels.

    ECE = Σ_b (n_b/n)·|mean_score_b − pos_rate_b|. Uses each bin's MEAN score
    (not the bin center) — the standard region-level form, more accurate at the
    few-hundred-sample scale than the 1000-bin pixel histogram in
    routenet_ceiling.py. Defined even when one class is absent. n==0 → nan.
    """
    s = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
    y = np.asarray(labels, dtype=int)
    n = s.size
    if n == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(s, edges[1:-1], right=False), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        cnt = int(m.sum())
        if cnt:
            ece += (cnt / n) * abs(float(s[m].mean()) - float(y[m].mean()))
    return float(ece)


def risk_coverage_curve(scores: Sequence[float], labels: Sequence[bool]) -> dict:
    """Selective-prediction risk vs coverage, ranked by confidence = |score-0.5|.

    Defer the least-confident predictions first; at coverage c (most-confident c
    fraction kept) risk = error rate (pred = score≥0.5 vs label) over the kept set.
    Returns coverage[]/risk[] plus AURC (mean risk over the per-sample coverage grid;
    lower=better) and risk_at_cov80 (risk at the smallest coverage ≥ 0.80). n==0 → empty/nan.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    n = s.size
    if n == 0:
        return {"coverage": [], "risk": [], "aurc": float("nan"), "risk_at_cov80": float("nan")}
    conf = np.abs(s - 0.5)
    err = ((s >= 0.5).astype(int) != y).astype(float)
    order = np.lexsort((np.arange(n), -conf))   # most-confident first; ties → lower idx first
    cum_err = np.cumsum(err[order])
    k = np.arange(1, n + 1)
    coverage = k / n
    risk = cum_err / k
    j = min(int(np.searchsorted(coverage, 0.80, side="left")), n - 1)
    return {
        "coverage": [round(float(c), 4) for c in coverage],
        "risk": [round(float(r), 4) for r in risk],
        "aurc": float(risk.mean()),       # mean over uniform 1/n…1 grid ≈ trapezoid to O(1/n)
        "risk_at_cov80": float(risk[j]),
    }


def compute_metrics(
    *,
    method: str,
    decisions: Sequence[str],
    scores: Sequence[float],
    labels: Sequence[bool],
    api_calls: Sequence[int],
    latencies: Sequence[float],
) -> MethodMetrics:
    n = len(labels)
    if n == 0:
        nan = float("nan")
        return MethodMetrics(
            method=method, n=0, pr_auc=nan, roc_auc=nan,
            ece=nan, aurc=nan, risk_at_cov80=nan,
            hit_rate=0.0, flag_rate=0.0, abstain_rate=0.0,
            total_api_calls=0, mean_latency_s=0.0,
        )

    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)

    # PR-AUC / ROC-AUC defined only when both classes appear
    pr_auc = float(average_precision_score(y, s)) if 0 < y.sum() < n else float("nan")
    roc_auc = float(roc_auc_score(y, s)) if 0 < y.sum() < n else float("nan")
    # Calibration / selective-prediction (defined even single-class; risk-coverage degenerate single-class)
    ece = expected_calibration_error(scores, labels, n_bins=10)
    rc = risk_coverage_curve(scores, labels)

    binary = np.asarray([d == "hotspot" for d in decisions], dtype=int)
    hit = int((binary == y).sum())
    flag = int(binary.sum())
    abstain = sum(1 for d in decisions if d == "abstain")

    return MethodMetrics(
        method=method,
        n=n,
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        ece=ece,
        aurc=rc["aurc"],
        risk_at_cov80=rc["risk_at_cov80"],
        hit_rate=hit / n,
        flag_rate=flag / n,
        abstain_rate=abstain / n,
        total_api_calls=int(sum(api_calls)),
        mean_latency_s=float(np.mean(latencies)) if latencies else 0.0,
    )


def win_tie_table(
    *,
    method_decisions: dict[str, Sequence[str]],
    labels: Sequence[bool],
) -> dict[str, dict[str, int]]:
    """For each pair of methods, count regions where method A correctly beat method B.

    "Win" for A vs B: A correct, B wrong.
    "Tie": both correct or both wrong.
    """
    methods = list(method_decisions.keys())
    table: dict[str, dict[str, int]] = {a: {b: 0 for b in methods} for a in methods}

    y = list(labels)
    for a in methods:
        a_correct = [(d == "hotspot") == lbl for d, lbl in zip(method_decisions[a], y)]
        for b in methods:
            if a == b:
                continue
            b_correct = [(d == "hotspot") == lbl for d, lbl in zip(method_decisions[b], y)]
            table[a][b] = sum(1 for ac, bc in zip(a_correct, b_correct) if ac and not bc)
    return table


__all__ = [
    "MethodMetrics",
    "compute_metrics",
    "win_tie_table",
    "expected_calibration_error",
    "risk_coverage_curve",
]
