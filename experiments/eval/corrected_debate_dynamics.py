"""Descriptive dynamics and stored-output sensitivity checks quoted in Section V-C.

Recomputes, from saved per-region files only (no model inference):
  * corrected debate: round-3 unanimity, majority-outcome changes from round 1 to round 3,
    share of round-3 hotspot votes, mean probability shift, hotspot-decision rate and
    precision, peer-attribution rate;
  * corrected debate: round-1 mean over all k-agent subsets (k=1..4), round-2 readouts,
    and fixed-round ECE with 5/10/15/20 bins;
  * historical debate: regions whose unanimous non-abstaining first round was nevertheless
    continued by the exact-three stopping rule.

Usage: PYTHONPATH=. python experiments/eval/corrected_debate_dynamics.py
"""
import gzip
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from experiments.eval.audited_paper_metrics import canonical, ece

RESULTS = Path(__file__).resolve().parents[1] / "results"


def load(path):
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text())
    with gzip.open(str(path) + ".gz", "rt") as fh:
        return json.load(fh)


def majority(msgs):
    votes = [m["vote"] for m in msgs]
    hot, cold = votes.count("hotspot"), votes.count("not_hotspot")
    return "hotspot" if hot > cold else "not_hotspot" if cold > hot else "tie"


def decision(msgs):
    votes = [m["vote"] for m in msgs]
    hot, cold = votes.count("hotspot"), votes.count("not_hotspot")
    if hot + cold < 2 or hot == cold:
        return "abstain"
    return "hotspot" if hot > cold else "not_hotspot"


def mean_p(msgs):
    return float(np.mean([m["p_hotspot"] for m in msgs]))


def corrected():
    rows = load(RESULTS / "corrected_debate_20260922_v3_n2000" / "per_region.json")
    y = np.array([r["is_hotspot"] for r in rows])
    for cfg in ("H", "X"):
        branches = {b: [r["configurations"][cfg]["branches"][b]["rounds"] for r in rows]
                    for b in ("own", "peer")}
        print(f"== configuration {cfg}")
        for b, rounds in branches.items():
            r3 = [rr[2] for rr in rounds]
            unanimous = np.mean([len({m["vote"] for m in m3}) == 1 and m3[0]["vote"] != "abstain"
                                 for m3 in r3])
            changed = np.mean([majority(rr[0]) != majority(rr[2]) for rr in rounds])
            hot_votes = np.mean([m["vote"] == "hotspot" for m3 in r3 for m in m3])
            shift = np.mean([mean_p(rr[2]) - mean_p(rr[0]) for rr in rounds])
            flagged = np.array([decision(m3) == "hotspot" for m3 in r3])
            line = (f"  {b:4s} unanimity {unanimous:.4f}  majority change {changed:.4f}  "
                    f"round-3 hotspot votes {hot_votes:.4f}  mean shift {shift:+.4f}  "
                    f"hotspot decisions {flagged.mean():.4f}  precision {y[flagged].mean():.4f}")
            if b == "peer":
                later = [m for rr in rounds for rd in rr[1:] for m in rd]
                line += f"  attribution {np.mean([bool(m['depends_on_agents']) for m in later]):.4f}"
            print(line)
            for rd in (2, 3):
                s = np.array([mean_p(rr[rd - 1]) for rr in rounds])
                bins = {b_: round(ece(s, y, b_), 3) for b_ in (5, 10, 15, 20)}
                print(f"       round {rd}: ROC {roc_auc_score(y, canonical(s)):.3f}  ECE by bins {bins}")
        r1 = [rr[0] for rr in branches["own"]]
        for k in range(1, 5):
            rocs = [roc_auc_score(y, canonical([np.mean([m[i]["p_hotspot"] for i in sub]) for m in r1]))
                    for sub in itertools.combinations(range(4), k)]
            print(f"  round-1 mean of k={k} agents (all subsets): ROC {np.mean(rocs):.3f}")


def historical():
    rows = load(RESULTS / "phase7_arms_v3gm_n2000_v2" / "per_region.json")
    for arm in ("A2", "A3"):
        count = 0
        for r in rows:
            blocks = [b for b in r["arms"][arm].get("raw_messages", [])
                      if isinstance(b, dict) and "round" in b and "messages" in b]
            if len(blocks) < 2:
                continue
            first = blocks[0]["messages"]
            labels = []
            for m in first:
                t = m.get("message_type")
                labels.append("hotspot" if t == "hotspot_proposal" else
                              "not_hotspot" if t == "not_hotspot" else "abstain")
            if len(labels) == 4 and len(set(labels)) == 1 and labels[0] != "abstain":
                count += 1
        print(f"== historical {arm}: unanimous non-abstaining first round, continued: {count}")


if __name__ == "__main__":
    corrected()
    historical()
