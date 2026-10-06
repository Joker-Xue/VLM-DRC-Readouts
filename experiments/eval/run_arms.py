"""⑦ Run the four hybrid arms (R0 / A1 / A2 / A3) over cached CNN evidence.

v2 Hybrid experiment harness (proposal-v2-hybrid.md §3, experiment-plan-phase7 §⑦.0).
Reads the pre-computed evidence cache (no CNN re-run), rebuilds per-region
``RegionFeatures``, runs each arm with a SWAPPABLE caller, and reports comparable
metrics — overall + stratified by ``label_strength``.

Arms (system-level information-matched; only the collaboration structure varies):
  R0  pure CNN threshold (no LLM)          — decision/score from ``cnn_prior``
  A1  single broad agent, 1 call, no debate — baseline_single_llm prompt + broad view
  A2  4 identical broad agents, 3-round debate (homogeneous views; forensics prompt ×4)
  A3  4 heterogeneous specialists, 3-round debate (cnn_prior → forensics only)  ← thesis
Mechanism proven ⇔ A3 > A2 > A1; R0 is the perception ceiling reference (NOT the
mechanism comparison). See notes/routenet-ceiling-R0-20260530.md for why GO.

Caller is swappable (``--caller``):
  mock      schema-valid, signal-driven fake (FREE, Mac) — the ⑦.0 step-5 plumbing gate
  deepseek  DeepSeek-chat via API (⑦.1, ~¥10-30, NO GPU)

Usage (from paper&experiment/)::

    PYTHONPATH=. python3 experiments/eval/run_arms.py --caller mock --sample 60     # ⑦.0 gate
    PYTHONPATH=. python3 experiments/eval/run_arms.py --caller deepseek --sample 400 # ⑦.1

Outputs (experiments/results/phase7_arms/)::
    per_region.json   every arm's decision + score per region
    summary.json      overall + stratified metrics + Win-Tie table
    summary.md        human-readable digest
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]          # paper&experiment/
sys.path.insert(0, str(ROOT))

from experiments.agents.baselines import (  # noqa: E402
    BaselineDecision,
    SINGLE_LLM_PROMPT_PATH,
    run_b2_single_llm,
)
from experiments.agents.orchestrator import (  # noqa: E402
    CollaborationTrace,
    Orchestrator,
    RegionFeatures,
)
from experiments.agents.protocol import AGENT_NAMES  # noqa: E402
from experiments.eval.metrics import (  # noqa: E402
    compute_metrics,
    risk_coverage_curve,
    win_tie_table,
)

PROMPTS_DIR = ROOT / "experiments" / "agents" / "prompts"
CACHE_DIR = ROOT / "data" / "evidence_cache"
RESULTS_DIR = ROOT / "experiments" / "results" / "phase7_arms"

ARMS = ["R0", "A1", "A2", "A3"]
BROAD_AGENT = "drc_pattern_forensics"   # the broad-view agent (all 9 ch + cnn_prior)


# ---------------------------------------------------------------------------
# Strength stratification (mirrors routenet_ceiling.strength_bucket)
# ---------------------------------------------------------------------------


def strength_bucket(s: float) -> str:
    if s < 0.01:
        return "clean(<0.01)"
    if s < 0.05:
        return "weak[0.01,0.05)"
    if s < 0.20:
        return "mid[0.05,0.20)"
    return "strong[>=0.20]"


BUCKET_ORDER = ["clean(<0.01)", "weak[0.01,0.05)", "mid[0.05,0.20)", "strong[>=0.20]"]


# ---------------------------------------------------------------------------
# Mock caller — schema-valid, signal-driven (the ⑦.0 step-5 plumbing gate)
# ---------------------------------------------------------------------------


def _region_signal(features: Any) -> float:
    """Extract a hotspot signal ∈ [0,1] from whatever view the caller received.

    Broad view → cnn_prior.peak (the learned perception signal); narrow view →
    max per-channel peak; legacy box-mean dict → max scalar. The mock thus reacts
    to the ACTUAL evidence threaded to each agent, so the gate catches wiring bugs
    (a mis-threaded view changes the numbers) instead of rubber-stamping them.
    """
    if isinstance(features, dict):
        prior = features.get("cnn_prior")
        if isinstance(prior, dict) and "peak" in prior:
            return float(prior["peak"])
        channels = features.get("channels")
        if isinstance(channels, dict) and channels:
            peaks = [float(c.get("peak", 0.0)) for c in channels.values() if isinstance(c, dict)]
            if peaks:
                return max(peaks)
        scalars = [float(v) for v in features.values() if isinstance(v, (int, float))]
        if scalars:
            return max(scalars)
    return 0.5


def _clamp01(v: float) -> float:
    return max(0.01, min(0.99, v))


def make_mock_caller(threshold: float = 0.1) -> Callable[[str, dict[str, Any]], dict[str, Any]]:
    """Build a deterministic mock LLM caller.

    Handles BOTH protocols off one signature ``(system, payload) -> message``:
      - single-LLM (B2/A1): payload has no ``agent`` → return {decision, confidence}.
      - orchestrator (A2/A3): payload has ``agent``/``round`` → return a schema-valid
        typed message honoring the round's allowed message types.

    Confidence is set so the downstream P(hotspot) score equals the region signal:
    decision=hotspot → score=confidence=sig; decision=not_hotspot → score=1-confidence=sig.
    """

    def mock(system: str, payload: dict[str, Any]) -> dict[str, Any]:
        sig = _region_signal(payload.get("features", {}))
        is_hot = sig >= threshold

        # ---- single-LLM branch (A1 / B-style) ----
        if "agent" not in payload:
            decision = "hotspot" if is_hot else "not_hotspot"
            conf = _clamp01(sig if is_hot else 1.0 - sig)   # → score == sig either way
            return {"decision": decision, "confidence": conf,
                    "rationale": f"mock: region signal peak={sig:.3f} vs thr={threshold:.2f}"}

        # ---- orchestrator branch (A2 / A3) ----
        agent = payload["agent"]
        rnd = int(payload.get("round", 1))
        region = payload.get("region")
        conf = _clamp01(sig if is_hot else 1.0 - sig)
        base: dict[str, Any] = {"agent": agent, "round": rnd, "confidence": conf,
                                "rationale": f"mock peak={sig:.3f}"}
        if isinstance(region, dict):
            base["region"] = region

        if rnd == 1:
            # R1 allows: hotspot_proposal / not_hotspot / abstain (no 'vote').
            base["message_type"] = "hotspot_proposal" if is_hot else "not_hotspot"
            return base
        # R2/R3: emit a 'vote'. depends_on_agents satisfies the R2 "vote-change must
        # credit" cross-field rule unconditionally (harmless in R3); valid enum name.
        base["message_type"] = "vote"
        base["vote"] = "hotspot" if is_hot else "not_hotspot"
        base["depends_on_agents"] = [BROAD_AGENT]
        return base

    return mock


# ---------------------------------------------------------------------------
# Trace → (decision, score) — reuse run_baselines._run_m1 extraction logic
# ---------------------------------------------------------------------------


def decision_and_score_from_trace(trace: CollaborationTrace) -> tuple[str, float]:
    """Map a collaboration trace to a (decision, P(hotspot) score) pair.

    Decision: 'hotspot' / 'not_hotspot' / 'abstain' (needs_human_review → abstain).
    Score: mean confidence of final-round agents whose vote matches the decision
    (1 - mean for not_hotspot; 0.5 for abstain). Identical to run_baselines._run_m1.
    """
    raw = trace.final_decision or "abstain"
    decision = ("hotspot" if raw == "hotspot"
                else "not_hotspot" if raw == "not_hotspot"
                else "abstain")

    last_round = trace.rounds[-1] if trace.rounds else None
    if last_round is None:
        return decision, 0.5

    relevant: list[float] = []
    for msg in last_round.messages:
        cf = msg.get("confidence")
        if cf is None:
            continue
        mt = msg.get("message_type")
        agent_decision = (
            "hotspot" if mt == "hotspot_proposal"
            else "not_hotspot" if mt == "not_hotspot"
            else msg.get("vote") or "abstain"
        )
        if agent_decision == raw:
            relevant.append(float(cf))

    if not relevant or decision == "abstain":
        score = 0.5
    elif decision == "hotspot":
        score = sum(relevant) / len(relevant)
    else:   # not_hotspot
        score = 1.0 - (sum(relevant) / len(relevant))
    return decision, score


def graded_score_from_trace(trace: CollaborationTrace) -> tuple[str, float]:
    """Graded (v3 visual) score: P(hotspot) = mean of the FINAL round's reported `confidence`.

    Here each agent's `confidence` IS its calibrated P(hotspot) (gate-style graded perception),
    so the team score is the ensemble mean probability — it varies with the image even when the
    binary votes collapse to a prior. Decision stays the majority vote (drives hit/flag/abstain);
    only the AUC/ECE-bearing score uses the graded signal.
    """
    raw = trace.final_decision or "abstain"
    decision = ("hotspot" if raw == "hotspot"
                else "not_hotspot" if raw == "not_hotspot" else "abstain")
    last = trace.rounds[-1] if trace.rounds else None
    if last is None:
        return decision, 0.5
    ps = [float(m["confidence"]) for m in last.messages
          if isinstance(m.get("confidence"), (int, float))]
    return decision, (sum(ps) / len(ps) if ps else 0.5)


# ---------------------------------------------------------------------------
# Per-arm runners
# ---------------------------------------------------------------------------


def _region_xy(rec: dict) -> dict[str, int]:
    return {"x": int(rec["x"]), "y": int(rec["y"]), "radius": int(rec["radius"])}


def run_r0(rec: dict, *, threshold: float, score_key: str) -> BaselineDecision:
    """Pure CNN ceiling: decision/score straight from cnn_prior. No LLM."""
    score = float(rec["cnn_prior"][score_key])
    return BaselineDecision(
        method="R0",
        region=_region_xy(rec),
        decision="hotspot" if score >= threshold else "not_hotspot",
        score=score,
        raw_messages=[{"cnn_prior": rec["cnn_prior"]}],
        api_calls=0,
        elapsed_seconds=0.0,
    )


def run_a1(rec: dict, *, caller: Callable | None, prompt_path: Path = SINGLE_LLM_PROMPT_PATH,
           graded: bool = False) -> BaselineDecision:
    """Single broad agent: baseline_single_llm prompt + broad view (9 ch + cnn_prior).

    graded=True (v3 visual): the prompt returns {"p_hotspot", "rationale"}; score = p_hotspot
    (calibrated probability), decision = p≥0.5. Bypasses run_b2's decision-conditional scoring.
    """
    if graded:
        system = prompt_path.read_text()
        payload = {"region": _region_xy(rec), "features": {"_patch_id": rec["patch_id"]},
                   "_temperature": 0.3}
        t0 = time.time()
        msg = caller(system, payload)         # VisualCaller (mode='broad') → forensics image
        dt = time.time() - t0
        p = msg.get("p_hotspot", msg.get("confidence"))
        try:
            p = max(0.0, min(1.0, float(p)))
        except (TypeError, ValueError):
            p = 0.5
        return BaselineDecision(
            method="A1", region=_region_xy(rec),
            decision="hotspot" if p >= 0.5 else "not_hotspot", score=p,
            raw_messages=[msg], api_calls=1, elapsed_seconds=dt,
        )
    broad_view = rec["evidence"][BROAD_AGENT]
    dec = run_b2_single_llm(
        region_features=broad_view,
        region_xy=_region_xy(rec),
        caller=caller,                       # None → DeepSeek single-LLM path
        prompt_path=prompt_path,
    )
    dec.method = "A1"
    return dec


def _run_orchestrated(orch: Orchestrator, region: RegionFeatures, *, method: str,
                      graded: bool = False) -> BaselineDecision:
    t0 = time.time()
    trace = orch.run(region)
    dt = time.time() - t0
    decision, score = (graded_score_from_trace(trace) if graded
                       else decision_and_score_from_trace(trace))
    return BaselineDecision(
        method=method,
        region={"x": region.x, "y": region.y, "radius": region.radius},
        decision=decision,
        score=score,
        raw_messages=[{"round": rd.round_num, "votes": rd.votes, "messages": rd.messages}
                      for rd in trace.rounds],
        api_calls=sum(len(rd.messages) for rd in trace.rounds),
        elapsed_seconds=dt,
    )


def run_a2(rec: dict, orch_homo: Orchestrator, *, graded: bool = False) -> BaselineDecision:
    """Homogeneous: every slot gets the broad view (= build_region_features(homogeneous=True)).

    For the visual arm the evidence dict only carries `_patch_id`; the VisualCaller routes the
    forensics image to every slot via its mode='broad'. The text payload is unused by the VLM.
    """
    broad_view = {"_patch_id": rec["patch_id"]} if graded else rec["evidence"][BROAD_AGENT]
    region = RegionFeatures(
        x=int(rec["x"]), y=int(rec["y"]), radius=int(rec["radius"]),
        evidence={a: broad_view for a in AGENT_NAMES},
    )
    return _run_orchestrated(orch_homo, region, method="A2", graded=graded)


def run_a3(rec: dict, orch_hetero: Orchestrator, *, graded: bool = False) -> BaselineDecision:
    """Heterogeneous: each agent keeps its own per-agent view (cnn_prior → forensics only)."""
    evidence = ({a: {"_patch_id": rec["patch_id"]} for a in AGENT_NAMES} if graded
                else rec["evidence"])
    region = RegionFeatures(
        x=int(rec["x"]), y=int(rec["y"]), radius=int(rec["radius"]),
        evidence=evidence,
    )
    return _run_orchestrated(orch_hetero, region, method="A3", graded=graded)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def load_records(split: str, region: str = "label-argmax") -> list[dict]:
    suffix = "_cnnargmax" if region == "cnn-argmax" else ""
    path = CACHE_DIR / f"{split}{suffix}_evidence.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"evidence cache not found: {path}\nbuild it: "
            f"python3 experiments/feature_extraction/build_evidence_cache.py --split {split} --region {region}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stratified_sample(records: list[dict], n: int, seed: int) -> list[dict]:
    """Proportional sample across the 4 label_strength buckets (each bucket represented,
    so both classes appear → PR-AUC well-defined). n<=0 or n>=len → all records."""
    if n <= 0 or n >= len(records):
        return list(records)
    rng = random.Random(seed)
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_bucket[strength_bucket(float(r["label_strength"]))].append(r)
    total = len(records)
    out: list[dict] = []
    for bucket, group in by_bucket.items():
        k = max(1, round(n * len(group) / total))
        rng.shuffle(group)
        out.extend(group[:k])
    rng.shuffle(out)
    return out


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------


def _metrics_for(arm: str, regions: list[dict]) -> dict:
    decisions = [r["arms"][arm]["decision"] for r in regions]
    scores = [r["arms"][arm]["score"] for r in regions]
    labels = [bool(r["is_hotspot"]) for r in regions]
    api_calls = [r["arms"][arm]["api_calls"] for r in regions]
    latencies = [r["arms"][arm]["elapsed_seconds"] for r in regions]
    return asdict(compute_metrics(
        method=arm, decisions=decisions, scores=scores, labels=labels,
        api_calls=api_calls, latencies=latencies,
    ))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--caller", choices=["mock", "deepseek", "vlm-local"], default="mock",
                    help="backend for A1/A2/A3 (R0 needs none). vlm-local = Qwen2.5-VL on M5 (v3 visual)")
    ap.add_argument("--visual", action="store_true",
                    help="agents see rendered images (prompts/visual/) instead of text descriptors; "
                         "implied by --caller vlm-local")
    ap.add_argument("--prompts-subdir", default="visual_v3_gatemirror",
                    help="subdir under experiments/agents/prompts/ for the visual prompt set "
                         "(e.g. 'visual_v2_neutral' for the gate-aligned ablation); only used when visual")
    ap.add_argument("--render-dir", default=str(ROOT / "data" / "render_cache"),
                    help="where per-agent view PNGs live (built by render_views.py)")
    ap.add_argument("--vlm-temp", type=float, default=0.7,
                    help="sampling temperature for the debate arms (A2/A3); A1 uses min(0.3,this)")
    ap.add_argument("--split", default="test_N28", choices=["test_N28", "train_N28"])
    ap.add_argument("--region", choices=["label-argmax", "cnn-argmax"], default="label-argmax",
                    help="which evidence cache to read (must match build_evidence_cache --region)")
    ap.add_argument("--sample", type=int, default=60,
                    help="stratified region count (0 = all in the split)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--r0-threshold", type=float, default=0.1,
                    help="cnn_prior score threshold for the R0 hotspot decision")
    ap.add_argument("--r0-score", default="peak", choices=["peak", "p90", "mean"],
                    help="which cnn_prior field drives the R0 score")
    ap.add_argument("--mock-threshold", type=float, default=0.1,
                    help="signal threshold for the mock caller's hotspot call")
    ap.add_argument("--out", default=str(RESULTS_DIR))
    a = ap.parse_args(argv)

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    visual = a.visual or a.caller == "vlm-local"
    prompts_src = (PROMPTS_DIR / a.prompts_subdir) if visual else PROMPTS_DIR
    a1_prompt_path = prompts_src / "baseline_single_llm.md"

    # ---- caller(s) ----
    vcallers: list = []          # VisualCaller instances needing per-region set_patch()
    if a.caller == "mock":
        caller = make_mock_caller(a.mock_threshold)
        a1_caller = caller
        a2_caller = a3_caller = caller
    elif a.caller == "deepseek":
        from experiments.agents.llm_clients import call_deepseek
        caller = call_deepseek
        a1_caller = None                       # A1 uses B2's own DeepSeek path
        a2_caller = a3_caller = caller
    else:                                       # vlm-local (v3 visual)
        from experiments.agents.vlm_local import LocalVLM, VisualCaller
        vlm = LocalVLM()
        calls_dir = out_dir / "calls"
        a1_caller = VisualCaller(vlm, a.render_dir, mode="broad", arm="A1",
                                 cache_dir=calls_dir, temperature=min(0.3, a.vlm_temp))
        a2_caller = VisualCaller(vlm, a.render_dir, mode="broad", arm="A2",
                                 cache_dir=calls_dir, temperature=a.vlm_temp)
        a3_caller = VisualCaller(vlm, a.render_dir, mode="own", arm="A3",
                                 cache_dir=calls_dir, temperature=a.vlm_temp)
        vcallers = [a1_caller, a2_caller, a3_caller]

    # ---- orchestrators (built once; reused per region) ----
    orch_homo = orch_hetero = None
    if "A2" in a.arms:
        forensics_prompt = (prompts_src / f"{BROAD_AGENT}.md").read_text()
        orch_homo = Orchestrator(llm_caller=a2_caller, prompts={a_: forensics_prompt for a_ in AGENT_NAMES})
    if "A3" in a.arms:
        hetero_prompts = {a_: (prompts_src / f"{a_}.md").read_text() for a_ in AGENT_NAMES}
        orch_hetero = Orchestrator(llm_caller=a3_caller, prompts=hetero_prompts)

    # ---- data ----
    records = load_records(a.split, a.region)
    sample = stratified_sample(records, a.sample, a.seed)
    n_pos = sum(int(r["is_hotspot"]) for r in sample)
    print(f"[run_arms] caller={a.caller} split={a.split} region={a.region} regions={len(sample)} "
          f"(hotspot={n_pos}/{len(sample)}={100*n_pos/max(len(sample),1):.1f}%) arms={a.arms}")
    bucket_counts = defaultdict(int)
    for r in sample:
        bucket_counts[strength_bucket(float(r["label_strength"]))] += 1
    print("           buckets: " + ", ".join(f"{b}={bucket_counts[b]}" for b in BUCKET_ORDER if bucket_counts[b]))

    # ---- run arms ----
    per_region: list[dict] = []
    for idx, rec in enumerate(sample, start=1):
        row: dict = {
            "patch_id": rec["patch_id"],
            "design": rec["design"],
            "region": _region_xy(rec),
            "label_strength": float(rec["label_strength"]),
            "bucket": strength_bucket(float(rec["label_strength"])),
            "is_hotspot": int(rec["is_hotspot"]),
            "arms": {},
        }
        for vc in vcallers:                       # visual mode: point callers at this patch's images
            vc.set_patch(rec["patch_id"])
        if "R0" in a.arms:
            row["arms"]["R0"] = asdict(run_r0(rec, threshold=a.r0_threshold, score_key=a.r0_score))
        if "A1" in a.arms:
            row["arms"]["A1"] = asdict(run_a1(rec, caller=a1_caller, prompt_path=a1_prompt_path,
                                              graded=visual))
        if "A2" in a.arms:
            row["arms"]["A2"] = asdict(run_a2(rec, orch_homo, graded=visual))
        if "A3" in a.arms:
            row["arms"]["A3"] = asdict(run_a3(rec, orch_hetero, graded=visual))
        per_region.append(row)

        if idx % 10 == 0 or idx == len(sample):
            tail = "  ".join(f"{arm}={row['arms'][arm]['decision'][:4]}/{row['arms'][arm]['score']:.2f}"
                             for arm in a.arms)
            print(f"  [{idx}/{len(sample)}] {rec['patch_id']} lbl={row['label_strength']:.3f} "
                  f"hot={row['is_hotspot']}  {tail}")

    # ---- metrics: overall + stratified ----
    overall = {arm: _metrics_for(arm, per_region) for arm in a.arms}
    stratified: dict[str, dict] = {}
    for bucket in BUCKET_ORDER:
        sub = [r for r in per_region if r["bucket"] == bucket]
        if sub:
            # NB: is_hotspot = label_strength > 0.005, so weak/mid/strong buckets are
            # entirely true-hotspots (single class) → PR-AUC/ROC-AUC are undefined there;
            # hit_rate IS meaningful (= recall on those buckets). Full per-arm metrics are
            # kept in JSON; the markdown headlines hit_rate. ECE/risk-coverage = step 3.
            stratified[bucket] = {"n": len(sub),
                                  "n_hotspot": int(sum(r["is_hotspot"] for r in sub)),
                                  "metrics": {arm: _metrics_for(arm, sub) for arm in a.arms}}

    arm_decisions = {arm: [r["arms"][arm]["decision"] for r in per_region] for arm in a.arms}
    labels = [bool(r["is_hotspot"]) for r in per_region]
    wt = win_tie_table(method_decisions=arm_decisions, labels=labels)

    # Risk-coverage curve points per arm (full sample — both classes; for later plotting).
    # AURC/risk@80 also live in overall[arm]; here we keep the full coverage/risk series.
    risk_coverage = {
        arm: risk_coverage_curve([r["arms"][arm]["score"] for r in per_region], labels)
        for arm in a.arms
    }

    summary = {
        "config": {
            "caller": a.caller, "split": a.split, "region": a.region, "n_regions": len(per_region),
            "seed": a.seed, "arms": a.arms, "r0_threshold": a.r0_threshold,
            "r0_score": a.r0_score, "mock_threshold": a.mock_threshold,
        },
        "label_distribution": {"n_hotspot": int(n_pos), "n_not_hotspot": int(len(per_region) - n_pos)},
        "overall": overall,
        "stratified_by_strength": stratified,
        "risk_coverage": risk_coverage,
        "win_table": wt,
    }

    (out_dir / "per_region.json").write_text(json.dumps(per_region, ensure_ascii=False, indent=2, default=str))
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    (out_dir / "summary.md").write_text(_render_markdown(summary, a))

    print("\n" + "=" * 64)
    for line in _render_markdown(summary, a).splitlines():
        print(line)
    print("=" * 64)
    print(f"written → {out_dir/'per_region.json'} | {out_dir/'summary.json'} | {out_dir/'summary.md'}")
    return 0


def _fmt(v) -> str:
    return "  nan" if v != v else f"{v:.3f}"   # NaN-safe


def _render_markdown(summary: dict, args) -> str:
    md: list[str] = []
    cfg = summary["config"]
    md.append("# Phase ⑦ Arms Summary\n")
    md.append(f"- caller=**{cfg['caller']}** · split={cfg['split']} · region={cfg.get('region','?')} "
              f"· regions=**{cfg['n_regions']}** · seed={cfg['seed']}")
    ld = summary["label_distribution"]
    md.append(f"- labels: {ld['n_hotspot']} hotspot / {ld['n_not_hotspot']} not_hotspot")
    md.append(f"- R0: score=cnn_prior.{cfg['r0_score']} threshold={cfg['r0_threshold']}")
    if cfg["caller"] == "mock":
        md.append("- ⚠️ **caller=mock**: numbers verify PLUMBING only. Arm ordering is a mock "
                  "artifact (signal-driven fake, not reasoning) — the A3>A2>A1 thesis is tested in "
                  "⑦.1 with a real LLM, not here.")
    md.append("")

    md.append("## Overall (mechanism holds ⇔ A3 > A2 > A1; R0 = ceiling ref) — ECE/AURC/R@80 lower=better\n")
    md.append("| Arm | n | PR-AUC | ROC-AUC | ECE | AURC | R@80 | Hit | Flag | Abstain | API calls | mean lat (s) |")
    md.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for arm in cfg["arms"]:
        m = summary["overall"][arm]
        md.append(f"| {arm} | {m['n']} | {_fmt(m['pr_auc'])} | {_fmt(m['roc_auc'])} | "
                  f"{_fmt(m['ece'])} | {_fmt(m['aurc'])} | {_fmt(m['risk_at_cov80'])} | "
                  f"{_fmt(m['hit_rate'])} | {_fmt(m['flag_rate'])} | {_fmt(m['abstain_rate'])} | "
                  f"{m['total_api_calls']} | {m['mean_latency_s']:.2f} |")

    md.append("\n## Hit-rate stratified by label_strength "
              "(weak/mid/strong are all true-hotspots → hit = recall; target = lift weak/mid recall)\n")
    buckets = [b for b in BUCKET_ORDER if b in summary["stratified_by_strength"]]
    md.append("| Bucket | n | hot% | " + " | ".join(cfg["arms"]) + " |")
    md.append("|---|--:|--:|" + "|".join("--:" for _ in cfg["arms"]) + "|")
    for b in buckets:
        s = summary["stratified_by_strength"][b]
        hot_frac = s["n_hotspot"] / s["n"] if s["n"] else float("nan")
        cells = " | ".join(_fmt(s["metrics"][arm]["hit_rate"]) for arm in cfg["arms"])
        md.append(f"| {b} | {s['n']} | {_fmt(hot_frac)} | {cells} |")

    md.append("\n## Win table — count of regions where row-arm correct AND col-arm wrong\n")
    arms = cfg["arms"]
    md.append("| A \\ B | " + " | ".join(arms) + " |")
    md.append("|---|" + "|".join("--" for _ in arms) + "|")
    for ra in arms:
        cells = " | ".join(str(summary["win_table"][ra][cb]) if ra != cb else "-" for cb in arms)
        md.append(f"| **{ra}** | {cells} |")
    return "\n".join(md) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
