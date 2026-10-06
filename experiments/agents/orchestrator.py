"""Four-agent × three-round (p0 p0 p1) collaboration orchestrator.

Entry point for the Phase 1 prompt-only baseline and Phase 2 LoRA co-training.
Phase 2 reuses this inference path, swapping LoRA adapters per agent via PEFT.

Agent identifiers (Phase 2-prep extension to 4 agents):
- ``congestion_analyst``         — sees 4 congestion overflow channels (narrow specialist)
- ``wire_demand_forecaster``     — sees 3 RUDY channels (narrow specialist)
- ``drc_pattern_forensics``      — sees all 9 channels + outputs rule_type tag
                                   (broad specialist; multi-task LoRA training)
- ``layout_arbiter``             — sees 2 layout channels + specialists' evidence
                                   (arbiter; final vote)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .protocol import (
    AGENT_NAMES,
    ARBITER_AGENTS,
    SPECIALIST_AGENTS,
    aggregate_vote,
    allowed_message_types,
    decide_next,
    validate_message,
    visible_context,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RegionFeatures:
    """Per-region feature view, sharded by the four agents' visibility.

    Field names follow the agent identifiers in :data:`AGENT_NAMES`.
    """

    x: int
    y: int
    radius: int
    congestion: dict[str, float] = field(default_factory=dict)   # congestion_analyst view (4 ch)
    wire_demand: dict[str, float] = field(default_factory=dict)  # wire_demand_forecaster view (3 ch)
    drc_pattern: dict[str, float] = field(default_factory=dict)  # drc_pattern_forensics view (all 9 ch)
    layout: dict[str, float] = field(default_factory=dict)       # layout_arbiter view (2 ch)
    # v2 Hybrid (2026-05-30): optional structured CNN evidence keyed by agent name. When set,
    # for_agent() returns the agent's structured view (cnn_evidence schema: per-channel descriptors
    # + cnn_prior for the broad agent) instead of the scalar box-mean dict. None → legacy box-mean
    # path (backward compatible; existing tests/callers unaffected).
    evidence: dict | None = None

    def for_agent(self, agent: str) -> dict[str, Any]:
        if self.evidence is not None and agent in self.evidence:
            return self.evidence[agent]
        if agent == "congestion_analyst":
            return self.congestion
        if agent == "wire_demand_forecaster":
            return self.wire_demand
        if agent == "drc_pattern_forensics":
            return self.drc_pattern
        if agent == "layout_arbiter":
            return self.layout
        raise ValueError(f"unknown agent: {agent}")


@dataclass
class RoundTrace:
    round_num: int
    messages: list[dict[str, Any]] = field(default_factory=list)
    votes: dict[str, int] = field(default_factory=dict)


@dataclass
class CollaborationTrace:
    region: RegionFeatures
    rounds: list[RoundTrace] = field(default_factory=list)
    final_decision: str | None = None  # 'hotspot' / 'not_hotspot' / 'needs_human_review'


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

LLMCaller = Callable[[str, dict[str, Any]], dict[str, Any]]
"""Signature: ``(system_prompt, user_payload) -> raw_json_message_dict``."""


class Orchestrator:
    """Three-agent × three-round main loop.

    Minimal contract:
        in:  :class:`RegionFeatures`
        out: :class:`CollaborationTrace` (three rounds of messages + final_decision)
    """

    def __init__(
        self,
        llm_caller: LLMCaller,
        prompts: dict[str, str],
        max_retries: int = 5,
        force_all_rounds: bool = False,
    ) -> None:
        self.llm_caller = llm_caller
        # Expected keys: 'congestion_analyst', 'wire_demand_forecaster', 'layout_arbiter'
        self.prompts = prompts
        self.max_retries = max_retries
        # force_all_rounds (2026-05-28): when True, always run all 3 rounds
        # even if consensus/divergence would trigger early_stop/escalate.
        # Required during PPO TRAINING so every agent produces exactly
        # max_rounds captures — otherwise early consensus yields 1-2 captures
        # per agent and trl.PPOTrainer.step raises "Batch size (3) … got 1".
        # Early stopping is an inference-time efficiency optimization; the
        # policy weights are identical either way, so forcing 3 rounds during
        # training only changes how many gradient samples we collect (and
        # gives richer R1/R2/R3 debate traces for the paper).
        self.force_all_rounds = force_all_rounds
        missing = set(AGENT_NAMES) - set(prompts.keys())
        if missing:
            raise ValueError(f"missing prompts for agents: {sorted(missing)}")

    # ------------------------------------------------------------------ public

    def run(self, region: RegionFeatures) -> CollaborationTrace:
        trace = CollaborationTrace(region=region)
        prev_counts: dict[str, int] | None = None

        for rnd in (1, 2, 3):
            round_trace = self._run_round(rnd, region, trace.rounds)
            trace.rounds.append(round_trace)

            current_counts = round_trace.votes
            action = decide_next(rnd, current_counts, prev_counts)

            # During training (force_all_rounds=True) we ignore early
            # termination so every agent contributes exactly max_rounds
            # captures to the PPO batch. We still record what the decision
            # WOULD have been (for analysis) but keep collecting rounds.
            if not self.force_all_rounds:
                if action == "early_stop":
                    trace.final_decision = self._majority(current_counts)
                    return trace
                if action == "escalate":
                    trace.final_decision = "needs_human_review"
                    return trace

            prev_counts = current_counts

        # Ran all three rounds without consensus
        last = trace.rounds[-1].votes
        if last.get("hotspot", 0) == last.get("not_hotspot", 0):
            trace.final_decision = "needs_human_review"
        else:
            trace.final_decision = self._majority(last)
        return trace

    # ----------------------------------------------------------------- helpers

    def _run_round(
        self,
        rnd: int,
        region: RegionFeatures,
        prior: list[RoundTrace],
    ) -> RoundTrace:
        messages: list[dict[str, Any]] = []
        # Specialists go first (in parallel logically — sequential here for
        # determinism); arbiter goes last so it can read their messages.
        for agent in sorted(SPECIALIST_AGENTS):
            msg = self._call_agent(agent, rnd, region, prior, messages)
            messages.append(msg)

        for agent in sorted(ARBITER_AGENTS):
            msg = self._call_agent(agent, rnd, region, prior, messages)
            messages.append(msg)

        return RoundTrace(round_num=rnd, messages=messages, votes=aggregate_vote(messages))

    def _call_agent(
        self,
        agent: str,
        rnd: int,
        region: RegionFeatures,
        prior_rounds: list[RoundTrace],
        current_round_messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Call one agent with up to ``max_retries`` validation retries.

        Falls back to an automatic ``abstain`` if every attempt fails schema.
        """
        context_keys = visible_context(agent, rnd)
        payload: dict[str, Any] = {
            "agent": agent,
            "round": rnd,
            "region": {"x": region.x, "y": region.y, "radius": region.radius},
            "features": region.for_agent(agent),
            "visible_history": self._assemble_history(
                agent, rnd, prior_rounds, current_round_messages, context_keys
            ),
            "allowed_message_types": sorted(allowed_message_types(rnd)),
        }

        last_raw: dict[str, Any] | None = None
        last_errors: list[str] = []
        for _attempt in range(self.max_retries):
            raw = self.llm_caller(self.prompts[agent], payload)
            ok, errors = validate_message(raw)
            if ok:
                return raw
            payload["last_errors"] = errors
            last_raw = raw
            last_errors = errors

        # Forensic logging (added 2026-05-28): capture WHY schema validation
        # kept failing so we can pinpoint the round-rule / field mismatch.
        # Best-effort; never raise from logging.
        try:
            import os
            import time
            import json as _json
            log_dir = os.environ.get(
                "LAYOUTGUARD_SCHEMA_FAIL_LOG_DIR",
                "/root/logs/schema_failures",
            )
            os.makedirs(log_dir, exist_ok=True)
            fname = f"{int(time.time() * 1000)}-{agent}-r{rnd}.txt"
            with open(os.path.join(log_dir, fname), "w", encoding="utf-8") as f:
                f.write(f"=== agent: {agent}\n=== round: {rnd}\n")
                f.write(f"=== errors:\n")
                for e in last_errors:
                    f.write(f"  {e}\n")
                f.write(f"=== last_raw message:\n")
                f.write(_json.dumps(last_raw, ensure_ascii=False, indent=2))
        except Exception:
            pass

        return {
            "message_type": "abstain",
            "agent": agent,
            "round": rnd,
            "rationale": f"protocol failure after {self.max_retries} retries",
        }

    @staticmethod
    def _assemble_history(
        agent: str,
        rnd: int,
        prior_rounds: list[RoundTrace],
        current_round_messages: list[dict[str, Any]],
        context_keys: list[str],
    ) -> list[dict[str, Any]]:
        history: list[dict[str, Any]] = []
        for key in context_keys:
            if key == "own_channel":
                continue
            if key == "specialists_round1" and prior_rounds:
                history.extend(
                    m for m in prior_rounds[0].messages
                    if m["agent"] in SPECIALIST_AGENTS
                )
            elif key == "all_round1" and len(prior_rounds) >= 1:
                history.extend(prior_rounds[0].messages)
            elif key == "all_round2" and len(prior_rounds) >= 2:
                history.extend(prior_rounds[1].messages)
        # Current-round messages already emitted by other agents (arbiter sees specialists)
        history.extend(m for m in current_round_messages if m["agent"] != agent)
        return history

    @staticmethod
    def _majority(counts: dict[str, int]) -> str:
        h = counts.get("hotspot", 0)
        n = counts.get("not_hotspot", 0)
        if h > n:
            return "hotspot"
        if n > h:
            return "not_hotspot"
        return "needs_human_review"


__all__ = [
    "AGENT_NAMES",
    "RegionFeatures",
    "RoundTrace",
    "CollaborationTrace",
    "Orchestrator",
    "LLMCaller",
]
