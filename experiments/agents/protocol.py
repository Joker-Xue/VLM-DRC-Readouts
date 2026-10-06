"""C1 — Evidence-Grounded Debate Protocol.

Implements the typed JSON message schema (proposal §3.3) and three-round
collaboration rules (proposal §3.4).

Agent identities (Phase 1.4 refresh, English-only for paper publication):
- ``congestion_analyst``         — Routing Congestion Analyst (4 overflow channels)
- ``wire_demand_forecaster``     — RUDY-based wire demand forecaster (3 RUDY channels)
- ``layout_arbiter``             — Static-layout arbiter (2 own channels + others' evidence)

Design principles:
- The protocol layer reads only structured JSON fields. Free-form natural language
  (``rationale``) is allowed but ignored by the arbiter.
- Messages that fail schema validation are rejected; the orchestrator retries
  (up to ``max_retries`` times) before falling back to an automatic ``abstain``.
- The arbiter votes based on others' ``evidence`` fields, never on ``rationale``.
"""
from __future__ import annotations

from typing import Any

try:
    from jsonschema import Draft7Validator
except ImportError:  # pragma: no cover - jsonschema is a Phase 0 dependency
    Draft7Validator = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Agent identifier registry
# ---------------------------------------------------------------------------

AGENT_NAMES: tuple[str, ...] = (
    "congestion_analyst",
    "wire_demand_forecaster",
    "drc_pattern_forensics",
    "layout_arbiter",
)

# Specialists see their own narrow data and speak first in each round.
# Note: drc_pattern_forensics is a *broad-view* specialist (sees all 9 channels);
# the other two are *narrow* specialists (3-4 channels each). All three speak
# before the arbiter.
SPECIALIST_AGENTS: set[str] = {
    "congestion_analyst",
    "wire_demand_forecaster",
    "drc_pattern_forensics",
}
NARROW_SPECIALIST_AGENTS: set[str] = {
    "congestion_analyst",
    "wire_demand_forecaster",
}
BROAD_SPECIALIST_AGENTS: set[str] = {"drc_pattern_forensics"}
ARBITER_AGENTS: set[str] = {"layout_arbiter"}


# ---------------------------------------------------------------------------
# CircuitNet-N28 DRC feature channels (matches feature_extraction.CHANNELS)
# ---------------------------------------------------------------------------

VALID_CHANNELS: list[str] = [
    "macro_region",
    "cell_density",
    "RUDY_long",
    "RUDY_short",
    "RUDY_pin_long",
    "congestion_eGR_horizontal_overflow",
    "congestion_eGR_vertical_overflow",
    "congestion_GR_horizontal_overflow",
    "congestion_GR_vertical_overflow",
]


# ---------------------------------------------------------------------------
# Schema (JSON Schema draft-07)
# ---------------------------------------------------------------------------

AGENT_MESSAGE_SCHEMA: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "AgentMessage",
    "type": "object",
    "required": ["message_type", "agent", "round"],
    "additionalProperties": False,
    "properties": {
        "message_type": {
            "enum": [
                "hotspot_proposal",
                "not_hotspot",
                "counter_claim",
                "vote",
                "evidence_request",
                "abstain",
            ]
        },
        "agent": {"enum": list(AGENT_NAMES)},
        "round": {"type": "integer", "minimum": 1, "maximum": 3},
        "region": {
            "type": "object",
            "required": ["x", "y", "radius"],
            "properties": {
                "x": {"type": "integer", "minimum": 0},
                "y": {"type": "integer", "minimum": 0},
                "radius": {"type": "integer", "minimum": 1},
            },
        },
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["channel", "value", "threshold"],
                "additionalProperties": False,
                "properties": {
                    "channel": {"enum": VALID_CHANNELS},
                    "value": {"type": "number"},
                    "threshold": {"type": "number"},
                    "exceeds_by": {"type": "number"},
                    "source_def_line": {"type": "integer"},
                },
            },
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "vote": {"enum": ["hotspot", "not_hotspot", "abstain"]},
        "depends_on_agents": {
            "type": "array",
            "items": {"enum": list(AGENT_NAMES)},
        },
        "rationale": {"type": "string", "maxLength": 1200},
        # Optional rule-type forensic tag — currently produced only by
        # `drc_pattern_forensics` when proposing/voting hotspot. Free-form string
        # to allow models to output rule names outside the canonical set; the
        # orchestrator/eval can validate against the 25-class taxonomy
        # downstream (see CircuitNet routability_features/DRC/DRC_seperated/).
        "rule_type": {"type": "string", "maxLength": 64},
        # Round-2 counter_claim must quote the rebutted channel/value; the
        # orchestrator performs the cross-agent reference check above the schema.
        "quotes": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["from_agent", "channel", "value"],
                "properties": {
                    "from_agent": {"enum": list(AGENT_NAMES)},
                    "channel": {"type": "string"},
                    "value": {"type": "number"},
                },
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Round-by-round message-type constraints (proposal §3.4)
# ---------------------------------------------------------------------------

ROUND_ALLOWED_TYPES: dict[int, set[str]] = {
    1: {"hotspot_proposal", "not_hotspot", "abstain"},
    2: {"counter_claim", "vote", "abstain"},
    3: {"vote", "abstain"},
}


# ---------------------------------------------------------------------------
# Visibility rules (proposal §3.4 — "information visibility" column)
# Consumed by the orchestrator to assemble per-round context for each agent.
# ---------------------------------------------------------------------------

VISIBILITY_RULES: dict[int, dict[str, list[str]]] = {
    1: {
        # All three specialists speak in parallel in R1; none sees the others.
        "congestion_analyst": ["own_channel"],
        "wire_demand_forecaster": ["own_channel"],
        "drc_pattern_forensics": ["own_channel"],
        # Arbiter goes last in R1 so it can read every specialist's R1 message.
        "layout_arbiter": ["own_channel", "specialists_round1"],
    },
    2: {
        "congestion_analyst": ["own_channel", "all_round1"],
        "wire_demand_forecaster": ["own_channel", "all_round1"],
        "drc_pattern_forensics": ["own_channel", "all_round1"],
        "layout_arbiter": ["own_channel", "all_round1"],
    },
    3: {
        "congestion_analyst": ["own_channel", "all_round1", "all_round2"],
        "wire_demand_forecaster": ["own_channel", "all_round1", "all_round2"],
        "drc_pattern_forensics": ["own_channel", "all_round1", "all_round2"],
        "layout_arbiter": ["own_channel", "all_round1", "all_round2"],
    },
}


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


def validate_message(msg: dict[str, Any]) -> tuple[bool, list[str]]:
    """Return ``(ok, errors)``.

    ``errors`` are short feedback strings suitable for prompting the LLM to
    retry. The orchestrator collects these and re-injects them into the next
    attempt's context.
    """
    if Draft7Validator is None:
        return False, ["jsonschema not installed: pip install jsonschema"]

    validator = Draft7Validator(AGENT_MESSAGE_SCHEMA)
    schema_errors = sorted(validator.iter_errors(msg), key=lambda e: e.path)
    errors: list[str] = []

    for err in schema_errors:
        path = ".".join(str(p) for p in err.path) or "(root)"
        errors.append(f"[schema] {path}: {err.message}")

    # Cross-field constraints (not expressible in plain JSON Schema)
    rnd = msg.get("round")
    mtype = msg.get("message_type")
    if rnd in ROUND_ALLOWED_TYPES and mtype not in ROUND_ALLOWED_TYPES[rnd]:
        allowed = sorted(ROUND_ALLOWED_TYPES[rnd])
        errors.append(
            f"[round-rule] round={rnd} disallows message_type={mtype}; allowed: {allowed}"
        )

    # R2 counter_claim must quote the rebutted evidence
    if mtype == "counter_claim" and not msg.get("quotes"):
        errors.append(
            "[round-rule] round=2 counter_claim must populate 'quotes' (cite the channel+value being rebutted)"
        )

    # An update vote (vote != prev_vote) must credit who changed your mind
    if mtype == "vote" and rnd == 2 and msg.get("vote") != msg.get("prev_vote"):
        # prev_vote is not part of the schema; the orchestrator injects it for
        # the cross-field check.
        if not msg.get("depends_on_agents"):
            errors.append(
                "[round-rule] round=2 vote change must populate 'depends_on_agents' (who convinced you)"
            )

    # Specialists may not 'vote' directly in R1 (only propose/not_hotspot/abstain)
    if rnd == 1 and msg.get("agent") in SPECIALIST_AGENTS and mtype == "vote":
        errors.append(
            "[round-rule] specialists in round=1 must use hotspot_proposal / not_hotspot / abstain (not 'vote')"
        )

    return (len(errors) == 0), errors


def allowed_message_types(round_num: int) -> set[str]:
    return ROUND_ALLOWED_TYPES.get(round_num, set())


def visible_context(agent: str, round_num: int) -> list[str]:
    return VISIBILITY_RULES.get(round_num, {}).get(agent, [])


# ---------------------------------------------------------------------------
# Early-stop / escalation state machine (C3)
# ---------------------------------------------------------------------------


def aggregate_vote(messages: list[dict[str, Any]]) -> dict[str, int]:
    """Aggregate one round's messages into vote counts."""
    counts = {"hotspot": 0, "not_hotspot": 0, "abstain": 0}
    for msg in messages:
        if msg.get("message_type") == "hotspot_proposal":
            counts["hotspot"] += 1
        elif msg.get("message_type") == "not_hotspot":
            counts["not_hotspot"] += 1
        elif msg.get("message_type") == "abstain":
            counts["abstain"] += 1
        elif msg.get("message_type") == "vote":
            v = msg.get("vote", "abstain")
            counts[v] = counts.get(v, 0) + 1
    return counts


def decide_next(
    round_num: int,
    current_counts: dict[str, int],
    prev_counts: dict[str, int] | None = None,
) -> str:
    """Return ``'early_stop'`` / ``'continue'`` / ``'must_continue_r3'`` / ``'escalate'``.

    Implements proposal §3.4 transition table.
    """
    h, n = current_counts.get("hotspot", 0), current_counts.get("not_hotspot", 0)
    if round_num == 1:
        if h == 3 or n == 3:
            return "early_stop"
        return "continue"
    if round_num == 2:
        if h == 3 or n == 3:
            return "early_stop"
        # Vote flip from R1 → R2 forces R3 reflection (proposal §3.4)
        if prev_counts and ((h > n) != (prev_counts.get("hotspot", 0) > prev_counts.get("not_hotspot", 0))):
            return "must_continue_r3"
        return "continue"
    if round_num == 3:
        if h == 3 or n == 3 or ((h + n) >= 2 and abs(h - n) >= 1):
            return "early_stop"
        return "escalate"
    return "continue"


__all__ = [
    "AGENT_MESSAGE_SCHEMA",
    "AGENT_NAMES",
    "ARBITER_AGENTS",
    "BROAD_SPECIALIST_AGENTS",
    "NARROW_SPECIALIST_AGENTS",
    "ROUND_ALLOWED_TYPES",
    "SPECIALIST_AGENTS",
    "VALID_CHANNELS",
    "VISIBILITY_RULES",
    "validate_message",
    "allowed_message_types",
    "visible_context",
    "aggregate_vote",
    "decide_next",
]
