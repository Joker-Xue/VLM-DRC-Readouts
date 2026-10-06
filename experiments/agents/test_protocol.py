"""Schema + state-machine tests for protocol.py."""
from __future__ import annotations

import pytest

from experiments.agents.protocol import (
    AGENT_NAMES,
    aggregate_vote,
    decide_next,
    validate_message,
)


def test_valid_r1_proposal():
    msg = {
        "message_type": "hotspot_proposal",
        "agent": "congestion_analyst",
        "round": 1,
        "region": {"x": 42, "y": 12, "radius": 3},
        "evidence": [
            {
                "channel": "congestion_GR_horizontal_overflow",
                "value": 0.78,
                "threshold": 0.5,
                "exceeds_by": 0.28,
            }
        ],
        "confidence": 0.78,
    }
    ok, errors = validate_message(msg)
    assert ok, errors


def test_r1_disallows_counter_claim():
    msg = {
        "message_type": "counter_claim",  # disallowed in R1
        "agent": "congestion_analyst",
        "round": 1,
    }
    ok, errors = validate_message(msg)
    assert not ok
    assert any("round-rule" in e for e in errors)


def test_r2_counter_must_quote():
    msg = {
        "message_type": "counter_claim",
        "agent": "wire_demand_forecaster",
        "round": 2,
        "region": {"x": 1, "y": 1, "radius": 1},
        # missing 'quotes' field
    }
    ok, errors = validate_message(msg)
    assert not ok
    assert any("quotes" in e for e in errors)


def test_unknown_message_type_rejected():
    msg = {
        "message_type": "rant",
        "agent": "congestion_analyst",
        "round": 1,
    }
    ok, errors = validate_message(msg)
    assert not ok


def test_unknown_agent_rejected():
    msg = {
        "message_type": "hotspot_proposal",
        "agent": "via_specialist",  # legacy name — must be rejected
        "round": 1,
        "region": {"x": 1, "y": 1, "radius": 1},
    }
    ok, errors = validate_message(msg)
    assert not ok


def test_unknown_channel_rejected():
    msg = {
        "message_type": "hotspot_proposal",
        "agent": "congestion_analyst",
        "round": 1,
        "region": {"x": 1, "y": 1, "radius": 1},
        "evidence": [
            {"channel": "gcell_overflow", "value": 0.5, "threshold": 0.3}  # legacy channel
        ],
    }
    ok, errors = validate_message(msg)
    assert not ok


def test_agent_names_registry_exposed():
    assert set(AGENT_NAMES) == {
        "congestion_analyst",
        "wire_demand_forecaster",
        "drc_pattern_forensics",
        "layout_arbiter",
    }


def test_drc_pattern_forensics_accepted():
    """4th agent must be a first-class member of the schema enum."""
    msg = {
        "message_type": "hotspot_proposal",
        "agent": "drc_pattern_forensics",
        "round": 1,
        "region": {"x": 10, "y": 10, "radius": 8},
        "evidence": [
            {"channel": "RUDY_pin_long", "value": 0.7, "threshold": 0.5, "exceeds_by": 0.2}
        ],
        "confidence": 0.7,
        "rule_type": "MetalShort",
    }
    ok, errors = validate_message(msg)
    assert ok, errors


def test_aggregate_vote_counts():
    messages = [
        {"message_type": "hotspot_proposal", "agent": "congestion_analyst", "round": 1},
        {"message_type": "not_hotspot", "agent": "wire_demand_forecaster", "round": 1},
        {"message_type": "hotspot_proposal", "agent": "drc_pattern_forensics", "round": 1},
        {"message_type": "hotspot_proposal", "agent": "layout_arbiter", "round": 1},
    ]
    counts = aggregate_vote(messages)
    assert counts == {"hotspot": 3, "not_hotspot": 1, "abstain": 0}


def test_decide_next_r1_unanimous_early_stop():
    counts = {"hotspot": 3, "not_hotspot": 0, "abstain": 0}
    assert decide_next(1, counts) == "early_stop"


def test_decide_next_r2_flip_must_continue_r3():
    prev = {"hotspot": 1, "not_hotspot": 2}  # R1 leans not_hotspot
    curr = {"hotspot": 2, "not_hotspot": 1}  # R2 flips
    assert decide_next(2, curr, prev) == "must_continue_r3"


def test_decide_next_r3_tied_escalates():
    counts = {"hotspot": 0, "not_hotspot": 0, "abstain": 3}
    assert decide_next(3, counts) == "escalate"


def _consensus_caller(system, payload):
    """Fake caller producing R1 consensus: 3 agents propose hotspot, 1 abstains.

    decide_next early-stops at R1 when exactly 3 votes land on one side
    (h == 3). For rounds 2/3 (only reached when force_all_rounds bypasses the
    early stop) every agent simply abstains — abstain is valid in all rounds,
    so the orchestrator runs to completion.
    """
    rnd = payload.get("round", 1)
    agent = payload["agent"]
    if rnd == 1 and agent != "wire_demand_forecaster":
        return {
            "message_type": "hotspot_proposal",
            "agent": agent,
            "round": 1,
            "region": payload["region"],
            "confidence": 0.9,
            "rationale": "hotspot",
        }
    return {
        "message_type": "abstain",
        "agent": agent,
        "round": rnd,
        "rationale": "abstain",
    }


def test_force_all_rounds_runs_three_rounds_despite_consensus():
    """force_all_rounds=True must run all 3 rounds even when R1 reaches
    unanimous consensus (which would normally early_stop).

    Regression guard (2026-05-28): early stopping yields <3 captures per
    agent, which breaks trl.PPOTrainer.step's fixed batch_size=max_rounds.
    """
    from experiments.agents.orchestrator import Orchestrator, RegionFeatures

    region = RegionFeatures(
        x=10, y=10, radius=8,
        congestion={"congestion_GR_horizontal_overflow": 0.8},
        wire_demand={"RUDY_long": 0.7},
        drc_pattern={"cell_density": 0.6},
        layout={"macro_region": 0.5},
    )
    prompts = {a: "sys" for a in [
        "congestion_analyst", "wire_demand_forecaster",
        "drc_pattern_forensics", "layout_arbiter",
    ]}

    # Default (early stop allowed): should stop at round 1
    orch_default = Orchestrator(llm_caller=_consensus_caller, prompts=prompts)
    trace_default = orch_default.run(region)
    assert len(trace_default.rounds) == 1, "consensus should early-stop at R1"

    # force_all_rounds=True: should run all 3 rounds
    orch_forced = Orchestrator(
        llm_caller=_consensus_caller, prompts=prompts, force_all_rounds=True,
    )
    trace_forced = orch_forced.run(region)
    assert len(trace_forced.rounds) == 3, (
        f"force_all_rounds must run 3 rounds, got {len(trace_forced.rounds)}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
