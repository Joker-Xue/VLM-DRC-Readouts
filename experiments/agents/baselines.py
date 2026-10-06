"""Single-LLM (B2/B3) and simple-voting (B4/B5) baselines.

These are the comparison points the proposal calls out (§4.2):
- **B2/B3** — one LLM sees the full 9-channel feature vector and emits a
  hotspot decision. No agent decomposition, no protocol layer.
- **B4/B5** — three LLM calls with the same prompt and a tiny diversity seed
  (so the cache hashes differ); aggregate by majority vote.

Outputs follow a uniform :class:`BaselineDecision` dataclass so the eval
runner can compare them apples-to-apples against the M1 multi-agent path.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from experiments.agents.llm_clients import _extract_json_object, retry_with_backoff


PROMPTS_DIR = Path(__file__).parent / "prompts"

SINGLE_LLM_PROMPT_PATH = PROMPTS_DIR / "baseline_single_llm.md"


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass
class BaselineDecision:
    """Uniform output for all baseline methods."""

    method: str                                   # 'B2', 'B4', 'M1', ...
    region: dict[str, int]                        # {'x','y','radius'}
    decision: str                                 # 'hotspot' / 'not_hotspot' / 'abstain'
    score: float                                  # predicted P(hotspot) ∈ [0, 1]
    raw_messages: list[dict[str, Any]] = field(default_factory=list)
    api_calls: int = 0
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# B2 / B3 — Single-LLM baseline
# ---------------------------------------------------------------------------


def _format_single_llm_messages(system: str, payload: dict[str, Any]) -> list[dict[str, str]]:
    user_content = (
        "Region under inspection (all 9 channel mean values for the cropped window):\n"
        f"```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```\n"
        "\nIssue your decision as JSON only."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


@retry_with_backoff(max_attempts=3, base_delay=1.0)
def _call_deepseek_single(system: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Direct DeepSeek call without the persona-cache decorator (we want temperature control)."""
    import os
    from openai import OpenAI

    api_key = os.environ["DEEPSEEK_API_KEY"]
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=_format_single_llm_messages(system, payload),
        response_format={"type": "json_object"},
        temperature=payload.get("_temperature", 0.3),
        max_tokens=512,
    )
    raw = response.choices[0].message.content or ""
    return _extract_json_object(raw)


def run_b2_single_llm(
    *,
    region_features: dict[str, float],
    region_xy: dict[str, int],
    caller: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    prompt_path: Path = SINGLE_LLM_PROMPT_PATH,
) -> BaselineDecision:
    """Run the B2 baseline once on a single region.

    Args:
        region_features: dict mapping each of the 9 channel names to its mean value.
        region_xy: ``{'x', 'y', 'radius'}`` for the region.
        caller: optional override (mainly for tests); default uses DeepSeek.
        prompt_path: persona file for the single-LLM judge.
    """
    if caller is None:
        caller = _call_deepseek_single

    system = prompt_path.read_text()
    payload = {
        "region": region_xy,
        "features": region_features,
        "_temperature": 0.3,
    }

    t0 = time.time()
    try:
        msg = caller(system, payload)
    except Exception as err:
        return BaselineDecision(
            method="B2",
            region=region_xy,
            decision="abstain",
            score=0.5,
            raw_messages=[{"error": f"{type(err).__name__}: {err}"}],
            api_calls=1,
            elapsed_seconds=time.time() - t0,
        )

    decision = str(msg.get("decision", "abstain"))
    confidence = float(msg.get("confidence", 0.5))
    # Convert decision + confidence into a P(hotspot) score
    if decision == "hotspot":
        score = confidence
    elif decision == "not_hotspot":
        score = 1.0 - confidence
    else:
        score = 0.5

    return BaselineDecision(
        method="B2",
        region=region_xy,
        decision=decision,
        score=score,
        raw_messages=[msg],
        api_calls=1,
        elapsed_seconds=time.time() - t0,
    )


# ---------------------------------------------------------------------------
# B4 / B5 — Simple-voting baseline (3 LLM, majority vote)
# ---------------------------------------------------------------------------


def run_b4_simple_voting(
    *,
    region_features: dict[str, float],
    region_xy: dict[str, int],
    n_voters: int = 3,
    caller: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    prompt_path: Path = SINGLE_LLM_PROMPT_PATH,
    temperature: float = 0.7,
) -> BaselineDecision:
    """Run the B4 baseline: ``n_voters`` LLM calls with diversity seed, majority vote."""
    if caller is None:
        caller = _call_deepseek_single

    system = prompt_path.read_text()
    raw_messages: list[dict[str, Any]] = []
    scores: list[float] = []
    decisions: list[str] = []

    t0 = time.time()
    for voter_idx in range(n_voters):
        payload = {
            "region": region_xy,
            "features": region_features,
            "_temperature": temperature,
            "voter_idx": voter_idx,  # forces cache miss + nudges diversity
        }
        try:
            msg = caller(system, payload)
        except Exception as err:
            msg = {"error": f"{type(err).__name__}: {err}", "voter_idx": voter_idx}
            decisions.append("abstain")
            scores.append(0.5)
            raw_messages.append(msg)
            continue

        raw_messages.append({**msg, "voter_idx": voter_idx})
        d = str(msg.get("decision", "abstain"))
        c = float(msg.get("confidence", 0.5))
        decisions.append(d)
        if d == "hotspot":
            scores.append(c)
        elif d == "not_hotspot":
            scores.append(1.0 - c)
        else:
            scores.append(0.5)

    hotspot_votes = sum(1 for d in decisions if d == "hotspot")
    not_hotspot_votes = sum(1 for d in decisions if d == "not_hotspot")
    if hotspot_votes > not_hotspot_votes:
        decision = "hotspot"
    elif not_hotspot_votes > hotspot_votes:
        decision = "not_hotspot"
    else:
        decision = "abstain"
    mean_score = sum(scores) / max(len(scores), 1)

    return BaselineDecision(
        method="B4",
        region=region_xy,
        decision=decision,
        score=mean_score,
        raw_messages=raw_messages,
        api_calls=n_voters,
        elapsed_seconds=time.time() - t0,
    )


__all__ = [
    "BaselineDecision",
    "SINGLE_LLM_PROMPT_PATH",
    "run_b2_single_llm",
    "run_b4_simple_voting",
]
