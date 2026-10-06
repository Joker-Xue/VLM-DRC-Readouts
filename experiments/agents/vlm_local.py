"""Ⅴ.2 (v3) — local VLM caller: agents decide from RENDERED IMAGES, not text descriptors.

Plugs into the existing Orchestrator / run_b2 path WITHOUT touching protocol.py: it has the
caller signature ``(system_prompt, payload) -> message_dict``, but instead of reading the
payload's text ``features`` it loads the agent's rendered view PNG and runs a local VLM
(Qwen2.5-VL via mlx-vlm, free on M5). Debate HISTORY stays textual (it is inherently
linguistic); only PERCEPTION is visual.

Image routing (the C2 visual information-isolation variable):
  - mode="own"   (A3 heterogeneous): agent sees data/render_cache/<pid>/<agent>.png
  - mode="broad" (A2 homogeneous, A1 single): every agent sees <pid>/drc_pattern_forensics.png
    (all 9 channels + the CNN hotspot-prior panel)

The current patch id is set per-region by run_arms via ``set_patch`` (run_arms is strictly
sequential, so a stateful caller is safe and avoids threading pid through the frozen protocol).

Reproducibility: debate arms sample at temperature>0 (so homogeneous agents actually diverge,
otherwise A2 collapses to 4 identical votes); every call is cached to disk keyed on
(system, user_text, image, agent) → a finished run is fixed and fully resumable.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from experiments.agents.llm_clients import _extract_json_object
from experiments.agents.protocol import AGENT_NAMES

BROAD_AGENT = "drc_pattern_forensics"
_SCHEMA_KEYS = {"message_type", "agent", "round", "region", "evidence", "confidence",
                "vote", "depends_on_agents", "rationale", "rule_type", "quotes"}
_VERDICTS = {"hotspot", "not_hotspot", "abstain"}
DEFAULT_MODEL = os.environ.get(
    "LG_VLM_MODEL",
    str(Path("~/.cache/modelscope/hub/models/mlx-community/Qwen2___5-VL-7B-Instruct-4bit").expanduser()),
)


# ---------------------------------------------------------------------------
# Local VLM (mlx-vlm), lazy-loaded once
# ---------------------------------------------------------------------------


class LocalVLM:
    def __init__(self, model_id: str = DEFAULT_MODEL):
        self.model_id = model_id
        self._loaded = False

    def _ensure(self):
        if self._loaded:
            return
        from mlx_vlm import load
        from mlx_vlm.utils import load_config
        self._model, self._proc = load(self.model_id)
        self._cfg = load_config(self.model_id)
        self._loaded = True
        print(f"[vlm_local] loaded {self.model_id}")

    def generate(self, system: str, user: str, image: Path, *, temperature: float,
                 max_tokens: int = 512) -> str:
        self._ensure()
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        full = f"{system}\n\n{user}"
        prompt = apply_chat_template(self._proc, self._cfg, full, num_images=1)
        out = generate(self._model, self._proc, prompt, image=[str(image)],
                       max_tokens=max_tokens, temperature=temperature, verbose=False)
        return out.text if hasattr(out, "text") else str(out)


# ---------------------------------------------------------------------------
# Payload → user text (region + debate history; perception comes from the image)
# ---------------------------------------------------------------------------


def _fmt_history(history: list[dict[str, Any]]) -> str:
    if not history:
        return "(you are first this round; no peer messages yet)"
    lines = []
    for m in history:
        who = m.get("agent", "?")
        mt = m.get("message_type", "?")
        verdict = (m.get("vote") or ("hotspot" if mt == "hotspot_proposal"
                   else "not_hotspot" if mt == "not_hotspot" else mt))
        conf = m.get("confidence")
        rat = (m.get("rationale") or "").strip().replace("\n", " ")
        if len(rat) > 160:
            rat = rat[:157] + "..."
        c = f" conf={conf}" if conf is not None else ""
        lines.append(f"  - {who}: {verdict}{c} — {rat}")
    return "\n".join(lines)


def build_user_text(payload: dict[str, Any]) -> str:
    region = payload.get("region", {})
    rnd = payload.get("round")
    allowed = payload.get("allowed_message_types")
    parts = [
        f"Round {rnd} of a 3-round visual DRC-hotspot inspection.",
        f"Region under inspection: center=({region.get('x')},{region.get('y')}), "
        f"radius={region.get('radius')} — marked by the RED BOX in your image; the bottom "
        f"panels zoom into exactly that region.",
        "Peers so far this debate:\n" + _fmt_history(payload.get("visible_history", [])),
    ]
    if allowed is not None:
        # hide counter_claim: it's protocol-allowed in R2 but visual agents never use it (it needs
        # numeric quotes they can't give); showing it just tempts the model into a schema failure.
        shown = sorted(t for t in allowed if t != "counter_claim")
        parts.append(f"Allowed message_type this round: {shown} — use exactly one of these.")
    if payload.get("last_errors"):
        parts.append("Your previous message was REJECTED by the schema validator. Fix exactly "
                     "these and resend valid JSON:\n  " + "\n  ".join(payload["last_errors"]))
    parts.append("Look at the image, then emit ONLY your typed JSON message per your instructions.")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Visual caller (Orchestrator / run_b2 compatible)
# ---------------------------------------------------------------------------


def normalize_message(msg: dict, rnd: int, agent: str) -> dict:
    """Coerce a small-VLM's raw output into a schema-valid typed message WITHOUT changing its verdict.

    Fixes the 7B-4bit's observed envelope quirks so they don't become forced abstains:
    re-proposal in R2/R3 → vote (verdict preserved) · R1 stray vote → proposal · wrong dep key
    (`dependency`→`depends_on_agents`) · missing/invalid `depends_on_agents` on R2/R3 votes →
    injected valid peer · confidence list/out-of-range → scalar in [0,1] · drop non-schema keys
    (critical: an extra key fails `additionalProperties:false`). The model's hotspot/not_hotspot
    decision and confidence are never invented — only the envelope is repaired.
    """
    if not isinstance(msg, dict):
        return {"message_type": "abstain", "agent": agent, "round": rnd}
    m = dict(msg)

    # wrong key name for dependencies
    if "depends_on_agents" not in m:
        for alt in ("dependency", "depends_on", "dependencies", "depends"):
            if alt in m:
                v = m.get(alt)
                m["depends_on_agents"] = v if isinstance(v, list) else ([v] if isinstance(v, str) and v else [])
                break
    if "depends_on_agents" in m:
        dep = m["depends_on_agents"]
        m["depends_on_agents"] = [x for x in dep if x in AGENT_NAMES] if isinstance(dep, list) else []

    mt = m.get("message_type")
    if mt in ("hotspot", "not hotspot", "nothotspot", "not-hotspot"):  # verdict used as type
        mt = "hotspot_proposal" if mt == "hotspot" else "not_hotspot"
        m["message_type"] = mt

    if rnd == 1 and mt == "vote":                          # R1 cannot vote → proposal
        m["message_type"] = "hotspot_proposal" if m.get("vote") == "hotspot" else "not_hotspot"
        m.pop("vote", None); mt = m["message_type"]
    if rnd >= 2 and mt in ("hotspot_proposal", "not_hotspot"):  # R2/R3 must vote (keep verdict)
        m["vote"] = "hotspot" if mt == "hotspot_proposal" else "not_hotspot"
        m["message_type"] = mt = "vote"

    if mt == "vote":
        if m.get("vote") not in _VERDICTS:
            m["vote"] = "abstain"
        if rnd >= 2 and not m.get("depends_on_agents"):    # protocol: every R2/R3 vote credits a peer
            m["depends_on_agents"] = [a for a in AGENT_NAMES if a != agent][:1]

    c = m.get("confidence")
    if isinstance(c, list) and c:
        c = c[0]
    if isinstance(c, (int, float)):
        m["confidence"] = max(0.0, min(1.0, float(c)))
    elif "confidence" in m:
        m.pop("confidence")

    if "vote" in m and m["vote"] not in _VERDICTS:         # stray invalid vote on a non-vote msg
        m.pop("vote")
    if "rule_type" in m and not isinstance(m["rule_type"], str):
        m.pop("rule_type")

    m["agent"] = agent
    m["round"] = rnd
    m = {k: v for k, v in m.items() if k in _SCHEMA_KEYS}  # additionalProperties:false safety
    m.setdefault("message_type", "abstain")
    return m


class VisualCaller:
    """A swappable ``(system, payload) -> message`` caller backed by a local VLM + rendered views."""

    def __init__(self, vlm: LocalVLM, render_dir: Path, *, mode: str, arm: str,
                 cache_dir: Path, temperature: float = 0.7, max_tokens: int = 512):
        assert mode in ("own", "broad")
        self.vlm = vlm
        self.render_dir = Path(render_dir)
        self.mode = mode
        self.arm = arm
        self.cache_dir = Path(cache_dir)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.pid: str | None = None

    def set_patch(self, pid: str) -> None:
        self.pid = pid

    def _image_for(self, agent: str) -> Path:
        view = BROAD_AGENT if self.mode == "broad" else agent
        return self.render_dir / self.pid / f"{view}.png"

    def __call__(self, system: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self.pid is None:
            raise RuntimeError("VisualCaller.set_patch(pid) must be called before use")
        is_orch = "agent" in payload                          # orchestrator path vs A1 (run_b2) path
        agent = payload.get("agent") or BROAD_AGENT
        rnd = int(payload.get("round", 1))
        image = self._image_for(agent)
        user = build_user_text(payload)
        # cache key: identity of THIS call (agent distinguishes A2's 4 same-image slots;
        # last_errors in user_text distinguishes retries → fresh sample, no infinite-fail loop)
        key = hashlib.sha256(
            f"{self.arm}|{self.pid}|{agent}|{system}|{user}|{image}".encode("utf-8")
        ).hexdigest()[:32]
        f = self.cache_dir / f"{self.arm}_{self.pid}_{agent}_{key}.json"
        if f.exists():
            return json.loads(f.read_text())

        try:
            raw = self.vlm.generate(system, user, image, temperature=self.temperature,
                                    max_tokens=self.max_tokens)
            msg = _extract_json_object(raw)
            if not isinstance(msg, dict):
                raise ValueError("non-dict JSON")
        except Exception:                                     # never crash the long run; degrade to abstain
            msg = ({"message_type": "abstain", "agent": agent, "round": rnd} if is_orch
                   else {"decision": "abstain", "confidence": 0.5, "rationale": "vlm parse/error"})
        if is_orch:
            # coerce to a schema-valid typed message (verdict preserved); MUST NOT carry extra keys —
            # the orchestrator validates with additionalProperties:false, so no _elapsed_s etc.
            msg = normalize_message(msg, rnd, agent)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(msg, ensure_ascii=False, indent=1))
        return msg


__all__ = ["LocalVLM", "VisualCaller", "build_user_text", "BROAD_AGENT", "DEFAULT_MODEL"]
