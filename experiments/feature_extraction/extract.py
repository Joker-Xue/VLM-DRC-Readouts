"""Feature extraction — CircuitNet-N28 (9 channels) → three-agent view.

Data sources:
1. **CircuitNet-N28** (primary benchmark): the (256, 256, 9) tensors produced by
   ``generate_training_set.py --task DRC``.
2. **OpenROAD-flow-scripts self-run** (Phase 2, deferred): own DRC runs from
   open-source RTL → 9-channel features.

Agent identifiers (Phase 1.4 refresh, matches ``protocol.AGENT_NAMES``):
- ``congestion_analyst``         — 4 congestion overflow channels
- ``wire_demand_forecaster``     — 3 RUDY channels
- ``layout_arbiter``             — 2 layout-geometry channels
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# CircuitNet-N28 actual 9 DRC feature channels
# (order matches generate_training_set.py output along the last axis)
# ---------------------------------------------------------------------------

CHANNELS: tuple[str, ...] = (
    "macro_region",                        # 0 macro/blockage geometry mask
    "cell_density",                        # 1 standard-cell occupancy density
    "RUDY_long",                           # 2 long-wire density estimate
    "RUDY_short",                          # 3 short-wire density estimate
    "RUDY_pin_long",                       # 4 long-wire density near pins
    "congestion_eGR_horizontal_overflow",  # 5 early Global Routing horizontal overflow
    "congestion_eGR_vertical_overflow",    # 6 early Global Routing vertical overflow
    "congestion_GR_horizontal_overflow",   # 7 detailed Global Routing horizontal overflow
    "congestion_GR_vertical_overflow",     # 8 detailed Global Routing vertical overflow
)

CHANNEL_INDEX = {name: i for i, name in enumerate(CHANNELS)}

# Label channel: DRC violation count clipped to [0, 200] then divided by 200.
LABEL_CHANNEL = "DRC_all"


# ---------------------------------------------------------------------------
# Four-agent ↔ nine-channel mapping (Phase 2-prep, Option F+):
#
# - congestion_analyst:       4 congestion overflow channels (narrow specialist)
# - wire_demand_forecaster:   3 RUDY channels                (narrow specialist)
# - drc_pattern_forensics:    ALL 9 channels (broad specialist; multi-task LoRA
#                             trained with rule-type auxiliary supervision)
# - layout_arbiter:           macro_region + cell_density    (arbiter; sees own
#                             2 channels + others' evidence fields)
#
# The three NARROW specialists ({congestion, wire_demand, layout_arbiter}) form
# a disjoint, exhaustive partition over the 9 channels. drc_pattern_forensics
# is the broad-view specialist that intentionally OVERLAPS them — its
# differentiation comes from (1) multi-task training (rule-type aux head) and
# (2) cross-channel pattern matching that narrow specialists structurally cannot do.
# ---------------------------------------------------------------------------

NARROW_AGENT_CHANNELS: dict[str, tuple[str, ...]] = {
    "congestion_analyst": (
        "congestion_eGR_horizontal_overflow",
        "congestion_eGR_vertical_overflow",
        "congestion_GR_horizontal_overflow",
        "congestion_GR_vertical_overflow",
    ),
    "wire_demand_forecaster": (
        "RUDY_long",
        "RUDY_short",
        "RUDY_pin_long",
    ),
    "layout_arbiter": (
        "macro_region",
        "cell_density",
    ),
}

AGENT_CHANNELS: dict[str, tuple[str, ...]] = {
    **NARROW_AGENT_CHANNELS,
    # Broad specialist sees the full nine-channel summary.
    "drc_pattern_forensics": CHANNELS,
}


@dataclass
class PatchFeatures:
    """A 256×256 CircuitNet patch with its full nine-channel feature stack."""

    patch_id: str
    design: str
    process: str  # 'N28' / 'N14'
    channels: dict[str, np.ndarray]
    drc_label: np.ndarray | None = None   # shape (256, 256), normalized DRC violation count

    def for_agent(self, agent: str) -> dict[str, np.ndarray]:
        keys = AGENT_CHANNELS[agent]
        return {k: self.channels[k] for k in keys if k in self.channels}

    def region_at(self, x: int, y: int, radius: int = 4) -> "RegionFeatures":
        """Crop a square region centred at (x, y) and aggregate per channel.

        Returns a :class:`RegionFeatures` populated with each agent's view
        (mean over the cropped window).
        """
        from experiments.agents.orchestrator import RegionFeatures  # avoid circular import

        def crop_mean(arr: np.ndarray) -> float:
            x0, x1 = max(0, x - radius), min(arr.shape[-1], x + radius + 1)
            y0, y1 = max(0, y - radius), min(arr.shape[-2], y + radius + 1)
            return float(arr[..., y0:y1, x0:x1].mean())

        def channel_view(agent: str) -> dict[str, float]:
            return {
                k: crop_mean(self.channels[k])
                for k in AGENT_CHANNELS[agent]
                if k in self.channels
            }

        return RegionFeatures(
            x=x,
            y=y,
            radius=radius,
            congestion=channel_view("congestion_analyst"),
            wire_demand=channel_view("wire_demand_forecaster"),
            drc_pattern=channel_view("drc_pattern_forensics"),
            layout=channel_view("layout_arbiter"),
        )

    def drc_label_in_region(self, x: int, y: int, radius: int = 4) -> float:
        """Mean DRC-violation label inside the cropped region (NaN if no label)."""
        if self.drc_label is None:
            return float("nan")
        x0, x1 = max(0, x - radius), min(self.drc_label.shape[-1], x + radius + 1)
        y0, y1 = max(0, y - radius), min(self.drc_label.shape[-2], y + radius + 1)
        return float(self.drc_label[..., y0:y1, x0:x1].mean())


# ---------------------------------------------------------------------------
# CircuitNet-N28 loader (output of generate_training_set.py)
# ---------------------------------------------------------------------------


def load_circuitnet_patch(feature_path: Path, label_path: Path | None = None) -> PatchFeatures:
    """Load one preprocessed CircuitNet patch.

    ``feature_path`` points to a ``(256, 256, 9)`` .npy whose channel order
    matches :data:`CHANNELS`. Optional ``label_path`` points to a
    ``(256, 256, 1)`` DRC_all label .npy.

    Patch filenames follow CircuitNet convention, e.g.
    ``8861-zero-riscy-a-3-c5-u0.75-m2-p7-f1.npy``.
    """
    feature_path = Path(feature_path)
    if not feature_path.exists():
        raise FileNotFoundError(feature_path)

    feat = np.load(feature_path, allow_pickle=False)
    if feat.ndim == 3 and feat.shape[-1] == len(CHANNELS):
        # (H, W, C) — generate_training_set output layout
        channels = {name: feat[..., i] for i, name in enumerate(CHANNELS)}
    elif feat.ndim == 3 and feat.shape[0] == len(CHANNELS):
        # (C, H, W) — fallback layout
        channels = {name: feat[i] for i, name in enumerate(CHANNELS)}
    else:
        raise ValueError(
            f"Unexpected feature shape {feat.shape}; expected (H, W, {len(CHANNELS)}) "
            f"or ({len(CHANNELS)}, H, W)"
        )

    drc_label = None
    if label_path is not None:
        label_path = Path(label_path)
        if label_path.exists():
            lbl = np.load(label_path, allow_pickle=False)
            drc_label = lbl.squeeze() if lbl.ndim == 3 else lbl

    # Extract design name from filename (e.g. "8861-zero-riscy-a-3-c5-..." → "zero-riscy")
    name_parts = feature_path.stem.split("-")
    design = "unknown"
    if len(name_parts) >= 3:
        # CircuitNet convention: <id>-<design-name>-..., design may contain "-"
        for i in range(2, len(name_parts)):
            if name_parts[i] in {"a", "b", "c"}:  # config-prefix marker
                design = "-".join(name_parts[1:i])
                break
        else:
            design = name_parts[1]

    return PatchFeatures(
        patch_id=feature_path.stem,
        design=design,
        process="N28",
        channels=channels,
        drc_label=drc_label,
    )


# ---------------------------------------------------------------------------
# OpenROAD-flow-scripts loader (self-curated data, deferred to Phase 2)
# ---------------------------------------------------------------------------


def load_openroad_run(run_dir: Path) -> PatchFeatures:
    """Extract 9-channel features from one OpenROAD-flow-scripts run directory."""
    raise NotImplementedError(
        "Phase 2 task: implemented after running OpenROAD-flow-scripts gcd/aes/ibex; "
        "needs to parse 6_final.def + congestion.rpt and compute RUDY / cell_density / "
        "macro_region locally."
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def load_features(source: str | Path, **kwargs: Any) -> PatchFeatures:
    source = Path(source)
    if source.suffix == ".npy":
        label_path = kwargs.pop("label_path", None)
        if label_path is None:
            # Default sibling layout: <root>/feature/<name>.npy ↔ <root>/label/<name>.npy
            sibling = source.parent.parent / "label" / source.name
            if sibling.exists():
                label_path = sibling
        return load_circuitnet_patch(source, label_path)
    if source.is_dir() and (source / "results").exists():
        return load_openroad_run(source)
    raise ValueError(f"unrecognized source: {source}")


__all__ = [
    "CHANNELS",
    "CHANNEL_INDEX",
    "LABEL_CHANNEL",
    "AGENT_CHANNELS",
    "NARROW_AGENT_CHANNELS",
    "PatchFeatures",
    "load_circuitnet_patch",
    "load_openroad_run",
    "load_features",
]
