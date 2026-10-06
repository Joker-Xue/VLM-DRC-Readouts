"""Sample regions from the CircuitNet-N28 test split for Phase 1 baseline evaluation.

Default strategy:
- Read ``test_N28.csv`` (3164 patches) from the cloned CircuitNet tools dir.
- Random-sample N patches (seeded).
- For each patch, pick the region with the highest DRC-label box-mean (radius=8).

This biases toward hotspot-containing regions, which is fine for a first PR-AUC
sanity check (and for keeping per-region label balanced).
"""
from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from experiments.feature_extraction.extract import load_circuitnet_patch


N28_ROOT = Path(__file__).parent.parent.parent / "data" / "circuitnet_n28"
N28_TRAINING_SET = N28_ROOT / "training_set" / "DRC"
TEST_SPLIT_CSV = (
    N28_ROOT
    / "raw"
    / "tools"
    / "routability_ir_drop_prediction"
    / "files"
    / "test_N28.csv"
)


@dataclass
class SampledRegion:
    patch_id: str
    design: str
    region_xy: dict[str, int]                     # {'x', 'y', 'radius'}
    features_per_agent: dict[str, dict[str, float]]  # {agent_name: {channel: value}}
    features_flat: dict[str, float]               # all 9 channels (for B2/B4)
    drc_label_strength: float                     # smoothed box mean of label
    is_hotspot: bool                              # binary ground truth at threshold


def _load_test_patch_ids() -> list[str]:
    if not TEST_SPLIT_CSV.exists():
        # Fallback: iterate all patches present in training_set/DRC/feature/
        feat_dir = N28_TRAINING_SET / "feature"
        return sorted(p.stem for p in feat_dir.glob("*.npy"))
    ids: list[str] = []
    with TEST_SPLIT_CSV.open() as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            # Format: feature/<patch_id>.npy,label/<patch_id>.npy
            stem = Path(row[0]).stem
            ids.append(stem)
    return ids


def _box_mean_argmax(label: np.ndarray, radius: int) -> tuple[int, int, float]:
    h, w = label.shape
    pad = np.pad(label, radius, mode="edge")
    cum = pad.cumsum(0).cumsum(1)
    cum = np.pad(cum, ((1, 0), (1, 0)))
    side = 2 * radius + 1
    s = (
        cum[side:, side:]
        - cum[:-side, side:]
        - cum[side:, :-side]
        + cum[:-side, :-side]
    )
    smoothed = s / (side * side)
    flat = int(smoothed.argmax())
    cy, cx = divmod(flat, w)
    return int(cx), int(cy), float(smoothed[cy, cx])


def sample_regions(
    *,
    n_patches: int = 100,
    seed: int = 42,
    radius: int = 8,
    hotspot_threshold: float = 0.005,
) -> Iterator[SampledRegion]:
    """Yield ``n_patches`` :class:`SampledRegion` instances.

    Args:
        n_patches: how many patches (≡ regions) to sample.
        seed: RNG seed for patch selection.
        radius: half-side of the box-mean region used to score each patch.
        hotspot_threshold: ``is_hotspot = label_strength > threshold``.
            Default 0.005 ≈ "≥ 1 violation per region cell on average".
    """
    feat_dir = N28_TRAINING_SET / "feature"
    lbl_dir = N28_TRAINING_SET / "label"
    if not feat_dir.exists():
        raise FileNotFoundError(f"{feat_dir} missing — did generate_training_set.py run?")

    all_ids = _load_test_patch_ids()
    rng = random.Random(seed)
    picks = rng.sample(all_ids, k=min(n_patches, len(all_ids)))

    for pid in picks:
        feat_path = feat_dir / f"{pid}.npy"
        lbl_path = lbl_dir / f"{pid}.npy"
        if not feat_path.exists() or not lbl_path.exists():
            continue
        patch = load_circuitnet_patch(feat_path, lbl_path)
        if patch.drc_label is None:
            continue
        cx, cy, strength = _box_mean_argmax(patch.drc_label, radius=radius)
        region = patch.region_at(x=cx, y=cy, radius=radius)
        per_agent = {
            "congestion_analyst": region.congestion,
            "wire_demand_forecaster": region.wire_demand,
            "layout_arbiter": region.layout,
        }
        flat = {**region.congestion, **region.wire_demand, **region.layout}

        yield SampledRegion(
            patch_id=patch.patch_id,
            design=patch.design,
            region_xy={"x": cx, "y": cy, "radius": radius},
            features_per_agent=per_agent,
            features_flat=flat,
            drc_label_strength=strength,
            is_hotspot=bool(strength > hotspot_threshold),
        )


__all__ = ["SampledRegion", "sample_regions", "N28_TRAINING_SET", "TEST_SPLIT_CSV"]
