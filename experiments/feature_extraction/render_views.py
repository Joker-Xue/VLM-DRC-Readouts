"""Ⅴ.0 (v3 pivot) — render per-agent VISUAL views of a patch+region for VLM agents.

v3 direction (decisions-log 2026-05-31 evening): agents see IMAGES natively instead of
serialized descriptors. Heterogeneity = each agent gets a different rendering of the same
die region (visual information isolation, contribution C2):
  - congestion_analyst     : 4 congestion overflow channels
  - wire_demand_forecaster : 3 RUDY channels
  - layout_arbiter         : 2 layout channels (macro_region, cell_density)
  - drc_pattern_forensics  : all 9 channels + the RouteNet hotspot-prior map (CNN evidence,
                             now rendered as an image — the Hybrid front-end is unchanged)

Each agent view is ONE PNG: for every channel a full-patch panel (region marked with a red
box) over a zoomed region-crop panel (nearest interpolation, 17×17 → readable). Channels
are min-max normalized per patch upstream; rendered with vmin=0, vmax=1 so intensity is
comparable across panels.

Region coords come from the cnn-argmax (label-FREE) evidence cache by default, so the
rendering pipeline stays closed-book. Labels go to meta.json only (never into prompts).

Usage (from paper&experiment/, all M5, free):
    PYTHONPATH=. python3 experiments/feature_extraction/render_views.py --patches 8789-zero-riscy-a-3-c5-u0.8-m2-p7-f0
    PYTHONPATH=. python3 experiments/feature_extraction/render_views.py --limit 100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

BASE = Path(__file__).resolve().parents[2]              # paper&experiment/
sys.path.insert(0, str(BASE))
from experiments.feature_extraction.extract import AGENT_CHANNELS, CHANNEL_INDEX  # noqa: E402

DATAROOT = BASE / "data/circuitnet_n28/training_set/DRC"
CACHE_DIR = BASE / "data/evidence_cache"
DEFAULT_OUT = BASE / "data/render_cache"

# Short panel titles (full channel names don't fit above ~2-inch panels)
SHORT = {
    "macro_region": "macro",
    "cell_density": "cell_density",
    "RUDY_long": "RUDY_long",
    "RUDY_short": "RUDY_short",
    "RUDY_pin_long": "RUDY_pin",
    "congestion_eGR_horizontal_overflow": "eGR_H",
    "congestion_eGR_vertical_overflow": "eGR_V",
    "congestion_GR_horizontal_overflow": "GR_H",
    "congestion_GR_vertical_overflow": "GR_V",
}
CNN_PANEL = "CNN_prior"      # forensics-only extra panel (RouteNet hotspot map)
CMAP = "magma"
MAX_COLS = 5                 # wrap wide views (forensics: 10 panels) into row-pairs


def _zoom_crop(arr: np.ndarray, x: int, y: int, radius: int) -> np.ndarray:
    h, w = arr.shape
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    return arr[y0:y1, x0:x1]


def render_agent_view(
    patch_chw: np.ndarray,
    agent: str,
    x: int,
    y: int,
    radius: int,
    out_path: Path,
    cnn_map: np.ndarray | None = None,
) -> Path:
    """Write one PNG: per-channel full view (red region box) + zoomed region crop."""
    panels: list[tuple[str, np.ndarray]] = [
        (SHORT[ch], patch_chw[CHANNEL_INDEX[ch]]) for ch in AGENT_CHANNELS[agent]
    ]
    if agent == "drc_pattern_forensics" and cnn_map is not None:
        panels.append((CNN_PANEL, cnn_map))

    # chunk panels into row-pairs of ≤ MAX_COLS (full-view row + zoom row per chunk)
    chunks = [panels[i:i + MAX_COLS] for i in range(0, len(panels), MAX_COLS)]
    ncols = max(len(c) for c in chunks)
    nrows = 2 * len(chunks)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.3 * ncols, 2.5 * nrows), squeeze=False)

    for ci, chunk in enumerate(chunks):
        for pi in range(ncols):
            ax_full, ax_zoom = axes[2 * ci][pi], axes[2 * ci + 1][pi]
            if pi >= len(chunk):
                ax_full.axis("off"); ax_zoom.axis("off")
                continue
            title, arr = chunk[pi]
            ax_full.imshow(arr, cmap=CMAP, vmin=0, vmax=1, interpolation="nearest")
            side = 2 * radius + 1
            ax_full.add_patch(Rectangle((x - radius, y - radius), side, side,
                                        fill=False, edgecolor="red", linewidth=1.6))
            ax_full.set_title(title, fontsize=9)
            ax_full.set_xticks([]); ax_full.set_yticks([])

            ax_zoom.imshow(_zoom_crop(arr, x, y, radius), cmap=CMAP, vmin=0, vmax=1,
                           interpolation="nearest")
            ax_zoom.set_title(f"{title} (region zoom)", fontsize=8)
            ax_zoom.set_xticks([]); ax_zoom.set_yticks([])

    fig.suptitle(f"{agent} view — region center=({x},{y}) radius={radius} "
                 f"(red box; intensities are per-patch normalized to [0,1])", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def render_patch(rec: dict, extractor=None, out_dir: Path = DEFAULT_OUT) -> dict:
    """Render all 4 agent views for one evidence-cache record; write meta.json.

    Labels stay in meta.json for eval only — they are never rendered into the images
    or handed to a model prompt (closed-book).
    """
    pid = rec["patch_id"]
    feat = np.load(DATAROOT / "feature" / f"{pid}.npy").transpose(2, 0, 1).astype(np.float32)
    cnn_map = extractor.hotspot_map(feat) if extractor is not None else None
    x, y, radius = int(rec["x"]), int(rec["y"]), int(rec["radius"])

    pdir = out_dir / pid
    paths = {}
    for agent in AGENT_CHANNELS:
        paths[agent] = str(render_agent_view(feat, agent, x, y, radius,
                                             pdir / f"{agent}.png", cnn_map=cnn_map))
    meta = {"patch_id": pid, "x": x, "y": y, "radius": radius,
            "region_select": rec.get("region_select", "?"),
            "is_hotspot": rec["is_hotspot"], "label_strength": rec["label_strength"],
            "cnn_prior_peak": rec["cnn_prior"]["peak"], "views": paths}
    (pdir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test_N28")
    ap.add_argument("--region", choices=["label-argmax", "cnn-argmax"], default="cnn-argmax")
    ap.add_argument("--patches", nargs="*", default=None, help="explicit patch ids")
    ap.add_argument("--limit", type=int, default=0, help="render first N cache records (0=need --patches)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    a = ap.parse_args(argv)

    suffix = "_cnnargmax" if a.region == "cnn-argmax" else ""
    cache = CACHE_DIR / f"{a.split}{suffix}_evidence.jsonl"
    recs = [json.loads(l) for l in cache.read_text().splitlines() if l.strip()]
    if a.patches:
        by_id = {r["patch_id"]: r for r in recs}
        todo = [by_id[p] for p in a.patches]
    elif a.limit:
        todo = recs[:a.limit]
    else:
        raise SystemExit("give --patches or --limit")

    from experiments.feature_extraction.cnn_evidence import CNNEvidenceExtractor
    ex = CNNEvidenceExtractor()
    print(f"renderer ready (device={ex.device}); {len(todo)} patches → {a.out}")
    for i, rec in enumerate(todo, 1):
        meta = render_patch(rec, extractor=ex, out_dir=Path(a.out))
        print(f"  [{i}/{len(todo)}] {rec['patch_id']} lbl={meta['label_strength']:.4f} "
              f"cnn_peak={meta['cnn_prior_peak']:.3f} → {len(meta['views'])} views")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
