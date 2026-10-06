"""CNN-backed evidence extractor (v2 Hybrid) — replaces the box-mean `.mean()` in
``extract.py:region_at`` that destroyed spatial pattern (see proposal-v2-hybrid.md).

For a patch + region it produces STRUCTURED, CITABLE spatial evidence the agents can
debate over, instead of one scalar mean per channel:
  - `cnn_prior`   : RouteNet (frozen, learned perception) hotspot prediction in the
                    region — peak / mean / p90 / area / peak-offset / n_blobs. This is
                    the *learned* perception signal; it is evidence, NOT the final call.
  - `per_agent`   : per-channel structured descriptors (peak, mean, p90, std, gradient,
                    peak-offset) for each agent's own channels (heterogeneous views).

Runs on M5 (MPS/CPU), free. Loads routenet.py directly to avoid the mmcv dep in
models/__init__ (via mavi.py). Channel names/order + agent→channel map are imported
from extract.py to stay in sync with the existing pipeline.

Demo:
    python3 experiments/feature_extraction/cnn_evidence.py   # OLD box-mean vs NEW evidence
"""
from __future__ import annotations
import importlib.util, sys
from pathlib import Path
import numpy as np
import torch

BASE = Path(__file__).resolve().parents[2]                       # paper&experiment/
sys.path.insert(0, str(BASE))
from experiments.feature_extraction.extract import CHANNELS, CHANNEL_INDEX, AGENT_CHANNELS

_CN = BASE / ".thirdparty/CircuitNet/routability_ir_drop_prediction"
RN_PY = _CN / "models/routenet.py"
DEF_WEIGHTS = BASE / ".thirdparty/CircuitNet/pretrained_weights/DRC.pth"


def _descr(crop: np.ndarray) -> dict:
    """Structured spatial descriptor of a region crop (preserves pattern, unlike mean)."""
    c = crop.shape[0] // 2
    pi = np.unravel_index(int(crop.argmax()), crop.shape)
    gy, gx = np.gradient(crop) if min(crop.shape) > 1 else (np.zeros_like(crop), np.zeros_like(crop))
    return {
        "peak": round(float(crop.max()), 4),
        "mean": round(float(crop.mean()), 5),
        "p90": round(float(np.percentile(crop, 90)), 4),
        "std": round(float(crop.std()), 4),
        "grad": round(float(np.sqrt(gx ** 2 + gy ** 2).mean()), 4),
        "peak_off": [int(pi[1] - c), int(pi[0] - c)],   # (dx, dy) from region center
    }


def _cnn_prior(crop: np.ndarray, thr: float = 0.1) -> dict:
    d = _descr(crop)
    above = crop >= thr
    d["area_frac>=%.2f" % thr] = round(float(above.mean()), 4)
    try:
        from scipy.ndimage import label as _label
        d["n_blobs"] = int(_label(above)[1])
    except Exception:
        d["n_blobs"] = None
    return d


class CNNEvidenceExtractor:
    def __init__(self, weights: Path = DEF_WEIGHTS, device: str | None = None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        spec = importlib.util.spec_from_file_location("routenet", str(RN_PY))
        rn = importlib.util.module_from_spec(spec); spec.loader.exec_module(rn)
        ckpt = torch.load(weights, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
        in_ch = next(v for k, v in sd.items() if v.ndim == 4).shape[1]
        self.model = rn.RouteNet(in_channels=in_ch, out_channels=1)
        self.model.load_state_dict(sd, strict=True)
        self.model.to(self.device).eval()
        self.in_channels = in_ch

    def hotspot_map(self, patch_chw: np.ndarray) -> np.ndarray:
        """patch_chw: (9, H, W) float → RouteNet DRC hotspot map (H, W) in [0,1]."""
        x = torch.from_numpy(np.ascontiguousarray(patch_chw)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            y = self.model(x).squeeze().clamp(0, 1)
        return y.float().cpu().numpy()

    def region_evidence(self, patch_chw: np.ndarray, x: int, y: int, radius: int = 8,
                        cnn_map: np.ndarray | None = None) -> dict:
        if cnn_map is None:
            cnn_map = self.hotspot_map(patch_chw)
        H, W = patch_chw.shape[-2:]
        x0, x1 = max(0, x - radius), min(W, x + radius + 1)
        y0, y1 = max(0, y - radius), min(H, y + radius + 1)
        crop = lambda a: a[y0:y1, x0:x1]
        cnn_prior = _cnn_prior(crop(cnn_map))
        per_agent: dict = {}
        for agent, chans in AGENT_CHANNELS.items():
            view = {"channels": {ch: _descr(crop(patch_chw[CHANNEL_INDEX[ch]])) for ch in chans}}
            # decision ① (2026-05-30): cnn_prior goes ONLY to the broad-view agent, so the
            # narrow specialists stay channel-isolated (preserves contribution C2). The
            # arbiter learns of it indirectly, via forensics' cited evidence in the debate.
            if agent == "drc_pattern_forensics":
                view["cnn_prior"] = cnn_prior
            per_agent[agent] = view
        return {"region": {"x": int(x), "y": int(y), "radius": int(radius)},
                "per_agent": per_agent,
                "_cnn_prior_full": cnn_prior}   # internal: logging/eval only, NOT shown to narrow agents


def build_region_features(extractor: "CNNEvidenceExtractor", patch_chw: np.ndarray,
                          x: int, y: int, radius: int = 8, cnn_map: np.ndarray | None = None,
                          homogeneous: bool = False):
    """Build an orchestrator.RegionFeatures carrying v2 structured CNN evidence.

    Drop-in replacement for the legacy box-mean RegionFeatures: the orchestrator's
    RegionFeatures.for_agent(agent) returns each agent's structured view.

    - homogeneous=False (default, arm A3): heterogeneous per-agent views; cnn_prior →
      drc_pattern_forensics only (information isolation, contribution C2).
    - homogeneous=True (arm A2): EVERY agent slot gets the SAME broad view (all 9 channels
      + cnn_prior) — i.e. the drc_pattern_forensics view replicated. Isolates "debate without
      view-heterogeneity" using the same 4-name roster (no orchestrator/protocol change)."""
    from experiments.agents.orchestrator import RegionFeatures
    ev = extractor.region_evidence(patch_chw, x, y, radius, cnn_map=cnn_map)
    per_agent = ev["per_agent"]
    if homogeneous:
        broad = per_agent["drc_pattern_forensics"]        # all 9 channels + cnn_prior
        per_agent = {agent: broad for agent in per_agent}  # replicate to every slot (A2)
    return RegionFeatures(x=int(x), y=int(y), radius=int(radius), evidence=per_agent)


# ---------------------------------------------------------------------------
# Demo: OLD box-mean vs NEW structured evidence, on a strong + a clean patch
# ---------------------------------------------------------------------------
def _demo() -> int:
    import glob, json
    dr = BASE / "data/circuitnet_n28/training_set/DRC"
    feats = sorted(glob.glob(str(dr / "feature/*.npy")))[:300]
    # find one strong + one clean patch by label max
    strong = clean = None
    for f in feats:
        name = Path(f).stem
        lab = np.load(dr / "label" / (name + ".npy"))
        mx = float(lab.max())
        if strong is None and mx >= 0.3: strong = (name, mx)
        if clean is None and mx < 0.01: clean = (name, mx)
        if strong and clean: break
    ex = CNNEvidenceExtractor()
    print(f"RouteNet evidence extractor ready (in_ch={ex.in_channels}, device={ex.device})\n")
    for tag, picked in [("STRONG hotspot", strong), ("CLEAN", clean)]:
        if not picked:
            continue
        name, mx = picked
        feat = np.load(dr / "feature" / (name + ".npy")).transpose(2, 0, 1).astype(np.float32)
        lab = np.load(dr / "label" / (name + ".npy"))[..., 0]
        cy, cx = np.unravel_index(int(lab.argmax()), lab.shape)
        r = 8
        y0, y1 = max(0, cy - r), cy + r + 1; x0, x1 = max(0, cx - r), cx + r + 1
        print(f"===== {tag}: {name}  (label max={mx:.3f}, region@({cx},{cy})) =====")
        print("OLD (box-mean — one scalar/channel, what we used to feed):")
        old = {ch: round(float(feat[CHANNEL_INDEX[ch], y0:y1, x0:x1].mean()), 5)
               for ch in AGENT_CHANNELS["congestion_analyst"]}
        print("  congestion_analyst:", old)
        ev = ex.region_evidence(feat, cx, cy, r)
        print("NEW (structured evidence — what agents now see):")
        print("  drc_pattern_forensics.cnn_prior (broad agent only):",
              json.dumps(ev["per_agent"]["drc_pattern_forensics"]["cnn_prior"], ensure_ascii=False))
        print("  congestion_analyst.channels (NO cnn_prior — C2 isolation):",
              json.dumps(ev["per_agent"]["congestion_analyst"]["channels"], ensure_ascii=False))
        print("  congestion_analyst has cnn_prior? ->", "cnn_prior" in ev["per_agent"]["congestion_analyst"])
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(_demo())
