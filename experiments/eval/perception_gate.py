"""Ⅴ.1 — VLM perception gate (v3 GO/NO-GO; decisions-log 2026-05-31 evening).

Question: can a VLM actually SEE discriminative structure in our abstract EDA heatmaps?
If not, the whole "agents look at images" direction dies here (cheaply). Three tasks,
all single-VLM (no debate), all local via mlx-vlm on M5 (free):

  T1 localization   — plain box-free channel render → "which 3x3 grid cell holds the
                      brightest cluster?" vs the channel's box-mean argmax. Chance ≈ 11%.
  T2 strong-vs-clean — forensics view (red-boxed region) → P(hotspot). AUC on
                      strong[>=0.20] vs clean(<0.01) patches. Sanity: should be easy.
  T3 weak-vs-clean   — same, weak[0.01,0.05) vs clean. THE battlefield (CNN ~0.5x-design;
                      text-A1 weak recall was 0.733). This decides the gate.

T2/T3 run in two image conditions: channels-only ("nocnn") and channels+CNN-prior panel
("withcnn") — isolating the VLM's own perception from CNN-evidence reading.

Per-call results are cached to disk (one JSON per call) → fully resumable; reruns are free.

Usage (from paper&experiment/):
    PYTHONPATH=. python3 experiments/eval/perception_gate.py --smoke          # 3 calls
    PYTHONPATH=. python3 experiments/eval/perception_gate.py                  # full gate
    PYTHONPATH=. python3 experiments/eval/perception_gate.py --report-only    # re-aggregate
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
from experiments.agents.llm_clients import _extract_json_object          # noqa: E402
from experiments.eval.region_sampler import _box_mean_argmax             # noqa: E402
from experiments.feature_extraction.extract import CHANNELS, CHANNEL_INDEX  # noqa: E402
from experiments.feature_extraction.render_views import render_agent_view   # noqa: E402

# Override with LG_VLM_MODEL (e.g. a local ModelScope snapshot path) — HF id is the default.
import os  # noqa: E402
MODEL_ID = os.environ.get("LG_VLM_MODEL", "mlx-community/Qwen2.5-VL-3B-Instruct-4bit")
DATAROOT = BASE / "data/circuitnet_n28/training_set/DRC"
CACHE = BASE / "data/evidence_cache/test_N28_cnnargmax_evidence.jsonl"
RENDER_DIR = BASE / "data/render_cache"
OUT_DIR = BASE / "experiments/results/v3_perception_gate"
RADIUS = 8

T1_PROMPT = (
    "This image is a 2D intensity heatmap (dark = low, bright yellow/white = high) from a "
    "chip-layout analysis tool. Divide the image into a 3x3 grid of equal cells: rows A (top), "
    "B (middle), C (bottom); columns 1 (left), 2 (middle), 3 (right). "
    "Which single cell contains the brightest / most concentrated high-intensity cluster? "
    'Answer with JSON only: {"cell": "<A1..C3>", "confidence": <0..1>}'
)

# NB (smoke finding, 2026-06-13): a base-rate hint ("most regions are NOT hotspots") made the
# 3B model parrot p=0 without looking. AUC needs ranking only → no base-rate prompt; instead
# force observation-first so the model must actually read the zoom panels.
T23_PROMPT = (
    "You are inspecting a chip-layout region for DRC (design-rule-check) hotspot risk. "
    "The image shows several feature-map panels of the SAME die area (top rows: full map with "
    "the inspected region marked by a RED BOX; bottom rows: zoom into that region). "
    "Panels include macro/cell-density, RUDY wire-demand, and routing-congestion overflow maps"
    "{cnn_hint}. Intensities are normalized per patch, so judge LOCALIZED structure (sharp "
    "spikes, dense clusters, bright bands inside the red box / zoom panels), not global "
    "brightness. First LOOK at the zoom panels and describe what is actually inside the region; "
    "then rate the risk. Answer with JSON only — observation must be ONE short sentence "
    "(a plain string, max 25 words, NOT per-panel): "
    '{{"observation": "<one sentence>", "p_hotspot": <single number 0..1>}}'
)
CNN_HINT = (
    "; the LAST panel (CNN_prior) is a learned hotspot-prediction map from a CNN — strong "
    "evidence when it shows a bright blob inside the red box, but it can miss subtle hotspots"
)


# ---------------------------------------------------------------------------
# Local VLM wrapper (mlx-vlm; lazy-loaded once)
# ---------------------------------------------------------------------------

_VLM = {}


def vlm_call(question: str, image_path: Path, max_tokens: int = 450) -> dict:
    if not _VLM:
        from mlx_vlm import load
        from mlx_vlm.utils import load_config
        model, processor = load(MODEL_ID)
        _VLM.update(model=model, processor=processor, config=load_config(MODEL_ID))
        print(f"[vlm] {MODEL_ID} loaded")
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    prompt = apply_chat_template(_VLM["processor"], _VLM["config"], question, num_images=1)
    out = generate(_VLM["model"], _VLM["processor"], prompt, image=[str(image_path)],
                   max_tokens=max_tokens, temperature=0.0, verbose=False)
    text = out.text if hasattr(out, "text") else str(out)
    try:
        return _extract_json_object(text)
    except Exception:
        return {"_parse_error": text[:300]}


# ---------------------------------------------------------------------------
# Sample selection + rendering
# ---------------------------------------------------------------------------


def load_cache() -> list[dict]:
    return [json.loads(l) for l in CACHE.read_text().splitlines() if l.strip()]


def pick_samples(recs: list[dict], n_loc: int, n_cls: int, seed: int):
    rng = random.Random(seed)
    strong = [r for r in recs if r["label_strength"] >= 0.20]
    weak = [r for r in recs if 0.01 <= r["label_strength"] < 0.05 and r["is_hotspot"]]
    clean = [r for r in recs if r["label_strength"] < 0.01 and not r["is_hotspot"]]
    for g in (strong, weak, clean):
        rng.shuffle(g)
    loc = rng.sample(recs, n_loc)
    return loc, strong[:n_cls], weak[:n_cls], clean[:n_cls]


def render_t1_image(pid: str, out: Path) -> tuple[Path, str]:
    """Box-free single-channel render + ground-truth 3x3 cell of its box-mean argmax."""
    import matplotlib.pyplot as plt
    feat = np.load(DATAROOT / "feature" / f"{pid}.npy").transpose(2, 0, 1).astype(np.float32)
    ci = int(np.argmax([feat[CHANNEL_INDEX[c]].std() for c in CHANNELS]))   # most-structured channel
    arr = feat[CHANNEL_INDEX[CHANNELS[ci]]]
    cx, cy, _ = _box_mean_argmax(arr, RADIUS)
    cell = "ABC"[min(2, int(cy / (arr.shape[0] / 3)))] + str(min(2, int(cx / (arr.shape[1] / 3))) + 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        plt.imsave(out, np.kron(arr, np.ones((2, 2))), cmap="magma", vmin=0, vmax=1)  # 512x512 nearest
    return out, cell


def render_t23_images(rec: dict) -> dict[str, Path]:
    """Forensics view with and without the CNN_prior panel (lazy, cached on disk)."""
    pid = rec["patch_id"]
    paths = {"withcnn": RENDER_DIR / pid / "drc_pattern_forensics.png",
             "nocnn": RENDER_DIR / pid / "drc_pattern_forensics_nocnn.png"}
    if not all(p.exists() for p in paths.values()):
        feat = np.load(DATAROOT / "feature" / f"{pid}.npy").transpose(2, 0, 1).astype(np.float32)
        x, y = int(rec["x"]), int(rec["y"])
        if not paths["nocnn"].exists():
            render_agent_view(feat, "drc_pattern_forensics", x, y, RADIUS, paths["nocnn"], cnn_map=None)
        if not paths["withcnn"].exists():
            from experiments.feature_extraction.cnn_evidence import CNNEvidenceExtractor
            ex = _VLM.setdefault("_cnn_ex", CNNEvidenceExtractor())
            render_agent_view(feat, "drc_pattern_forensics", x, y, RADIUS, paths["withcnn"],
                              cnn_map=ex.hotspot_map(feat))
    return paths


# ---------------------------------------------------------------------------
# Cached call layer (resumable)
# ---------------------------------------------------------------------------


PROMPT_VER = "v3"   # bump on any prompt change — keys the call cache, prevents stale mixing


def cached_call(key: str, question: str, image: Path, calls_dir: Path) -> dict:
    f = calls_dir / f"{PROMPT_VER}_{key}.json"
    if f.exists():
        return json.loads(f.read_text())
    t0 = time.time()
    ans = vlm_call(question, image)
    ans["_elapsed_s"] = round(time.time() - t0, 1)
    calls_dir.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(ans, ensure_ascii=False, indent=1))
    return ans


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    from sklearn.metrics import roc_auc_score
    y = [1] * len(pos) + [0] * len(neg)
    return float(roc_auc_score(y, pos + neg))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-loc", type=int, default=60)
    ap.add_argument("--n-cls", type=int, default=40, help="per class (strong/weak/clean)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="3 calls only, verify plumbing")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args(argv)

    # cache calls per-model: answers from one model must never be reused for another
    model_slug = Path(MODEL_ID).name.lower().replace("___", "-")
    calls_dir = OUT_DIR / f"calls_{model_slug}"
    recs = load_cache()
    if a.smoke:
        a.n_loc, a.n_cls = 1, 1
    loc, strong, weak, clean = pick_samples(recs, a.n_loc, a.n_cls, a.seed)
    print(f"[gate] T1 n={len(loc)} | T2/T3 per-class n={len(strong)}/{len(weak)}/{len(clean)} "
          f"| smoke={a.smoke}")

    results = {"t1": [], "t23": []}
    if not a.report_only:
        # T1 localization
        for i, rec in enumerate(loc, 1):
            pid = rec["patch_id"]
            img, gt = render_t1_image(pid, OUT_DIR / "t1_images" / f"{pid}.png")
            ans = cached_call(f"t1_{pid}", T1_PROMPT, img, calls_dir)
            results["t1"].append({"pid": pid, "gt": gt, "ans": ans})
            if i % 10 == 0 or a.smoke:
                print(f"  T1 [{i}/{len(loc)}] gt={gt} -> {ans.get('cell')} ({ans.get('_elapsed_s')}s)")
        # T2/T3 classification, two conditions
        groups = [("strong", strong), ("weak", weak), ("clean", clean)]
        for cond in ("nocnn", "withcnn"):
            q = T23_PROMPT.format(cnn_hint=CNN_HINT if cond == "withcnn" else "")
            for gname, grp in groups:
                for i, rec in enumerate(grp, 1):
                    pid = rec["patch_id"]
                    imgs = render_t23_images(rec)
                    ans = cached_call(f"t23_{cond}_{pid}", q, imgs[cond], calls_dir)
                    results["t23"].append({"pid": pid, "group": gname, "cond": cond,
                                           "label": int(rec["is_hotspot"]), "ans": ans})
                    if i % 10 == 0 or a.smoke:
                        print(f"  T23[{cond}/{gname} {i}/{len(grp)}] -> p={ans.get('p_hotspot')} "
                              f"({ans.get('_elapsed_s')}s)")
    else:
        # rebuild results from the call cache (current PROMPT_VER only)
        for f in sorted(calls_dir.glob(f"{PROMPT_VER}_t1_*.json")):
            pid = f.stem[len(PROMPT_VER) + 4:]
            img, gt = render_t1_image(pid, OUT_DIR / "t1_images" / f"{pid}.png")
            results["t1"].append({"pid": pid, "gt": gt, "ans": json.loads(f.read_text())})
        by_id = {r["patch_id"]: r for r in recs}
        for f in sorted(calls_dir.glob(f"{PROMPT_VER}_t23_*.json")):
            _, _, cond, pid = f.stem.split("_", 3)
            rec = by_id[pid]
            g = ("strong" if rec["label_strength"] >= 0.20
                 else "weak" if rec["label_strength"] >= 0.01 and rec["is_hotspot"]
                 else "clean")
            results["t23"].append({"pid": pid, "group": g, "cond": cond,
                                   "label": int(rec["is_hotspot"]), "ans": json.loads(f.read_text())})

    # ---- aggregate ----
    rep: dict = {"model": MODEL_ID, "n": {"t1": len(results["t1"])}}
    t1v = [r for r in results["t1"] if isinstance(r["ans"].get("cell"), str)]
    hit = sum(1 for r in t1v if r["ans"]["cell"].strip().upper() == r["gt"])
    adj = 0
    for r in t1v:
        try:
            pr, pc = "ABC".index(r["ans"]["cell"][0].upper()), int(r["ans"]["cell"][1]) - 1
            gr, gc = "ABC".index(r["gt"][0]), int(r["gt"][1]) - 1
            adj += (abs(pr - gr) <= 1 and abs(pc - gc) <= 1)
        except Exception:
            pass
    rep["t1"] = {"n_valid": len(t1v), "top1_acc": round(hit / max(len(t1v), 1), 3),
                 "adjacent_acc": round(adj / max(len(t1v), 1), 3), "chance": 0.111}

    def _coerce_p(v):
        # 3B occasionally emits a list ("[0.0, 0.0]") — take the first element
        if isinstance(v, list) and v and isinstance(v[0], (int, float)):
            v = v[0]
        return float(v) if isinstance(v, (int, float)) else None

    def pvals(group, cond):
        out = []
        for r in results["t23"]:
            if r["group"] == group and r["cond"] == cond:
                p = _coerce_p(r["ans"].get("p_hotspot"))
                if p is not None:
                    out.append(p)
        return out
    rep["t23"] = {}
    for cond in ("nocnn", "withcnn"):
        s, w, c = pvals("strong", cond), pvals("weak", cond), pvals("clean", cond)
        rep["t23"][cond] = {
            "n": {"strong": len(s), "weak": len(w), "clean": len(c)},
            "auc_strong_vs_clean": round(auc(s, c), 3),
            "auc_weak_vs_clean": round(auc(w, c), 3),
            "mean_p": {"strong": round(float(np.mean(s)), 3) if s else None,
                       "weak": round(float(np.mean(w)), 3) if w else None,
                       "clean": round(float(np.mean(c)), 3) if c else None},
        }
    n_fail = sum(1 for r in results["t1"] + results["t23"] if "_parse_error" in r["ans"])
    rep["parse_failures"] = n_fail

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report.json").write_text(json.dumps(rep, indent=2))
    print("\n===== Ⅴ.1 perception gate report =====")
    print(json.dumps(rep, indent=2))
    print(f"saved → {OUT_DIR/'report.json'}")
    print("\nGATE read-out: T1 top1≫0.11 → VLM sees spatial structure; "
          "T3 weak-vs-clean AUC is the battlefield (text-A1 reference ≈ weak recall 0.733; "
          "CNN cross-design weak ROC-AUC ≈ 0.52-0.56).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
