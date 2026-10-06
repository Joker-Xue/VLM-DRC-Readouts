"""R0 ceiling measurement — pretrained RouteNet on CircuitNet-N28 DRC test split.

Runs entirely on M5 (MPS/CPU), no AutoDL/GPU cloud cost. Loads the official
pretrained DRC.pth into RouteNet (bypassing models/__init__ → avoids the mmcv
dep that mavi.py pulls), runs inference over test_N28.csv, and reports
pixel-level ROC-AUC / PR-AUC / best-F1 / ECE (calibration) — overall, stratified
by per-patch hotspot strength, and by design. Memory-safe via per-class score
histograms (no storing 207M pixels).

Usage (from paper&experiment/):
    python3 experiments/eval/routenet_ceiling.py --limit 150      # quick smoke
    python3 experiments/eval/routenet_ceiling.py                  # full 3164
"""
from __future__ import annotations
import argparse, importlib.util, json, os, time
from pathlib import Path
import numpy as np
import torch

BASE = Path(__file__).resolve().parents[2]          # paper&experiment/
CN = BASE / ".thirdparty/CircuitNet/routability_ir_drop_prediction"
DEF_WEIGHTS = BASE / ".thirdparty/CircuitNet/pretrained_weights/DRC.pth"
DEF_CSV = CN / "files/test_N28.csv"
DEF_DATAROOT = BASE / "data/circuitnet_n28/training_set/DRC"
BINS = 1000
np.seterr(divide="ignore", invalid="ignore")   # histogram divides are masked by np.where


def load_routenet(weights: Path, device: str):
    spec = importlib.util.spec_from_file_location("routenet", str(CN / "models/routenet.py"))
    rn = importlib.util.module_from_spec(spec); spec.loader.exec_module(rn)
    ckpt = torch.load(weights, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    in_ch = next(v for k, v in sd.items() if v.ndim == 4).shape[1]
    model = rn.RouteNet(in_channels=in_ch, out_channels=1)
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval(), in_ch


def design_of(name: str) -> str:
    b = name.split("-")
    return b[1] if len(b) > 1 else "?"


def strength_bucket(mx: float) -> str:
    if mx < 0.01: return "clean(<0.01)"
    if mx < 0.05: return "weak[0.01,0.05)"
    if mx < 0.20: return "mid[0.05,0.20)"
    return "strong[>=0.20]"


def metrics_from_hist(hp: np.ndarray, hn: np.ndarray) -> dict:
    """ROC-AUC, PR-AUC, best-F1, ECE from per-class score histograms (hi→lo sweep)."""
    centers = (np.arange(BINS) + 0.5) / BINS
    P, N = hp.sum(), hn.sum()
    if P == 0 or N == 0:
        return dict(n_pos=int(P), n_neg=int(N), base_rate=float(P / max(P + N, 1)),
                    roc_auc=None, pr_auc=None, best_f1=None, best_thr=None, ece=None)
    # cumulative from high score → low (threshold descending)
    cum_p = np.cumsum(hp[::-1])[::-1]   # TP at threshold = bin score
    cum_n = np.cumsum(hn[::-1])[::-1]   # FP
    tpr = cum_p / P
    fpr = cum_n / N
    prec = np.where(cum_p + cum_n > 0, cum_p / (cum_p + cum_n), 1.0)
    rec = tpr
    # integrate (thresholds go high→low so tpr/fpr increase as index decreases → sort by fpr)
    order = np.argsort(fpr)
    roc_auc = float(np.trapz(tpr[order], fpr[order]))
    order_r = np.argsort(rec)
    pr_auc = float(np.trapz(prec[order_r], rec[order_r]))
    f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    bi = int(np.argmax(f1))
    # ECE: per-bin |mean-pred − empirical-pos-rate|, weighted by bin mass
    tot = hp + hn
    with np.errstate(invalid="ignore", divide="ignore"):
        acc = np.where(tot > 0, hp / tot, 0.0)
    w = tot / tot.sum()
    ece = float(np.sum(w * np.abs(centers - acc)))
    return dict(n_pos=int(P), n_neg=int(N), base_rate=float(P / (P + N)),
                roc_auc=roc_auc, pr_auc=pr_auc, best_f1=float(f1[bi]),
                best_thr=float(centers[bi]), ece=ece)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DEF_WEIGHTS))
    ap.add_argument("--csv", default=str(DEF_CSV))
    ap.add_argument("--dataroot", default=str(DEF_DATAROOT))
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--label-thr", type=float, default=0.0, help="hotspot pixel = label > thr")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--out", default=str(BASE / "experiments/results/routenet_ceiling_N28.json"))
    a = ap.parse_args()

    model, in_ch = load_routenet(Path(a.weights), a.device)
    rows = [l.strip() for l in open(a.csv) if l.strip()]
    if a.limit:
        rows = rows[:a.limit]
    print(f"RouteNet in_channels={in_ch} | device={a.device} | patches={len(rows)} | label_thr={a.label_thr}")

    H = {}  # group -> [hist_pos, hist_neg]
    def acc(group, pred, pos_mask):
        hp, hn = H.setdefault(group, [np.zeros(BINS, np.int64), np.zeros(BINS, np.int64)])
        b = np.clip((pred * BINS).astype(np.int64), 0, BINS - 1)
        np.add.at(hp, b[pos_mask], 1)
        np.add.at(hn, b[~pos_mask], 1)

    dr = Path(a.dataroot); t0 = time.time(); n_lbl_stat = 0
    for i, line in enumerate(rows):
        fp, lp = line.split(",")
        feat = np.load(dr / fp).transpose(2, 0, 1).astype(np.float32)   # (9,H,W)
        lab = np.load(dr / lp).transpose(2, 0, 1).astype(np.float32)[0]  # (H,W)
        x = torch.from_numpy(feat).unsqueeze(0).to(a.device)
        with torch.no_grad():
            pred = model(x).squeeze().clamp(0, 1).float().cpu().numpy()  # (H,W)
        pos = lab > a.label_thr
        if n_lbl_stat < 3:
            print(f"  sample {i}: label max={lab.max():.4f} mean={lab.mean():.5f} pos%={100*pos.mean():.3f} | pred max={pred.max():.4f}")
            n_lbl_stat += 1
        acc("ALL", pred, pos)
        acc("design:" + design_of(Path(fp).stem), pred, pos)
        acc("strength:" + strength_bucket(float(lab.max())), pred, pos)
        if (i + 1) % 200 == 0:
            print(f"  ...{i+1}/{len(rows)} ({(i+1)/(time.time()-t0):.1f} patch/s)")

    report = {g: metrics_from_hist(hp, hn) for g, (hp, hn) in sorted(H.items())}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"config": vars(a), "in_channels": in_ch, "n_patches": len(rows),
               "metrics": report}, open(a.out, "w"), indent=2)

    def show(g):
        m = report.get(g)
        if not m or m["roc_auc"] is None:
            print(f"  {g:28s}  (n_pos={m['n_pos'] if m else 0})  — degenerate"); return
        print(f"  {g:28s}  ROC-AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}  "
              f"F1*={m['best_f1']:.4f}@{m['best_thr']:.3f}  ECE={m['ece']:.4f}  base={m['base_rate']*100:.3f}%")
    print("\n===== R0 RouteNet ceiling (pixel-level) =====")
    show("ALL")
    print("--- by hotspot strength ---")
    for g in sorted(report):
        if g.startswith("strength:"): show(g)
    print("--- by design ---")
    for g in sorted(report):
        if g.startswith("design:"): show(g)
    print(f"\nsaved → {a.out}  (elapsed {time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
