"""Compare recomputed outputs with the archived reference outputs, value by value."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAIRS = [("outputs/recomputed-metrics.json", "reference_outputs/recomputed-metrics.json"),
         ("outputs/corrected-debate-evaluation.json", "reference_outputs/corrected-debate-evaluation.json")]
FIGURES = ["fig1_system_overview", "fig3_perception_scale", "fig4_prompt_sensitivity", "fig5_corrected_debate",
           "fig5_debate_failure", "fig6_ece_calibration", "fig7_risk_coverage", "fig8_bootstrap_forest",
           "fig10_reliability_diagram", "fig11_performance_vs_calls", "fig9_cross_design_convergence"]


def flatten(value, path=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from flatten(item, f"{path}/{key}")
    elif isinstance(value, list) and value and isinstance(value[0], (dict, list)):
        for index, item in enumerate(value):
            yield from flatten(item, f"{path}[{index}]")
    else:
        yield path, value


def main() -> None:
    failures, total = 0, 0
    for new, ref in PAIRS:
        a = dict(flatten(json.loads((ROOT / new).read_text())))
        b = dict(flatten(json.loads((ROOT / ref).read_text())))
        differing = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        total += len(b)
        failures += len(differing)
        print(f"{new}: {len(b)} values, {len(differing)} differ")
        for key in differing[:10]:
            print(f"   {key}: {a.get(key)!r} != {b.get(key)!r}")
    missing = [f for f in FIGURES if not (ROOT / "figures/out" / f"{f}.png").exists()]
    if missing:
        print("missing figures:", ", ".join(missing))
    if failures or missing:
        sys.exit("REPRODUCTION FAILED")
    print(f"REPRODUCTION OK: {total} values identical; {len(FIGURES)} figures regenerated in figures/out/")


if __name__ == "__main__":
    main()
