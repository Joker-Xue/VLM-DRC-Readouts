# VLM Readouts for DRC Hotspot Detection — Code, Prompts and Results

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23206403.svg)](https://doi.org/10.5281/zenodo.23206403)

This repository accompanies the article

> X. Lin and Z. Wang, "Vision-Language Readouts for DRC Hotspot Detection: A Calibrated CNN Baseline and a Matched-Control Test of Multi-Agent Debate," submitted to *IEEE Access*, 2026.

It contains the code, the exact prompts, and the saved per-region results needed to **recompute the metrics in Tables II–XI and XIII–XVI and Figs. 4–11 without running any model** (Table XII, the historical call counts in Table XVI and the Section VI-B disagreement statistics follow directly from the same saved per-region files; the Section V-C trajectory counts are produced by `PYTHONPATH=. python experiments/eval/debate_transcript_stats.py phase7_arms_v3gm_n2000_v2`), plus the scripts that produced those results from CircuitNet-N28. Table I and Fig. 3 are transcribed from the screening reports (score-level logs were retained only for Gemma-3-12B; the other rows are recorded summary values); Table XVII and Fig. 12 combine the recomputable n=2000 estimate with two pilot estimates whose per-region outputs were not retained; Figs. 1–2 are illustrative.

| What you want | Time | Needs |
|---|---|---|
| **Recompute the paper's metrics and result figures** from the saved results | ~1 min | Python 3.10+ and `requirements.txt` |
| Re-run the VLM experiments | days | Apple Silicon Mac, CircuitNet-N28, `requirements-inference.txt` |

## Quick start (recompute the paper)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make reproduce      # unpack data → tests → recompute metrics → regenerate figures → compare with references
```

`make reproduce` ends with `REPRODUCTION OK` when every recomputed value equals the archived reference outputs in `reference_outputs/` (1,155 values, including all bootstrap intervals) and every figure has been regenerated in `figures/out/`.

The individual steps are:

```bash
make data       # decompress the corrected-debate per-region file (64 MB)
make test       # unit tests for the debate protocol and validators
make metrics    # outputs/recomputed-metrics.json and outputs/corrected-debate-evaluation.json
make figures    # figures/out/*.png|pdf
make check      # compare outputs/ with reference_outputs/
```

## Where each result comes from

All metric definitions (ROC-AUC, average precision, ECE, Brier, NLL, AURC, R@80) are in `experiments/eval/audited_paper_metrics.py` and follow Eqs. (3)–(7) of the paper. Bootstrap intervals use 3,000 region resamples with seed 42.

| Paper item | Content | Produced by | Key in output |
|---|---|---|---|
| Table I, Fig. 3 | Five-model perception screen | `experiments/eval/perception_gate.py` → `experiments/results/m1_percept_*_report.json`, `v3_perception_gate_gemma3_12b/` | — |
| Table II | CNN vs. VLM, transferred isotonic calibration | `audited_paper_metrics.py` | `cnn_calibration/cnn-argmax`, `ensemble` |
| Table III | Label-argmax selection sensitivity | `audited_paper_metrics.py` | `cnn_calibration/label-argmax` |
| Table IV | CNN-prior panel ablation | `audited_paper_metrics.py` | `no_prior` |
| Table V, Fig. 4 | Prompt packages v1 / v3 / v4 | `audited_paper_metrics.py` | `prompts` |
| **Tables VI–VII, Fig. 5** | **Corrected matched-control debate** | `experiments/eval/eval_corrected_debate.py`, `figures/plot_corrected_debate.py` | `corrected-debate-evaluation.json` |
| Tables VIII–XII, Fig. 6 | Historical debate configurations | `audited_paper_metrics.py` | `historical_debate`, `historical_debate_bootstrap`, `committed_matched`, `sample_size_history` |
| Tables XIII–XIV, Figs. 7–10 | Repeated sampling (self-ensembling) | `audited_paper_metrics.py` | `ensemble` |
| Table XV | Temperature scaling of the single reader | `audited_paper_metrics.py` | `vlm_temperature` |
| Table XVI, Fig. 11 | N = 1/3/5/10 and logical calls | `audited_paper_metrics.py` | `ensemble/*/n{1,3,5,10}_mean` |
| Table XVII, Fig. 12 | Cross-design ROC gap | `experiments/eval/bootstrap_ci.py` (see below) | — |
| Pixel-level CNN ROC-AUC (Sec. V-A) | RouteNet on the official test split | `experiments/eval/routenet_ceiling.py` → `experiments/results/routenet_ceiling_N28.json` | — |

Fig. 1 (schematic) and Fig. 2 (rendered examples, `figures/static/`) are not computed from results.

Output file names in `figures/out/` and the corresponding figures of the article:

| File | Article | File | Article |
|---|---|---|---|
| `fig1_system_overview` | Fig. 1 | `fig6_ece_calibration` | Fig. 7 |
| `figures/static/fig2_rendered_views` | Fig. 2 | `fig7_risk_coverage` | Fig. 8 |
| `fig3_perception_scale` | Fig. 3 | `fig8_bootstrap_forest` | Fig. 9 |
| `fig4_prompt_sensitivity` | Fig. 4 | `fig10_reliability_diagram` | Fig. 10 |
| `fig5_corrected_debate` | Fig. 5 | `fig11_performance_vs_calls` | Fig. 11 |
| `fig5_debate_failure` | Fig. 6 | `fig9_cross_design_convergence` | Fig. 12 |

Table XVII interval:

```bash
PYTHONPATH=. python experiments/eval/bootstrap_ci.py \
  experiments/results/phase7_selfensemble_traincross/per_region.json \
  --compare experiments/results/phase7_selfensemble/per_region.json \
  --keys single0 ens_median ens_mean --n-boot 3000
```

## Repository layout

```
experiments/
  feature_extraction/   CNN evidence (RouteNet), candidate selection, image rendering
  agents/               VLM reader, historical debate orchestrator/protocol, corrected debate (audited_debate.py)
    prompts/            visual (v1), visual_v3_gatemirror (v3), visual_v3_gatemirror_no_cnn, visual_v4_subtlebias (v4)
  eval/                 experiment runners and the paper's metric/evaluation code
  results/              saved per-region results of every experiment reported in the paper
  tests/                unit tests
data/evidence_cache/    per-candidate CNN evidence for test_N28 and train_N28 (derived from CircuitNet-N28)
figures/                figure scripts; output goes to figures/out/
reference_outputs/      archived outputs used by `make check`
```

### Saved results

| Directory | Experiment (paper section) |
|---|---|
| `corrected_debate_20260922_v3_n2000/` | Corrected matched-control debate, H and X, n=2000 (V-C). `manifest.json` records code, prompt, image and model-file hashes; `per_region.json.gz` holds every logical message. |
| `phase7_arms_v3gm_n2000_v2/` | Historical debate A1/A2/A3, n=2000 (V-C) |
| `phase7_arms_v3gm_n450/`, `phase7_arms_v3graded_v3gatemirror/` | Historical debate at n=451 and n=150 (V-C) |
| `phase7_arms_v3graded/`, `phase7_arms_v3graded_v4subtlebias/` | Prompt packages v1 and v4, n=150 (V-B) |
| `phase7_ablation_no_cnn_prior/` | CNN-prior panel ablation (V-A) |
| `phase7_selfensemble/`, `phase7_selfensemble_traincross/` | Repeated sampling on test_N28 and train_N28, ten stored samples per candidate (V-D) |

The raw generation records of the corrected debate (all 81,882 attempts: raw model text, parsed output, validation errors, seeds and timing; `corrected_debate_raw_generations.tar.gz`, 40 MB) are archived as a separate file of the Zenodo release (<https://doi.org/10.5281/zenodo.23206403>); they are not needed to recompute the paper. The historical runs did not retain raw generations; see Section III-D and the Limitations of the paper.

## Re-running the experiments

These steps need the CircuitNet-N28 data and the official RouteNet weights, which are not redistributed here.

1. Obtain CircuitNet-N28 from the CircuitNet project (<https://github.com/circuitnet/CircuitNet>) and place the DRC training set under `data/circuitnet_n28/training_set/DRC/` (`feature/`, `label/`).
2. Clone CircuitNet into `.thirdparty/CircuitNet/` so that the RouteNet code and `pretrained_weights/DRC.pth` are available.
3. `pip install -r requirements-inference.txt`. The VLM is `mlx-community/Qwen2.5-VL-7B-Instruct-4bit`, run with MLX on Apple Silicon; pass its local path with `--model` where needed.
4. Build evidence and images: `experiments/feature_extraction/build_evidence_cache.py`, then `render_views.py` (and `render_no_cnn_prior.py` for the ablation).
5. Run `experiments/eval/run_arms.py` (historical configurations and prompt packages), `run_selfensemble.py` (repeated sampling) and `run_corrected_debate.py --sample 2000 --out <dir>` (corrected debate; resumable, about 40 calls per candidate).

Provenance files (manifests, perception-gate reports) contain absolute paths from the original machine; they are kept unchanged so that recorded hashes remain valid.

## Licence and citation

Code is released under the MIT License (`LICENSE`). Saved results, prompts and the evidence cache are released under CC BY 4.0 (`LICENSE-DATA`). The evidence cache is derived from CircuitNet-N28 with the official RouteNet weights; use of CircuitNet itself is subject to its own licence. Please cite the article and this archive, *VLM Readouts for DRC Hotspot Detection: code, prompts and results* (version 1.1.0), Zenodo, <https://doi.org/10.5281/zenodo.23206403> (`CITATION.cff`). The Zenodo archive is a snapshot of tag `v1.1.0` of this repository.
