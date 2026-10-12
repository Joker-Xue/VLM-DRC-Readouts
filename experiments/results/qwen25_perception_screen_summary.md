# Qwen2.5-VL perception screens (Table I, rows 1–2; Fig. 3)

The two Qwen2.5-VL screens were run with `experiments/eval/perception_gate.py` (60 localization
items, 40 regions per discrimination tier, seed 42, with and without the CNN-prior panel). Their
report files were later overwritten when the same output directory was reused, so the values
below are the summaries recorded in the authors' experiment log when each screen was run
(7B: 2026-06-13; 3B: 2026-07-06). No score-level logs exist for these two screens.

| Model (4-bit) | Localization top-1 | Adjacent rate | AUC strong vs clean | AUC weak vs clean | Parse failures |
|---|---|---|---|---|---|
| Qwen2.5-VL-3B | 2/60 (0.033) | not recorded | 0.46–0.51 | 0.46–0.50 | 4 |
| Qwen2.5-VL-7B | 16/60 (0.267) | 0.95 | 0.77–0.80 | 0.74–0.79 | 0 |

AUC ranges span the with- and without-CNN-prior conditions; only the range was recorded.
The three later models (`m1_percept_*_report.json`) and the Gemma-3-12B per-call outputs
(`v3_perception_gate_gemma3_12b/`) are archived as produced.
