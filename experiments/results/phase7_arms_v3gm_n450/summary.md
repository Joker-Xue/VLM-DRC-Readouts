# Phase ⑦ Arms Summary

- caller=**vlm-local** · split=test_N28 · region=cnn-argmax · regions=**451** · seed=42
- labels: 157 hotspot / 294 not_hotspot
- R0: score=cnn_prior.peak threshold=0.1

## Overall (mechanism holds ⇔ A3 > A2 > A1; R0 = ceiling ref) — ECE/AURC/R@80 lower=better

| Arm | n | PR-AUC | ROC-AUC | ECE | AURC | R@80 | Hit | Flag | Abstain | API calls | mean lat (s) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| R0 | 451 | 0.986 | 0.992 | 0.158 | 0.031 | 0.080 | 0.922 | 0.417 | 0.000 | 0 | 0.00 |
| A1 | 451 | 0.563 | 0.778 | 0.098 | 0.139 | 0.219 | 0.732 | 0.523 | 0.000 | 451 | 16.00 |
| A2 | 451 | 0.350 | 0.490 | 0.200 | 0.545 | 0.584 | 0.532 | 0.306 | 0.667 | 4716 | 224.95 |
| A3 | 451 | 0.391 | 0.554 | 0.245 | 0.564 | 0.535 | 0.375 | 0.938 | 0.035 | 3528 | 120.13 |

## Hit-rate stratified by label_strength (weak/mid/strong are all true-hotspots → hit = recall; target = lift weak/mid recall)

| Bucket | n | hot% | R0 | A1 | A2 | A3 |
|---|--:|--:|--:|--:|--:|--:|
| clean(<0.01) | 304 | 0.033 | 0.885 | 0.668 | 0.655 | 0.099 |
| weak[0.01,0.05) | 44 | 1.000 | 1.000 | 0.864 | 0.205 | 0.955 |
| mid[0.05,0.20) | 69 | 1.000 | 1.000 | 0.870 | 0.246 | 1.000 |
| strong[>=0.20] | 34 | 1.000 | 1.000 | 0.853 | 0.441 | 0.824 |

## Win table — count of regions where row-arm correct AND col-arm wrong

| A \ B | R0 | A1 | A2 | A3 |
|---|--|--|--|--|
| **R0** | - | 94 | 197 | 251 |
| **A1** | 8 | - | 166 | 186 |
| **A2** | 21 | 76 | - | 186 |
| **A3** | 4 | 25 | 115 | - |
