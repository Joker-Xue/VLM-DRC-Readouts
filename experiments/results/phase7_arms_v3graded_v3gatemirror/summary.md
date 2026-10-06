# Phase ⑦ Arms Summary

- caller=**vlm-local** · split=test_N28 · region=cnn-argmax · regions=**150** · seed=42
- labels: 52 hotspot / 98 not_hotspot
- R0: score=cnn_prior.peak threshold=0.1

## Overall (mechanism holds ⇔ A3 > A2 > A1; R0 = ceiling ref) — ECE/AURC/R@80 lower=better

| Arm | n | PR-AUC | ROC-AUC | ECE | AURC | R@80 | Hit | Flag | Abstain | API calls | mean lat (s) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| R0 | 150 | 0.982 | 0.989 | 0.156 | 0.034 | 0.075 | 0.920 | 0.413 | 0.000 | 0 | 0.00 |
| A1 | 150 | 0.539 | 0.765 | 0.111 | 0.138 | 0.250 | 0.733 | 0.533 | 0.000 | 150 | 11.53 |
| A2 | 150 | 0.354 | 0.531 | 0.197 | 0.478 | 0.575 | 0.587 | 0.307 | 0.660 | 1536 | 161.54 |
| A3 | 150 | 0.418 | 0.560 | 0.248 | 0.551 | 0.525 | 0.373 | 0.960 | 0.007 | 1124 | 86.34 |

## Hit-rate stratified by label_strength (weak/mid/strong are all true-hotspots → hit = recall; target = lift weak/mid recall)

| Bucket | n | hot% | R0 | A1 | A2 | A3 |
|---|--:|--:|--:|--:|--:|--:|
| clean(<0.01) | 101 | 0.030 | 0.881 | 0.663 | 0.713 | 0.079 |
| weak[0.01,0.05) | 15 | 1.000 | 1.000 | 0.733 | 0.400 | 1.000 |
| mid[0.05,0.20) | 23 | 1.000 | 1.000 | 1.000 | 0.261 | 1.000 |
| strong[>=0.20] | 11 | 1.000 | 1.000 | 0.818 | 0.364 | 0.909 |

## Win table — count of regions where row-arm correct AND col-arm wrong

| A \ B | R0 | A1 | A2 | A3 |
|---|--|--|--|--|
| **R0** | - | 32 | 57 | 84 |
| **A1** | 4 | - | 48 | 64 |
| **A2** | 7 | 26 | - | 65 |
| **A3** | 2 | 10 | 33 | - |
