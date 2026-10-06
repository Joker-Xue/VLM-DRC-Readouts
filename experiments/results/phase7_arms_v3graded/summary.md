# Phase ⑦ Arms Summary

- caller=**vlm-local** · split=test_N28 · region=cnn-argmax · regions=**150** · seed=42
- labels: 52 hotspot / 98 not_hotspot
- R0: score=cnn_prior.peak threshold=0.1

## Overall (mechanism holds ⇔ A3 > A2 > A1; R0 = ceiling ref) — ECE/AURC/R@80 lower=better

| Arm | n | PR-AUC | ROC-AUC | ECE | AURC | R@80 | Hit | Flag | Abstain | API calls | mean lat (s) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| R0 | 150 | 0.982 | 0.989 | 0.156 | 0.034 | 0.075 | 0.920 | 0.413 | 0.000 | 0 | 0.00 |
| A1 | 150 | 0.401 | 0.613 | 0.130 | 0.261 | 0.350 | 0.627 | 0.187 | 0.000 | 150 | 11.72 |
| A2 | 150 | 0.373 | 0.492 | 0.219 | 0.522 | 0.542 | 0.500 | 0.620 | 0.087 | 1368 | 134.30 |
| A3 | 150 | 0.302 | 0.416 | 0.188 | 0.581 | 0.592 | 0.373 | 0.920 | 0.013 | 996 | 69.28 |

## Hit-rate stratified by label_strength (weak/mid/strong are all true-hotspots → hit = recall; target = lift weak/mid recall)

| Bucket | n | hot% | R0 | A1 | A2 | A3 |
|---|--:|--:|--:|--:|--:|--:|
| clean(<0.01) | 101 | 0.030 | 0.881 | 0.822 | 0.416 | 0.099 |
| weak[0.01,0.05) | 15 | 1.000 | 1.000 | 0.200 | 0.533 | 0.933 |
| mid[0.05,0.20) | 23 | 1.000 | 1.000 | 0.261 | 0.609 | 0.957 |
| strong[>=0.20] | 11 | 1.000 | 1.000 | 0.182 | 1.000 | 0.909 |

## Win table — count of regions where row-arm correct AND col-arm wrong

| A \ B | R0 | A1 | A2 | A3 |
|---|--|--|--|--|
| **R0** | - | 50 | 68 | 82 |
| **A1** | 6 | - | 54 | 78 |
| **A2** | 5 | 35 | - | 41 |
| **A3** | 0 | 40 | 22 | - |
