# Phase ⑦ Arms Summary

- caller=**vlm-local** · split=test_N28 · region=cnn-argmax · regions=**2000** · seed=42
- labels: 704 hotspot / 1296 not_hotspot
- R0: score=cnn_prior.peak threshold=0.1

## Overall (mechanism holds ⇔ A3 > A2 > A1; R0 = ceiling ref) — ECE/AURC/R@80 lower=better

| Arm | n | PR-AUC | ROC-AUC | ECE | AURC | R@80 | Hit | Flag | Abstain | API calls | mean lat (s) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A1 | 2000 | 0.570 | 0.793 | 0.130 | 0.114 | 0.222 | 0.741 | 0.605 | 0.000 | 2000 | 19.85 |

## Hit-rate stratified by label_strength (weak/mid/strong are all true-hotspots → hit = recall; target = lift weak/mid recall)

| Bucket | n | hot% | A1 |
|---|--:|--:|--:|
| clean(<0.01) | 1349 | 0.039 | 0.620 |
| weak[0.01,0.05) | 196 | 1.000 | 0.995 |
| mid[0.05,0.20) | 305 | 1.000 | 0.993 |
| strong[>=0.20] | 150 | 1.000 | 0.980 |

## Win table — count of regions where row-arm correct AND col-arm wrong

| A \ B | A1 |
|---|--|
| **A1** | - |
