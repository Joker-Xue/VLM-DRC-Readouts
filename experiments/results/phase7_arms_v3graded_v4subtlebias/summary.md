# Phase ⑦ Arms Summary

- caller=**vlm-local** · split=test_N28 · region=cnn-argmax · regions=**150** · seed=42
- labels: 52 hotspot / 98 not_hotspot
- R0: score=cnn_prior.peak threshold=0.1

## Overall (mechanism holds ⇔ A3 > A2 > A1; R0 = ceiling ref) — ECE/AURC/R@80 lower=better

| Arm | n | PR-AUC | ROC-AUC | ECE | AURC | R@80 | Hit | Flag | Abstain | API calls | mean lat (s) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A1 | 150 | 0.518 | 0.742 | 0.173 | 0.170 | 0.267 | 0.673 | 0.460 | 0.000 | 150 | 11.94 |

## Hit-rate stratified by label_strength (weak/mid/strong are all true-hotspots → hit = recall; target = lift weak/mid recall)

| Bucket | n | hot% | A1 |
|---|--:|--:|--:|
| clean(<0.01) | 101 | 0.030 | 0.663 |
| weak[0.01,0.05) | 15 | 1.000 | 0.600 |
| mid[0.05,0.20) | 23 | 1.000 | 0.696 |
| strong[>=0.20] | 11 | 1.000 | 0.818 |

## Win table — count of regions where row-arm correct AND col-arm wrong

| A \ B | A1 |
|---|--|
| **A1** | - |
