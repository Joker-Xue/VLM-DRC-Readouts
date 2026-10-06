# Phase ⑦ Arms Summary

- caller=**vlm-local** · split=test_N28 · region=cnn-argmax · regions=**2000** · seed=42
- labels: 704 hotspot / 1296 not_hotspot
- R0: score=cnn_prior.peak threshold=0.1

## Overall (mechanism holds ⇔ A3 > A2 > A1; R0 = ceiling ref) — ECE/AURC/R@80 lower=better

| Arm | n | PR-AUC | ROC-AUC | ECE | AURC | R@80 | Hit | Flag | Abstain | API calls | mean lat (s) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| R0 | 2000 | 0.985 | 0.992 | 0.164 | 0.035 | 0.090 | 0.927 | 0.413 | 0.000 | 0 | 0.00 |
| A1 | 2000 | 0.575 | 0.791 | 0.094 | 0.133 | 0.216 | 0.742 | 0.508 | 0.000 | 2000 | 4.64 |
| A2 | 2000 | 0.363 | 0.525 | 0.182 | 0.504 | 0.561 | 0.556 | 0.328 | 0.644 | 20536 | 65.73 |
| A3 | 2000 | 0.366 | 0.515 | 0.245 | 0.593 | 0.559 | 0.355 | 0.935 | 0.041 | 15664 | 36.00 |

## Hit-rate stratified by label_strength (weak/mid/strong are all true-hotspots → hit = recall; target = lift weak/mid recall)

| Bucket | n | hot% | R0 | A1 | A2 | A3 |
|---|--:|--:|--:|--:|--:|--:|
| clean(<0.01) | 1349 | 0.039 | 0.896 | 0.687 | 0.661 | 0.089 |
| weak[0.01,0.05) | 196 | 1.000 | 0.974 | 0.806 | 0.352 | 0.867 |
| mid[0.05,0.20) | 305 | 1.000 | 1.000 | 0.872 | 0.311 | 0.970 |
| strong[>=0.20] | 150 | 1.000 | 1.000 | 0.887 | 0.380 | 0.827 |

## Win table — count of regions where row-arm correct AND col-arm wrong

| A \ B | R0 | A1 | A2 | A3 |
|---|--|--|--|--|
| **R0** | - | 408 | 837 | 1160 |
| **A1** | 37 | - | 693 | 887 |
| **A2** | 95 | 322 | - | 859 |
| **A3** | 15 | 113 | 456 | - |
