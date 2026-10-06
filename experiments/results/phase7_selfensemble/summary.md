# Experiment A — strong-base self-ensemble (N=10, temp=0.5)
- n=2000 hot=704 · prompts=visual_v3_gatemirror · region=cnn-argmax
- reference: A1 single (v3 gate-mirror) ROC 0.765 / ECE 0.111

| readout | ROC-AUC | PR-AUC | ECE | AURC | Flag | Hit |
|---|--:|--:|--:|--:|--:|--:|
| single0 | 0.771 | 0.561 | 0.099 | 0.155 | 0.460 | 0.721 |
| ens_median | 0.833 | 0.624 | 0.096 | 0.091 | 0.517 | 0.755 |
| ens_mean | 0.837 | 0.651 | 0.069 | 0.088 | 0.480 | 0.759 |

## mean ens score by bucket
| bucket | n | single0 | ens_mean |
|---|--:|--:|--:|
| clean(<0.01) | 1349 | 0.296 | 0.296 |
| weak[0.01,0.05) | 196 | 0.561 | 0.555 |
| mid[0.05,0.20) | 305 | 0.611 | 0.609 |
| strong[>=0.20] | 150 | 0.624 | 0.616 |
