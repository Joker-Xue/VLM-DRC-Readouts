# Experiment A — strong-base self-ensemble (N=10, temp=0.5)
- n=2000 hot=1006 · prompts=visual_v3_gatemirror · region=cnn-argmax
- reference: A1 single (v3 gate-mirror) ROC 0.765 / ECE 0.111

| readout | ROC-AUC | PR-AUC | ECE | AURC | Flag | Hit |
|---|--:|--:|--:|--:|--:|--:|
| single0 | 0.746 | 0.677 | 0.117 | 0.223 | 0.516 | 0.711 |
| ens_median | 0.819 | 0.743 | 0.055 | 0.108 | 0.614 | 0.778 |
| ens_mean | 0.819 | 0.752 | 0.119 | 0.107 | 0.563 | 0.758 |

## mean ens score by bucket
| bucket | n | single0 | ens_mean |
|---|--:|--:|--:|
| clean(<0.01) | 1088 | 0.310 | 0.323 |
| weak[0.01,0.05) | 234 | 0.529 | 0.543 |
| mid[0.05,0.20) | 403 | 0.570 | 0.587 |
| strong[>=0.20] | 275 | 0.642 | 0.635 |
