"""Fig. 5: corrected matched-control debate (no inference; reads saved run + frozen evaluation)."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'experiments/results/corrected_debate_20260922_v3_n2000'
OUTDIR = ROOT / 'figures/out'
OUTDIR.mkdir(parents=True, exist_ok=True)
E = json.loads((ROOT / 'outputs/corrected-debate-evaluation.json').read_text())
ROWS = json.loads((RUN / 'per_region.json').read_text())
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8, 'legend.fontsize': 7,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'pdf.fonttype': 42})


def unanimity(config, branch):
    n = 0
    for row in ROWS:
        votes = [m['vote'] for m in row['configurations'][config]['branches'][branch]['rounds'][2]]
        n += len(set(votes)) == 1 and votes[0] != 'abstain'
    return n / len(ROWS)


fig, (ax, bx) = plt.subplots(2, 1, figsize=(3.25, 4.4), layout='constrained')
readouts = [('single', 'Single\ndraw'), ('r1', 'Round-1\nmean'), ('own_fixed', 'Own\n(no comm.)'), ('peer_fixed', 'Peer\n(comm.)')]
x = np.arange(len(readouts)); w = 0.36
for k, (cfg, col, lab) in enumerate([('H', '#1873b8', 'Homogeneous (H)'), ('X', '#16845a', 'Heterogeneous (X)')]):
    vals = [E['configurations'][cfg]['readouts'][r]['roc'] for r, _ in readouts]
    ax.bar(x + (k - 0.5) * w, vals, w, color=col, label=lab)
ax.axhline(0.5, color='#555', ls='--', lw=0.8)
ax.set(xticks=x, xticklabels=[l for _, l in readouts], ylabel='ROC-AUC', ylim=(0, 0.85),
       title='(a) Matched readouts, n=2000 test candidates')
ax.legend(loc='upper left', frameon=False, ncol=2)

metrics = [('Unanimous\nround 3', None), ('Hotspot\ndecisions', 'flag'), ('Precision of\nhotspot decisions', 'prec')]
x2 = np.arange(len(metrics))
for k, (branch, col, lab) in enumerate([('own', '#9aa5b1', 'Own history'), ('peer', '#c0392b', 'Peer history')]):
    vals = []
    for name, key in metrics:
        hv, xv = [], []
        for cfg in ('H', 'X'):
            r = E['configurations'][cfg]['readouts'][f'{branch}_fixed']
            if key is None:
                v = unanimity(cfg, branch)
            elif key == 'flag':
                v = r['flag']
            else:
                dec = [row['configurations'][cfg]['branches'][branch]['fixed']['decision'] for row in ROWS]
                tp = sum(d == 'hotspot' and row['is_hotspot'] for d, row in zip(dec, ROWS))
                v = tp / max(1, sum(d == 'hotspot' for d in dec))
            hv.append(v)
        vals.append(hv)
    vals = np.array(vals)
    bx.bar(x2 + (k - 0.5) * w, vals.mean(1), w, color=col, label=lab)
    for j, mk in enumerate(('o', '^')):  # H circles, X triangles
        bx.scatter(x2 + (k - 0.5) * w, vals[:, j], s=10, color='black', zorder=3, marker=mk, linewidths=0)
bx.set(xticks=x2, xticklabels=[n for n, _ in metrics], ylabel='Rate', ylim=(0, 0.8),
       title='(b) Round-3 decisions (bars: H/X mean; ● H, ▲ X)')
bx.legend(loc='upper right', frameon=False)
for ext in ('png', 'pdf'):
    fig.savefig(OUTDIR / f'fig5_corrected_debate.{ext}', dpi=300, bbox_inches='tight', pad_inches=.05)
print('saved fig5_corrected_debate')
