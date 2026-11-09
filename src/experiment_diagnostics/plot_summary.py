"""Server-only teaching figure from saved metrics; no training or model imports."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--selected', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    rows = [json.loads(x) for x in (args.training/'metrics.jsonl').read_text().splitlines()]
    selected = [json.loads(x) for x in (args.selected/'metrics.jsonl').read_text().splitlines()]
    selected = {x['arm']: x for x in selected if x.get('kind') == 'selected_checkpoint'}
    curves = {arm: [x for x in rows if x.get('kind') == 'epoch' and x['arm'] == arm]
              for arm in ('Risk-A', 'Risk-B', 'Risk-C', 'Risk-D')}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for arm, records in curves.items():
        axes[0, 0].plot([r['step'] for r in records], [r['metrics']['select_brier'][2] for r in records], label=arm)
    axes[0, 0].set(title='Full China select: 6h risk error', xlabel='Optimizer updates', ylabel='Brier (dimensionless, lower is better)')
    axes[0, 0].legend()
    records = curves['Risk-C']
    for role in ('fit', 'select'):
        axes[0, 1].plot([r['step'] for r in records], [r['metrics'][role+'_brier'][2] for r in records], label=role)
    best = json.loads((args.training/'Risk-C'/'best.json').read_text())
    axes[0, 1].axvline(best['step'], color='gray', linestyle=':', label='selected checkpoint')
    axes[0, 1].set(title='Risk-C: learning vs model selection', xlabel='Optimizer updates', ylabel='Brier (dimensionless, lower is better)')
    axes[0, 1].legend()
    exported = {'curves': {a: [{'step':r['step'],'fit':r['metrics']['fit_brier'][2],'select':r['metrics']['select_brier'][2]} for r in rr] for a,rr in curves.items()}, 'selected_shadows':{}}
    for ax, arm in zip(axes[1], ('Risk-C', 'Risk-D')):
        row = selected[arm]
        roles = row['diagnostic']['result']['roles']
        names = [n for n,v in roles['fit']['shadows'].items() if v.get('brier_shadow_minus_full')]
        positions = np.arange(len(names))
        exported['selected_shadows'][arm] = {}
        for offset, role in zip((-.18, .18), ('fit','select')):
            values = [roles[role]['shadows'][n]['brier_shadow_minus_full'][2] for n in names]
            ax.bar(positions+offset, values, width=.36, label=role)
            exported['selected_shadows'][arm][role] = dict(zip(names,values))
        ax.axhline(0,color='black',linewidth=.8)
        ax.set_xticks(positions, [n.replace('_', '\n') for n in names])
        ax.set(title=f'{arm}: selected epoch {row["epoch"]}, fixed fit/select groups', ylabel='Brier after intervention - full model')
        ax.legend()
    for ax in axes.flat:
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('Protected-risk diagnostics | saved evidence, no retraining\nBottom panels: positive = intervention worsens error; negative = improves error. Shadow evidence only.', fontsize=12)
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output/'diagnostics_overview.png',dpi=150)
    plt.close(fig)
    exported['limits'] = 'Fit/select diagnostics only. Overlapping windows are not independent floods; single seed. No test access.'
    (args.output/'plot_source.json').write_text(json.dumps(exported,indent=2))


if __name__ == '__main__':
    main()
