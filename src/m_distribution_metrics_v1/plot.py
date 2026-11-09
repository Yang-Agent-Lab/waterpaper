"""One CPU-rendered double panel of pointwise lead CRPS and censored CRPS."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

def plot(root,cfg):
    root=Path(root);rows=list(csv.DictReader((root/'lead_metrics.csv').open()))
    fonts=font_manager.findSystemFonts()
    candidates=[p for p in fonts if any(k in p.lower() for k in ('notosanscjk','sourcehansans','wenquanyi','wqy'))]
    if cfg.get('font_file') and Path(cfg['font_file']).is_file():
        font_manager.fontManager.addfont(cfg['font_file']);candidates.insert(0,cfg['font_file'])
    chinese=bool(candidates)
    if chinese:plt.rcParams['font.family']=font_manager.FontProperties(fname=candidates[0]).get_name()
    plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    colors={'M':'#bc2833','D500':'#214c86','C':'#d18d22','native_TCN':'#18866b'}
    for model in cfg['plot_models']:
        group=sorted([r for r in rows if r['model']==model],key=lambda r:int(r['lead_hour']))
        for ax,metric in zip(axes,('crps_q95','q95_twcrps_q95')):
            if group:ax.plot([int(r['lead_hour']) for r in group],[float(r[metric]) for r in group],label=model,color=colors.get(model),lw=1.7,marker='o',markevery=[0,2,5,11,23],ms=3)
            else:ax.plot([],[],color='gray',ls='--',label=model+('：缺失' if chinese else ': unavailable'))
    for ax,title in zip(axes,('逐小时 CRPS/Q95','逐小时 Q95-twCRPS/Q95') if chinese else ('Pointwise CRPS / fit Q95','Pointwise Q95-twCRPS / fit Q95')):
        ax.set_title(title);ax.set_xlabel('预测提前量（小时）' if chinese else 'Forecast lead (hours)');ax.set_ylabel('误差（dimensionless，越低越好）' if chinese else 'Error (dimensionless; lower is better)');ax.set_xticks([1,3,6,12,24]);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('中国开发 select：1706 起报 / 9 站 / 17 来源单位；K256' if chinese else 'China development select: 1706 origins / 9 stations / 17 source units; K256',fontsize=11)
    fig.savefig(root/'lead_crps_twcrps.png',dpi=180);fig.savefig(root/'lead_crps_twcrps.pdf');plt.close(fig)
    (root/'plot_metadata.txt').write_text('Font='+str(candidates[0] if chinese else 'DejaVu Sans; CJK font unavailable')+'\nSource=lead_metrics.csv; station/source/origin equal; overlapping windows not independent events\nPointwise lead curves, not peak-window curves; no formal model promotion\n')

if __name__=='__main__':
    import argparse,json,shutil
    from .run import save,sha
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--config',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    source=Path(a.source);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(source/'lead_metrics.csv',out/'lead_metrics.csv')
    plot(out,json.loads(Path(a.config).read_text()))
    save(out/'render_receipt.json',dict(source_csv=str(source/'lead_metrics.csv'),source_csv_sha256=sha(source/'lead_metrics.csv'),copied_csv_sha256=sha(out/'lead_metrics.csv'),numeric_scoring_repeated=False,reason='replace a missing fullwidth slash glyph with ASCII; preserve original rendered output'))
