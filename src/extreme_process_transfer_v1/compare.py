"""Explicit, fixed comparisons of complete saved checkpoints."""
import argparse
import json
from pathlib import Path
from .worker import GROUPS, save_json
from .paths import checked_output

METRICS = {
    'tw95':('all','twcrps95_q95'), 'tw99':('all','twcrps99_q95'),
    'h6_peak_mae':('peak_h6','mae'), 'h24_peak_mae':('peak_h24','mae'),
    'ordinary_mae':('ordinary_cell','mae'), 'ordinary_rmse':('ordinary_cell','rmse'),
    'ordinary_crps':('ordinary_cell','crps_q95'), 'ordinary_brier95':('ordinary_cell','brier95'),
    'ordinary_brier99':('ordinary_cell','brier99'), 'overall_mae':('all','mae'),
    'overall_brier95':('all','brier95'), 'overall_brier99':('all','brier99')}


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs-m',required=True);p.add_argument('--runs-nb',required=True);p.add_argument('--reference',required=True);p.add_argument('--out',required=True)
    args=p.parse_args(); records={}
    checked_output(args.out)
    reference=json.loads((Path(args.reference)/'B_M.json').read_text())['scores']
    for name in GROUPS:
        root=Path(args.runs_nb if name.startswith('NB_') else args.runs_m)
        folder=root/name/'artifacts'; complete=json.loads((folder/'completion.json').read_text())
        if complete['state'] != 'complete':
            raise ValueError('incomplete group '+name)
        records[name]=json.loads((folder/'final_scores.json').read_text())['checkpoints']
    output={}
    for role in ('fixed_last','best_brier'):
        values={}
        for name, checkpoints in records.items():
            selected=[v for v in checkpoints.values() if role in v['roles']]
            if len(selected)!=1:raise ValueError('missing or ambiguous checkpoint role')
            values[name]={metric:selected[0]['scores'][section][field] for metric,(section,field) in METRICS.items()}
        values['B_M']={metric:reference[section][field] for metric,(section,field) in METRICS.items()}
        comparisons={}
        for metric in METRICS:
            v={name:items[metric] for name,items in values.items()}
            def diff(a,b):
                return dict(absolute=v[a]-v[b],reference=b,
                            relative_improvement_pct=(100*(v[b]-v[a])/v[b]) if v[b] else None)
            comparisons[metric]=dict(foreign_effect_without_center=diff('M_X_E','M_N_E'),
                foreign_effect_with_center=diff('M_X_ME','M_N_ME'),
                center_effect_natural=diff('M_N_ME','M_N_E'),center_effect_foreign=diff('M_X_ME','M_X_E'),
                interaction=(v['M_X_ME']-v['M_N_ME'])-(v['M_X_E']-v['M_N_E']),
                combo_vs_simple=diff('M_X_ME','H_X_MU_ONLY'),combo_vs_nb_n=diff('M_X_ME','NB_N'),
                combo_vs_nb_x=diff('M_X_ME','NB_X'),combo_vs_start=diff('M_X_ME','B_M'))
        output[role]=dict(values=values,comparisons=comparisons)
    save_json(args.out,dict(state='exploratory_single_seed',roles=output,
              note='Negative relative improvement means deterioration; N-BEATSx objective and budget differ.'))


if __name__=='__main__':main()
