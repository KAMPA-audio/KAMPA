"""Corrected result aggregation for the 20261114-v2 runner (2026-11-16).

feedback_core.aggregate merged m['test'] after m['recipe'], so the test-clip
count `n` overwrote the label budget `n`. N=200 and N=2000 runs fell into one
group, and the 2x2 effect step stopped with TypeError after the first N=2000
run. feedback_core.py is left byte-identical because its digest is part of each
run identity; changing it would invalidate every completed run. This file
replaces feedback_core.aggregate at launch time instead.

    python KAMPA_결과집계.py --package <unpacked ZIP> --config <launch_config.json>
    python KAMPA_결과집계.py --package <unpacked ZIP> --aggregate <output root>
"""
from pathlib import Path
import argparse, json, sys
import pandas as pd

METRICS=['accuracy','macro_precision','macro_recall','macro_f1']
RUN_KEY=['stage','n','split_seed','variant','seed']

def load_core(package):
    sys.path.insert(0,str(Path(package).resolve()))
    import feedback_core
    return feedback_core

def collect(out,core):
    rows=[];runtimes=set()
    for p in sorted((Path(out)/'runs').glob('*/complete.json')):
        core.validate_complete(p.parent)
        spec=json.loads((p.parent/'protocol.json').read_text());runtimes.add(json.dumps(spec['runtime'],sort_keys=True))
        m=json.loads((p.parent/'metrics.json').read_text());r=m['recipe']
        expected=f"split{r['split_seed']}_n{r['n']}_s{r['seed']}_{r['variant']}"
        if p.parent.name!=expected:raise RuntimeError(f'Run folder {p.parent.name} does not match its recipe {expected}')
        test={('n_test' if k=='n' else k):v for k,v in m['test'].items() if k!='confusion'}
        rows.append({**test,**r,'best_epoch':m['best_epoch']})
    if len(runtimes)>1:raise RuntimeError('Results from different hardware/library configurations must not be pooled')
    return rows

def aggregate(out,core):
    rows=collect(out,core)
    if not rows:return
    frame=pd.DataFrame(rows)
    frame=frame[RUN_KEY+METRICS+['n_test','best_epoch']]
    if frame.duplicated(RUN_KEY).any():raise RuntimeError('Duplicate runs for one recipe')
    core.write_csv(Path(out)/'results_all_runs.csv',frame)
    group=['stage','n','split_seed','variant']
    summary=frame.groupby(group)[METRICS].agg(['count','mean','std']).reset_index()
    summary.columns=['_'.join(c).rstrip('_') if isinstance(c,tuple) else c for c in summary.columns]
    core.write_csv(Path(out)/'results_summary.csv',summary)
    contrasts=[]
    for keys,part in frame.groupby(['stage','n','split_seed']):
        reference='kampa' if keys[0]=='1_3' else 'efficientat'
        ref=part[part.variant==reference][['seed']+METRICS]
        for variant,other in part.groupby('variant'):
            paired=other.merge(ref,on='seed',how='inner',suffixes=('','_reference'))
            for _,r in paired.iterrows():
                contrasts.append(dict(stage=keys[0],n=keys[1],split_seed=keys[2],variant=variant,reference=reference,seed=int(r.seed),
                    **{m+'_delta':r[m]-r[m+'_reference'] for m in METRICS}))
    core.write_csv(Path(out)/'paired_seed_differences.csv',pd.DataFrame(contrasts))
    effects=[]
    for keys,part in frame[frame.stage=='1_3'].groupby(['n','seed']):
        p=part.set_index('variant')
        if {'fixed_single','fixed_set','meta_single','kampa'}.issubset(p.index):
            r=dict(n=keys[0],seed=keys[1])
            for metric in METRICS:
                fs,ft,ms,mt=[float(p.loc[v,metric]) for v in ['fixed_single','fixed_set','meta_single','kampa']]
                r.update({metric+'_meta_effect_single':ms-fs,metric+'_meta_effect_set':mt-ft,
                          metric+'_set_effect_fixed':ft-fs,metric+'_set_effect_meta':mt-ms,
                          metric+'_interaction':(mt-ms)-(ft-fs)})
            effects.append(r)
    if effects:core.write_csv(Path(out)/'ablation_factorial_effects.csv',pd.DataFrame(effects))
    privileged=[]
    for keys,part in frame[frame.stage=='1_3'].groupby(['n','seed']):
        p=part.set_index('variant')
        if {'supervised','oracle'}.issubset(p.index):
            privileged.append(dict(n=keys[0],seed=keys[1],**{m+'_delta':float(p.loc['oracle',m]-p.loc['supervised',m]) for m in METRICS}))
    if privileged:core.write_csv(Path(out)/'oracle_vs_supervised.csv',pd.DataFrame(privileged))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--package',required=True)
    mode=ap.add_mutually_exclusive_group(required=True);mode.add_argument('--config');mode.add_argument('--aggregate')
    a=ap.parse_args();core=load_core(a.package)
    core.aggregate=lambda out:aggregate(out,core)
    if a.config:core.run_suite(json.loads(Path(a.config).read_text()))
    else:core.aggregate(a.aggregate)

if __name__=='__main__':main()
