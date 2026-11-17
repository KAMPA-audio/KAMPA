"""Tables and figures for feedback items 1-4 from the 20261114-v2 runs (2026-11-16).

    python 만들기_1234번_자료.py --collect <Drive KAMPA_feedback_20261114_v2/<runtime id>>
    python 만들기_1234번_자료.py

--collect checks every downloaded run file against complete.json, recomputes the
four test metrics from test_predictions.csv and writes 원자료/. Without it, 표/ and
그림/ are rebuilt from 원자료/ only, so GitHub alone reproduces every number.
The hidden U labels in src/provenance_active.csv are read only here, after
training, to describe pseudo-label quality. No significance tests are computed.
"""
from pathlib import Path
import argparse, hashlib, json, shutil
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
RAW, TAB, FIG = BASE/'원자료', BASE/'표', BASE/'그림'
SRC = BASE.parent/'src'
CLASSES = ['heavy_impact','dragging','construction','machine_noise','talking','water_toilet','water_shower','dog_bark','others']
METRICS = ['accuracy','macro_precision','macro_recall','macro_f1']
METRIC_NAMES = ['Accuracy','Macro Precision','Macro Recall','Macro F1']
TAU1, TAU2 = .9, .95
ORDER = ['supervised','fixed_single','fixed_set','meta_single','kampa','meta_top1_matched','soft_selected','soft_all','oracle','efficientat']
NAMES = {'supervised':'Labeled only','fixed_single':'F-S','fixed_set':'F-T','meta_single':'M-S','kampa':'KAMPA (M-T)',
         'meta_top1_matched':'KAMPA samples, top-1 targets','soft_selected':'Soft-selected','soft_all':'Soft-all',
         'oracle':'Oracle (+ teacher TRAIN labels)','efficientat':'EfficientAT'}
SPLITS = {101:'Re-split 1',102:'Re-split 2',103:'Re-split 3'}
WIDTH = 16.8  # cm, the printed width of the v8 figures

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def scores(y,pred):
    cm=np.zeros((9,9));np.add.at(cm,(y,pred),1);tp=np.diag(cm)
    p=np.divide(tp,cm.sum(0),out=np.zeros(9),where=cm.sum(0)>0);r=np.divide(tp,cm.sum(1),out=np.zeros(9),where=cm.sum(1)>0)
    f=np.divide(2*p*r,p+r,out=np.zeros(9),where=(p+r)>0)
    return dict(accuracy=tp.sum()/cm.sum(),macro_precision=p.mean(),macro_recall=r.mean(),macro_f1=f.mean()),f

def write(path,frame):
    path.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(path,index=False,encoding='utf-8-sig')

def pseudo_stats(frame,hidden,keys):
    """Adoption follows make_targets: q1>=TAU1 single, else q1+q2>=TAU2 two-candidate."""
    u=frame[frame.filename.isin(hidden.index)].reset_index(drop=True);assert len(u)==len(hidden)==2000
    q=u[['q_'+c for c in CLASSES]].to_numpy(float);order=np.argsort(-q,axis=1,kind='stable');ii=np.arange(len(q))
    a,b=order[:,0],order[:,1];q1=q[ii,a];one=q1>=TAU1;two=(~one)&(q1+q[ii,b]>=TAU2)
    used=u.weight.to_numpy()>0;truth=np.array([CLASSES.index(c) for c in hidden.loc[u.filename]])
    assert (u['first'].to_numpy()[used]==a[used]).all()
    if keys['variant'] in ('fixed_set','kampa'):assert ((u.second.to_numpy()>=0)==(two&used)).all() and (used==(one|two)).all()
    if keys['variant'].endswith('_single'):assert (used==one).all()
    if keys['variant']=='soft_all':assert used.all()
    row=dict(**keys,unlabeled=len(u),used=int(used.sum()),single=int((one&used).sum()),two_candidate=int((two&used).sum()),
             other_used=int((used&~one&~two).sum()),excluded=int((~used).sum()),weight_sum=float(u.weight[used].sum()),
             single_agreement=float((a==truth)[one&used].mean()) if (one&used).any() else np.nan,
             two_candidate_coverage=float(((a==truth)|(b==truth))[two&used].mean()) if (two&used).any() else np.nan,
             top1_agreement_used=float((a==truth)[used].mean()))
    classes=[dict(**keys,predicted_class=c,single=int((one&used&(a==i)).sum()),two_candidate=int((two&used&(a==i)).sum())) for i,c in enumerate(CLASSES)]
    return row,classes,used,u.weight.to_numpy()

def collect(drive):
    drive=Path(drive);prov=pd.read_csv(SRC/'provenance_active.csv',keep_default_na=False)
    hidden=prov[(prov.version=='mark5.2')&(prov.split=='unlabeled')].set_index('filename').label
    runs,preds,pseudo,pclasses,training,files=[],[],[],[],[],{};runtimes=set();accepted={}
    for stage in ('1_3','4'):
        for d in sorted((drive/stage/'runs').iterdir()):
            complete=json.loads((d/'complete.json').read_text())
            for p in sorted(d.iterdir()):
                if p.name=='complete.json':continue
                assert sha(p)==complete['files'][p.name],p
                files[f'{stage}/runs/{d.name}/{p.name}']=complete['files'][p.name]
            m=json.loads((d/'metrics.json').read_text());r=m['recipe']
            assert r['stage']==stage and d.name==f"split{r['split_seed']}_n{r['n']}_s{r['seed']}_{r['variant']}",d
            runtimes.add(json.dumps(json.loads((d/'protocol.json').read_text())['runtime'],sort_keys=True))
            keys=dict(stage=stage,split_seed=r['split_seed'],n=r['n'],variant=r['variant'],seed=r['seed'])
            p=pd.read_csv(d/'test_predictions.csv',keep_default_na=False)
            got,_=scores(np.array([CLASSES.index(c) for c in p.label]),np.array([CLASSES.index(c) for c in p.prediction]))
            for k in METRICS:assert abs(got[k]-m['test'][k])<1e-12,(d,k)
            assert len(p)==m['test']['n']
            history=pd.read_csv(d/'history.csv')
            runs.append(dict(**keys,**{k:m['test'][k] for k in METRICS},n_test=len(p),best_epoch=m['best_epoch'],
                             epochs_run=len(history),best_val_loss=m['best_val_loss']))
            p.insert(0,'run',d.name);p[[c for c in p.columns if c.startswith('prob_')]]=p.filter(like='prob_').round(6);preds.append((stage,p))
            if r['variant'] not in ('supervised','oracle','efficientat'):
                row,classes,used,weight=pseudo_stats(pd.read_csv(d/'pseudo_targets.csv',keep_default_na=False),hidden,keys)
                pseudo.append(row);pclasses+=classes;accepted[(stage,r['split_seed'],r['n'],r['seed'],r['variant'])]=(used,weight)
            if (d/'train_manifest.csv').exists():
                t=pd.read_csv(d/'train_manifest.csv',keep_default_na=False);teacher=t.origin_split=='teacher_train'
                assert (~teacher).sum()==r['n'] and t.path.is_unique
                row=dict(**keys,labeled=int((~teacher).sum()),teacher_train=int(teacher.sum()),total=len(t))
                if teacher.any():row.update(teacher_original=int((~t.augmented[teacher].astype(str).str.lower().eq('true')).sum()),
                                            teacher_source_groups=int(t.group[teacher].nunique()))
                row.update({'class_'+c:int((t.label==c).sum()) for c in CLASSES});training.append(row)
    assert len(runtimes)==1,'runs from different runtimes'
    # The matched and soft-selected controls must keep KAMPA's accepted rows and weights.
    for (stage,split,n,seed,variant),(used,weight) in accepted.items():
        if variant in ('meta_top1_matched','soft_selected'):
            ref_used,ref_weight=accepted[(stage,split,n,seed,'kampa')]
            assert (used==ref_used).all() and np.array_equal(weight,ref_weight)
    RAW.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(runs).sort_values(['stage','split_seed','n','variant','seed'])
    assert len(frame)==150 and not frame.duplicated(['stage','split_seed','n','variant','seed']).any()
    write(RAW/'runs.csv',frame)
    for stage in ('1_3','4'):write(RAW/f'test_predictions_{stage}.csv',pd.concat([p for s,p in preds if s==stage],ignore_index=True))
    write(RAW/'pseudo_labels.csv',pd.DataFrame(pseudo));write(RAW/'pseudo_label_classes.csv',pd.DataFrame(pclasses))
    write(RAW/'training_sets.csv',pd.DataFrame(training))
    composition=[]
    for split in sorted(int(p.name) for p in (drive/'4'/'splits').iterdir()):
        for part in ('train','val','test','unlabeled'):
            t=pd.read_csv(drive/'4'/'splits'/str(split)/f'{part}.csv',keep_default_na=False)
            files[f'4/splits/{split}/{part}.csv']=sha(drive/'4'/'splits'/str(split)/f'{part}.csv')
            composition.append(dict(split_seed=split,partition=part,clips=len(t),source_groups=t.group.nunique(),
                                    **{'class_'+c:int((t.label==c).sum()) for c in CLASSES} if part!='unlabeled' else {}))
    write(RAW/'resplit_composition.csv',pd.DataFrame(composition))
    audit=pd.read_csv(drive/'1_3'/'audit'/'teacher_oracle_audit.csv',keep_default_na=False)
    files['1_3/audit/teacher_oracle_audit.csv']=sha(drive/'1_3'/'audit'/'teacher_oracle_audit.csv')
    write(RAW/'teacher_train_accounting.csv',audit.assign(exclude_reason=audit.exclude_reason.replace('','kept'))
          .groupby(['exclude_reason','label']).size().rename('rows').reset_index())
    for stage in ('1_3','4'):
        shutil.copyfile(drive/stage/'audit'/'preflight.json',RAW/f'preflight_{stage}.json')
        files[f'{stage}/audit/preflight.json']=sha(drive/stage/'audit'/'preflight.json')
    (RAW/'source_files.json').write_text(json.dumps(dict(
        drive_folder=f'NewProject/codes/experiments/KAMPA_feedback_20261114_v2/{drive.name}',runtime=json.loads(runtimes.pop()),
        hidden_unlabeled_labels='src/provenance_active.csv (mark5.2 unlabeled rows), read after training for pseudo-label quality only',
        sha256=files),ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print(f'Collected {len(frame)} runs and {len(files)} verified source files into {RAW}')

def num(v):
    """Signed difference with a true minus sign; a difference that rounds to zero is unsigned."""
    s=f'{v:+.2f}';return '0.00' if float(s)==0 else s.replace('-','−')
def ms(values):return f'{np.mean(values):.2f} ± {np.std(values,ddof=1):.2f}'
def dms(values):return f'{num(np.mean(values))} ± {np.std(values,ddof=1):.2f}'
def cnt(values,digits=0):
    """Counts (digits=0) and rates that are equal for all five seeds are written once."""
    values=pd.Series(values,dtype=float);spread=max(digits,1)
    if values.nunique()==1:return f'{values.iloc[0]:,.{digits}f}'
    return f'{values.mean():,.{spread}f} ± {values.std(ddof=1):.{spread}f}'
def ordered(frame,*keys):return frame.assign(_o=frame.variant.map(ORDER.index)).sort_values([*keys,'_o']+(['seed'] if 'seed' in frame else [])).drop(columns='_o')

def paired(frame,a,b,n,split=0,stage='1_3'):
    """Per-seed a - b in percentage points, one row per seed."""
    part=frame[(frame.stage==stage)&(frame.n==n)&(frame.split_seed==split)]
    x=part[part.variant==a].set_index('seed')[METRICS];y=part[part.variant==b].set_index('seed')[METRICS]
    assert list(x.index)==list(y.index)==[1,2,3,4,5];return (x-y)*100

def perf(frame,variant,n,split=0,stage='1_3'):
    part=frame[(frame.stage==stage)&(frame.n==n)&(frame.split_seed==split)&(frame.variant==variant)]
    assert len(part)==5;return part[METRICS]*100

def tables(frame,pseudo,pclasses,sets,composition,accounting,preds):
    out={}
    rows=[]
    design={'supervised':('none','none'),'fixed_single':('fixed odds','single label only'),'fixed_set':('fixed odds','single + two-candidate'),
            'meta_single':('meta-classifier','single label only'),'kampa':('meta-classifier','single + two-candidate'),
            'meta_top1_matched':('meta-classifier','KAMPA rows, top-1 target')}
    for n in (200,2000):
        for v,(integ,target) in design.items():
            p=perf(frame,v,n);rows.append({'N':n,'Condition':NAMES[v],'Score integration':integ,'Pseudo-label targets':target,
                                          **{name:ms(p[k]) for k,name in zip(METRICS,METRIC_NAMES)}})
    out['표1_피드백1_2x2_성능']=pd.DataFrame(rows)
    effects=[('Integration effect, single label','M-S − F-S','meta_single','fixed_single'),
             ('Integration effect, two-candidate','M-T − F-T','kampa','fixed_set'),
             ('Two-candidate effect, fixed integration','F-T − F-S','fixed_set','fixed_single'),
             ('Two-candidate effect, meta-classifier','M-T − M-S','kampa','meta_single'),
             ('Two-candidate loss on the same rows','M-T − KAMPA samples top-1','kampa','meta_top1_matched'),
             ('Adding two-candidate rows as top-1','KAMPA samples top-1 − M-S','meta_top1_matched','meta_single'),
             ('KAMPA vs labeled only','M-T − labeled only','kampa','supervised')]
    rows=[]
    for n in (200,2000):
        for label,contrast,a,b in effects:
            d=paired(frame,a,b,n);rows.append({'N':n,'Effect':label,'Contrast':contrast,**{name:dms(d[k]) for k,name in zip(METRICS,METRIC_NAMES)}})
        d=(paired(frame,'kampa','meta_single',n)-paired(frame,'fixed_set','fixed_single',n))
        rows.insert(len(rows)-3,{'N':n,'Effect':'Interaction','Contrast':'(M-T − M-S) − (F-T − F-S)',**{name:dms(d[k]) for k,name in zip(METRICS,METRIC_NAMES)}})
    out['표2_피드백1_효과']=pd.DataFrame(rows)
    rows=[]
    for n in (200,2000):
        for v in ['fixed_single','fixed_set','meta_single','kampa','meta_top1_matched','soft_selected','soft_all']:
            p=pseudo[(pseudo.stage=='1_3')&(pseudo.n==n)&(pseudo.variant==v)];assert len(p)==5
            rows.append({'N':n,'Condition':NAMES[v],'Unlabeled clips':'2,000','Used':cnt(p.used),'Single label':cnt(p.single),
                         'Two-candidate':cnt(p.two_candidate),'Other (soft-all only)':cnt(p.other_used),'Excluded':cnt(p.excluded),
                         'Use rate (%)':cnt(p.used/p.unlabeled*100,2),'Single-label agreement (%)':cnt(p.single_agreement*100,2),
                         'Two-candidate coverage (%)':cnt(p.two_candidate_coverage*100,2) if p.two_candidate.sum() else '—',
                         'Top-1 agreement of used rows (%)':cnt(p.top1_agreement_used*100,2),'Weight sum':cnt(p.weight_sum,2)})
    out['표3_피드백1_가짜정답_채택과_품질']=pd.DataFrame(rows)
    rows=[]
    for n in (200,2000):
        for v,target in [('kampa','single label or two-candidate set'),('soft_selected','full 9-class q (T = 1), KAMPA rows and weights'),('soft_all','full 9-class q (T = 1), all 2,000 clips, uniform weight')]:
            p=perf(frame,v,n);u=pseudo[(pseudo.stage=='1_3')&(pseudo.n==n)&(pseudo.variant==v)]
            row={'N':n,'Condition':NAMES[v],'Target':target,'Unlabeled clips used':cnt(u.used),**{name:ms(p[k]) for k,name in zip(METRICS,METRIC_NAMES)}}
            if v!='kampa':d=paired(frame,v,'kampa',n);row.update({'Δ '+name+' vs KAMPA':dms(d[k]) for k,name in zip(METRICS,METRIC_NAMES)})
            rows.append(row)
    out['표4_피드백2_소프트타깃']=pd.DataFrame(rows).fillna('—')
    o=sets[sets.variant=='oracle'];kept=accounting[accounting.exclude_reason=='kept'].rows.sum()
    rows=[{'Item':'Teacher TRAIN rows (8 teachers)','N':'—','Clips':f'{int(accounting.rows.sum()):,}'}]
    for reason in ('student_eval_source_group','student_eval_file_hash','duplicate_teacher_file_hash'):
        rows.append({'Item':f'Excluded: {reason}','N':'—','Clips':f'{int(accounting[accounting.exclude_reason==reason].rows.sum()):,}'})
    assert set(accounting.exclude_reason)<={'kept','student_eval_source_group','student_eval_file_hash','duplicate_teacher_file_hash'}
    rows.append({'Item':'Unique teacher TRAIN clips with nine-class labels','N':'—','Clips':f'{int(kept):,}'})
    for c in CLASSES:rows.append({'Item':f'  {c}','N':'—','Clips':f'{int(accounting[(accounting.exclude_reason=="kept")&(accounting.label==c)].rows.sum()):,}'})
    for n in (200,2000):
        g=o[o.n==n];assert len(g)==5
        rows+=[{'Item':'Removed: same file as a labeled training clip','N':n,'Clips':cnt(kept-g.teacher_train)},
               {'Item':'Teacher TRAIN clips added','N':n,'Clips':cnt(g.teacher_train)},
               {'Item':'  original recordings','N':n,'Clips':cnt(g.teacher_original)},
               {'Item':'  source groups','N':n,'Clips':cnt(g.teacher_source_groups)},
               {'Item':'Total supervised training clips','N':n,'Clips':cnt(g.total)}]
    out['표5_피드백3_oracle_자료구성']=pd.DataFrame(rows)
    rows=[]
    for n in (200,2000):
        g=o[o.n==n]
        for v,extra,unl in [('supervised','0','0'),('kampa','0','2,000'),('oracle',cnt(g.teacher_train),'0')]:
            p=perf(frame,v,n);rows.append({'N':n,'Condition':NAMES[v],'Labeled clips':f'{n:,}','Teacher TRAIN labels added':extra,'Unlabeled clips':unl,
                                          **{name:ms(p[k]) for k,name in zip(METRICS,METRIC_NAMES)}})
        for label,a,b in [('KAMPA − labeled only','kampa','supervised'),('Oracle − labeled only','oracle','supervised'),('Oracle − KAMPA','oracle','kampa')]:
            d=paired(frame,a,b,n);rows.append({'N':n,'Condition':label,**{name:dms(d[k]) for k,name in zip(METRICS,METRIC_NAMES)}})
        k,s,r=[perf(frame,v,n).mean() for v in ('kampa','supervised','oracle')]
        rows.append({'N':n,'Condition':'Share of the oracle gain reached by KAMPA (%)',**{name:f'{(k[m]-s[m])/(r[m]-s[m])*100:.1f}' for m,name in zip(METRICS,METRIC_NAMES)}})
    out['표6_피드백3_oracle_성능']=pd.DataFrame(rows).astype(object).fillna('')
    c=composition.copy();c.insert(0,'Split',c.split_seed.map(SPLITS))
    c=c.drop(columns='split_seed').rename(columns={'partition':'Partition','clips':'Clips','source_groups':'Source groups',**{'class_'+k:k for k in CLASSES}})
    c['Partition']=c.Partition.replace({'train':'train candidates','unlabeled':'unlabeled (original U)'})
    for k in ['Clips','Source groups',*CLASSES]:c[k]=c[k].map(lambda v:'' if pd.isna(v) else f'{int(v):,}')
    out['표7_피드백4_재분할구성']=c
    rows=[]
    for n in (200,2000):
        for split in SPLITS:
            for v in ('kampa','efficientat'):
                p=perf(frame,v,n,split,'4');rows.append({'N':n,'Split':SPLITS[split],'Model':NAMES[v].replace(' (M-T)',''),**{name:ms(p[k]) for k,name in zip(METRICS,METRIC_NAMES)}})
            d=paired(frame,'kampa','efficientat',n,split,'4')
            rows.append({'N':n,'Split':SPLITS[split],'Model':'KAMPA − EfficientAT',**{name:dms(d[k]) for k,name in zip(METRICS,METRIC_NAMES)},
                         'Runs with higher Macro F1 (KAMPA)':f'{int((d.macro_f1>0).sum())} / 5'})
        for v in ('kampa','efficientat'):
            p=pd.concat([perf(frame,v,n,s,'4') for s in SPLITS]);rows.append({'N':n,'Split':'Mean of 3 re-splits','Model':NAMES[v].replace(' (M-T)',''),**{name:f'{p[k].mean():.2f}' for k,name in zip(METRICS,METRIC_NAMES)}})
        d=pd.concat([paired(frame,'kampa','efficientat',n,s,'4') for s in SPLITS])
        rows.append({'N':n,'Split':'Mean of 3 re-splits','Model':'KAMPA − EfficientAT',**{name:num(d[k].mean()) for k,name in zip(METRICS,METRIC_NAMES)},
                     'Runs with higher Macro F1 (KAMPA)':f'{int((d.macro_f1>0).sum())} / 15'})
    out['표8_피드백4_성능']=pd.DataFrame(rows).fillna('')
    a=frame.copy();a[METRICS]=(a[METRICS]*100).map(lambda v:f'{v:.2f}');a['condition']=a.variant.map(NAMES)
    out['부록표A_회차별_전체결과']=ordered(a,'stage','split_seed','n')[['stage','split_seed','n','condition','variant','seed']+METRICS+['n_test','best_epoch','epochs_run']]
    names=frame.assign(run='split'+frame.split_seed.astype(str)+'_n'+frame.n.astype(str)+'_s'+frame.seed.astype(str)+'_'+frame.variant).set_index(['stage','run'])
    rows=[]
    for (stage,run),p in preds.groupby(['stage','run']):
        _,f=scores(np.array([CLASSES.index(c) for c in p.label]),np.array([CLASSES.index(c) for c in p.prediction]))
        r=names.loc[(stage,run)];rows.append(dict(stage=stage,n=r.n,variant=r.variant,**{c:f[i]*100 for i,c in enumerate(CLASSES)}))
    assert len(rows)==150
    f=pd.DataFrame(rows).groupby(['stage','n','variant'])[CLASSES].mean().map(lambda v:f'{v:.2f}').reset_index()
    f.insert(3,'condition',f.variant.map(NAMES));out['부록표B_클래스별_F1']=ordered(f,'stage','n')
    p=pclasses[pclasses.stage=='1_3'].groupby(['n','variant','predicted_class'])[['single','two_candidate']].mean().round(1).reset_index()
    p=p[p.variant.isin(['fixed_single','fixed_set','meta_single','kampa'])].copy();p.insert(2,'condition',p.variant.map(NAMES))
    p=p.assign(_c=p.predicted_class.map(CLASSES.index),_o=p.variant.map(ORDER.index)).sort_values(['n','_o','_c']).drop(columns=['_c','_o'])
    out['부록표C_클래스별_가짜정답_채택수']=p
    for name,t in out.items():write(TAB/f'{name}.csv',t)
    return out

def style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,'xtick.labelsize':8.5,'ytick.labelsize':8.5,
                         'legend.fontsize':8.5,'axes.linewidth':.65,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    return plt

def limits(lo,hi):
    lo,hi=np.floor(lo-.4),np.ceil(hi+.4);step=1 if hi-lo<=8 else 2 if hi-lo<=16 else 3;return lo,hi,step

def place(fig,top,height,left,width):
    """Axes placed in cm from the top-left corner, so type sizes are the printed sizes."""
    w,h=fig.get_size_inches()*2.54
    return fig.add_axes([left/w,1-(top+height)/h,width/w,height/h])

def errorbars(ax,x,mean,sd,color,marker,filled,label,size=4.5):
    ax.errorbar(x,mean,yerr=sd,fmt=marker,color=color,markersize=size,markerfacecolor=color if filled else 'white',
                markeredgewidth=1,capsize=2.6,capthick=.9,elinewidth=1.05,label=label,zorder=3)

def finish(ax):
    ax.grid(axis='y',color='#E4E4E4',linewidth=.55,zorder=0);ax.spines[['right','top']].set_visible(False)

def metric_panels(fig,frame,series,top):
    """series: (variant, label, color, marker, filled). x = four metrics, one 6.95-cm panel per N."""
    axes=[]
    for j,n in enumerate((200,2000)):
        ax=place(fig,top,5.0,1.45+8.15*j,6.95);axes.append(ax)
        data=[perf(frame,v,n) for v,*_ in series];offsets=np.linspace(-.27,.27,len(series))
        lo,hi,step=limits(min((d.mean()-d.std(ddof=1)).min() for d in data),max((d.mean()+d.std(ddof=1)).max() for d in data))
        for d,(v,label,color,marker,filled),off in zip(data,series,offsets):
            errorbars(ax,np.arange(4)+off,d.mean().to_numpy(),d.std(ddof=1).to_numpy(),color,marker,filled,label)
        ax.set_xticks(np.arange(4),['Accuracy','Macro\nPrecision','Macro\nRecall','Macro F1']);ax.set_xlim(-.55,3.55);ax.tick_params(axis='x',length=0,pad=5)
        ax.set_ylim(lo,hi);ax.set_yticks(np.arange(lo,hi+.01,step));ax.set_ylabel('Score (%)' if j==0 else '')
        ax.set_title(f'N = {n:,} labeled training clips',fontsize=9,pad=6);finish(ax)
    return axes

def legend(fig,ax,ncol=1):
    w,h=fig.get_size_inches()*2.54
    fig.legend(*ax.get_legend_handles_labels(),loc='upper left',bbox_to_anchor=(1.25/w,1-.2/h),frameon=False,ncol=ncol,
               handlelength=1.6,columnspacing=1.6,labelspacing=.45,borderaxespad=0)

def gain_table(fig,top,title,rows):
    """rows: (label, four N = 200 values, four N = 2,000 values) in percentage points."""
    w,h=fig.get_size_inches()*2.54;left,width,first=.6,15.95,.36;col=(1-first)/8
    fig.text(left/w,1-top/h,title,fontsize=8.5,weight='bold',va='top')
    for k,n in enumerate((200,2000)):
        fig.text((left+width*(first+col*(4*k+2)))/w,1-(top+.62)/h,f'N = {n:,}',fontsize=8,ha='center',va='center',color='#444444')
    ax=place(fig,top+.85,(1+len(rows))*.42,left,width);ax.axis('off')
    cells=[['']+['Acc','P','R','F1']*2]+[[label]+[num(v) for v in a]+[num(v) for v in b] for label,a,b in rows]
    table=ax.table(cellText=cells,cellLoc='center',colWidths=[first]+[col]*8,bbox=[0,0,1,1]);table.auto_set_font_size(False);table.set_fontsize(7.6)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor('white');cell.set_facecolor('white');cell.PAD=.02
        if c==0:cell.get_text().set_ha('left')
        if r==0:cell.get_text().set_color('#444444')

def check_layout(fig):
    """Every label must lie inside the figure and every table entry inside its cell (the first draft clipped both)."""
    from matplotlib.table import Table
    from matplotlib.text import Text
    fig.canvas.draw();renderer=fig.canvas.get_renderer();box=fig.bbox
    for t in fig.findobj(Text):
        if t.get_visible() and t.get_text().strip():
            e=t.get_window_extent(renderer)
            assert box.x0-1<=e.x0 and e.x1<=box.x1+1 and box.y0-1<=e.y0 and e.y1<=box.y1+1,('outside figure',t.get_text(),e.bounds)
    for table in fig.findobj(Table):
        for cell in table.get_celld().values():
            if cell.get_text().get_text():
                assert cell.get_text().get_window_extent(renderer).width<=cell.get_window_extent(renderer).width,('cell overflow',cell.get_text().get_text())

def save(fig,name,provenance):
    check_layout(fig);FIG.mkdir(parents=True,exist_ok=True)
    # Fixed metadata keeps the PDF/SVG hashes stable between rebuilds.
    for ext,meta in (('png',None),('pdf',{'CreationDate':None}),('svg',{'Date':None})):fig.savefig(FIG/f'{name}.{ext}',dpi=600,facecolor='white',metadata=meta)
    provenance[name]={ext:sha(FIG/f'{name}.{ext}') for ext in ('png','pdf','svg')}

def mean_gain(frame,a,b,n,split=0,stage='1_3'):return paired(frame,a,b,n,split,stage).mean().to_numpy()

def figures(frame):
    plt=style();provenance={};cm=1/2.54
    fig=plt.figure(figsize=(WIDTH*cm,13.4*cm),facecolor='white')
    series=[('fixed_single','F-S: fixed integration, single labels only','#8C8C8C','o',False),
            ('fixed_set','F-T: fixed integration, single labels + two-candidate sets','#3D3D3D','s',False),
            ('meta_single','M-S: meta-classifier, single labels only','#5FA8CC','^',False),
            ('kampa','KAMPA (M-T): meta-classifier, single labels + two-candidate sets','#0077A5','D',True)]
    axes=metric_panels(fig,frame,series,top=2.75);legend(fig,axes[0])
    rows=[(label,mean_gain(frame,a,b,200),mean_gain(frame,a,b,2000)) for label,a,b in [
        ('Integration, single (M-S − F-S)','meta_single','fixed_single'),('Integration, two-cand. (M-T − F-T)','kampa','fixed_set'),
        ('Two-cand., fixed (F-T − F-S)','fixed_set','fixed_single'),('Two-cand., meta (M-T − M-S)','kampa','meta_single')]]
    rows.append(('Interaction',*[(paired(frame,'kampa','meta_single',n)-paired(frame,'fixed_set','fixed_single',n)).mean().to_numpy() for n in (200,2000)]))
    rows.append(('Two-cand. loss on the same rows',mean_gain(frame,'kampa','meta_top1_matched',200),mean_gain(frame,'kampa','meta_top1_matched',2000)))
    gain_table(fig,9.25,'Mean paired differences (percentage points)',rows)
    save(fig,'그림1_피드백1_2x2_기여도',provenance);plt.close(fig)

    fig=plt.figure(figsize=(WIDTH*cm,11.3*cm),facecolor='white')
    series=[('kampa','KAMPA: single label or two-candidate set','#0077A5','D',True),
            ('soft_selected','Soft-selected: full q on KAMPA rows and weights','#3D3D3D','s',False),
            ('soft_all','Soft-all: full q on all 2,000 unlabeled clips, uniform weight','#8C8C8C','o',False)]
    axes=metric_panels(fig,frame,series,top=2.35);legend(fig,axes[0])
    rows=[(label,mean_gain(frame,a,b,200),mean_gain(frame,a,b,2000)) for label,a,b in [('Soft-selected − KAMPA','soft_selected','kampa'),('Soft-all − KAMPA','soft_all','kampa')]]
    gain_table(fig,8.85,'Mean paired differences (percentage points)',rows)
    save(fig,'그림2_피드백2_소프트타깃',provenance);plt.close(fig)

    fig=plt.figure(figsize=(WIDTH*cm,11.7*cm),facecolor='white')
    series=[('supervised','Labeled only','#8C8C8C','o',False),('kampa','KAMPA: labeled + 2,000 unlabeled clips','#0077A5','D',True),
            ('oracle','Oracle reference: labeled + 12,477 teacher TRAIN clips with labels','#3D3D3D','s',False)]
    axes=metric_panels(fig,frame,series,top=2.35);legend(fig,axes[0])
    rows=[(label,mean_gain(frame,a,b,200),mean_gain(frame,a,b,2000)) for label,a,b in [('KAMPA − labeled only','kampa','supervised'),('Oracle − labeled only','oracle','supervised'),('Oracle − KAMPA','oracle','kampa')]]
    gain_table(fig,8.85,'Mean paired differences (percentage points)',rows)
    save(fig,'그림3_피드백3_oracle',provenance);plt.close(fig)

    fig=plt.figure(figsize=(WIDTH*cm,13.9*cm),facecolor='white')
    models=[('kampa','KAMPA','#0077A5','D',True),('efficientat','EfficientAT','#3D3D3D','s',False)]
    for i,n in enumerate((200,2000)):
        stats={v:[perf(frame,v,n,s,'4') for s in SPLITS] for v,*_ in models};top=1.35+4.4*i
        lo,hi,step=limits(min((d.mean()-d.std(ddof=1)).min() for ds in stats.values() for d in ds),max((d.mean()+d.std(ddof=1)).max() for ds in stats.values() for d in ds))
        fig.text(.3/WIDTH,1-(top+1.7)/13.9,f'N = {n:,}',rotation=90,ha='center',va='center',fontsize=9)
        for j,(k,name) in enumerate(zip(METRICS,METRIC_NAMES)):
            ax=place(fig,top,3.4,1.75+3.7875*j,3.4375)
            for (v,label,color,marker,filled),off in zip(models,(-.13,.13)):
                errorbars(ax,np.arange(3)+off,[d[k].mean() for d in stats[v]],[d[k].std(ddof=1) for d in stats[v]],color,marker,filled,label,4.2)
            ax.set_xticks(np.arange(3),['1','2','3']);ax.set_xlim(-.5,2.5);ax.set_ylim(lo,hi);ax.set_yticks(np.arange(lo,hi+.01,step));ax.tick_params(axis='x',length=0,pad=3)
            if i==0:ax.set_title(name,fontsize=9,pad=5)
            if i==1:ax.set_xlabel('Re-split',fontsize=8.5,labelpad=2)
            if j==0:ax.set_ylabel('Score (%)')
            else:ax.tick_params(axis='y',labelleft=False)
            finish(ax)
            if i==0 and j==0:first=ax
    legend(fig,first,ncol=2)
    rows=[(SPLITS[s],mean_gain(frame,'kampa','efficientat',200,s,'4'),mean_gain(frame,'kampa','efficientat',2000,s,'4')) for s in SPLITS]
    rows.append(('Mean of 3 re-splits',*[pd.concat([paired(frame,'kampa','efficientat',n,s,'4') for s in SPLITS]).mean().to_numpy() for n in (200,2000)]))
    gain_table(fig,10.45,'KAMPA − EfficientAT, mean paired differences (percentage points)',rows)
    save(fig,'그림4_피드백4_재분할',provenance);plt.close(fig)
    provenance['source']={'원자료/runs.csv':sha(RAW/'runs.csv')}
    (FIG/'그림_해시.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--collect');a=ap.parse_args()
    if a.collect:collect(a.collect)
    # '1_3' would otherwise be read as the integer 13.
    read=lambda name:pd.read_csv(RAW/name,dtype={'stage':str})
    frame,pseudo,pclasses,sets=[read(f) for f in ('runs.csv','pseudo_labels.csv','pseudo_label_classes.csv','training_sets.csv')]
    composition=pd.read_csv(RAW/'resplit_composition.csv')
    accounting=pd.read_csv(RAW/'teacher_train_accounting.csv',keep_default_na=False)
    preds=pd.concat([pd.read_csv(RAW/f'test_predictions_{s}.csv',keep_default_na=False).assign(stage=s) for s in ('1_3','4')],ignore_index=True)
    tables(frame,pseudo,pclasses,sets,composition,accounting,preds)
    figures(frame)
    print(f'Wrote {len(list(TAB.glob("*.csv")))} tables and {len(list(FIG.glob("*.png")))} figures')

if __name__=='__main__':main()
