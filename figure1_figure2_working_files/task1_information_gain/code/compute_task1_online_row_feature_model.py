#!/usr/bin/env python3
import argparse, ast, itertools, json, math
from pathlib import Path
import numpy as np
import pandas as pd

DIMS=['dim1','dim2','dim3','dim4']; LEVELS=[1,2,3]; ROWS=range(3)
PAIRS=list(itertools.combinations(range(4),2)); NAMES=[DIMS[a]+DIMS[b] for a,b in PAIRS]

def parse(v):
    if pd.isna(v): return []
    return v if isinstance(v,list) else ast.literal_eval(str(v))

def dim_values(row,d): return [int(x) for x in parse(row[d])]

def row_counts(row,d):
    vals=dim_values(row,d); out=[]
    for r in ROWS:
        g=vals[r*3:r*3+3]; out.extend([g.count(k) for k in LEVELS])
    return out

def pattern(row,d):
    vals=dim_values(row,d)
    return int(sum(max(sum(x==k for x in vals[r*3:r*3+3]) for k in LEVELS) for r in ROWS))

def ent(p,eps):
    p=np.clip(p,eps,1); p/=p.sum(); return float(-(p*np.log2(p)).sum())

def bh(q,eps):
    q=min(max(float(q),eps),1-eps); return float(-(q*math.log2(q)+(1-q)*math.log2(1-q)))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--learning-rate',type=float,default=.001); ap.add_argument('--sigma',type=float,default=None); ap.add_argument('--epsilon',type=float,default=1e-8); args=ap.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(args.input); nraw=len(raw); df=raw.dropna(subset=['subject','round','score','reward_relevent_feature','dim4']).copy(); df.score=pd.to_numeric(df.score,errors='coerce'); df=df.dropna(subset=['score']); sigma=args.sigma or max(float(df.score.std())/2,1.)
    nfeat=1+4*3*3; prior0=np.ones(6)/6; rows=[]; primary=[]; trial_rows=[]
    for subject,sub in df.groupby('subject',sort=False):
        p=prior0.copy(); theta={n:np.zeros(nfeat) for n in NAMES}
        for _,r in sub.sort_values('round').iterrows():
            x=np.array([1.0]+sum((row_counts(r,d) for d in DIMS),[]),dtype=float); y=float(r.score); preds=[]; errors=[]
            for n,(a,b) in zip(NAMES,PAIRS):
                mask=np.zeros(nfeat); mask[0]=1
                for d in (a,b): mask[1+d*9:1+(d+1)*9]=1
                preds.append(float(theta[n]@(x*mask))); errors.append(y-preds[-1])
            ll=-.5*np.square(errors)/(sigma**2); z=np.log(np.clip(p,args.epsilon,1))+ll; z-=z.max(); post=np.exp(z); post/=post.sum()
            trial_rows.append({'subject':subject,'round':int(r['round']),'prior_entropy_bits':ent(p,args.epsilon),'posterior_entropy_bits':ent(post,args.epsilon),'trial_entropy_reduction_bits':ent(p,args.epsilon)-ent(post,args.epsilon)})
            relevant={int(v) for v in parse(r.reward_relevent_feature)}
            metrics=[]
            for d,dim in enumerate(DIMS):
                q0=sum(p[i] for i,pair in enumerate(PAIRS) if d in pair); q1=sum(post[i] for i,pair in enumerate(PAIRS) if d in pair); der=bh(q0,args.epsilon)-bh(q1,args.epsilon); rel=d in relevant
                ci=max(0,math.log2(max(q1,args.epsilon)/max(q0,args.epsilon))) if rel else 0
                ei=max(0,math.log2(max(1-q1,args.epsilon)/max(1-q0,args.epsilon))) if not rel else 0
                pat=pattern(r,dim); item={'subject':subject,'round':int(r['round']),'dimension':dim,'is_relevant':rel,'pattern':pat,'q_prior':q0,'q_post':q1,'dimension_entropy_reduction_bits':der,'correct_confirmation_bits':ci,'correct_exclusion_bits':ei,'correct_information_bits':ci+ei}; rows.append(item); metrics.append((dim,pat,ci,ei,rel))
            maxpat=max(x[1] for x in metrics)
            for dim,pat,ci,ei,rel in metrics:
                if pat==maxpat: primary.append({'subject':subject,'round':int(r['round']),'dimension':dim,'is_relevant':rel,'pattern':pat,'correct_confirmation_bits':ci,'correct_exclusion_bits':ei,'correct_information_bits':ci+ei})
            for i,n in enumerate(NAMES):
                a,b=PAIRS[i]; mask=np.zeros(nfeat); mask[0]=1
                for d in (a,b): mask[1+d*9:1+(d+1)*9]=1
                e=errors[i]; theta[n]-=args.learning_rate*(-2*e*x*mask)
            p=post
    d=pd.DataFrame(rows); pr=pd.DataFrame(primary); tr=pd.DataFrame(trial_rows)
    def summary(z): return z.groupby('pattern').agg(n=('correct_information_bits','size'),mean_correct_confirmation=('correct_confirmation_bits','mean'),mean_correct_exclusion=('correct_exclusion_bits','mean'),mean_correct_information=('correct_information_bits','mean'),sem_correct_information=('correct_information_bits','sem')).reset_index()
    d.to_csv(out/'task1_online_row_feature_dimension_information.csv',index=False); pr.to_csv(out/'task1_online_row_feature_primary_information.csv',index=False); tr.to_csv(out/'task1_4d_trial_entropy_reduction.csv',index=False); summary(d).to_csv(out/'task1_online_row_feature_pattern_summary.csv',index=False); summary(pr).to_csv(out/'task1_online_row_feature_primary_pattern_summary.csv',index=False)
    (out/'task1_online_row_feature_config.json').write_text(json.dumps({'model':'online row-by-feature parameters; theta represents row_weight times feature_value','candidate_pairs':NAMES,'patterns':[3,5,6,7,9],'sigma':sigma,'n_4d_rows':len(df),'n_excluded_non_4d_rows':nraw-len(df)},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
