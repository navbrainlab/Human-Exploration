#!/usr/bin/env python3
import argparse, ast, itertools, math
from pathlib import Path
import numpy as np
import pandas as pd

DIMS = ['dim1', 'dim2', 'dim3']
LEVELS = [1, 2, 3]
PAIRS = list(itertools.combinations(range(3), 2))

def parse(v):
    return v if isinstance(v, list) else ast.literal_eval(str(v))

def values(row, dim):
    return [int(x) for x in parse(row[dim])]

def row_counts(row, dim):
    v = values(row, dim)
    return [v[i:i+3].count(k) for i in (0, 3, 6) for k in LEVELS]

def pattern(row, dim):
    v = values(row, dim)
    return sum(max(v[i:i+3].count(k) for k in LEVELS) for i in (0, 3, 6))

def binary_entropy(q, eps):
    q = np.clip(q, eps, 1-eps)
    return -(q*np.log2(q) + (1-q)*np.log2(1-q))

def entropy(p, eps):
    p=np.clip(np.asarray(p,dtype=float),eps,1); p=p/p.sum()
    return float(-(p*np.log2(p)).sum())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--learning-rate', type=float, default=.001)
    ap.add_argument('--epsilon', type=float, default=1e-8)
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    df = df[(df['phase'] == 'P1') & df['dim4'].isna()].copy()
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    df = df.dropna(subset=['subject', 'round', 'score', 'reward_relevent_feature'])
    sigma = max(float(df['score'].std())/2, 1.0)
    nfeat = 1 + 3*9
    rows = []; trial_rows=[]
    for subject, sub in df.groupby('subject', sort=False):
        probs = np.ones(3)/3
        theta = [np.zeros(nfeat) for _ in PAIRS]
        for _, row in sub.sort_values('round').iterrows():
            x = np.array([1.0] + sum((row_counts(row, d) for d in DIMS), []))
            y = float(row['score']); preds = []; errors = []; masks = []
            for j, (a, b) in enumerate(PAIRS):
                mask = np.zeros(nfeat); mask[0] = 1
                for d in (a, b): mask[1+d*9:1+(d+1)*9] = 1
                masks.append(mask); preds.append(theta[j] @ (x*mask)); errors.append(y-preds[-1])
            z = np.log(np.clip(probs, args.epsilon, 1)) - np.square(errors)/(2*sigma**2)
            z -= z.max(); post = np.exp(z); post /= post.sum()
            trial_rows.append({'subject':subject,'round':int(row['round']),'prior_entropy_bits':entropy(probs,args.epsilon),'posterior_entropy_bits':entropy(post,args.epsilon),'trial_entropy_reduction_bits':entropy(probs,args.epsilon)-entropy(post,args.epsilon)})
            relevant = {int(v) for v in parse(row['reward_relevent_feature'])}
            for d, dim in enumerate(DIMS):
                q0 = sum(probs[i] for i, pair in enumerate(PAIRS) if d in pair)
                q1 = sum(post[i] for i, pair in enumerate(PAIRS) if d in pair)
                rel = d in relevant
                ci = max(0, math.log2(max(q1,args.epsilon)/max(q0,args.epsilon))) if rel else 0
                ei = max(0, math.log2(max(1-q1,args.epsilon)/max(1-q0,args.epsilon))) if not rel else 0
                rows.append({'subject':subject,'round':int(row['round']),'dimension':dim,'is_relevant':rel,'pattern':pattern(row,dim),'q_prior':q0,'q_post':q1,'dimension_entropy_reduction_bits':binary_entropy(q0,args.epsilon)-binary_entropy(q1,args.epsilon),'correct_confirmation_bits':ci,'correct_exclusion_bits':ei,'correct_information_bits':ci+ei})
            for j in range(3): theta[j] -= args.learning_rate*(-2*errors[j]*x*masks[j])
            probs = post
    result = pd.DataFrame(rows)
    result.to_csv(out/'task1_3d_dimension_information.csv', index=False)
    pd.DataFrame(trial_rows).to_csv(out/'task1_3d_trial_entropy_reduction.csv',index=False)
    result.groupby('pattern')['correct_information_bits'].agg(['mean','sem','size']).reset_index().to_csv(out/'task1_3d_correct_information_summary.csv', index=False)

if __name__ == '__main__': main()
