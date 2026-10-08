#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--prefix', required=True)
    ap.add_argument('--mode', choices=['primary_only','all_dimensions'], default='primary_only')
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(args.input)
    keys = ['subject', 'round']
    d['trial_max_pattern'] = d.groupby(keys)['pattern'].transform('max')
    kept = d[d['pattern'] == d['trial_max_pattern']].copy() if args.mode == 'primary_only' else d.copy()
    trial = kept.groupby(keys, as_index=False).agg(
        pattern=('trial_max_pattern', 'first'),
        n_tied_max_dimensions=('dimension', 'size'),
        correct_information_bits=('correct_information_bits', 'mean'),
    )
    subject = trial.groupby(['subject', 'pattern'], as_index=False).agg(
        correct_information_bits=('correct_information_bits', 'mean'),
        n_trials=('round', 'size'),
    )
    summary = subject.groupby('pattern', as_index=False).agg(
        mean=('correct_information_bits', 'mean'),
        sem=('correct_information_bits', 'sem'),
        n_subjects=('subject', 'size'),
    )
    trial.to_csv(out / f'{args.prefix}_trial_primary.csv', index=False)
    subject.to_csv(out / f'{args.prefix}_subject_primary.csv', index=False)
    summary.to_csv(out / f'{args.prefix}_primary_summary.csv', index=False)

if __name__ == '__main__':
    main()
