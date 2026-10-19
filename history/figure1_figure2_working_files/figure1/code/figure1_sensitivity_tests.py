"""
Sensitivity / exploratory analyses for Figure 1 group comparisons:

  1. Perfect-score rate with chi-square tests (with and without Yates
     continuity correction), compared against the primary Fisher exact test.
  2. Time to first score >= 90 (instead of >= 95): KM + log-rank, same window
     and censoring rules as the >=95 analysis.
  3. Round of first attainment of the personal best score (within the window):
     lower = reached one's own best performance earlier; MWU between groups.

Window: Task 1 rounds <= 50, Task 2 trials <= 60 (identical to the main figure).

Outputs a single CSV with all results; nothing is overwritten.
"""

from pathlib import Path
import sys

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu

from figure1_integrated_four_panels_with_stats import load_task1, load_task2, OUTPUT_DIR
from figure1_perfect_rate_and_first95_survival import (
    WINDOW, perfect_rate_by_group, km_fit, logrank_test,
)

THRESHOLD_ALT = 90.0
THRESHOLD_85 = 85.0


def chi_square_rate(r_a, r_b, yates=True):
    table = [
        [r_a['n_perfect'], r_a['n_subjects'] - r_a['n_perfect']],
        [r_b['n_perfect'], r_b['n_subjects'] - r_b['n_perfect']],
    ]
    chi2, p, dof, expected = chi2_contingency(table, correction=yates)
    return float(chi2), float(p)


def max_rounds_per_subject(df, *, subject_col, trial_col):
    """Maximum trial number each subject completed (raw data, no window truncation)."""
    return df.groupby(subject_col)[trial_col].max().to_numpy(dtype=float)


def personal_best_round(df, *, subject_col, trial_col, score_col, window):
    """First trial (within window) at which the subject achieved their window max."""
    d = df.loc[df[trial_col] <= window]
    rounds = []
    for _, sdf in d.groupby(subject_col):
        sdf = sdf.sort_values(trial_col)
        best = sdf[score_col].max()
        first = sdf.loc[sdf[score_col] >= best, trial_col].min()
        rounds.append(float(first))
    return np.asarray(rounds)


def summarize(vals):
    return {
        'n': len(vals),
        'median': float(np.median(vals)),
        'q1': float(np.percentile(vals, 25)),
        'q3': float(np.percentile(vals, 75)),
        'mean': float(np.mean(vals)),
        'std': float(np.std(vals, ddof=1)),
    }


def main():
    task1_df, _, _, _, _, t1_labels, _, _ = load_task1()
    task2_df, _, _, _, _, t2_labels, _, _ = load_task2()

    rows = []

    # --- 1. chi-square on perfect-score rate --------------------------------
    t1_rates = {g: perfect_rate_by_group(task1_df, subject_col='subject', group_col='phase',
                                         trial_col='round', score_col='score', group=g,
                                         window=WINDOW['task1']) for g in ['P2', 'P2-only']}
    t2_rates = {g: perfect_rate_by_group(task2_df, subject_col='subject_id', group_col='plot_group',
                                         trial_col='trial', score_col='score_noisy', group=g,
                                         window=WINDOW['task2']) for g in ['FDS-Obs', 'FDS-NoObs']}
    for panel, rates, pair in [('e_task1', t1_rates, ('P2', 'P2-only')),
                               ('f_task2', t2_rates, ('FDS-Obs', 'FDS-NoObs'))]:
        for yates in (True, False):
            chi2, p = chi_square_rate(rates[pair[0]], rates[pair[1]], yates=yates)
            rows.append({
                'analysis': f'perfect_rate_chi2_yates={yates}', 'panel': panel,
                'comparison': f'{pair[0]} vs {pair[1]}',
                'group_a_rate': rates[pair[0]]['rate'], 'group_b_rate': rates[pair[1]]['rate'],
                'statistic': chi2, 'p': p,
            })

    # --- 2. time to first score >= 90 ---------------------------------------
    for panel, df, subject_col, group_col, trial_col, score_col, groups, window in [
        ('e_task1', task1_df, 'subject', 'phase', 'round', 'score', ['P2', 'P2-only'], WINDOW['task1']),
        ('f_task2', task2_df, 'subject_id', 'plot_group', 'trial', 'score_noisy',
         ['FDS-Obs', 'FDS-NoObs'], WINDOW['task2']),
    ]:
        fits = {g: km_fit(df.loc[df[group_col] == g], subject_col=subject_col, trial_col=trial_col,
                          score_col=score_col, window=window, threshold=THRESHOLD_ALT)
                for g in groups}
        stat, p = logrank_test(fits[groups[0]], fits[groups[1]])
        for g in groups:
            fit = fits[g]
            rows.append({
                'analysis': 'time_to_first90_km', 'panel': panel, 'comparison': g,
                'group_a_rate': fit['pct_reaching'],  # proportion reaching >=90
                'group_b_rate': fit['median'] if not np.isnan(fit['median']) else 'not reached',
                'statistic': '', 'p': '',
            })
        rows.append({
            'analysis': 'time_to_first90_logrank', 'panel': panel,
            'comparison': f'{groups[0]} vs {groups[1]}',
            'group_a_rate': '', 'group_b_rate': '',
            'statistic': stat, 'p': p,
        })

    # --- 2b. time to first score >= 85 --------------------------------------
    for panel, df, subject_col, group_col, trial_col, score_col, groups, window in [
        ('e_task1', task1_df, 'subject', 'phase', 'round', 'score', ['P2', 'P2-only'], WINDOW['task1']),
        ('f_task2', task2_df, 'subject_id', 'plot_group', 'trial', 'score_noisy',
         ['FDS-Obs', 'FDS-NoObs'], WINDOW['task2']),
    ]:
        fits = {g: km_fit(df.loc[df[group_col] == g], subject_col=subject_col, trial_col=trial_col,
                          score_col=score_col, window=window, threshold=THRESHOLD_85)
                for g in groups}
        stat, p = logrank_test(fits[groups[0]], fits[groups[1]])
        for g in groups:
            fit = fits[g]
            rows.append({
                'analysis': 'time_to_first85_km', 'panel': panel, 'comparison': g,
                'group_a_rate': fit['pct_reaching'],
                'group_b_rate': fit['median'] if not np.isnan(fit['median']) else 'not reached',
                'statistic': '', 'p': '',
            })
        rows.append({
            'analysis': 'time_to_first85_logrank', 'panel': panel,
            'comparison': f'{groups[0]} vs {groups[1]}',
            'group_a_rate': '', 'group_b_rate': '',
            'statistic': stat, 'p': p,
        })

    # --- 3. round of first personal-best attainment --------------------------
    for panel, df, subject_col, group_col, trial_col, score_col, groups, window in [
        ('e_task1', task1_df, 'subject', 'phase', 'round', 'score', ['P2', 'P2-only'], WINDOW['task1']),
        ('f_task2', task2_df, 'subject_id', 'plot_group', 'trial', 'score_noisy',
         ['FDS-Obs', 'FDS-NoObs'], WINDOW['task2']),
    ]:
        vals = {g: personal_best_round(df.loc[df[group_col] == g], subject_col=subject_col,
                                       trial_col=trial_col, score_col=score_col, window=window)
                for g in groups}
        res = mannwhitneyu(vals[groups[0]], vals[groups[1]], alternative='two-sided')
        r_rb = 1 - 2 * res.statistic / (len(vals[groups[0]]) * len(vals[groups[1]]))
        for g in groups:
            s = summarize(vals[g])
            rows.append({
                'analysis': 'personal_best_round', 'panel': panel, 'comparison': g,
                'group_a_rate': s['median'], 'group_b_rate': f"IQR {s['q1']:.0f}-{s['q3']:.0f}, mean {s['mean']:.1f}±{s['std']:.1f}",
                'statistic': '', 'p': '',
            })
        rows.append({
            'analysis': 'personal_best_round_mwu', 'panel': panel,
            'comparison': f'{groups[0]} vs {groups[1]}',
            'group_a_rate': f"rank-biserial r = {r_rb:.3f}", 'group_b_rate': '',
            'statistic': float(res.statistic), 'p': float(res.pvalue),
        })

    # --- 3b. maximum task rounds per subject --------------------------------
    for panel, df, subject_col, group_col, trial_col, groups in [
        ('e_task1', task1_df, 'subject', 'phase', 'round', ['P2', 'P2-only']),
        ('f_task2', task2_df, 'subject_id', 'plot_group', 'trial', ['FDS-Obs', 'FDS-NoObs']),
    ]:
        vals = {g: max_rounds_per_subject(df.loc[df[group_col] == g], subject_col=subject_col,
                                          trial_col=trial_col)
                for g in groups}
        res = mannwhitneyu(vals[groups[0]], vals[groups[1]], alternative='two-sided')
        r_rb = 1 - 2 * res.statistic / (len(vals[groups[0]]) * len(vals[groups[1]]))
        for g in groups:
            s = summarize(vals[g])
            rows.append({
                'analysis': 'max_task_rounds', 'panel': panel, 'comparison': g,
                'group_a_rate': s['median'], 'group_b_rate': f"IQR {s['q1']:.0f}-{s['q3']:.0f}, mean {s['mean']:.1f}±{s['std']:.1f}",
                'statistic': '', 'p': '',
            })
        rows.append({
            'analysis': 'max_task_rounds_mwu', 'panel': panel,
            'comparison': f'{groups[0]} vs {groups[1]}',
            'group_a_rate': f"rank-biserial r = {r_rb:.3f}", 'group_b_rate': '',
            'statistic': float(res.statistic), 'p': float(res.pvalue),
        })

    out = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / 'figure1_sensitivity_analyses.csv'
    out.to_csv(out_path, index=False)

    pd.set_option('display.width', 200)
    pd.set_option('display.max_colwidth', 60)
    print(out.to_string(index=False))
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
