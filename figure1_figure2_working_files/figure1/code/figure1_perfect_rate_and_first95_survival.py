"""
Figure 1 supplementary analyses (NHB style), same visual style as the original panels:

  1. Perfect-score rate (ever achieved score == 100 within a unified window):
     - Task 1 (panel e): 4D-E (P2) vs 4D-NE (P2-only), Fisher's exact test
     - Task 2 (panel f): Obs (FDS-Obs) vs NObs (FDS-NoObs), Fisher's exact test
     Bars show the rate with 95% Wilson score CIs; the tested pair gets a
     significance bracket with an exact P value.

  2. Time-to-first-trial with score >= 95 (within the same window):
     Kaplan-Meier curves with censoring (participants who never reached 95 are
     censored at their last available trial); between-group difference assessed
     with a two-sided log-rank test.

Window: Task 1 rounds <= 50, Task 2 trials <= 60 (identical to the main figure);
participants with fewer available trials contribute their last trial (censoring).

Outputs use dedicated names and never overwrite the original or previous figures.
"""

from pathlib import Path
import sys

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, chi2

from publication_plot_style import TASK1_COLORS, TASK1_LABELS, TASK2_COLORS, TASK2_LABELS
from figure1_integrated_four_panels_with_stats import (
    STYLE, lighten_color, style_panel_axis, format_pvalue, load_task1, load_task2, OUTPUT_DIR,
)

WINDOW = {'task1': 50, 'task2': 60}
THRESHOLD = 95.0
PERFECT = 100.0

PANEL_FIGSIZE = (4.6, 3.25)
COMBINED_FIGSIZE = (9.6, 3.45)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)


def perfect_rate_by_group(df, *, subject_col, group_col, trial_col, score_col, group, window):
    d = df.loc[(df[group_col] == group) & (df[trial_col] <= window)]
    best = d.groupby(subject_col)[score_col].max()
    n = int(best.shape[0])
    k = int((best >= PERFECT).sum())
    lo, hi = wilson_ci(k, n)
    return {
        'group': group, 'n_subjects': n, 'n_perfect': k,
        'rate': k / n if n else np.nan,
        'wilson_ci_lo': lo, 'wilson_ci_hi': hi,
    }


def fisher_test_rates(r_a, r_b):
    table = [
        [r_a['n_perfect'], r_a['n_subjects'] - r_a['n_perfect']],
        [r_b['n_perfect'], r_b['n_subjects'] - r_b['n_perfect']],
    ]
    odds_ratio, p = fisher_exact(table, alternative='two-sided')
    return float(odds_ratio), float(p)


def km_fit(df, *, subject_col, trial_col, score_col, window, threshold=THRESHOLD):
    """Per-subject event/censor times + Kaplan-Meier curve with Greenwood SE."""
    times, events = [], []
    for _, sdf in df.groupby(subject_col):
        sdf = sdf.loc[sdf[trial_col] <= window]
        if sdf.empty:
            continue
        reached = sdf.loc[sdf[score_col] >= threshold, trial_col]
        if not reached.empty:
            times.append(float(reached.min()))
            events.append(1)
        else:
            times.append(float(sdf[trial_col].max()))
            events.append(0)
    times = np.asarray(times)
    events = np.asarray(events)

    event_times = np.sort(np.unique(times[events == 1]))
    t_out, s_out, se_out = [0.0], [1.0], [0.0]
    S = 1.0
    greenwood = 0.0
    for t in event_times:
        n_risk = np.sum(times >= t)
        d = np.sum((times == t) & (events == 1))
        if n_risk <= 0:
            continue
        S *= (1 - d / n_risk)
        if n_risk - d > 0:
            greenwood += d / (n_risk * (n_risk - d))
        t_out.append(t)
        s_out.append(S)
        se_out.append(S * np.sqrt(greenwood))
    curve = pd.DataFrame({'time': t_out, 'survival': s_out, 'se': se_out})

    censor_points = pd.DataFrame({'time': times[events == 0]})

    def km_at(t_query):
        vals = curve.loc[curve['time'] <= t_query, 'survival']
        return float(vals.iloc[-1]) if not vals.empty else 1.0

    censor_points['survival'] = censor_points['time'].map(km_at)

    median = np.nan
    below = curve.loc[curve['survival'] <= 0.5, 'time']
    if not below.empty:
        median = float(below.iloc[0])
    pct_reaching = float(1 - curve['survival'].iloc[-1])

    return {
        'times': times, 'events': events, 'curve': curve,
        'censor_points': censor_points,
        'median': median, 'pct_reaching': pct_reaching,
    }


def logrank_test(fit_a, fit_b):
    """Two-sided log-rank test from per-subject times/events."""
    t_a, e_a = fit_a['times'], fit_a['events']
    t_b, e_b = fit_b['times'], fit_b['events']
    event_times = np.sort(np.unique(np.concatenate([t_a[e_a == 1], t_b[e_b == 1]])))
    o1 = e_a.sum()
    e1 = 0.0
    v1 = 0.0
    for t in event_times:
        n1 = np.sum(t_a >= t)
        n2 = np.sum(t_b >= t)
        n = n1 + n2
        d1 = np.sum((t_a == t) & (e_a == 1))
        d2 = np.sum((t_b == t) & (e_b == 1))
        d = d1 + d2
        if n <= 1:
            continue
        e1 += d * n1 / n
        v1 += n1 * n2 * d * (n - d) / (n * n * (n - 1))
    if v1 == 0:
        return np.nan, np.nan
    stat = (o1 - e1) ** 2 / v1
    p = float(chi2.sf(stat, 1))
    return float(stat), p


# ---------------------------------------------------------------------------
# Plotting (same style system as the original panels)
# ---------------------------------------------------------------------------

def plot_perfect_rate_panel(ax, rates, *, group_order, colors, labels, test_pair=None,
                            fisher_result=None, bar_width=0.52):
    slot_count = max(len(group_order), 3)
    start = STYLE['distribution_start'] + (slot_count - len(group_order)) * STYLE['group_step'] / 2
    centers = np.arange(len(group_order)) * STYLE['group_step'] + start

    for center, group in zip(centers, group_order):
        rate = rates[group]
        pct = 100 * rate['rate']
        lo = 100 * rate['wilson_ci_lo']
        hi = 100 * rate['wilson_ci_hi']
        ax.bar(
            center,
            pct,
            width=bar_width,
            color=lighten_color(colors[group], amount=0.18),
            edgecolor='black',
            linewidth=STYLE['box_line_width'],
            zorder=3,
        )
        ax.errorbar(
            center,
            pct,
            yerr=[[pct - lo], [hi - pct]],
            color='black',
            linewidth=STYLE['box_line_width'],
            capsize=4,
            capthick=STYLE['box_line_width'],
            zorder=4,
        )
        ax.text(
            center,
            pct + (hi - pct) + 3,
            f"{rate['n_perfect']}/{rate['n_subjects']}\n({pct:.0f}%)",
            ha='center',
            va='bottom',
            fontsize=STYLE['stats_fontsize'] - 2.5,
            zorder=5,
        )

    ax.set_xticks(centers)
    ax.set_xticklabels([labels[g] for g in group_order], fontsize=STYLE['tick_size'])
    style_panel_axis(
        ax,
        xlabel='',
        ylabel='Perfect-score rate (%)',
        y_ticks=[0, 25, 50, 75, 100],
        xlim=(1.0 - STYLE['distribution_left_margin'], STYLE['distribution_start'] + (slot_count - 1) * STYLE['group_step'] + 0.26),
        ylim=(0, 122),
    )
    ax.set_xticks(centers)
    ax.set_xticklabels([labels[g] for g in group_order], fontsize=STYLE['tick_size'])
    ax.spines['bottom'].set_bounds(centers[0], centers[-1])

    if test_pair is not None and fisher_result is not None:
        idx_a = group_order.index(test_pair[0])
        idx_b = group_order.index(test_pair[1])
        x_a, x_b = centers[idx_a], centers[idx_b]
        y_b, y_t = 109, 112
        ax.plot([x_a, x_a, x_b, x_b], [y_b - 2, y_b, y_b, y_b - 2],
                color='black', linewidth=1.3, clip_on=False, zorder=6)
        label = format_pvalue(fisher_result[1]) if fisher_result[1] < 0.05 else 'n.s.'
        ax.text((x_a + x_b) / 2, y_t, label, ha='center', va='bottom',
                fontsize=STYLE['stats_fontsize'], zorder=6)
        ax.text(
            0.03, 0.97,
            "Fisher's exact test (two-sided)",
            ha='left', va='top',
            fontsize=STYLE['stats_fontsize'] - 2.5,
            style='italic',
            transform=ax.transAxes,
        )


def plot_survival_panel(ax, fits, *, group_order, colors, labels, window, test_pair,
                        logrank_result, xlabel):
    for group in group_order:
        fit = fits[group]
        curve = fit['curve']
        color = colors[group]
        ax.step(
            curve['time'], 100 * curve['survival'],
            where='post',
            color=color,
            linewidth=STYLE['line_width'],
            label=labels[group],
            zorder=3,
        )
        ci_hi = np.clip(100 * (curve['survival'] + 1.96 * curve['se']), 0, 100)
        ci_lo = np.clip(100 * (curve['survival'] - 1.96 * curve['se']), 0, 100)
        ax.fill_between(
            curve['time'], ci_lo, ci_hi,
            step='post',
            color=color,
            alpha=STYLE['ribbon_alpha'],
            linewidth=0,
            zorder=2,
        )
        censors = fit['censor_points']
        if not censors.empty:
            ax.plot(
                censors['time'], 100 * censors['survival'],
                linestyle='none',
                marker='|',
                markersize=11,
                markeredgewidth=1.8,
                color=color,
                zorder=4,
            )

    style_panel_axis(
        ax,
        xlabel=xlabel,
        ylabel='Not reaching ' + r'$\geq$' + '95 (%)',
        x_ticks=np.arange(0, window + 1, 10),
        y_ticks=[0, 25, 50, 75, 100],
        xlim=(-2, window + 2),
        ylim=(0, 118),
    )

    stat, p = logrank_result
    lines = ['Log-rank test:', format_pvalue(p) if p < 0.05 else f"{format_pvalue(p)} (n.s.)", 'Median time to ' + r'$\geq$' + '95:']
    for group in group_order:
        fit = fits[group]
        if np.isnan(fit['median']):
            med_txt = 'not reached'
        else:
            med_txt = f"{fit['median']:.0f} ({100 * fit['pct_reaching']:.0f}% reached)"
        lines.append(f"{labels[group]}: {med_txt}")
    ax.text(
        0.03, 0.50, '\n'.join(lines),
        ha='left', va='top',
        fontsize=STYLE['stats_fontsize'] - 2.5,
        transform=ax.transAxes,
        zorder=6,
    )

    legend = ax.legend(
        frameon=False,
        fontsize=STYLE['legend_size'],
        loc='lower left',
        handlelength=2.0,
        borderaxespad=0.2,
    )
    for line in legend.get_lines():
        line.set_linewidth(STYLE['line_width'])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    task1_df, _, _, _, task1_order, task1_labels, task1_colors, _ = load_task1()
    task2_df, _, _, _, task2_order, task2_labels, task2_colors, _ = load_task2()

    # --- 1. perfect-score rate ------------------------------------------------
    t1_rates = {
        g: perfect_rate_by_group(
            task1_df, subject_col='subject', group_col='phase',
            trial_col='round', score_col='score', group=g, window=WINDOW['task1'],
        )
        for g in task1_order
    }
    t2_rates = {
        g: perfect_rate_by_group(
            task2_df, subject_col='subject_id', group_col='plot_group',
            trial_col='trial', score_col='score_noisy', group=g, window=WINDOW['task2'],
        )
        for g in task2_order
    }
    t1_fisher = fisher_test_rates(t1_rates['P2'], t1_rates['P2-only'])
    t2_fisher = fisher_test_rates(t2_rates['FDS-Obs'], t2_rates['FDS-NoObs'])

    rate_rows = []
    for panel, rates, fisher, pair in [
        ('e_task1', t1_rates, t1_fisher, ('P2', 'P2-only')),
        ('f_task2', t2_rates, t2_fisher, ('FDS-Obs', 'FDS-NoObs')),
    ]:
        for g, r in rates.items():
            rate_rows.append({'panel': panel, **r})
        rate_rows.append({
            'panel': panel, 'group': f"{pair[0]} vs {pair[1]}",
            'n_subjects': '', 'n_perfect': '',
            'rate': '', 'wilson_ci_lo': '', 'wilson_ci_hi': '',
            'fisher_odds_ratio': fisher[0], 'fisher_p': fisher[1],
        })
    rate_summary = pd.DataFrame(rate_rows)
    rate_summary.to_csv(OUTPUT_DIR / 'figure1_perfect_score_rate_tests.csv', index=False)

    print('Perfect-score rates (window: Task1 <=50 rounds, Task2 <=60 trials):')
    for g, r in t1_rates.items():
        print(f"  Task1 {g}: {r['n_perfect']}/{r['n_subjects']} = {100*r['rate']:.1f}%")
    print(f"  Fisher 4D-E vs 4D-NE: OR={t1_fisher[0]:.3f}, P={t1_fisher[1]:.4g}")
    for g, r in t2_rates.items():
        print(f"  Task2 {g}: {r['n_perfect']}/{r['n_subjects']} = {100*r['rate']:.1f}%")
    print(f"  Fisher Obs vs NObs: OR={t2_fisher[0]:.3f}, P={t2_fisher[1]:.4g}")

    # --- 2. time to first score >= 95 ----------------------------------------
    t1_fits = {
        g: km_fit(
            task1_df.loc[task1_df['phase'] == g],
            subject_col='subject', trial_col='round', score_col='score',
            window=WINDOW['task1'],
        )
        for g in ['P2', 'P2-only']
    }
    t2_fits = {
        g: km_fit(
            task2_df.loc[task2_df['plot_group'] == g],
            subject_col='subject_id', trial_col='trial', score_col='score_noisy',
            window=WINDOW['task2'],
        )
        for g in ['FDS-Obs', 'FDS-NoObs']
    }
    t1_logrank = logrank_test(t1_fits['P2'], t1_fits['P2-only'])
    t2_logrank = logrank_test(t2_fits['FDS-Obs'], t2_fits['FDS-NoObs'])

    km_rows = []
    surv_rows = []
    for panel, fits, groups, logrank, window, label_map in [
        ('e_task1', t1_fits, ['P2', 'P2-only'], t1_logrank, WINDOW['task1'], task1_labels),
        ('f_task2', t2_fits, ['FDS-Obs', 'FDS-NoObs'], t2_logrank, WINDOW['task2'], task2_labels),
    ]:
        for g in groups:
            fit = fits[g]
            curve = fit['curve'].copy()
            curve['group'] = label_map[g]
            km_rows.append(curve.assign(panel=panel))
            surv_rows.append({
                'panel': panel,
                'group': label_map[g],
                'n': len(fit['times']),
                'n_events': int(fit['events'].sum()),
                'pct_reaching_95': fit['pct_reaching'],
                'median_time_to_95': fit['median'],
            })
        surv_rows.append({
            'panel': panel, 'group': ' vs '.join(label_map[g] for g in groups),
            'n': '', 'n_events': '', 'pct_reaching_95': '',
            'median_time_to_95': '',
            'logrank_chi2': logrank[0], 'logrank_p': logrank[1],
        })
    pd.concat(km_rows).to_csv(OUTPUT_DIR / 'figure1_first95_km_curves.csv', index=False)
    pd.DataFrame(surv_rows).to_csv(OUTPUT_DIR / 'figure1_first95_logrank_tests.csv', index=False)

    print('\nTime to first score >=95 (KM / log-rank):')
    for g, fit in t1_fits.items():
        med = f"{fit['median']:.0f}" if not np.isnan(fit['median']) else 'not reached'
        print(f"  Task1 {g}: {int(fit['events'].sum())}/{len(fit['times'])} reached, median={med}")
    print(f"  Log-rank 4D-E vs 4D-NE: chi2={t1_logrank[0]:.3f}, P={t1_logrank[1]:.4g}")
    for g, fit in t2_fits.items():
        med = f"{fit['median']:.0f}" if not np.isnan(fit['median']) else 'not reached'
        print(f"  Task2 {g}: {int(fit['events'].sum())}/{len(fit['times'])} reached, median={med}")
    print(f"  Log-rank Obs vs NObs: chi2={t2_logrank[0]:.3f}, P={t2_logrank[1]:.4g}")

    # --- figures ----------------------------------------------------------------
    # Figure 1: perfect-score rate, 1x2
    fig, axes = plt.subplots(1, 2, figsize=COMBINED_FIGSIZE)
    fig.subplots_adjust(wspace=0.55)
    plot_perfect_rate_panel(
        axes[0], t1_rates, group_order=task1_order, colors=task1_colors, labels=task1_labels,
        test_pair=('P2', 'P2-only'), fisher_result=t1_fisher,
    )
    plot_perfect_rate_panel(
        axes[1], t2_rates, group_order=task2_order, colors=task2_colors, labels=task2_labels,
        test_pair=('FDS-Obs', 'FDS-NoObs'), fisher_result=t2_fisher,
    )
    out = OUTPUT_DIR / 'figure1_perfect_score_rate_panels_with_stats.png'
    fig.savefig(out, dpi=STYLE['dpi'], bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved: {out}')

    # Figure 2: KM survival, 1x2
    fig, axes = plt.subplots(1, 2, figsize=COMBINED_FIGSIZE)
    fig.subplots_adjust(wspace=0.55)
    plot_survival_panel(
        axes[0], t1_fits, group_order=['P2', 'P2-only'], colors=task1_colors, labels=task1_labels,
        window=WINDOW['task1'], test_pair=('P2', 'P2-only'), logrank_result=t1_logrank,
        xlabel='Round',
    )
    plot_survival_panel(
        axes[1], t2_fits, group_order=['FDS-Obs', 'FDS-NoObs'], colors=task2_colors, labels=task2_labels,
        window=WINDOW['task2'], test_pair=('FDS-Obs', 'FDS-NoObs'), logrank_result=t2_logrank,
        xlabel='Trial',
    )
    out = OUTPUT_DIR / 'figure1_first95_km_survival_panels_with_stats.png'
    fig.savefig(out, dpi=STYLE['dpi'], bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')

    # standalone panels
    standalone = [
        ('figure1_panel_e_task1_perfect_score_rate_with_stats.png',
         lambda ax: plot_perfect_rate_panel(
             ax, t1_rates, group_order=task1_order, colors=task1_colors, labels=task1_labels,
             test_pair=('P2', 'P2-only'), fisher_result=t1_fisher)),
        ('figure1_panel_f_task2_perfect_score_rate_with_stats.png',
         lambda ax: plot_perfect_rate_panel(
             ax, t2_rates, group_order=task2_order, colors=task2_colors, labels=task2_labels,
             test_pair=('FDS-Obs', 'FDS-NoObs'), fisher_result=t2_fisher)),
        ('figure1_panel_e_task1_first95_km_with_stats.png',
         lambda ax: plot_survival_panel(
             ax, t1_fits, group_order=['P2', 'P2-only'], colors=task1_colors, labels=task1_labels,
             window=WINDOW['task1'], test_pair=('P2', 'P2-only'), logrank_result=t1_logrank,
             xlabel='Round')),
        ('figure1_panel_f_task2_first95_km_with_stats.png',
         lambda ax: plot_survival_panel(
             ax, t2_fits, group_order=['FDS-Obs', 'FDS-NoObs'], colors=task2_colors, labels=task2_labels,
             window=WINDOW['task2'], test_pair=('FDS-Obs', 'FDS-NoObs'), logrank_result=t2_logrank,
             xlabel='Trial')),
    ]
    for filename, draw in standalone:
        fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
        draw(ax)
        out = OUTPUT_DIR / filename
        fig.savefig(out, dpi=STYLE['dpi'], bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out}')


if __name__ == '__main__':
    main()
