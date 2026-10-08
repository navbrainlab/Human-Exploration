"""
Optimized standalone panels e-j for the new Figure 1 layout:

  e  Task 1 learning curve        (with cluster-permutation stats)
  f  Task 1 perfect-score rate    (with Fisher exact stats, narrow bars)
  g  Task 1 personal-best round   (with MWU stats, tighter group gaps)
  h  Task 2 learning curve        (NO stats annotation)
  i  Task 2 perfect-score rate    (NO stats annotation, narrow bars)
  j  Task 2 personal-best round   (with MWU stats, tighter group gaps)

Optimizations vs previous versions:
  - e/f/h/i use a squarer figure size (3.9 x 3.5 instead of 4.6 x 3.25)
  - f/i bars are narrower (0.34 instead of 0.52)
  - g/j groups are closer together (group_step 0.55 instead of 0.82)

Overwrites figure1_panel_e_task1_learning_curve_with_stats.png and writes the
five other panels under their new panel-letter names.
"""

from pathlib import Path
import sys

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

from publication_plot_style import TASK1_COLORS, TASK1_LABELS, TASK2_COLORS, TASK2_LABELS
from figure1_integrated_four_panels_with_stats import (
    STYLE, lighten_color, style_panel_axis, format_pvalue, load_task1, load_task2,
    learning_curve_cluster_test, OUTPUT_DIR, plot_learning_panel,
)
from figure1_perfect_rate_and_first95_survival import (
    perfect_rate_by_group, fisher_test_rates, plot_perfect_rate_panel,
)
from figure1_sensitivity_tests import personal_best_round

SQUARE_FIGSIZE = (3.9, 3.5)       # e, f, h, i: squarer than the old 4.6 x 3.25
DIST_FIGSIZE = (4.6, 3.25)        # g, j
BAR_WIDTH = 0.34                  # f, i narrow bars
GROUP_STEP_TIGHT = 0.55           # g, j tighter group spacing


def plot_personal_best_panel_v2(ax, df, *, subject_col, group_col, trial_col, score_col,
                                group_order, colors, labels, window, mwu_result,
                                test_pair, ylabel='Round of personal best',
                                ylim=None, y_ticks=None, group_alignment='center',
                                group_step=GROUP_STEP_TIGHT):
    vals = {
        g: personal_best_round(df.loc[df[group_col] == g], subject_col=subject_col,
                               trial_col=trial_col, score_col=score_col, window=window)
        for g in group_order
    }
    data = [vals[g] for g in group_order]
    slot_count = max(len(group_order), 3)
    slot_span = (slot_count - 1) * group_step
    if group_alignment == 'left':
        start = STYLE['distribution_start']
    else:
        start = STYLE['distribution_start'] + (slot_count - len(group_order)) * group_step / 2
    centers = np.arange(len(group_order)) * group_step + start
    scatter_positions = centers + STYLE['scatter_offset']

    box = ax.boxplot(
        data, positions=centers, widths=STYLE['box_width'], patch_artist=True,
        showfliers=False,
        medianprops=dict(color='black', linewidth=STYLE['median_width']),
        whiskerprops=dict(color='black', linewidth=STYLE['box_line_width']),
        capprops=dict(color='black', linewidth=STYLE['box_line_width']),
        boxprops=dict(edgecolor='black', linewidth=STYLE['box_line_width']),
    )
    for patch, group in zip(box['boxes'], group_order):
        patch.set_facecolor(lighten_color(colors[group], amount=0.18))
        patch.set_alpha(0.90)

    rng = np.random.default_rng(2026)
    for pos, values, group in zip(scatter_positions, data, group_order):
        if len(values) == 0:
            continue
        jitter = rng.uniform(-STYLE['point_jitter'], STYLE['point_jitter'], size=len(values))
        ax.scatter(
            pos + jitter, values,
            s=STYLE['point_size'], color=colors[group], alpha=STYLE['point_alpha'],
            edgecolor='white', linewidth=0.45, zorder=4,
        )

    ax.set_xticks(centers)
    ax.set_xticklabels([labels[g] for g in group_order], fontsize=STYLE['tick_size'])
    style_panel_axis(
        ax, xlabel='', ylabel=ylabel,
        y_ticks=y_ticks,
        xlim=(1.0 - STYLE['distribution_left_margin'], STYLE['distribution_start'] + slot_span + 0.26),
        ylim=ylim,
    )
    ax.set_xticks(centers)
    ax.set_xticklabels([labels[g] for g in group_order], fontsize=STYLE['tick_size'])
    ax.spines['bottom'].set_bounds(centers[0], centers[-1])

    idx_a = group_order.index(test_pair[0])
    idx_b = group_order.index(test_pair[1])
    x_a, x_b = centers[idx_a], centers[idx_b]
    y_b = ylim[1] - 0.09 * (ylim[1] - ylim[0])
    y_t = ylim[1] - 0.035 * (ylim[1] - ylim[0])
    ax.plot([x_a, x_a, x_b, x_b],
            [y_b - 0.02 * (ylim[1] - ylim[0]), y_b, y_b, y_b - 0.02 * (ylim[1] - ylim[0])],
            color='black', linewidth=1.3, clip_on=False, zorder=6)
    label = format_pvalue(mwu_result['p']) if mwu_result['p'] < 0.05 else 'n.s.'
    ax.text((x_a + x_b) / 2, y_t, label, ha='center', va='bottom',
            fontsize=STYLE['stats_fontsize'], zorder=6)
    ax.text(
        0.03, 0.97, 'Mann-Whitney U (two-sided)',
        ha='left', va='top', fontsize=STYLE['stats_fontsize'] - 2.5,
        style='italic', transform=ax.transAxes,
    )


def mwu(vals_a, vals_b):
    res = mannwhitneyu(vals_a, vals_b, alternative='two-sided')
    return {'p': float(res.pvalue)}


def save(fig, name):
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=STYLE['dpi'], bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {path}')


def main():
    task1_df, task1_extended, task1_curves, _, task1_order, task1_labels, task1_colors, _ = load_task1()
    task2_df, task2_extended, task2_curves, _, task2_order, task2_labels, task2_colors, _ = load_task2()

    # ---- statistics ---------------------------------------------------------
    print('Running Task 1 cluster test (10000 permutations)...')
    _, t1_clusters = learning_curve_cluster_test(
        task1_extended.loc[task1_extended['phase'].isin(['P2', 'P2-only'])],
        subject_col='subject', group_col='phase', trial_col='round', score_col='score',
        group_a='P2', group_b='P2-only', max_trial=50, n_permutations=10000,
    )
    t1_rates = {g: perfect_rate_by_group(task1_df, subject_col='subject', group_col='phase',
                                         trial_col='round', score_col='score', group=g, window=50)
                for g in task1_order}
    t2_rates = {g: perfect_rate_by_group(task2_df, subject_col='subject_id', group_col='plot_group',
                                         trial_col='trial', score_col='score_noisy', group=g, window=60)
                for g in task2_order}
    t1_fisher = fisher_test_rates(t1_rates['P2'], t1_rates['P2-only'])
    t1_pb_mwu = mwu(
        personal_best_round(task1_df.loc[task1_df['phase'] == 'P2'], subject_col='subject',
                            trial_col='round', score_col='score', window=50),
        personal_best_round(task1_df.loc[task1_df['phase'] == 'P2-only'], subject_col='subject',
                            trial_col='round', score_col='score', window=50),
    )
    t2_pb_mwu = mwu(
        personal_best_round(task2_df.loc[task2_df['plot_group'] == 'FDS-Obs'], subject_col='subject_id',
                            trial_col='trial', score_col='score_noisy', window=60),
        personal_best_round(task2_df.loc[task2_df['plot_group'] == 'FDS-NoObs'], subject_col='subject_id',
                            trial_col='trial', score_col='score_noisy', window=60),
    )

    # ---- panel e: Task 1 learning curve (stats) -----------------------------
    fig, ax = plt.subplots(figsize=SQUARE_FIGSIZE)
    plot_learning_panel(
        ax, task1_curves, group_order=task1_order, colors=task1_colors, labels=task1_labels,
        ylim=STYLE['task1_learning_ylim'], y_ticks=STYLE['task1_learning_y_ticks'],
        show_legend=True, significant_clusters=t1_clusters, max_trial=50,
    )
    save(fig, 'figure1_panel_e_task1_learning_curve_with_stats.png')

    # ---- panel f: Task 1 perfect-score rate (stats, narrow bars) ------------
    fig, ax = plt.subplots(figsize=SQUARE_FIGSIZE)
    plot_perfect_rate_panel(
        ax, t1_rates, group_order=task1_order, colors=task1_colors, labels=task1_labels,
        test_pair=('P2', 'P2-only'), fisher_result=t1_fisher, bar_width=BAR_WIDTH,
    )
    save(fig, 'figure1_panel_f_task1_perfect_score_rate_with_stats.png')

    # ---- panel g: Task 1 personal-best round (stats, tight gaps) ------------
    fig, ax = plt.subplots(figsize=DIST_FIGSIZE)
    plot_personal_best_panel_v2(
        ax, task1_df, subject_col='subject', group_col='phase', trial_col='round', score_col='score',
        group_order=task1_order, colors=task1_colors, labels=task1_labels, window=50,
        mwu_result=t1_pb_mwu, test_pair=('P2', 'P2-only'),
        ylim=(0, 58), y_ticks=[10, 20, 30, 40, 50],
    )
    save(fig, 'figure1_panel_g_task1_personal_best_round_with_stats.png')

    # ---- panel h: Task 2 learning curve (NO stats) --------------------------
    fig, ax = plt.subplots(figsize=SQUARE_FIGSIZE)
    plot_learning_panel(
        ax, task2_curves, group_order=task2_order, colors=task2_colors, labels=task2_labels,
        ylim=STYLE['task2_learning_ylim'], y_ticks=STYLE['task2_learning_y_ticks'],
        show_legend=True, significant_clusters=None, max_trial=60,
    )
    save(fig, 'figure1_panel_h_task2_learning_curve.png')

    # ---- panel i: Task 2 perfect-score rate (NO stats, narrow bars) ---------
    fig, ax = plt.subplots(figsize=SQUARE_FIGSIZE)
    plot_perfect_rate_panel(
        ax, t2_rates, group_order=task2_order, colors=task2_colors, labels=task2_labels,
        test_pair=None, fisher_result=None, bar_width=BAR_WIDTH,
    )
    save(fig, 'figure1_panel_i_task2_perfect_score_rate.png')

    # ---- panel j: Task 2 personal-best round (stats, tight gaps) ------------
    fig, ax = plt.subplots(figsize=DIST_FIGSIZE)
    plot_personal_best_panel_v2(
        ax, task2_df, subject_col='subject_id', group_col='plot_group', trial_col='trial',
        score_col='score_noisy',
        group_order=task2_order, colors=task2_colors, labels=task2_labels, window=60,
        mwu_result=t2_pb_mwu, test_pair=('FDS-Obs', 'FDS-NoObs'),
        ylim=(0, 70), y_ticks=[15, 30, 45, 60], group_alignment='left',
    )
    save(fig, 'figure1_panel_j_task2_personal_best_round_with_stats.png')


if __name__ == '__main__':
    main()
