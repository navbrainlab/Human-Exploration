"""
Assemble the full Figure 1 (panels a-j) with statistics.

Layout:
  Row 1: panel a (Task 1 design)      | panel b (Task 2 design)
  Row 2: panel c (Task 1 grouping)    | panel d (Task 2 grouping)
  Row 3: panel e (Task 1 learning)    | f (perfect-score rate) | g (personal-best round)
  Row 4: panel h (Task 2 learning)    | i (perfect-score rate) | j (personal-best round)

Panels a-d are raster renders of the design PPTX files (cropped). Panel a's
source file is not present in this working folder; place it at
figure1/output/_pptx_render/task1_design.png to include it (otherwise a clearly
marked placeholder box is drawn).

Statistic panels reuse the exact style system of the original figure
(same colors, fonts, axes) and show:
  learning curves: cluster-based permutation test annotation
  perfect-score rate: Fisher's exact test bracket (95% Wilson CIs)
  personal-best round: Mann-Whitney U bracket

Outputs: figure1/output/figure1_full_panels_a_j_with_stats.png / .pdf (300 dpi)
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
from PIL import Image
from scipy.stats import mannwhitneyu

from publication_plot_style import TASK1_COLORS, TASK1_LABELS, TASK2_COLORS, TASK2_LABELS
from figure1_integrated_four_panels_with_stats import (
    STYLE, lighten_color, style_panel_axis, format_pvalue, add_panel_label,
    load_task1, load_task2, compute_learning_curves,
    learning_curve_cluster_test, best_score_mann_whitney, extend_after_first_perfect,
    OUTPUT_DIR,
)
from figure1_perfect_rate_and_first95_survival import (
    WINDOW, perfect_rate_by_group, fisher_test_rates, plot_perfect_rate_panel,
)
from figure1_sensitivity_tests import personal_best_round

RENDER_DIR = OUTPUT_DIR / '_pptx_render'
TASK1_DESIGN_IMG = RENDER_DIR / 'task1_design.png'   # panel a asset (may not exist)
GROUPING_RENDER = RENDER_DIR / 'grouping_design_editable.pptx.png'
TASK2_RENDER = RENDER_DIR / 'task2_design_editable.pptx.png'

FIGSIZE = (16.0, 13.2)
PANEL_LABELS_ROW3 = ['e', 'f', 'g']
PANEL_LABELS_ROW4 = ['h', 'i', 'j']


# ---------------------------------------------------------------------------
# image helpers
# ---------------------------------------------------------------------------

def autotrim(img, thresh=245, pad=12):
    gray = np.asarray(img.convert('L'))
    mask = gray < thresh
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    x0 = max(int(xs.min()) - pad, 0)
    y0 = max(int(ys.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad, img.width)
    y1 = min(int(ys.max()) + pad, img.height)
    return img.crop((x0, y0, x1, y1))


def crop_region(img, box):
    return autotrim(img.crop(box))


# ---------------------------------------------------------------------------
# personal-best round raincloud panel (same style as best-score distribution)
# ---------------------------------------------------------------------------

def plot_personal_best_panel(ax, df, *, subject_col, group_col, trial_col, score_col,
                             group_order, colors, labels, window, mwu_result,
                             test_pair, ylabel='Round of personal best',
                             ylim=None, y_ticks=None, group_alignment='center'):
    vals = {
        g: personal_best_round(df.loc[df[group_col] == g], subject_col=subject_col,
                               trial_col=trial_col, score_col=score_col, window=window)
        for g in group_order
    }
    data = [vals[g] for g in group_order]
    slot_count = max(len(group_order), 3)
    slot_span = (slot_count - 1) * STYLE['group_step']
    if group_alignment == 'left':
        start = STYLE['distribution_start']
    else:
        start = STYLE['distribution_start'] + (slot_count - len(group_order)) * STYLE['group_step'] / 2
    centers = np.arange(len(group_order)) * STYLE['group_step'] + start
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

    # MWU bracket
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
        0.02, 0.03, 'Mann-Whitney U (two-sided)',
        ha='left', va='bottom', fontsize=STYLE['stats_fontsize'] - 2.5,
        style='italic', transform=ax.transAxes,
    )


def mwu(vals_a, vals_b, name_a, name_b):
    res = mannwhitneyu(vals_a, vals_b, alternative='two-sided')
    return {
        'group_a': name_a, 'group_b': name_b,
        'n_a': len(vals_a), 'n_b': len(vals_b),
        'U': float(res.statistic), 'p': float(res.pvalue),
        'rank_biserial_r': float(1 - 2 * res.statistic / (len(vals_a) * len(vals_b))),
    }


# ---------------------------------------------------------------------------
# main assembly
# ---------------------------------------------------------------------------

def main():
    # ---------------- data & statistics ----------------
    task1_df, task1_extended, task1_curves, task1_score_df, task1_order, task1_labels, task1_colors, task1_max_trial = load_task1()
    task2_df, task2_extended, task2_curves, task2_score_df, task2_order, task2_labels, task2_colors, task2_max_trial = load_task2()

    n_perm = 10000
    print(f'Running Task 1 cluster test ({n_perm} permutations)...')
    _, t1_clusters = learning_curve_cluster_test(
        task1_extended.loc[task1_extended['phase'].isin(['P2', 'P2-only'])],
        subject_col='subject', group_col='phase', trial_col='round', score_col='score',
        group_a='P2', group_b='P2-only', max_trial=50, n_permutations=n_perm,
    )
    print(f"Task 1 significant clusters: {len(t1_clusters.loc[t1_clusters['cluster_p'] < 0.05])}")
    print('Running Task 2 cluster test...')
    _, t2_clusters = learning_curve_cluster_test(
        task2_extended,
        subject_col='subject_id', group_col='plot_group', trial_col='trial', score_col='score_noisy',
        group_a='FDS-Obs', group_b='FDS-NoObs', max_trial=60, n_permutations=n_perm,
    )
    print(f"Task 2 significant clusters: {len(t2_clusters.loc[t2_clusters['cluster_p'] < 0.05])}")

    t1_rates = {g: perfect_rate_by_group(task1_df, subject_col='subject', group_col='phase',
                                         trial_col='round', score_col='score', group=g, window=50)
                for g in task1_order}
    t2_rates = {g: perfect_rate_by_group(task2_df, subject_col='subject_id', group_col='plot_group',
                                         trial_col='trial', score_col='score_noisy', group=g, window=60)
                for g in task2_order}
    t1_fisher = fisher_test_rates(t1_rates['P2'], t1_rates['P2-only'])
    t2_fisher = fisher_test_rates(t2_rates['FDS-Obs'], t2_rates['FDS-NoObs'])

    t1_pb = {g: personal_best_round(task1_df.loc[task1_df['phase'] == g], subject_col='subject',
                                    trial_col='round', score_col='score', window=50)
             for g in ['P2', 'P2-only']}
    t2_pb = {g: personal_best_round(task2_df.loc[task2_df['plot_group'] == g], subject_col='subject_id',
                                    trial_col='trial', score_col='score_noisy', window=60)
             for g in ['FDS-Obs', 'FDS-NoObs']}
    t1_pb_mwu = mwu(t1_pb['P2'], t1_pb['P2-only'], 'P2', 'P2-only')
    t2_pb_mwu = mwu(t2_pb['FDS-Obs'], t2_pb['FDS-NoObs'], 'FDS-Obs', 'FDS-NoObs')
    print(f"Personal-best MWU: Task1 P={t1_pb_mwu['p']:.4g}, Task2 P={t2_pb_mwu['p']:.4g}")

    # ---------------- design crops ----------------
    grouping_img = Image.open(GROUPING_RENDER)
    task2_img = Image.open(TASK2_RENDER)
    # rough regions in the rendered thumbnails, then auto-trim
    crop_c = crop_region(grouping_img, (150, 870, 1075, 1245))   # Task 1 grouping (excl. broken image placeholder)
    crop_d = crop_region(grouping_img, (150, 250, 1100, 560))     # Task 2 grouping (2-group version)
    crop_b = autotrim(task2_img)                                   # Task 2 design
    if TASK1_DESIGN_IMG.exists():
        crop_a = autotrim(Image.open(TASK1_DESIGN_IMG))
        have_a = True
    else:
        have_a = False
        print(f'WARNING: panel a asset not found at {TASK1_DESIGN_IMG}; drawing placeholder.')

    # ---------------- figure ----------------
    fig = plt.figure(figsize=FIGSIZE)
    fig.patch.set_facecolor('white')

    stat_h = 0.17
    row1_y, row1_h = 0.735, 0.215
    row2_y, row2_h = 0.560, 0.155
    row3_y = 0.335
    row4_y = 0.065
    col_w = 0.265
    gap = 0.085
    x0 = 0.065
    col_x = [x0, x0 + col_w + gap, x0 + 2 * (col_w + gap)]
    half_x = [0.06, 0.54]
    half_w = 0.42

    # headers
    fig.text(half_x[0] + half_w / 2, 0.975, 'Task 1', ha='center', va='top',
             fontsize=20, fontweight='bold')
    fig.text(half_x[1] + half_w / 2, 0.975, 'Task 2', ha='center', va='top',
             fontsize=20, fontweight='bold')

    # row 1: a | b
    ax_a = fig.add_axes([half_x[0], row1_y, half_w, row1_h])
    if have_a:
        ax_a.imshow(np.asarray(crop_a))
    else:
        ax_a.set_facecolor('#f2f2f2')
        ax_a.text(0.5, 0.5, 'panel a\n(Task 1 design)\nasset pending', ha='center', va='center',
                  fontsize=14, color='#888888')
    ax_a.axis('off')
    add_panel_label(ax_a, 'a')

    ax_b = fig.add_axes([half_x[1], row1_y, half_w, row1_h])
    ax_b.imshow(np.asarray(crop_b))
    ax_b.axis('off')
    add_panel_label(ax_b, 'b')

    # row 2: c | d
    ax_c = fig.add_axes([half_x[0], row2_y, half_w, row2_h])
    ax_c.imshow(np.asarray(crop_c))
    ax_c.axis('off')
    add_panel_label(ax_c, 'c')

    ax_d = fig.add_axes([half_x[1], row2_y, half_w, row2_h])
    ax_d.imshow(np.asarray(crop_d))
    ax_d.axis('off')
    add_panel_label(ax_d, 'd')

    # row 3: e f g (Task 1 stats)
    ax_e = fig.add_axes([col_x[0], row3_y, col_w, stat_h])
    from figure1_integrated_four_panels_with_stats import plot_learning_panel
    plot_learning_panel(
        ax_e, task1_curves, group_order=task1_order, colors=task1_colors, labels=task1_labels,
        ylim=STYLE['task1_learning_ylim'], y_ticks=STYLE['task1_learning_y_ticks'],
        show_legend=True, significant_clusters=t1_clusters, max_trial=50,
    )
    add_panel_label(ax_e, 'e')

    ax_f = fig.add_axes([col_x[1], row3_y, col_w, stat_h])
    plot_perfect_rate_panel(
        ax_f, t1_rates, group_order=task1_order, colors=task1_colors, labels=task1_labels,
        test_pair=('P2', 'P2-only'), fisher_result=t1_fisher,
    )
    add_panel_label(ax_f, 'f')

    ax_g = fig.add_axes([col_x[2], row3_y, col_w, stat_h])
    plot_personal_best_panel(
        ax_g, task1_df, subject_col='subject', group_col='phase', trial_col='round', score_col='score',
        group_order=task1_order, colors=task1_colors, labels=task1_labels, window=50,
        mwu_result=t1_pb_mwu, test_pair=('P2', 'P2-only'),
        ylim=(0, 58), y_ticks=[10, 20, 30, 40, 50],
    )
    add_panel_label(ax_g, 'g')

    # row 4: h i j (Task 2 stats)
    ax_h = fig.add_axes([col_x[0], row4_y, col_w, stat_h])
    plot_learning_panel(
        ax_h, task2_curves, group_order=task2_order, colors=task2_colors, labels=task2_labels,
        ylim=STYLE['task2_learning_ylim'], y_ticks=STYLE['task2_learning_y_ticks'],
        show_legend=True, significant_clusters=t2_clusters, max_trial=60,
    )
    add_panel_label(ax_h, 'h')

    ax_i = fig.add_axes([col_x[1], row4_y, col_w, stat_h])
    plot_perfect_rate_panel(
        ax_i, t2_rates, group_order=task2_order, colors=task2_colors, labels=task2_labels,
        test_pair=('FDS-Obs', 'FDS-NoObs'), fisher_result=t2_fisher,
    )
    add_panel_label(ax_i, 'i')

    ax_j = fig.add_axes([col_x[2], row4_y, col_w, stat_h])
    plot_personal_best_panel(
        ax_j, task2_df, subject_col='subject_id', group_col='plot_group', trial_col='trial',
        score_col='score_noisy',
        group_order=task2_order, colors=task2_colors, labels=task2_labels, window=60,
        mwu_result=t2_pb_mwu, test_pair=('FDS-Obs', 'FDS-NoObs'),
        ylim=(0, 70), y_ticks=[15, 30, 45, 60], group_alignment='left',
    )
    add_panel_label(ax_j, 'j')

    png_path = OUTPUT_DIR / 'figure1_full_panels_a_j_with_stats.png'
    pdf_path = OUTPUT_DIR / 'figure1_full_panels_a_j_with_stats.pdf'
    fig.savefig(png_path, dpi=STYLE['dpi'], facecolor='white')
    fig.savefig(pdf_path, facecolor='white')
    print(f'Saved: {png_path}')
    print(f'Saved: {pdf_path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
