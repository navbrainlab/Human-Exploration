"""
Figure 1 panels e & f with between-group statistics (NHB style).

Adds to the original figure1_integrated_four_panels notebook:
  1. Best-score panels: two-sided Mann-Whitney U test (rank-biserial effect size r),
     visualized as a significance bracket with an exact P value.
  2. Learning-curve panels: trial-level Mann-Whitney U + cluster-based permutation
     test (10,000 label permutations, cluster-forming alpha = 0.05), visualized as a
     significance bar under the x-axis spanning significant trial clusters.

Comparisons:
  - Task 1 (panel e): P2 (4D-E) vs P2-only (4D-NE)
  - Task 2 (panel f): FDS-Obs (Obs) vs FDS-NoObs (NObs)

Outputs use a `_with_stats` suffix and never overwrite the original figures.
"""

from pathlib import Path
import sys

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FixedLocator, FormatStrFormatter
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from publication_plot_style import TASK1_COLORS, TASK1_LABELS, TASK2_COLORS, TASK2_LABELS

DATA_DIR = Path(__file__).resolve().parents[2] / 'data'
OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TASK1_DATA_PATH = DATA_DIR / 'choice_category_uniform_105.csv'
TASK2_DATA_PATH = DATA_DIR / 'summary_data_0723_task2.csv'

STYLE = {
    'learning_figsize': (4.6, 3.25),
    'distribution_figsize': (4.6, 3.25),
    'combined_figsize': (17.8, 3.45),
    'dpi': 300,
    'axis_width': 1.8,
    'tick_width': 1.6,
    'tick_length': 5.8,
    'line_width': 2.2,
    'box_line_width': 1.6,
    'median_width': 2.0,
    'label_size': 17,
    'tick_size': 14.5,
    'legend_size': 10.5,
    'panel_label_size': 13,
    'point_size': 22,
    'point_alpha': 0.88,
    'point_jitter': 0.060,
    'box_width': 0.24,
    'scatter_offset': -0.24,
    'box_offset': 0.0,
    'group_step': 0.82,
    'ribbon_alpha': 0.25,
    'task1_learning_y_ticks': [50, 65, 80, 95],
    'task2_learning_y_ticks': [45, 60, 75, 90],
    'task1_learning_ylim': (50, 95),
    'task2_learning_ylim': (45, 90),
    'learning_xlim': (-0.06, 1.0),
    'distribution_group_slots': 3,
    'distribution_start': 1.08,
    'distribution_left_margin': 0.42,
    'score_y_ticks': [70, 80, 90, 100],
    'score_ylim': (70, 104),  # extended upward to host the significance bracket
    # statistics annotation style
    'stats_fontsize': 11.5,
    'bracket_height': 101.2,
    'bracket_text_height': 102.0,
    'cluster_bar_height': -0.075,   # axes fraction, below the x-axis
    'cluster_label_height': -0.155, # axes fraction
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.grid': False,
})

# ---------------------------------------------------------------------------
# Original notebook helpers (unchanged logic)
# ---------------------------------------------------------------------------

def lighten_color(color, amount=0.25):
    rgb = np.array(mcolors.to_rgb(color))
    return tuple((1 - amount) * rgb + amount)


def centered_rolling_mean(values, window=5):
    return (
        pd.Series(np.asarray(values, dtype=float))
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def normalize_progress(trial_values, min_trial, max_trial):
    trial_values = np.asarray(trial_values, dtype=float)
    if max_trial <= min_trial:
        return np.zeros_like(trial_values, dtype=float)
    return (trial_values - min_trial) / (max_trial - min_trial)


def extend_after_first_perfect(df, *, subject_col, group_col, trial_col, score_col, max_trial_by_group):
    df = df.copy()
    perfect_score = df[score_col].max()
    extended_rows = []

    for _, subject_df in df.groupby([subject_col, group_col], dropna=False):
        subject_df = subject_df.sort_values(trial_col).copy()
        if subject_df.empty:
            continue

        group_value = subject_df[group_col].iloc[0]
        max_trial = int(max_trial_by_group[group_value])
        subject_df = subject_df[subject_df[trial_col] <= max_trial].copy()
        existing_trials = set(subject_df[trial_col].astype(int).tolist())
        subject_parts = [subject_df]

        perfect_hits = subject_df.loc[subject_df[score_col] >= perfect_score, trial_col]
        if not perfect_hits.empty:
            first_perfect_trial = int(perfect_hits.min())
            fill_trials = [
                trial for trial in range(first_perfect_trial + 1, max_trial + 1)
                if trial not in existing_trials
            ]
            if fill_trials:
                fill_df = pd.DataFrame({
                    subject_col: subject_df[subject_col].iloc[0],
                    group_col: group_value,
                    trial_col: fill_trials,
                    score_col: perfect_score,
                })
                subject_parts.append(fill_df)

        extended_rows.append(pd.concat(subject_parts, ignore_index=True))

    if not extended_rows:
        raise ValueError('No data available after extending perfect-score trials.')
    return pd.concat(extended_rows, ignore_index=True)


def compute_learning_curves(df, *, subject_col, group_col, trial_col, score_col, group_order, max_trial_by_group):
    extended_df = extend_after_first_perfect(
        df,
        subject_col=subject_col,
        group_col=group_col,
        trial_col=trial_col,
        score_col=score_col,
        max_trial_by_group=max_trial_by_group,
    )

    curve_dict = {}
    for group in group_order:
        sub = extended_df.loc[extended_df[group_col] == group].copy()
        if sub.empty:
            curve_dict[group] = pd.DataFrame()
            continue

        stats = (
            sub.groupby(trial_col)[score_col]
            .agg(['mean', 'std', 'count'])
            .reset_index()
            .sort_values(trial_col)
        )
        stats['sem'] = (stats['std'] / np.sqrt(stats['count'])).fillna(0)
        stats['x'] = normalize_progress(stats[trial_col], 1, max_trial_by_group[group])
        stats['mean_smooth'] = centered_rolling_mean(stats['mean'], window=5)
        stats['sem_smooth'] = centered_rolling_mean(stats['sem'], window=5)
        curve_dict[group] = stats

    return curve_dict


def style_panel_axis(ax, *, xlabel='', ylabel='', x_ticks=None, y_ticks=None, xlim=None, ylim=None, x_tick_format=None):
    ax.set_title('')
    ax.set_xlabel(xlabel, fontsize=STYLE['label_size'], fontweight='bold', labelpad=6)
    ax.set_ylabel(ylabel, fontsize=STYLE['label_size'], fontweight='bold', labelpad=6)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if x_ticks is not None:
        ax.xaxis.set_major_locator(FixedLocator(x_ticks))
    if y_ticks is not None:
        ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    if x_tick_format is not None:
        ax.xaxis.set_major_formatter(FormatStrFormatter(x_tick_format))

    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(STYLE['axis_width'])
    ax.spines['bottom'].set_linewidth(STYLE['axis_width'])
    ax.spines['bottom'].set_position(('outward', 6))

    if y_ticks is not None and len(y_ticks) >= 2:
        ax.spines['left'].set_bounds(y_ticks[0], y_ticks[-1])
    if x_ticks is not None and len(x_ticks) >= 2:
        ax.spines['bottom'].set_bounds(x_ticks[0], x_ticks[-1])

    ax.tick_params(
        axis='both',
        which='major',
        labelsize=STYLE['tick_size'],
        direction='out',
        length=STYLE['tick_length'],
        width=STYLE['tick_width'],
        top=False,
        right=False,
        pad=4,
    )


def add_panel_label(ax, label):
    ax.text(
        -0.16,
        1.08,
        label,
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=STYLE['panel_label_size'],
        fontweight='bold',
        clip_on=False,
    )


# ---------------------------------------------------------------------------
# Statistics: best score (Mann-Whitney U) and learning curve (cluster permutation)
# ---------------------------------------------------------------------------

def format_pvalue(p):
    """NHB-style exact P value formatting."""
    if p < 0.001:
        return r'$P < 0.001$'
    return rf'$P = {p:.3f}$'


def best_score_mann_whitney(df, *, group_col, score_col, group_a, group_b):
    """Two-sided Mann-Whitney U test with rank-biserial effect size."""
    a = df.loc[df[group_col] == group_a, score_col].dropna().to_numpy(dtype=float)
    b = df.loc[df[group_col] == group_b, score_col].dropna().to_numpy(dtype=float)
    res = mannwhitneyu(a, b, alternative='two-sided', method='auto')
    n1, n2 = len(a), len(b)
    # rank-biserial correlation, signed: positive => group A tends to score higher
    r_rb = 1.0 - 2.0 * res.statistic / (n1 * n2)
    return {
        'test': 'Mann-Whitney U (two-sided)',
        'group_a': group_a,
        'group_b': group_b,
        'n_a': n1,
        'n_b': n2,
        'median_a': float(np.median(a)),
        'median_b': float(np.median(b)),
        'mean_a': float(np.mean(a)),
        'mean_b': float(np.mean(b)),
        'U': float(res.statistic),
        'p': float(res.pvalue),
        'rank_biserial_r': float(r_rb),
    }


def _find_clusters(sig_mask):
    """Contiguous True runs in a boolean mask -> list of (start_idx, end_idx)."""
    clusters = []
    start = None
    for idx, flag in enumerate(sig_mask):
        if flag and start is None:
            start = idx
        if start is not None and (not flag or idx == len(sig_mask) - 1):
            end = idx if flag and idx == len(sig_mask) - 1 else idx - 1
            clusters.append((start, end))
            start = None
    return clusters


def _cluster_mass(p_values, start, end):
    return float(np.nansum(-np.log10(np.clip(p_values[start:end + 1], 1e-300, None))))


def _trial_level_pvalues(matrix_a, matrix_b):
    """Mann-Whitney p-value per trial column (NaN-aware)."""
    pvals = np.full(matrix_a.shape[1], np.nan)
    for j in range(matrix_a.shape[1]):
        a = matrix_a[:, j]
        b = matrix_b[:, j]
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]
        if len(a) == 0 or len(b) == 0:
            continue
        pvals[j] = mannwhitneyu(a, b, alternative='two-sided').pvalue
    return pvals


def learning_curve_cluster_test(
    extended_df,
    *,
    subject_col,
    group_col,
    trial_col,
    score_col,
    group_a,
    group_b,
    max_trial,
    n_permutations=10000,
    alpha=0.05,
    seed=42,
):
    """
    Trial-level Mann-Whitney U + cluster-based permutation test.

    Builds subject x trial score matrices (aligned on raw trial numbers, using the
    same extended data that feeds the plotted curves), finds clusters of adjacent
    trials with trial-level p < alpha, and evaluates each cluster with a
    max-mass permutation procedure over 10,000 group-label permutations.
    """
    series_by_key = {}
    for (subject, group), sdf in extended_df.groupby([subject_col, group_col]):
        sdf = sdf.sort_values(trial_col)
        series_by_key[(group, subject)] = sdf.set_index(trial_col)[score_col]

    trials = np.arange(1, int(max_trial) + 1)

    def build_matrix(group):
        subjects = sorted({k[1] for k in series_by_key if k[0] == group})
        mat = np.full((len(subjects), len(trials)), np.nan)
        for i, subject in enumerate(subjects):
            s = series_by_key[(group, subject)]
            mat[i, :] = [s.get(t, np.nan) for t in trials]
        return mat

    mat_a = build_matrix(group_a)
    mat_b = build_matrix(group_b)

    p_obs = _trial_level_pvalues(mat_a, mat_b)
    obs_clusters = _find_clusters(~np.isnan(p_obs) & (p_obs < alpha))
    obs_masses = [_cluster_mass(p_obs, s, e) for s, e in obs_clusters]

    # permutation null distribution of the maximum cluster mass
    pooled = np.vstack([mat_a, mat_b])
    n_a = mat_a.shape[0]
    rng = np.random.default_rng(seed)
    null_max = np.zeros(n_permutations)
    for i in range(n_permutations):
        order = rng.permutation(pooled.shape[0])
        p_perm = _trial_level_pvalues(pooled[order[:n_a]], pooled[order[n_a:]])
        perm_clusters = _find_clusters(~np.isnan(p_perm) & (p_perm < alpha))
        if perm_clusters:
            null_max[i] = max(_cluster_mass(p_perm, s, e) for s, e in perm_clusters)

    cluster_rows = []
    for cid, ((s, e), mass) in enumerate(zip(obs_clusters, obs_masses), start=1):
        cluster_rows.append({
            'cluster_id': cid,
            'trial_start': int(trials[s]),
            'trial_end': int(trials[e]),
            'n_trials': int(e - s + 1),
            'cluster_mass': mass,
            'cluster_p': float((np.sum(null_max >= mass) + 1) / (n_permutations + 1)),
        })

    trial_rows = []
    for j, t in enumerate(trials):
        row = {
            'trial': int(t),
            'x_progress': (t - 1) / (max_trial - 1),
            'p_mannwhitney': p_obs[j],
            'n_a': int(np.sum(~np.isnan(mat_a[:, j]))),
            'n_b': int(np.sum(~np.isnan(mat_b[:, j]))),
            'median_a': float(np.nanmedian(mat_a[:, j])) if np.sum(~np.isnan(mat_a[:, j])) else np.nan,
            'median_b': float(np.nanmedian(mat_b[:, j])) if np.sum(~np.isnan(mat_b[:, j])) else np.nan,
        }
        membership = [cid for cid, (s, e) in enumerate(obs_clusters, start=1) if s <= j <= e]
        row['in_significant_cluster'] = bool(membership)
        row['cluster_id'] = membership[0] if membership else 0
        trial_rows.append(row)

    return pd.DataFrame(trial_rows), pd.DataFrame(cluster_rows)


# ---------------------------------------------------------------------------
# Plotting with statistical annotations
# ---------------------------------------------------------------------------

def plot_learning_panel(ax, curve_dict, *, group_order, colors, labels, ylim, y_ticks=None,
                        show_legend=True, significant_clusters=None, max_trial=None):
    if y_ticks is None:
        y_ticks = STYLE['task1_learning_y_ticks']

    for group in group_order:
        curve = curve_dict[group]
        if curve.empty:
            continue

        x = curve['x'].to_numpy(dtype=float)
        y = curve['mean_smooth'].to_numpy(dtype=float)
        sem = curve['sem_smooth'].to_numpy(dtype=float)
        color = colors[group]

        ax.plot(x, y, color=color, linewidth=STYLE['line_width'], label=labels[group], zorder=3)
        ax.fill_between(
            x,
            y - sem,
            y + sem,
            color=color,
            alpha=STYLE['ribbon_alpha'],
            linewidth=0,
            zorder=2,
        )

    style_panel_axis(
        ax,
        xlabel='Trial Progression',
        ylabel='Mean Score',
        x_ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        y_ticks=y_ticks,
        xlim=STYLE['learning_xlim'],
        ylim=ylim,
        x_tick_format='%.1f',
    )

    # significant-cluster annotation in the empty upper-left corner of the panel
    if significant_clusters is not None and max_trial is not None:
        sig = significant_clusters.loc[significant_clusters['cluster_p'] < 0.05]
        if sig.empty:
            note = 'Cluster-based permutation test:\nno significant between-group clusters'
        else:
            lines = [f"trials {int(r.trial_start)}\u2013{int(r.trial_end)}: {format_pvalue(r.cluster_p)}"
                     for r in sig.itertuples()]
            note = 'Cluster-based permutation test\n(significant clusters):\n' + '\n'.join(lines)
        ax.text(
            0.03,
            0.935,
            note,
            ha='left',
            va='top',
            fontsize=STYLE['stats_fontsize'] - 2.5,
            transform=ax.transAxes,
            zorder=6,
        )

    if show_legend:
        legend = ax.legend(
            frameon=False,
            fontsize=STYLE['legend_size'],
            loc='lower right',
            handlelength=2.0,
            borderaxespad=0.2,
        )
        for line in legend.get_lines():
            line.set_linewidth(STYLE['line_width'])


def plot_raincloud_panel(
    ax,
    df,
    *,
    group_col,
    score_col,
    group_order,
    colors,
    labels,
    ylabel='Best score',
    show_violin=True,
    show_scatter=False,
    group_alignment='center',
    test_pair=None,
    best_score_result=None,
):
    data = [df.loc[df[group_col] == group, score_col].dropna().to_numpy(dtype=float) for group in group_order]
    slot_count = max(len(group_order), STYLE['distribution_group_slots'])
    slot_span = (slot_count - 1) * STYLE['group_step']
    if group_alignment == 'left':
        start = STYLE['distribution_start']
    else:
        start = STYLE['distribution_start'] + (slot_count - len(group_order)) * STYLE['group_step'] / 2
    group_centers = np.arange(len(group_order)) * STYLE['group_step'] + start
    scatter_positions = group_centers + STYLE['scatter_offset']
    box_positions = group_centers + STYLE['box_offset']

    box = ax.boxplot(
        data,
        positions=box_positions,
        widths=STYLE['box_width'],
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color='black', linewidth=STYLE['median_width']),
        whiskerprops=dict(color='black', linewidth=STYLE['box_line_width']),
        capprops=dict(color='black', linewidth=STYLE['box_line_width']),
        boxprops=dict(edgecolor='black', linewidth=STYLE['box_line_width']),
    )
    for patch, group in zip(box['boxes'], group_order):
        patch.set_facecolor(lighten_color(colors[group], amount=0.18))
        patch.set_alpha(0.90)

    if show_violin:
        violin_positions = group_centers + STYLE['violin_offset']
        violins = ax.violinplot(
            data,
            positions=violin_positions,
            widths=STYLE['violin_width'],
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for idx, body in enumerate(violins['bodies']):
            group = group_order[idx]
            body.set_facecolor(lighten_color(colors[group], amount=0.30))
            body.set_edgecolor('black')
            body.set_linewidth(STYLE['box_line_width'])
            body.set_alpha(0.70)
            vertices = body.get_paths()[0].vertices
            vertices[:, 0] = np.maximum(vertices[:, 0], violin_positions[idx])

        for pos, values in zip(violin_positions, data):
            if len(values) > 0:
                ax.vlines(pos, np.min(values), np.max(values), color='black', linewidth=1.15, zorder=2)

    if show_scatter:
        rng = np.random.default_rng(2026)
        for pos, values, group in zip(scatter_positions, data, group_order):
            if len(values) == 0:
                continue
            jitter = rng.uniform(-STYLE['point_jitter'], STYLE['point_jitter'], size=len(values))
            ax.scatter(
                pos + jitter,
                values,
                s=STYLE['point_size'],
                color=colors[group],
                alpha=STYLE['point_alpha'],
                edgecolor='white',
                linewidth=0.45,
                zorder=4,
            )

    ax.set_xticks(group_centers)
    ax.set_xticklabels([labels[group] for group in group_order], fontsize=STYLE['tick_size'])
    style_panel_axis(
        ax,
        xlabel='',
        ylabel=ylabel,
        y_ticks=STYLE['score_y_ticks'],
        xlim=(1.0 - STYLE['distribution_left_margin'], STYLE['distribution_start'] + slot_span + 0.26),
        ylim=STYLE['score_ylim'],
    )
    ax.set_xticks(group_centers)
    ax.set_xticklabels([labels[group] for group in group_order], fontsize=STYLE['tick_size'])
    ax.spines['bottom'].set_bounds(group_centers[0], group_centers[-1])

    # significance bracket for the tested pair
    if test_pair is not None and best_score_result is not None:
        idx_a = group_order.index(test_pair[0])
        idx_b = group_order.index(test_pair[1])
        x_a = group_centers[idx_a]
        x_b = group_centers[idx_b]
        y_b = STYLE['bracket_height']
        y_t = STYLE['bracket_text_height']
        ax.plot(
            [x_a, x_a, x_b, x_b],
            [y_b - 0.9, y_b, y_b, y_b - 0.9],
            color='black',
            linewidth=1.3,
            clip_on=False,
            zorder=6,
        )
        label = format_pvalue(best_score_result['p']) if best_score_result['p'] < 0.05 else 'n.s.'
        ax.text(
            (x_a + x_b) / 2,
            y_t,
            label,
            ha='center',
            va='bottom',
            fontsize=STYLE['stats_fontsize'],
            zorder=6,
        )
        ax.text(
            0.02,
            0.03,
            'Mann-Whitney U (two-sided)',
            ha='left',
            va='bottom',
            fontsize=STYLE['stats_fontsize'] - 2.5,
            style='italic',
            transform=ax.transAxes,
        )


# ---------------------------------------------------------------------------
# Data loading (same pipeline as the original notebook)
# ---------------------------------------------------------------------------

def load_task1():
    task1_df = pd.read_csv(TASK1_DATA_PATH, index_col=0)
    task1_df['round'] = pd.to_numeric(task1_df['round'], errors='coerce')
    task1_df['score'] = pd.to_numeric(task1_df['score'], errors='coerce')
    task1_df = task1_df.dropna(subset=['phase', 'subject', 'round', 'score']).copy()

    task1_order = ['P1', 'P2', 'P2-only']
    task1_labels = {group: TASK1_LABELS.get(group, group) for group in task1_order}
    task1_colors = {group: TASK1_COLORS.get(group, '#7f7f7f') for group in task1_order}
    task1_max_trial = {'P1': 30, 'P2': 50, 'P2-only': 50}

    task1_learning_df = task1_df[task1_df['phase'].isin(task1_order)].copy()
    task1_curves = compute_learning_curves(
        task1_learning_df,
        subject_col='subject',
        group_col='phase',
        trial_col='round',
        score_col='score',
        group_order=task1_order,
        max_trial_by_group=task1_max_trial,
    )

    task1_score_records = []
    for phase in task1_order:
        phase_df = task1_df.loc[task1_df['phase'] == phase].copy()
        window = task1_max_trial[phase]
        for subject, subject_df in phase_df.groupby('subject'):
            window_df = subject_df.sort_values('round').tail(window)
            if not window_df.empty:
                task1_score_records.append({
                    'phase': phase,
                    'subject': subject,
                    'best_score': window_df['score'].max(),
                })
    task1_score_df = pd.DataFrame(task1_score_records)

    # extended per-trial data used for the cluster test (identical to curve input)
    task1_extended = extend_after_first_perfect(
        task1_learning_df,
        subject_col='subject',
        group_col='phase',
        trial_col='round',
        score_col='score',
        max_trial_by_group=task1_max_trial,
    )
    return task1_df, task1_extended, task1_curves, task1_score_df, task1_order, task1_labels, task1_colors, task1_max_trial


def load_task2():
    task2_raw = pd.read_csv(TASK2_DATA_PATH)
    task2_df = task2_raw.rename(columns={'trial_id': 'trial'}).copy()
    task2_df['trial'] = pd.to_numeric(task2_df['trial'], errors='coerce')
    task2_df['score_noisy'] = pd.to_numeric(task2_df['score_noisy'], errors='coerce')

    task2_group_map = {
        'observation': 'FDS-Obs',
        'non_observation': 'FDS-NoObs',
        'disrupt FDS': 'NoFDS-Obs',
    }
    task2_df['plot_group'] = task2_df['group'].map(task2_group_map)
    task2_df = task2_df.dropna(subset=['subject_id', 'trial', 'score_noisy', 'plot_group']).copy()

    task2_order = ['FDS-Obs', 'FDS-NoObs']
    task2_df = task2_df[task2_df['plot_group'].isin(task2_order)].copy()
    task2_labels = {group: TASK2_LABELS.get(group, group) for group in task2_order}
    task2_colors = {group: TASK2_COLORS.get(group, '#7f7f7f') for group in task2_order}
    task2_max_trial = {group: 60 for group in task2_order}

    group_check = task2_df.groupby('subject_id')['plot_group'].nunique(dropna=True)
    bad_subjects = group_check[group_check > 1].index.tolist()
    if bad_subjects:
        raise ValueError(f'Some Task 2 subjects appear in more than one group: {bad_subjects[:10]}')

    task2_curves = compute_learning_curves(
        task2_df,
        subject_col='subject_id',
        group_col='plot_group',
        trial_col='trial',
        score_col='score_noisy',
        group_order=task2_order,
        max_trial_by_group=task2_max_trial,
    )

    task2_score_df = (
        task2_df.groupby(['subject_id', 'plot_group'], as_index=False)
        .agg(best_score=('score_noisy', 'max'))
    )

    task2_extended = extend_after_first_perfect(
        task2_df,
        subject_col='subject_id',
        group_col='plot_group',
        trial_col='trial',
        score_col='score_noisy',
        max_trial_by_group=task2_max_trial,
    )
    return task2_df, task2_extended, task2_curves, task2_score_df, task2_order, task2_labels, task2_colors, task2_max_trial


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    n_permutations = 10000

    task1_df, task1_extended, task1_curves, task1_score_df, task1_order, task1_labels, task1_colors, task1_max_trial = load_task1()
    task2_df, task2_extended, task2_curves, task2_score_df, task2_order, task2_labels, task2_colors, task2_max_trial = load_task2()

    # --- statistics: best score -------------------------------------------
    task1_best = best_score_mann_whitney(
        task1_score_df, group_col='phase', score_col='best_score', group_a='P2', group_b='P2-only'
    )
    task2_best = best_score_mann_whitney(
        task2_score_df, group_col='plot_group', score_col='best_score', group_a='FDS-Obs', group_b='FDS-NoObs'
    )

    best_summary = pd.DataFrame([
        {'panel': 'e_task1_best_score', **task1_best},
        {'panel': 'f_task2_best_score', **task2_best},
    ])
    best_summary.to_csv(OUTPUT_DIR / 'figure1_best_score_group_tests.csv', index=False)

    print('Best-score tests:')
    for row in best_summary.to_dict('records'):
        print(
            f"  {row['panel']}: {row['group_a']} vs {row['group_b']}, "
            f"U={row['U']:.1f}, P={row['p']:.4g}, r_rb={row['rank_biserial_r']:.3f}"
        )

    # --- statistics: learning curve cluster tests --------------------------
    print(f'Running cluster-based permutation tests ({n_permutations} permutations each)...')

    task1_trial_stats, task1_clusters = learning_curve_cluster_test(
        task1_extended.loc[task1_extended['phase'].isin(['P2', 'P2-only'])],
        subject_col='subject',
        group_col='phase',
        trial_col='round',
        score_col='score',
        group_a='P2',
        group_b='P2-only',
        max_trial=50,
        n_permutations=n_permutations,
    )
    task1_trial_stats.to_csv(OUTPUT_DIR / 'figure1_panel_e_task1_learning_curve_trial_stats.csv', index=False)
    task1_clusters.to_csv(OUTPUT_DIR / 'figure1_panel_e_task1_learning_curve_clusters.csv', index=False)

    task2_trial_stats, task2_clusters = learning_curve_cluster_test(
        task2_extended,
        subject_col='subject_id',
        group_col='plot_group',
        trial_col='trial',
        score_col='score_noisy',
        group_a='FDS-Obs',
        group_b='FDS-NoObs',
        max_trial=60,
        n_permutations=n_permutations,
    )
    task2_trial_stats.to_csv(OUTPUT_DIR / 'figure1_panel_f_task2_learning_curve_trial_stats.csv', index=False)
    task2_clusters.to_csv(OUTPUT_DIR / 'figure1_panel_f_task2_learning_curve_clusters.csv', index=False)

    print('Learning-curve cluster tests:')
    print(f"  panel e (P2 vs P2-only): {len(task1_clusters)} cluster(s)")
    if not task1_clusters.empty:
        print(task1_clusters.to_string(index=False))
    print(f"  panel f (FDS-Obs vs FDS-NoObs): {len(task2_clusters)} cluster(s)")
    if not task2_clusters.empty:
        print(task2_clusters.to_string(index=False))

    # --- combined four-panel figure ----------------------------------------
    fig = plt.figure(figsize=STYLE['combined_figsize'])
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 1.0], wspace=0.55)
    axes = [fig.add_subplot(gs[0, idx]) for idx in range(4)]

    plot_learning_panel(
        axes[0], task1_curves,
        group_order=task1_order, colors=task1_colors, labels=task1_labels,
        ylim=STYLE['task1_learning_ylim'], y_ticks=STYLE['task1_learning_y_ticks'],
        show_legend=True, significant_clusters=task1_clusters, max_trial=50,
    )
    plot_raincloud_panel(
        axes[1], task1_score_df,
        group_col='phase', score_col='best_score',
        group_order=task1_order, colors=task1_colors, labels=task1_labels,
        show_violin=False, show_scatter=True,
        test_pair=('P2', 'P2-only'), best_score_result=task1_best,
    )
    plot_learning_panel(
        axes[2], task2_curves,
        group_order=task2_order, colors=task2_colors, labels=task2_labels,
        ylim=STYLE['task2_learning_ylim'], y_ticks=STYLE['task2_learning_y_ticks'],
        show_legend=True, significant_clusters=task2_clusters, max_trial=60,
    )
    plot_raincloud_panel(
        axes[3], task2_score_df,
        group_col='plot_group', score_col='best_score',
        group_order=task2_order, colors=task2_colors, labels=task2_labels,
        show_violin=False, show_scatter=True, group_alignment='left',
        test_pair=('FDS-Obs', 'FDS-NoObs'), best_score_result=task2_best,
    )

    combined_path = OUTPUT_DIR / 'figure1_integrated_four_panels_with_stats.png'
    combined_pdf_path = OUTPUT_DIR / 'figure1_integrated_four_panels_with_stats.pdf'
    fig.savefig(combined_path, dpi=STYLE['dpi'], bbox_inches='tight')
    fig.savefig(combined_pdf_path, bbox_inches='tight')
    print(f'Saved: {combined_path}')
    print(f'Saved: {combined_pdf_path}')
    plt.close(fig)

    # --- standalone panels --------------------------------------------------
    panel_specs = [
        ('figure1_panel_e_task1_learning_curve_with_stats.png', STYLE['learning_figsize'],
         lambda ax: plot_learning_panel(
             ax, task1_curves, group_order=task1_order, colors=task1_colors,
             labels=task1_labels, ylim=STYLE['task1_learning_ylim'],
             y_ticks=STYLE['task1_learning_y_ticks'], show_legend=True,
             significant_clusters=task1_clusters, max_trial=50,
         )),
        ('figure1_panel_e_task1_best_score_distribution_with_stats.png', STYLE['distribution_figsize'],
         lambda ax: plot_raincloud_panel(
             ax, task1_score_df, group_col='phase', score_col='best_score',
             group_order=task1_order, colors=task1_colors, labels=task1_labels,
             show_violin=False, show_scatter=True,
             test_pair=('P2', 'P2-only'), best_score_result=task1_best,
         )),
        ('figure1_panel_f_task2_learning_curve_with_stats.png', STYLE['learning_figsize'],
         lambda ax: plot_learning_panel(
             ax, task2_curves, group_order=task2_order, colors=task2_colors,
             labels=task2_labels, ylim=STYLE['task2_learning_ylim'],
             y_ticks=STYLE['task2_learning_y_ticks'], show_legend=True,
             significant_clusters=task2_clusters, max_trial=60,
         )),
        ('figure1_panel_f_task2_best_score_distribution_with_stats.png', STYLE['distribution_figsize'],
         lambda ax: plot_raincloud_panel(
             ax, task2_score_df, group_col='plot_group', score_col='best_score',
             group_order=task2_order, colors=task2_colors, labels=task2_labels,
             show_violin=False, show_scatter=True, group_alignment='left',
             test_pair=('FDS-Obs', 'FDS-NoObs'), best_score_result=task2_best,
         )),
    ]

    for filename, figsize, draw_func in panel_specs:
        panel_fig, panel_ax = plt.subplots(figsize=figsize)
        draw_func(panel_ax)
        panel_path = OUTPUT_DIR / filename
        panel_fig.savefig(panel_path, dpi=STYLE['dpi'], bbox_inches='tight')
        plt.close(panel_fig)
        print(f'Saved: {panel_path}')


if __name__ == '__main__':
    main()
