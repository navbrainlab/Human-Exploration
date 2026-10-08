"""Shared Fig. 5 data/statistics helpers; no standalone figure export."""
from __future__ import annotations
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import mannwhitneyu
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
RESULTS_ROOT = PROJECT_ROOT
PACKAGE_ROOT = PROJECT_ROOT
CSV_ROOT = PROJECT_ROOT / 'data'
FIGURE_SIZE = (180.0 / 25.4, 6.65)
DPI = 350
INK = '#252B30'
MUTED = '#68717A'
LIGHT_EDGE = '#B8BDC3'
PURPLE = '#6F55A5'
PURPLE_LIGHT = '#D9D0E8'
ORANGE = '#E67E28'
YELLOW = '#F3C316'
BLUE = '#2F5FAE'
DATA_PATHS = {'direct_metadata': CSV_ROOT / 'direct_panel_metadata.csv', 'direct_values': CSV_ROOT / 'direct_panel_subject_values.csv', 'heatmap_matrix': CSV_ROOT / 'heatmap_no_baseline_matrix.csv', 'heatmap_stats': CSV_ROOT / 'heatmap_no_baseline_diagonal_stats.csv', 'heatmap_metadata': CSV_ROOT / 'heatmap_metadata.csv', 'timecourse_values': CSV_ROOT / 'timecourse_subject_values.csv', 'timecourse_metadata': CSV_ROOT / 'timecourse_metadata.csv', 'obs_nobs_exploration': CSV_ROOT / 'obs_nobs_behavior_0930.csv'}
PANEL_MAPPING = {'a': 'Schematic gaze-density examples for DIS and non-DIS trials', 'b': 'Gaze-coverage area on DIS and non-DIS trials', 'c': 'Fixation proportion by DIS dimension and viewed task dimension', 'd': 'Mean gaze duration per covered image on DIS and non-DIS trials', 'e': 'Gaze-shift speed on DIS and non-DIS trials', 'f': 'Gaze-covered task dimensions on DIS and non-DIS trials', 'g': 'Explored dimensions with and without prior observation', 'h': 'Pre-task gaze-covered dimensions with and without observation', 'i': 'Early-task noticed dimensions with and without observation', 'j': 'Score-relevant dimensions from the initial to final task stage'}
PLOT_RC = {'font.family': 'DejaVu Sans', 'font.size': 5.5, 'font.weight': 'normal', 'axes.labelsize': 5.5, 'axes.labelweight': 'normal', 'axes.titlesize': 5.5, 'axes.titleweight': 'normal', 'xtick.labelsize': 5.0, 'ytick.labelsize': 5.0, 'legend.fontsize': 5.0, 'text.color': INK, 'axes.edgecolor': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK, 'axes.grid': False, 'figure.facecolor': 'white', 'savefig.facecolor': 'white'}

def _stars(p_value: float) -> str:
    if p_value < 0.001:
        return '***'
    if p_value < 0.01:
        return '**'
    if p_value < 0.05:
        return '*'
    return 'ns'

def _style_axis(axis: plt.Axes, *, categorical: bool=True) -> None:
    axis.spines[['top', 'right']].set_visible(False)
    axis.spines['left'].set_linewidth(0.65)
    axis.spines['bottom'].set_linewidth(0.65)
    axis.tick_params(axis='both', width=0.65, length=2.5, pad=2.0)
    if categorical:
        axis.spines['bottom'].set_visible(False)
        axis.tick_params(axis='x', length=0, pad=3.0)
        axis.plot([0.0, 0.0, 1.0, 1.0], [-0.145, -0.185, -0.185, -0.145], transform=axis.get_xaxis_transform(), color=INK, linewidth=0.65, clip_on=False)

def _whisker_limits(values: np.ndarray) -> tuple[float, float]:
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    kept = values[(values >= q1 - 1.5 * iqr) & (values <= q3 + 1.5 * iqr)]
    return (float(kept.min()), float(kept.max()))

def _direct_inputs(panel_id: str, metadata: pd.DataFrame, values: pd.DataFrame) -> tuple[pd.Series, list[np.ndarray]]:
    panel = metadata.loc[metadata['panel_id'].eq(panel_id)].iloc[0]
    groups = [values.loc[values['panel_id'].eq(panel_id) & values['group_label'].eq(label), 'value'].to_numpy(dtype=float) for label in (panel['left_label'], panel['right_label'])]
    if any((group.size == 0 for group in groups)):
        raise ValueError(f'Missing group values for {panel_id}')
    return (panel, groups)

def _draw_timecourse(axis: plt.Axes, values: pd.DataFrame) -> None:
    data = values.loc[values['plot_id'].eq('03_obs_vs_nobs_score_relevance')].copy()
    styles = {'Obs': (ORANGE, 'Obs'), 'NObs': (YELLOW, 'NObs')}
    for group, (color, label) in styles.items():
        block = data.loc[data['group_label'].eq(group)]
        summary = block.groupby(['stage_order', 'stage_label'], sort=True)['value'].agg(['mean', 'sem']).reset_index()
        x = summary['stage_order'].to_numpy(dtype=float) - 1.0
        mean = summary['mean'].to_numpy(dtype=float)
        sem = summary['sem'].to_numpy(dtype=float)
        axis.plot(x, mean, color=color, linewidth=0.85, zorder=2)
        axis.errorbar(x, mean, yerr=sem, fmt='o', markersize=4.0, markerfacecolor=color, markeredgecolor='white', markeredgewidth=0.45, ecolor=color, elinewidth=0.85, capsize=2.0, capthick=0.85, label=label, zorder=3)
    axis.set_xlim(-0.35, 1.35)
    axis.set_xticks([0, 1], ['Initial', 'Final'])
    axis.set_ylim(3.2, 6.4)
    axis.set_yticks([3.2, 4.8, 6.4])
    axis.set_ylabel('Score-relevant\ndimensions', labelpad=3.0)
    _style_axis(axis)
    axis.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=4.8, handlelength=0.8, handletextpad=0.25, columnspacing=0.65, borderaxespad=0)

def _compute_statistics(metadata: pd.DataFrame, values: pd.DataFrame, exploration: pd.DataFrame, heatmap_stats: pd.DataFrame | None=None) -> pd.DataFrame:
    panel_ids = {'b': ('02_dis_vs_ndis_19', 'paired two-sided Wilcoxon signed-rank', 'none'), 'd': ('02_dis_vs_ndis_08', 'paired two-sided Wilcoxon signed-rank', 'FDR across gaze metrics'), 'e': ('02_dis_vs_ndis_03', 'paired two-sided Wilcoxon signed-rank', 'none'), 'f': ('02_dis_vs_ndis_01', 'paired two-sided Wilcoxon signed-rank', 'FDR across gaze metrics'), 'h': ('03_obs_vs_nobs_01', 'two-sided Mann–Whitney U', 'Benjamini–Hochberg across eight metrics'), 'i': ('03_obs_vs_nobs_02', 'two-sided Mann–Whitney U', 'none')}
    rows: list[dict[str, object]] = []
    for panel_letter, (panel_id, test_name, adjustment) in panel_ids.items():
        panel, groups = _direct_inputs(panel_id, metadata, values)
        rows.append({'panel': panel_letter, 'comparison': f"{panel['left_label']} vs {panel['right_label']}", 'n_left': int(groups[0].size), 'n_right': int(groups[1].size), 'left_mean': float(groups[0].mean()), 'right_mean': float(groups[1].mean()), 'p_value': float(panel['p_value']), 'significance': _stars(float(panel['p_value'])), 'test': test_name, 'p_adjustment': adjustment, 'source': panel_id})
    obs = exploration.loc[exploration['game_mode'].eq('with_obs'), 'n_dims_with_k_feats'].to_numpy(dtype=float)
    nobs = exploration.loc[exploration['game_mode'].eq('without_obs'), 'n_dims_with_k_feats'].to_numpy(dtype=float)
    test = mannwhitneyu(obs, nobs, alternative='two-sided')
    rows.append({'panel': 'g', 'comparison': 'Obs vs NObs', 'n_left': int(obs.size), 'n_right': int(nobs.size), 'left_mean': float(obs.mean()), 'right_mean': float(nobs.mean()), 'p_value': float(test.pvalue), 'significance': _stars(float(test.pvalue)), 'test': 'two-sided Mann–Whitney U', 'p_adjustment': 'none', 'source': 'obs_nobs_behavior_0930.csv'})
    if heatmap_stats is not None:
        for heatmap_row in heatmap_stats.itertuples(index=False):
            rows.append({'panel': f'c-{heatmap_row.metric_key}', 'comparison': str(heatmap_row.comparison), 'n_left': int(heatmap_row.n), 'n_right': int(heatmap_row.n), 'left_mean': float(heatmap_row.left_mean), 'right_mean': float(heatmap_row.right_mean), 'p_value': float(heatmap_row.p_value), 'significance': str(heatmap_row.stars), 'test': 'one-sided Wilcoxon signed-rank (greater)', 'p_adjustment': 'none recorded in source metadata', 'source': 'heatmap_no_baseline_diagonal_stats.csv'})
    return pd.DataFrame(rows).sort_values('panel').reset_index(drop=True)
