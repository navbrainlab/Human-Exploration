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
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from matplotlib.ticker import FixedLocator, LogFormatterMathtext, NullLocator, PercentFormatter
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import gaussian_kde
Bounds = tuple[float, float, float, float]
CODE_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = CODE_ROOT.parent
CSV_ROOT = PACKAGE_ROOT / 'data'
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
from attention_panel import draw_attention_temperature_bins
FIGURE_SIZE = (7.2, 6.65)
DPI = 350
OUTPUT_DIR = PACKAGE_ROOT / 'output'
OUTPUT_PATH = OUTPUT_DIR / 'fig6_nhb_typography.png'
MANIFEST_DIR = PACKAGE_ROOT / 'output'
MANIFEST_PATH = PACKAGE_ROOT / 'fig6_nhb_typography_manifest.json'
STATISTICS_PATH = CSV_ROOT / 'panel_statistics.csv'
CAPTION_PATH = PACKAGE_ROOT / 'caption.md'
METHODS_PATH = PACKAGE_ROOT / 'methods.md'
SOURCE_PROVENANCE_PATH = PACKAGE_ROOT / 'source_provenance.json'
PANEL_MAPPING = {'a': 'DGEM feature-to-action computational flow (vector artwork)', 'b': 'Grouped-PL likelihood comparison, Task 1 and Task 2', 'c': 'Effective attention temperature versus maximum dimension attention weight', 'd': 'Maximum dimension attention weight versus DIS ratio, with group distributions', 'e': 'Task-specific best-performance full-score proportion versus DIS ratio', 'f': 'Entity-level simulation-versus-fitting attention-temperature distributions and significance tests', 'g': 'Fitted-parameter simulation DIS ratio and DGEM temperature by round', 'h': 'Two-stage dimension and feature selection with direct candidate-item scoring and reward propagation', 'i': 'Digitized TDGE reference comparison across contextual-bandit baselines'}
INK = '#252B30'
MUTED = '#68717A'
LIGHT_EDGE = '#B8BDC3'
DGEM_PURPLE = '#6F55A5'
PLOT_RC = {'font.family': 'DejaVu Sans', 'font.size': 5.5, 'font.weight': 'normal', 'axes.labelsize': 5.5, 'axes.labelweight': 'normal', 'axes.titlesize': 6.0, 'axes.titleweight': 'normal', 'xtick.labelsize': 5.0, 'ytick.labelsize': 5.0, 'legend.fontsize': 5.0, 'text.color': INK, 'axes.edgecolor': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK, 'axes.grid': False, 'figure.facecolor': 'white', 'savefig.facecolor': 'white'}
DATA_PATHS = {'fit_summary': CSV_ROOT / 'fit_task1.csv', 'fit_significance': CSV_ROOT / 'fit_task1_significance.csv', 'attention_bins': CSV_ROOT / 'attention_bins.csv', 'attention_points': CSV_ROOT / 'attention_trial_coordinates.csv', 'attention_curve': CSV_ROOT / 'attention_dis_logit_curve.csv', 'attention_distribution_shape': CSV_ROOT / 'attention_dis_distribution_shape.csv', 'attention_distribution_summary': CSV_ROOT / 'attention_dis_distribution_summary.csv', 'best_performance': CSV_ROOT / 'performance_task1.csv', 'dis_by_round': CSV_ROOT / 'dis_by_round.csv', 'simulation_temperature': CSV_ROOT / 'temperature_by_round.csv', 'temperature_entities': CSV_ROOT / 'temperature_entity_means.csv', 'panel_i_reference': CSV_ROOT / 'tdge_digitized_reference.csv', 'task2_fit': CSV_ROOT / 'fit_task2.csv', 'task2_performance': CSV_ROOT / 'performance_task2.csv', 'panel_statistics': STATISTICS_PATH}

def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()

def _significance_stars(p_value: float) -> str:
    if not np.isfinite(p_value) or p_value >= 0.05:
        return 'ns'
    if p_value < 0.001:
        return '***'
    if p_value < 0.01:
        return '**'
    return '*'

def _compute_panel_statistics() -> pd.DataFrame:
    """Load the three exact comparisons exported with the plotting values."""
    statistics = pd.read_csv(STATISTICS_PATH)
    expected = {'d_DIS_vs_Non-DIS', 'f_3D_simulation_vs_fitting', 'f_4D_simulation_vs_fitting'}
    if set(statistics['panel_comparison']) != expected:
        raise ValueError('Packaged panel statistics are incomplete')
    if not statistics['p_value'].between(0, 1).all():
        raise ValueError('Packaged panel statistics have invalid p-values')
    return statistics

def _flow_box(axis: plt.Axes, bounds: Bounds, text: str, *, edgecolor: str=LIGHT_EDGE, facecolor: str='white', textcolor: str=INK, fontsize: float=6.2, linewidth: float=0.8, fontweight: str='normal') -> None:
    left, bottom, width, height = bounds
    patch = Rectangle((left, bottom), width, height, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, zorder=3)
    patch.set_gid('flow-box')
    axis.add_patch(patch)
    label = axis.text(left + width / 2, bottom + height / 2, text, ha='center', va='center', fontsize=fontsize, fontweight=fontweight, color=textcolor, linespacing=1.12, zorder=4)
    label.set_gid('flow-label')

def _orthogonal_arrow(axis: plt.Axes, points: tuple[tuple[float, float], ...], *, color: str=MUTED, linewidth: float=0.7) -> None:
    """Draw a square-corner route with an arrow only on its final segment."""
    if len(points) < 2:
        raise ValueError('An orthogonal arrow needs at least two points')
    if len(points) > 2:
        x, y = zip(*points[:-1])
        axis.plot(x, y, color=color, linewidth=linewidth, solid_capstyle='butt', solid_joinstyle='miter', zorder=2)
    _arrow(axis, points[-2], points[-1], color=color, linewidth=linewidth)

def _arrow(axis: plt.Axes, start: tuple[float, float], end: tuple[float, float], *, color: str=MUTED, linewidth: float=0.7, connectionstyle: str='arc3') -> None:
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=4.8, linewidth=linewidth, color=color, shrinkA=0.0, shrinkB=0.0, connectionstyle=connectionstyle, capstyle='butt', joinstyle='miter', zorder=2))

def _operator(axis: plt.Axes, center: tuple[float, float], symbol: str, *, edgecolor: str=MUTED) -> None:
    circle = Circle(center, radius=0.022, facecolor='white', edgecolor=edgecolor, linewidth=0.75, zorder=4)
    axis.add_patch(circle)
    axis.text(*center, symbol, ha='center', va='center', fontsize=7.0, color=edgecolor, zorder=5)

def draw_model_flow_panel(figure: plt.Figure, bounds: Bounds) -> None:
    """Draw panel a using the reference diagram's fixed rectangular structure."""
    axis = figure.add_axes(bounds)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_axis_off()
    start_box = (0.01, 0.405, 0.14, 0.175)
    uncertainty_box = (0.21, 0.475, 0.16, 0.275)
    value_box = (0.21, 0.165, 0.16, 0.17)
    sharpness_box = (0.3, 0.765, 0.16, 0.22)
    weight_box = (0.48, 0.7, 0.18, 0.26)
    feature_box = (0.49, 0.39, 0.13, 0.2)
    action_score_box = (0.7, 0.385, 0.13, 0.21)
    policy_box = (0.845, 0.405, 0.085, 0.17)
    action_box = (0.945, 0.385, 0.054, 0.21)
    _flow_box(axis, start_box, 'For each feature ($f$)\nin dimension ($d$)', fontsize=6.0)
    _flow_box(axis, uncertainty_box, 'Feature uncertainty\n($U_{d,f}$)', fontsize=6.2)
    _flow_box(axis, value_box, 'Feature value ($V_{d,f}$)', fontsize=6.0)
    _flow_box(axis, sharpness_box, 'Softmax with\nattention sharpness\n($\\tau_{att}$)', edgecolor=DGEM_PURPLE, facecolor='white', textcolor=DGEM_PURPLE, linewidth=1.0, fontweight='semibold')
    _flow_box(axis, weight_box, 'Attention weights on\nfeature dimensions\n($w_d$)', edgecolor=DGEM_PURPLE, facecolor='white', textcolor=DGEM_PURPLE, linewidth=1.0, fontweight='semibold')
    _flow_box(axis, feature_box, 'Feature score\n($S_{d,f}$)', fontsize=6.0)
    _flow_box(axis, action_score_box, 'Action Score\n($S_{A_i}$)', fontsize=6.0)
    _flow_box(axis, policy_box, 'Softmax\npolicy', fontsize=5.7)
    _flow_box(axis, action_box, 'Action\n($A_t$)', fontsize=5.8)
    plus_center = (0.43, 0.49)
    multiply_center = (0.665, 0.49)
    _operator(axis, plus_center, '+')
    _operator(axis, multiply_center, '$\\times$', edgecolor=INK)
    _orthogonal_arrow(axis, ((0.15, 0.493), (0.18, 0.493), (0.18, 0.613), (0.21, 0.613)))
    _orthogonal_arrow(axis, ((0.15, 0.493), (0.18, 0.493), (0.18, 0.25), (0.21, 0.25)))
    _orthogonal_arrow(axis, ((0.37, 0.585), (0.39, 0.585), (0.39, 0.508), (0.407, 0.508)))
    _orthogonal_arrow(axis, ((0.37, 0.25), (0.39, 0.25), (0.39, 0.472), (0.407, 0.472)))
    _arrow(axis, (0.453, 0.49), (0.49, 0.49))
    _orthogonal_arrow(axis, ((0.29, 0.75), (0.29, 0.86), (0.3, 0.86)), color=DGEM_PURPLE, linewidth=0.9)
    _arrow(axis, (0.46, 0.86), (0.48, 0.86), color=DGEM_PURPLE, linewidth=0.9)
    _orthogonal_arrow(axis, ((0.66, 0.83), (0.665, 0.83), (0.665, 0.514)), color=DGEM_PURPLE, linewidth=0.9)
    _arrow(axis, (0.62, 0.49), (0.642, 0.49))
    _arrow(axis, (0.688, 0.49), (0.7, 0.49))
    _arrow(axis, (0.83, 0.49), (0.845, 0.49))
    _arrow(axis, (0.93, 0.49), (0.945, 0.49))

def _style_data_axis(axis: plt.Axes, *, left: bool=True, bottom: bool=True, linewidth: float=0.65) -> None:
    axis.grid(False)
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['left'].set_visible(left)
    axis.spines['bottom'].set_visible(bottom)
    if left:
        axis.spines['left'].set_linewidth(linewidth)
    if bottom:
        axis.spines['bottom'].set_linewidth(linewidth)
    axis.tick_params(axis='both', which='major', direction='out', width=linewidth, length=2.4, pad=2.0)

def _style_bracketed_categorical_x_axis(axis: plt.Axes, positions: tuple[float, ...], labels: tuple[str, ...]) -> None:
    """Apply the bracketed categorical x-axis used in the supplied reference."""
    axis.set_xticks(positions, labels)
    axis.tick_params(axis='x', bottom=False, length=0, labelsize=5.8, pad=7.0)
    for label in axis.get_xticklabels():
        label.set_fontweight('bold')
        label.set_linespacing(0.95)
    left = positions[0]
    right = positions[-1]
    transform = axis.get_xaxis_transform()
    axis.plot([left, left, right, right], [-0.045, -0.015, -0.015, -0.045], transform=transform, color=INK, linewidth=0.65, solid_capstyle='butt', solid_joinstyle='miter', clip_on=False, zorder=6)
    for label in (*axis.get_xticklabels(), *axis.get_yticklabels()):
        label.set_fontweight('normal')

def _draw_fitting_comparison(figure: plt.Figure) -> None:
    task1_summary = pd.read_csv(DATA_PATHS['fit_summary']).set_index('model')
    task2_summary = pd.read_csv(DATA_PATHS['task2_fit'])
    task2_summary['model'] = task2_summary['model'].replace({'fRL': 'fRL-decay'})
    task2_summary = task2_summary.set_index('model')
    significance = pd.read_csv(DATA_PATHS['fit_significance'])
    panel_models = (('DGEM', 'fRL-decay', 'naiveRL', 'Bayesian'), ('DGEM', 'fRL-decay', 'naiveRL', 'Bayesian', 'ACL'))
    colors = {'DGEM': DGEM_PURPLE, 'fRL-decay': '#D8B938', 'naiveRL': '#8A6A3D', 'Bayesian': '#D88A42', 'ACL': '#4C956C'}
    y_by_model = {'DGEM': 4.0, 'fRL-decay': 3.0, 'naiveRL': 2.0, 'Bayesian': 1.0, 'ACL': 0.0}
    axes = (figure.add_axes((0.073, 0.445, 0.152, 0.19)), figure.add_axes((0.273, 0.445, 0.162, 0.19)))
    panel_values = (pd.DataFrame({'mean': task1_summary['mean_geom_likelihood'].mul(1000.0), 'se': task1_summary['sem_geom_likelihood'].mul(1000.0)}), pd.DataFrame({'mean': task2_summary['mean'].mul(1000.0), 'se': task2_summary['se'].mul(1000.0)}))
    x_bracket = 14.7
    task1_stars = '***'
    if 'significance' in significance:
        ordered = significance['significance'].astype(str)
        if not ordered.empty:
            task1_stars = min(ordered, key=lambda value: {'ns': 0, '*': 1, '**': 2, '***': 3}.get(value, 0))
    task2_stars = _significance_stars(float(task2_summary['p_vs_dgem'].dropna().max()))
    panel_stars = (task1_stars, task2_stars)
    for index, (axis, title, models, values, stars) in enumerate(zip(axes, ('Task 1', 'Task 2'), panel_models, panel_values, panel_stars)):
        mean = values['mean']
        sem = values['se']
        rightmost_baseline = max((float(mean[m] + sem[m]) for m in models[1:]))
        shared_x = rightmost_baseline + 0.5
        dgem_end = float(mean['DGEM'] + sem['DGEM'] + 0.2)
        for model in models:
            y = y_by_model[model]
            axis.barh(y, mean[model], xerr=sem[model], height=0.54, color=colors[model], edgecolor='white', linewidth=0.35, capsize=1.8, error_kw={'elinewidth': 0.7, 'capthick': 0.7, 'ecolor': INK}, zorder=3)
        axis.axvline(1000.0 / 1680.0, color='#777777', linewidth=0.6, linestyle=(0, (3, 3)), zorder=1)
        shared_bottom = y_by_model['Bayesian'] - 0.34 if index == 0 else y_by_model['ACL'] - 0.34
        shared_top = y_by_model['fRL-decay'] + 0.34
        shared_mid = (shared_top + shared_bottom) / 2.0
        axis.plot([dgem_end, x_bracket, x_bracket, shared_x], [y_by_model['DGEM'], y_by_model['DGEM'], shared_mid, shared_mid], color='#666666', linewidth=0.65, solid_capstyle='butt', clip_on=False, zorder=5)
        axis.plot([shared_x, shared_x], [shared_bottom, shared_top], color='#666666', linewidth=0.65, solid_capstyle='butt', clip_on=False, zorder=4)
        axis.text(15.85, (y_by_model['DGEM'] + shared_mid) / 2.0, stars, rotation=90, ha='center', va='center', fontsize=7.0, fontweight='bold', color=INK)
        axis.set_xlim(0, 17.0)
        axis.set_ylim(-0.6, 5.0)
        axis.set_xticks([0, 5, 10, 15])
        axis.set_yticks([4, 3, 2, 1, 0])
        if index == 0:
            axis.set_yticklabels(['DGEM', 'fRL-decay', 'naiveRL', 'Bayesian', ''])
        else:
            axis.set_yticklabels(['', '', '', '', 'ACL'])
            axis.get_yticklabels()[-1].set_color('#666666')
        axis.set_title(title, loc='left', pad=4.0, fontsize=6.0)
        axis.text(0.76, 4.72, 'chance = 1/1680', ha='left', va='top', fontsize=5.0, color='#666666')
        _style_data_axis(axis)
    figure.text(0.254, 0.408, 'Mean likelihood per trial (×10⁻³)', ha='center', va='center', fontsize=5.5, color=INK)

def _draw_attention_panels(figure: plt.Figure, statistics: pd.DataFrame) -> None:
    plotted_points = pd.read_csv(DATA_PATHS['attention_points'])
    bins = pd.read_csv(DATA_PATHS['attention_bins'])
    bins['in_main_panel'] = bins['in_main_panel'].astype(str).str.lower().eq('true')
    left = figure.add_axes((0.495, 0.432, 0.19, 0.215))
    inset = draw_attention_temperature_bins(left, plotted_points, bins, temperature_column='concrete_attention_temp', color=DGEM_PURPLE, ink=INK)
    _style_data_axis(left)
    _style_data_axis(inset, linewidth=0.55)
    left.tick_params(labelsize=5.0)
    inset.tick_params(labelsize=5.0, length=1.8, width=0.55, pad=1.5)
    for text in left.texts:
        if text.get_text().startswith('Binned mean'):
            text.set_visible(False)
    inset.set_title('Full range', fontsize=5.0, fontweight='normal', pad=2)
    for label in (*inset.get_xticklabels(), *inset.get_yticklabels()):
        label.set_fontweight('normal')
    curve = pd.read_csv(DATA_PATHS['attention_curve'])
    probability = figure.add_axes((0.76, 0.512, 0.215, 0.135))
    outcomes = figure.add_axes((0.76, 0.432, 0.215, 0.067), sharex=probability)
    probability.fill_between(curve['dim_attention_max'], curve['ci_lower'], curve['ci_upper'], color=DGEM_PURPLE, alpha=0.18, linewidth=0, label='95% CI')
    probability.plot(curve['dim_attention_max'], curve['observed_dis_probability'], color=DGEM_PURPLE, linewidth=1.0, label='Binomial logit')
    probability.set_xlim(0.25, 1.12)
    probability.set_ylim(0.0, 1.0)
    probability.set_yticks([0.0, 1 / 3, 2 / 3, 1.0])
    probability.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    probability.set_ylabel('DIS ratio')
    probability.tick_params(axis='x', bottom=False, labelbottom=False)
    probability.legend(frameon=False, loc='lower right', fontsize=5.0, handlelength=1.5, borderaxespad=0.2, labelspacing=0.25)
    _style_data_axis(probability, bottom=False)
    distribution = pd.read_csv(DATA_PATHS['attention_distribution_shape'])
    summary = pd.read_csv(DATA_PATHS['attention_distribution_summary'])
    for row in summary.itertuples(index=False):
        position = int(row.observed_dis)
        point_color = DGEM_PURPLE if position == 1 else '#737A82'
        block = distribution.loc[distribution['observed_dis'].eq(position)]
        grid = block['dim_attention_max'].to_numpy(dtype=float)
        width = block['half_width'].to_numpy(dtype=float)
        outcomes.fill_between(grid, position - width, position + width, facecolor=point_color, edgecolor=point_color, alpha=0.3, linewidth=0.6, zorder=1)
        outcomes.plot([row.q1, row.q3], [position, position], color=INK, linewidth=1.5, solid_capstyle='round', zorder=3)
        outcomes.plot(row.median, position, 'o', markerfacecolor='white', markeredgecolor=INK, markeredgewidth=0.7, markersize=2.9, zorder=4)
    outcomes.set_xlim(0.25, 1.12)
    outcomes.set_ylim(-0.45, 1.45)
    outcomes.set_xticks([0.25, 0.5, 0.75, 1.0])
    outcomes.set_yticks([0, 1], ['Non-DIS', 'DIS'])
    outcomes.set_xlabel('Max dimension attention weight')
    outcomes.tick_params(axis='y', length=0, labelsize=5.0)
    _style_data_axis(outcomes, left=False)
    d_row = statistics.loc[statistics['panel_comparison'].eq('d_DIS_vs_Non-DIS')].iloc[0]
    bracket_x = 1.025
    hook = 0.025
    outcomes.plot([bracket_x - hook, bracket_x, bracket_x, bracket_x - hook], [0.0, 0.0, 1.0, 1.0], color=INK, linewidth=0.65, solid_capstyle='butt', clip_on=False, zorder=6)
    outcomes.text(1.09, 0.5, str(d_row['significance']), rotation=90, ha='center', va='center', fontsize=7.0, fontweight='bold', color=INK, clip_on=False, zorder=7)

def _draw_best_performance(figure: plt.Figure) -> None:
    task1_summary = pd.read_csv(DATA_PATHS['best_performance'])
    task1_summary = task1_summary.loc[task1_summary['phase'].eq('P1')].set_index('model')
    task2_summary = pd.read_csv(DATA_PATHS['task2_performance'])
    task2_summary['model'] = task2_summary['model'].replace({'DGEM (winner)': 'DGEM', 'fRL': 'fRL-decay'})
    task2_summary = task2_summary.set_index('model')
    order = ('Human', 'DGEM', 'Bayesian', 'fRL-decay', 'naiveRL', 'ACL')
    colors = {'Human': '#2F5FAE', 'DGEM': DGEM_PURPLE, 'Bayesian': '#F57C00', 'fRL-decay': '#F3C316', 'naiveRL': '#8B5A20', 'ACL': '#4C956C'}
    axes = (figure.add_axes((0.073, 0.075, 0.152, 0.215)), figure.add_axes((0.273, 0.075, 0.162, 0.215)))
    for model in order:
        if model not in task1_summary.index:
            continue
        row = task1_summary.loc[model]
        axes[0].errorbar(float(row['fds_ratio']), float(row['full_score_rate']), xerr=float(row['fds_ratio_sem']), yerr=float(row['full_score_rate_sem']), fmt='o', markersize=3.1, color=colors[model], markeredgewidth=0.5, elinewidth=0.65, capsize=1.6, zorder=3)
    for model in order:
        row = task2_summary.loc[model]
        xerr = float(row['DIS_rate_se']) if pd.notna(row['DIS_rate_se']) else None
        yerr = float(row['full_score_rate_se']) / 100.0 if pd.notna(row['full_score_rate_se']) else None
        axes[1].errorbar(float(row['DIS_rate_mean']), float(row['full_score_rate_mean']) / 100.0, xerr=xerr, yerr=yerr, fmt='o', markersize=3.1, color=colors[model], markeredgewidth=0.5, elinewidth=0.65, capsize=1.6, zorder=3)
    for index, (current_axis, title) in enumerate(zip(axes, ('Task 1', 'Task 2'))):
        current_axis.set_xscale('log', base=10)
        current_axis.set_xlim(0.006, 1.0)
        current_axis.set_xticks([0.01, 0.03, 0.1, 0.3, 1.0], ['0.01', '0.03', '0.1', '0.3', '1.0'])
        current_axis.xaxis.set_minor_locator(NullLocator())
        current_axis.set_ylim(0, 1.05)
        current_axis.set_yticks([0.0, 0.3, 0.6, 1.0])
        current_axis.text(0.02, 0.98, title, transform=current_axis.transAxes, ha='left', va='top', fontsize=6.0)
        _style_data_axis(current_axis)
        if index == 0:
            current_axis.set_ylabel('Full-score proportion')
        else:
            current_axis.tick_params(axis='y', left=False, labelleft=False)
            current_axis.spines['left'].set_visible(False)
    figure.text(0.254, 0.041, 'DIS ratio', ha='center', va='center', fontsize=5.5)
    handles = [Line2D([0], [0], marker='o', linestyle='none', markersize=3.0, color=colors[model], label=model) for model in order]
    axes[0].legend(handles=handles, frameon=False, loc='upper left', bbox_to_anchor=(0.02, 0.88), fontsize=5.0, borderaxespad=0, handlelength=0.8, handletextpad=0.35, labelspacing=0.22)

def _rolling(values: np.ndarray, window: int=7) -> np.ndarray:
    radius = window // 2
    return np.asarray([values[max(0, index - radius):min(values.size, index + radius + 1)].mean() for index in range(values.size)])

def _draw_dis_by_round(figure: plt.Figure) -> None:
    points = pd.read_csv(DATA_PATHS['dis_by_round'])
    temperature = pd.read_csv(DATA_PATHS['simulation_temperature'])
    order = ('Human', 'DGEM fitted simulation', 'Bayesian', 'fRL-decay fitted simulation', 'naiveRL fitted simulation')
    labels = {'Human': 'Human', 'DGEM fitted simulation': 'DGEM', 'Bayesian': 'Bayesian', 'fRL-decay fitted simulation': 'fRL-decay', 'naiveRL fitted simulation': 'naiveRL'}
    colors = {'Human': '#2F5FAE', 'DGEM fitted simulation': DGEM_PURPLE, 'Bayesian': '#F57C00', 'fRL-decay fitted simulation': '#F3C316', 'naiveRL fitted simulation': '#8B5A20'}
    axis = figure.add_axes((0.76, 0.14, 0.215, 0.15))
    temp_axis = figure.add_axes((0.76, 0.075, 0.215, 0.04), sharex=axis)
    for model in order:
        block = points.loc[points['model'].eq(model)].sort_values('round')
        if block.empty:
            raise ValueError(f'Missing DIS-by-round data for {model}')
        x = block['round'].to_numpy(dtype=float)
        raw_mean = block['dis_ratio'].to_numpy(dtype=float)
        raw_lower = block['ci_lower'].to_numpy(dtype=float)
        raw_upper = block['ci_upper'].to_numpy(dtype=float)
        mean = _rolling(raw_mean)
        low_sem = _rolling(np.maximum((raw_mean - raw_lower) / 1.96, 0.0))
        high_sem = _rolling(np.maximum((raw_upper - raw_mean) / 1.96, 0.0))
        axis.plot(x, mean, color=colors[model], linewidth=0.9, solid_capstyle='round', label=labels[model])
        axis.fill_between(x, mean - low_sem, mean + high_sem, color=colors[model], alpha=0.14, linewidth=0)
    axis.set_xlim(0.6, 50.4)
    axis.set_xticks([1, 20, 35, 50])
    axis.tick_params(axis='x', labelbottom=False)
    axis.set_ylim(0, 0.8)
    axis.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    axis.set_ylabel('DIS ratio')
    axis.legend(frameon=False, loc='upper right', bbox_to_anchor=(1.0, 0.99), ncol=2, fontsize=5.0, handlelength=1.2, handletextpad=0.3, columnspacing=0.65, labelspacing=0.2, borderaxespad=0.2)
    _style_data_axis(axis)
    tx = temperature['round'].to_numpy(dtype=float)
    ty = _rolling(temperature['temperature_norm'].to_numpy(dtype=float))
    terr = _rolling(temperature['temperature_sem_norm'].to_numpy(dtype=float))
    temp_axis.plot(tx, ty, color=DGEM_PURPLE, linewidth=0.9, solid_capstyle='round')
    temp_axis.fill_between(tx, np.clip(ty - terr, 0, 1), np.clip(ty + terr, 0, 1), color=DGEM_PURPLE, alpha=0.14, linewidth=0)
    temp_axis.set_xlim(0.6, 50.4)
    temp_axis.set_xticks([1, 20, 35, 50])
    temp_axis.set_ylim(0, 1)
    temp_axis.set_yticks([0, 1])
    temp_axis.set_ylabel('temp.', labelpad=2)
    temp_axis.set_xlabel('Round', labelpad=2)
    _style_data_axis(temp_axis)

def _draw_temperature_comparison(figure: plt.Figure, statistics: pd.DataFrame) -> None:
    points = pd.read_csv(DATA_PATHS['temperature_entities'])
    axis = figure.add_axes((0.495, 0.075, 0.215, 0.215))
    phase_position = {'P1': 0.0, 'P2': 1.0}
    source_offset = {'simulation': -0.16, 'fitting': 0.16}
    colors = {'simulation': '#2A9DB5', 'fitting': DGEM_PURPLE}
    labels = {'simulation': 'Simulation', 'fitting': 'Fitting replay'}
    group_values: dict[tuple[str, str], np.ndarray] = {}
    axis.set_yscale('log', base=10)
    for phase in ('P1', 'P2'):
        for source in ('simulation', 'fitting'):
            values = points.loc[points['phase'].eq(phase) & points['source'].eq(source), 'mean_effective_attention_temperature'].to_numpy(dtype=float)
            group_values[phase, source] = values
            center = phase_position[phase] + source_offset[source]
            log_values = np.log10(values)
            if np.ptp(log_values) > 0:
                log_grid = np.linspace(log_values.min(), log_values.max(), 256)
                density = gaussian_kde(log_values, bw_method='scott')(log_grid)
                half_width = 0.105 * density / density.max()
                axis.fill_betweenx(10 ** log_grid, center - half_width, center + half_width, facecolor=colors[source], edgecolor='none', alpha=0.24, zorder=1)
                axis.plot(center - half_width, 10 ** log_grid, color=colors[source], linewidth=0.45, alpha=0.7, zorder=2)
            axis.boxplot([values], positions=[center], widths=0.065, whis=1.5, patch_artist=True, manage_ticks=False, showfliers=False, boxprops={'facecolor': 'white', 'edgecolor': colors[source], 'linewidth': 0.7}, medianprops={'color': colors[source], 'linewidth': 0.9}, whiskerprops={'color': colors[source], 'linewidth': 0.6}, capprops={'color': colors[source], 'linewidth': 0.6}, zorder=3)
    for phase, label in (('P1', '3D'), ('P2', '4D')):
        row = statistics.loc[statistics['panel_comparison'].eq(f'f_{label}_simulation_vs_fitting')].iloc[0]
        x_left = phase_position[phase] + source_offset['simulation']
        x_right = phase_position[phase] + source_offset['fitting']
        maximum = max(float(group_values[phase, 'simulation'].max()), float(group_values[phase, 'fitting'].max()))
        log_height = min(np.log10(maximum) + 0.55, 4.1)
        bracket_y = 10 ** log_height
        lower_y = bracket_y / 1.45
        axis.plot([x_left, x_left, x_right, x_right], [lower_y, bracket_y, bracket_y, lower_y], color=INK, linewidth=0.65, solid_capstyle='butt', clip_on=False, zorder=5)
        axis.text((x_left + x_right) / 2.0, bracket_y * 1.42, str(row['significance']), ha='center', va='bottom', fontsize=7.0, fontweight='bold', color=INK, zorder=6)
    axis.set_ylim(7e-05, 30000.0)
    axis.yaxis.set_major_locator(FixedLocator([0.0001, 1.0, 10000.0]))
    axis.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    axis.yaxis.set_minor_locator(NullLocator())
    axis.set_xlim(-0.48, 1.48)
    _style_bracketed_categorical_x_axis(axis, (0.0, 1.0), ('3D', '4D'))
    axis.set_ylabel('Mean attention temperature')
    handles = [Line2D([0], [0], color=colors[source], linewidth=2.4, label=labels[source]) for source in ('simulation', 'fitting')]
    axis.legend(handles=handles, frameon=False, loc='upper left', bbox_to_anchor=(0.02, 0.98), ncol=1, fontsize=5.0, handlelength=1.2, handletextpad=0.35, labelspacing=0.25, borderaxespad=0.15)
    _style_data_axis(axis, bottom=False)

def _draw_hierarchical_selection(figure: plt.Figure) -> None:
    """Draw a minimal conceptual hierarchy for the main-text figure."""
    axis = figure.add_axes((0.025, 0.055, 0.62, 0.885))
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis('off')
    dimension_color = '#3C78A8'
    feature_color = '#3A8F62'
    score_color = '#9B7800'
    action_color = '#D05B51'
    reward_color = DGEM_PURPLE
    neutral_fill = '#F7F7F6'

    def box(x: float, y: float, width: float, height: float, label: str, *, edge: str=LIGHT_EDGE, face: str='white', text_color: str=INK, linewidth: float=0.6, fontsize: float=5.2) -> None:
        axis.add_patch(Rectangle((x, y), width, height, facecolor=face, edgecolor=edge, linewidth=linewidth, zorder=2))
        axis.text(x + width / 2, y + height / 2, label, ha='center', va='center', fontsize=fontsize, color=text_color, zorder=3)

    def arrow(start: tuple[float, float], end: tuple[float, float], *, color: str=MUTED, linewidth: float=0.6, linestyle: str='-') -> None:
        axis.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=4.8, linewidth=linewidth, linestyle=linestyle, color=color, shrinkA=0, shrinkB=0, capstyle='butt', joinstyle='miter', zorder=4))
    title_y = 0.72
    axis.text(0.06, title_y, 'User', ha='center', va='bottom', fontsize=5.1, color=MUTED)
    axis.text(0.24, title_y, '1  Dimensions\n' + 'top-$K_1$', ha='center', va='bottom', fontsize=5.1, linespacing=1.08)
    axis.text(0.455, title_y, '2  Features\n' + 'top-$K_2$', ha='center', va='bottom', fontsize=5.1, linespacing=1.08)
    axis.text(0.675, title_y, '3  Item score', ha='center', va='bottom', fontsize=5.1, color=score_color)
    axis.text(0.835, title_y, 'Action', ha='center', va='bottom', fontsize=5.1, color=action_color)
    axis.text(0.943, title_y, 'Reward', ha='center', va='bottom', fontsize=5.1, color=reward_color)
    box(0.015, 0.43, 0.09, 0.14, '$\\mathbf{u}_t$', face='#F3F5F7', fontsize=5.4)
    axis.add_patch(Rectangle((0.17, 0.35), 0.14, 0.3, facecolor='white', edgecolor=LIGHT_EDGE, linewidth=0.55, zorder=1))
    for index, y in enumerate((0.585, 0.53, 0.475, 0.42, 0.365)):
        selected = index in (1, 3)
        axis.add_patch(Rectangle((0.195, y), 0.09, 0.036, facecolor='#EAF2F8' if selected else neutral_fill, edgecolor=dimension_color if selected else LIGHT_EDGE, linewidth=0.75 if selected else 0.4, zorder=2))
    axis.add_patch(Rectangle((0.36, 0.35), 0.19, 0.3, facecolor='white', edgecolor=LIGHT_EDGE, linewidth=0.55, zorder=1))
    for row, y in enumerate((0.52, 0.405)):
        axis.add_patch(Rectangle((0.372, y - 0.012), 0.166, 0.072, fill=False, edgecolor=LIGHT_EDGE, linewidth=0.4, zorder=1))
        for column, x in enumerate((0.381, 0.42, 0.459, 0.498)):
            selected = (row, column) in {(0, 0), (0, 2), (1, 1), (1, 3)}
            axis.add_patch(Rectangle((x, y), 0.03, 0.045, facecolor='#EAF5EE' if selected else neutral_fill, edgecolor=feature_color if selected else LIGHT_EDGE, linewidth=0.75 if selected else 0.4, zorder=2))
    box(0.615, 0.44, 0.12, 0.12, '$S_t(a)$', edge='#D1B556', face='#FFFBEF', text_color=score_color, linewidth=0.75, fontsize=5.4)
    box(0.8, 0.43, 0.07, 0.14, '$a_t$', edge=action_color, face='#FDECEA', text_color=action_color, linewidth=0.8, fontsize=5.6)
    box(0.915, 0.43, 0.055, 0.14, '$r_t$', edge=reward_color, face='#F0EAF8', text_color=reward_color, linewidth=0.8, fontsize=5.6)
    arrow((0.105, 0.5), (0.17, 0.5))
    arrow((0.31, 0.5), (0.36, 0.5), color=dimension_color)
    arrow((0.55, 0.5), (0.615, 0.5), color=feature_color)
    arrow((0.735, 0.5), (0.8, 0.5), color='#B38A17')
    arrow((0.87, 0.5), (0.915, 0.5), color=action_color)
    feedback_y = 0.165
    axis.plot([0.24, 0.943], [feedback_y, feedback_y], color=reward_color, linewidth=0.58, linestyle=(0, (3, 2)), zorder=0)
    arrow((0.943, 0.43), (0.943, feedback_y), color=reward_color, linewidth=0.58, linestyle=(0, (3, 2)))
    arrow((0.24, feedback_y), (0.24, 0.35), color=reward_color, linewidth=0.58, linestyle=(0, (3, 2)))
    arrow((0.455, feedback_y), (0.455, 0.35), color=reward_color, linewidth=0.58, linestyle=(0, (3, 2)))
    axis.text(0.35, 0.07, 'Update selected dimensions and features', ha='center', va='bottom', fontsize=5.0, color=reward_color)

def _draw_tdge_reference(figure: plt.Figure) -> None:
    """Draw the user-supplied TDGE reference comparison from digitized values."""
    data = pd.read_csv(DATA_PATHS['panel_i_reference'])
    axis = figure.add_axes((0.7, 0.2, 0.27, 0.625))
    order = ('LinUCB', 'NeuralUCB', 'NeuralTS')
    seed_color = '#9A88C6'
    mean_color = DGEM_PURPLE
    baseline_color = '#777D82'
    rng = np.random.default_rng(20260928)
    for x, algorithm in enumerate(order):
        block = data.loc[data['algorithm'].eq(algorithm)].copy()
        if block.empty:
            raise ValueError(f'Missing digitized panel-i values for {algorithm}')
        baseline = float(block['baseline_mean'].iloc[0])
        mean = float(block['tdge_mean'].iloc[0])
        improvement = float(block['improvement_pct'].iloc[0])
        jitter = rng.uniform(-0.105, 0.105, len(block))
        axis.plot([x - 0.23, x + 0.23], [baseline, baseline], color=baseline_color, linewidth=1.0, linestyle=(0, (4, 3)), solid_capstyle='butt', zorder=1)
        axis.scatter(x + jitter, block['seed_reward'], s=10, marker='o', facecolor=seed_color, edgecolor='white', linewidth=0.35, alpha=0.85, zorder=2)
        axis.scatter([x], [mean], s=42, marker='D', facecolor=mean_color, edgecolor='white', linewidth=0.5, zorder=3)
        axis.text(x, min(0.892, max(float(block['seed_reward'].max()), mean) + 0.012), f'+{improvement:.1f}%', ha='center', va='bottom', fontsize=5.5, color=INK)
    axis.set_xlim(-0.5, 2.5)
    axis.set_ylim(0.74, 0.9)
    _style_bracketed_categorical_x_axis(axis, (0.0, 1.0, 2.0), ('Linear\nUCB', 'Neural\nUCB', 'Neural\nTS'))
    axis.set_yticks([0.75, 0.8, 0.85, 0.9])
    axis.set_ylabel('Average reward\nper interaction', labelpad=3.0)
    handles = [Line2D([0], [0], color=baseline_color, linewidth=1.0, linestyle=(0, (4, 3)), label='Baseline'), Line2D([0], [0], marker='o', linestyle='none', markersize=3.5, color=seed_color, label='Seeds'), Line2D([0], [0], marker='D', linestyle='none', markersize=3.8, color=mean_color, label='Mean')]
    axis.legend(handles=handles, frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.04), ncol=3, fontsize=5.0, handlelength=1.4, handletextpad=0.35, columnspacing=0.7, borderaxespad=0)
    _style_data_axis(axis, bottom=False)

def assemble(*, output_stem: str, figure_finalize) -> tuple[Path, Path]:
    """Render the established edition or a separately named styling variant."""
    if output_stem is not None and (not output_stem or Path(output_stem).name != output_stem):
        raise ValueError('output_stem must be a filename stem')
    output_path = OUTPUT_PATH if output_stem is None else OUTPUT_DIR / f'{output_stem}.png'
    manifest_path = MANIFEST_PATH if output_stem is None else MANIFEST_DIR / f'{output_stem}_manifest.json'
    edition_audit = {}
    for path in (*DATA_PATHS.values(), CAPTION_PATH, METHODS_PATH, SOURCE_PROVENANCE_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)
    statistics = _compute_panel_statistics()
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(PLOT_RC):
        figure = plt.figure(figsize=FIGURE_SIZE, dpi=DPI)
        layout = figure.add_gridspec(2, 1, height_ratios=(5.2, 1.45), hspace=0.015)
        main_figure = figure.add_subfigure(layout[0])
        extension_figure = figure.add_subfigure(layout[1])
        draw_model_flow_panel(main_figure, (0.025, 0.705, 0.95, 0.27))
        _draw_fitting_comparison(main_figure)
        _draw_attention_panels(main_figure, statistics)
        _draw_best_performance(main_figure)
        _draw_dis_by_round(main_figure)
        _draw_temperature_comparison(main_figure, statistics)
        _draw_hierarchical_selection(extension_figure)
        _draw_tdge_reference(extension_figure)
        label_positions = {'a': (0.008, 0.978), 'b': (0.008, 0.681), 'c': (0.469, 0.681), 'd': (0.733, 0.681), 'e': (0.008, 0.324), 'f': (0.469, 0.324), 'g': (0.733, 0.324)}
        for panel, (x, y) in label_positions.items():
            main_figure.text(x, y, panel, ha='left', va='top', fontsize=8, fontweight='bold', color=INK, zorder=20)
        for panel, (x, y) in {'h': (0.008, 0.98), 'i': (0.655, 0.98)}.items():
            extension_figure.text(x, y, panel, ha='left', va='top', fontsize=8, fontweight='bold', color=INK, zorder=20)
        if figure_finalize is not None:
            edition_audit = figure_finalize(figure, main_figure, extension_figure, statistics) or {}
        rendered_size_inches = list(figure.get_size_inches())
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=DPI, facecolor='white', edgecolor='none', metadata={'Software': 'Matplotlib; DGEM publication assembly'})
        plt.close(figure)
    with Image.open(output_path) as rendered:
        rendered.convert('RGB').save(output_path, dpi=(DPI, DPI))
    with Image.open(output_path) as output_image:
        output_pixels = list(output_image.size)
        output_dpi = [float(value) for value in output_image.info.get('dpi', (DPI, DPI))]
        output_mode = output_image.mode
    manifest = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'figure': {'path': str(output_path.resolve()), 'sha256': _sha256(output_path), 'format': 'PNG only', 'size_inches': rendered_size_inches, 'pixels': output_pixels, 'dpi': output_dpi, 'color_mode': output_mode, 'font_family': 'DejaVu Sans', 'axis_and_tick_style': 'All panels are re-rendered from committed source tables with 5-7 pt sans-serif text, 0.55-0.70 pt axes, regular labels, and bold 8 pt panel letters'}, 'edition_audit': edition_audit, 'panel_mapping': PANEL_MAPPING, 'documentation': {'caption': {'path': str(CAPTION_PATH.resolve()), 'sha256': _sha256(CAPTION_PATH)}, 'panel_h_methods': {'path': str(METHODS_PATH.resolve()), 'sha256': _sha256(METHODS_PATH)}}, 'vector_panel_a': {'bounds': [0.025, 0.705, 0.95, 0.27], 'primary_color': DGEM_PURPLE, 'line_width_pt': '0.70-0.90', 'structure': 'Fixed reference layout with nine sharp rectangular nodes, two circular operators, and orthogonal attention routes'}, 'source_provenance': {'path': str(SOURCE_PROVENANCE_PATH.resolve()), 'sha256': _sha256(SOURCE_PROVENANCE_PATH)}, 'data_sources': {key: {'path': str(path.resolve()), 'sha256': _sha256(path)} for key, path in DATA_PATHS.items()}, 'panel_statistics': {'path': str(STATISTICS_PATH.resolve()), 'sha256': _sha256(STATISTICS_PATH), 'comparisons': int(len(statistics)), 'methods': ['paired two-sided Wilcoxon for panel d', 'two-sided Mann-Whitney U for panel f']}, 'notes': ['Only the packaged plot-facing CSV files are required to redraw this figure.', 'Panels b-g are re-rendered from the packaged plotting values with text sizes specified in points; final width is recorded in figure.size_inches.', 'Panel f shows log-domain violin densities, box summaries, and significance brackets.', 'Panel h is a vector redraw of the user-supplied hierarchical-selection reference diagram.', 'Panel i values were digitized from the user-supplied reference image because no source table or plotting code was found; they are not newly simulated mini_DG results.', 'No PDF is generated.']}
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return (output_path, manifest_path)
