"""Panel layout and statistics for the retained final Fig. 5. No legacy exports."""
from __future__ import annotations
from datetime import datetime, timezone
from hashlib import sha256
import base64
import io
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_rgba
from matplotlib.patches import Rectangle
from matplotlib.ticker import StrMethodFormatter
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import mannwhitneyu
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'Data mining/vis_result/results/DGEM/fig/publication/main_figure/fig5/code'))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import fig5_plot_helpers as original
from draw_data_source_icons import add_source_icon
RESULTS_ROOT = original.RESULTS_ROOT
FIG5A_ROOT = PROJECT_ROOT / 'assets/gaze'
FIG5A_FIGURES = FIG5A_ROOT
FIG5A_CSV = PROJECT_ROOT / 'data/sampled_trials_fixge200_fdsD_vs_nonfds_seed234.csv'
DIS_OVERLAY = FIG5A_FIGURES / 'fds_d_fixation_density_fixge200_area1_thr02.png'
NONDIS_OVERLAY = FIG5A_FIGURES / 'nonfds_randommatched_to_fds_d_fixation_density_fixge200_area1_seed234_thr02.png'
GAZE_BACKGROUND = FIG5A_ROOT / 'background.png'
OUTPUT_PATH = PROJECT_ROOT / 'output/fig5_nhb_evidence_layout.png'
DATA_DIR = PROJECT_ROOT / 'output'
STATISTICS_PATH = DATA_DIR / 'fig5_nhb_refined_typography_statistics.csv'
MANIFEST_PATH = DATA_DIR / 'fig5_nhb_refined_typography_manifest.json'
CAPTION_PATH = PROJECT_ROOT / 'caption.md'
METHODS_PATH = PROJECT_ROOT / 'methods.md'
G_DISTRIBUTION_PATH = DATA_DIR / 'fig5g_distribution_counts.csv'
FIGURE_SIZE = (180.0 / 25.4, 141.0 / 25.4)
COLUMN_LEFTS = (0.095, 0.3383333333333333, 0.5816666666666667, 0.825)
COLUMN_WIDTHS = (0.155, 0.155, 0.155, 0.155)
PLOT_HEIGHT = 33.3 / 141.0
ROW_BOTTOMS = (0.735, 0.393, 0.064)
ROW_LETTER_TOPS = (0.985, 0.65, 0.342)
LETTER_LEFTS = tuple((left - 0.075 for left in COLUMN_LEFTS))
HEATMAP_LEFT_SHIFT_MM = 2.5
HEATMAP_COLOR_MAX = 0.3
GAZE_IMAGE_EXTRA_WIDTH_MM = 5.0
OTHER_IMAGE_LEFT_SHIFT_MM = 2.0
COLORBAR_WIDTH = 0.008
STAGE_MEAN_MARKER_SIZE = 5.2
PURPLE = original.PURPLE
INK = original.INK
OTHERS_GREY = '#8D9399'
GAZE_CMAP = 'RdBu_r'
TRIAL_LABELS = ('DIS', 'others')
OBSERVATION_LABELS = ('Obs', 'NObs')
SOURCE_ICON_MAPPING = {**{letter: 'eye_tracking' for letter in 'abcdefh'}, 'g': 'task_behavior', 'i': 'survey', 'j': 'survey'}
PANEL_MAPPING = {**original.PANEL_MAPPING, 'a': 'Fixation density on plant-dimension DIS trials and trial-count-matched other trials', 'c': 'Gaze dwell-time proportion by DIS dimension and viewed task dimension', 'j': 'Within-participant change in score-relevant dimensions, Final minus Initial'}

def _panel_bounds(row: int, column: int) -> tuple[float, float, float, float]:
    return (COLUMN_LEFTS[column], ROW_BOTTOMS[row], COLUMN_WIDTHS[column], PLOT_HEIGHT)

def _annotate_comparison(axis: plt.Axes, p_value: float) -> None:
    low, high = axis.get_ylim()
    span = high - low
    y = high - 0.052 * span
    axis.plot([0, 0, 1, 1], [y - 0.022 * span, y, y, y - 0.022 * span], color=INK, linewidth=0.65, clip_on=False, solid_capstyle='butt')
    axis.text(0.5, y + 0.012 * span, original._stars(p_value), ha='center', va='bottom', fontsize=6.5, fontweight='bold', color=INK)

def _detached_category_axis(axis: plt.Axes) -> None:
    """Short horizontal axis joining category centres, with downward end ticks."""
    original._style_axis(axis, categorical=False)
    axis.spines['bottom'].set_visible(True)
    axis.spines['bottom'].set_bounds(0, 1)
    axis.spines['bottom'].set_position(('outward', 3))
    axis.tick_params(axis='x', direction='out', length=2.3, pad=2.5)
    axis.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

def _axis_frame_and_text_bbox(axis: plt.Axes, renderer) -> Bbox:
    """Measure visible layout content without empty boxplot-flier artifacts."""
    bounds = [axis.get_window_extent(renderer)]
    texts = [axis.xaxis.label, axis.yaxis.label, *axis.get_xticklabels(), *axis.get_yticklabels(), *axis.texts]
    bounds.extend((text.get_window_extent(renderer) for text in texts if text.get_visible() and text.get_text().strip()))
    legend = axis.get_legend()
    if legend is not None and legend.get_visible():
        bounds.append(legend.get_window_extent(renderer))
    return Bbox.union(bounds)

def _balance_bottom_row(figure: plt.Figure, panel_axes: dict) -> dict:
    """Use the same four columns and widths as the gaze-sampling row."""
    for column, letter in enumerate('ghij'):
        panel_axes[letter].set_position(_panel_bounds(2, column))
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    contents = [_axis_frame_and_text_bbox(panel_axes[k], renderer) for k in 'ghij']
    gaps = [(b.x0 - a.x1) * 25.4 / figure.dpi for a, b in zip(contents, contents[1:])]
    if min(gaps) < 1.5:
        raise ValueError(f'Insufficient bottom-row label clearance: {gaps}')
    return {'clear_gaps_mm': gaps, 'plot_widths_mm': [27.9] * 4, 'note': 'Bottom plot rectangles align exactly with the four middle-row columns.'}

def _align_bottom_annotations(figure: plt.Figure, panel_axes: dict) -> dict:
    """Give equal-height plots a shared annotation band above their frames."""
    stars = {}
    for letter in 'ghi':
        axis = panel_axes[letter]
        brackets = [line for line in axis.lines if len(line.get_xdata()) == 4 and np.allclose(line.get_xdata(), [0, 0, 1, 1])]
        labels = [text for text in axis.texts if text.get_text() in ('*', '**', '***', 'ns')]
        if len(brackets) != 1 or len(labels) != 1:
            raise ValueError(f'Expected one significance bracket and label in panel {letter}')
        brackets[0].set_transform(axis.get_xaxis_transform())
        brackets[0].set_ydata([1.045, 1.07, 1.07, 1.045])
        labels[0].set_transform(axis.get_xaxis_transform())
        labels[0].set_position((0.5, 1.088))
        labels[0].set_clip_on(False)
        stars[letter] = labels[0]
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    top = stars['h'].get_window_extent(renderer).y1
    axis = panel_axes['j']
    legend = axis.get_legend()
    current_anchor = axis.transAxes.inverted().transform(legend.get_bbox_to_anchor().p0)
    delta = (top - legend.get_window_extent(renderer).y1) / axis.get_window_extent(renderer).height
    legend.set_bbox_to_anchor((0.5, current_anchor[1] + delta))
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    mm_per_pixel = 25.4 / figure.dpi
    frames = {letter: panel_axes[letter].get_window_extent(renderer) for letter in 'ghij'}
    heights = {letter: frame.height * mm_per_pixel for letter, frame in frames.items()}
    np.testing.assert_allclose(list(heights.values()), 33.3, atol=0.01)
    np.testing.assert_allclose([frame.y0 for frame in frames.values()], frames['h'].y0, atol=0.1)
    np.testing.assert_allclose([frame.y1 for frame in frames.values()], frames['h'].y1, atol=0.1)
    annotation_tops = {letter: text.get_window_extent(renderer).y1 * mm_per_pixel for letter, text in stars.items()}
    annotation_tops['j'] = legend.get_window_extent(renderer).y1 * mm_per_pixel
    np.testing.assert_allclose(list(annotation_tops.values()), top * mm_per_pixel, atol=0.01)
    return {'plot_heights_mm': heights, 'annotation_tops_mm_from_canvas_bottom': annotation_tops, 'significance_bracket_height_axes': 1.07, 'note': 'Common plot tops/baselines; g–i brackets and j legend share a reserved annotation band. Data limits unchanged.'}

def _mark_truncated_axis(axis: plt.Axes) -> None:
    """Explicitly mark the retained nonzero lower bounds on b and f."""
    centers = (0.018, 0.045)
    if np.any(np.isclose(axis.get_yticks(), axis.get_ylim()[0])):
        centers = (-0.065, -0.037)
        axis.plot([0, 0], [-0.09, 0], transform=axis.transAxes, color=INK, linewidth=axis.spines['left'].get_linewidth(), clip_on=False, solid_capstyle='butt', zorder=3)
    for center in centers:
        for color, width in (('white', 2.4), (INK, 0.7)):
            axis.plot([-0.022, 0.022], [center - 0.01, center + 0.01], transform=axis.transAxes, color=color, linewidth=width, clip_on=False, solid_capstyle='butt', zorder=10)

def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()

def _holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm's step-down familywise correction, in the input order."""
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted_sorted = np.maximum.accumulate((len(values) - np.arange(len(values))) * values[order])
    result = np.empty_like(values)
    result[order] = np.minimum(adjusted_sorted, 1.0)
    return result

def _bootstrap_mean_ci(left: np.ndarray, right: np.ndarray, *, paired: bool, seed: int, n_boot: int=10000) -> tuple[float, float]:
    """Percentile CI for right minus left, resampling independent subjects."""
    rng = np.random.default_rng(seed)
    if paired:
        delta = np.asarray(right, dtype=float) - np.asarray(left, dtype=float)
        draws = delta[rng.integers(0, len(delta), size=(n_boot, len(delta)))].mean(axis=1)
    else:
        left = np.asarray(left, dtype=float)
        right = np.asarray(right, dtype=float)
        left_draws = left[rng.integers(0, len(left), size=(n_boot, len(left)))].mean(axis=1)
        right_draws = right[rng.integers(0, len(right), size=(n_boot, len(right)))].mean(axis=1)
        draws = right_draws - left_draws
    return tuple((float(x) for x in np.quantile(draws, [0.025, 0.975])))

def _paired_values(panel_id: str, values: pd.DataFrame) -> pd.DataFrame:
    block = values.loc[values['panel_id'].eq(panel_id)].copy()
    paired = block.pivot(index='subject_id', columns='group_label', values='value')
    return paired[['DIS', 'NDIS']].dropna().sort_index()

def _recolor_gaze_overlay(image_path: Path) -> tuple[np.ndarray, dict]:
    """Remap the existing raster's display colours, not the underlying gaze data.

    The saved SVG has a composite RGBA image and the package supplies its exact
    background. Fit its displayed RGB to a jet/background blend, then replace
    only that palette coordinate, retaining its estimated opacity. This is an
    approximate display conversion; its coordinates are never used as density
    measurements or statistical inputs.
    """
    svg_path = image_path.with_suffix('.svg')
    elements = [element for element in ET.parse(svg_path).iter() if element.tag.endswith('image')]
    if len(elements) != 1:
        raise ValueError(f'Expected one composite image in {svg_path}')
    encoded = elements[0].attrib['{http://www.w3.org/1999/xlink}href'].split(',', 1)[1]
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as embedded:
        rgba = np.asarray(embedded.convert('RGBA'))[::-1].copy()
    observed = rgba[:, :, :3].astype(float) / 255
    with Image.open(GAZE_BACKGROUND) as background:
        backdrop = np.asarray(background.convert('RGB').resize((rgba.shape[1], rgba.shape[0]), Image.Resampling.BICUBIC)).astype(float) / 255
    mask = (np.max(np.abs(observed - backdrop), axis=2) > 6 / 255) & (rgba[:, :, 3] == 255)
    levels = np.linspace(0, 1, 256)
    old_colors = plt.get_cmap('jet')(levels)[:, :3]
    new_colors = plt.get_cmap(GAZE_CMAP)(levels)[:, :3]
    foreground, backgrounds = (observed[mask], backdrop[mask])
    replacements, residuals = ([], [])
    for start in range(0, len(foreground), 2048):
        bg = backgrounds[start:start + 2048]
        delta = foreground[start:start + 2048] - bg
        direction = old_colors[None, :, :] - bg[:, None, :]
        opacity = np.clip(np.einsum('nij,nj->ni', direction, delta) / np.sum(direction ** 2, axis=2), 0, 1)
        residual = np.sum((direction * opacity[:, :, None] - delta[:, None, :]) ** 2, axis=2)
        indices = np.argmin(residual, axis=1)
        rows = np.arange(len(indices))
        alpha = opacity[rows, indices, None]
        residuals.extend(np.sqrt(residual[rows, indices]) * 255)
        replacements.append(bg * (1 - alpha) + new_colors[indices] * alpha)
    quantiles = np.quantile(residuals, [0.5, 0.95, 0.99])
    if quantiles[-1] > 6:
        raise ValueError(f'Source colour reconstruction is too inaccurate: {quantiles}')
    recolored = rgba.copy()
    recolored[mask, :3] = np.round(np.clip(np.concatenate(replacements), 0, 1) * 255).astype(np.uint8)
    if not np.array_equal(recolored[~mask], rgba[~mask]):
        raise AssertionError('Palette conversion modified the background')
    return (recolored, {'source_svg': str(svg_path), 'recolored_pixels': int(mask.sum()), 'source_rgb_reconstruction_error_p50_p95_p99': quantiles.tolist(), 'background_unchanged': True, 'note': 'Approximate palette-only conversion of the saved composite; no raw density was recovered or analyzed.'})

def _draw_gaze_overlay(axis: plt.Axes, image_path: Path) -> dict:
    recolored, audit = _recolor_gaze_overlay(image_path)
    axis.imshow(recolored, interpolation='none', aspect='equal')
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_edgecolor('#B8BDC3')
        spine.set_linewidth(0.55)
    return audit

def _box_points(axis: plt.Axes, groups: tuple[np.ndarray, np.ndarray], *, labels: tuple[str, str], ylabel: str, ylim: tuple[float, float], yticks: list[float], p_value: float, paired: bool, seed: int, zero_line: bool=False, annotate_significance: bool=True, box_width: float=0.33) -> None:
    rng = np.random.default_rng(seed)
    colors = (PURPLE, OTHERS_GREY) if paired else (original.ORANGE, original.YELLOW)
    boxes = axis.boxplot(groups, positions=[0, 1], widths=box_width, patch_artist=True, showfliers=not paired, manage_ticks=False, boxprops={'linewidth': 0.6, 'edgecolor': INK}, medianprops={'linewidth': 0.95, 'color': INK}, whiskerprops={'linewidth': 0.55, 'color': INK}, capprops={'linewidth': 0.55, 'color': INK}, flierprops={'marker': 'o', 'markersize': 2.0, 'markeredgewidth': 0.4, 'markeredgecolor': INK, 'markerfacecolor': 'white', 'alpha': 0.65, 'clip_on': False})
    for box, color in zip(boxes['boxes'], colors):
        box.set_facecolor(color)
        box.set_alpha(0.3 if paired else 0.58)
    if paired:
        if len(groups[0]) != len(groups[1]):
            raise ValueError('Paired panel must have the same number of values in both groups')
        jitter = rng.uniform(-0.09, 0.09, len(groups[0]))
        for offset, left, right in zip(jitter, groups[0], groups[1]):
            axis.plot([offset, 1 + offset], [left, right], color='#899198', linewidth=0.3, alpha=0.065, zorder=1)
        for x, group, color in zip((0, 1), groups, colors):
            axis.scatter(x + jitter, group, s=3.7, color=color, alpha=0.32, linewidths=0, zorder=3)
    axis.plot([0, 1], [float(np.mean(group)) for group in groups], color=INK, linewidth=1.05, linestyle='-' if paired else 'none', marker='D', markersize=3.0, markeredgecolor='white', markeredgewidth=0.45, zorder=5)
    if zero_line:
        axis.axhline(0, color='#AAB1B7', linewidth=0.55, linestyle=(0, (2.2, 2.2)), zorder=0)
    axis.set_xlim(-0.43, 1.43)
    axis.set_xticks([0, 1], labels)
    axis.set_ylim(*ylim)
    axis.set_yticks(yticks)
    axis.set_ylabel(ylabel, labelpad=2.5)
    _detached_category_axis(axis)
    if ylim[0] > 0:
        _mark_truncated_axis(axis)
    if annotate_significance:
        _annotate_comparison(axis, p_value)

def _draw_heatmap(figure: plt.Figure, matrix_long: pd.DataFrame, adjusted_p: np.ndarray, *, dis_on_x: bool=True) -> plt.Axes:
    left, bottom, width, height = (0.71, 0.944 - PLOT_HEIGHT, 0.185, PLOT_HEIGHT)
    bounds = (left, bottom, width, height)
    axis = figure.add_axes(bounds)
    color_axis = figure.add_axes((bounds[0] + bounds[2] + 0.009, bounds[1], COLORBAR_WIDTH, bounds[3]), label='fig5c_colorbar')
    dimensions = ['A', 'B', 'C', 'D', 'E']
    classes = ['sofa', 'table', 'carpet', 'potted_plant', 'painting']
    class_labels = ['Sofa', 'Table', 'Carpet', 'Plant', 'Painting']
    matrix = matrix_long.pivot(index='fds_dimension', columns='fixated_class', values='mean_value').reindex(index=dimensions, columns=classes).to_numpy(dtype=float)
    matrix = matrix.T
    if np.nanmax(matrix) > HEATMAP_COLOR_MAX:
        raise ValueError('Heatmap values exceed the declared 0–0.3 display range')
    image = axis.imshow(matrix, cmap=GAZE_CMAP, vmin=0, vmax=HEATMAP_COLOR_MAX, aspect='equal')
    axis.set_xticks(range(5), class_labels, rotation=35, ha='right', rotation_mode='anchor')
    axis.set_yticks(range(5), class_labels)
    axis.tick_params(width=0.55, length=2, pad=1.5, labelsize=5.0)
    axis.set_xlabel('DIS dimension', labelpad=2)
    axis.set_ylabel('Fixated dimension', labelpad=2.5)
    for spine in axis.spines.values():
        spine.set_linewidth(0.6)
    for index, p_value in enumerate(adjusted_p):
        axis.add_patch(Rectangle((index - 0.5, index - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=0.65))
        rgba = image.cmap(image.norm(matrix[index, index]))
        luminance = sum((weight * channel for weight, channel in zip((0.2126, 0.7152, 0.0722), rgba[:3])))
        axis.text(index, index, original._stars(float(p_value)), ha='center', va='center', fontsize=6, fontweight='bold', color='white' if luminance < 0.48 else INK)
    bar = figure.colorbar(image, cax=color_axis)
    ticks = [0, 0.1, 0.2, 0.3]
    bar.set_ticks(ticks)
    np.testing.assert_allclose(bar.ax.get_ylim(), image.get_clim())
    bar.ax.set_yticklabels([f'{value:.1f}' for value in ticks])
    bar.ax.tick_params(width=0.45, length=1.8, labelsize=5.0, pad=1.5)
    bar.outline.set_linewidth(0.45)
    bar.ax.set_title('Dwell-time\nproportion', fontsize=5.0, pad=3.0, loc='right')
    return axis

def _timecourse_change(timecourse: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    block = timecourse.loc[timecourse['plot_id'].eq('03_obs_vs_nobs_score_relevance')]
    pivot = block.pivot(index=['subject_id', 'group_label'], columns='stage_label', values='value')
    change = (pivot['Final'] - pivot['Initial']).dropna()
    obs = change.xs('Obs', level='group_label').to_numpy(dtype=float)
    nobs = change.xs('NObs', level='group_label').to_numpy(dtype=float)
    p_value = float(mannwhitneyu(obs, nobs, alternative='two-sided').pvalue)
    return (obs, nobs, p_value)

def _statistics(metadata: pd.DataFrame, values: pd.DataFrame, exploration: pd.DataFrame, heatmap_stats: pd.DataFrame, timecourse: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, float]:
    stats = original._compute_statistics(metadata, values, exploration, heatmap_stats).copy()
    stats['p_value_displayed'] = stats['p_value']
    stats['display_significance'] = stats['significance']
    stats['effect_mean_right_minus_left'] = np.nan
    stats['effect_ci95_low'] = np.nan
    stats['effect_ci95_high'] = np.nan
    stats['diagonal_minus_other_mean'] = np.nan
    stats['diagonal_minus_other_ci95_low'] = np.nan
    stats['diagonal_minus_other_ci95_high'] = np.nan
    heatmap_rows = stats.loc[stats['panel'].str.startswith('c-')].sort_values('panel')
    adjusted_p = _holm_adjust(heatmap_rows['p_value'].to_numpy(dtype=float))
    for panel_name, adjusted in zip(heatmap_rows['panel'], adjusted_p):
        idx = stats.index[stats['panel'].eq(panel_name)][0]
        stats.loc[idx, 'p_value_displayed'] = adjusted
        stats.loc[idx, 'display_significance'] = original._stars(adjusted)
        stats.loc[idx, 'p_adjustment'] = 'Holm across five diagonal dimensions'
        source = heatmap_stats.loc[heatmap_stats['metric_key'].eq(panel_name[-1])].iloc[0]
        stats.loc[idx, ['left_mean', 'right_mean']] = np.nan
        stats.loc[idx, 'diagonal_minus_other_mean'] = float(source['mean_diff'])
        stats.loc[idx, 'diagonal_minus_other_ci95_low'] = float(source['ci95_low'])
        stats.loc[idx, 'diagonal_minus_other_ci95_high'] = float(source['ci95_high'])
    for n, (panel, panel_id) in enumerate((('b', '02_dis_vs_ndis_19'), ('d', '02_dis_vs_ndis_08'), ('e', '02_dis_vs_ndis_03'), ('f', '02_dis_vs_ndis_01')), start=1):
        paired = _paired_values(panel_id, values)
        left, right = (paired['DIS'].to_numpy(), paired['NDIS'].to_numpy())
        low, high = _bootstrap_mean_ci(left, right, paired=True, seed=100 + n)
        idx = stats.index[stats['panel'].eq(panel)][0]
        stats.loc[idx, 'effect_mean_right_minus_left'] = float(np.mean(right - left))
        stats.loc[idx, 'effect_ci95_low'] = low
        stats.loc[idx, 'effect_ci95_high'] = high
    obs_g = exploration.loc[exploration['game_mode'].eq('with_obs'), 'n_dims_with_k_feats'].to_numpy(dtype=float)
    nobs_g = exploration.loc[exploration['game_mode'].eq('without_obs'), 'n_dims_with_k_feats'].to_numpy(dtype=float)
    for n, (panel, left, right) in enumerate((('g', obs_g, nobs_g), ('h', *original._direct_inputs('03_obs_vs_nobs_01', metadata, values)[1]), ('i', *original._direct_inputs('03_obs_vs_nobs_02', metadata, values)[1])), start=1):
        low, high = _bootstrap_mean_ci(left, right, paired=False, seed=200 + n)
        idx = stats.index[stats['panel'].eq(panel)][0]
        stats.loc[idx, 'effect_mean_right_minus_left'] = float(np.mean(right) - np.mean(left))
        stats.loc[idx, 'effect_ci95_low'] = low
        stats.loc[idx, 'effect_ci95_high'] = high
    obs_change, nobs_change, j_p = _timecourse_change(timecourse)
    low, high = _bootstrap_mean_ci(obs_change, nobs_change, paired=False, seed=301)
    stats = pd.concat([stats, pd.DataFrame([{'panel': 'j', 'comparison': 'Obs vs NObs within-participant change (Final − Initial)', 'n_left': len(obs_change), 'n_right': len(nobs_change), 'left_mean': float(np.mean(obs_change)), 'right_mean': float(np.mean(nobs_change)), 'p_value': j_p, 'p_value_displayed': j_p, 'significance': original._stars(j_p), 'display_significance': original._stars(j_p), 'test': 'two-sided Mann–Whitney U on subject-level changes; exploratory', 'p_adjustment': 'none; exploratory post-hoc comparison', 'source': 'timecourse_subject_values.csv', 'effect_mean_right_minus_left': float(np.mean(nobs_change) - np.mean(obs_change)), 'effect_ci95_low': low, 'effect_ci95_high': high}])], ignore_index=True)
    return (stats.sort_values('panel').reset_index(drop=True), adjusted_p, j_p)

def assemble(*, output_stem: str, figure_finalize, icon_size_mm: float=4.8) -> tuple[Path, Path, Path]:
    """Render the retained arrangement with the final publication style."""
    bottom_panels = 'boxes_timecourse'
    dis_on_x = True
    if output_stem is not None and (Path(output_stem).name != output_stem or not output_stem):
        raise ValueError('output_stem must be a filename stem')
    output_path = OUTPUT_PATH if output_stem is None else OUTPUT_PATH.with_name(f'{output_stem}.png')
    statistics_path = STATISTICS_PATH if output_stem is None else DATA_DIR / f'{output_stem}_statistics.csv'
    manifest_path = MANIFEST_PATH if output_stem is None else DATA_DIR / f'{output_stem}_manifest.json'
    g_distribution_path = G_DISTRIBUTION_PATH if output_stem is None else DATA_DIR / f'{output_stem}_g_counts.csv'
    edition_audit = {}
    active_stem = output_stem or OUTPUT_PATH.stem
    trajectory_letter, exploration_letter = ('j', 'i')
    timecourse_summary_path = DATA_DIR / f'{active_stem}_{trajectory_letter}_timecourse_summary.csv'
    box_summary_path = DATA_DIR / f'{active_stem}_{exploration_letter}_box_summary.csv'
    source_icon_mapping = dict(SOURCE_ICON_MAPPING)
    panel_mapping = dict(PANEL_MAPPING)
    source_icon_mapping.update(g='survey', j='task_behavior')
    panel_mapping.update(g='Early-stage and final-stage self-reported score-relevant dimensions, mean ± SEM', j='Explored dimensions with and without prior observation; boxplots')
    g_distribution_path = DATA_DIR / f'{active_stem}_{exploration_letter}_counts.csv'
    sources = {**original.DATA_PATHS, 'fig5a_dis_overlay': DIS_OVERLAY, 'fig5a_nondis_overlay': NONDIS_OVERLAY, 'fig5a_sampled_trials': FIG5A_CSV, 'fig5a_dis_svg': DIS_OVERLAY.with_suffix('.svg'), 'fig5a_others_svg': NONDIS_OVERLAY.with_suffix('.svg'), 'fig5a_background': GAZE_BACKGROUND}
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = pd.read_csv(original.DATA_PATHS['direct_metadata'])
    values = pd.read_csv(original.DATA_PATHS['direct_values'])
    heatmap = pd.read_csv(original.DATA_PATHS['heatmap_matrix'])
    heatmap_stats = pd.read_csv(original.DATA_PATHS['heatmap_stats'])
    timecourse = pd.read_csv(original.DATA_PATHS['timecourse_values'])
    exploration = pd.read_csv(original.DATA_PATHS['obs_nobs_exploration'])
    sampled = pd.read_csv(FIG5A_CSV)
    sample_summary = {group: {'trials': int(len(block)), 'participants': int(block['subject_id'].nunique())} for group, block in sampled.groupby('condition')}
    if sample_summary.get('FDS', {}).get('trials') != sample_summary.get('nonFDS', {}).get('trials'):
        raise ValueError('Fig. 5a overlay pair does not have matched trial counts')
    stats, adjusted_p, j_p = _statistics(metadata, values, exploration, heatmap_stats, timecourse)
    stats = stats.loc[~stats['panel'].eq('j')].copy()
    stats.loc[stats['panel'].eq('g'), 'panel'] = 'j'
    block = timecourse.loc[timecourse['plot_id'].eq('03_obs_vs_nobs_score_relevance')]
    timecourse_summary = block.groupby(['group_label', 'stage_order', 'stage_label'])['value'].agg(n='count', mean='mean', sem='sem').reset_index()
    stage_rows = []
    for stage in ('Initial', 'Final'):
        summary = timecourse_summary.loc[timecourse_summary['stage_label'].eq(stage)].set_index('group_label')
        stage_rows.append({'panel': f'g-{stage.lower()}', 'comparison': f'Obs / NObs at {stage}', 'n_left': summary.loc['Obs', 'n'], 'n_right': summary.loc['NObs', 'n'], 'left_mean': summary.loc['Obs', 'mean'], 'right_mean': summary.loc['NObs', 'mean'], 'left_sem': summary.loc['Obs', 'sem'], 'right_sem': summary.loc['NObs', 'sem'], 'test': 'Descriptive mean ± SEM; no significance annotation', 'source': 'timecourse_subject_values.csv'})
    stats = pd.concat([stats, pd.DataFrame(stage_rows)], ignore_index=True).sort_values('panel').reset_index(drop=True)
    with plt.rc_context(original.PLOT_RC):
        figure = plt.figure(figsize=FIGURE_SIZE, dpi=original.DPI)
        panel_axes = {}
        base_image_width = PLOT_HEIGHT * FIGURE_SIZE[1] / FIGURE_SIZE[0] * (770 / 577)
        extra_width = GAZE_IMAGE_EXTRA_WIDTH_MM / (FIGURE_SIZE[0] * 25.4)
        image_width = 33.3 / 180.0 * (770 / 577)
        image_height = image_width * FIGURE_SIZE[0] / FIGURE_SIZE[1] * (577 / 770)
        image_top = 0.944
        image_bottom = image_top - image_height
        image_lefts = (0.08, 0.345)
        image_bounds = [(left, image_bottom, image_width, image_height) for left in image_lefts]
        overlay_audit = {'DIS': _draw_gaze_overlay(figure.add_axes(image_bounds[0]), DIS_OVERLAY), 'others': _draw_gaze_overlay(figure.add_axes(image_bounds[1]), NONDIS_OVERLAY)}
        gaze_bar_height = PLOT_HEIGHT
        gaze_bar_bounds = (image_lefts[1] + image_width + 0.009, image_bottom + (image_height - gaze_bar_height) / 2, COLORBAR_WIDTH, gaze_bar_height)
        gaze_color_axis = figure.add_axes(gaze_bar_bounds, label='fig5a_shared_colorbar')
        gaze_bar = figure.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=GAZE_CMAP), cax=gaze_color_axis)
        gaze_bar.set_ticks([0, 1], labels=['Low', 'High'])
        gaze_bar.ax.tick_params(width=0.45, length=1.8, pad=1.5, labelsize=5.0)
        gaze_bar.outline.set_linewidth(0.45)
        gaze_bar.set_label('Fixation density', fontsize=5.0, fontweight='normal', rotation=270, labelpad=0, va='center')
        gaze_color_axis.yaxis.set_label_coords(2.3, 0.5)
        image_titles = ('DIS trials', 'others')
        image_title_colors = (PURPLE, OTHERS_GREY)
        title_gap_pt = 3.0
        title_y = image_top + title_gap_pt / (72 * FIGURE_SIZE[1])
        for left, label, color in zip(image_lefts, image_titles, image_title_colors):
            figure.text(left + image_width / 2, title_y, label, ha='center', va='bottom', fontsize=6.0, fontweight='normal', color=color)
        paired_specs = (('b', '02_dis_vs_ndis_19', _panel_bounds(1, 0), 'Gaze coverage area (deg²)', (100, 620), [150, 300, 450, 600]), ('d', '02_dis_vs_ndis_08', _panel_bounds(1, 1), 'Mean dwell time\nper fixated image (ms)', (0, 4500), [0, 1500, 3000, 4500]), ('e', '02_dis_vs_ndis_03', _panel_bounds(1, 2), 'Gaze-shift speed (deg s⁻¹)', (0, 600), [0, 200, 400, 600]), ('f', '02_dis_vs_ndis_01', _panel_bounds(1, 3), 'Task dimensions\nfixated (count)', (2, 5.5), [2, 3, 4, 5]))
        stats_indexed = stats.set_index('panel')
        for n, (letter, panel_id, bounds, ylabel, ylim, yticks) in enumerate(paired_specs):
            paired = _paired_values(panel_id, values)
            panel_axes[letter] = figure.add_axes(bounds)
            _box_points(panel_axes[letter], (paired['DIS'].to_numpy(), paired['NDIS'].to_numpy()), labels=TRIAL_LABELS, ylabel=ylabel, ylim=ylim, yticks=yticks, p_value=float(stats_indexed.loc[letter, 'p_value_displayed']), paired=True, seed=600 + n)
        panel_axes['c'] = _draw_heatmap(figure, heatmap, adjusted_p, dis_on_x=dis_on_x)
        obs = exploration.loc[exploration['game_mode'].eq('with_obs'), 'n_dims_with_k_feats'].to_numpy(dtype=float)
        nobs = exploration.loc[exploration['game_mode'].eq('without_obs'), 'n_dims_with_k_feats'].to_numpy(dtype=float)
        g_bounds = list(_panel_bounds(2, 0))
        panel_axes['g'] = figure.add_axes(g_bounds)
        original._draw_timecourse(panel_axes['g'], timecourse)
        axis = panel_axes['g']
        for line in list(axis.lines):
            if len(line.get_xdata()) == 4 and np.allclose(line.get_xdata(), [0, 0, 1, 1]):
                line.remove()
        _detached_category_axis(axis)
        axis.yaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
        axis.set_ylabel('Reported score-relevant\ndimensions', labelpad=2.5)
        axis.set_xticks([0, 1], ['Early stage', 'Final stage'])
        for errorbars in axis.containers:
            errorbars.lines[0].set_markersize(STAGE_MEAN_MARKER_SIZE)
        for label in axis.get_legend().get_texts():
            label.set_fontsize(5)
        independent_specs = (('h', tuple(original._direct_inputs('03_obs_vs_nobs_01', metadata, values)[1]), _panel_bounds(2, 1), 'Dimensions fixated\nbefore the task (count)', (0, 8), [0, 2, 4, 6, 8]), ('i', tuple(original._direct_inputs('03_obs_vs_nobs_02', metadata, values)[1]), _panel_bounds(2, 2), 'Initially noticed\ndimensions (self-report)', (0, 20), [0, 6, 12, 18]))
        for n, (letter, groups, bounds, ylabel, ylim, yticks) in enumerate(independent_specs):
            panel_axes[letter] = figure.add_axes(bounds)
            _box_points(panel_axes[letter], groups, labels=OBSERVATION_LABELS, ylabel=ylabel, ylim=ylim, yticks=yticks, p_value=float(stats_indexed.loc[letter, 'p_value_displayed']), paired=False, seed=700 + n, box_width=0.29)
        panel_axes['j'] = figure.add_axes(_panel_bounds(2, 3))
        axis = panel_axes['j']
        p = float(stats_indexed.loc['j', 'p_value_displayed'])
        _box_points(axis, (obs, nobs), labels=OBSERVATION_LABELS, ylabel='Dimensions explored (count)', ylim=(0, 5), yticks=list(range(6)), p_value=p, paired=False, seed=800, box_width=0.29, annotate_significance=False)
        for line in axis.lines:
            line.set_clip_on(False)
        axis.plot([0, 0, 1, 1], [1.045, 1.07, 1.07, 1.045], transform=axis.get_xaxis_transform(), color=INK, linewidth=0.65, clip_on=False, solid_capstyle='butt')
        axis.text(0.5, 1.088, original._stars(p), transform=axis.get_xaxis_transform(), ha='center', va='bottom', fontsize=6.5, fontweight='bold', color=INK)
        g_distribution = pd.DataFrame([{'group': label, 'dimensions_explored': level, 'participants': int(np.sum(vals == level)), 'group_n': len(vals), 'percentage': float(np.sum(vals == level) / len(vals) * 100)} for label, vals in zip(OBSERVATION_LABELS, (obs, nobs)) for level in range(6)])
        box_summary = pd.DataFrame([{'group': label, 'n': len(vals), 'mean': np.mean(vals), 'q1': np.percentile(vals, 25), 'median': np.median(vals), 'q3': np.percentile(vals, 75), 'whisker_low': original._whisker_limits(vals)[0], 'whisker_high': original._whisker_limits(vals)[1]} for label, vals in zip(OBSERVATION_LABELS, (obs, nobs))])
        for mapping in (panel_axes, source_icon_mapping, panel_mapping):
            mapping['g'], mapping['j'] = (mapping['j'], mapping['g'])
        stats['panel'] = stats['panel'].replace({'j': 'g', 'g-initial': 'j-initial', 'g-final': 'j-final'})
        stats = stats.sort_values('panel').reset_index(drop=True)
        for mapping in (panel_axes, source_icon_mapping, panel_mapping):
            old = dict(mapping)
            mapping.update(b=old['c'], c=old['b'], g=old['h'], h=old['i'], i=old['g'])
        stats['panel'] = stats['panel'].map(lambda value: {'b': 'c', 'c': 'b', 'g': 'i', 'h': 'g', 'i': 'h'}.get(value.split('-')[0], value.split('-')[0]) + ('-' + value.split('-', 1)[1] if '-' in value else ''))
        stats = stats.sort_values('panel').reset_index(drop=True)
        panel_axes['i'].set_ylabel('Dimensions explored\nthrough DIS (count)', labelpad=2.5)
        panel_axes['i'].set_xlabel('Trials 1–25', fontsize=5, labelpad=2.5)
        edition_audit['bottom_row_spacing'] = _balance_bottom_row(figure, panel_axes)
        edition_audit['bottom_row_height'] = _align_bottom_annotations(figure, panel_axes)
        label_positions = {'a': (LETTER_LEFTS[0], ROW_LETTER_TOPS[0]), 'b': (0.635, ROW_LETTER_TOPS[0]), **{letter: (LETTER_LEFTS[column], ROW_LETTER_TOPS[1]) for column, letter in enumerate('cdef')}, **{letter: (LETTER_LEFTS[column], ROW_LETTER_TOPS[2]) for column, letter in enumerate('ghij')}}
        label_positions.update({letter: (panel_axes[letter].get_position().x0 - 0.075, ROW_LETTER_TOPS[2]) for letter in 'ghij'})
        source_icon_bounds = {}
        for letter, (x, y) in label_positions.items():
            figure.text(x, y, letter, ha='left', va='top', fontsize=8, fontweight='bold', color=INK)
            icon_axis = add_source_icon(figure, source_icon_mapping[letter], left=x + 3.8 / (FIGURE_SIZE[0] * 25.4), top=y, size_mm=icon_size_mm)
            source_icon_bounds[letter] = list(icon_axis.get_position().bounds)
        if figure_finalize is not None:
            edition_audit.update(figure_finalize(figure, panel_axes, label_positions, stats) or {})
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        if gaze_color_axis.yaxis.label.get_window_extent(renderer).overlaps(panel_axes['b'].yaxis.label.get_window_extent(renderer)):
            raise ValueError("Panel a colour-bar label overlaps panel b's y-axis label")
        c_color_axis = next((axis for axis in figure.axes if axis.get_label() == 'fig5c_colorbar'))
        axes_layout = {letter: list(axis.get_position().bounds) for letter, axis in panel_axes.items()}
        shared_columns = tuple(zip('cdef', 'ghij'))
        for upper, lower in shared_columns:
            np.testing.assert_allclose(np.array(axes_layout[upper])[[0, 2, 3]], np.array(axes_layout[lower])[[0, 2, 3]], atol=1e-09)
        np.testing.assert_allclose([axes_layout[k][3] for k in 'cdefghij'], PLOT_HEIGHT, atol=1e-09)
        for previous, following in zip('ghi', 'hij'):
            if _axis_frame_and_text_bbox(panel_axes[previous], renderer).overlaps(_axis_frame_and_text_bbox(panel_axes[following], renderer)):
                raise ValueError(f'Bottom panels {previous}/{following} overlap')
        skill_scripts = Path(__file__).resolve().parent
        if str(skill_scripts) not in sys.path:
            sys.path.insert(0, str(skill_scripts))
        from audit_panel_alignment import require_matplotlib_panel_alignment
        alignment_path = DATA_DIR / f'{active_stem}_alignment.json'
        alignment = require_matplotlib_panel_alignment(figure, axes=list(panel_axes.values()), panel_ids={axis: label for label, axis in panel_axes.items()}, row_groups=[{'id': 'middle', 'panels': list('cdef')}, {'id': 'bottom', 'panels': list('ghij')}], column_groups=[{'id': f'column-{a}', 'panels': [a, b]} for a, b in zip('cdef', 'ghij')], exemptions=[], json_out=alignment_path, strict=True, tolerance_pt=1.5, gutter_tolerance_pt=1.5)
        edition_audit['alignment_verdict'] = alignment['verdict']
        edition_audit['alignment_audit'] = str(alignment_path.resolve())
        edition_audit['export_scope'] = 'PNG only; no PDF/SVG collision audit'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=original.DPI, facecolor='white', edgecolor='none', metadata={'Software': 'Matplotlib; mini_DG Figure 5 NHB revision'})
        plt.close(figure)
    with Image.open(output_path) as image:
        image.convert('RGB').save(output_path, dpi=(original.DPI, original.DPI))
        pixels = list(image.size)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    stats.to_csv(statistics_path, index=False)
    g_distribution.to_csv(g_distribution_path, index=False)
    timecourse_summary.to_csv(timecourse_summary_path, index=False)
    box_summary.to_csv(box_summary_path, index=False)
    manifest = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'figure': {'path': str(output_path.resolve()), 'sha256': _sha256(output_path), 'pixels': pixels, 'size_inches': list(FIGURE_SIZE), 'dpi': original.DPI, 'format': 'PNG', 'color_mode': 'RGB'}, 'output_formats': ['PNG'], 'edition_audit': edition_audit, 'panel_mapping': panel_mapping, 'bottom_panel_arrangement': bottom_panels, 'layout': {'plot_bounds_fraction': axes_layout, 'panel_letter_positions': label_positions, 'column_lefts': COLUMN_LEFTS, 'column_widths': COLUMN_WIDTHS, 'plot_height_mm': PLOT_HEIGHT * FIGURE_SIZE[1] * 25.4, 'heatmap_left_shift_mm': HEATMAP_LEFT_SHIFT_MM, 'note': 'Shared columns d/h, e/i, b/f/j; c and its colour bar shifted left 2.5 mm for label clearance; statistical frames equal height; a images enlarged with original aspect ratio.'}, 'display_style': {'condition_labels': {'DIS': 'DIS', 'NDIS': 'others', 'Obs': 'Obs', 'NObs': 'NObs'}, 'source_icons': {'mapping': source_icon_mapping, 'size_mm': icon_size_mm, 'bounds_fraction': source_icon_bounds, 'note': "Grayscale native Matplotlib symbols redrawn from the user's reference; data-source key is in the caption."}, 'category_axis_style': 'Short horizontal axis from first to second category centre, downward ticks, detached from y axis', 'statistical_annotations': 'Asterisks from the unchanged P values and correction definitions; numerical values in caption and data', 'truncated_axes': 'Visible discontinuity marks on b and f; f marks sit below the lowest tick (2)', 'others_color': OTHERS_GREY, 'panel_a_c_colormap': GAZE_CMAP, 'panel_c_color_limits': [0, HEATMAP_COLOR_MAX], 'panel_c_axes': {'x': 'DIS dimension', 'y': 'Fixated dimension', 'transposed_from_source_pivot': dis_on_x}, 'paired_panels': 'All paired points, faint individual connections, solid mean connection with diamonds', 'independent_panels': 'h–j: boxplots (1.5 IQR whiskers), outlier points, mean diamonds; no jittered point cloud', 'panel_g': {}, 'comparison_column_width_fraction': COLUMN_WIDTHS[1], 'first_column_width_fraction': COLUMN_WIDTHS[0], 'b_ylim': [100, 620], 'f_ylim': [2, 5.5]}, 'panel_a': {'dimension': 'D', 'fixation_threshold_ms': 200, 'image_bounds_fraction': dict(zip(('DIS', 'others'), image_bounds)), 'image_size_mm': [image_width * FIGURE_SIZE[0] * 25.4, image_height * FIGURE_SIZE[1] * 25.4], 'linear_scale_change': image_width / base_image_width, 'other_image_left_shift_mm': OTHER_IMAGE_LEFT_SHIFT_MM, 'shared_colorbar': {'bounds_fraction': gaze_bar_bounds, 'placement': 'Right of Other trials; label on the right, vertically centred', 'size_mm': [COLORBAR_WIDTH * FIGURE_SIZE[0] * 25.4, PLOT_HEIGHT * FIGURE_SIZE[1] * 25.4], 'label': 'Fixation density', 'endpoints': ['Low', 'High'], 'meaning': 'Qualitative source display-palette key; absolute density values are unavailable in the saved composites.'}, 'titles': list(image_titles), 'title_fontsize_pt': 6.0, 'title_colors': dict(zip(image_titles, image_title_colors)), 'title_weight': 'normal', 'title_gap_pt': title_gap_pt, 'density_area_normalized': True, 'peak_display_threshold': 0.2, 'non_fds_sample_seed': 234, 'sample_summary': sample_summary, 'recolor_audit': overlay_audit, 'note': 'Equal trial counts, not matched participants; original composite colours are approximately remapped without changing spatial sampling or background.'}, 'statistics': {'path': str(statistics_path.resolve()), 'sha256': _sha256(statistics_path)}, 'inputs': {name: {'path': str(path.resolve()), 'sha256': _sha256(path)} for name, path in sources.items()}}
    manifest['display_style']['panel_g'] = {'display': 'Early-stage to final-stage connected means with ±1 SEM, using the original Fig. 5 panel-j observations', 'stage_labels': {'Initial': 'Early stage', 'Final': 'Final stage'}, 'mean_marker_size_pt': STAGE_MEAN_MARKER_SIZE, 'width_mm': axes_layout[trajectory_letter][2] * FIGURE_SIZE[0] * 25.4, 'y_limits': [3.2, 6.4], 'y_ticks': [3.2, 4.8, 6.4], 'summary_csv': str(timecourse_summary_path.resolve()), 'statistics': 'Descriptive trajectory; original timecourse source has no significance annotation'}
    manifest['display_style']['panel_j'] = {'display': 'Boxplot matching h: median, IQR, 1.5 IQR whiskers, open outliers, mean diamonds', 'y_limits': [0, 5], 'y_ticks': list(range(6)), 'box_width': 0.29, 'summary_csv': str(box_summary_path.resolve()), 'frequency_csv': str(g_distribution_path.resolve()), 'statistics': 'Original exploration Mann–Whitney U test retained; P = 0.03313122238856'}
    style = manifest['display_style']
    style['panel_i'], style['panel_j'] = (style.pop('panel_j'), style.pop('panel_g'))
    style['source_icons'] = {'enabled': True, 'mapping': source_icon_mapping, 'bounds_fraction': source_icon_bounds, 'size_mm': icon_size_mm}
    style['truncated_axes'] = 'Visible discontinuity marks on c and f'
    style['panel_a_colormap'] = style.pop('panel_a_c_colormap')
    style['panel_b_colormap'] = GAZE_CMAP
    style['panel_b_color_limits'] = style.pop('panel_c_color_limits')
    style['panel_b_axes'] = style.pop('panel_c_axes')
    style['c_ylim'] = style.pop('b_ylim')
    style['independent_panels'] = 'g–i: boxplots (1.5 IQR whiskers), outlier points, mean diamonds; no jittered point cloud'
    manifest['layout']['note'] = 'Evidence-led layout: a gaze examples; b dimension selectivity; c–f gaze sampling; g–j observation in temporal order. Bottom row shares top and baseline; all four plots are 33.3 mm high, with a common annotation band above them. All eight c–j plots are 27.9 mm wide and 33.3 mm high, aligned in four shared columns. Source icons follow their evidence modality.'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return (output_path, statistics_path, manifest_path)
