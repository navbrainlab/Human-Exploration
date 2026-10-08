"""Render isolated Figure 6 no-bonus edition from its packaged CSV values."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib.ticker import NullLocator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE
sys.path.insert(0, str(ROOT))
import publication_style as final
spec = importlib.util.spec_from_file_location('fig6_nouncertainty_layout', PACKAGE / 'code/figure6_layout.py')
layout = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = layout
spec.loader.exec_module(layout)

def draw_slope(main, statistics):
    participants = pd.read_csv(PACKAGE / 'data/participant_temperature_slope.csv', dtype={'subject': str})
    tests = pd.read_csv(PACKAGE / 'data/slope_phase_tests.csv')
    assert len(participants) == 156 and participants.subject.nunique() == 105
    axis = main.add_axes((0.495, 0.075, 0.215, 0.215))
    purple = layout.DGEM_PURPLE
    phases = ['P1', 'P2', 'P2-only']
    labels = ['3D', '4D-E', '4D-NE']
    paired = participants.loc[participants.phase.isin(['P1', 'P2'])].pivot(index='subject', columns='phase', values='temperature_slope')
    assert len(paired) == 51 and paired.notna().all().all()
    counts = []
    for i, phase in enumerate(phases):
        group = participants.loc[participants.phase.eq(phase)].sort_values('subject')
        values = group.temperature_slope.to_numpy(float)
        counts.append(len(values))
        assert (values > 0).all()
        axis.boxplot([values], positions=[i], widths=0.45, showfliers=False, patch_artist=True, zorder=3, boxprops={'facecolor': matplotlib.colors.to_rgba(purple, 0.17), 'edgecolor': purple, 'linewidth': 0.8}, medianprops={'color': purple, 'linewidth': 1.1}, whiskerprops={'color': purple, 'linewidth': 0.7}, capprops={'color': purple, 'linewidth': 0.7})
    axis.set_yscale('log')
    axis.set_ylim(0.007, 200)
    axis.set_xlim(-0.45, 2.45)
    axis.set_yticks([0.01, 0.1, 1, 10], ['0.01', '0.1', '1', '10'])
    axis.yaxis.set_minor_locator(NullLocator())
    axis.set_xticks([0, 1, 2], labels)
    axis.set_xlabel('Task phase')
    axis.set_ylabel('Temperature slope')
    axis.spines['bottom'].set_bounds(0, 2)
    from matplotlib.transforms import blended_transform_factory
    bracket_transform = blended_transform_factory(axis.transData, axis.transAxes)
    brackets = []
    stars = []
    for left, right, height in [(0, 1, 0.73), (1, 2, 0.73), (0, 2, 0.93)]:
        row = tests.loc[tests.phase_a.eq(phases[left]) & tests.phase_b.eq(phases[right])]
        assert len(row) == 1
        star = str(row.iloc[0].significance)
        line, = axis.plot([left, left, right, right], [height - 0.018, height, height, height - 0.018], transform=bracket_transform, color=layout.INK, linewidth=0.65, clip_on=False)
        text = axis.text((left + right) / 2, height + 0.018, star, ha='center', va='bottom', transform=bracket_transform, fontsize=6, fontweight='normal' if star == 'ns' else 'bold', color=layout.INK)
        brackets.append(line)
        stars.append(text)
    layout._style_data_axis(axis)
    axis._slope_display_spec = {'source_counts': counts, 'brackets': brackets, 'stars': stars, 'individual_points_displayed': False, 'fliers_displayed': False}
    return axis

def finalize(figure, main, extension, statistics):
    axes = dict(zip(('a', 'b1', 'b2', 'c', 'd_upper', 'd_lower', 'e1', 'e2', 'g_upper', 'g_lower', 'f', 'h', 'i'), figure.axes))
    assert len(figure.axes) == 13
    final._recolor(figure)
    style = final._fig6_typography(figure, axes)
    scale = 180 / (figure.get_figwidth() * 25.4)
    figure.set_size_inches(*figure.get_size_inches() * scale)
    qa = final._render_qa(figure, PACKAGE / 'fig6_nouncertainty_render_qa.json')
    alignment = final.require_matplotlib_panel_alignment(figure, axes=[axes[k] for k in ('b1', 'b2', 'c', 'd_upper', 'd_lower', 'e1', 'e2', 'f', 'g_upper', 'g_lower')], panel_ids={axes[k]: k for k in ('b1', 'b2', 'c', 'd_upper', 'd_lower', 'e1', 'e2', 'f', 'g_upper', 'g_lower')}, row_groups=[{'id': 'fit-tasks', 'panels': ['b1', 'b2']}, {'id': 'performance-and-temperature', 'panels': ['e1', 'e2', 'f']}], column_groups=[{'id': 'task1', 'panels': ['b1', 'e1']}, {'id': 'task2', 'panels': ['b2', 'e2']}, {'id': 'dis-association', 'panels': ['d_upper', 'd_lower']}, {'id': 'round-trajectories', 'panels': ['g_upper', 'g_lower']}], exemptions=[{'panels': ['e2', 'f'], 'checks': ['panel-width', 'horizontal-gutter'], 'reason': 'Established unequal task-specific and parameter-panel spans'}], json_out=PACKAGE / 'fig6_nouncertainty_alignment.json', strict=True, tolerance_pt=1.5, gutter_tolerance_pt=1.5)
    return {'typography': style, 'font_floor_pt': qa['font_floor_pt'], 'qa_status': 'PASS', 'alignment_status': 'PASS', 'no_bonus_data': 'panels b Task1,c,d,e DGEM,f,g DGEM; original Task2 fit unavailable'}

def main():
    layout._draw_temperature_comparison = draw_slope
    layout.PANEL_MAPPING['f'] = 'No-bonus fitted temperature-gate slopes across 3D, 4D-E and 4D-NE'
    layout.CAPTION_PATH = PACKAGE / 'caption.md'
    layout.METHODS_PATH = PACKAGE / 'methods.md'
    layout.SOURCE_PROVENANCE_PATH = PACKAGE / 'source_provenance.json'
    result = layout.assemble(output_stem='fig6_nouncertainty', figure_finalize=finalize)
    print('\n'.join((str(x) for x in result)))
if __name__ == '__main__':
    raise SystemExit('Reusable slope drawing: run render_fig6_full_joint90_free_beta.py instead.')
