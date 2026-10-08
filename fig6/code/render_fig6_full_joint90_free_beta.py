"""Render a separate edition: performance, phase slopes, DIS trajectories."""
from __future__ import annotations
import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
PACKAGE = Path(__file__).resolve().parents[1]
MAIN = PACKAGE / 'output'
ROOT = PACKAGE
sys.path.insert(0, str(ROOT))

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
base = load('fig6_free_beta_base', PACKAGE / 'code/full_joint90_native.py')
slope = load('fig6_free_beta_requested_slope', PACKAGE / 'code/no_bonus_slope.py')
layout = base.layout
layout.CSV_ROOT = PACKAGE / 'data'
layout.DATA_PATHS = {key: PACKAGE / 'data' / path.name for key, path in layout.DATA_PATHS.items() if key != 'temperature_entities'}
layout.DATA_PATHS.update(participant_parameters=PACKAGE / 'data/participant_fitted_parameters.csv', slope_participants=PACKAGE / 'data/participant_temperature_slope.csv', slope_tests=PACKAGE / 'data/slope_phase_tests.csv')
layout.STATISTICS_PATH = PACKAGE / 'data/panel_statistics.csv'
layout.CAPTION_PATH = PACKAGE / 'caption.md'
layout.METHODS_PATH = PACKAGE / 'methods.md'
layout.SOURCE_PROVENANCE_PATH = PACKAGE / 'source_provenance.json'
layout.MANIFEST_DIR = PACKAGE / 'output'
layout.OUTPUT_DIR = MAIN
base.AUDIT = PACKAGE / 'output'
slope.PACKAGE = PACKAGE
slope.layout = layout
layout._draw_temperature_comparison = slope.draw_slope
layout.PANEL_MAPPING.update(b='Task1 seven-parameter DGEM free-beta fit; Task2 original fixed-beta reference', c='Free-beta fitted-data replay: effective attention temperature and dimension weight', d='Free-beta fitted-data replay: maximum attention and observed DIS', e='Original independently validated full-DGEM best performance, Task1 and Task2', f='No-bonus fitted temperature slopes: 3D, 4D-E, 4D-NE; boxes only, without individual points or paired lines', g='Free-beta fitted-parameter simulations: DIS; lower trace is fitted-data replay temperature', h='Dimension-guided movie recommendation: semantic tag clusters, selected tags and candidates; Movie B highlighted as selected')

def statistics():
    frame = pd.read_csv(layout.STATISTICS_PATH)
    assert set(frame.panel_comparison) == {'d_DIS_vs_Non-DIS'}
    assert frame.p_value.between(0, 1).all()
    return frame
layout._compute_panel_statistics = statistics
original_b = layout._draw_fitting_comparison

def draw_fitting(figure):
    before = len(figure.axes)
    original_b(figure)
    first, second = figure.axes[before:]
    values = pd.read_csv(layout.DATA_PATHS['fit_summary']).set_index('model')
    dgem = values.loc['DGEM']
    end = 1000 * (dgem.mean_geom_likelihood + dgem.sem_geom_likelihood) + 0.2
    bracket = end + 1.0
    step = float(np.ceil((end + 3.0) / 3))
    first.set_xlim(0, 3 * step)
    first.set_xticks(np.arange(4) * step)
    for line in first.lines:
        x, y = (np.asarray(line.get_xdata()), np.asarray(line.get_ydata()))
        if len(x) == 4 and y[0] == 4 and (y[1] == 4) and (y[2] == y[3]):
            line.set_xdata([end, bracket, bracket, x[-1]])
    for text in first.texts:
        if text.get_text() in ('ns', '*', '**', '***'):
            text.set_x(bracket + 1.0)
    second.set_title('Task 2 reference', loc='left', pad=4.0, fontsize=6.0)
layout._draw_fitting_comparison = draw_fitting

def finalize(figure, main, extension, tests):
    fitting1, fitting2 = figure.axes[1:3]
    attention = figure.axes[3]
    _, bottom, _, height = attention.get_position().bounds
    old_bottom = fitting1.get_position().y0
    for axis in (fitting1, fitting2):
        left, _, width, _ = axis.get_position().bounds
        axis.set_position((left, bottom, width, height))
    for text in main.texts:
        if text.get_text().startswith('Mean likelihood per trial'):
            text.set_y(text.get_position()[1] + bottom - old_bottom)
    performance1, performance2 = figure.axes[6:8]
    dis, temperature = figure.axes[8:10]
    phase = figure.axes[10]
    phase.set_position((0.495, 0.075, 0.215, 0.215))
    dis.set_position((0.76, 0.14, 0.215, 0.15))
    temperature.set_position((0.76, 0.075, 0.215, 0.04))
    performance1.set_position((0.073, 0.075, 0.152, 0.215))
    performance2.set_position((0.273, 0.075, 0.162, 0.215))
    for text in main.texts:
        label = text.get_text()
        if label == 'e':
            text.set_position((0.008, 0.324))
        elif label == 'f':
            text.set_position((0.469, 0.324))
        elif label == 'g':
            text.set_position((0.732, 0.324))
        elif label == 'DIS ratio':
            text.set_x(0.254)
    result = base.finalize(figure, main, extension, tests)
    result.pop('full_dgem_90both', None)
    result['scientific_scope'] = {'b_Task1_c_d_g': 'DGEM with individually fitted beta_init and six other parameters', 'b_Task2': 'unchanged original fixed-beta reference; not re-fitted', 'e': 'unchanged full-DGEM best-performance 5000-agent results, beta_init fixed20', 'f': 'original no-bonus fitted slopes and tests, without paired visual lines'}
    figure.canvas.draw()
    factor = 25.4 / figure.dpi
    fitting_attention = {'status': 'PASS', 'plot_heights_mm': {name: axis.bbox.height * factor for name, axis in [('b1', fitting1), ('b2', fitting2), ('c', attention)]}, 'top_spread_mm': float(np.ptp([axis.bbox.y1 for axis in (fitting1, fitting2, attention)])) * factor, 'bottom_spread_mm': float(np.ptp([axis.bbox.y0 for axis in (fitting1, fitting2, attention)])) * factor}
    assert fitting_attention['top_spread_mm'] < 0.01 and fitting_attention['bottom_spread_mm'] < 0.01
    result['fitting_attention_height_qa'] = fitting_attention
    top = [phase.bbox.y1, dis.bbox.y1, performance1.bbox.y1, performance2.bbox.y1]
    bottom = [phase.bbox.y0, temperature.bbox.y0, performance1.bbox.y0, performance2.bbox.y0]
    assert max(top) - min(top) < figure.dpi * 1.5 / 72
    assert max(bottom) - min(bottom) < figure.dpi * 1.5 / 72
    result['third_row_edges_aligned'] = True
    from matplotlib.collections import PathCollection
    counts = [len(item.get_offsets()) for item in phase.collections if isinstance(item, PathCollection)]
    assert not counts, 'No individual scatter points should appear in panel f'
    spec = phase._slope_display_spec
    assert spec['source_counts'] == [51, 51, 54] and len(phase.patches) == 3
    diagonal_pairs = [line for line in phase.lines if len(line.get_xdata()) == 2 and np.ptp(line.get_xdata()) > 0 and (np.ptp(line.get_ydata()) > 0)]
    assert not diagonal_pairs, 'Participant connecting lines must be absent'
    from matplotlib.path import Path as MplPath
    renderer = figure.canvas.get_renderer()
    adjacent_heights = [line.get_transform().transform(line.get_xydata())[1, 1] for line in spec['brackets'][:2]]
    assert abs(adjacent_heights[0] - adjacent_heights[1]) < 1e-06
    boxes = [text.get_window_extent(renderer) for text in spec['stars']]
    for i, box in enumerate(boxes):
        for other in boxes[i + 1:]:
            assert min(box.x1, other.x1) <= max(box.x0, other.x0) or min(box.y1, other.y1) <= max(box.y0, other.y0)
        for line in spec['brackets']:
            assert not MplPath(line.get_transform().transform(line.get_xydata())).intersects_bbox(box, filled=False)
    result['slope_panel'] = {'source_counts': spec['source_counts'], 'displayed_point_counts': counts, 'box_count': len(phase.patches), 'connecting_subject_lines': 0, 'statistics_unchanged': True, 'display': 'boxes only; individual points and fliers hidden', 'significance_annotation_collisions': 0, 'adjacent_comparison_brackets_same_height': True}
    spec = figure.axes[11]._movie_selection_spec
    assert len(spec['containers']) == 12 and len(spec['strokes']) == 3
    result['h_schematic'] = {'display': 'selected Movie B within candidates; no separate Action box', 'boxes': 12, 'arrows': 3, 'selected_movie': 'Movie B', 'action_box': False, 'other_tags_muted_block': True, 'text_labels': len(spec['texts']), 'details_in_methods': True}
    result['public_panel_order'] = ['e: performance tasks', 'f: slope distributions', 'g: DIS and temp.']
    return result

def main():
    output, manifest = layout.assemble(output_stem='fig6_full_joint90_free_beta_performance_first', figure_finalize=finalize)
    print(json.dumps({'output': str(output), 'manifest': str(manifest)}, indent=2))
if __name__ == '__main__':
    main()
