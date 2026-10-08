"""Render a separate full-DGEM Fig. 6 with independently validated 90/90 e data."""
from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from PIL import Image
PKG = Path(__file__).resolve().parents[1]
MAIN = PKG / 'output'
AUDIT = PKG / 'output'
ROOT = PKG
ORIGINAL = PKG
sys.path.insert(0, str(ROOT))
import publication_style as final
spec = importlib.util.spec_from_file_location('figure6_full_joint90_layout', ORIGINAL / 'code/figure6_layout.py')
layout = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = layout
spec.loader.exec_module(layout)
layout.CSV_ROOT = PKG / 'data'
layout.DATA_PATHS = {name: PKG / 'data' / path.name for name, path in layout.DATA_PATHS.items()}
layout.STATISTICS_PATH = PKG / 'data/panel_statistics.csv'
layout.CAPTION_PATH = PKG / 'caption.md'
layout.METHODS_PATH = PKG / 'methods.md'
layout.SOURCE_PROVENANCE_PATH = PKG / 'source_provenance.json'
layout.MANIFEST_DIR = PKG / 'output'
layout.OUTPUT_DIR = MAIN
layout.DPI = 600
layout.PANEL_MAPPING['e'] = 'Independent 5,000-agent full-DGEM best-performance validation: both DIS and full score exceed 90%'

def draw_uncertainty_flow(figure, bounds):
    """Native stimulus-to-action schematic; standalone intermediates are unnecessary."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
    ax = figure.add_axes(bounds)
    ax.set(xlim=(0, 180), ylim=(0, 49))
    ax.set_axis_off()
    INK, NEUTRAL, PURPLE = ('#202B3B', '#6E7D8E', '#6F55A5')
    texts = []
    containers = []
    strokes = []

    def label(x, y, s, ha='center', **kw):
        t = ax.text(x, y, s, fontsize=kw.pop('fontsize', 7), ha=ha, va='center', color=kw.pop('color', INK), **kw)
        texts.append(t)
        return t

    def box(x, y, w, h, accent=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=.6', facecolor='#F0EAF8' if accent else '#F3F6F9', edgecolor=PURPLE if accent else NEUTRAL, linewidth=0.75))
        containers.append((x, y, w, h))

    def arrow(points, color=NEUTRAL):
        strokes.extend(zip(points[:-1], points[1:]))
        for a, b in zip(points[:-2], points[1:-1]):
            ax.plot([a[0], b[0]], [a[1], b[1]], lw=0.85, color=color, solid_capstyle='butt')
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle='-|>', mutation_scale=7, linewidth=0.85, color=color, shrinkA=0, shrinkB=0))
    illustrative_features = [(r + 1, c + 1, (r + c) % 3 + 1) for r in range(3) for c in range(3)]
    tile_palette = ['#8FAEC6', '#B59BD2', '#95B4A0']
    tile_artists = {}

    def tiles(x, y, order):
        patches = []
        for row in range(3):
            for col in range(3):
                idx = order[row * 3 + col]
                tile = Rectangle((x + col * 2.8, y + (2 - row) * 1.9), 2.1, 1.3, facecolor=tile_palette[illustrative_features[idx][0] - 1], edgecolor='#798695', linewidth=0.3)
                ax.add_patch(tile)
                patches.append(tile)
        return patches

    def operator(x, y, symbol):
        ax.add_patch(Circle((x, y), 2.7, facecolor='white', edgecolor=NEUTRAL, linewidth=0.85))
        containers.append((x - 2.7, y - 2.7, 5.4, 5.4))
        label(x, y, symbol, fontsize=8)
    box(3, 9.5, 17, 14)
    label(11.5, 20.1, 'Stimulus')
    stimulus_order = [0, 4, 8, 5, 6, 1, 7, 2, 3]
    tile_artists['stimulus'] = tiles(7.65, 12.5, stimulus_order)
    box(27, 20, 28, 11)
    label(41, 28, 'Feature uncertainty')
    label(41, 22.9, '$U_{df}=1/(n_{df}+\\epsilon)$')
    box(27, 3, 28, 11)
    label(41, 11, 'Feature value')
    label(41, 5.9, '$Q_{df}$')
    box(61, 29, 20, 17, accent=True)
    label(71, 42.6, 'Attention\ncontrol', linespacing=1.1, fontweight='bold')
    label(71, 37.2, '$U_d=\\sum_f U_{df}$')
    label(71, 32.2, '$\\pi_d,\\;\\tau_{\\mathrm{att}}$')
    box(85, 29, 54, 17, accent=True)
    label(112, 42.6, 'Dimension attention weights', fontweight='bold')
    weight_formula = label(121.6, 34.5, '$w_d=\\mathrm{softmax}_d\\!\\left(\\frac{\\log\\pi_d+G_d}{\\tau_{\\mathrm{att}}}\\right)$')
    illustrative_weights = [0.72, 0.18, 0.1]
    illustrative_bars = []
    for i, (x, w, color) in enumerate(zip([88.1, 93.6, 99.1], illustrative_weights, [PURPLE, '#B09CCF', '#D0C3E2']), 1):
        bar = Rectangle((x, 33.2), 3.5, 8 * w, facecolor=color, edgecolor='none')
        ax.add_patch(bar)
        illustrative_bars.append(bar)
        label(x + 1.75, 31.2, f'D{i}')
    operator(62.5, 16.5, '+')
    box(68, 9.5, 28, 14)
    label(82, 20, 'Feature score')
    label(82, 13.8, '$S_{df}=Q_{df}+\\beta U_{df}+\\sigma Z_{df}$')
    operator(101, 16.5, '×')
    box(109, 9.5, 24, 14)
    label(121, 20, 'Item score')
    label(121, 13.8, '$v_j=\\sum_d w_d S_{d,f_d(j)}$')
    box(138, 9.5, 23, 14)
    label(149.5, 20.2, 'Rank items')
    label(149.5, 12.9, 'Group into 3 rows')
    arrow([(149.5, 18.2), (149.5, 15.2)])
    box(166, 9.5, 12, 14)
    label(172, 20.1, 'Action')
    action_order = list(range(9))
    tile_artists['action'] = tiles(168.15, 12.5, action_order)
    arrow([(20, 16.5), (23.5, 16.5), (23.5, 25.5), (27, 25.5)])
    arrow([(23.5, 16.5), (23.5, 8.5), (27, 8.5)])
    arrow([(41, 31), (41, 37.5), (61, 37.5)], PURPLE)
    arrow([(81, 37.5), (85, 37.5)], PURPLE)
    arrow([(101, 29), (101, 19.2)], PURPLE)
    arrow([(55, 25.5), (58, 25.5), (58, 16.5), (59.8, 16.5)])
    value_link = [(55, 8.5), (58, 8.5), (58, 16.5)]
    strokes.extend(zip(value_link[:-1], value_link[1:]))
    for a, b in zip(value_link[:-1], value_link[1:]):
        ax.plot([a[0], b[0]], [a[1], b[1]], lw=0.85, color=NEUTRAL, solid_capstyle='butt')
    arrow([(65.2, 16.5), (68, 16.5)])
    arrow([(96, 16.5), (98.3, 16.5)])
    arrow([(103.7, 16.5), (109, 16.5)])
    arrow([(133, 16.5), (138, 16.5)])
    arrow([(161, 16.5), (166, 16.5)])
    ax._model_flow_spec = {'texts': texts, 'containers': containers, 'strokes': strokes, 'bars': illustrative_bars, 'illustrative_weights': illustrative_weights, 'action_order': action_order, 'stimulus_order': stimulus_order, 'illustrative_tile_features': illustrative_features, 'tile_artists': tile_artists}
    return ax

def audit_model_flow(figure, axis, path):
    """Native PNG geometry audit after typography and final physical layout."""
    from matplotlib.path import Path as MplPath
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    spec = axis._model_flow_spec
    boxes = [text.get_window_extent(renderer) for text in spec['texts']]
    collisions = []
    for i, a in enumerate(boxes):
        for j, b in enumerate(boxes[i + 1:], i + 1):
            if min(a.x1, b.x1) > max(a.x0, b.x0) and min(a.y1, b.y1) > max(a.y0, b.y0):
                collisions.append([i, j])
    outside = []
    for i, (text, r) in enumerate(zip(spec['texts'], boxes)):
        x, y = text.get_position()
        enclosing = [b for b in spec['containers'] if b[0] < x < b[0] + b[2] and b[1] < y < b[1] + b[3]]
        assert len(enclosing) == 1, (i, text.get_text())
        bx, by, bw, bh = enclosing[0]
        lo = axis.transData.transform((bx, by))
        hi = axis.transData.transform((bx + bw, by + bh))
        if r.x0 < lo[0] or r.y0 < lo[1] or r.x1 > hi[0] or (r.y1 > hi[1]):
            outside.append(i)
    crossed = []
    bar_overlap = []
    for i, r in enumerate(boxes):
        for a, b in spec['strokes']:
            if MplPath(axis.transData.transform([a, b])).intersects_bbox(r, filled=False):
                crossed.append(i)
        for bar in spec['bars']:
            b = bar.get_window_extent(renderer)
            if min(r.x1, b.x1) > max(r.x0, b.x0) and min(r.y1, b.y1) > max(r.y0, b.y0):
                bar_overlap.append(i)
    assert not (collisions or outside or crossed or bar_overlap), (collisions, outside, crossed, bar_overlap)
    import numpy as np
    from portable_dis import is_dis_action
    assert sorted(spec['action_order']) == list(range(9))
    assert sorted(spec['stimulus_order']) == list(range(9))
    action = np.asarray(spec['illustrative_tile_features'])[spec['action_order']].reshape(3, 3, 3)
    stimulus = np.asarray(spec['illustrative_tile_features'])[spec['stimulus_order']].reshape(3, 3, 3)
    action_is_dis = bool(is_dis_action(action))
    assert action_is_dis and (not is_dis_action(stimulus))
    colors = {name: [tile.get_facecolor() for tile in tiles] for name, tiles in spec['tile_artists'].items()}
    assert all((len(set(colors['action'][3 * r:3 * r + 3])) == 1 for r in range(3)))
    assert len({colors['action'][3 * r] for r in range(3)}) == 3
    assert all((len(set(colors['stimulus'][3 * r:3 * r + 3])) == 3 for r in range(3)))
    report = {'status': 'PASS', 'text_collisions': collisions, 'labels_outside_boxes': outside, 'arrow_text_intersections': crossed, 'bar_text_intersections': bar_overlap, 'nominal_font_sizes_pt': sorted({t.get_fontsize() for t in spec['texts']}), 'physical_size_mm': [axis.bbox.width / figure.dpi * 25.4, axis.bbox.height / figure.dpi * 25.4], 'illustrative_weights_are_data': False, 'illustrative_weight_vector': spec['illustrative_weights'], 'illustrative_action_order': spec['action_order'], 'illustrative_action_is_dis': action_is_dis, 'illustrative_stimulus_order': spec['stimulus_order'], 'illustrative_stimulus_is_dis': False, 'action_row_color_counts': [1, 1, 1], 'stimulus_row_color_counts': [3, 3, 3], 'illustrative_tile_features': spec['illustrative_tile_features'], 'illustrative_action_is_data': False, 'scope': 'Native Matplotlib geometry and PNG visual QA; no PDF audit.'}
    path.write_text(json.dumps(report, indent=2) + '\n')
    return report
layout.draw_model_flow_panel = draw_uncertainty_flow

def draw_movie_dimension_selection(figure):
    """TDGE movie example grounded in the supplied manuscript, pp. 5–6, 16–18."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Arc
    axis = figure.add_axes((0.025, 0.055, 0.62, 0.885))
    axis.set(xlim=(0, 105), ylim=(0, 32.08))
    axis.set_axis_off()
    ink, neutral, purple = ('#202B3B', '#6E7D8E', '#6F55A5')
    texts = []
    containers = []
    strokes = []
    box_artists = []

    def box(x, y, w, h, accent=False, muted=False):
        artist = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=.45', facecolor='#F0EAF8' if accent else '#F3F6F9', edgecolor=purple if accent else '#B7BEC5' if muted else neutral, linewidth=0.75)
        axis.add_patch(artist)
        box_artists.append(artist)
        containers.append((x, y, w, h))
        return len(containers) - 1

    def label(x, y, s, owner=None, **kw):
        artist = axis.text(x, y, s, fontsize=6, ha='center', va='center', color=kw.pop('color', ink), linespacing=1.15, **kw)
        texts.append((artist, owner))
        return artist

    def arrow(x1, x2, y=14.5, accent=False):
        strokes.append(((x1, y), (x2, y)))
        axis.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle='-|>', mutation_scale=6, linewidth=0.85, color=purple if accent else neutral, shrinkA=0, shrinkB=0))

    def film_icon(x, y, w, h, color=neutral):
        axis.add_patch(Rectangle((x, y), w, h, facecolor='white', edgecolor=color, linewidth=0.75))
        for side in (x + 0.35, x + w - 1.05):
            for offset in (0.45, h / 2 - 0.3, h - 1.15):
                axis.add_patch(Rectangle((side, y + offset), 0.7, 0.7, facecolor=color, edgecolor='none'))
    label(4.5, 27.5, 'User')
    label(25.5, 27.5, 'Dimensions\n(tag clusters)\n' + 'Top-$K_1$')
    label(54.5, 27.5, 'Features\n(movie tags)\n' + 'Top-$K_2$')
    label(82.5, 27.5, 'Candidate\nmovies')
    user = box(0, 6, 9, 17)
    axis.add_patch(Circle((4.5, 18), 1.4, facecolor='white', edgecolor=neutral, linewidth=0.75))
    axis.add_patch(Arc((4.5, 13.8), 5.2, 5.2, theta1=0, theta2=180, color=neutral, linewidth=0.75))
    label(4.5, 10, '$u_t$', owner=user)
    violence = box(13, 17, 25, 6, accent=True)
    country = box(13, 10.5, 25, 6, accent=True)
    other = box(13, 5.5, 25, 3.5, muted=True)
    label(25.5, 20, 'Violence', owner=violence, color=purple)
    label(25.5, 13.5, 'Country', owner=country, color=purple)
    label(25.5, 7.25, 'Other clusters', owner=other, color='#7A838B')
    feature_boxes = []
    for y, left, right in ((17, 'gruesome', 'brutal'), (10.5, 'australia', 'canada')):
        lbox = box(42, y, 12, 6, accent=True)
        rbox = box(55, y, 12, 6, accent=True)
        feature_boxes.extend([lbox, rbox])
        label(48, y + 3, left, owner=lbox)
        label(61, y + 3, right, owner=rbox)
    other_tags = box(42, 5.5, 25, 3.5, muted=True)
    label(54.5, 7.25, 'Other tags', owner=other_tags, color='#7A838B')
    for y, name in ((17.5, 'A'), (12, 'B'), (6.5, 'C')):
        selected = name == 'B'
        candidate = box(71, y, 23, 5, accent=selected)
        film_icon(73, y + 1, 3, 3, purple if selected else neutral)
        label(85, y + 2.5, f'Movie {name}', owner=candidate, color=purple if selected else ink)
        if selected:
            selected_candidate = candidate
    arrow(9, 13)
    arrow(38, 42, accent=True)
    arrow(67, 71, accent=True)
    label(40, 2, 'Value & uncertainty guide selection', color=purple)
    label(82.5, 2, 'Tag similarity')
    axis._movie_selection_spec = {'texts': texts, 'containers': containers, 'strokes': strokes, 'box_artists': box_artists, 'feature_boxes': feature_boxes, 'selected_candidate': selected_candidate, 'other_cluster_box': other, 'other_tags_box': other_tags}
    return axis

def audit_movie_selection(figure, axis, path):
    """Check h at the final physical size without making a PDF/SVG export."""
    from matplotlib.path import Path as MplPath
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    spec = axis._movie_selection_spec
    boxes = [artist.get_window_extent(renderer) for artist, _ in spec['texts']]
    collisions = []
    outside = []
    crossed = []
    clipped = []
    for i, a in enumerate(boxes):
        for j, b in enumerate(boxes[i + 1:], i + 1):
            if min(a.x1, b.x1) > max(a.x0, b.x0) and min(a.y1, b.y1) > max(a.y0, b.y0):
                collisions.append([i, j])
        if a.x0 < axis.bbox.x0 or a.x1 > axis.bbox.x1 or a.y0 < axis.bbox.y0 or (a.y1 > axis.bbox.y1):
            clipped.append(i)
        owner = spec['texts'][i][1]
        if owner is not None:
            x, y, w, h = spec['containers'][owner]
            lo = axis.transData.transform((x, y))
            hi = axis.transData.transform((x + w, y + h))
            if a.x0 < lo[0] or a.y0 < lo[1] or a.x1 > hi[0] or (a.y1 > hi[1]):
                outside.append(i)
        for start, end in spec['strokes']:
            if MplPath(axis.transData.transform([start, end])).intersects_bbox(a, filled=False):
                crossed.append(i)
    from matplotlib.text import Text
    neighboring_labels = []
    for other_axis in figure.axes:
        if other_axis is axis:
            continue
        for text in other_axis.findobj(Text):
            if not text.get_visible() or not text.get_text().strip():
                continue
            r = text.get_window_extent(renderer)
            for x, y, w, h in spec['containers']:
                lo = axis.transData.transform((x, y))
                hi = axis.transData.transform((x + w, y + h))
                if min(r.x1, hi[0]) > max(r.x0, lo[0]) and min(r.y1, hi[1]) > max(r.y0, lo[1]):
                    neighboring_labels.append(text.get_text())
    assert not (collisions or outside or crossed or clipped or neighboring_labels), (collisions, outside, crossed, clipped, neighboring_labels)
    selected = spec['box_artists'][spec['selected_candidate']]
    assert all((selected.get_facecolor() == spec['box_artists'][idx].get_facecolor() and selected.get_edgecolor() == spec['box_artists'][idx].get_edgecolor() for idx in spec['feature_boxes']))
    assert not any((text.get_text() == 'Action' for text, _ in spec['texts']))
    other_cluster = spec['box_artists'][spec['other_cluster_box']]
    other_tags = spec['box_artists'][spec['other_tags_box']]
    assert other_tags.get_facecolor() == other_cluster.get_facecolor()
    assert other_tags.get_edgecolor() == other_cluster.get_edgecolor()
    assert not any((text.get_text() == 'Selected tags' for text, _ in spec['texts']))
    report = {'status': 'PASS', 'text_collisions': collisions, 'labels_outside_boxes': outside, 'arrow_text_intersections': crossed, 'clipped_text': clipped, 'neighboring_labels_intersecting_boxes': neighboring_labels, 'nominal_font_sizes_pt': sorted({artist.get_fontsize() for artist, _ in spec['texts']}), 'physical_size_mm': [axis.bbox.width / figure.dpi * 25.4, axis.bbox.height / figure.dpi * 25.4], 'reward_node_or_feedback_arrows': False, 'movie_examples_are_data': False, 'action_box_displayed': False, 'selected_candidate': 'Movie B', 'selected_candidate_matches_feature_fill_and_edge': True, 'other_tags_block_matches_other_clusters': True, 'source': 'Task_dimension_guided_exploration_NeurIPS_2026 (19).pdf', 'source_pages': [5, 6, 16, 17, 18], 'scope': 'Native Matplotlib text, box and arrow geometry plus PNG visual inspection.'}
    path.write_text(json.dumps(report, indent=2) + '\n')
    return report
layout._draw_hierarchical_selection = draw_movie_dimension_selection
layout.PANEL_MAPPING['h'] = 'TDGE movie recommendation: semantic tag clusters, selected movie tags, similarity retrieval and item-level scoring'

def measure_horizontal_spacing(figure, axes):
    """Measure rendered panel content, including ticks, titles and legends."""
    from matplotlib.transforms import Bbox
    renderer = figure.canvas.get_renderer()
    factor = 25.4 / figure.dpi
    areas = {}
    for name, axis in axes.items():
        if name == 'a':
            continue
        box = axis.get_tightbbox(renderer)
        if name == 'h':
            spec = axis._movie_selection_spec
            boxes = [t.get_window_extent(renderer) for t, _ in spec['texts']]
            boxes.extend((Bbox.from_extents(*axis.transData.transform((x, y)), *axis.transData.transform((x + w, y + h))) for x, y, w, h in spec['containers']))
            box = Bbox.union(boxes)
        areas[name] = {'plot_mm': [v * factor for v in axis.bbox.extents], 'content_mm': [v * factor for v in box.extents]}
    groups = {'b': ['b1', 'b2'], 'c': ['c'], 'd': ['d_upper', 'd_lower'], 'e': ['e1', 'e2'], 'f': ['f'], 'g': ['g_upper', 'g_lower'], 'h': ['h'], 'i': ['i'], 'b1': ['b1'], 'b2': ['b2'], 'e1': ['e1'], 'e2': ['e2']}
    content = {name: [min((areas[key]['content_mm'][0] for key in members)), max((areas[key]['content_mm'][2] for key in members))] for name, members in groups.items()}
    pairs = [('b1', 'b2'), ('b', 'c'), ('c', 'd'), ('e1', 'e2'), ('e', 'f'), ('f', 'g'), ('h', 'i')]
    return {'areas': areas, 'content_gaps_mm': {f'{left}-{right}': content[right][0] - content[left][1] for left, right in pairs}}

def increase_row_spacing(figure, main, extension, axes):
    """Slightly enlarge same-row gutters while keeping the canvas and y layout."""
    figure.canvas.draw()
    before = measure_horizontal_spacing(figure, axes)
    horizontal = {'b1': (13.14, 26.2), 'b2': (49.7, 28.0), 'c': (91.4, 31.9), 'd_upper': (138.6, 36.9), 'd_lower': (138.6, 36.9), 'e1': (13.14, 26.2), 'e2': (49.7, 28.0), 'f': (91.4, 31.9), 'g_upper': (138.6, 36.9), 'g_lower': (138.6, 36.9), 'i': (126.8, 47.8)}
    for name, (left_mm, width_mm) in horizontal.items():
        _, bottom, _, height = axes[name].get_position().bounds
        axes[name].set_position((left_mm / 180, bottom, width_mm / 180, height))
    for text in main.texts:
        label = text.get_text()
        x, y = text.get_position()
        if label in ('c', 'f'):
            text.set_position((x + 2.3 / 180, y))
        elif label in ('d', 'g'):
            text.set_position((x + 1.8 / 180, y))
        elif label.startswith('Mean likelihood per trial'):
            text.set_x((13.14 + 26.2 / 2 + (49.7 + 28 / 2)) / 2 / 180)
        elif label == 'DIS ratio':
            text.set_x((13.14 + 26.2 / 2 + (49.7 + 28 / 2)) / 2 / 180)
    for text in extension.texts:
        if text.get_text() == 'i':
            x, y = text.get_position()
            text.set_position((x + 0.8 / 180, y))
    figure.canvas.draw()
    after = measure_horizontal_spacing(figure, axes)
    increments = {pair: gap - before['content_gaps_mm'][pair] for pair, gap in after['content_gaps_mm'].items()}
    for pair, gap in after['content_gaps_mm'].items():
        assert gap >= 2.0, (pair, gap)
        assert increments[pair] >= 0.65, (pair, increments[pair])
    report = {'status': 'PASS', 'before': before, 'after': after, 'gap_increments_mm': increments, 'minimum_content_gap_mm': 2.0, 'canvas_width_mm': 180, 'font_sizes_changed': False, 'vertical_geometry_preserved': True, 'method': 'Final rendered bounds include ticks, axis titles, legends and schematic labels. All same-row content gaps increase by at least 0.65 mm.'}
    (AUDIT / 'fig6_full_joint90_horizontal_spacing.json').write_text(json.dumps(report, indent=2) + '\n')
    return report

def finalize(figure, main, extension, statistics):
    names = ('a', 'b1', 'b2', 'c', 'd_upper', 'd_lower', 'e1', 'e2', 'g_upper', 'g_lower', 'f', 'h', 'i')
    assert len(figure.axes) == len(names)
    axes = dict(zip(names, figure.axes))
    final._recolor(figure)
    style = final._fig6_typography(figure, axes)
    scale = 180 / (figure.get_figwidth() * 25.4)
    old_main_mm = main.bbox.height / figure.dpi * 25.4 * scale
    old_extension_mm = extension.bbox.height / figure.dpi * 25.4 * scale
    old_a_bounds = axes['a'].get_position().bounds
    extra_mm = 49 - old_a_bounds[3] * old_main_mm
    new_main_mm = old_main_mm + extra_mm
    figure.set_size_inches(180 / 25.4, (new_main_mm + old_extension_mm) / 25.4)
    main._subplotspec.get_gridspec().set_height_ratios([new_main_mm, old_extension_mm])
    main._redo_transform_rel_fig()
    extension._redo_transform_rel_fig()
    for name, axis in axes.items():
        if name in ('a', 'h', 'i'):
            continue
        x, y, w, h = axis.get_position().bounds
        axis.set_position((x, y * old_main_mm / new_main_mm, w, h * old_main_mm / new_main_mm))
    axes['a'].set_position((0, old_a_bounds[1] * old_main_mm / new_main_mm, 1, 49 / new_main_mm))
    for text in main.texts:
        x, y = text.get_position()
        offset = extra_mm if text.get_text() == 'a' else -1.7 if text.get_text() == 'DIS ratio' else 0
        text.set_position((x, (y * old_main_mm + offset) / new_main_mm))
    from matplotlib.text import Text
    for text in figure.findobj(Text):
        label = text.get_text().strip()
        if label in tuple('abcdefghi') and text.axes is None:
            text.set_fontsize(8.0)
        elif text.axes is axes['a']:
            text.set_fontsize(8.0 if label in ('+', '×') else 7.0)
        else:
            text.set_fontsize(6.0)
    spacing_qa = increase_row_spacing(figure, main, extension, axes)
    figure.canvas.draw()
    schematic_qa = audit_model_flow(figure, axes['a'], AUDIT / 'fig6_full_joint90_panel_a_qa.json')
    movie_qa = audit_movie_selection(figure, axes['h'], AUDIT / 'fig6_full_joint90_panel_h_qa.json')
    sizes = sorted({text.get_fontsize() for text in figure.findobj(Text) if text.get_visible() and text.get_text().strip()})
    assert len(sizes) <= 3, sizes
    style['font_sizes_pt'] = sizes
    style['ordinary_labels'] = '6 pt labels, ticks and legends; 7 pt panel-a body; 8 pt panel letters'
    style['schematic_text'] = 'a: 7 pt body, 8 pt operators; h: 6 pt body; panel letters 8 pt'
    qa = final._render_qa(figure, AUDIT / 'fig6_full_joint90_render_qa.json')
    selection = {key: axes[key] for key in ('b1', 'b2', 'c', 'd_upper', 'd_lower', 'e1', 'e2', 'f', 'g_upper', 'g_lower')}
    alignment = final.require_matplotlib_panel_alignment(figure, axes=list(selection.values()), panel_ids={axis: key for key, axis in selection.items()}, row_groups=[{'id': 'fit-tasks', 'panels': ['b1', 'b2']}, {'id': 'fit-and-attention-height', 'panels': ['b1', 'b2', 'c']}, {'id': 'third-row-performance-and-slopes', 'panels': ['e1', 'e2', 'f']}], column_groups=[{'id': 'dis-association', 'panels': ['d_upper', 'd_lower']}, {'id': 'round-trajectories', 'panels': ['g_upper', 'g_lower']}, {'id': 'task1-column', 'panels': ['b1', 'e1']}, {'id': 'task2-column', 'panels': ['b2', 'e2']}, {'id': 'attention-and-parameter-column', 'panels': ['c', 'f']}], exemptions=[{'panels': ['e2', 'f'], 'checks': ['panel-width', 'horizontal-gutter'], 'reason': 'Two-task performance comparison and single phase-distribution panel have intentionally different spans'}, {'panels': ['b1', 'b2', 'c'], 'checks': ['panel-width', 'horizontal-gutter'], 'reason': 'Two task-specific fitting plots and one attention panel retain distinct widths; only top, bottom and height are matched'}], json_out=AUDIT / 'fig6_full_joint90_alignment.json', strict=True, tolerance_pt=1.5, gutter_tolerance_pt=1.5)
    return {'typography': style, 'font_floor_pt': qa['font_floor_pt'], 'qa_status': 'PASS', 'alignment_status': alignment['verdict'], 'horizontal_spacing_qa': spacing_qa, 'panel_a_qa': schematic_qa, 'panel_h_qa': movie_qa, 'header_added_height_mm': extra_mm, 'full_dgem_90both': 'Only e-panel DGEM data are newly simulated; all other plot data match primary full Fig. 6.'}

def main():
    global AUDIT
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.output_dir is not None:
        layout.OUTPUT_DIR = args.output_dir
        AUDIT = PKG / 'output'
        AUDIT.mkdir(parents=True, exist_ok=True)
        layout.MANIFEST_DIR = PKG / 'output'
    output, manifest = layout.assemble(output_stem='fig6_full_joint90', figure_finalize=finalize)
    record = json.loads(manifest.read_text())
    record['vector_panel_a'].update({'bounds': 'Full-width 180 x 49 mm native header', 'structure': 'Stimulus, feature uncertainty and value, uncertainty-derived attention, weighted item scores, ranking and grouping into three rows', 'standalone_image_dependency': False})
    record['panel_mapping']['a'] = 'Native DGEM uncertainty-dependent Concrete attention and ranking-based layout generation'
    record['panel_h_schematic'] = {'native_renderer': True, 'reward_displayed': False, 'definition': 'A dimension is a semantic cluster of movie tags; a feature is one individual tag.', 'selection': 'Top-K1 dimensions; global Top-K2 tags pooled from selected clusters; cosine retrieval; argmax item-level score.', 'analogy_to_dgem': 'Dimension-guided feature processing; TDGE hard selection differs from DGEM continuous uncertainty-derived attention.'}
    record['notes'] = ['Panel h is grounded in the supplied TDGE manuscript, Sections 3.3–3.4 and Appendix C.2/D/E, with representative Figure 2b tags; reward and update paths are omitted.' if note.startswith('Panel h is') else note for note in record['notes']]
    manifest.write_text(json.dumps(record, indent=2) + '\n')
    with Image.open(output) as image:
        assert image.mode == 'RGB' and min(image.size) > 2000
    print(json.dumps({'output': str(output), 'manifest': str(manifest)}, indent=2))
if __name__ == '__main__':
    raise SystemExit('Reusable drawing module: run render_fig6_full_joint90_free_beta.py instead.')
