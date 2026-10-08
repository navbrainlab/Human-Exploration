"""Render a separate evidence-led Fig. 5; keep the current primary PNG intact."""
from pathlib import Path
import sys, json, hashlib
from matplotlib.text import Text
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import publication_style as final
import figure5_evidence_layout as layout
PKG = Path(__file__).resolve().parents[1]
layout.CAPTION_PATH = PKG / 'caption.md'
layout.METHODS_PATH = PKG / 'methods.md'
layout.original.DPI = 600
STEM = 'fig5_nhb_evidence_layout'
STAT_PLOT_HEIGHT_MM = 30.0
previous_height = layout.PLOT_HEIGHT
layout.PLOT_HEIGHT = STAT_PLOT_HEIGHT_MM / 141
layout.ROW_BOTTOMS = tuple((bottom if row == 0 else bottom + previous_height - layout.PLOT_HEIGHT for row, bottom in enumerate(layout.ROW_BOTTOMS)))

def finalize(figure, panels, labels, statistics):
    final._recolor(figure)
    from matplotlib.collections import PathCollection
    import numpy as np
    for name in 'cdefghi':
        axis = panels[name]
        for patch in axis.patches:
            rgb = patch.get_facecolor()[:3]
            patch.set_alpha(None)
            patch.set_facecolor((*rgb, 0.4 if name in 'cdef' else 0.58))
            patch.set_edgecolor('#4C535A')
            patch.set_linewidth(0.75)
        if name in 'cdef':
            for line in axis.lines:
                if line.get_alpha() == 0.065:
                    line.set_alpha(0.14)
                    line.set_linewidth(0.4)
                elif np.isclose(line.get_linewidth(), 0.55):
                    line.set_linewidth(0.75)
                elif np.isclose(line.get_linewidth(), 1.05):
                    line.set_linewidth(1.1)
            for collection in axis.collections:
                if isinstance(collection, PathCollection) and collection.get_alpha() == 0.32:
                    collection.set_alpha(0.48)
    for line in panels['j'].lines:
        if np.isclose(line.get_linewidth(), 0.85):
            line.set_linewidth(1.0)
    for collection in panels['j'].collections:
        collection.set_linewidth(0.9)
    panels['c'].set_ylabel('Gaze coverage area\n(deg²)', labelpad=2.5)
    panels['e'].set_ylabel('Gaze-shift speed\n(deg s⁻¹)', labelpad=2.5)
    for text in figure.texts:
        if text.get_text() in ('DIS trials', 'others'):
            text.set_color(final.INK)
    final._readable_axes([axis for axis in figure.axes if axis.axison])
    figure.canvas.draw()
    for text in figure.findobj(Text):
        text.set_fontsize(8 if text.axes is None and text.get_text().strip() in tuple('abcdefghij') else 6 if text.get_fontsize() >= 6 else 5)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    top = 0.944
    width = 44.0 / 180
    height = 44.0 * 577 / 770 / 141
    target_bottom = top - height
    image_axes = [axis for axis in figure.axes if axis.images and axis is not panels['b']]
    assert len(image_axes) == 2
    for axis, left in zip(image_axes, (0.07, 0.33)):
        axis.set_position((left, target_bottom, width, height))
    for text in figure.texts:
        if text.get_text() in ('DIS trials', 'others'):
            left = 0.07 if text.get_text() == 'DIS trials' else 0.33
            text.set_x(left + width / 2)
    density = next((axis for axis in figure.axes if axis.get_label() == 'fig5a_shared_colorbar'))
    density.set_position((0.33 + width + 0.01, target_bottom, 0.008, height))
    density.yaxis.set_label_position('right')
    density.yaxis.label.set_rotation(270)
    density.yaxis.set_label_coords(2.3, 0.5)
    density.yaxis.label.set_fontsize(5)
    panels['b'].set_aspect('auto')
    panels['b'].set_position((0.71, top - 30.6 / 141, 37.0 / 180, 30.6 / 141))
    selectivity_bar = next((axis for axis in figure.axes if axis.get_label() == 'fig5c_colorbar'))
    selectivity_bar.set_position((0.71 + 37.0 / 180 + 0.01, panels['b'].get_position().y0, 0.008, panels['b'].get_position().height))
    selectivity_bar.set_title('', loc='right')
    selectivity_bar.yaxis.set_label_position('right')
    selectivity_bar.set_ylabel('Dwell-time proportion', fontsize=5, fontweight='normal', rotation=270, labelpad=6, va='center')
    for text in figure.texts:
        if text.get_text() == 'a':
            text.set_x(0.05)
        if text.get_text() == 'b':
            text.set_x(0.635)
    for axis in figure.axes:
        if axis.get_label().startswith('source_eye_tracking_') and abs(axis.get_position().y1 - 0.985) < 0.01 and (axis.get_position().x0 < 0.2):
            pos = axis.get_position()
            axis.set_position((0.05 + 3.8 / 180, pos.y0, pos.width, pos.height))
    for axis in figure.axes:
        if axis.get_label().startswith('source_eye_tracking_') and abs(axis.get_position().y1 - 0.985) < 0.01 and (axis.get_position().x0 > 0.6):
            pos = axis.get_position()
            axis.set_position((0.635 + 3.8 / 180, pos.y0, pos.width, pos.height))
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for axis in image_axes:
        assert abs(axis.get_window_extent(renderer).y1 - panels['b'].get_window_extent(renderer).y1) < 0.5
    image_audit = {'reference_proportions': True, 'top_aligned_to_b': True, 'height_mm': height * 141, 'bounds': [list(axis.get_position().bounds) for axis in image_axes], 'source_aspect_ratio_preserved': True, 'b_size_mm': [37.0, 30.6]}
    label_box = selectivity_bar.yaxis.label.get_window_extent(renderer)
    tick_boxes = [text.get_window_extent(renderer) for text in selectivity_bar.get_yticklabels() if text.get_visible() and text.get_text().strip()]
    assert all((label_box.x0 > bbox.x1 for bbox in tick_boxes))
    assert label_box.x1 < figure.bbox.x1
    colorbar_audit = {'label': 'Dwell-time proportion', 'placement': 'Right of tick labels, vertically centred', 'rotation_degrees': 270, 'fontsize_pt': 5, 'tick_label_gap_mm': (label_box.x0 - max((b.x1 for b in tick_boxes))) * 25.4 / figure.dpi, 'right_canvas_margin_mm': (figure.bbox.x1 - label_box.x1) * 25.4 / figure.dpi, 'normalization': [0, 0.3], 'tick_values': [0, 0.1, 0.2, 0.3]}
    annotation = layout._align_bottom_annotations(figure, panels)
    plot_bounds = {name: list(panels[name].get_position().bounds) for name in 'cdefghij'}
    heights = {name: bounds[3] * 141 for name, bounds in plot_bounds.items()}
    widths = {name: bounds[2] * 180 for name, bounds in plot_bounds.items()}
    np.testing.assert_allclose(list(heights.values()), STAT_PLOT_HEIGHT_MM, atol=0.01)
    np.testing.assert_allclose(list(widths.values()), 27.9, atol=0.01)
    renderer = figure.canvas.get_renderer()
    icons = [axis for axis in figure.axes if axis.get_label().startswith('source_')]
    label_icon_overlaps = [{'panel': name, 'icon': icon.get_label()} for name in 'cdefghij' for icon in icons if panels[name].yaxis.label.get_window_extent(renderer).overlaps(icon.get_window_extent(renderer))]
    assert not label_icon_overlaps, label_icon_overlaps
    geometry = {'plot_heights_mm': heights, 'plot_widths_mm': widths, 'row_top_edges_mm_from_bottom': {row: [panels[name].get_position().y1 * 141 for name in row] for row in ('cdef', 'ghij')}, 'top_edges_preserved': True, 'all_eight_equal_height': True, 'ylabel_icon_overlaps': label_icon_overlaps, 'wrapped_ylabels': {'c': 'Gaze coverage area\n(deg²)', 'e': 'Gaze-shift speed\n(deg s⁻¹)'}}
    qa = final._render_qa(figure, layout.DATA_DIR / (STEM + '_render_qa.json'))
    sizes = sorted({t.get_fontsize() for t in figure.findobj(Text) if t.get_visible() and t.get_text().strip()})
    assert sizes == [5, 6, 8]
    return {'panel_a_alignment': image_audit, 'panel_b_colorbar_label': colorbar_audit, 'statistical_panel_geometry': geometry, 'primary_style_matched': {'paired_box_alpha': 0.4, 'observation_box_alpha': 0.58, 'paired_line_alpha': 0.14, 'paired_point_alpha': 0.48}, 'font_sizes_pt': sizes, 'font_floor_pt': qa['font_floor_pt'], 'bottom_row_height': annotation, 'panel_order_from_primary': {'a': 'a', 'b': 'c', 'c': 'b', 'd': 'd', 'e': 'e', 'f': 'f', 'g': 'h', 'h': 'i', 'i': 'g', 'j': 'j'}}

def main():
    paths = layout.assemble(output_stem=STEM, figure_finalize=finalize)
    manifest = json.loads(paths[-1].read_text())
    image_layout = manifest['edition_audit']['panel_a_alignment']
    manifest['panel_a']['image_bounds_fraction'] = dict(zip(('DIS', 'others'), image_layout['bounds']))
    manifest['panel_a']['image_size_mm'] = [image_layout['bounds'][0][2] * 180, image_layout['height_mm']]
    manifest['panel_a']['shared_colorbar']['bounds_fraction'] = [0.33 + image_layout['bounds'][0][2] + 0.01, image_layout['bounds'][0][1], 0.008, image_layout['bounds'][0][3]]
    manifest['panel_a']['shared_colorbar']['size_mm'] = [1.44, image_layout['height_mm']]
    manifest['layout']['panel_letter_positions']['b'][0] = 0.635
    manifest['documentation'] = {name: {'path': str((PKG / name).resolve()), 'sha256': hashlib.sha256((PKG / name).read_bytes()).hexdigest()} for name in ('caption.md', 'methods.md')}
    manifest['renderer'] = {'path': str(Path(__file__).resolve()), 'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    manifest['code_sources'] = [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in (Path(__file__).resolve(), Path(layout.__file__))]
    paths[-1].write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'outputs': [str(p) for p in paths]}, indent=2))
if __name__ == '__main__':
    main()
