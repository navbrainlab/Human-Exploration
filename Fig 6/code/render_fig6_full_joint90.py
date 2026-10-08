"""Render a separate full-DGEM Fig. 6 with independently validated 90/90 e data."""
from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from PIL import Image
PKG = Path(__file__).resolve().parents[1]
MAIN = PKG / 'output'
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

def draw_reference_flow(figure, bounds):
    """Reconstruct the supplied panel-a reference with native editable artists."""
    from matplotlib.patches import FancyBboxPatch, Ellipse
    axis = figure.add_axes(bounds)
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.set_axis_off()
    ink, edge, purple = ('#17243B', '#64768B', '#6F55A5')
    ratio = figure.bbox.width * bounds[2] / (figure.bbox.height * bounds[3])

    def box(x, y, w, h, label, attention=False):
        patch = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.004', linewidth=0.75, edgecolor=purple if attention else edge, facecolor='#F0EAF8' if attention else '#F1F6FA', zorder=3)
        patch.set_gid('flow-box')
        axis.add_patch(patch)
        text = axis.text(x + w / 2, y + h / 2, label, ha='center', va='center', fontsize=6, color=ink, fontweight='semibold' if attention else 'normal', linespacing=1.12, zorder=4)
        text.set_gid('flow-label')
        text._flow_width = w

    def arrow(*points, attention=False):
        layout._orthogonal_arrow(axis, points, color=purple if attention else edge, linewidth=0.8 if attention else 0.65)

    def operator(x, y, symbol):
        axis.add_patch(Ellipse((x, y), 0.026, 0.026 * ratio, facecolor='white', edgecolor=edge, linewidth=0.75, zorder=4))
        axis.text(x, y, symbol, ha='center', va='center', fontsize=8, color=ink, zorder=5)
    box(0.005, 0.28, 0.14, 0.16, 'For each feature ($f$)\nin dimension ($d$)')
    box(0.198, 0.4, 0.13, 0.16, 'Feature uncertainty\n($U_{d,f}$)')
    box(0.198, 0.2, 0.13, 0.1, 'Feature value ($V_{d,f}$)')
    box(0.362, 0.6, 0.13, 0.22, 'Softmax with\nattention sharpness\n($\\tau_{att}$)', True)
    box(0.525, 0.6, 0.142, 0.22, 'Attention weights on\nfeature dimensions\n($w_d$)', True)
    box(0.465, 0.27, 0.118, 0.16, 'Feature score\n($S_{d,f}$)')
    box(0.697, 0.27, 0.102, 0.16, 'Action score\n($S_A$)')
    box(0.827, 0.27, 0.074, 0.16, 'Softmax\npolicy')
    box(0.926, 0.27, 0.064, 0.16, 'Action\n($A_t$)')
    operator(0.402, 0.35, '+')
    operator(0.637, 0.35, '$\\times$')
    arrow((0.145, 0.375), (0.174, 0.375), (0.174, 0.48), (0.198, 0.48))
    arrow((0.145, 0.335), (0.174, 0.335), (0.174, 0.25), (0.198, 0.25))
    arrow((0.328, 0.48), (0.357, 0.48), (0.357, 0.37), (0.389, 0.37))
    arrow((0.328, 0.25), (0.357, 0.25), (0.357, 0.33), (0.389, 0.33))
    arrow((0.415, 0.35), (0.465, 0.35))
    arrow((0.583, 0.35), (0.624, 0.35))
    arrow((0.65, 0.35), (0.697, 0.35))
    arrow((0.799, 0.35), (0.827, 0.35))
    arrow((0.901, 0.35), (0.926, 0.35))
    arrow((0.328, 0.525), (0.338, 0.525), (0.338, 0.72), (0.362, 0.72), attention=True)
    arrow((0.492, 0.72), (0.525, 0.72), attention=True)
    arrow((0.667, 0.72), (0.682, 0.72), (0.682, 0.49), (0.637, 0.49), (0.637, 0.35 + 0.013 * ratio), attention=True)
layout.draw_model_flow_panel = draw_reference_flow

def finalize(figure, main, extension, statistics):
    names = ('a', 'b1', 'b2', 'c', 'd_upper', 'd_lower', 'e1', 'e2', 'g_upper', 'g_lower', 'f', 'h', 'i')
    assert len(figure.axes) == len(names)
    axes = dict(zip(names, figure.axes))
    final._recolor(figure)
    style = final._fig6_typography(figure, axes)
    scale = 180 / (figure.get_figwidth() * 25.4)
    figure.set_size_inches(*figure.get_size_inches() * scale)
    from matplotlib.text import Text
    figure.canvas.draw()
    for text in figure.findobj(Text):
        label = text.get_text().strip()
        if label in tuple('abcdefghi') and text.axes is None:
            text.set_fontsize(8.0)
        else:
            text.set_fontsize(6.0 if text.get_fontsize() >= 6.0 else 5.0)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for text in axes['a'].texts:
        width = getattr(text, '_flow_width', None)
        if width is not None and text.get_window_extent(renderer).width > axes['a'].bbox.width * width * 0.98:
            text.set_fontsize(5.0)
    sizes = sorted({text.get_fontsize() for text in figure.findobj(Text) if text.get_visible() and text.get_text().strip()})
    assert len(sizes) <= 3, sizes
    style['font_sizes_pt'] = sizes
    style['ordinary_labels'] = '6 pt main text; 5 pt compact labels, ticks and legends'
    style['schematic_text'] = '5 or 6 pt; panel letters 8 pt'
    qa = final._render_qa(figure, layout.OUTPUT_DIR / 'fig6_full_joint90_render_qa.json')
    selection = {key: axes[key] for key in ('b1', 'b2', 'c', 'd_upper', 'd_lower', 'e1', 'e2', 'f', 'g_upper', 'g_lower')}
    alignment = final.require_matplotlib_panel_alignment(figure, axes=list(selection.values()), panel_ids={axis: key for key, axis in selection.items()}, row_groups=[{'id': 'fit-tasks', 'panels': ['b1', 'b2']}, {'id': 'performance-and-temperature', 'panels': ['e1', 'e2', 'f']}], column_groups=[{'id': 'task1', 'panels': ['b1', 'e1']}, {'id': 'task2', 'panels': ['b2', 'e2']}, {'id': 'dis-association', 'panels': ['d_upper', 'd_lower']}, {'id': 'round-trajectories', 'panels': ['g_upper', 'g_lower']}], exemptions=[{'panels': ['e2', 'f'], 'checks': ['panel-width', 'horizontal-gutter'], 'reason': 'Established unequal task-specific and parameter-panel spans'}], json_out=layout.OUTPUT_DIR / 'fig6_full_joint90_alignment.json', strict=True, tolerance_pt=1.5, gutter_tolerance_pt=1.5)
    return {'typography': style, 'font_floor_pt': qa['font_floor_pt'], 'qa_status': 'PASS', 'alignment_status': alignment['verdict'], 'full_dgem_90both': 'Only e-panel DGEM data are newly simulated; all other plot data match primary full Fig. 6.'}

def main():
    output, manifest = layout.assemble(output_stem='fig6_full_joint90', figure_finalize=finalize)
    with Image.open(output) as image:
        assert image.mode == 'RGB' and min(image.size) > 2000
    print(json.dumps({'output': str(output), 'manifest': str(manifest)}, indent=2))
if __name__ == '__main__':
    main()
