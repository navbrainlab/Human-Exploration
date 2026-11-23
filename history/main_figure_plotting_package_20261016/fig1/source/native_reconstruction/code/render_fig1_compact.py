"""Fig1 with original PPTX panels a/b and compound behavioral panels c/e."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from matplotlib.text import Text
from matplotlib.transforms import Bbox

import render_fig1_bar_box as legacy
import pptx_original_panels as pptx_task_design
from axis_break_marks import audit_yaxis_breaks, mark_nonzero_yaxis
from detached_axes import detach_axes, audit_detached_axes

base = legacy.base
NAME = legacy.NAME
SOURCE_PANEL_MAP = dict(zip("abcdefgh", "abcdeghj"))
LAYOUT_PANEL_MAP = dict(zip("abcdefgh", "ababcdef"))
DISPLAY_PANEL_MAP = dict(zip("abcdefgh", "ababccee"))
PUBLIC_SOURCE_PANEL_MAP = {"a": ["a", "c"], "b": ["b", "d"],
                           "c": ["e", "g"], "e": ["h", "j"]}
PUBLIC_PANEL_LETTERS = list("abce")
SOURCE_ICON_SIZE_MM = 4.8
SOURCE_ICON_OFFSET_MM = 3.8
LOWER_KEYS = "efgh"
TASK1_LEARNING_WIDTH_MM = 36.
TASK2_LEARNING_WIDTH_MM = 33.
BOX_WIDTH_MM = 27.9
PLOT_HEIGHT_MM = 30.
CONTENT_GAP_MM = 2.
TASK_GAP_MM = 5.
GLOBAL_X_SHIFT_MM = .85
PPTX_X_SHIFT_MM = 24 * 25.4 / 600
# Integer-pixel shifts preserve the original task-flow raster grid.
TIER_SHIFT_MM = 71 * 25.4 / 600
PPTX_ADDED_HEIGHT_MM = 820 * 25.4 / 600
CANVAS_HEIGHT_MM = 206. - 1276 * 25.4 / 600 + PPTX_ADDED_HEIGHT_MM


def check_inputs():
    hashes = json.loads((base.ROOT / "plot_input_hashes.json").read_text())
    for name, value in hashes.items():
        assert hashlib.sha256((base.CSV / name).read_bytes()).hexdigest() == value, name
    return hashes


def heading(fig, source, x, y, title):
    before = len(fig.texts)
    base.panel_heading(fig, LAYOUT_PANEL_MAP[source], x, y, title)
    fig._headings[source] = fig.texts[before:]
    fig._headings[source][0].set_gid("nhb-panel-letter")


def row_bounds(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    scale = 25.4 / fig.dpi
    bounds = {}
    for key in LOWER_KEYS:
        boxes = [fig._panels[key].get_tightbbox(renderer)]
        boxes.extend(t.get_window_extent(renderer) for t in fig._headings[key])
        bounds[key] = (np.asarray(Bbox.union(boxes).extents) * scale).tolist()
    return dict(bounds_mm=bounds, gaps_mm=[bounds[right][0] - bounds[left][2]
                for left, right in zip(LOWER_KEYS, LOWER_KEYS[1:])])


def shift_figure_x(fig, millimetres):
    shift = millimetres / 180
    for axis in fig._panels.values():
        x, y, width, height = axis.get_position().bounds
        axis.set_position((x + shift, y, width, height))
    for item in fig.texts:
        x, y = item.get_position()
        item.set_position((x + shift, y))


def raster_ink_margins(path):
    with base.Image.open(path) as raster:
        pixels = np.asarray(raster.convert("RGB"))
    occupied = np.flatnonzero(np.any(pixels < 245, axis=2).sum(axis=0) > 20)
    if len(occupied) == 0:
        raise ValueError("Rendered figure contains no visible ink")
    millimetres_per_pixel = 25.4 / 600
    return [float(occupied[0] * millimetres_per_pixel),
            float(180 - occupied[-1] * millimetres_per_pixel)]


def balance_row(fig):
    initial = row_bounds(fig)
    previous_right = initial["bounds_mm"]["e"][2]
    shifts = {"e": 0.}
    for key in "fgh":
        left, _, right, _ = initial["bounds_mm"][key]
        gap = TASK_GAP_MM if key == "g" else CONTENT_GAP_MM
        dx = previous_right + gap - left
        shifts[key] = float(dx)
        axis = fig._panels[key]
        x, y, width, height = axis.get_position().bounds
        axis.set_position((x + dx / 180, y, width, height))
        for text in fig._headings[key]:
            x, y = text.get_position()
            text.set_position((x + dx / 180, y))
        previous_right = right + dx
    final = row_bounds(fig)
    np.testing.assert_allclose(final["gaps_mm"], [CONTENT_GAP_MM, TASK_GAP_MM, CONTENT_GAP_MM], atol=1e-7)
    assert final["bounds_mm"]["h"][2] <= 179, final
    page_margins = [final["bounds_mm"]["e"][0], 180 - final["bounds_mm"]["h"][2]]
    assert min(page_margins) > 1., page_margins
    fig._compact_spacing = dict(before=initial, after=final, shifts_mm=shifts,
                               page_margins_mm=page_margins,
                               definition="Visible axes/ticks/labels/legends/headings, not just plot rectangles")


def group_outcome_panels(fig):
    """Remove legacy headers after layout to keep all data panels in place."""
    source = base.ROOT / "assets/approved_source_icons.py"
    spec = importlib.util.spec_from_file_location("fig1_approved_source_icons", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fig._source_icons = {}
    for key in LOWER_KEYS:
        keep = fig._headings[key][:1] if key in "eg" else []
        for text in fig._headings[key][len(keep):]:
            text.remove()
        fig._headings[key] = keep
        if not keep:
            continue
        letter = keep[0]
        letter.set_text(DISPLAY_PANEL_MAP[key])
        x, y = letter.get_position()
        fig._source_icons[key] = module.add_source_icon(
            fig, "task_behavior", left=x + SOURCE_ICON_OFFSET_MM / 180,
            top=y, size_mm=SOURCE_ICON_SIZE_MM)
    fig._legacy_layout_spacing = fig._compact_spacing
    visible = row_bounds(fig)
    assert min(visible["gaps_mm"]) >= CONTENT_GAP_MM - 1e-7
    fig._compact_spacing = dict(
        visible=visible, physical_plot_positions_preserved=True,
        definition="Title-free axes/ticks/labels/legends and shared c/e letters")


def build_figure():
    check_inputs()
    pptx_task_design.check_assets()
    fig = base.make_canvas(CANVAS_HEIGHT_MM)
    fig._headings = {}
    for letter, x, draw in [("a", 3, pptx_task_design.task1),
                             ("b", 94, pptx_task_design.task2)]:
        heading(fig, letter, x, 3, "")
        axis = base.axis(fig, letter, x, 8.5, pptx_task_design.PANEL_WIDTH_MM,
                         pptx_task_design.PANEL_HEIGHT_MM, schematic=True)
        draw(axis)
    for task, learning_key, box_key, left in [(1, "e", "f", 12.), (2, "g", "h", 101.)]:
        heading(fig, learning_key, left - 10, 99 - TIER_SHIFT_MM + PPTX_ADDED_HEIGHT_MM, f"Task {task} | learning")
        learning_width = TASK1_LEARNING_WIDTH_MM if task == 1 else TASK2_LEARNING_WIDTH_MM
        axis = base.axis(fig, learning_key, left, 108 - TIER_SHIFT_MM + PPTX_ADDED_HEIGHT_MM,
                         learning_width, PLOT_HEIGHT_MM)
        base.learning(axis, task)
        if task == 1:
            axis.set(xlim=(0, 50), xticks=legacy.TASK1_LEARNING_TICKS)
        heading(fig, box_key, left + 36, 99 - TIER_SHIFT_MM + PPTX_ADDED_HEIGHT_MM, "Personal best")
        axis = base.axis(fig, box_key, left + 46, 108 - TIER_SHIFT_MM + PPTX_ADDED_HEIGHT_MM, BOX_WIDTH_MM, PLOT_HEIGHT_MM)
        legacy.personal_best_box(axis, task)
    fig._rows = [["a", "b"], list(LOWER_KEYS)]
    fig._columns = []
    legacy.match_fig5_style(fig)
    for key in LOWER_KEYS:
        mark_nonzero_yaxis(fig._panels[key], panel=DISPLAY_PANEL_MAP[key])
    shift_figure_x(fig, GLOBAL_X_SHIFT_MM)
    for key in "ab":
        ax = fig._panels[key]
        x, y, width, height = ax.get_position().bounds
        ax.set_position((x + PPTX_X_SHIFT_MM / 180, y, width, height))
        for text in fig._headings[key]:
            x, y = text.get_position()
            text.set_position((x + PPTX_X_SHIFT_MM / 180, y))
    balance_row(fig)
    group_outcome_panels(fig)
    for key in LOWER_KEYS:
        detach_axes(fig._panels[key])
    visible = row_bounds(fig)
    assert min(visible["gaps_mm"]) >= CONTENT_GAP_MM - 1e-7, visible
    fig._compact_spacing = dict(visible=visible, physical_plot_positions_preserved=True,
        definition="Final detached axes/ticks/labels/legends and shared c/e letters")
    return fig


def audit(fig):
    """Verify frozen values and native artists independently of public letters."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    stats = legacy.check_statistics()
    curves = base.read("fig1_learning_display.csv")
    personal = base.read("fig1_reference_personal_best.csv")
    counts, tests, fingerprints, widths = {}, {}, {}, {}
    for task, learning_key, box_key, source_box in [(1, "e", "f", "g"), (2, "g", "h", "j")]:
        config = base.TASK_CONFIG[task]
        learning, box = [fig._panels[key] for key in (learning_key, box_key)]
        fingerprints[learning_key] = []
        for group, line, band in zip(config["groups"], learning.lines, learning.collections):
            rows = curves.loc[curves.task.eq(task) & curves.group.eq(group)]
            np.testing.assert_allclose(line.get_xdata(), rows.trial)
            np.testing.assert_allclose(line.get_ydata(), rows.mean_smooth)
            vertices = band.get_paths()[0].vertices
            np.testing.assert_allclose([vertices[:, 1].min(), vertices[:, 1].max()],
                [(rows.mean_smooth - rows["sem"]).min(), (rows.mean_smooth + rows["sem"]).max()])
            fingerprints[learning_key].append(dict(group=group,
                line_sha256=hashlib.sha256(np.column_stack([line.get_xdata(), line.get_ydata()]).tobytes()).hexdigest(),
                band_sha256=hashlib.sha256(vertices.tobytes()).hexdigest()))
        counts[box_key] = []
        assert not box.collections
        for index, group in enumerate(config["groups"]):
            values = personal.loc[personal.task.eq(task) & personal.group.eq(group), "first_best_trial"].to_numpy()
            counts[box_key].append(len(values))
            artists = box._restored_boxes
            vertices = artists["boxes"][index].get_path().vertices
            np.testing.assert_allclose([vertices[:, 1].min(), vertices[:, 1].max()], np.percentile(values, [25, 75]))
            np.testing.assert_allclose(artists["medians"][index].get_ydata(), np.median(values))
        assert all(len(line.get_xdata()) == 0 for line in artists["fliers"])
        tests[box_key] = {**stats["tests"][source_box], "previous_panel": source_box}
        assert [text.get_text() for text in box.texts] == [base.stars(tests[box_key]["p_raw"])]
        widths[box_key] = [patch.get_window_extent(renderer).width * 25.4 / fig.dpi for patch in box.patches]
        np.testing.assert_allclose(widths[box_key], 4.95, atol=1e-7)
    assert counts == {"f": [51, 51, 54], "h": [40, 40]}
    np.testing.assert_allclose(fig._panels["e"].get_ylim(), [50, 100])
    np.testing.assert_allclose(fig._panels["g"].get_ylim(), [45, 90])
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    sizes = sorted({t.get_fontsize() for t in texts})
    assert sizes == [5., 6., 8.], sizes
    letters = [t.get_text() for t in fig.texts if t.get_gid() == "nhb-panel-letter"]
    assert letters == PUBLIC_PANEL_LETTERS, letters
    assert all(t.get_gid() == "nhb-panel-letter" or not t.get_text().strip()
               for t in fig.texts), "Outcome titles must be absent"
    source_icons = {}
    for key, icon in fig._source_icons.items():
        letter = fig._headings[key][0]
        anchor = letter.get_transform().transform(letter.get_position())
        offset = (np.array([icon.bbox.x0, icon.bbox.y1]) - anchor) * 25.4 / fig.dpi
        np.testing.assert_allclose(offset, [SOURCE_ICON_OFFSET_MM, 0], atol=1e-7)
        size = np.array([icon.bbox.width, icon.bbox.height]) * 25.4 / fig.dpi
        np.testing.assert_allclose(size, SOURCE_ICON_SIZE_MM, atol=1e-7)
        for text in texts:
            assert not icon.bbox.overlaps(text.get_window_extent(renderer)), text.get_text()
        assert icon.bbox.y0 > fig._panels[key].bbox.y1
        source_icons[DISPLAY_PANEL_MAP[key]] = dict(
            kind="task_behavior", size_mm=size.tolist(),
            letter_offset_mm=offset.tolist(), native_render=True,
            embedded_raster_images=len(icon.images),
            code="assets/approved_source_icons.py",
            code_sha256=hashlib.sha256((base.ROOT / "assets/approved_source_icons.py").read_bytes()).hexdigest())
    collisions = []
    for index, first in enumerate(texts):
        a = first.get_window_extent(renderer)
        for second in texts[index + 1:]:
            b = second.get_window_extent(renderer)
            if min(a.x1, b.x1) - max(a.x0, b.x0) > .8 and min(a.y1, b.y1) - max(a.y0, b.y0) > .8:
                collisions.append([first.get_text(), second.get_text()])
    assert not collisions, collisions
    geometry = {key: dict(left_mm=ax.get_position().x0 * 180,
        top_mm=(1 - ax.get_position().y1) * fig._height_mm,
        width_mm=ax.get_position().width * 180, height_mm=ax.get_position().height * fig._height_mm)
        for key, ax in fig._panels.items()}
    np.testing.assert_allclose([geometry["e"]["width_mm"], geometry["g"]["width_mm"]],
                               [TASK1_LEARNING_WIDTH_MM, TASK2_LEARNING_WIDTH_MM])
    return dict(status="PASS", source_type="behavior and illustrative task schematics",
        canvas_mm=[180, CANVAS_HEIGHT_MM], font_sizes_pt=sizes,
        public_panel_map=PUBLIC_SOURCE_PANEL_MAP, internal_source_panel_map=SOURCE_PANEL_MAP,
        display_panel_map=DISPLAY_PANEL_MAP, public_panel_letters=letters,
        compound_panels={"a": ["a"], "b": ["b"], "c": ["e", "f"], "e": ["g", "h"]},
        source_icons=source_icons, outcome_titles_removed=True,
        preceding_layout_spacing=fig._legacy_layout_spacing,
        quantitative_row_shift_down_mm=PPTX_ADDED_HEIGHT_MM,
        source_panel_shift_right_mm=PPTX_X_SHIFT_MM,
        source_typography=pptx_task_design.check_assets()["typography_scope"],
        source_font_substitutions=pptx_task_design.check_assets()["source_font_substitutions"],
        task_design_geometry="Complete source slides; common uniform scale and bottom whitespace crop",
        global_x_shift_mm=GLOBAL_X_SHIFT_MM,
        geometry_mm=geometry, spacing=fig._compact_spacing, text_collisions=collisions,
        personal_best_counts=counts, tests=tests, box_widths_mm=widths,
        public_personal_best_counts={DISPLAY_PANEL_MAP[k]: v for k, v in counts.items()},
        public_tests={DISPLAY_PANEL_MAP[k]: v for k, v in tests.items()},
        learning_artist_fingerprints=fingerprints, learning_sem="Across individually smoothed participant trajectories",
        fullscore_panels_moved_to="supplementary_figure/figS1_fullscore_rates_20261013.png",
        time_estimand="First trial attaining each participant's observed personal maximum; includes non-completers",
        inherited_inputs_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in base.CSV.iterdir() if p.is_file()},
        task_design_pptx_sha256=pptx_task_design.check_assets()["pptx_sha256"],
        task_design_source_image_count=len(pptx_task_design.check_assets()["images"]),
        y_axis_breaks=audit_yaxis_breaks({key: fig._panels[key] for key in LOWER_KEYS}),
        detached_axes=audit_detached_axes({key: fig._panels[key] for key in LOWER_KEYS}),
        detached_axes_helper_sha256=hashlib.sha256((base.ROOT / "code/detached_axes.py").read_bytes()).hexdigest(),
        no_fitting_or_simulation=True)


def render(output_dir, qa_dir, replace=False):
    output_dir, qa_dir = Path(output_dir), Path(qa_dir)
    output_dir.mkdir(parents=True, exist_ok=True); qa_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{NAME}.png"
    if output.exists() and not replace:
        raise FileExistsError(f"Use --replace to overwrite {output}")
    base.OUT, base.QA = output_dir, qa_dir
    fig = build_figure()
    report = audit(fig)
    (qa_dir / f"{NAME}_native_style.json").write_text(json.dumps(report, indent=2) + "\n")
    exemptions = [dict(panels=list(LOWER_KEYS), checks=["panel-width", "horizontal-gutter"],
        reason="Task 1 learning needs a36-mm plot, Task 2 learning33 mm; endpoint plots are27.9 mm, with2-mm within-task and5-mm between-task gaps")]
    base.finish(fig, NAME, nominal_font_sizes=(5, 6, 8), alignment_exemptions=exemptions)
    ink_margins = raster_ink_margins(output)
    assert abs(ink_margins[0] - ink_margins[1]) < .15, ink_margins
    path = qa_dir / f"{NAME}_verification.json"
    verification = json.loads(path.read_text())
    verification.update(status="PASS_source_backed_compact_fig1", public_panel_map=PUBLIC_SOURCE_PANEL_MAP,
        display_panel_map=DISPLAY_PANEL_MAP, public_panel_letters=PUBLIC_PANEL_LETTERS,
        nominal_canvas_mm=report["canvas_mm"], actual_canvas_mm=report["canvas_mm"],
        source_typography=report["source_typography"],
        source_font_substitutions=report["source_font_substitutions"],
        source_data_and_tests_unchanged=True, plot_input_hashes=report["inherited_inputs_sha256"],
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        raster_ink_margins_mm=ink_margins,
        current_caption="captions_and_methods_compact.md", moved_panels=report["fullscore_panels_moved_to"])
    path.write_text(json.dumps(verification, indent=2) + "\n")
    return output


def verify(output_dir, qa_dir):
    output_dir, qa_dir = Path(output_dir), Path(qa_dir)
    target = output_dir / f"{NAME}.png"
    stored = json.loads((qa_dir / f"{NAME}_native_style.json").read_text())
    fig = build_figure()
    try:
        fresh = audit(fig)
        for key in ["public_panel_map", "display_panel_map", "compound_panels", "geometry_mm", "spacing",
                    "personal_best_counts", "tests", "learning_artist_fingerprints", "inherited_inputs_sha256",
                    "box_widths_mm", "task_design_pptx_sha256", "task_design_source_image_count", "source_icons",
                    "detached_axes", "detached_axes_helper_sha256"]:
            assert stored[key] == fresh[key], key
    finally:
        base.plt.close(fig)
    saved = json.loads((qa_dir / f"{NAME}_verification.json").read_text())
    assert saved["plot_input_hashes"] == check_inputs()
    assert saved["public_panel_map"] == PUBLIC_SOURCE_PANEL_MAP
    assert saved["public_panel_letters"] == PUBLIC_PANEL_LETTERS
    assert saved["renderer_sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert saved["sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    ink_margins = raster_ink_margins(target)
    np.testing.assert_allclose(saved["raster_ink_margins_mm"], ink_margins, atol=1e-8)
    assert abs(ink_margins[0] - ink_margins[1]) < .15, ink_margins
    with base.Image.open(target) as im:
        assert im.mode == "RGB" and im.size == (4251, 4410)
        assert min(im.info["dpi"]) > 599
    print("PASS: compact Fig1 native content, frozen inputs, source tests and exported PNG")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=base.ROOT.parent)
    parser.add_argument("--qa-dir", type=Path, default=base.ROOT / "qa/bar_box")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify(args.output_dir, args.qa_dir)
    else:
        render(args.output_dir, args.qa_dir, args.replace)
