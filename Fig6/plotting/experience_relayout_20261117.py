"""Author-requested Fig. 6 layout from frozen behavioural artists.

No statistics, data coordinates, marker sizes or uncertainty values are changed.
Panel letters are sequential a–h; the PowerPoint is deliberately not exported.
"""
import gc
import json
import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.transforms import Affine2D, ScaledTranslation


def axis_panel(R, source_number, part, index, letter, left, width,
               height=24, letter_gap=4.42):
    package = R.PUB / 'main_figure' / R.NAMES[source_number - 1]
    fig, path = R.load(package, part)
    before = R.science(fig)
    R.style(fig)
    axis = fig.axes[index]
    fig.canvas.draw()
    old = axis.bbox.frozen()
    old_data = axis.transData.frozen()
    # Source packages contain explicitly frozen display transforms. Capture
    # those transforms before moving an axis; changing its position alone
    # would otherwise leave the scientific marks at their old locations.
    artists = list(axis.lines) + list(axis.collections) + list(axis.patches) + list(axis.texts)
    transforms = [(obj, obj.get_transform().frozen(),
                   obj.get_offset_transform().frozen() if isinstance(obj, PathCollection) else None)
                  for obj in artists]
    for other in fig.axes:
        other.set_visible(other is axis)
    for obj in fig.texts:
        obj.set_visible(False)
    for obj in fig.legends:
        obj.set_visible(False)
    fig.set_size_inches(180 / 25.4, 55 / 25.4, forward=True)
    axis.set_position([left / 180, 15 / 55, width / 180, height / 55])
    # Some frozen source axes also retain fixed axis/tick transforms. Rebuild
    # their coordinate system and tick artists at the destination rectangle.
    axis._set_lim_and_transforms()
    axis.patch.set_transform(axis.transAxes)
    for spine in axis.spines.values():
        spine.set_position(spine.get_position())
    new = axis.bbox.frozen()
    move = Affine2D().translate(-old.x0, -old.y0).scale(new.width / old.width, new.height / old.height).translate(new.x0, new.y0)
    for obj, transform, offset in transforms:
        if offset is not None:
            obj.set_offset_transform(offset + move)
        else:
            obj.set_transform(transform + move)
        if obj.get_clip_on():
            obj.set_clip_path(axis.patch)
    checked = 0
    largest_residual = 0.0
    for obj, transform, offset in transforms:
        if isinstance(obj, PathCollection):
            points = np.asarray(obj.get_offsets(), dtype=float)
            original_transform = offset
            actual_transform = obj.get_offset_transform()
        elif obj in axis.lines:
            points = np.column_stack([obj.get_xdata(), obj.get_ydata()]).astype(float)
            original_transform = transform
            actual_transform = obj.get_transform()
        else:
            continue
        if not len(points) or not np.isfinite(points).all():
            continue
        # Only genuine data-coordinate artists enter this check. Axis break
        # marks and axes-relative annotations retain their own coordinates.
        if np.max(np.abs(original_transform.transform(points) - old_data.transform(points))) < 1e-5:
            residual = float(np.max(np.abs(actual_transform.transform(points) - axis.transData.transform(points))))
            assert residual < 1e-5, (letter, residual)
            largest_residual = max(largest_residual, residual)
            checked += 1
    assert checked > 0, ('No scientific display transforms verified', letter)
    # Keep physical y-axis break marks attached to the actual outward spine.
    for line in axis.lines:
        x = np.asarray(line.get_xdata(), dtype=float)
        y = np.asarray(line.get_ydata(), dtype=float)
        if len(x) == 2 and np.max(np.abs(x)) < .1 and np.max(np.abs(y)) < .15:
            position = axis.spines['left'].get_position()
            outward = position[1] if position[0] == 'outward' else 0
            line.set_transform(axis.transAxes + ScaledTranslation(-outward / 72, 0, fig.dpi_scale_trans))
    axis.xaxis.set_label_coords(.5, -.22)
    axis.yaxis.set_label_coords(-.25, .5)
    label = fig.text((left - 4.68) / 180, (15 + height + letter_gap) / 55, letter,
             va='top', fontsize=10, fontweight='bold', gid='nhb-panel-letter')
    from .compact_axis_labels import apply as compact_labels
    compact_labels(fig)
    assert R.science(fig) == before
    out = R.render(fig)
    visible = np.any(out < 250, axis=2)
    ys, xs = np.nonzero(visible)
    assert xs.min() > 0 and xs.max() < out.shape[1] - 1
    assert ys.min() > 0 and ys.max() < out.shape[0] - 1
    R.QA['native_parts'].append({
        'source': str(path.relative_to(R.PUB)), 'sha256': R.sha(path),
        'source_axis': index, 'panel': letter,
        'scientific_arrays_unchanged': True, 'science_hash': before,
        'plot_area_mm': [left, 15, width, height], 'font_sizes': [6, 7, 10],
        'canvas_vertical_anchors_px': [out.shape[0] - axis.bbox.y1,
                                       out.shape[0] - axis.bbox.y0,
                                       out.shape[0] - label.get_transform().transform(label.get_position())[1]],
        'source_kind': 'frozen behavioural summaries and participant observations',
        'display_transform_rebased': True,
        'data_artist_transforms_verified': checked,
        'largest_data_projection_residual_px': largest_residual,
    })
    R.plt.close(fig)
    gc.collect()
    return out


def native_lower_row_anchors(R):
    package = R.PUB / 'main_figure' / R.NAMES[4]
    composition = json.loads((package / 'rendering.json').read_text())['composition']
    anchors = {}
    for part, letter in [('hi_heatmaps', 'g'), ('hij', 'h')]:
        fig, _ = R.load(package, part)
        entry = next(item for item in composition if item['part'] == part)
        crop_y = entry.get('crop_xy_pixels', [0, 0])[1]
        origin = entry['offset_xy_pixels'][1] - crop_y + fig.canvas.get_width_height()[1]
        bounds = fig.axes[0].bbox
        anchors[letter] = [origin - bounds.y1, origin - bounds.y0]
        R.plt.close(fig)
    overlay, _ = R.load(package, 'modality_icon_overlay')
    entry = next(item for item in composition if item['part'] == 'modality_icon_overlay'
                 and item['crop_y_pixels'][0] > 0)
    origin = entry['offset_xy_pixels'][1] - entry['crop_y_pixels'][0] + overlay.canvas.get_width_height()[1]
    for source_letter, letter in [('h', 'g'), ('i', 'h')]:
        label, = [label for label in overlay.texts if label.get_text() == source_letter
                  and label.get_gid() == 'nhb-panel-letter' and label.get_visible()]
        anchors[letter].append(origin - label.get_transform().transform(label.get_position())[1])
    R.plt.close(overlay)
    assert np.max(np.abs(np.asarray(anchors['g']) - anchors['h'])) <= 1
    return anchors


def rows(R):
    definitions = [
        (1, 'bottom', 1, 'a', 13, 18.0766666667),
        (4, 'cdef__upper', 14, 'b', 48, 22.5),
        (4, 'cdef__upper', 15, 'c', 82, 22.5),
        (4, 'cdef__upper', 16, 'd', 116, 22.5),
        (4, 'cdef__upper', 0, 'e', 150, 25),
    ]
    top = None
    occupied = None
    for definition in definitions:
        panel = axis_panel(R, *definition)
        mask = np.any(panel < 250, axis=2)
        if top is None:
            top = panel
            occupied = mask
        else:
            assert not np.any(mask & occupied), ('Panel overlap', definition[3])
            top = np.minimum(top, panel)
            occupied |= mask
    # Entire pre-existing heatmap/report panels, including source icons and
    # legends, are only translated. Native pixels and scientific geometry stay
    # exact; the editorial note in the supplied screenshot is not a data panel.
    source = R.current(5, {'h': 'g', 'i': 'h', 'j': 'h'},
                       parts=['hij', 'hi_heatmaps', 'modality_icon_overlay'])
    tail = R.row(source, 4620, None, name='old5hi_to_new6gh')
    tail_top = R.QA['row_crops'][-1]['source_crop'][1]
    anchors = native_lower_row_anchors(R)
    top_px, bottom_px, label_px = np.mean(list(anchors.values()), axis=0)
    best = axis_panel(R, 1, 'bottom', 3, 'f', 13, 18.0766666667,
                      height=(bottom_px - top_px) / R.PPM, letter_gap=(top_px - label_px) / R.PPM)
    best_anchors = np.asarray(R.QA['native_parts'][-1]['canvas_vertical_anchors_px'])
    # Coordinates here are relative to the 180-mm row, after the A4 margin
    # has been removed. The measured 106.21–112.56-mm gutter is fully white.
    cut = round(110 * R.PPM)
    mask = np.any(tail < 250, axis=2)
    assert not np.any(mask[:, cut - 5:cut + 5])
    pieces = []
    for start, end in [(0, cut), (cut, tail.shape[1])]:
        piece = tail[:, start:end]
        x0, y0, x1, y1 = R.bounds(piece)
        pieces.append(piece[:, x0:x1])
    best = R.trim_native_row(best, 'new6f')
    best_crop_top = R.QA['row_crops'][-1]['source_crop'][1]
    best_top = round(top_px - tail_top - (best_anchors[0] - best_crop_top))
    row_top = min(0, best_top)
    row_bottom = max(best_top + len(best), len(tail))
    bx0, _, bx1, _ = R.bounds(best)
    box = best[:, bx0:bx1]
    lower = np.full((row_bottom - row_top, R.WIDTH, 3), 255, np.uint8)
    x = round(1.5 * R.PPM)
    for piece, y in [(box, best_top - row_top), *((piece, -row_top) for piece in pieces)]:
        assert x + piece.shape[1] < R.WIDTH
        lower[y:y + len(piece), x:x + piece.shape[1]] = piece
        x += piece.shape[1] + round(5 * R.PPM)
    placement = {'f': (best_anchors - best_crop_top + best_top - row_top).tolist(),
                 **{letter: (np.asarray(values) - tail_top - row_top).tolist()
                    for letter, values in anchors.items()}}
    alignment_px = np.ptp(np.asarray(list(placement.values())), axis=0)
    assert np.max(alignment_px) <= 1.5, (placement, alignment_px)
    R.QA['experience_relayout'] = {
        'panel_order': 'abcdefgh', 'top': ['personal_best_task1', 'onset', 'coverage', 'patterns', 'transfer'],
        'bottom': ['personal_best_task2', 'probe_DIS_joint_distributions', 'reported_dimensions'],
        'original_tests_and_correction_families_retained': True,
        'bottom_panels_rendered_without_resampling': True,
        'bottom_alignment': {'order': ['plot_top', 'plot_bottom', 'letter_top'],
                             'row_anchors_px': placement, 'range_px': alignment_px.tolist()},
        'removed_from_figure': ['group_DIS_trajectories', 'group_learning_trajectories'],
        'powerpoint_updated': False,
    }
    return [R.trim_native_row(top, 'new6abcde'), R.trim_native_row(lower, 'new6fgh')]
