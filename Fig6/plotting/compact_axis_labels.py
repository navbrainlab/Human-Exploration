"""Place native axis labels near the rendered tick envelope, without moving plots."""
import numpy as np


def visible(text):
    return text.get_visible() and bool(text.get_text().strip())


def quantitative(ax):
    return ax.get_visible() and ax.axison and ax.name != '3d' and not hasattr(ax, '_colorbar') \
        and any(s.get_visible() for s in ax.spines.values())


def spine_edge(ax, which):
    axis = ax.xaxis if which == 'x' else ax.yaxis
    side = axis.get_label_position()
    if which == 'x':
        side = 'top' if side == 'top' else 'bottom'
    else:
        side = 'right' if side == 'right' else 'left'
    spine = ax.spines[side]
    points = spine.get_transform().transform(spine.get_path().vertices)
    edge = float(np.mean(points[:, 1 if which == 'x' else 0]))
    half_stroke = spine.get_linewidth() * ax.figure.dpi / 144
    return side, edge + half_stroke * (1 if side in ('top', 'right') else -1)


def records(fig, shared=()):
    """shared contains explicit (Text, [owner_axes], 'x'|'y') tuples."""
    entries = []
    seen = set()
    for ax in fig.axes:
        if not quantitative(ax):
            continue
        for which, axis in [('x', ax.xaxis), ('y', ax.yaxis)]:
            text = axis.label
            if visible(text):
                entries.append((text, [ax], which))
                seen.add(id(text))
    for text, axes, which in shared:
        if visible(text) and id(text) not in seen:
            entries.append((text, axes, which))
            seen.add(id(text))
    return entries


def measure(fig, entries, *, tick_gap_mm=.8, no_tick_gap_mm=1.5):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    factor = fig.dpi / 25.4
    result = []
    for text, axes, which in entries:
        side = spine_edge(axes[0], which)[0]
        edges = [spine_edge(a, which)[1] for a in axes]
        spine = min(edges) if side in ('left', 'bottom') else max(edges)
        bbox = text.get_window_extent(renderer)
        tick_boxes = []
        for ax in axes:
            axis = ax.xaxis if which == 'x' else ax.yaxis
            if not axis.get_visible():
                continue
            tick_boxes += [t.get_window_extent(renderer) for t in
                           [*axis.get_ticklabels(minor=False), *axis.get_ticklabels(minor=True)] if visible(t)]
            offset = axis.get_offset_text()
            if visible(offset):
                ob = offset.get_window_extent(renderer)
                overlap = min(ob.x1,bbox.x1)-max(ob.x0,bbox.x0) if which=='x' else min(ob.y1,bbox.y1)-max(ob.y0,bbox.y0)
                if overlap>0:tick_boxes.append(ob)
        bbox = text.get_window_extent(renderer)
        if side == 'bottom':
            nearest = bbox.y1
            tick_edge = min(b.y0 for b in tick_boxes) if tick_boxes else None
            target = tick_edge-tick_gap_mm*factor if tick_boxes else spine-no_tick_gap_mm*factor
            clearance = (tick_edge-nearest)/factor if tick_boxes else None
            spine_gap = (spine-nearest)/factor
        elif side == 'top':
            nearest = bbox.y0
            tick_edge = max(b.y1 for b in tick_boxes) if tick_boxes else None
            target = tick_edge+tick_gap_mm*factor if tick_boxes else spine+no_tick_gap_mm*factor
            clearance = (nearest-tick_edge)/factor if tick_boxes else None
            spine_gap = (nearest-spine)/factor
        elif side == 'left':
            nearest = bbox.x1
            tick_edge = min(b.x0 for b in tick_boxes) if tick_boxes else None
            target = tick_edge-tick_gap_mm*factor if tick_boxes else spine-no_tick_gap_mm*factor
            clearance = (tick_edge-nearest)/factor if tick_boxes else None
            spine_gap = (spine-nearest)/factor
        else:
            nearest = bbox.x0
            tick_edge = max(b.x1 for b in tick_boxes) if tick_boxes else None
            target = tick_edge+tick_gap_mm*factor if tick_boxes else spine+no_tick_gap_mm*factor
            clearance = (nearest-tick_edge)/factor if tick_boxes else None
            spine_gap = (nearest-spine)/factor
        result.append(dict(text=text.get_text(), role=which, side=side,
            owners=[fig.axes.index(a) for a in axes], shared=len(axes)>1,
            font_pt=text.get_fontsize(), label_spine_gap_mm=spine_gap,
            tick_text_clearance_mm=clearance, target_tick_text_clearance_mm=tick_gap_mm,
            target_error_mm=abs(target-nearest)/factor,
            shift_pixels=target-nearest))
    return result


def apply(fig, *, shared=(), tick_gap_mm=.8, no_tick_gap_mm=1.5, tolerance_mm=.15):
    """Move label Text only. Preserve font, ticks, plot rectangles and data."""
    entries = records(fig, shared)
    before = measure(fig, entries, tick_gap_mm=tick_gap_mm, no_tick_gap_mm=no_tick_gap_mm)
    for (text, axes, which), record in zip(entries, before):
        position = np.array(text.get_transform().transform(text.get_position()), dtype=float)
        position[1 if which == 'x' else 0] += record['shift_pixels']
        if len(axes) == 1 and text is (axes[0].xaxis.label if which == 'x' else axes[0].yaxis.label):
            axis = axes[0].xaxis if which == 'x' else axes[0].yaxis
            x, y = fig.transFigure.inverted().transform(position)
            axis.set_label_coords(x, y, transform=fig.transFigure)
        else:
            text.set_transform(fig.transFigure)
            text.set_position(fig.transFigure.inverted().transform(position))
    after = measure(fig, entries, tick_gap_mm=tick_gap_mm, no_tick_gap_mm=no_tick_gap_mm)
    failures = [r for r in after if r['target_error_mm']>tolerance_mm or
                (r['tick_text_clearance_mm'] is not None and r['tick_text_clearance_mm']<tick_gap_mm-tolerance_mm)]
    return dict(status='PASS' if not failures else 'NEEDS_LAYOUT',before=before,after=after,
        failures=failures,tick_text_gap_mm=tick_gap_mm,no_tick_spine_gap_mm=no_tick_gap_mm,
        tolerance_mm=tolerance_mm,
        rule='Nearest rendered label edge follows visible tick-text envelope; fixed6.5/8mm spine gaps superseded.')
