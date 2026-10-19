"""Set visible Cartesian spine endpoints at the outer displayed major ticks.

The data rectangle, limits, tick values and data artists are unchanged.
Zero-omission extensions are decorative exceptions and still connect to the
first displayed y tick. Apply after the figure has fixed its ticks and limits.
"""
from __future__ import annotations
import numpy as np


def displayed_ticks(axis, orientation):
    limits = axis.get_xlim() if orientation == 'x' else axis.get_ylim()
    ticks = axis.get_xticks() if orientation == 'x' else axis.get_yticks()
    low, high = sorted(limits)
    tolerance = max(1., abs(low), abs(high)) * 1e-10
    return np.sort(np.unique(np.asarray(ticks)[
        (np.asarray(ticks) >= low-tolerance) & (np.asarray(ticks) <= high+tolerance)]))


def bound_spines_to_ticks(axis, *, sides=('bottom', 'left', 'right')):
    for side in sides:
        spine = axis.spines[side]
        if not spine.get_visible() or not axis.axison:
            continue
        orientation = 'x' if side in ('bottom', 'top') else 'y'
        ticks = displayed_ticks(axis, orientation)
        if len(ticks) >= 2:
            spine.set_bounds(float(ticks[0]), float(ticks[-1]))
    # Existing double slashes sit below the data rectangle. Connect the
    # extension to the shortened main y spine without moving the slashes.
    if hasattr(axis, '_publication_y_break_artists') and axis.spines['left'].get_visible():
        lower = axis.spines['left'].get_bounds()[0]
        fraction = axis.transAxes.inverted().transform(axis.transData.transform((0, lower)))[1]
        extension = next(line for line in axis._publication_y_break_artists
                         if line.get_gid() == 'publication-y-break-extension')
        start = float(extension.get_ydata()[0])
        extension.set_ydata([start, float(fraction)])
        axis._publication_y_break['extension_connects_to_y_tick'] = float(lower)
    return axis


def audit_tick_bounded_spines(axes, *, sides=('bottom', 'left', 'right')):
    reports = {}
    for name, axis in axes.items():
        axis.figure.canvas.draw()
        records = {}
        for side in sides:
            spine = axis.spines[side]
            if not spine.get_visible() or not axis.axison:
                continue
            orientation = 'x' if side in ('bottom', 'top') else 'y'
            ticks = displayed_ticks(axis, orientation)
            if len(ticks) < 2:
                records[side] = {'status': 'EXEMPT', 'reason': 'Fewer than two displayed major ticks'}
                continue
            bounds = spine.get_bounds()
            np.testing.assert_allclose(bounds, [ticks[0], ticks[-1]], rtol=0, atol=1e-10)
            values = spine.get_path().vertices[:, 0 if orientation == 'x' else 1]
            np.testing.assert_allclose(np.sort(values), [ticks[0], ticks[-1]], rtol=0, atol=1e-10)
            records[side] = {'status': 'PASS', 'bounds': [float(value) for value in bounds],
                             'outer_displayed_major_ticks': [float(ticks[0]), float(ticks[-1])]}
        reports[name] = records or {'status': 'EXEMPT', 'reason': 'No visible Cartesian spine'}
    return reports
