"""Measure and compact vertical categorical boxes/bars using display margins.

Keep category/data coordinates and physical widths. Translate the categorical
x viewport at unchanged span, so boxes, ticks and brackets move together.
Never apply this operation to numerical scatter/trajectory x coordinates.
"""
from __future__ import annotations
import numpy as np


def category_gaps(axis, boxes):
    axis.figure.canvas.draw()
    renderer=axis.figure.canvas.get_renderer()
    scale=25.4/axis.figure.dpi
    spine=axis.spines['left']
    vertices=spine.get_transform().transform(spine.get_path().vertices)
    spine_right=float(vertices[:,0].mean()+spine.get_linewidth()*axis.figure.dpi/144)
    edges=[box.get_window_extent(renderer).x0-box.get_linewidth()*axis.figure.dpi/144 for box in boxes]
    left=float(min(edges))
    label=axis.yaxis.label
    label_right=label.get_window_extent(renderer).x1 if label.get_visible() and label.get_text() else None
    return {'box_to_y_spine_mm':(left-spine_right)*scale,
            'ylabel_to_box_mm':None if label_right is None else (left-label_right)*scale,
            'xlim':list(map(float,axis.get_xlim())),
            'box_widths_mm':[float(box.get_window_extent(renderer).width*scale) for box in boxes],
            'left_spine_position':list(spine.get_position())}


def compact_category_margin(axis, boxes, *, target_mm=3.5, acceptable_mm=(2.,5.)):
    """Only call with explicit native artists from a vertical categorical panel."""
    boxes=list(boxes)
    if not boxes or not axis.spines['left'].get_visible():
        return {'status':'EXEMPT','reason':'No vertical category boxes/bars or hidden shared y spine'}
    before=category_gaps(axis,boxes)
    gap=before['box_to_y_spine_mm']
    low,high=map(float,axis.get_xlim())
    assert low<high and axis.get_xscale()=='linear','Requires an increasing linear categorical display'
    delta=0.
    if gap<acceptable_mm[0] or gap>acceptable_mm[1]:
        shift_mm=gap-target_mm
        delta=shift_mm/(axis.bbox.width*25.4/axis.figure.dpi)*(high-low)
        axis.set_xlim(low+delta,high+delta)
    after=category_gaps(axis,boxes)
    np.testing.assert_allclose(after['box_widths_mm'],before['box_widths_mm'],rtol=0,atol=1e-9)
    np.testing.assert_allclose(np.diff(after['xlim']),np.diff(before['xlim']),rtol=0,atol=1e-12)
    assert acceptable_mm[0]-1e-6<=after['box_to_y_spine_mm']<=acceptable_mm[1]+1e-6
    return {'status':'PASS','before':before,'after':after,
            'categorical_view_translation':float(delta),'target_box_to_y_spine_mm':target_mm,
            'soft_ylabel_target_mm':[6,12],
            'scope':'Categorical display-only translation; data/category positions, intervals and mark widths retained'}
