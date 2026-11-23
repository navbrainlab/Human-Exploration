"""Portable Cartesian axis spacing matching mini_DG Fig. 4's 5/3-pt spines.

Only style transforms change. Hidden spines remain hidden; bounds, data,
limits, tests and tick values are never changed. Call after omission marks.
"""
from __future__ import annotations
import numpy as np
from matplotlib.transforms import ScaledTranslation, Bbox

Y_OUTWARD_PT=5.0
X_OUTWARD_PT=3.0


def detach_axes(axis, *, y_pt=Y_OUTWARD_PT, x_pt=X_OUTWARD_PT, right=False):
    if axis.spines['left'].get_visible():axis.spines['left'].set_position(('outward',y_pt))
    if axis.spines['bottom'].get_visible():axis.spines['bottom'].set_position(('outward',x_pt))
    if right and axis.spines['right'].get_visible():axis.spines['right'].set_position(('outward',y_pt))
    # Omission strokes must move with the true left spine, never stay at x=0.
    if hasattr(axis,'_publication_y_break_artists'):
        pos=axis.spines['left'].get_position()
        outward=float(pos[1]) if pos[0]=='outward' else 0.0
        transform=axis.transAxes+ScaledTranslation(-outward/72.0,0,axis.figure.dpi_scale_trans)
        for line in axis._publication_y_break_artists:line.set_transform(transform)
        axis._publication_y_break['left_spine_outward_pt']=outward
    return axis


def audit_detached_axes(axes, *, minimum_mm=.5):
    reports={}
    for name,a in axes.items():
        left,bottom=a.spines['left'],a.spines['bottom']
        if not (left.get_visible() and bottom.get_visible()):
            reports[name]={'status':'EXEMPT','reason':'A Cartesian corner is not drawn: shared/hidden/categorical one-axis view.'}
            continue
        a.figure.canvas.draw();renderer=a.figure.canvas.get_renderer();scale=25.4/a.figure.dpi
        lv=left.get_transform().transform(left.get_path().vertices)
        bv=bottom.get_transform().transform(bottom.get_path().vertices)
        y_end=lv[np.argmin(lv[:,1])];x_end=bv[np.argmin(bv[:,0])]
        radius=(left.get_linewidth()+bottom.get_linewidth())/2*25.4/72
        distance=float(np.linalg.norm(y_end-x_end)*scale)
        gap=distance-radius
        assert gap>=minimum_mm,(name,'Axis corner gap',gap)
        item={'status':'PASS','left_position':list(left.get_position()),'bottom_position':list(bottom.get_position()),
              'corner_endpoint_distance_mm':distance,'conservative_visible_corner_gap_mm':gap}
        if hasattr(a,'_publication_y_break_artists'):
            slashes=[line for line in a._publication_y_break_artists if str(line.get_gid()).startswith('publication-y-break-slash')]
            assert len(slashes)==2,(name,'Expected two omission slashes')
            vertices=[line.get_transform().transform(line.get_xydata()) for line in slashes]
            expected_x=lv[:,0].mean()
            assert all(np.isclose(v[:,0].mean(),expected_x) for v in vertices),(name,'Omission marks detached from actual y spine')
            right_edges=[v[:,0].max()+line.get_linewidth()/2*a.figure.dpi/72 for line,v in zip(slashes,vertices)]
            slashgap=(x_end[0]-max(right_edges))*scale-bottom.get_linewidth()/2*25.4/72
            assert slashgap>=minimum_mm,(name,'Omission slash to x endpoint',slashgap)
            item['slash_to_x_endpoint_visible_gap_mm']=float(slashgap)
            boxes=[]
            for line,v in zip(slashes,vertices):
                half=line.get_linewidth()/2*a.figure.dpi/72
                boxes.append(Bbox.from_extents(v[:,0].min()-half,v[:,1].min()-half,v[:,0].max()+half,v[:,1].max()+half))
            collisions=[t.get_text() for t in a.get_xticklabels() if t.get_visible() and t.get_text().strip()
                        and any(b.overlaps(t.get_window_extent(renderer)) for b in boxes)]
            assert not collisions,(name,'Slash/x tick text overlap',collisions)
            item['slash_tick_text_collisions']=[]
        reports[name]=item
    return reports
