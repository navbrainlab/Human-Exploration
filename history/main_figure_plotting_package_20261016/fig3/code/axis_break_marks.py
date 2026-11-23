"""Portable physical-size y-axis omission marks, matching the approved Fig5f.

Apply only to linear quantitative axes with a positive lower bound. Log axes,
categories and domains containing zero are not zero-omission breaks. Markers
are decoration: they never change the scale, limits, ticks or data artists.
"""
from __future__ import annotations
import numpy as np
from matplotlib.transforms import Bbox

VERSION = "fig5f-double-slash-20261012"
WIDTH_MM = 1.2276
SLASH_HEIGHT_MM = .6
CENTERS_MM = (-1.95, -1.11)
EXTENSION_MM = 2.7

def mark_nonzero_yaxis(axis, *, panel=None, ink="#252B30"):
    existing = getattr(axis, "_publication_y_break", None)
    if existing is not None:
        for artist in axis._publication_y_break_artists:
            artist.remove()
        del axis._publication_y_break
        panel = panel or existing.get("panel")
    low, high = (float(v) for v in axis.get_ylim())
    report = dict(panel=panel, ylim=[low,high], scale=axis.get_yscale(),
                  version=VERSION, marked=False)
    if axis.get_yscale() != "linear":
        return {**report, "reason":"Nonlinear scale; no omitted-zero marker"}
    if not 0 < low < high:
        return {**report, "reason":"Domain includes zero or is not increasing"}
    figure = axis.figure
    bbox = axis.get_window_extent()
    width_mm, height_mm = np.asarray([bbox.width,bbox.height])/figure.dpi*25.4
    x = np.array([-WIDTH_MM/2, WIDTH_MM/2])/width_mm
    extension, = axis.plot([0,0],[-EXTENSION_MM/height_mm,0],
        transform=axis.transAxes, color=ink,
        linewidth=axis.spines["left"].get_linewidth(),clip_on=False,
        solid_capstyle="butt", zorder=3, gid="publication-y-break-extension")
    # Reserve the omitted interval below the plotted baseline. If its marks
    # would touch x tick text, enlarge tick padding instead of moving data.
    figure.canvas.draw()
    renderer=figure.canvas.get_renderer()
    unit=figure.dpi/25.4
    region=Bbox.from_extents(axis.bbox.x0-1.04*unit,axis.bbox.y0-2.68*unit,
                            axis.bbox.x0+1.04*unit,axis.bbox.y0-.38*unit)
    touching=[text for text in axis.get_xticklabels() if text.get_visible()
              and text.get_text().strip() and region.overlaps(text.get_window_extent(renderer))]
    if touching:
        axis.tick_params(axis="x",pad=max(7.0,axis.xaxis.get_major_ticks()[0].get_pad()))
    strokes=[]
    for index,center in enumerate(CENTERS_MM):
        y=np.array([center-SLASH_HEIGHT_MM/2,center+SLASH_HEIGHT_MM/2])/height_mm
        for color,lw,kind in (("white",2.4,"mask"),(ink,.7,"slash")):
            line,=axis.plot(x,y,transform=axis.transAxes,color=color,linewidth=lw,
                clip_on=False,solid_capstyle="butt",zorder=10,
                gid=f"publication-y-break-{kind}-{index}")
            strokes.append(line)
    assert (low,high)==axis.get_ylim(),"Omission decoration changed limits"
    report.update(marked=True,slash_count=2,width_mm=WIDTH_MM,
                  slash_height_mm=SLASH_HEIGHT_MM,centers_mm=list(CENTERS_MM),
                  extension_mm=EXTENSION_MM,
                  x_tick_pad_pt=float(axis.xaxis.get_major_ticks()[0].get_pad()),
                  meaning="Zero-to-lower-bound interval omitted below first displayed y value")
    axis._publication_y_break=report
    axis._publication_y_break_artists=[extension,*strokes]
    return report

def audit_yaxis_breaks(axes):
    """Check rendered stroke dimensions and that omitted-zero marks are distinct."""
    reports={}
    for panel,axis in axes.items():
        item=getattr(axis,"_publication_y_break",None)
        if item is None:continue
        assert axis.get_yscale()=="linear" and axis.get_ylim()[0]>0
        assert list(axis.get_ylim())==item["ylim"]
        slashes=[line for line in axis._publication_y_break_artists
                 if str(line.get_gid()).startswith("publication-y-break-slash")]
        assert len(slashes)==2
        measured=[]
        for line,center in zip(slashes,CENTERS_MM):
            pixels=line.get_transform().transform(line.get_xydata())
            delta=(pixels[1]-pixels[0])*25.4/axis.figure.dpi
            center_mm=(pixels[:,1].mean()-axis.bbox.y0)*25.4/axis.figure.dpi
            np.testing.assert_allclose(delta,[WIDTH_MM,SLASH_HEIGHT_MM],atol=1e-5)
            np.testing.assert_allclose(center_mm,center,atol=1e-5)
            measured.append(dict(width_mm=float(delta[0]),height_mm=float(delta[1]),center_mm=float(center_mm)))
        renderer=axis.figure.canvas.get_renderer()
        unit=axis.figure.dpi/25.4
        region=Bbox.from_extents(axis.bbox.x0-1.04*unit,axis.bbox.y0-2.68*unit,
                                axis.bbox.x0+1.04*unit,axis.bbox.y0-.38*unit)
        overlaps=[t.get_text() for t in axis.get_xticklabels() if t.get_visible()
                  and t.get_text().strip() and region.overlaps(t.get_window_extent(renderer))]
        assert not overlaps,(panel,"Break overlaps x tick text",overlaps)
        reports[panel]={**item,"x_tick_text_collisions":overlaps,"panel":panel,"rendered_strokes":measured,"geometry_verdict":"PASS"}
    return reports
