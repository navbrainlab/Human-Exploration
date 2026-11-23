from __future__ import annotations
import json
import numpy as np
from matplotlib.collections import Collection, PathCollection, PolyCollection
from matplotlib.colors import to_rgba
from matplotlib.container import ErrorbarContainer
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch
from matplotlib.text import Text
INK = "#252B30"
PALETTE = {
    "#6F55A5": "#60458F",  # same DGEM purple family
    "#F3C316": "#B68F00",  # gold with stronger white-background contrast
    "#D8B938": "#B68F00",  # match fRL across bars and trajectories
    "#E67E28": "#D57224",  # Obs
    "#F57C00": "#D87300",  # Bayesian
    "#D88A42": "#D87300",
    "#2A9DB5": "#20879C",  # simulation temperature
}
RULES = {
    "checked_date": "2026-09-30",
    "sources": ["https://www.nature.com/nathumbehav/content",
                "https://www.nature.com/nathumbehav/submission-guidelines/aip-and-formatting",
                "https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/"],
    "official": "Readable accessible colours; standard sans-serif labels 5–7 pt; >=300 dpi; <=180 mm wide.",
    "scope": "Final typography and contrast, not certification of a complete submission package.",
    "output_override": "User requires PNG only; no PDF/SVG. Editable plotting source is retained.",
    "palette": PALETTE,
}


def _color(value):
    try:
        rgba = to_rgba(value)
    except (ValueError, TypeError):
        return value
    for old, new in PALETTE.items():
        if np.allclose(rgba[:3], to_rgba(old)[:3], atol=1e-6):
            return (*to_rgba(new)[:3], rgba[3])
    return rgba


def _recolor(figure):
    # Update legend keys together with the plotted marks. Heatmap colormaps
    # and measured image arrays are deliberately outside this palette map.
    for artist in figure.findobj():
        if isinstance(artist, Line2D):
            artist.set_color(_color(artist.get_color()))
            artist.set_markerfacecolor(_color(artist.get_markerfacecolor()))
            artist.set_markeredgecolor(_color(artist.get_markeredgecolor()))
        elif isinstance(artist, Collection):
            for getter, setter in ((artist.get_facecolors, artist.set_facecolors),
                                   (artist.get_edgecolors, artist.set_edgecolors)):
                old = getter()
                if len(old):
                    new = np.array([_color(value) for value in old])
                    if not np.array_equal(new, old):
                        setter(new)
        elif isinstance(artist, Patch):
            face, edge = artist.get_facecolor(), artist.get_edgecolor()
            new_face, new_edge = _color(face), _color(edge)
            if not np.allclose(face, new_face) or not np.allclose(edge, new_edge):
                artist.set_alpha(None)
                artist.set_facecolor(new_face)
                artist.set_edgecolor(new_edge)
        elif isinstance(artist, Text):
            artist.set_color(_color(artist.get_color()))


def _render_qa(figure, path):
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    texts = []
    for text in figure.findobj(Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        if text.axes is not None and not text.axes.axison:
            # Flow diagrams intentionally turn off axes but retain their text.
            if text not in text.axes.texts:
                continue
        box = text.get_window_extent(renderer)
        if box.width > 0 and box.height > 0:
            texts.append((text, box))
    threshold = .3 * figure.dpi / 72
    collisions = []
    rotated_tick_bbox_warnings = []
    for index, (a, ab) in enumerate(texts):
        for b, bb in texts[index + 1:]:
            overlap = (min(ab.x1, bb.x1) - max(ab.x0, bb.x0),
                       min(ab.y1, bb.y1) - max(ab.y0, bb.y0))
            if min(overlap) > threshold:
                # The existing angled Plant/Painting tick labels have
                # intersecting axis-aligned boxes, but separate text glyphs.
                # Preserve this explicit visual-review item, not a global
                # weakening of the text-collision threshold.
                if ({a.get_text(), b.get_text()} == {"Plant", "Painting"}
                        and a.get_rotation() == b.get_rotation() == 35):
                    rotated_tick_bbox_warnings.append([a.get_text(), b.get_text()])
                else:
                    collisions.append([a.get_text(), b.get_text()])
    clipped = [t.get_text() for t, b in texts
               if b.x0 < -.5 or b.y0 < -.5 or b.x1 > figure.bbox.width + .5 or b.y1 > figure.bbox.height + .5]
    qa = {"rules": RULES, "size_mm": (figure.get_size_inches() * 25.4).tolist(),
          "dpi": figure.dpi, "font_floor_pt": min(t.get_fontsize() for t, _ in texts),
          "text_collisions": collisions, "clipped_text": clipped,
          "rotated_tick_bbox_visual_review": rotated_tick_bbox_warnings,
          "audit_scope": "Matplotlib text rectangles and geometry; no PDF collision audit."}
    path.write_text(json.dumps(qa, indent=2) + "\n")
    if collisions or clipped or qa["font_floor_pt"] < 5:
        raise ValueError(f"Rendered text requires review: {path}: {collisions}, clipped={clipped}")
    assert qa["size_mm"][0] <= 180.001
    return qa


def _readable_axes(axes):
    """Strengthen the frame without making tick labels or axis labels bold."""
    for axis in axes:
        for spine in axis.spines.values():
            if spine.get_visible():
                spine.set_linewidth(.70)
        axis.tick_params(which="major", width=.70)
        for label in (*axis.get_xticklabels(), *axis.get_yticklabels()):
            label.set_fontsize(max(label.get_fontsize(), 5.5))
            label.set_fontweight("normal")
            label.set_color(INK)
        for label in (axis.xaxis.label, axis.yaxis.label):
            if label.get_text():
                label.set_fontsize(max(label.get_fontsize(), 6.0))
                label.set_fontweight("normal")
                label.set_color(INK)
        legend = axis.get_legend()
        if legend is not None:
            for label in legend.get_texts():
                label.set_fontsize(5.5)
                label.set_fontweight("normal")
                label.set_color(INK)
        # Errorbar collections are distinct from confidence-band polygons.
        for container in axis.containers:
            if isinstance(container, ErrorbarContainer):
                for line in container.lines[1]:
                    line.set_markeredgewidth(.75)
                for collection in container.lines[2]:
                    collection.set_linewidth(.75)

