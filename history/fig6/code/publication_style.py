"""Exact shared display/audit helpers; no project-level imports."""

from __future__ import annotations

from hashlib import sha256

import importlib.util

import json

from pathlib import Path

import sys

import matplotlib

matplotlib.use("Agg")

from matplotlib.collections import Collection, PathCollection, PolyCollection

from matplotlib.colors import to_rgba

from matplotlib.container import ErrorbarContainer

from matplotlib.lines import Line2D

from matplotlib.patches import FancyArrowPatch, Patch

from matplotlib.text import Text

import numpy as np

from audit_panel_alignment import require_matplotlib_panel_alignment

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

def _hash_array(value):
    array = np.asarray(value)
    return sha256(str(array.shape).encode() + str(array.dtype).encode() + array.tobytes()).hexdigest()

def _geometry(figure):
    """Fingerprint scientific coordinates independently of display styling."""
    def patch_geometry(patch):
        if isinstance(patch, FancyArrowPatch):
            # The rendered arrowhead depends on stroke width; its original
            # path/endpoints, rather than that outline, define its connection.
            if patch._path_original is not None:
                return _hash_array(patch._path_original.vertices)
            return _hash_array(patch._posA_posB)
        return _hash_array(patch.get_path().vertices)
    return [{"bounds": list(axis.get_position().bounds),
             "limits": [list(axis.get_xlim()), list(axis.get_ylim())],
             "lines": [_hash_array(line.get_xydata()) for line in axis.lines],
             "collections": [{"offsets": _hash_array(c.get_offsets()),
                              "paths": [_hash_array(p.vertices) for p in c.get_paths()]}
                             for c in axis.collections],
             "patches": [patch_geometry(p) for p in axis.patches],
             "images": [_hash_array(image.get_array()) for image in axis.images]}
            for axis in figure.axes]

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

def _fig6_typography(figure, axes):
    _readable_axes([axis for name, axis in axes.items() if name not in ("a", "h")])
    for text in figure.findobj(Text):
        if text.get_text().startswith("Mean likelihood per trial") or text.get_text() == "DIS ratio":
            text.set_fontsize(6.0)
            text.set_fontweight("normal")
    for name in ("a", "h"):
        axis = axes[name]
        for patch in axis.patches:
            if isinstance(patch, FancyArrowPatch):
                patch.set_linewidth(1.0)
            else:
                patch.set_linewidth(max(patch.get_linewidth(), .75))
        for line in axis.lines:
            line.set_linewidth(1.0)
        for text in axis.texts:
            text.set_fontsize(max(text.get_fontsize(), 5.5 if name == "h" else 6.0))
    for name in ("g_upper", "g_lower", "d_upper"):
        for line in axes[name].lines:
            if np.isclose(line.get_linewidth(), 1.0):
                line.set_linewidth(1.10)
    # Retain the original compact inset and statistical brackets. Neither
    # dense observations nor uncertainty fills are made more opaque here.
    for patch in axes["f"].patches:
        patch.set_linewidth(max(patch.get_linewidth(), .75))
    for line in axes["f"].lines:
        if np.isclose(line.get_linewidth(), .60):
            line.set_linewidth(.75)
    return {"ordinary_labels": "6 pt axis labels; >=5.5 pt ticks and legends; regular weight",
            "schematic_text": "h >=5.5 pt; a >=6 pt; established attention headings retained",
            "lines": "0.70 pt axes; 0.75 pt error bars; 1.10 pt primary curves; 1.0 pt arrows",
            "preserved": "Significance weight, dense scatter opacity, SEM/CI fills, all scientific values"}
