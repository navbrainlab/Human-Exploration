"""Compact grayscale data-source symbols, redrawn from the user's reference.

Native Matplotlib paths keep the symbols sharp and reproducible without SVG
files, external fonts or changes to any quantitative panel.
"""

from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.path import Path
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch, Polygon


INK = "#454545"
PALE = "#ECECEC"
MID = "#A5A5A5"
LINE_WIDTH = .55


def _line(axis: Axes, vertices: list[tuple[float, float]]) -> None:
    x, y = zip(*vertices)
    axis.plot(x, y, color=INK, linewidth=LINE_WIDTH,
              solid_capstyle="round", solid_joinstyle="round")


def _box(axis: Axes, x: float, y: float, width: float, height: float,
         fill: str = PALE, radius: float = 2.0) -> None:
    axis.add_patch(FancyBboxPatch(
        (x, y), width, height, boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fill, edgecolor=INK, linewidth=LINE_WIDTH,
    ))


def _path(axis: Axes, vertices: list[tuple[float, float]], codes: list[int],
          fill: str = "none") -> None:
    axis.add_patch(PathPatch(Path(vertices, codes), facecolor=fill,
                            edgecolor=INK, linewidth=LINE_WIDTH,
                            capstyle="round", joinstyle="round"))


def _task_behavior(axis: Axes) -> None:
    for x, y, fill in ((6, 60, PALE), (36, 60, MID), (6, 30, PALE)):
        _box(axis, x, y, 25, 24, fill)
    # One extended index finger, three bent fingers and the thumb/palm.
    vertices = [
        (71, 6), (68, 15), (58, 18), (53, 24),
        (48, 30), (39, 42), (36, 47),
        (32, 55), (38, 61), (44, 54),
        (49, 48), (38, 74),
        (34, 83), (44, 88), (49, 78),
        (59, 57), (60, 62), (67, 63), (70, 56),
        (73, 60), (79, 58), (81, 52),
        (85, 56), (90, 51), (91, 44),
        (96, 32), (88, 25), (95, 10), (97, 6),
    ]
    codes = ([Path.MOVETO] + [Path.CURVE4] * 9 + [Path.LINETO] * 2
             + [Path.CURVE4] * 3 + [Path.LINETO] + [Path.CURVE4] * 12
             + [Path.LINETO])
    _path(axis, vertices, codes, PALE)
    for start, end in (((69, 82), (69, 94)), ((79, 79), (87, 89)), ((87, 69), (98, 73))):
        _line(axis, [start, end])


def _eye_tracking(axis: Axes) -> None:
    _path(axis, [(7, 51), (29, 79), (67, 79), (91, 51),
                 (68, 22), (29, 22), (7, 51)],
          [Path.MOVETO] + [Path.CURVE4] * 6, "white")
    axis.add_patch(Circle((49, 51), 17, facecolor="#999999", edgecolor="none"))
    axis.add_patch(Circle((49, 51), 7, facecolor=INK, edgecolor="none"))
    axis.add_patch(Polygon([(50, 51), (96, 30), (70, 7)], closed=True,
                           facecolor="#D5D5D5", edgecolor="none", alpha=.9, zorder=4))
    for start, end in (((47, 83), (44, 94)), ((61, 83), (67, 95)), ((75, 78), (87, 85))):
        _line(axis, [start, end])


def _survey(axis: Axes) -> None:
    _box(axis, 10, 9, 67, 77, "white", radius=6)
    axis.add_patch(Circle((43, 91), 6, facecolor=MID, edgecolor=INK, linewidth=LINE_WIDTH))
    _box(axis, 27, 81, 32, 11, MID)
    for y in (58, 40, 22):
        _box(axis, 23, y, 11, 11, "white", radius=.6)
        _line(axis, [(43, y + 5.5), (62, y + 5.5)])
    _line(axis, [(25, 65), (29, 61), (37, 71)])
    # Pencil runs outside the form and remains legible at publication size.
    axis.add_patch(Polygon([(67, 15), (83, 60), (95, 55), (79, 11), (64, 4)],
                           closed=True, facecolor=MID, edgecolor=INK,
                           linewidth=LINE_WIDTH, joinstyle="round", zorder=4))
    _line(axis, [(67, 15), (79, 11)])
    _line(axis, [(81, 53), (92, 48)])


def add_source_icon(figure: Figure, kind: str, *, left: float, top: float,
                    size_mm: float = 4.8) -> Axes:
    """Place a square source icon at a figure-relative left/top anchor."""
    drawers = {"task_behavior": _task_behavior, "eye_tracking": _eye_tracking,
               "survey": _survey}
    width = size_mm / (figure.get_figwidth() * 25.4)
    height = size_mm / (figure.get_figheight() * 25.4)
    axis = figure.add_axes((left, top - height, width, height), label=f"source_{kind}_{left}_{top}")
    axis.set(xlim=(0, 104), ylim=(0, 104), aspect="equal")
    axis.set_axis_off()
    axis.patch.set_alpha(0)
    drawers[kind](axis)
    return axis
