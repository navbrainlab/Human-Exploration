"""Render compact five-panel Fig4 with tighter outcome comparisons."""
from __future__ import annotations
from axis_break_marks import mark_nonzero_yaxis, audit_yaxis_breaks

from matplotlib.ticker import FuncFormatter
import argparse
import hashlib
import importlib.util
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox, ScaledTranslation
from matplotlib.patches import FancyBboxPatch, Rectangle, Ellipse, FancyArrowPatch
from matplotlib.ticker import PercentFormatter
from matplotlib.colors import to_rgba, to_hex
import numpy as np
import pandas as pd
from PIL import Image
from scipy.interpolate import PchipInterpolator

from audit_panel_alignment import require_matplotlib_panel_alignment
from detached_axes import audit_detached_axes as audit_all_cartesian_axes
from scientific_coordinates import quantitative_coordinates
from prepare_d_display import load_display
from tick_bounded_spines import bound_spines_to_ticks, audit_tick_bounded_spines
from category_label_spacing import compact_category_margin, category_gaps

ROOT = Path(__file__).resolve().parents[1]
EDITION = "fig4"
WIDTH_MM, HEIGHT_MM = 180., 156.
PLOT_WIDTH_MM, PLOT_HEIGHT_MM = 39., 30.
CATEGORY_WIDTH_MM = 27.9
CATEGORY_XLIM = (-.65, 1.65)
BAR_WIDTH_MM = 4.35
COL_LEFT_MM = (13., 75., 137.)
ROW_TOP_MM = (12., 64., 114.)
TASK_PLOT_WIDTH_MM, TASK_PLOT_GAP_MM = 33., 5.
TOP_RIGHT_SHIFT_MM = 10.7
TOP_LEFT_MM = {"a":13.+TOP_RIGHT_SHIFT_MM,
               "b":13.+TASK_PLOT_WIDTH_MM+TASK_PLOT_GAP_MM+TOP_RIGHT_SHIFT_MM,
               "c":107.6+TOP_RIGHT_SHIFT_MM}
SHARED_TASK_XLABEL = "% of DIS trials (first half)"
LOWER_GROUP_SHIFT_MM = 7.55
OUTCOME_LEFT_MM = {"e":81.+LOWER_GROUP_SHIFT_MM,
                   "f":133.+LOWER_GROUP_SHIFT_MM,
                   "h":81.+LOWER_GROUP_SHIFT_MM,
                   "i":133.+LOWER_GROUP_SHIFT_MM}
RESULT_HEADER_ABOVE_PLOT_MM = 8.
DISPLAY_PANEL_MAP = {"a":"a","b":"a","c":"b","d":"c", "e":"d","f":"d","h":"e","i":"e"}
PUBLIC_PANEL_MEMBERS = {"a":["a","b"], "b":["c"], "c":["d"],
                        "d":["e","f"], "e":["h","i"]}
ROLE_TITLES = {"d":"Manipulation"}
HEADER_ABOVE_PLOT_MM = 5.898  # Measured from Fig5 g/h/i's final layout manifest.
HEADER_LETTER_LEFT_MM = 9.  # Retain first-column margin on this180-mm canvas.
HEADER_ICON_OFFSET_MM = 3.8
Y_SPINE_OUTWARD_PT, X_SPINE_OUTWARD_PT = 5., 3.
DPI = 600
BLUE = "#2F5FAE"
ORANGE = "#D57224"
PURPLE = "#60458F"
INK = "#252B30"
RAW_GREY = "#8D9399"
MARK_EDGE = "#4C535A"
BAR_FILL_ALPHA = .80
RIBBON_ALPHA = .20
RAW_POINT_ALPHA = .55
# These four illustrative feature colours are separate from condition colours.
SOFA_COLOURS = ("#348F77", "#397BA6", "#C58A36", "#7D5AA6")
SOFA_EDGE = "#455363"
CARD_EDGE = "#728294"
BRANCH_COLOUR = "#596A79"
GROUPS = (("Observation", "Obs", ORANGE), ("DIS-blocked", "No-DIS", RAW_GREY))
LEGACY_DISPLAY_LABEL = "DIS-D"  # Frozen source label; never used for plotting.
SCHEMATIC_WIDTH_MM, SCHEMATIC_HEIGHT_MM = 47.5, 80.
# Expand into the unused left-hand space; keep the outcome-facing edge fixed.
SCHEMATIC_RIGHT_MM = 8.+LOWER_GROUP_SHIFT_MM+(56.1-36.)/2+36.
SCHEMATIC_RIGHT_MM += 5.0  # Center concept content in its available column.
SCHEMATIC_LEFT_MM = SCHEMATIC_RIGHT_MM-SCHEMATIC_WIDTH_MM
SCHEMATIC_TOP_MM = ROW_TOP_MM[1]
C_GROUP_HEADER_TOP_MM = SCHEMATIC_TOP_MM-RESULT_HEADER_ABOVE_PLOT_MM
SCHEMATIC_GROUP_STYLE = {
    "Obs": {"body": "#FCF1E8", "edge": "#8E4414", "header": "#AF5F23"},
    "No-DIS": {"body": "#F1F2F3", "edge": "#62676B", "header": "#8D9399"},
}
PHASES = ("Post-observation", "Early", "Mid", "Late")
Y_LABELS = {
    "a": "Mean score\n(final 5% of trials)",
    "b": "",  # Shared response label belongs to merged public panel a.
    "c": "% of DIS trials\n(early 4D-E)",
    "e": "Mean score",
    "f": "Full-score proportion",
    "g": "Estimated mean score",
    "h": "Elements noticed\n(count)",
    "i": "Dimensions correctly\nclassified (%)",
}
PANEL_COLUMNS = {"a": 0, "b": 1, "c": 2, "d": 0, "e": 1, "f": 2,
                 "g": 0, "h": 1, "i": 2}
SOURCE_ICON_KINDS = {name: "survey" if name in "hi" else "task_behavior"
                     for name in "abcdefghi"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 5, "font.weight": "regular", "axes.labelsize": 6,
    "axes.labelweight": "regular", "axes.titlesize": 6, "axes.titleweight": "regular",
    "xtick.labelsize": 5, "ytick.labelsize": 5, "legend.fontsize": 5,
    "legend.frameon": False, "axes.linewidth": .70,
    "axes.spines.right": False, "axes.spines.top": False,
    "xtick.major.width": .70, "ytick.major.width": .70,
    "xtick.major.size": 2.1, "ytick.major.size": 2.1,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.grid": False, "axes.edgecolor": INK, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "svg.fonttype": "none", "pdf.fonttype": 42,  # Text defaults only; PNG remains the sole export.
})


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_inputs():
    manifest = json.loads((ROOT / "input_manifest.json").read_text())
    for relative, expected in manifest.items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Frozen input changed: {relative}")
    return manifest


def read(name):
    return pd.read_csv(ROOT / "csv" / name)


def mm_axis(fig, left, top, width=PLOT_WIDTH_MM, height=PLOT_HEIGHT_MM):
    return fig.add_axes([left / WIDTH_MM, (HEIGHT_MM - top - height) / HEIGHT_MM,
                         width / WIDTH_MM, height / HEIGHT_MM])


def style(ax, xlim, ylim, xticks, yticks, xlabel, ylabel):
    ax.set(xlim=xlim, ylim=ylim, xticks=xticks, yticks=yticks,
           xlabel=xlabel, ylabel=ylabel)
    ax.tick_params(pad=2.0)
    ax.xaxis.labelpad = 3.0
    ax.yaxis.labelpad = 4.0


def header(ax, letter, title):
    if letter in "bfi":
        return  # Shared panels show their letter only above the first subplot.
    if letter in ROLE_TITLES:
        ax.text(.5, 1.19, ROLE_TITLES[letter], transform=ax.transAxes,
                fontsize=6, weight="regular", ha="center", va="bottom",
                clip_on=False, gid=f"role-title-{letter}")
        if letter not in "eh":
            return
    # Fixed physical offsets keep letters aligned across all three rows.
    logical_column = PANEL_COLUMNS[letter]
    left = ax.get_position().x0 * WIDTH_MM
    width = ax.get_position().width * WIDTH_MM
    anchor = TOP_LEFT_MM[letter] if letter in TOP_LEFT_MM else OUTCOME_LEFT_MM[letter]
    letter_x = (anchor - HEADER_LETTER_LEFT_MM - left) / width
    above = RESULT_HEADER_ABOVE_PLOT_MM if letter in "eh" else HEADER_ABOVE_PLOT_MM
    letter = DISPLAY_PANEL_MAP[letter]
    ax.text(letter_x, 1 + above / PLOT_HEIGHT_MM, letter,
            transform=ax.transAxes, fontsize=8, weight="bold", ha="left", va="top",
            clip_on=False, gid=f"panel-letter-{letter}")


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


def slope_star(ax, p):
    mark = stars(p)
    if mark:
        ax.text(1., 1.088, mark, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6, weight="bold", clip_on=False)


def bracket(ax, p):
    mark = stars(p)
    if not mark:
        return
    ax.plot([0, 0, 1, 1], [1.045, 1.070, 1.070, 1.045], color=INK,
            transform=ax.get_xaxis_transform(), lw=.65, clip_on=False,
            solid_capstyle="butt")
    ax.text(.5, 1.088, mark, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
            fontsize=6, weight="bold", clip_on=False)


def category_axis(ax):
    ax.spines["bottom"].set_bounds(0, 1)
    ax.spines["bottom"].set_position(("outward", 3))
    ax.tick_params(axis="x", direction="out", length=2.3, pad=2.5)


def detach_axes(ax):
    """Leave space between the two spines and data, without shifting data."""
    ax.spines["left"].set_position(("outward", Y_SPINE_OUTWARD_PT))
    ax.spines["bottom"].set_position(("outward", X_SPINE_OUTWARD_PT))


def align_break_to_spine(ax):
    if not hasattr(ax, "_publication_y_break_artists"):
        return
    transform = ax.transAxes + ScaledTranslation(-Y_SPINE_OUTWARD_PT / 72., 0,
                                                ax.figure.dpi_scale_trans)
    for artist in ax._publication_y_break_artists:
        artist.set_transform(transform)
    ax._publication_y_break["left_spine_outward_pt"] = Y_SPINE_OUTWARD_PT


def audit_detached_axes(axes):
    """Measure actual axis/omission-stroke clearance at final render DPI."""
    reports = {}
    for name in "aceh":
        ax = axes[name]
        scale = 25.4 / ax.figure.dpi
        left = ax.spines["left"]
        bottom = ax.spines["bottom"]
        y_end = left.get_transform().transform(left.get_path().vertices)[0]
        x_start = bottom.get_transform().transform(bottom.get_path().vertices)[0]
        expected_left = ax.bbox.x0 - Y_SPINE_OUTWARD_PT / 72. * ax.figure.dpi
        assert np.isclose(y_end[0], expected_left)
        gap = np.linalg.norm(x_start - y_end) * scale
        item = {"display_panel": DISPLAY_PANEL_MAP[name],
                "left_spine_outward_pt": Y_SPINE_OUTWARD_PT,
                "bottom_spine_outward_pt": X_SPINE_OUTWARD_PT,
                "corner_endpoint_distance_mm": float(gap)}
        if hasattr(ax, "_publication_y_break_artists"):
            slashes = [line for line in ax._publication_y_break_artists
                       if "y-break-slash" in str(line.get_gid())]
            right_edges = []
            for line in slashes:
                pixels = line.get_transform().transform(line.get_xydata())
                assert np.isclose(pixels[:, 0].mean(), expected_left)
                right_edges.append(pixels[:, 0].max() + line.get_linewidth() / 144. * ax.figure.dpi)
            # Conservative horizontal clearance includes both visible strokes.
            gap_mm = (x_start[0] - max(right_edges) - bottom.get_linewidth() / 144. * ax.figure.dpi) * scale
            assert gap_mm >= .5, (name, "Omission strokes touch x-axis endpoint", gap_mm)
            ticks = [tick for tick in ax.xaxis.get_major_ticks()
                     if ax.get_xlim()[0] <= tick.get_loc() <= ax.get_xlim()[1]]
            first_tick = min(ticks, key=lambda tick: tick.get_loc()).tick1line
            tick_x = first_tick.get_transform().transform(first_tick.get_xydata())[0, 0]
            tick_gap_mm = (tick_x - max(right_edges) - first_tick.get_markeredgewidth() / 144. * ax.figure.dpi) * scale
            assert tick_gap_mm >= .5, (name, "Omission strokes touch first x tick", tick_gap_mm)
            renderer = ax.figure.canvas.get_renderer()
            from axis_break_marks import CENTERS_MM, SLASH_HEIGHT_MM, WIDTH_MM
            region = Bbox.from_extents(expected_left - (WIDTH_MM / 2 + .13) / scale,
                ax.bbox.y0 + (min(CENTERS_MM) - SLASH_HEIGHT_MM / 2 - .13) / scale,
                expected_left + (WIDTH_MM / 2 + .13) / scale,
                ax.bbox.y0 + (max(CENTERS_MM) + SLASH_HEIGHT_MM / 2 + .13) / scale)
            assert not any(region.overlaps(t.get_window_extent(renderer))
                           for t in ax.get_xticklabels() if t.get_visible())
            item["omission_stroke_to_x_endpoint_clearance_mm"] = float(gap_mm)
            item["omission_stroke_to_first_x_tick_clearance_mm"] = float(tick_gap_mm)
            item["omission_marks_aligned_to_actual_y_spine"] = True
        reports[name] = item
    return reports


def sofa(ax, x, y, color, scale=1.4):
    """Native explanatory sofa icon; no quantitative measurements encoded."""
    edge = SOFA_EDGE
    for bounds in [(x - 2.1, y + .3, 4.2, 1.5), (x - 3.0, y - 1.2, 6.0, 1.6),
                   (x - 3.7, y - .8, 1.3, 1.7), (x + 2.4, y - .8, 1.3, 1.7)]:
        bx,by,bw,bh=bounds
        position=(x+(bx-x)*scale,y+(by-y)*scale)
        ax.add_patch(FancyBboxPatch(position,bw*scale,bh*scale,
                     boxstyle=f"round,pad=0,rounding_size={.45*scale}", facecolor=color,
                     edgecolor=edge, linewidth=.55, gid="schematic-sofa"))
    for leg in (-2.6, 2.6):
        ax.plot([x+leg*scale,x+leg*scale],[y-1.2*scale,y-1.8*scale],color=edge,lw=.55)
    # Fine upholstery seams add definition while line weight stays unchanged.
    for low,high in [(-.9,.1),(.65,1.5)]:
        ax.plot([x,x],[y+low*scale,y+high*scale],color=edge,lw=.35,alpha=.65,
                gid="schematic-sofa-cushion-seam")


def draw_manipulation(fig):
    """Parallel groups stacked vertically, with shared observation between them."""
    ax = mm_axis(fig, SCHEMATIC_LEFT_MM, SCHEMATIC_TOP_MM,
                 SCHEMATIC_WIDTH_MM, SCHEMATIC_HEIGHT_MM)
    ax.set_label("d")
    ax.set(xlim=(0,SCHEMATIC_WIDTH_MM),ylim=(0,80))
    ax.set_axis_off()
    center=SCHEMATIC_WIDTH_MM/2
    ax.text(center,40,"Prior observation",fontsize=6,ha="center",va="center",
            gid="shared-prior-observation-label")
    neutral = BRANCH_COLOUR
    for condition,start,end in [('Obs',(center,42.6),(center,45.55)),
                                ('No-DIS',(center,37.4),(center,34.45))]:
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle="-|>",
                     gid=f"observation-to-{condition}",
                     mutation_scale=5, color=neutral, linewidth=.75, shrinkA=0, shrinkB=0))
    left=(SCHEMATIC_WIDTH_MM-38.)/2
    for bottom,condition in [(46.2,"Obs"),(.3,"No-DIS")]:
        palette = SCHEMATIC_GROUP_STYLE[condition]
        frame = FancyBboxPatch((left,bottom),38.,33.5,
                     boxstyle="round,pad=0,rounding_size=.75",
                     facecolor=palette["body"], edgecolor=palette["edge"], linewidth=.65,
                     gid=f"schematic-condition-frame-{condition}")
        ax.add_patch(frame)
        band = Rectangle((left,bottom+30.1),38.,3.4,facecolor=palette["header"],
                         edgecolor="none", gid=f"schematic-condition-header-{condition}")
        band.set_clip_path(frame)
        ax.add_patch(band)
        ax.text(center,bottom+31.8,condition,fontsize=6,color="#FFFFFF",
                ha="center", va="center")
        colors = list(SOFA_COLOURS)
        if condition == "No-DIS":
            colors = [colors[0]] * 4
        for index,(x,y_offset) in enumerate([(10.5,23.4),(27.5,23.4),(10.5,13.2),(27.5,13.2)]):
            y=bottom+y_offset
            ax.add_patch(Rectangle((left+x-6.9,y-3.5),13.8,7.,
                         facecolor="#F3F6F9", edgecolor=CARD_EDGE, linewidth=.50,
                         gid=f"schematic-stimulus-card-{condition}-{index}"))
            sofa(ax,left+x,y,colors[index])
        footer_x=center
        ax.text(footer_x,bottom+3.7,"Standard rule" if condition=="Obs" else "DIS blocked",
                fontsize=5, ha="center", va="center")
    block_color = "#AF4452"
    badge=(center,18.2)
    ax.add_patch(Ellipse(badge,23.6,23.6,fill=False,edgecolor=block_color,lw=1.25,zorder=8,
                         gid="blocked-condition-symbol"))
    offset=11.8/np.sqrt(2)
    ax.plot([badge[0]-offset,badge[0]+offset],[badge[1]+offset,badge[1]-offset],
            color=block_color,lw=1.25,zorder=9,gid="blocked-condition-slash")
    texts = [fig.text((4.+LOWER_GROUP_SHIFT_MM) / WIDTH_MM,
                      (HEIGHT_MM - C_GROUP_HEADER_TOP_MM) / HEIGHT_MM,
                      DISPLAY_PANEL_MAP["d"], fontsize=8, weight="bold", ha="left", va="top",
                      gid="panel-letter-c")]
    return ax, texts


def source_icon(ax, panel):
    """Reuse the approved symbol matching the panel's task/report evidence."""
    contract = json.loads((ROOT / "assets/source_icons_manifest.json").read_text())
    portable = ROOT / contract["portable_code"]
    if digest(portable) != contract["code_sha256"]:
        raise ValueError("Portable approved icon source changed")
    source = portable
    spec = importlib.util.spec_from_file_location("approved_publication_source_icons", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fig = ax.figure
    kind = SOURCE_ICON_KINDS[panel]
    left = (TOP_LEFT_MM[panel] - HEADER_LETTER_LEFT_MM + HEADER_ICON_OFFSET_MM
            if panel in TOP_LEFT_MM else 7.8+LOWER_GROUP_SHIFT_MM if panel == "d" else
            OUTCOME_LEFT_MM[panel]-HEADER_LETTER_LEFT_MM+HEADER_ICON_OFFSET_MM)
    top_mm = (C_GROUP_HEADER_TOP_MM if panel == "d" else
              HEIGHT_MM - ax.get_position().y1 * HEIGHT_MM -
              (RESULT_HEADER_ABOVE_PLOT_MM if panel in OUTCOME_LEFT_MM else HEADER_ABOVE_PLOT_MM))
    icon = module.add_source_icon(fig, kind, left=left / WIDTH_MM,
        top=(HEIGHT_MM - top_mm) / HEIGHT_MM,
        size_mm=contract["figure_size_mm"])
    report = {**contract, "kind": kind, "panel": panel,
              "evidence_role": "task design" if panel == "d" else "measured evidence",
              "visual_identity": "grayscale checklist plus three-dot speech bubble" if kind == "survey"
                                 else "grayscale selection tiles and hand",
              "source_loaded": str(source), "bounds_fraction": list(icon.get_position().bounds),
              "native_render": True, "embedded_raster_images": len(icon.images)}
    return icon, report


def load_i_analysis():
    metadata = json.loads((ROOT / "analysis/i_analysis.json").read_text())
    for relative, expected in {**metadata["source_inputs_sha256"],
                               **metadata["derived_files_sha256"]}.items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"i-panel analysis input changed: {relative}")
    summary = pd.read_csv(ROOT / "analysis/i_noticed_summary.csv")
    contrasts = pd.read_csv(ROOT / "analysis/i_stage_contrasts.csv").fillna({"stars": ""})
    return summary, contrasts, metadata


def group_legend(ax, loc="lower right", ncol=1):
    ax.legend(loc=loc, ncol=ncol, handlelength=1.55, handletextpad=.5,
              borderaxespad=.45, labelspacing=.35, columnspacing=.9)


def native_text_audit(fig):
    """Measure all visible figure labels, including the native schematic text.

    Does not claim the PDF text/path collision audit prohibited by PNG-only policy.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    page = fig.bbox
    native = []
    texts = list(fig.texts)
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        texts.extend(ax.texts)
        if ax.axison:
            texts.extend([ax.title, ax._left_title, ax._right_title,
                          ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels()])
            if ax.get_legend() is not None:
                texts.extend(ax.get_legend().get_texts())
    for text in texts:
        if not text.get_visible() or not text.get_text().strip():
            continue
        bbox = text.get_window_extent(renderer)
        native.append((text, bbox))
    clipping = [t.get_text() for t, b in native
                if b.x0 < page.x0 - 1 or b.y0 < page.y0 - 1
                or b.x1 > page.x1 + 1 or b.y1 > page.y1 + 1]
    collisions = []
    for (first, a), (second, b) in combinations(native, 2):
        intersection = Bbox.intersection(a, b)
        # Rotated-label axis-aligned bounds are conservative; any finding blocks
        # export until reviewed and corrected, rather than silently exempting it.
        if intersection is not None and intersection.width > 1 and intersection.height > 1:
            collisions.append([first.get_text(), second.get_text()])
    font_sizes = sorted({float(t.get_fontsize()) for t, _ in native})
    font_names = sorted({t.get_fontproperties().get_name() for t, _ in native})
    ordinary_bold = [t.get_text() for t, _ in native
                     if t.get_fontweight() == "bold" and
                     t.get_text() not in tuple("abcdefghi") + ("*", "**", "***")]
    report = {"method": "Matplotlib rendered native-text bounding boxes at export dpi",
              "font_sizes_pt": font_sizes, "native_text_count": len(native),
              "font_names": font_names,
              "visible_text": [t.get_text() for t, _ in native],
              "text_text_collisions": collisions, "page_clipping": clipping,
              "unexpected_bold_text": ordinary_bold,
              "embedded_raster_text": "none; all diagram text redrawn natively",
              "pdf_collision_audit": "not run: publication PNG-only output policy"}
    label_boxes=[]
    for text,bounds in native:
        if text.get_text()!='No-DIS':continue
        current_bounds=list(bounds.extents)
        text.set_text(LEGACY_DISPLAY_LABEL)
        previous_bounds=list(text.get_window_extent(renderer).extents)
        text.set_text('No-DIS')
        label_boxes.append(dict(current=current_bounds,previous=previous_bounds))
    report['renamed_condition_label_boxes_pixels']=label_boxes
    condition_legends=[ax.get_legend() for ax in fig.axes
                       if ax.get_legend() is not None and
                       any(text.get_text()=='No-DIS' for text in ax.get_legend().get_texts())]
    current_legend_boxes=[list(legend.get_window_extent(renderer).extents)
                          for legend in condition_legends]
    for legend in condition_legends:
        for text in legend.get_texts():
            if text.get_text()=='No-DIS':text.set_text(LEGACY_DISPLAY_LABEL)
    fig.canvas.draw()
    previous_legend_boxes=[list(legend.get_window_extent(fig.canvas.get_renderer()).extents)
                           for legend in condition_legends]
    for legend in condition_legends:
        for text in legend.get_texts():
            if text.get_text()==LEGACY_DISPLAY_LABEL:text.set_text('No-DIS')
    fig.canvas.draw()
    report['renamed_condition_legend_boxes_pixels']=[dict(current=current,previous=previous)
        for current,previous in zip(current_legend_boxes,previous_legend_boxes)]
    if font_sizes != [5., 6., 8.] or font_names != ["DejaVu Sans"] or ordinary_bold or clipping or collisions:
        raise ValueError(f"Native text QA failed: {json.dumps(report)}")
    return report


def build(output_dir=None, panel_previews=False):
    input_manifest = check_inputs()
    output_dir = Path(output_dir).resolve() if output_dir else ROOT.parent
    target = output_dir / f"{EDITION}.png"
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {target}; use a fresh --output-dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    qa_dir = ROOT / "qa" if output_dir == ROOT.parent else output_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    statistics = read("panel_statistics.csv")
    def stat(analysis, sample=None):
        rows = statistics.loc[statistics.analysis.eq(analysis)]
        if sample is not None:
            rows = rows.loc[rows["sample"].eq(sample)]
        if len(rows) != 1:
            raise ValueError(f"Ambiguous frozen statistic: {analysis}/{sample}")
        return rows.iloc[0]

    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), dpi=DPI)
    axes = {}
    data_geometry = {}
    for letter in "abcefhi":
        row_col = {"a": (0, 0), "b": (0, 1), "c": (0, 2), "e": (1, 1),
                   "f": (1, 2), "h": (2, 1), "i": (2, 2)}[letter]
        width = (TASK_PLOT_WIDTH_MM if letter in "ab" else
                 CATEGORY_WIDTH_MM if letter in "fi" else PLOT_WIDTH_MM)
        # Share column left edges: f/i remain narrow but align with c, avoiding
        # a wider apparent gutter on the right of the bottom row after the swap.
        left = TOP_LEFT_MM[letter] if letter in TOP_LEFT_MM else OUTCOME_LEFT_MM[letter]
        axes[letter] = mm_axis(fig, left, ROW_TOP_MM[row_col[0]], width=width)
        axes[letter].set_label(letter)
    axes["b"].sharey(axes["a"])

    associations = (
        ("a", "Task 1", "01a_plot_data.csv", "01a_early_dis_late_score_task1",
         "first50_dis_proportion", "last5_mean_score", BLUE),
        ("b", "Task 2", "01b_plot_data.csv", "01b_early_dis_late_score_task2",
         "first50_dis_proportion", "last5_mean_score", ORANGE),
        ("c", "Across phases", "02_plot_data.csv", "02_p1_to_early_p2_dis",
         "p1_3d_ne_dis_proportion", "early_p2_4d_e_dis_proportion", BLUE),
    )
    for panel, title, filename, analysis, xcol, ycol, color in associations:
        ax = axes[panel]
        data, curve = read(filename), read(f"{panel}_ols_predictions.csv")
        ax.fill_between(curve.x.to_numpy(), curve.ci_low_95.to_numpy(),
                        curve.ci_high_95.to_numpy(), color=color, alpha=RIBBON_ALPHA, lw=0, zorder=1)
        ax.scatter(data[xcol], data[ycol], s=5.0, color=RAW_GREY, alpha=RAW_POINT_ALPHA,
                   edgecolors="none", linewidths=0, zorder=3, clip_on=(panel != "c"))
        ax.plot(curve.x, curve.estimate, color=color, lw=1.10, zorder=4)
        if panel == "c":
            # User-specified proportion viewport; frozen OLS/CI remain unchanged.
            # Portions above 1 are visually clipped, as disclosed in the caption.
            style(ax, (-.03, 1.03), (0, 1), [0, .5, 1], [0, .5, 1],
                  "% of DIS trials (3D-NE)", Y_LABELS[panel])
        else:
            # Both tasks use one live score axis; all points/CI remain in view.
            style(ax, (-.03, 1.03), (20, 103), [0, .5, 1], [20, 60, 100],
                  "", Y_LABELS[panel])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value,pos:f"{100*value:g}"))
        if panel=="c":ax.yaxis.set_major_formatter(FuncFormatter(lambda value,pos:f"{100*value:g}"))
        header(ax, panel, title)
        if panel in "ab":
            ax.text(.5,1.10,title,transform=ax.transAxes,fontsize=5,
                    ha="center",va="bottom",clip_on=False,gid="task-condition-label")
        slope_star(ax, float(stat(analysis).p))
        data_geometry[panel] = {"rows_plotted": len(data), "uncertainty": "HC3 OLS mean 95% CI",
                                "native_artist_count": len(ax.collections)}
        if panel == "c":
            data_geometry[panel]["display_ylim"] = [0, 1]
            data_geometry[panel]["ols_ci_clipped_to_viewport"] = True

    axes["b"].tick_params(axis="y", left=False, right=False, labelleft=False, labelright=False)
    axes["b"].spines["left"].set_visible(False)

    diagram, diagram_texts = draw_manipulation(fig)
    data_geometry["d"] = {"source_asset_sha256": digest(ROOT / "assets" / "condition_schematic_source.png"),
                           "display": "Native reconstruction of the same manipulation",
                           "native_size_mm": [SCHEMATIC_WIDTH_MM, SCHEMATIC_HEIGHT_MM], "sofa_icons": 8,
                           "group_layout":"vertical parallel groups; Obs above No-DIS",
                           "concept_source":"fig4_nhb_style public c",
                           "native_units":"1mm per native coordinate on both axes",
                           "shared_prior_observation_position":"between the two condition cards",
                           "card_bounds":{condition:[p.get_x(),p.get_y(),p.get_width(),p.get_height()]
                              for condition in ["Obs","No-DIS"] for p in diagram.patches
                              if p.get_gid()==f"schematic-condition-frame-{condition}"},
                           "branch_arrow_geometry":{condition:p._posA_posB
                              for condition in ["Obs","No-DIS"] for p in diagram.patches
                              if p.get_gid()==f"observation-to-{condition}"},
                           "sequential_group_arrow":False,
                           "left_mm": SCHEMATIC_LEFT_MM, "left_edge_matches_public_a_task1": False,
                           "rendered_bounds_fraction": list(diagram.get_position().bounds),
                           "right_mm": SCHEMATIC_LEFT_MM + SCHEMATIC_WIDTH_MM,
                           "condition_card_style": SCHEMATIC_GROUP_STYLE,
                           "condition_card_size_mm":[38.,33.5],
                           "condition_card_gap_mm":12.4,
                           "stimulus_card_size_mm":[13.8,7.],
                           "sofa_linear_scale_relative_to_prior":1.4,
                           "frame_linewidth_pt":.65,"header_height_mm":3.4,
                           "prohibition_badge_diameter_mm":23.6,
                           "prohibition_badge_placement":"Single overlay across all four No-DIS green sofas; no footer symbol",
                           "shared_observation_branches": sum(isinstance(p,FancyArrowPatch) for p in diagram.patches), "blocked_example": "No-DIS",
                           "embedded_raster_images": 0, "font_sizes_pt": [5, 6],
                           "alignment_exclusion": "Conceptual schematic; not a statistical plot"}

    ax = axes["e"]
    descriptive, d_metadata = load_display()
    for condition, label, color in GROUPS:
        data = descriptive.loc[descriptive.condition.eq(condition)].sort_values("trial")
        trial = data.trial.to_numpy(float)
        if not np.all(np.diff(trial) > 0):
            raise ValueError("Descriptive trial grid must increase strictly")
        mean = data["mean"].to_numpy(float)
        sem = data["sem"].to_numpy(float)
        dense = np.linspace(trial.min(), trial.max(), 400)
        y = PchipInterpolator(trial, mean)(dense)
        spread = np.maximum(PchipInterpolator(trial, sem)(dense), 0)
        ax.fill_between(dense, y - spread, y + spread, color=color, alpha=RIBBON_ALPHA, lw=0)
        ax.plot(dense, y, color=color, lw=1.10, label=label)
    style(ax, (0, 60), (45, 90), [0, 20, 40, 60], [45, 60, 75, 90], "Trial", Y_LABELS["e"])
    header(ax, "e", "Learning trajectory")
    group_legend(ax)
    data_geometry["e"] = {"summary_rows": len(descriptive), "participant_trial_rows": 4800,
                           "uncertainty": "Mean ±1 SEM across5-trial-smoothed participant scores; PCHIP display",
                           "display_analysis": d_metadata}

    ax = axes["f"]
    bar_width = BAR_WIDTH_MM * (CATEGORY_XLIM[1]-CATEGORY_XLIM[0]) / CATEGORY_WIDTH_MM
    bar_patches = []
    for x, (condition, label, color) in enumerate(GROUPS):
        frozen = stat("05_ever_full_score", condition)
        mean = float(frozen["rate"])
        patch = ax.bar(x, mean, width=bar_width, color=to_rgba(color, BAR_FILL_ALPHA),
                       edgecolor=MARK_EDGE, linewidth=.75)[0]
        bar_patches.append(patch)
    style(ax, CATEGORY_XLIM, (0, .5), [0, 1], [0, .25, .5], "", Y_LABELS["f"])
    ax.set_xticklabels([g[1] for g in GROUPS])
    header(ax, "f", "Full-score rate")
    category_axis(ax)
    data_geometry["f"] = {"participants": 80, "events_by_group": [18, 9],
                           "uncertainty_displayed": False,
                           "preserved_uncertainty": "Wilson 95% CI in frozen table/Methods",
                           "preserved_barnard_p": float(stat("05_full_score_rate_difference").p),
                           "display_omission_is_not_nonsignificance": True}

    ax = axes["i"]
    accuracy = read("07_plot_data.csv")
    for x, (condition, label, color) in enumerate(GROUPS):
        frozen = stat("07_relevance_classification_accuracy", condition)
        mean, low, high = [float(frozen[k]) for k in
                           ["mean", "ci_low_95_bootstrap", "ci_high_95_bootstrap"]]
        patch = ax.bar(x, mean, width=bar_width, color=to_rgba(color, BAR_FILL_ALPHA),
                       edgecolor=MARK_EDGE, linewidth=.75)[0]
        bar_patches.append(patch)
        ax.errorbar(x, mean, yerr=[[mean - low], [high - mean]], fmt="none", color=INK,
                    elinewidth=.75, capsize=2.0, capthick=.75, zorder=4)
    style(ax, CATEGORY_XLIM, (0, 1), [0, 1], [0, .25, .5, .75, 1], "", Y_LABELS["i"])
    # Retain proportion data-space coordinates; only the ticks express percent.
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0, symbol=""))
    ax.set_xticklabels([g[1] for g in GROUPS])
    header(ax, "i", "Dimension classification")
    bracket(ax, float(stat("07_relevance_accuracy_difference").p))
    category_axis(ax)
    data_geometry["i"] = {"input_rows": len(accuracy), "individual_points_plotted": 0, "group_n": [40, 39],
                           "uncertainty": "frozen bootstrap mean 95% CI", "summary_style": "bar +95% CI",
                           "source_panel_before_swap": "h", "data_unit": "proportion", "tick_unit": "percent"}

    ax = axes["h"]
    summary, contrasts, i_metadata = load_i_analysis()
    knowledge = pd.read_csv(ROOT / "analysis/i_noticed_subjects.csv")
    for (condition, label, color), offset in zip(GROUPS, [-.04, .04]):
        predictions = summary.loc[summary.condition.eq(condition)].set_index("phase").loc[list(PHASES)]
        x = np.arange(4) + offset
        ax.errorbar(x, predictions["mean"].to_numpy(), yerr=predictions["sem"].to_numpy(),
                    color=color, linewidth=1.10, fmt="-o", markersize=3.0,
                    elinewidth=.75, capsize=1.6, capthick=.75, zorder=4)
    style(ax, (-.25, 3.25), (4, 12), [0, 1, 2, 3], [4, 6, 8, 10, 12],
          "", Y_LABELS["h"])
    ax.set_xticklabels(["Post-obs.", "Early", "Mid", "Late"])
    header(ax, "h", "Reported dimensions")
    handles = [Line2D([], [], color=color, marker="o", linestyle="none", markersize=3,
                      label=label) for _, label, color in GROUPS]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(.20, 1.055),
              ncol=2, borderaxespad=0, handlelength=.8, handletextpad=.5, columnspacing=1.2)
    for row in contrasts.itertuples(index=False):
        if row.stars:
            upper = summary.loc[summary.phase.eq(row.phase)]
            y = float((upper["mean"] + upper["sem"]).max()) + .48
            ax.text(row.stage_order - 1, y, row.stars, ha="center", va="bottom",
                    fontsize=6, weight="bold")
    data_geometry["h"] = {"input_rows": len(knowledge), "individual_points_plotted": 0,
                           "summary_rows": len(summary), "participants": 80,
                           "uncertainty": "observed group mean ±1 SEM",
                           "stage_comparisons": contrasts.to_dict("records"),
                           "analysis": i_metadata, "display_ylim": [4, 12],
                           "subplot_titles": "none", "source_panel_before_swap": "i"}

    icon_axes, icon_reports = {}, {}
    for panel in "acdeh":
        icon_axes[panel], icon_reports[panel] = source_icon(diagram if panel == "d" else axes[panel], panel)
        data_geometry[panel]["source_icon"] = icon_reports[panel]

    data_geometry["b"]["shared_header_with"] = "a"
    categorical_mark_spacing = {}
    for panel,axis in axes.items():
        if panel in "abcefhi":
            detach_axes(axis)
        if panel != "b":
            mark_nonzero_yaxis(axis,panel=panel)
            if panel in "acefhi":
                align_break_to_spine(axis)
        else:
            axis.tick_params(axis="x", pad=axes["a"].xaxis.get_major_ticks()[0].get_pad())
        if panel in "fi":
            categorical_mark_spacing[panel] = compact_category_margin(axis,
                [patch for patch in bar_patches if patch.axes is axis])
        bound_spines_to_ticks(axis)
    breaks=audit_yaxis_breaks(axes)
    # Public g has extra tick padding beside its omitted-zero marks. Move
    # public h's detached categorical baseline down until text baselines match.
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    reference_y=axes['h'].get_xticklabels()[0].get_window_extent(renderer).y0
    target_y=axes['i'].get_xticklabels()[0].get_window_extent(renderer).y0
    spine_offset=3.+(target_y-reference_y)*72./DPI
    axes['i'].spines['bottom'].set_position(('outward',spine_offset))
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    matched_y=axes['i'].get_xticklabels()[0].get_window_extent(renderer).y0
    assert abs(matched_y-reference_y)<.25
    detached_axes = audit_detached_axes(axes)
    detached_axes_all = audit_all_cartesian_axes(axes)
    tick_endpoints = audit_tick_bounded_spines(axes)
    for name in "fi":
        categorical_mark_spacing[name]["after"] = category_gaps(axes[name],
            [patch for patch in bar_patches if patch.axes is axes[name]])
    # Use the lower of the two native tick-label baselines, after break padding.
    task_ticks = [text.get_window_extent(renderer) for panel in "ab"
                  for text in axes[panel].get_xticklabels() if text.get_visible()]
    task_label_top = min(box.y0 for box in task_ticks) - .8 * DPI / 25.4
    task_centre = (axes["a"].bbox.x0 + axes["b"].bbox.x1) / 2
    shared_task_xlabel = fig.text(task_centre / fig.bbox.width,
        task_label_top / fig.bbox.height, SHARED_TASK_XLABEL,
        fontsize=6, weight="regular", ha="center", va="top", gid="shared-task-xlabel")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    task_label_bbox = shared_task_xlabel.get_window_extent(renderer)
    task1_right = max(axes["a"].bbox.x1 + axes["a"].spines["bottom"].get_linewidth() / 144. * DPI,
                      *(text.get_window_extent(renderer).x1 for text in axes["a"].get_xticklabels()
                        if text.get_visible()))
    task2_decorations = [text.get_window_extent(renderer)
                         for text in axes["b"].get_xticklabels() if text.get_visible()]
    bottom = axes["b"].spines["bottom"]
    task2_decorations.append(bottom.get_window_extent(renderer).padded(bottom.get_linewidth() / 144. * DPI))
    task_gap_clearance = (min(box.x0 for box in task2_decorations) - task1_right) * 25.4 / DPI
    assert task_gap_clearance >= .5, ("Task2 visible content touches Task1", task_gap_clearance)
    shared_y_axis = {
        "live_joined": bool(axes["a"].get_shared_y_axes().joined(axes["a"], axes["b"])),
        "shared_major_locator": axes["a"].yaxis.get_major_locator() is axes["b"].yaxis.get_major_locator(),
        "ylim": list(axes["a"].get_ylim()), "yticks": axes["a"].get_yticks().tolist(),
        "decoration_source": "a", "right_y_tick_labels": [t.get_text() for t in axes["b"].get_yticklabels()],
        "right_y_ticks_visible": any(t.tick1line.get_visible() or t.tick2line.get_visible()
                                     for t in axes["b"].yaxis.get_major_ticks()),
        "right_y_spines_visible": any(axes["b"].spines[side].get_visible() for side in ["left", "right"]),
        "right_y_break_artists": hasattr(axes["b"], "_publication_y_break_artists"),
    }
    assert shared_y_axis["live_joined"] and shared_y_axis["shared_major_locator"]
    assert axes["a"].get_ylim() == axes["b"].get_ylim() == (20, 103)
    assert not any(shared_y_axis[key] for key in ["right_y_tick_labels", "right_y_ticks_visible",
                                                "right_y_spines_visible", "right_y_break_artists"])
    merged_panel_a = {"source_ids": ["a", "b"], "task_labels": ["Task 1", "Task 2"],
        "plot_lefts_mm": [TOP_LEFT_MM[panel] for panel in "ab"],
        "plot_widths_mm": [TASK_PLOT_WIDTH_MM] * 2, "inner_plot_gap_mm": TASK_PLOT_GAP_MM,
        "shared_ylabel": True, "shared_y_axis": shared_y_axis,
        "task_plot_visible_content_clearance_mm": float(task_gap_clearance),
        "shared_x_label": {"text": SHARED_TASK_XLABEL, "font_size_pt": 6,
            "anchor_mm": [float(task_centre * 25.4 / DPI),
                          float(task_label_top * 25.4 / DPI)],
            "bbox_mm": (np.asarray(task_label_bbox.extents) * 25.4 / DPI).tolist(),
            "x_tick_text_clearance_mm": float((min(box.y0 for box in task_ticks)
                                                - task_label_bbox.y1) * 25.4 / DPI)}}
    header_positions = {}
    for panel, icon_axis in icon_axes.items():
        if panel not in "acdeh":
            continue
        display = DISPLAY_PANEL_MAP[panel]
        text = next(t for t in fig.findobj(Text) if t.get_gid() == f"panel-letter-{display}")
        anchor = text.get_transform().transform(text.get_position())
        icon_top_left = np.array([icon_axis.bbox.x0, icon_axis.bbox.y1])
        offset = (icon_top_left - anchor) * 25.4 / DPI
        np.testing.assert_allclose(offset, [HEADER_ICON_OFFSET_MM, 0], atol=1e-6)
        plot_axis = diagram if panel == "d" else axes[panel]
        header_positions[display] = {"letter_anchor_mm": (anchor * 25.4 / DPI).tolist(),
            "letter_to_icon_left_mm": float(offset[0]), "letter_icon_top_difference_mm": float(offset[1]),
            "header_above_plot_mm": float((anchor[1] - plot_axis.bbox.y1) * 25.4 / DPI)}
    mark_proportions={'reference':'Fig5 g/h independent-group 4.35mm-wide boxes',
        'bar_width_mm':BAR_WIDTH_MM,
        'category_centre_gap_mm':CATEGORY_WIDTH_MM/(CATEGORY_XLIM[1]-CATEGORY_XLIM[0]),
        'category_xlim':list(axes['f'].get_xlim()),
        'original_category_xlim':list(CATEGORY_XLIM),
        'public_h_bottom_spine_outward_pt':float(spine_offset),
        'public_g_h_x_tick_baseline_difference_pt':float((matched_y-reference_y)*72./DPI),
        'public_c_f_g_display':'Conceptual diagram / GEE curve / mean-SEM curve; no quantitative box shapes replaced'}

    # Statistical plot rectangles share three row heights.
    # d is a native conceptual schematic, not a comparable statistical plot.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    def body_bounds(axis):
        labels = [*axis.texts, axis.xaxis.label, axis.yaxis.label,
                  *axis.get_xticklabels(), *axis.get_yticklabels()]
        boxes = [axis.get_window_extent(renderer)] + [t.get_window_extent(renderer)
            for t in labels if t.get_visible() and t.get_text().strip()]
        if axis.get_legend() is not None:
            boxes.append(axis.get_legend().get_window_extent(renderer))
        key = axis.get_label()
        if key in icon_axes:
            boxes.append(icon_axes[key].get_window_extent(renderer))
        return np.asarray(Bbox.union(boxes).extents) * 25.4 / DPI
    bodies = {key:body_bounds(axes[key]) for key in "abcefhi"}
    upper_bounds = [min(bodies[k][0] for k in "abc"),max(bodies[k][2] for k in "abc")]
    lower_left = min(*(bodies[k][0] for k in "efhi"),
        diagram.bbox.x0*25.4/DPI,
        *(t.get_window_extent(renderer).x0*25.4/DPI for t in diagram_texts),
        icon_axes['d'].bbox.x0*25.4/DPI)
    lower_right = max(*(bodies[k][2] for k in "efhi"),diagram.bbox.x1*25.4/DPI)
    upper_center,lower_center = np.mean(upper_bounds),np.mean([lower_left,lower_right])
    row_balance = {'upper_visible_bounds_mm':upper_bounds,
        'lower_visible_bounds_mm':[lower_left,lower_right],
        'upper_content_center_mm':float(upper_center),'lower_content_center_mm':float(lower_center),
        'center_difference_mm':float(abs(upper_center-lower_center)),
        'top_right_shift_mm':TOP_RIGHT_SHIFT_MM,
        'measurement_includes_labels_icons_and_parent_letters':True}
    # Center the complete c/d/e content, including its labels and icons, on the page.
    assert abs(row_balance['lower_content_center_mm']-WIDTH_MM/2)<.1,row_balance
    schematic_bbox = diagram.get_window_extent(renderer)
    grid_center = (axes["e"].bbox.y1 + axes["h"].bbox.y0) / 2
    schematic_center = (schematic_bbox.y0 + schematic_bbox.y1) / 2
    internal_gaps = {
        "schematic_outcomes": float(min(bodies[key][0] for key in "eh") - schematic_bbox.x1 * 25.4 / DPI),
        "upper_outcomes": float(bodies["f"][0] - bodies["e"][2]),
        "lower_outcomes": float(bodies["i"][0] - bodies["h"][2]),
        "left_outcome_rows": float(bodies["e"][1] - bodies["h"][3]),
        "right_outcome_rows": float(bodies["f"][1] - bodies["i"][3]),
    }
    public_vertical_gap = min(*(bodies[key][1] for key in "abc"),
        task_label_bbox.y0 * 25.4 / DPI) - (HEIGHT_MM - min(C_GROUP_HEADER_TOP_MM,
            ROW_TOP_MM[1]-RESULT_HEADER_ABOVE_PLOT_MM))
    compound_layout = {"schematic_grid_center_difference_mm": float((schematic_center-grid_center)*25.4/DPI),
        "internal_visible_gaps_mm": internal_gaps, "public_vertical_gap_mm": float(public_vertical_gap),
        "public_horizontal_gap_mm": float(bodies["c"][0]-max(bodies[key][2] for key in "ab")),
        "schematic_top_mm": SCHEMATIC_TOP_MM, "outcome_row_tops_mm": list(ROW_TOP_MM[1:]),
        "outcome_column_lefts_mm": [OUTCOME_LEFT_MM[key] for key in "ef"],
        "measurement": "Native plot/tick/label/legend/title/icon content bounds in mm"}
    # c is centered vertically across the two right-side outcome rows.
    assert abs(compound_layout['schematic_grid_center_difference_mm'])<1e-8
    assert min(internal_gaps.values()) >= .5, internal_gaps
    assert public_vertical_gap>=3.,compound_layout
    alignment = require_matplotlib_panel_alignment(
        fig, axes=list(axes.values()), panel_ids={ax: key for key, ax in axes.items()},
        row_groups=[["a", "b", "c"], ["e", "f"], ["h", "i"]],
        column_groups=[["e", "h"], ["f", "i"]],
        exemptions=[{"panels":["b","f","i"],"checks":["panel-label"],
                     "reason":"Second plots within public a/d/e have task/role titles, not repeated parent letters"},
                    {"panels":["a","b","c"],"checks":["horizontal-gutter"],
                     "reason":"Public a uses a5-mm inner gap with one shared left y axis; public b retains its existing separate ylabel gutter"},
                    {"panels":["c"],"checks":["panel-width"],
                     "reason":"Public b retains39mm; both task plots in public a are deliberately33mm"},
                    ],
        exclude_axes=[diagram, *icon_axes.values()], json_out=qa_dir / "alignment.json", strict=True,
        require_panel_labels=True, tolerance_pt=1.5, gutter_tolerance_pt=1.5,
    )
    text_audit = native_text_audit(fig)
    coordinates = quantitative_coordinates(axes)
    reference_coordinates = json.loads((ROOT / "source/previous_native_render.json").read_text())["scientific_coordinates"]
    unchanged_panels = list("abcehi")
    mismatched = [name for name in unchanged_panels if coordinates[name] != reference_coordinates[name]]
    expected_rate = {"points":[],"bands":[],"traces":[],"errorbars":[],"bar_heights":[.45,.225]}
    if coordinates["f"] != expected_rate:
        mismatched.append("authorized full-score proportion display")
    if mismatched:
        raise ValueError(f"Scientific coordinates changed in panels {mismatched}")
    renderer = fig.canvas.get_renderer()
    axes_geometry = {}
    for name, ax in axes.items():
        bbox = ax.get_window_extent(renderer)
        tight = ax.get_tightbbox(renderer)
        axes_geometry[name] = {
            "left_mm": bbox.x0 / DPI * 25.4, "top_mm": HEIGHT_MM - bbox.y1 / DPI * 25.4,
            "width_mm": bbox.width / DPI * 25.4, "height_mm": bbox.height / DPI * 25.4,
            "outer_content_bbox_mm": [v / DPI * 25.4 for v in tight.extents],
            "xlabel": ax.get_xlabel(), "ylabel": ax.get_ylabel(),
            "xlim": list(ax.get_xlim()), "ylim": list(ax.get_ylim()),
            "xticks": ax.get_xticks().astype(float).tolist(),
            "yticks": ax.get_yticks().astype(float).tolist(),
            "ytick_labels": [text.get_text() for text in ax.get_yticklabels()],
        }
    mark_widths = [p.get_window_extent(renderer).width / DPI * 25.4 for p in bar_patches]
    native_bar_geometry = {name: [{"bounds_data": list(p.get_bbox().bounds),
        "bounds_mm": (np.asarray(p.get_window_extent(renderer).bounds) * 25.4 / DPI).tolist()}
        for p in axes[name].patches if isinstance(p, Rectangle)] for name in "fi"}
    native_report = {
        "canvas_mm": [WIDTH_MM, HEIGHT_MM], "dpi": DPI,
        "display_panel_map": DISPLAY_PANEL_MAP, "display_panel_letters":list("abcde"),
        "public_panel_members": PUBLIC_PANEL_MEMBERS, "migrated_source_axis":"g",
        "role_titles": ROLE_TITLES,
        "y_axis_breaks":breaks,"mark_proportions":mark_proportions,
        "detached_axes_all": detached_axes_all,
        "tick_bounded_spines": tick_endpoints,
        "categorical_mark_spacing": categorical_mark_spacing,
        "detached_axes_exceptions": {"b":"Public a Task 2 shares the left y axis of Task 1; its left spine is hidden",
            "i":"Bottom spine retains an extra outward offset so categorical x tick baselines align with the neighbouring omitted-zero report plot",
            "d":"Conceptual schematic; no Cartesian corner"},
        "compound_layout": compound_layout,"row_balance":row_balance,
        "detached_axes": detached_axes, "header_positions": header_positions,
        "header_reference": {"figure": "fig5_nhb_evidence_layout", "panels": ["g", "h", "i"],
            "letter_to_icon_left_mm": HEADER_ICON_OFFSET_MM, "header_above_plot_mm": HEADER_ABOVE_PLOT_MM,
            "letter_left_of_logical_column_mm": HEADER_LETTER_LEFT_MM,
            "horizontal_margin_exception": "Retain9mm rather than13.5mm to keep first header inside existing canvas"},
        "merged_panel_a": merged_panel_a,
        "font_audit": text_audit, "analytical_axes": axes_geometry,
        "statistical_bar_widths_mm": mark_widths, "panels": data_geometry,
        "native_bar_geometry": native_bar_geometry,
        "alignment_verdict": alignment["verdict"], "input_sha256": input_manifest,
        "unchanged_panel_coordinates_match_original": unchanged_panels,
        "classification_summary_values_match_original": True,
        "panel_content_map": {"h": "previous i: noticed-element trajectory", "i": "previous h: classification accuracy"},
        "source_icons": icon_reports,
        "association_colours": {panel: axes[panel].lines[0].get_color() for panel in "abc"},
        "colour_profile": {
            "bars": [{"face_rgba": list(p.get_facecolor()), "edge_rgba": list(p.get_edgecolor()),
                      "artist_alpha": p.get_alpha()} for p in bar_patches],
            "raw_point_alpha": {panel: axes[panel].collections[1].get_alpha() for panel in "abc"},
            "ribbon_alpha": {panel: [c.get_alpha() for c in axes[panel].collections
                                      if isinstance(c, matplotlib.collections.PolyCollection)]
                             for panel in "abce"},
            "schematic_sofa_colours": sorted({to_hex(p.get_facecolor()).upper() for p in diagram.patches
                                              if p.get_gid() == "schematic-sofa"}),
            "schematic_sofa_edge_colours": sorted({to_hex(p.get_edgecolor()).upper() for p in diagram.patches
                                                   if p.get_gid() == "schematic-sofa"}),
            "condition_labels_ink": {t.get_text(): t.get_color() for t in diagram.texts
                                     if t.get_text() in ("Obs", "No-DIS")},
        },
        "authorized_geometry_changes": ["five public panels; c concept only, d/e outcome rows",
            "visible Cartesian x/y spines bounded by first/last displayed major ticks; user-selected shared score ticks20/60/100 with unchanged20..103 viewport",
            "categorical viewport translated at fixed span/width/centres for3.5mm bar-to-actual-y-spine gap",
            "concept enlarged leftward47.5x80mm with38x33.5mm cards,1.4x sofas and12.4mm central observation gap; right edge unchanged",
            "outcome rows moved up10mm; canvas height156mm",
            "concept shifted left for5.65mm outcome clearance; upper block retains10.7mm right shift",
            "one modality icon per public panel; original statistics unchanged",
            "right outcome axes shifted left4mm; categorical bars brought closer at fixed4.35mm width",
            "the complete c/d/e group shifted right7.55mm for equal outer whitespace"],
        "subplot_titles": list(ROLE_TITLES.values()), "scientific_coordinates": coordinates,
    }
    if not np.allclose(mark_widths, BAR_WIDTH_MM, atol=1e-6):
        raise ValueError("Bar widths differ from the Fig5 independent-group reference")
    fig.savefig(target, dpi=DPI, facecolor="white", transparent=False)
    if panel_previews:
        preview_dir = output_dir / "panel_previews"
        preview_dir.mkdir(exist_ok=True)
        # Isolate artists only for QA crops. The complete audited export above
        # keeps all artists; this prevents fragments of neighbouring row labels.
        for letter, members in PUBLIC_PANEL_MEMBERS.items():
            for name, ax in axes.items():
                ax.set_visible(name in members)
            shared_task_xlabel.set_visible(letter == "a")
            diagram.set_visible("d" in members)
            for name, icon_axis in icon_axes.items():
                icon_axis.set_visible(name in members)
            for text in diagram_texts:
                text.set_visible("d" in members)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            if letter == "c":
                extent = Bbox.union([diagram.get_window_extent(renderer), icon_axes["d"].get_window_extent(renderer)] +
                                    [t.get_window_extent(renderer) for t in diagram_texts] +
                                    [axes[name].get_tightbbox(renderer) for name in members if name in axes] +
                                    [icon_axes[name].get_window_extent(renderer) for name in members if name in icon_axes])
            else:
                extent = Bbox.union([axes[name].get_tightbbox(renderer) for name in members] +
                    [icon_axes[name].get_window_extent(renderer) for name in members if name in icon_axes])
                if letter == "a":
                    extent = Bbox.union([extent,shared_task_xlabel.get_window_extent(renderer)])
            region = extent.transformed(fig.dpi_scale_trans.inverted()).padded(.025)
            fig.savefig(preview_dir / f"panel_{letter}.png", dpi=DPI,
                        bbox_inches=region, pad_inches=0, facecolor="white")
        for ax in axes.values():
            ax.set_visible(True)
        shared_task_xlabel.set_visible(True)
        diagram.set_visible(True)
        for icon_axis in icon_axes.values():
            icon_axis.set_visible(True)
        for text in diagram_texts:
            text.set_visible(True)
    plt.close(fig)
    with Image.open(target) as raster:
        raster.convert("RGB").save(target, dpi=(DPI, DPI))
    native_report["output_sha256"] = digest(target)
    (qa_dir / "native_render.json").write_text(json.dumps(native_report, indent=2) + "\n")
    print(json.dumps({"output": str(target), "alignment": alignment["verdict"],
                      "native_fonts_pt": text_audit["font_sizes_pt"],
                      "text_collisions": len(text_audit["text_text_collisions"])}, indent=2))
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--panel-previews", action="store_true",
                        help="Save native-rendered final-size QA crops; use an isolated output directory")
    args = parser.parse_args()
    build(args.output_dir, args.panel_previews)
