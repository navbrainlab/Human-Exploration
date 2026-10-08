"""Shared publication styling for the six figure-drawing notebooks.

The module changes presentation only. It does not transform data or statistics.
"""
from __future__ import annotations

import colorsys
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PathCollection, PolyCollection
from matplotlib.ticker import MaxNLocator


TASK1_LABELS = {"P1": "3D-NE", "P2": "4D-E", "P2-only": "4D-NE"}
# TASK1_COLORS = {"P1": "#EDF6B6", "P2": "#93CDBD", "P2-only": "#4781B6"}
TASK1_COLORS = {"P1": "#079E6C", "P2": "#0965C0", "P2-only": "#14BAEC"}

TASK2_LABELS = {
    "FDS-Obs": "Obs",
    "FDS-NoObs": "NObs",
    "NoFDS-Obs": "DIS-D",
    "DisFDS": "DIS-D",
    "DIS-D": "DIS-D",
    "Dis-FDS": "DIS-D",
    "Dis-DIS": "DIS-D",
    "Obs": "Obs",
    "NObs": "NObs",
    "DIS": "DIS",
    "NDIS": "NDIS",
}
TASK2_COLORS = {
    # "FDS-Obs": "#E67E28",
    # "FDS-NoObs": "#A10A05",
    # "NoFDS-Obs": "#FCC30B",
    "FDS-Obs": "#E67E28",
    # "FDS-Obs": "#EB6D06",
    "FDS-NoObs": "#F7BA00",
    # "NoFDS-Obs": "#7048A0",
    "DisFDS": "#8657BE",
    "NoFDS-Obs": "#8657BE",
    "Obs": "#E67E28",
    "NObs": "#F7BA00",
}

# Additional palettes used by several panels.  Keeping them here prevents the
# same semantic group from silently changing colour between notebooks.


RT_COLORS = {
    "selection_time": "#4C72B0",
    "thinking_time": "#DD8452",
    "total_time": "#55A868",
}

FDS_OBSERVATION_COLORS = {
    "Obs-Sys": "#C44E52",
    "Obs-NonSys": "#E39A9D",
    "NObs-Sys": "#4C72B0",
    "NObs-NonSys": "#9CBAD3",
}

# Survey categories are ordered, rather than experimental groups.  They use a
# separate sequential palette so that Task 1/Task 2 colours retain one meaning.
SURVEY_COLORS_2 = ["#56CFE1", "#1D8FE1"]
SURVEY_COLORS_3 = ["#A6EACB", "#56CFE1", "#1D8FE1"]
SURVEY_COLORS = ["#A6EACB", "#56CFE1", "#1D8FE1", "#0E5CAD", "#083D77"]

NEUTRAL_COLORS = {
    "random": "#9E9E9E",
    "edge": "#333333",
    "light": "#D9D9D9",
}

# Shared model and strategy palettes. Notebooks should copy these mappings rather
# than redefining hexadecimal colors in individual cells.
# HUMAN_COLOR = "#2F3CAF"
# MODEL_COLORS = {
#     "Human": HUMAN_COLOR,
#     "human": HUMAN_COLOR,
#     "naiveRL": "#49C3D3",
#     "fRL": "#6BB2F5",
#     "DGE": "#6050A7",
#     "Bayesian": "#3B71E6",
# }

# 红橙+蓝紫配色
# HUMAN_COLOR = "#CA1919EB"
# MODEL_COLORS = {
#     "Human": HUMAN_COLOR,
#     "human": HUMAN_COLOR,
#     "naiveRL": "#6050A7",
#     "fRL": "#6BB2F5",
#     "DGE": "#FA813B",
#     "Bayesian": "#3B71E6",
# }

# 蓝紫+橙黄配色
HUMAN_COLOR = "#1951CAEB"
MODEL_COLORS = {
    "Human": HUMAN_COLOR,
    "human": HUMAN_COLOR,
    "naiveRL": "#9E6E15",
    "fRL": "#FDD212",
    "DGE": "#6050A7",
    "Bayesian": "#F78113",
}

FDS_COLORS = {
    "Sys": "#E47D1D",
    "Non-sys": "#F1C40F",
}

PHASE_LABELS = TASK1_LABELS

LABEL_ALIASES = {
    "Phase": "Task condition",
    "Task phase": "Task condition",
    "Group": "Participant group",
    "Trial": "Trial number",
    "Round": "Trial number",
    "Rounds": "Number of trials",
    "Total rounds": "Number of trials completed",
    "Mean Score": "Mean score",
    "best_score": "Best score",
    "Best Score": "Best score",
    "Full Score Rate (%)": "Participants reaching maximum score (%)",
    "Full score rate": "Participants reaching maximum score (%)",
    "FDS ratio": "Proportion of trials using DIS",
    "FDS_ratio": "Proportion of trials using DIS",
    "FDS frequency": "Proportion of trials using DIS",
    "FDS round ratio": "Proportion of trials using DIS",
    "FDS_round_ratio": "Proportion of trials using DIS",
    "Proportion of rounds with FDS": "Proportion of trials using DIS",
    "Proportion of FDS Trials": "Proportion of trials using DIS",
    "FDS rounds": "Number of trials using DIS",
    "FDS_rounds": "Number of trials using DIS",
    "Number of rounds adopting FDS": "Number of trials using DIS",
    "# rounds adopting FDS": "Number of trials using DIS",
    "First FDS round index": "Trial of first DIS use",
    "Index of first FDS round": "Trial of first DIS use",
    "first FDS round index": "Trial of first DIS use",
    "FDS dimensions": "Number of dimensions explored using DIS",
    "FDS_dimensions": "Number of dimensions explored using DIS",
    "FDS dimension count": "Number of dimensions explored using DIS",
    "FDS_dimension_count": "Number of dimensions explored using DIS",
    "FDS counts per dms": "Mean number of DIS trials per explored dimension",
    "FDS_counts_per_dms": "Mean number of DIS trials per explored dimension",
    "FDS patterns per dms": "Mean unique DIS patterns per explored dimension",
    "FDS_patterns_per_dms": "Mean unique DIS patterns per explored dimension",
    "FDS_counts_within_dms": "Number of DIS trials within a dimension",
    "Max round reached": "Maximum trial reached",
    "Maximum Round": "Maximum trial reached",
    "Maximum Trial": "Maximum trial reached",
    "Best Score in Final Rounds": "Best score in final trials",
    "Best score at the final rounds": "Best score in final trials",
    "Last-Round Score": "Final-trial score",
    "Round index": "Trial number",
    "Relative trial from FDS onset": "Trial relative to DIS onset",
    "Subject mean time": "Mean response time (s)",
    "Subject-phase mean time": "Mean response time (s)",
    "Dimension": "Number of dimensions",
    "Count": "Number of participants",
    "T anxiety": "Trait anxiety score (STAI-T)",
    "Models and Human": "Models and human participants",
    "Number of FDS trials per subject": "Number of DIS trials per participant",
    "selection_time": "Selection time",
    "thinking_time": "Thinking time",
    "total_time": "Total response time",
}


def apply_publication_defaults() -> None:
    """Apply consistent defaults suitable for small multi-panel figures."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 16,
        "axes.labelsize": 19,
        "axes.labelweight": "normal",
        "axes.titlesize": 18,
        "axes.titleweight": "normal",
        "axes.linewidth": 2.8,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.major.width": 2.4,
        "ytick.major.width": 2.4,
        "legend.fontsize": 14,
        "legend.title_fontsize": 14,
        "legend.frameon": False,
        "lines.linewidth": 2.5,
        "patch.linewidth": 1.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.dpi": 120,
    })


def _clean_label(label: str) -> str:
    label = (label or "").strip()
    if label in TASK1_LABELS:
        return TASK1_LABELS[label]
    if label in TASK2_LABELS:
        return TASK2_LABELS[label]
    if label in LABEL_ALIASES:
        return LABEL_ALIASES[label]
    # Convert raw snake_case variable names when they leak into a figure.
    if "_" in label and re.fullmatch(r"[A-Za-z0-9_#%() /.-]+", label):
        label = label.replace("_", " ").strip()
        label = re.sub(r"\s+", " ", label)
        return label[:1].upper() + label[1:]
    return label


def _numeric_ticklabels(labels) -> bool:
    values = [t.get_text().strip().replace("−", "-") for t in labels]
    values = [v for v in values if v]
    if not values:
        return False


def _saturated_rgba(colors, minimum_saturation: float = 0.58):
    """Return chromatic artist colours with a publication-safe saturation floor."""
    rgba = np.asarray(colors, dtype=float).copy()
    if rgba.size == 0:
        return rgba
    one_color = rgba.ndim == 1
    rgba = np.atleast_2d(rgba)
    for row in rgba:
        red, green, blue = row[:3]
        hue, saturation, value = colorsys.rgb_to_hsv(red, green, blue)
        # Leave structural neutral colours and near-white fills unchanged.
        if saturation < 0.08 or value > 0.97:
            continue
        saturation = max(saturation, minimum_saturation)
        value = min(value, 0.90)
        row[:3] = colorsys.hsv_to_rgb(hue, saturation, value)
    return rgba[0] if one_color else rgba
    try:
        for value in values:
            float(value.replace("%", ""))
        return True
    except ValueError:
        return False


def _preserve_artist_style(artist) -> bool:
    gid_getter = getattr(artist, "get_gid", None)
    gid = gid_getter() if callable(gid_getter) else None
    if gid in {"preserve-style", "preserve-alpha"}:
        return True
    return bool(getattr(artist, "_preserve_style", False))


def _preferred_fontweight(text_artist, default: str = "normal") -> str:
    """Keep an explicitly heavier weight instead of flattening it to normal."""
    current_weight = getattr(text_artist, "get_fontweight", lambda: default)()
    if current_weight is None:
        return default
    weight_text = str(current_weight).strip().lower()
    if weight_text in {"bold", "semibold", "demibold", "heavy", "extra bold", "ultrabold", "black"}:
        return current_weight
    return default


def style_axis(ax, *, trim_spines: bool = True, tidy_ticks: bool = True) -> None:
    """Normalize one axis without altering plotted data."""
    ax.set_xlabel(
        _clean_label(ax.get_xlabel()),
        fontsize=19,
        fontweight=_preferred_fontweight(ax.xaxis.label),
        labelpad=7,
    )
    ax.set_ylabel(
        _clean_label(ax.get_ylabel()),
        fontsize=19,
        fontweight=_preferred_fontweight(ax.yaxis.label),
        labelpad=7,
    )
    if ax.get_title():
        ax.set_title(
            _clean_label(ax.get_title()),
            fontsize=18,
            fontweight=_preferred_fontweight(ax.title),
            pad=8,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(2.8)

    ax.tick_params(axis="both", which="major", direction="out", length=6,
                   width=2.4, labelsize=16, pad=5, top=False, right=False)
    ax.tick_params(axis="both", which="minor", direction="out", length=3,
                   width=1.2, top=False, right=False)

    # Replace internal group codes wherever they appear as categorical ticks.
    raw_ticklabels = [tick.get_text() for tick in ax.get_xticklabels()]
    canonical_ticklabels = [
        TASK1_LABELS.get(label, TASK2_LABELS.get(label, label))
        for label in raw_ticklabels
    ]
    if canonical_ticklabels != raw_ticklabels:
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(canonical_ticklabels, fontsize=16)

    # Binary recognition variables must never appear as unexplained 0/1 labels.
    context = " ".join([ax.get_xlabel(), ax.get_ylabel(), ax.get_title()]).lower()
    tick_set = {label.strip() for label in canonical_ticklabels}
    if tick_set == {"0", "1"} and "recogn" in context:
        recognized = ["Not recognized" if x.strip() == "0" else "Recognized"
                      for x in canonical_ticklabels]
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(recognized, fontsize=14)

    # Long factorial group labels are much easier to read on two horizontal
    # lines than as six steeply rotated labels in a small panel.
    current_xticklabels = [tick_label.get_text() for tick_label in ax.get_xticklabels()]
    if any(" + " in label for label in current_xticklabels):
        wrapped_labels = [label.replace(" + ", "\n") for label in current_xticklabels]
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(wrapped_labels, rotation=0, ha="center", fontsize=11)

    # Keep categorical comparisons compact, especially two-group panels.
    visible_labels = [label for label in ax.get_xticklabels() if label.get_text().strip()]
    if 1 < len(visible_labels) <= 4 and not _numeric_ticklabels(visible_labels):
        ax.margins(x=0.025)

    # Keep categorical marks saturated and legible. Error ribbons created by
    # fill_between remain translucent because their PolyCollection usually has
    # a single long path; scatter points, bars and boxes are made opaque enough
    # for consistent multi-panel reproduction.
    for patch in ax.patches:
        if _preserve_artist_style(patch):
            continue
        patch.set_facecolor(_saturated_rgba(patch.get_facecolor()))
        patch.set_edgecolor(_saturated_rgba(patch.get_edgecolor()))
        alpha = patch.get_alpha()
        if alpha is not None and alpha < 0.82:
            patch.set_alpha(0.88)
    for collection in ax.collections:
        if _preserve_artist_style(collection):
            continue
        if len(collection.get_facecolors()):
            collection.set_facecolors(_saturated_rgba(collection.get_facecolors()))
        if len(collection.get_edgecolors()):
            collection.set_edgecolors(_saturated_rgba(collection.get_edgecolors()))
        if isinstance(collection, PathCollection):
            alpha = collection.get_alpha()
            if alpha is not None and alpha < 0.82:
                collection.set_alpha(0.88)
        elif isinstance(collection, PolyCollection):
            paths = collection.get_paths()
            # Violin bodies are compact closed paths; retain transparency for
            # broad confidence/error ribbons spanning many x positions.
            if paths and len(paths[0].vertices) < 250:
                alpha = collection.get_alpha()
                if alpha is not None and alpha < 0.72:
                    collection.set_alpha(0.82)
    for line in ax.lines:
        try:
            line.set_color(_saturated_rgba(mpl.colors.to_rgba(line.get_color())))
        except (TypeError, ValueError):
            pass

    # Reduce only genuinely dense numeric axes; categorical and heat-map labels stay intact.
    if tidy_ticks and not ax.images:
        if len(ax.get_xticks()) > 7 and _numeric_ticklabels(ax.get_xticklabels()):
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
        if len(ax.get_yticks()) > 7 and _numeric_ticklabels(ax.get_yticklabels()):
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))

    legend = ax.get_legend()
    if legend is not None:
        legend.set_frame_on(False)
        title = legend.get_title().get_text()
        if title in {"phase", "Phase"}:
            legend.set_title("Task condition")
        for text in legend.get_texts():
            text.set_fontsize(14)
            raw = text.get_text()
            if raw in TASK1_LABELS:
                text.set_text(TASK1_LABELS[raw])
            elif raw in TASK2_LABELS:
                text.set_text(TASK2_LABELS[raw])
        legend.get_title().set_fontsize(14)

    if trim_spines and not ax.images:
        # Matplotlib may return a Python list for categorical axes and an array
        # for numeric axes. Normalize both before boolean indexing.
        yticks = np.asarray(ax.get_yticks(), dtype=float)
        xticks = np.asarray(ax.get_xticks(), dtype=float)
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()
        valid_y = yticks[(yticks >= min(ylim)) & (yticks <= max(ylim))]
        valid_x = xticks[(xticks >= min(xlim)) & (xticks <= max(xlim))]
        if len(valid_y) >= 2:
            ax.spines["left"].set_bounds(valid_y[0], valid_y[-1])
        if len(valid_x) >= 2:
            ax.spines["bottom"].set_bounds(valid_x[0], valid_x[-1])


def style_figure(fig=None) -> None:
    """Apply the shared final-pass style to every axis in a figure."""
    apply_publication_defaults()
    fig = fig or plt.gcf()
    for ax in fig.axes:
        # Colorbar axes are deliberately left structurally intact.
        is_colorbar = getattr(ax, "_colorbar", None) is not None
        style_axis(ax, trim_spines=not is_colorbar, tidy_ticks=not is_colorbar)
    for text_artist in fig.texts:
        current = text_artist.get_text()
        cleaned = _clean_label(current)
        if cleaned != current:
            text_artist.set_text(cleaned)
            text_artist.set_fontsize(17)
            text_artist.set_fontweight(_preferred_fontweight(text_artist))
    # Do not call fig.align_labels() here: composite figures often use a
    # different semantic y-label for each panel, and global alignment can stack
    # those labels on top of one another at the far-left margin.


def save_figure(fig, path, **kwargs) -> Path:
    """Style and save a figure with consistent vector-friendly defaults."""
    style_figure(fig)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs.setdefault("dpi", 300)
    kwargs.setdefault("bbox_inches", "tight")
    fig.savefig(path, **kwargs)
    return path


apply_publication_defaults()
