"""Empirical attention trends with a central view and full-range inset."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, NullLocator


def _plot_mean_intervals(axis, summary, *, color, markersize, linewidth):
    """Draw empirical means and percentile intervals without a fitted curve."""
    x = summary["temperature_median"].to_numpy(dtype=float)
    mean = summary["weight_mean"].to_numpy(dtype=float)
    axis.plot(x, mean, color=color, lw=linewidth, zorder=3)
    axis.vlines(
        x,
        summary["ci_lower"],
        summary["ci_upper"],
        color=color,
        linewidth=0.9,
        zorder=4,
    )
    axis.plot(
        x, mean, "o", color=color, markersize=markersize,
        markeredgecolor="white", markeredgewidth=0.35, zorder=5,
    )


def draw_attention_temperature_bins(
    axis: plt.Axes,
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    temperature_column: str,
    color: str,
    ink: str,
) -> plt.Axes:
    """Show 0–6 on a linear axis and every trial in a log-scale inset.

    The underlying binned trial means are the same in both views. The line
    segments connect observations and must not be described as a regression.
    """
    temperature = frame[temperature_column].to_numpy(dtype=float)
    weight = frame["dim_attention_max"].to_numpy(dtype=float)
    if np.any(temperature <= 0):
        raise ValueError("the full-range log inset requires positive temperatures")
    central = temperature <= 6.0
    if not central.any():
        raise ValueError("the 0–6 panel requires observations in that interval")
    if int(summary["n_trials"].sum()) != len(frame):
        raise ValueError("attention bins must account for every plotted trial")

    axis.scatter(
        temperature[central],
        weight[central],
        s=5.0,
        color="#8C939A",
        alpha=0.14,
        edgecolors="none",
        rasterized=True,
        zorder=1,
    )
    _plot_mean_intervals(
        axis, summary.loc[summary["in_main_panel"]],
        color=color, markersize=3.6, linewidth=0.75,
    )
    axis.set(
        xlim=(0, 6), ylim=(0.235, 1.0),
        xlabel="Attention temperature", ylabel="Max dimension attention weight",
    )
    axis.set_xticks([0, 2, 4, 6])
    axis.set_yticks([0.25, 0.50, 0.75, 1.00])
    axis.set_yticklabels(["0.25", "0.50", "0.75", "1.00"])
    axis.text(
        0.035, 0.96, "Binned mean, 95% CI",
        transform=axis.transAxes, ha="left", va="top", fontsize=7,
        color=ink, fontweight="normal",
    )

    inset = axis.inset_axes([0.49, 0.53, 0.47, 0.31])
    inset.set_facecolor("white")
    inset.scatter(
        temperature, weight, s=1.4, color="#8C939A", alpha=0.10,
        edgecolors="none", rasterized=True, zorder=1,
    )
    _plot_mean_intervals(
        inset, summary, color=color, markersize=2.1, linewidth=0.65,
    )
    inset.set_xscale("log")
    lower = min(1e-4, float(temperature.min()))
    upper = max(1e4, float(temperature.max()))
    inset.set(xlim=(lower / 1.15, upper * 1.35), ylim=(0.23, 1.0))
    inset.set_title("Full range (log x)", fontsize=6.4, fontweight="normal", pad=4)
    inset.xaxis.set_major_locator(FixedLocator([1e-4, 1, 1e4]))
    inset.set_xticklabels([r"$10^{-4}$", "1", r"$10^{4}$"])
    inset.set_yticks([0.25, 1.00], ["0.25", "1.00"])
    inset.xaxis.set_minor_locator(NullLocator())
    inset.yaxis.set_minor_locator(NullLocator())
    inset.tick_params(labelsize=6, length=2.2, width=0.6, pad=2)
    for spine in ("top", "right"):
        inset.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        inset.spines[spine].set_linewidth(0.6)
    inset.grid(False)
    return inset
