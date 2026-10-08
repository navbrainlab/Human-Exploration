from __future__ import annotations

"""Build the final organized B/C1/C2 figure set.

B = occurrence above random baseline.
C1 = stay rate above shuffled baseline.
C2 = run length above shuffled baseline.

All three standalone figures and the combined three-panel figure are rendered
without panel letters. Existing source folders are read-only inputs.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from scipy.stats import t, wilcoxon

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
    }
)

import build_bc_main_figure as base


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
OUT = PACKAGE_ROOT / "figure2" / "output"
SUPPORT = PACKAGE_ROOT / "data" / "support_tables"
RNG_SEED = base.RNG_SEED + 404

B_GROUP_ORDER = [
    ("SHJ", "SHJ"),
    ("2D exploration", "沿坐标轴方向探索"),
    ("2D exploration", "Local grouping"),
    ("MASC", "MASC"),
    ("Build-an-Icon", "Build-an-Icon 1D"),
    ("Build-an-Icon", "Build-an-Icon 2D"),
    ("Build-an-Icon", "Build-an-Icon 3D"),
]

C_GROUP_ORDER = base.GROUP_ORDER

B_LABELS = [
    "SHJ",
    "Along\ncoordinate axes",
    "Local\ngrouping",
    "MASC",
    "Build\n1D",
    "Build\n2D",
    "Build\n3D",
]

C_LABELS = [
    "SHJ",
    "Along\ncoordinate axes",
    "Local\ngrouping",
    "MASC",
    "Build\n1D",
    "Build\n2D",
    "Build\n3D",
]

B_LABELS_COMBINED = [
    "SHJ",
    "Along\naxes",
    "Local\ngrouping",
    "MASC",
    "Build\n1D",
    "Build\n2D",
    "Build\n3D",
]

C_LABELS_COMBINED = [
    "SHJ",
    "Along\naxes",
    "Local\ngrouping",
    "MASC",
    "Build\n1D",
    "Build\n2D",
    "Build\n3D",
]

C_GROUP_POSITIONS = np.arange(1, len(C_GROUP_ORDER) + 1, dtype=float)
B_GROUP_POSITIONS = np.arange(1, len(B_GROUP_ORDER) + 1, dtype=float)


def summarize_rows(rows: list[dict]) -> pd.DataFrame:
    raw = pd.DataFrame(rows)
    grouped = (
        raw.groupby(["panel", "dataset", "group", "subject_key"], as_index=False)
        [["observed", "baseline"]]
        .mean()
    )
    grouped["difference"] = grouped["observed"] - grouped["baseline"]
    return grouped


def build_shj_occurrence_rows() -> list[dict]:
    """Per-trial dominant-dimension gaze share vs a matched random null.

    For each SHJ trial, the number of observed gaze events n is retained. The
    null assigns those n events independently and uniformly to the three
    dimensions, then takes the largest dimension share. Thus the baseline is
    n-specific and is generally above 1/3 for finite n; it is not a fixed 1/3
    category-frequency baseline.
    """
    b_rows, _ = base.build_shj_rows(np.random.default_rng(RNG_SEED))
    for row in b_rows:
        row["difference"] = row["observed"] - row["baseline"]
    return b_rows


def load_final_subject_rows() -> pd.DataFrame:
    """Combine corrected B, existing C1, and validated C2 subject tables."""
    occurrence = pd.read_csv(SUPPORT / "bc_occurrence_box_subject_level_v2.csv")
    occurrence_b = occurrence.loc[occurrence["panel"].eq("b")].copy()
    occurrence_b = occurrence_b.loc[~occurrence_b["dataset"].eq("SHJ")]
    # The three Build-an-Icon labels are task conditions (1/2/3 relevant
    # dimensions), but the occurrence null samples from all three candidate
    # feature dimensions.  Therefore P(K=1) = C(3,1) * .5^3 = 3/8 for all
    # three plotted conditions.
    build_mask = occurrence_b["dataset"].eq("Build-an-Icon")
    occurrence_b.loc[build_mask, "baseline"] = 3 / 8
    occurrence_b.loc[build_mask, "difference"] = (
        occurrence_b.loc[build_mask, "observed"]
        - occurrence_b.loc[build_mask, "baseline"]
    )
    occurrence_b = occurrence_b.to_dict("records")
    shj_b = build_shj_occurrence_rows()

    stay = pd.read_csv(SUPPORT / "bc_stay_box_subject_level.csv")
    stay_c1 = stay.loc[stay["panel"].eq("b")].copy()
    stay_c1["panel"] = "c1"

    run = pd.read_csv(SUPPORT / "bc_occurrence_box_subject_level_v2.csv")
    run_c2 = run.loc[run["panel"].eq("c")].copy()
    run_c2["panel"] = "c2"

    final = pd.concat(
        [
            pd.DataFrame(shj_b + occurrence_b),
            stay_c1,
            run_c2,
        ],
        ignore_index=True,
    )
    return final[["panel", "dataset", "group", "subject_key", "observed", "baseline", "difference"]]


def p_to_stars(p: float) -> str:
    if not np.isfinite(p):
        return "n.s."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def make_stats(subject_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    panel_orders = {"b": B_GROUP_ORDER, "c1": C_GROUP_ORDER, "c2": C_GROUP_ORDER}
    for panel, groups in panel_orders.items():
        for dataset, group in groups:
            values = subject_rows.loc[
                (subject_rows["panel"] == panel)
                & (subject_rows["dataset"] == dataset)
                & (subject_rows["group"] == group),
                "difference",
            ].dropna()
            n = len(values)
            mean_diff = float(values.mean()) if n else np.nan
            sem = float(values.sem()) if n > 1 else np.nan
            ci = float(t.ppf(0.975, n - 1) * sem) if n > 1 and np.isfinite(sem) else np.nan
            try:
                p_value = float(wilcoxon(values.to_numpy(), alternative="greater").pvalue) if n >= 3 and not np.allclose(values, 0) else np.nan
            except ValueError:
                p_value = np.nan
            subset = subject_rows.loc[
                (subject_rows["panel"] == panel)
                & (subject_rows["dataset"] == dataset)
                & (subject_rows["group"] == group)
            ]
            rows.append(
                {
                    "panel": panel,
                    "dataset": dataset,
                    "group": group,
                    "n_subject_units": n,
                    "mean_observed": float(subset["observed"].mean()) if n else np.nan,
                    "mean_baseline": float(subset["baseline"].mean()) if n else np.nan,
                    "mean_difference": mean_diff,
                    "ci95_low": mean_diff - ci if np.isfinite(ci) else np.nan,
                    "ci95_high": mean_diff + ci if np.isfinite(ci) else np.nan,
                    "wilcoxon_p_greater": p_value,
                    "stars": p_to_stars(p_value),
                }
            )
    return pd.DataFrame(rows)


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.6)
    ax.spines["bottom"].set_linewidth(1.6)
    ax.grid(False)
    ax.tick_params(axis="both", labelsize=12, width=1.3, length=5)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))


def values_for(subject_rows, panel, dataset, group):
    return subject_rows.loc[
        (subject_rows["panel"] == panel)
        & (subject_rows["dataset"] == dataset)
        & (subject_rows["group"] == group),
        "difference",
    ].dropna()


def add_dataset_legend(ax_or_fig, *, combined: bool = False):
    handles = [
        Patch(facecolor=color, edgecolor="none", label=dataset)
        for dataset, color in base.DATASET_COLORS.items()
    ]
    if combined:
        ax_or_fig.legend(handles=handles, frameon=False, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.03), fontsize=11)
    else:
        # Standalone panels use a figure-level legend so the title remains
        # separated from the legend after tight_layout/bbox_inches="tight".
        fig = ax_or_fig.get_figure()
        fig.legend(handles=handles, frameon=False, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.995), fontsize=11)


def add_violin(ax, values_by_group, positions, group_order, rng):
    parts = ax.violinplot(
        [values.to_numpy(dtype=float) for values in values_by_group],
        positions=positions,
        widths=0.24,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    for body, (dataset, _) in zip(parts["bodies"], group_order):
        color = base.DATASET_COLORS[dataset]
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.34)
        body.set_linewidth(0.8)
    if "cmedians" in parts:
        parts["cmedians"].set_color("#222222")
        parts["cmedians"].set_linewidth(1.2)

    for position, values, (dataset, _) in zip(positions, values_by_group, group_order):
        if len(values) == 0:
            continue
        color = base.DATASET_COLORS[dataset]
        q1, median, q3 = np.percentile(values.to_numpy(dtype=float), [25, 50, 75])
        ax.vlines(position, q1, q3, color="#333333", linewidth=3.0, zorder=4)
        ax.scatter([position], [median], s=16, color="#222222", zorder=5)
        jitter = rng.uniform(-0.055, 0.055, size=len(values))
        ax.scatter(np.full(len(values), position) + jitter, values, s=13, color=color, alpha=0.32, linewidths=0, zorder=3)


def add_stars(ax, position, values, stats_table, panel, group):
    if len(values) == 0:
        return
    stat = stats_table.loc[(stats_table["panel"] == panel) & (stats_table["group"] == group)]
    if stat.empty:
        return
    y = float(values.max()) + max(0.03, 0.08 * max(np.ptp(values.to_numpy()), 0.05))
    ax.text(position, y, str(stat["stars"].iloc[0]), ha="center", va="bottom", fontsize=11, color="#222222", fontweight="bold")


def configure_axis(ax, positions, xlabels, ylabel, xlabel="Exploration condition"):
    ax.axhline(0, color="#666666", linewidth=1.5, linestyle=(0, (2, 2)), zorder=1)
    ax.set_xticks(positions, xlabels)
    left = float(np.min(positions))
    right = float(np.max(positions))
    ax.set_xlim(left - 0.55, right + 0.55)
    ax.spines["bottom"].set_bounds(left, right)
    ax.set_ylabel(ylabel, fontsize=14, fontweight="bold", labelpad=9)
    ax.set_xlabel(xlabel, fontsize=13, fontweight="bold", labelpad=9)
    ax.tick_params(axis="x", labelsize=12)
    style_axis(ax)


def plot_b(subject_rows, stats_table, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 7.0), dpi=320)
    positions = B_GROUP_POSITIONS
    rng = np.random.default_rng(RNG_SEED + 1)
    for position, (dataset, group) in zip(positions, B_GROUP_ORDER):
        values = values_for(subject_rows, "b", dataset, group)
        if len(values) == 0:
            continue
        color = base.DATASET_COLORS[dataset]
        ax.scatter(np.full(len(values), position) + rng.uniform(-0.10, 0.10, len(values)), values, s=17, color=color, alpha=0.30, linewidths=0, zorder=2)
        mean = float(values.mean())
        sem = float(values.sem()) if len(values) > 1 else np.nan
        ci = float(t.ppf(0.975, len(values) - 1) * sem) if len(values) > 1 else 0.0
        ax.errorbar(position, mean, yerr=ci, fmt="o", color=color, markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.8, markersize=6.5, capsize=2.5, elinewidth=1.3, zorder=4)
        add_stars(ax, position, values, stats_table, "b", group)
    configure_axis(ax, positions, B_LABELS, "Dominant-dimension occurrence difference\n(observed − random baseline)")
    ax.set_title("Dominant-dimension occurrence\nrelative to a random baseline", fontsize=16, pad=12, fontweight="bold")
    add_dataset_legend(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.76))
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_violin_panel(subject_rows, stats_table, panel, path: Path, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 7.0), dpi=320)
    positions = C_GROUP_POSITIONS
    values = [values_for(subject_rows, panel, dataset, group) for dataset, group in C_GROUP_ORDER]
    add_violin(ax, values, positions, C_GROUP_ORDER, np.random.default_rng(RNG_SEED + (10 if panel == "c1" else 20)))
    for position, vals, (_, group) in zip(positions, values, C_GROUP_ORDER):
        add_stars(ax, position, vals, stats_table, panel, group)
    configure_axis(ax, positions, C_LABELS, ylabel)
    ax.set_title(title, fontsize=16, pad=12, fontweight="bold")
    add_dataset_legend(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.76))
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_combined(subject_rows, stats_table, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 7.4), dpi=320)

    # Occurrence panel.
    ax = axes[0]
    positions = B_GROUP_POSITIONS
    rng = np.random.default_rng(RNG_SEED + 31)
    for position, (dataset, group) in zip(positions, B_GROUP_ORDER):
        values = values_for(subject_rows, "b", dataset, group)
        if len(values) == 0:
            continue
        color = base.DATASET_COLORS[dataset]
        ax.scatter(np.full(len(values), position) + rng.uniform(-0.09, 0.09, len(values)), values, s=12, color=color, alpha=0.28, linewidths=0, zorder=2)
        mean = float(values.mean())
        sem = float(values.sem()) if len(values) > 1 else np.nan
        ci = float(t.ppf(0.975, len(values) - 1) * sem) if len(values) > 1 else 0.0
        ax.errorbar(position, mean, yerr=ci, fmt="o", color=color, markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.7, markersize=5.5, capsize=2, elinewidth=1.1, zorder=4)
        add_stars(ax, position, values, stats_table, "b", group)
    configure_axis(
        ax,
        positions,
        B_LABELS_COMBINED,
        "Dominant-dimension occurrence difference\n(observed − random baseline)",
        xlabel="Exploration condition",
    )
    ax.set_title("Dominant-dimension occurrence\nrelative to a random baseline", fontsize=14, pad=12, fontweight="bold")

    # Persistence panels.
    for ax, panel, title, ylabel, seed in [
        (axes[1], "c1", "Stay-rate persistence\nrelative to a shuffled baseline", "Stay-rate difference\n(observed − shuffled baseline)", 41),
        (axes[2], "c2", "Run-length persistence\nrelative to a shuffled baseline", "Run-length difference\n(observed − shuffled baseline)", 51),
    ]:
        vals_by_group = [values_for(subject_rows, panel, dataset, group) for dataset, group in C_GROUP_ORDER]
        add_violin(ax, vals_by_group, C_GROUP_POSITIONS, C_GROUP_ORDER, np.random.default_rng(RNG_SEED + seed))
        for pos, vals, (_, group) in zip(C_GROUP_POSITIONS, vals_by_group, C_GROUP_ORDER):
            add_stars(ax, pos, vals, stats_table, panel, group)
        configure_axis(ax, C_GROUP_POSITIONS, C_LABELS_COMBINED, ylabel, xlabel="Exploration condition")
        ax.set_title(title, fontsize=14, pad=12, fontweight="bold")

    add_dataset_legend(fig, combined=True)
    fig.subplots_adjust(left=0.045, right=0.995, bottom=0.20, top=0.80, wspace=0.32)
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def write_readme() -> None:
    text = """# Final organized figures

Only the four final PNGs are retained in this folder:

- `figure_b_occurrence.png`: occurrence above random baseline.
- `figure_c1_stay_rate.png`: stay rate above shuffled baseline.
- `figure_c2_run_length.png`: run length above shuffled baseline.
- `figure_b_c1_c2_combined.png`: the three panels together.

## B occurrence definition

- SHJ is the within-trial share of gaze events belonging to the most-gazed
  dimension (`max eye_pri_norm`). For each trial, the null keeps its observed
  gaze count n, assigns every gaze independently and uniformly to the three
  dimensions, and takes the maximum dimension share. The resulting baseline is
  n-specific and above 1/3 for finite n; for small n it can be substantially
  higher (for example, n=1 gives 1 and n=2 gives 2/3). Type 1/2/4/6 are averaged
  within subject, before trials only.
- 2D is the occurrence rate of along-coordinate-axis and local-grouping moves
  above random generated paths.
- MASC is the within-trial most-fixated-dimension share above its random AOI
  baseline.
- Build-an-Icon is the rate of raw one-dimensional selection trials above a
  random subset baseline. All three candidate feature dimensions are in the
  null, so independent p=0.5 selection gives P(K=1)=C(3,1)/2^3=3/8 for
  each of the 1D/2D/3D relevant-dimension conditions. These are separate
  condition groups and are not summed.

## C persistence definitions

- C1 is the stay-rate difference from the shuffled sequence baseline.
- C2 is the mean run-length difference from the shuffled sequence baseline.
- C1 and C2 are violin plots with raw subject/worker-game points, median
  markers, and IQR bars. The C2 y-axis uses sequence units because the
  underlying unit differs across tasks.

Exact statistics are in `final_b_c1_c2_stats.csv`; subject-level values are in
`final_b_c1_c2_subject_level.csv`. Supporting scripts and intermediate CSVs
are retained for provenance, but obsolete PNGs were removed.
"""
    (OUT / "README_final_figures.md").write_text(text, encoding="utf-8")


def main() -> None:
    subject_rows = load_final_subject_rows()
    stats_table = make_stats(subject_rows)
    subject_rows.to_csv(OUT / "final_b_c1_c2_subject_level.csv", index=False)
    stats_table.to_csv(OUT / "final_b_c1_c2_stats.csv", index=False)
    plot_b(subject_rows, stats_table, OUT / "figure_b_occurrence.png")
    plot_violin_panel(subject_rows, stats_table, "c1", OUT / "figure_c1_stay_rate.png", "Stay rate above shuffled baseline", "Observed − shuffled stay rate")
    plot_violin_panel(subject_rows, stats_table, "c2", OUT / "figure_c2_run_length.png", "Run length above shuffled baseline", "Run-length difference (sequence units)")
    plot_combined(subject_rows, stats_table, OUT / "figure_b_c1_c2_combined.png")
    write_readme()
    print(stats_table.to_string(index=False))


if __name__ == "__main__":
    main()
