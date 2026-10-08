from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle

from publication_plot_style import apply_publication_defaults, save_figure, style_figure


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PACKAGE_ROOT / "figure2" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DENSE_HEATMAP_OUTPUT_PATH = OUTPUT_DIR / "task1_task2_feature_enrichment_vs_trial_progress_mode_heatmap.png"
LONG_CURVE_OUTPUT_PATH = OUTPUT_DIR / "task1_task2_feature_enrichment_vs_trial_progress_mode_curve_long.png"
LONG_GAPPED_HEATMAP_OUTPUT_PATH = (
    OUTPUT_DIR / "task1_task2_feature_enrichment_vs_trial_progress_mode_gapped_heatmap_long.png"
)

BASE_N_BINS = 10
LONG_N_BINS = 24
MIN_PROP_TO_SHOW = 0.03
MIN_OVERALL_PROP_TO_KEEP = 0.03
TASK1_COLOR = "#005BBB"
TASK2_COLOR = "#F28E2B"


def _sum_category_digits_task1(value):
    if pd.isna(value):
        return np.nan
    digits = [int(ch) for ch in str(value).strip() if ch.isdigit()]
    return int(sum(digits)) if digits else np.nan


def compute_human_trial_enrichment_task1(row: pd.Series):
    dim_cols = ["dim1_category", "dim2_category", "dim3_category"]
    if pd.notna(row.get("dim4_category", np.nan)):
        dim_cols.append("dim4_category")
    dim_scores = [_sum_category_digits_task1(row[col]) for col in dim_cols if col in row.index]
    dim_scores = [score for score in dim_scores if pd.notna(score)]
    return int(max(dim_scores)) if dim_scores else np.nan


def parse_selected_objects_task2(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    return ast.literal_eval(value)


def compute_feature_enrichment_task2(selected_objects):
    try:
        objects = parse_selected_objects_task2(selected_objects)
    except (ValueError, SyntaxError, TypeError):
        return np.nan
    if not isinstance(objects, list) or len(objects) != 4:
        return np.nan
    if any(not isinstance(obj, str) or not obj.strip() for obj in objects):
        return np.nan
    features = [feature.strip() for obj in objects for feature in obj.split("_") if feature.strip()]
    return max(Counter(features).values()) if features else np.nan


def build_task1_trial_df():
    trial_df = pd.read_csv(PACKAGE_ROOT / "data" / "choice_category_uniform_105.csv", index_col=0)
    trial_df["human_enrichment"] = trial_df.apply(compute_human_trial_enrichment_task1, axis=1)
    trial_df = trial_df.dropna(subset=["subject", "phase", "human_enrichment", "round"]).copy()
    trial_df["human_enrichment"] = trial_df["human_enrichment"].astype(int)
    trial_df["round"] = pd.to_numeric(trial_df["round"], errors="coerce")
    return trial_df.dropna(subset=["round"]).copy()


def build_task2_trial_df():
    raw_df = pd.read_csv(PACKAGE_ROOT / "data" / "summary_data_0723_task2.csv")
    trial_df = raw_df.loc[
        raw_df["group"].isin(["observation", "non_observation"]) & raw_df["score_noisy"].notna(),
        ["subject_id", "trial_id", "selected_objects"],
    ].copy()
    trial_df["human_enrichment"] = trial_df["selected_objects"].apply(compute_feature_enrichment_task2)
    trial_df = trial_df.dropna(subset=["subject_id", "human_enrichment", "trial_id"]).copy()
    trial_df["human_enrichment"] = trial_df["human_enrichment"].astype(int)
    trial_df["trial_id"] = pd.to_numeric(trial_df["trial_id"], errors="coerce")
    return trial_df.dropna(subset=["trial_id"]).copy()


def add_progress_bins(df: pd.DataFrame, unit_cols: list[str], order_col: str, n_bins: int = 10):
    use_df = df[list(unit_cols) + [order_col, "human_enrichment"]].copy()
    use_df = use_df.sort_values(list(unit_cols) + [order_col]).copy()
    use_df["trial_rank"] = use_df.groupby(unit_cols, observed=False).cumcount() + 1
    n_trials = use_df.groupby(unit_cols, observed=False)["human_enrichment"].transform("size")
    use_df["trial_progress"] = (use_df["trial_rank"] - 1) / (n_trials - 1).where(n_trials > 1, 1)
    use_df["trial_progress"] = use_df["trial_progress"].fillna(0.0).clip(0.0, 1.0)
    use_df["progress_bin"] = np.minimum((use_df["trial_progress"] * n_bins).astype(int), n_bins - 1)
    use_df["progress_center"] = (use_df["progress_bin"] + 0.5) / n_bins
    return use_df


def build_heatmap_and_mode(progress_df: pd.DataFrame, n_bins: int):
    all_enrich_values = sorted(progress_df["human_enrichment"].astype(int).unique().tolist())
    counts = (
        progress_df.groupby(["progress_bin", "human_enrichment"], observed=False)
        .size()
        .rename("count")
        .reset_index()
    )
    totals = counts.groupby("progress_bin", observed=False)["count"].transform("sum")
    counts["proportion"] = counts["count"] / totals

    all_bins = pd.Index(range(n_bins), name="progress_bin")
    all_enrichment = pd.Index(all_enrich_values, name="human_enrichment")
    heatmap = (
        counts.set_index(["progress_bin", "human_enrichment"])["proportion"]
        .reindex(pd.MultiIndex.from_product([all_bins, all_enrichment]))
        .unstack("progress_bin")
        .fillna(0.0)
    )
    heatmap = heatmap.loc[all_enrich_values, list(range(n_bins))]

    mode_df = (
        counts.sort_values(["progress_bin", "proportion", "human_enrichment"], ascending=[True, False, False])
        .groupby("progress_bin", as_index=False, observed=False)
        .first()
        .sort_values("progress_bin")
        .reset_index(drop=True)
    )
    mode_df["progress_center"] = (mode_df["progress_bin"] + 0.5) / n_bins

    overall_props = (
        counts.groupby("human_enrichment", observed=False)["count"].sum() / counts["count"].sum()
    )
    keep_values = sorted(
        {
            int(value)
            for value, prop in overall_props.items()
            if prop >= MIN_OVERALL_PROP_TO_KEEP
        }
        | set(mode_df["human_enrichment"].astype(int).tolist())
    )

    if keep_values:
        heatmap = heatmap.loc[keep_values]
    return heatmap, mode_df


def make_colormap(base_color: str):
    base = np.array(mcolors.to_rgb(base_color))
    white = np.array([1.0, 1.0, 1.0])
    colors = [
        tuple(white),
        tuple(white * 0.98 + base * 0.02),
        tuple(white * 0.85 + base * 0.15),
        tuple(white * 0.65 + base * 0.35),
        tuple(white * 0.35 + base * 0.65),
        tuple(base),
    ]
    return mcolors.LinearSegmentedColormap.from_list(f"enrichment_{base_color[1:]}", colors)


def map_progress_to_axis(progress_values, x_start=0.0, x_end=1.0):
    progress_values = np.asarray(progress_values, dtype=float)
    return x_start + progress_values * (x_end - x_start)


def set_clean_spines(ax, x_tick_positions, y_tick_positions):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_bounds(float(np.min(y_tick_positions)), float(np.max(y_tick_positions)))
    ax.spines["bottom"].set_bounds(float(np.min(x_tick_positions)), float(np.max(x_tick_positions)))


def get_display_y_ticks(enrich_values):
    enrich_values = np.asarray(sorted(np.unique(enrich_values)), dtype=float)
    return np.arange(int(enrich_values.min()), int(enrich_values.max()) + 1, dtype=float)


def get_y_edges(enrich_values):
    enrich_values = np.asarray(sorted(np.unique(enrich_values)), dtype=float)
    if enrich_values.size == 1:
        return np.array([enrich_values[0] - 0.5, enrich_values[0] + 0.5], dtype=float)
    midpoints = 0.5 * (enrich_values[:-1] + enrich_values[1:])
    return np.concatenate(
        [
            [enrich_values[0] - 0.5],
            midpoints,
            [enrich_values[-1] + 0.5],
        ]
    )


def draw_dense_heatmap_panel(ax, heatmap: pd.DataFrame, mode_df: pd.DataFrame, title: str, base_color: str):
    matrix = heatmap.to_numpy(dtype=float)
    matrix = np.where(matrix >= MIN_PROP_TO_SHOW, matrix, 0.0)
    enrich_values = heatmap.index.to_numpy(dtype=float)

    x_edges = np.linspace(0.0, 1.0, heatmap.shape[1] + 1)
    y_edges = get_y_edges(enrich_values)
    y_ticks = get_display_y_ticks(enrich_values)
    norm = mcolors.Normalize(vmin=0.0, vmax=max(0.45, float(matrix.max())))

    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        matrix,
        cmap=make_colormap(base_color),
        norm=norm,
        shading="flat",
        linewidth=0.0,
    )

    mode_y = mode_df["human_enrichment"].to_numpy(dtype=float)
    x_ticks = np.array([0.0, 0.5, 1.0])

    ax.plot(
        mode_df["progress_center"],
        mode_y,
        color=base_color,
        linewidth=2.0,
        marker="o",
        markersize=4.2,
        markerfacecolor="white",
        markeredgewidth=1.2,
        markeredgecolor=base_color,
        zorder=3,
    )

    ax.set_title(title, fontsize=13, pad=6)
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_ylim(float(y_edges[0]), float(y_edges[-1]))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(int(v)) for v in y_ticks])
    ax.grid(False)
    set_clean_spines(ax, x_ticks, y_ticks)
    return mesh, norm


def draw_mode_curve_panel(ax, mode_df: pd.DataFrame, title: str, base_color: str, y_values):
    x_ticks = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    y_values = get_display_y_ticks(y_values)

    ax.plot(
        mode_df["progress_center"],
        mode_df["human_enrichment"].to_numpy(dtype=float),
        color=base_color,
        linewidth=2.4,
        marker="o",
        markersize=3.6,
        markerfacecolor="white",
        markeredgewidth=1.1,
        markeredgecolor=base_color,
        zorder=3,
    )

    ax.set_title(title, fontsize=13, pad=6)
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_ylim(float(y_values.min() - 0.55), float(y_values.max() + 0.55))
    ax.set_yticks(y_values)
    ax.set_yticklabels([str(int(v)) for v in y_values])
    ax.grid(False)
    set_clean_spines(ax, x_ticks, y_values)


def draw_gapped_heatmap_panel(
    ax,
    heatmap: pd.DataFrame,
    mode_df: pd.DataFrame,
    title: str,
    base_color: str,
    x_start: float = 0.05,
    row_height: float = 0.42,
):
    matrix = heatmap.to_numpy(dtype=float)
    matrix = np.where(matrix >= MIN_PROP_TO_SHOW, matrix, 0.0)
    enrich_values = heatmap.index.to_numpy(dtype=float)
    y_ticks = get_display_y_ticks(enrich_values)
    n_bins = heatmap.shape[1]
    x_edges = np.linspace(x_start, 1.0, n_bins + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])

    cmap = make_colormap(base_color)
    norm = mcolors.Normalize(vmin=0.0, vmax=max(0.45, float(matrix.max())))

    for row_idx, y_val in enumerate(enrich_values):
        for col_idx, proportion in enumerate(matrix[row_idx]):
            if proportion <= 0:
                continue
            ax.add_patch(
                Rectangle(
                    (x_edges[col_idx], y_val - row_height / 2),
                    x_edges[col_idx + 1] - x_edges[col_idx],
                    row_height,
                    facecolor=cmap(norm(proportion)),
                    edgecolor="none",
                    linewidth=0,
                    zorder=1,
                )
            )

    ax.plot(
        x_centers,
        mode_df["human_enrichment"].to_numpy(dtype=float),
        color=base_color,
        linewidth=2.1,
        marker="o",
        markersize=3.4,
        markerfacecolor="white",
        markeredgewidth=1.0,
        markeredgecolor=base_color,
        zorder=3,
    )

    tick_progress = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    tick_positions = map_progress_to_axis(tick_progress, x_start=x_start, x_end=1.0)

    ax.set_title(title, fontsize=13, pad=6)
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_ylim(float(y_ticks.min() - 0.70), float(y_ticks.max() + 0.70))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(int(v)) for v in y_ticks])
    ax.grid(False)
    set_clean_spines(ax, tick_positions, y_ticks)
    return ScalarMappable(norm=norm, cmap=cmap)


def make_dense_heatmap_figure(task1_heatmap, task1_mode, task2_heatmap, task2_mode):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), constrained_layout=True)
    mesh1, _ = draw_dense_heatmap_panel(axes[0], task1_heatmap, task1_mode, "Task 1", TASK1_COLOR)
    mesh2, _ = draw_dense_heatmap_panel(axes[1], task2_heatmap, task2_mode, "Task 2", TASK2_COLOR)
    axes[0].set_ylabel("Feature enrichment", fontsize=12)
    axes[0].set_xlabel("Trial progress", fontsize=12)
    axes[1].set_xlabel("Trial progress", fontsize=12)
    style_figure(fig)
    cbar1 = fig.colorbar(mesh1, ax=axes[0], fraction=0.046, pad=0.03)
    cbar2 = fig.colorbar(mesh2, ax=axes[1], fraction=0.046, pad=0.03)
    cbar1.set_label("Selection proportion", fontsize=10)
    cbar2.set_label("Selection proportion", fontsize=10)
    cbar1.ax.tick_params(labelsize=9)
    cbar2.ax.tick_params(labelsize=9)
    save_figure(fig, DENSE_HEATMAP_OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_long_curve_figure(task1_mode, task1_heatmap, task2_mode, task2_heatmap):
    fig, axes = plt.subplots(1, 2, figsize=(18.0, 2.9), constrained_layout=True)
    draw_mode_curve_panel(axes[0], task1_mode, "Task 1", TASK1_COLOR, task1_heatmap.index.tolist())
    draw_mode_curve_panel(axes[1], task2_mode, "Task 2", TASK2_COLOR, task2_heatmap.index.tolist())
    axes[0].set_ylabel("Feature enrichment", fontsize=12)
    axes[0].set_xlabel("Trial progress", fontsize=12)
    axes[1].set_xlabel("Trial progress", fontsize=12)
    style_figure(fig)
    save_figure(fig, LONG_CURVE_OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_long_gapped_heatmap_figure(task1_heatmap, task1_mode, task2_heatmap, task2_mode):
    fig, axes = plt.subplots(1, 2, figsize=(18.0, 4.3), constrained_layout=True)
    sm1 = draw_gapped_heatmap_panel(axes[0], task1_heatmap, task1_mode, "Task 1", TASK1_COLOR)
    sm2 = draw_gapped_heatmap_panel(axes[1], task2_heatmap, task2_mode, "Task 2", TASK2_COLOR)
    axes[0].set_ylabel("Feature enrichment", fontsize=12)
    axes[0].set_xlabel("Trial progress", fontsize=12)
    axes[1].set_xlabel("Trial progress", fontsize=12)
    style_figure(fig)
    cbar1 = fig.colorbar(sm1, ax=axes[0], fraction=0.020, pad=0.02)
    cbar2 = fig.colorbar(sm2, ax=axes[1], fraction=0.020, pad=0.02)
    cbar1.set_label("Selection proportion", fontsize=10)
    cbar2.set_label("Selection proportion", fontsize=10)
    cbar1.ax.tick_params(labelsize=9)
    cbar2.ax.tick_params(labelsize=9)
    save_figure(fig, LONG_GAPPED_HEATMAP_OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    apply_publication_defaults()

    task1_base_df = add_progress_bins(
        build_task1_trial_df(), unit_cols=["subject", "phase"], order_col="round", n_bins=BASE_N_BINS
    )
    task2_base_df = add_progress_bins(
        build_task2_trial_df(), unit_cols=["subject_id"], order_col="trial_id", n_bins=BASE_N_BINS
    )
    task1_long_df = add_progress_bins(
        build_task1_trial_df(), unit_cols=["subject", "phase"], order_col="round", n_bins=LONG_N_BINS
    )
    task2_long_df = add_progress_bins(
        build_task2_trial_df(), unit_cols=["subject_id"], order_col="trial_id", n_bins=LONG_N_BINS
    )

    task1_heatmap, task1_mode = build_heatmap_and_mode(task1_base_df, n_bins=BASE_N_BINS)
    task2_heatmap, task2_mode = build_heatmap_and_mode(task2_base_df, n_bins=BASE_N_BINS)
    task1_heatmap_long, task1_mode_long = build_heatmap_and_mode(task1_long_df, n_bins=LONG_N_BINS)
    task2_heatmap_long, task2_mode_long = build_heatmap_and_mode(task2_long_df, n_bins=LONG_N_BINS)

    make_dense_heatmap_figure(task1_heatmap, task1_mode, task2_heatmap, task2_mode)
    make_long_curve_figure(task1_mode_long, task1_heatmap_long, task2_mode_long, task2_heatmap_long)
    make_long_gapped_heatmap_figure(task1_heatmap_long, task1_mode_long, task2_heatmap_long, task2_mode_long)

    print(f"Saved figure to: {DENSE_HEATMAP_OUTPUT_PATH}")
    print(f"Saved figure to: {LONG_CURVE_OUTPUT_PATH}")
    print(f"Saved figure to: {LONG_GAPPED_HEATMAP_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
