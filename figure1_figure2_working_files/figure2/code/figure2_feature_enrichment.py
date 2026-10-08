# Figure 2 feature-enrichment plots (panels 2b and the SEM progress curve)
# Extracted from cells 71-72 of the original analysis notebook.
# Run from the package root; reads data/ and writes to figure2/output/.

# Task 1 / Task 2 subject-level difference-versus-random violin figures
# New versions only: keep the bar-based difference plots untouched.
from collections import Counter
from pathlib import Path
import ast

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from publication_plot_style import apply_publication_defaults, style_figure, save_figure

apply_publication_defaults()

OUTPUT_DIR = Path("figure2/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TASK1_VIOLIN_OUTPUT = OUTPUT_DIR / "task1_human_vs_random_feature_enrichment_distribution_all_phases_diff_vs_random_violin_mean.png"
TASK2_VIOLIN_OUTPUT = OUTPUT_DIR / "task2_human_vs_random_feature_enrichment_distribution_all_phases_degree_2_to_4_diff_vs_random_violin_mean.png"

EDGE_COLOR = "#000000"
TASK1_VIOLIN_COLOR = tuple(np.array(mcolors.to_rgb("#005BBB")) + (1 - np.array(mcolors.to_rgb("#005BBB"))) * 0.54)
TASK2_VIOLIN_COLOR = tuple(np.array(mcolors.to_rgb("#F28E2B")) + (1 - np.array(mcolors.to_rgb("#F28E2B"))) * 0.42)


def p_to_stars_violin(p):
    if pd.isna(p):
        return "n.s."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def fdr_bh_violin(pvals):
    pvals = np.asarray(pvals, dtype=float)
    out = np.full(len(pvals), np.nan)
    valid = np.isfinite(pvals)
    if valid.sum() == 0:
        return out
    valid_idx = np.flatnonzero(valid)
    ranked_idx = np.argsort(pvals[valid])
    ranked_p = pvals[valid][ranked_idx]
    n = len(ranked_p)
    adjusted = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        adj = min(prev, ranked_p[i] * n / rank)
        adjusted[i] = min(adj, 1.0)
        prev = adj
    restored = np.empty(n, dtype=float)
    restored[ranked_idx] = adjusted
    out[valid_idx] = restored
    return out


def subject_level_diff_table(trial_df, subject_col, human_label_col, random_label_col, order):
    rows = []
    for subject_id, sub_df in trial_df.groupby(subject_col, observed=False):
        for label in order:
            human_pct = 100 * (sub_df[human_label_col] == label).mean()
            random_pct = 100 * (sub_df[random_label_col] == label).mean()
            rows.append(
                {
                    "subject_id": subject_id,
                    "category": label,
                    "human_pct": human_pct,
                    "random_pct": random_pct,
                    "diff_pct": human_pct - random_pct,
                }
            )
    return pd.DataFrame(rows)


def subject_level_stats(diff_df, order):
    rows = []
    raw_pvals = []
    for label in order:
        vals = diff_df.loc[diff_df["category"] == label, "diff_pct"].dropna().astype(float)
        p_raw = 1.0
        if len(vals) > 0 and not np.allclose(vals.to_numpy(), 0.0):
            try:
                p_raw = wilcoxon(vals, zero_method="wilcox", alternative="two-sided", method="auto").pvalue
            except ValueError:
                p_raw = 1.0
        rows.append(
            {
                "category": label,
                "mean_diff": vals.mean() if len(vals) > 0 else np.nan,
                "median_diff": vals.median() if len(vals) > 0 else np.nan,
                "n_subjects": len(vals),
                "p_raw": p_raw,
            }
        )
        raw_pvals.append(p_raw)
    stats_df = pd.DataFrame(rows)
    stats_df["p_fdr"] = fdr_bh_violin(raw_pvals)
    stats_df["sig"] = stats_df["p_fdr"].apply(p_to_stars_violin)
    return stats_df


def symmetric_axis_spec(diff_df):
    max_abs = float(np.nanmax(np.abs(diff_df["diff_pct"].to_numpy(dtype=float)))) if len(diff_df) > 0 else 10.0
    limit = max(20.0, np.ceil((max_abs + 6.0) / 10.0) * 10.0)
    step = 10.0 if limit <= 30 else 20.0 if limit <= 60 else 25.0 if limit <= 75 else 50.0
    tick_max = np.floor(limit / step) * step
    ticks = np.arange(-tick_max, tick_max + 0.5 * step, step)
    return limit, ticks


def draw_violin_mean_diff(diff_df, stats_df, order, color, out_path, figsize, x_step, violin_width, x_pad):
    fig, ax = plt.subplots(figsize=figsize)
    x_positions = np.arange(len(order), dtype=float) * x_step

    axis_limit, y_ticks = symmetric_axis_spec(diff_df)

    for x_pos, label in zip(x_positions, order):
        vals = diff_df.loc[diff_df["category"] == label, "diff_pct"].dropna().astype(float).to_numpy()
        stat_row = stats_df.loc[stats_df["category"] == label].iloc[0]
        mean_val = float(stat_row["mean_diff"])
        sig_text = stat_row["sig"]

        if len(vals) >= 2 and np.ptp(vals) > 1e-8:
            parts = ax.violinplot(
                [vals],
                positions=[x_pos],
                widths=violin_width,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for body in parts["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor(EDGE_COLOR)
                body.set_linewidth(1.2)
                body.set_alpha(0.8)
        else:
            ax.plot(
                [x_pos - violin_width * 0.28, x_pos + violin_width * 0.28],
                [mean_val, mean_val],
                color=EDGE_COLOR,
                linewidth=1.2,
                zorder=3,
            )

        ax.scatter(
            [x_pos],
            [mean_val],
            s=34,
            facecolors="white",
            edgecolors=EDGE_COLOR,
            linewidths=1.2,
            zorder=4,
        )

        high_anchor = max(np.max(vals), mean_val) if len(vals) > 0 else mean_val
        low_anchor = min(np.min(vals), mean_val) if len(vals) > 0 else mean_val
        text_pad = max(2.0, axis_limit * 0.04)
        if mean_val >= 0:
            text_y = min(high_anchor + text_pad, axis_limit - 1.5)
            va = "bottom"
        else:
            text_y = max(low_anchor - text_pad, -axis_limit + 1.5)
            va = "top"
        ax.text(
            x_pos,
            text_y,
            sig_text,
            ha="center",
            va=va,
            fontsize=10,
            color=EDGE_COLOR,
            clip_on=False,
            zorder=5,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(order)
    ax.set_xlabel("Feature Enrichment Degree", labelpad=3)
    ax.set_ylabel("Difference vs Random (%)", labelpad=3)
    ax.set_title("")

    style_figure(fig)
    ax.grid(False)
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        bottom=True,
        top=False,
        left=True,
        right=False,
        length=6.2,
        width=2.0,
        pad=5,
        colors=EDGE_COLOR,
    )
    ax.tick_params(axis="both", which="minor", bottom=False, left=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(EDGE_COLOR)
    ax.spines["bottom"].set_color(EDGE_COLOR)
    ax.spines["left"].set_linewidth(2.4)
    ax.spines["bottom"].set_linewidth(2.4)
    ax.spines["left"].set_bounds(float(y_ticks[0]), float(y_ticks[-1]))
    ax.spines["bottom"].set_bounds(x_positions[0], x_positions[-1])
    ax.spines["bottom"].set_position(("outward", 8))
    ax.set_xlim(x_positions[0] - x_pad, x_positions[-1] + x_pad)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_yticks(y_ticks)
    ax.axhline(0, color=EDGE_COLOR, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)

    fig.tight_layout(pad=0.35)
    save_figure(fig, out_path, dpi=300, bbox_inches="tight")
    plt.show()


def _sum_category_digits_task1_violin(value):
    if pd.isna(value):
        return np.nan
    digits = [int(ch) for ch in str(value).strip() if ch.isdigit()]
    return int(sum(digits)) if digits else np.nan


def _parse_block_options_task1_violin(value):
    if pd.isna(value):
        return []
    parsed = ast.literal_eval(value) if isinstance(value, str) else value
    return [str(item) for item in parsed]


def compute_human_trial_enrichment_task1_violin(row):
    dim_cols = ["dim1_category", "dim2_category", "dim3_category"]
    if pd.notna(row.get("dim4_category", np.nan)):
        dim_cols.append("dim4_category")
    dim_scores = [_sum_category_digits_task1_violin(row[col]) for col in dim_cols if col in row.index]
    dim_scores = [score for score in dim_scores if pd.notna(score)]
    return int(max(dim_scores)) if dim_scores else np.nan


def compute_random_trial_enrichment_task1_violin(row, rng):
    all_options = []
    for block_col in ["block1", "block2", "block3"]:
        if block_col in row.index:
            all_options.extend(_parse_block_options_task1_violin(row[block_col]))
    if len(all_options) != 9:
        return np.nan
    shuffled = list(rng.permutation(all_options))
    groups = [shuffled[i:i + 3] for i in range(0, 9, 3)]
    dim_count = 4 if pd.notna(row.get("dim4_category", np.nan)) else 3
    dim_enrichment_scores = []
    for dim_idx in range(dim_count):
        dim_total = 0
        for group in groups:
            dim_values = [str(option)[dim_idx] for option in group]
            dim_total += Counter(dim_values).most_common(1)[0][1]
        dim_enrichment_scores.append(dim_total)
    return int(max(dim_enrichment_scores)) if dim_enrichment_scores else np.nan


def build_task1_trial_level_violin():
    df_human = pd.read_csv(Path("data") / "choice_category_uniform_105.csv", index_col=0)
    rng = np.random.default_rng(42)
    trial_df = df_human.copy()
    trial_df["human_enrichment"] = trial_df.apply(compute_human_trial_enrichment_task1_violin, axis=1)
    trial_df["random_enrichment"] = trial_df.apply(
        lambda row: compute_random_trial_enrichment_task1_violin(row, rng),
        axis=1,
    )
    trial_df = trial_df.dropna(subset=["subject", "phase", "human_enrichment", "random_enrichment"]).copy()
    trial_df["human_label"] = trial_df["human_enrichment"].astype(int).apply(lambda value: "<6" if value < 6 else str(value))
    trial_df["random_label"] = trial_df["random_enrichment"].astype(int).apply(lambda value: "<6" if value < 6 else str(value))
    return trial_df


def parse_selected_objects_task2_violin(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    return ast.literal_eval(value)


def compute_feature_enrichment_task2_violin(selected_objects):
    try:
        objects = parse_selected_objects_task2_violin(selected_objects)
    except (ValueError, SyntaxError, TypeError):
        return np.nan
    if not isinstance(objects, list) or len(objects) != 4:
        return np.nan
    if any(not isinstance(obj, str) or not obj.strip() for obj in objects):
        return np.nan
    features = [feature.strip() for obj in objects for feature in obj.split("_") if feature.strip()]
    return max(Counter(features).values()) if features else np.nan


def parse_images_this_trial_task2_violin(value):
    if isinstance(value, list):
        paths = value
    elif pd.isna(value):
        return []
    else:
        try:
            paths = ast.literal_eval(value)
        except (ValueError, SyntaxError, TypeError):
            return []
    if not isinstance(paths, list):
        return []
    return [Path(path).stem for path in paths if isinstance(path, str) and path.strip()]


def build_task2_trial_level_violin():
    raw_df = pd.read_csv(Path("data") / "summary_data_0723_task2.csv")
    rng = np.random.default_rng(42)
    trial_df = raw_df.loc[
        raw_df["group"].isin(["observation", "non_observation"]) & raw_df["score_noisy"].notna(),
        ["subject_id", "trial_id", "group", "score_noisy", "selected_objects", "images_this_trial"],
    ].copy()
    trial_df["available_objects"] = trial_df["images_this_trial"].apply(parse_images_this_trial_task2_violin)
    valid_option_pool = trial_df["available_objects"].apply(lambda objects: len(objects) == 16 and len(set(objects)) == 16)
    trial_df = trial_df.loc[valid_option_pool].copy()
    trial_df["random_selected_objects"] = trial_df["available_objects"].apply(
        lambda objects: rng.choice(objects, size=4, replace=False).tolist()
    )
    trial_df["human_enrichment"] = trial_df["selected_objects"].apply(compute_feature_enrichment_task2_violin)
    trial_df["random_enrichment"] = trial_df["random_selected_objects"].apply(compute_feature_enrichment_task2_violin)
    trial_df = trial_df.dropna(subset=["subject_id", "human_enrichment", "random_enrichment"]).copy()
    trial_df["human_label"] = trial_df["human_enrichment"].astype(int)
    trial_df["random_label"] = trial_df["random_enrichment"].astype(int)
    return trial_df


task1_order_violin = ["<6", "6", "7", "9"]
task1_trial_violin = build_task1_trial_level_violin()
task1_diff_subject = subject_level_diff_table(
    task1_trial_violin,
    subject_col="subject",
    human_label_col="human_label",
    random_label_col="random_label",
    order=task1_order_violin,
)
task1_stats_violin = subject_level_stats(task1_diff_subject, task1_order_violin)
print("Task 1 subject-level difference summary (Wilcoxon vs 0, BH-FDR corrected):")
print(task1_stats_violin.to_string(index=False))
draw_violin_mean_diff(
    task1_diff_subject,
    task1_stats_violin,
    order=task1_order_violin,
    color=TASK1_VIOLIN_COLOR,
    out_path=TASK1_VIOLIN_OUTPUT,
    figsize=(4.8, 2.35),
    x_step=0.72,
    violin_width=0.25,
    x_pad=0.18,
)
print(f"Saved Task 1 violin difference figure to: {TASK1_VIOLIN_OUTPUT.resolve()}")


task2_order_violin = [2, 3, 4]
task2_trial_violin = build_task2_trial_level_violin()
task2_diff_subject = subject_level_diff_table(
    task2_trial_violin,
    subject_col="subject_id",
    human_label_col="human_label",
    random_label_col="random_label",
    order=task2_order_violin,
)
task2_stats_violin = subject_level_stats(task2_diff_subject, task2_order_violin)
print("Task 2 subject-level difference summary (Wilcoxon vs 0, BH-FDR corrected):")
print(task2_stats_violin.to_string(index=False))
draw_violin_mean_diff(
    task2_diff_subject,
    task2_stats_violin,
    order=task2_order_violin,
    color=TASK2_VIOLIN_COLOR,
    out_path=TASK2_VIOLIN_OUTPUT,
    figsize=(4.1, 2.35),
    x_step=0.72,
    violin_width=0.25,
    x_pad=0.18,
)
print(f"Saved Task 2 violin difference figure to: {TASK2_VIOLIN_OUTPUT.resolve()}")


# SEM feature-enrichment curve over normalized trial progress
# Task 1 / Task 2 human feature-enrichment trajectories over normalized trial progress
# Reuses the same human_enrichment metric as the violin figures above.
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from publication_plot_style import apply_publication_defaults, style_figure, save_figure

apply_publication_defaults()

OUTPUT_DIR = Path("figure2/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROGRESS_OUTPUT = OUTPUT_DIR / "task1_task2_feature_enrichment_vs_trial_progress_sem.png"

TASK1_PROGRESS_COLOR = tuple(
    np.array(mcolors.to_rgb("#005BBB")) + (1 - np.array(mcolors.to_rgb("#005BBB"))) * 0.10
)
TASK2_PROGRESS_COLOR = tuple(
    np.array(mcolors.to_rgb("#F28E2B")) + (1 - np.array(mcolors.to_rgb("#F28E2B"))) * 0.10
)
EDGE_COLOR = "#000000"
N_BINS = 10


def summarize_progress_curve(df, unit_cols, order_col, metric_col="human_enrichment", n_bins=10):
    keep_cols = list(unit_cols) + [order_col, metric_col]
    use_df = df[keep_cols].dropna().copy()
    use_df[order_col] = pd.to_numeric(use_df[order_col], errors="coerce")
    use_df = use_df.dropna(subset=[order_col]).copy()
    use_df = use_df.sort_values(list(unit_cols) + [order_col]).copy()

    use_df["trial_rank"] = (
        use_df.groupby(list(unit_cols), observed=False).cumcount() + 1
    )
    n_trials = use_df.groupby(list(unit_cols), observed=False)[metric_col].transform("size")
    use_df["trial_progress"] = (use_df["trial_rank"] - 1) / (n_trials - 1).where(n_trials > 1, 1)
    use_df["trial_progress"] = use_df["trial_progress"].fillna(0.0).clip(0.0, 1.0)
    use_df["progress_bin"] = np.minimum(
        (use_df["trial_progress"] * n_bins).astype(int),
        n_bins - 1,
    )
    use_df["progress_center"] = (use_df["progress_bin"] + 0.5) / n_bins

    unit_bin_df = (
        use_df.groupby(list(unit_cols) + ["progress_bin", "progress_center"], as_index=False, observed=False)[metric_col]
        .mean()
    )

    summary_df = (
        unit_bin_df.groupby(["progress_bin", "progress_center"], as_index=False, observed=False)
        .agg(
            mean_enrichment=(metric_col, "mean"),
            sem_enrichment=(
                metric_col,
                lambda s: s.std(ddof=1) / np.sqrt(s.notna().sum()) if s.notna().sum() > 1 else np.nan,
            ),
            n_units=(metric_col, "count"),
        )
        .sort_values("progress_center")
        .reset_index(drop=True)
    )
    summary_df["lower"] = summary_df["mean_enrichment"] - summary_df["sem_enrichment"]
    summary_df["upper"] = summary_df["mean_enrichment"] + summary_df["sem_enrichment"]
    summary_df["plot_progress"] = np.linspace(0.0, 1.0, len(summary_df))
    return use_df, unit_bin_df, summary_df


task1_progress_df = build_task1_trial_level_violin()
task2_progress_df = build_task2_trial_level_violin()

_, task1_unit_bin_df, task1_progress_summary = summarize_progress_curve(
    task1_progress_df,
    unit_cols=["subject", "phase"],
    order_col="round",
    metric_col="human_enrichment",
    n_bins=N_BINS,
)
_, task2_unit_bin_df, task2_progress_summary = summarize_progress_curve(
    task2_progress_df,
    unit_cols=["subject_id"],
    order_col="trial_id",
    metric_col="human_enrichment",
    n_bins=N_BINS,
)

FIGSIZE = (5.6, 3.4)
LINE_WIDTH = 2.2
MARKER_SIZE = 4.0
BAND_ALPHA = 0.20
AXIS_LABEL_SIZE = 12
TICK_LABEL_SIZE = 11
SPINE_WIDTH = 1.6
N_Y_TICKS = 3
X_OFFSET = 0.000
TASK1_DISPLAY_SHIFT = -0.10
TASK2_DISPLAY_SHIFT = 0.00
TICK_MARGIN_FRAC = 0.15
X_LIM = (0.0, 1.0)

def _next_nice_step(step):
    exponent = np.floor(np.log10(step))
    fraction = step / (10 ** exponent)
    nice_fractions = np.array([1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0])
    idx = np.searchsorted(nice_fractions, fraction, side="right")
    if idx >= len(nice_fractions):
        exponent += 1
        idx = 0
    return float(nice_fractions[idx] * (10 ** exponent))


def _nice_step(raw_step):
    exponent = np.floor(np.log10(raw_step))
    fraction = raw_step / (10 ** exponent)
    if fraction <= 1.0:
        nice_fraction = 1.0
    elif fraction <= 2.0:
        nice_fraction = 2.0
    elif fraction <= 2.5:
        nice_fraction = 2.5
    elif fraction <= 3.0:
        nice_fraction = 3.0
    elif fraction <= 4.0:
        nice_fraction = 4.0
    elif fraction <= 5.0:
        nice_fraction = 5.0
    elif fraction <= 6.0:
        nice_fraction = 6.0
    elif fraction <= 8.0:
        nice_fraction = 8.0
    else:
        nice_fraction = 10.0
    return float(nice_fraction * (10 ** exponent))


def step_decimals(step):
    for decimals in range(4):
        if np.isclose(step, np.round(step, decimals)):
            return decimals
    return 4


def compute_display_ticks(lower, upper, n_ticks):
    raw_step = max((upper - lower) / max(n_ticks - 1, 1), 1e-9)
    exponent = np.floor(np.log10(raw_step))
    nice_fractions = np.array([1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0])
    candidates = []
    for exp in (exponent - 1, exponent, exponent + 1):
        candidates.extend((nice_fractions * (10 ** exp)).tolist())
    candidates = sorted({float(step) for step in candidates if step > 0})

    mid = 0.5 * (lower + upper)
    best = None
    for step in candidates:
        if step < raw_step * 0.75 or step > raw_step * 3.0:
            continue
        half_span = 0.5 * step * (n_ticks - 1)
        start_raw = mid - half_span
        grid = step / 2.0
        start = np.round(start_raw / grid) * grid
        ticks = start + step * np.arange(n_ticks)
        if ticks[0] > lower + 1e-9 or ticks[-1] < upper - 1e-9:
            continue
        extra = (lower - ticks[0]) + (ticks[-1] - upper)
        score = (extra, step)
        if best is None or score < best[0]:
            best = (score, ticks, step)

    if best is None:
        step = _nice_step(raw_step)
        half_span = 0.5 * step * (n_ticks - 1)
        start = np.round((mid - half_span) / (step / 2.0)) * (step / 2.0)
        ticks = start + step * np.arange(n_ticks)
    else:
        _, ticks, step = best

    decimals = max(0, step_decimals(step))
    ticks = np.round(ticks, decimals + 1)
    return ticks, step


fig, ax_left = plt.subplots(figsize=FIGSIZE)
ax_right = ax_left.twinx()

task1_x = task1_progress_summary["progress_center"].to_numpy(dtype=float) - X_OFFSET
task2_x = task2_progress_summary["progress_center"].to_numpy(dtype=float) + X_OFFSET
left_lower = float(np.nanmin(task1_progress_summary["lower"]))
left_upper = float(np.nanmax(task1_progress_summary["upper"]))
right_lower = float(np.nanmin(task2_progress_summary["lower"]))
right_upper = float(np.nanmax(task2_progress_summary["upper"]))
task1_mean_plot = task1_progress_summary["mean_enrichment"] + TASK1_DISPLAY_SHIFT
task1_lower_plot = task1_progress_summary["lower"] + TASK1_DISPLAY_SHIFT
task1_upper_plot = task1_progress_summary["upper"] + TASK1_DISPLAY_SHIFT
task2_mean_plot = task2_progress_summary["mean_enrichment"] + TASK2_DISPLAY_SHIFT
task2_lower_plot = task2_progress_summary["lower"] + TASK2_DISPLAY_SHIFT
task2_upper_plot = task2_progress_summary["upper"] + TASK2_DISPLAY_SHIFT
left_ticks, left_step = compute_display_ticks(float(np.nanmin(task1_lower_plot)), float(np.nanmax(task1_upper_plot)), N_Y_TICKS)
right_ticks, right_step = compute_display_ticks(float(np.nanmin(task2_lower_plot)), float(np.nanmax(task2_upper_plot)), N_Y_TICKS)
left_ylim = (float(left_ticks[0] - TICK_MARGIN_FRAC * left_step), float(left_ticks[-1] + TICK_MARGIN_FRAC * left_step))
right_ylim = (float(right_ticks[0] - TICK_MARGIN_FRAC * right_step), float(right_ticks[-1] + TICK_MARGIN_FRAC * right_step))

ax_left.plot(
    task1_x,
    task1_mean_plot,
    color=TASK1_PROGRESS_COLOR,
    linewidth=LINE_WIDTH,
    linestyle="-",
    marker="o",
    markersize=MARKER_SIZE,
    markerfacecolor="white",
    markeredgecolor=TASK1_PROGRESS_COLOR,
    markeredgewidth=1.2,
    zorder=4,
)
ax_left.fill_between(
    task1_x,
    task1_lower_plot,
    task1_upper_plot,
    color=TASK1_PROGRESS_COLOR,
    alpha=BAND_ALPHA,
    linewidth=0,
    zorder=2,
)

ax_right.plot(
    task2_x,
    task2_mean_plot,
    color=TASK2_PROGRESS_COLOR,
    linewidth=LINE_WIDTH,
    linestyle=(0, (4, 2)),
    marker="s",
    markersize=MARKER_SIZE,
    markerfacecolor="white",
    markeredgecolor=TASK2_PROGRESS_COLOR,
    markeredgewidth=1.2,
    zorder=5,
)
ax_right.fill_between(
    task2_x,
    task2_lower_plot,
    task2_upper_plot,
    color=TASK2_PROGRESS_COLOR,
    alpha=BAND_ALPHA,
    linewidth=0,
    zorder=1,
)

ax_left.set_xlabel("Trial progress", fontsize=AXIS_LABEL_SIZE, labelpad=8)
ax_left.set_ylabel("Feature enrichment (Task 1)", fontsize=AXIS_LABEL_SIZE, labelpad=10, color=TASK1_PROGRESS_COLOR)
ax_right.set_ylabel("Feature enrichment (Task 2)", fontsize=AXIS_LABEL_SIZE, labelpad=10, color=TASK2_PROGRESS_COLOR)
ax_left.set_xlim(*X_LIM)
ax_right.set_xlim(*X_LIM)
ax_left.set_xticks([0.0, 0.5, 1.0])
ax_left.set_ylim(*left_ylim)
ax_right.set_ylim(*right_ylim)
ax_left.set_yticks(left_ticks)
ax_right.set_yticks(right_ticks)
left_decimals = step_decimals(left_step)
right_decimals = step_decimals(right_step)
ax_left.set_yticklabels([f"{tick:.{left_decimals}f}" for tick in left_ticks])
ax_right.set_yticklabels([f"{tick:.{right_decimals}f}" for tick in right_ticks])

for ax in (ax_left, ax_right):
    ax.grid(False)
    ax.spines["top"].set_visible(False)

ax_left.spines["right"].set_visible(False)
ax_right.spines["left"].set_visible(False)
ax_right.spines["bottom"].set_visible(False)
ax_right.spines["right"].set_visible(True)
ax_left.spines["bottom"].set_position(("outward", 6))
ax_left.spines["left"].set_position(("axes", 0.0))
ax_right.spines["right"].set_position(("axes", 1.0))
ax_left.spines["left"].set_linewidth(SPINE_WIDTH)
ax_left.spines["bottom"].set_linewidth(SPINE_WIDTH)
ax_right.spines["right"].set_linewidth(SPINE_WIDTH)
ax_left.spines["left"].set_color(EDGE_COLOR)
ax_left.spines["bottom"].set_color(EDGE_COLOR)
ax_right.spines["right"].set_color(EDGE_COLOR)
ax_right.spines["right"].set_zorder(10)
ax_left.spines["left"].set_bounds(left_ticks[0], left_ticks[-1])
ax_right.spines["right"].set_bounds(right_ticks[0], right_ticks[-1])
ax_left.spines["bottom"].set_bounds(0.0, 1.0)

ax_left.tick_params(
    axis="x",
    which="major",
    direction="out",
    bottom=True,
    top=False,
    length=4.5,
    width=1.4,
    pad=4,
    colors=EDGE_COLOR,
    labelsize=TICK_LABEL_SIZE,
)
ax_left.tick_params(
    axis="y",
    which="major",
    direction="out",
    left=True,
    right=False,
    length=4.5,
    width=1.4,
    pad=4,
    colors=TASK1_PROGRESS_COLOR,
    labelsize=TICK_LABEL_SIZE,
)
ax_right.tick_params(
    axis="y",
    which="major",
    direction="out",
    left=False,
    right=True,
    length=4.5,
    width=1.4,
    pad=4,
    colors=TASK2_PROGRESS_COLOR,
    labelsize=TICK_LABEL_SIZE,
)
ax_right.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
ax_left.tick_params(axis="both", which="minor", bottom=False, left=False)
ax_right.tick_params(axis="y", which="minor", right=False)

fig.subplots_adjust(left=0.17, right=0.83, bottom=0.26, top=0.96)
# Save this twin-axis panel directly: the shared final-pass helper is
# designed for single-axis figures and suppresses the right spine.
fig.savefig(PROGRESS_OUTPUT, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved feature-enrichment progress figure to: {PROGRESS_OUTPUT.resolve()}")
print(
    "Task 1 units (subject x phase):",
    task1_unit_bin_df[["subject", "phase"]].drop_duplicates().shape[0],
)
print(
    "Task 2 units (subject):",
    task2_unit_bin_df[["subject_id"]].drop_duplicates().shape[0],
)
plt.show()

