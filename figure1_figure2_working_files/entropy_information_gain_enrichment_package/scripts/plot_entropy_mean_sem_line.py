#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PATTERN_ORDER = ["4-0", "3-1", "2-2", "2-1-1", "1-1-1-1"]
DISPLAY_ORDER = ["1", "2", "3", "4"]
DISPLAY_GROUPS = {
    "4": ["4-0"],
    "3": ["3-1"],
    "2": ["2-2", "2-1-1"],
    "1": ["1-1-1-1"],
}
COLORS = {
    "dimension_level": "#7B3FB2",
    "trial_primary": "#7B3FB2",
}
Y_LIMITS = (0.02, 0.14)
Y_TICKS = [0.02, 0.06, 0.10, 0.14]


def read_subject_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pattern = row["pattern_label"]
            if pattern not in PATTERN_ORDER:
                continue
            rows.append(
                {
                    "subject_id": row["subject_id"],
                    "pattern_label": pattern,
                    "balanced_correct_evidence_bits": float(row["balanced_correct_evidence_bits"]),
                }
            )
    return rows


def read_pattern_summary(path: Path) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pattern = row["pattern_label"]
            if pattern not in PATTERN_ORDER:
                continue
            if not row["balanced_correct_evidence_bits"]:
                continue
            rows[pattern] = {
                "mean": float(row["balanced_correct_evidence_bits"]),
                "n": float(row["n_dim_trials"]),
            }
    return rows


def grouped_plot_rows(
    pattern_rows: dict[str, dict[str, float]],
    subject_rows: list[dict[str, object]],
) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for display_label, source_patterns in DISPLAY_GROUPS.items():
        weighted_sum = 0.0
        total_n = 0.0
        for pattern in source_patterns:
            if pattern not in pattern_rows:
                continue
            weighted_sum += pattern_rows[pattern]["mean"] * pattern_rows[pattern]["n"]
            total_n += pattern_rows[pattern]["n"]
        if total_n == 0:
            continue

        by_subject: dict[str, list[float]] = {}
        for row in subject_rows:
            if row["pattern_label"] in source_patterns:
                by_subject.setdefault(str(row["subject_id"]), []).append(float(row["balanced_correct_evidence_bits"]))
        subject_values = np.array([np.mean(values) for values in by_subject.values()], dtype=float)
        sem = float(subject_values.std(ddof=1) / np.sqrt(len(subject_values))) if len(subject_values) > 1 else 0.0

        rows[display_label] = {
            "mean": weighted_sum / total_n,
            "sem": sem,
            "n": float(len(subject_values)),
        }
    return rows


def values_for_plot(rows: dict[str, dict[str, float]]) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    labels = [label for label in DISPLAY_ORDER if label in rows]
    means = np.array([rows[label]["mean"] for label in labels], dtype=float)
    sems = np.array([rows[label]["sem"] for label in labels], dtype=float)
    ns = np.array([rows[label]["n"] for label in labels], dtype=float)
    return labels, means, sems, ns


def style_axes(ax: plt.Axes, x_min: float, x_max: float, spine_right: float) -> None:
    ax.set_facecolor("white")
    ax.figure.patch.set_facecolor("white")
    ax.set_xlabel("Specific Feature Enrichment", fontsize=20, fontweight="bold", labelpad=12)
    ax.set_ylabel("information gain (bits)", fontsize=20, fontweight="bold", labelpad=12)
    ax.tick_params(axis="both", labelsize=18, width=2.4, length=8, direction="out")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2.6)
    ax.spines["bottom"].set_linewidth(2.6)
    ax.spines["left"].set_position(("outward", 12))
    ax.spines["bottom"].set_position(("outward", 12))
    ax.spines["bottom"].set_bounds(0, spine_right)
    ax.grid(False)
    ax.set_ylim(*Y_LIMITS)
    ax.set_yticks(Y_TICKS)
    ax.set_xlim(x_min, x_max)


def draw_single(rows: dict[str, dict[str, float]], label: str, color: str, out_base: Path) -> None:
    patterns, means, sems, _ = values_for_plot(rows)
    x = np.arange(len(patterns), dtype=float)

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.errorbar(
        x,
        means,
        yerr=sems,
        color=color,
        marker="o",
        markersize=10,
        markerfacecolor=color,
        markeredgecolor="black",
        markeredgewidth=2.0,
        linewidth=4.2,
        elinewidth=2.4,
        capsize=6,
        capthick=2.4,
        zorder=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(patterns)
    style_axes(ax, -0.28, max(x) + 0.28 if len(x) else 1.0, max(x) if len(x) else 1.0)
    fig.subplots_adjust(left=0.28, right=0.94, bottom=0.24, top=0.94)

    fig.savefig(out_base.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def draw_combined(
    dimension_rows: dict[str, dict[str, float]],
    primary_rows: dict[str, dict[str, float]],
    out_base: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.0))

    for rows, label, color, offset in [
        (dimension_rows, "Dimension level", COLORS["dimension_level"], -0.04),
        (primary_rows, "Trial primary", COLORS["trial_primary"], 0.04),
    ]:
        patterns, means, sems, _ = values_for_plot(rows)
        x = np.array([DISPLAY_ORDER.index(pattern) for pattern in patterns], dtype=float) + offset
        ax.errorbar(
            x,
            means,
            yerr=sems,
            label=label,
            color=color,
            marker="o",
            markersize=10,
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=2.0,
            linewidth=4.2,
            elinewidth=2.4,
            capsize=6,
            capthick=2.4,
            zorder=3,
        )

    ax.set_xticks(np.arange(len(DISPLAY_ORDER)))
    ax.set_xticklabels(DISPLAY_ORDER)
    style_axes(ax, -0.32, len(DISPLAY_ORDER) - 1 + 0.32, len(DISPLAY_ORDER) - 1)
    fig.subplots_adjust(left=0.27, right=0.94, bottom=0.24, top=0.94)

    fig.savefig(out_base.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw entropy information gain as mean + SEM line plots.")
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Entropy output directory. Defaults to ../outputs relative to this script.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Plot output directory. Defaults to input-dir.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    input_dir = Path(args.input_dir).expanduser().resolve() if args.input_dir else script_dir.parent / "outputs"
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dimension_subject_rows = read_subject_rows(
        input_dir / "dimension_level_entropy_subject_level_balanced_correct_evidence_bits.csv"
    )
    primary_subject_rows = read_subject_rows(
        input_dir / "trial_primary_entropy_subject_level_balanced_correct_evidence_bits.csv"
    )
    dimension_pattern_rows = read_pattern_summary(
        input_dir / "dimension_level_entropy_pattern_summary.csv"
    )
    primary_pattern_rows = read_pattern_summary(
        input_dir / "trial_primary_entropy_pattern_summary.csv"
    )
    dimension_rows = grouped_plot_rows(dimension_pattern_rows, dimension_subject_rows)
    primary_rows = grouped_plot_rows(primary_pattern_rows, primary_subject_rows)

    draw_single(
        dimension_rows,
        "Dimension-Level Entropy Information Gain",
        COLORS["dimension_level"],
        output_dir / "dimension_level_entropy_mean_sem_line",
    )
    draw_single(
        primary_rows,
        "Trial-Primary Entropy Information Gain",
        COLORS["trial_primary"],
        output_dir / "trial_primary_entropy_mean_sem_line",
    )
    draw_combined(
        dimension_rows,
        primary_rows,
        output_dir / "entropy_information_gain_mean_sem_line_combined",
    )

    print(f"Saved line plots to: {output_dir}")


if __name__ == "__main__":
    main()
