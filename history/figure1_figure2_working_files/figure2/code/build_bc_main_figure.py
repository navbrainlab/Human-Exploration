from __future__ import annotations

"""Build the publication-style frequency/persistence panels B and C.

This script is intentionally self-contained and writes only to the directory
that contains it.  It reuses source data and existing analysis conventions from
the four project folders without modifying those folders.
"""

from collections import defaultdict
from importlib.util import module_from_spec, spec_from_file_location
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from scipy.io import loadmat
from scipy.stats import t, wilcoxon


ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent

SHJ_DIR = ROOT / "shj_dimension_exploration"
GRID_DIR = ROOT / "gridsearch_review"
MASC_DIR = ROOT / "MASC"
SONG_DIR = ROOT / "song_task1_preliminary_results"
SONG_SOURCE = Path(
    r"C:/Users/DELL/Desktop/humans-combine-value-learning-and-hypothesis-testing-main/data/data_all_wClickInfo.csv"
)

FIRST_FRACTION = 0.30
N_RANDOM = 2000
N_SHUFFLES = 2000
N_SONG_SHUFFLES = 1000
RNG_SEED = 20260920

DATASET_COLORS = {
    "SHJ": "#4C78A8",
    "2D exploration": "#2A9D8F",
    "MASC": "#E07A5F",
    "Build-an-Icon": "#8064A2",
}

GROUP_ORDER = [
    ("SHJ", "SHJ"),
    ("2D exploration", "沿坐标轴方向探索"),
    ("2D exploration", "Local grouping"),
    ("MASC", "MASC"),
    ("Build-an-Icon", "Build-an-Icon 1D"),
    ("Build-an-Icon", "Build-an-Icon 2D"),
    ("Build-an-Icon", "Build-an-Icon 3D"),
]


def p_to_stars(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "n.s."
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def mean_true_run_length(sequence: np.ndarray) -> float:
    lengths: list[int] = []
    current = 0
    for value in sequence:
        if bool(value):
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return float(np.mean(lengths)) if lengths else 0.0


def mean_label_run_length(sequence: np.ndarray) -> float:
    if len(sequence) == 0:
        return np.nan
    changes = np.concatenate(([True], sequence[1:] != sequence[:-1]))
    starts = np.flatnonzero(changes)
    ends = np.concatenate((starts[1:], [len(sequence)]))
    return float(np.mean(ends - starts))


def first_fraction_trials(
    frame: pd.DataFrame, *, group_cols: list[str], trial_col: str = "trial_index"
) -> pd.DataFrame:
    keys = group_cols + [trial_col]
    order = frame[keys].drop_duplicates().sort_values(keys).copy()
    order["trial_order"] = order.groupby(group_cols, observed=True).cumcount() + 1
    order["n_trials"] = order.groupby(group_cols, observed=True)[trial_col].transform("count")
    order["cutoff"] = np.ceil(order["n_trials"] * FIRST_FRACTION).astype(int)
    keep = order.loc[order["trial_order"] <= order["cutoff"], keys]
    return frame.merge(keep, on=keys, how="inner")


def wilcoxon_greater(difference: pd.Series) -> float:
    values = difference.dropna().to_numpy(dtype=float)
    if len(values) < 3 or np.allclose(values, 0):
        return np.nan
    try:
        return float(wilcoxon(values, alternative="greater").pvalue)
    except ValueError:
        return np.nan


def summarize_subject_rows(rows: list[dict]) -> pd.DataFrame:
    """Collapse repeated task conditions only after subject-level metrics exist."""
    raw = pd.DataFrame(rows)
    grouped = (
        raw.groupby(["panel", "dataset", "group", "subject_key"], as_index=False)
        [["observed", "baseline"]]
        .mean()
    )
    grouped["difference"] = grouped["observed"] - grouped["baseline"]
    return grouped


def random_max_share(n_observations: int, rng: np.random.Generator, cache: dict) -> float:
    if n_observations <= 0:
        return np.nan
    if n_observations not in cache:
        counts = rng.multinomial(
            n_observations, [1 / 3, 1 / 3, 1 / 3], size=N_RANDOM
        )
        cache[n_observations] = float(np.mean(counts.max(axis=1) / n_observations))
    return cache[n_observations]


def build_shj_rows(rng: np.random.Generator) -> tuple[list[dict], list[dict]]:
    eye = pd.read_csv(SHJ_DIR / "results" / "intermediate" / "shj_eye_long.csv")
    eye = eye.loc[eye["period"].eq("before")].copy()
    eye = first_fraction_trials(eye, group_cols=["s", "type"])

    random_cache: dict[int, float] = {}
    trial_rows = []
    for keys, group in eye.groupby(["s", "type", "trial_index"], observed=True):
        n_obs = int(round(group["eye_obs"].sum()))
        observed = float(group["eye_pri_norm"].max()) if n_obs > 0 else np.nan
        baseline = random_max_share(n_obs, rng, random_cache)
        trial_rows.append(
            {
                "s": int(keys[0]),
                "type": int(keys[1]),
                "trial_index": int(keys[2]),
                "observed": observed,
                "baseline": baseline,
            }
        )

    trial = pd.DataFrame(trial_rows).dropna(subset=["observed", "baseline"])
    subject_type = (
        trial.groupby(["s", "type"], as_index=False)[["observed", "baseline"]].mean()
    )
    subject = subject_type.groupby("s", as_index=False)[["observed", "baseline"]].mean()
    b_rows = [
        {
            "panel": "b",
            "dataset": "SHJ",
            "group": "SHJ",
            "subject_key": f"S{int(row.s)}",
            "observed": float(row.observed),
            "baseline": float(row.baseline),
        }
        for row in subject.itertuples(index=False)
    ]

    dominant = pd.read_csv(
        SHJ_DIR / "results" / "intermediate" / "shj_dominant_dimensions.csv"
    )
    dominant = dominant.loc[dominant["period"].eq("before")].copy()
    dominant = first_fraction_trials(dominant, group_cols=["s", "type"])
    run_rows = []
    for keys, group in dominant.groupby(["s", "type"], observed=True):
        sequence = (
            group.sort_values("trial_index")
            .loc[group.sort_values("trial_index")["focused"] & group.sort_values("trial_index")["dominant_dim"].notna(), "dominant_dim"]
            .to_numpy(dtype=int)
        )
        if len(sequence) < 2:
            continue
        shuffled = [
            mean_label_run_length(rng.permutation(sequence))
            for _ in range(N_SHUFFLES)
        ]
        run_rows.append(
            {
                "s": int(keys[0]),
                "type": int(keys[1]),
                "observed": mean_label_run_length(sequence),
                "baseline": float(np.mean(shuffled)),
            }
        )
    run_df = pd.DataFrame(run_rows)
    subject_run = run_df.groupby("s", as_index=False)[["observed", "baseline"]].mean()
    c_rows = [
        {
            "panel": "c",
            "dataset": "SHJ",
            "group": "SHJ",
            "subject_key": f"S{int(row.s)}",
            "observed": float(row.observed),
            "baseline": float(row.baseline),
        }
        for row in subject_run.itertuples(index=False)
    ]
    return b_rows, c_rows


def scenario_short(value: str) -> str:
    value = str(value)
    if value.startswith("Avg"):
        return "Avg.R"
    if value.startswith("Max"):
        return "Max.R"
    return value


def grid_random_occurrence(
    n_steps: int,
    category: str,
    rng: np.random.Generator,
    cache: dict[tuple[int, str], float],
) -> float:
    if n_steps <= 0:
        return np.nan
    key = (n_steps, category)
    if key in cache:
        return cache[key]
    coords = rng.integers(0, 11, size=(N_RANDOM, n_steps + 1, 2))
    diffs = np.diff(coords, axis=1)
    nonrepeat = np.logical_or(diffs[:, :, 0] != 0, diffs[:, :, 1] != 0)
    dx, dy = diffs[:, :, 0], diffs[:, :, 1]
    if category == "axis":
        flags = ((dx != 0) & (dy == 0)) | ((dx == 0) & (dy != 0))
    else:
        flags = np.abs(dx) + np.abs(dy) <= 2
    rates = np.divide(
        (flags & nonrepeat).sum(axis=1),
        nonrepeat.sum(axis=1),
        out=np.full(N_RANDOM, np.nan, dtype=float),
        where=nonrepeat.sum(axis=1) > 0,
    )
    cache[key] = float(np.nanmean(rates))
    return cache[key]


def grid_category_flags(dx: np.ndarray, dy: np.ndarray, category: str) -> np.ndarray:
    if category == "axis":
        return ((dx != 0) & (dy == 0)) | ((dx == 0) & (dy != 0))
    return np.abs(dx) + np.abs(dy) <= 2


def grid_round_rows(
    data: pd.DataFrame,
    source_dataset: str,
    rng: np.random.Generator,
    random_cache: dict[tuple[int, str], float],
) -> list[dict]:
    data = data.sort_values(["subject", "round", "trial"]).copy()
    data["scenario_short"] = data["scenario"].map(scenario_short)
    group_cols = ["row_index", "subject", "scenario_short", "horizon", "round"]
    if "kernel" in data.columns:
        group_cols.append("kernel")

    rows = []
    for keys, group in data.groupby(group_cols, sort=False, observed=True):
        group = group.sort_values("trial")
        coords = group[["x", "y"]].to_numpy(dtype=float)
        diffs = np.diff(coords, axis=0)
        steps = group["trial"].to_numpy(dtype=int)[1:] - 1
        horizon = int(group["horizon"].iloc[0])
        in_phase = steps <= horizon * FIRST_FRACTION
        diffs = diffs[in_phase]
        diffs = diffs[np.logical_or(diffs[:, 0] != 0, diffs[:, 1] != 0)]
        dx, dy = diffs[:, 0], diffs[:, 1]
        key_map = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        for category in ["axis", "local"]:
            flags = grid_category_flags(dx, dy, category)
            observed_occurrence = float(flags.mean()) if len(flags) else np.nan
            shuffled_runs = []
            for _ in range(N_SHUFFLES):
                shuffled_runs.append(mean_true_run_length(rng.permutation(flags)))
            rows.append(
                {
                    "source_dataset": source_dataset,
                    "subject": int(key_map["subject"]),
                    "category": category,
                    "observed_occurrence": observed_occurrence,
                    "random_occurrence": grid_random_occurrence(
                        len(flags), category, rng, random_cache
                    ),
                    "observed_run_length": mean_true_run_length(flags),
                    "shuffled_run_length": float(np.mean(shuffled_runs))
                    if shuffled_runs
                    else np.nan,
                }
            )
    return rows


def build_grid_rows(rng: np.random.Generator) -> tuple[list[dict], list[dict]]:
    random_cache: dict[tuple[int, str], float] = {}
    round_rows = []
    for source_dataset, filename in [
        ("experiment2d", "tidy_experimentData2D.csv"),
        ("exp3", "tidy_exp3.csv"),
    ]:
        data = pd.read_csv(GRID_DIR / filename)
        round_rows.extend(grid_round_rows(data, source_dataset, rng, random_cache))

    round_df = pd.DataFrame(round_rows)
    subject = (
        round_df.groupby(["source_dataset", "subject", "category"], as_index=False)
        [[
            "observed_occurrence",
            "random_occurrence",
            "observed_run_length",
            "shuffled_run_length",
        ]]
        .mean()
    )
    b_rows = []
    c_rows = []
    for row in subject.itertuples(index=False):
        group = "沿坐标轴方向探索" if row.category == "axis" else "Local grouping"
        common = {
            "dataset": "2D exploration",
            "group": group,
            "subject_key": f"{row.source_dataset}:S{int(row.subject)}",
        }
        b_rows.append(
            {
                "panel": "b",
                **common,
                "observed": float(row.observed_occurrence),
                "baseline": float(row.random_occurrence),
            }
        )
        c_rows.append(
            {
                "panel": "c",
                **common,
                "observed": float(row.observed_run_length),
                "baseline": float(row.shuffled_run_length),
            }
        )
    return b_rows, c_rows


def load_masc_module():
    path = MASC_DIR / "analysis" / "analyze_masc_attention.py"
    spec = spec_from_file_location("masc_attention_analysis", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def random_masc_max_share(n_fix: int, rng: np.random.Generator, cache: dict[int, float]) -> float:
    if n_fix <= 0:
        return np.nan
    if n_fix not in cache:
        dims = rng.integers(0, 3, size=(N_RANDOM, n_fix))
        counts = np.stack([(dims == dim).sum(axis=1) for dim in range(3)], axis=1)
        cache[n_fix] = float(np.mean(counts.max(axis=1) / n_fix))
    return cache[n_fix]


def build_masc_rows(rng: np.random.Generator) -> tuple[list[dict], list[dict]]:
    module = load_masc_module()
    random_cache: dict[int, float] = {}
    b_trial_rows = []
    for domain in module.DOMAINS:
        loaded = module.load_domain(domain)
        all_fix = loaded["sumStats"]["allFix"]
        n_trials, n_subjects = loaded["difficulty"].shape
        first_n = int(np.ceil(n_trials * FIRST_FRACTION))
        for subject_idx in range(n_subjects):
            for trial_idx in range(first_n):
                sequence = module.clean_seq(all_fix, trial_idx, subject_idx)
                if len(sequence) == 0:
                    continue
                dims = module.aoi_to_dim(sequence) - 1
                counts = np.bincount(dims, minlength=3)
                b_trial_rows.append(
                    {
                        "subject": subject_idx + 1,
                        "domain": domain,
                        "observed": float(counts.max() / len(dims)),
                        "baseline": random_masc_max_share(len(dims), rng, random_cache),
                    }
                )

    b_trial = pd.DataFrame(b_trial_rows)
    b_subject = b_trial.groupby("subject", as_index=False)[["observed", "baseline"]].mean()
    b_rows = [
        {
            "panel": "b",
            "dataset": "MASC",
            "group": "MASC",
            "subject_key": f"S{int(row.subject)}",
            "observed": float(row.observed),
            "baseline": float(row.baseline),
        }
        for row in b_subject.itertuples(index=False)
    ]

    trial_metrics = pd.read_csv(MASC_DIR / "analysis" / "tables" / "trial_level_metrics.csv")
    trial_metrics = trial_metrics.loc[trial_metrics["trial_index"] <= 36]
    c_subject = trial_metrics.groupby("subject", as_index=False)[
        ["mean_run_length", "shuffled_mean_run_length"]
    ].mean()
    c_rows = [
        {
            "panel": "c",
            "dataset": "MASC",
            "group": "MASC",
            "subject_key": f"S{int(row.subject)}",
            "observed": float(row.mean_run_length),
            "baseline": float(row.shuffled_mean_run_length),
        }
        for row in c_subject.itertuples(index=False)
    ]
    return b_rows, c_rows


def expected_random_subset_max_share(n_relevant: int) -> float:
    if n_relevant <= 0:
        return np.nan
    denominator = 2**n_relevant - 1
    return float(
        sum(
            comb(n_relevant, k) / denominator / k
            for k in range(1, n_relevant + 1)
        )
    )


def add_song_flags(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    dims = ["color", "shape", "pattern"]
    selected_cols = [f"selectedFeature_{dim}" for dim in dims]
    random_cols = [f"randomlySelectedFeature_{dim}" for dim in dims]
    out["valid_behavior_trial"] = out["rt"].notna() & out["numSelectedFeatures"].notna()
    out["n_selected_dims"] = out[selected_cols].notna().sum(axis=1)
    out["n_random_dims"] = out[random_cols].notna().sum(axis=1)
    out["one_dim_randomselect"] = (
        out["valid_behavior_trial"]
        & out["numSelectedFeatures"].eq(1)
        & out["n_selected_dims"].eq(1)
        & out["n_random_dims"].eq(2)
    )
    out["single_selected_dim"] = pd.NA
    out["single_selected_feature"] = pd.NA
    for dim in dims:
        mask = out["one_dim_randomselect"] & out[f"selectedFeature_{dim}"].notna()
        out.loc[mask, "single_selected_dim"] = dim
        out.loc[mask, "single_selected_feature"] = out.loc[
            mask, f"selectedFeature_{dim}"
        ]

    ordered = out.sort_values(["workerId", "game", "trial"]).copy()
    group_break = ordered[["workerId", "game"]].ne(
        ordered[["workerId", "game"]].shift()
    ).any(axis=1)
    previous_one_dim = ordered["one_dim_randomselect"].shift(fill_value=False)
    selected_dim = ordered["single_selected_dim"].fillna("__none__")
    dim_switch = selected_dim.ne(selected_dim.shift(fill_value="__none__"))
    new_bout = ordered["one_dim_randomselect"] & (
        group_break | ~previous_one_dim | dim_switch
    )
    bout_id = new_bout.cumsum()
    ordered["one_dim_bout_id"] = np.where(
        ordered["one_dim_randomselect"], bout_id, np.nan
    )
    ordered["bout_feature_nunique"] = np.nan
    mask = ordered["one_dim_randomselect"]
    if mask.any():
        bouts = ordered.loc[mask].groupby(
            ["workerId", "game", "one_dim_bout_id"], sort=False
        )
        ordered.loc[mask, "bout_feature_nunique"] = bouts[
            "single_selected_feature"
        ].transform("nunique")
    ordered["one_dim_exploration"] = ordered["one_dim_randomselect"] & ordered[
        "bout_feature_nunique"
    ].gt(1)
    return ordered


def build_song_rows(rng: np.random.Generator) -> tuple[list[dict], list[dict]]:
    data = add_song_flags(pd.read_csv(SONG_SOURCE))
    data = data.loc[data["trial"].between(1, 9)].copy()

    b_rows = []
    for (worker, game), group in data.groupby(["workerId", "game"], sort=True):
        valid = group.loc[group["valid_behavior_trial"]].copy()
        if valid.empty:
            continue
        rd = int(valid["numRelevantDimensions"].iloc[0])
        valid = valid.loc[valid["n_selected_dims"] > 0]
        if valid.empty:
            continue
        observed = float((1 / valid["n_selected_dims"]).mean())
        baseline = expected_random_subset_max_share(rd)
        b_rows.append(
            {
                "panel": "b",
                "dataset": "Build-an-Icon",
                "group": f"Build-an-Icon {rd}D",
                "subject_key": f"W{int(worker)}:G{int(game)}",
                "observed": observed,
                "baseline": baseline,
            }
        )

    c_rows = []
    for (worker, game), group in data.groupby(["workerId", "game"], sort=True):
        valid = group.loc[group["valid_behavior_trial"]].sort_values("trial")
        if len(valid) < 2:
            continue
        rd = int(valid["numRelevantDimensions"].iloc[0])
        sequence = valid["one_dim_exploration"].to_numpy(dtype=bool)
        shuffled = [
            mean_true_run_length(rng.permutation(sequence))
            for _ in range(N_SONG_SHUFFLES)
        ]
        c_rows.append(
            {
                "panel": "c",
                "dataset": "Build-an-Icon",
                "group": f"Build-an-Icon {rd}D",
                "subject_key": f"W{int(worker)}:G{int(game)}",
                "observed": mean_true_run_length(sequence),
                "baseline": float(np.mean(shuffled)),
            }
        )
    return b_rows, c_rows


def make_stats(subject_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for panel in ["b", "c"]:
        for dataset, group in GROUP_ORDER:
            group_rows = subject_rows.loc[
                (subject_rows["panel"] == panel)
                & (subject_rows["dataset"] == dataset)
                & (subject_rows["group"] == group)
            ]
            diff = group_rows["difference"].dropna()
            n = len(diff)
            mean_diff = float(diff.mean()) if n else np.nan
            sem = float(diff.sem()) if n > 1 else np.nan
            ci = float(t.ppf(0.975, n - 1) * sem) if n > 1 and np.isfinite(sem) else np.nan
            p_value = wilcoxon_greater(diff)
            rows.append(
                {
                    "panel": panel,
                    "dataset": dataset,
                    "group": group,
                    "n_subject_units": n,
                    "mean_observed": float(group_rows["observed"].mean()) if n else np.nan,
                    "mean_baseline": float(group_rows["baseline"].mean()) if n else np.nan,
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
    ax.grid(False)
    ax.tick_params(axis="both", labelsize=11, length=4)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))


def plot_panel(
    subject_rows: pd.DataFrame,
    stats_table: pd.DataFrame,
    panel: str,
    output_path: Path,
    *,
    combined: bool = False,
) -> None:
    panel_rows = subject_rows.loc[subject_rows["panel"] == panel].copy()
    panel_stats = stats_table.loc[stats_table["panel"] == panel].copy()
    fig, ax = plt.subplots(figsize=(12.4, 5.2), dpi=320)
    positions = np.arange(1, len(GROUP_ORDER) + 1)
    rng = np.random.default_rng(RNG_SEED + (11 if panel == "b" else 17))

    for position, (dataset, group) in zip(positions, GROUP_ORDER):
        values = panel_rows.loc[
            (panel_rows["dataset"] == dataset) & (panel_rows["group"] == group),
            "difference",
        ].dropna()
        color = DATASET_COLORS[dataset]
        if len(values):
            jitter = rng.uniform(-0.10, 0.10, size=len(values))
            ax.scatter(
                np.full(len(values), position) + jitter,
                values,
                s=19,
                color=color,
                alpha=0.32,
                linewidths=0,
                zorder=2,
            )
            mean = float(values.mean())
            sem = float(values.sem()) if len(values) > 1 else np.nan
            ci = float(t.ppf(0.975, len(values) - 1) * sem) if len(values) > 1 else 0.0
            ax.errorbar(
                position,
                mean,
                yerr=ci,
                fmt="o",
                color=color,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.8,
                markersize=7,
                capsize=3,
                elinewidth=1.5,
                zorder=4,
            )
            # Significance labels are retained in the combined main figure
            # and in bc_stats.csv; the standalone panel stays uncluttered.

    ax.axhline(0, color="#777777", linewidth=1.0, linestyle=(0, (2, 2)), zorder=1)
    labels = [
        "SHJ",
        "Along\ncoordinate axes",
        "Local\ngrouping",
        "MASC",
        "Build\n1D",
        "Build\n2D",
        "Build\n3D",
    ]
    ax.set_xticks(positions, labels)
    ax.set_xlim(0.55, len(GROUP_ORDER) + 0.45)
    ax.spines["bottom"].set_bounds(0.8, len(GROUP_ORDER) + 0.2)
    if panel == "b":
        ax.set_ylabel("Observed − random baseline", fontsize=12)
        panel_title = "Frequency above random baseline"
    else:
        ax.set_ylabel("Observed − shuffled baseline", fontsize=12)
        panel_title = "Run length above shuffled baseline"
    ax.set_xlabel("Task / exploration pattern", fontsize=12, labelpad=8)
    style_axis(ax)

    handles = [
        Patch(facecolor=color, edgecolor="none", label=dataset)
        for dataset, color in DATASET_COLORS.items()
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=4,
        fontsize=10,
    )
    ax.tick_params(axis="x", labelsize=10)
    fig.suptitle(panel_title, fontsize=14, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_combined(subject_rows: pd.DataFrame, stats_table: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 5.4), dpi=320, sharex=True)
    rng = np.random.default_rng(RNG_SEED + 31)
    positions = np.arange(1, len(GROUP_ORDER) + 1)
    labels = [
        "SHJ",
        "Along\ncoordinate axes",
        "Local\ngrouping",
        "MASC",
        "Build\n1D",
        "Build\n2D",
        "Build\n3D",
    ]
    for ax, panel, letter, title, ylabel in zip(
        axes,
        ["b", "c"],
        ["B", "C"],
        ["Frequency above random baseline", "Run length above shuffled baseline"],
        ["Observed − random baseline", "Observed − shuffled baseline"],
    ):
        panel_rows = subject_rows.loc[subject_rows["panel"] == panel]
        panel_stats = stats_table.loc[stats_table["panel"] == panel]
        for position, (dataset, group) in zip(positions, GROUP_ORDER):
            values = panel_rows.loc[
                (panel_rows["dataset"] == dataset) & (panel_rows["group"] == group),
                "difference",
            ].dropna()
            if not len(values):
                continue
            color = DATASET_COLORS[dataset]
            jitter = rng.uniform(-0.09, 0.09, size=len(values))
            ax.scatter(
                np.full(len(values), position) + jitter,
                values,
                s=16,
                color=color,
                alpha=0.30,
                linewidths=0,
                zorder=2,
            )
            mean = float(values.mean())
            sem = float(values.sem()) if len(values) > 1 else np.nan
            ci = float(t.ppf(0.975, len(values) - 1) * sem) if len(values) > 1 else 0.0
            ax.errorbar(
                position,
                mean,
                yerr=ci,
                fmt="o",
                color=color,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.8,
                markersize=6.5,
                capsize=2.5,
                elinewidth=1.4,
                zorder=4,
            )
            stat = panel_stats.loc[panel_stats["group"] == group]
            if not stat.empty:
                y_range = max(float(values.max() - values.min()), 0.05)
                ax.text(
                    position,
                    float(values.max()) + 0.08 * y_range,
                    str(stat["stars"].iloc[0]),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    color=color,
                )
        ax.axhline(0, color="#777777", linewidth=0.9, linestyle=(0, (2, 2)), zorder=1)
        ax.set_title(title, fontsize=13, pad=14)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xticks(positions, labels)
        ax.set_xlim(0.55, len(GROUP_ORDER) + 0.45)
        ax.spines["bottom"].set_bounds(0.8, len(GROUP_ORDER) + 0.2)
        style_axis(ax)
        ax.tick_params(axis="x", labelsize=10)
        ax.text(-0.10, 1.04, letter, transform=ax.transAxes, fontsize=15, fontweight="bold")
    axes[0].set_xlabel("Task / exploration pattern", fontsize=11, labelpad=8)
    axes[1].set_xlabel("Task / exploration pattern", fontsize=11, labelpad=8)
    handles = [
        Patch(facecolor=color, edgecolor="none", label=dataset)
        for dataset, color in DATASET_COLORS.items()
    ]
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.19, top=0.82, wspace=0.23)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def write_readme() -> None:
    text = """# Main-figure panels B and C

Generated by `build_bc_main_figure.py`. Only this folder is written by the
script; the four source analysis folders are read-only inputs.

## Common plotting convention

- All analyses use the first 30% of trials/search steps.
- Each point is one subject-level unit (or one worker-game unit for
  Build-an-Icon, matching its existing persistence analysis).
- Panel B shows observed frequency/concentration minus a random-generated
  baseline. Panel C shows observed mean run length minus a shuffled baseline.
- Error bars are 95% confidence intervals around the mean difference.
- Stars are one-sided Wilcoxon signed-rank tests of the subject-level
  differences against zero; exact p values are in `bc_stats.csv`.

## Dataset-specific definitions

- SHJ: before trials only; types 1, 2, 4, and 6 are calculated separately and
  then averaged within subject. B is the mean within-trial attention share of
  the primary observed dimension (`max eye_pri_norm`). Its random baseline
  assigns the same number of eye observations in each trial uniformly to the
  three dimensions. C is the mean run length of consecutive focused dominant
  dimensions, with the focused sequence shuffled within subject-type.
- 2D exploration: Experiment 2D and Experiment 3 are pooled as one task.
  `Along coordinate axes` is the union of strict x-axis and y-axis moves and
  is recomputed as one sequence; `Local grouping` is Manhattan distance <= 2.
  B uses random paths on the 0..10 grid with matched active-step counts. C
  shuffles the binary category sequence within each round.
- MASC: Phone and Hotel, all difficulty levels, and all dimensions are pooled.
  B is the mean within-trial fixation share of the most fixated dimension;
  random AOI sequences with matched fixation counts provide the baseline. C
  uses the existing trial-level dimension run length and its within-trial AOI
  shuffle baseline, restricted to trials 1-36.
- Build-an-Icon: known and unknown trials are pooled within 1D/2D/3D
  relevant-dimension groups. B is the mean share of the primary selected
  dimension (`1 / n_selected_dims`); the random baseline samples a nonempty
  subset of relevant dimensions uniformly. C uses the existing binary
  feature-switching one-dimensional exploration sequence and its 1000-shuffle
  run-length baseline.

The two panel-specific PNGs and a combined `figure_bc_main.png` are exported
at 320 dpi with a white background.
"""
    (OUT / "README_bc_main_figure.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    shj_b, shj_c = build_shj_rows(rng)
    grid_b, grid_c = build_grid_rows(rng)
    masc_b, masc_c = build_masc_rows(rng)
    song_b, song_c = build_song_rows(rng)

    subject_rows = summarize_subject_rows(
        shj_b + grid_b + masc_b + song_b + shj_c + grid_c + masc_c + song_c
    )
    stats_table = make_stats(subject_rows)

    subject_rows.to_csv(OUT / "bc_subject_level.csv", index=False)
    stats_table.to_csv(OUT / "bc_stats.csv", index=False)
    plot_panel(subject_rows, stats_table, "b", OUT / "figure_b_frequency_above_random.png")
    plot_panel(subject_rows, stats_table, "c", OUT / "figure_c_run_length_above_shuffled.png")
    plot_combined(subject_rows, stats_table, OUT / "figure_bc_main.png")
    write_readme()

    print("Wrote:")
    for path in [
        OUT / "figure_b_frequency_above_random.png",
        OUT / "figure_c_run_length_above_shuffled.png",
        OUT / "figure_bc_main.png",
        OUT / "bc_subject_level.csv",
        OUT / "bc_stats.csv",
        OUT / "README_bc_main_figure.md",
    ]:
        print(path)
    print("\nStats:")
    print(stats_table.to_string(index=False))


if __name__ == "__main__":
    main()
