#!/usr/bin/env python3
"""Freeze the full reference package's f/i/g/j endpoints from plotting inputs.

Personal best is the earliest observed trial at each participant's own maximum,
including non-completers. This is distinct from the first-full-score endpoint.
No raw identifying information, model fitting or simulations are required.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "csv"
GROUPS = {1: ["P1", "P2", "P2-only"], 2: ["FDS-Obs", "FDS-NoObs"]}
INPUTS = ["fig1_task1_personal_best.csv", "fig1_task1_learning_extended.csv",
          "fig1_task2_observed_scores.csv", "fig1_task1_perfect_rate.csv",
          "fig1_task2_perfect_rate.csv"]


def calculate():
    t1 = pd.read_csv(CSV / INPUTS[0], dtype={"subject": str})
    extended = pd.read_csv(CSV / INPUTS[1], dtype={"subject": str})
    rows = []
    for row in t1.itertuples(index=False):
        # The frozen learning input only extends missing trials after full score;
        # neither that first occurrence nor a non-completer's maximum is changed.
        part = extended[extended.phase.eq(row.phase) & extended.subject.eq(row.subject)]
        assert part.score.max() == row.best_score
        assert part.loc[part.score.eq(row.best_score), "round"].min() == row.first_best_trial
        rows.append(dict(task=1, group=row.phase, subject=row.subject,
                         best_score=row.best_score, first_best_trial=int(row.first_best_trial),
                         completed=row.best_score == 100, horizon=30 if row.phase == "P1" else 50))
    t2 = pd.read_csv(CSV / INPUTS[2], dtype={"subject_id": str})
    assert len(t2) == 4111 and t2.trial.between(1, 60).all()
    assert not t2.duplicated(["subject_id", "trial"]).any()
    for (group, subject), part in t2.groupby(["plot_group", "subject_id"], sort=True):
        best = part.score_noisy.max()
        rows.append(dict(task=2, group=group, subject=subject, best_score=float(best),
                         first_best_trial=int(part.loc[part.score_noisy.eq(best), "trial"].min()),
                         completed=best == 100, horizon=60))
    personal = pd.DataFrame(rows).sort_values(["task", "group", "subject"]).reset_index(drop=True)
    assert len(personal) == 236 and not personal.duplicated(["task", "group", "subject"]).any()
    assert personal.first_best_trial.between(1, personal.horizon).all()
    expected_counts = {1: [51, 51, 54], 2: [40, 40]}
    expected_successes = {1: [17, 32, 22], 2: [18, 12]}
    for task in [1, 2]:
        rates = pd.read_csv(CSV / f"fig1_task{task}_perfect_rate.csv")
        for i, group in enumerate(GROUPS[task]):
            part = personal[personal.task.eq(task) & personal.group.eq(group)]
            assert len(part) == expected_counts[task][i] == rates.iloc[i]["n"]
            assert part.completed.sum() == expected_successes[task][i] == rates.iloc[i].n_perfect
            assert np.isclose(part.completed.mean(), rates.iloc[i].rate)
    tests = {}
    for panel, task, groups in [("f", 1, ["P2", "P2-only"]), ("i", 2, GROUPS[2])]:
        parts = [personal[personal.task.eq(task) & personal.group.eq(g)] for g in groups]
        table = [[int(part.completed.sum()), int((~part.completed).sum())] for part in parts]
        result = fisher_exact(table, alternative="two-sided")
        tests[panel] = dict(test="Fisher exact", groups=groups, contingency_table=table,
                            odds_ratio=float(result.statistic), p_raw=float(result.pvalue),
                            displayed=panel == "f", alternative="two-sided", adjustment="none")
    for panel, task, groups in [("g", 1, ["P2", "P2-only"]), ("j", 2, GROUPS[2])]:
        values = [personal.loc[personal.task.eq(task) & personal.group.eq(g), "first_best_trial"]
                  for g in groups]
        result = mannwhitneyu(*values, alternative="two-sided", method="auto")
        tests[panel] = dict(test="Mann–Whitney U", groups=groups, n=[len(v) for v in values],
                            U=float(result.statistic), p_raw=float(result.pvalue), displayed=True,
                            alternative="two-sided", adjustment="none",
                            method="SciPy auto: asymptotic tie-corrected with continuity correction")
    assert np.allclose([tests[p]["p_raw"] for p in ["f", "g", "i", "j"]],
                       [.031850093396471726, .020714828307657238,
                        .24806112143075693, .045259782513881706])
    stats = dict(endpoint="First observed trial attaining the participant's own window maximum",
                 excludes_noncompleters=False, counts=expected_counts, fullscore_counts=expected_successes,
                 trials="Observed Task 1 scores; observed Task 2 score_noisy; no normalized coordinates",
                 reference="reference/figure1_figure2_files_full/figure1/code/figure1_panels_ej_optimized.py",
                 interpretation="Reference visualizations include their personal-best endpoint and selected unadjusted tests",
                 tests=tests, input_sha256={name: hashlib.sha256((CSV/name).read_bytes()).hexdigest()
                                           for name in INPUTS})
    return personal, stats


def build():
    personal, stats = calculate()
    personal.to_csv(CSV / "fig1_reference_personal_best.csv", index=False)
    (CSV / "fig1_reference_panel_statistics.json").write_text(json.dumps(stats, indent=2) + "\n")
    print("Reference personal-best records:", len(personal), "— no non-completers excluded")
    for panel, test in stats["tests"].items():
        print(panel, test["test"], "P =", test["p_raw"], "displayed:", test["displayed"])


if __name__ == "__main__":
    build()
