"""Cache participant-smoothed Fig4d scores, matching the Fig1e display rule.

Uses the frozen extended table, including its original perfect-score filling.
No tests, fits, simulated observations, or source-table mutations are made.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "csv/04_06_extended_learning_data.csv"
DISPLAY = "analysis/d_learning_display_5trial.csv"
METADATA = "analysis/d_learning_display_5trial.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    expected = json.loads((ROOT / "input_manifest.json").read_text())[SOURCE]
    assert sha(ROOT / SOURCE) == expected
    scores = pd.read_csv(ROOT / SOURCE).sort_values(["condition", "participant", "trial_id"])
    assert len(scores) == 4800 and scores.participant.nunique() == 80
    assert not scores.duplicated(["condition", "participant", "trial_id"]).any()
    scores["smoothed_score"] = scores.groupby(["condition", "participant"])["score"].transform(
        lambda s: s.rolling(5, center=True, min_periods=1).mean())
    summary = scores.groupby(["condition", "trial_id"])["smoothed_score"].agg(
        mean="mean", sd="std", n="count").reset_index().rename(columns={"trial_id": "trial"})
    summary["sem"] = summary.sd / np.sqrt(summary.n)
    assert summary.n.eq(40).all() and len(summary) == 120
    summary[["condition", "trial", "mean", "sem", "n"]].to_csv(ROOT / DISPLAY, index=False)
    metadata = {
        "display_panel": "d", "source_analysis_id": "e", "window_trials": 5,
        "centered": True, "edge_rule": "Use available trials; min_periods=1",
        "summary": "Across-participant mean and sample SD/sqrt(n), after individual smoothing",
        "source_rows": 4800, "participants": 80, "participants_per_condition": 40,
        "source_sha256": {SOURCE: expected}, "display_sha256": {DISPLAY: sha(ROOT / DISPLAY)},
        "interpolation": "PCHIP to400 trial-grid nodes, without extrapolation",
        "reference": "Fig1e, fig1_fig2_reconstruction/code/build_outcome_panels.py",
        "tests_changed": False, "fits_changed": False,
        "source_assumption": "Original100-score carry-forward after stopping, through trial60",
    }
    (ROOT / METADATA).write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


def load_display():
    metadata = json.loads((ROOT / METADATA).read_text())
    for relative, expected in {**metadata["source_sha256"], **metadata["display_sha256"]}.items():
        assert sha(ROOT / relative) == expected, f"Fig4d display input changed: {relative}"
    return pd.read_csv(ROOT / DISPLAY), metadata


if __name__ == "__main__":
    prepare()
