# Task 1 enrichment analysis package

This package contains only analyses comparing information gain across enrichment levels. Trial-series, experiment-progress-bin, and action-total-over-time analyses are excluded.

## Included analyses

### All-pattern analysis

Each trial–dimension receives its own enrichment value. Its dimension-level correct information gain is assigned to that enrichment level.

### Primary-only analysis (`只归入`)

Each action is classified by its maximum enrichment. Only dimensions having that maximum enrichment are included. Tied maximum dimensions are averaged within the action.

### All-dimensions-assigned analysis (`只算`)

Each action is classified by its maximum enrichment. All dimension-level information values are averaged within the action and assigned to the maximum enrichment.

## Enrichment definition

Each dimension has nine feature values arranged as three rows of three. For each row, take the largest repeated-feature count, then sum the three row maxima. Possible enrichment values are `3, 5, 6, 7, 9`.

## Contents

- `data/`: source behavioral data.
- `code/`: final 3D/4D model, enrichment summarization, and plotting scripts.
- `results/3d/`: 3D dimension-level, primary-only, and all-dimensions-assigned CSV/PNG/SVG results.
- `results/4d/`: 4D dimension-level, primary-only, all-dimensions-assigned, and five-metric enrichment results.

No trial-by-trial or experimental-progress plots are included.
