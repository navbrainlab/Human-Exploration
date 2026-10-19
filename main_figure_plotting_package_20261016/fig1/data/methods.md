## Display refinement, 2026-10-14

Only panel a's schematic feedback layout changes: the score-strip height decreases from75 to37 units, the selection frame grows from127×113 to148×149, and each displayed bundle grows from30 to40 units. The same nine bundle identities, their order, feature assets, row rewards and score78 are retained; feature-space zoom connectors attach to the enlarged frame. The outer feedback box and all scientific result pixels are unchanged. A complete current package/PNG snapshot is preserved locally. This renderer performs no new statistical analysis, fitting or simulation.

## Current display convention

Main x/y spine endpoints coincide with first/last displayed major ticks. The c-right box plot adds a zero tick to cover its smallest observed trials. Task 2's categorical x viewport shifts from (-0.5,1.5) to (-0.1202060932,1.8797939068), retaining the span of 2, category positions 0.2/0.8, physical box widths, y coordinates and statistics. Its boxes/ticks/bracket move left together; no numeric observations or group definitions change. The visible box-to-y-spine/ylabel gaps are 3.5/8.512972 mm. Task 1 box margins are retained. All source sample eligibility, uncertainty definitions and comparisons remain as documented in the original caption. The native renderer verifies exact pre-revision pixels before adjusting the display and preserves every scientific artist coordinate.

# Fig. 1 display revision

Scientific definitions, eligible participants, score smoothing, completion filling, the personal-best endpoint, box/whisker conventions and original Mann–Whitney comparisons are retained from `caption.md` and `source/original_results_caption.md`. No estimation or hypothesis test is recomputed.

The native frozen-data reconstruction uses `source/native_reconstruction/code/render_fig1_compact.py`. Its unmodified lower region, with the parent's already-approved e-right ylabel and y ticks, must reproduce `source/results_axis_parent.png` pixel-for-pixel. Only the e-right bottom-spine bounds then change to the first and last category ticks (Obs/NObs). Its data coordinates, view limits, tick positions and labels remain fixed. This matches the existing c-right category-axis presentation. The extended data viewport is retained around the boxes; the main spine stops at its category ticks.

The figure is assembled at native 600 dpi with the approved upper source and the newly rendered lower results. The upper region and all lower content outside the e-right x-spine change remain identical. Font classes are 5/6/8 pt; the copied source drawing code and CSVs permit a portable native rebuild. The parent image and full package are preserved, with hashes in `provenance/current_parent_snapshot.json`.

### Display update, 15 October 2026

Learning-curve progress is (trial − 1)/(scheduled horizon − 1), separately for each condition (30/50/50/60/60 trials); no resampling, new smoothing or statistical tests. Personal-best box plots retain absolute trial counts.
