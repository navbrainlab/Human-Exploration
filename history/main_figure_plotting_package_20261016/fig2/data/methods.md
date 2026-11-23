# Fig. 2 reordered layout and retained source methods, 2026-10-14

## Current public panel mapping

The layout moves the original c/e metrics to the upper-right b/c positions, original b examples to the middle-left d position, and original d sequence to the middle-right e position. Panels a/f/g/h retain their identities. Internal table names and source-axis identifiers remain frozen: public b←source c, c←source e, d←source b, e←source d. This change does not alter any data, interval definitions or tests. The caption follows the new public letters. Two upper quantitative areas are matched28.5×32mm. Final native content gaps between the four tiers are2/3/3mm.

## Scope and preserved sources

This revision follows the supplied source analyses for the quantitative panels.
The previous complete figure/package is frozen in
`../fig2_nhb_endpoint_callouts_20261014/revisions/20261012_before_source_consistency_update`. Its full-sample/Holm
analyses are preserved and are historical sensitivity results, not the current
star definitions. All 19 inherited plotting CSVs and the original sources retain
their recorded bytes. No model fitting or random trajectory generation was run.

The source-consistency audit compares the current a-h panels by content. The
supplied working package uses some different panel letters: its dimension-
identification analysis is labelled h, and its corpus analysis i.

## Current f-axis display

Only the y-axis tick density is reduced. The four f views use three equally spaced major ticks: (0,0.075,0.15), (0,0.6,1.2), (0,0.05,0.1) and (0,0.6,1.2), respectively. Numeric ranges, normalization, visible 32 mm y-spine heights, physical plot rectangles, observations, statistical summaries, random means and saved P values remain unchanged. Standard ylabel clearance follows the changed tick-label widths to avoid overlapping text. Earlier tick lists below describe historical editions.

## a: per-dimension FEL counts and visits

The source aggregate comprises 24 clusters, derived from 1,368 P1 actions by 51
participants. Source visit counts and mean normalized within-participant visit
ranks are retained. Marker area is proportional to visit count; rings identify
DIS vertices. A native linear projection displays the three per-dimension FEL
counts; palette and marker sizes are display choices. The redundant coloured strips and
miniature feature legends are removed from a. Its three full action examples
follow actual Task 1 coding: D1 dessert, D2 main dish and D3 drink, correcting
the illustrative D2/D3 ordering inherited from the source PPT.

The current coordinates are raw feature enrichment level (FEL) counts:

FEL_d = sum_r max_k n[d,k,r].

Here n[d,k,r] is the number of stimuli with feature k of dimension d in row r.
The most frequent feature is counted separately in each row; the three row
counts are then summed. FEL_d ranges from 3 (no enrichment) to 9 (a DIS choice).
Panel a displays FEL in D1/D2/D3. Its definition note expands the FEL abbreviation,
positioned below and to the right of the FEL in D2 label;
the full count definition is given in the figure caption. A short horizontal
colour key in the upper-left blank area shows the unchanged relative visit
order from Early to Late. Below it, two differently sized grey dots illustrate
choice frequency: the pooled number of observed choices in each FEL pattern,
not unique participants. A gold empty ring illustrates DIS membership.

The compact "Visit order" title is centred above the colourbar, not over the
whole legend. Colour uses the frozen `relative_visit_order` column. The
24 FEL patterns are ordered by `mean_rank_normalized_order`, which aggregates
normalized within-participant visit ranks, and their `display_order_rank`
values are rescaled as (rank - 1) / (24 - 1). This is a relative ordering
of aggregate patterns, not an individual's raw trial index or visit count.
Neither the values nor the colour normalization changes in this revision.

The Non-DIS inset reproduces the row membership of subject 2024102201,
P1 trial5, from frozen source/f_behavior/data_3d_processed.csv. Row-wise
dominant-feature counts are (2,2,1) in D1, (2,2,3) in D2 and (1,1,1) in D3,
giving FEL=(5,7,3). D1 and D2 are partially enriched; no dimension reaches9.
The inset points to the corresponding aggregate pattern (55 choices), representing all
arrangements with this FEL triplet; no new point or visit is added. Existing
Task 1 food icons are copied byte-for-byte from the Fig. 1 PPTX component
package and assembled natively. The icon identities represent stimulus
features and do not infer the participant's internal attentional state.
Source row membership, asset hashes and callout coordinates are saved in the
native manifest and assets/non_dis_example.json.

DIS is defined as dimension-isolating sampling in the figure caption. It
describes feature-homogeneous rows along one dimension in Task1, or selections
sharing a feature value along one dimension in Task2. The graphic retains the
compact DIS abbreviation in a and a shared full-name heading in d. DIS is
a behavioural classification, not direct
evidence of an unobserved attentional state. Non-DIS includes partially
enriched arrangements as well as completely mixed ones.

The inherited source CSV records squared-count energy
E_d = sum_{r,k} n[d,k,r]^2 / 3 - 3. Its bytes remain frozen. For balanced Task 1
tables, each row and feature totals three. Enumerating all 55 such 3-by-3 count
tables establishes an exact one-to-one map: energies 0, 4/3, 2, 10/3 and 6
recover raw FEL counts 3, 5, 6, 7 and 9. This recovery is not interpolated and
is not assumed to generalize to other tasks or unbalanced stimulus sets.

The displayed planar coordinates are a linear projection of these recovered
counts: xy = origin + sum_d ((FEL_d - 3) / 6) vector_d. The origin is
(37.5, 28) mm; the three vectors are (0, -19), (-16.5, 16) and (22, 2) mm in
panel-local coordinates with downward-positive y. Subtracting the minimum
count and scaling by six are display normalization only, not an energy
transform. The same projection applies to all 24 aggregate patterns; all remain
distinct. Their visits, visit-order colours and DIS membership are unchanged.
The native manifest records every FEL triplet and displayed coordinate.

## b: feature enrichment level and highest-FEL frequency

For Task 1, an action places nine stimuli into three rows of three. FEL is
max_d sum_r max_f n[d,f,r], with maximum 9. Dominant features may differ between
rows. Task 2 FEL is the largest count of a feature label among four selected
scenes, with maximum 4. Thus Task 1 b/c use the maximum across dimensions of
the same per-dimension raw FEL count displayed separately along a's axes.

Source human and frozen random trial values reproduce for all 5,323/4,111
Task 1/2 trials. Individual highest-FEL human-minus-random differences are
shown in percentage points for 105/80 participants. Source two-sided Wilcoxon
(wilcox) and within-task Benjamini–Hochberg correction across Highest/Lower
are retained. Lower is the exact negative and remains in source data without
redundant display. Highest adjusted P values are 8.77343044655385e-19 and
1.14521640300789e-14. Stars test each task against zero, not between tasks.

## c: original aggregation with explicit display transformations

Trial progress is rank-normalized, with ten bin centres 0.05-0.95. The source
1,559/798 participant-unit bin means and ten-bin summaries reproduce within
floating-point rounding. Task 1 includes 105 people and 156 participant-phases,
with 155 units in bin 4 and 156 elsewhere; Task 2 includes 79 units in bins 3/6
and 80 elsewhere. The source means and SEMs are retained. Task 1 SEM is
across participant-phase means and is descriptive, not an inferential interval
that assumes repeated phases are independent people.

The supplied progress plot's arbitrary -0.10 Task 1 display shift is removed.
Unshifted means and SEMs are then multiplied by 100/M (M=9/4 for Task 1/2).
Removing that shift corresponds to +1.111111 percentage points compared with
a shifted-and-scaled Task 1 display; it is not a change to raw trial values.
Scaling does not subtract chance or equate task floors/null distributions.
The full saved mean±SEM envelope is 69.3873-90.8246%, within the 68-93% viewport.
There is no added temporal or task-difference significance test.

## d and e: retained illustrative material

The two DIS example image areas (public d; source b) now occupy the left half of the second row. Their source pixels and aspect ratios are unchanged. Their native sofa locators retain normalized image coordinates and0.85pt outlines. The physical image width is36.8mm, within matched38×30.5mm plotting rectangles.

The sequence diagram (public e; source d) occupies the second row's right half in an84×25.5mm rectangle. It keeps the original virtual source coordinates and all187 extracted source shape records, including the81/28 selected Task1/Task2 squares, feature categories and order. Its representative dimension icons and source callouts remain. The highlighted single-dimension subrun illustrates the run concept; it does not establish eligibility under f3's distinct whole-bout rule.

Task1's original cookie, main dish and drink represent D1/D2/D3. Task2 uses native sofa, table and carpet glyphs. Each icon identifies a dimension; feature values remain encoded by shade. These are source illustrations rather than new participant data. The gray source icon stays omitted beside public e; all six internal dimension icons remain.

## f: source participant endpoints and native complete displays

All valid numerical endpoint summaries contribute to native boxplots and
arithmetic mean diamonds. Individual and outlier points are hidden at the
author's request. The earlier retained raster density crop omitted an Obs
value of 8.6 trials in f3; it remains in the complete source table, summaries
and tests, although its individual outlier marker is now hidden. Old artwork
is retained as provenance, not used as a substitute for observations.

The current f viewports are 0–0.15, 0–1.2, 0–5 and 0–6, respectively.
The f3 viewport follows the displayed whiskers and means, rather than the
hidden individual maximum of 8.6. This display choice does not exclude any
valid endpoint from quartiles, means or tests. The f4 axis explicitly shows
the upper endpoint of 6; the empirical dimensional maximum remains 5.

Task 1 f1 is the mean of maximal consecutive same-dimension DIS runs within
a participant-phase, divided by 30/50 trials for 3D/4D. f2 averages the number
of distinct dimensions in each uninterrupted DIS bout, divided by 3/4.
Switching dimension ends a run but need not end a bout; a non-DIS trial ends
both. Source groups contain 51/51/54 participant-phases. One 3D-NE phase has
no DIS and undefined endpoint means, leaving 50/51/54 valid values.

Task 2 f3 retains only whole uninterrupted DIS bouts confined throughout to
one dimension and averages their lengths within each participant. A bout
that changes dimension is omitted, not split into subruns. Five/two Obs/NObs
participants have no eligible bout, leaving 35/38 of 40/40. f4 counts distinct
dimensions selected through DIS across the entire game, including all 40/40
participants and zeros. These tasks' endpoint definitions differ, so this
figure adds no direct cross-task metric comparison.

Dashed lines show the supplied simulated-participant summaries' arithmetic
means. The previously displayed 2.5th-97.5th percentile bands are hidden at
the author's request; their values remain in the source data and metadata.
They are not confidence intervals for the mean or rank-sum rejection regions.
No new simulation is run. The common legend now reads Random mean only.

Task 1's supplied pooled random reference contains 90 eligible 3D and 14
eligible 4D summaries, from 319/26 trajectories. All eligible f1 summaries
are 1/30 or 1/50; f2 summaries are 1/3 or 1/4. Thus the pooled percentile
width reflects normalization constants as well as reference selection.
This is an unbalanced conditional reference, not a newly validated
condition-matched chance law. Task 2 f3 includes 973 eligible random
trajectories: 972 summaries equal 1 and one equals 1.5. Its percentile range
is exactly 1-1 and is not artificially widened; mean=1.000513874614594.
All 1,999 trajectories contribute to f4, with percentile range 0-2 and
mean=0.6098049024512256. Eligibility and undefined-value exclusions are
applied equally according to the supplied source definitions.

Each human group is compared with its supplied random-reference summaries
using two-sided Mann–Whitney U, normal approximation with tie and continuity
corrections. Benjamini–Hochberg correction is within each f panel (three
Task 1 or two Task 2 comparisons), matching the supplied script and saved
statistics. The preceding Holm(10) table remains historical. Stars are group-
versus-random contrasts, not comparisons among human groups. The very small
f3 P values are asymptotic tail approximations; their numerical precision
should not be interpreted as exact finite-sample calibration.

| Panel | Group | n human/random | U | Raw P | Source BH P | Mark |
|---|---|---|---|---|---|---|
|f1|3D-NE|50/104|4975|1.076966213e-25|3.230898638e-25|***|
|f1|4D-E|51/104|4158|1.317105223e-10|1.975657834e-10|***|
|f1|4D-NE|54/104|3504|0.004691526667|0.004691526667|**|
|f2|3D-NE|50/104|4660|3.613232123e-21|1.083969637e-20|***|
|f2|4D-E|51/104|3943|2.647788707e-08|2.647788707e-08|***|
|f2|4D-NE|54/104|4210|5.068772606e-09|7.603158909e-09|***|
|f3|Obs|35/973|31121|4.348154178e-176|8.696308357e-176|***|
|f3|NObs|38/973|33070|2.712034486e-168|2.712034486e-168|***|
|f4|Obs|40/1999|77658.5|1.665719672e-29|3.331439343e-29|***|
|f4|NObs|40/1999|77446.5|3.434184353e-29|3.434184353e-29|***|

The endpoint/exclusion audit reproduces the supplied metric functions
independently. Source Task 2 duplicate subject-trial keys keep the first,
removing 4,695 rows; 30 conflicting keys are outside target Obs/NObs groups.
862 remaining rows outside those groups are omitted. Each Task 1 random CSV
has a malformed last row that is skipped as in the source. Their last valid
trajectories are incomplete (16/30 and 37/50 trials). Excluding these
incomplete trajectories does not change star categories. Saved generator
code/seeds are incomplete; independent simulated trajectories are assumed.

## g: source test trimming and original displayed pairs

All matched participants and their source identification grouping are
retained: Task 1 None/Partial/Perfect n=7/15/82; Task 2 Partial/Perfect n=23/56.
Boxplots summarize these full samples, without individual point markers.
As in the source, tests retain
values between each group's 5th and 95th percentiles inclusive if n≥6;
otherwise the full group is used. Tested group counts are therefore
5/13/73 and 20/50, distinct from plot n. Trimming is source-specified
exploratory processing, not a newly preregistered or journal-mandated rule.

Two-sided Mann–Whitney U tests use SciPy method=auto, with no multiplicity
correction in the supplied analysis. Current stars reproduce its three
original displayed pairs. The computed None–Perfect comparison remains in
Source Data and is omitted from the figure as in the original; this is a
display selection, not absence of a computed result.

| Task | Comparison | n tested | U | Unadjusted P | Source display |
|---|---|---|---|---|---|
|1|None–Partial|5/13|8.5|0.02033788742|*|
|1|None–Perfect|5/73|25.5|0.001406222183|Not drawn; **|
|1|Partial–Perfect|13/73|272.5|0.0151112785|*|
|2|Partial–Perfect|20/50|318|0.01825912167|*|

The preceding untrimmed Holm(4) sensitivity table is preserved. It gives
Task 1 None–Perfect adjusted P=.0106933 and the other three adjusted P=.1268532.
The analyses therefore support different pairwise conclusions; the current
source-reproduction labels are not evidence of robustness to exclusion or
multiplicity choices. Identification-group associations are observational.

## h: source paired Wilcoxon without added correction

The supplied corpus script is reproduced: 72 participant pairs per task,
Task 1 within-participant phase averages across 109 source rows, Task 2 one
row per participant. The metrics count dimension terms in DIS passages and
the whole interview, which includes those passages. This is conditional
lexical enrichment, not DIS versus a disjoint Others corpus.

Two-sided paired Wilcoxon uses zero_method=wilcox and method=auto. There is
no added cross-task multiplicity correction in the source. Task 1 has two
zero differences omitted from the ranks, not from the plot, leaving 70
nonzero differences. Source results are W=437, P=2.4298046512211917e-6;
Task 2 W=657, P=.00022701800668101054. Both remain ***. Historical Holm(2)
and zsplit caches are preserved but do not define current stars.
The supplied Task 1 original historical notebook is unavailable; the
provided reproducible subject-average analysis is explicitly identified.
Lexical preprocessing is not assumed identical across the two tasks.

## Marks, exports and verification

Boxes show medians and IQRs; whiskers extend to the most extreme points within
1.5×IQR. All valid participant summaries, including boxplot outliers, remain
in box-summary and statistical calculations. Individual and outlier points
are hidden in b/f/g; h retains its participant points. Mean diamonds are arithmetic
means. Each b/f/g/h box is 5.05645 mm wide, matching the Fig5c box-to-axis
width ratio of4.95/27.9 on the unchanged28.5-mm plot area. h lines connect participant pairs and
means. Stars use the panel's stated P values, not a shared correction rule.

The source-processing audit, exact source-consistent f/g/h statistics, raw
summaries and preserved earlier analyses are available in analysis/.
Review scope is bounded to this figure; reproducibility of supplied processing
is distinct from judging optimal statistical methodology or manuscript
acceptability. Independent frozen reviews and an editorial synthesis are in
qa/source_consistency_review/.

Python/Matplotlib; 180×170 mm, RGB PNG-only, native 600 dpi, one font family
(DejaVu Sans) and three nominal sizes (5/6/8 pt). g sample counts are retained
in the caption and source metadata rather than category tick labels.
Final render checks native text/icon clearance, complete f input counts,
absence of individual b/f/g markers, complete c SEM envelopes, Fig5c-proportioned boxes, fixed 1.5 pt axes alignment,
and 2/3/3 mm row clearances. The PNG-only contract excludes PDF glyph/stroke
collision audits. Original sources and historical figure bytes are verified
separately from current standalone rendering and statistical checks.

## Shared omitted-zero marks (2026-10-12)

Double slashes indicate an omitted zero-to-lower-bound interval. The native marks match Fig5f in physical size. They change decoration and, where necessary, x tick padding; plotted values, limits and tests are retained. Logarithmic scales and domains containing zero remain continuous.

## Box-to-axis ratio revision (2026-10-12)

The current Fig5c native geometry supplies the reference: a4.95-mm box in a
27.9-mm plot area, giving a width fraction of0.17741935. Fig2 retains all
28.5-mm plot widths, category positions and x limits; only the horizontal
widths of its21box bodies, medians and caps are adjusted to this fraction.
Vertical summaries, whisker endpoints, mean values, individual observations,
source tables and the17significance records are unchanged. All non-box-panel
pixels remain identical. The prior full package is preserved outside
main_figure in the Fig2 history folder, under20261012_before_box_ratio_fig5c.
This is a source-rendered style revision, not a reanalysis or model fit.

## Tick endpoints and compact categorical margins (2026-10-14)

The endpoint-tick values below describe the preceding spacing revision;
the regular-tick refinement immediately below supersedes b/c display values.

All10native Cartesian views have visible bottom and left spines bounded by
their first/last displayed major ticks. Public b adds−10within its unchanged
−10–110view to cover its existing negative lower whisker. Public c adds68and93 endpoint ticks
within the unchanged68–93view to enclose its original69.387–90.825SEM envelope;
h1 adds.14within the unchanged0–.14view to enclose its original individual
points. The schematics, illustrative image panels and colour key are exempt.
The omitted-zero decoration remains connected to the real y-spine.

The public b, f3, f4, g2, h1 and h2 categorical x viewports are translated
by+0.218239648 at unchanged2.1-unit span. This shifts their boxes, category
ticks and brackets left2.961824mm while retaining every numerical/category
coordinate and the5.05645mm physical box width. The left y-spines retain
their5pt outward position. Nearest visible box-edge to actual y-spine-edge
gaps are3.5mm; existing4.05399mm gaps in f1/f2/g1 are retained. Positive
ylabel padding changes only in f1/h1, with final ylabel-right-edge to nearest
box gaps7.21475–9.9mm. All original participant-marker x edges remain
inside their views; exact-zero observations retain baseline contact.

The current source renderer and display-only verifier use local frozen
inputs and the complete pre-revision package/PNG snapshot. Native artist
geometry/styles, source-image arrays, sample groups, uncertainty intervals,
all17saved significance records, colours, panel mapping and y view ranges
remain unchanged. This revision runs no preparation, fitting, simulation or
statistical test. Verification outputs are in `qa/tick_endpoints.json`,
`qa/category_label_spacing.json` and `qa/tick_spacing_revision_verification.json`.

## Regular ticks and explicit percentage-point unit (2026-10-14)

The subsequent explicit b tick request below supersedes only this section's
b viewport/ticks; c's regular ticks and the full percentage-point unit remain.

Public c now uses the65–95y display with uniformly spaced65/75/85/95major
ticks (10units). This contains the complete unchanged69.387285934–90.824605235
SEM envelope and all original mean/curve values. Public b now uses−20–110
with−20/0/20/40/60/80/100major ticks (20percentage points), preserving the
original zero reference, negative lower whisker and104-level significance
annotations. Its ylabel is “Excess highest-FEL choices (percentage points)”.
The original human-minus-random differences already use percentage points;
neither multiplication/division nor a change to percentage interpretation
is applied. All first/last spine endpoints follow the displayed major ticks.

Only these authorized b/c y viewports/ticks and b wording change. Native raw
data artists, intervals, annotations, saved P values, groups, colours, box
widths, categorical x viewports and all other panels retain the preceding
version. The complete350-file package/PNG snapshot is preserved in
`revisions/20261014_before_regular_ticks_feedback_refinement/`, including
the previous spacing backup. Current display-only verification is
`code/verify_regular_ticks_feedback.py`, producing
`qa/regular_ticks_feedback_verification.json` and a standalone deny-project
render check. This refinement performs no new analysis, fit or simulation.

## Explicit requested b ticks (2026-10-14)

The later three-tick request below supersedes this section's four-tick list;
the−10–110view and all source data remain unchanged.

The user explicitly requested public b major ticks−10/0/50/100, overriding
the default preference for equal major-tick intervals. The y viewport is
−10–110and the visible y-spine begins at−10and ends at100. This retains the
original−2lower whisker and95upper whisker, zero reference and104-level
significance annotations. All native data/interval/annotation coordinates,
groups, source images,17saved significance records, phase colours, ylabel
“Excess highest-FEL choices (percentage points)” and categorical x geometry
remain unchanged. Public c retains65–95with65/75/85/95ticks and its complete
original SEM envelope.

The full707-file package and PNG snapshot, including all earlier backups,
is preserved in `revisions/20261014_before_requested_ticks_wordclouds/`.
Current scoped verification is `code/verify_requested_ticks_wordclouds.py`,
with results in `qa/requested_ticks_wordclouds_verification.json`. The scope
allows only public b's ticks/spine and the authorized y-view transform of
its unchanged marks; all pixels outside its native axis/view region are
protected. This display-only task runs no preparation, analysis, statistical
test, fitting or simulation.

## Three requested b ticks with the negative margin retained (2026-10-14)

Public b now shows exactly0/50/100major ticks. Its−10–110viewport is retained
so the original−2lower whisker remains visible at the exact same rendered
coordinate. The visible y-spine starts at0and ends at100. This is a tick/spine
decoration change only: no data, interval, source annotation, P value, label,
group, phase colour, categorical x coordinate or view range is changed.
No omitted-zero slash is appropriate because this viewport includes zero
and negative values. Public c retains65/75/85/95on65–95.

The complete1,421-file package and PNG snapshot, including all earlier
revisions, is preserved in `revisions/20261014_before_box_fliers_temperature/`.
Current verification is `code/verify_box_fliers_temperature.py`, with native
and standalone checks in `qa/box_fliers_temperature_verification.json`.
All pixels outside b's left tick/spine/label gutter are protected; the
scientific b marks also retain their exact native rendered coordinates.
This task performs no new analysis, statistical test, fit or simulation.

## g/h categorical x-axis alignment (2026-10-12)

Each h bottom spine uses the same3.5pt outward offset as its corresponding
g subplot. This shifts only the x-axis line, tick marks and category text;
plot-area geometry, data coordinates, box widths and statistical annotations
are retained. The native renderer verifies both the spine baseline and
tick-text baseline differences to within0.03pt. Previous PNG/code/data/QA
are stored outside main_figure in the Fig2 history archive under
20261012_before_h_axis_alignment. The change adds no fitting or simulation.


## Public b ylabel clarification (2026-10-15)

The ylabel is now “Highest-FEL choice rate / Human − random”. The unit, percentage points, is stated in the caption rather than on the axis. Values remain 100 times the participant-specific human-minus-random difference in highest-FEL choice proportions. No rescaling, changes to trials, samples, boxes, significance tests, ticks or limits are made. Historical label descriptions above refer to prior revisions. Current verification: `code/verify_b_ylabel.py`.

### Display update, 15 October 2026

DIS-frequency tick labels are converted from source proportions to percentages by a factor of 100; all native scientific coordinates, intervals and statistical results remain unchanged. Raw field names such as `round` are retained for source compatibility, while displayed temporal labels use Trial.

## Current display refinement, 2026-10-15

Panel f plotting rectangles now match public b at 28.5 × 32 mm; canvas height is 145 mm. Task 2 f3 and f4 divide every displayed box, whisker, median, mean and random reference by 60 and 5, respectively. These common positive scale changes preserve ranks; the saved Mann–Whitney tests are retained without recalculation. All input tables, endpoint eligibility and hidden outlier policy remain unchanged. Task 1 within-dimension subruns differ from the whole single-dimension bouts retained in Task 2; Task 1 coverage is within-bout, Task 2 coverage spans the game. The shared labels are normalized display names, not a redefinition or new between-task test. Panel c removes the unit from its ylabel only. Panel d Task 2 removes the source image’s two-pixel perimeter crop border, preserving room content, source scale and native sofa locators.
# 15 October 2026: reference label and equal y-spine heights

Panel f's shared dashed-line legend now reads Random. The reference still denotes the saved random-selection mean, with the previously documented normalization. Task2's run-length display uses 0–0.10 with 0.025-spaced major ticks, giving its visible y-spine the same 32 mm height as the other three f axes. All source values, interval definitions, endpoint eligibility and saved statistical comparisons are retained. This is a display-only revision; no random reference, statistical test, fitting or simulation was rerun.

