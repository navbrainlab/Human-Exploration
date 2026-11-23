## Display-only concept refinement, 2026-10-14

Publicc alone is enlarged into unused left whitespace; right-facing boundary and all statistical panels remain. The same illustrative eight sofas and original feature/condition colours are drawn natively. Their linear dimensions increase40%; cards are38×33.5mm with a12.4mm observation gap. Thin frames and smaller headers retain5/6/8pt type. The red prohibition badge beside DISblocked denotes the forbidden combination without obscuring stimuli. Two arrows branch independently from the shared prior-observation label; no sequential group transition is implied. All data arrays, exposure/score definitions, sample counts, coefficients/intervals/Pvalues and source eligibility remain unchanged. This revision introduces no statistical test, refit or simulation. Source snapshots, nativegeometry and pixel protection are recorded in the current QA.

# Fig4 Tighter Outcome Panels: Methods And Source Boundaries

Current requested sparse ticks,2026-10-14: public a shows exactly20/60/100,
with its existing20–103 view and20/100 spine endpoints preserved. This only
changes the y-tick gutter; every point, curve, interval, group, P value,
font, colour, width and layout stays fixed. Frozen-data native verification
is `code/verify_requested_ticks_wordclouds.py` and
`qa/requested_ticks_wordclouds_verification.json`. It includes exact native
ticks, mark/interval/source protection,600dpi/fonts/alignment/spine checks,
pixel comparison and byte-identical portable redraw. The complete previous
package/PNG and earlier history are under
`revisions/20261014_before_requested_ticks_wordclouds/`. No new analysis,
statistical test, fitting, simulation or resampling runs.

Preceding display refinement,2026-10-14: public a uses regular20-unit y ticks
20/40/60/80/100 on its live shared Task1/Task2 score axis. Its display limits
expand30–103→20–103, retaining the upper103 padding and all frozen score values
34–100, OLS means and95% intervals. The main shared y spine spans20–100 and
the actual omitted-zero marks retain their original physical placement.
Scientific inputs, statistics, category spacing, colours and every other
panel are unchanged. The full previous package/PNG, including earlier
revisions, is archived in `revisions/20261014_before_regular_ticks_feedback_refinement/`.
Current acceptance is `code/verify_regular_ticks_feedback_refinement.py`
and `qa/regular_ticks_feedback_refinement_verification.json`; it reads frozen
statistics and runs no new analyses, statistical tests, fitting or simulation.
Historical display dimensions and prior QA below retain their stated scope.

This edition copies frozen inputs from the three-compound-panel edition
byte-for-byte. It retains the original public c manipulation from `fig4_nhb_style`, now
stacked vertically, and retains five groups a/b/c/d/e.
Upper a/b keep their previous10.7-mm right shift; outcome rows move up10mm.
Statistical values, numerical limits and graph dimensions remain unchanged. The canvas
shrinks180x166->180x156mm to retain bottom margins; graphs/fonts are not shrunk.
Pixel equivalence after geometric translation is not claimed. Source g's S4 migration and source f's
interval-free display are inherited. Source IDs are historical analysis IDs,
not public letters. No model fitting, simulation, statistical contrast,
sample exclusion, numerical summary or confidence interval is changed.

## Associations (Public a/b; Source a/b/c)

Task1 retains105 P2/P2-only participants; Task2 retains80 Obs/NObs participants.
The predictor is DIS in the first floor(0.5 x observed trials); the response
is mean score in the final ceil(0.05 x observed trials), containing1-3 recorded
trials. Both frozen OLS-HC3 grids are reused, not refitted. Slopes and their
original95% coefficient intervals are12.649592600646482 [6.5891552228974595,18.710029978395504]
and16.744417760847938 [1.7408101235718725,31.748025398124003]. Original slope
P values are.00004296710116626994 and.028715453811434016.

The two33x30-mm axes actually share a Matplotlib y group and major locator,
with common30-103 limits and40/60/80/100 ticks. Only the left axis displays
y ticks/label/spine/omission marks. Samples are not pooled. Both x-axis
spines/ticks remain, with one6pt `% of DIS trials (first half)` label.
Source a/b/c lefts are now23.7/61.7/118.3mm instead of13/51/107.6mm. The
shared x label and a/b headers move by the same10.7mm. Source a/b still actually
share their y group and major locator.

The cross-phase association retains51 matched Task1 participants: full-phase
3D-NE DIS versus first ceil(0.25 x observed P2 trials)4D-E DIS. It is not
Task1-to-Task2 transfer. OLS-HC3 slope.6201716714535519,95% interval
[.30273659097949474,.937606751927609],P=.00012857410640989409. Its original
unbounded fit/CI coordinates are preserved; the0-1 viewport clips the upper
fit/interval without clamping the underlying values or fitting a bounded model.

## Concept c And Outcome Rows d/e (Sources d/e/f/h/i)

The native schematic preserves the same8 sofa examples, observation branches,
extra DIS-blocking rule and condition colours. It is illustrative, not new
quantitative data. Its36×80-mm native plot spans the d/e outcome rows from top64to144mm.
The d/e outcome subplots use their axis labels without separate role subtitles.
The schematic is36×80mm at left25.6/top64mm. Obs is above No-DIS. Shared prior
observation is shown between the cards, with separate upward and downward
arrows into the two conditions. Four sofa examples per condition remain. The groups are not a sequence or a new
task-phase definition. Public d groups source e learning and source f full score;
public e groups source h reported elements and source i classification. Each
group has one appropriate modality icon rather than repeated right-column icons.
Evidence types and statistical results remain unchanged.

Learning (source e) retains80 participants,40/condition, and4800 extended
participant-trial rows. After first achieving100, missing scheduled trials are
filled with100 through trial60, as in the supplied derivation. This retained
assumption differs from using observed rows only. Each participant is smoothed
with a centered5-trial available-window mean. Group means/sample SEMs are
computed after participant smoothing. The frozen display uses400-node PCHIP
curves and mean plus/minus1 SEM ribbons, beginning at trial1 without inventing
trial0. Independent verification recomputes each participant window, not a new
statistical analysis. `analysis/d_learning_display_5trial.csv/json` retain bytes.

Full score (source f) counts each participant once if any recorded score is at
least100. Obs18/40=.45; No-DIS9/40=.225. Plot bars at these raw proportions on
a0-1 axis, not45/22.5 on a percent axis. The approved display removes only
error artists, stars and brackets. Original Wilson95% intervals are
Obs[.3070530781090492,.6017090820106794] and
No-DIS[.12316091323540743,.37503096742310027]. The original two-sided Barnard
unconditional exact comparison has P=.03989111970448495, difference.225
(22.5 percentage points). These original intervals/test remain in
`csv/panel_statistics.csv` and are independently checked by the new verifier.
Not displaying them does not imply a nonsignificant test; no uncertainty or
P value is recalculated or replaced.

Noticed elements (source h) retain the corrected80-participant320-row source.
Stages are coded post-observation counts and structured q2_1/q2_2/q2_3 noticed
lists, excluding q2_4's union. Counts are reported elements, not classification
accuracy or necessarily the five ground-truth dimensions. Group means/sample
SEMs and the complete two-sided stage-wise Mann-Whitney family remain frozen.
Holm(4) P values are.4817271743190612,.4817271743190612,.02068892879319218,
.020958444786923283. Stars remain only at Mid/Late. These are exploratory
stage contrasts, not an interaction test. Local source lists and means/SEMs
are independently verified. The historical mixed-model P=.1411 is not used
for the corrected noticed endpoint.

Classification (source i) retains40 Obs/39 No-DIS participants. A task dimension
is correct only when exclusively assigned to its true relevant/irrelevant
category in q2_4; missing/contradictory assignments are incorrect. Means remain
.595/.3641025641025641; bootstrap95% intervals [.505,.690] and
[.3230769230769231,.4051282051282051]. Original10000 participant resamples
(seed20261010) are not regenerated. Original Mann-Whitney P=.00036589285526049723
and its*** annotation remain. Scientific coordinates stay proportions; only
ticks display percent0-100. All source observations remain in the local table.

## GEE Migration (Source g; Not In This Main Display)

The frozen360-row Gaussian GEE prediction cache and original test remain local
for traceability and are assigned to the separate new S4 alternative. The model
uses80 participant clusters, an exchangeable working correlation and cubic
B-spline progress terms. Frozen Wald chi-square(4)=11.646427589375037,
P=.02018363396061409; ribbons are population-mean pointwise95% intervals,
not SEM. The same post-perfect carry-forward assumption applies. No GEE
refitting or copying another endpoint's test is performed. The new S4 package
is `figureS4_DIS_evidence_with_gee_20261013`; its panel e and local QA preserve
the original GEE means, both confidence bounds and participant-clustered test.

## Rendering And Protection

Five public letters a/b/c/d/e. Concept c's header is79.5mm; d remains56mm and
e106mm. c's schematic/letter/icon translate together by23.5mm, centering its
80-mm concept on104mm, the midpoint of the d/e plot envelope64–144mm.
The two right-hand statistical axes (source f/i) retain their4-mm reduction of
the within-row gap. The entire lower c/d/e group then translates right7.55mm;
upper a/b remain pixel-identical to the preceding edition.
Association headers retain their5.898-mm offset;
outcome/concept headers use8mm to align with metric titles/annotation bands.
Visible-content centers include axes, labels, ticks, titles, legends and parent
headers: upper84.46294792976937mm, lower90mm. The lower group's visible
horizontal extent is11.55–168.45mm, leaving equal11.55-mm page margins.
The upper group is not moved by this lower-group centering request.
Strict1.5pt alignment tolerance is not relaxed.

Compared with the previous five-panel edition, visible upper/lower separation
is15.206->5.206mm and concept/outcome clearance2.749->5.649mm. The concept
is deliberately centered across both outcome rows, with no numerical or size changes to the
two-row grid. Vertical separation need not exceed horizontal within-row gaps;
that previous assumption is removed to meet the user's compact-layout request,
not by relaxing the strict geometry tolerance or reducing any font.

Native Matplotlib180x156-mm white RGB PNG at600dpi,4251x3685px, DejaVu Sans
5/6/8pt. Parent letters a/b/c/d/e are8pt bold; d/e role subtitles are omitted.
Associated scientific marks, intervals and statistics keep their prior data
coordinates. Source f's historical percent-to-proportion display is inherited.
Independent groups have more visible whitespace than internal subplots.
All statistical plot areas retain30-mm heights, approved widths/colours and4.35-mm bars.
The d/e visible subplot gaps decrease from8.184/6.391mm to4.184/2.391mm;
the two categorical bar centres decrease from15.000mm to12.130mm apart.
Only the arbitrary categorical x viewport widens to(-.65,1.65); bar heights,
error bars, category labels and numerical y coordinates do not change.

The original1.5pt alignment/gutter tolerances remain strict. Specific exemptions
cover no separate public letters for compound children, unequal top-row widths
and inner/outer top gutters. Same-role grid columns and row baselines are checked.
Double slashes on a/e/h retain1.2276x.6-mm physical strokes and actual outward
y-spine alignment, with0.903-mm visible clearance to the x-axis endpoint.
Whole-figure native text collision/clipping and final raster DPI/mode checks
must pass. The renderer refuses an existing PNG before any QA writes. The current
named PNG was replaced from a verified native render after the prior PNG and
metadata were archived under `revisions/20261013_before_page_centering/`.
The subsequent d/e subtitle removal has its own immediately prior snapshot at
`revisions/20261013_before_de_subtitle_removal/`.

`source/protected_source_hashes.json` preserves the coordinator's original-file
snapshot. The verifier checks every available original and requires all of
them with `--require-originals`. A copied package can render without reading
old live figures; unavailable external originals are reported explicitly by
the verifier. Numeric evidence snapshots are local immutable copies, not
runtime imports of an old renderer. PNG-only checks are not PDF path-collision
or journal editable-vector certification. Historical source packages remain unchanged.



## Vertical concept style verification (2026-10-14)

Only public c and its header placement change relative to the immediately
preceding edition. All statistical plots are pixel-identical outside the
concept region, and native numeric coordinates, intervals, colours, tests and
input hashes remain unchanged. The shared-observation text and two arrows
are between the cards; each arrow feeds one parallel condition. Ordinary
rendering and `verify_vertical_concept.py` do not refit, resample or reanalyse
any result. Historical styling dimensions above refer to earlier editions
where explicitly recorded.

## Tick endpoints and categorical display margins (2026-10-14, current)

All visible Cartesian main spines are bounded by the first and last displayed
major ticks, using the portable `tick_bounded_spines.py` helper. Numerical
view limits, quantitative artist coordinates and uncertainty endpoints are
preserved. The shared Task1/Task2 score axis adds30 to40/60/80/100, so its main
y spine spans30–100 while the original30–103 viewport retains marker clearance.
The omitted-zero extension still joins the true y spine; slashes retain their
physical dimensions and their outward5pt alignment. Shared Task2 y decorations
remain hidden. The approved0–1 clipping of public b's unbounded OLS mean/CI
remains disclosed above. Phase group offsets ±.04 retain their original margin.

The categorical-only `category_label_spacing.py` helper measures the actual
rendered mark and outward y-spine strokes. Source f/i x viewports are translated
by+0.3064941756272406 at an unchanged2.3-unit span. Their final view limits are
[-0.34350582437275945,1.9564941756272405]. Group centres remain0/1, data bar
width remains0.3586021505376344 and physical width remains4.35mm. All y values,
bootstrap intervals, native group colours and raw source labels stay fixed.
Ticks, bars, error bars and the significance bracket move together; the star
uses the category-x transform to stay centred on its existing bracket.

The visible bar-to-actual-y-spine gaps decrease7.217908→3.500000mm. The
ylabel-right-edge-to-bar gaps decrease13.003463→9.285556mm for full score and
13.564380→9.846472mm for classification. Numerical scatter/trajectory artists
are not translated. All five numerical plot rectangles and the latest vertical
concept are pixel-identical to the immediately prior PNG. Canvas180×156mm,
RGB4251×3685px,600dpi,DejaVu Sans5/6/8pt, regular ordinary text and strict1.5pt
alignment remain. Minimum verified Cartesian corner gap is1.810086mm and
slash-to-x-endpoint gap is0.903144mm. The7.3pt classification baseline continues
to align its category labels with the neighbouring report plot.

The previous active package plus PNG are archived at
`revisions/20261014_before_tick_endpoints_box_spacing/`. Earlier revisions are
preserved in place and hashed separately. Current verification is
`code/verify_tick_endpoints_box_spacing.py`: frozen file hashes, native
coordinates, original intervals/statistics, spacing/endpoints, typography,
alignment, numeric/concept pixel protection and portable rendering with all
project reads denied. This verifier never invokes historical questionnaire
analyses, new resampling, fitting or simulations.

### Display update, 15 October 2026

DIS-frequency tick labels are converted from source proportions to percentages by a factor of 100; all native scientific coordinates, intervals and statistical results remain unchanged. Raw field names such as `round` are retained for source compatibility, while displayed temporal labels use Trial.
No-DIS/Non-DIS uses the Fig5 others base grey, #8D9399; transparent uncertainty/fill marks retain their existing alpha. No-DIS intervention and Non-DIS trial categories remain distinct scientific definitions.

The 2026-10-15 schematic revision places one 23.6-mm red prohibition circle and slash over the four green sofas in the No-DIS condition. The footer symbol is removed and the DIS blocked text is centred. This is an illustrative manipulation symbol; all quantitative panels, original inputs and tests remain unchanged.


Display refinement, 15 October 2026: public d full-score proportion uses limits 0–0.5 and ticks 0, 0.25, 0.5; observed proportions remain 18/40 and 9/40. Public c concept artwork shifts 5 mm right; no quantitative panel or source observation changes.
