# Methods and statistical details for Fig. 7

## Edition and fitted model

This separate edition retains the original architecture and grouped Plackett–Luce action generator, but Task 1 fitted panels use the completed joint seven-parameter `free_bonus` analysis. The free parameters are eta, beta_decay, temperature_threshold, temperature_slope, attention_inertia, action_softmax_temp and beta_init. The initial direct feature-utility bonus is beta_init in [0,40], and beta_t=beta_init×beta_decay^(t−1). It is neither fixed 20 nor forced 0 in these fitted panels. Fixed attention/feedback/count settings retain the existing DGEM.3 configuration. All 156 all-data fits (105 people;51/51/54 phases) are retained. The beta_init median is 0.499 (IQR 0.062–3.261), with five exact zeros.

The archived run used two independent DE/Powell starts with finite budgets, then independently rescored the saved parameters using 8,192 particles×8 seeds. This figure reuses that completed fit; it does not claim a new optimizer run or global optimum. The optimizer protocol, fit table and high-particle score table are separate provenance inputs. Source files have evolved since the original optimizer protocol; the saved parameters and score outputs are frozen, and current replay/simulation sources are independently hashed. Endpoint tests confirm the free model retains the existing full and no-bonus kernels when beta_init is 20 or 0.

The figure contains distinct model scopes: b uses retained seven-parameter DGEM fits in both tasks; c/d/f/g use the same Task 1 free-beta analysis. Task 1 b/e retains the verified 2026-10-12 RL policies. Task 2 b/e now uses the supplied V2 alternatives, while DGEM and Human references are retained. Panel e uses separately selected performance parameters, including fixed beta_init=20 for DGEM. Its Task 2 sources have different feedback environments, detailed below. This transfer copies saved results and performs no fitting, simulation or new statistical test.

## Panel a: uncertainty-dependent attention and action construction

Stimulus and Action show the same nine illustrative items. Colours depict the three levels of one feature dimension. Stimulus colours are shuffled in a 3×3 grid; each Action row has a single colour, with different levels across the three rows, illustrating DIS. The example is checked with the project's DIS classifier. Feature assignments are schematic rather than empirical data. This illustration does not constrain the model to generate DIS on every trial.

The schematic follows the shared DGEM V11.9.4.3 kernel. Feature uncertainty is U_df = 1/(n_df + 0.1), and dimension uncertainty sums across the three features. Dimension priorities depend on uncertainty and previous attention, including a uniform lapse component. The smoothed total-uncertainty gate determines the Concrete temperature. Learned feature values do not directly enter attention control. Independent standard Gumbel dimension shocks generate continuous normalized attention weights.

Feature utility is S_df = Q_df + beta_t U_df + sigma Z_df, where beta_t = beta_init rho^(t-1) and Z_df is a standard Gumbel shock shared across items containing the same feature within a trial. Item utility sums attention-weighted feature utilities. The displayed ranking step uses the existing Plackett–Luce kernel with attention-dependent residual item noise. Successive groups of three ranked items form three ordered rows; within-row order is ignored. The three attention bars are a schematic example, not fitted estimates or measured observations. The seven-parameter scope is documented above; initialization, learning, grouped-PL choice and particle filtering retain the DGEM.3 kernel.

## Panel h: TDGE movie recommendation and its relation to DGEM

Panel h illustrates the separate task-dimension-guided exploration (TDGE) application described in the author-supplied manuscript *Human-inspired, Task-Dimension-Guided Exploration for Efficient Learning in High Dimensions*, `reference/Task_dimension_guided_exploration_NeurIPS_2026 (19).pdf`. The mechanism is grounded in Sections 3.3–3.4 (pp. 5–6), Algorithm 1 (p. 16), and Appendices D–E (pp. 16–18). It is not the stimulus-layout action kernel used by DGEM in panel a.

For movies, a **feature** is one item-descriptive MovieLens tag, such as `gruesome`, `brutal`, `australia` or `canada`. A **dimension** is a semantic cluster of related tags. Sentence-BERT embeddings are clustered by Ward hierarchical clustering; a constrained Kelley–Gardner–Sutcliffe criterion selects the cut. The preprocessing yields 47 dimensions from 1,128 MovieLens tags. Each movie is associated with its ten highest-relevance tag-genome descriptors. The restored diagram shows representative semantic clusters (Violence and Country), selected tags (gruesome, brutal, australia and canada), candidate Movie A–C, with Movie B highlighted as the illustrative recommendation within the candidate list. The separate Action node is omitted; Movie B uses exactly the same purple fill and border as the selected feature boxes. The grey Other tags block denotes other feature options and matches the grey Other clusters block in fill, border, dimensions and vertical position; purple tag boxes remain the illustrative selections. Cluster titles summarize the supplied manuscript examples; tags do not represent the complete selected set. Numerical scores and feedback paths remain omitted. Movie identities and icons are schematic rather than observed recommendations.

At trial $t$, independent contextual-bandit agents score dimensions, features and items from contexts combining the current user representation with the corresponding semantic representation. LinUCB and NeuralUCB combine estimated value with an uncertainty bonus; NeuralTS samples uncertainty-aware scores. TDGE selects $K_1$ dimensions and pools their tags before selecting $K_2$ features **globally across that pool**:

$$
D_t=\operatorname{TopK}_{c\in\mathcal C}(s^{\mathrm{dim}}_{c,t},K_1),\qquad
G_t=\bigcup_{c\in D_t} C_c,\qquad
F_t=\operatorname{TopK}_{j\in G_t}(s^{\mathrm{feat}}_{j,t},K_2).
$$

The selected tags define an intent vector, and cosine similarity retrieves candidate movies from the valid item set. The item-level bandit then recommends the highest-scoring retrieved movie:

$$
\bar v_t=\operatorname{norm}\!\left(\frac{1}{|F_t|}\sum_{j\in F_t}v_j^{\mathrm{feat}}\right),\qquad
P_t=\operatorname{TopK}_{i\in I_t^{\mathrm{valid}}}\!\left(\cos(v_i^{\mathrm{item}},\bar v_t),K\right),\qquad
a_t\in\arg\max_{i\in P_t}s^{\mathrm{item}}_{i,t}.
$$

The manuscript uses $(K_1,K_2)=(2,12)$ for all three MovieLens backbones. The schematic shows representative tags and candidate movies rather than complete selections or numerical scores. It does not use the previously described direct sum of selected feature scores as the final item score: retrieval and contextual item scoring are distinct steps in the supplied manuscript.

The analogy with DGEM is **dimension-guided feature processing that structures action selection**. DGEM allocates continuous attention weights derived from feature uncertainty and weights feature utilities before constructing a ranked stimulus layout. TDGE makes hard dimension/tag selections from value-and-uncertainty-dependent bandit scores, then retrieves and scores movies. Purple identifies the dimension-guided stages, without implying shared parameters, identical attention equations or a newly validated DGEM recommender.

The reward node and feedback arrows are omitted to focus this panel on action construction. In the source algorithm, learning still follows the recommendation: the item agent updates from the observed reward, and higher-level credit is restricted to selected tags present in the recommended movie and their contributing clusters. This omission does not remove reward learning from TDGE.

## Panels b–g: summaries, uncertainty and statistical tests

**b, Model fitting.** Task 1 first averages each person’s geometric mean trial likelihood across their available phases, then computes the mean and s.e.m. across 105 people. The DGEM values come from the independently evaluated 8192-particle × 8-bank free-beta all-data score, not the optimizer’s training-bank likelihood. fRL-decay and naiveRL scores are newly estimated using the definitions and budgets below; Bayesian retains its original source. Three paired two-sided Wilcoxon comparisons against DGEM use Holm correction; exact values and matched samples are saved in csv/fit_task1_significance.csv. Bayesian has 92 matched participants; fRL-decay and naiveRL each have 105. Task 2 retains seven-parameter DGEM including beta_init and imports the saved V2 baseline fits, cohort audit and statistical comparisons specified below. The shared line summarizes separate model comparisons, not a pooled-group test. Chance is 1/1,680 for Task 1 and 1/1,820 for Task 2.

**c, Attention.** All 5,323 observed human trials are replayed using their participant-phase seven-parameter fits with 2048 particles. A deterministic new seed is assigned to each sequence; observed layouts and rewards update the latent states. The fast diagnostic replay omits only unused enumeration of all DIS layouts. Grey coordinates are prior-particle-weighted effective attention temperature and maximum dimension attention weight. Bins and 5,000 participant-clustered bootstrap resamples follow the existing attention-bin definitions. All trials are included; the inset displays the full range. This is conditional fitted-data replay, not free simulation.

**d, DIS and attention.** A participant-clustered binomial-logit relationship and 95% confidence band are recomputed from the free-beta replay. The lower shapes are trial distributions; they are not participant densities. The separate paired inferential test averages maximum attention weight within each participant and observed action class. A two-sided Wilcoxon test uses n=105 pairs, W=102, P=1.043431982e-17. Exact values are in csv/panel_statistics.csv. Observed DIS labels are derived from human layouts and are not assigned from an attention threshold.

**e, Best-performance simulations.** DGEM uses the full V11.9.4.3 model with beta_init fixed at 20, separately selected from participant fits. Task 1 displays only the frozen 4D results. The DGEM candidate (threshold 4.063, beta decay 0.971) achieved DIS proportion 0.9103859789 and full-score proportion 0.9268 across 5,000 independent validation agents. Its saved bootstrap DIS 95% CI is 0.9089–0.9118 and Wilson success 95% CI is 0.9192–0.9337; the plotted errors are SEM, not these CIs. Task 1 Human is the independent 4D-NE cohort (n=54); fRL-decay and naiveRL each use 1,000 agents, and Bayesian is a historical 100-agent reference. Task 2 combines saved V2 alternatives (80 episodes each) with retained DGEM (1,000 episodes) and Human (80 people); feedback environments differ, as specified below. No performance significance test is introduced.

**f, Seven-parameter fitted temperature slope.** All 156 participant-phase values come from the same free_bonus fit table used in b Task 1,c,d,g. beta_init was fitted jointly with eta, beta_decay, temperature_threshold, temperature_slope, attention_inertia and action_softmax_temp. Five fits reach the beta_init=0 boundary; they remain full-model fits, not independently fitted no-bonus models. Phase labels 3D,4D-E and 4D-NE correspond to P1,P2 and P2-only (n=51,51,54). The parameter temperature_slope is the uncertainty-gate slope, not effective attention temperature. Boxes summarize all values with median, IQR and 1.5-IQR whiskers on a log axis; individual points, outlier markers and participant connecting lines are omitted.

The three exploratory two-sided phase contrasts form one Holm(3) family. The 3D–4D-E comparison retains 51 subject pairs and uses Wilcoxon signed-rank (W=331; raw P=0.0018583016423124122; adjusted P=0.0037166032846248243). Independent Mann–Whitney comparisons use U=1322 for 3D–4D-NE and U=1916 for 4D-E–4D-NE. Exact values are authoritative in csv/slope_phase_tests.csv and slope_analysis.json. Medians are 0.039614,0.079323,0.040127 for 3D,4D-E,4D-NE. These are fitted-parameter contrasts, not a demonstration of simulation recovery of group behaviour or lower effective temperature. No model was refitted and no simulation was rerun for this correction.

The 3D–4D-NE adjusted P is 0.7267713238249085; the 4D-E–4D-NE raw and adjusted P values are 0.00055529076830236254 and 0.0016658723049070875. Significance labels are **/ns/** for 3D–4D-E,3D–4D-NE,4D-E–4D-NE.

**g, Trial-wise trajectories.** Each of the 156 seven-parameter all-data fits generates 20 independently seeded production-model simulations (3,120 runs). beta_init and beta_decay come from the corresponding fit. Task horizons remain 30 trials for 3D and 50 for 4D; agents stop on first full score. DIS ratios use only real simulated actions; no full-score padding is added to DIS. Repetitions are averaged within participant before phase summaries, and phases are equally weighted where available. Human, Bayesian and both RL curves are retained references. The established plotting convention uses seven-round centered moving means and s.e.m. shading reconstructed from the saved 95% summary intervals. The lower temperature trace is from the new fitted-data replay, pooled over available observed participant-phase trials each round. It is min–max normalized to [0,1], with the same linear scaling applied to its trial-level s.e.m. This is not simulation temperature, and later trials include fewer contributing observed trajectories.

## Panel i: reference provenance

The contextual-bandit comparison is digitized from the user-supplied TDGE figure, not newly generated mini_DG simulation data. The plotting table contains six LinUCB, five NeuralUCB and six NeuralTS seed summaries. TDGE means, baseline means and relative improvements reproduce the supplied summary; individual digitized seed values are approximate. No inferential test is added to this descriptive reference comparison.

## Final layout spacing

The 180-mm canvas, 6/7/8-pt typography and numerical axes are retained. The two b task plots match c’s plot-area top, bottom and height; the shared b x-label is shifted to retain its gap from the ticks. Other vertical geometry is retained. Horizontal positions and widths are adjusted slightly to enlarge gaps between rendered panel content, including ticks, axis titles, legends and schematic labels. The b–c gap increases from approximately 1.08 to 3.98 mm; all same-row gaps are at least 2 mm and increase by at least 0.65 mm relative to the preceding geometry. Exact measurements are recorded in fig6_full_joint90_horizontal_spacing.json. These are layout changes only; no fitting, replay or simulation is rerun.

## Source performance-first edition and current correction

The source edition is fig6_full_joint90_free_beta_reordered, which is preserved along with its code, CSVs and documentation. Only the third-row panel permutation changes: source g → new e (best performance), source e → new f (no-bonus phase slopes), source f → new g (fitted-parameter DIS and replay temperature). The source data and inferential tests are byte-identical. The compact third-row performance pair shares horizontal plot bounds with the fitting pair above; the phase panel aligns with c and the trajectory pair occupies the right column. Model scopes remain distinct.

## Retained Task 1 RL definitions (2026-10-12)

The retained Task 1 version is `rl_feature_item_20261012`. naiveRL maintains a value for each complete stimulus; fRL-decay maintains a dimension-by-feature value table. Both retain the original task-specific credit assignment. There is no uncertainty bonus, fitted reward baseline or dimension-attention mechanism in either RL baseline.

For item i with feature f(i,d) in dimension d, choice scores are

$$Z_i^{\mathrm{naive}}=Q(x_i)+\sigma H_i,\qquad
Z_i^{\mathrm{fRL}}=\sum_d[W_{d,f(i,d)}+\sigma G_{d,f(i,d)}]+\sigma H_i.$$

Each trial draws one independent standard Gumbel H per item. fRL additionally draws one standard Gumbel G per dimension–feature pair; items sharing a feature share that perturbation. G and H are independent. The same sigma multiplies both noise components in fRL, fixing their relative scale at one; no additional noise-ratio parameter is fitted. Value gain beta is fixed at one to avoid its scaling redundancy with sigma. These plain feature sums do not introduce a 1/D attention multiplier. naiveRL has no feature noise and no decay. Ranking yields three ordered groups of three in Task 1 (within-row permutations ignored), with chance 1/1,680. The former Task 2 policy is superseded by the V2 definitions below.

Learning uses pre-update values. In Task1, let v_i=Q(x_i) for naiveRL and v_i=sum_d W_df(i) for fRL. The prediction error is delta_i=r/9−v_i, with signed row coefficients c=(0.5,0,−0.5). naiveRL updates Q(x_i) by eta c_row(i) delta_i. fRL updates each item's constituent feature by the same increment, after multiplying W_df by [1−lambda(1−E_df)]. The original row-weighted exposure is E_df=sum_r (a_r/3) sum_(i in row r) 1[f(i,d)=f], a=(10,5,1)/16. No individual hidden feature reward is supplied.

### Retained Task 1 fitting and performance protocol

Task 1 includes 156 observed phase trajectories (51/51/54; 105 people). naiveRL has three free parameters (eta, initial Q, sigma); fRL has four (eta, initial W, sigma, decay). Fit bounds are eta [1e−4,1], sigma [1e−4,5], initial value [0,20] and decay [0,0.2]. Conditional on feature perturbations, independent item Gumbels yield an exact PL probability; a subset dynamic program sums within-row permutations. fRL integrates over feature particles, while naiveRL is scored analytically. No probability floor or extra lapse is added.

The saved Task 1 RL search used two starts, differential-evolution population 32 for 15 generations with 128 feature particles, then Powell refinement with 256 particles and 128 objective evaluations. Four candidates were selected using two independent 1,024-particle banks and independently scored with eight 16,384-particle banks. DGEM retains its 8,192-particle × 8-bank reference. Finite optimizer budgets do not establish global maxima. Participant-level summaries and Holm(3) tests remain byte-identical to the original Task 1 files.

Task 1 RL performance used 320 global proposals plus anchors and 160 local proposals, screening 64 episodes, selecting among 24 finalists on 256 independent episodes, then locking parameters before 1,000 fresh validation episodes per phase. Performance bounds were eta [1e−5,1], sigma [1e−6,5], initial value [0,15] and fRL decay [0,0.2]. The objective prioritized success, mean best score and capped completion time. Only the 4D phase (50-trial limit) is plotted. Full-score proportions are 0.799 for fRL-decay and 0.313 for naiveRL; DIS proportions are 0.0241010216 and 0.0149569571, respectively. Human proportions are DIS 0.4042499199 and success 0.4074074074; historical Bayesian values are 0.0236210181 and 0.25. Agents stop at first full score; DIS uses actual executed trials, with no post-stop padding. Both errors are saved within-4D SEMs, not phase-average SEMs. `performance_task1_4d.csv` is the active input; mixed-phase `performance_task1.csv` is historical.

Task 1 production sources are agents/rl_feature_item.py and agents/rl_feature_item_fast.py, with saved results under data/fitting/rl_feature_item_20261012 and data/best_performance/rl_feature_item_20261012. The previous Task 2 analysis from that run is retained only as history. Do not run the old build_rl_feature_item.py over the current tables; use import_v2_be.py for the approved frozen data transfer.

## Retained Task 2 DGEM

DGEM has the same seven free parameters as Task 1, including beta_init in [0,40]. Its feature uncertainty, smoothed uncertainty gate, Concrete attention, attentional inertia, shared feature Gumbels, attention-dependent residual item Gumbels and uncertainty-bonus decay are retained. Because Task 2 supplies one aggregate set score, its prediction is the unweighted selected-feature fraction sum; the prediction error uses (score-50)/50 and attention weights the selected-feature update. Exposure counts update with attention-weighted selected-feature fractions divided by (1+gate). There is no fabricated item-specific or Task 1 row-specific feedback. Five dimensions and four feature levels replace Task 1's three/four dimensions and three levels; the threshold/slope bounds are rescaled to the twenty-feature uncertainty total.

## Current Task 2 V2 alternatives, 15 October 2026

## Experimental-task mapping and sources

The supplied `reference/model_version2_1012_for_jiahui` bundle contains only the
five-dimensional experimental Task 2, with 16 available objects and an unordered
selection of four. Its analysis script uses **Task1** to mean its fitting analysis
and **Task2** to mean its optimal-parameter simulation analysis. Those internal
names are not the experiment's Task 1 and Task 2. We use the bundle's four V2
baselines only in experimental Task 2.

Experimental Task 1 is retained from the current formal Fig. 7 package. Its
fitting bars average each participant over their available 3D, 4D-E and 4D-NE
phases. Its performance panel is **4D only**, using the current saved 4D model
summaries and the independent 4D-NE human reference. These retained tables and
their statistical tests are verified byte-identical; they are not recomputed here.
The Task 1 fitting and performance panels therefore have different phase scopes.

For experimental Task 2, the DGEM fitting reference is the current seven-parameter
free-bonus fit, with fitted `beta_init`. All DGEM and human performance summaries
are retained from the current formal package. Best-performance parameters are
separate from fitted parameters; this package does not present them as recovery
simulations from human fits.

## Fitting data and probability definition

V2 baseline participant likelihoods are read from the saved
`analysis/task1_subject_likelihood.csv`; they are checked against the corresponding
subject-level fitted-parameter JSON log likelihoods and trial counts. No fitting
or likelihood replay is rerun. Each plotted participant value is
`exp(total_log_likelihood / n_trials)`; bars are its across-participant arithmetic
mean, and error bars are across-participant s.e.m. (`sd(ddof=1)/sqrt(n)`).

The V2 four baselines use one ActionScore-softmax over all `C(16,4)=1,820`
unordered four-object sets. The action score is the sum of the four object
values. DGEM retains its particle-integrated grouped Plackett–Luce probability
for the same unordered selected-set event. These are different choice laws over
the same observed outcome space, not the same latent policy. The chance lines
are 1/1,680 for Task 1 and 1/1,820 for Task 2.

The fitted parameter counts in experimental Task 2 are DGEM 7, fRL-decay 4,
naiveRL 3, Bayesian 1 and V2 ACL 5. V2 ACL includes `q0`; it is not the four-parameter
ACL from the previous formal comparison. This is an in-sample fit comparison,
without a parameter-complexity correction or a new held-out validation.

We checked all 80 subject identifiers and 4,111 trial records in the V2 processed
data against the current DGEM behavioural source. Trial keys, selected unordered
sets, displayed feedback, relevant dimensions, candidate order, and candidate
scores all match. Only participant-level plot inputs and cohort counts are
exported; raw behaviour, stimulus schedules, gaze tables and pickle files remain
in their original locations.

V2 ACL conditions on real same-trial gaze when available and uses uniform
attention when it is missing. Of the 4,111 trials, 3,811 have a positive gaze
record and 300 use the missing-record fallback; 75 participants have some real
gaze and five have none. All 80 participants and all 4,111 trials are included.
This differs from the previous formal ACL's 75-person, 3,736-valid-trial cohort. No old ACL
test or valid-gaze sample count is reused. The ACL result is labelled as a
gaze-conditional/fallback reference, not a purely behavioural baseline.

## Statistical comparisons

For Task 2, DGEM and each V2 behavioural baseline are paired by subject across
the same 80 participants. We retain the already computed two-sided Wilcoxon signed-rank tests
(`zero_method='wilcox'`, `method='auto'`) on participant geometric likelihoods,
with Holm correction across fRL-decay, naiveRL and Bayesian. The comparison
against ACL is a separate single two-sided paired comparison because it uses
additional contemporaneous gaze information with a uniform missing-data fallback.
The shared significance display does not imply one pooled test against all
alternatives. Stars are `* P<0.05`, `** P<0.01`, `*** P<0.001`.

The updated Task 2 tests are:

| Comparator | Pairs | W | Display P | Adjustment |
|---|---:|---:|---:|---|
| fRL-decay | 80 | 35 | 2.913292e-14 | Holm, 3 behavioural comparisons |
| naiveRL | 80 | 3 | 2.445708e-14 | Holm, 3 behavioural comparisons |
| Bayesian | 80 | 1 | 2.445708e-14 | Holm, 3 behavioural comparisons |
| ACL | 80 | 43 | 3.916274e-14 | Separate single conditional/fallback comparison |

The V2 source's original tests were against ACL rather than DGEM. They are
preserved in `csv/source_v2_likelihood_vs_acl_statistics.csv` for provenance,
but their stars are not used in this figure.

## Saved optimal-parameter simulation outcomes

The V2 source optimised full-score rate using 100 search games and saved a
separate set of 80 simulation games per baseline, with at most 60 trials each.
We use the saved `optimal/*_choices.csv` and `*_games_summary.csv`; no new
simulation or parameter optimisation is performed. A DIS trial is a selected
four-object set with a shared feature value in at least one dimension. For each
episode, DIS ratio is its number of DIS trials divided by its actually executed
trials, including the terminal successful trial. Successful episodes stop at
the first score of 100. We do not pad DIS ratios after stopping.

Performance points are means across the 80 episodes. Both horizontal DIS-ratio
and vertical binary full-score-rate uncertainties are s.e.m. across episodes,
calculated directly from the episode metrics. The input bundle's bootstrap 95%
intervals are retained separately in
`csv/source_v2_performance_bootstrap_ci.csv`; they are not relabelled as s.e.m.
The V2 full-score percentages are fRL-decay 85.0, naiveRL 11.25, Bayesian 65.0 and
ACL 65.0. The ACL optimal parameters have `u_et=0`, so its saved optimal simulation
has uniform dimension weights despite the supplied real-gaze sampling pool.

**Environment limitation:** V2 uses deterministic normalized score feedback.
The retained DGEM performance reference uses the current noisy feedback rule
(±1 or ±2 away from protected endpoints), while Human is observed behaviour.
DGEM's 1,000-agent and Human's 80-participant current references are retained.
All markers keep the original filled style at the user's request; source roles are explained here. This is a transparent version
display, not a controlled comparison under identical simulation environments or
a claim of global parameter optimality. No performance significance tests are
introduced.

### Import and redraw

`code/import_v2_be.py` copies exact CSVs and saved tests from `fig6_b_e_v2_comparison_20261015`; it does not execute its prepare_data.py, tests or model code. `v2_be_revision.json` maps file hashes. Frozen source documentation and its original analysis provenance are in `provenance/v2_be_20261015/`. Its statements that the source comparison is separate refer to that frozen source edition; the user subsequently authorized this Fig. 7 transfer. `code/render_fig6_full_joint90_free_beta.py` directly renders the full figure at 600 dpi. Original filled bar/point styles are retained at the user’s request. The e legend uses naiveRL and ACL without suffixes; equal y ticks are 0, 0.5 and 1. Source-role differences are documented in the caption rather than encoded by marker fill. Old ACL 75-person/3,736-trial tables and old RL Task 2 parameter tables are history, not inputs for current b/e. All other panel inputs are unchanged.

### Remaining recovery scope

Panel g is unchanged: DGEM uses individually fitted seven-parameter free-beta simulations and conditional temperature replay; the two RL curves remain historical references (including the earlier feature/no-decay naive label). They do not constitute recovery checks for the current b/e RL definitions. A future current-policy recovery run must use the new fitted parameters. All other panels retain their sources and tests. The complete pre-revision PNG/code/CSV/docs are saved under revisions/20261012_before_frl_feature_item_naive_item.

## Shared omitted-zero marks (2026-10-12)

Double slashes indicate an omitted zero-to-lower-bound interval. The native marks match Fig5f in physical size. They change decoration and, where necessary, x tick padding; plotted values, limits and tests are retained. Logarithmic scales and domains containing zero remain continuous.

Full range inset exception (2026-10-12): panel c’s inset intentionally has no double slashes at the user’s request. The main c plot and panel i retain their omitted-zero markers. Inset data and ranges are unchanged.

## Panel h horizontal translation (2026-10-12)

The native movie-selection schematic is translated1.5mm rightward without
rescaling its axis, altering its vertical placement or changing any label,
container, arrow or conceptual quantity. The lower-row h panel letter keeps
its original aligned anchor. The final plot left edge is6.0mm rather than
4.5mm; its width and height are retained. All statistical panels are unchanged.
Native diagram/text collision and final h-to-i spacing checks are rerun, and
the previous PNG and full package are retained in
`revisions/20261012_before_h_shift/`. No fitting, simulation or inference is
performed for this layout revision.

### Display update, 15 October 2026

DIS-frequency tick labels are converted from source proportions to percentages by a factor of 100; all native scientific coordinates, intervals and statistical results remain unchanged. Raw field names such as `round` are retained for source compatibility, while displayed temporal labels use Trial.
No-DIS/Non-DIS uses the Fig5 others base grey, #8D9399; transparent uncertainty/fill marks retain their existing alpha. No-DIS intervention and Non-DIS trial categories remain distinct scientific definitions.

## Simplified architecture display (15 October 2026; restored layout)

Panel a restores the box positions and connectors from the edition archived in `source_packages/20261015_before_editorial_layout`, retaining the stimulus-to-action pathway and illustrative weight bars. The feature-score label shows the deterministic component S_df = Q_df + beta_t U_df; the complete uncertainty definition and feature/item shocks remain as specified above.

The user-requested schematic removes the inertia coefficient and previous-attention term from the displayed relation:

$$w_{d,t}\propto\operatorname{softmax}_d(\mathbf U_t/\tau_t).$$

Here, $\mathbf U_t=(U_{1,t},\ldots,U_{D,t})$ is the vector of dimension uncertainties, $U_{d,t}=\sum_f U_{df,t}$. The scalar total uncertainty $U_t=\sum_d U_{d,t}=\sum_{df}U_{df,t}$ drives the temperature gate. The attention temperature $\tau_t$ is the quantity previously labelled $\tau_{\mathrm{att}}$. Time indices elsewhere in the diagram are suppressed for readability. The schematic shows that higher relative dimension uncertainty increases attention preference, while lower temperature concentrates weights. After normalization, the proportional expression is simply an equality because softmax already sums to one.

This is a deterministic special case, not a replacement for the fitted model or its conditional expected weight. It sets attention inertia $\lambda=0$, uniform lapse $\epsilon=0$, dimension shocks to zero and priority scale $T_U=1$. In this limit, normalizing $[\operatorname{softmax}_d(\mathbf U_t)]^{1/\tau_t}$ yields exactly $\operatorname{softmax}_d(\mathbf U_t/\tau_t)$. Multiplying the previous weight by this softmax would instead impose a different update; the implemented DGEM uses an additive mixture with previous attention before temperature scaling.

The complete implemented rule, including the priority scale $T_U$, lapse $\epsilon$ and standard Gumbel shocks $G_{d,t}$, remains

$$p_{d,t}=\operatorname{softmax}_d(\mathbf U_t/T_U),$$
$$\pi_{d,t}=(1-\epsilon)\left[\lambda w_{d,t-1}+(1-\lambda)p_{d,t}\right]+\epsilon/D,$$
$$w_{d,t}=\frac{\left(\pi_{d,t}e^{G_{d,t}}\right)^{1/\tau_t}}{\sum_{e=1}^{D}\left(\pi_{e,t}e^{G_{e,t}}\right)^{1/\tau_t}}.$$

Thus the compact figure isolates the uncertainty–temperature mechanism without changing the complete model; its previous-attention term remains in the equations above. The restored schematic connectors summarize information flow; they do not imply that dimension uncertainty influences attention exclusively through the scalar gate. Feature values do not directly enter the attention rule. No fitting, simulation, statistical result or empirical panel is changed.

The actual gate is raw_gate = sigmoid[s(threshold − U)], U = sum_df U_df; gate_t = alpha * raw_gate_t + (1 − alpha) * gate_(t−1). Temperature is the gate odds, gate/(1 − gate), clipped to [10⁻⁴, 10⁴]. Thus, for positive slope and fixed history, increasing U decreases temperature. The schematic’s U-up / tau-down relation reflects this sign; it does not claim an inverse-linear relationship. Source: agents/dgem_v11_9_4_3_core.py, prepare_choice and concrete_temperature_from_gate.

Panels c/d/f/g are labelled Task 1 explicitly. No model parameters, fitting results, simulated trajectories, confidence intervals or P values were recomputed.


## Task-scope labels, 2026-10-15

The data for b Task1, c, d, f and g contain 3D/P1, 4D-E/P2 and 4D-NE/P2-only. Their headers read Task 1; the phase composition remains documented here rather than repeated in those titles. Panel f retains all three exact phase labels. Panel e is Task 1 (4D), with the observed Human reference drawn specifically from 4D-NE. Model simulations in e are fresh independent 4D agents, not a pooled experience cohort. All previously cached c/d/f/g estimates, confidence intervals and P values remain unchanged.
