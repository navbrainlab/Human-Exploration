# Methods and statistical details for Fig. 6

## Panel h: hierarchical action construction

At trial $t$, DGEM uses the current user representation to rank feature dimensions and retains the top $K_1$. Within each retained dimension, it ranks the available features and retains the top $K_2$. The resulting active feature set is denoted by $F_t$.

The candidate pool contains every previously rated item together with a sampled subset of unrated items. Let $F(a)$ denote the features present in candidate item $a$, $s_t(f)$ the current relevance score of feature $f$, and $w_{a,f}$ the item–feature weight. Each candidate is scored as

$$
S_t(a)=\sum_{f\in F_t\cap F(a)} s_t(f)w_{a,f}.
$$

A candidate that shares no feature with $F_t$ receives a score of zero. The recommended item satisfies $a_t\in\arg\max_a S_t(a)$; ties are resolved randomly.

After observing reward $r_t$, the active features for learning are the intersection of the selected features and the features present in the recommended item, $F_t\cap F(a_t)$. Reward is propagated to those active features and their contributing dimensions. The dimension update uses the maximum active feature relevance within that dimension, whereas the feature update uses the corresponding item–feature relevance.

## Panels b–g: summaries, uncertainty and statistical tests

**b, Model fitting.** Task 1 first averages each participant’s geometric likelihood per trial across the three phases (3D, 4D-E and 4D-NE), then computes group means and s.e.m. Task 1 comparisons versus DGEM use two-sided paired Wilcoxon tests with Holm correction. Task 2 uses the supplied all-game-mode means, standard errors and p-versus-DGEM values; its test specification is not documented in the supplied table. Those values are fRL, 4.001513232334431 × 10⁻⁷; naiveRL, 2.0380473891232933 × 10⁻⁶; Bayesian, 6.482994004557716 × 10⁻⁷; and ACL, 6.661101011777581 × 10⁻⁷. The shared bracket abbreviates separate comparisons with DGEM, not a test against a pooled group of alternative models. Chance likelihood is 1/1,680. The plot-facing Task 1 table retains the significance categories; exact Task 1 P values are not present in that reduced table.

**c, Attention.** Each grey point represents a trial in fitted-data replay. Purple points are temperature-bin means, with participant-clustered 95% confidence intervals. The inset displays the full temperature range. These observations come from replay of fitted human trials, rather than newly simulated behaviour.

**d, DIS and attention.** The upper panel uses the saved binomial-logit curve and its 95% confidence interval. The lower distributions describe trial-level maximum attention weights for DIS and non-DIS actions. The inferential unit is the participant: a two-sided paired Wilcoxon test compares within-participant means across action classes (105 pairs, W = 98, P = 9.337056 × 10⁻¹⁸).

**e, Best-performance simulations.** The new DGEM points are from the full V11.9.4.3 model with direct initial feature-uncertainty bonus fixed at 20. They are **not** from the no-bonus model or participant-fitted simulations. A threshold-focused five-parameter search first screened 64 cases and selected on 200 disjoint cases for each task. The 3D 500-case holdout narrowly missed the strict 90% full-score target on a later independent 5,000-case validation (89.96%); it was not used for tuning. A separately planned six-parameter 3D refinement, including the original beta-decay parameter, used new 96-case screening and 500-case selection banks, then locked one candidate for an untouched 5,000-case validation. The selected threshold was 3.926 and beta decay 0.872. It achieved DIS 0.9158 (agent bootstrap 95% CI 0.9144–0.9171) and full-score proportion 0.9538 (Wilson 95% CI 0.9476–0.9593). The 4D full-model candidate, threshold 4.063 and beta decay 0.971, was validated independently on 5,000 agents: DIS 0.9104 (bootstrap 95% CI 0.9089–0.9118) and full-score proportion 0.9268 (Wilson 95% CI 0.9192–0.9337). Both outcomes in both tasks have 95% interval lower bounds above 0.90. Means and s.e.m. in the figure use each agent as the unit. The alternative-model and human points remain the original reference cohorts (100 simulated agents for the Task 2 alternatives); they are not matched to these new DGEM case banks, so no direct paired tests are claimed. Exact parameter vectors, case IDs, scripts and SHA-256 hashes are in the dated run manifests. These are exploratory best-performance simulations; the fitting and fitted-parameter simulation panels remain unchanged.

**f, Effective attention temperature.** Each entity contributes its mean effective attention temperature: one value per simulation agent or participant. Each dimensionality contains 100 simulation agents and 51 participants. Violin densities are estimated in the log domain; embedded boxes show medians and interquartile ranges, with whiskers reaching observations within 1.5 interquartile ranges. Two-sided Mann–Whitney U tests compare simulation and fitted-data replay (3D: U = 156, P = 4.637194 × 10⁻²¹; 4D: U = 640, P = 5.786386 × 10⁻¹⁴). Exact values are retained in `csv/panel_statistics.csv`.

**g, Round-wise trajectories.** Human behaviour and fitted-parameter simulations are averaged across the three Task 1 phases (source labels P1, P2 and P2-only). Curves use seven-round moving means; shaded bands show ±1 s.e.m. The lower trace shows the normalized effective attention temperature of DGEM, also as mean ±1 s.e.m. The saved plotting summaries are `csv/dis_by_round.csv` and `csv/temperature_by_round.csv`.

## Panel i: reference provenance

The contextual-bandit comparison is digitized from the user-supplied TDGE figure, not newly generated mini_DG simulation data. The plotting table contains six LinUCB, five NeuralUCB and six NeuralTS seed summaries. TDGE means, baseline means and relative improvements reproduce the supplied summary; individual digitized seed values are approximate. No inferential test is added to this descriptive reference comparison.
