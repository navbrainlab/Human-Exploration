# Fig. 6 Results and caption

## Results

Motivated by the gaze selectivity accompanying DIS, we developed the Dimension-Guided Exploration Model (DGEM) to examine how uncertainty-guided dimensional attention could generate DIS (Fig. 6a). Summed feature uncertainty sets dimensional priorities, while total uncertainty controls a trial-specific attention temperature $\tau_t$. The attention prior $\pi_{d,t}$ combines uncertainty-based priority, previous attention and a uniform lapse component. Dimension attention weights are sampled as

$$
w_{d,t}=\frac{\exp[(\log\pi_{d,t}+G_{d,t})/\tau_t]}{\sum_{e=1}^{D}\exp[(\log\pi_{e,t}+G_{e,t})/\tau_t]}.
$$

Here, $D$ is the number of stimulus dimensions and $G_{d,t}$ are independent standard Gumbel shocks. Lower temperatures concentrate each attention sample on a dimension, whereas higher temperatures distribute attention across dimensions. Attention then weights utilities derived from feature values and uncertainty:

$$
\begin{aligned}
s_{df,t}&=Q_{df,t}+\beta_t u_{df,t}+\sigma_{\mathrm{choice}}Z_{df,t},\\
v_{j,t}&=\sum_{d=1}^{D}w_{d,t}s_{d,f_d(j),t}.
\end{aligned}
$$

Here, $Q_{df,t}$ and $u_{df,t}$ denote learned feature value and uncertainty, $\beta_t$ controls the exploration bonus, and $f_d(j)$ identifies stimulus $j$'s feature on dimension $d$. The scale $\sigma_{\mathrm{choice}}$ controls standard Gumbel feature shocks $Z_{df,t}$, each drawn once per trial and shared by all stimuli containing that feature.

Independent Gumbel stimulus noise is then added with scale $\sigma_{\mathrm{item},t}=\sigma_{\mathrm{choice}}\sqrt{1-\sum_{d=1}^{D}w_{d,t}^{2}}$, which decreases as attention concentrates. Ranking the perturbed stimulus utilities and assigning successive groups of three to ordered rows produces an action. In the single-dimension limit, stimulus noise vanishes and stimuli sharing the attended feature have identical utility; because each feature occurs three times, distinct feature utilities yield DIS. Feedback updates feature values and exposure-based uncertainty according to attention. Complete learning and choice rules are specified in Methods.

In Task 1, the seven-parameter DGEM assigned higher likelihood to human choices than the tested alternatives. Mean geometric trial likelihood was 0.0170 for DGEM, compared with 0.00206–0.00432 for the alternatives. Participant-paired Wilcoxon comparisons favoured DGEM (all Holm-adjusted P ≤ 2.18 × 10⁻¹⁶; Fig. 6b). The retained Task 2 reference also favoured DGEM (reported P ≤ 2.04 × 10⁻⁶), although its initial bonus was fixed.

Replaying participants’ observed trials linked the model’s attentional state to behaviour. Lower effective attention temperatures were associated with more concentrated dimension weights (Fig. 6c). Greater maximum dimension weight was associated with a higher probability of observed DIS. A participant-clustered binomial-logit model supported this relationship (slope = 7.68, P = 3.90 × 10⁻³⁵; Fig. 6d).

With parameters selected for performance and a fixed initial bonus, DGEM achieved mean DIS proportions of 91.6% in 3D and 91.0% in 4D. Full-score rates reached 95.4% and 92.7% (5,000 independently simulated agents per task; Fig. 6e). In the separate no-bonus model, fitted temperature-gate slopes were higher in 4D-E than in 3D or 4D-NE. The corresponding Holm-adjusted P values were 6.00 × 10⁻⁵ and 0.00336 (Fig. 6f). Evidence for a 3D–4D-NE difference was inconclusive (adjusted P = 0.703).

Simulations using participants’ fitted parameters reproduced the broad decline in DIS during learning (Fig. 6g). A separate replay, pooled over available observed trials, showed increasing effective attention temperature across rounds. This descriptive correspondence was consistent with a shift towards more diffuse dimensional attention as learning progressed.

We also examined the dimension-guided principle in task-dimension-guided exploration (TDGE), a separate movie-recommendation model [TDGE reference]. Semantic clusters of movie tags served as dimensions, and individual tags served as features. The recommender selected dimensions and tags before retrieving and scoring candidate movies (Fig. 6h). Digitized MovieLens benchmark results showed 13.1%, 11.3% and 13.5% higher mean rewards than Linear UCB, Neural UCB and Neural Thompson sampling, respectively (Fig. 6i). These descriptive gains extend the dimension-guided principle to recommendation through a different action-selection mechanism.

## Figure caption

**Fig. 6 | A computational account of dimension-guided exploration.** **a,** Feature uncertainty guides dimensional attention; total uncertainty controls attention temperature. Attention weights feature utilities before items are ranked into three rows; colours illustrate DIS. **b,** Mean geometric trial likelihood: seven-parameter Task 1 fits (free initial uncertainty-bonus strength, β₀) and original Task 2 reference (β₀ fixed); dashed lines, chance (1/1,680). **c,** Effective attention temperature versus maximum dimension weight during replay (5,323 trials). Grey points denote trials; purple points, bin means and participant-clustered bootstrap 95% confidence intervals; inset, full temperature range. **d,** Observed DIS probability (binomial-logit fit and participant-clustered 95% confidence band, upper) and trial distributions with medians and interquartile ranges (lower), against maximum dimension weight. **e,** Full-DGEM performance simulations (β₀ = 20; 5,000 agents per task), with human and alternative-model reference cohorts. **f,** No-bonus temperature-gate slopes (β₀ = 0; n = 51, 51 and 54 for 3D, 4D-E and 4D-NE). Boxes show medians, interquartile ranges and 1.5-IQR whiskers; points and outliers are omitted. **g,** Descriptive DIS trajectories (upper; 3,120 fitted-parameter DGEM simulations) and normalized fitted-data replay temperature (lower); seven-round moving means. **h,** TDGE recommendation: semantic tag clusters define dimensions; movie tags define features. Selected tags guide retrieval before item-level scoring; Movie B is illustrative. **i,** Digitized MovieLens TDGE reference [TDGE reference]. Points denote approximate seed summaries (n = 6, 5 and 6 for LinUCB, NeuralUCB and NeuralTS); diamonds, reported TDGE means; dashed lines, baseline means.

Task 1 comprises 156 participant-phase sequences from 105 people. Two-sided tests in Task 1 b, d and f are paired Wilcoxon with Holm correction in b (105 RL pairs; 92 Bayesian pairs), participant-paired Wilcoxon in d (105 pairs), and Holm-adjusted paired Wilcoxon (3D–4D-E) or independent Mann–Whitney U (other comparisons) in f. Task 2 significance follows reference summaries. Bars/bands in b, e and g denote s.e.m. Exact statistics are in Methods and Source Data. \*P < 0.05, \*\*P < 0.01, \*\*\*P < 0.001; ns, not significant. DGEM, Dimension-Guided Exploration Model; TDGE, task-dimension-guided exploration; DIS, dimension-invariant search; 4D-E/4D-NE, 4D with/without prior 3D experience; s.e.m., standard error of the mean.
