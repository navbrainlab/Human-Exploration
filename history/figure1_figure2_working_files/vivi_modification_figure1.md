# vivi_modification.md — Figure 1 panels e & f 组间统计检验修改记录

日期：2026-09-30

## 1. 修改目标

对 Figure 1 的图 e（Task 1）和图 f（Task 2）增加组间差异统计检验并在图上标注，按 *Nature Human Behaviour* (NHB) 出版标准执行：

- **图 e**：4D-E（P2，n=51）vs 4D-NE（P2-only，n=54）——Mean Score on trial progression（学习曲线）与 Best score（分布）
- **图 f**：Obs（FDS-Obs，n=40）vs NObs（FDS-NoObs，n=40）——Mean Score on trial progression 与 Best score

## 2. 统计检验方案（及选择理由）

### 2.1 Best score（独立样本两组比较）
- **检验**：双侧 **Mann–Whitney U 检验**（`scipy.stats.mannwhitneyu`, two-sided, method='auto'）。
- **理由**：得分有上界（存在 100 分天花板效应），分布不服从正态假设；样本量为中等（n=40~54），非参数检验稳健且不依赖分布假设，符合 NHB 对行为数据的要求。
- **效应量**：rank-biserial correlation *r*（有符号，正号表示第一组得分更高），与 P 值一并输出到统计表。

### 2.2 Mean Score on trial progression（试次级重复测量比较）
- **检验**：逐试次 Mann–Whitney U（cluster-forming α = 0.05）+ **cluster-based permutation test**（10,000 次组标签置换，cluster mass = Σ−log₁₀(p)，置换零分布取 max cluster mass）。
- **理由**：学习曲线是试次级重复测量，逐点检验存在多重比较问题。Cluster-based permutation 控制族系错误率（FWE），是时序/神经科学数据（含学习曲线类）的主流严谨做法，比逐点 BH-FDR 更保守、更受审稿人认可。
- **统计所用数据**：与绘图完全一致——同一批经「首次满分后按组最大试次补齐」扩展后的逐试次逐被试分数（未使用平滑后的曲线值做检验）。

### 2.3 图上标注方式（NHB 风格）
- **分布面板**：被比较两组之间画显著性括号线（bracket），标注精确 P 值（*P* = 0.003 格式；*P* < 0.001 时写 `P < 0.001`；P ≥ 0.05 时标注 `n.s.`）；左下角斜体注明检验名称「Mann–Whitney U (two-sided)」。NHB 要求报告精确 P 值而非仅以星号代替，故未采用 */**/*** 星号体系。
- **学习曲线面板**：显著试次簇以图内左上角文字块标注（如「trials 19–24: *P* = 0.026」），无显著簇时明确标注「no significant between-group clusters」；避免在密集曲线上叠加元素。
- 所有标注使用 Arial、斜体 *P*，与 `publication_plot_style.py` 的出版样式一致。

## 3. 对脚本的修改操作

**未改动任何原有文件**。所有修改以新增文件实现：

| 操作 | 文件 | 说明 |
|---|---|---|
| 新建 | `figure1/code/figure1_integrated_four_panels_with_stats.py` | 基于原 notebook `figure1_integrated_four_panels.ipynb` 的全部绘图逻辑（数据加载、首次满分补齐、滑动平均曲线、raincloud 分布）改写为独立脚本，并新增：①`best_score_mann_whitney()`（MWU + rank-biserial r）；②`learning_curve_cluster_test()`（逐试次 MWU + cluster-based permutation, 10,000 次置换, seed=42）；③绘图函数新增统计标注参数（`test_pair`/`best_score_result`/`significant_clusters`）；④`score_ylim` 由 (70, 102) 扩展为 (70, 104) 以容纳显著性括号线 |
| 新建 | 本文件 `vivi_modification.md` | 修改记录 |

复现命令（项目根目录执行，约 3 分钟，含 2×10,000 次置换）：

```bash
python3 figure1/code/figure1_integrated_four_panels_with_stats.py
```

## 4. 生成的新文件（均在 `figure1/output/`，未覆盖任何原图）

### 图（全部 300 dpi，PNG + 组合图另存 PDF）
- `figure1_integrated_four_panels_with_stats.png` / `.pdf` — 带统计标注的 1×4 组合图
- `figure1_panel_e_task1_learning_curve_with_stats.png` — 图 e 学习曲线
- `figure1_panel_e_task1_best_score_distribution_with_stats.png` — 图 e Best score
- `figure1_panel_f_task2_learning_curve_with_stats.png` — 图 f 学习曲线
- `figure1_panel_f_task2_best_score_distribution_with_stats.png` — 图 f Best score

### 统计结果表（CSV，可直接用于论文 Methods/图注）
- `figure1_best_score_group_tests.csv` — 两次 MWU 检验的完整结果（U、P、rank-biserial r、中位数/均值、n）
- `figure1_panel_e_task1_learning_curve_trial_stats.csv` — 图 e 逐试次统计（每试次 P、各组 n、中位数、所属簇）
- `figure1_panel_e_task1_learning_curve_clusters.csv` — 图 e 全部试次簇及簇水平置换 P
- `figure1_panel_f_task2_learning_curve_trial_stats.csv` — 图 f 逐试次统计
- `figure1_panel_f_task2_learning_curve_clusters.csv` — 图 f 全部试次簇及簇水平置换 P

## 5. 统计结果摘要

### Best score（Mann–Whitney U, two-sided）
| 面板 | 比较 | n | U | P | rank-biserial r |
|---|---|---|---|---|---|
| e (Task 1) | 4D-E vs 4D-NE | 51 vs 54 | 1629.0 | 0.081 | −0.183（4D-E 略低，不显著） |
| f (Task 2) | Obs vs NObs | 40 vs 40 | 924.5 | 0.219 | −0.156（Obs 略低，不显著） |

→ 两个 Best score 比较均为 **n.s.**（图中括号线已标注 n.s.）。

### Mean Score on trial progression（cluster-based permutation, 10,000 permutations）
| 面板 | 比较 | 显著簇 | 簇水平 P |
|---|---|---|---|
| e (Task 1) | 4D-E vs 4D-NE | trials 19–24 | 0.026 |
| | | trials 26–31 | 0.032 |
| f (Task 2) | Obs vs NObs | 无显著簇（最小簇 P = 0.095, trials 43–44） | — |

→ Task 1 的 4D-E 组在学习中段（约第 19–31 试次）显著高于 4D-NE 组；Task 2 两组学习曲线差异未达显著（尽管 Obs 组均值全程略高，误差带重叠明显）。

## 6. 供论文 Methods/图注使用的表述建议

> **Panel e**: Between-group differences in best score were assessed with a two-sided Mann–Whitney U test. Trial-by-trial group differences in learning curves were assessed with a cluster-based permutation test (10,000 group-label permutations; cluster-forming threshold P < 0.05, Mann–Whitney U per trial; cluster-level α = 0.05). 4D-E participants significantly outperformed 4D-NE participants during the middle phase of learning (trials 19–24, P = 0.026; trials 26–31, P = 0.032), whereas best scores did not differ significantly between groups (P = 0.081).
>
> **Panel f**: The same procedures were applied: best scores did not differ between Obs and NObs groups (P = 0.219), and no significant trial clusters were detected in the learning curves (cluster-based permutation test).

（注：Task 2 的 score 为含噪得分 `score_noisy`，与原图保持一致；P1/3D-NE 组仅用于 Task 1 曲线展示，不参与组间检验——如需三组间检验可另行补充 Kruskal–Wallis + post-hoc。）


---

# 追加记录（2026-09-30）：满分率 Fisher 精确检验 + 首次 ≥95 分 KM/log-rank 生存分析

## 7. 新增分析目标

在 Best score 组间差异不显著（图 e: P = 0.081；图 f: P = 0.219）后，追加两个补充分析：

1. **满分率**（窗口内是否获得过 100 分）：4D-E vs 4D-NE、Obs vs NObs，Fisher 精确检验。
2. **首次获得 ≥95 分的轮数**（time-to-event）：KM 生存曲线 + log-rank 检验，未达标者按删失处理。

## 8. 统计方法

- **满分率**：被试水平二分类（窗口内 `max(score) == 100`），组间比较用 **Fisher 精确检验**（双侧；`scipy.stats.fisher_exact`），报告优势比 OR、精确 P。误差线为 **Wilson score 95% CI**（小样本率区间比正态近似更稳）。每根柱标注 `k/n (%)`。
- **首次 ≥95 分轮数**：事件 = 窗口内首个 `score ≥ 95` 的试次；**从未达标者在其最后一个可用试次删失**（避免仅比较有达标者造成的幸存者偏差）。KM 曲线（Greenwood SE 置信带），组间差异用 **双侧 log-rank 检验**（手工实现 O−E 方差公式，χ²(1)）；报告中位达标时间（未达 50% 时报告 "not reached"）与达标比例。
- **统一分析窗口**：Task 1 ≤ 50 轮、Task 2 ≤ 60 试次（与主图一致），保证组间机会均等；试次不足者按删失/其实际试次计入。P1（3D-NE）仅展示不参与检验（其任务仅 30 轮，与 50 轮窗口不可比）。

## 9. 对脚本的修改操作

**未改动任何已有文件**，新增：

| 操作 | 文件 | 说明 |
|---|---|---|
| 新建 | `figure1/code/figure1_perfect_rate_and_first95_survival.py` | 满分率（`perfect_rate_by_group` + `fisher_test_rates` + `wilson_ci`）、KM/log-rank（`km_fit` + `logrank_test`）、两套绘图函数（`plot_perfect_rate_panel` / `plot_survival_panel`，复用原样式系统与配色）。数据加载复用 `figure1_integrated_four_panels_with_stats.py` 的 `load_task1/load_task2` |

复现命令：

```bash
python3 figure1/code/figure1_perfect_rate_and_first95_survival.py
```

## 10. 生成的新文件（均在 `figure1/output/`，不覆盖任何已有图）

### 图（300 dpi，配色/字体/轴线样式与原 Figure 1 面板一致）
- `figure1_perfect_score_rate_panels_with_stats.png` — 满分率 1×2 组合图
- `figure1_first95_km_survival_panels_with_stats.png` — KM 生存曲线 1×2 组合图
- `figure1_panel_e_task1_perfect_score_rate_with_stats.png` / `figure1_panel_f_task2_perfect_score_rate_with_stats.png` — 单面板
- `figure1_panel_e_task1_first95_km_with_stats.png` / `figure1_panel_f_task2_first95_km_with_stats.png` — 单面板

### 统计表（CSV）
- `figure1_perfect_score_rate_tests.csv` — 各组 n、满分数、满分率、Wilson CI 及 Fisher OR/P
- `figure1_first95_logrank_tests.csv` — 各组 n、事件数、达标比例、中位达标时间及 log-rank χ²/P
- `figure1_first95_km_curves.csv` — KM 曲线逐点数据（time, survival, SE, 组，面板）

## 11. 统计结果摘要

### 满分率（Fisher 精确检验，双侧）
| 面板 | 组 | 满分率 | 比较 | OR | P |
|---|---|---|---|---|---|
| e (Task 1) | 4D-E | 32/51 (62.7%) | 4D-E vs 4D-NE | 2.45 | **0.032**（显著） |
| | 4D-NE | 22/54 (40.7%) | | | |
| | (3D-NE 参考) | 17/51 (33.3%) | | | |
| f (Task 2) | Obs | 18/40 (45.0%) | Obs vs NObs | 1.91 | 0.248（n.s.） |
| | NObs | 12/40 (30.0%) | | | |

→ **Task 1 中 4D-E 组满分率显著高于 4D-NE 组**，为组间差异提供了 best score 之外的显著证据；Task 2 方向一致（Obs 更高）但未达显著。

### 首次 ≥95 分轮数（KM + log-rank）
| 面板 | 组 | 达标比例 | 中位轮数 | log-rank P |
|---|---|---|---|---|
| e (Task 1) | 4D-E | 43/51 (84%) | 23 | 0.126（n.s.） |
| | 4D-NE | 43/54 (80%) | 32 | |
| f (Task 2) | Obs | 18/40 (45%) | 未达到 | 0.181（n.s.） |
| | NObs | 13/40 (33%) | 未达到 | |

→ 两组均 n.s.：Task 1 中 4D-E 中位快约 9 轮（23 vs 32）但变异大；Task 2 两组中位均未达到（>50% 被试未在窗口内达标 95 分）。

## 12. 供论文 Methods/图注使用的表述建议

> **Perfect-score rate** (supplementary to Fig. 1e,f): For each participant we determined whether a perfect score (100) was achieved within the analysed window (Task 1: rounds 1–50; Task 2: trials 1–60). Between-group differences in the proportion of perfect scorers were assessed with two-sided Fisher's exact tests; error bars show 95% Wilson score confidence intervals. In Task 1, the perfect-score rate was significantly higher in the 4D-E group than in the 4D-NE group (62.7% vs 40.7%, odds ratio = 2.45, P = 0.032); in Task 2 the difference was not significant (Obs 45.0% vs NObs 30.0%, P = 0.248).
>
> **Time to first score ≥ 95** (supplementary to Fig. 1e,f): The first trial on which each participant scored at least 95 was analysed as a time-to-event outcome (Kaplan–Meier estimator; participants who never reached 95 within the window were censored at their last available trial). Between-group differences were assessed with two-sided log-rank tests. Neither comparison reached significance (Task 1: P = 0.126, median 23 vs 32 rounds; Task 2: P = 0.181, medians not reached).


---

# 追加记录（2026-09-30 晚）：敏感性/探索性分析（卡方、≥90 阈值、个人最好成绩轮数）

## 13. 分析内容

1. 满分率的 **卡方检验**（含/不含 Yates 校正），与主分析 Fisher 精确检验对照。
2. 「首次 ≥95 分」改为「**首次 ≥90 分**」：KM + log-rank（窗口与删失规则不变：Task 1 ≤50 轮、Task 2 ≤60 试次）。
3. **首次达到个人最好成绩（personal best）的轮数**：窗口内取每个被试的最高分首次出现的轮次（越低说明越快学到自身最好水平，所有人都有定义、无删失）；组间 **Mann–Whitney U**（双侧，+ rank-biserial r）。

## 14. 脚本与输出

| 操作 | 文件 | 说明 |
|---|---|---|
| 新建 | `figure1/code/figure1_sensitivity_tests.py` | 三个分析一次运行，复用既有数据加载与统计函数 |
| 新建 | `figure1/output/figure1_sensitivity_analyses.csv` | 全部统计结果（各组描述统计 + 检验统计量与 P 值） |

复现：`python3 figure1/code/figure1_sensitivity_tests.py`

## 15. 结果摘要

### 15.1 满分率：卡方 vs Fisher
| 面板 | 比较 | 卡方 P（Yates 校正） | 卡方 P（未校正） | Fisher P（主分析） |
|---|---|---|---|---|
| e (Task 1) | 4D-E vs 4D-NE | 0.039 | 0.024 | 0.032 |
| f (Task 2) | Obs vs NObs | 0.248 | 0.166 | 0.248 |

→ **换成卡方检验，Obs vs NObs 依然不显著**（45% vs 30% 的差异量本身不足以达到显著）；Task 1 结论不变。卡方未校正版 P 值偏小是已知偏差，n≈40 时仍推荐 Fisher/Yates 版作为主报告。

### 15.2 首次 ≥90 分（KM + log-rank）
| 面板 | 组 | ≥90 达标比例 | 中位轮数 | log-rank P |
|---|---|---|---|---|
| e (Task 1) | 4D-E | 94.1% | 10 | 0.380（n.s.） |
| | 4D-NE | 98.1% | 15 | |
| f (Task 2) | Obs | 80.0% | 31 | **0.046（显著）** |
| | NObs | 62.5% | 47 | |

→ 阈值放宽到 90 后，**Task 2 的 Obs 组首次达标速度显著快于 NObs**（中位 31 vs 47 试次）；Task 1 因两组几乎都很快达到 90（94–98% 达标）而无差异。

### 15.3 首次达到个人最好成绩的轮数（MWU）
| 面板 | 组 | 中位轮数 (IQR) | 均值±SD | MWU P | rank-biserial r |
|---|---|---|---|---|---|
| e (Task 1) | 4D-E | 26 (18–36) | 27.0±9.7 | **0.021** | 0.262 |
| | 4D-NE | 33.5 (23–40) | 32.0±11.2 | | |
| f (Task 2) | Obs | 31.5 (20–43) | 32.4±15.5 | **0.045** | 0.261 |
| | NObs | 43.5 (28–50) | 39.1±14.4 | | |

→ **两个任务均显著**：4D-E 与 Obs 组被试显著更早达到自己的最好成绩。

## 16. 解读与注意事项

- 现在图 e、图 f 各自都有显著的组间证据：图 e 有学习曲线中段簇（P = 0.026/0.032）、满分率（P = 0.032）、个人最好成绩轮数（P = 0.021）；图 f 有首次 ≥90 分速度（P = 0.046）与个人最好成绩轮数（P = 0.045）。
- ⚠️ 15.2 与 15.3 的 Task 2 结果 P 值在 0.04–0.05 边缘，且属于探索性分析；若同一图中报告多个检验，建议在文中说明为探索性结果，或考虑对单一任务内的检验家族做 BH-FDR 校正后再下结论（校正后 Task 2 的 P ≈ 0.045 两个检验均在 0.05 附近，结论可能不稳定）。
- 卡方与 Fisher 结论一致，主分析维持 Fisher 精确检验即可。


---

# 追加记录（2026-09-30 晚 2）：最大任务轮数 + 首次 ≥85 分

## 17. 分析内容与方法

1. **最大任务轮数**（每名被试实际完成的轮数 = 其最后一个有记录的试次编号，原始数据、不做窗口截断）：组间 **Mann–Whitney U**（双侧 + rank-biserial r）。若任务设有"掌握后提前结束"的停止规则，完成轮数越少可解释为越快达到掌握标准。
2. **首次 ≥85 分**：KM + log-rank（窗口与删失规则同前）。

## 18. 结果

### 最大任务轮数（MWU）
| 面板 | 组 | 中位轮数 (IQR) | 均值±SD | MWU P | r |
|---|---|---|---|---|---|
| e (Task 1) | 4D-E | 36 (18–50) | 33.9±14.7 | **0.007** | 0.286 |
| | 4D-NE | 50 (32–50) | 41.2±12.6 | | |
| f (Task 2) | Obs | 60 (35–60) | 48.7±15.6 | 0.109 | 0.178 |
| | NObs | 60 (54–60) | 54.0±12.0 | | |

→ **Task 1 显著**：4D-E 组完成任务的轮数显著少于 4D-NE（中位 36 vs 50，方向符合"轮数越少学习越有效"的假设）；Task 2 不显著（Obs 完成轮数略少，36% 的 Obs 被试提前结束）。

### 首次 ≥85 分（log-rank）
| 面板 | 组 | ≥85 达标率 | 中位轮数 | log-rank P |
|---|---|---|---|---|
| e (Task 1) | 4D-E | 100% | 6 | 0.063（趋势，n.s.） |
| | 4D-NE | 100% | 9 | |
| f (Task 2) | Obs | 82.5% | 29 | 0.192（n.s.） |
| | NObs | 75.0% | 41 | |

→ **Task 1 不显著**（P = 0.063，边缘趋势，4D-E 中位快 3 轮）：85 分门槛对 Task 1 太低，两组被试 100% 都能达到，仅速度有微弱差异。Task 2 也不显著。

## 19. 解读与注意事项

- 阈值扫描小结（Task 1 / Task 2 首次达标的 log-rank P）：≥85 → 0.063 / 0.192；≥90 → 0.380 / **0.046**；≥95 → 0.126 / 0.181。阈值并非越低越显著：门槛过低时两组几乎无差别（天花板），Task 2 的显著性恰好出现在 90 分这一最能区分两组的水平。这属于多阈值探索，正式报告中建议只预设一个阈值（或明确标注为探索性），避免「挑显著阈值」的质疑。
- 最大任务轮数在 Task 1 显著（P = 0.007）且方向符合假设，是图 e 组间差异的又一证据；但该指标受任务停止规则影响，论文中引用前建议核实实验程序（被试是固定 50 轮还是达到标准可提前结束），并在 Methods 中写明规则。
- 累计来看，图 e（Task 1）的组间差异证据非常强且一致：学习曲线簇（P=0.026/0.032）、满分率（P=0.032）、最好成绩轮数（P=0.021）、完成轮数（P=0.007）；图 f（Task 2）证据较弱且集中在边缘显著（≥90 速度 P=0.046、最好成绩轮数 P=0.045），建议作为探索性结果报告并考虑多重比较校正说明。


---

# 追加记录（2026-10-08）：Figure 1 完整组装（panel a–j）

## 20. 目标版式

```
              Task 1                              Task 2
  ┌──────────────────────────┐      ┌──────────────────────────┐
a │ Task 1 实验设计图        │ b    │ Task 2 实验设计图        │
  ├──────────────────────────┤      ├──────────────────────────┤
c │ Task 1 分组流程          │ d    │ Task 2 分组流程          │
  ├──────────┬──────────┬────┤      ├──────────┬──────────┬────┤
e │ 学习曲线  │ f 满分率 │ g  │      │ h 学习曲线│ i 满分率 │ j  │
  │ (permut.) │ (Fisher) │ 最好成绩轮数(MWU)     │(permut.) │(Fisher)│ 最好成绩轮数(MWU)
  └──────────┴──────────┴────┘      └──────────┴──────────┴────┘
```

## 21. 实现

| 操作 | 文件 | 说明 |
|---|---|---|
| 新建 | `figure1/code/figure1_full_assembly.py` | 计算全部统计（cluster permutation 10,000 次 ×2、Fisher ×2、MWU ×2），绘制 e–j 六个统计子图（完全复用原样式系统与配色），与 panel a–d 设计图组装为整图 |
| 新建 | `figure1/output/figure1_full_panels_a_j_with_stats.png/.pdf` | 整图成品（PNG 300 dpi + 矢量 PDF），版面 16×13.2 in，行布局 a/b、c/d、e–g、h–j，含 Task 1/Task 2 标题与 a–j 面板标签 |

复现：`python3 figure1/code/figure1_full_assembly.py`（约 3 分钟，含置换检验）。

panel b/c/d 素材渲染：因 PowerPoint 中正打开用户自己的演示文稿（避免干扰未走 GUI 自动化），改用 macOS 内置 `qlmanage`（Quick Look）将 `grouping_design_editable.pptx`、`task2_design_editable.pptx` 渲染为 PNG 后裁剪（PIL 自动去白边 + 手工区域裁剪，已剔除缺失链接图片的占位框）。

## 22. 已知的素材缺口与注意事项

1. **panel a（Task 1 实验设计图）源文件不在本工作文件夹中**（仅有 panel b 的 `task2_design_editable.pptx` 与 c/d 的 `grouping_design_editable.pptx`）。当前整图中 panel a 为灰色占位框。**请将 Task 1 设计图导出为 PNG 放到 `figure1/output/_pptx_render/task1_design.png`，重跑组装脚本即可自动嵌入。**
2. panel b 的 Quick Look 渲染中竖排文字 "Dimension"/"value" 出现断字（Quick Look 字体度量问题）；如需完美效果，建议在 PowerPoint 中将该页另存为 PNG 后覆盖 `figure1/output/_pptx_render/task2_design_editable.pptx.png`，再重跑组装脚本。
3. `grouping_design_editable.pptx` 中 Survey & Interview 图标为外部链接图片（未嵌入，任何机器打开均显示为断裂占位框），d 面板已裁剪剔除；如需该图标需找回原图。
4. panel a–d 为栅格素材，e–j 统计面板在 PDF 中为矢量；全部文字为 Arial。

## 23. 图中统计结果（与第 5/11/15 节一致）

- **e**：4D-E vs 4D-NE 学习曲线簇检验显著（trials 19–24, P=0.026；26–31, P=0.032）
- **f**：满分率 4D-E 62.7% vs 4D-NE 40.7%，Fisher P=0.032
- **g**：个人最好成绩轮数 中位 26 vs 33.5，MWU P=0.021
- **h**：Obs vs NObs 无显著簇（permutation test）
- **i**：满分率 45.0% vs 30.0%，n.s.（P=0.248）
- **j**：个人最好成绩轮数 中位 31.5 vs 43.5，MWU P=0.045


---

# 追加记录（2026-10-08 晚）：panel e–j 独立子图优化版

## 24. 需求与实现

按新 Figure 1 编号重新生成 6 个独立子图（脚本 `figure1/code/figure1_panels_ej_optimized.py`，复现约 1.5 分钟）：

| 面板 | 内容 | 统计标注 | 图幅 | 其他优化 |
|---|---|---|---|---|
| e | Task 1 学习曲线 | 保留（cluster permutation：trials 19–24 P=0.026；26–31 P=0.032） | 3.9×3.5 in（更方） | — |
| f | Task 1 满分率 | 保留（Fisher：P=0.032 括号线） | 3.9×3.5 in | bar 宽度 0.52→0.34 |
| g | Task 1 最好成绩轮数 | 保留（MWU：P=0.021 括号线） | 4.6×3.25 in | 组间距 0.82→0.55 |
| h | Task 2 学习曲线 | **无** | 3.9×3.5 in（更方） | — |
| i | Task 2 满分率 | **无** | 3.9×3.5 in | bar 宽度 0.52→0.34 |
| j | Task 2 最好成绩轮数 | 保留（MWU：P=0.045 括号线） | 4.6×3.25 in | 组间距 0.82→0.55 |

另将检验名称注释（"Fisher's exact test"/"Mann-Whitney U"）从图内左下角（与数据重叠）移至左上角空白区（`figure1_perfect_rate_and_first95_survival.py` 与优化脚本同步修改）。

## 25. 输出文件（figure1/output/）

- `figure1_panel_e_task1_learning_curve_with_stats.png`（覆盖旧文件）
- `figure1_panel_f_task1_perfect_score_rate_with_stats.png`（新命名）
- `figure1_panel_g_task1_personal_best_round_with_stats.png`（新）
- `figure1_panel_h_task2_learning_curve.png`（新，无统计标注）
- `figure1_panel_i_task2_perfect_score_rate.png`（新，无统计标注）
- `figure1_panel_j_task2_personal_best_round_with_stats.png`（新）

注：旧编号命名的独立图 `figure1_panel_e_task1_perfect_score_rate_with_stats.png`、`figure1_panel_f_task2_perfect_score_rate_with_stats.png`、`figure1_panel_e/f_task{1,2}_first95_km_with_stats.png` 仍保留在 output 中（内容对应旧编号），如需清理可手动删除。
