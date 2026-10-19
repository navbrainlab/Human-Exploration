# vivi_modification_figure2.md — Figure 2 修改记录

> 本文档记录所有对 Figure 2 相关代码的修改过程、输出结果与统计结论。
> Figure 1 的修改记录见 `vivi_modification_figure1.md`。

日期：2026-10-08（创建）

---

## 0. Figure 2 现有代码与输出盘点（修改前基线）

### 0.1 目标版式（依据 `figure2/output/figure2_layout_sketch.png`）

```
a  富集轨迹散点图(E1→E2→E3)   b  特征富集度人vs随机(小提琴, Task1/Task2)
c  DIS 模式示意图(Task1/Task2) d  information gain 曲线
e  Mode Feature Enrichment Level 随试次进度(Task1/Task2 双条带)
f  Task1 概念密度/DIS 箱线图    g  Task2 同左
h  维度识别准确率×DIS 比例(Task1/Task2)
i  词汇分析(Task1/Task2, DIS vs Full corpus)
右侧: B/C1/C2 跨数据集面板 + Figure A 任务设计总览(SHJ/2D grid/MASC/Build-an-Icon)
```

### 0.2 脚本清单（`figure2/code/`）

| 脚本 | 对应面板 | 功能 | 输入 | 输出（`figure2/output/`） |
|---|---|---|---|---|
| `figure2a_reproduction_source.zip` | a | 完整复现包：原始行为→动作空间查找→输入表→`aggregate_cluster_relative_visit_order.*` | 包内自带 `source_data/task1_behavior_raw.csv` | `aggregate_cluster_relative_visit_order.{png,pdf,svg}`（成品已保留） |
| `figure2_feature_enrichment.py` | b + SEM 曲线 | 人 vs 随机的特征富集度差值小提琴图（Wilcoxon+BH-FDR）+ 富集度随进度 SEM 曲线 | `data/choice_category_uniform_105.csv`、`data/summary_data_0723_task2.csv` | `task1_human_vs_random_..._violin_mean.png`、`task2_human_vs_random_..._violin_mean.png`、`task1_task2_feature_enrichment_vs_trial_progress_sem.png` |
| `make_feature_enrichment_progress_mode_heatmap.py` | e（mode 曲线/热图） | 富集度众数随进度变化（10/24 bin） | 同上两个 csv | `task1_task2_feature_enrichment_vs_trial_progress_mode_curve_long.png`、`..._mode_heatmap.png`、`..._mode_gapped_heatmap_long.png` |
| `build_bc_main_figure.py` | B/C 源头 | 跨 4 外部数据集（SHJ/2D grid/MASC/Build-an-Icon）出现率/游程长度 vs 基线 | ⚠️ 依赖本包外原工作区数据（`shj_eye_long.csv` 等），本包内不可直接跑 | `bc_subject_level.csv`、`figure_b_frequency_above_random.png`、`figure_c_run_length_above_shuffled.png`、`figure_bc_main.png` |
| `build_final_b_c1_c2.py` | 最终 B/C1/C2 | 读支持表重绘三联图（散点+CI/小提琴，Wilcoxon 标星） | `data/support_tables/*.csv` | `figure_b_occurrence.png`、`figure_c1_stay_rate.png`、`figure_c2_run_length.png`、`figure_b_c1_c2_combined.png` + stats csv |
| `figure_drawing_6_survey_text_anaysis.ipynb` | h + 问卷清洗 | 问卷准确率×DIS 比例双面板图；产出问卷合并表 | `..._with_FDS_pattern_within_dms.csv`、`p1p2-endsurvey-sumup.xlsx`、`analysis_dataset_questionnaire_DIS_task2.csv` | `task1_task2_..._dualpanel_v1.png` + 清洗后 csv（写回 `data/`） |
| `game2_manual_fds_analysis_0917_JW.ipynb` | i（Task2 语料） | 人工 FDS 语句 vs 全语料词频（箱线图/词云/Top20 词分布） | `data/interview_data/*` | `game2_manual_fds_vs_all_boxplot_totaldim.png`、`game2_..._word_dist.png`、`game2_manual_fds_metrics0715.csv` |
| `build_figure_publication.mjs` | Figure A | 任务设计总览 PPTX（4 面板） | ⚠️ 硬编码 Windows 路径 | `figureA_task_design_overview_publication_v9.{pptx,png}` |
| `publication_plot_style.py` | 全局 | 与 figure1 相同的共享样式模块 | — | — |

### 0.3 已知缺口（来自根 README 与盘点）

- **Figure 2d（information gain 曲线）**：脚本未找到，仅存在于 sketch 中；
- **Figure 2f/2g（概念密度/DIS 箱线图）**：代码缺失，仅 sketch 中有；
- **Figure 2i 的 Task1 版本**：原 notebook 缺失，仅保留产物 `game1_manual_fds_vs_all_boxplot_foodfreq.png` + `data/game1_manual_fds_metrics.csv`；
- `build_bc_main_figure.py` 与 `build_figure_publication.mjs` 依赖原工作区路径，本包内只能使用其保留产物。

### 0.4 共用样式

- 调色板：`TASK1_COLORS`（3D-NE 绿 #079E6C / 4D-E 蓝 #0965C0 / 4D-NE 浅蓝 #14BAEC）、`TASK2_COLORS`（Obs 橙 #E67E28 / NObs 黄 #F7BA00）、FDS 紫系、调查类顺序色板；
- `LABEL_ALIASES` 术语统一为 DIS；Arial、300 dpi、pdf.fonttype=42、去上/右脊线。

---

<!-- 后续修改记录从第 1 节开始追加 -->
