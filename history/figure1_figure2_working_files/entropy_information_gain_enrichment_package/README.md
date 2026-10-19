# 信息增益-特征富集程度图：复现包

本文件夹整理了“不同特征富集程度下的信息增益”均值加 SEM 折线图所需的脚本、数据和最终图片。图中将选择模式转换为特征富集程度：

| 原始选择模式 | 图中富集程度 |
| --- | --- |
| `4-0` | `4` |
| `3-1` | `3` |
| `2-2`、`2-1-1` | `2`（合并） |
| `1-1-1-1` | `1` |

## 信息增益口径

每个 trial、每个维度先计算二元维度熵下降：

`DER(t,d) = h(q(t-1,d)) - h(q(t,d))`

- 若维度与分数相关，只保留“正确确认”信息：`correct_confirmation_bits`。
- 若维度与分数无关，只保留“正确排除”信息：`correct_exclusion_bits`。
- 图中的信息增益为 `balanced_correct_evidence_bits`：先分别在相关与无关维度内取均值，再对这两个均值取平均，以平衡两类维度的数量差异。

## 统计与误差条

- 折线点的均值：pooled trial-dimension 数据的均值。对富集程度 `2`，先将 `2-2` 与 `2-1-1` 按各自 `n_dim_trials` 加权合并。
- 误差条：被试层面 `balanced_correct_evidence_bits` 的 SEM。对富集程度 `2`，同一被试的 `2-2` 与 `2-1-1` 先合并后再计算跨被试 SEM。
- `dimension_level`：每个 trial 的每个维度均按该维度自己的选择模式归类。
- `trial_primary`：每个 trial 按最大特征富集程度对应的主模式归类。

## 文件结构

- `scripts/plot_entropy_mean_sem_line.py`：绘图脚本。
- `data/plot_inputs/`：复现绘图直接需要的被试级 CSV 与模式汇总 CSV。
- `data/source_outputs/trial_dimension_identity_update_entropy.csv`：每 trial × 维度的原始熵计算结果。
- `data/source_outputs/dimension_identity_update_entropy_config.json`：熵计算配置。
- `figures/`：最终图，均同时提供 PNG 与可编辑 SVG。

## 重新绘图

在本文件夹根目录运行：

```bash
MPLBACKEND=Agg python3 scripts/plot_entropy_mean_sem_line.py \
  --input-dir data/plot_inputs \
  --output-dir figures
```

脚本会重新生成：

- `figures/dimension_level_entropy_mean_sem_line.png` / `.svg`
- `figures/trial_primary_entropy_mean_sem_line.png` / `.svg`
- `figures/entropy_information_gain_mean_sem_line_combined.png` / `.svg`
