# Fig. 5 可独立重绘包

本包对应当前 `main_figure/fig5`（2026-10-15 活动版本），按当前 native Matplotlib renderer 重建。a：原 Fig2g 的 Task 1 维度辨认；b：原 Fig2h 两任务访谈维度词频；c–f：原 Fig5g–j 的观察、早期维度意识、DIS 覆盖和奖励相关维度报告。

## 使用

```bash
python -m pip install -r requirements.txt
python run.py
python verify.py
```

从任意目录调用 `python /路径/fig5/run.py` 亦可；输出为 `output/fig5.png`。`verify.py` 在 Python 审计钩子下拒绝读取原 mini_DG 项目的其他文件，再重绘并逐 RGB 像素比较 `reference/fig5.png`。包可移动到项目之外；不依赖项目源码、Codex skill 或联网资源。Python 3.11 是已验证环境。

## 文件

- `code/`：当前重组入口与实际调用的 native renderer、坐标轴/布局 helper。仅将路径和 import 改为包内，并提取实际使用的函数；数据、样式、坐标和统计显示保持原样。
- `data/old5_inputs/`：旧 Fig5 native assembly 实际读取的冻结绘图 CSV、既有 composite PNG/SVG 和背景。为保持当前 native artist 再组装流程，此目录保留 assembly 所需输入；不是重新处理原始眼动数据。
- `data/old5/`：冻结的统计缓存、配对变化及已核验分析结果。统计星号读取现有缓存，不重新检验。
- `data/old2/`：Fig5a/b 对应的被试级绘图数据与原显示统计缓存。
- `reference/`：活动版本参考图；`output/`：源码重绘 PNG 与 native QA。
- `caption.md`、`methods.md`：活动版本说明；`provenance.json`：逐文件原始路径及 hash；`verification.json`：实际独立重绘结果。

## 边界与说明

不重新 fitting、simulation 或进行新的统计推断；仅从冻结输入计算原有描述性显示（如 boxplot、mean/SEM）。没有修改分组、观测值、显著性或面板视觉。

当前共享 renderer 先建立原 Fig5 native artists，再保留、移动所需 panel；因此 Fig5 的包内包含这一 assembly 读取的配套输入。未包含修订历史、旧最终图、无关分析脚本和模型拟合流水线。

已验证输出为 600 dpi、RGB；当前 Fig5 与活动版本逐 RGB 像素一致。相同 pinned Python 库和 DejaVu Sans 渲染环境可精确复现；跨库/字体版本可能产生抗锯齿差异。
