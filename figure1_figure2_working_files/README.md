# Figure 1–2 作图材料

这里汇总了 Figure 1 和 Figure 2 的图、代码及作图数据，原项目文件保持原样。

## 目录

- `data/`：两张主图共用及各分析所需的输入表。
- `figure1/code/`、`figure1/output/`：Figure 1 作图代码和当前图稿/可编辑结构图。
- `figure2/code/`、`figure2/output/`：Figure 2 作图代码和当前图稿/可编辑插图。

## 常用入口

- Figure 1：从项目包根目录打开 `figure1/code/figure1_integrated_four_panels.ipynb`。
- Figure 2b 及富集进程 SEM 图：在包根目录运行 `python figure2/code/figure2_feature_enrichment.py`。
- Figure 2c/2e 曲线：在包根目录运行 `python figure2/code/make_feature_enrichment_progress_mode_heatmap.py`；插图本身可在 `figure2/output/DIS_illustration_material.pptx` 中编辑。
- Figure 2a：完整复现材料已压在 `figure2/code/figure2a_reproduction_source.zip`，解压后按其中 README 操作。
- Figure 2h/2i：源 notebook 放在 `figure2/code/`，设包根目录为工作目录后运行。

2d 还在定位；2f/2g 暂留空位。Figure 2i 的 Task 1 原始作图 notebook 未找到，当前图和指标表已保留。Figure A 源 PPTX 可直接编辑；其构建脚本仍带原工作区路径。C1/C2 图及支持表已放入本包，若从跨项目原始数据开始重跑，仍需要原 SHJ datasets 工作区。

重跑绘图只会把新输出写进本包相应的 `output/`。
