# mini_DG 当前主图绘图包

打包日期：2026-10-16。对应 `publication/main_figure` 当前正式 Fig. 1–7；每张图放在独立的 `fig1` 至 `fig7` 文件夹，可单独复制、运行。

## 快速运行

已验证：Python 3.11.5、Matplotlib 3.7.2、NumPy 1.24.3、pandas 2.0.3、Pillow 9.4.0、SciPy 1.11.1。当前项目环境可直接使用；新环境安装根目录依赖后运行：

```bash
python -m pip install -r requirements.txt
python run_all.py
```

只画指定图：

```bash
python run_all.py --figures 2 5 7
```

单图运行：在相应文件夹执行 `python run.py`，或从任意位置调用该文件的绝对路径。输出位置均为 `figN/output/figN.png`；当前正式参考图为 `figN/reference/figN.png`。

## 图件与来源

| 文件夹 | 当前正式来源 | 内容 |
|---|---|---|
| fig1 | fig1_nhb_tick_endpoints_20261014 | 两套任务、先前经验与学习表现 |
| fig2 | fig2_nhb_reordered_examples_20261014 | 特征富集、DIS 示例、持续性与覆盖 |
| fig3 | fig3_nhb_tick_endpoints_size_key_20261014 | 不确定性、EI 示意与探索／语言证据 |
| fig4 | fig4_nhb_five_panels_tight_outcomes_20261013 | DIS 与表现、跨阶段关联及阻断实验 |
| fig5 | fig5_nhb_dimension_awareness_20261015 | 维度知识、观察与早期维度探索 |
| fig6 | fig6_nhb_gaze_selectivity_20261015 | 注视选择性、注视覆盖及停留时间 |
| fig7 | fig7_dgem_free_beta_consistent_20261015 | DGEM、拟合、回放、独立性能优化与应用 |

Fig3 使用活动目录中的最新 EI 定义与三层布局；不使用另一个 compact 候选目录。Fig5/6 的旧编号原生源码及所需冻结数据已随各图本地化。Fig7 历史代码文件名仍含 fig6，但输出为当前 Fig7。

## 每张图包含什么

- `run.py`：独立入口；`code/`：原生绘图器及实际调用的依赖。
- `data/`、`csv/` 或 `source/`：绘图所需的冻结表、已保存统计／曲线和源资产；具体映射见单图 README。
- `reference/`：当前正式 PNG；`output/`：本次重绘 PNG 及原生 QA。
- caption、Methods、`provenance.json`：科学定义、原文件来源、SHA-256 和包内路径修改记录。
- `requirements.txt`：固定库版本；需要时随包提供字体与许可。

Fig1 上半部保留已获批准的完整任务示意栅格，同时提供原任务 PPTX 及相关资产用于编辑；统计结果面板由数据与原生代码重绘。Fig6a 使用既有 gaze composite 的显示层；原始密度估计不在重绘时重算。其余来源和具体限制见各图 caption/Methods。

本包读取冻结绘图输入，保留当前数据、检验结果、模型来源、布局及 600 dpi RGB 输出。主模型的拟合／回放沿用七参数 free_bonus；独立性能优化、历史参考与数字化 TDGE 应用保持各自来源说明。运行入口不执行模型拟合、模拟、原始眼动处理或新统计检验。

单图入口的像素核对用于复现冻结版本。后续改图可编辑各图 code/data；如果主动改变数据或样式，参考比较会报告差异，部分冻结输入校验也需同步更新。具体底层 renderer 入口在单图说明中列出。

## 独立复现核验

```bash
python verify_package.py --jobs 2
```

该检查将每个 fig 文件夹复制到项目之外的临时目录，移除既有 output，阻止读取原 mini_DG 项目及 Codex/agents 目录，再执行 `run.py` 并与当前正式参考图比较 RGB 像素、尺寸、颜色模式及 DPI。环境依赖使用已安装的固定 Python 库。

结果：`qa/independent_verification.json`；每图运行日志：`qa/figN_relocated.log`。`original_png_hashes.json` 用于在原工作环境核对正式 PNG 保持不变；离开原工作环境后这些路径仅作来源记录，不参与绘图。

`file_manifest.json` 与 `SHA256SUMS.txt` 记录交付时逐文件校验值；重绘生成 QA 文件后，动态输出文件的校验值可能更新。外层 ZIP 附完整性检查记录和 SHA-256。
