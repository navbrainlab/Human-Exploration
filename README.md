# Fig. 1–7 绘图复现包

本包对应 `narrative_layout_20261113` 当前的七张图。每个 Fig 目录可单独复制和运行，内含对应的 `render.py`、绘图代码、冻结输入、Arial 字体和校验清单。渲染从 Matplotlib 绘图对象生成新 PNG。各图使用 600 dpi RGB 和 4960 × 7015 px 画布。

| 图号 | 目录 | 内容 | 输出文件 |
| --- | --- | --- | --- |
| Fig. 1 | `Fig1/` | 多维学习任务 | `fig1_multidimensional_tasks.png` |
| Fig. 2 | `Fig2/` | 维度隔离式采样 | `fig2_dimensional_sampling_pattern.png` |
| Fig. 3 | `Fig3/` | 探索特性 | `fig3_exploratory_properties.png` |
| Fig. 4 | `Fig4/` | 视觉采样与维度识别 | `fig4_visual_sampling_and_dimension_awareness.png` |
| Fig. 5 | `Fig5/` | DIS 与学习表现 | `fig5_sampling_and_learning.png` |
| Fig. 6 | `Fig6/` | 经验与 DIS | `fig6_experience_and_sampling.png` |
| Fig. 7 | `Fig7/` | 计算模型与参考应用 | `fig7_computational_account_and_reference_application.png` |

## 运行

用 Linux x86_64 下的 Conda 锁文件创建本次验证使用的环境。环境包含 Python 3.11.5、Matplotlib 3.7.2、NumPy 1.24.3、Pillow 9.4.0、cloudpickle 2.2.1 和相应的 FreeType、zlib 版本。

```bash
conda create --name mini-dg-figures --file environment-linux-64.lock
conda activate mini-dg-figures
python Fig1/render.py --verify
```

将 `Fig1` 换成对应目录即可重绘其他图。PNG 和核验记录保存在各图的 `output/`。在压缩包解压目录下可以批量运行。

```bash
for figure in Fig1 Fig2 Fig3 Fig4 Fig5 Fig6 Fig7; do
    python "$figure/render.py" --verify || break
done
```
