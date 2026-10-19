# Fig. 7 独立绘图包

在本目录执行：

```bash
python -m pip install -r requirements.txt
python run.py
```

已验证环境为 Python 3.11；当前环境可直接运行。输出 `output/fig7.png`，并与 `reference/fig7.png` 逐像素比较。`verification.json` 记录尺寸、RGB、600 dpi 和比较结果。可移动整个 fig7 文件夹到其他位置运行；不需要原 mini_DG 项目或 Codex skill。

- `code/`：当前 native renderer、布局、坐标轴及审计 helper。历史入口文件名保留 `render_fig6_full_joint90_free_beta.py`，但生成的是当前 Fig. 7。
- `data/`：冻结的绘图 CSV，包括当前 V2 b/e 数据、七参数拟合与回放汇总、独立性能优化汇总，以及数字化 TDGE 参考数据。
- `assets/fonts/`：匹配当前绘图环境的 DejaVu 字体和许可。
- `caption.md`、`methods.md`、`docs/results.md`：当前图的科学定义、来源与解释边界。
- `provenance.json`：源文件位置及 SHA-256、包内文件及修改说明。
- `source_provenance.json`、`v2_be_revision.json`：原始科学来源记录；其中绝对路径仅作溯源，不在重绘时访问。

重绘只读取保存的绘图表和精确统计，不重新拟合、模拟或执行统计检验。为使包独立运行，路径改为包内相对路径，并提取了实际调用的样式函数和 DIS 布尔判定；绘图函数与科学数值保持不变。与重绘无关的历史 PNG 保护检查改为空范围，原项目 PNG 的保护由总包 QA 独立核验。

当前 b/e 的任务 2 V2 基线与保留的 DGEM 性能参考反馈条件不同；g 中 RL 曲线是历史参考；i 为独立 TDGE 应用的数字化参考奖励。这些来源边界沿用当前 caption/Methods，不在打包时改写。

`run.py` 默认验证冻结版复现，改变代码或数据后若图片不同会报告差异。直接调用 `code/render_fig6_full_joint90_free_beta.py --output-dir PATH` 可用于后续改图；冻结参考图用于对照。
