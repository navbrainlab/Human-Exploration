# Fig. 1 独立绘图包

在本目录执行：

```bash
python run.py
```

建议 Python 3.11。当前环境可直接运行；新环境先 `python -m pip install -r requirements.txt`。
输出在 `output/fig1.png`，参考图在 `reference/fig1.png`；程序验证 RGB、600 dpi、逐像素一致，并禁止读取原 mini_DG 项目目录。`verification.json` 记录结果。

`code/` 和 `source/` 保留实际原生绘图器及相对依赖；`data/` 内有当前 caption/Methods。图表只使用冻结绘图表与保存统计，不重新 fitting、simulation 或统计检验。`provenance.json` 保留原文件映射及 SHA256。源码副本仅本地化路径及历史验收依赖。

`assets/fonts/` 随包提供匹配的 DejaVu Sans 字体。Fig1 的 a/b 保留作者批准的完整任务示意栅格，结果 panel 从原生代码重绘；并保留原始 PPTX 与字体供任务图编辑。
