# Fig. 3 独立绘图包

在当前目录运行：

```bash
python -m pip install -r requirements.txt
python run.py
```

输出：`output/fig3.png`；参考图：`reference/fig3.png`。`verification.json` 记录原图与重绘图的 RGB 像素比较，绘图器的原生布局检查保存在 `output/` 内。

- `code/`：当前正式图的 renderer 及直接依赖；只调整输出命名和便携完整性校验。
- `data/`：冻结绘图输入、已保存统计结果/曲线；不重新做拟合、模拟或统计检验。
- `assets/fonts/`：渲染环境实际使用的 DejaVu Sans 字体；字体许可随包提供。
- `documentation/`：当前 caption 和 Methods；`provenance.json` 给出来源和 SHA-256 映射。

可以移动整个 fig3 文件夹到其他位置后运行，不需要原项目。`run.py` 会阻止原项目其它目录的文件读取。`reference/` 仅用于渲染完成后的像素核对，不作为绘图输入。

此包用于复现当前主图，冻结的是绘图所需数据而非全项目的上游分析数据库。保留原科学定义、原统计、原布局及 600 dpi；不包含历史修订、缓存、额外拟合或模拟流程。精确像素复现建议使用所列 Python 3.11 / 固定依赖版本。

Fig. 3 包含新增 EI 概念图 d 和底部语言面板 g；旧缓存命名保持不变。词频字号 5–8 pt 是原图已获准的数据编码例外。
