# Fig. 1 · 多维学习任务

`render.py` 定义本图的面板选择和版式。`plotting/` 提供原生绘图函数。`data/` 包含本图使用的 Matplotlib 对象、版式和 Arial 字体。`manifest.json` 记录面板来源和校验值。

在 Linux x86_64 上可用包内的 Conda 锁文件创建环境。如果已创建同一环境，直接激活即可。

```bash
conda create --name mini-dg-figures --file environment-linux-64.lock
conda activate mini-dg-figures
python render.py --verify
```

运行后在 `output/` 中得到 `fig1_multidimensional_tasks.png` 和渲染核验记录。支持 `--output` 指定输出路径。用 `--verify` 可以比较 600 dpi RGB 像素与项目当前图片。冻结的绘图对象包含已计算的数据、图层和素材，可直接运行绘图。需要调整版式时可编辑 `render.py`。

| 面板 | 原项目中的来源 |
| --- | --- |
| a | current Fig.1a |
| b | current Fig.1b |
| c | current Fig.1c, Task 1 learning curves: EXP+ 3D, EXP+ 4D and EXP− 4D |
| d | current Fig.1e, Task 2 learning curves: Obs and NObs |
