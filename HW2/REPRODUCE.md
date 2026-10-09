# 复现步骤

建议 Python 3.12，使用独立环境。本次 Windows 实测依赖见 `requirements.txt`；完整已安装版本在 `requirements-lock.txt`。`fasttext-wheel==0.9.2` 是第三方提供的 fastText 预编译分发包，调用的是 `fasttext.train_supervised`，不是 Gensim 分类 API。NumPy 固定为 1.26.4，避免该接口与 NumPy 2 的数组复制兼容问题。

## Windows PowerShell

在本作业目录运行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --no-cache-dir --index-url https://pypi.org/simple -r requirements.txt
.\.venv\Scripts\python.exe -u scripts/download_data.py
$env:PYTHONHASHSEED = '0'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe -u scripts/run_experiment.py
.\.venv\Scripts\python.exe -u scripts/check_convergence.py
.\.venv\Scripts\python.exe scripts/verify_results.py
```

## Linux / macOS

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -u scripts/download_data.py
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u scripts/run_experiment.py
PYTHONHASHSEED=0 .venv/bin/python -u scripts/check_convergence.py
.venv/bin/python scripts/verify_results.py
```

若所用平台没有 `fasttext-wheel` 的对应 wheel，需要可用的 C++ 编译环境或自行安装兼容的官方 fastText。改变分发包或依赖版本后应重新记录环境，不能预先保证数值逐位相同。

下载脚本核验缓存哈希并记录固定来源。训练脚本按相同顺序重新生成清洗、划分和词向量；进行 2 个表示模型训练、18 个分类条件训练，以及 1 次 8 类预训练分类器的相同种子重复。三种分类随机种子为 42、43、44；表示模型仅使用种子 42。跨平台的浮点和线程实现可能造成差异，重复实验只能验证本次环境内的复现。

训练文件与模型的目录是 `data/`、`models/`；日志、逐条预测、分数、混淆矩阵和图在 `results/`。运行结束执行 `verify_results.py`，它从逐条预测重新计算指标，并检查数量、混淆矩阵、嵌套划分和训练／测试隔离。

本机首次分类尝试遇到中文绝对路径无法打开的问题，修复为从作业目录传相对 ASCII 路径。最终分类运行用 `--reuse-embeddings` 继续，重建的数据哈希和已导出 `.vec` 的哈希均核对一致，因此未重训已完成的表示模型。正常首次复现不加该参数即可；已有相同输入和词向量时可以使用。调试记录见 `results/path_compatibility_issue.log`。

主流程的 19 次分类训练之外，`check_convergence.py` 新训练 8 类从零模型的 100／250 轮两个条件，并加载已有 25 轮模型评估。总计 21 次分类训练。该补充由低分基线触发，用于诊断训练不足，不用同一测试集作无偏模型选择。
