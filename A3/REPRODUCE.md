# 复现实验

原始运行环境：Windows 10、Python 3.12.14、NumPy 2.3.5、16 个逻辑 CPU。计时只用于描述该机器上的本次运行。

## 1. Python 环境

在仓库根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux/macOS 激活环境的命令为 `source .venv/bin/activate`。

## 2. KenLM 工具

Windows x86-64 可安装与原实验相同的第三方构建：

```powershell
python A3/setup_kenlm.py
```

脚本下载指定版本，核验 SHA256，只解包所需的 `lmplz.exe`、`build_binary.exe` 和 `query.exe`。来源见 [工具清单](tools/tool_manifest.json)。如果已经有工具归档，可以离线安装：

```powershell
python A3/setup_kenlm.py --archive "path/to/kenlm-x86_64-windows.zip"
```

Linux/macOS 请按 [KenLM 上游说明](https://github.com/kpu/kenlm)构建，使用 `--kenlm-bin /path/to/kenlm/build/bin` 指定目录。三个工具也可以加入 PATH。原实验仅在 Windows 构建上验证；不同编译器或平台可能改变时间和模型字节哈希。

## 3. 语料与运行

准备 [DATA.md](DATA.md) 中描述的文件；本仓库不附完整语料。在根目录运行：

```powershell
python A3/run_experiment.py --source "path/to/199801.txt" --stage all --repeats 3
```

每阶重复三次，顺序轮换。也可以分阶段运行：

```powershell
python A3/run_experiment.py --source "path/to/199801.txt" --stage prepare
python A3/run_experiment.py --stage train --repeats 3
python A3/run_experiment.py --stage analyze
```

`analyze` 需要已生成的模型和数据。要把新产物与已记录结果分开，给所有阶段传入同一个输出目录：

```powershell
python A3/run_experiment.py --source "path/to/199801.txt" --stage all --repeats 3 --output-dir A3/rerun
```

## 4. 结果口径

- 清洗保留词边界与标点；未识别格式先报错。
- 按文章 ID 的确定性哈希划分，再对段落去重；不把测试词补入训练词表。
- 主表 PPL 包含 OOV 和段落末 EOS。
- 续写最多 48 个新 token，遇到 EOS 停止；固定提示、种子和参数见报告。
- Python ARPA 评分会与 KenLM `query` 进行核对。
- 计时包含训练过程和 ARPA 写出，不含清洗与二进制转换。

默认运行会覆盖 `A3/results` 的机器可读结果和部分 Markdown 汇总。报告正文记录 2026-10-08 的原始实验；重新运行后应根据新结果同步更新正文与参数统计。
