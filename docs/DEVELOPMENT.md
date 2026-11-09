# 开发环境与可运行范围

## 环境

基础依赖见 `requirements.txt`。历史逐时基准使用过 NumPy 2.2.6、SciPy 1.15.3、PyTorch 2.7.1 及 CUDA 11.8；本仓库保留已记录的部分版本，但未提供完整锁文件，也未在全新环境中完成训练复现。选择 CUDA 安装方式时按实际服务器和驱动准备 PyTorch。

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-benchmarks.txt  # 开发外部基准时
python -m pip install -r requirements-dev.txt         # 运行 pytest 类检查时
python scripts/check_public_tree.py
```

数据读取/采集的可选依赖在 `requirements-data.txt`。CDS 等服务的认证由开发者在本机安全配置，不能写入代码、命令参数、配置样例或提交历史。

## 固定版本作者代码

DiffPTS、PPM 适配器按指定提交读取作者实现，不把整个上游代码库或数据重新上传到本仓库。准备工具只下载这两个源码提交，不下载数据、不安装依赖、不训练：

```bash
python scripts/fetch_author_sources.py
export WATERPAPER_DIFFPTS_SOURCE="$PWD/reference/supporting/code/DiffPTS/e9a7d1d56bd40b4bcfced29b430f6c5be6f2a424"
export WATERPAPER_PPM_SOURCE="$PWD/reference/supporting/code/PPM/6b4206a8c6bd3d298d8bdc58ebc88e5f5f6c13f9"
```

准备后的路径与 `src/pointwise_benchmark_v1/vendor.py` 使用的版本一致：

| 模型 | 官方仓库 | 提交 |
| --- | --- | --- |
| DiffPTS | https://github.com/wwy155/DiffPTS | `e9a7d1d56bd40b4bcfced29b430f6c5be6f2a424` |
| PPM | https://github.com/ljl8336/PPM | `6b4206a8c6bd3d298d8bdc58ebc88e5f5f6c13f9` |

`torch-timeseries` 在原适配器中用于指定神经网络层，并保留源文件完整性检查；不要为处理依赖问题而跳过这些检查。

## 配置与数据

从仓库根运行命令。公开导出仅替换了原机器的项目路径前缀；出现 `./external-home/` 的位置代表未分发的外部本地依赖。`configs/` 是随源代码保留的历史配置，含旧运行名、旧权重 SHA、固定数据清单和预算，需按具体任务准备真实输入并显式记录变动，不能直接当成通用默认配置。

数据准备与核心训练入口可先查看帮助：

```bash
python src/prepare_china_windows_v1.py --help
python -m src.pointwise_benchmark_v1.runner --help
python -m src.extreme_process_transfer_v1.worker --help
```

不同模型路径要求的数据格式并不完全相同。准备 `fit/select` 数组、清单、索引及阈值时以实际加载器为准，见 [data/README.md](../data/README.md)。本次不含原模型权重和真实数组，历史预测比较不能在克隆后直接重现。

公开适配只将迁移模块的输出范围检查改为本 checkout 下的固定 run，并让条件轨迹入口核对本 checkout 的绝对根路径；原私人服务器目录结构不再是这两处检查的前提。计算公式、样本身份、预算和选模规则保持原样。所需数组读取辅助代码也从保留的 Git 版本补齐，具体来源见 `SOURCE_MANIFEST.json`。

## 内部监控依赖

部分 B2、数据混合、D500 过程适配、增强和路由训练入口依赖内部 `sciencelab.training.TrainingRecorder`；其中若干历史入口还包含发送训练状态的子进程。该组件和采集端凭据未随本仓库公开。开发这些入口前，需要由维护者另行提供允许分享的组件，或审阅并实现独立的本地记录适配；不要安装名称相似的未知包或关闭来源检查。

核心模型、损失、指标、数据准备实现可分别阅读和修改。这里的限制针对相应历史训练入口，不表示所有模块都依赖内部监控。

## 正式运行与验证

先与维护者确定数据、预算和计算位置。代码提交且工作区干净后，沿以下入口执行已约定的任务：

```text
python src/run_versioned.py --run-dir results/experiments/<unique-run-id> -- <command and arguments>
```

该工具记录版本、命令、日志与退出状态，不验证研究结论。正式运行不应覆盖旧目录；失败与负结果保留。图也保存到该 run，并记录单位、数据角色和来源。

发布前静态检查只覆盖源码语法、JSON 和公开文件范围，不代表依赖安装、真实数据处理、模型训练或独立测试已通过。
