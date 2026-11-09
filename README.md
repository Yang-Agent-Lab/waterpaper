# Waterpaper · 洪水预测研究

使用起报时已获得的当前观测和历史信息，预测未来各小时的流量分布与洪水发生风险，并研究极端事件数据增强及跨流域迁移能否改善中国流域的预测。

这是用于共同开发的研究代码仓库。包含基准、候选方法、历史实验和失败诊断；不能把某个目录的存在理解为该方法已获验证。逐时洪水概率、窗口内曾发生洪水的概率和首次发生时间是不同目标。

## 从哪里开始

| 任务 | 入口 |
| --- | --- |
| 看研究目标、方法分工和历史版本 | [研究与代码导览](docs/RESEARCH_CONTEXT.md) |
| 配环境、准备外部作者代码 | [开发环境](docs/DEVELOPMENT.md) |
| 了解数据来源、格式与划分 | [数据说明](data/README.md) |
| 修改当前逐时基准 | `src/pointwise_benchmark_v1/` |
| 极端数据增强、过程适配与迁移 | `src/d500_process_adapter_v1/`、`src/m_extreme_augmentation_v1/`、`src/extreme_process_transfer_v1/` |
| 讨论修改、提交 Pull Request | [协作约定](CONTRIBUTING.md) |

## 开始开发

在准备开发的机器上克隆仓库。正式数据处理、训练和绘图使用项目维护者指定的计算环境。

```bash
git clone https://github.com/Yang-Agent-Lab/waterpaper.git
cd waterpaper
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/check_public_tree.py
```

开发 DiffPTS、PPM 或 NeuralForecast 对比模型时，还需按[开发环境](docs/DEVELOPMENT.md)准备可选依赖与固定版本的作者代码。仓库不含真实数据、预训练权重及原实验输出，克隆后不能直接重现历史分数。

## 目录

```text
src/                    数据处理、模型、训练、评价与报告代码
tests/                  随源代码保留的检查与测试
configs/                历史技术规格与配置；使用前核对路径、预算和数据身份
data/catalog/           公开数据来源与变量说明
docs/                   开发和研究导览
scripts/                公共源码检查与作者代码准备工具
reference/              第三方来源说明；下载的作者代码由 Git 忽略
results/experiments/    运行输出约定；不进入 Git
SOURCE_MANIFEST.json    导出来源版本、文件哈希与路径替换记录
```

## 本次公开版本

主体代码来自本地保留提交 `f8bf638229dcd9206c835c895c8e90b3305c0e70`，补充了 `cffc1b13be29cd2bcab26d666f7c20100a90668b` 中的预警代码与相关核查脚本。公开副本采用新的提交历史，机器专用路径已归一化；原始来源与变更文件见 [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)。本次未核对服务器独有的最新修改。

部分历史训练入口使用内部 `sciencelab.training.TrainingRecorder`、原实验模型权重及数据清单；这些依赖未包含在公开仓库中，具体限制见[开发环境](docs/DEVELOPMENT.md)。它们不影响阅读和修改核心模型、损失、数据处理及评价实现。

第三方来源与已保留的许可文本见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
