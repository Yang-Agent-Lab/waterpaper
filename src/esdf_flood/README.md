# ESDF洪水适配

参考 Lu et al., *Electricity load forecasting under extreme heat and cold waves*, Nature Communications (2026), DOI 10.1038/s41467-026-77014-0。
作者代码 https://github.com/hkuedl/Heat-Cold_Wave_Load_Forecasting ，提交 a271801501a5c45660c3da71c0c4f32105a0680b，Copyright (c) 2025 EDL@HKU，MIT。许可全文见 `UPSTREAM_LICENSE.txt`；上游目录保持原样。

## 输入与模型

`--data-root` 显式指向 `data/processed/china_first_flow_v1/`。每目标独立读 `training_transforms.json` 和相应角色的 `*.jsonl.gz`，不重做划分、尺度或阈值。预测输入 `[B,12,2]` 按 P/Q 排列，输出未来 Q `[B,12]`；离线未来 P 和高流量标签不进入预测器。扩散联合样本 `[B,2,24]` 包含历史与离线未来 P/Q，类别为训练 Q95 定义的高流量窗口。

基线和直接合成增强均用相同 LSTM（长短期记忆网络，用历史序列预测未来）；合成增强每 epoch 从混合池随机抽取真实训练集相同数量的窗口，保持相同最大优化更新预算和早停规则，日志记录实际更新及真实/合成抽样次数。三分支采用共享 LSTM、真实专属 MLP、合成专属 MLP（多层全连接网络），共享 LSTM 从训练完成的基线复制权重。共享重构器恢复历史 P/Q；真实推断仅用共享和真实分支。三分支参数容量较大，结果不能直接解释为纯生成收益。

保留作者固定五个平方核带宽的 MMD（衡量两域共享特征分布差异）与跨样本矩阵正交公式；不同批大小使用各核块独立均值，零范数用机器精度保护。重构 L1 按各域各变量相加。预测/重构/MMD/正交权重初值 1/.1/1/.5，仅是显式初值。所有预测 checkpoint 仅按真实验证集未来12小时平均绝对误差选取；生成器仅按训练损失选取。

必要改造包括 168h/24h 电力布局变为12h/12h雨洪布局、删除未来天气预测输入、二维扩散卷积改一维、两类高流量标签、非零对数类别概率引导、删除 [0,1] 上界裁剪、按事件角色隔离、去除每 epoch 测试和作者顶层执行副作用。生成样本反变换后严格检查有限、P/Q非负、未来Q超训练Q95；不投影、裁剪、填补或隐瞒拒绝。达到原始抽样上限仍不足所需样本即 `sample_failed`，保留已接受样本及拒绝计数。

## 配置与命令

JSON 配置为扁平字段；正式值以批准的项目配置为准。初次接口示例：

```json
{
  "targets": "all", "seed": 2, "device": "cuda:0", "walltime_seconds": 21600,
  "hidden": 64, "batch_size": 64, "learning_rate": 0.001, "patience": 15,
  "baseline_epochs": 100, "diffusion_epochs": 100, "naive_epochs": 100, "separation_epochs": 100,
  "synthetic_ratio": 1, "sample_max_multiplier": 5, "sample_batch_size": 64,
  "sigma_min": 0.01, "sigma_max": 50, "sampling_steps": 100,
  "guidance_scale": 1, "corrector_snr": 0.2, "corrector_steps": 1,
  "prediction_weight": 1, "reconstruction_weight": 0.1, "mmd_weight": 1, "orthogonal_weight": 0.5
}
```

先提交，再用版本记录工具运行。下列输出目录必须尚不存在；wrapper 的记录目录和内部输出目录分开。

```sh
PYTHONPATH=src python3 src/run_versioned.py --run-dir results/experiments/esdf_check_001 -- python3 -m esdf_flood.check_pipeline --out-dir results/experiments/esdf_check_001/check --data-root ./data/processed/china_first_flow_v1
PYTHONPATH=src python3 src/run_versioned.py --run-dir results/experiments/esdf_china_001 -- python3 -m esdf_flood.train --config configs/esdf_flood_first_run_v1.json --data-root ./data/processed/china_first_flow_v1 --out-dir results/experiments/esdf_china_001/models --stage all --evaluate-test
```

支持 `--stage baseline|diffusion|sample|naive|separation|all`，`--target <folder>` 可重复。独立后续阶段用 `--from-run <prior-models-dir>` 只读已完成前置；缺少或未完成即失败，不自动补训。无法恢复中途优化器/RNG状态，本版只能从已完成阶段接续；`last.pt` 保留中断模型与优化器供诊断。`all` 顺序完成每个目标，单目标失败后继续其他目标，耗尽总预算则记剩余目标未启动。所有阶段与采样共享整个运行的 walltime；PC采样每步检查预算。

默认不读真实 test；显式 `--evaluate-test` 只在目标所选训练阶段全部完成后加载一次最终真实测试集。验证/测试报告各提前量 MAE/RMSE（平均绝对误差/均方根误差）、Q95/Q99条件误差和窗口数、持续性基线及逐窗口观测预测。无高流量窗口的单元格为 null / `NOT_EVALUABLE`。`window_peak` 只衡量12小时窗口内最大流量及其首次出现位置，不是完整事件洪峰；窗口数量也不是独立洪水数量。

`check_pipeline` 用保留在指定输出目录的自造窗口检查数据角色、未来信息隔离、尺寸和梯度、初始化、损失公式、无裁剪过滤、指标、非零引导方向及导入无执行副作用；可对真实 train/validation 完整只读检查，不读 test。PASS 只说明实现检查通过，不证明预测收益或生成质量。
