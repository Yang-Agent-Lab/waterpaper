# 研究与代码导览

## 研究问题

中国本土洪水预测是最终应用目标。国际资料用于检验能否提高中国流域的预测泛化，不能用国际资料上的结果替代中国结果。河流洪水、山洪与城市内涝也不能混作同一个预测标签。

模型应在起报时仅使用已经获得的观测和历史信息，输出未来各小时的流量分布，并由当地有依据的阈值得到风险。历史跨度和预测时长是独立参数；12 小时历史、6 小时峰值或四个窗口均不是永久任务定义。

## 方法与目录

原始改进基准是 DiffPTS，N-BEATSx、TCN 等是外部对比模型。工作主线借鉴 Lu 等的极端事件生成、真实/合成特征分离与下游预测收益验证；具体改造需要实际雨洪预测收益支持。

| 目录 | 职责 |
| --- | --- |
| `src/pointwise_flood_risk_v1/` | 逐时风险接口 |
| `src/pointwise_benchmark_v1/` | 统一数据接口、DiffPTS/PPM/外部模型适配、状态条件与评价 |
| `src/d500_process_adapter_v1/` | 冻结预测基准上的过程适配候选 |
| `src/m_extreme_augmentation_v1/` | 极端过程生成与增强探索 |
| `src/extreme_process_transfer_v1/` | 国外极端过程迁移、采样及常态保护探索 |
| `src/moe_lead_routing_v1/` | 专家和提前量路由探索 |
| `src/real_extreme_v1/` | 真实极端样本对照 |
| `src/conditional_trajectory_v1/` | 条件轨迹候选 |
| `src/five_country_model_data_v1/` | 多国资料模型化准备 |
| `src/training_report_v1/` | 统一训练与预测报告 |
| `src/esdf_flood/` | 早期极端生成—分离—预测适配；任务定义与新逐时版本不同 |
| `src/rl_warning_v1/` | 固定预测器之后的预警策略探索；与预测模型收益分开 |

目录名中 `D500`、`M` 等是已有研究版本标记，不是本次发布新增的方法。公开代码同时保留历史与候选路径，不为其重新指定有效性或性能排名。历史配置内的授权、运行状态和服务器记录只描述当时条件。

## 阅读顺序

1. 阅读 `configs/pointwise_flood_risk_v1.md` 和 `configs/paper_metrics_v1.md`，确认预测对象及指标。
2. 从 `src/pointwise_benchmark_v1/dataset.py`、`models.py`、`metrics.py` 理解数据流。
3. 按负责方向阅读过程适配、增强或迁移模块及其配置。
4. 先做最小可辨别对照。观察到信号后，再与维护者确定充分训练和独立验证范围。

主要参考为 Lu et al., *Electricity load forecasting under extreme heat and cold waves*, [DOI: 10.1038/s41467-026-77014-0](https://doi.org/10.1038/s41467-026-77014-0)。引用及第三方代码来源另见 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。本次导出没有重新检索论文或重新训练模型。
