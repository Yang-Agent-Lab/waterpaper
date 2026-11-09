# WP-20261018-POINTWISE-DATA-MIX-V1

状态：已批准的实现与短跑准备；smoke 通过且主控审核后才启动 9 个正式 run。正式训练总量为 27,000 次更新。

## 研究问题与边界

在同一原始 DiffPTS、同一批中国开发样本和同一更新预算下，训练数据的国家组成是否改变中国逐时洪水风险表现？本轮只比较数据配置，不修改架构，也不训练外部对比模型。它不能证明国际数据带来因果迁移收益：A 的中国样本曝光量是 B/C 的两倍；B→C 在中国曝光固定时用冰岛、美国、日本样本替换一部分德国样本。

此前阶段分析中，保留版 B3b 虽然全量 Brier 有收益，Q95 起涨初期和持续中段却比 B2 分别退步 7.11% 和 2.92%；共享两参数概率校准也使 B3b 总 Brier 退步 1.51%、持续中段退步 11.88%。CORN 有序风险头没有胜过同容量的普通二元头。本轮不把这些负结果当成新的结构收益证据；改用同一个原始 DiffPTS，先检验数据混合是否能改变逐时全过程表现。

目标仍是当前/历史 P、Q 到未来 24 个小时逐时 Q95/Q99 超越概率，包含洪水起涨和持续状态，不是窗口内曾发生概率、首次发生时间或时间累积最大概率。困难特大洪水仍需在可比训练预算下直接与外部强对照比较；本轮不含该比较，也不因范围限制免除后续比较要求。历史 B0/B3b/TCN 分数因训练条件和预算不同，只能另列为历史结果，不能充作本轮公平对照。

## 固定数据及站点规则

唯一输入为已接受 PQ bank：服务器 `./results/experiments/pointwise_full_pack_108f9a0/artifacts/pq`，manifest SHA-256 `2957aabab9a36247168e3ee05a6cd0f4490ac2f31395b2cd53c384668d7ce81c`。仅使用 bank 内的 fit 和 select。训练使用历史 24 小时 P/Q，预测未来 24 小时 Q；不输入未来降雨。复用原 fit 归一化、Q95/Q99 阈值、源身份和事件/来源分组，不重新估计或更改标签。

中国 fit 固定为 bank 中已有 8 个站。德国、冰岛、美国、日本的候选站只从 fit 行按 `canonical_station` 去重；对 `SHA256(change_id + NUL + country + NUL + canonical_station)` 做升序排列，按下表取前 N 个。排序不读取 select 分数或事件难度。列表生成一次，三 seed 共用；输出中记录 station identity、hash 和 fit 行数。没有足额候选站即停止，不降低数量、不下载新资料。

|组|Fit 站点|每个 64 条 batch 的国家抽样比例|
|---|---|---|
|A|全部 8 个中国 fit 站|CN 100%|
|B|同一 8 个 CN + 固定 30 个 DE|CN 50%，DE 50%|
|C|同一 8 个 CN + 同一 30 个 DE + 固定 20 个 IS、20 个 US、1 个 JP|CN 50%，DE 20%，IS 10%，US 15%，JP 5%|

每国按 canonical 站点均匀、站内 source-event 均匀、event 内 origin 均匀有放回抽样，沿用既有 station→event→origin 公平方法。B/C 使用同一个 seed 对应的中国抽样序列；每轮各有 32 条 CN 样本，因此中国曝光逐条一致。正式 3000 updates 中，A 为 192,000 次 CN 曝光；B/C 各为 96,000 次 CN 曝光。曝光数是训练抽样次数，不是独立样本量；重复抽到的窗口不增加独立事件或站点数。C 的正式总曝光按国家精确分配为 CN 96,000、DE 38,400、IS 19,200、US 28,800、JP 9,600。smoke 的整数配额按最大余数法取整，仅验证路径，不解释为实验结果。

三组共享完整中国 select：9 站、17 个 source-event、1,706 个 24 小时预测起报；不得抽小或删难例。只从 `y_q_m3_s` float64 原始监督与 bank 原 fit 阈值构造观测标签；不从 float32 重建标签。select 的 Q95/Q99 主评分按中国站点→source-event→origin 等权。不得读取 test、calibration 或 reserved 数据。

旧李青店洪水 6 / 栾川洪水 9 的巨大峰值诊断不属于当前 select；当前 select 对应李青店洪水 5 / 栾川洪水 1，身份不可混用。

## 模型、预算与检查点

只训练原始 DiffPTS：沿用 pointwise benchmark 的作者实现和原生损失；mean 编码器读取历史 P/Q，variance 模块与去噪器只读取目标 Q。显式使用已对齐的 `variance_floor=1e-8`。不启用 state 或新增 risk head，不训练 TCN/N-BEATSx/其他模型。每个 run 使用 seed 2、13、42 之一；同 seed 的 A/B/C 从相同初值开始，保存并核对初始权重 SHA-256。

每 run 使用 Adam（lr 0.001，默认 betas/eps）、batch 64、梯度裁剪 10、共 3 个 1,000-update 区段。每个区段末保存候选权重并以完整中国 select 的 128 draws 评价；用 station→event→origin 等权的 24 个 lead × Q95/Q99 平均 Brier 选择最早最低分。最终选中权重对完整中国 select 做 256 draws，保存逐起报预测、canonical observation、scores、行身份、所选权重和 loss/report。9 个正式 run 为 3 组 × 3 seed，不超出 27,000 updates。

每组 smoke 最多 2 个 updates，共 6 updates，单独标为工程检查，不占正式预算，不据此解释模型效果。smoke 覆盖银行身份/hash、站点清单、国家与站点/event 抽样、初值一致、训练、三个主程序阶段中 smoke 可达的候选选择/最终评分与报告写出；如 smoke 的 select 128/final 256 draws 对完整 CN select 不能完成，记录失败并查明原因，不减少正式 select 范围。

## 指标与比较

主终点：完整中国 select 上 24 lead × Q95/Q99 的 station→source-event→origin 等权平均 Brier，越低越好。Q95 起涨 first-three-positive 与持续中段（仅正例连续时长大于 6 小时的 event，去掉最初和最后 3 小时）为阶段次终点；另报 Q99、每 lead 与 CRPS。阶段定义、配对样本和 canonical truth 复用已接受 `pointwise_phase_analysis_v1`，不重新挑事件。

对每 seed 先算组间配对差及相对改进率；低值优指标的改进率为 `(参照值 - 当前值) / 参照值 × 100%`。报告 A/B/C 的三 seed mean、sample SD、min–max，并列 A 为同轮数据参照、C 对 B 的直接数值差。三 seed 变异只表示这三个训练种子的波动，不是站点或洪水事件泛化置信区间；不能把 origin、小时或重复采样曝光当独立 n。Q95/Q99 及起涨/持续冲突时报告 mixed，不隐藏退步。

## 运行与记录

本机仅编辑和 Git；服务器唯一根 `.`。本轮新 worktree 为 `worktrees/pointwise_data_mix_v1`，分支 `codex/pointwise-data-mix-v1`，代码起点 `e924e7e6f504420f2d5b582542fa1c5e1b74ab2f`。使用既有 `./environments/pointwise_benchmark_v1/bin/python`、vendor checkout 和 GPU 排他锁 `./artifacts/pointwise_gpu0.lock`。正式与 smoke 命令均由 `src/run_versioned.py` 记录版本、argv、stdout/stderr 和真实退出码；训练进度与产物保存在新的 `results/experiments/` 目录，不覆盖既有 run 或 bank。

先完成代码提交、服务器自检和 6-update smoke，再交主控检查；主控检查前不启动 formal。正式 run 使用已存在的 `TrainingRecorder` 和固定 sender，显式白名单只含模型、组别、超参、train loss、select Brier 与提交身份；smoke 不上传监控事件。上传若未获确认，保留 `partial` 状态和 sender 回执交主控判断。不得创建/恢复 cron、wakeup、App automation 或新聊天。正式结果和必要图在服务器生成；后续阶段分析复用 phase-analysis 实现，不再新建训练/报告系统。Science/飞书双台账由主控维护；本次不写外部 ledger。
