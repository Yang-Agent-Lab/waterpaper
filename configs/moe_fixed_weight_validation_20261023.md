# RESEARCH EXECUTION SPEC：M专家混合固定权重小验证

记录号：WP-20261023-MOE-FIXED-WEIGHT-V1。revision1.0。Status: APPROVED。
授权：用户明确“暂停极端生成G，保留D500主干验证M”；接续已说明的四种固定权重小验证。不训练新模型，不实施截图中的新路由/时间监督。

## 问题与允许的结论

在既有M权重下，学习路由是否优于三个专家等权，专家分支是否比保留历史读取和共享卷积的模型提供收益？允许观察固定权重依赖与性能线索，不能认领去模块重训贡献、MoE原创性、稳定泛化或正式模型晋级。阴性或混合结果均可完成，不以改指标或增量挽救。

## 起点、身份与允许修改

- 源码基线7e02d9e090c356f5b7ecaeb1a93c08c3e7dc07d0；本地复用干净独立工作树 .，创建codex/moe-fixed-weight-validation-20261023分支，旧分支保留。
- 仅新增诊断工具及本规格/必要配置、轻量结果记录；不改原D500、M、G、训练/采样/评分实现，不改权重。严禁覆盖主目录用户暂存修改。
- 服务器lj，权威项目根.；新诊断工作树放该根worktrees/moe-fixed-weight-validation-20261023；环境 .venv/bin/python，复用 runtime/d500_process_adapter_v1_deps_20261022 及原作者输入，先核实际环境与资源。
- 原M0权重：./results/experiments/d500_process_adapter_v1_20261022_G1/finalize/artifacts/M/best.pt，SHA256 3cfdf9fe702db06b5cb6df4da51a805e324f6be2f3289508e3cc304bccfbc9ce。
- D500 SHA256 bbd6f87e0ad452873f1d4b357a67cf2fbd823bb2dd3bdc36e56ab7b2d06c4a3a，位置沿既有configs/m_extreme_augmentation_v1_20261023.json。
- 已有配对参考根：./results/experiments/m_extreme_augmentation_v1_20261023/pilot_prepare_envfix/artifacts，复用M0_K64、D500_K64、manifest、subset_selection。不能选R/E/G新权重替换M0。

## 四种对照与小验证范围

1. dynamic：原M0路由，直接复用同口径K64参考。
2. uniform：M的两块各将3专家路由严格固定为[1/3,1/3,1/3]，其余计算不变。
3. no_experts：两块混合专家贡献严格为0；保留Bridge历史读取、共享时序卷积、残差、归一化和末端投影。不是关闭整个M。
4. no_M：新增去噪残差严格为0，复用同口径D500_K64参考。

同262起报、9站、17来源单位、原global_index/sample_id顺序、K64、seed900002、20步、原chunk与NoisePlan；原全select阶段标签映射至子集，不稀疏重定义起涨。只用起报时已知历史P/Q；不生成训练数据，不访问test，不扩到1706/K256。统计按原站→来源→起报及提前量口径，重叠窗口不当独立洪水。

参数/优化器更新0，新增正式模型推断最多uniform和no_experts各一次262×K64；工程小检查最多原两起报×K64，覆盖dynamic/no_M复现和两个新模式的语义/写出。若缓存配对或原路径复现不一致，先定位与报告，不自动重做完整参照或扩大推断。工程失败只补相关失败检查，成功部分不重复；不得通过裁剪/替换非有限值掩盖异常。

## 指标与取舍

主要报告uniform−dynamic、no_experts−dynamic的原值/差值/相对变化，原Q95/Q99起涨Brier与总体Brier并列，真实超阈/未超阈分层、CRPS/Q95及已有提前量分层作为必要代价。CRPS全时段不能叫极端条件CRPS；首次越阈时间监督/评分不在本轮新增范围。保留旧峰及ES指标可复用的输出，不新增训练消融。

动态优于等权：支持当前权重利用了动态分配；等权优于无专家：支持当前专家组合有作用；无专家不差：专家必要性减弱。差异小、指标取舍或支持不足时保留inconclusive，不设事后显著性阈值。现有单seed/已暴露开发子集只提供诊断线索。

## 检查、运行、验收与停止

- 核cached样本/真值/阈值/权重/噪声/K/seed/模块字节身份，保存manifest。预检查原两个起报dynamic/no_M要复现对应缓存；概率要求精确一致，若情景浮点非逐位必须定位并披露实际最大差，不能放宽后将其冒充精确复现。
- 检查uniform两块路由和为1且逐值1/3；no_experts两块混合贡献确为0；其它模块/参数、eval模式和随机噪声保持；参数指纹前后相同、优化器步数0；保存干预位置与输出证据。不只检查最后输出形状。
- 独立复核未参与实现者核不可变commit、干预语义和工程证据后正式放行。主会话或独立reviewer从保存情景复算关键Brier/CRPS与配对样本；PASS只指执行可信。
- 先提交后服务器run_versioned，正式入口保存commit/argv/cwd、配置、日志、退出状态、预测/指标、模式/源与权重SHA、缓存引用和完成状态。最多1GPU、4线程；无小时硬限，无自动新增组/种子/训练。
- 新输出独立 results/experiments/moe_fixed_weight_validation_20261023。两种正式推断完成即保存暂停。短诊断可直接完成；若实际变成长任务，按项目后台启动规则一次核实真实进展后交接，不持续LLM轮询、不新增监控或唤醒。
- 结束后输出简明中文结论，说明动态分配/专家组合/整个M分别得到何种支持。只在方案完成后做必要集中归档与同步；期间仅本地/服务器Git、配置与回执。不自动上传每一修复，不恢复G、PPO、v2、其它种子或旧任务。
