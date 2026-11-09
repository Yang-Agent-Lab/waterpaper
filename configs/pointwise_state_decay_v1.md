# RESEARCH EXECUTION SPEC: 状态作用随未来提前量衰减

Status: APPROVED。2026-10-16，用户批准开展状态逐提前量衰减比较；随后主控在授权范围内固定 tau=6 小时，并加入等均值常数门控控制。不增加训练组、不搜索 tau。

Question and falsifiable hypothesis: 已完成的 `state_pq` 在1–3h Brier优于原版、13–24h Brier较差。检验固定 (w(h)=\exp(-(h-1)/6)) 是否保留短提前量收益并减轻远期代价；最强控制为全24小时都使用同一 (w_c=\mathrm{mean}_{h=1..24}w(h)\approx0.2664)。如果衰减组未优于原版全24h主指标、没有保留短时收益，或常数控制表现相当，则不支持“按提前量衰减有额外价值”。单种子结果只作探索证据。

Allowed scientific claim and credible negative outcome: 报告固定衰减与等算术均值常数门控在指定开发集上的差异。不得将其称为中国业务预报、校准风险阈值或当地指南；未改善原版、远期仍退步、或常数控制相当都是有效负结果。高低提前量单独改善不能替代全24小时主指标。

## Starting point

- Repository, baseline branch, and exact commit: 项目仓库 `waterpaper`；独立分支 `codex/pointwise-state-decay`，基于 `codex/five-country-model-data-v1` 的 `ff67ea4c186588fed229dbb636fdd6e97bf8c3cb`。旧全程状态源码为 `150f5371fbed9bb9a7f17f2bfa93c6e16cdad159`；原版方差下限对照为 `e2c409801e501a3c1df6743cf7fa7a5e02eaceca`。
- Existing evidence and limitations: `data/catalog/pointwise_alignment_state_results_20261016.md` 已记录旧state_pq 1–3h线索、13–24h退步和全24h mixed结果。单seed2、中国现有select开发数据，不是独立test。
- Intended change and fixed comparator: 同一19维历史P/Q状态编码器、相同32维状态层及三个零初始化残差头；仅将状态残差分别乘固定lead权重。对照为同bank、同seed、同训练配置复用的floor原版与全程state_pq；另训练衰减组和常数均值门控组。
- Explicit non-goals: 不增加状态特征、模型参数、训练组、seed、tau搜索或未来输入；不重训原版/state_pq；不改损失、阈值、样本、数据划分、Q95/Q99定义或选模方法；不访问旧test/calibration/reserved；不恢复唤醒、定时或自动续训。

## Data and protocol

- Data provenance, permitted inputs, and forbidden or held-out inputs: 只读既有pq样本bank的fit/select文件，manifest SHA256 `2957aabab9a36247168e3ee05a6cd0f4490ac2f31395b2cd53c384668d7ce81c`。历史输入为起报时已有P/Q，H24；预测未来逐小时Q概率，F24，不使用未来P。Q95/Q99为各训练站点fit阈值代理。禁止calibration、reserved、test及任何新数据。
- Independent unit, groups, eligibility, and split or pairing: 评分使用原有中国select sample IDs；层级权重为国家内站点等权、站点内继承事件等权、事件内起报等权。保存sample_id、country、canonical_station、event身份散列并校验四模型一致；同暴雨多站及重叠起报不视作独立统计重复。
- Information allowed at each analysis or prediction step, when relevant: 状态编码只取过去24小时P/Q；(w(h))只由固定未来提前量决定，不依赖未来观测、站点、事件或select结果。逐时风险保持各小时边际含义，不做时间累计最大；门控系数递减不保证训练后的状态修正幅度递减，因为24个lead各有可学习的均值和方差残差输出，模型可能部分补偿固定权重。用固定样本的门控前后残差RMS检查这种补偿；观察到正结果也不能仅凭此归因于门控。
- Baselines, controls, confounders, and fair selection budget: 两个新组均seed2、50 epochs×200 updates、batch64、Adam lr0.001、clip10、128 select draws、256 final draws；使用同一原始pq fit曝光、sample bank、选模与全量select评分。原版与state_pq仅复用e2c4098/150f537完成结果。常数控制只匹配24个lead权重的算术均值，不声称完全排除正则化影响。
- Primary endpoint, effect/uncertainty measure, and decision rule: 中国station→inherited-event层级加权、未来24h逐时Q95/Q99 Brier均值（无单位、越低越好）；每组由相同128-draw规则选epoch，256 draws作最终评分。四模型结果表报告相对floor原版的全24h百分比差、1–3h及13–24h Brier；13–24h分别对比原版和全程state_pq。可描述短时收益保留比例=(原版1–3h Brier−新组1–3h Brier)/(原版1–3h Brier−state_pq 1–3h Brier)，仅作描述，不是选模指标或阈值。仅单seed开发结果，不作显著性或泛化保证。
- Secondary outcomes and diagnostics: 固定分层1–3、4–6、7–12、13–24h Brier；旧协议的流量预测分布CRPS（按fit Q95归一）；Q95/Q99洪水时刻与非洪水时刻概率Brier；固定困难案例预测、在真实值最高lead上比较训练报告可复核的中心值与final边际样本；在相同固定样本上保存均值/方差状态残差门控前后逐lead RMS，并与全程state_pq同样计算结果对照；统一training_report_v1三页及其同定义train/fit/select loss曲线。强上涨案例若训练报告无对应真实值则记unknown。过拟合仅作曲线形态解释，不能由此单独定因。

## Prechecks and execution

- Domain-specific validity checks, including leakage or degenerate-result checks: 先做w=0/1端点均值、方差、denoiser、loss分项和固定随机数samples逐值一致检查；检查batch重复顺序、逐lead广播、常数门控均值、参数量不变及有限非零梯度。runner只允许fit/select bank且校验prior checkpoint、completion、bank与sample IDs。
- Branch/worktree, compute environment, and resource budget: 本机独立Git worktree位于公司waterpaper `.worktrees/pointwise-state-decay`；服务器源码worktree唯一路径 `.`。检查、训练、评分和绘图只在teacherbox `.`下的固定环境执行；命令环境显式设置 `WATERPAPER_DIFFPTS_SOURCE` 至作者commit `e9a7d1d56bd40b4bcfced29b430f6c5be6f2a424`、`WATERPAPER_PPM_SOURCE` 至作者commit `6b4206a8c6bd3d298d8bdc58ebc88e5f5f6c13f9`，沿用项目已核启动脚本配置。本轮新增正式训练共20,000 updates；端到端smoke每组2 updates、batch4；不增加其他训练。
- Required commands or methods to discover and run: 端点自检、两组smoke（含保存预测、选模评分、统一报告与概要）、检查实际完成后，再以`src/run_versioned.py`提交版本身份运行正式两组。正式任务detach后台运行；无自动化/cron。运行前核对服务器代码commit及worktree干净状态、设备和已有GPU任务。
- Required tables, logs, manifests, versions, and true exit codes: 保存run_versioned记录/日志、conditions、runtime、checkpoint、逐epoch曲线、原始预测、scores、select_rows、training_report_v1、两组比较身份、compact manifest、review JSON和固定案例/总览图。训练成功只表示执行完成，科学判断依据实际select保存预测与复算结果。
- Diagnosis required for negative, weak, abnormal, or non-evaluable results: 保留失败/部分完成状态；核对保存样本预测及概率、Brier、CRPS复算；报告有效/不可评价案例与训练曲线。不得调tau、换指标、重挑事件或追加训练挽救结果。

## Acceptance and authority

- Completion criteria for the worker: commit规格和源码；端点检查通过；smoke每组恰2 updates且有保存预测、评分、报告与概要；正式两组均在唯一服务器worktree从该干净commit启动并保留真实退出状态、每组epoch/update/score状态及产物；总览图最多一张，另保留原固定案例图。实现执行者完成一轮后停止。
- Independent acceptance: required；主控未参与实现，由主控核对compact manifest、固定样本与主要分数，执行完成不等于假设通过。
- Correction scope and stopping conditions (time, compute, or material blocker): 只修复本规格内的实现错误；若数据身份、先验对照、预算、scope或设备授权发生材料变化即停止并回报。训练失败或negative均结束本轮，不改tau、不补seed、不重训。
- Post-acceptance continue/stop/revise criteria for the parent: 只有实际结果支持全程主指标和目标提前量收益后，主控才讨论下一步；混合或负结果保持候选失败/mixed，不自动派生新实验。
- Authorized external effects: 本地/服务器Git提交、将本次版本传到teacherbox、在指定服务器worktree运行本次smoke与两组正式比较、保存项目内结果及同步少量图供本机Preview查看。
- Forbidden external effects: 不写Science或飞书（Science由主控处理；飞书按既有决定partial）、不推送公共远程、不发布、不读held-out数据、不迁移或删除已有资料、不付费、不新增唤醒或自动化。
