# RESEARCH EXECUTION SPEC: 冻结状态表示上的有序洪水风险读出

Status: APPROVED。2026-10-16，用户已批准CORN尝试与两头比较；主控确认本spec的冻结checkpoint、特征、损失、初始化、公平条件和报告适配。编号：`WP-20261016-POINTWISE-CORN-READOUT-V1`。

Question and falsifiable hypothesis: 在冻结的state_variance_only预测表示上，直接学习逐时Q95/Q99风险，是否比独立二元读出更好地利用阈值有序信息，并在不改变已有流量主干时改善中国风险Brier？比较CORN的“超过Q95”概率与“超过Q99 | 已超过Q95”条件概率，和同表示/同容量的独立Q95、Q99 BCE头。若CORN不改善主Brier、或其排序一致性没有带来风险收益，则本次证据不支持该读出值得保留。单次同表示读出实验不估计端到端联合训练的上限。

Allowed scientific claim and credible negative outcome: 仅报告固定state_variance_only表示上的两种风险读出差异及与已有冻结模型概率的对照。不得声称CORN独立导致全模型洪水预报或连续流量分布改善，也不得据此推断状态/方差接口的因果机制或跨种子稳定性。负结果、混合结果和两种头都未优于基准均可完整报告。

## Starting point
- Repository, baseline branch, and exact commit: 公司工作树 `.worktrees/pointwise-corn-risk-v1`，基于前轮工作树报告提交 `1a25c4ae842c37456b49550df3920ae79eaf5541`；已运行模型实现源 `82d9d485275c49d0d3552302381fa0ea8c4a9cd0`。Git工作树为独立分支；primary及前轮运行源码保持只读。
- Existing evidence and limitations: 前轮variance-only模型选择epoch23，checkpoint SHA256 `a27ed514a4e95b933891e891ac48b2a3e4aae09f1d7408f2c7d71be038324c8f`，同源训练50×200更新；全24h Q95/Q99 Brier `.020824`，单seed探索。该checkpoint完整预测接口含mu、方差和19→32→32历史状态编码器；源记录在 `./results/experiments/state_interface_formal_82d9d48/artifacts/state_variance_only_pq/`。
- Intended change and fixed comparator: 固定加载上述checkpoint，冻结主干、state encoder及所有buffer，全部保持eval且只用no-grad前向。每个未来时刻的风险head输入为 `[z(32), signedlog1p(mu)(1), log(max(g,1e-8))(1), lead/24(1)]` 共35维；这些固定的可逆/数值稳定变换不改原始P/Q或主干，不声称解决旧native loss震荡。CORN和独立二元头均使用 `35→32→2` MLP、各1218参数，同参数数、同seed2随机隐藏层初始化及同训练暴露。CORN输出 `p95=sigmoid(a)`、`p99=p95×sigmoid(b)`；独立头输出两个未经排序/校正的sigmoid概率。已有原版floor、全接口tau6及state_variance_only的保存预测只读复用；state_variance_only是同冻结主干的采样概率参照。
- Explicit non-goals: 不解冻/重训主干；不改P/Q输入、阈值定义、sample bank、样本划分、未来逐时概率含义或流量损失；不加入weighted loss、类别重采样、后验排序/校准、额外模型、seed、τ或郑州新样本；不读取旧calibration/reserved/test；不把风险head包装为连续流量预测器或新增CRPS。

## Data and protocol
- Data provenance, permitted inputs, and forbidden or held-out inputs: 只读五国原pq bank fit/select，bank identity `2957aabab9a36247168e3ee05a6cd0f4490ac2f31395b2cd53c384668d7ce81c`。输入仅起报时已有过去24h P/Q及冻结网络的确定性预测表示；监督标签为未来24h逐时Q与站点拟合Q95/Q99阈值。不得读未来降雨或held-out roles。
- Independent unit, groups, eligibility, and split or pairing: 同一country、canonical station、inherited event和sample_id；主结果为中国select，按站点再按事件等权；已知为1706起报/9站/17站点-来源事件，不冒充1706独立洪水事件。两组使用相同fit/select rows和同一固定曝光数组。
- Information allowed at each analysis or prediction step, when relevant: 对每个样本未来时刻：`class=0` 若 `Q≤Q95`；`class=1` 若 `Q95<Q≤Q99`；`class=2` 若 `Q>Q99`。CORN首任务估计 `q1=P(class>0)`；次任务只在 `class>0` 风险集拟合 `q2=P(class>1|class>0)`；无条件风险为 `p95=q1`、`p99=q1*q2`。空次级风险集安全跳过，不形成NaN。独立头分别拟合 `P(Q>Q95)` 与 `P(Q>Q99)`，不排序、不修正，并报告越序比例。风险只沿严重等级轴嵌套，时间轴允许升降。
- Baselines, controls, confounders, and fair selection budget: 每个风险头训练50 epochs×200 updates、batch64、seed2、Adam lr=.001、clip norm10，总20,000 head updates。冻结checkpoint来自同已完成variance-only结果；三种旧模型只读。CORN按原文条件子集及总有效riskset样本数归一；首任务对全部样本算稳定logit BCE，次任务只对Q95-positive子集算BCE，空次级子集跳过。实现用公式对应的stable BCE；服务器若已有作者`corn_loss`可直接调用则另做输入/梯度数值核对，不安装完整新库。二元控制对两个边际目标计算无类别权重BCE并取平均。两组用相同35→32→2容量、曝光序列和训练预算；因目标/loss定义本身不同，不能称其优化目标完全相同。初始输出采用全fit、国家等权且国家内站点→事件等权的类别先验；三个等级概率加固定微小平滑后得到共同的p95/p99，分别转换成CORN的q1/q2和二元头bias，使两头初始边际概率完全匹配；输出层权重置零，隐藏层seed2初始化相同。伪计数只用于有限logit初始化，不参与训练损失。
- Primary endpoint, effect/uncertainty measure, and decision rule: 中国station→inherited-event加权、未来24h逐时Q95/Q99 Brier均值；每epoch用同一select primary Brier选checkpoint，直接头概率确定、不使用128/256抽样。对比两风险头、已有variance-only采样概率、原版floor和全接口tau6；报告差值和相对原版百分比，不设事后阈值或显著性声称。单seed开发比较。
- Secondary outcomes and diagnostics: 固定1–3、4–6、7–12、13–24h Q95/Q99 Brier；Q95/Q99正、负时刻条件误差；独立二元头越序比例；逐h概率可靠性；两固定case逐时风险曲线（sample IDs `40b87956858b6939584c4742b552fc801f3d6db5829124a3e4afb3b3f956128f`、`03775f54c578324cd679632e06629242291e0e0a205a71b1077e0ede2e95282b`）；CORN/BCE各自同定义训练与fit/select loss及Brier学习曲线。新风险head的连续流量CRPS记为 `NOT_APPLICABLE`；旧state_variance_only保留其已保存draws得到的CRPS并标清来源/256 draws，不与风险头声称连续分布等价。

## Prechecks and execution
- Domain-specific validity checks, including leakage or degenerate-result checks: 检查所有fit/select阈值 `Q99≥Q95`、标签仅来自既有未来Q、冻结表示无NaN且相同输入重复推理完全一致；验证同checkpoint hash和bank/sample identities。每个训练步骤确保主干所有参数及buffer始终位级不变且无grad；仅风险head更新。自检CORN概率恒满足`p99≤p95`但不累积时间轴；独立头保留并测量违序；手工检查CORN首/次风险集及空次集loss有限、梯度有限，并用小手算例与论文公式/可用作者实现核对数值。smoke两头各2 updates，覆盖选择评分、概率保存、报告与失败记录。
- Branch/worktree, compute environment, and resource budget: 本机源码只在公司waterpaper `.worktrees/pointwise-corn-risk-v1` 编辑/Git；服务器唯一新项目根 `.`。teacherbox既有pointwise环境；全部检查、训练、评分、绘图只在teacherbox；两头正式训练合计20,000 updates，不自动追加训练。
- Required commands or methods to discover and run: commit后server核对干净HEAD；以run_versioned记录冻结主干/风险head接口自检及每组2-update端到端smoke；通过后后台detach两组正式任务。全程使用既有GPU lock，不创建wake/cron/App自动化。
- Required tables, logs, manifests, versions, and true exit codes: run_record、完整条件、固定base checkpoint identity/hash、fit/select/exposure与case IDs hash、冻结weights/buffers before-after hash、两风险模型checkpoint、逐epoch损失/Brier、所有select行和真实概率文件、独立review及compact manifest、risk-only training_report_v1 bundle和必要图。保存确定性概率，不生成MC samples。
- Diagnosis required for negative, weak, abnormal, or non-evaluable results: 失败与`NOT_EVALUABLE`状态如实保留。CORN无次级riskset时跳过该任务；若整个head无有限loss/更新则失败并保留证据。若新头Brier降低但越序/正负时刻误差显示代价，完整汇报；不改阈值、训练组或比较metric挽救结果。

## Acceptance and authority
- Completion criteria for the worker: APPROVED spec、实现和配置先commit；focused self-check及两组端到端smoke通过；正式两头训练各50×200并结束评分；真实run_record exit、预测与报告、冻结base identity相互一致；提交compact manifest并停止。
- Independent acceptance: required；父会话直接从新两组保存逐时概率复算主Brier与固定分层，核对风险概率、sample identities和原读出参照。PASS只表示执行与报告可信，不表示方法成功。
- Correction scope and stopping conditions (time, compute, or material blocker): 只修复本规格实现/工程问题；如checkpoint、银行、阈值身份或授权预算改变，先停并报告。负结果、阻断或smoke失败不增加训练组、seed或类别重采样。
- Post-acceptance continue/stop/revise criteria for the parent: 依据完整主Brier、各段误差、越序比例、正负误差与固定案例决定是否值得进一步开发；即使CORN优于二元头，也只支持这个冻结表示上的风险读出，而非端到端收益或稳定性。
- Authorized external effects: 公司项目隔离worktree内提交、teacherbox指定worktree同步、本规格授权的smoke和两组正式训练、保存本实验产物及有限图。
- Forbidden external effects: 不改primary或前轮源码/结果；不触碰calibration/reserved/test；不加数据/郑州样本；不写Science或飞书；不推送/发布/合并/删除；不恢复唤醒/cron；不使用GUI/鼠标键盘。

## Method sources
- CORN paper, defining chain-rule probabilities, conditional risk sets, and summed conditional BCE: https://arxiv.org/html/2111.08851v3
- Author implementation entry point (`corn_loss`); numerical behavior must be checked on the server before use: https://raw.githubusercontent.com/Raschka-research-group/coral-pytorch/main/coral_pytorch/losses.py
