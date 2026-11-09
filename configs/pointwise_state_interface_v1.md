# RESEARCH EXECUTION SPEC: 逐时状态注入接口定位

Status: APPROVED。2026-10-16，用户批准只均值、只方差、只去噪状态注入三组同批对照；沿用上一轮固定的 tau=6 小时状态衰减和全部评估条件。

Question and falsifiable hypothesis: 上轮三接口衰减组在全24h中国select Brier上优于原版，但残差RMS提示方差头可能补偿门控。检验固定衰减下哪些单个接口仍能保留预测收益。若只有一组接近完整衰减组，其他接口可能不是必要条件；若三组均未复现，则当前证据不支持任何单接口足够，交互或多接口联合仍可能重要。单接口失败不能证明该接口在联合模型中无用。

Allowed scientific claim and credible negative outcome: 仅报告相同训练/选模协议下，单接口模型相对floor原版及已完成全接口衰减模型的开发集差异。不得据此认领唯一因果模块，亦不能排除接口间协同。负结果、混合结果与单接口均未复现都有效。

## Starting point

- Repository, baseline branch, and exact commit: 本机独立工作树分支 `codex/pointwise-state-interface` 基于报告分支 `74e0bb1ec93710d764c3168c2e7d60ffc61cbc7d`；原版floor参考源为 `e2c409801e501a3c1df6743cf7fa7a5e02eaceca`，全接口tau6参考运行源为 `d70880cbe7297c2a05900e548602ff234f3c3f7b`。
- Existing evidence and limitations: 上一轮formal compact manifest/review及state_decay、state_constant_mean预测留在独立只读工作树 `pointwise_state_decay_v1`。Decay全24h Brier为0.020883，短时收益保留约86.3%，13–24h仍比原版高约0.366%；方差残差显示固定门控可被训练部分补偿。详细结果以原运行文件和主控独立复算为准。
- Intended change and fixed comparator: 训练 `state_mean_only`、`state_variance_only`、`state_denoise_only` 三组。所有组保留完全相同的19→32→32状态编码器、均值/方差/128维隐藏层三个零初始化heads及固定tau6门控；只让指定的一个接口进入前向损失，其余head保留参数但不参与图计算。共同对照是floor原版DiffPTS与已完成全接口tau6模型，仅复用预测/评分。
- Explicit non-goals: 不改状态特征、数据/标签/阈值、模型主干、损失、门控、optimizer、seed或选模规则；不新增残差头、参数搜索、训练组或交互组合；不重训原版/全接口参照；不读取calibration、reserved、test，不纳入新郑州样本，不混合河流洪水和城市内涝标签。

## Data and protocol

- Data provenance, permitted inputs, and forbidden or held-out inputs: 只读既有pq bank的fit/select，manifest SHA256 `2957aabab9a36247168e3ee05a6cd0f4490ac2f31395b2cd53c384668d7ce81c`；H24起报历史P/Q，未来F24逐时Q与Q95/Q99 fit代理。无未来P。禁止held-out数据、下载或增加样本。
- Independent unit, groups, eligibility, and split or pairing: 原始中国select origins和原event/station层级不变；同银行、同sample_id、country、canonical_station、event，用同一物理站/事件权重配对。保存身份hash并与两个既有对照核对。
- Information allowed at each analysis or prediction step, when relevant: 状态向量只由过去24小时真实P/Q构成；各组使用相同固定 (w(h)=\exp(-(h-1)/6))，仅调制被指定的那一个残差接口。逐时概率保持边际含义，不做时间累计最大。单接口设计不能隔离其与另外两个接口的潜在交互。
- Baselines, controls, confounders, and fair selection budget: 三新组均seed2、同初始state_dict、50 epochs×200 updates、batch64、Adam lr0.001、clip10、select128 draws、final256 draws；统一原pq exposure与公平国家/站点/事件采样。未激活heads保留相同参数但不连入loss、不更新；报告完整参数量、有效接口与梯度状态。它们的未观察梯度是预期状态，不代表运行失败；活跃头的梯度按通常条件检查。随机输入由每步固定generator种子控制，状态分支无随机算子，因此各组共享原序列的随机数消耗。
- Primary endpoint, effect/uncertainty measure, and decision rule: 中国station→inherited-event加权的未来24h Q95/Q99 Brier均值；相同128-draw选epoch、256-draw最终评分。报告相对原版与全接口tau6的差值，不设新的成功阈值。单seed开发比较，不作统计显著或外推保证。
- Secondary outcomes and diagnostics: 固定1–3、4–6、7–12、13–24h Brier，Q95/Q99正负时刻条件Brier，流量CRPS（按fit Q95归一），训练/同定义fit-select loss与选模曲线，固定案例逐时预测。强洪峰阈值与应急分级仍为候选讨论，不用于改本轮训练标签。

## Prechecks and execution

- Domain-specific validity checks, including leakage or degenerate-result checks: 检查四种状态模型（完整与3个单接口）总参数量及state_dict初始化相同；每个单接口只改变对应输出，另外两条路径严格等同原版；用同generator seed核对扩散训练噪声一致、前向有限且活跃head梯度有限。smoke每组恰2 updates，要求每组真实预测、评分、training_report及活跃/停用head状态完整。
- Branch/worktree, compute environment, and resource budget: 本机源码工作树为公司waterpaper `.worktrees/pointwise-state-interface`；服务器工作树唯一路径 `.`。检查、训练、分析、绘图只在teacherbox的固定pointwise环境运行；显式设置项目固定的DiffPTS/PPM作者代码checkout。正式训练合计30,000 updates；smoke为每组2 updates、batch4；无额外种子/训练。
- Required commands or methods to discover and run: 源码/spec/config commit后，服务器核对干净HEAD；通过run_versioned运行单接口隔离自检与端到端smoke，再以run_versioned后台detach正式3组。GPU运行使用既有pointwise lock；不创建唤醒、cron或App自动化。
- Required tables, logs, manifests, versions, and true exit codes: 保留run_record、stdout/stderr、各组condition/epoch/health/checkpoint/select_predictions/scores/select_rows/三页training_report、comparison review JSON、compact manifest、最多一张overview及已有固定案例图。父会话对新组预测做一次独立复算。
- Diagnosis required for negative, weak, abnormal, or non-evaluable results: 组失败同时保存run-root与model目录failure.json，review可读失败原因；不覆盖已有结果，不更改数据、tau或指标来挽救失败。区分未激活head的`declared_inactive`/无梯度和活跃head的真实数值故障。

## Acceptance and authority

- Completion criteria for the worker: commit批准规格和实现；接口自检及smoke通过；正式3组各50×200并完成选模/256-draw最终评分；run_versioned状态与compact manifest和预测文件一致；结束后停止本轮。
- Independent acceptance: required；主控独立读取新三组saved draws复算，工程PASS不代表研究假设通过。
- Correction scope and stopping conditions (time, compute, or material blocker): 只修复本规格内实现/工程问题；数据身份、服务器权限、已批准预算或对照发生材料变化则停并回报。正式失败不追加预算、种子、tau或新模块。
- Post-acceptance continue/stop/revise criteria for the parent: 以新三组与原版/全接口衰减的完整主指标、时段分层和案例决定下一方向；单接口不足只能说明本次改法单独不够，不能断言无交互价值。
- Authorized external effects: 公司项目内独立Git提交、teacherbox指定worktree同步、运行本次smoke及3组正式训练、保存结果与少量图。
- Forbidden external effects: 不写Science或飞书、不推送/发布、不访问held-out数据、不更改旧结果、不删除/迁移资料、不新增郑州样本、不新建唤醒或自动化，不使用GUI/鼠标键盘。
