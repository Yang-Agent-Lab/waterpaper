# RESEARCH EXECUTION SPEC: 李青店/栾川本地适应 × 未来雨量

Status: APPROVED，revision 1.0；2026-10-13（Asia/Taipei）。
变更编号：WP-20261013-LOCAL-RAIN-2X2-V1。
授权：用户对上一轮具体S2起点、两阶段3200更新、单种子最小方案回复“可以，按照你说的。开始来做吧”。本规格把已批准方案写成执行合同，常规实现细节由执行者处理，不扩展训练范围。

## Question and falsifiable hypothesis

区分两站六小时巨大洪峰低估中的本地响应适应不足与未来降雨信息不足，并观察两者是否交互。允许两站结果不同。只改善训练或概率而未改善留出峰幅，不支持巨大洪峰已解决；四条件均弱也是有效阴性，不能据此宣布所有微调或未来雨量无用。

## Starting point

- 代码基础：独立工作树`.`的干净提交`620c0eb82fab3710e652db5cd21dd234ade1e5e5`；S2核心model/runner与原运行`97c39309ac38bfe43f228f3ae790986647c662af`一致。主仓库指标记录提交`1d6ad6f`，复制其`configs/paper_metrics_v1.md`入实施分支，不覆盖其他历史。
- 实施分支：`codex/local-rain-2x2-v1`；复用上述隔离工作树。主仓库既有AGENTS修改及未跟踪文件不碰。服务器通过`teacher-web`，独立checkout `._local_rain_<code_sha7>`；环境`./.venv/bin/python`。
- 起始权重：`._multiwindow_97c3930/results/experiments/multiwindow_formal_97c3930_20261011/artifacts/seed_2/S2/best.pt`，SHA256 `b1061ac7fad0f9f5a54406974bfac7b395d2c0272f241419033d1eb0fe6bb79f`。原选中第1轮，不宣称已充分优化；固定seed2，不按开发困难峰挑种子。
- 原模型、权重、原始资料与旧运行保持只读。新增模块放`src/local_rain_2x2_v1/`，新增配置`configs/local_rain_2x2_v1.json`；复用已有加载、模型数学、分布评分、诊断、MLflow工具，不修改旧模块行为。

## Data and information boundaries

- prepared：`._prob_flood_ab720ba/results/experiments/prob_flood_prepare_ab720ba_20261010/artifacts/data`；manifest SHA256 `e482500d17c6cd900b985c67ff3c6a274b43a892d5f779316032cdf3bfb0ecb6`。复用原来源、前缀排除、窗口身份、分组与尺度，不重划分、不补回unknown前12行。
- 原source_manifest：`._multiwindow_97c3930/results/experiments/multiwindow_formal_97c3930_20261011/artifacts/source_manifest.json`；其源protected风险source_manifest及配置可用来恢复P0/normalization/attribute位置。
- 共同fit原28814窗（中国4554、德国12199、冰岛12061）。select中国2150窗。calibration中国1853、development中国1356，只在所有四条件选模停止后进入最终评价。test禁止读取。
- 目标：`da_tl_cn:Liqingdian`、`da_tl_cn:Luanchuan`。fit分别162/137窗、各3来源事件；select分别74/57窗、各1来源事件；calibration56/64窗，development65/88窗，各1来源事件。重叠窗口不当独立洪水。栾川三个fit事件不满足原完整主峰前6h诊断资格，但保留合法滚动窗训练，不改找峰规则。
- 所有未来Q仅监督/评价，不进输入。新增允许未来P：fit/select阶段只按对应角色window_id恢复随后1–6h降雨；calibration/development未来P在所有选模结束后才恢复。绝不访问test未来P/Q。
- 未来P使用旧窗口文件的`offline_generation_future_p`，沿原prepared manifest记录的来源文件、旧角色、original_pscale恢复物理量，再用本站fit-only pscale归一化。当前fit/select等新角色可能混在旧train文件中，必须在json.loads前按window_id筛选，核对station/source_rows/source_times及历史P/Q和标签Q物理对应。可参考`src/train_gap_audit_v1.py`的fit-only实现，但不可直接调用会画图或扩大角色的主入口。
- 只保存所需6小时未来P，数据与大缓存留服务器。中国P为作者提供雨量，时区/累计区间语义unknown；德国包含作者填补、冰岛为再分析，不统一称实测。新增条件统一标`oracle_rain_diagnostic`，不能当实际天气预报或严格理论上界。不需获取ECMWF/新外部资料。
- 沿用现有已核实峰资格JSON及window_id可避免重复读取完整源Q；核对角色、站点、事件和prepared成员对应，并保存来源hash。若复用原恢复函数，保持事件allowlist和原边界/并列峰规则。
- 所有比较采用同一可配对成员；不因模型表现删窗。缺失未来P如仅非关键少量成员，可对四条件共同限制并记录覆盖；若两站任一必要select/dev主起报点不可配对，停止相关正式比较并报告needs-input，不改划分或从dev挑替代峰。

## Model and fixed comparator

- S2原P0永久冻结、eval；更新现有adapter（雨/流TCN、query/key/value、adapt）、decoder、head。所有条件可更新模块相同，损失继续原S2的四段平均NLL＋权重1的四窗口CRPS，不改分布族、不封顶峰值。
- 新增`Linear(6,64)`残差加到64维历史表示，权重和偏置零初始化。历史条件输入历史最后6小时P；oracle条件输入未来1–6h P。新增支路两侧结构、容量、初值相同；起始输出与原S2一致。未来12h仍只获前6h雨量，M12仅补充，不据其阴性判断信息充分性。
- A：历史输入，原共同fit采样继续学习；B：历史输入，两站各自本站fit微调；C：oracle输入，原共同fit采样继续学习；D：oracle输入，两站各自本站fit微调。

## Two stages and exact selection budget

1. 公共适配：历史、oracle两条轨迹，均从原S2及同一零分支出发；各1000更新，batch128，Adam lr3e-4，clip10。使用相同原country→station→window采样序列及独立配对随机流。1000更新末轮作为固定分叉起点，不在此阶段选有利checkpoint；select仅诊断，不看dev。
2. 分叉：A/C各一条共同继续学习，B/D各两条本站学习，合计6轨迹；各200更新，batch32，fresh Adam lr1e-4，clip10。A/B从同一历史公共权重、C/D从同一oracle公共权重复制；优化器重置规则一致。A/C采样序列配对，本站B/D各自使用相同本站均匀窗口采样序列。更多本站重复曝光是微调变化，须报告，不声称全曝光相同。
3. 总正式更新数上限3200；seed只用2。不得增加种子、学习率网格、解冻范围搜索、损失搜索、扩散或数据增强。无购买资源、不使用多GPU；耗时为估计，不因超过估计擅自增加/重置范围。
4. 分叉候选步数固定0/50/100/150/200。主选模为两目标站各合法select主峰前6h起点的M6中位数绝对误差/本站原fit尺度，两站等权，越低越好。同分取更早。B、D在相同候选步数联合评价两个本站权重，并为每个条件选一个共同步数；不能分别按两个事件择优增加组合数。A/C也用同一两站指标及5次机会。0步可胜出并如实记录。
5. train情景16、select情景128、最终情景512。独立固定随机流，同checkpoint评分不推进训练/采样RNG。样本、初始化、权重hash及追加更新数落盘。工程smoke使用独立短轨迹（公共2更新、各分叉2更新），不占正式候选、不复用为科学结果。

## Endpoints and scientific decision

遵循`configs/paper_metrics_v1.md`，本轮研究范围预先限李青店/栾川，主终点E6及逐站成对差、物理峰值与偏差同时报告。A/C额外报告其他中国站结果观察共同更新代价；B/D只用于本站，不把本站微调强行推广其他站。保留原S2零新增更新作背景参照，不把额外更新收益误认本地适应。

- 主要对照：B-A、C-A、D-B、D-C以及(E_B-E_D)-(E_A-E_C)交互线索，保留两站分别结果。单种子、每站单开发事件只提供探索证据，不虚构置信区间或显著性。
- 风险：四窗口Brier/可靠性，current-low/current-high分列，原Q95与5%calibration报警规则，实际FPR和事件发现/提前时间分开。两站单cal事件的支持量不足保留NOT_EVALUABLE，不从dev定线。地方模型的校准仅对应本站，汇总时两站等权；A/C同样用相应目标站calibration，不用额外站点的校准支持制造优势。
- 分布：CRPS、中央90%覆盖/宽度；普通、上涨、已高升级、退水代价。S2无逐小时流量曲线，NSE/RMSE/MAE逐小时与精确峰时对S2为NOT_APPLICABLE，原P0过程结果只作原参照，不虚构新曲线。
- B仅fit改善而dev不改善：本地记忆更可信；C和D改善而B弱：未来雨量信息有诊断价值，下一阶段才比较实际发布预报；仅D改善：交互值得继续；均弱：检查oracle在fit是否会用雨量及响应误差，不自动追加训练。

## Focused validity checks and outputs

原数据准备/旧模型评分已接受，不重复泛查。新增检查限未来P身份/单位/角色隔离、零残差初始等价、冻结P0不变、更新模块范围、同条件分叉初值与配对样本/RNG、5次联合选模、全部选择完成后最终资料访问、四窗口风险单调以及输出/日志/接续。

允许一次真实fit/select端到端smoke覆盖2个公共适配、6个分叉、联合选模和最终报告stand-in；calibration/development以fit/select替代，标明实际角色，不提前读取最终角色。保留失败，修复仅限同规格工程缺陷；科学阴性不触发重试。

服务器代码/配置提交后用`python3 src/run_versioned.py --run-dir results/experiments/<唯一名称> -- <命令>`。数据准备、smoke及formal分别有run_record；版本不同须分别记录，不在运行中修改源码。formal后台持久运行，文件锁防重复、保存阶段checkpoint、实际进度、失败与完成记录；接续只完成本规格，不恢复旧automations或旧训练。

产物：配置/差异/版本和输入manifest、未来P逐角色访问与配对报告、初值与模块范围、训练/select/健康诊断、公共末轮及6轨迹checkpoint、5候选联合选模表、最终逐窗口/事件成对结果及困难与普通案例、实际耗时/退出码、MLflow run关联；大数组/模型留服务器。

复用现有每50更新健康与固定fit/select诊断（初始、公共末轮、分叉选中状态），区分无梯度、裁剪、非有限、分布中心和宽度。MLflow新experiment `waterpaper-local-rain-2x2-v1`，按阶段/条件/站点/seed分run，不混入旧实验，不重造平台；断线保留本地待同步状态，不重复训练。

服务器作少量关键图：配对训练/select曲线、两站实测峰与四条件预测峰/区间、事件配对差及普通代价；图保存results对应figures，源表与脚本保留。同步小产物并聊天展示；按项目要求同批Preview查看，不打开Finder，不制造多个预览窗口。

## Acceptance and authority

执行者一名，仅写新增包/配置及本轮receipt；主会话拥有本规格、Science与研究判断。未参与实施者核对immutable提交及实际产物，复算主终点/联合选模/主要对照，检查新雨量角色隔离；执行PASS不等于有效。

先记录Science `hzwwxlaxyxnwsxwpjwpt` 的waterpaper同编号并回读，代码准备后补实际提交再进行正式数据处理/训练。飞书仍按原决定暂缓，双台账partial。不得写personal-os，不推送/发布/删除，不访问test或下载外部天气数据。

用户已授权本规格范围的实现、数据配对、smoke、正式运行、诊断、绘图和结果归档。科学选择不再重复询问；出现真实数据比较失效、额外费用或扩大范围时保留partial/needs-input并说明。完成后停止，不追加路线。
