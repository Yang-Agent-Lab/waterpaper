# RESEARCH EXECUTION SPEC: 外国极端过程使用 × M条件中心连接先导

Status: APPROVED
Change ID: WP-20261101-EXTREME-PROCESS-TRANSFER-PILOT-V1
Version: r1.4
Approval: 用户在2026-11-01对前述四组探索、普通时段保护、N-BEATSx对应数据对照及简单中心头必要对照明确回复“可以，按照你说的来做”。本文件将已批准方向具体化为有限小试；不授权全量训练/测试或自动后续。

## Question and hypothesis

在不增加中国资料与可得信息的条件下，国外极端过程的均衡使用，能否通过M直接修正扩散条件均值的通路，提高中国尾部分布与峰幅，同时维持普通时段精度？因素D为国外抽样，因素C为中心修正连接。
可信负结果：仅抽样有效、仅中心头有效、组合无增量、正常精度退化、简单头足够，或N-BEATSx同样/更好获益。不得按结果更改主指标、删困难站点、换种子或扩大范围。

## Starting source and ownership

- Repository: waterpaper独立Git；实现基线c9776c73618b4bd5199598e207fb51d562514497。
- 本机受控工作树 .，branch codex/extreme-process-transfer-20261101。
- 父会话负责本spec及主任务轻量回执；科研工程师独占src/extreme_process_transfer_v1/、configs/extreme_process_transfer_pilot_20261101.json；如wrapper确实不支持指定输出根，允许在新工作树内最小增加已接受的显式output-root接口并做逃逸拒绝检查，不改旧运行树。
- 旧模型代码优先只读继承/包装。需要改旧模块时仅本工作树、明确diff、维持旧行为默认值；不得原位改变旧run源码。
- 固定B链：lj ./results/experiments/real_extreme_AB_diffpts_gpu2_20261028/B_diffpts/artifacts/ 下C/state/gate/D500/M最佳权重及fit-only统计。加载实际工厂链，逐文件完整SHA写manifest；不得拼接旧实验FrozenD500权重。
- M四主组和简单中心头均从同一个B-M已选权重热启动；冻结同一B-D500全部参数。
- 强外部N-BEATSx使用同一原作者概率配置/neuralforecast==3.1.0，两个采样组从相同seed2初始化各训练3000步。无需等待yxz旧权重即可独立重建该有限对照；不是伪称已恢复旧权重。旧B-NBEATSx已验收分数可作为明确标记的历史参考，不能替代当前预测或普通分层指标。

## Data and exposure

- 唯一训练输入：teacherbox ./data/processed/real_extreme_ABC_v1_20261028/ 的完整B合法fit及固定262 select；实际manifest、fit/rows/indices/select SHA与形状核实。
- 已确认fit 268009行：CN3388、foreign264621；history [N,24,2] P/Q、future_q [N,24]、q95/q99/flow_scale；原fit阈值、尺度、身份不重拟。
- 只使用已有五国：中国.50、德国.20、冰岛.10、美国.15、日本.05。输入历史24h P/Q，未来24h Q监督；预测时不使用未来实测P/Q、severity或early_extreme标签。
- 原B层级基准概率p_B、原severity=0/1/2（未来24h峰不超Q95／超Q95不超Q99／超Q99）来自fit监督，可用于训练抽样。保留early_extreme作曝光统计，不另加第三种抽样因素。
- natural条件使用p_B。foreign-balanced条件仅在各外国国内采用0.5*p_B(i|country)+0.5*(1/n_available_classes)*p_B(i|country)/P_B(class(i)|country)。中国保持原p_B，不用旧p_C（旧p_C也改变中国）。
- 国家槽位与中国索引序列在各组逐位置相同；不同数据条件仅外国索引不同。同一条件下M与NB共用相同暴露序列前缀；NB扩展到3000*64时按同一规则继续。
- 拟合和普通保持项使用逐样本w=p_B/p_sampler；中国w=1。验证精确概率归一化与加权期望恒等式，记录真实/加权类曝光，不裁剪权重代替正确实现。保留原路由balance项定义并说明它不是逐样本严格等价目标。
- 不能用102459行旧执行缓存做新抽样。需要传输完整必要数据时只在项目服务器根/本run inputs下，校完整SHA；允许scp -3网络流经Mac，不在Mac落大数组，不传凭据。
- 统计单位保留起报、唯一站点小时、来源过程三层；262/9站/17来源不是262独立洪水。无新test/全1706/国家/种子。

## Seven finite configurations

| ID | 数据条件 | 可训练部分 | 更新数 |
|---|---|---|---:|
| M_N_E | natural | 原M去噪适配器，原中心 | 1200 |
| M_X_E | foreign-balanced | 原M去噪适配器，原中心 | 1200 |
| M_N_ME | natural | 原M适配器+中心头 | 1200 |
| M_X_ME | foreign-balanced | 原M适配器+中心头 | 1200 |
| H_X_MU_ONLY | foreign-balanced | 简单中心头；旧M和D500参数均冻结 | 1200 |
| NB_N | natural | 原作者N-BEATSx概率模型 | 3000 |
| NB_X | foreign-balanced | 同一N-BEATSx概率模型 | 3000 |

seed固定2。M批量64、辅助分布反向至多前8起报/K8/完整20反向步，沿原实际优化器、学习率、clip和精度；配置写出真实继承值。M已是成熟B权重，辅助CRPS_U/ES_U系数从起始即各0.1，balance0.01，不重复旧冷启动warmup。
NB批量64、17个原分位数（包括.99/.995）及原learning-rate/optimizer/架构，不为新结果增加调参。总M正式更新6000、NB正式更新6000；工程更新另外记录。
无墙钟硬上限，不抢占别的作业；固定配置/步数结束即退出，next_job=NONE。

## Center connection and gradient semantics

- delta_mu = small_head(existing_M_history_context, lead)，只从历史表示/起报可得条件生成未来各h修正；末层全部零初始化，起始mu_new=mu_base。
- 首轮不新增动态路由、专家、相似性学习模块、额外未来输入或独立尾部loss。
- 原D500 conditions可no_grad；delta_mu必须在其外计算。mu_new进入同一个完整采样初态、所有反向步骤、adapter.prepare/view和训练q_sample/denoiser；不能只改报告/损失中的mu。
- 原g保持固定父模型定义；不得把方差/标准差混用。Q95归一化条件下q95=1的旧代码语义保持。
- 冻结参数不阻断对mu_new输入的梯度；simple中心头组也保留通过旧M和全部采样步骤对mu_new的导数。
- zerohead原预测等价、两个头梯度可达、完整辅助分布损失到中心头梯度、原D500参数和缓冲未变化必须实测。

## Common ordinary-process protection

普通训练起报定义为未来24h最大真实Q<=原fit Q95；标签仅训练期使用，推断不提供。
冻结起始B-M作共同教师。每批复用同一x/t/noise，由教师mu/g构造同一y_t，比较学生与教师的epsilon输出；加mu_new与mu_teacher差的平方（已在Q95单位）。
L_keep = 0.1 * weighted_mean[ordinary_mask * (mean((epsilon_student-epsilon_teacher)^2)+mean((mu_new-mu_teacher)^2))]。
与原M预测目标同样按p_B/p_sampler加权、按整个基准批量归一；无普通行时该项明确0，不跳过优化其他项。教师输出detach且参数/缓冲不更新。
这是局部去噪/中心一致性代理，不声称严格保持完整分布；真实普通预测指标独立评价。
四主M组与简单头组共用此约束和权重。NB使用原生分位数拟合和同一概率修正，不强行套扩散教师；这是原生强外部路线，预算/训练目标差异如实标注。

## Checkpoints, selection and evaluation

- 主因子比较固定末步：M1200、NB3000，避免按新尾部结果选择有利轮次。
- 同时保存原协议兼容的best_Brier辅助轨迹：M在0/400/800/1200；NB在0/1000/2000/3000，K32相同逐时Q95/Q99 Brier均值、最早最小值。不得逐指标挑checkpoint。
- 终评K64：各组fixed_last及best_Brier，若为同一个权重则去重；两种选模口径分表。旧强外部选模不能因不利而省略，不能只超过NB末步就宣称超过最强NB。
- 主冻结B-M/D500参考在同一262/NoisePlan下只准备一次；同身份旧预测可复用但必须实际核噪声、sample_id、GT和形状，不匹配则在本run唯一生成。
- 所有M抽样使用共同可续NoisePlan、相同样本ID与K/20步，保存种子、规划/权重SHA；NB仅原边缘分位数的确定性K32/K64网格，不伪造联合情景。
- 主结果：全样本twCRPS95/99和H6/H24点中位数曲线峰幅MAE，四组因子效应、绝对差/百分比及组合相对单因素；同时展示普通和总体代价，不事后改指标。
- 普通目标单元Q<=Q95的MAE/RMSE/CRPS及Q95/Q99 Brier；另列全窗口普通起报的支持及同类指标，缺支持NOT_EVALUABLE。
- 完整约定指标保留：全体MAE/MSE/RMSE、有效NSE、CRPS物理及/Q95、tw95/99、总体/起涨Brier、fit-Q95/Q99条件及eval-Top10/5/1、H6/H24峰幅MAE/相对误差/偏差和峰时MAE/偏差、90%覆盖/目标偏离/宽度。先复用已验证数学与层级加权；单seed无SD或稳定性主张；NSE精确常值先识别，不能浮点伪SST。
- per-h1..24保存，预列1/3/6/12/24展示；逐站/来源/困难案例保留，不能只挑改善案例。联合情景分数只给有资格的扩散模型，NB为NOT_EVALUABLE。
- 常态保护判断不预设有利阈值：逐项报告普通误差与风险的变化；尾部/峰幅改善但普通指标退化则mixed，不自动晋级。
- 新增组核心数值齐备且可复查即可完成；负/混合结果同样结束，不自动扩种子/重训/调损失。结果后由用户决定继续。

## Runtime, versioning, output and visibility

- 首选lj .，源码 worktrees/extreme-process-transfer-20261101/。
- 唯一远端output-dir ./results/experiments/extreme_process_transfer_pilot_20261101_v1/；inputs/checks/smoke/runs/figures/logs/tmp均在此run。若转用teacherbox，必须在运行前将其实际项目根/输出路径记录为明确修订，不把Mac路径当服务器。
- 复用lj environments/real_extreme_v1；NF缺依赖时直接建本任务隔离环境并补齐neuralforecast3.1.0及必要兼容依赖，不升级无关共享环境。环境和实际版本入manifest。
- 完整可续队列最多2张空闲GPU（首选0/1），每次实际申请前核空闲并用项目可用锁，不杀其他作业；后台脱离当前SSH，有限配置，无轮询唤醒。
- 先检查改动范围并commit，再正式数据准备/科学smoke/训练/评分；通过src/run_versioned.py启动，保存实际HEAD/dirty/cwd/argv、stdout/stderr、状态、权重、数据SHA、真实退出码。原输入/权重不入Git。
- 必须提供真实持续watch命令（阶段/步数/进度条、CPU与墙钟、最近验证指标单位和方向/更新时间），不是cat一次JSON。
- 统一报告复用training_report_v1，学习loss与同定义验证loss、真实案例、概率/预警及必要分层；健康/梯度保存数值，不额外凑诊断图。主聊天全指标用“绝对变化+相对改善/退步”，强外部同表。
- 本机轻量回执 ./results/experiments/extreme_process_transfer_pilot_20261101_v1/；源/配置/必要说明可回传，不回传大数组/模型环境。已登记begin-run，收尾check-run；这不代替服务器写出检查。
- 新方案未结束前不做Science/Drive/Docs中途上传。已读唯一原生Docs，modified2026-10-23、最后review WP-M-DIST-20261023-FINAL-001基于6d10d9e已答复/同步；不是当前c9776c7新改造的未处理审查。本spec用户新批准不来自旧ACCEPT。

## Engineering checks and launch authorization

实现者负责一次真实端到端smoke，目标服务器、同正式入口、所有7配置及模式：
1. 数据身份/窗口角色、外国-only概率、中国序列共同、权重期望；真实262身份和GT保持。
2. 零头输出等价、每步mu路由、两类头与完整20步分布梯度；simple头与全冻结参数身份；数值finite，不用裁剪非有限或fallback掩盖问题。
3. 教师同y_t/noise/普通掩码与权重，教师与原D500无参数/缓冲更新；有普通/极端/零普通/各国的真实小批。
4. 至少一次每主模式优化与真实validation、保存/重载、预测/评分/报告、best/last去重；NB真实GPU分位数训练/预测，不以mock替代。
5. 3+1与4步恢复等价（RNG/optimizer/采样/NoisePlan）；队列从上一配置结束转下一配置、失败退出、next_job=NONE；独立入口实际输出不散写。
工程更新与正式更新分账，不复用工程权重冒充正式起点。实现修复按原scope自主进行；主要数据/研究问题/组数/指标变化回主控，不由工程师重写。
独立reviewer核真实源码、spec/config、source/data/权重SHA及受影响真实路径。PASS只指实现可用于本试验，不是预测有效。
用户已授权：同一spec通过自检、独立复核和CEO针对性验收后，工程师直接后台启动有限正式队列，无需再问许可。主控确认一次真实进展、无启动错误、脱离连接、版本/预算/日志/watch入口后回报并结束主动监督；不等首配置或整批结束。
若NF环境准备独立较慢，可先让它准备，同时M工程检查继续；正式队列不得假装未就绪分支已运行。是否分批启动由主控按已验证依赖决定。

## r1.1：teacherbox准备输出位置（工程接续，不改科学条件）

- lj正式训练根/七组/步数/数据与评价不变。现有lj传输与新SSH命令等待、teacherbox可连接，因此允许先在teacherbox完成完整B的输入身份/抽样概率/共同曝光、轻量数学检查与NF隔离依赖准备。
- teacherbox唯一准备用output-dir为 ./results/experiments/extreme_process_transfer_pilot_20261101_v1/；准备产物放其preparation/、checks/、logs/、tmp/，不要写原data/processed或旧run。
- teacherbox源码可位于 ./，先提交/核版本，再用同一versioned入口写上述准备run；依赖环境位于该项目environments/extreme_process_transfer_pilot_20261101/。
- teacherbox产物在manifest中标role=preparation_only、host=teacherbox。正式run仍为lj同名run，传入时校SHA；不把两host路径混写成同一次训练结果。
- 此条不批准在teacherbox绕过缺失B权重做替代训练；若后续需要把真实模型smoke或正式训练转teacherbox，再由主控明确修订实际权重/环境/输出路径。Mac仍不运行科学检查。

## r1.2：本轮eval-Top定义澄清（沿原数学，仅固定262人口）

- 原已接受实现是development_metrics_complete_v1/metrics.py的eval_top_thresholds（源协议commit1496264593a68893b406467ac5bbd7fc5baef94f）：以站点+origin_time+(h+1)小时为真实目标时刻，先去重；重复真值须np.isclose(rtol=2e-6,atol=2e-6)一致，否则失败。
- 本轮只在固定262的未来24小时真值并集中，按站点去重后用np.quantile(values,[.90,.95,.99])默认linear方法；采用严格truth>threshold，等于阈值不计入，不能改成ceil取固定个数。并列/小样本可使比例不恰好10/5/1%，如实保存支持。
- 阈值与membership一次确定、全部模型/选模口径共用；不访问全1706真值来重算阈值，不把本轮Top分数与旧完整1706 Top分数混表。fit-Q95/Q99条件依然沿原训练阈值，二者分别命名。
- 此项仅确定spec已经约定的本轮评价人口与沿用数学，不增指标或改变训练、主终点、数据和用户已批范围。

## r1.3：已验证yxz通道与外部对照执行位置（工程安排）

- 用户在本轮明确展示原样ssh yxz成功；主控随后用同一别名的交互SSH和普通ssh yxz pwd均成功，实际B源码对象经git cat-file确认可读，4张4090D当时各1MiB占用。当前yxz/lj别名均为原用户Tailscale配置，不写入新ProxyCommand、不重启连接服务、不删除复用socket。
- 允许在yxz执行完整数据准备、通用检查和两组NB的真实smoke/正式任务；对应唯一output-dir ./results/experiments/extreme_process_transfer_pilot_20261101_v1/，源码 ./。
- yxz已有 environments/real_extreme_v1/bin/python 已实际导入torch2.5.1+cu118、neuralforecast3.1.0，可优先直接复用，不为外部对照在lj重复安装一套环境。依然先核所有实际版本与输入SHA，再科学运行。
- NB_N/NB_X在manifest中明确host=yxz；M五组仍以lj同名run为主要目标，固定B链身份不变。不得在两个host重复启动相同配置。若M需要转yxz，必须先取得并核对同一套B权重，在运行前记录实际位置；不以其他实验M或重训代替。
- 多host只改变执行位置，不改变七配置、正式步数、抽样/选模、数据或评价口径。汇总按同一change ID与配置ID关联，并核物理GT/阈值/样本/NoisePlan身份，不混淆两台机器的路径或GPU。

## r1.4：就地计算，teacherbox主执行（工程安排，覆盖前述分host默认）

- teacherbox已在906a231的真实完整B上通过正式入口完成曝光准备，data/exposure身份已有回执；完整B向yxz的传输长时间仍不完整。为避免搬运80MB级原始准备包阻止科学运行，主控将七配置正式主执行目标改为teacherbox，就地读取同一完整B原文件，不重新生成数据或改变抽样。
- teacherbox唯一run/output-dir仍为 ./results/experiments/extreme_process_transfer_pilot_20261101_v1/；原已完成preparation保留原位。正式source为该项目worktrees/extreme-process-transfer-20261101-<最终短SHA>/，不得覆盖运行中的checkout；所有log/inputs/checks/smoke/runs/figures/tmp留本run。
- NB可先独立做真实smoke与验收：复用teacher兼容环境；NF3.1.0确缺时只在本任务项目环境补齐。无需等待M权重即可测试NB两组；M五组只从lj复制同一套已固定B权重及fit统计到teacher本run/inputs/b_chain/，逐原SHA核对，不换教师权重、不重训基座。
- teacherbox正式worker仅使用实际空闲GPU0，最多一个正式训练进程。七组设计、1200/3000步、seed2、损失、父模型、输入/评价/选模都不变。记录真实设备/环境；不得为显存换batch/K/截断梯度，若真实不适配则只将对应组转回原服务器并记录。
- paths/queue允许teacherbox执行已批准七组的显式子集。release必须绑定当前host、精确requested_groups、每个组的实际smoke、源码/data/exposure身份；requested_groups须是全七组的子集，不能仅凭NB release带跑M。队列恢复与跳过已完成组同样核全部身份。
- yxz/lj上的旧准备和传输产物保留原位，不因转主执行端清理。它们不得并行启动同一配置；此前没有任何正式训练启动，后续以teacherbox release配置ID为唯一排程。可以结束本任务已确认的慢传输客户端以省流量，但不得结束其他会话/作业，不删除部分文件。
- 曝光唯一小时的NumPy时区表示警告只限制该覆盖统计；原训练样本、阈值、角色和标签不变，不把此统计诊断当成训练前置。按已接受origin/valid-time语义记录转换和unknown；若发现真实样本/标签/角色错误则仅停依赖项。
- 本条为同数据/同权重/同研究设计的服务器安排修订，不增加配置、训练或数据范围；仍须真实smoke和独立运行审查后才正式launch。

