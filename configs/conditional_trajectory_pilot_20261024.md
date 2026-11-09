# RESEARCH EXECUTION SPEC：条件轨迹修正三组小验证

record_key: WP-20261024-CONDITIONAL-TRAJECTORY-PILOT-V1
revision: 1.1
Status: APPROVED_SCOPE / IMPLEMENTATION_AND_INDEPENDENT_RELEASE_PENDING

用户2026-10-24明确授权：实施推荐的条件轨迹修正，保留位置—尺度与解冻专家两个对照；先写清结构、预算、标准，独立复核后在teacherbox运行，不扩大数据/种子，不恢复G/PPO/v2。本规格在该授权内固定工程和先导参数；正式900更新须等待精确提交的独立research PASS和CEO release，不再另问训练许可。

## 问题、已有证据与可推翻假设

问题：同一M0起点下，直接修正输出情景的位置、非对称幅度及跨小时关系，能否改善真实高流量Top与中位数点曲线峰幅，并保留整体预测质量？

83c2acc正式R0/R1各1200及262/K64终评已完成。R1较R0的Top两档平均退0.3470%、6/24h峰幅平均退0.0752%，整体CRPS仅改善0.1559%；没有继续信号。新M0 producer=2ca8594、真实5000、SHA3fb45967…；旧3cf M及1706/K256报告只作背景，不混排数值。源码显示冻结μ/g不等于最终分布中心固定；专家/输出冻结的路由试验不能否定专家内容适配。

主假设：输出端的条件非线性修正能比简单位置尺度和专家适配更好地修复Top/峰幅。竞争解释：简单偏差足够；专家内容适配不足；峰时不确定或输入信息不足，增强输出表达仍无收益。有效负结果可结束本轮，不自动搜参或补种子。

## 起点、所有权与来源

- baseline commit：83c2accd6019e2c85fe102b117ba0f1a30499cd4；新分支codex/conditional-trajectory-pilot-20261024。
- 复用已结束任务的干净隔离树：本机 `.`；teacher `.`。只新增本轮package/config/spec，不改旧采样、adapter、评分、schema或旧运行原件；主仓用户staged/untracked不动。
- 仅 `ssh teacherbox`，项目根.。复用环境environments/pointwise_benchmark_v1/bin/python、CPU4线程、单GPU worker串行和既有GPU0锁。源码提交后才运行，run_versioned；无墙钟硬限。
- 资料机制：Radial Tail Repair，首发2026-10-12，预印本 https://arxiv.org/pdf/2610.16851 （径向修正不能保证条件CRPS或中位数点峰）；Neural Spline Flows，NeurIPS2019 https://proceedings.neurips.cc/paper_files/paper/2019/hash/7ac71d433f282034e088473244df8c02-Abstract.html （条件样条耦合的经典依据）。Flexible Tails(ICML2025，首发2024-06-22)只作输出尾部机制参考，不声称本轮实现该论文的TTF或具备渐近重尾保证。条件组织与水文适配均待验证，不预认领新颖性。

## 输入与数据边界

- 同一CN真实fit3388起报/8站/44来源单位；同一metadata-only开发子集262起报/9站/17来源单位、原global_index/IDs/顺序，不重选。来源/窗口不是独立洪水。
- 同原历史24h P/Q→当前24h未来流量情景及逐时超阈概率；不增加未来实测降雨、国家、站点、时段或test，不评分全1706。fit阈值、变换、层级权重及物理单位沿原件。
- M0权重：旧formal_2ca8594/runs/M0_REBUILT/artifacts/M0_REBUILT.pt，SHA256=3fb459672412f51b1c1a98583055d930aca6844807aceda51ecc7e7498bfa68d，producer=2ca8594485f9c3a57768dec8059491c68c44489f，实际5000。D500 SHA=bbd6f87e0ad452873f1d4b357a67cf2fbd823bb2dd3bdc36e56ab7b2d06c4a3a。核原producer queue/binding/run_record/completion和checkpoint/张量指纹，M0只读复用、不重训。
- M0/D500完整262 K32/K64参考来自旧83c formal runs/references/artifacts，核NPZ/rows/阈值/NoisePlan/SHA后复用，不用e7的32行smoke参考。基准仍DiffPTS，D500为保留参照，M0为本轮共同起点；L/C/E均内部对照或候选。
- 模型seed仅2。生成一次300×8加权fit曝光矩阵（seed2、同站→来源→起报抽样权重），记录可重复的2400次起报抽样、唯一ID数及SHA，三组共用。每个update固定B8×K8、完整20步的NoisePlan，seed沿2+10000000+update的派生规则，三个组共享实际噪声数组/SHA，不是增加重复种子。
- M0在这300个确切minibatch上的K8情景缓存只生成一次，共2400次起报抽样、0 optimizer；保存实际NoisePlan与原始有限值/公共非负投影计数，C/L读取该缓存，E每步使用同一噪声做可微采样。缓存producer、consumer、数据/曝光/噪声/权重/格式版本分别记录，不把旧产物改写成新提交生产。正式release绑定已复核的cache manifest SHA。

## 固定三组结构

共同预测对象z为Q/Q95。原M0情景按旧物理非负投影后输入修正；新输出在评分前沿相同投影，投影前先查NaN/Inf并记录负值，不以投影修复非有限值。所有条件来自历史x、冻结D500的μ和g；g是方差（原采样用sqrt(g)）。明确换算：r=q_scale/Q95，mu95=mu*r，g95=g*r^2，条件尺度特征为log(g95)，不是log(g)/Q95或log(g/Q95)。条件特征为历史24×2、mu95的24维、log(g95)的24维，共96维；只用fit估计标准化，记录统计量SHA；条件编码器96→64→64、SiLU。历史P/Q沿原数据各自fit变换，不把降雨量强除以流量Q95。不拼未来真值或Top/峰时标签。r1.1仅将r1.0的尺度意图写成明确方差单位公式，其余结构/目标/预算/判据不变。

### L：位置—尺度修正

冻结全部D500/M0；条件编码器+共享lead读出（context64+归一化lead1→32→2）生成a_h(c)、正尺度b_h(c)，z'=a_h+b_h*z。末层初始化使a=0、b=1；尺度用有明确初值偏置的softplus保证正值，位置/尺度不设置为了赢分而挑选的幅度硬上限。

### C：完整条件轨迹修正

同L的位置—尺度前层，再接两层交替偶/奇小时mask的条件有理二次样条耦合。每层conditioner读取context64与12个未变坐标，76→64→对应12小时样条参数；8个bins，区间[-8,8]（Q95单位），宽/高正且归一化，导数正。区间外连续连接可学习正斜率的线性尾，初始斜率1；上下尾可分别调整。两层后24小时均可变，后层读取前层结果，从而具备跨小时交互。

整体恒等初始化：L前层a0/b1；样条均匀宽高、导数1。最小bin宽/高/导数仅用于定义合法单调变换，工程实现须公开常数；不在结果出现后改变。该结构是有限范围内的非对称条件联合变换，不将线性外尾称为严格Pareto/TTF重尾模型，也不声称精确计算原M0隐式密度。

### E：解冻专家与输出层

从同一M0原件开始，不附加输出变换。只解冻ProcessAdapter的experts.*与output.*；routes/bridge/input/shared/time/norms及整个D500均不更新。完整20步输入导数保留，不能用no_grad包住冻结主干。记录实际可训练/冻结参数名单和数量。

## 共同目标、预算与验证

- 三组各一个配置、300次正式optimizer更新，共900；batch8、K8，AdamW lr0.0003、weight_decay0.0001、clip_grad_norm1，恒定学习率，无选最优/早停/追加。L/C共有参数同初值，额外层构造不消耗配对噪声序列。
- 三组共同下游目标：J=CRPS_U + 0.1 ES_U + 0.5 twCRPS_U(Q95)。均在Q95归一化情景上、全batch/全部提前量评分；ES除sqrt(H)。tw用v(z)=max(z-1,0)同时作用预测与真值，再算同U-statistic。训练U评分可出现负估计，不能因此误报数值错误；最终仍沿原V评分。
- 这是显式记录的新共同训练条件：输出修正不能使用epsilon MSE，故E也采用同一预测分布目标，不沿旧noise-MSE路线；不把新旧loss差异冒称结构增益。所有训练样本都参与，非只挑真实Top小时训练。原旧objective源码保持。
- 100/200/300做原262 K32过程评分，固定末步300唯一K64比较，不挑最好成绩。固定验证loss用相同J定义、原262顺序前32及K32预测的前8情景，明确该抽样方式；不同于独立生成K8计划，不冒称同噪声。L/C直接变换合法M0参考缓存；E按原B32/chunk32/20步、K32 seed600002、K64 seed900002产生相同计划新情景，核NoisePlan身份。
- 预算只按真实optimizer更新计；缓存/评价0更新，实耗单独记录。更新报错后保留实际returned/completed/uncertain与最近checkpoint，不重置预算。单配置未收敛标optimization_limited，不将它写成整个方法无效，也不自动延长。

## 判断标准

主指标：真实Q>fitQ95/Q99的条件CRPS/Q95均值；逐小时lower-median点曲线的6/24h窗口峰幅MAE/Q95均值。两档/两窗及每提前量、来源反例必须分别列，不改成median(情景峰)或P90点预测。保留原峰分布CRPS、Q95-twCRPS、峰时、起涨及整体CRPS/Brier；支持不足保留NOT_EVALUABLE。

候选C相对M0的继续信号：Top两档平均与峰幅两窗平均均下降，且整体CRPS及Q95/Q99 Brier均不高于M0+1e-8；这是探索信号，不是显著性或最终晋级证明。复杂结构的增量：与L、E各作同口径配对比较；若C在两类主汇总均不优于有效简单对照，优先简化或转E，不以仅整体平均变好认领Top修复。各分项/来源退步照实报告，不为过线挑来源。

若只改善尾部覆盖/情景峰而未改善Top与原点峰，则不认领主张；若只能用明显整体概率代价换Top，则按mixed/negative处理。结束后next_job NONE，无自动搜索、新种子或G/PPO/v2。

## 工程检查与独立放行

先commit，teacher真实入口smoke。建议每组连续4次真实更新，保存live S3，完整model/Adam/RNG严格恢复，再同S3续第4次（5/组，共15工程更新）；不复制不必要的独立prefix。明确工程与正式900分账。临时验证/终评触发在actual4，不伪造100/300；参考/评价采用原顺序首个完整B32，记录真实范围，不重算有效完整baseline缓存。

覆盖：共享曝光/NoisePlan/训练缓存生产与消费、L/C恒等与非对称/跨小时能力、未来标签交换不影响推断、前向/三项loss/反向/预期更新与冻结、E完整20步梯度、验证→Recorder→合法report_bundle→checkpoint→K64预测/评分→汇总→nextNONE，三组自动切换都真实执行。恒等RQS新数学FP32预声明atol1e-6/rtol1e-6；完整状态重载exact、续步沿原atol1e-7/rtol1e-5，不为通过改容差。零初始化导致早期隐藏层零梯度、无尾支持时tw梯度零均按机制解释，不错误要求每个参数每步非零。

复用training_report_v1 schema与实际离线TrainingRecorder，model/experiment身份及max_updates300正确，工程角色记录actual4/formal0；共享loss定义字符串与数学相同，metadata含实际forecast_hours=24/input_group=pq。只记录不启动sender，中途不做Drive/Science同步；图表按方案结束需求整理，不为启动美化。

科研工程师独占新增src/conditional_trajectory_v1、执行JSON与必要回执；CEO拥有本规格/判断，独立reviewer只读核不可变提交及真实smoke。已复核且输入/语义未变的缓存和模型证据可按producer/consumer与SHA明确引用，不伪造samecommit，也不无故重跑。独立PASS+CEO一次核查后直接后台有限队列；确认一次真实进展、版本/argv/cwd/脱离SSH、日志/监控入口后结束驻留，不等300/整批，不恢复唤醒。旧代码、失败、用户修改与原权重始终保留；不推GitHub、不发布、不写论文、不迁移/删除数据。
