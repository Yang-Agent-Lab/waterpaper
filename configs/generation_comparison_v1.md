# RESEARCH EXECUTION SPEC: 五组生成策略比较 v1
Status: APPROVED
批准依据：用户2026-10-01明确“做好了就开始训练验证”，承接configs/architecture_candidates_v1_20261001.md。首阶段种子2、9中国目标、单空闲GPU、正式训练/采样/评价总walltime上限3600秒；不自动追加种子或预算。

## 问题
历史条件生成是否比原联合生成、规范条件补全及真实极端重采样，更能改善真实validation洪峰预测？负结果或生成不足均是有效结果；不宣称条件扩散为新发明、不以运行完成证明创新。

## 起点与范围
本地基线880bd2c；实施使用包含本规格的独立worktree，服务器用对应不可变提交detached worktree。父会话负责科学验收，工作者负责实现与执行。主参考骨干保留，不做S局部门控/国际迁移/新数据/新loss创新，不修改作者副本。
允许train和validation；禁止test时序/权重选型，温度/未来实测P禁止进入预测器。数据固定teacherbox:./data/processed/china_first_flow_v1，各目标独立；使用既定训练变换、Q95/Q99和事件划分，不重新划分。validation全部窗口按起点滚动预测；事后峰前切片是评价而非输入。

## 五组与比较公平性
R0仅真实普通训练；R1真实高Q95事件重采样；R2新训练的原联合扩散；R3使用同一R2权重、VE相容已知前段加噪注入的条件补全；R4显式历史条件扩散。R2与R4从头训练使用同一训练窗口、批次计划、100epoch与lr0.001，不拿历史权重少训的R2对比新R4。末轮固定生成权重并保存逐轮loss，不以随机训练loss最低轮作效果证据。模型hidden64，batch64；R4保留原卷积骨干，在时间embedding中加入历史P/Q编码，score与分类器同时条件化，只噪声未来/只未来score损失与引导更新。

R3是在VE边际noise level sigma_t下把H+sigma_t*z注入已知前段，未知位置作原predictor-corrector更新，下一noise level重新处理已知部分，最终前段严格等于H。限定为VE条件补全对照，不冒充完整RePaint重采样日程。R4分类器明确训练在未加噪历史+加噪未来，不调用不兼容旧类别接口。

各目标desired=len(train)，attempt上限5*desired，sampling_steps100、guidance_scale1、corrector_snr0.2、corrector_steps1、sigma_min0.01、sigma_max50，source归属及失败日志保留。R3/R4共同使用固定、事件平衡的train高Q95历史尝试序列，记录attempt_id和anchor行；优先用共同接受attempt匹配历史，再取N=min(desired,R2接受数,共同接受数)，保留所有尝试/实际使用索引。R1的额外真实样本取相同历史锚点对应真实未来，记录接受选择后的事件分布，不再声称最终支持严格事件均衡。R2没有相同历史锚点，不宣称五组完全配对生成。N=0时该目标生成比较NOT_EVALUABLE，不用真实填补；R0/R1仍可输出清楚命名的真实参照，但不得与缺失增强组成完整五组结论。N<desired保留不足状态并报告数量匹配。

预测网络全部使用相同Separation结构及真实推断路径、同种子初始化；可共同从相同真实LSTM预训练初始化共享编码器。mmd_weight=0，其他预测/重构/正交保持现有值1/0.1/0.5，默认配置不修改。为排除额外分支/曝光差异，五组均采用相同base真实批次和aux批次结构：R0的aux来自普通真实train，R1来自上述真实极端池，R2–R4来自对应合成池；代码的synthetic分支在本实验中表示aux来源，真实推断分支不变，需在报告中解释R0只包含真实数据。每组100epoch、相同optimizer更新数和batch64，不早停（保留每轮validation并按同一真实validation MAE选择checkpoint）。所有组记录base/aux总曝光、实际步数、参数量、训练/采样时间。不能从R0对比单独归因生成，R4对R1/R3为关键增量。
推断仅输入x历史P/Q；R3/R4未来P仅离线训练目标。真实锚点的非负性以已验证train来源为准，明确浮点回转边界；生成部分严格有限/非负/未来超Q95，不偷偷投影裁剪。报告各组接受率与共同支持筛选，不把选择更容易样本当机制成功。

## 评价和结果
保存每窗口origin/event_uid/真值/预测，使用既有验证事件重建与峰评价逻辑，不另选有利事件。主结果为同事件同提前量1/3/6/12h峰值APE和峰时绝对误差，逐事件→目标汇总及成对R4vsR1/R3；同时报告全部滚动窗口max误差、小时MAE/RMSE、高Q95/Q99、边界峰/并列峰/范围不足。持续性可作参照，但恒定输出峰时不作有意义定位。
补充实际训练子集的future粗糙度/high95持续时长、Q_origin=q[11]、future_peak-origin与future范围、未来最近邻/复制；历史直接复制不算生成质量提升。不做因果或独立事件的虚假推断。原test已曾评估，开发暴露历史保留，本轮不再读取。
单种子结果只作方向筛查；关键对照幅值和峰时取舍不清为inconclusive。报告正/负/失败全部流域，不因候选差而扫参数。

## 执行与输出
允许修改独立worktree中的src/esdf_flood/conditional_diffusion.py、新src/run_generation_comparison_v1.py、新src/check_generation_comparison_v1.py、configs/generation_comparison_v1.json及本规格必要具体化；仅确需复用调整时修改diffusion.py的共享前向helper，旧forward行为须保持可测一致。不修改既有评价公式/原始数据/作者代码。输出忽略的results/experiments/generation_comparison_v1_<时间提交>/；小型报告/验收/启动记录在data/catalog/generation_comparison_v1_*。父会话更新RESEARCH/CONTEXT。
代码配置先commit，再用run_versioned正式运行检查；read-only preflight不需commit。前置检查含条件mask/score梯度、known前段保存、正确noise-level输入、分类引导0vs非0、未来信息不进predictor、五组初始化/曝光/步数一致、固定例子指标及生成失败路径。工作者交检查通过的精确commit给父会话，父会话针对性代码复核后发执行指令；无需再次请用户批准既定范围。
teacherbox既有独立.venv，读取空闲GPU后用一张；OMP/BLAS线程1；不影响其他进程。SSH失败一次有界复查，不循环重连/自动换数据；保留真实partial状态。
正式任务后台启动并保存PID、run_record、stage进度、完成标记与日志；应有完成唤醒或本地等待包装的事件回传，不空转轮询。超过3600秒停止新增任务并保存partial，未完成不伪称complete。不删除worktree/历史、安装新服务、推送远端公共仓库或发布。

## 验收
独立验收由未参与实现的父会话或只读审查者完成，含不可变版本、输入角色、生成/训练公平性、主结果从保存预测定向复算。PASS只说明执行可信；效果另记positive/negative/inconclusive/NOT_EVALUABLE。失败修复限定实现缺陷、保留失败记录；不自动重启完整一小时预算。首阶段结束后汇报，不自动新增模块或多种子。

