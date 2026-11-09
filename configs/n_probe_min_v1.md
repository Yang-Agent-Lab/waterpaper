# RESEARCH EXECUTION SPEC: 六小时上涨信息定位，最小探针批次

Status: APPROVED — 2026-10-09 用户明确批准委派写代码、独立验收并部署服务器。
Change ID: WP-20261009-N-PROBE-MIN-V1
Revision: 1.1

操作性修订：首个smoke中CPU与原GPU前向最大差1.90735e-6，身份/真值/尺度/Qnow一致且尚未拟合probe。允许常规后端浮点误差，采用以下allclose及主指标双检查；不为通过检查更换backend、改变输出或覆盖原预测。此修订不改变科学条件、拟合预算和评价目标，原失败记录保留。

## 问题与唯一范围

同一六小时上涨目标，原始历史＋固定分时段特征，与N既有内部特征，哪一种更能识别上涨、读出涨幅？首选解释为N的表示/输出未充分利用已有上涨线索；竞争解释为可用信息/训练事件覆盖不足。一次单权重探索，不把探针失败当信息不存在，不把探针收益当完整预测模型收益。

本批只做两个信息来源×两个读出任务，共四个拟合模型，不加其他种子、MLP、主模型更新、未来雨输入、生成器或GPU。

## 源码、输入与隔离

- 实现工作树：`.`，起点`f1cef7dfe80c0aa7c5b543f33cd6f7d2f90cc134`。沿用独立工作树，不修改主仓未提交的AGENTS或其他文件。
- 项目最新用户指令以主仓AGENTS和本规格为准；旧停止规则已由本次批准在此有界探针范围内替代，不恢复Multica旧小队或任务。
- N原执行版本`28ea910557cea4b7f9ca16e6168fad9e9a5b6288`；唯一权重teacherbox `./external-home/vega_material_formal_candidate_20261007/results/experiments/material_route_formal_28ea910_20261007/artifacts/seed_2/N/best.pt`，selection原epoch8。只读、map_location=cpu、eval/no_grad；任何N参数不得更新。
- 原dev预测同目录`validation_predictions.jsonl.gz`；CPU回放按window_id对齐全部中国1356窗，真值/尺度/起报Q一致；预测np.allclose(atol=3e-6,rtol=1e-6)，且重算原12h窗口峰主指标差≤3e-6才进入拟合。保存最大绝对差、差异分布及对应窗口，不舍入掩盖差异。
- 数据版本`e50e3cfb1413338f876ba3475771a2116c25fe83`；manifest `./external-home/pooled_esdf_d781a3f_v2/results/experiments/esdf_projection_full_d781a3f_20261005/artifacts/model_manifest.json`，只读取China的train/validation文件。8557train/1356dev，保留全部16开发事件与9站。
- 禁读test、混合preparation cache、外国数值、任何未来P。来源时间/事件元数据可读。两近邻train案例只核对源行、源事件、被排前缀相对位置和6h/12h时段；不扩展为全数据审计，不因来源语义unknown自称真实水文不可辨识。

## 固定数据划分与目标

1. 原train按temporal_overlap_component分组；缺组件时不可悄悄改成窗口随机拆分，报NOT_EVALUABLE。GroupShuffleSplit(n_splits=1,test_size=0.2,random_state=20261009)，约80%探针拟合、20%报警阈值校准。所有站共享组件划分，同事件/源行/24h区间不得跨两侧；保存每窗split。
2. 不重分开发集。N曾用全部原train训练且dev选best，因此本批是开发诊断，校准子集不是encoder从未见过的数据，不能宣称独立盲测。
3. qscale沿原train Q95；连续目标r6=max(0,max(y_q[:6])-x_q[-1])，完全在归一化Q尺度。另保存真实第6小时Q、窗口峰和起报Q，不能混同这些目标。
4. 强上涨标签：每站仅在探针fit部分的正r6上计算90%分位cutoff，label=(r6>cutoff)。无正r6站的分类为NOT_EVALUABLE，不借用calibration/dev标签或换阈值；该站回归仍保留。保存每站阈值/正负窗口数/正负来源事件数。未来高95标签只作描述分层，不另拟合分类器。

## 两个信息来源与四个拟合模型

A: 原始历史12×2展平24维＋18维分时段特征=42维。分段相对起报为[-2,0]、[-5,-3]、[-11,-6]。每段6项：P之和、P最大值、P>0格数、Q均值、Q末减首、段内逐小时正增量最大值（max(0,max(diff Q)))。原P/Q已经使用原训练尺度归一化，原始序列始终保留。

B: N real分支共享64维与real专属64维拼接后的128维，即real_predictor输入，非最后64维。只读提取，不把shared/specific加法重构向量当预测表示。

- 每种信息各拟合一个LogisticRegression(C=1,penalty=l2,solver=lbfgs,max_iter=2000,tol=1e-6)强上涨分类器；无class_weight额外平衡，无超参搜索。
- 每种信息各拟合Ridge(alpha=1)预测连续r6；无目标log变换。保存raw输出及max(0,raw)；主回归指标使用截到0的输出，两种读出器同样处理。
- 特征均值/标准差仅用fit部分，零方差scale=1。样本权重按窗口→事件→站点等权，fit权重归一化至均值1；特征标准化也使用fit权重。保持相同划分和拟合预算。
- 如果分类fit缺任一类别，相关分类不可评价；不能把概率常数伪装为成功分类器。收敛警告显式记录，不增加迭代/换模型救结果；regression可继续。

## 参照、评价与判断

- 参照为原N seed2输出派生rhat6=max(0,max(pred[:6])-Qnow)，以及持续性rhat6=0。N连续rhat6也用作风险排序分数；分类另报告fit正例比例常数参照和当前Q/近3h涨幅简单分数，不拟合新模型。
- 在calibration负例上选择使事件/站点加权FPR≤5%的最低报警阈值，严格score>threshold；并列采用保守阈值，保存实际FPR。校准不足时阈值指标NOT_EVALUABLE，AP仍按可评价条件报告。
- 主回归评价：dev r6 MAE，窗口→事件→站点等权；同时报真实r6>0和强上涨子集、物理m³/s偏差、平方误差、全部事件/站点、最困难Liqingdian/Luanchuan。无rise子集报NE不填0。
- 识别：明确使用sklearn average_precision_score及事件/站点样本权重，报告正例率、AP、实际dev FPR/召回/精确率；起报Q≤1子集单列。AP/召回提升不等于峰幅预测提升，不新增任意晋级线。
- 两张服务器图足够：全部事件成对识别/涨幅误差；困难事件按起报时间的真实/原N/两探针涨幅与风险（标训练确定报警线）。展示成功和失败，不能只挑有利案例。
- 若A的识别和幅度均较B好，优先提出历史直通的最小干预；B能读幅度而N明显偏低，优先预测头/目标；训练好开发差提示跨事件问题；都弱时保留信息/覆盖/探针容量未决。本批没有第二轮自动模型选择或新架构训练。

## 执行、验收和预算

- 服务器teacherbox，优先`./.venv/bin/python`，需要sklearn时允许安装固定版本到本批隔离依赖目录/环境；不破坏旧运行环境，记录依赖版本。
- 单CPU线程，CUDA_VISIBLE_DEVICES为空，内存≤4GiB，所有本批smoke/正式计算累计墙钟≤3600秒，GPU0。实现编写时间不冒充计算时间。超预算保存partial，不自动重置。
- 实现/配置先commit，所有新验证和正式运行经`src/run_versioned.py`。新输出目录不得覆盖。先一次真实输入的小规模smoke，覆盖feature读取、组划分、四种拟合路径、全量dev回放一致性、保存预测/评分和出口；smoke结果不能选超参。
- 独立代码与smoke验收通过后，由主会话唯一启动正式全量CPU任务。保存progress/completion/exit，预期短任务无需监控；若实测超过10分钟，按本次有界任务建立一次性完成接续，不恢复旧小队或周期训练监控。
- 输出manifest（输入文件/weight及code版本）、split/label inventory、feature定义与标准化参数、四个拟合模型、逐窗原始预测和分数、事件/站点汇总、图/来源表、运行记录、资源与科学限制。原权重内容前后保持不变。
- 独立验收者未参与实现，复算主指标/一个完整困难事件与分类阈值，确认边界与权重不变。PASS只证明检查范围内实现可信，不代表方法改进。
- 外部效果：按变更编号写入Science research_experiments与已存在的飞书Base，写前查重/核schema、写后回读。Base仅简明中文，不写路径/提交/命令。无法定位一端时标partial，不编造同步成功，不因记录端缺口无限延误有界实验。
- 禁止：主模型/扩散训练、GPU、新seed/超参网格/test、修改原始数据/旧权重、发布、删除、GitHub推送或恢复旧Multica任务。
