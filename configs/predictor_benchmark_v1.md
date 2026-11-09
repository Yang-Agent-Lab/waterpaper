# RESEARCH EXECUTION SPEC: 三预测器公平比较与统一训练报告

Status: APPROVED_EXPLORATORY，2026-10-14。用户明确回复“可以，按照你说的做”，批准当前预测器、端到端概率LSTM、时序卷积三种对照，内置模块诊断与固定四页报告；沿既有单seed三组规模实施，不增加组/种子/下载。编号WP-20261014-PREDICTOR-BENCHMARK-V1。

## 问题、比较与解释

当前固定P0+adapter编码组合是否比直接端到端历史编码更适合洪水概率预测？当前多次改进共用P0和输出结构，尚不能判断编码器组合的优劣。三个模型共用已有S2的四段峰概率解码/分布族/NLL+CRPS，隔离历史编码路径；本轮是预测器组合的实用比较，不认领单个模块因果贡献。若两挑战模型均无收益，不能推断所有模型无效：共同的输出假设/输入/学习目标仍是竞争解释。

三组：current_s2 = 原固定P0 LSTM64 + TemporalAdapter；direct_lstm = 一层LSTM(2,64)末时刻状态，从头端到端训练，不含P0/adapter；direct_tcn = 因果残差一维卷积，通道64、kernel3、dilation1/2/4、末时刻状态，从头训练。三组都输出64维条件表示，共用原GRUCell(68,64)+Linear(64,2) S2概率头，不把更换概率头混入本轮。保持四段(0,1]/(1,3]/(3,6]/(6,12]与累计窗口峰语义，不能画成逐小时流量轨迹。共同解码器/头用独立共同种子初始化并核hash；编码器结构不同，不宣称参数量完全相同。P0已有预训练开销单独注明；三组本轮曝光可比，不宣称历史总算力相等。

## 数据与角色

起点357be9da7bbc795dd1ae03cecf8c20f3ecf36826。独立工作树.。原始和旧结果只读。服务器训练首选yxz，已核25个固定输入位于._inputs/data_org_v1；Python./.venv/bin/python。teacherbox可用于既有元数据/CPU工作，不在Mac运行分析/测试/渲染。

prepared producer ab720ba，manifest SHA256 e482500d17c6cd900b985c67ff3c6a274b43a892d5f779316032cdf3bfb0ecb6，P0 SHA256 0908370cc4f1ed487495e3f49fa3bd881e5a1cf3b9156c45eff141a6b6896d58。沿旧source manifest/path_map，不改原数组、尺度、Q95/Q99、事件划分。输入只有历史12h P/Q；f接口保持但三组不得新增额外信息；标签未来12h Q只作监督。

fit共28814：中国4554/德国12199/冰岛12061。现有select中国2150/德国4528/冰岛3896；calibration中国1853/德国3273/冰岛4043；development只有中国1356。新loader显式支持国家选择，不再无声过滤外源select。训练/选模阶段只读fit/select；三组选择全部停止后才按calibration→阈值保存→development顺序执行。外国calibration本轮不拟合报警线；外国select报告原概率/误差/区间，事件发现与提前时间无对应资格时NOT_EVALUABLE。历史ESDF使用过更宽来源范围，不能称外国select整个课题从未接触；本轮称同站开发留出。

禁止test观测、未来P、美国/日本训练或评分、新下载、生成增强、重新切分、额外模型/种子/超参搜索、恢复定时或采集器。中德冰fit参与共同学习；各国分层输出不能把fit指标冒充留出验证。

## 训练预算与选模

seed2，三组各100轮，每轮28928窗/226更新/batch128，合计67800更新；不按结果加训或更换种子。沿共同country→station→window均匀采样A，用src/multiwindow_peak_v1/data.py slots构造一次并三组完全复用；无需重跑旧B/C采样。Adam0.001，clip10，训练16情景、select128、final512。数值错误立即失败并保留，不以裁剪样本/改分布救结果；记录实际时间，不抢占他人GPU。

本轮所有模型重新训练，checkpoint统一按中国select H6窗口峰CRPS/fitQ95（先事件后站等权，越低越好），同分取早。它衡量整个预测分布，与旧E6不同；此次共同变更已提前记录，不把新current_s2与旧ABC绝对差认领为编码器收益。外国select作补充开发评价，不参与代码中的checkpoint选择。训练曲线保存每轮train total/NLL/CRPS，以及同定义评估的fit_probe/select total/NLL/CRPS；fit_probe为事前固定的三国小组，明确不是全fit。评估样本数/抽样数/归一化保存在元数据，避免不同定义曲线混称validation loss。

主研究判断：中国development H6合并Q99 Brier与事件发现、实际提前量、连续误报及普通代价并列；沿EVENT-RISK/ALLSTATE-Q99的既有资格/校准算法，中国cal阴性按5%加权FPR定线，2/10%仅既有敏感性。Q95及1/3/12h、CRPS、区间、原E6为补充。只有概率改善且关键事件/误报没有明显冲突才列为优先候选；冲突报mixed。单seed不声称稳定优势/显著性/最终选定架构；当前模型若胜出也须保留P0既有预训练优势的解释。

## 训练内诊断

每50更新保存可训练模块梯度范数、全局裁剪前范数/裁剪状态、可获取激活的方差/非有限比例及损失分项；冻结模块梯度为NOT_APPLICABLE，缺记录为missing，不填0。不存在跨架构统一健康分数，报告状态与原始趋势。

每10轮及最终选中权重在固定fit/select小组做无梯度临时干预，隔离RNG/权重/优化器/模块模式；缓存键含代码/权重/数据角色/样本/算法身份。小组每国最多256窗，固定hash/轮转方式覆盖station/block，每站最多2块、每块最多8窗，IDs在训练前保存；外国source_event_id为空时使用event时间块身份，不称独立洪水。本上限只用于fit_probe及干预，不减少完整select评价或训练曝光。三模型共同input_rain_zero/input_flow_zero用于输入敏感性；可中和的模型模块另列adapter_zero/encoder_zero。decoder/head若无有意义替代标NOT_EVALUABLE，不硬凑贡献百分比。贡献差值=临时干预后指标−完整模型指标，保存H6 CRPS/Brier与峰幅MAE；正值表示关闭后变差，不冒称去模块重训的独立收益。不同干预语义不可混排。每次训练自动汇总，不结束后另开全量诊断。

## 固定报告接口与四页

唯一入口python -m training_report_v1 --bundle <report_bundle.json> --out <new_dir>。版本schema_version=1。核心JSON顶层：run_id,models(list of {id,label}),epochs,cases,risk,health,interventions,metadata。每行带model/role/country；不可用数值为null且有state/reason。新增字段向后兼容，旧source适配器独立。

epochs行：epoch,step,train_loss,train_nll,train_crps,fit_loss,fit_nll,fit_crps,select_loss,select_nll,select_crps,selected；若为分国行country明确，不和汇总重复。cases行：case_id,station,event_id,country,role,model,x_kind,x,truth,median,lower,upper；当前S2只允许origin_time_h6_peak（横轴起报时间、纵轴未来6h窗口最高流量）或horizon_window_peak（横轴1/3/6/12h），不得伪造逐时预测。risk行：model,country,role,horizon,target,brier,crps,detected,events,false_alarm_segments,lead_hours,reliability_bins(list of {mean_probability,observed_fraction,n})。health行：model,module,epoch,step,gradient_norm,output_variance,nonfinite_fraction,state。interventions行：model,module,kind,role,country,metric,horizon,delta,n,checkpoint,state。完整版本、来源、定义和限制放metadata/附件。

固定输出01_learning.png、02_predictions.png、03_risk.png、04_modules.png和同四页PDF；颜色/页序/坐标语义固定、中文短标题，不堆技术图注。第一页必须直接看见loss及最佳轮，缺失曲线写未记录；第二页依次固定China development李青店困难、栾川困难、泸溪普通、朱溪坡困难，metadata.preferred_cases可配置，缺资格在原位置标NE；旧ABC适配器可用现成预测和事件表，新比较沿同案例身份，不能按模型各选好例；第三页概率可靠性、检出/误报/提前量与国别分层；第四页模块健康和临时干预变化。必要的角色、单位、样本量保留，长日志/版本不塞进图面。

## 验证、运行与接受

报告取例最终规则（正式训练前确定）：本项目新旧结果第二页统一horizon_window_peak，即同一起报点的1/3/6/12h窗口峰，避免旧图用时间轴、新图又换横轴。上述三个困难站选development实测H6窗口峰最大起报点、并列取时间最早；泸溪普通选时间最早且truthH6<=fitQ95的起报点。各模型共用window_id，标题站名/类型及简短起报时间；完整ID留附件。取例仅依赖已有观测与身份，不随预测优劣变化。四窗口truth可沿保存point_windows/原prepared角色连接，不新推断。

两个执行者并行但文件分开：benchmark_impl仅src/predictor_benchmark_v1/及对应tests与execution；report_impl仅src/training_report_v1/及对应tests、report_demo_receipt。父拥有spec/研究状态/Science/独立接受。测试和运行先提交后在服务器run_versioned；先边界测试，再CPU真实小试三模型各2轮/每轮2更新，fit/select替身完成保存→选择→评分→四页报告，不能只测函数；必须验证续跑不会重复已完成组与风险角色访问顺序。

绘图脚本先用旧ABC保存输出生成完整四页演示，缺失数据明确显示；禁止为了补旧图重新推断/训练。必要health原日志只从既有服务器读取。父检查真实产物、数值及图，再对新三模型CPU/GPU小试独立接受。正式前重新核查单GPU空闲/唯一进程与锁、输入hash/代码commit；使用空闲yxz或lj既有资源，若均忙仅保持ready，不动他人任务。只有通过真实小试后唯一正式启动，禁止重复启动/并发写同run；无自动重试改变科学条件。

长任务持久后台保存progress/completion/failure与run_record；不恢复abc/q99，不建立循环监听服务。可用既有一次性完成回执机制时沿用，不能声称不存在的自动通知。最终父从保存结果核主要分数、模块身份和选模、国内外角色范围，执行可信与科学收益分开。Science同编号记录，飞书仍暂缓；不发布/推送/删除。
