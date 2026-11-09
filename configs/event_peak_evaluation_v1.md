# RESEARCH EXECUTION SPEC: 中国事件记录峰值的预报起点评价

Status: APPROVED — 用户2026-10-01同意明确洪峰预报起点、覆盖范围并依据误差再决定预测目标/网络。实施选择写于执行前。

## 问题与范围
当前模型对中国验证事件记录中的主峰，提前1/3/6/12小时预测有多大峰值和峰时误差？12小时预报范围和历史不足限制了多少事件？基础提交 afd90edddbeec4a3b4faa113440f0cd5cdd01864。不以特征提取作为创新，不新增训练、不改网络、不重新采样、不得读test时序或预测。

## 输入及峰值资格
- 仅已有9目标 validation.jsonl.gz、training_transforms.json（只读），事件目录china_event_catalog_v1.csv及既有目标划分metadata；禁止打开train/test时序或全站标准化文件。
- 用validation窗口的x_q+y_q、source_rows及origin_source_time重建各作者事件的已观测Q记录，按source_row/时间去重，重复值一致性校验；反归一化只用既有训练尺度。把窗口记录并集的行数、起止时间、间隔与事件目录逐项核对。目录显示本轮21个验证事件均无缺口/冲突，但实际并集仍须验证。不能把重叠窗口当独立洪水。
- 并集覆盖不全、目录有缺测/冲突、记录峰在首/末边界，保留明确原因，不纳入主峰误差汇总。峰值并列时以首次最大值为参考，另报并列小时数、是否多个分离峰段；全平/边界峰不得宣称完整主峰可评价。
- 完整覆盖作者事件记录不证明物理上完整洪水起止。报告统一使用“事件记录主峰”，physical_event_completeness=unknown；不得默认为21场独立且完整的自然洪水。

## 两种固定起点协议
A. retrospective_peak_anchored：以每事件首次实测主峰时刻t_peak为离线评分参照，分别查找origin=t_peak-h（h=1/3/6/12）的已保存12h预测。这是事后定位峰时的诊断，不是预报时知道未来峰时。每事件每h最多一个预测起点。预测器没有接收未来信息；选评价起点使用了事后真值，必须披露。
B. earliest_available_origin：不看峰值，选择该事件validation中最早的origin，即现有事件记录内首次具备12h输入且有完整12h目标的时刻。报告事件记录峰值是否落在(origin, origin+12h]；范围外只报未覆盖，不拿短期预测错比未来范围外洪峰。原作者事件起止仍是事后选取，因此它是事件条件下固定起点评价，不称全年自动预警。
两协议分别汇总，不混在同一排行榜。对A缺少origin，区分历史不足、后续12h记录不足与其他缺失；B峰已发生、超过范围、边界/数据不全分别记录。没有origin时不得拼凑多个起点的预测或补零/插值。

## 对照与评分
- 复用保存的validation预测：原真实baseline、alignment_b33e953的none/unconditional/high95，另用持续性预测（未来Q等于origin最新观测值）。不读取任何test文件或权重，不运行模型。
- 所有方法同事件、同origin、同12h范围；核对模型预测记录event_uid/origin/source_rows和observed_q是否与准备数据相符（float32回转误差允许明确小容差，例如rtol1e-6/atol1e-3 m3/s；超过即失败，不能扩大容差救结果）。不得按各模型好坏选择事件或起点。
- 预测主峰取该次12h输出最大值及首次出现时刻，与落在范围内的事件记录实测主峰比较；主要报告峰值绝对误差m3/s、绝对百分比误差和有符号百分比偏差（实测峰<=0则比例NOT_EVALUABLE）、峰时绝对及有符号小时误差。单独报告预测最大值在输出边界、预测并列峰（持续性全部并列）的计数，不能掩盖峰时信息不足。
- 先逐事件列出，再在各目标内等事件权重平均；跨目标比较用目标中位数（百分比峰值误差、小时峰时误差），各h报告可评事件/目标数，不直接平均不同尺度m3/s误差。先全事件，训练Q95以上记录峰值作辅助子集；无子集保留NOT_EVALUABLE。
- 保留逐事件成对差异，比较none vs baseline、none vs unconditional、high95 vs none，不凭总体均值选方法。小样本、单种子、验证集多轮使用，不做窗口级显著性或最终效果声明。
- 不为改善评价删除难事件。任何资格排除都保留在清单与分母中。数据/范围不足为有效结果。

## 实现、执行和验收
- 主会话负责RESEARCH/CONTEXT和本规格；一个worker负责src/evaluate_event_peaks_v1.py和src/check_event_peaks_v1.py，无额外框架或依赖，优先标准库；按规格先写手算用例。
- 定向检查覆盖：完整并集去重、缺口/不全、峰在首尾、并列峰、固定起点不依赖未来峰值、t_peak-h边界、完整12h目标资格、峰值/峰时手算和目标层汇总。不得调用真实test输入。
- 先提交代码再用run_versioned执行检查/评价。teacherbox新detached worktree，输出results/experiments/event_peak_evaluation_v1，CPU最多600秒，不用GPU，不覆盖已有目录。SSH连接失败不反复尝试，报告未核实。
- 原预测路径：./results/experiments/china_first_full_v1/artifacts/<target>/baseline/validation_predictions.jsonl.gz；三组在./results/experiments/alignment_comparison_v1/artifacts/<target>/<arm>/validation_predictions.jsonl.gz。数据在./data/processed/china_first_flow_v1。
- 小型JSON/CSV摘要、逐事件峰值评分/资格清单可回本地，小时序列与完整预测留服务器。主会话独立复算关键起点/排除原因及评分一次；已验证的训练不重复运行。
- 输出为资格清单、两协议每h/目标/模型指标、逐事件配对值、来源和状态记录。最终决定：数据不支持则明确所缺前置/范围；能评价且明显低估或峰时偏差则据此提出下一轮针对性预测目标方案。不自动启动新训练、改默认模型、下载、联系机构、删除资料或发布。
