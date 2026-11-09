# 洪峰失败诊断 v1（2026-10-07）

状态：APPROVED_DIAGNOSIS_ONLY。用户请求diagnose后再针对问题设计。本次只分析已允许的train/validation与已训练权重，没有新训练、优化器步骤、模型选择、网格、test或新数据。过去W/E结果已经看过，本次是探索性诊断，不追溯称预注册。

## 问题及竞争解释

1. 局部训练目标/上涨幅度覆盖不足，或者现有选中模型对训练池高响应也拟合不好。对9个中国流域核对训练目标最大Q、历史最大Q、正的未来最大涨幅分布；同时给所有48557训练窗的全局归一化范围，避免只看本流域便说模型从未见过。固定D/W/E各3seed best权重，在中国train/validation前向回放，比较同口径高流量/高涨幅误差。训练池不等于每条都被抽中过，也不是泛化评价；不据回放误差直接认定欠训练或继续加轮。
2. 起报前信息对未来强响应的提示不足，或压缩没有利用已有提示。读取9个旧6h事件各自历史12h P/Q与未来12h Q，核对历史P与上涨程度在该站训练输入分布中的位置。若角色文件有offline_generation_future_p，仅用于事后降雨时序诊断和图示，绝不作为模型输入/门控特征；未知保留NA。未来雨多/历史看起来平不能证明不可预测，更不证明未来P是唯一原因。
3. 两专家预测范围不够，或者当前gate/二分类标签与回归任务有错配。对已保存3seed中国E共4068条记录计算：真峰超过两专家任一预测峰的幅值下界；当前低估中还有多少在专家输出范围内；用未来真值求每窗口单一g∈[0,1]使12点无权MAE最小的精确oracle（候选为0/1及(y-a)/(h-a)落在区间内的折点），与当前gate和真实Q95类别硬路由比较。oracle只是诊断，不是可部署成绩或新模型，不改排名。

## 固定输入与输出

原D：./external-home/three_branch_bc94649/results/experiments/three_branch_formal_bc94649_20261006；W/E：./external-home/peak_specialization_983710c/results/experiments/peak_specialization_formal_983710c_20261006；执行源码基础983710c。

角色manifest与属性用原正式input_sources/run_record指向；仅原train/validation窗口文件，不读test或多角色cache，不读大canonical源CSV。中国train8557/validation1356，九站均保留。国际train仅流式取归一化目标/涨幅范围，不做新模型推断。中国P/Q/属性特征沿用原代码，流域/国家ID不进网络。

读取原best.pt，eval()+torch.inference_mode，无反向、无梯度或参数更新。D/W/E各seed2/17/42（总9个已有模型），仅中国train/val回放；不读last作选择。用已有primary_metrics的中国事件/站点指标对照回放的validation结果，允许明确记录的float32/CPU-GPU舍入，不重复已完成全量旧结果验收。保存紧凑NPZ预测、共同ID/真值/标签/来源和汇总，使父会话可复算。

高流量仍采用原权威high95_hours。新增“高涨幅”仅为诊断标签：每个站训练窗口max(futureQ)-Q_now的正值Q95，严格大于；无正值则NOT_EVALUABLE，不用validation定阈值。保留连续涨幅及分母，不把此切点当已批准的新训练标签。按原来源事件内平均→站点等权，同时保留站点和seed。

gate诊断复用既有report源JSON中的中国预测/sidecar，不再为此重新扫全部预测。记录underestimation=max(true_peak-mixture_peak,0)、head_envelope_deficit=max(true_peak-max(regular_peak,high_peak),0)，后者是给定两专家的低估下界。二者差只表示理论可减少的低估，不保证时刻/全曲线正确。scalar-MAE oracle用整条曲线优化，不声称同时最优峰值或峰时。

降雨案例固定沿用上轮三例Liqingdian洪水6、Zhuxipo洪水32/35，另输出全部9个旧6h事件的小表，避免只挑故事。历史输入可见区间与事后未来降雨严格标色。总P/峰前未来P及位置只作描述。

## 执行和交付

主会话维护本计划/研究判断；three_branch_impl只新增src/diagnose_peak_failure_v1.py和必要的同前缀绘图文件，不改原模型/评分/数据文件。先提交；teacherbox新checkout、新run_versioned目录，原正式run只读。CPU单线程优先，可用空闲GPU但不停止别人任务。允许Torch收集与既有Matplotlib环境渲染分开，全部绘图和导出在服务器。

少量图优先2页：训练/验证范围与固定权重训练池回放；三个案例的历史/未来P-Q和可用信息界线（未来雨只作事后解释）。如门控诊断比其中一图更有信息可替换，不增加美化循环。合并PDF在同一Preview窗口，脚本与源表保留。父会话核对一组手算oracle、计数/尺度、val回放和若干原案例后给研究解释，无新训练或自动追加探索。

本次完成标准：给出观测支持哪些解释、不能区分哪些解释，以及推荐下一项最小设计；不是保证找到唯一原因。SSH失败有界停下，不退回本地画图，不自动重启旧任务。尚无新训练授权。

流程参考：使用hypothesis-generation组织观察、竞争解释和可区分诊断。Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065 （官方arXiv元数据已于2026-10-07核验；该引用是工作流程来源，不是水文结论证据）。
