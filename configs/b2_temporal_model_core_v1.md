# B2模型侧实现选择（APPROVED r1.0的工程记录）

仅实现五组状态方差侧支路，不改变训练/数据/评分设计。

- stats_b2：19→32→32→24，2488个活跃参数。
- temporal_summary：3→14→14的两层卷积、14→32→24，2490个活跃参数。
- temporal_lead：相同卷积，提前量1→14→14查询、14→14 无bias key/value、14→32→1共享读出，2375个活跃参数。
- fixed_segments / dynamic_segments：上述结构增加14→14变化query及2→14历史中心/持续长度投影，2627个活跃参数；两组参数结构完全相同。
- 两层卷积kernel5、dilation1/5、padding2/10，感受范围25点；相对历史位置和提前量均为确定坐标。P/Q采用signed log1p，不拟合缩放。
- 两种片段组共同使用相邻key/query投影余弦变化（第一个位置为0），变化归一化加权汇总；变化总和≤1e-8时均匀平均。固定组严格六个连续4h片段；动态组变化降序选最多6个位置、同分较早优先，抑制距离≤2的位置，就近分组同距选较早位置。原始P/Q各自时间不变（容差1e-8）或没有正变化候选时只用中心index11的一个片段。实际片段数和成员的真实中心/时长逐批保存在state_branch.last_segment_counts / last_segment_coordinates，供runner记录。
- 片段保留真实历史坐标和持续长度，补齐位置用attention mask；全部24个逐小时记忆始终参与读出。
- 部分机制适配来源：https://arxiv.org/html/2605.14014v2 第3.2/3.3节；未实现小波和重构目标。
- 原DiffPTS参数冻结且mean/variance/diffusion在model.train()后仍为eval；去噪前向不使用no_grad，以保留方差侧支路梯度。只新增state_branch参数可训练。
- 输出头严格零初始化。初始编码器梯度为0是零输出头的数学结果；CPU检查在临时非零输出头下核所有活跃参数的损失可达性，随后恢复，不执行optimizer.step。
- API继承原loss、_samples、predict_samples、conditions；构造严格加载C checkpoint state_dict。后续侧支路checkpoint需先构造模型再load_state_dict，不能将其冒充原C输入。

提前量attention的key偏置会被softmax平移抵消，故key统一不带bias，避免统计与损失无关参数。片段两组仍共用相同key结构。

CPU初始均值采用rtol=1e-6、atol=1e-8（float32）检查，并记录实际差值与重复调用一致性；5672379实测同参数指纹、相同no_grad上下文的差max1.79e-7、relative2.20e-7，重复调用差0。逐位差异不能作为训练或实质预测改变。
