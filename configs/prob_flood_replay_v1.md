# RESEARCH EXECUTION SPEC: 已有P0/P5权重的中国大上涨回放

变更编号：WP-20261010-PEAK-REPLAY-V1。状态：APPROVED。
授权：用户2026-10-10在“已有模型回放，区分见过也不会与新事件失败”的讨论后明确“可以，按照你说的来做”。本轮只执行上一报告第1项；不同时执行指标重筛或校准改动。

## 问题与决策
固定已训练模型时，训练中见过的大上涨是否仍被严重低估？最佳轮次与第100轮是否不同？
- 若最佳与末轮在fit大上涨上都明显失效，优先考虑学习组织/共享表示/目标或输入歧义，不能唯一归因容量不足。
- 若末轮能拟合而最佳不能，增强选模取舍解释；不据此重新选择开发最优权重。
- 若fit能拟合而dev失败，增强覆盖/外推/可用信息解释，但也可能只是记忆，不能证明历史输入足够。
训练与开发事件组成不同；角色均值差不是严格匹配的泛化损失。统计单位为来源事件、站点；窗口和512条轨迹不冒充独立洪水。

## 已有来源与固定比较
原模型提交48d2929b483476021c435286651f369899d34cd4。
原正式产物：
teacherbox:._prob_flood_48d2929/results/experiments/prob_flood_formal_48d2929_20261010/artifacts
只加载seed_2/P0与seed_2/P5的best.pt、last.pt（last内model字段），参数只读，记录文件hash及真实checkpoint轮次。
P0 best=28、P5 best=57；last=100，以原selection/checkpoint实际字段再核对，不制造新选模。

固定准备数据：
teacherbox:._prob_flood_ab720ba/results/experiments/prob_flood_prepare_ab720ba_20261010/artifacts/data
manifest SHA256=e482500d17c6cd900b985c67ff3c6a274b43a892d5f779316032cdf3bfb0ecb6。
仅使用中国fit4554窗与原中国development1356窗；读取现有fit/development的NPZ和rows、fit已确定尺度/强上涨cutoff及数据manifest。混合fit文件加载后立即按country=China筛选，不评分国外；不读test，不重新划分/归一化，不读取未来P。select/calibration只允许原权重选择元信息，不加载其样本或拟合任何映射。

四个固定状态：P0-best、P0-last、P5-best、P5-last，外加当前Q持续性。P0一条曲线；P5每窗512条自由轨迹，CPU、eval、inference_mode、无优化器和梯度更新。历史12条P/Q及原8指标定义不变。
为保证最佳/末轮可配对，P5各状态同一role/窗口顺序、同一固定随机数和batch切分；建议沿用evaluation batch32、role固定推断seed，数值写入配置。推断seed不是新增训练seed。CPU与旧GPU随机序列可能不同，不把Monte Carlo差异当成权重变化；保留前256/全512差异。不得为了节时暗减轨迹数，时间不足留partial。

## 固定分析
主看fit已有strong_rise_cutoff定义的强上涨子集，兼报全部、高流量、当前未高/已高与退水；cutoff来自既有fit，不新调阈值。保留每站/来源事件及计数，空类为NOT_EVALUABLE。
输出：
- Q6、前6h/12h过程、峰6幅值与峰时误差；分别给原物理单位和原fit Q95规范值。
- 真实/预测signed_change6与rise6、峰6样本中位与5/95分位、覆盖；P0/持续性不虚构概率区间。
- 全部合法起报点及全部来源事件的结果；逐事件标签峰的最大值、合法真实主峰前6h起报、相应预测。若最大标签峰没有合法峰前6h起报，显式不可评价，不把历史中见过峰当作模型已监督到该预测标签。
- 四状态严格同一成员；持续性同一口径；最佳与末轮按同窗口/事件配对。
- all/checkpoint/station/event层次表；重大失败与改善都保留。不根据dev误差筛案例，不更改原18组排名或结论。
无需重做概率校准；本轮问题是连续上涨/峰幅拟合，不将风险概率校准错误混入峰幅诊断。

图最多两份，服务器生成：角色与状态比较，以及训练大上涨和两困难开发案例的曲线。
病例选择固定：每站fit来源事件的最大真实未来标签峰中，取具有合法峰前6h起报的最大事件，再跨站取物理峰最大的3站；另取所有合格fit事件中按真实峰排序的中位事件作普通参考；固定dev Liqingdian洪水6、Luanchuan洪水9。只按观测/资格选，不按模型优劣选；未合格者表内保留。横轴相对起报小时、纵轴m³/s，图表明确fit或dev及best/last。所有源事件仍在完整表，图仅解释。清楚即可，不反复美化。

## 执行、预算与停止
现有隔离worktree：.，当前8f40826（实施者启动前核对）。只新增本任务前缀源码、配置、检查与执行状态；不改原prob_flood_v1模型代码、数据、权重、旧结果。父会话负责主RESEARCH/CONTEXT/Science和结论，实施者唯一部署/运行。
允许文件：src/prob_flood_replay_v1.py；configs/prob_flood_replay_v1.json；tests/prob_flood_replay_v1_checks.py；data/catalog/prob_flood_replay_execution_20261010.json；对应忽略的results目录。需补小helper保持同前缀。
单CPU线程，CUDA_VISIBLE_DEVICES为空，torch/BLAS/OMP均1线程；整个实际计算累计不超过1200秒（含fixture/小检查、推断、评分及绘图），0 GPU，0训练。代码准备和网络等待另记，不拿预算剩余扩大范围。使用独立CPU时间账，所有尝试累计不重置，到限partial；不得转GPU或自动加预算。
代码/配置先commit，再在新服务器checkout中经run_versioned执行；原48源码与产物只读。排他锁防重复。一次端到端小检查覆盖4状态加载、CPU自由采样、角色映射、持续性/评分、写出和累计预算；通过后同一实施者执行完整固定范围，不重复旧18组smoke。
实际原权重hash前后核对；CPU设备/参数不变/模型源码和配置来源、运行命令、依赖、真实耗时、退出状态、完整成员与输出hash可复查。非有限值、成员不一致、权重被改、意外GPU或预算不足停止，保留失败；工程修复限本规格。
若预计超过10分钟，后台执行并仅用本线程有界接续；旧已完成训练或旧监控不恢复。

## 完成和独立接受
实施者回传精确提交、真实运行记录、源/权重manifest、完整汇总与困难事件、服务器图和本地同步位置。不能仅凭进程退出码宣称模型学会洪峰。
父会话未参与实现，独立核对加载状态/角色与若干保存预测、主要聚合和病例，再给研究结论。预算partial不等于科学阴性。
Science新编号先登记、完成后更新并回读；飞书继续暂缓。图同步后在同一macOS Preview窗口查看。任务完成即停，下一新训练或校准方案另行批准。

