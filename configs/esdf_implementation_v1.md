# ESDF洪水版首轮实现约定

状态：用户已授权代码改造及正确性检查，并于2026-10-01明确同意中国9流域、单空闲GPU、一个种子、总6小时的首轮训练；配置见esdf_flood_first_run_v1.json。不是论文有效性结论。
起点：项目ae9c765；作者a271801501a5c45660c3da71c0c4f32105a0680b，MIT。保留上游原样，改造位于src/esdf_flood/，带作者归属与原许可。

## 问题与范围

问题：已有中国事件数据能否正确贯通真实预测、Q95定向生成、直接合成增强及三分支特征分离预测？运行失败、合成样本无效或没有预测收益均如实记录。国际源数据未形成正式训练窗口，本轮不暗中混入；中国本土评价仍为主线。

沿用configs/china_first_flow_v1.json的9个独立目标流域、已保存的事件划分和训练变换。历史12小时→未来12小时，评价1/3/6/12小时。Q95用于生成，Q99补充评价，二者只能读取已在训练事件拟合的阈值。温度不加入，不重新拟合尺度/划分。

## 组件接口（并行实施共同约定）

- data.py读取服务器data/processed/china_first_flow_v1/<station>/，加载角色必须明确；训练阶段禁止读取test。预测x为[B,12,2]，通道顺序P,Q；y为[B,12]，只含未来Q。
- joint为[B,2,24]：前12时刻历史P/Q，后12时刻离线未来P/Q；labels为[B]整数0/1，沿用generation_high95；不把joint未来部分或标签传给预测器。
- diffusion.py提供ScoreModel(hidden=64,sigma_min=0.01,sigma_max=50.0)，forward(x,t)返回同形score及[B,2] logits；diffusion_loss(model,joint,labels,config)返回loss/score_loss/classifier_loss张量字典；sample(model,n,config,device)返回[B,2,24]。config含sampling_steps、guidance_scale、corrector_snr、corrector_steps、sigma_min/max。
- 扩散保留VE连续时间去噪score目标+带噪类别交叉熵；将二维卷积适配为一维时间卷积，明确记录此必要结构变化。PC采样使用非零梯度log p(class|x,t)引导，guidance_scale可设0用于对照；不做[0,1]上界裁剪。
- forecasting.py提供真实LSTM基线及共享LSTM+真实MLP+合成MLP三分支模型。共享编码器从已训练基线复制初始化；保留预测头与共享重构器的不同职责。重构仅历史P/Q；真实预测只使用共享+真实专属表示。
- 保留预测L1、重构L1、MMD和正交约束；原实现LSTM系数1、0.1、1、0.5为显式初始设置，不称最优。直接合成增强与真实基线采用相同预测结构。
- 生成样本须反变换后检查有限值、P/Q非负及未来Q超过训练Q95；报告拒绝比例，不能偷偷裁平洪峰或用真实样本填补生成失败。抽样量有上限，零有效合成须明确失败。

## 执行与评价

train.py提供阶段baseline/diffusion/sample/naive/separation/all，目标可选单个或全部9流域；每阶段配置、种子、时间、状态及checkpoint保存到独立运行目录，禁止覆盖旧运行。阶段间加载本轮权重，明确缺少前置时失败。导入不启动任务。

代码实现先支持显式参数，正式预算写入配置后再运行。预测阶段按中国真实validation的MAE选模型，test只在最终模型评价时加载。扩散训练只train，可按训练损失保存；不拿test选择引导强度或合成比例。训练循环写每epoch日志；统一wall-time预算，耗尽保存当前状态为budget_exhausted，不宣称complete。若无法恢复中途优化状态，明确说明只可从已完成阶段继续。

输出各提前量的MAE/RMSE、Q95/Q99条件误差及计数，持续性基线；保留逐样本事件ID、预报起点、真实/预测Q。12小时窗内峰值和峰时误差可提供，但必须命名为window_peak，不能宣称完整洪水事件洪峰成绩。没有高流量样本的单元格为null/NOT_EVALUABLE，不写0。

## 正确性验收

一次针对性检查：数据角色隔离、未来字段不影响预测输入、张量尺寸、训练损失与梯度有限、共享初始化确实一致、引导0/非0确实改变采样方向、样本过滤无上界裁剪、指标小例子正确、模块导入无训练副作用。检查只能说明实现可执行，不是科学收益。先commit，再由run_versioned执行检查；修复后重新commit，仅复查受影响项。

禁止：改写原始/标准化数据和作者原版；读取test调参；混合各目标划分建立共享中国模型；改变Q95/Q99定义；无记录的大规模超参数搜索；对外发布。
