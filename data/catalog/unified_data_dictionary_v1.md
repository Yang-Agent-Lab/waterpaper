# 首批数据统一与清洗约定

## 当前实际来源

- 中国DA-TL：9个basin_data/*.xlsx观测文件，列time/Q/P/T/events，分场洪水片段；合成与迁移预训练文件另存，不能混入观测真值。
- 德国CAMELS-DE-1h：timeseries下1611个站点CSV，7类属性CSV，另含边界及benchmark_models模拟结果。流域属性按gauge_id关联，模拟结果不加入观测表。
- 压缩包解开总约53.42GB，其中小时CSV约51.46GB；当前从ZIP直接流式读取，不全量解压。
- 德国单位和时区依据发布包附带的2026-045_Dolich-et-al_data-description.pdf第4页；实际字段为water_level_obs，说明表写water_level，已经按实际CSV字段映射。

## 两张核心表

1. stations_unified_v1.csv：每个来源站点/流域一行，键station_uid=来源名:站点ID，包含国家、原成员文件、数据是连续记录还是事件片段、单位/时区核实状态、坐标/面积及原来源可用性摘要。德国1611行、中国9行。中国缺失的属性不猜补。
2. 每站一个压缩CSV：station_uid、source_dataset、source_row、event_id、time_source、time_utc、q_value、p_value、p_gapfilled_value、t_value、water_level_m、precipitation_uncovered_pct、qc_flags。按来源/站点保留边界，不能首尾拼接成一条全球序列。

字段单位从站点表读取。德国Q=m3/s，P=mm/h，T=摄氏度，time_utc=UTC；原水位cm转换为m。德国降雨代表前一小时累计，温度为小时起点值；不擅自移动时标。中国Q/P/T单位尚未最终核实，保留原数值；time_utc为空，time_source保留原时间。不得因T接近290就擅自从K转摄氏度。

## 本轮处理规则

- 不修改原始包；新输出写服务器data/interim/standardized_v1_pilot/。
- 统一列名、站点命名空间、数值表示；德国只做有文件依据的单位/时区转换。
- 缺失、非有限或不可解析的选中数值输出为空并加标记；不填零、不插值。
- 负降雨/负流量保留并标记；高洪峰不按分位数或箱线图直接剔除。
- 重复时间、冲突记录和事件内非小时间隔保留并标记，不静默平均/删除。
- precipitation_mean 与 precipitation_mean_gapfilled 分开，不能把作者填补数据冒充原始降雨。
- 按中国事件编号检查事件内部连续性，不填补两场洪水之间几个月/几年空档。
- 不做归一化、不设正常/极端阈值、不划分训练集、不训练扩散模型。

## 本轮范围与尚未完成项

统一站点目录覆盖现有两套数据；时序整理先覆盖中国9流域及德国按文件名顺序选的3站，作为处理流程验证，不代表正式研究样本。其余德国站点仍保留在原ZIP中。

本轮完成不等于全量清洗完成或数据可直接训练。后续须核对中国单位/时区、时间异常和德国观测质量；再确定样本范围、完整事件测试划分，训练期内计算阈值及归一化参数。站点表内发布方基于全时段计算的水文统计只作资料说明，不直接作为模型输入。
