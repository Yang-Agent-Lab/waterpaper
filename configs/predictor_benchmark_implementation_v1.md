# Predictor Benchmark Implementation Plan

Goal: 同一输入与概率输出下比较三预测器，并始终交付同四页训练报告。
Spec: configs/predictor_benchmark_v1.md。
Architecture: 新包显式适配旧纯函数，历史代码与产物不改。模型训练生成统一report_bundle，绘图只消费保存结果。全部计算/测试/渲染在服务器，先commit再run_versioned。

## 任务与接口

- [ ] 报告执行者：src/training_report_v1/{__main__,schema,plot,legacy}.py。入口读取spec规定schema，旧ABC适配器只读既有epochs/selected_diagnostic/summary/event/predictions，输出bundle及来源manifest。先写并在服务器观察缺字段/null/role混用/概率峰值语义检查失败，再实现；重点检查缺失不画0、颜色稳定、不同模型/国家不串、输出始终四页。用旧保存ABC做真实演示，不重推断。
- [ ] 模型执行者：src/predictor_benchmark_v1/{model,data,diagnostics,training,score,runner}.py。统一build_model(name,p0,seed)与encode/distribution/sample/loss接口；共用旧S2纯数学。先测试三模型输出合法、共同头初始化一致、TCN因果、P0冻结、梯度流与诊断前后权重/RNG不变，再实现。新loader读取全部现有三国select但不访问test，保持cal/dev阶段隔离。固定采样数组、checkpoint指标、恢复身份和报告bundle接口。
- [ ] 端到端小试：已核真实fit/select，小试2轮×2更新×3模型，cal/dev使用明确fit/select替身；保存预测、选择、一次停止续跑验证、最终四页报告齐全。人工图查只修可读性，不反复美化。
- [ ] 父独立接受：核common head/sample hashes、实际更新/角色、CRPS与干预差值的保存数据复算、四页语义；通过后再唯一正式训练。初始研究判断、全部正负结果与实际版本写Science，主仓库仅加入口不混入既有未提交修改。

Review focus: 中国过滤不能悄悄丢掉外国select；旧验证loss缺失不伪造；四窗口峰不能画成小时水文曲线；冻结P0无梯度不判故障；临时干预不是正式模块贡献；新旧选模不同不直接归因；重复接续不增加训练预算。
