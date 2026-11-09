# 校准补充：Sankouzhen小样本拟合的低步长检查

Status: APPROVED_CALIBRATION_DIAGNOSTIC_ONLY
依据用户已批准的训练充分性校准任务与A真实结果：b82ab9e阶段A的8目标达到预先参考线，Sankouzhen在同8样本/2000步常数lr=.001后归一化MAE=.02869889885187149，未达.02；500/1000/1500/2000步误差有波动。不能把它静默当通过，也不能据此直接判定模型容量不足。

问题：原参数更新步长是否限制了这8个固定样本的后期拟合？只做一次降学习率的针对性检查，不换样本、真值、归一化、模型、seed或判定线。

输入为A原Sankouzhen/C12/small_sample_selection.json、small_sample_predictions.json及原normalization.json。新独立脚本复用同Baseline26508，seed2、CPU单线程、Adam .001，完整重放原2000步以恢复当时模型与optimizer状态（旧检查未保存optimizer）；须先验证重放归一化MAE与原值一致（允许同环境FP32绝对差1e-6）。不一致则停止，不进入降步长阶段。

随后保持optimizer状态，将lr仅降为.0001，再训练最多500步，每50步记录同一误差与保存预测；不更改.02参考线，不增加第二次学习率尝试。达到参考线可停止，未达则保留NOT_REACHED。此检查仅说明8个训练样本的可拟合性，不证明完整训练收敛或泛化。

实现只新增src/refine_reliability_small_fit_v2.py，不改b82ab9e原运行脚本和结果。先commit再run_versioned；在新独立远端checkout运行，CPU/BLAS线程1，限120秒，从既有A+B7200秒总计算预算扣除，本次A及补充校准总和不得超过1200秒。保存run_record/logs、重复对齐检查、前后误差和全部所选原输入/真值/预测。保留原NOT_REACHED和A结果，禁止覆盖。

父会话独立复算小样本误差后记录B决策：无实现问题、参考检查达到且A最佳checkpoint均<400时，B cap500。若仍未达，明确说明该目标优化未决及进入/不进入B的理由，不悄悄剔除。B仍运行原b82ab9e源码，CLI的a-completion可指向独立的A阶段累计预算回执，其elapsed_seconds必须是真实A+补充计算耗时，附两次原completion/run_record路径，不能伪造或改写原日志。

不读test，不改原数据或B的学习率协议，不运行新的架构/生成器，不追加额外优化试验，不扩展总预算。
