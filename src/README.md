# 研究代码

当前已实现服务器数据下载、观测统一整理、事件核查和中国首版预测窗口准备；尚未实现或运行预测/扩散模型。

- `catalog_china_events.py`：中国事件与连续窗口可用性清单；全局时间划分仅作候选比较。
- `compare_event_splits.py`：按目标流域分别比较时间划分及国际源数据年代资格。
- `prepare_china_windows_v1.py`：按 `configs/china_first_flow_v1.json` 生成中国首版样本，阈值和尺度只拟合训练部分。

时序处理在 teacherbox 完成，原始资料保持不动。本地仅保留代码与小型清单。运行前先读 `RESEARCH.md` 和配置；已有输出不覆盖。

## 提交后运行

修改后先检查并 `git commit`。在干净的项目工作区执行：

```bash
python3 src/run_versioned.py --run-dir results/experiments/<唯一运行名> -- python3 src/<任务脚本>.py
```

`run_record.json`记录启动提交号、实际命令、时间与退出状态，`stdout.log`和`stderr.log`保留输出。有未提交改动或已有同名运行记录时拒绝启动。仅检查启动时状态，运行期间不得改该任务源码；记录命令时不要传入密码或密钥。失败记录保留，修复提交后使用新运行名重试。

工具验证：先提交工具版本，再运行 `python3 src/check_run_versioned.py`；测试只使用独立Git样例，样例保留在results/experiments。
