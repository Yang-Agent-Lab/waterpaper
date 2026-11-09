# RESEARCH EXECUTION SPEC: 来源未核实前缀的覆盖敏感性

Status: APPROVED_CPU_DIAGNOSTIC_ONLY
2026-10-03用户批准继续数据来源核查及分阶段数据验证。本小步将已观察到的来源不确定性标为元数据并计算窗口覆盖，不改原数据或训练。原始数据真实性与前缀成因仍unknown；只作保守敏感性分析，不认定填补，不扩大数据或费用。

问题：当来源事件最前12条记录暂不用于输入和标签时，已有H6/12/24、K12窗口和峰前起点会损失多少？可信的负结果为严重覆盖损失/NOT_EVALUABLE；不据此判断预测优劣。

输入仅上一阶段559cb7a已验收的100个train事件、12672行train_input.csv及对应event_response.csv/角色清单。默认路径为本工作区results/experiments/china_response_audit_v2_20261003。禁止读取validation/test、原始文件改写、新数据获取、归一化/阈值重拟合、模型训练、发布或删除。

按每个event_uid原source_row数值排序，前12条标记source_prefix_unverified=true，包括原无效行，不能在有效行筛选后重新数12条。除此之外不把任何后续行标作已确认真实观测。标记仅在新清单中产生，原time/P/Q/source_row/role不改。

完整连续窗口沿用原pq_numeric_valid与相邻1h规则。H={6,12,24}，K=12：A保持已有窗口，报告每窗历史/未来各使用多少标记行；B只保留两侧标记数均0的窗口，保留A/B共同起点与A独有起点计数，不能把删除源数据与窗口资格改变混同。记录主峰真值固定使用559cb7a对原有效记录的主峰，不能按B剩余数据重新找峰。L={1,3,6,12}资格同时报告完整窗口及唯一内部峰，基准88个唯一内部峰；无资格保留原因，不插值/补零。

输出新目录results/experiments/china_prefix_coverage_v1_20261003：prefix_row_manifest.csv、window_prefix_exposure.csv、event_coverage.csv、basin_coverage.csv、peak_origin_eligibility.csv、summary.json，以及run_versioned记录。均为train-only。无需新图，沿用Mermaid/小表解释两种假设即可。

实现仅新增src/diagnose_china_prefix_coverage_v1.py，必要专用self-test；不改559cb7a原脚本。先提交后run_versioned，单线程CPU硬上限120秒，无新依赖；自检须覆盖原行排序/无效前缀计数/历史和未来标记/缺口/峰真值不改变/原H窗口数10316等核对。运行失败产物保留。计算若实际需超过边界则停止，不自动扩展。

未参与实现的主会话用独立时间戳集合与source_row标记复算全部H窗口计数、逐事件峰前资格及标记计数。PASS只接受覆盖敏感性结论。父会话将真实发现写入data/catalog与RESEARCH/CONTEXT并commit；代码仍在独立工作区，不合并。下一阶段真实基线训练另写执行规格，不在本小步启动。
