# DATA REGISTRY EXECUTION SPEC: 五国资料、分布与分类依据

Status: APPROVED_DATA_PREPARATION_V1，2026-10-14。用户明确批准检查现有国外资料的分布/特征、借鉴两篇论文的分组分类、结构化写入数据库以复用。编号WP-20261014-MULTICOUNTRY-DATA-REGISTRY-V1。

## 目标与边界

交付可查询的Science数据目录、站点/既有事件身份、带来源范围的质量/分布特征、分类规则与资料缺口，以及基于输入身份和方法版本的缓存。此项是资料准备，非新模型效果实验；不能预写跨国泛化已成立。中国仍为主应用，USA/Japan先登记candidate_external_profiled；本次描述统计不用于拟合训练尺度、分组切点、选站调参或修改既有ABC。

范围为现有中国DA-TL、德国CAMELS-DE-1h、冰岛LamaH-Ice、美国CAMELSH Hourly2、日本MLIT利根川Q。国内新增Hei/Luoma日资料、鄱阳/长江样例、广西实况只登记数据集级资格，不混入小时预测。无新下载/天气请求/采集器修复或恢复、训练/推断/test观测读取、阈值改动、数据删除/迁移、定时任务恢复。不要读取中国既有test的P/Q值；原事件目录的角色/身份元数据可以登记。美国日本全期统计是inventory描述，不可冒称fit统计或未曾查看数据的盲测。

复用独立工作树`.`，base58e0c00。服务器teacherbox=zxy，优先已有`./.venv/bin/python`，无GPU。所有解析、测试、统计、图均服务器执行。父负责本规格、DB schema/写入/回读、研究解释与接受；执行者仅`src/data_registry_v1/`及`data/catalog/data_registry_execution_20261014.json`，不改旧脚本。先commit，再独立server checkout/run_versioned。长运行持久后台，记录进度，可按已完成站缓存接续；本次不新建/恢复automation，不重复启动。不得把shell/密码/密钥写入DB。

## 输入与复用

- 美国`./data/Hourly2.zip`，5767 NC、未压缩约19.76GB；只逐成员内存读取，不全包解压。查每成员streamflow/time单位、fill和scale属性；Q已知m3/s、hours since1980-01-01 UTC仅为预检，不替代逐文件核验。无P；不能写成配对P/Q已就绪。
- 日本`data/raw/japan_mlit_tone_Q_2015_2024/chunks`，约98MB/32014CSV；站点JSON `data/catalog/japan_mlit_station_list_tone_20261013.json`、Q_ids和launch元数据。空10字节CSV是header_only，不是有效观测；缺失'-'非0。不能以旧ok/skip/error0作资格。站名先与请求/native ID及目录核对，歧义为unresolved_station_mapping，不能猜。
- 中国/德国已有`data/catalog/integrated_station_index_v2.csv`、`cleaning_v2_stations.jsonl`、accepted清洗回执可复用覆盖统计，标其原计算范围/时间。中国基础150事件身份仅从catalog读取，test元数据可登记但不得读取其观测。
- 冰岛必须沿1fa15b9修复后的registry/completion及父接受；禁止使用60ce3e7错误有效数。根`./external-home/multicountry_QC_fix_1fa15b9/results/experiments/multicountry_harmonization_full_1fa15b9_20261005/artifacts`须实查，不猜实际文件。
- 已接受ab720ba prepared fit元数据/数组可用于三国当前fit-window-union分布（only history P、history/label Q）；标明sampled_fit_windows_not_full_archive，按source row/time去重，不当全档案分布。不要重新计算原模型分数。
- 中国既有67条fit/dev事件Q形态及修正event table沿17b26fe/7a22804，可导入现成特征，不重算旧事件分析；完整洪量/历时/Ts仍NE。真实源文件若在另checkout，精确路径由既有回执确认。
- 国内辅助目录及当前receipt只用于dataset/coverage登记。字段质量疑点保留。只读现有状态，不启动任何采集器。

## 统计与处理

美国先取中央目录排序首3个NC作为真实解析核对，随后缓存这3个并继续全部NC；记录实际时间/资源，按站向量化，不逐小时Python循环。日本全量读取现有非空chunks，按已确认station ID+time去重，重复同值计数、不同值冲突排除出有效统计并保留冲突数；只在已确认同一站时间轴上判断连续性。未知映射不混入别站。

每站存：文件/行/有效Q/有效P及配对数（有明确配对才能填）；原时间跨度和有效Q首末时刻；时间间隔/最长有效连续段/最长缺测段；缺失、负值、零值、重复/冲突计数；有效非负Q的min/mean/std/q05/q25/q50/q75/q95/q99/max；最大正/负1h变化及对应前后值和时间（必须严格相邻1h且两端有效，不跨缺口）。单位和时区保存；不能把未知置0。覆盖分母区分完整声明时间轴和仅下载到的观察片段，不把两者混为同一完整率。无独立逐行QC时标basic_numeric_only，不宣称全部合格观测。

旧QC/已有特征可直接导入，写original_method/原computed_at与本次observed_at；统计范围区分fit、development、full_available_inventory、historical_QC。描述性q95/q99绝不替代模型fit阈值。中国事件片段与国外连续记录不混为同一个总体；流域面积未知时不把原单位流量直接作跨流域严重程度排名。

缓存键至少含dataset+native entity+scope+method_version+source_fingerprint。小文件用SHA256，ZIP成员用路径/CRC/大小及全包stat构成签名并注明不是重新计算全包MD5；目录文件按排序路径/大小/mtime形成索引签名，只有变化部分才读取/hash。源变化或方法变化产生新profile；旧profile保留。缓存重复执行应0新增原始成员解析，不能只做无效重复全扫。保存缓存manifest、processed/skipped/failed数和每站失败原因；解析失败局限于相应站，别伪造成功或无限重试。

## 分类依据

参考Tang et al.,2025,10.1029/2025EF006060与Tripathi et al.,2026,10.1029/2026EF008397。前者SMOTE是增强方法，非缺测补值/洪水成因标签；后者流域背景/相似流域组和事件雨洪过程类型分开登记。国家只是来源/验证分层，不是洪水类型。

本版分类规则登记availability（hourly_Q/PQ、daily_auxiliary、sample_only、unknown）、background（area/annualP特征可用性；气候-蓄水类别需PET/P和蓄水证据）、event_morphology（已接受连续Q特征，不硬设类别）、rain_soil_process（长雨/短雨/excess-rain候选，原论文所需雨量历时、前期土壤湿度等缺失时NOT_EVALUABLE）、legacy_flow_severity（仅既有fit尺度与既有角色事件，不估重现期）。没有可靠字段/完整阈值或尚未验证小时尺度适用性时label=null并列出missing_requirements，不用代理冒充实测湿度、不把日尺度阈值硬套小时。

## DB payload契约（JSONL，父写Science）

全部含project_slug=waterpaper。键稳定，schema表为：

1. datasets.jsonl → research_datasets：dataset_key,name,country_code,source_url,source_version,data_kind,nominal_interval_minutes,time_zone,variables(json),server_uri,source_fingerprint,fingerprint_method,inventory_status,model_readiness,planned_role,summary(json),observed_at。
2. entities.jsonl → research_data_entities：entity_key,dataset_key,entity_type(dataset/station/event),native_id,station_key(nullable父station entity_key),name,role_label,time_start/text,time_end/text,attributes(json),status,observed_at。dataset节点用于总览profile，station原生ID不因别国同名合并。event只用现成来源事件，不新造POT洪水事件；metadata_only明确。
3. profiles.jsonl → research_data_profiles：profile_key,entity_key,dataset_key,profile_kind,scope,method_version,source_fingerprint,fingerprint_method,n_rows/n_valid_q/n_valid_p(nullable bigint),time_start/time_end(text),statistics(json),quality(json),provenance(json),status,computed_at。n_valid_p未知不能填0，已知无P可0并说明。统计字段优先标准键q_min/q_mean/q_std/q05/q25/q50/q75/q95/q99/q_max、valid_q_fraction、max_rise_1h/max_recession_1h、longest_valid_run_hours、longest_missing_run_hours，单位与参考范围在provenance。
4. rules.jsonl → research_data_classification_rules：rule_key,name,level,method_version,status,definition(json),required_fields(json array),source_refs(json array),notes。由父给定五类规则/执行者按本节实现记录，不虚构原论文数字阈值。
5. classifications.jsonl → research_data_classifications：assignment_key,entity_key,profile_key(nullable),rule_key,label(nullable),status,evidence(json),assigned_at。只给有依据的availability标签，其余使用features_available、descriptor_only或NOT_EVALUABLE及missing_requirements。

JSON中非有限数必须null，不能NaN；键与FK逐项验证，unit/scope/unknown不能丢。profile key不可只用station ID覆盖历史；同输入重复导出保持业务键和内容一致。父分批upsert元数据、新profile append-on-conflict-do-nothing；不将逐小时数十亿行塞进Supabase。服务端SQLite/JSON缓存与原始资料只在服务器；Mac回收小型payload/manifest/图，不复制原始Q/P。

## 完成与验证

服务器一CPU聚焦检查：小时缺口不跨算涨速、NaN/负数/fill/zero区别、重复冲突、JP空表/映射歧义、timezones、事件role边界、稳定key与缓存复用、JSON无NaN/FK完整。无必要重跑旧测试。美国/日本实际阶段有真实run_record，原始数据只读；网络断线不启动第二份。一次小图展示各来源数据范围/缺测或可用性，统计总体不同须显式区分。

父核对至少美国1个站、日本1个非空站/空文件、已有中国事件profile和冰岛修复来源；核实际DB键/数量、query示例、RLS/访问边界以及重复导入不新增记录。最终可用性、未知和partial逐来源说明；工程成功不能认领分类或跨国预测有效。无新模型训练、额外下载、timer或发布。

## 2026-10-14 检查中发现的日本时间解析修复

首版Japan最长连续有效段的站点中位数和p95均为23h。父要求两文件跨日行核查后，确认raw中有23:00→24:00→次日01:00及有效Q；datetime.fromisoformat拒绝24:00导致解析伪缺口。这使首版日本覆盖、连续性、跨午夜涨速及分布只描述排除了24点的子集，不能解释成真实缺测。其它四国不受影响。

本任务范围内修正合法当日24:00为次日00:00 JST，无插值、不修复或恢复采集器。新代码先提交，在新派生输出复制旧JP去重SQLite；仅扫描现有非空CSV补入24点，按原唯一站名映射和冲突规则批事务处理，再重算JP profile。原始数据、旧cache和首版快照保留。方法版本inventory-jp-clock-v1.1；其它国家直接复用保存profile，不重读原始数据。核对样例跨日1h、唯一有效午夜补入数与validQ增量、冲突/重复、原样接续不重复插入。其它malformed仍标未解释，不将97647条全部归因于24点。该修复影响资料统计，不改变任何已完成模型实验。新旧结果和实际版本关联写入执行回执及Science同编号记录。
