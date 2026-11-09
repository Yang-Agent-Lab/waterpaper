# 日本小时P/Q实际取得（有界首批）

WP-20261015-JP-HOURLY-ACQUIRE-V1，用户批准实际取得并存服务器。执行者1人，teacher-web=zxy，独立checkout/run_versioned，不触已有训练、环境、进程，不新训练/绘图/监听，不联系作者。只有新src/jp_hourly_acquire_v1/、本规格、data/catalog/jp_hourly_acquire_receipt_20261015.json属于本执行者。

作者成品优先核查：arXiv2111.04732的正式版本已定位10.1016/j.jenvman.2024.120931，J Environ Manage 359（2024）；作者机构存档https://open.metu.edu.tr/handle/11511/109654。保存官方出版/机构页面、预印本原文与GitHub仓库搜索回执；若未有具体成品URL，不臆造下载包。Ishikari若取得为新增候选，不替换Tone。单个作者包≤100MiB且本次总读取≤100MiB，不下载大气象格点包。

现有Q主身份栗橋303031283301030，已有2019年10月CSV只读复用，不再重下全Q；原始24:00规范到次日00:00 JST。Q元数据官方位置36°08′38″N,139°42′09″E。P候选至多栗橋103031283301020、川俣103031283301030、八斗島103031283301040；同利根川干流，川俣/八斗島为上游方向候选，栗橋点雨量位于Q附近。核官方SiteInfo坐标、站名与单站月表；上游多边形/面积权重未确定，禁止称流域面雨量或完整配对ready。

不同于上次多站生成端点，这次用本次独立cookie jar依次访问contents、SiteInfo、单站DspRainData.exe（20191001—31），必要时同单站HTTP作为协议回退；不原样重试旧SrchSiteSuiData2 URL。首版KIND=1后来由真实SiteInfo链接确认是任意期间小时表而非月表；月表用KIND=2，修订接续见下方。若取得，保留原HTML/下载链接与质量符号。必要一次单站Q时刻表DspWaterData.exe KIND=6作单位/时间对照，不抓全国。

JMA备用：前橋（prec_no42,block_no47624）2019-10-12小时表view=p1，实际成功后才串行取2019年10月其余天。该站位于Tone上游区域的候选点雨量；未核上游边界就不认面雨量。最多31个日页、小月，缺失和质量符号保留，不用日值插小时。

curl IPv4、Mozilla UA、connect15秒、单请求墙钟75秒、总600秒（超过即结束本collector）；单body≤10MiB，总body含失败≤100MiB；每个具体URL至多一次，仅协议回退是不同有效路线，JMA任一后续页失败则止该月接续。本次自己的cookie不与用户/其它会话共享、不输出值。失败body和stderr保留，不以exit0或HTML存在当有效数据。成功样例逐行时间/数值/单位/角色回执；未得到则列准确障碍与下一步。脚本/配置先commit，独立服务器执行自检后采集，所有原始数据只在服务器。

实际请求后已取得八斗島官方SiteInfo，北纬36°15′53″、东经139°11′48″；href明确SrchRainData.exe?ID=103031283301040&KIND=2&PAGE=0是月表。KIND=1的川俣响应只含任意期間時間雨量标题/雨量(mm/h)/质量说明，没有行，不计成功。修订recovery.py在新commit/新checkout仅取八斗島KIND=2设置页和2019年10月月表，保留form controls；不再重复其它雨量站/旧多站请求。JMA www端点若未取到，改官方ds.data.jma.go.jp/stats与/obd/stats两入口先核10月12日，24行成立才在同一有效入口接续该月。两个入口仍是同一前橋站和同月，未扩大站点/日期范围。

首版JMA www端点已实际得到小时页并推进整月。若首版完成，recovery用--reuse-jma读取其已存小时P JSON，不再次请求JMA；网络仅剩有真实链接依据的八斗島月表。新版本先commit后执行，旧run使用的源码不改。

JMA首版已取得31页744小时；原代码把所有非纯数字原样保留null。官方remark.html实查明确“--=无相应现象”，不能把无雨误算缺测；“×/ ////空白”等缺失继续null，“0.0”保留官方小于0.5mm的记录精度说明。normalize.py先实取官方符号说明后才按该语义生成新派生版本，原HTML/JSON不改，不重抓JMA月页；附取官方JST说明及当前前橋站位置JSON。现有Q首两日位于20190925八日chunk，补读仅这个既有邻接chunk，仍过滤2019年10月，不新下载Q。三相邻小时样例固定2019-10-12 10/11/12 JST；规范符号是源语义解释，不是缺失填零。上游面雨量资格仍partial。

离线规范最终按源月表时钟2019-10-01 01:00—2019-11-01 00:00 JST对齐Q（含原10月31日24点，避免把它排出月份制造假缺口），官方符号说明/站点位置从已保存metadata复用，不再请求网络。保存连续有效段与H12/K6候选窗口清单，仅历史P/Q和未来Q资格，不读未来P作为预测输入、不启动模型。所有窗口保留point_P_only_spatial_support_partial。

用户追加批准：调查官方JMA obsdl CSV批量接口，先同前橋一站一个年度；接口稳定且本任务总量<100MiB才可继续2015—2024，不逐日抓数千页。batch_csv.py先仅保存官方前端及至多8个同JMA域非通用库脚本，查真实station/element/hourly-period/日期请求字段，不猜接口后直接拉十年。此阶段仅discovery，无年度请求；后续确认真实字段后另提交年度入口再执行。既有2019年10月744页数据复用不重抓。整体已取得约6.4MB，后续总读取账继续累计，不能给每阶段重置100MiB。

已从官方top.bundle.js核实：时別値aggrgPeriod=9，降水量element=101且标“前1時間”，POST目标Qs+show/table；字段interAnnualType、option、csvFlag等由真实Vy函数定义，不沿旧接口猜201/1。annual_csv.py先取同官方模块whatsnewHistory.js确认API基址，再warm本任务session；首年只请求前橋2018年，必须台站/降水header、8760行、全时间连续唯一通过才继续同站2015–2024其它段。2019年10月已取得，2019只请求1–9月与11–12月；总11个批量请求，无逐日新增。起始账继承既有6093811+44213+248806+566203=6953033B，100MiB封顶。CSV原始字段/质量代码全部保留，解析只核时间/行数，不预判quality代码或填0；若首年不合格立即停止，不用日资料替代。新版本commit与自检后才实际POST。

若同站CSV年度实取完成，merge_span.py仅合并已存年度/两段CSV与已存10月P，保存完整2015–2024点P；Q复用旧launch/278个原生ID/4站一组→70组，按栗橋ID定位仅对应group10原CSV，拒绝名字歧义、冲突Q保留null，不重扫其它站Q、不重新下载。源quality由官方help3.html核8正常、5准正常、4资料不足、2疑问、1缺失、0非观测；raw全部留存，资格主清单仅CSV quality8，HTML月未带符号的数值/无雨记录另标html_unflagged，不把两种编码混为同一质量代码。输出同钟CSV、连续有效段与候选H12/K6身份清单，不生成模型张量/新训练、不把点雨量升级成面雨量。100MiB账继承年度receipt，合并阶段只多取官方质量说明小页。

逐年/逐月 coverage按原表1–24时归属（末日24时属于原日），分开P实际小时、P质量合格小时、Q快照记录小时、Q数值小时、没有取得Q行、已取得行中的缺失标记、双数值资格小时。header_only/无Q行只能称取得不全或源可用性unknown，不称物理缺测；原行'-'等为源快照缺失标记，未重新查官方现行记录时不断言永久无观测。先保存Q_missing_month_manifest，再视其具体月份决定有限单站补取，不无界补全全国。

用户追加要求补2,088小时未取得Q，即使不是整月全空。q_gaps.py先以已生成多年CSV及SHA256固定缺小时、连续缺区间和16个涉及月份manifest，再新run按官方Q SiteInfo实际href（SrchWaterData KIND=6为流量月表）补对应单站月表，优先缺720h的2018年1月。每月HTTPS一次，失败时HTTP一次；串行，connect15s/单请求75s/本轮600s，承接累计9652146B、100MiB总限。原缓存不改；月表虽然含已存行，只给原先没有取得的clock新增记录，已存41缺失行不填；已存数值与新表不同另出duplicate_value_conflicts.json、保留旧值并从资格统计排除冲突，不静默择一。新表原缺失标记与仍没取到的时间分开报告；源站/单位/行数/月份网格不合格不得补入。原月表HTML、curl失败和时间均保留。

父复核指出HTML质量码未经独立核对，不应与CSV quality8合称统一正常码。用户追加批准2019-10这一条同站JMA批量CSV例外重取（几十KiB）并与已存HTML逐时核差，旧月页/JSON保持不变；后续finalize.py统一采用所有年度/月份CSV品质码8作为资格，5/4/1等仍保留原值但不进入正常资格。必须在Q补取结束后串行执行，累计预算继承Q补取receipt；生成新表SHA256、年度已取/源标记/未取/冲突及连续段与候选身份，不实际构造模型输入张量。该改动是来源质量规范，不是新模型实验。

JMA help3.html说明均质番号每次选期从1重新编号；跨11个请求不得视作全期统一同质标签。保留每行P_source与原P_homogeneity，不据全部为1声称2015–2024观测环境不变，也不因此重下全十年。

Q月表首批已获得多个实际HTML，发现先前monthly_rows仅接受日号，而官方真实首列YYYY/MM/DD，造成已取得HTML未被识别为行。旧run与0行状态保留为parser_limited，不解释成源无数据。修正共享解析器支持并严格核对年月日，新版本优先离线复用旧run已成功HTML，不重取这些月份；仅原两路线都失败、无有效body的月份允许再有界HTTPS→HTTP一次，预算继承旧receipt。2020年1月实表明确m³/s，1月1日空白而1月2日起有观测，保留源未记录/取得不全之区别。

父提供yxz既有Python3 ._native_baselines_v1/.venv/bin/python（旧waterpaper/.venv也可），仅标准库、不改环境、不导入GPU库；SSH复用master，ControlMaster=no/BatchMode=yes/ConnectTimeout=10。先在teacher-web新run --cache-only离线解析成功HTML，再把同一manifest和该cache receipt（小元数据）送yxz新独立版本目录；yxz_months.py仅请求未成功缓存的月份，跳过已取月表，HTTPS→HTTP至多一次/协议，75s/请求与600s/本轮，承接同一100MiB账。新原件先存yxz，SSH流传回teacher-web新目录（Mac不落原件），最后--additional-raw-cache --cache-only合并两地成功原件。不得触碰native训练源码/环境/进程；原41小时缺失仍不填。
# Partial monthly HTML qualification (2026-10-15)

Reuse dated, 24-cell daily rows from a truncated monthly response when station ID,
flow variable and m3/s unit match. Only verified clock rows fill previously absent
Q; unlisted days remain acquisition unknown. 2023-01 cached response ends at the
2023/01/30 row opener: Jan 1–29 contain 696 verified hours, Jan 30–31 are unlisted.
This does not reinterpret unlisted rows as source missing. No re-fetch of this
successfully cached partial month; original Q missing markers remain unchanged.

