# US/JP小时降雨实际样例：批准的小范围采集

WP-20261015-USJP-RAIN-SAMPLE-V1。来源样例核验，不是模型实验。执行者1人、teacher-web=zxy，独立目录，不安装依赖、不训练、不重下Q、不访问中国test、不写数据库、不创建定时。

美国只读现有Hourly2.zip中央目录确定一个USGS ID；只请求Zenodo旧forcing元数据、7z32字节头和至多4MiB下一目录头。必须206且body在单请求/总量上限内；忽略Range立即拒读整包。目录头不证明站点P可独立解压，不通过下载21GB解决。不能取得一个站的3小时P则保留partial并说明最小后续范围。

日本以既有栗橋Q身份303031283301030:0202为参考，取同利根川水系栗橋/川俣/八斗島雨量ID：103031283301020:0101、103031283301030:0101、103031283301040:0101；2019-10-08—15（小月内8日，沿官方单请求8天限制）。复用已有MLIT请求/临时dat方式但不复用旧解析器的24:00问题。无上游边界只认同水系点雨量实表，不认流域面雨量或已配对资格。若本请求失败不扩大全国/年份范围。

下载串行，timeout25秒，每端点最多2次，1秒间隔。累计上限64MiB，应用实际读取payload包含失败/IncompleteRead，加每请求16KiB保守header余量；不宣称精确网卡线速计量。单页面2MiB、7z目录4MiB、7z起始头64B。原始成功/部分body及请求headers保留；未传完不当成功。所有原质量符号原样存，缺失不填0，24:00→次日00:00，时区/单位从实表核，不从名称推断。

实现src/us_jp_rain_sample_v1.py，标准库parser selfcheck核24:00/缺失/相邻小时；先提交规格/代码，再同步全新独立server checkout，run_versioned先自检再采集。结果服务器results/experiments/us_jp_rain_sample_v1_20261015/artifacts；回收仅receipt到data/catalog/us_jp_rain_sample_receipt_20261015.json。来源报告保持原样；失败、unknown、partial保留。当前脚本仅元数据与少量内容，禁止读取Q序列。

首run实际获得32B/40B的206 Range头，40B为EncodedHeader，指向31376B LZMA1目录。增量版只复用首run原始header与receipt，另取该31KiB目录并以标准库解码，核NetCDF文件名、既有USGS ID是否存在和packed-stream数；累计预算从原receipt继续扣，不重抓元数据/日本P。只支持此已核布局，其他结构拒绝。源码另提交后在新checkout运行，不改首run源码；不解压数据流、不取Q值或整包。

首日本HTTP端点两次25秒超时，增量版仅换同站/同8日HTTPS端点（每端点仍至多2次）；原ID的分隔符沿已知采集器原样保留。不加站、不加年份。原美国文件名01011000_hourly.nc规范USGS ID为01011000，历史首receipt保留原记录，增量receipt注明更正。累计额外保守扣首run两次失败请求header余量32768B。

目录实取3166个文件，首文件01011000.nc与现有Q身份匹配；8个LZMA2 packed streams，首块2347101695B。用户追加允许原64MiB累计限内仅取首块前缀增量解首文件：每次最多4MiB、Range必须206，解码max_length限制首文件字节，不解第二站。首文件完整才用已有waterpaper/.venv中的h5py核P变量属性和time前3小时，绝不读取同文件Streamflow值；未完整则partial并停止。沿目录receipt累计预算，不重取美国目录，不再请求日本（HTTP/HTTPS均2次25秒超时）。本版本需新commit和新server checkout/run_versioned自检/执行；不安装h5py。

前缀首版暴露urllib socket timeout不限制慢速持续传输总时长。只中断本任务下载child（没有停止训练或其它进程），保留exit130与3440640B原始部分响应，累计3628587B。修正为SIGALRM每请求25秒墙钟期限；新版本复用该部分响应并延续累计预算，后续块缩为256KiB，避免重下已取得前缀。仍只解首文件、不再请求日本；失败停止保留partial。

末轮ebb7b09实际62.426秒结束，未触及父要求120秒总限，第三个256KiB请求触发25秒墙钟超时后停止，没有再重试。后来补全Fetcher显式120秒deadline，SIGALRM取单请求25秒与总剩余时长较小值；不再运行采集。最终code仅该总限补全未经另跑网络，原实验执行版本仍ebb7b09。预算回执保守将旧4MiB请求与新失败256KiB请求均按请求全额扣除，累计5217835B（含headers余量）<64MiB。未得到完整首NC/P值，不声称预算不足，而是本次墙钟/传输失败停止。
