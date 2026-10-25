# 第六轮：嵌套温度校准融合

2026-10-25。固定80%四视角SVC、20%ModernBERT/Gaussian融合，新增训练折内部温度校准。原120个prompt / 2159行开发集，不是新独立验证集。**开发BA 0.370895580454404，预登记开发门槛全部通过；条件最终训练、四次跨平台人工检查和Linux运行时迁移核验已完成。私有资产已上传，Kaggle Version 1保存运行成功；一次正式提交已成功，Kaggle Public Score 为0.365。**

## 候选与验证隔离

预登记计划为`docs/WINDOWS_ROUND6_PLAN_20261025.md`，开发配置`config/windows_calibrated_round6_20261025.json`。运行前已推送`ef18f9d`，条件最终入口在运行前推送`1a85d36`，部署四次人工检查合约在执行前推送`06253ba`。分支仍为`exp/windows-4060-20261023`。

每个外层训练折约96个prompt，内部三折GroupKFold分别拟合SVC与Gaussian，生成内层OOF。两个正逆温度分别最小化内层平均负对数似然，beta固定范围[0.05,50]、最多100次目标评估。外层验证使用哈希锁定的R5/R3对应外层模型；重算其未校准概率并与旧OOF完全核对，再应用训练折内部温度。没有使用旧全局OOF的训练行替代内层OOF，避免间接使用外层验证题目。

模型配方、权重、原外层五折、冻结编码器和特征不变。旧30-prompt留出不训练、不评分、不选模。开发新增30次模型fit、10次温度fit，共40次尝试；无新开发编码器前向。

## 真实开发结果

| 指标 | 结果 |
| --- | --- |
| pooled BA | 0.370895580454404 |
| 五折 BA | 0.3726851852 / 0.3680555556 / 0.3750000000 / 0.3981481481 / 0.3407809984 |
| BA描述性95%区间 | [0.3505483582, 0.3917613292] |
| 相对R3 Gaussian 0.2986772487 | +0.0722183318；95%配对区间[+0.0500076507,+0.0940006419]；5/5折改善 |
| 相对R5风格对照0.3593253968 | +0.0115701836；95%配对区间[+0.0041627762,+0.0189814815]；5/5折改善 |
| 监督执行 | 退出0；190.287109秒；抽样进程树RSS峰值545263616 bytes（0.508GiB） |
| 预算实际 | 30/30模型fit、10/10温度fit，170次目标函数评估；编码器0次 |

`finite_complete / pooled / each_fold / paired_gain / paired_interval / improved_folds / not_worse_than_style_control`七项均为true。温度校准不改变单个模型argmax，但改变融合时的分数幅度；五个外层风格温度约0.1303–0.1337，Gaussian温度约1.9418–2.0437。各折内层NLL均下降。所有区间为原开发集反复选模后的描述性证据，不能消除自适应开发偏差。

| 类别 | 第六轮OOF召回 |
| --- | --- |
| gpt-5.5 | 0.275000 |
| gpt-5.6-luna | 0.325000 |
| gpt-5.6-sol | 0.233333 |
| gpt-5.6-terra | 0.166667 |
| gpt-6-astra | 0.650000 |
| gemini-3.1-pro-preview | 0.850000 |
| gemini-3.6-flash | 0.408333 |
| gemini-3.7-flash | 0.375000 |
| gemini-3.8-flash | 0.316667 |
| claude-fable-5 | 0.134454 |
| claude-opus-4-6 | 0.475000 |
| claude-opus-4-8 | 0.208333 |
| claude-opus-5 | 0.641667 |
| claude-sonnet-5 | 0.450000 |
| qwen3.8-max | 0.491667 |
| glm-5.3 | 0.175000 |
| glm-5.3-flash | 0.216667 |
| kimi-k3 | 0.283333 |

## 条件最终训练

全部开发gate经主线程和独立审计核验后，对2159行开发集做三折内部OOF，6次模型fit+2次温度fit，再各拟合一个最终SVC/Gaussian，合计8次模型fit、2次温度fit。耗时70.490101秒，峰值RSS845901824 bytes（0.788GiB），supervisor退出0。

最终风格温度`0.1369934354093213`，Gaussian温度`1.8974942665741952`。最终模型导出并回读分数最大误差0；校准OOF仍仅2159行/120prompt。最终训练没有产生新的泛化成绩。全2699行fit=0、旧留出评分=0、新开发编码器前向=0。

## 离线部署状态

推理入口`inference_round6.py`只加载冻结模型，不训练。固定CPython3.11、NumPy2.2.6、SciPy1.16.3、sklearn1.7.2、Torch2.8、Transformers4.56.2。Linux CPU专用Torch从PyTorch官方索引下载，其余依赖来自PyPI；与Kaggle宿主Python隔离。编码器仍为原固定ModernBERT revision，最后四层attention-mask mean拼接3072维、max_length256、SDPA、关闭reference_compile。

部署使用CPU Float32，开发特征来自CUDA BF16。另有运行前固定的4次人工文本检查：Windows CUDA、Windows CPU、Linux CPU、Linux倒序；每次一批4条人工文本，最多一次前向，无失败重试。要求argmax与CUDA一致，融合概率最大差≤0.005，跨平台SVC分数差≤1e-10，Linux倒序差≤1e-8。四项全部完成且通过，恰好4次编码器前向。Windows CPU最大概率差`0.0047747640368526745`，Linux CPU为`0.004774891755220556`，均小于事先固定的0.005；四个argmax均与CUDA相同，跨平台SVC分数差0、Linux倒序恢复概率差0。不能据人工检查宣称所有真实文本预测完全相同。

上传资产只允许冻结权重、分类头、温度、标签映射、源码、依赖运行时与校验清单；不含比赛原始数据、开发特征、OOF标签或凭据。只创建新的私有Kaggle资产与Notebook，最多一次保存运行和一次正式提交。test≤4096行、≤9小时、CPU、Internet Off，生成`submission.parquet`，恰好保留每个test ID和18类合法标签。

实际离线包为`outputs/kaggle_r6_package/ntl_r6_payload.tgz.bin`，954,912,317 bytes（约955MB），含27,835个归档成员。已在Linux临时目录解压原包，用其中自带的CPython3.11.17验证固定依赖导入、资产哈希和冻结分类头；对保存的人工特征重算概率最大差0，新增编码器前向0、训练fit0。该核验不读取真实test或旧留出。Notebook结构、空输出和代码AST通过；冻结Notebook/打包与迁移脚本提交为`ff39b2d`。

开发和最终训练的`summary.json`是各阶段结束时冻结的记录，其中`deployment_validation_complete=false`、`linux_offline_validation_complete=false`保留原值及原SHA，未用后续进展覆盖。后来的人工兼容性与迁移结论分别以`kaggle_r6_preflight/verification.json`和`kaggle_r6_package/relocation_verification.json`为证据；它们覆盖上述限定范围，不等于Kaggle正式执行成功。

Kaggle新资产上传=1、新保存运行=1、新正式提交=1。私有资产[NTL R6 Assets 20261025](https://www.kaggle.com/datasets/zephyrpong/ntl-r6-assets-20261025)为Version 1，包含同一954,912,317-byte冻结包。已导入冻结Notebook，挂载官方比赛输入与该资产，确认Private、CPU/None、Internet Off、No persistence。浏览器确认当前账号ZephyrPong登录且规则此前已接受。

[私有Notebook Version 1 / 362038537](https://www.kaggle.com/code/zephyrpong/ntl-calibrated-r6-20261025?scriptVersionId=362038537)命名为`R6 calibrated frozen offline v1`，2026-10-25 06:34:23 UTC保存运行，页面显示**Successfully ran in 96.5s**。实际宿主为Python3.13.15，隔离推理Python3.11.17及固定依赖成功加载；对公开占位test的3行完成1次前向，模型fit=0、温度fit=0、旧留出评分=0。生成813-byte `submission.parquet`，SHA256为`dfbd16bb00885281f680204ab9508fbf9f10401487ff1896f388562ff732bb18`。这只证明保存运行与输出契约通过，不能把3行当作隐藏测试数量或正式评分证据。

2026-10-25 06:39:47 UTC从该唯一版本正式提交一次；07:08 UTC在[Submissions](https://www.kaggle.com/competitions/name-that-llm/submissions)及该条Submission Details核实**Succeeded、Public Score 0.365**，Notebook版本仍为362038537。06:39:47来自详情页提交时间，07:08是本地观察成功的时间；没有平台精确完成时间、隐藏行数或逐batch日志，不把约29分钟等待或96.5秒保存运行写成隐藏评测耗时。保存运行和正式提交各一次的预算均已使用，没有重复提交或事后调参。

本账号历史提交Version360768689显示0.351，本次0.365，按显示精度增加0.014（1.4个百分点）。这是完整方案在同一比赛public部分的历史比较，候选配方不同，不能单独归因于校准。队友Version360258830的历史0.601仍高于本轮，本轮没有刷新队伍最佳。正式public 0.365、本地开发CV 0.370895580454404、3行公开占位test保存运行与尚未知的private最终成绩分别记录。没有访问隐藏标签或把公榜当作新的调参集。

此前的上传失败记录保留：两次文件选择器等待超时，随后明确返回Chrome ChatGPT扩展未启用“Allow access to file URLs”；编辑器也曾持续“Loading Editor”，页面提示reCAPTCHA服务连接异常。用户在06:29:22 UTC确认已开启扩展文件权限，重新连接后实际上传及编辑器导入均成功。没有自动解决CAPTCHA或绕过正常页面；未改变模型、重训或重跑四次人工检查。

## 核验与证据

6项温度/嵌套合成测试、3项最终准入测试、3项输入输出契约测试通过。主线程`verify_round6_results.py`和独立审计从保存数组重算行/标签/分组/折身份、内外层隔离、校准公式、固定融合、指标、配对bootstrap与gate；最终导出另经独立哈希和schema审计。旧claims保持原状，失败历史保留。

| 产物 | SHA256 |
| --- | --- |
| 开发summary.json | e6f468b0f7d05144ba5cc5636a043dc006ae74c1385f4749106108fe83466843 |
| 开发oof.npz | e0f9670cb8d8cdeba5295852913635726fc22841554383c21f3c38d982aa3eb6 |
| 最终summary.json | f009f8a592accd6933383cbd1fa46edb2fc5c8bc950a805ae6982cc4b2ae017c |
| 最终style.joblib | 1e0cfdc294e3daf0205bb2f9ee04affd4301edfdda0f14adbfdef8955e026998 |
| 最终gaussian.npz | cbb896c30cf45327a1a94c6b94c110d341d60544ed092f6efdee48f36be2fd3c |
| 最终calibration.json | b0d6129a012cff03836e1ac9f33db6cf1c5dd6c06124bdff55f4ba8bba1fdcbb |
| 四次人工检查verification.json | f7440b4e7471a9bbbc1f7d10f7211d0c824016de252fb10add47f29e55f87701 |
| ntl_r6_payload.tgz.bin | 6d3798ba4b82ae69d049f131f373ff6629002b758d4e348a73d18e9d2cbde153 |
| calibrated_round6_offline.ipynb | 74a2fb8762117ea9a4e184c457d180087829d171d34dfde41c30e8b1b964ecf1 |

原始证据在`outputs/windows_calibrated_20261025_r6/`、`outputs/windows_calibrated_final_20261025_r6/`、`outputs/kaggle_r6_preflight/`，部署资产在`outputs/kaggle_r6_assets/`，最终包及迁移收据在`outputs/kaggle_r6_package/`。浏览器证据位于`outputs/windows_4060/`：`round6_kaggle_private_assets_created_20261025.png`、`round6_kaggle_version1_output_20261025.png`、`round6_kaggle_version1_visible_logs_20261025.txt`；此前`round6_kaggle_editor_loading_20261025.png`和`round6_kaggle_upload_pending_20261025.png`保留为已解决的失败历史。这些本地二进制产物不进入Git；源码、预登记、测试、空输出Notebook源码与聚合报告随本人分支发布。只读复核可运行`python scripts/verify_round6_results.py`；已结束的训练入口、人工前向和exclusive打包/迁移入口不得重跑。

正式成功证据另存为`round6_kaggle_public_score_20261025.jpg`与`round6_kaggle_submission_visible_state_20261025.txt`，待评分历史截图`round6_kaggle_formal_running_20261025.png`保留。原`round6_deliverable_integrity_20261025.json`记录提交后待评分时的文档哈希，保持不变；出分后文档与证据使用新的`round6_scored_closeout_20261025.json`收据。此次一次性实验与正式评分已完成，后续不重跑既有入口。
