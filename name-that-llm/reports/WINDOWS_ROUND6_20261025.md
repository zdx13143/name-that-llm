# 第六轮：嵌套温度校准融合

2026-10-25。固定80%四视角SVC、20%ModernBERT/Gaussian融合，新增训练折内部温度校准。原120个prompt / 2159行开发集，不是新独立验证集。**开发BA 0.370895580454404，预登记开发门槛全部通过，条件最终训练已完成。离线部署验证正在进行，尚无本轮Kaggle分数。**

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

部署使用CPU Float32，开发特征来自CUDA BF16。另有运行前固定的4次人工文本检查：Windows CUDA、Windows CPU、Linux CPU、Linux倒序；每次一批4条人工文本，最多一次前向，无失败重试。要求argmax与CUDA一致，融合概率最大差≤0.005，跨平台SVC分数差≤1e-10，Linux倒序差≤1e-8。已完成前两项，Windows CPU最大概率差`0.0047747640368526745`，四个argmax均相同。Linux检查待完成；不能据人工检查宣称所有真实文本预测完全相同。

上传资产只允许冻结权重、分类头、温度、标签映射、源码、依赖运行时与校验清单；不含比赛原始数据、开发特征、OOF标签或凭据。只创建新的私有Kaggle资产与Notebook，最多一次保存运行和一次正式提交。test≤4096行、≤9小时、CPU、Internet Off，生成`submission.parquet`，恰好保留每个test ID和18类合法标签。

当前Kaggle新保存运行=0、新正式提交=0。历史本账号0.351、队友0.601不能当作本轮分数。浏览器已确认当前账号登录且规则此前已接受；页面的reCAPTCHA服务连接提示尚未阻止规则读取，未尝试自动解决挑战。

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

原始证据在`outputs/windows_calibrated_20261025_r6/`、`outputs/windows_calibrated_final_20261025_r6/`、`outputs/kaggle_r6_preflight/`，部署资产在`outputs/kaggle_r6_assets/`。这些本地二进制产物不进入Git；源码、预登记、测试与聚合报告随本人分支发布。只读复核可运行`python scripts/verify_round6_results.py`；已结束的训练入口不得重跑。
