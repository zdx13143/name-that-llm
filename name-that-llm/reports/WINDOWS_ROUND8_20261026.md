# 第八轮：分析队友0.642，固定Sinkhorn组件未确认增益

2026-10-26。已完成一次固定组件实验与独立结果复核；**开发BA从0.4459383754到0.4473117025，+0.1373个百分点，gate失败关闭**。没有最终训练、资产上传、Kaggle运行或正式提交。本账号正式成绩仍是R7的0.414；本报告的0.447312是开发CV。

## 队友最新成绩与源码结论

[ntl-e642 V2 / 362382146](https://www.kaggle.com/code/zdx13143/ntl-e642?scriptVersionId=362382146)实际Succeeded、Public Score **0.642**。收尾重新查看[Submissions](https://www.kaggle.com/competitions/name-that-llm/submissions)，`ntl-e643` V1 / 362503121仍Notebook Running，其描述“0.643、Qwen/helper权重0.10→0.15”不是已返回的成绩。队友0.601到0.642的观察差为4.1个百分点，多个组件同时改变，不能作单组件因果归因。

固定下载Notebook SHA256 `2f1f717a5cf8384fffc958831c78d45fda85858c8f1a3e5079e8939781545cd8`；提取代码SHA `b88cb13e2f48746af014b966546d87c7cb1e398f9c37b2c6af0ab8e3c1bc9e9e`。本机证据在`outputs/team642_review_20261026/`，含`source.py`与`latest_submissions_visible_state.txt`/`latest_submissions.jpg`。源码点验位置如下（指冻结source.py行号）：

| 部分 | 实际内容 | 对继续推进的意义 |
| --- | --- | --- |
| 基础与complementary | 四视角SVC、多个decoder/encoder、plain/twin与Gaussian等读出；参考与complementary按85/15混合（4293–4331） | 我们的独立SVC只是其中一个组件，0.414与0.642不是同等模型预算的比较 |
| 题目结构 | 当前query文本TF-IDF、平均链接聚类，distance cut=.85（1173–1180）；多处对2–18行组做残差中心化（1913–1919等） | 测试没有真实prompt_id，分组质量是必要诊断；不能把真实题组送给预测函数 |
| ATLAS | 独立Qwen3-8B likelihood expert；仅低margin<.3行混入10%（4333–4342），小批<50行全部调用 | 低置信样本上的互补机制值得学习，但需要自己的无泄漏OOF和资源验证 |
| spectral helper | Qwen3第2/4/6/8/12/16/20/24层拼接32768维；八头包含Fourier/Laplace/RBF/KNN及word/char残差PCA；在ATLAS后混10%（1903–2028、4343–4352） | 不只是“增加一个分类器”；表示、去题与非线性读出共同变化。元数据要求2699官方+4404合成，现成full-fit银行不能当开发OOF |
| Qwen主增量 | 27B、NF4、64层全前向、最后8层adapter、全长、batch3、要求2张GPU；对**所有query**混10%（3966–4181、4360–4377） | 不是只处理低置信20%；本机8GB4060不能直接照搬这套双卡入口。混合后的名义份额为81%此前复合prior、9%spectral、10%Qwen |
| 最终平衡 | 300次列/行归一化，列目标n/18，<50行绕过（1206–1214、4377） | 是本轮可低成本独立验证的固定组件；均衡概率总量不等于硬预测每类数量相等 |
| 工程证据 | exact三条公开占位样例直接输出格式预览，不加载完整模型（13–37、4406–4452）；RuntimeBudget.remaining为inf且require为空（4186–4241） | 保存运行23.8秒不能代表隐藏全集耗时；参数名“预算”不等于实际超时保护 |

源码中的最终全量权重/元数据声明与我们原120prompt开发协议不同；没有下载执行这些重模型、没有借它们在已训练过的官方行上制造高OOF分数。当前运行“0新fit”也不代表所加载资产从未见过官方数据。`adaptive_svc_topic_correction`和旧`select_low_margin`虽存在于源码，不能仅凭定义就声称当前主入口调用了它们。

## 固定实验与实际结果

预登记[原计划](../docs/WINDOWS_ROUND8_PLAN_20261026.md)在`0eb0600`推送后才执行。唯一输入为R7已保存权重4 SVC的OOF，原2159行/120prompt/5折；按R2身份再次比对。不读官方原始文本、外部语料或旧30prompt，不训练。每个验证折独立平衡，候选只接概率，不接truth/groups；固定温度1、floor1e-300、300轮及50行阈值。

| 指标 | R7冻结对照 | R8平衡候选 |
| --- | --- | --- |
| pooled balanced accuracy | 0.44593837535014 | 0.4473117024587613 |
| 普通accuracy | 0.4460398332561371 | 0.4474293654469662 |
| 正确条数 / 2159 | 963 | 966 |
| fold0 BA | 0.4236111111 | 0.4166666667 |
| fold1 BA | 0.4351851852 | 0.4259259259 |
| fold2 BA | 0.4490740741 | 0.4560185185 |
| fold3 BA | 0.4629629630 | 0.4629629630 |
| fold4 BA | 0.4589371981 | 0.4750402576 |

变化346条，纠正93条、改错90条，净多正确3条；2折改善、1折持平、2折下降。配对增益 **0.0013733271**，2000次整prompt重采样、seed2026102608，描述性95%区间 **[-0.0106481481,+0.0129815398]**，即[-1.0648,+1.2982]个百分点。它是**给定已固定转导预测**的描述性区间，没有在每次重采样中重新执行平衡，不能推广成对任意批次或独立holdout的置信保证。

| 预登记条件 | 实际 | 结果 |
| --- | --- | --- |
| pooled BA ≥0.42 | 0.447312 | 通过 |
| 每折BA ≥0.35 | 最低0.416667 | 通过 |
| 对R7增益 ≥0.01 | 0.001373 | 失败 |
| 配对区间下界 >0 | -0.010648 | 失败 |
| 至少4/5折改善 | 2/5；一折仅浮点尾差，按持平解释 | 失败 |

完整性通过不替代性能门槛。列质量最大误差1.42e-14，行反序最大概率误差2.20e-15以内且argmax相同；这只证明计算与顺序一致，不证明算法提高识别能力。整个批次的构成会影响预测，不能声称切块/单条不变。均匀类别先验是部署假设，隐藏数据分布尚未验证。

## 18类开发召回

| 标签 | 支持数 | R7召回 | R8召回 | 差值/百分点 |
| --- | --- | --- | --- | --- |
| gpt-5.5 | 120 | 0.416667 | 0.550000 | +13.333 |
| gpt-5.6-luna | 120 | 0.400000 | 0.400000 | +0.000 |
| gpt-5.6-sol | 120 | 0.216667 | 0.183333 | -3.333 |
| gpt-5.6-terra | 120 | 0.325000 | 0.350000 | +2.500 |
| gpt-6-astra | 120 | 0.700000 | 0.783333 | +8.333 |
| gemini-3.1-pro-preview | 120 | 0.791667 | 0.891667 | +10.000 |
| gemini-3.6-flash | 120 | 0.650000 | 0.616667 | -3.333 |
| gemini-3.7-flash | 120 | 0.166667 | 0.175000 | +0.833 |
| gemini-3.8-flash | 120 | 0.675000 | 0.525000 | -15.000 |
| claude-fable-5 | 119 | 0.226891 | 0.193277 | -3.361 |
| claude-opus-4-6 | 120 | 0.658333 | 0.633333 | -2.500 |
| claude-opus-4-8 | 120 | 0.400000 | 0.433333 | +3.333 |
| claude-opus-5 | 120 | 0.591667 | 0.716667 | +12.500 |
| claude-sonnet-5 | 120 | 0.658333 | 0.641667 | -1.667 |
| qwen3.8-max | 120 | 0.558333 | 0.508333 | -5.000 |
| glm-5.3 | 120 | 0.125000 | 0.108333 | -1.667 |
| glm-5.3-flash | 120 | 0.175000 | 0.208333 | +3.333 |
| kimi-k3 | 120 | 0.291667 | 0.133333 | -15.833 |

8类提升、9类下降、1类持平。不能只取gpt-5.5或claude-opus-5的改善；kimi与gemini-3.8等下降抵消了大部分收益。

## 启动失败、资源与核验

原run在读取评分数据/任何变换前被父进程检查拒绝，attempted/completed transforms=0，监督耗时0.75秒；原claim、stderr、contract和源码快照完整保留。Windows venv多一层redirector，原R8直接父PID比较遗漏该层。复用R7已核验的原生路径/父链检查，真实子进程回归确认正常链通过、错误PID/时间/状态/已消费计数拒绝。[唯一启动修正](../docs/WINDOWS_ROUND8_STARTUP_20261026.md)在`52d42b1`推送后执行；独立claim，不修改原科学配方或放宽门槛，剩余119秒。

成功ID `windows-frozen-svc-sinkhorn-dev-20261026-r8-startup-1`：**10/10变换（5主+5反序），1.312秒，抽样进程树RSS 94,420,992 bytes约90.05MiB，exit0**；两次监督合计2.062秒，小于120秒。0fit、0模型predict、0encoder、0旧留出。原12个claim与R7/R2输入SHA保持，原失败claim另锁定。7项针对性合成/真实进程测试通过。

根代理运行[verify_round8_results.py](../scripts/verify_round8_results.py)，仅从保存数组独立重算BA、逐类混淆、五折、bootstrap、纠错/改错、概率质量与gate，核对源码快照/合同/输入/旧claim，全部通过；没有重新调用平衡或模型。完成后再次默认readiness实际返回blocked、拒绝已有claim/output。独立只读探子另从保存数组重算确认数值、哈希和执行边界；另一探子核对文档/学习HTML源SHA/相对链接，未发现确定错误。不删除记录重跑。

| 证据 | SHA256 |
| --- | --- |
| summary.json | `93ef68efc07db703c4e349e8e1394d617171a5abf97fbd88a34f0bb5edca0158` |
| candidate_oof.npz | `d85ebb9909701185abb3eef37988036eb4e57f181abd166d5ddf9a9753d7a47a` |
| supervisor.json | `61963e6b58feae7bf4a089126d7fc15a2b74f2bb426fbf4d2d96fa8226f36a33` |
| verification.json | `dae58317d911900a72ee99cbba371d7bbc99f027adbd1f9889c806ff97ee144c` |
| 原零变换failed claim | `5337905da538d3b8ce6dfaad2d9416fca79462a6b71b96229e13d8e794aacaad` |
| completed startup claim | `cf8647670781eb63dc40b3d9600e0c94cff8ed5acab79696e0aa7adde2cd487b` |

输出根为`outputs/windows_sinkhorn_20261026_r8_startup1/`，原失败在`outputs/windows_sinkhorn_20261026_r8/`。所有行级预测、源Notebook、claim和日志保持本地受忽略；Git只有自己的代码、配置、测试与聚合文档。

## 继续方向与边界

1. **降低后处理调参优先级。** 本轮固定Sinkhorn没有足够净收益，不追加温度/强度/迭代网格，不用公榜尝试修正开发失败。它在弱SVC上的结果也不能反推其在完整0.642中的贡献。
2. **下一项优先研究题目结构与读出。** 先在原120prompt开发范围诊断从文本推断的簇覆盖率、大小和题目一致性；真实prompt仅作诊断，绝不送入验证预测。旧中心化实验2149/2159行落入单例，不能跳过这一步。再另登记一个训练折内残差降维/非线性读出的固定候选，复用可用冻结特征并与当前R7对照，避免同时改变多个因素。现有ModernBERT表示不等同于队友32768维Qwen表示，收益未知。
3. **要接近完整强基线，先解决同协议资产。** 需要确切版本及每折未训练过验证题目的编码器/adapter/readout和OOF，或者明确登记新的折内重建资源。队友full-fit银行和最终2699行adapter不能替代这些资产；不向队友自动发送消息，不把缺少OOF写成已验证增量。
4. **重模型另作资源论证。** 27B双GPU全长路径、ATLAS和多专家不能直接放进当前8GB卡；正式迁移前先核对GPU/时间/离线依赖，真正实现超时保护。当前完整隐藏运行日志未取得，不根据23.8秒预览估算总耗时。

本轮已关闭，没有新提交候选。原30prompt、原R7一次提交预算和各轮claim继续封存。后续是新的预登记研究问题，不是重开R8；开发CV、多轮自适应选模、外部数据筛选和转导批次假设的局限全部保留。

已执行的审计命令（从模块目录；真实`--run`不要重复）：

```powershell
.\.venv-gpu\Scripts\python.exe -m unittest discover -s tests -p 'test_sinkhorn_round8*.py' -v
.\.venv-gpu\Scripts\python.exe scripts\windows_sinkhorn_round8.py --run
.\.venv-gpu\Scripts\python.exe scripts\windows_sinkhorn_round8_startup.py --run
.\.venv-gpu\Scripts\python.exe scripts\verify_round8_results.py
```
