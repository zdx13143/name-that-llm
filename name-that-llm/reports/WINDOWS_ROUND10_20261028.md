# 第十轮：组内分数去偏与一次Kaggle提交

2026-10-28，本人分支`exp/windows-4060-20261023`。用户要求继续提升并在本轮结束后提交一次观察分数。预登记、配置、实现与4项合成测试在真实执行前推送为`a15821e`。R9已关闭的增益门槛失败结论保持。

**开发候选通过全部预登记条件，选用R10。** BA **0.47740429505135384**，相对R9的0.4542794895736073增加**0.0231248054777467（2.3125个百分点）**，5/5折改善，按prompt配对的描述性95%区间**[+0.0101124058,+0.0375097934]**。相对R7的0.44593837535014，总增益**0.0314659197012138（3.1466个百分点）**。

当前部署状态：最终模型、Linux确切包迁移验证与最大行数资源检查全部通过，Kaggle上传/保存运行/正式提交均尚未执行。最终公榜待实际评分，本地BA不能冒充公榜。

## 固定方法与真实结果

本轮不重训开发模型，直接复用R9原五折margin和text-only推断组。对每个2..18条的组、每个类别的margin，减去该组该类别平均margin的一半。单例、过大组保持原样。系数固定0.5，没有新增温度、Sinkhorn、Hungarian、融合或参数搜索。原120prompt/2159行/GroupKFold身份保持。

| 批次 | R9 BA | R10 BA | 增益 | 改善折数 | 正确条数 | 纠正 / 改错 |
| --- | --- | --- | --- | --- | --- | --- |
| 完整，唯一候选 | 0.4542794896 | **0.4774042951** | **+0.0231248055** | **5/5** | **1031** | **120 / 70** |
| 两个半批，固定压力测试 | 0.4510348584 | 0.4681528167 | +0.0171179583 | 5/5 | 1011 | 120 / 83 |
| 单例，固定压力测试 | 0.4431489262 | 0.4431489262 | 0 | 0/5 | 957 | 0 / 0 |

完整批385条预测变化，净多正确50条；半批396条变化，净多37条。单例不做margin修正，因而逐条与R9单例相同。该结果仍依赖同批同题伙伴，不能据反序一致性声称任意分块不变。相对R7主候选共有456条变化，纠正147、改错79，净多68条。

| Fold | R9 BA | R10 BA | R10-R9 |
| --- | --- | --- | --- |
| 0 | 0.4305555556 | 0.4444444444 | +0.0138888889 |
| 1 | 0.4444444444 | 0.4490740741 | +0.0046296296 |
| 2 | 0.4629629630 | 0.4861111111 | +0.0231481481 |
| 3 | 0.4722222222 | 0.5023148148 | +0.0300925926 |
| 4 | 0.4613526570 | 0.5051328502 | +0.0437801932 |

门槛：BA>=.45、每折>=.35、对R9增益>=.005、配对区间下界>0、至少4折改善、对R7总增益>=.01、半批对R9半批>=-.01、单例对R9单例>=-.02，全部通过。预登记的“若数值gate失败则提交原R9”分支没有使用。没有因结果追加第二套系数。

## 预算与最终模型

开发20/20次分数变换（5折×完整/反序/半批/单例）、0新模型fit/监督词表fit/分组/encoder/旧留出，监督退出0；4.360秒，采样进程树峰值263,602,176 bytes。主线程保存数组复算与独立探子复算一致；两项静态审查未发现数学、选择规则或runner的可执行错误。

独立最终claim完成**1/1次SVC fit、0新监督词表fit**，官方2159行/120prompt加固定38,724外部行。复用R7最终四个词表前核验官方row identity、外部IDs与标点筛选；训练和R9保持相同word alpha=.5、组大小2..18、SVC参数和4/1权重。191.328秒、RSS峰值5,117,530,112 bytes、exit0、34迭代。旧30prompt/540行不训练不评分、全2699行fit=0、encoder=0。

80条人工文本覆盖40个双条组；模型导出/回读/反序的最大margin误差0。最终预测整批分组后进行word中心化和margin修正，最多4096行；没有沿用R7的128行切块。

开发输出`outputs/windows_margin_20261028_r10/`，最终输出`outputs/windows_topic_word_final_20261028_r10/`。所有已完成claim不得重跑或删除恢复预算。旧15个continuation claims及冻结输入逐一核验不变。

## 工程验证与资产边界

4项新数学/合成分组测试通过；3项复用输入schema/ID顺序/4096行上限/manifest测试通过。独立保存数组核对BA、18类召回、五折、bootstrap、纠错/改错、softmax与scores、输入/源码快照/旧claims及操作计数。静态部署审查确认整批分组、模型schema、白名单、隔离runtime和Parquet两列链路。

`external_source_manifest.json`是R7冻结的**14源审核目录**，不是实际入池来源的简化名单；其中Hailey/Michael两源虽被记录和审核，但没有进入训练池。实际排除证据在`outputs/round7_reference_20261026/prepared_pool_receipt.json`；不因清单仍有两源名字而将其当成训练输入。本轮没有新下载或更改池。

包只包含冻结模型、必要推理源、标签、来源审核目录、人工文本及六依赖Linux运行时；原始比赛文本、旧留出、OOF、凭证不上传。Kaggle采用新私有资产/新私有Notebook、CPU/Internet Off，最多一次Save & Run及一次正式提交。当前账号Zephyr Pong与5次当日剩余额度已在页面核验，没有R10重复提交。

确切payload **211,255,835 bytes**，SHA `b91faaec404360805899bcebd65e31c8f960237a2e9615d57ab5399b38a9cd6d`；manifest SHA `3ba4692bc460d53feb64b2b48c7b9ed15cc880b28115b6b994143052e715ae46`。Linux解包后80条人工文本的完整/反序/单例共3次预测调用，最大跨平台误差0，2次无标签内容词表fit尝试、0监督fit/encoder。4096人工文本的一次分组检查恢复228组，175.822秒、RSS峰值524,451,840 bytes。Notebook SHA `70f3d5b9ea52e2c001ce09fa27bb55b95d151b469c5dd9a006cef90ee329ebc9`，代码编译通过、无执行输出；这些是兼容性和资源证据，不是测试准确率。

## 局限与复现入口

这是反复使用原120prompt的开发CV；外部池保留R7全开发无标签排重、上游来源选择及潜在语义重叠局限。配对bootstrap2000次、seed20261028，在固定转导预测上按真实prompt重采样，不在抽样中重新分组，因此区间只作描述性证据。没有证明改善队友完整0.651集成，也不能保证隐藏批次题组构成相同。

```powershell
# 以下真实执行均已完成；不得再次运行 --run：
.\name-that-llm\.venv-gpu\Scripts\python.exe name-that-llm/scripts/windows_margin_round10.py development --run
.\name-that-llm\.venv-gpu\Scripts\python.exe name-that-llm/scripts/windows_margin_round10.py final --run
# 只读check()可复算；main首次独占写verification.json，已存在时不要删除重写：
.\name-that-llm\.venv-gpu\Scripts\python.exe name-that-llm/scripts/verify_round10_results.py
```

固定方法见[预登记](../docs/WINDOWS_ROUND10_PLAN_20261028.md)、[配置](../config/windows_margin_round10_20261028.json)；保留[R9原报告](WINDOWS_ROUND9_20261028.md)。下列逐类召回和哈希从保存文件生成。

## 18类召回

| 类别 | R9 | R10 | 变化（百分点） |
| --- | --- | --- | --- |
| gpt-5.5 | 0.425000 | 0.550000 | +12.5000 |
| gpt-5.6-luna | 0.391667 | 0.416667 | +2.5000 |
| gpt-5.6-sol | 0.258333 | 0.225000 | -3.3333 |
| gpt-5.6-terra | 0.333333 | 0.333333 | +0.0000 |
| gpt-6-astra | 0.708333 | 0.791667 | +8.3333 |
| gemini-3.1-pro-preview | 0.825000 | 0.891667 | +6.6667 |
| gemini-3.6-flash | 0.650000 | 0.691667 | +4.1667 |
| gemini-3.7-flash | 0.150000 | 0.175000 | +2.5000 |
| gemini-3.8-flash | 0.683333 | 0.666667 | -1.6667 |
| claude-fable-5 | 0.243697 | 0.193277 | -5.0420 |
| claude-opus-4-6 | 0.658333 | 0.741667 | +8.3333 |
| claude-opus-4-8 | 0.425000 | 0.400000 | -2.5000 |
| claude-opus-5 | 0.600000 | 0.741667 | +14.1667 |
| claude-sonnet-5 | 0.658333 | 0.683333 | +2.5000 |
| qwen3.8-max | 0.566667 | 0.550000 | -1.6667 |
| glm-5.3 | 0.141667 | 0.133333 | -0.8333 |
| glm-5.3-flash | 0.166667 | 0.208333 | +4.1667 |
| kimi-k3 | 0.291667 | 0.200000 | -9.1667 |

## 文件身份

| 文件 | SHA256 |
| --- | --- |
| outputs/windows_margin_20261028_r10/summary.json | `df146123b15bdcb4fbd5ea5ced983b1fff8f545d846550b3f4c584031e098e0c` |
| outputs/windows_margin_20261028_r10/supervisor.json | `8110151a96b4641fe898a48e9c33c42252e59aa540ee643aee81d8fae44bee9d` |
| outputs/windows_margin_20261028_r10/verification.json | `022371b525f6e3cefbaf5079c93c92cae2c29eaf2e32ef446bd8da4364020f74` |
| outputs/windows_margin_20261028_r10/candidate_oof.npz | `c7dbf7d8d67b1e7ef8a9bde1adc26dcd9fed2b09053e1f241444d5deaf9ed1f6` |
| outputs/windows_topic_word_final_20261028_r10/summary.json | `cc4110fdad9c04ad9efc29e72909136d409218bdbb818fcb02e370aabb05a1d4` |
| outputs/windows_topic_word_final_20261028_r10/topic_svc.joblib | `160b5612fa8487668b5ad5f0ad252288bdb58b03879a731abb7bd04b6b513c9b` |
