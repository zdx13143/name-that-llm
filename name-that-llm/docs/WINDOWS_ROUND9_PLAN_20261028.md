# 第九轮预登记：固定词视角部分去题

登记日期：2026-10-28。用户要求继续推进一轮、寻找提升。仅在本人 `exp/windows-4060-20261023` 分支运行一个开发候选，保留全部历史失败与计数。真实计算前提交本计划、配置、实现和合成测试。

## 问题与证据

R7官方权重4的四视角SVC在原120prompt/2159行的开发BA为0.44593837535014；R8固定Sinkhorn只到0.4473117024587613，增量门槛失败。旧中心化实验用word 1–2 gram、距离阈值0.65，2149/2159验证行是单例，不能据此认定有效恢复了题组。

本次只读核对远端main为`dc62708e7116af472257d8f9159424fec2078f01`。其e651 Notebook SHA256为`d87c8394b7ad8b3cff1d7aa20687699a29d08ebb92eab684dfe5b1e61c4dc835`。新源码的内容聚类过滤文档频率过高的词，再做平均链接、每簇最多18条；借鉴该无标签分组方法，但不复制完整大模型集成。R7 fold views只用该折官方训练文本和固定外部池拟合，可复用；全量full-fit模型和未绑定本地折的v5 adapter不能冒充OOF。

## 唯一候选

- 继续原120prompt/2159行、同一5折GroupKFold；旧30prompt/540行不训练、不评分。唯一对照是R7已保存的weight4 OOF，不重新fit。
- 固定40,261行外部池及每折标点筛选保持R7身份；不新下载、重选或再清洗池。所有14个历史continuation claims与22个输入文件SHA写入配置并在前后验证。
- 复用每折四个TF-IDF词表。仅word视角减去0.5倍同组均值，不重新归一化；char/shape/function保持原值。训练侧仅用当前训练折真实prompt和外部canonical group，两个命名空间隔开；只有2..18条的组进行中心化，单例和过大组原样保留。报告两类覆盖率。
- 验证侧函数只接收文本、唯一行ID及固定配置。先按ID稳定排序，使用英文停用词word unigram TF-IDF、sublinear TF、min_df=2、max_df=0.05；余弦相似度做有18条上限的average linkage，最低合并相似度0.15。少于40条或空词表回退单例。原始源码在相同次序上的合并公式保持，稳定ID排序是本地确定性补充。没有真题组数、label、Hungarian或Sinkhorn输入。
- LinearSVC与R7一致：C=0.5、balanced、max_iter=5000、tol=1e-4、random_state=42、dual=auto；官方/外部样本权重4/1。无融合、校准、阈值或权重搜索。

## 评估与停止规则

主比较：候选对R7的pooled balanced accuracy、五折BA、18类recall、混淆矩阵、纠正/改错数；按真实prompt整组配对bootstrap2000次，seed=20261028。真实prompt只在预测数组落盘后用于分组质量诊断（ARI、pair precision/recall、singleton/eligible/oversize覆盖），不反馈预测。bootstrap条件于已完成的转导预测，不在重采样内重新分组；多轮开发与外部池筛选使其只是描述性证据。

每折只进行4次分组调用：完整批次、反序批次、按ID稳定切出的两个半批；合计20次内容TF-IDF分组尝试。反序分组及margin必须一致（误差≤1e-7）。单条退化的检查直接以全部单例组预测，等价于逐行推理；不会额外fit词表。半批和单例均为固定压力检查，不能事后改选为候选。

所有条件同时通过才标记开发候选可采用：

1. 完整、有限输出，监督者退出0且所有次数精确符合预算；源、输入、历史claim哈希不变。
2. 主pooled BA≥0.45；每折BA≥0.35。
3. 相对R7增益≥0.01；配对95%区间下界>0；至少4/5折改善（>1e-12）。
4. 半批pooled BA相对R7不低于-0.01；单例不低于-0.02。

失败即关闭本轮，保留模型、预测、日志、源码和失败claim，不改参数、不回退候选、不追加实验。通过也只说明本地开发证据；完整强集成的增量、最终训练、部署和Kaggle提交是后续独立阶段，本预登记不启动它们。

## 预算与执行

新ID：`windows-topic-word-dev-20261028-r9`；固定输出`outputs/windows_topic_word_20261028_r9/`与同名continuation claim。最多5次SVC fit尝试、20次分组尝试；监督特征词表fit=0、encoder=0、最终fit=0、旧留出评估=0、Kaggle上传/保存运行/正式提交=0。两线程、墙钟3600秒、进程树RSS上限12GiB，Windows原生父链检查兼容venv redirector；自动重试0。失败也消耗次数；已有claim/output拒绝启动。

仓库根命令：

```powershell
.\name-that-llm\.venv-gpu\Scripts\python.exe -m unittest discover -s name-that-llm/tests -p test_topic_word_round9.py -v
.\name-that-llm\.venv-gpu\Scripts\python.exe name-that-llm/scripts/windows_topic_word_round9.py
# 仅允许首次，运行后不得再次执行：
.\name-that-llm\.venv-gpu\Scripts\python.exe name-that-llm/scripts/windows_topic_word_round9.py --run
```

外部池沿用R7全开发文本无标签去复制的局限，上游来源选择及未知语义重叠未消除。本轮不能拿其120prompt组件CV与队友150prompt完整CV或公开榜0.651直接比较。
