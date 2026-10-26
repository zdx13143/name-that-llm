# 第七轮：按用户明确要求提交一次

2026-10-26 14:44:49（Asia/Singapore），用户明确要求：**“请你提交一次到kaggle我看看分数成果”**。这覆盖原R7“增量gate失败则不进入提交”的停止条件，仅授权这次固定候选的必要最终拟合、部署与一次正式评分。原开发实验、失败门槛、OOF和所有历史claim保持原样；不将此次提交称为开发门槛通过。

## 固定候选与预算

仍是R7唯一候选：官方样本权重4、外部1，原四视角TF-IDF与LinearSVC全部参数不变；不改选等权对照、不加融合、Sinkhorn、Hungarian或测试时拟合。开发BA仍为0.44593837535014，原配对增量gate仍为false。

- 新final身份：`windows-external-weighting-requested-final-20261026-r7`，新输出`outputs/windows_external_requested_final_20261026_r7/`。
- 最终数据：原2159行/120个开发prompt，加固定外部池经全开发训练侧标点规则筛选后的行。旧30prompt不训练、不评分；不读取真实test来选择参数。
- 最多4次vectorizer fit、1次SVC fit，总5次尝试；1800秒、进程树RSS 12GiB、2线程。失败保留claim，不能恢复或自动重试。
- 保存四个人工文本的参考margin，仅用于Windows/Linux加载与批量/顺序一致性核验；不新增开发评估。
- 独立私有Kaggle资产及Notebook，CPU/Internet Off，最多1次Save Version & Run、1次正式提交；test≤4096行、运行≤9小时。平台对正式提交的隐藏重跑属于该一次正式提交。
- 打包只含冻结模型、推理源码、标签、合成人工核验及必要离线运行时/许可；不包含官方或外部原始文本、OOF、凭据。

## 执行前的实际核对

Kaggle当前账号Zephyr Pong，Name That LLM已接受规则；提交对话框显示今天剩余4次。没有R7已提交记录；本账号R6正式public 0.365与旧0.351可见。队友`ntl-e642` Version 2正独立评分，保持不动；不重复提交历史Notebook。官方规则允许满足可访问性要求的外部数据，当前12来源均来自已固定版本的公开Kaggle数据。

现有Linux CPython 3.11.17与固定依赖可复用，R7仅SVC不需要ModernBERT、Torch或大语言模型。最终推理严格读取`id,text`，输出`id,model`的`submission.parquet`，保持test ID顺序与18标签映射。Kaggle保存运行及正式评分分别记录，不把3行占位test通过写成正式分数。

新入口`scripts/windows_requested_final_round7.py`核对本次请求配置SHA、已完成R7 claim/summary/verification SHA，并把`development_gate_passed=false`写入新契约。原条件入口仍按原gate拒绝。配置与代码先提交推送，再做真实fit；后续部署和评分证据追加记录，不覆盖原第七轮关闭结论。

## 部署核验固定步骤

最终拟合已按`b5d3516`完成：4/4词表、1/1分类器，241.078303秒，抽样进程树RSS峰值4,690,862,080字节，监督退出0。新模型仍只训练2159官方开发行和38,724条筛选外部行。原开发门槛保持false。

推理包使用原训练时的已拟合float32 TF-IDF对象及两个原预处理函数，固定margin argmax。现有Linux CPython3.11.17仅打包numpy2.2.6、scipy1.16.3、scikit-learn1.7.2、joblib1.5.2、threadpoolctl3.6.0、pyarrow21.0.0及其发行版文件/许可，保留Python标准库，不携带Torch/Transformers/ModernBERT。打包和推理源码另行提交后再做迁移验证。

唯一Linux迁移验证从确切payload解包到临时目录，仅对四条保存的人工文本执行六次调用：整批一次、反序一次、单条四次；与Windows参考margin比较，最大绝对误差≤1e-7、argmax一致，并检验id/model Parquet回读。最多900秒，0fit、0编码器、0真实开发/留出/test评估。保存独立claim、stdout/stderr与验证收据；不覆盖失败。随后Notebook只加载同一SHA固定的payload，官方test按128行分批推理，输出顺序不变。
