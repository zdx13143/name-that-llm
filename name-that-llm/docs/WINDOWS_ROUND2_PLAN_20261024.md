# Windows ModernBERT 第二轮独立预算

2026-10-24，Asia/Singapore；用户明确要求登记下一轮独立预算、保留失败记录、不重开旧留出，完成修复后的五折评估，有合格新候选时上传一次 Kaggle。本文件与配置在真实计算前提交到自己的 `exp/windows-4060-20261023` 分支。

## 既有证据和不变边界

上一轮已关闭：265次forward尝试、264次持久完成记录、0次fit，无完整特征或OOF。原claim曾被路径默认参数错误改写，当前两个claim的真实字节和法证副本全部保留，不回滚、不删除、不重新执行旧入口。新配置固定两条现存claim的SHA256，在新轮启动、工作进程核验及完成时检查其未变。

新实验ID为 `windows-modernbert-frozen-dev-20261024-r2`，独立固定claim及输出路径。`--round2` 与旧零计算恢复开关互斥，不提供任意ID、输出路径、覆盖、续跑或失败重试选项。新的540次batch预算不抹掉旧轮265次尝试；两轮历史分别记录。

仍只使用原120 prompt / 2,159行开发子集。官方CSV字节、labels、重建的120/30分区均验证原SHA。读取全文件hash与id/prompt元数据不等于评估保护集；旧30 prompt / 540行的文本与标签不进入特征、训练、评分或最终分类器。原291次CPU fit和完成的GPU合成probe不重跑。

## 固定模型、预算与门槛

固定ModernBERT-base revision、BF16/SDPA、长度256、batch=4、最后4层mask mean、3072维FP32；模型冻结，单次开发提取最多540个batch。固定GroupKFold(5, shuffle=False)、每折训练部分StandardScaler、LR C=0.1/lbfgs/balanced/max_iter=5000/random_state=0。最多5次CV拟合尝试，不调参数、融合权重或fold。

外部监督限1,800秒、采样进程树RSS上限8GiB、每batch检查CUDA allocated峰值6GiB、数学库2线程。每次forward/fit前持久计数；Windows临时文件替换的有限重试仅重写相同receipt字节，不重做计算。收敛警告、身份错误、缺失OOF、资源超限或进程失败均停止并保留日志。

提交资格在看结果前固定为：**pooled balanced accuracy ≥ 0.36，且每折 balanced accuracy ≥ 0.30**，同时五折完整、概率/行序/标签契约与supervisor资源检查全部通过。历史冻结75/25候选同开发子集BA=0.348673只作描述性参考；没有历史OOF，不计算配对增益，也不凭聚合分数宣称统计显著改善。该开发集被多次使用，bootstrap区间不代表未参与选择的独立测试。

## 达标后的条件预算

只有上述条件全部通过，才允许在独立固定最终head claim下再尝试**1次**StandardScaler+同配置LR拟合，训练仍仅2,159行，复用本轮验证过的特征。因此本轮总计最多5次CV fit + 1次条件最终fit；不在Kaggle重新训练，不合并旧保护集。

导出数值参数并核验推理与已拟合head一致，再制作独立私有离线Notebook/模型输入，绑定精确版本与哈希。条件部署预算为最多4次新的合成前向（验证部署差异，不重开旧probe）、1次Kaggle公开样例保存运行、1次正式隐藏测试提交；隐藏test超过4,096行拒绝执行，Kaggle运行不超过官方9小时。正式提交前必须验证离线依赖、资源、输出ID/18类、Notebook与模型身份及公开运行成功。不用榜单分数回调方案或重复提交旧候选。

若门槛不通过，本轮停止：不最终拟合、不上传、不提交、不追加搜索。若部署核验失败，保留失败证据，不扩大上述预算。0.351旧正式提交和队友0.601与本轮CV及未来正式分数分别记录。

## 唯一执行入口

从仓库根执行，默认仅核验，不消耗计算：

```powershell
.\name-that-llm\.venv-gpu\Scripts\python.exe .\name-that-llm\scripts\windows_modernbert_dev.py --round2
```

本轮真实执行在上述命令末尾加 `--run`。新claim或输出已存在即拒绝；任何失败均不得再次运行该入口。
