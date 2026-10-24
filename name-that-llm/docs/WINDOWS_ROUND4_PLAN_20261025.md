# 第四轮预登记：让编码器学习作者标签

2026-10-25，Asia/Singapore。用户授权继续一轮，有达到要求的进步候选再提交一次Kaggle。本轮在本人`exp/windows-4060-20261023`分支，新ID为`windows-modernbert-finetune-dev-20261025-r4`。保留全部已执行记录，不重开旧30-prompt留出。

## 假设与选择

第二轮冻结ModernBERT+LR BA=0.289904，第三轮题目残差PCA+Gaussian BA=0.298677，配对增益区间跨0。两轮都没有让编码器利用作者标签学习。本轮选择**全参数监督微调一个ModernBERT-base**，固定末层mask mean池化与18类分类头，检验任务相关训练是否优于继续改变冻结特征后的统计头。没有承诺提升，也不从队友历史0.601继承收益。

队友根`src/train_clf.py`有全参数分类微调参考，但直接使用全量150prompt、外部数据，并以外层每epoch验证分数选择最佳epoch。不能原样运行。本轮仅复用思路，另建隔离入口：每折从同一固定预训练revision重新开始，只接收原120prompt开发数据；不使用外部数据、不读test、不在外层验证上选择epoch。

已核对本地transformers4.56.2的ModernBERT分类实现和官方[模型文档](https://huggingface.co/docs/transformers/v4.56.2/en/model_doc/modernbert)、PyTorch2.8的[混合精度说明](https://docs.pytorch.org/docs/2.8/amp.html)。使用原官方权重revision`8949b909ec900327062f0ebf497f51aef5e6f0c8`，权重SHA不变；classification head的初始化与加载缺失键记录在每折报告中。

## 固定训练与验证

- 原120prompt / 2,159行，严格核对train、labels、split SHA和原行身份；五折身份与第二轮完全一致。旧30prompt文本/标签不进入训练、评估或部署。
- 每折重新加载原预训练权重；固定seed为20261025+fold，固定4个epoch，无早停、无最佳epoch选择。只有固定最终epoch模型预测验证折一次。
- `ModernBertForSequenceClassification`，末层mask mean pooling，classifier dropout=0.1。参数FP32、CUDA BF16 autocast、SDPA，reference_compile=false，梯度检查点use_reentrant=false。
- max_length=256，固定padding，micro-batch=4，累积8批（有效batch通常32）；不丢最后一批。每次累积按实际样本数加权，最后不足32条也完整训练。
- AdamW，encoder lr=2e-5、非encoder head lr=1e-4、weight_decay=0.01（bias/1D norm不衰减），foreach=false、fused=false；10% warmup后线性衰减，梯度裁剪1.0；无类别权重、无额外数据。
- 每epoch只记录训练loss、样本数和步数。五折全部完成后才计算pooled/fold/per-class BA、混淆矩阵和与第三轮固定OOF的配对bootstrap；不能看中间分数改训练设置。
- 保存5个最终fold checkpoint及OOF，但不把这5个模型在同一开发行上的平均预测当作合法OOF；每条OOF只来自未训练该prompt的对应模型。

## 独立预算与停止规则

真实运行前先登记并推送本计划、配置与代码。最多5次开发fit尝试。每fold为432个micro-batch/epoch × 4epoch，共8640个训练micro-batch、1080个optimizer update；验证共540次forward。

本轮另有**一次新的反向传播资源预检**：在4条人工文本上做1次训练forward/backward/optimizer step，然后2次eval forward检查倒序一致性。它验证全参数训练与Adam状态的显存开销，不重跑旧冻结提取probe；预检模型立即丢弃，每个正式fold仍重新加载原权重。预检计入同一新claim的独立计数，失败即关闭整轮，不改小batch重试。

总上限：9183次逻辑模型forward（8640训练+540验证+3预检）、8641次backward、1081次optimizer step。梯度检查点在单次backward内部的层重计算不另算顶层forward调用；报告保持该计数口径。每个操作之前持久记录attempt，之后记录complete，失败消耗预算。

外部supervisor限3600秒、抽样进程树RSS12GiB、数值库2线程；CUDA allocated最多5GiB、reserved最多5.5GiB，开始至少4GiB空闲；磁盘至少10GiB空闲。当前4060总8188MiB、桌面约占2093MiB；不终止其他用户程序。显存检查在forward/backward/step后执行，OOM、非有限loss/gradient或任何身份/资源异常立即失败关闭，无自动重试。

第一轮两个claim、第二轮和第三轮claim SHA在配置固定；启动与结束核对，不覆盖/删除旧claim，不把独立预算当作重新打开旧实验的许可。旧留出评估和全量2699行最终fit均为0。

## 提交门槛与条件预算

继续使用绝对门槛pooled BA≥0.36、每折≥0.30；相对第三轮必须配对增益≥0.02、2000次整prompt配对bootstrap描述性95%下界>0、至少4/5折改善。完整性、完成收据与资源检查必须全部通过。开发集已被反复使用，区间只是描述性证据，不是新的独立测试结果。

只有全部通过，才允许额外一次最终微调：仍只用2159行/120prompt，同样4epoch固定配方、最多2160训练micro-batch和272optimizer step。条件阶段使用单独固定claim；最多4次部署合成forward、1次Kaggle公开保存运行、1次正式提交，测试上限4096行、断网运行9小时内。先验证离线依赖、checkpoint身份、数值行为、输出契约与公开运行成功，再正式提交。不能在Kaggle重新用全量训练数据拟合。

门槛失败就停止，不改变epoch/lr/pooling、追加融合或以旧候选替代提交。更新根`PROJECT_LEARNING.md`和`PROJECT_LEARNING.html`，解释本轮学习到的监督微调、梯度、epoch和验证隔离。
