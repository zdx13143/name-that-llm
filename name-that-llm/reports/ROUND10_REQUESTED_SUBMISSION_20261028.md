# R10正式提交：冻结候选与平台证据

2026-10-28。用户明确要求本轮改进完成后提交一次观察分数。开发、最终训练及确切离线包已完成，代码/报告冻结提交`669fb20`。只在本人`exp/windows-4060-20261023`分支工作。

**已完成：正式隐藏评测Succeeded，Public Score 0.416。** 比R7的0.414高**0.002（0.2个百分点）**，仍低于队友完整方案0.651。唯一Save & Run和唯一正式Submit均已执行，本轮关闭，不追加调参或提交。

## 固定候选与本地证据

R9词视角半中心化基础上，对2..18行的文本推断组，每类margin固定减0.5倍组均值。开发BA **0.47740429505135384**，相对R9 **+0.0231248054777467**、5/5折改善，描述性95%区间[+0.0101124058,+0.0375097934]，R10预登记门槛全部通过。R9历史幅度门槛失败保持。完整结果、批次限制和18类召回见[开发报告](WINDOWS_ROUND10_20261028.md)。

开发20次分数变换、0新fit；最终仅1次SVC fit、0新监督词表fit，官方2159行/120prompt加固定38,724外部行。旧30prompt/540行未训练或评分。原120prompt已经多轮用于开发，CV和描述性区间不是盲测或完整队友0.651集成的增量证明。

离线包约211.26 MB；80条人工文本Windows/Linux完整、反序、单例共3次预测核验最大误差0。4096人工文本分组175.822秒、峰值约500.16MiB，恢复228组。上述均为兼容性/资源检查，不是隐藏准确率证据。整批text-only分组具有批次依赖，少于40行回退单例。

## Kaggle保存运行

- 账号：Zephyr Pong（`zephyrpong`）；提交前页面显示当日剩余5次。
- 私有资产：[NTL R10 Assets 20261028](https://www.kaggle.com/datasets/zephyrpong/ntl-r10-assets-20261028)，datasetVersion **21345327**。
- 私有Notebook：[NTL Topic Margin R10 20261028，Version 1](https://www.kaggle.com/code/zephyrpong/ntl-topic-margin-r10-20261028?scriptVersionId=363085022)，scriptVersion **363085022**，版本名`R10 frozen topic-margin offline v1`。
- 平台metadata：`accelerator=none`、`isGpuEnabled=false`、`isInternetEnabled=false`；官方competition ID **170484**及上述私有datasetVersion。
- 从Version History页面读取保存源码JSON，两cell的类型与源码逐字等于本地冻结Notebook；平台metadata/序列化变化使完整JSON SHA不同，不能冒称整个文件字节相同。
- 保存运行日志显示 **Successfully ran in 47.1s**；公开占位test **3行**，3个单例、eligible rows=0。模型fit/监督词表fit/encoder/旧留出评估均0，1次预测，0次分组词表fit尝试。
- 输出`submission.parquet` **808 bytes**、列`id,model`，SHA `aeb717f50c7b150da2f8d5c3fbe423fbf504cd94e6bd084004740d596fb3371e`。该3行占位输出与R7相同，不代表隐藏输出相同，也没有在这3行上检验非单例中心化。

## 正式隐藏评测

从上述确切Version 1选择根目录`submission.parquet`提交一次；页面解释平台将用隐藏test重新运行该版本。经过Notebook Running，**2026-10-28 07:35:15 UTC存证时已Succeeded，Public Score 0.416**。这是观察时间，不是平台完成时间。隐藏测试行数和实际运行耗时未知，不从3行/47.1秒或页面“5m ago”推断。最终选用checkbox未调整，队友版本未操作。保存运行与正式提交预算各1/1已用完。

| 同一阶段的比较 | R7 | R10 | 显示增益 |
| --- | --- | --- | --- |
| 原120prompt开发BA | 0.4459383754 | 0.4774042951 | +3.1466个百分点 |
| 正式Public Score | 0.414 | 0.416 | +0.2个百分点 |

本次公榜只显示很小的提高，开发收益没有等幅迁移。开发样本反复使用、验证与隐藏批次构成差异、题组大小及分组质量、训练/测试分布差异都是待检验的可能原因，当前没有隐藏标签或分组诊断，不能判定哪一项导致差距。R9没有单独公榜提交，R7到R10同时包含word中心化与margin修正，因而不能把0.002单独归因于R10分数修正；公榜也不是private最终成绩。后续若继续，应另登记针对泛化与批次差异的对照，而非据这次公榜调整系数。

## 哈希与本地收据

本机证据目录：`outputs/kaggle_r10_submission_20261028/`。包含ledger、保存运行可见日志、保存版本源码JSON及源码核验收据、正式Running可见状态，以及`public_score_0416.jpg`、`public_score_visible_state.txt`、`public_score_details.txt`。模型、运行时、OOF、原始数据及这些本机输出不入Git。

| 文件/身份 | SHA256 |
| --- | --- |
| payload（211,255,835 bytes） | `b91faaec404360805899bcebd65e31c8f960237a2e9615d57ab5399b38a9cd6d` |
| manifest | `3ba4692bc460d53feb64b2b48c7b9ed15cc880b28115b6b994143052e715ae46` |
| 本地冻结Notebook | `70f3d5b9ea52e2c001ce09fa27bb55b95d151b469c5dd9a006cef90ee329ebc9` |
| 平台保存源码JSON | `22ffb1e17655df2eeb855d6d9fb3ba7c675625709ab7f726dfec13aefbb54a76` |
| 确切包Linux迁移verification | `9fd8b0f975af22866cb45a1078ca280b12dfb4ec82232f499ffb363aa6f46523` |
| 4096行分组stress verification | `8f9f346fc416c7a330e660cdb5c69678aab01d5f3cf77e646f1d3897132dccfc` |
| 正式0.416详情截图 | `f3196defed38c846dd25660bbf928ec2440a3a5d0ab36f1e20dd56eb25e70c20` |
| 公榜列表可见状态 | `0cc0a0992d4fd7385df514387d212c5365d080bd69e00c03d37360154061278f` |
| 正式Submission Details可见状态 | `cf24e9abf88a2825f807fb36ffa87a7773e46f1b3dc63fbd1b42c9cddc3aafd3` |

本次没有重跑已完成开发/final claims。独立只读审计复核平台两cell源码、metadata、保存日志、输入版本及ledger一致。
