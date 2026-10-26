# 第七轮一次提交：Kaggle正式成功，Public Score 0.414

2026-10-26。用户在06:44:49 UTC明确要求“请你提交一次到kaggle我看看分数成果”。固定权重4候选已完成最终拟合、Linux迁移核验、私有资产上传、一次保存运行和一次正式提交。07:37 UTC核实同一版本状态为**Succeeded，Public Score 0.414**。相比第六轮显示分数0.365增加0.049（4.9个百分点），仍低于队友另一完整方案0.601。原开发增量gate继续为false。

## 授权与科学结论

这次请求覆盖原“gate失败则不提交”的执行停止条件，仅观察同一候选的正式分数。开发BA仍为0.44593837535014，对照0.4454831932773109，增量gate仍为false；未改选对照、增加融合或追加开发搜索。完整开发指标与18类召回保留在[第七轮报告](WINDOWS_ROUND7_20261026.md)。新入口使用独立claim/output，原条件入口继续按原gate拒绝，原OOF和历史claim不改写。

授权、预算与入口在真实fit前推送为`b5d3516`；部署代码为`b6489ba`，资产构建Windows路径键修正为`d5e521e`，Linux验证后的冻结Notebook为`51e6bb6`。固定范围见[请求登记](../docs/ROUND7_REQUESTED_SUBMISSION_20261026.md)。

## 最终拟合和迁移

- 仅2159条官方开发行/120prompt，加38,724条筛选外部行，共40,883行；官方/外部权重4/1。旧30prompt不训练、不评分，全2699行fit为0。
- 4次词表fit、1次SVC fit全部完成；517,992维、37次SVC迭代。241.078303秒、抽样进程树RSS峰值4,690,862,080字节、监督退出0，无编码器前向。
- 模型回读及人工文本反序、单条/批量检查误差0。未重新进行开发集评分。
- 210,522,119字节离线包仅包含冻结模型、标签、推理源码、来源清单、人工参考、CPython3.11.17和六个依赖：NumPy2.2.6、SciPy1.16.3、scikit-learn1.7.2、joblib1.5.2、threadpoolctl3.6.0、pyarrow21.0.0。
- 从确切payload迁移到Linux临时目录后，四个人工文本共六次预测调用全部完成；与Windows参考margin最大误差0，argmax一致、Parquet回读通过。这只验证四条人工文本兼容性，不是准确率或隐藏测试性能。
- 19项R7针对性测试通过。独立只读核验确认模型、payload、manifest、Notebook哈希链与计数；主线程补核开发/最终两个claim的确切路径与SHA。

原summary的`deployment_validation_complete=false`和打包时receipt的`packaged_not_yet_validated`保留生成时状态；后续验证以独立`relocation_verification.json`与`deployment_ready.json`为准，不回写历史产物。

## Kaggle正式结果与运行证据

一次上传已成功创建私有[NTL R7 Assets 20261026](https://www.kaggle.com/datasets/zephyrpong/ntl-r7-assets-20261026)，页面确认Private。仅含`ntl_r7_payload.tgz.bin`与`README.txt`；没有原始语料、旧留出、OOF或凭据。

[私有Notebook Version 1（362398792）](https://www.kaggle.com/code/zephyrpong/ntl-weighted-svc-r7-20261026?scriptVersionId=362398792)，版本名`R7 frozen weighted SVC offline v1`，仅一次Save & Run All。页面确认Private、CPU/None、Internet Off，挂载官方比赛170484及私有资产版本21251756。保存运行成功39.0秒；宿主Python3.13.15，隔离推理Python3.11.17。公开占位test为3行，预测调用1次，模型fit、编码器前向、旧留出评分均0；输出808-byte `submission.parquet`，SHA `aeb717f50c7b150da2f8d5c3fbe423fbf504cd94e6bd084004740d596fb3371e`。

约07:34 UTC正式提交上述确切版本及`submission.parquet`一次，页面先显示`Notebook Running`，07:37:16 UTC核实**Succeeded / 0.414**。该时间是观察时间，不是平台提供的完成时间。正式分数来自隐藏测试的public部分；未取得隐藏运行逐batch日志、确切行数或耗时，不用保存运行的3行/39秒代替。最终选用checkbox未调整。

此前编辑器持续`Editor Loading`、导入禁用，刷新及重新打开未恢复。用户随后确认手动打开已能正常显示，继续同一草稿后成功导入[冻结Notebook](../notebooks/weighted_round7_offline.ipynb)并完成后续流程。资源页曾有reCAPTCHA服务连接提示，但上传实际成功；未自动处理CAPTCHA，也没有证据证明该提示是编辑器故障原因。旧失败截图保留，已不是当前阻塞。

本请求的保存运行1/1、正式提交1/1均已消耗；没有重复fit、上传、运行或提交，也没有据公榜追加调参。0.414高于R6的0.365，但配方同时改变了外部数据、特征与融合方式，不能把全部提升归因于权重4。权重1对照没有正式提交，本次公榜也不能替代本轮配对开发比较。队友`ntl-e642` Version2在最终观察时仍独立运行，未操作；不能把其名称中的0.642写成已得分。

## 证据身份

本地证据目录为`outputs/windows_external_requested_final_20261026_r7/`、`outputs/kaggle_r7_package/`、`outputs/kaggle_r7_submission_20261026/`。最后一处保存ledger、正式`public_score_0414.jpg`/`public_score_visible_state.txt`、保存运行日志/截图、接收提交时截图及历史编辑器失败截图。原始输出受Git忽略。

| 产物 | SHA256 |
| --- | --- |
| 最终contract规范JSON对象 | `76f08316464e7d99eb882c79e691e519718b4fea1395c5155f06cf7d1652271b` |
| 最终contract.json文件字节 | `3f01ee10fe2da4b5ffc80fdc27390a2766e9f829f5361e53c0d083058ae8a2b` |
| 最终summary.json | `31d2f072d12d088a15ca75b2989a94ab23db76ade7bea5e381402bc617ca3a46` |
| 最终completed claim | `3972a43cf20e282e53c99f4e8bc4d5e4036edb4eb434d1ed4c5bf2cbb4534917` |
| weighted_svc.joblib | `5667dcdc58cb51bf71fdd5b96aa8dba35c0dc43771388a44521220562c398a8c` |
| 资产manifest.json | `d3cdf995074e9b0352da18f94bbe6155f6d11e22a238830213c96620e5474e01` |
| ntl_r7_payload.tgz.bin | `19e424acf23278c366014bdc4cdaf4bede928b13e6c68e2ebd383a67edf6fadc` |
| relocation_verification.json | `8f1b436c80f9277b88466d9ef481b88052653f601e889277b393e22664afcb44` |
| weighted_round7_offline.ipynb | `01b7a9756ade69b607b4bfb5672f275e756fb6bdfc46e9c630084271cc13260a` |
| 正式public分数截图 | `334cbb9a58be6911c7e7c72bfc554a4210a2520feee6e1881b159429be7e68fb` |
| 正式public页面可见状态 | `0c4b0a62c3b7734c335d5d2cd82693fb020366c41c946d4151d5e04e1556494f` |

最终claim为`outputs/continuation_claims/windows-external-weighting-requested-final-20261026-r7.json`。开发claim为另一个`windows-external-weighting-dev-20261026-r7-startup-1.json`，SHA为`d397abf349eafd17eb01e5a8c4b85bfbb8f754a683bfc2ee87c7648e2d4970c8`；两者不能混用。规范对象hash与JSON原始字节hash也分别记录。
