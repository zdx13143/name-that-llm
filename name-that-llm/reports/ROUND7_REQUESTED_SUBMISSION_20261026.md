# 第七轮一次提交：本地完成，等待Kaggle编辑器恢复

2026-10-26。用户在06:44:49 UTC明确要求“请你提交一次到kaggle我看看分数成果”。固定权重4候选已完成最终拟合、Linux迁移核验和私有资产上传。独立Notebook草稿目前持续停在`Editor Loading`，导入不可用；**新Kaggle保存运行0次、正式提交0次，尚无R7公榜分数**。刷新和重新打开同一草稿仍未恢复。

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

## Kaggle当前状态与恢复入口

一次上传已成功创建私有[NTL R7 Assets 20261026](https://www.kaggle.com/datasets/zephyrpong/ntl-r7-assets-20261026)，页面确认Private。仅含`ntl_r7_payload.tgz.bin`与`README.txt`；没有原始语料、旧留出、OOF或凭据。

独立草稿：[NTL Weighted SVC R7 20261026](https://www.kaggle.com/code/zephyrpong/ntl-weighted-svc-r7-20261026/edit)。已核对Accelerator=None、No persistence，并设置Internet Off；但编辑器持续Loading，导入菜单禁用。添加官方输入未得到成功确认，不能假定输入已挂载。尚未导入冻结Notebook、启动session、Save & Run或正式Submit。

资源页曾出现reCAPTCHA服务连接提示，但上传实际成功；未自动处理CAPTCHA，也没有证据证明该提示导致编辑器失败。刷新及重新打开同一草稿未恢复，已请求用户确认手动打开状态或恢复页面所需网络连接。

恢复后只继续同一草稿：导入[冻结Notebook](../notebooks/weighted_round7_offline.ipynb)，挂载官方比赛与已建私有资源，重新核对Private/CPU/Internet Off，最多一次Save Version & Run；确认成功版本和合法`submission.parquet`后，最多一次正式提交并等待分数。状态不明先查版本/提交记录，不重复点击。最终拟合、打包及Linux人工验证已完成，不重跑。

开始前页面显示剩余4次提交。R6正式0.365是既有结果，队友0.601来自另一个完整方案；队友`ntl-e642` Version2当时独立运行，未操作。不能以这些历史分数或0.445938开发CV代替本次公榜。

## 证据身份

本地证据目录为`outputs/windows_external_requested_final_20261026_r7/`、`outputs/kaggle_r7_package/`、`outputs/kaggle_r7_submission_20261026/`；最后一处保存ledger、`private_asset_created.jpg`和`editor_loading.jpg`。原始输出受Git忽略。

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

最终claim为`outputs/continuation_claims/windows-external-weighting-requested-final-20261026-r7.json`。开发claim为另一个`windows-external-weighting-dev-20261026-r7-startup-1.json`，SHA为`d397abf349eafd17eb01e5a8c4b85bfbb8f754a683bfc2ee87c7648e2d4970c8`；两者不能混用。规范对象hash与JSON原始字节hash也分别记录。
