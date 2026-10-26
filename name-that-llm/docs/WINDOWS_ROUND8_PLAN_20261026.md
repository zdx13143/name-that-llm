# 第八轮预登记：固定R7分数的批量概率平衡

2026-10-26。用户要求分析队友最新高分并继续推进。本计划在任何R8候选评分前固定，使用独立claim/output；已结束R1—R7和旧30prompt留出保持关闭。

## 依据与唯一问题

实际已评分的是[队友ntl-e642 V2，362382146](https://www.kaggle.com/code/zdx13143/ntl-e642?scriptVersionId=362382146)，Succeeded/public **0.642**。`ntl-e643` V1（362503121）在最初核实时仍Running，名称/描述中的0.643不是已验证成绩。GitHub连接器对私有仓库返回404，已用原Git连接只读同步main；main仍692ac87，本人分支a84f885。Git受限凭据组件首次失败，普通权限只读fetch成功；没有修改/合并队友分支。

固定下载Notebook SHA256 `2f1f717a5cf8384fffc958831c78d45fda85858c8f1a3e5079e8939781545cd8`，提取源码SHA `b88cb13e2f48746af014b966546d87c7cb1e398f9c37b2c6af0ab8e3c1bc9e9e`。本机只读快照在`outputs/team642_review_20261026/`。主线程亲读Notebook说明并点验关键源码；三个独立源码探子核对分组、spectral/ATLAS、Qwen/调度，另一个定位本地开发资产。

该方案是参考decoder/encoder与complementary集成、选择性ATLAS、八个spectral读出、全量query Qwen3.8-27B及最后一次Sinkhorn。ATLAS之后的名义混合为81%参考、9%spectral、10%Qwen。spectral银行使用2699官方+4404合成样本；Qwen最后8层rank16、64层完整前向、NF4、batch3，要求两张GPU。现成全官方拟合资产不能伪装成120prompt开发OOF。23.8秒保存运行来自exact三条公开样例的格式预览，没有加载模型；完整隐藏推理时间未获取。`RuntimeBudget`的remaining为inf、require为空，不能把其参数写成真正超时保护。分数提升4.1个百分点来自多项同时改变，无法单独归因。

**本轮只回答：对当前R7权重4 SVC做相同的固定Sinkhorn平衡，是否改善原五折开发结果？** 不把这个组件实验称为0.642完整复现，不下载/执行27B、ATLAS或full-fit银行。

## 数据与算法

- 原120prompt/2159行、五个GroupKFold保持不变；只读取已核验R7候选OOF的`softmax(margins)`和其身份数组，以R2原始OOF元数据再核对。不开官方原始CSV、外部语料或旧留出。
- 对照：R7冻结OOF，BA 0.44593837535014。候选：每个外层验证折单独执行300次列/行归一化；18类列目标质量为该批行数/18，float64、floor=1e-300。小于50行只行归一化。温度固定1，没有温度拟合。
- 与来源`source.py:1206-1214,4377`一致；五折不混合，平衡函数只接概率，不接truth、prompt_id或fold外数据。labels/groups仅用于身份核对、评估与bootstrap。
- 五次主变换与五次反序一致性检查，总10次；不增加模型fit、predict或encoder调用。输出保持原ID顺序；不声称单行/分块不变，整个批次组成会影响输出。
- 列概率质量接近相等不保证argmax硬计数恰好相等。本方法假定部署批次类先验近似均匀；这不是已验证的隐藏测试分布，分布漂移是风险。

## 指标、门槛、预算

记录pooled BA、五折BA、18类召回/混淆矩阵、预测类数、变化/纠正/改错数、概率列质量、行反序误差。按120prompt做2000次配对bootstrap，seed=2026102608；它是**固定转导预测条件下的描述性区间**，不重新估计每个bootstrap批次的平衡，不等于独立盲测或对所有批次分布的置信保证。既往多轮自适应开发及外部数据去重局限继续保留。

唯一候选须同时满足pooled BA≥0.42、每折≥0.35、相对R7至少+0.01、配对描述性95%区间下界>0、至少4/5折改善。检查完整性不替代性能门槛。失败后不调温度/强度/迭代次数、不改选别的候选、不提交。

原生Windows现有`.venv-gpu`，仅CPU两线程；120秒、进程树RSS 2GiB。0fit、0模型predict、0encoder、0旧留出、0自动重试。新身份`windows-frozen-svc-sinkhorn-dev-20261026-r8`，输出`outputs/windows_sinkhorn_20261026_r8/`。现有12个claim和输入文件SHA全部锁定，先做合成测试/只读readiness并推送预登记，再真实执行一次。通过门槛后才检查同一候选的部署准入；R7此前一次例外提交已消耗，不作为本轮绕过gate的授权。

```powershell
.\.venv-gpu\Scripts\python.exe -m unittest discover -s tests -p test_sinkhorn_round8.py -v
.\.venv-gpu\Scripts\python.exe scripts\windows_sinkhorn_round8.py
.\.venv-gpu\Scripts\python.exe scripts\windows_sinkhorn_round8.py --run
```
