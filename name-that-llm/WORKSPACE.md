# 独立实验入口

2026-10-26 当前：**R8固定Sinkhorn完成，增量gate失败关闭**。队友最新已核实public为`ntl-e642` V2/362382146的0.642，`ntl-e643` V1仍Running。复用R7原五折OOF，BA **0.44593837535014→0.4473117024587613**，增益0.0013733271、描述性区间[-0.0106481481,+0.0129815398]、2折改善/1持平/2下降。5次主变换+5次反序检查，0fit/encoder/旧留出/新Kaggle操作。原零变换启动失败保留，唯一修正`r8-startup-1`完成1.312秒/90.05MiB；7项测试、保存结果独立重算通过。`0eb0600`原预登记、`52d42b1`启动修正均在各自执行前推送。两个claim不得重跑，无最终候选或新提交；本账号正式0.414保持。完整分析、18类召回、证据与下一方向见[第八轮报告](reports/WINDOWS_ROUND8_20261026.md)。以下为R7及以前历史状态。

2026-10-26 后续明确请求已完成：**R7正式Succeeded，Public Score 0.414**。固定权重4候选用独立claim完成最终4词表+1SVC fit（2159官方开发行+38,724外部行，241.078303秒），Linux四个人工文本六次调用误差0，私有资源已创建。编辑器恢复后，Version 1（362398792）CPU/Internet Off保存运行成功39秒，随后一次正式提交于07:37 UTC观察成功；保存运行/正式提交各1/1已用完。比R6公榜0.365高4.9个百分点，原开发gate仍false，旧30prompt未训练/评分。详情及全部SHA见[本次提交报告](reports/ROUND7_REQUESTED_SUBMISSION_20261026.md)。以下R7开发“final不存在/未提交”为新请求前的历史检查点；不重跑已完成claim或据公榜追加调参。

2026-10-26 最新状态：**第七轮真实五折完成，增量gate失败关闭**。锁定队友0.601版本360258830的源码后，重建40,261条外部池，检验官方样本权重4对1。对照BA **0.4454831932773109**，候选 **0.44593837535014**；候选对R6+0.0750427949、5/5改善，但对本轮对照仅+0.0004551821、配对95%区间[-0.0060400005,+0.0069444444]、3/5改善，三个增量条件失败。10/10分类器fit+20/20词表fit，892.042534秒、RSS峰值4.353306GiB，监督退出0。原零fit启动失败保留，唯一修正使用新claim并保持原科学配方和总预算；13项测试及主线程OOF重算通过。最终入口实际返回blocked，final claim/output不存在，最终fit/旧留出评分/新Kaggle上传运行提交均为0。不是完整0.601复现，不把组件CV当公榜；固定去重池与上游选择局限见[第七轮报告](reports/WINDOWS_ROUND7_20261026.md)、[预登记](docs/WINDOWS_ROUND7_PLAN_20261026.md)。不得重跑两个R7 claim或事后改选对照提交。

2026-10-25 第六轮历史状态：**第六轮嵌套温度校准完成，固定80/20融合BA 0.370895580454404，开发gate全部通过**。相对R3+0.072218、95%配对区间[+0.050008,+0.094001]；相对R5较强风格对照+0.011570、区间[+0.004163,+0.018981]，均5/5折改善。开发30次模型fit+10次温度fit/190.287109秒，条件最终8次模型fit+2次温度fit/70.490101秒，二者均退出0，无新开发编码器前向。原2159行/120prompt、旧30prompt和历史claims边界不变。固定4次人工文本离线检查全部通过，Linux隔离运行时迁移后的分类头输出差为0。约955MB冻结包已上传到新的私有Kaggle资产，Notebook Version 1（362038537）保存运行成功，耗时96.5秒；一次正式提交已成功，Kaggle Public Score 为0.365。新保存运行与正式提交各一次的预算均已使用，不重复执行。此前浏览器权限/编辑器问题已恢复，失败记录保留。详见[第六轮报告](reports/WINDOWS_ROUND6_20261025.md)、[预登记](docs/WINDOWS_ROUND6_PLAN_20261025.md)和HANDOFF末节。

2026-10-25 第五轮历史状态：四视角风格SVC对照BA **0.359325**；唯一80/20融合候选 **0.323681**，pooled<.36且弱于风格对照，gate失败关闭。5/5次fit、89.948856秒、无新编码器前向。没有事后改选对照、最终fit或Kaggle提交。详见[第五轮报告](reports/WINDOWS_ROUND5_20261025.md)；预登记`65d7e54`在真实fit前推送。

2026-10-25 第四轮历史状态：全参数ModernBERT监督微调**五折完成，性能下降，提交门槛失败**。pooled BA **0.264858**，相对第三轮下降0.033820，配对95%区间[-0.054189,-0.014333]，0/5折改善。5/5 fit、9183逻辑forward、8641backward、1081optimizer step均完成（含人工反向预检）；supervisor退出0，2031.168秒。主线程与独立OOF复核一致，四个历史claim哈希不变。9项训练合成测试及5项条件准入测试通过，但不能据此称模型有效。条件最终入口已准备但被gate阻断，final claim/output不存在；最终训练、旧留出评分、新Kaggle上传/运行/提交均为0。详见[第四轮实际结果](reports/WINDOWS_ROUND4_20261025.md)和[预登记](docs/WINDOWS_ROUND4_PLAN_20261025.md)。该轮已关闭，不重跑、不降低门槛或追加参数搜索。

2026-10-24 历史状态：第三轮**五折已完成，提交门槛未通过**。分析第二轮并核对GitHub队友源码后，选择题目残差PCA128 + 收缩高斯头；复用原120 prompt / 2,159行特征，新增编码器前向0次、5/5次fit。pooled BA **0.298677**，比第二轮高0.008773；配对95%区间[-0.006992,+0.024969]跨0、仅3/5折改善，绝对与配对门槛均未达。无最终fit、上传或Kaggle提交，旧留出评分=0。6项新增合成测试及实际OOF/分类头独立复核通过，supervisor退出0。详见[第三轮报告](reports/WINDOWS_ROUND3_20261024.md)及HANDOFF末节。

当前独立分支 `exp/windows-4060-20261023`，工作目录 `D:\kaggle\name_that_llm`。第四轮预登记及代码在运行前推送为`78d529d`，条件阶段准备与结果复核脚本为`660460a`；第三轮预登记为`c85a5a4`，第二轮见[原报告](reports/WINDOWS_ROUND2_20261024.md)。原账号已登录、官方数据已核验，无Mac材料依赖。已核实历史正式Kaggle为本账号0.351、队友0.601，本轮无新公榜分数。第一轮265/264、0fit失败和法证记录保留，四个先前claim哈希均未变。旧291次fit和已用30-prompt留出未重跑；所有已结束入口保持关闭。

**后续维护要求：每轮同步更新仓库根[PROJECT_LEARNING.md](../PROJECT_LEARNING.md)与[PROJECT_LEARNING.html](../PROJECT_LEARNING.html)**，记录假设、预算、实际动作、结果、停止原因和证据，同时面向AI初学者解释本轮知识。Markdown为文字源，根`scripts/render_project_learning.py`生成无外部依赖HTML；保留历史失败，教学虚构例子与真实指标分开。

以下保留2026-10-23交接记录，历史分支：`exp/windows-handoff-20261023`。2026-10-23 已补充 [Windows 交接](docs/WINDOWS_HANDOFF.md)和[下一对话 prompt](../WINDOWS_NEXT_CHAT_PROMPT.md)。用户已手动创建比赛 Notebook 草稿，但是否运行/提交尚未确认，接续先核对历史，不能重复提交。下文“一次提交准备”的未上传表述是原自动化执行记录，不否定用户后来的手动操作。原候选来源分支为 `exp/single-submission-20261021`。用户已明确授权一次Kaggle提交；固定75% original_style + 25% partial候选的[提交计划](docs/SINGLE_SUBMISSION_PLAN.md)和[独立Notebook](notebooks/candidate_blend_offline.ipynb)已准备。本地全量最终训练与公开占位test接口检查通过，尚未上传、运行或提交Kaggle：本轮读取页面仍被保存的域名权限拒绝。下一步须等访问限制解除，再核对当前规则、账号/队伍、额度，使用独立私有Notebook提交至多一次。不得通过其它渠道绕过拒绝，或重复运行已经完成的本地训练。详细命令与证据见[HANDOFF末节](HANDOFF.md#2026-10-21-一次提交准备当前停止点)。

上一轮[有限中心化实验](docs/PROMPT_CENTERING_PLAN.md)与[4060迁移评估](docs/WINDOWS_4060_ASSESSMENT.md)已结束；保留原始开发协议与结果，不据此次最终训练重新选模。环境位于**仓库根 `.venv/`**，Notebook检查依赖为本模块 `requirements-notebook-check.txt`（包含既有CPU依赖与nbformat）；它与旧原始工作区环境隔离。旧30-prompt最终留出已用，不得重新评分。

以下为原上传快照说明，历史日期和来源保持不变。

2026-10-21。仓库：[https://github.com/zdx13143/name-that-llm](https://github.com/zdx13143/name-that-llm)；上传分支：`exp/cpu-stylegram-handoff-20261021`。

本模块是原 `kaggle-titanic` 仓库 `9d293904365f54acdfaa22775c0647d080d1b062` 的 Name That LLM 文件快照。目标主分支基准为 `692ac87d6b7b3e16d0b74bdc85c476d47daac059`。原文件清单与 hash 见 [SOURCE_SNAPSHOT.json](reports/SOURCE_SNAPSHOT.json)，本次不导入旧双赛 Git 历史或 CASMI 代码。

项目目录是仓库根下的 `name-that-llm/`。现有根目录方案照常保留；本模块的固定 120-prompt 开发验证与 30-prompt 一次留出单独报告。先读 [HANDOFF](HANDOFF.md)、[实验结果](reports/CPU_EXPERIMENTS.md)和[每类召回](reports/PER_CLASS_RECALL.md)。

新 clone 不含原始数据、环境、完整 OOF、claim、权重或预测；它不是可自动续跑旧搜索的实验状态副本。最终留出在原实验已用一次，不可因本机缺少 claim 再次打开。原有本机检查点保留在原工作区；本次只上传文件，没有训练或 Kaggle 操作。

从仓库根运行工具测试的可复现命令：

```sh
python -m venv name-that-llm/.venv
name-that-llm/.venv/bin/python -m pip install -r name-that-llm/requirements.txt
name-that-llm/.venv/bin/python -m unittest discover -s name-that-llm/tests -v
```

测试使用临时合成数据，无需下载比赛数据。真实数据需另按 [数据说明](data/README.md)取得。已导出的 [离线 Notebook](notebooks/cpu_offline.ipynb) 内嵌冻结代码；将来使用它会另行全量拟合，应单独记录资源和执行环境，不能冒充本次上传已经在 Kaggle 执行。
