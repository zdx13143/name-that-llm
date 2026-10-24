# 独立 CPU 实验入口

2026-10-24 最新状态：原账号已登录，官方ZIP及三个成员已取得并通过CRC/SHA核验。Windows真实开发**失败并关闭**：265次forward尝试、264次持久完成记录、0次fit，无特征缓存/OOF/新分数。启动器、原子日志替换、恢复claim默认参数路径三个问题已修复，30项相关测试通过；修复后没有重跑真实实验。两条claim现状和重建的首次零计算失败档案均保留，禁止删除、回滚或重新执行本轮入口。详情以 [本轮报告](reports/WINDOWS_ROUND_20261024.md)和HANDOFF末节为准。

当前独立分支 `exp/windows-4060-20261023`，工作目录 `D:\kaggle\name_that_llm`。已有Kaggle正式记录已核实：本账号 `NTL Blend 20261021` Version1为Succeeded、**0.351**；队友 `ntl-ensemble-600` Version1为Succeeded、**0.601**。前者输出provenance与冻结75/25候选一致。本轮没有新增上传/运行/提交；旧291次fit和已用30-prompt留出未重跑。现在不缺官方数据，也不需要Mac文件；缺的是完成的新开发结果。新一轮必须先记录新预算，不复用本轮失败入口。

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
