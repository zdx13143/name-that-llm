# Windows 接续交接

更新：2026-10-23，Asia/Singapore。保存分支 `exp/windows-handoff-20261023`，来源 `exp/single-submission-20261021@4760f67fd06f59c602389595caafae090606331d`。此次补充接续文档及该候选的 LF 检出规则，CV 变化为 0；没有训练、Kaggle 操作、系统安装或 Windows 实测。

## 获取代码

私有仓库需要你有访问权限的 GitHub 账号。从 Windows PowerShell 的目标父目录执行：

```powershell
git clone --branch exp/windows-handoff-20261023 --single-branch https://github.com/zdx13143/name-that-llm.git
cd name-that-llm
git status --short --branch
git rev-parse HEAD
git switch -c exp/windows-4060-20261023
```

若已存在 clone，不要覆盖或删目录，先查看未提交改动再 fetch/switch。分支名已存在时选新的唯一后缀。两台机器不要同时推进同一分支，不向 main 推送或自动合并。

打开根目录 [WINDOWS_NEXT_CHAT_PROMPT.md](../../WINDOWS_NEXT_CHAT_PROMPT.md)，将正文交给 Windows Codex。当前默认分支可能没有这份交接，务必显式选择上述交接分支。

## 已保存与未包含的内容

已在 Git 保存源码、未执行的候选 Notebook、工具测试、聚合实验报告、一次提交计划、4060 评估与本交接。Mac 本地原始数据、模型、OOF、claims、.venv、预测、执行 Notebook、CLI 手动包和浏览器故障日志未上传，仍在原机器保留。Git 是代码与报告快照，不是实验运行状态的完整迁移。

手动 CLI 包的可移植核心是原候选 Notebook；其中配置用户名仍为占位值，不能直接复制 Mac 的绝对路径或假定 Windows 已有 Kaggle 凭证。此次不上传本机配置、token 或完整对话记录。

## 当前已知状态

- 冻结候选：[candidate_blend_offline.ipynb](../notebooks/candidate_blend_offline.ipynb)。SHA256 `82931fb6a28441e01b884c2ce62c16ced6f35c39bc9ff092c2f53277308b3df8`。固定 75/25 融合，CPU、Internet Off，无外部权重。
- 历史本地全量最终拟合与公开 3 行占位 test 接口验证已通过；历史工具测试 79 项通过。此次未重跑训练或测试。完整证据见 HANDOFF 的 2026-10-21 一次提交章节。
- 旧 30-prompt 最终留出已使用；新 clone 缺少 claim 不等于重新可用。累计 291 次本模块真实 fit 为历史记录，不含队友或合成测试。
- 用户已手动创建一个绑定 Name That LLM 比赛输入的 Notebook 草稿；是否已导入候选、完成运行或提交，尚无后续证据。没有已确认的候选 Notebook URL、成功版本、submission ref 或成绩。Windows 接续先核对，不能假定从未提交。
- 截至 2026-10-22 最后一次访问检查，原 Mac 工具仍返回 origin approval gate 拒绝发起 Kaggle 网络请求。本地配置 allow 与运行时拒绝冲突未修复；不得将换机器作为绕过方式。按正式权限恢复途径处理，未解除期间只做独立本地工作。

`.gitattributes` 仅为这份冻结候选固定 LF 换行，避免 Windows 的 core.autocrlf 转换导致 SHA256 改变；候选源码内容未改动。

## 第一阶段：仅核对硬件和文件

从仓库根运行，先用现有 Python，不安装训练包、不运行模型：

```powershell
py -3 --version
nvidia-smi
New-Item -ItemType Directory -Force name-that-llm/outputs/windows_4060
py -3 name-that-llm/scripts/probe_gpu_environment.py --output name-that-llm/outputs/windows_4060/preflight_initial.json
Get-FileHash name-that-llm/notebooks/candidate_blend_offline.ipynb -Algorithm SHA256
```

没有 Python launcher 时先检查实际 Python 安装；不要盲换命令到不明环境。探测文件拒绝覆盖，复查使用新文件名。脚本默认只记录当前设备和已安装依赖，不下载模型、不创建训练 tensor；PyTorch 未安装可以先记录为未安装。其输出在 outputs/，不要提交 Git。

数据另行传输到 `name-that-llm/data/raw/`。按 [INITIAL_DATA_AUDIT](../reports/INITIAL_DATA_AUDIT.md) 验证每个原始文件。训练 CSV SHA256 必须是 `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`；否则停下调查来源，不能改候选的 hash 检查让它通过。不要复制或重建私密凭证。后续旧 split 元数据可用已有 restore_team_split.py 恢复，先阅读其参数与输入校验，不执行旧 search/finalize。

## 第二阶段：制定新 GPU 实验

[WINDOWS_4060_ASSESSMENT](WINDOWS_4060_ASSESSMENT.md) 是 2026-10-21 的方案评估，不是目标机实测。原生 Windows 先完成独立 GPU 探测；当前部分 CPU runner 使用 Unix resource、SIGALRM 和 setitimer，复用时优先评估 WSL2，不能承诺全部脚本原生可用。

安装新环境前记录目标 Python/驱动，重新核对 PyTorch 官方安装选择器，创建独立环境，不直接套用 Mac 环境。CUDA smoke 成功才表示 CUDA 小运算可运行，不代表训练、速度或分数已验证。先按既有建议登记 ModernBERT-base 冻结特征的有限方案及完整 revision/输入顺序/缓存契约，供用户 review；不直接启动 8B/14B 全套或按公榜调参。开发只用既有 120 prompt，旧留出始终封存。

## Kaggle 接续

只有在正常权限恢复后，先确认已有草稿与提交历史、账号和队伍、官方规则及剩余额度。尚未使用的一次授权可继续；如用户已经手动提交，不重复消耗次数。候选普通运行可能只生成 3 行占位输出；代码赛需绑定成功 Notebook 版本的 submission.parquet，由平台隐藏测试重跑评分。保存、运行、提交、评分分别记证据。遇到未知提交状态先查历史，不自动重试。

候选对 test 有 4096 行门槛；失败需保留日志后另行评估，不删除执行 claim、不静默修改模型。当前没有证明本候选优于队友历史 0.601 方案，Windows GPU 也不能保证提升分数。
