# Windows 接续 prompt

把下面正文复制到 Windows 上的 Codex 对话；先将该仓库分支克隆到 Windows，并把 Codex 工作目录指向仓库根。

---

请继续推进 Name That LLM，仅负责这个比赛，不操作 CASMI。

仓库：https://github.com/zdx13143/name-that-llm
交接分支：exp/windows-handoff-20261023
先读 README.md、name-that-llm/AGENTS.md、WORKSPACE.md、COLLABORATION.md、HANDOFF.md，以及 name-that-llm/docs/WINDOWS_HANDOFF.md。其中后四个文件都在 name-that-llm/ 子目录。先核对实际分支和 Git SHA，再创建自己的 exp/windows-4060-<日期或唯一后缀> 分支；不要修改 main、队友分支或强推。

本轮首先做接续审计与 RTX 4060 环境探测，不自动启动大模型训练。核验 GPU 型号、实际显存、驱动、Python、系统 RAM/磁盘及 CUDA 可用性，运行已有 probe_gpu_environment.py。原生 Windows 不可盲跑依赖 resource/SIGALRM 的旧 CPU runner；需要复用时评估 WSL2。不要复制 Mac .venv。

已完成的 CPU 搜索和最终训练不要重复。旧 30-prompt 最终留出已使用并封存，新 clone 缺少 claim 也不能重新评估它。累计本模块 291 次真实拟合为历史值。下一阶段正式开发仍限既定 120 prompt / 2159 行并按 prompt 分组；新模型先记录配置、来源、预算和停止规则供用户 review，再执行。可先参考已有 WINDOWS_4060_ASSESSMENT.md 中的 ModernBERT-base 冻结特征方案，但 Windows/CUDA 和分数收益尚未实测。

冻结提交候选为 name-that-llm/notebooks/candidate_blend_offline.ipynb，SHA256：
82931fb6a28441e01b884c2ce62c16ced6f35c39bc9ff092c2f53277308b3df8
它是固定 75% original_style + 25% partial 的 CPU 离线 Notebook，本地接口验证已完成，公开 test 仅 3 行。不要把本地 3 行 Parquet 当正式预测；不要将本地验证分数称为 Kaggle 成绩。候选 test 大于 4096 行会停止，不能静默改阈值。

Kaggle 状态：原对话没有自动上传、运行或正式提交成功记录；用户后来手动创建了带比赛输入的新 Notebook 草稿，并收到导入/Save & Run All/提交指导，但尚无后续版本、submission ref 或评分证据。不要断言用户尚未提交。先核对用户已有页面/记录，避免重复上传与提交。原有至多一次正式提交授权仍需结合实际历史判断是否已经使用。

原 Mac 的浏览器工具存在 unresolved origin-policy 拒绝，即使截图和本地配置 allow；不得换电脑、CLI、浏览器或对话绕过这一拒绝。Windows 可以继续本地准备、数据审计和环境检查；只有有效访问限制通过正式途径解除后，才能正常核查 Kaggle 账号、队伍、当前规则、配额与历史，并继续尚未完成的一次提交。不自动发送消息给队友。

Git 中不含数据、模型、OOF、claims、凭证或环境。先盘点缺少的本地材料；数据只从用户授权的单独传输或正常获准的官方渠道取得。不要因为缺少旧输出而重跑完成的实验或重开留出。参考 INITIAL_DATA_AUDIT.md 核对原始文件 hash，并只恢复已有 split 元数据。

每一步区分本机检查、训练、Kaggle 普通运行、正式提交和评分。有依赖/权限阻塞时保留准确证据；不将 Windows 可运行或 Git push 当作分数提升。修改完成后更新本模块 HANDOFF，只暂存明确路径，保持数据和本地输出不进 Git。
