# Windows ModernBERT 冻结特征预登记

日期：2026-10-23，Asia/Singapore。基点 `25c9a02ccd18d36391e31e15ffda268ab735dd8e`；独立分支 `exp/windows-4060-20261023`。本计划先于模型下载、前向和真实拟合记录。机器与材料的实际结果另见 [Windows 检查报告](../reports/WINDOWS_4060_PREFLIGHT.md)。

2026-10-24 执行修订：用户要求只依托 GitHub 仓库推进。新增 [独立开发协议](WINDOWS_STANDALONE_DEV.md)和原生Windows入口，不再以Mac文件/历史OOF为执行前提；只需官方train.csv即可恢复并核验原分区。沿用同一experiment_id及原五次fit总预算，本轮只评估单模型，暂停下文的配对融合与候选采用部分。其余模型、保护集和预算约束不变。下文保留2026-10-23原始预登记，不代表当前入口仍未实现。

## 目标与当前边界

先回答原生 Windows RTX 4060 是否能运行固定 ModernBERT-base 冻结特征，再在材料齐全后判断其对当前 75/25 开发候选的互补。所有数值是本地开发证据；不以队友历史 0.601、公榜或旧留出调参。旧 30-prompt 最终留出已使用，禁止重开，原 291 次真实拟合不重复。

本轮先完成独立环境、CUDA 小矩阵和 4 条固定合成英文文本的前向检查，不读取比赛文本、不训练分类器。真实开发入口尚未实现，不能把此计划或 smoke 脚本当成可执行的完整训练管线。数据与历史 OOF 缺失时停在工程准备，不能为了补缓存重训基线。

用户已要求常规可逆工作自主推进，后续 Kaggle 提交也由代理接手；历史手动操作是否已消耗提交次数仍未知。原访问限制未正常解除前不访问 Kaggle；正常恢复后先检查版本与提交历史，再按授权执行，不能重复提交状态未知的候选。

## 固定模型与环境

机器实测为 RTX 4060 Laptop GPU / 8,188 MiB / 驱动 596.36；Python 3.11.6。使用模块内 `.venv-gpu`，不复制 Mac 环境、不改变系统 Python 或已有 WSL。原生 Windows 只承担独立探测和冻结前向；旧 POSIX CPU runner 不能直接导入。后续若复用其评估代码，先评估已有 Ubuntu WSL2，或将必要的数据契约隔离为可移植模块，不伪造 `resource`/定时器来通过导入。

固定配置：[windows_modernbert_20261023.json](../config/windows_modernbert_20261023.json)。

| 项 | 预先固定值 |
| --- | --- |
| 来源 | 官方 `https://huggingface.co/answerdotai/ModernBERT-base`；不使用根代码默认镜像 |
| 模型及 tokenizer revision | `8949b909ec900327062f0ebf497f51aef5e6f0c8` |
| 权重 | `model.safetensors`，598,635,032 字节；SHA256 `340ac08b74eef0d7bdec2d7981a6a3d4249bf0e6aab60634b72ad02c2b8023a9` |
| 推理 | `AutoModel`，eval + requires_grad false + inference_mode；无 MLM head 训练 |
| 层/池化 | 最后 4 层逐层 attention-mask mean（包含特殊 token、排除 padding），拼接 3,072 维 |
| 文本 | tokenizer 默认规范化，不另清洗；max_length=256，右侧截断，padding=max_length |
| 批量/精度 | batch=4；BF16 前向、FP32 池化和缓存；SDPA，reference_compile=false |
| 环境 | Python 3.11.6；torch 2.8.0 CUDA 12.8 wheel；Transformers 4.56.2；其余直接依赖见 requirements-windows-gpu.txt |

PyTorch 版本是此次选择的固定可复现组合，不称最新版本。使用官方 CUDA wheel；不安装独立 CUDA toolkit、驱动、FlashAttention、bitsandbytes 或大模型。保存完整实际 `pip freeze`；依赖安装失败不默默换模型/镜像。模型仅下载 README、config、tokenizer 与 safetensors；下载后逐文件核验上游 blob/LFS 身份并记录 SHA256。加载时 `local_files_only=True`、`trust_remote_code=False`。

## 合成前向与资源停止条件

1. 默认 stdlib 探测已完成后，在新环境执行现有 `probe_gpu_environment.py --cuda-smoke`。通过才下载模型和执行合成前向。
2. 合成输入固定 4 条：短句、标点差异句、较长句和重复长句；其中最后一条超过 256 token 用于验证截断。没有比赛 id、prompt 或标签。
3. 最多两次模型 forward：原顺序一次、反序一次。恢复顺序后验证 shape、有限值、mask 池化与非空 token；比较 batch 反序结果（FP32 embedding 最大绝对差不超过 0.02）。padding 在池化中必须无贡献。
4. 单次 smoke 子进程墙钟上限 300 秒；CUDA 分配峰值超过 6 GiB、非有限值、维度错误或输出不一致则失败停止。不得自动改短序列、改精度、换层/模型、清除失败记录或无限重试。
5. 工程下载总预算 6 GiB，新环境和缓存 12 GiB；失败保留日志。下载不超过 15 分钟，单次安装不超过 15 分钟。不需要比赛材料即可完成的检查继续执行。

合成前向通过只证明该配置在这台机器上可以运行，不能估计完整开发速度、训练容量、分数或 Kaggle 兼容性。

## 后续真实开发协议（材料齐全后实施）

- 只使用原 120 prompt / 2,159 行。原始 CSV 和 split SHA 必须匹配固定配置。读取 CSV 可为 hash/元数据扫描，但 30-prompt 文本/标签不得进入特征缓存、训练或评分。
- split 内的历史 `development_fold` 是早期 seed 自定义分折，不能作为本次 fold。按固定开发行序重建 `GroupKFold(5, shuffle=False)`，与历史 original_style/partial OOF 的 fold_id 完全相同才继续。
- 取回 `original_style_oof.npz` 和 partial batch OOF，核验文件 SHA、规范 contract SHA、row_index/id/group/truth/label 顺序、fold、有限且归一化的概率。NPZ 用 `allow_pickle=False`。缺失时不重训已有模型，不把同 shape 缓存当相同身份。
- 特征提取最多一个完整开发 pass / 540 个 batch。记录输入文本 SHA、行序 SHA、train/split/labels SHA、model/tokenizer 完整 revision、所有下载文件 SHA、层/池化/长度/dtype/实现和依赖版本。缓存复用必须完整契约一致。
- 标准化与 LR 只在各训练 fold 拟合；固定 `C=0.1, solver=lbfgs, class_weight=balanced, max_iter=5000, random_state=0`。五折最多 **5 次 fit 尝试**，每次尝试前持久写计数；失败也消耗预算。不做 epoch、学习率、C、长度、层或融合权重搜索。
- 主要对照为既有 `0.75 original_style + 0.25 partial` OOF；主要新候选固定为 `0.75 对照 + 0.25 ModernBERT`。独立 ModernBERT 与 original_style 比较仅作描述，不从多个结果挑最好来替代主假设。
- 报告 pooled BA、每折 BA、18 类 recall/confusion；按 prompt 配对 bootstrap 2,000 次、seed=20261023。保留候选须同时满足 BA 增益≥0.005、95% 配对区间下界>0、至少 4/5 折改善（差>1e-12）、每类 recall 下降≤0.10；阈值不在看结果后更改。该区间是有限开发比较的描述性不确定性，不是新独立留出或榜单收益保证。
- 端到端开发墙钟最多 1,800 秒，2 个数值库线程、主机 RSS 8 GiB、CUDA allocated 6 GiB；以外部进程 deadline 实施 Windows 超时。缺 input、契约不一致、OOM、非有限值、收敛警告、超时/预算都停止并保留证据，不改阈值补跑。
- 新实验在固定 experiment_id 下独占 claim，失败/完成均禁止换输出目录重置。现有 claim 与历史完成记录保留；`final_holdout_evaluations=0`、全量最终 fit=0、测试预测=0。

## 材料移交

当前 Git checkout 无数据、split、OOF、claims、权重；远端材料核查详见检查报告。用户未提供单独传输位置，不能假定有原 Mac 文件。原始数据可从原机器经用户授权复制到 `name-that-llm/data/raw/`；或在正常 Kaggle 权限恢复后由获准渠道取得。旧实验状态仍须从原机器恢复。

| 材料 | 固定 SHA256 / 用途 |
| --- | --- |
| train.csv | `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707` |
| test.csv | `4545aa3a3459f4c0724f97ad5bd06c4caa3506bdfea5fab905e8203184fc0c57`；非本次开发输入 |
| sample_submission.parquet | `ccf86796b80fd7f83d8463664603fd1208e17559abcf416f198776b819646188`；非本次开发输入 |
| split_manifest.json | `c8c24921f617ad483b9e14cb23d0d64a0fbbde44c0214868445e8d0e30d6e1db`；仅此文件可用现有 restore_team_split.py 从 hash 正确的 train 恢复 |
| original_style_oof.npz | `40cf86da51cfaccc2ad6af396b62e1363c72ab27a871a3a1dca60c9be5feed75`；原 outputs/team_dev_20261021/recovery_01/ |
| partial batch OOF | `2317f0f02dbcfa24d9fb183e794076470c182a821ad7c31ac719a066aa562aac`；原 outputs/prompt_centering_20261021/，以内容和 contract 识别 |
| team contract | 规范 JSON hash `4f8c614736e544884a3ccc362b015a4cd8356f375f93880e2b00dff08b466250` |
| centering contract | 规范 JSON hash `0cd3a6605134525cb428b78a4ead0d00c0c8007234a43c50ed503e05ff322e15` |
| 已用 holdout claim | `5a93d4efc1590e6acc63bdd3d40b3f3ebf860aa85e04d6af5bda582638c68db0`；只作状态证据，不重新评估 |
| 历史开发 claims | outputs/continuation_claims/ 中两个 completed claim；保留失败记录和来源，不伪造缺失文件 |

只传本比赛所需材料，不传凭证、Mac .venv 或其它项目。原 291 次 fit 历史保持封存。禁止运行旧 search/finalize/verify_candidate_notebook 来补缓存。

## 官方来源（2026-10-23 实际读取）

- [ModernBERT 模型卡](https://huggingface.co/answerdotai/ModernBERT-base)与[固定 revision API](https://huggingface.co/api/models/answerdotai/ModernBERT-base/revision/8949b909ec900327062f0ebf497f51aef5e6f0c8?blobs=true)：模型身份、许可及权重元数据。
- [Transformers 4.56.2 ModernBERT 文档](https://huggingface.co/docs/transformers/v4.56.2/en/model_doc/modernbert)：SDPA 与 reference_compile 配置。
- [PyTorch 官方历史安装组合](https://pytorch.org/get-started/previous-versions/#v280)：Windows CUDA 12.8 / torch 2.8.0 wheel。本轮没有修改驱动或系统 CUDA。
