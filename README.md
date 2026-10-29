# Name That LLM

Kaggle Featured 竞赛 **Name That LLM** 的队伍工作仓库。

> **最新 P0 进展：已准备必需 S7 成员的折隔离源码候选，并核验 30 个训练范围及原采样器的训练行边界；7/7 新人工测试通过。** 执行入口仍关闭，没有训练或新分数。官方固定配置给出的 24 层 FP16 基础权重下界为 **9.784 GiB**，已超过本机 8 GiB 总显存，因此没有强行试跑或下载权重。见[S7 适配及资源结论](docs/R15_P0_S7_ADAPTATION_20261029.md)和[回执](configs/r15_p0e_receipt_20261029.json)。完整成员来源、外部数据排除、校准和执行环境仍待补齐。

> **已完成的数据准入：经用户明确授权，一次性隔离导出并准入原 2,159 行／120 prompts 开发数据。** 原有行、文本、五折哈希通过，独立审查完成，已生成 5 个外折和 25 个内折计划；准入入口最终 9/9 人工测试通过。本次确实读取完整 CSV 一次，在内存中解析旧 30 prompts 的 540 行两遍（1,080 次记录解析事件）；旧留出写出、训练和评估均为 0，之后未再读原 CSV。当前仍缺完整训练来源、外部数据排除与校准证据，没有新增精度结果。见[导出及准入结果](docs/R15_P0_DATA_ADMISSION_20261029.md)和[回执](configs/r15_p0d_receipt_20261029.json)。下列 P0 零读取／待授权说明是本次授权前的历史记录。

> **本实验分支 R15：已实现完整 R14 集成后的单变量排版消融（强度 1.0 → 0.5）、按既定 120 prompts / 5 folds 重放的库和资产校验；15/15 合成测试通过。缺少完整集成 OOF 与逐成员训练排除证据，真实预测评估未开始，没有提分结论。** 见 [R15 报告](docs/R15_REPORT_20261029.md)、[预登记](docs/R15_PLAN_20261029.md)、[资产契约](docs/R15_BUNDLE_CONTRACT_20261029.md)和[执行回执](configs/r15_execution_receipt_20261029.json)。本轮只做本地工作与只读远端核验，0 push、0 导入、0 Save、0 正式提交；R14 冻结候选保留不变。

> **R15 后续查找（2026-10-29）：已查到公开组件 OOF 和 nested validation cache，但没有可准入的完整 bundle。ATLAS 明确为全部 2699 行训练；部分融合参数也需补训练范围证明。** 详见[查找与准入结论](docs/R15_ASSET_SEARCH_20261029.md)、[最小重建方案及资源上限](docs/R15_REBUILD_PLAN_20261029.md)和[本轮查找回执](configs/r15_asset_search_receipt_20261029.json)。固定消融未运行，重建仅是预案。先前执行回执的文件哈希对应其所在提交 `81c64c7`；本次文档更新由新回执单独绑定。

> **R15 优先级 0 准备：已保存四份固定版本训练源码及配置证据，新增 development-only 导出准入与内层五折规划，11/11 人工测试通过。** 旧 loader 会解析完整 CSV，未用于恢复文本；Instyle 外部语料没有按验证 prompt 排除。严格融合方案采用每外折内的五折校准，但完整训练源码、外部谱系、原 2159 行安全导出及精确校准过程仍阻塞。见 [P0 报告及下一步](docs/R15_P0_REPORT_20261029.md)、[源码归档](reference_sources/r15_p0/README.md)、[P0 回执](configs/r15_p0_receipt_20261029.json)。0 GPU、0 官方评分、0 旧留出读取、0 push/Save/提交；原 R15 协议未启用。

> **P0 继续查找：已补取 Nishant part-two 原始配置、ATLAS 配置及源码清单，归档 14 份来源材料；扩展本地、Drive 和公开 Kaggle 元数据检索仍未找到原 2159 行独立导出。** 新下载的 Nishant V2 是 SVC 训练加 LoRA 推理，不是神经训练源码；Claude trainer 的准确链接目前不可访问。已明确 main.deb 与 COMP 冻结 DeBERTa 的区别，并记录 ATLAS 两份 model.py 哈希不一致。最新授权允许后续低负载本机 GPU，但本次未启动，未新增评分。见[继续查找报告](docs/R15_P0_CONTINUATION_20261029.md)、[新增来源](reference_sources/r15_p0b/README.md)和[回执](configs/r15_p0b_receipt_20261029.json)。没有重跑已完成测试。

> **P0 输入兼容修复：发现历史开发标签顺序与 R14 预测列顺序冲突，已保留原哈希并显式映射评分索引；修改后相关测试 28/28 通过。** 登录态 R6/R7/R10 私有资产也未提供独立开发文本。已准备默认关闭的一次性原始 CSV 导出方案，8/8 人工测试通过；它会读取旧留出所在字节，须用户明确调整供数边界后才可执行，当前未读取真实数据、未准入或训练。见[修复与待决操作](docs/R15_P0_INPUT_COMPATIBILITY_20261029.md)及[回执](configs/r15_p0c_receipt_20261029.json)。

> **R14 历史成果：冻结公开实得 0.660 的 V32 / 363222852 完整配方，本地 14/14 合成测试通过，尚无正式分数。** 2026-10-29 本轮只读复核时，已有私有草稿可以显示上游代码，版本历史仍为 `Starting Fresh`，Session off，底部仍有 reCAPTCHA 连接提示；候选未导入，0 个 saved version。页面 T4 ×2 / Internet off 只是设置。我们的已确认最佳仍为 R12 V1 / 363188587 的 **Public Score 0.654**；外部 0.660 不是我们的成绩，也不是本地 CV。旧 30-prompt 留出保持关闭，本轮未新建草稿或自动跟踪。R14 原记录见 [报告](docs/R14_REPORT_20261029.md)、[回执](configs/r14_execution_receipt_20261029.json)与[方案及范围修订](docs/R14_PLAN_20261029.md)。

> R12 于 2026-10-28 19:47:36 UTC 确认正式评测 Succeeded / 0.654，相比队友 0.651 高 0.003，与外部 V4 在页面显示精度上一致。R12 共一次保存、一次正式提交；详见 [R12 报告](docs/R12_REPORT_20261028.md) 与 [回执](configs/r12_execution_receipt_20261028.json)。精确服务器评分时间未公开，所记为确认时间；尚无新的完整集成本地 CV 或 private 成绩。

**目标：银牌以上**（Featured 竞赛银牌 ≈ 队伍排名前 5%）。

> **队友基线记录：公开榜 0.651；原交接记录的名次为第 147 / 3,542 队（前 4.15%），本轮未重新核验实时名次。**
> 完整实验记录见 [EXPERIMENTS.md](EXPERIMENTS.md)。
>
> ⚠️ **成绩构成说明**：0.651 来自**复现社区公开方案**，不是原创模型。
> 上游作者：Dhruv Tibarewal（`nlm-v27`，0.651 的原始配方的作者）、
> avazaku（`verified reproduction`，在免费 T4×2 上验证可复现）、
> 以及更早的 Vladislav Lavrentev（`lightningv08`）、Lucas Boesen、
> Zbigniew Kirchner、kirilltym、nishantdahal、josephadamski91、
> pjmathematician（UltraBERT 权重）、seyitkaangunes（prompt 库）。
>
> 我们自己做的是：**修好 Kaggle 挂载路径变更**（否则这批 0.64+ 笔记本全部跑不起来）、
> 打通默认镜像依赖、在多次失败后判断方向，以及 v1/v2 的原创探索（均被证伪）。
>
> **队友基线成绩曲线**：0.381 → 0.339 → 0.547 → 0.554 → 0.601 → 0.642 → 0.643 → **0.651**；本独立 R12 分支固定复现：**0.654**。

---

## 赛题一句话

给一段 40–250 词的英文段落，判断它由 18 个前沿大模型中的**哪一个**写的。
这是**作者归属（authorship attribution）**问题，不是「AI 文本检测」。

## 关键事实（已核实官方页面与数据）

| 项 | 值 |
|---|---|
| 训练集 | 2,699 条 = 18 模型 × 150 prompt（`claude-fable-5` 少 1 条，为 149） |
| 隐藏测试集 | ≈4,000 条，**prompt 在训练集中从未出现** |
| 字段 | train: `id, prompt_id, text, model`；test: `id, text` |
| 指标 | **Balanced accuracy**（18 类召回率取算术平均） |
| 榜切分 | public ≈ 40%，private ≈ 60%，**private 决定最终排名** |
| 提交 | Code Competition，跑时 ≤9 小时，**断网**，写 `submission.parquet` |
| 外部数据 | 允许公开可用、对所有参赛者平等可得的外部数据与预训练模型 |
| 提交次数 | 每天 5 次，最终可选 2 份计分 |
| 时间线 | 2026-10-14 开赛 → 2027-01-06 报名/合并截止 → 2027-01-13 最终提交（UTC） |

18 个标签：OpenAI 5（gpt-5.5 / 5.6-luna / 5.6-sol / 5.6-terra / 6-astra）、
Google 4（gemini-3.1-pro-preview / 3.6-flash / 3.7-flash / 3.8-flash）、
Anthropic 5（claude-fable-5 / opus-4-6 / opus-4-8 / opus-5 / sonnet-5）、
Qwen 1（qwen3.8-max）、ZAI 2（glm-5.3 / glm-5.3-flash）、Moonshot AI 1（kimi-k3）。

---

## ⚠️ 验证规范（必须遵守）

**测试集的 prompt 全都没在训练集里出现过。** 因此：

1. **本地验证必须按 `prompt_id` 做 GroupKFold**，绝不能用随机 KFold。
   随机切分会让同一个 prompt 的文本同时出现在训练和验证两侧，
   分数会虚高十几个点，上榜就崩。
2. **一切以本地 GroupKFold 分数为准**，不要用 public LB 调参——
   public 只占 40%，且各模型构成与 private 不同。
3. 报告结果时必须同时给出**每类的召回率**。指标是 balanced accuracy，
   均摊到 18 类，只看总体准确率会掩盖短板类。

## 已知的胜负手（来自数据指纹分析）

标点是最强的可解释信号，且能分层：

- **弯双引号 `“` 只出现在 OpenAI 的模型上**（其余 13 个模型几乎为 0）→ 近乎完美的机构检测器
- **em dash `—`**：Anthropic（0.63–0.83）和 ZAI（0.79–0.82）用得极多；
  **Google 四个模型全部 ≤0.01，几乎不用**；OpenAI 内部呈梯度（luna 0.37 > sol 0.27 > terra 0.23 > 5.5 0.13 > astra 0.09）
- **弯撇号 `’`**：Anthropic 与 ZAI **精确为 0**；Google 内部呈版本梯度（3.1-pro 0.06 < 3.6-flash 0.19 < 3.8-flash 0.25 < 3.7-flash 0.31）
- **最难的是 Anthropic + ZAI 这 9 个模型**，四个指标全部挤在一起 → 这是分数的真正瓶颈

跑 `python scripts/01_data_check.py` 可复现上述统计。

---

## 目录结构

```
name-that-llm/
├── data/raw/            # Kaggle 原始数据（不入库）
├── src/                 # 可复用的库代码
│   ├── common.py        # 路径、标签常量、数据读取
│   ├── features.py      # 指纹特征工程
│   ├── cv.py            # prompt 分组 CV + 评估 + 混淆分析
│   └── run_baseline.py  # 基线入口
├── scripts/             # 一次性分析脚本
├── notebooks/           # 探索用 notebook
└── requirements.txt
```

## 环境与运行

```bash
pip install -r requirements.txt

# 拉数据（需要 ~/.kaggle/access_token）
kaggle competitions download -c name-that-llm -p data/raw

python scripts/01_data_check.py     # 数据体检 + 标点指纹统计
python src/run_baseline.py          # 基线 + prompt 分组 CV
```

## 协作约定

- 主分支 `main`；实验性改动开分支，命名 `exp/<主题>`。
- 提交信息写清「改了什么 + CV 分数变化」，例如 `feat: 加入 em dash 特征, CV 0.385 -> 0.41`。
- 任何「提升了分数」的结论都要附上可复现的命令。
- 严禁提交 `.env`、凭据、以及 `data/` 下的原始数据。
