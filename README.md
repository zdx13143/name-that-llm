# Name That LLM

Kaggle Featured 竞赛 **Name That LLM** 的队伍工作仓库。

> **最新范围：用户确认本轮只推进本地改进与验证。R14 成果保留在本地独立分支，未推送；远端导入、保存和提交暂缓。** 下述 Kaggle 草稿已在该确认到达前创建，保存与正式提交均为 0；原先的一次保存/一次提交方案仅作后续参考。

> **本实验分支 R14：已冻结公开实得 0.660 的 V32 / 363222852 完整配方，本地 14/14 合成测试通过。当前仅创建一个私有 Kaggle 草稿，编辑器持续 `Editor Loading` 并报告无法连接 reCAPTCHA 服务，候选尚未导入；0 次保存运行、0 次正式提交、暂无本轮分数。** 页面设置为 T4 ×2、Internet off，仍待实际保存运行核验。我们的已确认最佳仍为 R12 V1 / 363188587 的 **Public Score 0.654**；外部 0.660 不是我们的成绩，也不是本地 CV。见 [R14 报告](docs/R14_REPORT_20261029.md)、[执行回执](configs/r14_execution_receipt_20261029.json)、[预登记方案](docs/R14_PLAN_20261029.md)。旧 30-prompt 留出保持关闭，R12 自动跟踪保持暂停，R14 尚未建立自动跟踪。

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
