# Name That LLM

Kaggle Featured 竞赛 **Name That LLM** 的队伍工作仓库。

**目标：银牌以上**（Featured 竞赛银牌 ≈ 队伍排名前 5%）。

> **队友2026-10-20实验记录：公开榜0.601，第86 / 2,084队。** 本分支尚未独立核验对应提交或当前排名；此前0.554为历史版本。完整记录见 [EXPERIMENTS.md](EXPERIMENTS.md)，版本证据缺口见[审查说明](name-that-llm/docs/TEAM_WORK_REVIEW.md)。
>
> **成绩构成说明**：记录中的0.554和0.601来自**复现社区公开方案**。
> 上游作者：Lucas Boesen（`lora-x-adaptive-centering`）、
> Vladislav Lavrentev（`lightningv08`）、Zbigniew Kirchner、kirilltym、
> pjmathematician（UltraBERT 权重）、seyitkaangunes（prompt 库）。
> 我们自己做的是：打通默认镜像依赖、修正路径、判断方向、以及下面 v1/v2 的原创探索。

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


## 独立 CPU 实验交接（2026-10-21）

独立分支导入的历史 CPU 实验位于 [name-that-llm/](name-that-llm/README.md)，包含源码、28 项工具测试、空输出离线 Notebook、[逐类召回率](name-that-llm/reports/PER_CLASS_RECALL.md)和[执行命令](name-that-llm/docs/CPU_RESUME.md)。既有根目录方案与实验记录保留。

该模块使用固定 120 个 prompt 作五折分组开发验证，另 30 个 prompt 作一次最终留出；不等同于本仓库全量 150-prompt GroupKFold，不以两者数值大小判断改进。本次仅上传既有成果，CV 变化为 0，未重新训练或提交 Kaggle。新 clone 不含旧输出和 claim，不能据此重新开启已用留出。

## 队友审查后的接续（2026-10-21）

独立分支 `exp/dev-comparison-20261021` 已修复组中心化只作用于首行和当前嵌入重复进入对照池的问题，修复不自动追认历史分数。新有限CPU实验只用原120-prompt开发子集，统一GroupKFold；原模型BA为0.340336，最佳固定融合为0.347281，差值区间跨0且仅3/5折改善，未达到预定采用标准，保留旧冻结交付。

共21次实际拟合尝试（1次概率校验失败、20次有效），未重开最终留出、未使用GPU或提交Kaggle。完整[结果、逐类召回及命令](name-that-llm/reports/TEAM_DEV_COMPARISON.md)、[预定计划](name-that-llm/docs/TEAM_CONTINUATION_PLAN.md)和[队友工作审查](name-that-llm/docs/TEAM_WORK_REVIEW.md)已记录。完成的本机状态不要重跑训练；新clone可用 `restore_team_split.py` 只恢复哈希一致的原划分元数据。
