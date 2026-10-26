# Name That LLM

> **2026-10-26 一次提交请求的最新状态：** 用户明确要求观察分数后，已对原权重4候选完成最终拟合与Linux四人工文本核验，私有资源上传成功。Kaggle草稿停在`Editor Loading`，尚未导入/保存运行/正式提交，暂无R7公榜分数。原开发gate仍为false，旧30prompt未训练或评分。见[提交记录与恢复入口](reports/ROUND7_REQUESTED_SUBMISSION_20261026.md)。下段为此请求前的开发阶段结论。

> **2026-10-26 最新结果：** 第七轮外部语料四视角SVC，权重1对照BA **0.445483**、权重4候选 **0.445938**。候选相对R6开发集提高7.5043个百分点，但对本轮对照仅+0.0455个百分点、95%区间跨0、3/5折改善，增量gate失败关闭。10次分类器+20次词表fit完成；最终训练、旧留出评分、新Kaggle上传/运行/提交均为0。完整0.601仅作为源码参考，不能把组件CV当成强集成或公榜提升。见[第七轮报告](reports/WINDOWS_ROUND7_20261026.md)、[预登记](docs/WINDOWS_ROUND7_PLAN_20261026.md)、[学习Markdown](../PROJECT_LEARNING.md)/[HTML](../PROJECT_LEARNING.html)。

> **2026-10-25 历史结果：** 第六轮嵌套温度校准固定80/20融合**BA 0.370896，开发门槛全部通过**。比第五轮风格对照高1.1570个百分点、五折均改善。开发30次模型fit+10次温度fit，条件最终8次模型fit+2次温度fit完成；四次跨平台人工检查和Linux迁移核验通过。私有Kaggle资产已上传，Version 1保存运行成功（96.5秒）；一次正式提交已成功，Kaggle Public Score 为0.365。旧留出未重开。详见[第六轮报告](reports/WINDOWS_ROUND6_20261025.md)、[当前入口](WORKSPACE.md)及根[学习Markdown](../PROJECT_LEARNING.md)/[离线HTML](../PROJECT_LEARNING.html)。第五轮0.323681、第四轮0.264858及以下日期更早的状态均保留为历史记录。

> **2026-10-21 接续结果：** 已在 `exp/dev-comparison-20261021` 分支审查队友方案、修复两处bug，并完成统一开发GroupKFold比较。原模型0.340336，最佳固定融合0.347281，增益区间跨0，未达到采用标准；原冻结交付保持不变。见[本轮报告与逐类召回](reports/TEAM_DEV_COMPARISON.md)及[队友审查](docs/TEAM_WORK_REVIEW.md)。这是新开发探索，不是新的最终留出或榜单成绩。新环境为仓库根 `.venv/`，使用 `requirements-team-cpu.txt`。

> 2026-10-21 上传说明：这是从旧双赛仓库 `9d293904365f54acdfaa22775c0647d080d1b062` 导入的独立 CPU 实验模块，保留 `name-that-llm/` 子目录，不替换仓库根部已有的 `src/`、`scripts/` 或模型方案。上传只改变文档与聚合交接，没有新增模型拟合。
> 请先读 [协作与运行入口](WORKSPACE.md)、[逐类召回率](reports/PER_CLASS_RECALL.md)及 [迁移说明](HANDOFF.md#2026-10-21-独立仓库上传)。本模块的 120-prompt 开发验证不等同于队友的全量 150-prompt GroupKFold，分数不能直接比较优劣。旧最终留出已使用一次；新 clone 没有本机 claim 并不代表它未使用。

从短文本判断生成它的语言模型，共 18 类；评价指标为 balanced accuracy，即各类召回率的平均值。

## 2026-10-18 历史 CPU 状态

以下审计、环境与原推进顺序保留为历史快照；当前实验分支、Kaggle结果与已关闭入口以顶部更新及WORKSPACE为准。

截至 2026-10-18，用户手动下载的官方文件已全量审计：2,699 条训练、150 个 prompt、18 类、3 条公开测试，所列完整性检查未发现问题。见 [真实数据审计](reports/INITIAL_DATA_AUDIT.md)。本模块记录独立 CPU 实验；三轮 CPU 搜索已完成，开发 BA 为 **0.346818**，冻结后一次最终留出 BA 为 **0.383333**（按 prompt bootstrap 95% 区间 **[0.338889, 0.429630]**）。这些都是本地结果，没有 Kaggle 运行或比赛提交。当前入口见 [HANDOFF.md](HANDOFF.md)。

最近三个既定优化方向未确认稳定提升，按工程规则结束本轮搜索；宽区间不足以证明全局或已尝试范围的收益平台。三轮 41 个配置 / 250 次 CV 拟合，最终评估、全量交付和 Notebook 复现另加 3 次，共 253 次。最终留出已使用，后续不可重开调参。详见 [实验报告](reports/CPU_EXPERIMENTS.md)。

[离线 CPU Notebook](notebooks/cpu_offline.ipynb) 已在本机顺序执行全部 4 个代码单元，重训完整训练集后，公开测试 ID、标签预测和 decision scores 与最终模型完全一致。交付 Notebook 清空运行输出；本地模型、预测与执行收据受 Git 忽略。Kaggle 运行环境、免费 GPU 配额和完整规则仍待访问条件允许后核对。

| 项目 | 位置或约定 |
| --- | --- |
| 比赛 | [Kaggle Name That LLM](https://www.kaggle.com/competitions/name-that-llm) |
| 比赛标识 | `name-that-llm` |
| 本项目目录 | `name-that-llm/` |
| 协作仓库 | [zdx13143/name-that-llm](https://github.com/zdx13143/name-that-llm)（私有，需仓库访问权限） |
| 独立开发分支 | 当前 `exp/windows-4060-20261023`；CPU历史 `exp/dev-comparison-20261021`，原上传 `exp/cpu-stylegram-handoff-20261021`（源于 `main@692ac87`） |
| 数据说明 | [data/README.md](data/README.md) |
| 下一阶段方案 | [docs/PLAN.md](docs/PLAN.md)、[整轮 CPU 优化及停止标准](docs/OPTIMIZATION_PLAN.md) |
| 交接与待核对项 | [HANDOFF.md](HANDOFF.md) |

仅在独立分支推进，经队友审阅后再考虑合并。原始数据、密钥、环境、模型权重和运行输出不随代码上传；本文件不代表远端已同步或 Kaggle 团队已合并。

## 运行数据审计（CPU）

将三个官方文件保存到本项目 `data/raw/` 后，从项目目录执行：

```sh
python -m pip install -r requirements.txt
python scripts/profile_data.py
python -m unittest discover -s tests -v
```

`python` 应指向你的项目环境。本机数据取得时的审计使用 CASMI 环境；现已建立本项目独立 `.venv`，Python 3.13.2 / pyarrow 26.0.0，分类依赖见 `requirements.txt`。尚未验证新的 Windows 环境。工具也支持显式 `--data-dir` 和 `--output`，可供 CPU Notebook 调用。

默认输出 `reports/local_data_profile.json`，受 Git 忽略。报告仅包含聚合统计，不包含原文、ID 或 prompt ID。退出码 `0` 表示完整审计且没有错误级发现（仍可能有需检查项），`1` 表示审计完成但有完整性错误，`2` 表示文件缺失/不可完整读取或配置无效。合成测试通过不代表真实竞赛数据已经审计。

## 已核对的比赛约束

- 必须通过 Kaggle Notebook 提交；CPU/GPU 提交运行均不超过 9 小时，互联网关闭。
- 输入测试路径为 `/kaggle/input/competitions/name-that-llm/test.csv`；输出必须为 `/kaggle/working/submission.parquet`，包含 `id`、`model` 两列，每个测试 ID 恰好一次。
- `model` 必须属于官方 18 个标签，不能有空值。隐藏测试约 4,000 条；公开测试样例只能检查流程。
- 官方 Overview 允许免费公开的外部数据和预训练模型；具体资源使用前仍需核对完整 Rules 和资源许可。

来源：[官方 Overview](https://www.kaggle.com/competitions/name-that-llm/overview/abstract)、[Rules](https://www.kaggle.com/competitions/name-that-llm/rules)。完整 Rules 本轮尚未完成核对，不填写未经确认的每日提交或队伍人数上限。

## 时间线存在冲突

Overview 时间线写：报名/合队截止 `2027-01-06 23:59 UTC`（新加坡 `2027-01-07 07:59`），最终提交截止 `2027-01-13 23:59 UTC`（新加坡 `2027-01-14 07:59`）。但页头 `Close` 提示显示新加坡 `2027-01-15 07:59`，相差一天。需等待官方确认；内部按较早日期安排，不将其写成已解决的规则。

## 推进顺序

真实数据审计已经完成。按 `prompt_id` 分组冻结开发验证与一次性最终留出，自主执行已提出的 CPU TF-IDF + 线性分类及有限优化方案。免费 Kaggle GPU 作为后续选项；当前尚未核对账号实际额度，不启动 GPU 会话。

官方说明 GPU 每周通常为 30 小时，可能按资源情况浮动；实际额度和硬件以账户页面为准。[GPU 使用指南](https://www.kaggle.com/docs/efficient-gpu-usage)、[Notebook 文档](https://www.kaggle.com/docs/notebooks)。
