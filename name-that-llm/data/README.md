# 数据目录

2026-10-18 用户手动下载比赛 ZIP 后，已解压至 `data/raw/` 并完成全部三个文件审计。训练 2,699 行、150 个 prompt、18 类；公开测试与示例提交各 3 行。文件 SHA-256、范围与发现见 [真实数据审计](../reports/INITIAL_DATA_AUDIT.md)。以下同时保留此前页面描述供核对。

## 页面描述的文件

| 文件 | 页面描述的字段 | 用途 |
| --- | --- | --- |
| `train.csv` | `id`, `prompt_id`, `text`, `model` | 有标签训练集；页面称 2,699 条文本、150 个 prompt。 |
| `test.csv` | `id`, `text` | 公开交互样例；评分时替换为隐藏测试，隐藏 prompt 不在训练集内。 |
| `sample_submission.parquet` | `id`, `model` | 提交格式示例。 |

来源：[官方 Data](https://www.kaggle.com/competitions/name-that-llm/data)。Data 页写公开测试有 3 行，Overview 写 1 行，两者不一致；本次实际文件为 3 行。代码按读取后的完整测试表处理，不固定行数。

评分时 Notebook 读取 `/kaggle/input/competitions/name-that-llm/test.csv`，写入 `/kaggle/working/submission.parquet`。隐藏测试约 4,000 条，每个 ID 应有一个合法模型标签。来源：[官方 Overview 与标签清单](https://www.kaggle.com/competitions/name-that-llm/overview/abstract)。

## 本地存放约定

- `data/raw/`：官方原始下载与解压文件，保留原文件名，不直接覆盖修改。
- `data/processed/`：后续生成的特征、分组索引或中间数据，需能从原始数据重建。
- 原始文件、处理后数据和样本文本不上传 GitHub；仓库仅保留说明、处理代码与不包含原文的汇总结果。

`data/raw/` 已建立并受 Git 忽略；原 ZIP 保留在用户 Downloads，不用虚构样例补齐正式文件。

## 首次审计清单

已实现 [CPU 审计工具](../scripts/profile_data.py)；从项目目录运行 `python scripts/profile_data.py`。8 项合成行为测试已经通过，真实文件全量审计也已成功，退出码为 0。完整聚合输出 `reports/local_data_profile.json` 保持本地。

- 记录来源、下载日期、文件大小与 SHA-256；验证 CSV/Parquet 可完整读取。
- 检查真实字段、类型、行数、空值、ID 唯一性和标签是否均属官方 18 类。
- 统计训练集各类数量、`prompt_id` 数量，以及每个 prompt 的类别覆盖。
- 检查文本完全重复与跨 prompt 重复；必要时调整分组，避免相同文本同时进入训练与验证。
- 按 `prompt_id` 分组验证；验证集与训练集的 prompt 交集必须为空，并检查每折类别覆盖。
- 检查测试字段与训练字段的差异；不要求测试存在 `prompt_id` 或 `model`。
- 校验提交 ID 与读入测试 ID 完全一致、无重复无空值，模型标签合法。

公开测试只能用于验证读取和输出流程，不能用于推断隐藏测试分布或验证模型效果。
