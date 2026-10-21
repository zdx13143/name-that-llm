# 首次真实数据审计

日期：2026-10-18（Asia/Singapore）。用户从比赛页手动下载 `name-that-llm.zip` 至 Downloads，本地检查 ZIP 成员、CRC 和解压前路径后，将三个原始文件保存到 `data/raw/`。浏览器工具仍被已保存的权限策略拒绝，本次没有绕过该限制。

## 实际文件

| 文件 | 字节数 | 全量行数 | 实际字段 |
|---|---:|---:|---|
| train.csv | 2,139,418 | 2,699 | id, prompt_id, text, model |
| test.csv | 2,218 | 3 | id, text |
| sample_submission.parquet | 899 | 3 | id, model（均为 large_string） |

CSV 字段按原始字符串读取，未改写 ID。三个文件全部读至末尾，字段与此前官方 Data 描述一致；没有多余或缺失字段，没有空值、空白字符串、重复 ID 或 CSV 结构错误。公开 test 实际为 **3 行**，解决本次下载版本的行数问题；此前 Overview 的 1 行描述仍是页面冲突，不据此推断隐藏测试的内容。

## 标签、分组与泄漏检查

- 18 个实际标签与保存的官方标签清单完全一致。17 类各 150 行，`claude-fable-5` 有 149 行。
- 共 150 个 prompt；149 个 prompt 各有完整的 18 类，1 个 prompt 有 17 类。没有重复的 prompt/label 对。
- 训练文本完全重复及 Unicode 空白折叠后的重复均为 0；未发现跨 prompt 的上述重复或标签冲突。
- train/test 的 ID 交集为 0，完全相同及空白归一文本交集均为 0。此检查不涵盖语义近重复或同一模板的改写。
- sample 的 3 个 ID 与 test 完全对应，每个 ID 恰好一行；其示例标签均合法。示例标签是格式参考，不是真实测试答案。

审计退出码 `0`，`audit_status=no_findings`。这表示所列完整性检查未发现问题，不是数据无偏、模型有效或隐藏测试表现的证明。

## 冻结的数据版本

| 文件 | SHA-256 |
|---|---|
| 下载 ZIP（874,800 字节） | `7c375db7dfeea6b36cefc7b48a290dd48c54c7fe03bfd402390728a6a2bb9386` |
| train.csv | `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707` |
| test.csv | `4545aa3a3459f4c0724f97ad5bd06c4caa3506bdfea5fab905e8203184fc0c57` |
| sample_submission.parquet | `ccf86796b80fd7f83d8463664603fd1208e17559abcf416f198776b819646188` |

哈希用于后续复现版本，不构成官方签名或下载链认证。用户提供的 ZIP 文件名、成员、实际字段和规模与此前官方页面证据一致。

## 复现与下一步

从双赛仓库根运行：

```sh
enveda-casmi-2026/.venv/bin/python name-that-llm/scripts/profile_data.py
```

本次使用现有 Python 3.13.2 / PyArrow 26.0.0，仅运行审计。完整聚合输出为 `reports/local_data_profile.json`，保持 Git 忽略。原始文本、行 ID、prompt ID 和文本指纹均不进入分享报告。

用户随后明确要求自主决定并自动推进，已批准此前提出的 CPU 实施及有限优化范围。按 prompt 分组预留最终留出集、在其余数据上固定五折开发验证；词表与 IDF 只在折内训练数据拟合。最终留出只在模型选择冻结后使用一次。模型结果另行报告，不与本审计混称。

来源：[官方 Data](https://www.kaggle.com/competitions/name-that-llm/data)、[官方 Overview](https://www.kaggle.com/competitions/name-that-llm/overview/abstract)为此前已核对的页面；本次结论的直接证据是用户下载的上述三个实际文件及本地全量审计。
