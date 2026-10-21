# CPU 实验完成与后续入口

记录日期：2026-10-18（Asia/Singapore）。先前暂停已恢复；第三轮、一次最终留出、全量训练及真实离线 Notebook 复现均已完成。当前没有训练进程或未完成 trial。不要按旧暂停记录重启训练。

## 当前冻结状态

- 冻结配置：`b316d0a5c48a`，字符 TF-IDF (2,5) + style 0.25 + LogisticRegression C=16。
- 开发 prompt 分组 BA **0.346818**；相对第二轮 +0.006019，95% CI [-0.002782, 0.014819]。三批未确认提升，按既定工程规则停止；尚不能证明平台期。
- 最终留出已经在冻结后打开**一次**：30 个 prompt /540 行，BA **0.383333**，组 bootstrap 95% CI [0.338889, 0.429630]。不允许重新选模后再把该留出当成未见数据。
- 累计 41 配置 /250 次 CV 拟合，加最终 2 次、Notebook 1 次，共 **253 次真实拟合**。
- 本地离线 Notebook 全部 4 个代码 cell 真实执行完成，公开 test 3 行预测及 decision score 与冻结模型完全一致。未在 Kaggle 执行、未正式提交、无榜单分数。
- 完整聚合证据见 [CPU_EXPERIMENTS.md](../reports/CPU_EXPERIMENTS.md)。

## 本次实际执行命令

以下是已经完成的历史命令，供复核溯源，**不是要求再次执行**：

```sh
# 从原双赛仓库根执行；下方是已完成的历史命令
name-that-llm/.venv/bin/python name-that-llm/scripts/train_cpu_round3.py search
name-that-llm/.venv/bin/python name-that-llm/scripts/finalize_cpu.py --output name-that-llm/outputs/cpu_round3
name-that-llm/.venv/bin/python name-that-llm/scripts/build_cpu_notebook.py --run-dir name-that-llm/outputs/cpu_round3
name-that-llm/.venv/bin/python name-that-llm/scripts/verify_cpu_notebook.py --run-dir name-that-llm/outputs/cpu_round3
name-that-llm/.venv/bin/python -m unittest discover -s name-that-llm/tests -v
```

第三轮搜索 302.1 秒、45 次主 CV +10 次新增稳定性拟合；第二轮 incumbent 的 10 次稳定性拟合按冻结 OOF 复用。最终留出拟合 8.7 秒、全量交付拟合 11.2 秒。Notebook 使用同一 `.venv`，顺序编译执行导出 cell，另全量训练一次约 10.9 秒；没有通过 Jupyter 内核或 Kaggle 执行。28 项测试涵盖有意义的失败路径，临时合成拟合不计真实数据实验预算。

## 后续优先只读核验

```sh
# 从原双赛仓库根执行；下方是已完成的历史命令
git status --short --branch
name-that-llm/.venv/bin/python -c 'import json; from pathlib import Path; p=Path("name-that-llm/outputs/cpu_round3"); print(json.loads((p/"frozen_selection.json").read_text())["selected_id"]); print(json.loads((p/"holdout_metrics.json").read_text())["balanced_accuracy"]); print(json.loads(Path("name-that-llm/outputs/notebook_reproduction/reproduction.json").read_text())["predictions_match"])'
```

本地证据（均受忽略）：

- `outputs/cpu_round1/`、`cpu_round2/`、`cpu_round3/`：原始 summary、配置、OOF、split、耗时、源码快照。
- `outputs/cpu_round3/artifact_contract.json`、`historical_registry.json`：恢复后追溯绑定和第三轮开跑前历史文件 SHA。
- `outputs/holdout_claims/`：固定项目级、按 train SHA 命名的一次性留出 claim；本次已 completed。旧轮也已写 marker，不能换输出目录绕过。
- `outputs/cpu_round3/frozen_selection.json`、`selected_config.json`、`holdout_metrics.json`、`holdout_predictions.npz`、`model.joblib`、`submission.parquet`。
- `outputs/notebook_reproduction/runtime.json`、`reproduction.json`、`executed_cpu_offline.ipynb`、`submission.parquet`。
- `outputs/PAUSED.json` 为先前暂停的历史记录，已标记被本次完成状态取代；`outputs/COMPLETED.json` 指向最终证据。

第三轮运行源码以其 `code_snapshot/` 与 summary SHA 为准。基准提交 `e434209` 不包含当时所有未提交代码；不要仅用 commit 推断运行版本。

## 复现和继续边界

交付 [cpu_offline.ipynb](../notebooks/cpu_offline.ipynb) 保持空输出。它内嵌冻结模型代码，使用真实 `test.csv` 行数，导出合法 `id,model` Parquet。生成器及复现工具会核对配置/summary/标签/模型源代码绑定，训练前核对训练文件 SHA；runtime 记录完整版本和额外拟合次数。

本地已验证环境：Python 3.13.2、scikit-learn 1.9.1、numpy 2.5.4、scipy 1.18.1、pyarrow 26.0.0、joblib 1.6.0。以后如果运行 Kaggle Notebook，先核对实际挂载路径、预装包与规则；免费 CPU 即可运行这个模型。版本不同要作为单独复现记录，不能沿用本地精确等价结论。GPU 配额与设备仍未检查，不应自动启动。

无需重跑已完成 search/finalize。统一入口会拒绝再次打开该训练数据的留出；不要删除 claim 或更换目录规避。如果将来继续新模型方向，需要先定义新的评估方案并明确现有留出已见，不能将本次留出继续用作独立选模依据。

正式比赛提交仍需用户明确授权。Git 上传、Kaggle Notebook 编辑和比赛提交是不同动作，不会自动同步；不要改队友分支、合并或上传任何本地数据与权重。
