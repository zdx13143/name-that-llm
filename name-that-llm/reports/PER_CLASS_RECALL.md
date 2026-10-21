# 冻结 CPU 模型的每类召回率

整理日期：2026-10-21。以下指标只从 2026-10-18 已保存的开发 OOF 和最终留出预测重新汇总，没有训练模型、生成新预测或再次执行最终留出评估。冻结配置为 `b316d0a5c48a`：字符 TF-IDF `(2,5)` + 文体统计权重 `0.25` + LogisticRegression `C=16`。

开发 OOF 覆盖 **120 个 prompt / 2,159 行**，balanced accuracy 为 **0.346818**；一次最终留出覆盖 **30 个 prompt / 540 行**，balanced accuracy 为 **0.383333**。Balanced accuracy 是下表 18 类召回率的算术平均，使用未四舍五入的值计算。最终留出已经使用，不能再次用于调参或作为新的独立验证集。

| 模型标签 | 开发 OOF 正确/样本 | 开发召回率 | 最终留出正确/样本 | 最终留出召回率 |
|---|---:|---:|---:|---:|
| `gpt-5.5` | 46/120 | 0.383333 | 14/30 | 0.466667 |
| `gpt-5.6-luna` | 26/120 | 0.216667 | 13/30 | 0.433333 |
| `gpt-5.6-sol` | 18/120 | 0.150000 | 5/30 | 0.166667 |
| `gpt-5.6-terra` | 21/120 | 0.175000 | 8/30 | 0.266667 |
| `gpt-6-astra` | 69/120 | 0.575000 | 16/30 | 0.533333 |
| `gemini-3.1-pro-preview` | 84/120 | 0.700000 | 22/30 | 0.733333 |
| `gemini-3.6-flash` | 42/120 | 0.350000 | 12/30 | 0.400000 |
| `gemini-3.7-flash` | 50/120 | 0.416667 | 16/30 | 0.533333 |
| `gemini-3.8-flash` | 43/120 | 0.358333 | 6/30 | 0.200000 |
| `claude-fable-5` | 15/119 | 0.126050 | 7/30 | 0.233333 |
| `claude-opus-4-6` | 70/120 | 0.583333 | 18/30 | 0.600000 |
| `claude-opus-4-8` | 25/120 | 0.208333 | 6/30 | 0.200000 |
| `claude-opus-5` | 67/120 | 0.558333 | 21/30 | 0.700000 |
| `claude-sonnet-5` | 45/120 | 0.375000 | 10/30 | 0.333333 |
| `qwen3.8-max` | 52/120 | 0.433333 | 11/30 | 0.366667 |
| `glm-5.3` | 23/120 | 0.191667 | 6/30 | 0.200000 |
| `glm-5.3-flash` | 24/120 | 0.200000 | 7/30 | 0.233333 |
| `kimi-k3` | 29/120 | 0.241667 | 9/30 | 0.300000 |

## 分组实现与可比较范围

历史实现是 [train_cpu.py](../scripts/train_cpu.py) 中的自定义 `grouped_folds`，**没有调用 `sklearn.model_selection.GroupKFold`**：先以 seed `20261018` 从全部 150 个 prompt 中冻结 30 个最终留出组；其余 120 个 prompt 经 `np.unique`、同一固定 seed 的随机排列及 `np.array_split(..., 5)`，分为五折。每折验证侧有 24 个完整 prompt，训练侧有另外 96 个；验证行数依次为 432、432、432、431、432。已核对各折训练/验证 prompt 互斥、OOF 每行恰好覆盖一次，开发集与最终留出也互斥。词表、IDF 和文体特征缩放均只在训练折拟合。

这一实现满足队伍验证规范中“同一 prompt 不得跨训练与验证”的隔离目的，但与队伍现有 [src/cv.py](../../src/cv.py) 的 `GroupKFold(n_splits=5)` 并非相同分折流程。队伍基线 [src/run_baseline.py](../../src/run_baseline.py) 使用完整训练表；本报告的开发 OOF 只覆盖预留最终留出之后的 2,159 行，且折分配与每折训练规模不同。因此，这些历史分数不能直接与队伍全量 GroupKFold 分数比较，也不代表当前队伍基线或公开榜获得提升。本次只交接独立历史实验，没有为统一分折重新训练。

最终留出的模型在全部 2,159 行开发数据上拟合后，对预留 540 行生成过一次预测。它与 OOF 的训练规模和查询 prompt 不同，二者分数之差不能解释成模型改进。完整实验范围和停止依据见 [CPU_EXPERIMENTS.md](CPU_EXPERIMENTS.md)。

## 只读复核

重新汇总时核对了训练文件、标签表、split、summary 和冻结配置的哈希；NPZ 的 `row_index`、`truth` 与冻结数据映射一致，逐类最终留出召回率与原 `holdout_metrics.json` 完全相符。报告不包含原始文本、样本 ID、prompt ID 或逐行预测。

| 本地证据 | SHA-256 |
|---|---|
| 开发 OOF：`b316d0a5c48a-20261018-oof.npz` | `0f09042aee5f4bfc9804eeff956047bb06eccaa71ca5909c1497d8057637a2df` |
| `holdout_predictions.npz` | `533fd1bef13308d9bc4852ab931e431cb617f9edf22ce4b28d934f65594cdcf3` |
| `split_manifest.json` | `c8c24921f617ad483b9e14cb23d0d64a0fbbde44c0214868445e8d0e30d6e1db` |
| `config/labels.json` | `d8c9dcad5979236fb0e5f5504dd2e53c676123bf75cbf06faa13bc1e86524b9d` |

以下命令仅适用于保留上述私有实验文件的原项目副本。从项目目录运行，它读取已有预测并打印聚合计数，不训练或调用 `finalize`。新克隆仓库不包含 `outputs/`，不能直接执行此命令；也不能因本地 claim 未随 Git 复制就视为最终留出尚未使用。

```sh
python - <<'PY'
import json
from pathlib import Path
import numpy as np

root = Path.cwd()
labels = json.loads((root / "config/labels.json").read_text())["labels"]
run = root / "outputs/cpu_round3"
paths = {
    "development_oof": run / "trials/b316d0a5c48a-20261018-oof.npz",
    "final_holdout_saved_predictions": run / "holdout_predictions.npz",
}
for scope, path in paths.items():
    with np.load(path, allow_pickle=False) as saved:
        truth, prediction = saved["truth"], saved["prediction"]
        recalls = []
        print(scope, "rows", len(truth))
        for index, label in enumerate(labels):
            mask = truth == index
            correct = int((prediction[mask] == index).sum())
            count = int(mask.sum())
            recall = correct / count
            recalls.append(recall)
            print(label, f"{correct}/{count}", f"{recall:.6f}")
        print("balanced_accuracy", float(np.mean(recalls)))
PY
```

本次没有 Kaggle 运行、GPU 使用或正式比赛提交。
