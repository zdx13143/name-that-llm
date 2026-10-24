# Windows ModernBERT 第二轮：五折完成，提交门槛未通过

2026-10-24，Asia/Singapore；分支 `exp/windows-4060-20261023`。

**修复后的真实五折评估已完成，pooled balanced accuracy为0.289904；没有合格的新提交候选，因此本轮最终拟合、Kaggle上传、保存运行及正式提交均为0。** 旧30-prompt留出没有重开。上一轮失败及265次forward尝试完整保留，本轮使用用户新授权的独立预算。

## 预登记与实际执行

[第二轮计划](../docs/WINDOWS_ROUND2_PLAN_20261024.md)、[固定配置](../config/windows_modernbert_round2_20261024.json)、独立入口及回归测试在任何本轮真实计算前提交为 **`42a444c`** 并推送同一独立分支。配置SHA为 `32bb45624a6969be47db8802be9c84a47d5cda78a9ad07e6453fcf6c89c8c5b5`。

固定实验ID `windows-modernbert-frozen-dev-20261024-r2`；新claim与新输出目录，沿用原120 prompt / 2,159行开发分区。ModernBERT-base固定revision、BF16/SDPA、长度256、batch=4、冻结最后4层mask mean拼接3072维FP32；每折仅训练部分拟合StandardScaler，LR固定C=0.1。五折为GroupKFold(5, shuffle=False)，没有参数搜索或融合权重选择。

| 项目 | 实际结果 |
| --- | ---: |
| 开发行 / prompt | 2,159 / 120 |
| forward尝试 / 完成 | **540 / 540** |
| CV fit尝试 / 完成 | **5 / 5** |
| 最终开发head fit | **0** |
| 旧保护集评分 / 全量最终fit | **0 / 0** |
| supervisor状态 / 退出码 | **completed / 0** |
| 总耗时 | 67.288秒 |
| 特征提取耗时 | 48.226秒 |
| 抽样进程树RSS峰值 | 1,938,956,288字节，约1.806GiB |
| Torch CUDA allocated峰值 | 367,231,488字节，约350.219MiB |
| Torch CUDA reserved峰值 | 375,390,208字节，358MiB |
| 新Kaggle保存运行 / 正式提交 | **0 / 0** |

GPU为RTX4060 Laptop；Torch intra/inter-op均为2线程。RSS是采样峰值，Torch allocated/reserved不等于整机显存占用。stderr只有`torch_dtype`弃用提示；没有收敛警告或执行失败。

## 全部开发结果与固定停止判断

| 指标 | 实测 | 预登记要求 |
| --- | ---: | ---: |
| pooled BA | **0.2899042951** | ≥0.36，未通过 |
| Fold0 BA，432行/24prompt | 0.2962962963 | ≥0.30，未通过 |
| Fold1 BA，432行/24prompt | 0.2824074074 | ≥0.30，未通过 |
| Fold2 BA，432行/24prompt | 0.3125000000 | ≥0.30，通过 |
| Fold3 BA，432行/24prompt | 0.3009259259 | ≥0.30，通过 |
| Fold4 BA，431行/24prompt | 0.2575483092 | ≥0.30，未通过 |

2,000次整prompt bootstrap描述性95%区间为 **[0.2704743425, 0.3088898615]**。该开发集经过多轮探索，区间不代表独立留出、榜单成绩或未参与选择的新数据证据。

历史冻结75/25融合在同120-prompt开发子集报告BA=0.348673，本轮数值更低。缺少历史逐行OOF，且Windows依赖版本与历史环境不完全一致，没有配对增益、显著性或逐fold优势结论。此结果只否定本轮固定配置的提交资格，不代表所有ModernBERT方案或其他模型均无效。

| 类别 | Recall |
| --- | ---: |
| gpt-5.5 | 0.225000 |
| gpt-5.6-luna | 0.241667 |
| gpt-5.6-sol | 0.183333 |
| gpt-5.6-terra | 0.175000 |
| gpt-6-astra | 0.483333 |
| gemini-3.1-pro-preview | 0.566667 |
| gemini-3.6-flash | 0.375000 |
| gemini-3.7-flash | 0.258333 |
| gemini-3.8-flash | 0.266667 |
| claude-fable-5 | 0.193277 |
| claude-opus-4-6 | 0.341667 |
| claude-opus-4-8 | 0.233333 |
| claude-opus-5 | 0.525000 |
| claude-sonnet-5 | 0.308333 |
| qwen3.8-max | 0.266667 |
| glm-5.3 | 0.125000 |
| glm-5.3-flash | 0.150000 |
| kimi-k3 | 0.300000 |

完整18×18 confusion matrix保存在本地summary。没有据上述逐类结果追加训练、选择新权重或放宽门槛。条件部署预算未触发，不拟合最终head、不生成或上传新Notebook、不重复提交旧版本。此前核实的本账号0.351、队友0.601仍是历史正式结果；本轮未刷新榜单，也没有新Kaggle分数。

## 验证与证据

真实运行前**33项相关合成测试全部通过，19.348秒，退出0**，覆盖Windows启动链、临时文件锁、当前claim路径、旧记录身份、Round2子进程协议、输入/缓存隔离、预算和门槛。没有重跑历史全仓POSIX测试。

```powershell
.\name-that-llm\.venv-gpu\Scripts\python.exe -m unittest discover -s name-that-llm/tests -p 'test_windows*.py' -v
```

完成后独立复核OOF重算pooled与五折BA，一致；主线程最终复核官方原开发行与NPZ逐行身份、snapshot/当前源码、所有哈希、资源、已退出的supervisor和两个旧claim。features为(2159,3072) float32；OOF为(2159,18) float64，有限且归一化，最大概率和误差4.44e-16；120个group各只属于一折。旧保护集文本/标签不进入特征、拟合或指标。

最终readiness只读检查退出2，准确拒绝已有claim/output；该检查不增加任何forward/fit，claim前后哈希一致。**本轮已完成并关闭，不能再次执行`--round2 --run`，不能删除claim或换目录解锁。**

本地完整产物位于 `outputs/windows_modernbert_dev_20261024_r2/`；最终核验脚本与收据在 `outputs/windows_4060/`，行级特征、OOF、原始日志与claims不进入Git。

| 证据 | SHA256 |
| --- | --- |
| contract canonical对象 | `973ca22331cd509d3117c7fc39b2f90da2c1949f1fa5e972ffa0335acd98396a` |
| features.npz | `f600b58712a6539ff84efe1630550de558dd76f1aedb41c84c13d768bdfcc7a0` |
| oof.npz | `105ce885457e15f5e70844e9941e519edd4c85f07deab17803f3d9833496a17b` |
| summary.json | `6724ef2428793099f66c655c14272e815cee1f75623128e83f88e25866802f57` |
| supervisor.json | `f84090d4e412802ebcb22765721c77aee1982d08413943b2c3b22b16e772f056` |
| round2_final_reconciliation_20261024.json | `3ccd02e9ee6659bb668a5a22b080023b939901c6052baada8ae75b228c7f49a1` |

上一轮两条claim当前SHA仍分别为 `fc06fd719de08f08bc283136104753e020bb6cb4b2abcb73387aea8f6a120507` 与 `86d89fc70b66e22fed20e8275d5b7af85309e731ab5e8c6ab2a1c36c0b8cc2c3`；原265/264计数、0次fit以及覆盖缺陷的法证与重建说明均保留，详见 [上一轮失败报告](WINDOWS_ROUND_20261024.md)。新的540/540、5/5属于独立第二轮，不能把旧失败改写为成功。

冻结75/25 Notebook SHA仍为 `82931fb6a28441e01b884c2ce62c16ced6f35c39bc9ff092c2f53277308b3df8`，labels SHA仍为 `d8c9dcad5979236fb0e5f5504dd2e53c676123bf75cbf06faa13bc1e86524b9d`。本轮只向自己的分支提交实现、预登记、测试和聚合报告。
