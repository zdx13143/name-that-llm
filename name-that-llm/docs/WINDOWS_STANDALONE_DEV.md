# Windows 独立 ModernBERT 开发协议修订

**2026-10-24执行后状态：本轮已失败并关闭，不能重新运行。** 官方数据已齐全；265次forward尝试、264次持久完成记录、0次fit，无新分数。具名零计算启动修正也已使用，详情见 [实际轮次报告](../reports/WINDOWS_ROUND_20261024.md)。下文保留预登记协议，不构成重开已使用入口的指令。

2026-10-24，Asia/Singapore。用户要求直接依据 GitHub 仓库推进，不再依赖 Mac 本地文件。本修订在任何真实开发前向或分类器拟合之前记录；替代原 [ModernBERT计划](WINDOWS_MODERNBERT_PLAN.md)中“先取得历史 OOF 再开发”的执行顺序。

## 本轮要回答的问题

在已固定的 120 prompt / 2,159 行开发子集上，评估固定 ModernBERT 冻结特征与 LR 的单模型表现。只报告 pooled balanced accuracy、五折 BA、18 类 recall/confusion 和 2,000 次 prompt bootstrap 的描述性区间。没有历史同协议 OOF 时，不计算配对增益、不选择融合权重、不宣称优于队友或原冻结候选，不替换提交 Notebook。

此修订沿用原实验 ID `windows-modernbert-frozen-dev-20261023` 和同一个固定 claim；**仍然总共最多 5 次新 fit 尝试**，不是另加五次。原配对融合方案本轮不执行。旧 291 次真实 fit 和已使用的 30-prompt 留出不重新打开；完成的合成 GPU probe 不重跑。

## 已有仓库资产与唯一数据缺口

模型、池化和分类器方案来自队友代码与原预登记。新入口是 [windows_modernbert_dev.py](../scripts/windows_modernbert_dev.py)，不导入 POSIX `resource` / `SIGALRM` runner。标准化每折只拟合训练部分；分类器固定 C=0.1、lbfgs、balanced、max_iter=5000、random_state=0，不做参数搜索。

真正运行只需官方 `train.csv`，SHA256 必须为 `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`。GitHub 的当前可见源码不包含该文件。无需原机器的 split、OOF、claims、模型或环境；旧已完成状态以仓库历史记录为约束，不伪造缺失的旧 claim。

从正确 CSV 的 id/prompt 元数据在内存中调用现有 `historical_manifest_bytes`，重建字节必须匹配原 split SHA `c8c24921f617ad483b9e14cb23d0d64a0fbbde44c0214868445e8d0e30d6e1db`。不写旧实验目录。随后只保留原开发索引的文本/标签。CSV hash 和元数据扫描会读取文件字节；保护集文本/标签不进入特征、训练或评分。

五折重新使用 `GroupKFold(5, shuffle=False)`；不使用旧 manifest 的 seeded `development_fold`。固定 scikit-learn 1.7.2 与现有 Windows NumPy 2.2.6；不声称与历史 Mac 依赖逐版本相同。新协议及来源版本完整写入 contract，不能凭历史聚合分数做同协议优越性结论。

## 执行预算与证据

- 单次开发提取，最多 540 个 batch，batch=4、长度256、BF16/SDPA、最后4层mask mean、拼接3072维FP32，完整模型/tokenizer revision 沿用原配置。
- 最多5次fit尝试；forward/fit之前持久增加次数，失败不回退。收敛警告、非有限值、身份不符或预算超限均停止；固定 claim 已存在就禁止重跑，没有恢复或替换输出目录选项。
- 外部 supervisor 限时1,800秒，每0.2秒抽样监测子进程树RSS，8GiB超限终止；CUDA allocated峰值在每批后检查，限6GiB；数值库2线程。抽样RSS不是硬性预分配上限，OOM也按失败保留。
- 合同绑定CSV/split/labels、文本与行序、fold、模型全部6个文件身份、配置、源码与依赖版本。特征以专属文件首次保存后读回核验，既有同形状缓存不自动接受；NPZ禁用pickle。
- 完成时保存开发OOF、每类/每折指标、特征/OOF哈希、源码快照、日志、资源记录和claim。失败保留已有文件及次数；全量最终fit、旧留出评分、测试预测均为0。

运行依赖 [requirements-windows-dev.txt](../requirements-windows-dev.txt)，安装到现有隔离的 `.venv-gpu`。不会覆盖系统Python或改动WSL。模型已下载并核验，不需要再次下载或运行合成前向。

以下从 `D:\kaggle\name_that_llm` 执行；默认入口只核验材料，不执行模型：

```powershell
.\name-that-llm\.venv-gpu\Scripts\python.exe -m pip install --index-url https://pypi.org/simple -r .\name-that-llm\requirements-windows-dev.txt
.\name-that-llm\.venv-gpu\Scripts\python.exe .\name-that-llm\scripts\windows_modernbert_dev.py
```

官方文件正常取得并通过readiness后，唯一真实入口是在第二条命令末尾加 `--run`。默认数据位置为 `name-that-llm/data/raw/train.csv`，也支持 `--train` 指向正式取得的同字节文件；路径变化不能重置claim。

执行后核对：Kaggle原账号可正常访问，官方数据已核验；已有冻结候选Version1正式Succeeded、0.351，队友记录0.601。本轮未形成新候选，没有新增提交或重复上传旧版本。原入口应拒绝已有claim/输出，具名恢复也已关闭；不得把换机器、缺失本地claim或合成测试通过当成重跑本轮的理由。
