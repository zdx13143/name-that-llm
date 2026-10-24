# Windows真实开发轮次与Kaggle历史核对

2026-10-24，Asia/Singapore；分支 `exp/windows-4060-20261023`。

**本轮真实开发失败并关闭，没有可用特征缓存、OOF、五折分数或新的Kaggle提交。** 官方数据已取得并核验；启动器、收据写入和claim路径问题已修复，30项相关测试通过。修复后没有重跑真实提取，测试通过不代表真实开发完成。

## 数据与固定协议

用户登录原参赛账号后，Kaggle正常可访问。官方ZIP自动跳转被Chrome以 `ERR_BLOCKED_BY_CLIENT` 拦截，由用户手动下载至仓库根 `name-that-llm.zip`；未换通道绕过拦截。ZIP为874,800字节、3个预期成员，CRC通过，ZIP及成员SHA均匹配 [INITIAL_DATA_AUDIT](INITIAL_DATA_AUDIT.md)。成员首次写入 `data/raw/`，原ZIP保留并精确忽略。

训练SHA为 `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`。readiness核对原split SHA并确认120-prompt / 2,159行开发子集。旧30-prompt留出没有进入特征、拟合或评分；旧CPU搜索、最终训练和完成的合成GPU probe均未重跑。

## 真实执行与失败核对

第一次启动在监督进程检查退出，所有计算计数为0。最小复现确认Windows venv redirector使直接父PID不等于逻辑监督者PID。修复检查PID、创建时间和仅一层与当前venv路径匹配的启动器；实际子进程回归通过。在任何真实forward之前登记了 [具名零计算启动修正](../docs/WINDOWS_STARTUP_RECOVERY.md)，同一experiment_id、数据、模型、数值配置和原总预算不变。

恢复运行在 `probe_modernbert.write_json()` 的 `os.replace()` 出现 `PermissionError: [WinError 5]`。临时文件测试可用Windows读取共享锁复现同样错误，但不能确定本次具体占用者。监督者退出1，worker已退出，运行105.022秒，抽样进程树RSS峰值1,734,283,264字节（约1.615GiB）。未生成 `features.npz`、`oof.npz`、`summary.json` 或成功supervisor收据。

交叉核对又发现恢复入口的路径缺陷：`consume(..., path=CLAIM)`、`fit_head(..., claim_path=CLAIM)` 的默认值在导入时绑定旧路径。恢复后的attempted计数误写原claim，completed计数写恢复claim。这是同一次执行的两条记录，不能相加为两轮。

| 项目 | 本轮核实结果 |
| --- | --- |
| forward尝试 | **265**，两条claim的最高持久尝试计数 |
| forward完成记录 | **264**，最高持久完成计数 |
| fit尝试 / 完成 | **0 / 0** |
| 旧留出评分 / 全量最终fit | **0 / 0** |
| 可用特征 / OOF / 五折及每类指标 | **无** |
| 新Kaggle运行 / 正式提交 | **0 / 0** |

worker错误发生在forward后的完成收据写入处；不把最后未成功持久化的完成数算作成功记录。此前进度消息中的264次尝试已由跨路径核对更正为265。未到540上限不代表允许重跑：真实提取已开始、固定入口已使用，且没有完整缓存可复用。

**原失败claim路径被bug覆盖，不能声称它始终未变。** 两份当前claim字节另行归档，现场路径不回滚、不重置。首次零计算失败的完整JSON按启动前已核实内容重建为独立档案，SHA严格匹配预先固定的 `cdbe811ad0c7664bbfc3389e4ec862cd0e4935b96781118f1f17f3b184289738`。归档明确标注重建来源。原路径遗留的 `started` 不代表仍在运行；进程已确认退出。

## 已修复与验证范围

- 监督者：支持实际Windows venv启动链，拒绝错误PID、过期创建时间、失败状态和已消耗claim。
- 原子写入：仅对Windows错误5/32/33重试同一份已fsync字节，最多8次replace、累计等待0.71秒；永久拒绝仍报错，保留旧收据和临时字节，不重试forward/fit。
- claim路径：按调用时解析当前路径，并在真实提取/拟合调用处显式传入。两项回归修复前失败、修复后通过；测试不写真实claim。
- 恢复配置：固定SHA、LF检出；有计算的失败、数据/参数改变、异常输出或原身份不符均拒绝。

最终**30项相关测试通过，9.793秒，退出0**，包含真实Windows子进程、临时文件锁和合成数据隔离/缓存/预算/失败检查。未重跑旧全仓POSIX测试。

```powershell
.\name-that-llm\.venv-gpu\Scripts\python.exe -m unittest discover -s .\name-that-llm\tests -p "test_windows*.py" -v
```

收尾只读检查：原入口退出2（claim/输出已存在）；具名恢复入口退出1（原claim身份已与启动前固定记录不符）。两次检查未改写claim、调用模型或增加预算。本轮不再运行 `--run`，不删除claim来解锁。当前源码是修复版本，失败运行以输出目录中的源码快照为准。

## 已有Kaggle正式结果

账号 `zephyrpong`，队伍 `zhouruijie`；队员页面列出 `zdx13143` 与 `zephyrpong`。官方[提交记录](https://www.kaggle.com/competitions/name-that-llm/submissions)显示：

| 已有提交 | 状态 | Public score |
| --- | --- | --- |
| [NTL Blend 20261021，Version 1，360768689](https://www.kaggle.com/code/zephyrpong/ntl-blend-20261021?scriptVersionId=360768689) | Succeeded | **0.351** |
| [队友 ntl-ensemble-600，Version 1，360258830](https://www.kaggle.com/code/zdx13143/ntl-ensemble-600?scriptVersionId=360258830) | Succeeded | **0.601** |

第一份Notebook保存运行89.5秒、无加速器；candidate=`blend_partial_0.25`，claim=completed，attempted/completed fits=2/2。Output中的summary、contract、7个source、generator、training hash及selection_scope与本地冻结Notebook元数据逐项一致。未下载远端Notebook字节，不能声称整个文件SHA已核对。

保存运行是2,699行train和3行公开test；其 `formal_submission=false` 描述Notebook自身执行，正式成功和0.351由提交页单独确认。未取得隐藏重跑日志/实际行数，不能把公开3行输出当成隐藏预测。队友0.601只核对提交记录，本轮未重新审计其完整代码/附件。

这些是已有结果。本轮没有新增上传、运行或正式提交，也没有重复提交已知0.351的同一版本；新ModernBERT没有形成可评估或可提交候选。0.351、0.601及旧开发分数不能用来宣称本轮改进。

## 证据和后续边界

原始证据在忽略目录 `outputs/windows_4060/`：官方导入收据、两次实际调用记录、`claim_path_forensics_20261024/`、最终测试日志和两份reconciliation；实际输出目录保留源码快照和stderr。

归档README中“neither live claim path was overwritten or reset”仅指事后归档/重建步骤，不能用来否定运行时覆盖缺陷；同目录 `CLARIFICATION.md`已明确此范围，保留原README字节以维持已记录哈希。

- `round_reconciliation_20261024.json` SHA：`c6c47947074b6c0c58781ae595a022952b59cf673e88e0310df5c88995c7de18`
- `kaggle_live_reconciliation_20261024.json` SHA：`ff4294e7ce1cef7e0ce577417e08e5d07d2170d12f6a580e2452596b3c08ac68`
- 冻结Notebook SHA仍为 `82931fb6a28441e01b884c2ce62c16ced6f35c39bc9ff092c2f53277308b3df8`；labels仍为 `d8c9dcad5979236fb0e5f5504dd2e53c676123bf75cbf06faa13bc1e86524b9d`。

只提交代码、测试、配置和聚合说明；ZIP、CSV、权重、环境、原始claims/logs及行级输出不进Git。后续新一轮必须先记录新预算和停止条件，保留本轮265次尝试及失败，不复用本轮入口或重开旧30-prompt留出。
