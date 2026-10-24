# Windows 零计算启动故障修正

**执行后状态：此一次入口已使用并失败关闭，不可再运行。** 实际265次forward尝试、264次持久完成记录、0次fit。恢复时默认参数绑定旧claim路径的缺陷使原claim被计数覆盖；已保留现场并修复，原零计算失败内容重建为单独档案且匹配原SHA，未回滚当前路径。下文是执行前的设计记录，其“原claim不会被覆盖”目标没有达成；以 [真实结果与修复报告](../reports/WINDOWS_ROUND_20261024.md)为准。

2026-10-24，任何真实开发 forward / fit 之前记录。用户已要求取得官方资料后继续本轮；本修订只修复启动链，沿用同一开发实验及其总预算。

首次真实入口在 `execute()` 的监督进程检查退出。Windows 3.11 venv redirector 使 Python worker 的直接父进程是启动器，原监督者在上一层。最小复现直接调用该检查，在修复前失败；修复后核验 PID、进程创建时间和最多一层路径匹配的 venv 启动器。20项相关测试通过，测试未使用比赛训练数据或GPU。

原 claim 为 `failed`，attempted/completed forward 和 fit 均为0，旧留出评分和最终全量fit也为0；尚未加载比赛数据或模型的worker在入口检查退出。原失败claim、contract、日志和源码快照原地保留，不删除、不改写。其字节身份固定于 [恢复配置](../config/windows_modernbert_startup_recovery_20261024.json)。

为完成已经授权的同一轮开发，增加一个显式、仅一次的 `--recover-zero-work-startup` 入口。它不是通用重试机制：必须匹配这条失败记录的完整哈希、全零计算计数和没有特征/OOF的原输出集合。恢复使用固定的子claim和输出路径，关联父失败claim；两个claim累计仍最多540次forward尝试和5次fit尝试。原入口继续被失败claim锁定，恢复入口一旦创建claim或输出也不能再次执行。

恢复contract要求数据、分区、fold、模型、依赖、所有数值配置及非runner源码与原contract完全相同；仅runner启动修复和本恢复记录可以改变。无需重新下载模型、重跑已完成合成probe、增加预算或打开旧30-prompt留出。任何发生了真实forward/fit的失败不适用本入口。脚本不自行重试；保留原配置 `automatic_failed_run_retries=0`。

从 `D:\kaggle\name_that_llm` 先做只读核验；确认ready后在同一命令末尾追加 `--run` 执行这次明确的启动修正：

```powershell
.\name-that-llm\.venv-gpu\Scripts\python.exe .\name-that-llm\scripts\windows_modernbert_dev.py --recover-zero-work-startup
```

这项修订不更换模型或根据分数选择参数；截至记录时没有任何新的开发分数可供选择。原失败日志始终作为本轮证据的一部分。
