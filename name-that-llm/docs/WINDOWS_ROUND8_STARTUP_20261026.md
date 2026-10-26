# R8 唯一启动修正

原R8在`0eb0600`推送后执行。worker在`load_inputs()`与任何评分/变换之前被直接父进程身份检查拒绝；原claim保留failed，attempted/completed transforms均为0，原源码快照/stderr保留。父进程监督耗时0.75秒。

本机Windows venv使用redirector，真实Python进程的直接父进程可能是venv launcher。R8原检查遗漏这层，故合法运行被拒绝。复用已经在R7验证的`supervisor_round7.require_live_supervisor`，只允许正确的直接父进程或恰好一层原生路径校验通过的venv launcher，仍严格核对PID/创建时间/未消费状态。使用真实子进程作无数据回归，正常链通过，错误PID、时间、状态和已消费计数拒绝。

独立启动身份`windows-frozen-svc-sinkhorn-dev-20261026-r8-startup-1`；新配置与入口、独立输出。旧claim不恢复，失败证据哈希固定；校验算法、输入、折、指标、门槛全部不变。剩余预算119秒，与失败0.75秒合计小于原120秒，仍10次变换、2GiB、2线程、0fit/encoder/holdout。此为产生任何候选结果之前的一次明确启动修正，替代原计划的无重试停止条件，非自动重试或增加候选；再次失败则停止，不再新增启动修正。修正先提交推送再执行。
