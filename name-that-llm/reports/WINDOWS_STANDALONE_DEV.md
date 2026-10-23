# Windows 独立开发入口验证记录

2026-10-24，Asia/Singapore；分支 `exp/windows-4060-20261023`。依据用户“直接根据GitHub仓库推进，不用管Mac本地文件”的指示，新增不依赖历史OOF的原生Windows开发入口。当前结论是**代码及合成检查完成，缺官方训练文件，未取得真实开发分数或Kaggle成绩**。

## 交付变化

- [脚本](../scripts/windows_modernbert_dev.py)只接收官方train.csv，以仓库已有恢复算法重建并核验原120/30分区；30-prompt已用留出不进入特征、fit或评分。
- [修订协议](../docs/WINDOWS_STANDALONE_DEV.md)沿用原ModernBERT模型与五次fit总预算。单模型描述性五折评估不再依赖原机器OOF；没有配对提升结论或候选替换。
- 绑定输入、split、labels、行序、文本、fold、模型/tokenizer、源代码与依赖身份。新特征缓存首次保存后读回核验，禁用pickle，不接纳仅shape相同的旧缓存。
- 每次forward/fit之前持久记录预算。原生Windows supervisor实施墙钟限制、RSS抽样监控、失败留痕；超时、收敛警告、身份或数值错误都停止。清理子进程异常也写入失败claim，禁止自动重试。

## 实际检查

隔离环境是本模块 `.venv-gpu`，沿用torch2.8.0+cu128、Transformers4.56.2、NumPy2.2.6，新增scipy1.16.3、scikit-learn1.7.2、joblib1.5.2、threadpoolctl3.6.0、psutil7.0.0，pip安装退出0，`pip check`通过。未修改系统Python、驱动或WSL。

最终命令从仓库根运行：

```powershell
.\name-that-llm\.venv-gpu\Scripts\python.exe -m unittest discover -s .\name-that-llm\tests -p "test_windows_modernbert*.py" -v
.\name-that-llm\.venv-gpu\Scripts\python.exe .\name-that-llm\scripts\windows_modernbert_dev.py
.\name-that-llm\.venv-gpu\Scripts\python.exe -m pip check
```

**18项测试通过，3.299秒，退出0**：7项既有模型身份/合成claim测试与11项新开发入口测试。测试覆盖原分区索引/ID精确一致、保护集哨兵排除、文件身份变化、错序/坏dtype/NaN/错维缓存、五折合成分类器OOF与指标、失败预扣次数、真实子进程超时终止、终止异常留痕及启动前失败原因。测试使用合成记录或临时进程，无真实比赛训练或GPU前向。默认沙箱的临时文件原子替换曾返回WinError5，最终主代理在正常文件权限下执行测试；不把权限失败误报为模型失败。

两项独立只读核验分别检查分区/缓存/指标和Windows预算/失败路径。主代理落实终止异常留痕、启动前错误记录、模型缺失的结构化blocked返回、Torch inter-op线程控制及加载设置核验，再执行上述最终测试。未重复全仓旧POSIX测试；相关代码没有变更。

默认readiness退出 **2**，无stderr，返回：

```json
{"state":"blocked","blockers":["Missing official training input: D:\\kaggle\\name_that_llm\\name-that-llm\\data\\raw\\train.csv"],"verified_model_files":6}
```

这确认环境版本、labels与模型6文件检查均通过，当前唯一运行材料缺口是官方train.csv。没有创建真实开发claim或输出目录。没有运行`--run`；新真实fit=0、新真实GPU前向=0、旧holdout评分=0、全量最终fit=0。18项合成检查不证明完整真实开发可以在30分钟/8GiB限制内完成。

原始日志、pip freeze与收据保留在忽略的 `outputs/windows_4060/standalone_*`，不上传Git。`standalone_reconciliation.json` SHA256：`c20b64a50273bb9c3dc36651e3121395ee4b5b67fbb8d739251e05b69bb50a8e`。

## 冻结交付与后续

原候选Notebook SHA仍为 `82931fb6a28441e01b884c2ce62c16ced6f35c39bc9ff092c2f53277308b3df8`，labels SHA仍为 `d8c9dcad5979236fb0e5f5504dd2e53c676123bf75cbf06faa13bc1e86524b9d`。没有修改75/25候选、4,096行门槛、旧CPU模型或队友分支。

无需Mac文件即可继续，但Git仓库中的报告不能替代原始训练CSV。待Kaggle原访问限制正常恢复后，从正式渠道取得官方数据并核验SHA；新开发程序才可按预登记运行。用户要求本轮至多一次正式提交的目标保留，恢复后先核查现有Notebook/提交历史，再执行原固定候选；未核实前不重复提交。当前没有Kaggle上传、运行、正式提交或新榜单分数。
