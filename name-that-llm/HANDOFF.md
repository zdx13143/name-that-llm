# Name That LLM 交接

更新：2026-10-28（Asia/Singapore）。本模块保存 Name That LLM 独立实验；下文保留历史运行记录，最新 Windows 接续与材料/提交边界见文末。

## 已完成与证据范围

| 状态 | 内容 |
| --- | --- |
| 已核对 | Kaggle 登录页面显示已报名后的 `Submit Prediction`、`Team`、`Submissions`。 |
| 已核对 | 官方 Overview：18 类、balanced accuracy、Notebook 提交、`submission.parquet`、关闭互联网、CPU/GPU 最长 9 小时。 |
| 已实测 | 真实训练集 2,699 条、150 个 prompt、18 类；公开 test/sample 各 3 行。字段见 [data/README.md](data/README.md)。 |
| 已完成 | 三个真实文件全量读取、SHA-256、schema、空值、重复及标签分布审计，退出码 0；详见 [审计报告](reports/INITIAL_DATA_AUDIT.md)。 |
| 历史 CPU 阶段 | 三轮CPU搜索、护栏修复、一次最终留出、全量模型及离线Notebook本地复现已完成；该阶段记录不代表后续GPU或Kaggle状态。 |
| 最新 Windows 阶段 | R9词视角半中心化BA 0.454279，对R7+0.8341个百分点、五折改善，但未达到1个百分点采用门槛；本账号正式0.414，队友已核实0.651。无R9新Kaggle提交，完整证据见文末。 |
| 已授权 | [CPU 方案](docs/PLAN.md)及[整轮停止标准](docs/OPTIMIZATION_PLAN.md)；不逐次询问常规调参。 |

## 待核对

1. **数据描述冲突**：Overview 写交互测试 1 行，Data 页写 3 行。本次真实文件已核对为 3 行；推理代码必须适配任意行数，不能把公开样例当隐藏测试。
2. **截止时间冲突**：Overview 最终提交为 `2027-01-13 23:59 UTC`；页头 `Close` 为新加坡 `2027-01-15 07:59`，对应 `2027-01-14 23:59 UTC`。保留两项证据，官方澄清前按较早日期安排。
3. **完整规则**：每日提交数、最终可选提交数、队伍人数和外部资源条件尚未完整核对。
4. **协作状态**：GitHub 分支隔离不等于 Kaggle 团队合并；如需要共享私有比赛材料，先核对实际队伍状态和规则。当前文档不声称已向队友发送内容。
5. **算力状态**：用户当前免费 GPU 余额、可用型号和排队情况未核实。启动实验前在 Notebook Settings 检查并记录。

## 接手后的顺序

1. 阅读本文件、[数据说明](data/README.md)与[方案](docs/PLAN.md)。
2. 本机数据已在 `data/raw/`；其他机器需另行下载，按审计报告核对哈希。不要将登录态、令牌或 Cookie 写入仓库。
3. 用审计脚本复核数据版本和分组约束。审计结果与页面描述分别记录。
4. 自主执行已批准 CPU 基线与有限优化；验证分数明确标为本地验证，不称为榜单分数。不要重用最终留出集调参。
5. CPU 结果揭示改进方向后，再决定是否使用 Kaggle 免费 GPU；训练资源与离线提交资源分别准备。

## Git 协作约定

当前上传仓库：[zdx13143/name-that-llm](https://github.com/zdx13143/name-that-llm)，分支 `exp/cpu-stylegram-handoff-20261021`，遵循目标 README 的 `exp/<主题>` 约定。旧 `codex/name-that-llm-optimize`、`e434209` 及 `9d29390` 属于原 [kaggle-titanic](https://github.com/zdx13143/kaggle-titanic) 历史，不是本仓库祖先；本次仅移植文件快照。仅提交明确属于本任务的代码和文档，不重写队友分支、不强推、不直接合并默认分支。上传前检查暂存文件；不上传原始比赛数据、凭据、环境、模型权重或输出。远端是否同步以实际 push 与远端 SHA 核验为准。

## 2026-10-18 持续优化准备（历史阶段）

用户要求持续优化至效果不明显。已准备[整轮 CPU 模型方案与停止规则](docs/OPTIMIZATION_PLAN.md)，按用户刚提供的 `AGENTS.md` 提交审核，模型实施和训练尚待确认。当前 Kaggle 浏览器访问被权限检查拒绝，CLI 也未认证；未尝试绕过拒绝，真实数据仍未取得。访问条件未满足不能记为模型收益平台期。

已实现 `scripts/profile_data.py` 和 `config/labels.json`，只做数据质量检查，不训练模型。可检查完整 CSV/Parquet、文件 SHA-256、schema/空值、ID 唯一性、类别与 prompt 覆盖、exact/空白归一重复及跨 prompt 标签冲突、train-test 重叠、sample ID 覆盖。标签表来自此前已核对的官方页面；与真实标签集不一致时报告差异，不擅自改写数据。

本轮在仓库根执行：

```sh
enveda-casmi-2026/.venv/bin/python -m unittest discover -s name-that-llm/tests -v
enveda-casmi-2026/.venv/bin/python name-that-llm/scripts/profile_data.py
```

8 项临时合成文件测试通过；第二条真实入口返回 `2`，正确报告缺少 `train.csv`、`test.csv`、`sample_submission.parquet`，未生成真实数据报告。默认输出 `reports/local_data_profile.json` 已确认受 Git 忽略。没有安装新依赖、生成预测或取得模型分数。

官方依据：[Overview](https://www.kaggle.com/competitions/name-that-llm/overview/abstract)、[Data](https://www.kaggle.com/competitions/name-that-llm/data)、[Rules](https://www.kaggle.com/competitions/name-that-llm/rules)、[GPU 使用指南](https://www.kaggle.com/docs/efficient-gpu-usage)。

## 2026-10-18 本地数据取得与自主实施授权

用户先授权重试浏览器，但工具仍明确报告该域名存在已保存拒绝设置或管理策略。未绕过限制；用户随后自行将比赛 ZIP 下载至 `~/Downloads/name-that-llm.zip`。检查压缩包成员恰为三个预期文件、大小合理、CRC 全通过，解压时只写指定目标；若目标文件已存在且字节不同会停止，未盲目覆盖。

从仓库根再次执行：

```sh
enveda-casmi-2026/.venv/bin/python name-that-llm/scripts/profile_data.py
```

本次成功退出 `0`，输出 `reports/local_data_profile.json`，审计时间 `2026-10-18T03:37:22Z`。训练 2,699 行、测试与 sample 各 3 行，必需字段完整、ID 唯一、标签合法；17 类各 150 行，`claude-fable-5` 为 149 行。150 个 prompt 中 149 个具有全部类别，另一个缺一类。完全/空白归一的重复文本以及 train/test 文本、ID 重叠均为 0。完整范围、哈希和局限见 [真实审计](reports/INITIAL_DATA_AUDIT.md)。

用户随后明确说“请你自己做决定自动推进不用询问我”。该新指示覆盖此前模型方案等待审核的阶段，当前自主执行已经列出的 CPU 基线、有限优化及停止规则；不需要重复批准每个超参数。正式比赛提交、付费资源及绕过浏览器权限不在本轮执行范围。模型实验报告会另行记录真实结果，不把本次数据审计称为模型提升。

## 2026-10-18 用户要求暂停（历史检查点，现已恢复）

用户最新要求“请你现在在合适位置暂停下工作，后续再继续推进”。第二轮搜索已自然退出，当前无活动模型进程，无未完成 trial；第三轮、最终留出和真实最终 Notebook 尚未启动。**本次是用户主动暂停，不能称优化已收敛。**

- 两轮合计 32 个配置、155 次主 CV 拟合和 40 次预定稳定性拟合，共 195 次真实分类器拟合。开发 BA 从初始最佳 0.281513，经首轮 0.295880，至文体统计增强模型 0.340799。两个稳定性种子的增益同向；这些均为开发验证，非最终留出或榜单结果。
- 固定 120 个 prompt / 2,159 行开发数据与 30 个 prompt / 540 行最终留出保持不变。最终留出未打开。原始数据、OOF、特征缓存、划分、代码快照和暂停检查点留在 `outputs/cpu_round1/`、`outputs/cpu_round2/`、`outputs/PAUSED.json`，受 Git 忽略。
- 已有 20 项审计、模型及 Notebook 合成行为测试通过。根任务另行只读复核首轮全部 24 个 trial artifact，重算 BA、核对 OOF 行序/标签及 prompt 隔离，全部一致；收据是 `reports/local_round1_reconciliation.json`。没有据此多跑真实拟合。
- 新的统一 `finalize_cpu.py` 是未测试、未执行草稿；旧 round2 最终评估的跨轮 marker 失败路径与 prior OOF 标签/truth 校验仍需修复。实际本轮文件已另行核对一致，现有指标未因此失效。**恢复时先补齐这些护栏，不要直接运行旧 finalize。**
- Notebook builder 的文体模型路径已通过合成端到端检查；真实全量模型、真实公开测试预测与真实最终 Notebook 尚未生成，不能写成 Kaggle 执行完成。

完整指标、范围、源码快照及局限见 [CPU_EXPERIMENTS.md](reports/CPU_EXPERIMENTS.md)；恢复核验与第三轮固定配置表见 [CPU_RESUME.md](docs/CPU_RESUME.md)。第三轮入口尚未实现，不存在可直接续跑第三轮的训练命令。不要重跑已结束的两个 search，也不要通过换输出目录重置计数。

## 2026-10-18 恢复后的第三轮 CPU 搜索

用户已解除暂停，并要求本对话后续集中推进 Name That LLM。CASMI 的扩展结果与独立环境已交接至 `<historical-casmi-worktree>`，分支 `codex/casmi-continue`；当前工作目录 `<historical-workspace>` 使用 `codex/name-that-llm-optimize`。二者从已推送的双赛共同检查点 `e434209` 分出。不要在另一目录切换分支或共写输出。

- 已修复统一最终评估入口；各轮旧入口都委托同一 `finalize_cpu.py`。按项目和训练文件 hash 创建持久独占 claim，先写 claim 再拟合；单轮 marker 写失败或留出预测后全量拟合失败也不能重开。所有 search 检查同一 claim，不能换输出目录重新调参。
- 历史输入、标签、split、CSV/OOF schema、行序、真实标签、预测范围与重算 BA 均纳入校验。历史 JSON 未改写；追溯绑定合同明确是恢复时新增的核验，不能冒充第一轮运行时已存在。第三轮还冻结每份历史 trial JSON/OOF 的 hash，复用前再次核对。
- 第三轮保持原 120 个 prompt / 2,159 行开发数据及 30 个 prompt / 540 行最终留出，已完成 9 个新配置、45 次主拟合、10 次新增稳定性拟合；旧 incumbent 两个种子的 10 次历史拟合直接复用，不重复计费。三轮合计 **41 个配置 / 250 次 CV 分类器拟合**，搜索墙钟约 **953.22 秒**。
- 选择配置 `b316d0a5c48a`：char `(2,5)` + 文体统计权重 `0.25` + LR `C=16`。开发 BA **0.346818**，相对第二轮最佳 **+0.006019**，按 prompt 配对 95% 区间 **[-0.002782, 0.014819]**。两个预定稳定性种子的差值均为正，所有主/稳定性拟合无收敛警告。
- 三个既定优化方向都未满足“确认提升”的完整判据，按事先工程规则结束这轮 CPU 搜索；区间仍允许超过 0.005 的收益，因此**不是已进入平台期或全局最优的证明**。不能用最终留出结果继续调参。

精确源码、registry、配置、OOF 与 artifact contract 保留在 `outputs/cpu_round3/`，受 Git 忽略。完整命令、逐批比较、稳定性和本次最终交付状态见 [恢复执行记录](docs/CPU_RESUME.md) 与 [实验报告](reports/CPU_EXPERIMENTS.md)。

## 2026-10-18 冻结、一次最终留出与离线交付（最新）

独立只读复核通过后，按既定开发选择冻结 `b316d0a5c48a`，只执行一次统一最终评估。最终留出 **540 行 / 30 个 prompt**，BA **0.383333**，2,000 次按 prompt bootstrap 的 95% 区间 **[0.338889, 0.429630]**。留出拟合 8.719 秒，随后全部 2,699 条训练数据拟合交付模型 11.206 秒；没有根据留出结果更改配置。项目级 claim 为 `completed`，三轮均保留同一次开启标记；不得删除或通过换目录再次打开。

实际命令从仓库根执行如下，均已经完成，**不要重复执行训练或 finalize**：

```sh
name-that-llm/.venv/bin/python name-that-llm/scripts/train_cpu_round3.py search
name-that-llm/.venv/bin/python name-that-llm/scripts/finalize_cpu.py --output name-that-llm/outputs/cpu_round3
name-that-llm/.venv/bin/python name-that-llm/scripts/build_cpu_notebook.py --run-dir name-that-llm/outputs/cpu_round3
name-that-llm/.venv/bin/python name-that-llm/scripts/verify_cpu_notebook.py --run-dir name-that-llm/outputs/cpu_round3
name-that-llm/.venv/bin/python -m unittest discover -s name-that-llm/tests -v
```

- [离线 Notebook](notebooks/cpu_offline.ipynb) 嵌入冻结配置与模型源码，交付文件无 cell 输出。已在当前本机 `.venv` 顺序执行导出的全部 4 个代码单元，完整重训一次，耗时 10.860 秒；这不是 Jupyter 内核或 Kaggle 会话执行。
- Notebook 的公开 test 3 行 ID、标签预测与冻结模型完全一致，decision score 最大绝对差为 0，`runtime.json` 可解析、环境版本差异为空。两份 Parquet SHA 均为 `30c13d5c20118765fa8dc4f2d3fbc29846597c500244c96f0d6d539121807c89`；公开样例只验证接口与复现，不是隐藏测试成绩。
- 总计 **250 次 CV + 2 次最终拟合 + 1 次 Notebook 复现 = 253 次真实拟合**。最终全套 **28 项测试通过**，涵盖一次性 claim 失败路径、标签/schema/OOF 校验和 Notebook 冻结配置篡改拒绝；`git diff --check` 通过。
- 独立只读审查从 OOF/holdout NPZ 重算开发与最终 BA、分组区间、逐类 recall/confusion；源码/registry/contract、开发与留出互斥、冻结选择、模型/Notebook hash 和逐行公开预测均匹配，没有重训或再次评分留出。
- 本地 `outputs/COMPLETED.json` 指向最终证据；历史 `PAUSED.json` 标记被本次完成状态取代。原始数据、环境、所有模型/预测、执行版 Notebook 与本地收据均受 Git 忽略，不上传。当前无活动训练进程。

本轮 CPU 搜索按工程规则结束，不能据此证明平台期。后续本对话仍只负责 Name That LLM；先处理实际 Kaggle 运行环境和规则核对。若要研究新模型，先记录新的评估方案并声明这 30 个 prompt 已见，不可作为新的独立选模留出。Kaggle 访问限制尚未解除，没有使用 GPU、正式提交或榜单分数。


## 2026-10-21 独立仓库上传

用户指定上传到私有仓库 `zdx13143/name-that-llm`，并要求遵守队友约定。已读取目标 `README.md` 和 `EXPERIMENTS.md`；当次查询的 Issues、PR 列表为空。README 要求实验分支为 `exp/<主题>`、提交说明注明改动及 CV 变化、报告每类召回并附复现命令，故按用户最新指示使用 **`exp/cpu-stylegram-handoff-20261021`**，覆盖旧工作区默认的 `codex/` 命名。没有修改目标 main 或队友工作分支。

- 目标基点：`main@692ac87d6b7b3e16d0b74bdc85c476d47daac059`。源快照：原 [kaggle-titanic](https://github.com/zdx13143/kaggle-titanic) 的 `9d293904365f54acdfaa22775c0647d080d1b062`。只导入 25 个受 Git 跟踪的 NTL 文件，不导入旧双赛 Git 历史、CASMI 项目、原始数据或本机输出。
- 保留 `name-that-llm/` 子目录作为独立模块；目标已有 `src/`、`scripts/`、`configs/`、根依赖和 `EXPERIMENTS.md` 不变，根 README 仅追加模块入口。所有原有 Python 代码、测试、依赖、标签配置和空输出 Notebook 逐字节一致。
- 适配本模块 README/HANDOFF/CPU_RESUME/PLAN 的仓库入口、根计划链接和个人绝对路径；新增本模块 AGENTS、WORKSPACE、COLLABORATION、局部忽略规则、NTL 原计划摘录、源文件 hash 清单及逐类指标。源/目标文件 hash 见 [SOURCE_SNAPSHOT.json](reports/SOURCE_SNAPSHOT.json)。
- 队友要求的逐类召回见 [PER_CLASS_RECALL.md](reports/PER_CLASS_RECALL.md)：只从已有预测重算聚合，没有重新训练或打开最终留出。本模块五折实现是完整 prompt 分组后按固定 seed 划分，并非目标 `src/cv.py` 的 sklearn GroupKFold；样本也为 120-prompt 开发子集，不能将它写成队伍 150-prompt 全量同协议成绩或新增改进。
- 上传前在目标目录用原已验证 Python 3.13.2 环境执行 `python -m unittest discover -s name-that-llm/tests -v`，**28 项通过，0.519 秒**；仅有临时合成数据拟合，没有新的真实数据实验。未在目标目录安装环境；Windows/Kaggle 未运行。
- 原始数据、凭据、`.venv`、weights、缓存、OOF、预测、执行版 Notebook 及旧 claim 都不随 Git 上传。逐项检查忽略规则，数据说明 `data/README.md` 可正常提交。新 clone 不包含完整实验状态，不得把缺少 claim 当作已使用的 30-prompt 留出可以重开的理由。

本次 CV 变化为 **0**；原开发 BA 0.346818 和一次留出 BA 0.383333 保持历史值。没有 Kaggle 操作、正式提交、付费资源或新模型优化。后续先读本模块 [WORKSPACE](WORKSPACE.md) 与目标仓库协作约定，避免盲目执行旧 search/finalize。

本次上传使用的 Git 命令（先审查明确路径暂存的 diff，随后仅推本分支）：

```sh
git switch -c exp/cpu-stylegram-handoff-20261021
git diff --cached --check
git push -u origin HEAD:refs/heads/exp/cpu-stylegram-handoff-20261021
git ls-remote --heads origin refs/heads/main refs/heads/exp/cpu-stylegram-handoff-20261021
```

实际是否上传成功以 push 返回值和远端 SHA 核对为准，不从本地 commit 推断已同步。

## 2026-10-21 上传后审查队友与开发比较（最新）

原上传已成功推送 `exp/cpu-stylegram-handoff-20261021@64ec4e7`。用户随后要求“上传完后分析队友的工作，在此基础上继续推进”，结合此前自主决策授权，本轮从它新建 **`exp/dev-comparison-20261021`**。遵循队友的 `exp/` 分支、prompt GroupKFold、逐类召回与可复现命令要求；main及队友分支不变。

队友成果和局限见 [TEAM_WORK_REVIEW](docs/TEAM_WORK_REVIEW.md)。确认其折内TF-IDF/指纹和分组验证可以复用；修复 `src/centered_stylegram.py` 只从组首行减均值、验证时使用真实组数，以及 `src/run_embed.py` 当前嵌入进入基线后又重复加入增强池的问题。中心化现在每行正确减均值、保留单例并处理空词表；n/18仍是待验证假设，未开展真实中心化训练。嵌入比较只称开发对照，不声称完整嵌套泛化估计。裸缓存身份、外部语料重叠、epoch选择、完整嵌套融合等审查发现仍保留，未假装本轮全部修复。

0.601有社区Notebook参考源码，但当前文件的合并版本/后处理与队友成功v8记录未绑定；旧打包入口只是早期CPU/ModernBERT方案。根README改为有日期的队友记录，不把它写成本轮核验的当前成绩。没有访问Kaggle、重新核对实际GPU配额或执行大模型；历史访问限制未在本轮证明解除，不通过其它入口绕过。

### 实际执行与结果

- 训练前方案为 [TEAM_CONTINUATION_PLAN](docs/TEAM_CONTINUATION_PLAN.md)，初版代码提交 `a88484a`。只用原**120 prompt / 2,159行开发数据**重新固定GroupKFold五折，旧**30 prompt / 540行最终留出**不训练、不评分；旧search/finalize没有执行。新代码没有全量拟合或测试预测入口。
- 四个预定模型：旧冻结文体模型、队友charword/C16、char13/C4、指纹/C0.25。六个预定融合只使用队友权重0.25/0.5，不新增训练。新同协议基线BA **0.340336**，最佳等权charword融合 **0.347281**，差值 **+0.006944**，描述性prompt配对95%区间 **[-0.006011, 0.018974]**，仅 **3/5折改善**，全部方案未达到采用门槛。
- 首次原模型首折已拟合返回，但float32概率行和被过严atol=1e-8拒绝，没有保存预测或得到CV分数。同管线合成数据复现合法float32误差2.384e-7；真实首次概率没有保存，不能事后逐值确认。修订为dtype适配检查后转float64归一化，配置/折/权重不变；提交 `ce92b24`，原failed claim/contract/source原样保留，只允许一次恢复且消耗计数不归零。
- **累计21次真实拟合尝试 = 1次失效首折 + 20次有效拟合**；恢复96.824秒、包含首次8.958秒后合计105.782秒。峰值RSS约0.954GiB，数学库2线程；20个有效fold无收敛警告。此前本模块253次拟合保持历史记录，本次新增21次；队友其它实验次数不在这个计数内。
- 独立核验不导入runner、不重训，从4个新OOF重算10方案BA、五折BA、18类recall和2,000次prompt配对CI，数值与原summary一致。原120/30组边界、输入顺序、概率及hash均匹配；旧最终claim SHA仍为 `5a93d4efc1590e6acc63bdd3d40b3f3ebf860aa85e04d6af5bda582638c68db0`，状态completed。
- 复核另发现两个非最佳方案的“改善折数”计入了等分的浮点残差：char13权重0.25为3而非4，权重0.5为2而非3。原summary不改写，修正在独立收据及报告中明示；future runner采用差值>1e-12并有回归测试。没有影响BA、CI、最佳方案或全部不采用结论，没有重训。

完整表格、每类recall、全部模型OOF hash、合成/真实证据界限见 [TEAM_DEV_COMPARISON](reports/TEAM_DEV_COMPARISON.md)。工程停止本组有限比较，保留原冻结Notebook/模型；不把区间跨0称为已经证明平台期。后续最有价值的是绑定队友成功强基线版本及同协议OOF，再评价新增互补，不继续盲目扩展这组CPU超参数。

### 环境、文件与命令

新环境为目标**仓库根 `.venv/`**，Python3.13.2；沿用旧numpy/scipy/sklearn版本，新增pandas3.0.6。旧原始工作区环境、数据、权重、输出、claim均未改动。依赖固定在 `requirements-team-cpu.txt`，没有修改根requirements。

以下从目标仓库根运行；`NTL_TRAIN`和`NTL_SPLIT`分别代表原NTL项目的 `data/raw/train.csv` 和 `outputs/cpu_round3/split_manifest.json`（用变量替代本机个人绝对路径）：

```sh
.venv/bin/python -m pip install -r name-that-llm/requirements-team-cpu.txt
.venv/bin/python name-that-llm/scripts/compare_team_cpu.py --train "$NTL_TRAIN" --split "$NTL_SPLIT"
# 以下两条已经执行完成，仅保留审计命令，不要重复：
.venv/bin/python name-that-llm/scripts/compare_team_cpu.py --train "$NTL_TRAIN" --split "$NTL_SPLIT" --run
.venv/bin/python name-that-llm/scripts/compare_team_cpu.py --train "$NTL_TRAIN" --split "$NTL_SPLIT" --run --resume-float32-check
.venv/bin/python -m unittest discover -s name-that-llm/tests -v
.venv/bin/python -m unittest discover -s tests -v
```

`restore_team_split.py --train "$NTL_TRAIN"` 已实际执行一次，只读取id/prompt元数据生成既有manifest；429,318字节，和原文件逐字节一致，SHA `c8c24921f617ad483b9e14cb23d0d64a0fbbde44c0214868445e8d0e30d6e1db`。新clone可用它恢复 `outputs/team_inputs/split_manifest.json` 后进行开发输入核验。它拒绝不同输入/重建hash和覆盖冲突文件，**不重开已用最终留出**。

最终工具测试共 **48项通过**（本模块42项、根修复6项），`git diff --check`通过。覆盖两处根bug、开发隔离/OOF契约、float32校验、一次恢复与预算、改善折数边界以及元数据恢复；合成测试拟合不混入真实比赛数据拟合计数。

本机证据：`outputs/team_dev_20261021/` 保留首次失败；`recovery_01/` 有4个OOF、完整summary与独立复核收据；父目录和恢复目录分别保留两次运行的源码/配置快照。共享claim `outputs/continuation_claims/team-dev-groupkfold5-20261021.json` 为completed。输出、环境及所有行级数据受Git忽略，仅上传代码、测试、计划与聚合报告。Git推送仍不等于Kaggle Notebook同步或提交。

## 2026-10-21 中心化与4060评估（已完成）

用户本轮要求继续提升分数并评估带4060的Windows电脑；结合自主执行授权，从 `7a468a8` 新建本人分支 **`exp/prompt-centering-20261021`**。训练前提交 `b34e91e2b68ac8db709335456b39e1e2bd123408` 固定[15-fit计划](docs/PROMPT_CENTERING_PLAN.md)、配置、源码和测试。队友分支与main未改动，CASMI及原工作区保留。

- 只用原120 prompt / 2,159行，沿用上一轮GroupKFold5和严格hash/身份匹配的original_style OOF；原30-prompt已用留出不训练、不评分。没有重跑旧search/finalize。
- 本轮3个模型仅改变组均值扣除比例0/0.5/1，另做固定0.25/0.5融合；分类器共15次真实拟合，全部有效、无收敛警告。耗时145.132秒，2线程，峰值RSS1.291GiB。历史253+21+本轮15=289次本模块真实拟合，不含队友其它运行和合成测试。
- 最佳开发候选：**75% original_style + 25% partial**，BA **0.348673**，对原style0.340336为 **+0.008337**，prompt配对95%区间 **[0.000926, 0.015741]**，5/5折提升，并通过预定半批/单例压力门槛。
- 相对上一轮最佳0.347281的事后增量仅 **+0.001393**，区间 **[-0.010648, 0.013889]**，2/5折改善；相对同特征不中心化融合的区间也跨0。不能把通过旧对照门槛写成相对最好已有方案已确认提升。
- 固定文本聚类在2,159验证行中只有10行进入双行簇，2,149行是单例。没有证明成功恢复题目或推理侧去除话题，不在看到结果后扩展阈值搜索。有限CPU计划已结束，登记候选而不替换冻结交付，不提交Kaggle；这是工程停止与证据不足，不是证明收敛。

完整结果、每折/每类recall、压力检查、hash及复现边界见 [PROMPT_CENTERING_COMPARISON](reports/PROMPT_CENTERING_COMPARISON.md)。70项测试通过（模块64、根6）；独立审计不导入runner、不重训，重算9份OOF/9个候选及全部指标，最大差异4.44e-16。旧holdout claim仍为completed且SHA `5a93d4efc1590e6acc63bdd3d40b3f3ebf860aa85e04d6af5bda582638c68db0` 不变。审计自身的配置序列化字节比较错误已更正并留痕，实验结果未改写。

本地 `outputs/prompt_centering_20261021/` 保存contract（规范hash `0cd3a6605134525cb428b78a4ead0d00c0c8007234a43c50ed503e05ff322e15`）、source_snapshot、9份OOF、summary和独立收据（SHA `91b4401fb32c4da160098fc6b6a9fb4a57c82dce42809cd3260d889b42dd8e25`）。新共享claim为 `outputs/continuation_claims/prompt-centering-dev-20261021.json`，completed。原始数据与行级输出不在Git；新clone也不含原baseline OOF，不能绕过hash或claim重跑本机协议。

实际命令从仓库根执行，`NTL_TRAIN`/`NTL_SPLIT`意义同上：

```sh
.venv/bin/python -m unittest discover -s name-that-llm/tests -v
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python name-that-llm/scripts/compare_prompt_centering.py --train "$NTL_TRAIN" --split "$NTL_SPLIT"
# 下条已完成，仅供审计，不再运行：
.venv/bin/python name-that-llm/scripts/compare_prompt_centering.py --train "$NTL_TRAIN" --split "$NTL_SPLIT" --run
.venv/bin/python name-that-llm/scripts/probe_gpu_environment.py --output name-that-llm/outputs/local_gpu_preflight_20261021.json
```

**4060判断：值得作为下一阶段小编码器GPU节点，无需把现有CPU工作整体迁移。** 普通4060桌面/笔记本均按8GB评估；ModernBERT-base冻结特征优先，有限微调其次，8B量化先实测，14B强集成不直接搬运。PyTorch/bitsandbytes当前官方支持Windows，现有CPU runner却用了Unix resource/定时器，复用它优先WSL2。官方来源、权重下界与具体检查命令见 [WINDOWS_4060_ASSESSMENT](docs/WINDOWS_4060_ASSESSMENT.md)。

GPU探测脚本为跨平台stdlib入口，12项风险测试通过；默认不分配tensor、不安装/下载，`--cuda-smoke`显式做小矩阵，`--output`拒绝覆盖。实际本机探测只确认Mac arm64/18GiB RAM且本环境Torch未安装；Windows目标机、实际4060可用显存、速度均未核验。没有下载神经模型或GPU训练。

下一步在4060上先做独立环境探测，再登记并冻结ModernBERT-base开发特征方案（模型完整revision、文本/行序hash、pooling/长度/层/dtype缓存契约），使用同120-prompt开发协议检验互补。先取得队友成功强基线精确版本及同协议OOF，才能讨论对其0.601记录的增量。当前Kaggle保存权限仍拒绝，没有核验配额/现行规则或绕过；Windows运行也不等于Kaggle离线Notebook复现。旧已用留出始终封存。

## 2026-10-21 一次提交准备（当前停止点）

用户随后明确要求“请提交一次到kaggle我看看你的优化成果”。按[固定计划](docs/SINGLE_SUBMISSION_PLAN.md)，从 `d0f6cbe803b4c89847a22bdb6f39f7a2f5b32549` 建立本人分支 **`exp/single-submission-20261021`**；候选与提交代码冻结于 `645be5110bf6b719b9eefa2333574fa51719666b`。授权覆盖最终训练及至多一次正式提交，不再请求同一提交授权；没有重新调参、重评旧留出或更改已有开发分数。

**尚未上传、执行或提交Kaggle，暂无本候选榜单分数。** 本轮从Chrome的比赛页再次请求读取，Browser Use自动审批审核拒绝：`www.kaggle.com`的保存权限或管理策略禁止访问。没有取得当前规则、账号/队伍、额度、Notebook版本或提交历史；没有用API/其它浏览器绕过。已请求用户解除该域名限制。解除后再走正常页面入口，新建独立私有CPU/Internet Off Notebook，挂官方比赛输入，确认成功版本后提交一次；不覆盖队友Notebook。若结果状态不明，先查提交历史，不能重复点击。

本轮交付 [candidate_blend_offline.ipynb](notebooks/candidate_blend_offline.ipynb)，SHA256 `82931fb6a28441e01b884c2ce62c16ced6f35c39bc9ff092c2f53277308b3df8`。它内嵌固定源码，75% original_style + 25% partial，不依赖Git下载或外部模型。训练数据hash固定，test按实际`id,text`读取并保留顺序，输出`id,model`的`submission.parquet`。为二次方聚类内存设置执行前行数门槛：4,096行以内，4,097行拒绝；这不是任意规模或实际隐藏test的资源保证。若在线数据超过门槛，应停止并另行评估，不能静默换模型或调整阈值。

本地仅执行一次真实候选流程，**2次分类器最终fit**，全部官方2,699行参与训练；这是固定模型最终拟合，不是留出验证。按已验证Notebook代码单元依序Python exec，**不是Jupyter内核或Kaggle运行**。公开test仅3条占位行，本地检查不说明榜单收益。总耗时30.044秒，峰值RSS1.438GiB，无警告、无数值库版本差异；从保存的两个分量独立重算融合，最大概率差0.0，类别映射、ID顺序、Parquet格式均通过。新执行claim为completed，旧holdout claim SHA仍为 `5a93d4efc1590e6acc63bdd3d40b3f3ebf860aa85e04d6af5bda582638c68db0`，未重评。累计本模块真实fit从289增至291，不含合成测试和队友运行。

工具测试共**79项通过**（模块73、根6）；nbformat结构验证通过，跟踪Notebook无执行输出。合成测试覆盖非3行输入、类别/融合、训练hash与schema、预算claim以及行数门槛。独立合成4,000条短文本聚类约0.381秒、峰值0.779GiB，没有拟合真实模型，不能当真实隐藏test性能。另生成了本地执行记录HTML预览，但浏览器访问localhost超时，**未完成视觉检查**；未影响代码单元与输出审计。

本地证据目录：`outputs/single_submission_20261021/local_reproduction/`，全部受Git忽略。

| 产物 | SHA256 |
| --- | --- |
| verification.json | `ea4491e29a302c13582e6141ec7021e2a7afbeb3365522900660d6837cb4c07d` |
| submission.parquet（3行公开占位检查） | `9975693f4e88f840597b655b0da083346f0955b766b418663261a7963bde5c00` |
| probabilities.npz | `ae7b47089db1f229e439da8d4f233ac8c85be719191f3a76c5f8e3da3e6580d4` |
| candidate_attempt.json | `f9dc368cdac606ba77ad8f6b5b0036ef05d76ecb5f9ec6058a877345fad6f4e4` |

实际命令从仓库根执行；`NTL_DATA`是本机官方raw数据目录，`NTL_HOLDOUT_CLAIM`是原工作区已用留出claim路径，不随Git分发：

```sh
.venv/bin/python -m pip install 'nbformat==5.10.4'
.venv/bin/python -m unittest discover -s name-that-llm/tests -v
.venv/bin/python -m unittest discover -s tests -v
# 下条真实训练已完成，只保留审计命令，不重复执行：
.venv/bin/python name-that-llm/scripts/verify_candidate_notebook.py --data-dir "$NTL_DATA" --protected-claim "$NTL_HOLDOUT_CLAIM"
```

新环境的检查依赖入口为 `requirements-notebook-check.txt`。Git仅上传源码、未执行Notebook、测试及汇总文档；环境、数据、权重、概率、提交文件与执行版Notebook保留本地。Git push与Kaggle上传/运行/正式提交分别记录，不能混用。


## 2026-10-23 Windows 保存与接续

用户要求先保存 GitHub，后续在 Windows 电脑继续。已从 `exp/single-submission-20261021@4760f67fd06f59c602389595caafae090606331d` 建立独立 `exp/windows-handoff-20261023`，遵循队友 exp/ 命名；不更改 main 或队友分支。新增 [Windows 交接](docs/WINDOWS_HANDOFF.md)和根目录 [WINDOWS_NEXT_CHAT_PROMPT.md](../WINDOWS_NEXT_CHAT_PROMPT.md)，更新入口；本次仅文档及候选 Notebook 的 LF 检出规则，CV 变化为 0，没有训练或 Kaggle 网络操作。

状态纠正：此前“尚未上传/运行/提交”仅指代理完成情况。用户随后手动创建带官方比赛输入的新 Notebook 草稿，并收到导入、保存运行及一次提交指导；尚未收到实际成功版本、submission ref 或评分。Windows 接续必须先核对已有状态，不能默认没有手动提交，也不能重复消耗一次授权。

2026-10-22 已只读核验 Mac 的 Kaggle 本地规则为 allow，但浏览器工具实际仍拒绝发起该域名网络请求；访问问题未修复，迁移不构成绕过授权。故障原始材料及 CLI 手动包留本机，不上传配置、凭证或完整对话。Windows 本地环境探测可独立继续；实际 Kaggle 操作须等正常限制解除。

此次检查：候选 Notebook 原字节 SHA256 与冻结值一致；新交接链接、命令中引用文件及 Git diff 做静态核验；为冻结候选添加精确路径的 .gitattributes LF 规则，并核对 core.autocrlf=true 检出字节哈希。不重跑已完成的 291 次真实拟合或已用最终留出。数据、OOF、claims、权重、环境、预测与 .local-backups 保留本地，未随 Git 迁移。Windows/CUDA 尚未运行。

保存命令（远端是否成功以工具返回和最后 SHA 核对为准）：

```sh
git switch -c exp/windows-handoff-20261023
git diff --check
git add .gitattributes WINDOWS_NEXT_CHAT_PROMPT.md README.md name-that-llm/WORKSPACE.md name-that-llm/HANDOFF.md name-that-llm/docs/WINDOWS_HANDOFF.md
git commit -m "docs: save Windows continuation handoff; CV unchanged"
git push -u origin HEAD:refs/heads/exp/windows-handoff-20261023
git ls-remote --heads origin refs/heads/main refs/heads/exp/windows-handoff-20261023
```

## 2026-10-23 Windows RTX 4060 接续

本机目录为 `D:\kaggle\name_that_llm`。从用户指定交接分支克隆，HEAD 精确匹配 `25c9a02ccd18d36391e31e15ffda268ab735dd8e`，初始工作区干净；已新建独立 **`exp/windows-4060-20261023`**。主分支、交接分支和队友分支均未改写，没有强推或自动合并。

详细检查、失败记录、材料与 GitHub 核查范围见 [WINDOWS_4060_PREFLIGHT](reports/WINDOWS_4060_PREFLIGHT.md)。计划在真实模型运行前由 `82fd612` 记录，合成 probe 和完整模型文件身份由 `861b809` 固定；[WINDOWS_MODERNBERT_PLAN](docs/WINDOWS_MODERNBERT_PLAN.md)规定唯一120-prompt开发协议、最多5次新分类器fit、对照与停止条件。原始已完成CPU搜索、最终训练及旧30-prompt留出都未重跑。

本机已确认 RTX 4060 Laptop GPU / 8,188 MiB / 驱动596.36、i9-14900HX、约31.73GiB RAM、Python3.11.6，初次D盘空闲约70.09GiB。已有Ubuntu WSL2处于Stopped；未启动或改动。独立环境在 `name-that-llm/.venv-gpu/`，不复制Mac环境、不覆盖系统Python。旧resource/SIGALRM/setitimer CPU runner仍不可直接视作原生Windows可用。

已在新环境确认torch2.8.0+cu128正常导入，CUDA12.8可用、设备计算能力8.9；既有probe的32×32 FP32小矩阵实测通过，最大绝对误差0。固定ModernBERT合成前向也已通过：4条合成文本、两次forward，特征4×3072、反序恢复差异0、峰值allocated约350.22MiB，总耗时50.931秒。完整6文件身份均已核验，claim为completed，结果SHA256为 `920300705ea14c0ca6b11adc6e9b324ed235566bce44990ecd35fc79c839db40`。12项原探测测试与7项新增身份/claim测试通过。这些工程检查不证明真实开发分数、完整吞吐或Kaggle成绩；不要重跑已完成的合成probe。

修复了迁移时可确定的字节问题：autocrlf将labels.json检出为CRLF，使原hash契约拒绝。现增加精确LF属性，并恢复原Git blob字节；标签内容不变、hash检查不放宽。冻结候选Notebook SHA256仍为 `82931fb6a28441e01b884c2ce62c16ced6f35c39bc9ff092c2f53277308b3df8`，75/25权重和4096行门槛都未改。

**2026-10-23材料盘点。** 用户说明没有另行传输、认为材料都在GitHub；只读核查当前clone、可见6个分支、数据/报告目录、Release和Actions页面，没有找到原始CSV、split、历史OOF、claims或模型。未取得完整API递归tree，范围限制已在报告记录。GitHub中的代码/报告不是这些文件本体。后续用户要求不依赖Mac，执行方案已按文末2026-10-24修订；不再要求从原机器转移历史OOF来启动单模型开发。缺缓存仍不是重跑旧实验的理由。

**Kaggle一次提交尚待正常访问和历史核查。** 用户本轮说手动状态“不确定”，要求后续代理自动提交，并明确要求本轮结束后提交一次看分数；本轮目标包含至多一次正式提交。该授权不证明旧操作未提交，也不解除原origin-policy拒绝。已请求确认原限制是否经正式设置解除及现有Notebook链接；未确认前只做独立本地工作。正常恢复后先核对Notebook版本、运行和提交历史、账号/队伍/规则/额度；未知状态不重复点提交，不覆盖队友Notebook。没有访问Kaggle、运行Notebook、正式提交或取得新成绩。

当前新增真实fit=0，旧holdout evaluation=0；历史291次fit及所有已登记开发/留出成绩保持原值。新GPU配置通过工程检查后仍需材料齐全才能实施真实开发，不能把合成前向包装为模型收益。只暂存明确代码、配置和聚合文档；数据、环境、模型、OOF、claims、安装日志与原始receipt留在忽略目录。

## 2026-10-24 只依托仓库的原生Windows开发入口

用户最新要求“直接根据GitHub仓库的东西推进，不用管Mac本地文件”。继续在本人 `exp/windows-4060-20261023` 分支实现 [windows_modernbert_dev.py](scripts/windows_modernbert_dev.py)及[修订协议](docs/WINDOWS_STANDALONE_DEV.md)：只需官方train.csv，不再要求原机器split、OOF或claims。现有恢复算法在内存重建原120/30分区并严格核对既有SHA；沿用固定ModernBERT、五折GroupKFold、折内StandardScaler和C=0.1 LR，只报告独立单模型开发指标，不进行旧OOF配对融合或候选替换。

修订与原计划共用 `windows-modernbert-frozen-dev-20261023` experiment_id及固定claim，仍然总共最多5次新fit，失败也消耗预算；不是另开五次。原生supervisor实施1,800秒deadline和8GiB进程树RSS抽样限制，forward后检查6GiB CUDA allocated；2线程。缓存绑定完整输入/行序/特征/模型/源码/依赖身份，保留失败日志与次数；旧291fits和30-prompt已用留出不重跑。

详细结果见 [WINDOWS_STANDALONE_DEV](reports/WINDOWS_STANDALONE_DEV.md)。隔离环境补齐固定Windows科学计算依赖，pip check通过。最终18项相关合成测试通过，3.299秒；独立只读审查后补齐终止异常留痕等边界。默认readiness退出2，只报告官方 `data/raw/train.csv` 缺失，模型6文件全部核验通过，未创建开发claim。新真实fit=0、新GPU forward=0、旧留出评分=0；此前成功合成probe未重跑。原冻结Notebook/labels字节SHA均未改变。

本地完整证据在忽略的 `outputs/windows_4060/standalone_*`，reconciliation SHA `c20b64a50273bb9c3dc36651e3121395ee4b5b67fbb8d739251e05b69bb50a8e`。这里只提交代码、配置、测试及聚合报告，不上传数据、权重、环境或行级输出。正常恢复Kaggle访问后可从官方取得训练文件，无需回Mac取材料；一次正式提交前仍要核查手动操作历史。当前无Kaggle运行、正式提交或新榜单分数。

## 2026-10-24 官方Kaggle访问恢复核查

用户新指示：“需要官方资料请直接从kaggle下载，现在继续推进一轮”。当前Chrome已成功正常读取官方[Overview](https://www.kaggle.com/competitions/name-that-llm)、[Rules](https://www.kaggle.com/competitions/name-that-llm/rules)与[Data](https://www.kaggle.com/competitions/name-that-llm/data)。本次没有发生旧域名策略拒绝，也没有通过API或其它渠道绕过拒绝；不要把此前历史拒绝继续当作本次实时状态。

当前页面显示Login/Sign Up，数据区域明确要求登录并确认参赛规则，尚不能取得账号/队伍、Notebook或提交历史。已打开原比赛返回地址的登录页并请求用户登录原参赛账号；未操作认证对话框、未索取密码/验证码/API key。只检查本机官方CLI及标准认证位置的存在性：CLI可用，标准凭据与认证环境变量均不存在，未读取或输出密钥。

本次官网核实：团队最多5人、每天最多5次提交、最终可选2份；外部公开资源允许但须符合可访问性要求，团队外私下共享受限制。数据页列出train.csv/test.csv/sample_submission.parquet三文件，合计约2.14MB；训练为2,699行/150prompt，公开test说明为3行，隐藏重跑约4,000行。Overview的单行示例与Data的3行表述、时间线与页头日期差异仍存在，沿用既有记录而不擅自解释。以上是网页说明，尚未下载本次文件或核验其字节。

独立只读复核冻结候选Notebook及生成/验证脚本、相关测试：SHA仍为 `82931fb6a28441e01b884c2ce62c16ced6f35c39bc9ff092c2f53277308b3df8`，官方输入路径、训练hash、75/25融合、4096通过/4097拒绝、id/model Parquet契约均保持一致，未发现确定提交阻塞，无网络或外部权重依赖。Kaggle镜像依赖版本、实际运行、成功版本、claim和隐藏测试行为仍须在线验证；没有重新运行已完成本地拟合。

截至此检查点，真实开发尚未启动，官方文件尚未下载，Kaggle运行/正式提交=0，无新分数。当前需要用户完成登录，随后继续官方文件下载与哈希核验、固定五折开发和至多一次正式提交的历史核查；不需要再次批准已授权的常规工作。

## 2026-10-24 官方数据、真实运行失败与历史提交确认（最新）

用户登录原账号并将官方ZIP放在仓库根。ZIP和3个成员的CRC/SHA全部匹配冻结记录，已首次导入 `data/raw/`。当前不缺官方数据，不需要Mac文件。原始ZIP保留并加入精确忽略规则。自动ZIP跳转曾被Chrome `ERR_BLOCKED_BY_CLIENT` 拦截，实际文件由用户手动下载；没有绕过访问拒绝。

实际执行按原120-prompt / 2,159行、同一ModernBERT配置与5fit/540batch总预算开始。首次零计算启动因Windows venv redirector父PID检查失败；修复并登记具名一次启动修正后，真实提取在收据原子替换处发生WinError5并停止。最终交叉核对为 **265次forward尝试、264次持久完成记录、0次fit、0次旧留出评分、0次全量最终fit**；无特征缓存、OOF、开发指标或新提交候选。恢复运行105.022秒，抽样RSS峰值约1.615GiB，进程已退出。

恢复入口另有默认参数路径缺陷：attempted计数误写回原claim路径，completed写入恢复claim。原失败claim路径因此被覆盖，不能声称始终原样保留；两个当前文件均另行归档。首次零计算失败JSON由启动前记录重建为单独档案，严格匹配原SHA `cdbe811ad0c7664bbfc3389e4ec862cd0e4935b96781118f1f17f3b184289738`，没有回滚/重置当前claim。原路径的started是遗留状态，不代表仍运行；两条记录属于同一次恢复执行，不能相加。**本轮及具名恢复入口均关闭，禁止删除claim、换目录或再次提取来解锁。**

已修复监督者身份、Windows暂时占用下的收据原子替换、当前claim路径解析；永久写入拒绝仍停止。最终30项相关测试通过，9.793秒；包含真实Windows venv与临时文件锁回归，全部无需比赛数据/GPU。修复后没有重跑真实实验。完整事实、局限及命令见 [WINDOWS_ROUND_20261024](reports/WINDOWS_ROUND_20261024.md)。原Notebook SHA、标签、75/25权重及4096门槛均未改，旧CPU及旧留出未重跑。

Kaggle账号zephyrpong、队伍zhouruijie已核实。[本账号NTL Blend 20261021 Version1](https://www.kaggle.com/code/zephyrpong/ntl-blend-20261021?scriptVersionId=360768689)正式状态Succeeded、public **0.351**；其保存运行claim=completed、fit=2/2，Output全部provenance字段与本地冻结候选元数据一致，但未核对远端完整Notebook字节SHA。[队友ntl-ensemble-600 Version1](https://www.kaggle.com/code/zdx13143/ntl-ensemble-600?scriptVersionId=360258830)已有Succeeded、public **0.601**。这些是既有结果；本轮新增Kaggle运行/提交=0，未重复上传0.351版本，未按公榜调参。

本地证据在忽略的 `outputs/windows_4060/` 与两次实际输出目录。`round_reconciliation_20261024.json` SHA为 `c6c47947074b6c0c58781ae595a022952b59cf673e88e0310df5c88995c7de18`；在线核对收据SHA为 `ff4294e7ce1cef7e0ce577417e08e5d07d2170d12f6a580e2452596b3c08ac68`。只推本人分支的明确代码/配置/测试/聚合文档，原始数据、日志、claims、环境和模型不入Git。后续新一轮先登记新的预算与停止条件，保留本轮失败与265次尝试，不能把未完成说成通过。

## 2026-10-24 独立第二轮已完成，未达到提交门槛

用户明确授权登记独立下一轮、保留失败、不重开旧留出，五折完成后有合格候选才提交一次。预登记、独立入口和测试已在真实运行前提交并推送为 `42a444c`。固定新ID `windows-modernbert-frozen-dev-20261024-r2`，原120 prompt / 2,159行、同一冻结ModernBERT配置与LR，不复用旧claim或旧计算预算；最多540个batch、5次CV fit，条件最终head最多1fit但仅在门槛通过后启用。

本轮实际 **540/540 forward、5/5 fit，supervisor completed/exit0，总耗时67.288秒**。pooled BA **0.2899042951**；五折为 **0.2962962963、0.2824074074、0.3125000000、0.3009259259、0.2575483092**。整prompt描述性95% bootstrap区间 **[0.2704743425,0.3088898615]**。预登记要求pooled≥0.36且每折≥0.30；pooled及3折未通过。因此最终head fit、新Kaggle上传/运行/正式提交均为0，无新榜单分数，不修改门槛或追加参数搜索。

33项相关合成测试通过，19.348秒。真实特征/OOF已保存，独立重算BA一致；主线程复核原开发行与缓存/OOF逐项身份、源码快照、哈希、资源与claim完成状态。抽样RSS峰值约1.806GiB，CUDA allocated约350.219MiB，监督进程已退出。再次只读readiness正确退出2拒绝已有claim/output，未增加计算。旧30prompt评分和全量最终fit均为0。

两个旧claim哈希与第二轮启动前完全一致，第一轮265/264、0fit失败及原路径覆盖的法证/重建档案均保留。冻结75/25 Notebook与labels哈希也未变。本轮开发数据多次用于探索，且无历史逐行OOF，不把0.289904与历史0.348673的聚合比较写成配对显著性；既有0.351/0.601仍分别属于历史正式结果。

详情见 [第二轮报告](reports/WINDOWS_ROUND2_20261024.md)，协议见 [预登记计划](docs/WINDOWS_ROUND2_PLAN_20261024.md)。本地产物 `outputs/windows_modernbert_dev_20261024_r2/`；最终收据 `outputs/windows_4060/round2_final_reconciliation_20261024.json` SHA为 `3ccd02e9ee6659bb668a5a22b080023b939901c6052baada8ae75b228c7f49a1`。本轮已关闭，不重跑`--round2 --run`，不删除claim；只将代码/配置/测试与聚合文档推送本人分支，原始数据、OOF、模型与环境保持本地。

## 2026-10-24 第三轮完成与根目录学习笔记

用户授权分析第二轮、结合GitHub队友方案登记新预算执行，同时要求后续根目录维护Markdown和HTML，让AI初学者理解项目。实时核对远端main仍为`692ac87`，参考代码无diff。开发诊断表明冻结表示中题目结构明显，12.737%文本超过256token，但现有分组准确率不支持把截断断定为主要原因；未做额外编码器前向。完整[诊断](reports/MODERNBERT_ROUND2_DIAGNOSIS_20261024.md)区分事实和解释。

独立第三轮`windows-prompt-gaussian-dev-20261024-r3`的[预登记](docs/WINDOWS_ROUND3_PLAN_20261024.md)、实现及6项合成测试已在真实拟合前提交并推送为`c85a5a4`。借鉴队友共享prompt Gaussian的统计思想，固定训练折StandardScaler、同题残差PCA128、4次双向均值与收缩协方差；未知题目逐行边缘预测，不接收验证prompt_id、不聚类或强制唯一分配。复用第二轮已核验的原120prompt特征和fold身份，没有复现完整0.601集成。

实际新增encoder forward=0、classifier fit=5/5；supervisor completed/exit0，35.15199秒、抽样RSS峰值610803712字节（约0.569GiB）。BA **0.2986772487**，五折0.296296/0.284722/0.305556/0.314815/0.291868。对第二轮配对点增益**0.0087729536**、描述性95%区间**[-0.0069917101,+0.0249690083]**，3/5折提升。绝对BA、最低折、配对点增益、区间及改善折数门槛均未达，故条件最终fit、新Kaggle上传/保存运行/正式提交均为0。未加搜参数或改门槛，无新公榜分数。

独立审查重算OOF及保存分类头推理，最大误差约2.11e-15；主线程核对真实开发行、缓存、源码快照、fold/labels、hash、supervisor退出与预算。三个先前claim逐字节哈希不变，第一轮失败与法证保留；旧30prompt评分和全量最终fit均为0。本轮入口已关闭，再次默认readiness退出2拒绝已有输出，不再次运行。产物`outputs/windows_prompt_gaussian_20261024_r3/`；最终收据`outputs/windows_4060/round3_final_reconciliation_20261024.json` SHA **`7ed26d4d23961379d5eefd403740c868c58a1295fce9b3c8667cc4dd7cb0cf9e`**。逐类召回、条件和身份见[实际结果](reports/WINDOWS_ROUND3_20261024.md)。

根目录新增[PROJECT_LEARNING.md](../PROJECT_LEARNING.md)为文字源，[PROJECT_LEARNING.html](../PROJECT_LEARNING.html)为离线阅读版，根`scripts/render_project_learning.py`仅重新生成页面，不读取训练数据。内容含项目时间线、任务与验证流程、第二轮原因、第三轮方法、BA计算器、术语及练习；虚构例子和实测指标明确分开。**以后每轮同步维护这两个文件**，记录问题、预算、实际动作、结果、停止原因与证据，并解释本轮学习点。不要只更新技术报告或只交付分数；历史CV与Kaggle分数不混比。

## 2026-10-25 第四轮监督微调完成，性能下降（历史）

用户授权继续一轮，达到门槛的进步候选才提交一次Kaggle。第四轮计划、配置与训练实现于真实执行前推送为`78d529d`。新ID `windows-modernbert-finetune-dev-20261025-r4`，固定原120prompt/2159行、原五折身份，每fold从同一ModernBERT-base重新初始化分类头并全参数微调4epoch；无外部数据、无外层验证选epoch。9项合成CPU测试和独立训练数学/预算边界静态审查通过。第四轮自己的人工文本反向资源预检计入新预算，没有重用旧probe额度。

真实结果：**5/5 fit完成，9183/9183逻辑forward、8641/8641backward、1081/1081optimizer step**；supervisor completed/exit0，耗时2031.168秒。抽样进程树RSS峰值2.911GiB，CUDA allocated/reserved峰值2.623/2.729GiB。pooled BA **0.2648576097**；五折 **0.2731481481、0.2569444444、0.2847222222、0.2384259259、0.2711352657**。对第三轮配对点增益 **-0.0338196390**，2000次整prompt描述性95%区间 **[-0.0541888971,-0.0143334118]**，0/5折改善。完整性通过，但全部性能条件失败。

独立OOF核验与主线程`verify_round4_results.py`复算一致；源码快照、五份receipt/checkpoint、行/标签/fold身份、概率、逐类召回、混淆矩阵、bootstrap、计数与资源已核验。四个历史claim字节哈希不变，原失败与法证保留。默认readiness返回blocked拒绝已有输出。最终复核收据为`outputs/windows_4060/round4_final_reconciliation_20261025.json`，SHA **`f50c3e0d25d2926d5f7013da591d734f542408979b6f7b659824a475a19b5cea`**。真实产物在`outputs/windows_modernbert_finetune_20261025_r4/`，完整指标和18类对照见[第四轮报告](reports/WINDOWS_ROUND4_20261025.md)。

成绩产生前已准备并推送条件最终入口`windows_modernbert_final_round4.py`（`660460a`），5项合成准入测试通过。真实准入检查因gate失败返回blocked，final claim/output不存在。**条件最终训练=0，旧30prompt评估=0，全量2699行fit=0，新Kaggle上传/保存运行/正式提交=0**。没有因loss下降就宣称进步，没有修改门槛、追加参数搜索或用旧0.351候选代替提交。本轮再次只读确认历史本账号Notebook仍显示public0.351；不存在第四轮新公榜分数。

根`PROJECT_LEARNING.md`与HTML已同步三轮BA对照和本轮失败解释，增加监督微调、loss/交叉熵、epoch、梯度累积，以及“训练loss下降不等于新题目识别更好”。本轮结论仅淘汰这套固定配方，不能从单次组合变化断定过拟合/欠拟合或所有微调方法无效。后续新实验仍需独立登记，不得重跑已结束claim或重开旧留出；全部原始输出、环境和权重保持本地，Git只发布本分支源码、测试及聚合文档。

## 2026-10-25 第五轮完成：风格对照提升，固定融合失败（历史）

用户要求参考队友继续优化一轮。亲读根EXPERIMENTS及相关Notebook源码后，登记四视角stylegram LinearSVC与第三轮Gaussian OOF固定80/20融合；纯风格仅作对照。运行前配置、源码、计划与测试已推送为`65d7e54`。未采用历史150-prompt调优融合权重，未复现完整0.601大模型集成。原style/partial完整OOF本地缺失，不能与其历史聚合分数作配对显著性比较。

第五轮`windows-stylegram-fusion-dev-20261025-r5`完成CPU五折5/5次fit，89.948856秒，进程树RSS抽样峰值534601728 bytes（0.498GiB），新增编码器forward=0。每折仅训练折拟合TF-IDF与SVC；真实120prompt/2159行和R2折身份不变。**风格对照BA 0.3593253968；融合候选BA 0.3236811391**。融合相对第三轮+0.0250038904，配对95%区间[+0.0175925926,+0.0328703704]，5/5折改善；但相对风格对照-0.0356442577，0/5折改善。pooled≥.36和不得弱于风格对照两项失败，本轮关闭。

风格对照相对第三轮+0.0606481481，描述性95%区间[+0.0384492702,+0.0828817499]，5/5折改善。对已保存OOF诊断发现风格softmax平均最高分0.0842，语义分数0.4933；融合在1969/2159条上跟随语义首选。当前固定融合相对风格纠错290条、改错367条。分数尺度是有依据的后续研究方向，但本轮未再试温度/权重、未做校准fit，不能声称问题已解决。

8项合成测试通过；独立静态审计后补强worker自身父进程/预算守护及子进程终止确认。真实OOF、源快照、五个模型哈希、每类召回、混淆矩阵、bootstrap、固定融合及配对增益由`verify_round5_results.py`重算通过；五个历史claim哈希保持不变。默认readiness返回blocked拒绝已有claim/output。产物`outputs/windows_stylegram_20261025_r5/`，额外诊断`outputs/windows_4060/round5_diagnostics_20261025.json`，完整指标与哈希见[第五轮报告](reports/WINDOWS_ROUND5_20261025.md)。

**条件最终fit=0、新Kaggle上传/保存运行/正式提交=0、旧30prompt评分=0。** 风格对照没有被事后改选为提交候选；其BA也低于.36，不能四舍五入判通过。条件部署入口/Notebook未构建，因开发门槛已失败，不需要启动该阶段。根学习Markdown/HTML同步新增TF-IDF、n-gram、LinearSVC、softmax、校准和“权重比例不等于决策影响比例”的解释。后续若另开一轮，需重新登记候选和预算；本轮不自动延伸搜索，不修改已冻结源文件或claim。

## 2026-10-25 第六轮：开发与部署通过，Kaggle正式成功0.365（历史）

用户要求继续推进，有成果上传Kaggle查看分数。登记一个新候选：两个模型来源分别在外层训练集内做三折OOF温度校准，再固定80/20融合。开发预登记`ef18f9d`，条件最终入口`1a85d36`，部署四次人工检查合约`06253ba`均在相应真实执行前推送。不得重跑已结束开发/最终claim。

开发30/30模型fit、10/10温度fit，BA **0.370895580454404**；五折[0.3726851852,0.3680555556,0.3750000000,0.3981481481,0.3407809984]。相对第三轮+0.0722183318、配对95%区间[+0.0500076507,+0.0940006419]；相对R5风格+0.0115701836、区间[+0.0041627762,+0.0189814815]，均5/5改善。7项gate均通过。190.287109秒、RSS0.508GiB、新开发编码器前向0。主线程与独立数组核验一致；原开发集重复使用，非新独立holdout。

条件最终仅2159行/120prompt：三折内部6模型fit+2温度fit，再2个最终模型fit，合计8模型+2温度。70.490101秒、RSS0.788GiB，导出回读误差0。最终温度style=0.1369934354093213、Gaussian=1.8974942665741952。全2699行fit=0、旧30prompt评分=0。最终summary SHA `f009f8a592accd6933383cbd1fa46edb2fc5c8bc950a805ae6982cc4b2ae017c`。

部署本地核验已完成：`inference_round6.py`只加载冻结模型；隔离Linux CPython3.11.17 + Torch2.8.0+cpu/Transformers4.56.2。恰好4次人工文本前向全部通过，无追加/重试；Win CPU对CUDA最大概率差0.0047747640、Linux CPU差0.0047748918，均≤0.005且argmax一致，Linux倒序差0，跨平台SVC差0。部署代码与输出契约经独立审计，3项合成契约测试通过。最终955MB包在Linux临时目录完整解压，固定依赖导入通过，复用已保存人工特征的分类头预测差0，无新增编码器前向/训练。

冻结交付：`notebooks/calibrated_round6_offline.ipynb` SHA `74a2fb8762117ea9a4e184c457d180087829d171d34dfde41c30e8b1b964ecf1`；本地`outputs/kaggle_r6_package/ntl_r6_payload.tgz.bin` 954,912,317 bytes，SHA `6d3798ba4b82ae69d049f131f373ff6629002b758d4e348a73d18e9d2cbde153`。Notebook与迁移脚本已推送`ff39b2d`，包受Git忽略；包不含原始数据、开发OOF或凭据。所有真实训练、四次前向、打包与迁移入口均已结束，不重跑。

Kaggle新私有资产[NTL R6 Assets 20261025](https://www.kaggle.com/datasets/zephyrpong/ntl-r6-assets-20261025) Version 1已创建，同一冻结Notebook已导入并挂载官方比赛输入，确认CPU/None、Internet Off、Private、No persistence。[Notebook Version 1 / 362038537](https://www.kaggle.com/code/zephyrpong/ntl-calibrated-r6-20261025?scriptVersionId=362038537)命名`R6 calibrated frozen offline v1`，06:34:23 UTC保存运行，实际成功96.5秒。宿主Python3.13.15、隔离推理Python3.11.17；公开占位test3行、1次前向、模型/温度fit0、旧留出评分0，合法813-byte parquet SHA `dfbd16bb00885281f680204ab9508fbf9f10401487ff1896f388562ff732bb18`。这不是隐藏test评分证据。

06:39:47 UTC从上述版本正式提交一次，07:08 UTC在[Submissions](https://www.kaggle.com/competitions/name-that-llm/submissions)核实同一版本状态**Succeeded、Public Score 0.365**。提交时间由详情页确认，07:08是观察时间，不是平台提供的完成时间；未获取隐藏运行逐batch日志、确切行数或耗时。本账号历史0.351到本次0.365，显示分数增加0.014；不同配方不能据此单独归因于温度校准。队友0.601仍为更高的历史成绩，private分数未知。新保存运行=1、新正式提交=1，预算均已消耗；不重复提交、不据公榜调参。≤4096测试行/9小时限制、最终选用checkbox和历史Notebook保持不变。

正式结果的本地证据为`outputs/windows_4060/round6_kaggle_public_score_20261025.jpg`与`round6_kaggle_submission_visible_state_20261025.txt`，与保存运行日志分开保留。此次用户授权的开发、部署、一次正式评分链已完成；后台评分跟踪在收尾后关闭。后续新实验需要另行登记问题、预算与停止条件，不能重开既有claim。完整指标、18类开发召回和证据路径见[第六轮报告](reports/WINDOWS_ROUND6_20261025.md)。

此前文件选择器两次超时，后续明确要求Chrome扩展启用“Allow access to file URLs”；编辑器曾持续加载并显示reCAPTCHA连接提示。用户06:29:22 UTC确认权限已开启，重新连接后实际上传、导入与执行成功。这些失败记录保留，但不再是当前阻塞。未自动解决CAPTCHA，未改变模型或追加人工前向。

## 2026-10-26 第七轮：借鉴0.601外部数据分支，权重4增量gate失败（开发阶段）

用户要求“参考队友0.601分的版本继续做提升”。本人分支仍为`exp/windows-4060-20261023`；没有合并或写main、没有修改队友Notebook。实际参考已下载锁定为`zdx13143/ntl-ensemble-600`版本360258830，页面Public Score 0.601、T4×2、3小时25分35秒。Notebook SHA `d9f0ccd827ef224784b0375c2e6721e2ed711962531a27854225c6df8db01fc1`、代码SHA `7ab8127fe82378dd6aad9d08c440b3bc491e1dc8eec9367bfed67a47748dcd56`，与本地社区参考代码一致。其“尚无分数”Markdown是旧描述；实际版本没有旧笔记所写的破折号硬禁用与Hungarian，根EXPERIMENTS已补注更正。

只选择其中外部语料+四视角SVC组件做独立验证，不将已经全2699行训练的LoRA伪装成折外模型。原34,940条pool仅审计；固定14来源原始版本后，排除说明不足的两源、2700条train_regen、170条pilot及质量不合格/重复/复制题组，形成40,261条/18类池，SHA `516f1d9b73d02eb52414e68e2e52d7b7b03c7b7c31413056539f42aba93d3e6e`。每折标点规则仅从官方训练折产生，五折外部量38645/38724/38724/38724/38724。原120prompt/2159开发行及折身份不变。复制检查使用全部开发文本无标签排除重叠外部题组，因此结果只属于固定去重池下的开发CV；上游来源选择、多轮开发与未知语义重叠限制保留。旧30prompt不训练不评分。

唯一对照为官方/外部权重1/1，候选4/1，其余共享。预登记`c0a027c`在fit前推送。首次启动因`psutil.exe()`丢失盘符而在任何fit前失败，原claim SHA `00a3029371b1ab3f65c7fb52c15fd446fb46b697db71ba662e27cc42469534ef`，计数全0，2.485466秒；最小无数据复现排除`-u`原因。唯一启动修正`0d868fe`改用原生QueryFullProcessImageNameW，仍严格核对父PID/创建时间/唯一venv中转与未用claim，模型数据门槛不变，剩余监督3597秒，原claim及快照未覆盖。

成功运行ID为`windows-external-weighting-dev-20261026-r7-startup-1`，输出`outputs/windows_external_weighting_20261026_r7_startup1/`；**10/10分类器fit、20/20词表fit、0编码器前向，892.042534秒，RSS峰值4.353306GiB，退出0**。权重1对照BA **0.4454831932773109**；权重4候选 **0.44593837535014**，五折[.4236111111,.4351851852,.4490740741,.4629629630,.4589371981]。候选对对照仅+.0004551821，配对区间[-.0060400005,+.0069444444]，3/5折改善；对R6+.0750427949，区间[+.0505016795,+.1000585512]，5/5改善。相对旧方案的大幅观察增益不能归因于权重4或外部数据单一因素。两组改变139条预测，候选纠正31条、改错30条，净多答对1条。

绝对BA≥.42、每折≥.35与对R6增益≥.02通过，但对本轮对照增益≥.01、配对区间下界>0、至少4/5改善均失败；**本轮关闭，不做最终fit、不改选对照、不降低门槛或追加权重搜索**。`scripts/windows_external_final_round7.py`已准备且实际返回blocked；final claim/output均不存在。13项针对性测试、主线程保存OOF与scores逐元素/指标重算通过。`scripts/verify_round7_results.py`仅复核已有文件，不读原始文本、不fit；缺少本地claim/输出的新clone不得据此重跑实验。

主证据：summary SHA `577fd5c9bdbdc1c3a314f37eb7cf56c45353b119a74369f1797aa7fe7cc4ff2f`；supervisor SHA `b23cfe0904bef760d203514d601b05ea03c08176c37e1c2e407a411c414b860e`；verification SHA `6e4895e31b7cca80570f953c4f94ee533861b7101128616cc4ce6356eade65f3`；candidate OOF SHA `4d9b7b2dff1a2f203238cca28b3295490a2986afc0c4b6f8d6c404e21fdecf3c`。9个历史受保护claim字节未变。所有数据、模型与运行输出留本地，不入Git。

独立只读数组重算已确认BA、折身份、配对bootstrap、正确963/962条、31纠正/30改错及9个历史claim哈希；独立文档核对未发现R7数字或边界冲突，模块README的旧“当前状态”已改为历史快照，当前分支名已更正。

本轮新Kaggle资产上传/保存运行/正式提交均0；仍只有第六轮历史正式public 0.365，队友0.601是另一完整方案的公榜，不与本轮CV直接比较。后续若继续，应针对公平可验证的强基线另登记新问题与预算；本轮不追加实验。完整18类召回与命令见[第七轮报告](reports/WINDOWS_ROUND7_20261026.md)，根PROJECT_LEARNING.md/HTML已同步外部数据、权重及验证局限的教学解释。

## 2026-10-26 用户明确要求提交一次：正式Succeeded，Public Score 0.414

06:44:49 UTC用户明确要求“请你提交一次到kaggle我看看分数成果”。该请求仅覆盖同一权重4候选的必要最终拟合与一次正式评分，原gate=false、开发结果及全部旧claim保留。授权与预算在fit前推送`b5d3516`，不改选对照、不加新参数搜索，原条件最终入口仍拒绝。

新的`windows-external-weighting-requested-final-20261026-r7`完成4/4词表fit+1/1SVC fit，2159官方开发行/120prompt加38,724外部行，241.078303秒、RSS峰值4,690,862,080字节。旧30prompt训练/评分、全2699行fit及编码器前向均0。模型SHA `5667dcdc58cb51bf71fdd5b96aa8dba35c0dc43771388a44521220562c398a8c`，最终claim SHA `3972a43cf20e282e53c99f4e8bc4d5e4036edb4eb434d1ed4c5bf2cbb4534917`。

Linux确切payload迁移验证四条人工文本、六次预测调用误差0，0训练/编码器/真实数据评分。包210,522,119字节，SHA `19e424acf23278c366014bdc4cdaf4bede928b13e6c68e2ebd383a67edf6fadc`；Notebook SHA `01b7a9756ade69b607b4bfb5672f275e756fb6bdfc46e9c630084271cc13260a`，已推送`51e6bb6`。19项R7测试通过，独立收据核验通过；canonical contract对象hash与文件字节hash分开记录，不混用。原summary/打包receipt的旧阶段状态不回写，后续收据另存。

私有[NTL R7 Assets 20261026](https://www.kaggle.com/datasets/zephyrpong/ntl-r7-assets-20261026)已创建成功，版本21251756。编辑器曾持续Loading，用户确认手动恢复后复用同一草稿，成功导入冻结Notebook并挂载官方比赛/私有资产。[Notebook Version 1（362398792）](https://www.kaggle.com/code/zephyrpong/ntl-weighted-svc-r7-20261026?scriptVersionId=362398792)名为`R7 frozen weighted SVC offline v1`，Private、CPU/None、Internet Off；唯一保存运行39.0秒成功。3行公开占位test输出808-byte Parquet，SHA `aeb717f50c7b150da2f8d5c3fbe423fbf504cd94e6bd084004740d596fb3371e`；模型fit/编码器/旧留出评分均0。此前Loading及reCAPTCHA连接提示保留为历史失败，不再阻塞；未自动解CAPTCHA，原因未证实。

约07:34 UTC从该版本正式提交一次，07:37:16 UTC观察到**Succeeded，Public Score 0.414**。观察时间不等于平台完成时间；隐藏测试行数/耗时未取得，不由3行/39秒推测。相比R6的0.365提高0.049（4.9个百分点），仍低于队友完整0.601；权重1对照没有正式提交，不把公榜提升归因于权重4。最终选用checkbox未调整，队友`ntl-e642`独立运行未操作。保存运行与正式提交预算各1/1已耗尽，不重复提交或据公榜调参。证据见[本次提交报告](reports/ROUND7_REQUESTED_SUBMISSION_20261026.md)和`outputs/kaggle_r7_submission_20261026/`中的ledger、`public_score_0414.jpg`、`public_score_visible_state.txt`与保存运行日志。根学习Markdown/HTML已同步；本次明确请求已完成，原开发gate=false和全部历史claim保持。

## 2026-10-26 第八轮：分析队友0.642，固定概率平衡未确认增益（当前）

用户要求分析队友最新高分并继续推进。已从实际Kaggle页面确认`ntl-e642` V2/362382146 Succeeded、public **0.642**；收尾时`ntl-e643` V1/362503121仍Notebook Running，未把其描述0.643当已评分。下载源码SHA与Notebook版本绑定；三个源码探子分别核对分组/readout、spectral/ATLAS、Qwen/调度，主线程亲读说明并点验关键逻辑，另一个探子定位本地可复用OOF。Git main仍692ac87，不修改队友Notebook或分支。

0.642是多专家/去题/非线性读出/选择性ATLAS/全量query双GPU27B及最终Sinkhorn的完整方案；full-fit银行与adapter使用全部2699官方行，不能充当本地开发OOF。公开3条exact preview的23.8秒不代表隐藏推理，源码RuntimeBudget未真正实施超时。实际逻辑、引用行号和资源边界见[第八轮报告](reports/WINDOWS_ROUND8_20261026.md)。

预登记`0eb0600`只检验固定Sinkhorn：原R7权重4 SVC OOF、2159行/120prompt、每外层折独立300次列/行归一化，50行阈值、温度1，无标签或真题组输入。首启动直接父PID检查遗漏venv redirector，0变换/0评分即失败；原claim永久failed且SHA锁定。`52d42b1`另登记唯一无数据启动修正并推送，复用已测原生父链检查、PID/创建时间/未用计数，算法和门槛不变，剩余119秒。

成功ID`windows-frozen-svc-sinkhorn-dev-20261026-r8-startup-1`：10/10变换、1.312秒、RSS94,420,992 bytes、exit0；失败0.75秒也计入原120秒总预算。BA **0.44593837535014→0.4473117024587613**，净多正确3条；346条变更、93纠正/90改错。配对增益0.0013733271、固定转导预测下描述性95%区间[-0.0106481481,+0.0129815398]，2/5折改善、一折持平。绝对条件通过，增益、区间下界、改善折数三项失败，gate=false。

7项合成/真实venv进程测试通过，主线程独立保存数组重算、完整性和原12个claim/input哈希核验通过；没有追加真实变换/模型fit/encoder/旧留出。summary SHA `93ef68efc07db703c4e349e8e1394d617171a5abf97fbd88a34f0bb5edca0158`，verification SHA `dae58317d911900a72ee99cbba371d7bbc99f027adbd1f9889c806ff97ee144c`。输出`outputs/windows_sinkhorn_20261026_r8_startup1/`和原失败目录均保留，新clone缺少它们也不能重开。

R8关闭，0最终fit、0新Kaggle上传/运行/提交，本账号正式仍0.414。下一优先项是先诊断文本题组覆盖率，再另登记训练折内残差/非线性readout；旧中心化几乎全单例的问题不能跳过。完整强基线需要同协议fold资产或明确新的重建资源，不能用full-fit权重冒充OOF。不根据`ntl-e643`待定名称调融合权重。根PROJECT_LEARNING.md/HTML同步解释Sinkhorn、转导批次、概率质量与硬类别计数的区别。

## 2026-10-28 第九轮：词视角半中心化五折改善，幅度门槛未过（当前）

用户要求继续推进一轮。远端只读核对main已到`dc62708e7116af472257d8f9159424fec2078f01`，提取e651 Notebook，SHA `d87c8394b7ad8b3cff1d7aa20687699a29d08ebb92eab684dfe5b1e61c4dc835`。Kaggle页面实际确认e651 V1/362577633 Succeeded **0.651**，e651b V1/362920738为0.649，e643 V1/362503121已0.643；本账号R7仍0.414。Git源码SHA与线上版本分数分别存证，没有假称逐字节绑定。v5 fold adapters缺少与本地120prompt折映射，不能据目录名直接用于本地OOF。

新ID`windows-topic-word-dev-20261028-r9`，预登记和代码`536f59f`、字节属性`303fce3`在真实运行前推送。复用R7每折词表及固定池/选择，仅word减0.5倍2..18行组均值，训练用已知训练题组、验证只用文本TF-IDF+cap18/相似度0.15聚类，另外三视角和SVC4/1权重保持。原120prompt/2159行和GroupKFold不变，旧30prompt不训练不评分。正常批次、反序、两半合计20次分组；单例为固定压力检查，不事后改选。

实际完成**5/5次SVC fit、20/20分组，800.376434秒、进程树RSS峰值5,057,384,448 bytes（4.710056GiB）、exit0**。主BA **0.4542794895736073**，对R7 **+0.0083411142234673**、描述性95%区间[+0.0009336096,+0.0153014106]、**5/5折改善**。正确981/2159，对照963；120条变化，35纠正/17改错。半批BA0.4510348584、单例0.4431489262。8项数值条件中7项通过，唯一失败是增益≥0.01；不能因区间下界为正或只差0.001659就降低门槛。

分组共332簇、90条单例、2069条可中心化；12,902个推断同组文本对全部真同题，pair precision1.0、recall0.7033745843。各折ARI约.764–.860。当前主要是同题碎裂，未观察跨题误合并；这只是开发批次诊断，不保证隐藏测试。训练外部池约一半行因组>18或单例保持原样，且单条推理收益消失，后续可据此另登记残差信息利用/批次稳健性问题，不在本轮追加搜索。

9项新合成测试+1项真实Windows venv父链测试通过。两个静态探子核对数学和runner；主线程`verify_round9_results.py`不读文本、不加载模型、不重新分组，从已有数组复核全部指标、scores→OOF、group diagnostics、预算、源快照、14个旧claim及22个输入哈希。默认readiness已实际blocked。输出`outputs/windows_topic_word_20261028_r9/`及claim不得重跑；缺输出的新clone不能重新开启。

summary SHA **`cb220a45fdd32b01c3a593159bdbdad362a2ac96b92fdfd95b32b6ed8d2b1cef`**，verification SHA **`dc103feda0d1af769e7c8d84a2ae2ad6da526a3e4ae26068173e00d1e5e636c2`**。**0新监督词表fit、0encoder、0旧留出评估、0最终fit、0Kaggle上传/运行/提交**；仅本地开发增益，没有新公榜分数。完整18类召回、逐折/批次诊断、来源与命令见[第九轮报告](reports/WINDOWS_ROUND9_20261028.md)，根学习MD/HTML同步讲解词残差、precision/recall和批次依赖。
