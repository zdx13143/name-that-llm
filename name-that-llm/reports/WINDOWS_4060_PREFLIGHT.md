# Windows RTX 4060 接续检查

日期：2026-10-23（Asia/Singapore）。本报告只负责 Name That LLM。原始本机收据在 `outputs/windows_4060/`，受 Git 忽略；本文件只保存环境摘要和操作边界。

## 仓库与冻结文件

- 本机仓库：`D:\kaggle\name_that_llm`；用户指定目录原先为空，按明确交接分支克隆。
- 初始 HEAD：`25c9a02ccd18d36391e31e15ffda268ab735dd8e`，与已核验交接提交一致；初始工作区干净。
- 当前独立分支：`exp/windows-4060-20261023`。没有修改远端 main、交接分支或队友分支，没有 force-push、merge 或自动提交 Kaggle。
- 预登记提交 `82fd612`；合成 probe 与完整文件身份固定于 `861b809`。这两项发生在新模型运行之前。
- 候选 Notebook SHA256 与冻结值 `82931fb6a28441e01b884c2ce62c16ced6f35c39bc9ff092c2f53277308b3df8` 一致；固定 75/25 模型及 4,096 行门槛未改。
- `core.autocrlf=true` 初次检出使 labels.json 变成 CRLF，文件 SHA 为 `72bab5b1ab11c89431bdb1aca22892d392ade42ee66e6a95f651ce3341251ab1`，不能通过原数据契约。已增加**仅该文件**的 LF 属性并恢复现有 Git blob 字节，原 SHA `d8c9dcad5979236fb0e5f5504dd2e53c676123bf75cbf06faa13bc1e86524b9d` 恢复；标签内容和既有 hash 检查不变。

## 初始硬件与软件实测

初始 stdlib probe 时间为 `2026-10-23T13:03:12Z`；CIM 查询时间为 `13:03:51Z`。容量使用 GiB，瞬时空闲值不保证下一次相同。

| 项 | 实际检查结果 |
| --- | --- |
| OS | Windows 11 家庭中文版，10.0.26200 / build 26200 |
| CPU | Intel Core i9-14900HX，24 核 / 32 逻辑处理器 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| VRAM | 总计 8,188 MiB；初次 nvidia-smi 空闲 6,374 MiB |
| 驱动 | 596.36 |
| 系统内存 | 34,070,192,128 bytes，约 31.73 GiB；初次可用约 14.46 GiB |
| C 盘 | 初次可用 71,236,595,712 bytes，约 66.34 GiB |
| D 盘 | 初次可用 75,256,102,912 bytes，约 70.09 GiB |
| Python | 实际 python.exe 为 CPython 3.11.6；pip 26.0.1 |
| launcher | py.exe 存在，但 `py -0p` 返回 No installed Pythons found；后续用已确认的 python.exe/独立 venv 绝对路径 |
| 初始模型依赖 | Torch、Transformers、HF Hub、nbformat 未安装；CUDA 实际运算当时未知 |
| 已有 WSL | 默认 Ubuntu，版本 2，状态 Stopped；本次没有启动、安装或修改发行版 |

初始系统已有 numpy 2.4.3、scipy 1.17.1、scikit-learn 1.9.0、pandas 3.0.5、pyarrow 25.0.1；不把这些当 Mac 历史实验的相同依赖，也不覆盖系统环境。新依赖只写模块 `.venv-gpu/`，已加入忽略规则。

## 权限失败及实际核验范围

首次沙箱 Git clone 在 credential-manager 的 MSYS 命名对象访问上失败；正常权限重试同一用户指定的 clone 成功。沙箱 CIM / WSL 查询返回 access denied，保留初次无效 `host_inventory_initial.json`（null，不用于结论）；正常权限只读查询成功，证据为 `host_inventory_normal_permissions.json` 与 `wsl_inventory.json`。这不是 Kaggle 访问恢复，也未绕过 Kaggle 拒绝。

新增纯合成文件测试首次在沙箱临时目录的 `os.replace` 遇到 WinError 5；正常权限运行原保护逻辑通过。未通过降低 claim 保护来绕过该测试失败。实际模型 probe 使用正常权限运行，以免沙箱的原子写限制破坏执行记录。

## 材料盘点与 GitHub 核查

用户确认没有单独传入材料，并认为所有东西都在 GitHub。检查当前 clone：原始 train/test/sample、split manifest、历史 OOF、claims、CPU 模型和神经权重均缺失；初始 outputs 只有本轮环境收据。数据缺失不授权重跑原 291 次拟合或重新使用 30-prompt 留出。

远端只读网页核验列出 6 个分支：

| 分支 | tip |
| --- | --- |
| main | `692ac87d6b7b3e16d0b74bdc85c476d47daac059` |
| exp/cpu-stylegram-handoff-20261021 | `64ec4e7e39dd0ff6290af89f006c934b8fb8351a` |
| exp/dev-comparison-20261021 | `7a468a8436c2002963efd839d1b9181a94a4577d` |
| exp/prompt-centering-20261021 | `d0f6cbe803b4c89847a22bdb6f39f7a2f5b32549` |
| exp/single-submission-20261021 | `4760f67fd06f59c602389595caafae090606331d` |
| exp/windows-handoff-20261023 | `25c9a02ccd18d36391e31e15ffda268ab735dd8e` |

各分支根目录、single-submission 的模块 data/reports 页面、Releases/Tags/Actions 页面和当前 clone 可达 Git 对象均未找到上述运行材料。data 页面只有 README；Release 页面无发布；Actions 显示初始配置页，无可见运行或 artifact。相关入口：[分支](https://github.com/zdx13143/name-that-llm/branches)、[数据说明](https://github.com/zdx13143/name-that-llm/tree/exp/single-submission-20261021/name-that-llm/data)、[Releases](https://github.com/zdx13143/name-that-llm/releases)、[Actions](https://github.com/zdx13143/name-that-llm/actions)。

范围限制：GitHub connector 未能访问该私有仓库，gh 认证也未通过；网页登录态可读。未取得 API 递归树/全部 artifact JSON，因此不宣称已穷尽所有已删除或不可见历史对象。现有证据支持“代码、报告和 Notebook 在 GitHub，原始实验状态未随交接上传”。不能据 README 中出现文件名就视为数据已经取得。

下一步需要原机器授权传输清单见 [预登记方案](../docs/WINDOWS_MODERNBERT_PLAN.md#材料移交)。只有 split 元数据允许由 hash 正确的 train 用既有工具恢复；OOF 缺失不能通过重训补齐。原生 Windows 不能直接导入依赖 resource/SIGALRM/setitimer 的 compare_team_cpu、compare_prompt_centering、旧 train/finalize 或 verify_candidate_notebook。

## 新 GPU 检查结果

已在独立 `.venv-gpu` 导入 torch `2.8.0+cu128`，`pip check` 无依赖冲突。初次安装会话因工具连接更新失去可轮询的 process id，安装日志停在 Installing，不能补写“原安装进程返回0”；之后对实际环境的导入、依赖检查和 CUDA 实测分别提供确认。

`preflight_cuda.json` 时间为 `2026-10-23T14:56:11Z`，CUDA build=12.8、available=true、设备 compute capability=8.9。32×32 FP32 CUDA 矩阵检查 passed，最大绝对误差 0.0，单次 matmul 0.048268 秒、峰值分配 8,531,968 bytes；这是含该次运行条件的小检查，不是吞吐 benchmark。当时 NumPy 尚未安装，Torch 发出 NumPy 初始化警告，但矩阵检查不依赖 NumPy并实际通过；后续模型环境单独安装固定 NumPy。

ModernBERT 合成前向已在 `2026-10-23T15:01:28Z` 启动并成功完成，独占claim为completed、attempted/completed forward均为2。未使用比赛文本或标签，未拟合分类器。

| 检查 | 实测 |
| --- | --- |
| 模型 | answerdotai/ModernBERT-base，revision `8949b909ec900327062f0ebf497f51aef5e6f0c8` |
| 文件身份 | README、config、tokenizer等6个文件均匹配预登记Git blob/LFS身份，另记录本机SHA256；权重SHA为 `340ac08b74eef0d7bdec2d7981a6a3d4249bf0e6aab60634b72ad02c2b8023a9` |
| 实际直接依赖 | torch2.8.0+cu128、transformers4.56.2、huggingface-hub0.34.4、tokenizers0.22.0、safetensors0.6.2、numpy2.2.6 |
| 前向设置 | 冻结BF16、SDPA、reference_compile=false、batch4、max_length256、最后4层FP32 mask mean拼接 |
| 输入 | 4条固定合成文本，原始token长度12/18/22/803，1条截断 |
| 输出 | 4×3072，有限值与显式非padding参考池化通过；原序/反序恢复后最大绝对差0.0 |
| 两次forward检查耗时 | 0.272567秒 / 0.024290秒，包含同步与池化核验；不外推完整开发吞吐 |
| CUDA峰值 | allocated 367,231,488 bytes（约350.22MiB）；reserved 375,390,208 bytes（358MiB） |
| 总耗时 | 50.931秒，含校验、import与加载；不含模型下载 |
| 限制/警告 | 仅有torch_dtype参数弃用提示，已保留日志；没有OOM、非有限值或一致性失败 |

下载与smoke进程均实际返回0，依赖安装返回0，最终pip check返回0。现有 GPU probe 12 项、此次身份/claim 保护7项通过；工具测试与真实CUDA/模型合成检查分开报告。没有重跑已完成的smoke来收集更好耗时。

本地关键证据：`preflight_initial.json`、`host_inventory_normal_permissions.json`、`wsl_inventory.json`、`preflight_cuda.json`、`modernbert_files.json`、`modernbert_smoke.json`、`modernbert_smoke.log`、`pip_freeze_windows_gpu.txt`，以及 `outputs/continuation_claims/windows-modernbert-synthetic-20261023.json`。模型检查结果文件SHA256为 `920300705ea14c0ca6b11adc6e9b324ed235566bce44990ecd35fc79c839db40`；claim绑定该hash、配置规范hash和执行脚本hash。

已执行命令（从仓库根；无需重跑已完成的download/smoke）：

```powershell
python -m venv name-that-llm/.venv-gpu
name-that-llm/.venv-gpu/Scripts/python.exe -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0
name-that-llm/.venv-gpu/Scripts/python.exe name-that-llm/scripts/probe_gpu_environment.py --cuda-smoke --output name-that-llm/outputs/windows_4060/preflight_cuda.json
name-that-llm/.venv-gpu/Scripts/python.exe -m pip install --no-cache-dir --index-url https://pypi.org/simple -r name-that-llm/requirements-windows-gpu.txt
name-that-llm/.venv-gpu/Scripts/python.exe name-that-llm/scripts/probe_modernbert.py download
name-that-llm/.venv-gpu/Scripts/python.exe name-that-llm/scripts/probe_modernbert.py smoke
```

上述安装/download/smoke实际由Python父进程调用，并分别使用900/900/300秒timeout、独占新日志捕获输出。普通直接命令本身不带外部deadline。源码与保护测试在Git，环境、权重及原始receipt不上传。

## 在队友工作的基础上改进

本轮亲自复查根 EXPERIMENTS.md 和既有 TEAM_WORK_REVIEW.md，结合源码定位作如下取舍，保留历史记录不改写：

1. 延续队友 ModernBERT-base 最后四层、3,072 维、折内 StandardScaler + LR 的可用路径；新的工作是固定模型/tokenizer与输入缓存身份、隔离120-prompt开发数据，并在当前75/25候选上做预定互补比较。根 run_embed.py 会读全量150 prompt且缓存按名称/shape复用，故不直接运行它。
2. 队友日志的0.3041单模型与0.3764融合均为另一协议的历史记录。旧run_embed对照池曾包含待比较嵌入，故“+0.0278”的因果增量不能直接移植；本轮固定一个融合权重并复用严格匹配的OOF。
3. 不重试已记录的无尺度密集/稀疏直接拼接、先机构后型号、全体簇按18类硬分配，或直接把弱基线Gemini门控搬到强集成。
4. 队友0.601仍缺成功v8源码/依赖版本/提交编号的完整绑定，当前参考Notebook含8B/14B且无已验证8GB回退。因此不在这台4060上直接启动完整强集成，也不宣称本轮ModernBERT会超过0.601。
5. 本轮一次Kaggle提交的具体版本必须基于已验证交付。新GPU实验未完成时，不能将合成probe包装成改进模型；现有固定75/25 CPU Notebook是已验证的候选。正常权限恢复后先核查该候选是否已提交，再执行尚未完成的一次操作。

本轮方案与停止条件见 [Windows ModernBERT 预登记](../docs/WINDOWS_MODERNBERT_PLAN.md)；未来相对队友强方案的改进还需要该强方案相同120-prompt协议的OOF，不可跨协议直接相减。

## Kaggle 与分数边界

用户本轮回复手动运行/提交状态“不确定”，并授权后续由代理自动提交；随后明确要求本轮结束后提交一次查看分数。将该授权与历史状态核查分开：访问限制经正常途径解除后先核对当前账号/队伍、Notebook 版本、运行结果、提交历史与余量；未知状态不重复提交。已请求用户确认原限制是否经正式设置解除。当前没有访问 Kaggle、核实规则/额度、运行 Notebook、正式提交或取得评分。

旧 CPU 开发 BA、30-prompt 历史留出分数、队友 0.601 历史记录均不变化。本轮新真实 fit=0、旧 holdout evaluation=0；GPU 合成工程结果不能称为新开发分数或榜单提升。
