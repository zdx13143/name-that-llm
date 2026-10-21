# RTX 4060 Windows 推进评估

评估日期：2026-10-21。建议把已有 RTX 4060 电脑作为后续编码器实验的 GPU 节点，当前 CPU 审计、稀疏模型和统计复核继续留在本机。没有必要为当前 CPU 轮次整体迁移；若下一步进入冻结编码器特征或小模型微调，迁移 GPU 部分值得做。GPU 能缩短某些实验的时间，但不能保证提高比赛分数。

本文件是官方资料与当前源码基础上的可行性评估，**尚未连接或运行目标 Windows 电脑**。台式/笔记本型号、系统内存、空闲磁盘、驱动和实际可用显存均未知；以下按普通 RTX 4060 8GB 评估。没有下载模型、训练 GPU 模型或提交 Kaggle。

## 硬件适用范围

NVIDIA 官方规格中，普通桌面 RTX 4060 与 RTX 4060 Laptop GPU 均为8GB显存；4060 Ti 的16GB型号是另一款产品。笔记本功耗/散热会影响持续速度，不能由“4060”名称推断与台式版等速。[桌面规格](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4060-4060ti/)、[笔记本规格](https://www.nvidia.com/en-us/geforce/laptops/40-series/)。

| 工作 | 8GB RTX 4060 的判断 | 本项目安排 |
|---|---|---|
| TF-IDF、文体统计、LR、OOF重算 | 现有代码主要由CPU/内存承担，不能直接获得CUDA加速 | 留在当前机器。上一组21次拟合尝试合计约106秒，迁移成本高于该轮计算成本；见[实际报告](../reports/TEAM_DEV_COMPARISON.md) |
| ModernBERT-base冻结特征 | 适合作为首个GPU任务；短序列、小batch应有充足的权重空间，实际峰值仍需测量 | 先冻结模型提取开发集特征，再用CPU做同协议分类器与融合比较 |
| ModernBERT-base有限微调 | 值得试验，小batch、混合精度、必要时梯度检查点；“可运行”不等于“优于现有强基线” | 冻结特征路线通过工程核验后再安排；单折可用于运行能力检查，不能充当完整CV结论 |
| ModernBERT-large冻结特征 | 参数量仍远小于8B；可作为后续候选，但应先证明base的增量价值 | 不为模型变大而直接扩大搜索 |
| Qwen3-8B 4-bit冻结推理/特征 | 有条件可行；要控制序列、batch、保留的hidden states，并实测显存 | 作为后续小规模可行性试验，不能保证现有Notebook原样适配 |
| Qwen3-8B QLoRA | 8GB非常紧张，取决于量化、序列、adapter、优化器及检查点策略 | 先完成独立前向/反向试验与耗时估计；不把五折训练列为首选迁移任务 |
| Qwen3-14B及队友完整强集成 | 不适合作为8GB机器的直接搬运目标 | 先绑定成功版本和权重；以后再考虑经核验的更大显存/多卡环境，或另行设计并验证的offload/量化版本 |

以上“适合/紧张”是根据参数规模和代码路径作出的工程判断，不是目标机benchmark。ModernBERT官方列出base为149M、large为395M参数。[模型卡](https://huggingface.co/answerdotai/ModernBERT-base)。Qwen3官方模型卡列出8B实际为8.2B、14B实际为14.8B参数。[8B模型卡](https://huggingface.co/Qwen/Qwen3-8B)、[14B模型卡](https://huggingface.co/Qwen/Qwen3-14B)。

仅按全部参数以同一精度存放计算，`参数量 × 每参数位数 / 8 / 2^30` 得到：

| 模型 | FP16/BF16原始权重 | 理想4-bit原始权重 |
|---|---:|---:|
| ModernBERT-base（149M） | 0.28GiB | 本轮不需要量化 |
| ModernBERT-large（395M） | 0.74GiB | 本轮不需要量化 |
| Qwen3-8B（8.2B） | 15.27GiB | 3.82GiB |
| Qwen3-14B（14.8B） | 27.57GiB | 6.89GiB |

这些是算术下界，**不是加载显存或可运行保证**。量化元数据、未量化模块、激活、临时工作区、CUDA上下文和显示占用还需空间；训练另需梯度/优化器/adapter状态。生成任务还会增加KV cache，特征任务则应关闭不需要的cache并避免保留全部层输出。8B的“3.82GiB”不能推导8GB必能完成训练，14B的“6.89GiB”也不能推导8GB必能推理。

队友0.601是仓库中的实验记录，成功v8的精确源码、附件版本和提交证据尚未完整绑定；当前参考Notebook的14B路径在未满足多卡条件时直接放到`cuda:0`，没有已验证的8GB回退。不能承诺转到4060就复现0.601。详见[队友审查](TEAM_WORK_REVIEW.md#0601-的源码与提交证据缺口)。量化或改offload也属于新运行条件，需要重新验证。

## Windows 原生与 WSL2

Windows原生不是CUDA或量化的禁区。PyTorch官方提供Windows CUDA安装；RTX4060的CUDA计算能力为8.9，bitsandbytes v0.50.2官方安装表列有Windows x86-64、CUDA及`sm89`构建支持。因此不能沿用“bitsandbytes不支持Windows”的旧说法。是否成功仍取决于目标机驱动和选定的实际软件组合。[PyTorch安装](https://pytorch.org/get-started/locally/)、[NVIDIA计算能力](https://developer.nvidia.com/cuda/gpus)、[bitsandbytes安装表](https://huggingface.co/docs/bitsandbytes/v0.50.2/en/installation)。

推荐分工：原生Windows可先运行独立GPU探测和未来独立的编码器提取器；如果要复用当前开发验证runner，优先WSL2。当前`compare_team_cpu.py`使用Unix的`resource`、`SIGALRM`和`setitimer`，不能把“PyTorch支持Windows”误写为本仓库所有脚本已支持Windows。没有必要仅为小编码器预先引入FlashAttention编译链；优先官方Transformers支持的SDPA路径。[ModernBERT文档](https://huggingface.co/docs/transformers/model_doc/modernbert)。

WSL2的CUDA由Windows NVIDIA驱动提供。按NVIDIA指南准备WSL2；不要在WSL内部另装Linux显示驱动。本评估没有替用户修改系统、安装WSL或驱动。[NVIDIA WSL指南](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)。

4-bit/8-bit微调应训练额外参数（例如LoRA），不能当作普通全参数低位训练；`device_map="auto"`的推理offload方式也不应直接套成训练配置。[Transformers量化说明](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes)、[PEFT量化训练说明](https://huggingface.co/docs/peft/en/developer_guides/quantization)。

## 迁移顺序与可执行检查

1. 在Windows独立clone本仓库的本人实验分支，记录准确Git SHA；后续Windows改动使用新的`exp/`分支，避免两台机器同时向同一分支推进。代码通过Git同步；原始数据、模型、缓存、claim、OOF和预测均不进入Git。Mac的`.venv`不能复制为Windows环境。
2. 单独传输比赛原始数据并核对SHA。只恢复既有split元数据，不运行旧search/finalize。新clone没有旧claim，不表示旧30-prompt最终留出重新可用。新的编码器开发工作只取既定120个prompt/2,159行；不提取或评分旧30-prompt留出。
3. 建立独立`.venv-gpu`，使用PyTorch官方选择器按目标Windows/WSL与驱动选择CUDA wheel；记录Python、torch/CUDA、transformers、huggingface-hub等实际版本。不要直接覆盖已验证CPU环境。量化任务才加入经测试的bitsandbytes/PEFT/accelerate组合。
4. 先完成下述无模型探测。只有CUDA设备和小矩阵运算确实成功，才开始模型下载和小batch前向检查。CUDA smoke只是环境证据，不是模型速度、显存容量或分数证明。

原生Windows PowerShell，从仓库根运行（Python环境已按上一步创建并安装CUDA版PyTorch）：

```powershell
nvidia-smi
New-Item -ItemType Directory -Force name-that-llm/outputs/windows_4060
.\.venv-gpu\Scripts\python.exe name-that-llm/scripts/probe_gpu_environment.py --cuda-smoke --output name-that-llm/outputs/windows_4060/preflight.json
```

WSL2/Linux等价检查：

```sh
nvidia-smi
mkdir -p name-that-llm/outputs/windows_4060
.venv-gpu/bin/python name-that-llm/scripts/probe_gpu_environment.py --cuda-smoke --output name-that-llm/outputs/windows_4060/preflight.json
```

探测输出是本地收据，拒绝覆盖既有文件；不同环境使用新的文件名。保留设备名、总显存、计算能力、驱动、torch CUDA版本、依赖版本和真实smoke结果；另从`nvidia-smi`记录当时的空闲显存。还需检查系统RAM与磁盘空闲；本轮不预先宣称目标电脑满足这些条件。

## 第一个 GPU 实验的建议边界

首先只做ModernBERT-base冻结特征，模型/分词器绑定完整Hub commit SHA，依赖固定；长度从256、batch从4开始检查，失败可降低batch并如实记录，不能悄悄截短后仍沿用原配置名称。先用少量合成文本核对前向、padding mask、pooling、行序、有限值及峰值显存。正式开发集提取前冻结长度、选层和pooling方案，记录截断比例；不以外层验证标签选择这些设置。

模型下载采用官方来源并固定`revision`。缓存键至少包含模型与tokenizer revision、配置/权重文件SHA、输入文本hash和行序、截断/选层/pooling、dtype、实现与依赖版本。当前根`src/embeddings.py`的缓存命名及默认镜像不能替代这些来源记录；不能仅因缓存文件同名就复用。Hugging Face支持按完整commit revision下载，离线执行时使用已核验的本地快照。[Hub下载文档](https://huggingface.co/docs/huggingface_hub/guides/download)。

预训练模型保持冻结且特征定义不使用本赛标签时，可以对固定开发文本统一提取，再让分类器、缩放和融合按同一GroupKFold做折内训练。若进行监督微调，每个外层fold必须从相同预训练快照重新开始，只训练该fold的训练prompt；epoch/学习率选择需要内层分组验证或预先冻结，不能反复查看外层标签选择最好epoch。模型变更仍需另记有限预算与停止规则后执行。

首批模型前向只回答“这台电脑能否稳定运行、峰值显存和耗时是多少”；完整五折和配对不确定性才回答“是否值得保留”。对队友强方案是否有增量，还需要该强方案相同开发协议的OOF。不得将开发改善直接称为公开榜提升，亦不得因更换电脑重新打开旧最终留出。

Kaggle免费GPU仍是可选执行环境，但当前保存的域名权限拒绝尚未解除，本轮没有核验配额、设备或现行规则，也没有通过API或其它浏览器绕过。完成Windows实验也不等于Kaggle离线Notebook已经复现；未来提交前还须核验官方限制和实际运行环境。
