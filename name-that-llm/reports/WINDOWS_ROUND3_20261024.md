# 第三轮实际结果：题目噪声高斯分类头

实验日期：2026-10-24（Asia/Singapore）。本轮五折完成，pooled BA **0.2986772487**，比第二轮高 **0.0087729536**（0.8773个百分点），但增益区间跨0、仅3/5折改善，未达到预登记要求。没有最终拟合、Kaggle上传、保存运行或正式提交；没有新公榜分数。

## 问题与方法

第二轮冻结ModernBERT + LR已完成但效果弱。开发诊断发现明显题目结构；截断存在但未被证实是主要损失来源。详细证据与因果局限见[原因分析](MODERNBERT_ROUND2_DIAGNOSIS_20261024.md)。

实时核对GitHub `main=692ac87d6b7b3e16d0b74bdc85c476d47daac059`，参考源文件与该commit无差异。借鉴队友仓库Gaussian参考中的共享prompt噪声建模，选择可复用现有向量的一组固定算法：训练折StandardScaler、同题残差PCA128、4次类别/题目均值交替估计、经有限组噪声校正及Ledoit–Wolf收缩的协方差，单条高斯边缘预测。不是复现完整8B/14B集成，不继承其榜单分数。

预登记、实现和测试在真实运行前提交并推送为 **`c85a5a4`**，详见[固定计划](../docs/WINDOWS_ROUND3_PLAN_20261024.md)。新ID为`windows-prompt-gaussian-dev-20261024-r3`；使用原120个prompt / 2,159行和第二轮完全相同的fold_id。scaler、PCA、类别与协方差估计均只看训练折；验证预测不接收prompt_id、不聚类、不强制18条唯一分配。第二轮LR没有重新拟合。

## 真实指标和决策

| 项目 | 第二轮 LR | 第三轮 Gaussian | 第三轮减第二轮 |
| --- | ---: | ---: | ---: |
| pooled BA | 0.2899042951 | 0.2986772487 | +0.0087729536 |
| fold 0 | 0.2962962963 | 0.2962962963 | 0 |
| fold 1 | 0.2824074074 | 0.2847222222 | +0.0023148148 |
| fold 2 | 0.3125000000 | 0.3055555556 | -0.0069444444 |
| fold 3 | 0.3009259259 | 0.3148148148 | +0.0138888889 |
| fold 4 | 0.2575483092 | 0.2918679549 | +0.0343196457 |

第三轮2,000次整prompt bootstrap的描述性95% BA区间为 **[0.2801470588, 0.3166683201]**。同一批行、同一fold的配对增益区间为 **[-0.0069917101, +0.0249690083]**，改善折数为3/5。fold0的浮点差约-1.2e-18，按零报告。

| 运行前门槛 | 实际 | 结论 |
| --- | --- | --- |
| pooled BA ≥ 0.36 | 0.298677 | 未通过 |
| 每折 BA ≥ 0.30 | 最低0.284722，3折低于0.30 | 未通过 |
| 配对点增益 ≥ 0.02 | 0.008773 | 未通过 |
| 配对95%区间下界 > 0 | -0.006992 | 未通过 |
| 至少4折改善 | 3折 | 未通过 |
| 完整性与supervisor完成 | 通过 | 仅工程检查通过 |

`candidate_development_gate_passed=false`，部署验证未启动。不能将“方向有统计动机”“观察分数略升”写成稳定收益。120个prompt已经多轮用于开发，bootstrap不是新的独立测试；历史75/25的0.348673仅是聚合参考，无逐行旧OOF的配对结论。历史Kaggle 0.351/0.601与本地CV分别报告。

## 每类召回率

| 类别 | 召回率 |
| --- | ---: |
| gpt-5.5 | 0.233333 |
| gpt-5.6-luna | 0.308333 |
| gpt-5.6-sol | 0.175000 |
| gpt-5.6-terra | 0.175000 |
| gpt-6-astra | 0.483333 |
| gemini-3.1-pro-preview | 0.583333 |
| gemini-3.6-flash | 0.333333 |
| gemini-3.7-flash | 0.383333 |
| gemini-3.8-flash | 0.258333 |
| claude-fable-5 | 0.142857 |
| claude-opus-4-6 | 0.366667 |
| claude-opus-4-8 | 0.266667 |
| claude-opus-5 | 0.441667 |
| claude-sonnet-5 | 0.291667 |
| qwen3.8-max | 0.275000 |
| glm-5.3 | 0.191667 |
| glm-5.3-flash | 0.200000 |
| kimi-k3 | 0.266667 |

## 预算与核验

- 实际5/5次classifier fit尝试/完成；每次包含scaler/PCA/统计参数整套拟合。新增编码器前向0次，第二轮特征直接复用。
- supervisor completed / exit0，总耗时35.1519905秒；抽样进程树RSS峰值610,803,712字节，约0.5689GiB；上限1,800秒、4GiB、2线程。
- 6项新增相关合成测试通过（0.229秒）。检查训练类覆盖、单条/批量/顺序一致性、导出预测一致性、claim与预算边界；测试不使用比赛数据，不做真实编码器前向。
- 独立只读审查核算OOF和保存分类头的预测，最大绝对差约2.11e-15；主线程最终核验官方开发行身份、fold、labels、缓存、源码快照、原始SHA、进程退出与完成收据。
- 默认readiness再次只读检查返回exit2，拒绝既有claim/output，没有重跑。旧30-prompt评分、全量最终fit、条件最终head fit、新Kaggle上传/运行/提交全部为0。
- 第一轮两个claim和第二轮claim的SHA均保持预登记值；保留第一轮265/264、0fit的失败与路径覆盖法证记录，不回写为成功。

实际运行命令已使用一次，**本轮关闭，不再次执行**：

```powershell
.\name-that-llm\.venv-gpu\Scripts\python.exe .\name-that-llm\scripts\windows_prompt_gaussian_round3.py --run
```

## 证据身份

完整本地产物在忽略目录`outputs/windows_prompt_gaussian_20261024_r3/`。代码、配置、测试和聚合文档入Git；原始数据、模型、OOF、数值分类头与环境留在本地。

| 证据 | SHA256 |
| --- | --- |
| 固定配置 | `be521933587ae0fd99086a2d1a52cb0dc67bea396ad81bdedfc9f9caa12f7091` |
| contract语义身份 | `e8fc011170cca61e50696ed6e91413544010e2a07388e096e59ed1f61af035f1` |
| oof.npz | `d0d8d3c413676c376eca0f74acfa03d4ee7a9959ddc1b57789b882948344e30f` |
| summary.json | `1f9ec3e48e51b7c1e6d70e666b68a5af5f12cbd9f0b99dda87906e6d31e58bf5` |
| supervisor.json | `1fe5e2c82a99ae2634dfe31f7eecb4f1771f08bd7e17544acac2762d325fa3b2` |
| 完成claim | `8708ede039c4ee84f81f7fc5d809b360fef63bbc70d840708b1254b8fb371b9e` |
| 最终核验收据 | `7ed26d4d23961379d5eefd403740c868c58a1295fce9b3c8667cc4dd7cb0cf9e` |

最终收据为`outputs/windows_4060/round3_final_reconciliation_20261024.json`，同时记录全部原claim SHA、真实fit次数、保留的旧Notebook/标签SHA与核验结果。

面向初学者的进展、术语、方法解释与练习已维护在根目录[PROJECT_LEARNING.md](../../PROJECT_LEARNING.md)及[离线HTML](../../PROJECT_LEARNING.html)。后续每轮同步更新，不能只交付分数和代码。

2026-10-25页面验收：10章文字与源文件SHA绑定，重复生成字节一致，所有本地链接存在，无外部资源依赖；桌面显示、目录、计算器两组数值与恢复按钮通过，浏览器未见脚本错误。390px视口DOM检查无整页横向溢出；该视口的浏览器截图返回空白，移动端视觉检查未确认，临时视口已还原。只读独立文档核验未发现指标/预算/解释实质错误。页面检查收据在`outputs/windows_4060/learning_artifact_validation_20261025.json`，没有新增训练或留出评估。
