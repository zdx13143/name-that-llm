# 本模块协作约定

遵循仓库根 [README](../README.md#协作约定)：实验分支用 `exp/<主题>`，提交说明包含改动及 CV 影响；报告给出每类召回和对应命令，验证必须按 prompt 分组，不能用随机行切分或公榜调参。原上传分支为 `exp/cpu-stylegram-handoff-20261021`；后续 `exp/dev-comparison-20261021` 和 `exp/prompt-centering-20261021` 实验均已完成。当前从后者建立 `exp/single-submission-20261021`，仅准备用户明确授权的一次固定候选提交，见 [SINGLE_SUBMISSION_PLAN](docs/SINGLE_SUBMISSION_PLAN.md)；此次CV变化为0。

原上传仅导入已有结果，CV 不发生变化。用户随后要求分析队友并继续推进，新的有限实验已单独定义于 [TEAM_CONTINUATION_PLAN](docs/TEAM_CONTINUATION_PLAN.md)，只对原120-prompt开发子集使用真正GroupKFold；不能把历史自定义分折的分数换标签。30-prompt最终留出已经使用，不能重新评分、调参或据新clone缺少claim而重开；也不能直接与队友全量五折或榜单数字比较。

本模块单独管理依赖，不修改仓库根 requirements、冻结模型或队友历史实验记录。当前授权范围包含根代码的两处确定bug修复；新训练代码仍放在本模块。用明确路径暂存，先检查 diff；只推自己的分支，不向 main 或队友分支推送、不强推、不自动合并。数据、凭据、环境、权重和本地输出均不提交。Git 上传不等于 Kaggle Notebook 更新或正式比赛提交。
