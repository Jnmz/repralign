# 当前限制

`repralign` 已经可以做真实实验，但仍然是一个较早期的研究工具。

## 现阶段限制

- token-level alignment 还不是一等公民
- 超大数据集还缺少更高效的磁盘缓存策略
- generation reference 工作流依赖目标 pipeline 是否暴露稳定的内部模块
- 想精确贴近某篇论文，还需要自己验证实验协议

## 一个重要提醒

`repralign` 支持的是和多模态论文里相同类别的分析方法，并不自动意味着结果等同于任何已发表论文。要做到强对齐，通常还需要一致的：

- candidate model
- reference model
- dataset
- preprocessing
- layer selection
- metric setting
