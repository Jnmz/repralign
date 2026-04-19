# repralign

`repralign` 是一个用于多模态模型分层表示对齐分析的可复用 Python 工具包。

它适合这样的工作流：

- 从 candidate model 和 reference model 中提取中间特征
- 对特征做池化和归一化
- 计算逐层 CKA / CKNNA 相似度
- 输出 CSV、JSON 和 PNG 图

## 当前支持

- 基于 forward hook 的分层特征提取
- `cls`、`mean_tokens`、`flatten_mean` 三种池化方式
- 线性 CKA 与 CKNNA
- 张量 batch 或图片数据集的分批提取
- 通过 Hugging Face vision encoder 做 semantic reference
- 通过 Diffusers pipeline 做 generation reference，例如 SD3
- CLI 工作流与 Python API

## 适合做什么

`repralign` 适合做论文风格的表示对齐分析，例如：

- candidate model 对 semantic reference
- candidate model 对 generation-oriented reference
- 分析不同层与参考模型的对齐曲线

## 对 TUNA 类分析的支持情况

当前版本已经可以支持和 TUNA 论文相关章节**同类型**的分析流程：

- 一个 candidate model
- 多个 reference model
- 逐层计算相似度
- 输出可视化曲线

但它不是某篇论文的现成实验包。要得到你真正想要的实验结果，仍然需要你自己提供：

- 真实的 candidate model adapter
- 想要分析的具体 layer names
- semantic reference checkpoint
- generation reference checkpoint
- 数据集和预处理方案

因此目前的状态可以概括为：

- 分析框架已经 ready
- TUNA 风格实验已经可行
- 具体实验设置仍然取决于你的模型和数据

## 从这里开始

- [安装](getting-started/installation.md)
- [CLI 工作流](guides/cli-workflow.md)
- [YAML 配置](guides/yaml-configs.md)
- [参考模型](guides/reference-models.md)
- [真实图片工作流](tutorials/real-image-workflow.md)
