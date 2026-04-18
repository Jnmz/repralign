[English](README.md) | [中文](README.zh-CN.md)

# repralign

[![PyPI version](https://img.shields.io/pypi/v/repralign)](https://pypi.org/project/repralign/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

`repralign` 是一个面向 PyTorch 与多模态研究工作流的轻量、可复用 Python 工具包，用于做分层表示对齐分析。它适合这样的实验：从候选模型和一个或多个参考模型中提取中间特征，计算逐层相似度曲线，并保存便于复查与写作的结果文件。

这个项目采用了论文启发式的分析工作流：把同一个候选模型分别和不同参考模型对比，观察对齐关系如何随层数变化，并用论文风格的图进行总结。它的定位是一个通用、可配置、可复用的研究工具包。

## 项目动机

很多多模态论文都会讨论：某个学到的统一表示更像语义编码器、生成导向编码器，还是介于两者之间。`repralign` 提供了一套通用基础设施来支持这类分析：

- 从可配置层提取中间特征
- 对特征做池化与归一化，得到可比较表示
- 用可复用指标计算逐层相似度
- 缓存提取后的特征，避免重复前向
- 保存 CSV、PNG 和 JSON 结果，方便分析与论文整理

## 安装

通过 PyPI 安装：

```bash
pip install repralign
```

如需同时安装 Hugging Face 支持：

```bash
pip install "repralign[huggingface]"
```

从源码安装用于本地开发：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

从源码安装并启用 Hugging Face 支持：

```bash
pip install -e .[dev,huggingface]
```

## 包结构

```text
repralign/
  src/repralign/
    adapters/
    metrics/
    cache.py
    cli.py
    config.py
    extract.py
    hooks.py
    plotting.py
    pooling.py
    registry.py
  tests/
  examples/
  configs/
```

## 指标的实用理解

### CKA

线性 CKA 用来衡量两个表示空间是否保留了相近的样本两两关系结构。实际使用时，它适合比较特征维度不同的层，因为它关注的是整体几何关系，而不是坐标逐项一致。

期望输入形状：

- `features_a`: `(n_samples, n_features_a)`
- `features_b`: `(n_samples, n_features_b)`

### CKNNA

`repralign` 中的 CKNNA 实现为一种邻域重叠指标：对每个样本，分别在两个表示空间里找 top-`k` 最近邻，然后比较两组邻居集合的重叠比例，最后再取平均。它适合回答这样的问题：即使全局几何不同，两层是否仍然诱导出相似的局部邻域结构。

期望输入形状：

- `features_a`: `(n_samples, n_features_a)`
- `features_b`: `(n_samples, n_features_b)`

局限性：

- 两个输入都要求样本按相同顺序对齐
- 当 batch 很小时，CKNNA 会比较不稳定
- v0.1 主要分析池化后的表示，还没有覆盖 token 级结构

## 快速开始

运行一个最小玩具示例：

```bash
python examples/minimal_compare.py
```

运行一个“论文启发式”的分析示例：

```bash
python examples/paper_figure_analysis.py
```

这两个示例都会把结果输出到 `outputs/`。

## CLI 用法

列出模型中可用层名：

```bash
repralign list-layers \
  --factory examples.model_factories:build_toy_candidate \
  --adapter generic_torch
```

通过 YAML 配置提取特征：

```bash
repralign extract-features --config configs/example_analysis.yaml
```

基于缓存特征运行相似度分析：

```bash
repralign analyze \
  --candidate-cache outputs/paper_like/candidate_features.npz \
  --reference-cache outputs/paper_like/semantic_reference_features.npz \
  --metric cka \
  --output-csv outputs/paper_like/cka_semantic.csv \
  --output-json outputs/paper_like/cka_semantic.json
```

绘制逐层对齐曲线：

```bash
repralign plot \
  --input-csv outputs/paper_like/cka_semantic.csv \
  --input-csv outputs/paper_like/cka_generation.csv \
  --output-png outputs/paper_like/cka_comparison.png \
  --title "Layer-wise Alignment"
```

## Python API

```python
from repralign.adapters.generic_torch import GenericTorchAdapter
from repralign.extract import extract_feature_dict
from repralign.metrics import linear_cka

adapter = GenericTorchAdapter(model, layer_names=["encoder.layers.0", "encoder.layers.1"])
features = extract_feature_dict(adapter=adapter, batch=batch, pooling="mean_tokens")
score = linear_cka(features["encoder.layers.0"], features["encoder.layers.1"])
```

## YAML 配置

v0.1 的 CLI 使用显式 YAML 配置，把模型工厂、层名、输入张量、池化方式和输出目录都放在一个清晰文件里。可运行示例见 [configs/example_analysis.yaml](/Users/jnmz/Desktop/code/repralign/configs/example_analysis.yaml)。

## 论文启发式工作流

本仓库借鉴的是公开论文中常见的分析结构，主要体现在：

- 将同一个候选模型与多个参考模型对比
- 使用逐层相似度曲线组织分析图
- 围绕“语义导向参考”和“生成导向参考”来设计实验

它**不应**被当作隐藏实现细节的来源。本仓库刻意把模型内部逻辑做成可配置，只依赖通用、公开的分析原语。

## TODO

- 增加不经过池化的 token 级相似度分析
- 增加张量文件以外的数据集加载方式
- 增加适配自定义多模态结构的更丰富 adapter，例如 Show-o2
- 增加 diffusion 与 DiT 风格参考模型 adapter
- 增加适合大规模数据的分批缓存写出能力
- 增加置信区间和多次运行聚合工具

## 许可证

MIT
