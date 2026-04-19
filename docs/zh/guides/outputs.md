# 输出内容

`repralign` 在实验过程中会写出几类结果。

## Feature Caches

后缀：

- `.npz`

按层名保存 pooled feature matrix，便于后续复用而不必重复前向。

## Metadata

后缀：

- `.json`

通常包含：

- 模型名
- adapter 名
- 选取的层
- pooling 方式
- normalize 选项
- dataset 配置

## Similarity Tables

后缀：

- `.csv`

通常每行包含：

- metric
- candidate layer index
- candidate layer name
- reference layer name
- score

## Plots

后缀：

- `.png`

`plot` 命令会把一个或多个相似度曲线画成论文风格图像。
