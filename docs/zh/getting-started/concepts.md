# 基本概念

## Candidate Model

你真正想研究内部表示的模型。

## Reference Model

用于外部比较的模型。在 `repralign` 里通常有两类：

- semantic reference
- generation-oriented reference

## Layer-Wise Alignment

对 candidate model 的每一层提取特征，并与一个或多个 reference feature 做比较，最后得到一条随层数变化的相似度曲线。

## Pooling

很多 hook 到的输出是 token 级或空间级特征，需要先变成 sample-by-feature 矩阵。当前支持：

- `cls`
- `mean_tokens`
- `flatten_mean`

## Metrics

### CKA

衡量两个表示空间是否保留了相似的样本几何关系。

### CKNNA

衡量两个表示空间在 top-k 邻居结构上的一致程度。
