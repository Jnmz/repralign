# CLI 工作流

CLI 入口是 `repralign`。

## 列出层名

```bash
repralign list-layers \
  --factory examples.model_factories:build_toy_candidate \
  --adapter generic_torch
```

## 提取特征

```bash
repralign extract-features --config configs/example_analysis.yaml
```

这一步会：

- 读取 candidate 和 reference 配置
- 从指定层提取特征
- 保存 `.npz` cache
- 写出 JSON metadata

## 计算相似度

```bash
repralign analyze \
  --candidate-cache outputs/paper_like/candidate_features.npz \
  --reference-cache outputs/paper_like/semantic_reference_features.npz \
  --metric cka \
  --output-csv outputs/paper_like/cka_semantic.csv \
  --output-json outputs/paper_like/cka_semantic.json
```

## 绘图

```bash
repralign plot \
  --input-csv outputs/paper_like/cka_semantic.csv \
  --input-csv outputs/paper_like/cka_generation.csv \
  --output-png outputs/paper_like/cka_comparison.png \
  --title "Layer-wise Alignment"
```
