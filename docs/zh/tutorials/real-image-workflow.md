# 真实图片工作流

这是从 toy example 走向真实实验时最该看的路径。

## 1. 从模板开始

使用：

- `configs/reference_alignment_template.yaml`

## 2. 换成真实 Candidate Model

把 toy candidate factory 替换成你的真实模型工厂，并填写真实 layer names。

## 3. 选择参考模型

常见组合：

- semantic reference：SigLIP 或其他 HF vision encoder
- generation reference：SD3-Medium + `sd3_reference`

## 4. 指向你的图片目录

```yaml
dataset:
  type: image_folder
  image_root: /path/to/images
  recursive: true
  batch_size: 2
  prompt: ""
```

如果 generation reference 需要更有意义的文本输入，建议改用 CSV / JSONL manifest。

## 5. 提取特征

```bash
repralign extract-features --config configs/reference_alignment_template.yaml
```

## 6. 分别对每个 Reference 做分析

```bash
repralign analyze \
  --candidate-cache outputs/reference_alignment/candidate_features.npz \
  --reference-cache outputs/reference_alignment/semantic_reference_features.npz \
  --metric cknna \
  --output-csv outputs/reference_alignment/semantic_cknna.csv \
  --output-json outputs/reference_alignment/semantic_cknna.json
```

然后对 generation reference 重复一次。

## 7. 绘图

```bash
repralign plot \
  --input-csv outputs/reference_alignment/semantic_cknna.csv \
  --input-csv outputs/reference_alignment/generation_cknna.csv \
  --output-png outputs/reference_alignment/cknna_comparison.png \
  --title "Representation Alignment"
```
