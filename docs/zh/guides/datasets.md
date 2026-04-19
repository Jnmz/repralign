# 数据集输入

`repralign` 当前支持多种输入方式。

## 1. Tensor Batch

当输入已经是 `.pt` tensor 或 tensor dict 时，可以用 `extract.batch_path`。

## 2. 图片目录

```yaml
dataset:
  type: image_folder
  image_root: path/to/images
  recursive: true
  batch_size: 4
  prompt: ""
```

每张图片会形成一条 record，通常包含：

- `id`
- `image_path`
- `prompt`

也可以从文件名自动生成 prompt：

```yaml
dataset:
  type: image_folder
  image_root: path/to/images
  prompt_template: "an image named {stem}"
```

## 3. Manifest

```yaml
dataset:
  type: manifest
  manifest_path: data/images.jsonl
  image_root: data
  image_key: image
  prompt_key: prompt
  batch_size: 8
```

支持格式：

- `.jsonl`
- `.csv`
