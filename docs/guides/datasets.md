# Dataset Inputs

`repralign` supports multiple input styles.

## 1. Tensor Batch

Use `extract.batch_path` in YAML when the input is already a `.pt` tensor or a dict of tensors.

## 2. Image Folder

Example:

```yaml
dataset:
  type: image_folder
  image_root: path/to/images
  recursive: true
  batch_size: 4
  prompt: ""
```

Each discovered image becomes a dataset record with:

- `id`
- `image_path`
- `prompt`

You can also generate prompts from filenames:

```yaml
dataset:
  type: image_folder
  image_root: path/to/images
  prompt_template: "an image named {stem}"
```

## 3. Manifest

Example:

```yaml
dataset:
  type: manifest
  manifest_path: data/images.jsonl
  image_root: data
  image_key: image
  prompt_key: prompt
  batch_size: 8
```

Supported manifest formats:

- `.jsonl`
- `.csv`

This is a good option when you want per-image prompts or richer experiment bookkeeping.
