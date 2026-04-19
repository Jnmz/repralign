# YAML 配置

`repralign` 的设计是显式配置驱动的。YAML 里会描述：

- 模型工厂
- adapter 类型
- processor 工厂
- adapter 参数
- layer names
- dataset 来源
- pooling / normalize
- 输出目录

## 张量 Batch 风格

参考：

- `configs/example_analysis.yaml`

适合：

- toy model
- synthetic test
- 已经准备好的 tensor batch

## 数据集驱动风格

参考：

- `configs/reference_alignment_template.yaml`

适合：

- 图片目录
- CSV / JSONL manifest
- semantic + generation reference 的完整实验

## 模型配置字段

```yaml
name: semantic_reference
adapter: hf_vision
factory: repralign.factories:load_siglip_vision_model
processor_factory: repralign.factories:load_siglip_image_processor
factory_kwargs:
  model_name: google/siglip-so400m-patch14-384
processor_kwargs:
  model_name: google/siglip-so400m-patch14-384
layers:
  - vision_model.encoder.layers.0
device: cpu
```

关键字段：

- `adapter`
- `factory`
- `processor_factory`
- `factory_kwargs`
- `processor_kwargs`
- `adapter_kwargs`
- `layers`
- `device`
