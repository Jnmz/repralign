# Python API

## 最小特征提取

```python
from repralign.adapters.generic_torch import GenericTorchAdapter
from repralign.extract import extract_feature_dict
from repralign.metrics import linear_cka

adapter = GenericTorchAdapter(model, layer_names=["encoder.layers.0", "encoder.layers.1"])
features = extract_feature_dict(adapter=adapter, batch=batch, pooling="mean_tokens")
score = linear_cka(features["encoder.layers.0"], features["encoder.layers.1"])
```

## 分批数据集提取

```python
from repralign.adapters.generic_torch import GenericTorchAdapter
from repralign.extract import extract_feature_dataset

adapter = GenericTorchAdapter(model, layer_names=["layers.0", "layers.1"])
features = extract_feature_dataset(
    adapter=adapter,
    batches=my_batches,
    pooling="mean_tokens",
    normalize=True,
)
```
