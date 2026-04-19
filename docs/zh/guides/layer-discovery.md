# 层发现

正确选层对实验质量影响很大。

## 列出模型的命名模块

```bash
repralign list-layers \
  --factory examples.model_factories:build_toy_candidate \
  --adapter generic_torch
```

做真实实验时，把 toy factory 替换成你自己的模型工厂即可。

## 一般 hook 哪些层

通常按模型类型来选：

- vision backbone：encoder blocks 或 transformer layers
- multimodal encoder：representation blocks 或 fusion blocks
- diffusion reference：denoising transformer 的 block

## 常见错误

- hook 到返回 tuple 的模块但没有检查输出结构
- 不小心把 token-level 特征和 pooled 特征混着比
- reference 和 candidate 的预处理不一致
