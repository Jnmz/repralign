# 常见问题

## 模型能加载，但 layer name 不对

先运行 `repralign list-layers`，从输出里复制精确名称。

## SD3 reference 太重，机器扛不住

先尝试降低：

- 图像分辨率
- batch size
- hook 层数
- 精度

也可以先只跑 semantic reference，把流程跑通后再加 generation reference。

## 结果图和论文里的不一样

通常是这些因素不一致：

- candidate model
- reference checkpoint
- dataset
- prompt 设置
- layer 选择
- metric / k 设置

## 我只有图片，没有 prompt

对 semantic reference，可以直接用空 prompt。  
对 generation reference，你需要自己决定：

- 空 prompt 是否合理
- 是否使用统一 default prompt
- 是否为每张图片提供单独 prompt
