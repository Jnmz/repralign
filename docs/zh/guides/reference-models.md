# 参考模型

## Semantic Reference

内置 semantic 路线是基于 Hugging Face vision encoder。

当前可直接用的工厂：

- `repralign.factories:load_siglip_vision_model`
- `repralign.factories:load_siglip_image_processor`
- `repralign.factories:load_hf_image_processor`

适合：

- SigLIP
- ViT 风格视觉骨干
- 其他带稳定 named modules 的 HF vision model

## Generation Reference

内置 generation 路线基于 Diffusers pipeline。

当前工厂：

- `repralign.factories:load_sd3_pipeline`

当前 adapter：

- `sd3_reference`

这个路径会：

- 用 VAE 编码图片
- 用 pipeline 的文本编码器编码 prompt
- hook SD3 transformer blocks
- 把 denoising transformer 作为 generation-oriented reference source

## 关于 SD3-Medium

仓库已经支持把 SD3 作为 generation reference，但具体实验仍然依赖这些选择：

- prompt 文本
- timestep
- 图像分辨率
- hook 哪些 transformer blocks
- Diffusers 版本和权重
