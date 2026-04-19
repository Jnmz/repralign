# 安装

## 通过 PyPI 安装

```bash
pip install repralign
```

## 安装 Hugging Face Vision 支持

```bash
pip install "repralign[huggingface]"
```

## 安装 Generation Reference 支持

这个 extra 用于 SD3 这类基于 Diffusers 的 generation reference。

```bash
pip install "repralign[generation]"
```

## 从源码安装用于本地开发

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## 本地预览文档站

```bash
pip install mkdocs mkdocs-material pymdown-extensions mkdocs-static-i18n
mkdocs serve
```
