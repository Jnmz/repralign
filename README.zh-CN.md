[English](README.md) | [中文](README.zh-CN.md)

# repralign

[![PyPI version](https://img.shields.io/pypi/v/repralign.svg)](https://pypi.org/project/repralign/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

`repralign` 是一个用于多模态模型分层表示对齐分析的可复用 Python 工具包。

文档网站：

- `https://jnmz.github.io/repralign/`

## 安装

通过 PyPI 安装：

```bash
pip install repralign
```

安装 Hugging Face vision 支持：

```bash
pip install "repralign[huggingface]"
```

安装 generation reference 支持：

```bash
pip install "repralign[generation]"
```

从源码安装用于本地开发：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## 许可证

MIT
