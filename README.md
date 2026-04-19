[English](README.md) | [中文](README.zh-CN.md)

# repralign

[![PyPI version](https://img.shields.io/pypi/v/repralign.svg)](https://pypi.org/project/repralign/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

`repralign` is a reusable Python toolkit for layer-wise representation alignment analysis in multimodal models.

Documentation:

- Docs site: `https://jnmz.github.io/repralign/`

## Installation

Install from PyPI:

```bash
pip install repralign
```

Install with Hugging Face vision support:

```bash
pip install "repralign[huggingface]"
```

Install with generation-reference support:

```bash
pip install "repralign[generation]"
```

Install from source for local development:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## License

MIT
