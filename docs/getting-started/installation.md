# Installation

## Install From PyPI

```bash
pip install repralign
```

## Install With Hugging Face Vision Support

```bash
pip install "repralign[huggingface]"
```

## Install With Generation-Reference Support

This extra is intended for Diffusers-based generation references such as Stable Diffusion 3 Medium.

```bash
pip install "repralign[generation]"
```

## Install From Source For Local Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Install Docs Tooling

If you want to build the documentation site locally:

```bash
pip install mkdocs mkdocs-material pymdown-extensions
```

Then run:

```bash
mkdocs serve
```
