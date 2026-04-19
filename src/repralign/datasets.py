"""Config-driven dataset loading utilities."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Optional

from PIL import Image


IMAGE_PATTERNS = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")


def load_dataset_records(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Load dataset records from a config dictionary."""

    dataset_type = str(spec.get("type", "image_folder"))
    if dataset_type == "image_folder":
        return load_image_folder_records(
            root=spec["image_root"],
            recursive=bool(spec.get("recursive", False)),
            patterns=tuple(spec.get("patterns", IMAGE_PATTERNS)),
            limit=spec.get("limit"),
            prompt=spec.get("prompt"),
            prompt_template=spec.get("prompt_template"),
        )
    if dataset_type == "manifest":
        return load_manifest_records(
            manifest_path=spec["manifest_path"],
            image_root=spec.get("image_root"),
            image_key=str(spec.get("image_key", "image")),
            prompt_key=str(spec.get("prompt_key", "prompt")),
            limit=spec.get("limit"),
        )
    raise ValueError(f"Unsupported dataset type: {dataset_type}")


def load_image_folder_records(
    root: str,
    recursive: bool = False,
    patterns: tuple[str, ...] = IMAGE_PATTERNS,
    limit: Optional[int] = None,
    prompt: Optional[str] = None,
    prompt_template: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Load image records from a folder of image files."""

    root_path = Path(root)
    globber = root_path.rglob if recursive else root_path.glob
    image_paths: list[Path] = []
    for pattern in patterns:
        image_paths.extend(sorted(globber(pattern)))

    deduped = sorted({path.resolve() for path in image_paths})
    if limit is not None:
        deduped = deduped[: int(limit)]

    records = []
    for path in deduped:
        stem = path.stem
        resolved_prompt = prompt
        if prompt_template is not None:
            resolved_prompt = prompt_template.format(stem=stem, filename=path.name, path=str(path))
        records.append(
            {
                "id": stem,
                "image_path": str(path),
                "prompt": resolved_prompt or "",
            }
        )
    return records


def load_manifest_records(
    manifest_path: str,
    image_root: Optional[str] = None,
    image_key: str = "image",
    prompt_key: str = "prompt",
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Load image records from a CSV or JSONL manifest."""

    manifest = Path(manifest_path)
    suffix = manifest.suffix.lower()
    if suffix == ".jsonl":
        rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif suffix == ".csv":
        with manifest.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        raise ValueError("Manifest datasets currently support .jsonl and .csv files")

    root = Path(image_root) if image_root is not None else None
    records = []
    for idx, row in enumerate(rows):
        image_value = row[image_key]
        image_path = Path(image_value)
        if root is not None and not image_path.is_absolute():
            image_path = root / image_path
        records.append(
            {
                "id": str(row.get("id", idx)),
                "image_path": str(image_path),
                "prompt": str(row.get(prompt_key, "")),
            }
        )

    if limit is not None:
        records = records[: int(limit)]
    return records


def iter_batch_records(records: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    """Yield lists of dataset records."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


def batch_to_image_records(batch: object) -> Optional[list[dict[str, Any]]]:
    """Normalize a raw batch into image records with loaded PIL images."""

    if not isinstance(batch, (list, tuple)):
        return None

    records: list[dict[str, Any]] = []
    for idx, item in enumerate(batch):
        if isinstance(item, dict):
            record = dict(item)
        elif isinstance(item, (str, Path)):
            record = {"id": str(idx), "image_path": str(item), "prompt": ""}
        else:
            return None

        if "image" not in record:
            image_path = record.get("image_path")
            if image_path is None:
                return None
            record["image"] = Image.open(image_path).convert("RGB")
        records.append(record)
    return records


def batch_to_images(batch: object) -> Optional[list[Image.Image]]:
    """Extract PIL images from a raw batch."""

    records = batch_to_image_records(batch)
    if records is None:
        return None
    return [record["image"] for record in records]
