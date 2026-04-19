from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

from repralign.datasets import (
    batch_to_image_records,
    iter_batch_records,
    load_dataset_records,
)


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (8, 8), color=color)
    image.save(path)


def test_load_image_folder_records(tmp_path: Path) -> None:
    _write_image(tmp_path / "a.png", (255, 0, 0))
    _write_image(tmp_path / "b.png", (0, 255, 0))

    records = load_dataset_records(
        {
            "type": "image_folder",
            "image_root": str(tmp_path),
            "batch_size": 2,
            "prompt_template": "describe {stem}",
        }
    )

    assert len(records) == 2
    assert records[0]["prompt"].startswith("describe")


def test_load_manifest_records_from_jsonl(tmp_path: Path) -> None:
    _write_image(tmp_path / "sample.png", (0, 0, 255))
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(json.dumps({"image": "sample.png", "prompt": "hello"}) + "\n", encoding="utf-8")

    records = load_dataset_records(
        {
            "type": "manifest",
            "manifest_path": str(manifest_path),
            "image_root": str(tmp_path),
            "batch_size": 1,
        }
    )
    assert records == [{"id": "0", "image_path": str(tmp_path / "sample.png"), "prompt": "hello"}]


def test_iter_batch_records_and_batch_to_image_records(tmp_path: Path) -> None:
    _write_image(tmp_path / "one.png", (10, 20, 30))
    records = [{"id": "one", "image_path": str(tmp_path / "one.png"), "prompt": ""}]
    batches = list(iter_batch_records(records, batch_size=1))

    assert len(batches) == 1
    loaded = batch_to_image_records(batches[0])
    assert loaded is not None
    assert loaded[0]["image"].size == (8, 8)
