"""Plotting utilities for layer-wise alignment curves."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Union

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def write_similarity_csv(path: Union[str, Path], rows: list[dict[str, object]]) -> Path:
    """Write similarity rows to CSV."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows provided for CSV output")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_similarity_csv(path: Union[str, Path]) -> list[dict[str, str]]:
    """Read similarity rows from CSV."""

    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def plot_layer_alignment(
    curve_groups: list[dict[str, object]],
    output_path: Union[str, Path],
    title: str,
    ylabel: str = "Similarity",
) -> Path:
    """Render one or more layer-wise similarity curves."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for idx, group in enumerate(curve_groups):
        x_values = group["x"]
        y_values = group["y"]
        label = str(group["label"])
        ax.plot(x_values, y_values, marker="o", linewidth=2.0, color=colors[idx % len(colors)], label=label)

    ax.set_title(title)
    ax.set_xlabel("Candidate Layer Index")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_csv_curves(csv_paths: list[Union[str, Path]], output_path: Union[str, Path], title: str) -> Path:
    """Read one or more CSV files and render them as curves."""

    curves: list[dict[str, object]] = []
    for csv_path in csv_paths:
        rows = read_similarity_csv(csv_path)
        curves.append(
            {
                "label": Path(csv_path).stem,
                "x": [int(row["candidate_layer_index"]) for row in rows],
                "y": [float(row["score"]) for row in rows],
            }
        )
    return plot_layer_alignment(curves, output_path=output_path, title=title)
