"""Reusable layer-wise representation alignment analysis."""

from repralign.extract import extract_feature_dataset, extract_feature_dict
from repralign.pooling import apply_pooling
from repralign.registry import create_adapter, compute_metric

__all__ = ["apply_pooling", "compute_metric", "create_adapter", "extract_feature_dataset", "extract_feature_dict"]

__version__ = "0.1.0"
