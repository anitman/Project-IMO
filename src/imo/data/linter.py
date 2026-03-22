"""Dataset linting and quality validation."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset

logger = logging.getLogger(__name__)


class QualityLevel(Enum):
    """Dataset quality levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXCELLENT = "excellent"


@dataclass
class LintResult:
    """Result of dataset linting."""

    quality_score: float
    quality_level: QualityLevel
    issues: list[str]
    warnings: list[str]
    stats: dict[str, Any]

    def is_acceptable(self, min_score: float = 0.7) -> bool:
        """Check if dataset meets minimum quality threshold."""
        return self.quality_score >= min_score and len(self.issues) == 0


class DatasetLinter:
    """Lint datasets for quality and format validation."""

    def __init__(self, min_quality_score: float = 0.7):
        self.min_quality_score = min_quality_score

    def lint(self, dataset: Dataset) -> LintResult:  # type: ignore
        """Lint a HuggingFace dataset."""
        if not isinstance(dataset, Dataset):
            raise TypeError(f"Expected Dataset, got {type(dataset).__name__}")

        assert isinstance(dataset, Dataset)

        issues: list[str] = []
        warnings: list[str] = []
        stats: dict[str, Any] = {}

        stats["num_samples"] = len(dataset)
        stats["num_columns"] = len(dataset.column_names)
        stats["columns"] = dataset.column_names

        if len(dataset) == 0:
            issues.append("Dataset is empty")

        self._check_schema(dataset, issues, warnings)
        self._check_encoding(dataset, issues, warnings)
        self._check_duplicates(dataset, issues, warnings)
        self._check_toxicity(dataset, issues, warnings)
        self._compute_statistics(dataset, stats)

        quality_score = self._compute_quality_score(issues, warnings, stats)
        quality_level = self._determine_quality_level(quality_score)

        return LintResult(
            quality_score=quality_score,
            quality_level=quality_level,
            issues=issues,
            warnings=warnings,
            stats=stats,
        )

    def lint_file(self, file_path: str | Path) -> LintResult:
        """Lint a dataset file (JSON, Parquet, CSV)."""
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".json":
            dataset = Dataset.from_pandas(pd.read_json(path))
        elif suffix == ".parquet":
            dataset = Dataset.from_parquet(path)
        elif suffix == ".csv":
            dataset = Dataset.from_pandas(pd.read_csv(path))
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        return self.lint(dataset)  # type: ignore

    def _check_schema(self, dataset: Dataset, issues: list[str], warnings: list[str]) -> None:
        """Validate dataset schema and column types."""
        required_columns = {"text", "content", "prompt", "messages"}

        has_required = bool(set(dataset.column_names) & required_columns)
        if not has_required:
            warnings.append(f"No standard text column found. Columns: {dataset.column_names}")

        for col in dataset.column_names:
            sample = dataset[0][col]
            if not isinstance(sample, (str, int, float, bool, type(None))):
                warnings.append(f"Column '{col}' has complex type: {type(sample).__name__}")

    def _check_encoding(self, dataset: Dataset, issues: list[str], warnings: list[str]) -> None:
        """Check for encoding issues and binary artifacts."""
        text_columns = [c for c in dataset.column_names if c in {"text", "content", "prompt"}]

        for col in text_columns:
            for i, row in enumerate(dataset):
                if not isinstance(row, dict):
                    continue
                value = row.get(col)
                if value is None:
                    continue
                if not isinstance(value, str):
                    continue

                if "\ufffd" in value:
                    warnings.append(f"Unicode replacement char found in {col} at row {i}")
                    break

                if any(ord(c) > 0xFFFF for c in value[:1000]):
                    warnings.append(f"Non-BMP characters found in {col}")
                    break

    def _check_duplicates(self, dataset: Dataset, issues: list[str], warnings: list[str]) -> None:
        """Check for near-duplicate samples."""
        text_columns = [c for c in dataset.column_names if c in {"text", "content", "prompt"}]

        if not text_columns:
            return

        col = text_columns[0]
        hashes = set()
        duplicates = 0

        for row in dataset:
            if not isinstance(row, dict):
                continue
            value = row.get(col, "")
            if not isinstance(value, str):
                continue

            text_hash = hash(value[:1000])
            if text_hash in hashes:
                duplicates += 1
            else:
                hashes.add(text_hash)

        dup_ratio = duplicates / len(dataset) if len(dataset) > 0 else 0
        if dup_ratio > 0.5:
            issues.append(f"High duplicate rate: {dup_ratio:.1%}")
        elif dup_ratio > 0.1:
            warnings.append(f"Duplicate samples detected: {dup_ratio:.1%}")

    def _check_toxicity(self, dataset: Dataset, issues: list[str], warnings: list[str]) -> None:
        """Screen for toxic content."""
        toxic_patterns = [
            r"\b(nigger|faggot|cunt)\b",
            r"(^|\s)(!?\w){3,}$",
        ]

        text_columns = [c for c in dataset.column_names if c in {"text", "content"}]
        if not text_columns:
            return

        col = text_columns[0]
        toxic_count = 0

        for row in dataset:
            if not isinstance(row, dict):
                continue
            value = row.get(col, "")
            if not isinstance(value, str):
                continue

            for pattern in toxic_patterns:
                if re.search(pattern, value, re.IGNORECASE):
                    toxic_count += 1
                    break

        if toxic_count > 0:
            warning_ratio = toxic_count / len(dataset) if len(dataset) > 0 else 0
            warnings.append(
                f"Potentially toxic content: {toxic_count} samples ({warning_ratio:.1%})"
            )

    def _compute_statistics(self, dataset: Dataset, stats: dict[str, Any]) -> None:
        """Compute statistical profile of dataset."""
        text_columns = [c for c in dataset.column_names if c in {"text", "content", "prompt"}]

        if not text_columns:
            return

        col = text_columns[0]
        lengths = []

        for row in dataset:
            if not isinstance(row, dict):
                continue
            value = row.get(col, "")
            if isinstance(value, str):
                lengths.append(len(value))

        if lengths:
            stats["avg_length"] = sum(lengths) / len(lengths)
            stats["min_length"] = min(lengths)
            stats["max_length"] = max(lengths)
            stats["length_std"] = (
                sum((x - stats["avg_length"]) ** 2 for x in lengths) / len(lengths)
            ) ** 0.5

    def _compute_quality_score(
        self, issues: list[str], warnings: list[str], stats: dict[str, Any]
    ) -> float:
        """Compute overall quality score."""
        score = 1.0

        score -= len(issues) * 0.2
        score -= len(warnings) * 0.05

        if stats.get("num_samples", 0) < 100:
            score -= 0.1

        avg_length = stats.get("avg_length", 0)
        if 0 < avg_length < 10:
            score -= 0.1
        elif avg_length > 100000:
            score -= 0.05

        return max(0.0, min(1.0, score))

    def _determine_quality_level(self, score: float) -> QualityLevel:
        """Map quality score to level."""
        if score >= 0.9:
            return QualityLevel.EXCELLENT
        elif score >= 0.75:
            return QualityLevel.HIGH
        elif score >= 0.6:
            return QualityLevel.MEDIUM
        else:
            return QualityLevel.LOW


class CleanlabLinter:
    """ML-powered data quality detection using Cleanlab.

    Complements rule-based DatasetLinter with learned quality signals:
    - Label error detection (confident learning)
    - Near-duplicate detection beyond exact hash matching
    - Outlier detection that understands the data distribution
    - Per-sample quality scores

    Cleanlab uses the model's own predictions to find samples where the
    given label disagrees with what the model has learned — a much stronger
    signal than simple heuristics.

    Requires: pip install cleanlab
    Reference: https://github.com/cleanlab/cleanlab
    """

    def __init__(
        self,
        label_column: str = "label",
        text_column: str = "text",
        confidence_threshold: float = 0.5,
    ):
        self.label_column = label_column
        self.text_column = text_column
        self.confidence_threshold = confidence_threshold

    def find_label_issues(
        self,
        labels: list[int],
        pred_probs: Any,
    ) -> LintResult:
        """Find label errors using Cleanlab's confident learning.

        Args:
            labels: Ground truth labels (integer-encoded).
            pred_probs: Model's predicted class probabilities (N x K numpy array).
                        Typically from cross-validated predictions.
        """
        try:
            from cleanlab.filter import (
                find_label_issues as cl_find_issues,  # type: ignore[import-not-found]
            )
            from cleanlab.rank import get_label_quality_scores  # type: ignore[import-not-found]
        except ImportError:
            return LintResult(
                quality_score=0.0,
                quality_level=QualityLevel.LOW,
                issues=["Cleanlab not installed — run: pip install cleanlab"],
                warnings=[],
                stats={},
            )

        import numpy as np

        labels_arr = np.array(labels)

        issue_mask = cl_find_issues(labels_arr, pred_probs)
        quality_scores = get_label_quality_scores(labels_arr, pred_probs)

        issue_indices = list(np.where(issue_mask)[0])
        low_quality_indices = list(np.where(quality_scores < self.confidence_threshold)[0])

        issues: list[str] = []
        warnings: list[str] = []
        stats: dict[str, Any] = {
            "num_label_issues": len(issue_indices),
            "label_issue_indices": issue_indices[:100],
            "avg_label_quality": float(np.mean(quality_scores)),
            "min_label_quality": float(np.min(quality_scores)),
            "num_low_quality": len(low_quality_indices),
        }

        issue_rate = len(issue_indices) / len(labels) if labels else 0
        if issue_rate > 0.2:
            issues.append(
                f"High label error rate: {issue_rate:.1%} ({len(issue_indices)} samples)"
            )
        elif issue_rate > 0.05:
            warnings.append(
                f"Label errors detected: {issue_rate:.1%} ({len(issue_indices)} samples)"
            )

        if stats["avg_label_quality"] < 0.5:
            issues.append(f"Low average label quality: {stats['avg_label_quality']:.3f}")

        quality_score = max(0.0, min(1.0, 1.0 - issue_rate - len(issues) * 0.15))

        return LintResult(
            quality_score=quality_score,
            quality_level=self._determine_quality_level(quality_score),
            issues=issues,
            warnings=warnings,
            stats=stats,
        )

    def find_outliers(
        self,
        features: Any,
    ) -> dict[str, Any]:
        """Find outlier samples using Cleanlab's OutOfDistribution detector.

        Args:
            features: Feature embeddings (N x D numpy array).
                      E.g. from a sentence-transformer or CLIP encoder.
        """
        try:
            from cleanlab.outlier import OutOfDistribution  # type: ignore[import-not-found]
        except ImportError:
            return {"error": "Cleanlab not installed — run: pip install cleanlab"}

        ood = OutOfDistribution()
        ood_scores = ood.fit_score(features=features)

        import numpy as np
        outlier_indices = list(np.where(ood_scores < self.confidence_threshold)[0])

        return {
            "num_outliers": len(outlier_indices),
            "outlier_indices": outlier_indices[:100],
            "avg_ood_score": float(np.mean(ood_scores)),
            "min_ood_score": float(np.min(ood_scores)),
        }

    def _determine_quality_level(self, score: float) -> QualityLevel:
        """Map quality score to level."""
        if score >= 0.9:
            return QualityLevel.EXCELLENT
        elif score >= 0.75:
            return QualityLevel.HIGH
        elif score >= 0.6:
            return QualityLevel.MEDIUM
        else:
            return QualityLevel.LOW
