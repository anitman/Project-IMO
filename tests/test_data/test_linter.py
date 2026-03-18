"""Tests for dataset linting and quality validation."""

from __future__ import annotations

import pytest
from datasets import Dataset

from imo.data.linter import DatasetLinter, QualityLevel


class TestDatasetLinter:
    """Test DatasetLinter functionality."""

    def test_lint_empty_dataset(self) -> None:
        """Test linting an empty dataset."""
        linter = DatasetLinter()
        dataset = Dataset.from_list([])
        result = linter.lint(dataset)

        assert result.quality_score < 1.0
        assert "Dataset is empty" in result.issues

    def test_lint_valid_dataset(self) -> None:
        """Test linting a valid dataset."""
        linter = DatasetLinter()
        dataset = Dataset.from_list(
            [
                {"text": "This is a sample text.", "label": 0},
                {"text": "Another sample.", "label": 1},
            ]
        )
        result = linter.lint(dataset)

        assert result.quality_score >= 0.7
        assert result.is_acceptable()

    def test_lint_dataset_with_duplicates(self) -> None:
        """Test linting dataset with duplicates."""
        linter = DatasetLinter()
        dataset = Dataset.from_list(
            [{"text": f"Text {i}"} for i in range(5)] + [{"text": "Same text"} for _ in range(5)]
        )
        result = linter.lint(dataset)

        assert "Duplicate" in str(result.issues + result.warnings)

    def test_lint_dataset_with_toxic_content(self) -> None:
        """Test linting dataset with toxic content."""
        linter = DatasetLinter()
        dataset = Dataset.from_list(
            [
                {"text": "Normal text"},
                {"text": "This contains bad words nigger here"},
            ]
        )
        result = linter.lint(dataset)

        assert len(result.warnings) > 0

    def test_quality_level_mapping(self) -> None:
        """Test quality level mapping."""
        assert QualityLevel.EXCELLENT.value == "excellent"
        assert QualityLevel.HIGH.value == "high"
        assert QualityLevel.MEDIUM.value == "medium"
        assert QualityLevel.LOW.value == "low"

    def test_lint_file_json(self, tmp_path) -> None:
        """Test linting a JSON file."""
        linter = DatasetLinter()
        file_path = tmp_path / "test.json"
        file_path.write_text('[{"text": "Sample text"}]')

        result = linter.lint_file(str(file_path))
        assert result.quality_score > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
