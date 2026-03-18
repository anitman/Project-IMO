"""Tests for dataset specifications."""

from __future__ import annotations

import pytest

from imo.data.dataset_spec import (
    DataType,
    DatasetRequirement,
    DatasetSpec,
    ModelCategory,
    get_model_category_description,
)


class TestDatasetSpec:
    """Test DatasetSpec functionality."""

    def test_get_llm_spec(self) -> None:
        """Test LLM dataset specification."""
        spec = DatasetSpec.from_category(ModelCategory.LLM, "test_llm")

        assert spec.model_category == ModelCategory.LLM
        assert len(spec.requirements) == 1
        assert spec.requirements[0].data_type == DataType.TEXT
        assert spec.requirements[0].required is True
        assert spec.requirements[0].min_samples == 1_000_000

    def test_get_vlm_spec(self) -> None:
        """Test Vision-Language Model dataset specification."""
        spec = DatasetSpec.from_category(ModelCategory.MULTIMODAL_VLM, "test_vlm")

        assert spec.model_category == ModelCategory.MULTIMODAL_VLM
        assert len(spec.requirements) == 2

        data_types = {r.data_type for r in spec.requirements}
        assert DataType.IMAGE in data_types
        assert DataType.TEXT in data_types

    def test_get_speech_recognition_spec(self) -> None:
        """Test speech recognition dataset specification."""
        spec = DatasetSpec.from_category(
            ModelCategory.AUDIO_SPEECH_RECOGNITION,
            "test_asr",
        )

        assert spec.model_category == ModelCategory.AUDIO_SPEECH_RECOGNITION
        assert len(spec.requirements) == 2

        data_types = {r.data_type for r in spec.requirements}
        assert DataType.AUDIO in data_types
        assert DataType.TEXT in data_types

    def test_validation_pass(self) -> None:
        """Test validation with valid dataset."""
        spec = DatasetSpec.from_category(ModelCategory.LLM, "test_llm")

        dataset_info = {
            "data_types": [DataType.TEXT],
            "num_samples": 2_000_000,
            "size_mb": 5000,
            "format": "jsonl",
        }

        is_valid, errors = spec.validate(dataset_info)
        assert is_valid
        assert len(errors) == 0

    def test_validation_fail_missing_data_type(self) -> None:
        """Test validation fails with missing data type."""
        spec = DatasetSpec.from_category(ModelCategory.VISION_CLASSIFICATION, "test")

        dataset_info = {
            "data_types": [DataType.LABEL],
            "num_samples": 100_000,
            "size_mb": 1000,
            "format": "parquet",
        }

        is_valid, errors = spec.validate(dataset_info)
        assert not is_valid
        assert any("image" in error.lower() for error in errors)

    def test_validation_fail_insufficient_samples(self) -> None:
        """Test validation fails with insufficient samples."""
        spec = DatasetSpec.from_category(ModelCategory.LLM, "test_llm")

        dataset_info = {
            "data_types": [DataType.TEXT],
            "num_samples": 100,
            "size_mb": 1000,
            "format": "jsonl",
        }

        is_valid, errors = spec.validate(dataset_info)
        assert not is_valid
        assert any("Insufficient samples" in error for error in errors)

    def test_validation_fail_format(self) -> None:
        """Test validation fails with unsupported format."""
        spec = DatasetSpec.from_category(ModelCategory.LLM, "test_llm")

        dataset_info = {
            "data_types": [DataType.TEXT],
            "num_samples": 2_000_000,
            "size_mb": 5000,
            "format": "csv",
        }

        is_valid, errors = spec.validate(dataset_info)
        assert not is_valid
        assert any("Format" in error for error in errors)

    def test_get_all_model_categories(self) -> None:
        """Test that all model categories have descriptions."""
        for category in ModelCategory:
            description = get_model_category_description(category)
            assert len(description) > 0

    def test_video_understanding_spec(self) -> None:
        """Test video understanding dataset specification."""
        spec = DatasetSpec.from_category(
            ModelCategory.VIDEO_UNDERSTANDING,
            "test_video",
        )

        assert spec.model_category == ModelCategory.VIDEO_UNDERSTANDING
        assert len(spec.requirements) == 2

        data_types = {r.data_type for r in spec.requirements}
        assert DataType.VIDEO in data_types
        assert DataType.TEXT in data_types


class TestDatasetRequirement:
    """Test DatasetRequirement functionality."""

    def test_requirement_creation(self) -> None:
        """Test creating a dataset requirement."""
        req = DatasetRequirement(
            data_type=DataType.TEXT,
            required=True,
            min_samples=1000,
            min_size_mb=100,
            accepted_formats=["jsonl", "parquet"],
            description="Test requirement",
        )

        assert req.data_type == DataType.TEXT
        assert req.required is True
        assert req.min_samples == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
