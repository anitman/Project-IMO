"""Dataset types and specifications for different model categories."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ModelCategory(Enum):
    """Model categories defining dataset requirements."""

    # Language Models
    LLM = "llm"  # Large Language Models
    LLM_CHAT = "llm_chat"  # Chat/Instruction-tuned LLMs
    LLM_CODE = "llm_code"  # Code generation LLMs

    # Vision Models
    VISION_CLASSIFICATION = "vision_classification"
    VISION_DETECTION = "vision_detection"
    VISION_SEGMENTATION = "vision_segmentation"
    VISION_GENERATION = "vision_generation"

    # Audio Models
    AUDIO_SPEECH_RECOGNITION = "audio_speech_recognition"
    AUDIO_TEXT_TO_SPEECH = "audio_text_to_speech"
    AUDIO_CLASSIFICATION = "audio_classification"
    AUDIO_SOUND_GENERATION = "audio_sound_generation"

    # Video Models
    VIDEO_CLASSIFICATION = "video_classification"
    VIDEO_UNDERSTANDING = "video_understanding"
    VIDEO_GENERATION = "video_generation"

    # Multimodal Models
    MULTIMODAL_VLM = "multimodal_vlm"  # Vision-Language Models
    MULTIMODAL_AUDIO_VLM = "multimodal_audio_vlm"  # Audio-Language Models
    MULTIMODAL_VIDEO_VLM = "multimodal_video_vlm"  # Video-Language Models

    # Embedding Models
    EMBEDDING_TEXT = "embedding_text"
    EMBEDDING_IMAGE = "embedding_image"
    EMBEDDING_MULTIMODAL = "embedding_multimodal"

    # Time-series Models
    TIME_SERIES_FORECASTING = "time_series_forecasting"
    TIME_SERIES_CLASSIFICATION = "time_series_classification"

    # Recommendation Models
    RECOMMENDATION = "recommendation"


class DataType(Enum):
    """Data types that can be in datasets."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    TABULAR = "tabular"
    EMBEDDING = "embedding"
    LABEL = "label"
    MASK = "mask"  # For segmentation
    BOUNDING_BOX = "bounding_box"  # For object detection
    KEYPOINTS = "keypoints"  # For pose estimation


@dataclass
class DatasetRequirement:
    """Requirement for a specific data type in a dataset."""

    data_type: DataType
    required: bool
    min_samples: int = 0
    min_size_mb: float = 0.0
    max_size_mb: float = 0.0
    accepted_formats: list[str] = field(default_factory=list)
    description: str = ""


# Dataset specifications for each model category
MODEL_DATASET_SPECS: dict[ModelCategory, list[DatasetRequirement]] = {
    # ==================== LANGUAGE MODELS ====================
    ModelCategory.LLM: [
        DatasetRequirement(
            data_type=DataType.TEXT,
            required=True,
            min_samples=1_000_000,
            min_size_mb=1000,
            accepted_formats=["jsonl", "parquet", "json"],
            description="Raw text data for pretraining",
        ),
    ],
    ModelCategory.LLM_CHAT: [
        DatasetRequirement(
            data_type=DataType.TEXT,
            required=True,
            min_samples=100_000,
            accepted_formats=["jsonl", "parquet", "json"],
            description="Multi-turn conversation data",
        ),
        DatasetRequirement(
            data_type=DataType.LABEL,
            required=False,
            description="Preference/reward labels for RLHF",
        ),
    ],
    ModelCategory.LLM_CODE: [
        DatasetRequirement(
            data_type=DataType.TEXT,
            required=True,
            min_samples=500_000,
            accepted_formats=["jsonl", "parquet"],
            description="Code snippets and documentation",
        ),
    ],
    # ==================== VISION MODELS ====================
    ModelCategory.VISION_CLASSIFICATION: [
        DatasetRequirement(
            data_type=DataType.IMAGE,
            required=True,
            min_samples=10_000,
            accepted_formats=["parquet", "tfrecord", "webdataset"],
            description="Images with class labels",
        ),
        DatasetRequirement(
            data_type=DataType.LABEL,
            required=True,
            description="Class labels (single or multi-class)",
        ),
    ],
    ModelCategory.VISION_DETECTION: [
        DatasetRequirement(
            data_type=DataType.IMAGE,
            required=True,
            min_samples=5_000,
            accepted_formats=["parquet", "coco", "voc"],
            description="Images for object detection",
        ),
        DatasetRequirement(
            data_type=DataType.BOUNDING_BOX,
            required=True,
            description="Bounding box annotations (x1, y1, x2, y2, class)",
        ),
    ],
    ModelCategory.VISION_SEGMENTATION: [
        DatasetRequirement(
            data_type=DataType.IMAGE,
            required=True,
            min_samples=1_000,
            accepted_formats=["parquet", "webdataset"],
            description="Images for segmentation",
        ),
        DatasetRequirement(
            data_type=DataType.MASK,
            required=True,
            description="Pixel-level segmentation masks",
        ),
    ],
    ModelCategory.VISION_GENERATION: [
        DatasetRequirement(
            data_type=DataType.IMAGE,
            required=True,
            min_samples=100_000,
            min_size_mb=10_000,
            accepted_formats=["parquet", "tfrecord"],
            description="Images for training generative models",
        ),
        DatasetRequirement(
            data_type=DataType.TEXT,
            required=False,
            description="Text captions for text-to-image models",
        ),
    ],
    # ==================== AUDIO MODELS ====================
    ModelCategory.AUDIO_SPEECH_RECOGNITION: [
        DatasetRequirement(
            data_type=DataType.AUDIO,
            required=True,
            min_samples=10_000,
            min_size_mb=1000,
            accepted_formats=["parquet", "webdataset"],
            description="Audio recordings",
        ),
        DatasetRequirement(
            data_type=DataType.TEXT,
            required=True,
            description="Transcriptions (text labels)",
        ),
    ],
    ModelCategory.AUDIO_TEXT_TO_SPEECH: [
        DatasetRequirement(
            data_type=DataType.TEXT,
            required=True,
            min_samples=5_000,
            description="Text input for synthesis",
        ),
        DatasetRequirement(
            data_type=DataType.AUDIO,
            required=True,
            min_samples=5_000,
            description="Corresponding audio output",
        ),
        DatasetRequirement(
            data_type=DataType.LABEL,
            required=False,
            description="Speaker identity labels",
        ),
    ],
    ModelCategory.AUDIO_CLASSIFICATION: [
        DatasetRequirement(
            data_type=DataType.AUDIO,
            required=True,
            min_samples=1_000,
            accepted_formats=["parquet", "webdataset"],
            description="Audio clips",
        ),
        DatasetRequirement(
            data_type=DataType.LABEL,
            required=True,
            description="Sound event/class labels",
        ),
    ],
    ModelCategory.AUDIO_SOUND_GENERATION: [
        DatasetRequirement(
            data_type=DataType.AUDIO,
            required=True,
            min_samples=50_000,
            min_size_mb=5000,
            accepted_formats=["parquet", "webdataset"],
            description="Audio samples for generation training",
        ),
        DatasetRequirement(
            data_type=DataType.TEXT,
            required=False,
            description="Text descriptions or prompts for text-to-audio",
        ),
    ],
    # ==================== VIDEO MODELS ====================
    ModelCategory.VIDEO_CLASSIFICATION: [
        DatasetRequirement(
            data_type=DataType.VIDEO,
            required=True,
            min_samples=1_000,
            min_size_mb=5000,
            accepted_formats=["parquet", "webdataset"],
            description="Video clips",
        ),
        DatasetRequirement(
            data_type=DataType.LABEL,
            required=True,
            description="Action/class labels",
        ),
    ],
    ModelCategory.VIDEO_UNDERSTANDING: [
        DatasetRequirement(
            data_type=DataType.VIDEO,
            required=True,
            min_samples=10_000,
            min_size_mb=10_000,
            accepted_formats=["parquet", "webdataset"],
            description="Video clips",
        ),
        DatasetRequirement(
            data_type=DataType.TEXT,
            required=True,
            description="Captions, descriptions, or Q&A pairs",
        ),
    ],
    ModelCategory.VIDEO_GENERATION: [
        DatasetRequirement(
            data_type=DataType.VIDEO,
            required=True,
            min_samples=100_000,
            min_size_mb=50_000,
            accepted_formats=["parquet", "tfrecord"],
            description="Video data for generation",
        ),
        DatasetRequirement(
            data_type=DataType.TEXT,
            required=False,
            description="Text prompts for text-to-video",
        ),
    ],
    # ==================== MULTIMODAL MODELS ====================
    ModelCategory.MULTIMODAL_VLM: [
        DatasetRequirement(
            data_type=DataType.IMAGE,
            required=True,
            min_samples=100_000,
            accepted_formats=["parquet", "webdataset"],
            description="Images",
        ),
        DatasetRequirement(
            data_type=DataType.TEXT,
            required=True,
            min_samples=100_000,
            description="Text captions, Q&A, or instructions",
        ),
    ],
    ModelCategory.MULTIMODAL_AUDIO_VLM: [
        DatasetRequirement(
            data_type=DataType.AUDIO,
            required=True,
            min_samples=50_000,
            accepted_formats=["parquet", "webdataset"],
            description="Audio data",
        ),
        DatasetRequirement(
            data_type=DataType.TEXT,
            required=True,
            min_samples=50_000,
            description="Text captions, transcriptions, or Q&A",
        ),
    ],
    ModelCategory.MULTIMODAL_VIDEO_VLM: [
        DatasetRequirement(
            data_type=DataType.VIDEO,
            required=True,
            min_samples=10_000,
            min_size_mb=10_000,
            accepted_formats=["parquet", "webdataset"],
            description="Video data",
        ),
        DatasetRequirement(
            data_type=DataType.TEXT,
            required=True,
            min_samples=10_000,
            description="Captions, descriptions, or Q&A",
        ),
    ],
    # ==================== EMBEDDING MODELS ====================
    ModelCategory.EMBEDDING_TEXT: [
        DatasetRequirement(
            data_type=DataType.TEXT,
            required=True,
            min_samples=100_000,
            accepted_formats=["jsonl", "parquet"],
            description="Text pairs or triples for contrastive learning",
        ),
        DatasetRequirement(
            data_type=DataType.LABEL,
            required=False,
            description="Similarity scores or relevance labels",
        ),
    ],
    ModelCategory.EMBEDDING_IMAGE: [
        DatasetRequirement(
            data_type=DataType.IMAGE,
            required=True,
            min_samples=100_000,
            accepted_formats=["parquet", "webdataset"],
            description="Images for contrastive learning",
        ),
    ],
    ModelCategory.EMBEDDING_MULTIMODAL: [
        DatasetRequirement(
            data_type=DataType.IMAGE,
            required=True,
            min_samples=100_000,
            accepted_formats=["parquet"],
            description="Images",
        ),
        DatasetRequirement(
            data_type=DataType.TEXT,
            required=True,
            min_samples=100_000,
            description="Corresponding text captions",
        ),
    ],
    # ==================== TIME SERIES ====================
    ModelCategory.TIME_SERIES_FORECASTING: [
        DatasetRequirement(
            data_type=DataType.TABULAR,
            required=True,
            min_samples=10_000,
            accepted_formats=["csv", "parquet"],
            description="Time-series data points",
        ),
        DatasetRequirement(
            data_type=DataType.LABEL,
            required=True,
            description="Future values to predict",
        ),
    ],
    ModelCategory.TIME_SERIES_CLASSIFICATION: [
        DatasetRequirement(
            data_type=DataType.TABULAR,
            required=True,
            min_samples=1_000,
            accepted_formats=["csv", "parquet"],
            description="Time-series sequences",
        ),
        DatasetRequirement(
            data_type=DataType.LABEL,
            required=True,
            description="Class labels",
        ),
    ],
    # ==================== RECOMMENDATION ====================
    ModelCategory.RECOMMENDATION: [
        DatasetRequirement(
            data_type=DataType.TABULAR,
            required=True,
            min_samples=100_000,
            accepted_formats=["parquet", "csv"],
            description="User-item interactions",
        ),
        DatasetRequirement(
            data_type=DataType.LABEL,
            required=True,
            description="Ratings, clicks, or purchase indicators",
        ),
    ],
}


@dataclass
class DatasetSpec:
    """Complete dataset specification for an IMO project."""

    model_category: ModelCategory
    name: str
    requirements: list[DatasetRequirement]
    additional_fields: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_category(cls, category: ModelCategory, name: str) -> DatasetSpec:
        """Create a dataset spec from a model category."""
        requirements = MODEL_DATASET_SPECS.get(category, [])
        return cls(
            model_category=category,
            name=name,
            requirements=requirements,
        )

    def validate(self, dataset_info: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate a dataset against this specification.

        Args:
            dataset_info: Dictionary containing dataset metadata with keys:
                - data_types: list of DataType present in dataset
                - num_samples: total number of samples
                - size_mb: total size in MB
                - format: file format

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors: list[str] = []

        dataset_types = set(dataset_info.get("data_types", []))
        num_samples = dataset_info.get("num_samples", 0)
        size_mb = dataset_info.get("size_mb", 0)
        dataset_format = dataset_info.get("format", "")

        for req in self.requirements:
            if req.required and req.data_type not in dataset_types:
                errors.append(
                    f"Missing required data type: {req.data_type.value}. {req.description}"
                )

            if req.data_type in dataset_types:
                if num_samples < req.min_samples:
                    errors.append(
                        f"Insufficient samples for {req.data_type.value}: "
                        f"have {num_samples}, need {req.min_samples}"
                    )

                if req.min_size_mb > 0 and size_mb < req.min_size_mb:
                    errors.append(f"Dataset too small: have {size_mb}MB, need {req.min_size_mb}MB")

                if req.accepted_formats and dataset_format not in req.accepted_formats:
                    errors.append(
                        f"Format '{dataset_format}' not accepted. Accepted: {req.accepted_formats}"
                    )

        return len(errors) == 0, errors


def get_model_category_description(category: ModelCategory) -> str:
    """Get human-readable description for a model category."""
    descriptions = {
        # Language Models
        ModelCategory.LLM: "Large Language Models - trained on massive text corpora",
        ModelCategory.LLM_CHAT: "Chat/Instruction-tuned LLMs - for conversational AI",
        ModelCategory.LLM_CODE: "Code Generation LLMs - trained on programming languages",
        # Vision Models
        ModelCategory.VISION_CLASSIFICATION: "Image Classification Models",
        ModelCategory.VISION_DETECTION: "Object Detection Models - with bounding boxes",
        ModelCategory.VISION_SEGMENTATION: "Image Segmentation Models - pixel-level",
        ModelCategory.VISION_GENERATION: "Image Generation Models - e.g., Stable Diffusion",
        # Audio Models
        ModelCategory.AUDIO_SPEECH_RECOGNITION: "Speech-to-Text Models",
        ModelCategory.AUDIO_TEXT_TO_SPEECH: "Text-to-Speech Models",
        ModelCategory.AUDIO_CLASSIFICATION: "Audio Classification - sound events",
        ModelCategory.AUDIO_SOUND_GENERATION: "Audio Generation Models",
        # Video Models
        ModelCategory.VIDEO_CLASSIFICATION: "Video Action Classification",
        ModelCategory.VIDEO_UNDERSTANDING: "Video Understanding - VLMs for video",
        ModelCategory.VIDEO_GENERATION: "Video Generation Models",
        # Multimodal Models
        ModelCategory.MULTIMODAL_VLM: "Vision-Language Models - image + text",
        ModelCategory.MULTIMODAL_AUDIO_VLM: "Audio-Language Models - audio + text",
        ModelCategory.MULTIMODAL_VIDEO_VLM: "Video-Language Models - video + text",
        # Embedding Models
        ModelCategory.EMBEDDING_TEXT: "Text Embedding Models",
        ModelCategory.EMBEDDING_IMAGE: "Image Embedding Models",
        ModelCategory.EMBEDDING_MULTIMODAL: "Multimodal Embedding Models",
        # Time-series
        ModelCategory.TIME_SERIES_FORECASTING: "Time-series Forecasting Models",
        ModelCategory.TIME_SERIES_CLASSIFICATION: "Time-series Classification Models",
        # Recommendation
        ModelCategory.RECOMMENDATION: "Recommendation Systems",
    }
    return descriptions.get(category, category.value)
