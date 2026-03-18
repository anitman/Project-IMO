"""Registry for papers, models, and datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from imo.protocol.imo import Paper, TrainingMode


@dataclass
class ModelRegistryEntry:
    """Entry for a trained model in the registry."""

    id: str
    imo_id: str
    architecture: str
    checkpoint_path: str
    training_config: dict[str, Any]
    performance_metrics: dict[str, float]
    created_at: str


@dataclass
class DatasetRegistryEntry:
    """Entry for a dataset in the registry."""

    id: str
    name: str
    source: str
    license: str
    num_samples: int
    quality_score: float
    ipfs_hash: str


class Registry:
    """Unified registry for papers, models, and datasets."""

    def __init__(self):
        self.papers: dict[str, Paper] = {}
        self.models: dict[str, ModelRegistryEntry] = {}
        self.datasets: dict[str, DatasetRegistryEntry] = {}

    def register_paper(self, paper: Paper) -> None:
        """Register a research paper."""
        self.papers[paper.id] = paper

    def get_paper(self, paper_id: str) -> Paper | None:
        """Get a paper by ID."""
        return self.papers.get(paper_id)

    def list_papers(self) -> list[Paper]:
        """List all registered papers."""
        return list(self.papers.values())

    def register_model(self, entry: ModelRegistryEntry) -> None:
        """Register a trained model."""
        self.models[entry.id] = entry

    def get_model(self, model_id: str) -> ModelRegistryEntry | None:
        """Get a model by ID."""
        return self.models.get(model_id)

    def list_models(self) -> list[ModelRegistryEntry]:
        """List all registered models."""
        return list(self.models.values())

    def register_dataset(self, entry: DatasetRegistryEntry) -> None:
        """Register a dataset."""
        self.datasets[entry.id] = entry

    def get_dataset(self, dataset_id: str) -> DatasetRegistryEntry | None:
        """Get a dataset by ID."""
        return self.datasets.get(dataset_id)

    def list_datasets(self) -> list[DatasetRegistryEntry]:
        """List all registered datasets."""
        return list(self.datasets.values())

    def find_models_by_architecture(
        self,
        architecture: str,
    ) -> list[ModelRegistryEntry]:
        """Find models by architecture type."""
        return [m for m in self.models.values() if m.architecture == architecture]

    def find_papers_for_training(
        self,
        training_mode: TrainingMode,
    ) -> list[Paper]:
        """Find papers suitable for a training mode."""
        return [p for p in self.papers.values() if p.training_spec.training_mode == training_mode]
