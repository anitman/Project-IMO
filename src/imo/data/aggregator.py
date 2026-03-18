"""Dataset aggregation and linking for multi-source training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from datasets import Dataset


@dataclass
class DatasetManifest:
    """Manifest describing how datasets should be aggregated."""

    id: str
    name: str
    datasets: list[dict[str, Any]]
    weights: dict[str, float]
    filtering_rules: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_json(self) -> str:
        """Serialize manifest to JSON."""
        return json.dumps(
            {
                "id": self.id,
                "name": self.name,
                "datasets": self.datasets,
                "weights": self.weights,
                "filtering_rules": self.filtering_rules,
                "created_at": self.created_at,
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, json_str: str) -> DatasetManifest:
        """Load manifest from JSON."""
        data = json.loads(json_str)
        return cls(
            id=data["id"],
            name=data["name"],
            datasets=data["datasets"],
            weights=data["weights"],
            filtering_rules=data.get("filtering_rules", []),
            created_at=data.get("created_at", ""),
        )

    def compute_hash(self) -> str:
        """Compute Merkle hash of the manifest."""
        content = json.dumps(self.datasets + self.filtering_rules, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()


@dataclass
class DatasetEntry:
    """A single dataset entry in the registry."""

    id: str
    source: str
    license: str
    num_samples: int
    language: str | None = None
    domain: str | None = None
    quality_score: float = 0.0
    hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class DatasetAggregator:
    """Aggregate multiple datasets according to a manifest."""

    def __init__(self):
        self.registry: dict[str, DatasetEntry] = {}

    def register(self, entry: DatasetEntry) -> None:
        """Register a dataset in the aggregation registry."""
        self.registry[entry.id] = entry

    def load_manifest(self, manifest: DatasetManifest) -> list[DatasetEntry]:
        """Load datasets from a manifest."""
        entries: list[DatasetEntry] = []

        for dataset_info in manifest.datasets:
            dataset_id = dataset_info.get("id", "")
            if dataset_id in self.registry:
                entries.append(self.registry[dataset_id])

        return entries

    def aggregate(
        self,
        datasets: list[Dataset],
        weights: list[float] | None = None,
        sampling_strategy: str = "proportional",
    ) -> Dataset:
        """Aggregate multiple datasets into one."""
        if not datasets:
            raise ValueError("No datasets to aggregate")

        if len(datasets) == 1:
            return datasets[0]

        if weights is None:
            weights = [1.0 / len(datasets)] * len(datasets)

        if len(weights) != len(datasets):
            raise ValueError("Weights must match number of datasets")

        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        if sampling_strategy == "proportional":
            return self._aggregate_proportional(datasets, normalized_weights)
        elif sampling_strategy == "balanced":
            return self._aggregate_balanced(datasets)
        else:
            raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")

    def _aggregate_proportional(self, datasets: list[Dataset], weights: list[float]) -> Dataset:
        """Aggregate with proportional sampling based on weights."""
        import numpy as np

        total_samples = sum(len(ds) for ds in datasets)

        sample_weights = []
        for i, ds in enumerate(datasets):
            ds_weight = weights[i] * len(ds) / total_samples
            sample_weights.extend([ds_weight] * len(ds))

        total_weight = sum(sample_weights)
        probabilities = [w / total_weight for w in sample_weights]

        all_samples = []
        for ds in datasets:
            all_samples.extend(ds)

        indices = np.random.choice(len(all_samples), size=len(all_samples), p=probabilities)
        sampled = [all_samples[i] for i in indices]

        return Dataset.from_list(sampled)

    def _aggregate_balanced(self, datasets: list[Dataset]) -> Dataset:
        """Aggregate with balanced sampling (equal representation)."""
        min_size = min(len(ds) for ds in datasets)

        balanced_samples = []
        for i in range(min_size):
            for ds in datasets:
                balanced_samples.append(ds[i])

        return Dataset.from_list(balanced_samples)

    def apply_filtering_rules(self, dataset: Dataset, rules: list[str]) -> Dataset:
        """Apply filtering rules to a dataset."""
        filtered_samples = []

        for sample in dataset:
            if self._passes_all_rules(sample, rules):
                filtered_samples.append(sample)

        return Dataset.from_list(filtered_samples)

    def _passes_all_rules(self, sample: dict[str, Any], rules: list[str]) -> bool:
        """Check if a sample passes all filtering rules."""
        for rule in rules:
            if not self._apply_rule(sample, rule):
                return False
        return True

    def _apply_rule(self, sample: dict[str, Any], rule: str) -> bool:
        """Apply a single filtering rule to a sample."""
        if rule.startswith("min_length:"):
            min_len = int(rule.split(":")[1])
            text = sample.get("text", sample.get("content", ""))
            return len(text) >= min_len

        elif rule.startswith("max_length:"):
            max_len = int(rule.split(":")[1])
            text = sample.get("text", sample.get("content", ""))
            return len(text) <= max_len

        elif rule.startswith("language:"):
            lang = rule.split(":")[1]
            return sample.get("language") == lang

        elif rule.startswith("domain:"):
            domain = rule.split(":")[1]
            return sample.get("domain") == domain

        elif rule == "remove_duplicates":
            return True

        return True

    def compute_contribution_shares(
        self,
        dataset: Dataset,
        source_datasets: list[DatasetEntry],
    ) -> dict[str, float]:
        """Compute contribution shares for each source dataset."""
        source_counts: dict[str, int] = {entry.id: 0 for entry in source_datasets}

        for sample in dataset:
            source_id = sample.get("_source_id", "")
            if source_id in source_counts:
                source_counts[source_id] += 1

        total = sum(source_counts.values())
        if total == 0:
            return {k: 0.0 for k in source_counts}

        return {k: v / total for k, v in source_counts.items()}
