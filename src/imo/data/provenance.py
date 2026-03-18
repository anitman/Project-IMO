"""Data provenance tracking and lineage management."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class TransformationType(Enum):
    """Types of data transformations."""

    FILTER = "filter"
    MAP = "map"
    SAMPLE = "sample"
    CONCAT = "concat"
    CLEAN = "clean"
    TOKENIZE = "tokenize"


@dataclass
class Transformation:
    """A data transformation in the lineage."""

    type: TransformationType
    name: str
    parameters: dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    output_hash: str = ""


@dataclass
class ProvenanceRecord:
    """Provenance record for a dataset."""

    id: str
    source: str
    timestamp: str
    license: str
    transformations: list[Transformation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""

    def add_transformation(self, t: Transformation) -> None:
        """Add a transformation to the lineage."""
        self.transformations.append(t)

    def compute_hash(self) -> str:
        """Compute content hash for integrity verification."""
        content = json.dumps(
            {
                "source": self.source,
                "license": self.license,
                "transformations": [
                    {
                        "type": t.type.value,
                        "name": t.name,
                        "parameters": t.parameters,
                    }
                    for t in self.transformations
                ],
            },
            sort_keys=True,
        )
        self.content_hash = hashlib.sha256(content.encode()).hexdigest()
        return self.content_hash

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(
            {
                "id": self.id,
                "source": self.source,
                "timestamp": self.timestamp,
                "license": self.license,
                "transformations": [
                    {
                        "type": t.type.value,
                        "name": t.name,
                        "parameters": t.parameters,
                        "timestamp": t.timestamp,
                        "output_hash": t.output_hash,
                    }
                    for t in self.transformations
                ],
                "metadata": self.metadata,
                "content_hash": self.content_hash,
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, json_str: str) -> ProvenanceRecord:
        """Load from JSON."""
        data = json.loads(json_str)
        transformations = [
            Transformation(
                type=TransformationType(t["type"]),
                name=t["name"],
                parameters=t["parameters"],
                timestamp=t.get("timestamp", ""),
                output_hash=t.get("output_hash", ""),
            )
            for t in data.get("transformations", [])
        ]
        return cls(
            id=data["id"],
            source=data["source"],
            timestamp=data["timestamp"],
            license=data["license"],
            transformations=transformations,
            metadata=data.get("metadata", {}),
            content_hash=data.get("content_hash", ""),
        )


class DataProvenance:
    """Track data provenance and lineage."""

    def __init__(self):
        self.records: dict[str, ProvenanceRecord] = {}

    def register(
        self,
        dataset_id: str,
        source: str,
        license: str,
        metadata: dict[str, Any] | None = None,
    ) -> ProvenanceRecord:
        """Register a new dataset."""
        record = ProvenanceRecord(
            id=dataset_id,
            source=source,
            timestamp=datetime.now(timezone.utc).isoformat(),
            license=license,
            metadata=metadata or {},
        )
        self.records[dataset_id] = record
        return record

    def get_record(self, dataset_id: str) -> ProvenanceRecord | None:
        """Get provenance record for a dataset."""
        return self.records.get(dataset_id)

    def add_transformation(
        self,
        dataset_id: str,
        t_type: TransformationType,
        name: str,
        parameters: dict[str, Any],
        output_hash: str = "",
    ) -> Transformation:
        """Add a transformation to a dataset's lineage."""
        if dataset_id not in self.records:
            raise ValueError(f"Dataset not found: {dataset_id}")

        transformation = Transformation(
            type=t_type,
            name=name,
            parameters=parameters,
            output_hash=output_hash,
        )
        self.records[dataset_id].add_transformation(transformation)
        return transformation

    def get_lineage(self, dataset_id: str) -> list[Transformation]:
        """Get the full transformation lineage for a dataset."""
        record = self.get_record(dataset_id)
        if record is None:
            raise ValueError(f"Dataset not found: {dataset_id}")
        return record.transformations

    def verify_integrity(self, dataset_id: str, content_hash: str) -> bool:
        """Verify dataset integrity against stored hash."""
        record = self.get_record(dataset_id)
        if record is None:
            return False
        return record.content_hash == content_hash

    def save_to_file(self, output_path: str) -> None:
        """Save all records to a JSON file."""
        data = {dataset_id: record.to_json() for dataset_id, record in self.records.items()}
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_file(cls, input_path: str) -> DataProvenance:
        """Load records from a JSON file."""
        with open(input_path) as f:
            data = json.load(f)

        provenance = cls()
        for dataset_id, json_str in data.items():
            record = ProvenanceRecord.from_json(json_str)
            provenance.records[dataset_id] = record
        return provenance
