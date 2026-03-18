"""IMO lifecycle management."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from imo.data.dataset_spec import ModelCategory


class IMOStatus(Enum):
    """IMO lifecycle status."""

    SUBMITTED = "submitted"
    VOTING = "voting"
    APPROVED = "approved"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"


class TrainingMode(Enum):
    """Training mode for IMO."""

    NEW_ARCHITECTURE = "new_architecture"
    FINE_TUNING = "fine_tuning"


@dataclass
class TrainingSpec:
    """Specification for training a model."""

    model_architecture: str
    model_category: ModelCategory
    dataset_ids: list[str]
    hyperparameters: dict[str, Any]
    max_steps: int
    training_mode: TrainingMode = TrainingMode.NEW_ARCHITECTURE

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(
            {
                "model_architecture": self.model_architecture,
                "model_category": self.model_category.value,
                "dataset_ids": self.dataset_ids,
                "hyperparameters": self.hyperparameters,
                "max_steps": self.max_steps,
                "training_mode": self.training_mode.value,
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, json_str: str) -> TrainingSpec:
        """Load from JSON."""
        data = json.loads(json_str)
        return cls(
            model_architecture=data["model_architecture"],
            model_category=ModelCategory(data["model_category"]),
            dataset_ids=data["dataset_ids"],
            hyperparameters=data["hyperparameters"],
            max_steps=data["max_steps"],
            training_mode=TrainingMode(data["training_mode"]),
        )


@dataclass
class Paper:
    """Research paper proposing a model to train."""

    id: str
    title: str
    authors: list[str]
    abstract: str
    ipfs_hash: str
    training_spec: TrainingSpec
    model_category: ModelCategory
    submitted_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class VotingRecord:
    """Voting record for an IMO."""

    voter_id: str
    vote: bool
    stake: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class IMO:
    """Initial Model Offering record."""

    id: str
    paper: Paper
    status: IMOStatus
    votes: list[VotingRecord] = field(default_factory=list)
    total_stake: float = 0.0
    quorum_required: float = 1000.0
    voting_deadline: str = ""
    training_start: str = ""
    training_end: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_vote(self, voter_id: str, vote: bool, stake: float) -> None:
        """Add a vote to the IMO."""
        self.votes.append(
            VotingRecord(
                voter_id=voter_id,
                vote=vote,
                stake=stake,
            )
        )
        if vote:
            self.total_stake += stake

    def voting_complete(self) -> bool:
        """Check if voting phase is complete."""
        if not self.voting_deadline:
            return False
        now = datetime.now(timezone.utc).isoformat()
        return now > self.voting_deadline

    def quorum_reached(self) -> bool:
        """Check if quorum has been reached."""
        return self.total_stake >= self.quorum_required

    def approve(self) -> None:
        """Approve the IMO and transition to training."""
        if not self.quorum_reached():
            raise RuntimeError("Quorum not reached")
        self.status = IMOStatus.APPROVED

    def start_training(self) -> None:
        """Start training phase."""
        if self.status != IMOStatus.APPROVED:
            raise RuntimeError("IMO not approved")
        self.status = IMOStatus.TRAINING
        self.training_start = datetime.now(timezone.utc).isoformat()

    def complete_training(self) -> None:
        """Complete training phase."""
        if self.status != IMOStatus.TRAINING:
            raise RuntimeError("Training not in progress")
        self.status = IMOStatus.COMPLETED
        self.training_end = datetime.now(timezone.utc).isoformat()


class IMORepository:
    """Repository for IMO records."""

    def __init__(self):
        self.imos: dict[str, IMO] = {}

    def add(self, imo: IMO) -> None:
        """Add an IMO to the repository."""
        self.imos[imo.id] = imo

    def get(self, imo_id: str) -> IMO | None:
        """Get an IMO by ID."""
        return self.imos.get(imo_id)

    def list_by_status(self, status: IMOStatus) -> list[IMO]:
        """List IMOs by status."""
        return [imo for imo in self.imos.values() if imo.status == status]

    def list_all(self) -> list[IMO]:
        """List all IMOs."""
        return list(self.imos.values())
