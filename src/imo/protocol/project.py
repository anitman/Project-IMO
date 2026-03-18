"""Training project management — the core coordination unit.

A Project is the central concept: someone proposes a model to train,
others contribute datasets and compute, and rewards flow back to all
contributors proportionally.

Flow:
    1. Proposer creates a Project (model spec + paper)
    2. Data contributors submit datasets → project's dataset pool grows
    3. Community votes to approve training
    4. Compute contributors join the training swarm
    5. Training completes → model evaluated → rewards distributed
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ProjectStatus(Enum):
    """Project lifecycle status."""

    DRAFT = "draft"
    OPEN_FOR_DATA = "open_for_data"
    VOTING = "voting"
    APPROVED = "approved"
    TRAINING = "training"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"


class TrainingMode(Enum):
    """Training mode for the project."""

    FROM_SCRATCH = "from_scratch"
    FULL_FINE_TUNE = "full_fine_tune"
    LORA = "lora"
    QLORA = "qlora"


@dataclass
class DatasetContribution:
    """A dataset contributed to a project by a participant."""

    id: str
    contributor_id: str
    project_id: str
    name: str
    num_samples: int
    size_mb: float
    quality_score: float
    ipfs_hash: str
    license: str
    data_types: list[str]
    format: str
    language: str | None = None
    domain: str | None = None
    uniqueness_factor: float = 1.0
    submitted_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    verified: bool = False

    def content_hash(self) -> str:
        """Compute content hash for deduplication."""
        data = f"{self.ipfs_hash}:{self.num_samples}:{self.size_mb}"
        return hashlib.sha256(data.encode()).hexdigest()


@dataclass
class ComputeContribution:
    """A compute node's contribution to a project's training."""

    node_id: str
    project_id: str
    vram_gb: int
    compute_hours: float
    steps_completed: int
    loss_reduction: float
    bandwidth_mbps: float
    verified: bool = False
    reputation: float = 1.0
    joined_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class ProjectSpec:
    """Specification of what the project aims to train."""

    model_architecture: str
    model_category: str
    training_mode: TrainingMode
    base_model: str | None = None  # For fine-tuning: HF model ID
    max_steps: int = 100_000
    target_loss: float | None = None
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    required_data_types: list[str] = field(default_factory=list)
    min_dataset_samples: int = 0
    min_dataset_quality: float = 0.7


@dataclass
class Project:
    """A training project that coordinates data + compute + paper contributors.

    This is the fundamental unit of IMO: a proposal to collectively train
    a model, with rewards flowing to all participants.
    """

    id: str
    title: str
    description: str
    proposer_id: str
    spec: ProjectSpec
    status: ProjectStatus = ProjectStatus.DRAFT

    # Paper / research
    paper_ipfs_hash: str = ""
    paper_authors: list[str] = field(default_factory=list)

    # Contributed datasets (aggregated pool)
    dataset_contributions: list[DatasetContribution] = field(default_factory=list)

    # Compute contributors
    compute_contributions: list[ComputeContribution] = field(default_factory=list)

    # Voting
    votes_for: float = 0.0
    votes_against: float = 0.0
    quorum_required: float = 1000.0
    voting_deadline: str = ""

    # Results
    final_model_ipfs_hash: str = ""
    quality_score: float = 0.0
    quality_multiplier: float = 0.0

    # Timestamps
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    training_started_at: str = ""
    completed_at: str = ""

    metadata: dict[str, Any] = field(default_factory=dict)

    # -- Dataset contribution --

    def contribute_dataset(self, contribution: DatasetContribution) -> None:
        """Add a dataset contribution to this project."""
        if self.status not in (ProjectStatus.DRAFT, ProjectStatus.OPEN_FOR_DATA):
            raise RuntimeError(
                f"Cannot contribute data in status {self.status.value}"
            )

        # Check for duplicates by content hash
        new_hash = contribution.content_hash()
        for existing in self.dataset_contributions:
            if existing.content_hash() == new_hash:
                raise ValueError(
                    f"Duplicate dataset: {contribution.name} matches {existing.name}"
                )

        self.dataset_contributions.append(contribution)

    def open_for_data(self) -> None:
        """Open the project for dataset contributions."""
        if self.status != ProjectStatus.DRAFT:
            raise RuntimeError("Can only open from draft status")
        self.status = ProjectStatus.OPEN_FOR_DATA

    @property
    def total_samples(self) -> int:
        """Total samples across all contributed datasets."""
        return sum(c.num_samples for c in self.dataset_contributions)

    @property
    def total_data_size_mb(self) -> float:
        """Total data size in MB."""
        return sum(c.size_mb for c in self.dataset_contributions)

    @property
    def num_data_contributors(self) -> int:
        """Number of unique data contributors."""
        return len({c.contributor_id for c in self.dataset_contributions})

    @property
    def num_compute_contributors(self) -> int:
        """Number of unique compute contributors."""
        return len({c.node_id for c in self.compute_contributions})

    # -- Compute contribution --

    def join_training(self, contribution: ComputeContribution) -> None:
        """Register a compute node for training."""
        if self.status not in (ProjectStatus.APPROVED, ProjectStatus.TRAINING):
            raise RuntimeError(
                f"Cannot join training in status {self.status.value}"
            )
        # Check if already joined
        existing_ids = {c.node_id for c in self.compute_contributions}
        if contribution.node_id in existing_ids:
            raise ValueError(f"Node {contribution.node_id} already joined")

        self.compute_contributions.append(contribution)

    def update_compute_contribution(
        self,
        node_id: str,
        steps: int,
        loss_reduction: float,
        compute_hours: float,
    ) -> None:
        """Update a compute contributor's metrics."""
        for contrib in self.compute_contributions:
            if contrib.node_id == node_id:
                contrib.steps_completed = steps
                contrib.loss_reduction = loss_reduction
                contrib.compute_hours = compute_hours
                return
        raise ValueError(f"Node {node_id} not found in project")

    # -- Lifecycle transitions --

    def start_voting(self, deadline: str, quorum: float = 1000.0) -> None:
        """Start the community voting phase."""
        if self.status != ProjectStatus.OPEN_FOR_DATA:
            raise RuntimeError("Must be open_for_data to start voting")
        if not self.dataset_contributions:
            raise RuntimeError("Need at least one dataset contribution")
        self.status = ProjectStatus.VOTING
        self.voting_deadline = deadline
        self.quorum_required = quorum

    def vote(self, stake: float, support: bool) -> None:
        """Cast a vote."""
        if self.status != ProjectStatus.VOTING:
            raise RuntimeError("Not in voting phase")
        if support:
            self.votes_for += stake
        else:
            self.votes_against += stake

    def resolve_voting(self) -> bool:
        """Resolve voting: approve if quorum reached."""
        if self.status != ProjectStatus.VOTING:
            return False
        if self.votes_for >= self.quorum_required:
            self.status = ProjectStatus.APPROVED
            return True
        self.status = ProjectStatus.FAILED
        return False

    def start_training(self) -> None:
        """Transition to training phase."""
        if self.status != ProjectStatus.APPROVED:
            raise RuntimeError("Project not approved")
        self.status = ProjectStatus.TRAINING
        self.training_started_at = datetime.now(timezone.utc).isoformat()

    def complete_training(self, model_ipfs_hash: str) -> None:
        """Mark training as complete."""
        if self.status != ProjectStatus.TRAINING:
            raise RuntimeError("Not in training phase")
        self.status = ProjectStatus.EVALUATING
        self.final_model_ipfs_hash = model_ipfs_hash

    def finalize(
        self,
        quality_score: float,
        quality_multiplier: float,
    ) -> None:
        """Finalize the project after evaluation."""
        if self.status != ProjectStatus.EVALUATING:
            raise RuntimeError("Not in evaluating phase")
        self.quality_score = quality_score
        self.quality_multiplier = quality_multiplier
        self.status = ProjectStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc).isoformat()

    # -- Serialization --

    def to_json(self) -> str:
        """Serialize project to JSON."""
        return json.dumps(
            {
                "id": self.id,
                "title": self.title,
                "description": self.description,
                "proposer_id": self.proposer_id,
                "status": self.status.value,
                "spec": {
                    "model_architecture": self.spec.model_architecture,
                    "model_category": self.spec.model_category,
                    "training_mode": self.spec.training_mode.value,
                    "base_model": self.spec.base_model,
                    "max_steps": self.spec.max_steps,
                    "hyperparameters": self.spec.hyperparameters,
                },
                "paper_authors": self.paper_authors,
                "total_samples": self.total_samples,
                "total_data_size_mb": self.total_data_size_mb,
                "num_data_contributors": self.num_data_contributors,
                "num_compute_contributors": self.num_compute_contributors,
                "votes_for": self.votes_for,
                "votes_against": self.votes_against,
                "quality_score": self.quality_score,
                "created_at": self.created_at,
            },
            indent=2,
        )


class ProjectRepository:
    """In-memory project repository."""

    def __init__(self) -> None:
        self.projects: dict[str, Project] = {}

    def create(self, project: Project) -> None:
        """Register a new project."""
        if project.id in self.projects:
            raise ValueError(f"Project {project.id} already exists")
        self.projects[project.id] = project

    def get(self, project_id: str) -> Project | None:
        """Get a project by ID."""
        return self.projects.get(project_id)

    def list_by_status(self, status: ProjectStatus) -> list[Project]:
        """List projects by status."""
        return [p for p in self.projects.values() if p.status == status]

    def list_open(self) -> list[Project]:
        """List projects accepting dataset contributions."""
        return self.list_by_status(ProjectStatus.OPEN_FOR_DATA)

    def list_training(self) -> list[Project]:
        """List projects currently training."""
        return self.list_by_status(ProjectStatus.TRAINING)

    def list_all(self) -> list[Project]:
        """List all projects."""
        return list(self.projects.values())
