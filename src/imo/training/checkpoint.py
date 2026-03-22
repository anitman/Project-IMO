"""Distributed checkpointing for training resilience."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


@dataclass
class CheckpointMetadata:
    """Metadata for a training checkpoint."""

    step: int
    timestamp: str
    model_state_hash: str
    optimizer_state_hash: str
    training_config: dict[str, Any]
    node_assignments: dict[str, list[int]]


class CheckpointManager:
    """Manage distributed model checkpoints."""

    def __init__(self, checkpoint_dir: str | Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        step: int,
        model_state: dict[str, torch.Tensor],
        optimizer_state: dict[str, Any],
        training_config: dict[str, Any],
        node_assignments: dict[str, list[int]],
    ) -> Path:
        """Save a training checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{step}.pt"
        metadata = CheckpointMetadata(
            step=step,
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_state_hash=self._compute_hash(model_state),
            optimizer_state_hash=self._compute_hash(optimizer_state),
            training_config=training_config,
            node_assignments=node_assignments,
        )

        metadata_path = self.checkpoint_dir / f"checkpoint_{step}_meta.json"
        with open(metadata_path, "w") as f:
            json.dump(
                {
                    "step": metadata.step,
                    "timestamp": metadata.timestamp,
                    "model_state_hash": metadata.model_state_hash,
                    "optimizer_state_hash": metadata.optimizer_state_hash,
                    "training_config": metadata.training_config,
                    "node_assignments": metadata.node_assignments,
                },
                f,
                indent=2,
            )

        torch.save(
            {
                "model_state": model_state,
                "optimizer_state": optimizer_state,
            },
            checkpoint_path,
        )

        return checkpoint_path

    def load(self, step: int) -> dict[str, Any]:
        """Load a training checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{step}.pt"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, weights_only=True)
        return {
            "step": step,
            "model_state": checkpoint["model_state"],
            "optimizer_state": checkpoint["optimizer_state"],
        }

    def load_latest(self) -> dict[str, Any]:
        """Load the latest checkpoint."""
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.pt"))

        if not checkpoint_files:
            raise FileNotFoundError("No checkpoints found")

        latest = max(
            checkpoint_files,
            key=lambda p: int(p.stem.split("_")[1]),
        )
        step = int(latest.stem.split("_")[1])
        return self.load(step)

    def list_checkpoints(self) -> list[int]:
        """List all available checkpoint steps."""
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
        return [int(f.stem.split("_")[1]) for f in checkpoint_files]

    def delete(self, step: int) -> None:
        """Delete a checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{step}.pt"
        metadata_path = self.checkpoint_dir / f"checkpoint_{step}_meta.json"

        if checkpoint_path.exists():
            checkpoint_path.unlink()
        if metadata_path.exists():
            metadata_path.unlink()

    def cleanup_old(self, keep_last: int = 5) -> list[int]:
        """Clean up old checkpoints, keeping only the most recent."""
        steps = self.list_checkpoints()
        steps.sort(reverse=True)

        deleted = []
        for step in steps[keep_last:]:
            self.delete(step)
            deleted.append(step)

        return deleted

    def _compute_hash(self, state: dict[str, Any]) -> str:
        """Compute hash of state dictionary."""
        import hashlib

        content = json.dumps(state, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()
