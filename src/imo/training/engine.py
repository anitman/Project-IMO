"""Core distributed training engine built on Hivemind.

Orchestrates the full lifecycle: peer discovery → data loading →
gradient-compressed collaborative training → checkpointing → reward settlement.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import hivemind
import torch
import torch.nn as nn

from imo.node.communicator import GradientCommunicator, TopKCompression
from imo.node.discovery import PeerDiscovery, PeerInfo
from imo.node.scheduler import VRAMScheduler
from imo.training.aggregator import AggregationStrategy, FederatedAveraging, TrimmedMean
from imo.training.checkpoint import CheckpointManager
from imo.training.security import PoisoningDetector
from imo.training.verifier import GradientVerifier

logger = logging.getLogger(__name__)


class TrainingStatus(Enum):
    """Status of a training run."""

    INITIALIZING = "initializing"
    DISCOVERING_PEERS = "discovering_peers"
    SCHEDULING = "scheduling"
    TRAINING = "training"
    CHECKPOINTING = "checkpointing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TrainingConfig:
    """Configuration for a distributed training run."""

    project_id: str
    model_architecture: str
    model_category: str

    # Training hyperparameters
    max_steps: int = 100_000
    batch_size: int = 32
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1

    # Distributed settings
    target_batch_size: int = 4096
    compression: str = "top_k"
    top_k_sparsity: float = 0.01
    comm_dtype: str = "float16"
    min_peers: int = 2
    matchmaking_time: float = 60.0

    # Hivemind settings
    hivemind_target_group_size: int = 16
    hivemind_averaging_expiration: float = 15.0
    hivemind_matchmaking_prefix: str = "imo_training"

    # Byzantine tolerance
    aggregation_strategy: str = "trimmed_mean"
    trim_ratio: float = 0.1
    poisoning_threshold: float = 2.0

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    checkpoint_interval: int = 1000
    keep_checkpoints: int = 5

    # Diffusion-specific (for image/video/audio generation)
    is_diffusion: bool = False
    num_diffusion_steps: int = 1000
    noise_schedule: str = "cosine"
    prediction_type: str = "epsilon"


@dataclass
class TrainingMetrics:
    """Metrics collected during training."""

    step: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    gradient_norm: float = 0.0
    samples_per_second: float = 0.0
    peer_count: int = 0
    aggregation_time_ms: float = 0.0
    compression_ratio: float = 0.0
    contributions: dict[str, float] = field(default_factory=dict)


class DistributedTrainingEngine:
    """Orchestrates distributed training across heterogeneous peers.

    Uses hivemind for decentralized gradient averaging, with Byzantine-robust
    aggregation and gradient compression for bandwidth efficiency.

    Lifecycle:
        1. Initialize model + optimizer
        2. Discover peers via DHT
        3. Schedule layers across peers by VRAM
        4. Run collaborative training loop:
           a. Local forward/backward pass
           b. Compress gradients (Top-K / SignSGD)
           c. All-reduce via hivemind.Optimizer
           d. Verify gradient integrity
           e. Apply aggregated update
        5. Checkpoint periodically
        6. Report contributions for reward settlement
    """

    def __init__(
        self,
        config: TrainingConfig,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ):
        self.config = config
        self.model = model
        self.status = TrainingStatus.INITIALIZING
        self.metrics = TrainingMetrics()

        # Optimizer
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        else:
            self.optimizer = optimizer

        self.scheduler = scheduler

        # Distributed components
        self.dht: hivemind.DHT | None = None
        self.hivemind_optimizer: hivemind.Optimizer | None = None
        self.discovery: PeerDiscovery | None = None
        self.communicator = GradientCommunicator(
            compression_method=TopKCompression(sparsity=config.top_k_sparsity),
        )
        self.checkpoint_manager = CheckpointManager(config.checkpoint_dir)

        # Security
        self.aggregation: AggregationStrategy = self._build_aggregation_strategy()
        self.poisoning_detector = PoisoningDetector(
            anomaly_threshold=config.poisoning_threshold,
        )
        self.verifier = GradientVerifier()

        # Tracking
        self._step = 0
        self._contributions: dict[str, dict[str, float]] = {}

    def _build_aggregation_strategy(self) -> AggregationStrategy:
        """Build aggregation strategy from config."""
        if self.config.aggregation_strategy == "trimmed_mean":
            return TrimmedMean(trim_ratio=self.config.trim_ratio)
        return FederatedAveraging()

    async def initialize(
        self,
        initial_peers: list[str] | None = None,
        node_info: PeerInfo | None = None,
    ) -> None:
        """Initialize DHT, discover peers, and set up hivemind optimizer."""
        self.status = TrainingStatus.DISCOVERING_PEERS
        logger.info("Initializing distributed training for project %s", self.config.project_id)

        # Start DHT
        self.dht = await hivemind.DHT.create(
            initial_peers=initial_peers or [],
            start=True,
        )

        # Set up hivemind collaborative optimizer
        run_id = f"{self.config.hivemind_matchmaking_prefix}_{self.config.project_id}"
        self.hivemind_optimizer = hivemind.Optimizer(
            dht=self.dht,
            run_id=run_id,
            params=self.optimizer.param_groups,
            optimizer=self.optimizer,
            target_batch_size=self.config.target_batch_size,
            batch_size_per_step=self.config.batch_size,
            matchmaking_time=self.config.matchmaking_time,
            averaging_timeout=self.config.hivemind_averaging_expiration,
            grad_compression=hivemind.ScaledFloat16Compression(),
            state_averaging_compression=hivemind.ScaledFloat16Compression(),
            verbose=True,
        )

        self.status = TrainingStatus.SCHEDULING
        logger.info("Hivemind optimizer initialized, waiting for peers...")

    async def train_step(
        self,
        batch: dict[str, torch.Tensor],
        loss_fn: Any = None,
    ) -> TrainingMetrics:
        """Execute one training step with gradient averaging.

        For standard models: forward → loss → backward → hivemind step.
        For diffusion models: sample noise → predict → loss → backward → hivemind step.
        """
        self.status = TrainingStatus.TRAINING
        self.model.train()

        if self.config.is_diffusion:
            loss = self._diffusion_forward(batch, loss_fn)
        else:
            loss = self._standard_forward(batch, loss_fn)

        loss.backward()
        self._step += 1

        # Gradient norm for monitoring
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Hivemind handles all-reduce and optimizer step
        if self.hivemind_optimizer is not None:
            self.hivemind_optimizer.step()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad()

        if self.scheduler is not None:
            self.scheduler.step()

        # Update metrics
        self.metrics.step = self._step
        self.metrics.loss = loss.item()
        self.metrics.gradient_norm = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        self.metrics.learning_rate = self.optimizer.param_groups[0]["lr"]

        # Checkpoint
        if self._step % self.config.checkpoint_interval == 0:
            await self._save_checkpoint()

        return self.metrics

    def _standard_forward(
        self,
        batch: dict[str, torch.Tensor],
        loss_fn: Any,
    ) -> torch.Tensor:
        """Standard forward pass for LLM / classification / embedding models."""
        if loss_fn is not None:
            outputs = self.model(**batch)
            return loss_fn(outputs, batch)

        # HuggingFace-style: model returns loss when labels provided
        outputs = self.model(**batch)
        if hasattr(outputs, "loss"):
            return outputs.loss
        raise ValueError("Model must return loss or provide loss_fn")

    def _diffusion_forward(
        self,
        batch: dict[str, torch.Tensor],
        loss_fn: Any,
    ) -> torch.Tensor:
        """Forward pass for diffusion models (image/video/audio generation).

        Implements the standard denoising objective:
        1. Sample random timestep t ~ Uniform(0, T)
        2. Sample noise ε ~ N(0, I)
        3. Create noisy input: x_t = √(ᾱ_t) * x_0 + √(1 - ᾱ_t) * ε
        4. Predict noise: ε̂ = model(x_t, t)
        5. Loss = MSE(ε, ε̂)
        """
        x_0 = batch["pixel_values"] if "pixel_values" in batch else batch["input"]
        bsz = x_0.shape[0]
        device = x_0.device

        # Sample timesteps
        timesteps = torch.randint(0, self.config.num_diffusion_steps, (bsz,), device=device)

        # Sample noise
        noise = torch.randn_like(x_0)

        # Compute noisy input using cosine schedule
        alpha_bar = self._cosine_alpha_bar(timesteps, device)
        alpha_bar = alpha_bar.view(-1, *([1] * (x_0.dim() - 1)))

        noisy = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise

        # Condition (text prompt embedding, etc.)
        condition = batch.get("encoder_hidden_states", batch.get("condition"))

        # Model prediction
        if condition is not None:
            pred = self.model(noisy, timesteps, condition)
        else:
            pred = self.model(noisy, timesteps)

        if hasattr(pred, "sample"):
            pred = pred.sample

        # Loss
        if self.config.prediction_type == "epsilon":
            target = noise
        elif self.config.prediction_type == "v_prediction":
            target = torch.sqrt(alpha_bar) * noise - torch.sqrt(1 - alpha_bar) * x_0
        else:
            target = x_0

        return torch.nn.functional.mse_loss(pred, target)

    def _cosine_alpha_bar(self, timesteps: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Cosine noise schedule: ᾱ(t) = cos²(π/2 · (t/T + s) / (1 + s))."""
        s = 0.008
        t = timesteps.float() / self.config.num_diffusion_steps
        return torch.cos(((t + s) / (1 + s)) * (3.14159265 / 2)) ** 2

    async def _save_checkpoint(self) -> None:
        """Save model checkpoint with metadata."""
        self.status = TrainingStatus.CHECKPOINTING

        self.checkpoint_manager.save(
            step=self._step,
            model_state=self.model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            training_config={
                "project_id": self.config.project_id,
                "model_architecture": self.config.model_architecture,
                "step": self._step,
                "loss": self.metrics.loss,
            },
            node_assignments={},
        )
        self.checkpoint_manager.cleanup_old(keep_last=self.config.keep_checkpoints)

        self.status = TrainingStatus.TRAINING
        logger.info("Checkpoint saved at step %d", self._step)

    def record_contribution(
        self,
        node_id: str,
        compute_time: float,
        loss_reduction: float,
        vram_gb: int,
    ) -> None:
        """Record a node's training contribution for reward settlement."""
        self._contributions[node_id] = {
            "compute_time": compute_time,
            "loss_reduction": loss_reduction,
            "vram_gb": vram_gb,
            "steps": self._step,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_contributions(self) -> dict[str, dict[str, float]]:
        """Get all recorded contributions."""
        return dict(self._contributions)

    async def shutdown(self) -> None:
        """Gracefully shut down the training engine."""
        logger.info("Shutting down training engine at step %d", self._step)
        await self._save_checkpoint()

        if self.hivemind_optimizer is not None:
            self.hivemind_optimizer.shutdown()
        if self.dht is not None:
            self.dht.shutdown()

        self.status = TrainingStatus.COMPLETED

    @property
    def step(self) -> int:
        return self._step

    @property
    def is_training(self) -> bool:
        return self.status == TrainingStatus.TRAINING
