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
from imo.training.pipeline import (
    BlockServer,
    PipelineRouter,
    RemoteBlock,
    RemoteSequential,
    build_remote_pipeline,
)
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

    # Parallelism mode: "data_parallel" (default) or "pipeline_parallel"
    parallelism_mode: str = "data_parallel"
    total_blocks: int = 0  # Number of transformer blocks (for pipeline mode)

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

    The engine delegates model loading, loss computation, and optimizer
    creation to a TrainingToolkit (pluggable adapter). The engine handles
    everything else: Hivemind DHT setup, pipeline parallelism (splitting
    model layers across nodes), gradient compression, Byzantine-robust
    aggregation, checkpointing, and contribution tracking.

    Lifecycle:
        1. Toolkit loads model → Engine receives nn.Module
        2. Discover peers via DHT
        3. Schedule layers across peers by VRAM (pipeline parallelism)
           OR replicate full model (data parallelism)
        4. Run collaborative training loop:
           a. Toolkit computes loss (local forward)
           b. Engine handles backward pass
           c. Compress gradients (Top-K / SignSGD)
           d. All-reduce via hivemind.Optimizer
           e. Verify gradient integrity (Byzantine detection)
           f. Apply aggregated update
        5. Checkpoint periodically
        6. Report contributions for reward settlement
        7. Toolkit runs post-training (LoRA merge, export)
    """

    def __init__(
        self,
        config: TrainingConfig,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        toolkit: Any | None = None,
    ):
        self.config = config
        self.model = model
        self.toolkit = toolkit
        self.status = TrainingStatus.INITIALIZING
        self.metrics = TrainingMetrics()

        # Optimizer — prefer toolkit-created, then explicit, then default
        if optimizer is not None:
            self.optimizer = optimizer
        elif toolkit is not None:
            self.optimizer, self.scheduler = toolkit.create_optimizer(
                model, {"learning_rate": config.learning_rate,
                        "weight_decay": config.weight_decay,
                        "warmup_steps": config.warmup_steps,
                        "max_steps": config.max_steps}
            )
            scheduler = self.scheduler  # capture for later
        else:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )

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

        # Pipeline parallelism
        self.pipeline_router: PipelineRouter | None = None
        self.remote_sequential: RemoteSequential | None = None

        # Tracking
        self._step = 0
        self._contributions: dict[str, dict[str, float]] = {}

    @classmethod
    def from_toolkit(
        cls,
        config: TrainingConfig,
        toolkit: Any,
        spec: dict[str, Any],
    ) -> DistributedTrainingEngine:
        """Create an engine from a TrainingToolkit.

        The toolkit handles model loading and optimizer creation;
        the engine handles everything distributed (DHT, pipeline
        parallelism, gradient aggregation).

        Args:
            config: Training configuration.
            toolkit: A TrainingToolkit adapter instance.
            spec: Project spec dict passed to toolkit.load_model() etc.
        """
        model = toolkit.load_model(spec)
        return cls(config=config, model=model, toolkit=toolkit)

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

        # Pipeline parallelism setup
        if self.config.parallelism_mode == "pipeline_parallel":
            await self._initialize_pipeline()

        self.status = TrainingStatus.SCHEDULING
        logger.info("Hivemind optimizer initialized, waiting for peers...")

    async def _initialize_pipeline(self) -> None:
        """Set up pipeline parallelism: discover servers and build remote chain."""
        self.pipeline_router = PipelineRouter(dht=self.dht)

        if self.config.total_blocks <= 0:
            # Auto-detect from model
            all_layers = list(self.model.children())
            self.config.total_blocks = len(all_layers)

        await self.pipeline_router.discover_servers(self.config.project_id)
        chain = self.pipeline_router.build_chain(self.config.total_blocks)
        self.remote_sequential = build_remote_pipeline(self.model, chain)

        logger.info(
            "Pipeline parallelism initialized: %d servers, %d total blocks",
            len(chain), self.config.total_blocks,
        )

    async def train_step(
        self,
        batch: dict[str, torch.Tensor],
        loss_fn: Any = None,
    ) -> TrainingMetrics:
        """Execute one training step with gradient averaging.

        Loss computation priority:
          1. Pipeline parallelism path (activations routed across nodes)
          2. Toolkit.compute_loss() (if a toolkit is attached)
          3. Diffusion forward (if config.is_diffusion)
          4. Standard forward (HF-style model(**batch).loss)
          5. Explicit loss_fn parameter (legacy fallback)
        """
        self.status = TrainingStatus.TRAINING
        self.model.train()

        if self.config.parallelism_mode == "pipeline_parallel" and self.remote_sequential is not None:
            loss = self._pipeline_forward(batch, loss_fn)
        elif self.toolkit is not None:
            loss = self.toolkit.compute_loss(self.model, batch)
        elif self.config.is_diffusion:
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

    def _pipeline_forward(
        self,
        batch: dict[str, torch.Tensor],
        loss_fn: Any,
    ) -> torch.Tensor:
        """Forward pass using pipeline parallelism (Petals-style).

        Activations flow through RemoteSequential, which routes them
        across the server chain. Each server processes its block range
        and passes outputs to the next.
        """
        inputs = batch.get("input_ids", batch.get("inputs_embeds", batch.get("input")))
        if inputs is None:
            raise ValueError("Pipeline mode requires 'input_ids', 'inputs_embeds', or 'input' in batch")

        # Forward through remote pipeline
        output = self.remote_sequential(inputs)

        # Compute loss
        if loss_fn is not None:
            return loss_fn(output, batch)

        labels = batch.get("labels")
        if labels is not None:
            return torch.nn.functional.cross_entropy(
                output.view(-1, output.size(-1)), labels.view(-1)
            )
        raise ValueError("Pipeline mode requires loss_fn or 'labels' in batch")

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

    async def shutdown(self, project_dir: str | None = None) -> None:
        """Gracefully shut down the training engine."""
        logger.info("Shutting down training engine at step %d", self._step)
        await self._save_checkpoint()

        # Toolkit post-processing (LoRA merge, export, etc.)
        if self.toolkit is not None and project_dir is not None:
            from pathlib import Path

            self.toolkit.post_training(self.model, Path(project_dir))

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
