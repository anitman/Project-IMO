"""Base class for training toolkits."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from imo.data.dataset_spec import ModelCategory


class ToolkitCapability:
    """Constants for training mode capabilities."""

    FROM_SCRATCH = "from_scratch"
    FINE_TUNE = "full_fine_tune"
    LORA = "lora"
    QLORA = "qlora"
    CONTINUAL_PRETRAIN = "continual_pretrain"
    DISTILLATION = "distillation"
    HYBRID = "hybrid"


@dataclass
class ToolkitInfo:
    """Metadata about a training toolkit."""

    name: str
    display_name: str
    description: str
    url: str
    pip_package: str
    supported_categories: list[ModelCategory]
    supported_modes: list[str]
    min_vram_gb: int = 4
    extra_dependencies: list[str] = field(default_factory=list)


class TrainingToolkit(ABC):
    """Abstract base for training toolkit adapters.

    Each toolkit wraps a specific training backend (HF Trainer, Unsloth, etc.)
    and provides a uniform interface for:
      - Project scaffolding (config files, scripts)
      - Model loading and preparation
      - Dataset preprocessing
      - Loss computation
      - Optimizer / scheduler creation
      - Post-training processing (LoRA merge, export)

    The DistributedTrainingEngine calls toolkit methods to obtain the model
    and loss function, then handles Hivemind DHT orchestration, pipeline
    parallelism (layer splitting across nodes), gradient compression, and
    Byzantine-robust aggregation around the toolkit-provided components.
    """

    # ── Metadata ──────────────────────────────────────────────

    @abstractmethod
    def info(self) -> ToolkitInfo:
        """Return toolkit metadata."""

    @abstractmethod
    def validate_environment(self) -> list[str]:
        """Check if the environment meets requirements.

        Returns a list of missing dependencies / issues. Empty = ready.
        """

    # ── Project scaffolding ───────────────────────────────────

    @abstractmethod
    def setup_project(self, project_dir: Path, spec: dict[str, Any]) -> None:
        """Scaffold training files in the project directory.

        Creates config files, scripts, and any toolkit-specific structure.
        """

    @abstractmethod
    def prepare_config(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Generate a training config dict from a project spec."""

    @abstractmethod
    def get_train_command(self, project_dir: Path) -> list[str]:
        """Return the shell command to launch training."""

    # ── Model lifecycle (used by DistributedTrainingEngine) ───

    @abstractmethod
    def load_model(self, spec: dict[str, Any]) -> nn.Module:
        """Load or construct the model from spec.

        For fine-tuning modes: loads pretrained weights from HuggingFace.
        For from_scratch: initializes a fresh model from architecture config.
        For LoRA/QLoRA: loads base model and wraps with PEFT adapters.

        The returned nn.Module is then handed to the Engine, which may:
          - Keep it whole (data parallelism)
          - Split its layers across nodes (pipeline parallelism)
        """

    @abstractmethod
    def prepare_dataset(
        self, spec: dict[str, Any], split: str = "train"
    ) -> Any:
        """Load and preprocess the dataset for training.

        Returns a PyTorch-compatible dataset (torch.utils.data.Dataset or
        HuggingFace datasets.Dataset) ready for the DataLoader.
        """

    @abstractmethod
    def compute_loss(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute the training loss for one batch.

        This is the toolkit's core training logic:
          - Standard LLMs: causal language modeling loss
          - Diffusion models: denoising objective (noise prediction)
          - Classification: cross-entropy
          - Distillation: KL-divergence between teacher/student

        The Engine calls this during both data-parallel and pipeline-parallel
        training. In pipeline mode, the Engine handles activation routing
        across nodes; this method only sees the local computation.
        """

    @abstractmethod
    def create_optimizer(
        self, model: nn.Module, spec: dict[str, Any]
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None]:
        """Create optimizer and optional LR scheduler.

        Toolkits can customize (e.g., Unsloth uses fused optimizers,
        HF Trainer uses AdamW with linear warmup).
        """

    def post_training(self, model: nn.Module, project_dir: Path) -> None:
        """Post-training processing (optional).

        E.g., merge LoRA weights, export to safetensors, upload to hub.
        Default: no-op.
        """

    # ── Convenience methods ───────────────────────────────────

    def get_install_command(self) -> str:
        """Return the pip install command for this toolkit."""
        ti = self.info()
        parts = [ti.pip_package] + ti.extra_dependencies
        return f"pip install {' '.join(parts)}"

    def supports_category(self, category: ModelCategory) -> bool:
        """Check if this toolkit supports the given model category."""
        return category in self.info().supported_categories

    def supports_mode(self, mode: str) -> bool:
        """Check if this toolkit supports the given training mode."""
        return mode in self.info().supported_modes
