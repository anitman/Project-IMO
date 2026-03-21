"""Tests for DistributedTrainingEngine + TrainingToolkit integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from imo.toolkits.base import ToolkitCapability, ToolkitInfo, TrainingToolkit
from imo.training.engine import DistributedTrainingEngine, TrainingConfig


class SimpleModel(nn.Module):
    """Minimal model for testing."""

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class StubToolkit(TrainingToolkit):
    """Stub toolkit for testing engine integration."""

    def info(self) -> ToolkitInfo:
        from imo.data.dataset_spec import ModelCategory

        return ToolkitInfo(
            name="stub",
            display_name="Stub Toolkit",
            description="Test stub",
            url="",
            pip_package="",
            supported_categories=[ModelCategory.LLM],
            supported_modes=[ToolkitCapability.FROM_SCRATCH],
        )

    def validate_environment(self) -> list[str]:
        return []

    def setup_project(self, project_dir: Path, spec: dict[str, Any]) -> None:
        pass

    def prepare_config(self, spec: dict[str, Any]) -> dict[str, Any]:
        return {}

    def get_train_command(self, project_dir: Path) -> list[str]:
        return ["echo", "stub"]

    def load_model(self, spec: dict[str, Any]) -> nn.Module:
        hidden = spec.get("hidden", 16)
        return SimpleModel(hidden=hidden)

    def prepare_dataset(self, spec: dict[str, Any], split: str = "train") -> Any:
        return None

    def compute_loss(
        self, model: nn.Module, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        x = batch["input"]
        output = model(x)
        target = batch["target"]
        return nn.functional.mse_loss(output, target)

    def create_optimizer(
        self, model: nn.Module, spec: dict[str, Any]
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None]:
        lr = spec.get("learning_rate", 1e-3)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        return optimizer, None

    def post_training(self, model: nn.Module, project_dir: Path) -> None:
        (project_dir / "training_complete.marker").touch()


class TestEngineToolkitIntegration:
    """Test that the engine correctly delegates to toolkits."""

    @pytest.fixture()
    def config(self) -> TrainingConfig:
        return TrainingConfig(
            project_id="test-project",
            model_architecture="simple",
            model_category="llm",
            max_steps=10,
            batch_size=4,
            learning_rate=1e-3,
        )

    @pytest.fixture()
    def toolkit(self) -> StubToolkit:
        return StubToolkit()

    def test_from_toolkit_creates_engine(
        self, config: TrainingConfig, toolkit: StubToolkit
    ) -> None:
        engine = DistributedTrainingEngine.from_toolkit(
            config=config, toolkit=toolkit, spec={"hidden": 16}
        )
        assert engine.model is not None
        assert isinstance(engine.model, SimpleModel)
        assert engine.toolkit is toolkit

    def test_from_toolkit_uses_toolkit_optimizer(
        self, config: TrainingConfig, toolkit: StubToolkit
    ) -> None:
        engine = DistributedTrainingEngine.from_toolkit(
            config=config, toolkit=toolkit, spec={"hidden": 16}
        )
        assert isinstance(engine.optimizer, torch.optim.SGD)

    @pytest.mark.asyncio
    async def test_train_step_uses_toolkit_loss(
        self, config: TrainingConfig, toolkit: StubToolkit
    ) -> None:
        engine = DistributedTrainingEngine.from_toolkit(
            config=config, toolkit=toolkit, spec={"hidden": 16}
        )
        batch = {
            "input": torch.randn(4, 16),
            "target": torch.randn(4, 16),
        }
        metrics = await engine.train_step(batch)
        assert metrics.loss > 0
        assert metrics.step == 1

    @pytest.mark.asyncio
    async def test_multiple_train_steps(
        self, config: TrainingConfig, toolkit: StubToolkit
    ) -> None:
        engine = DistributedTrainingEngine.from_toolkit(
            config=config, toolkit=toolkit, spec={"hidden": 16}
        )
        for i in range(3):
            batch = {
                "input": torch.randn(4, 16),
                "target": torch.randn(4, 16),
            }
            metrics = await engine.train_step(batch)
            assert metrics.step == i + 1

    @pytest.mark.asyncio
    async def test_training_reduces_loss(
        self, config: TrainingConfig, toolkit: StubToolkit
    ) -> None:
        torch.manual_seed(42)
        engine = DistributedTrainingEngine.from_toolkit(
            config=config, toolkit=toolkit, spec={"hidden": 16, "learning_rate": 0.01}
        )
        # Fixed input and target so loss should decrease
        fixed_input = torch.randn(4, 16)
        target = torch.zeros(4, 16)
        losses = []
        for _ in range(50):
            batch = {"input": fixed_input, "target": target}
            metrics = await engine.train_step(batch)
            losses.append(metrics.loss)
        # Average of last 5 should be lower than average of first 5
        assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5

    def test_engine_without_toolkit_still_works(
        self, config: TrainingConfig
    ) -> None:
        model = SimpleModel(hidden=16)
        engine = DistributedTrainingEngine(config=config, model=model)
        assert engine.toolkit is None
        assert isinstance(engine.optimizer, torch.optim.AdamW)

    @pytest.mark.asyncio
    async def test_post_training_called_on_shutdown(
        self, config: TrainingConfig, toolkit: StubToolkit, tmp_path: Path
    ) -> None:
        config.checkpoint_interval = 999999  # avoid checkpoint during test
        engine = DistributedTrainingEngine.from_toolkit(
            config=config, toolkit=toolkit, spec={"hidden": 16}
        )
        await engine.shutdown(project_dir=str(tmp_path))
        assert (tmp_path / "training_complete.marker").exists()

    def test_record_contribution(
        self, config: TrainingConfig, toolkit: StubToolkit
    ) -> None:
        engine = DistributedTrainingEngine.from_toolkit(
            config=config, toolkit=toolkit, spec={"hidden": 16}
        )
        engine.record_contribution("node-1", compute_time=100.0, loss_reduction=0.5, vram_gb=24)
        contribs = engine.get_contributions()
        assert "node-1" in contribs
        assert contribs["node-1"]["compute_time"] == 100.0


class TestEngineToolkitPipeline:
    """Test pipeline parallelism with toolkit-loaded models."""

    @pytest.fixture()
    def config(self) -> TrainingConfig:
        return TrainingConfig(
            project_id="pipeline-test",
            model_architecture="simple",
            model_category="llm",
            parallelism_mode="pipeline_parallel",
            total_blocks=3,
        )

    def test_pipeline_config(self, config: TrainingConfig) -> None:
        assert config.parallelism_mode == "pipeline_parallel"
        assert config.total_blocks == 3

    def test_model_has_splittable_layers(self) -> None:
        model = SimpleModel(hidden=16)
        layers = list(model.layers.children())
        assert len(layers) == 3  # Linear, ReLU, Linear — splittable
