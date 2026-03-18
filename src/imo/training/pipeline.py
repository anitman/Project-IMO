"""Pipeline parallelism for distributed training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PipelineStage:
    """A stage in the pipeline."""

    node_id: str
    layer_range: tuple[int, int]
    model: nn.Module


class PipelineParallelism:
    """Manage pipeline parallelism across nodes."""

    def __init__(self, model: nn.Module, stages: list[PipelineStage]):
        self.model = model
        self.stages = stages

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run forward pass through pipeline."""
        output = inputs

        for stage in self.stages:
            start, end = stage.layer_range
            stage_model = self._get_submodule(start, end)
            output = stage_model(output)

        return output

    def _get_submodule(
        self,
        start_layer: int,
        end_layer: int,
    ) -> nn.Module:
        """Get a sub-module covering the specified layer range."""
        layers = list(self.model.children())
        return nn.Sequential(*layers[start_layer:end_layer])

    def get_stage_for_layer(self, layer_index: int) -> PipelineStage | None:
        """Get the stage responsible for a layer."""
        for stage in self.stages:
            start, end = stage.layer_range
            if start <= layer_index < end:
                return stage
        return None


class MicrobatchScheduler:
    """Schedule microbatches across pipeline stages."""

    def __init__(self, num_microbatches: int = 4):
        self.num_microbatches = num_microbatches

    def split_batch(self, batch: torch.Tensor) -> list[torch.Tensor]:
        """Split a batch into microbatches."""
        return torch.chunk(batch, self.num_microbatches)

    def schedule(
        self,
        microbatches: list[torch.Tensor],
        stages: list[PipelineStage],
    ) -> list[tuple[int, torch.Tensor]]:
        """Create schedule for processing microbatches."""
        schedule: list[tuple[int, torch.Tensor]] = []

        for stage_idx, stage in enumerate(stages):
            for mb_idx, microbatch in enumerate(microbatches):
                schedule.append((stage_idx, microbatch))

        return schedule


def create_pipeline_model(
    model: nn.Module,
    node_assignments: dict[str, list[int]],
) -> PipelineParallelism:
    """Create a pipeline parallel model from node assignments."""
    stages: list[PipelineStage] = []

    for node_id, layers in node_assignments.items():
        if not layers:
            continue

        start_layer = min(layers)
        end_layer = max(layers) + 1

        stages.append(
            PipelineStage(
                node_id=node_id,
                layer_range=(start_layer, end_layer),
                model=model,
            )
        )

    stages.sort(key=lambda s: s.layer_range[0])
    return PipelineParallelism(model, stages)
