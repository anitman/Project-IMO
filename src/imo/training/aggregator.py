"""Gradient aggregation for distributed training."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeAlias

import torch

Gradients: TypeAlias = dict[str, torch.Tensor]


class AggregationStrategy(ABC):
    """Base class for gradient aggregation strategies."""

    @abstractmethod
    def aggregate(
        self,
        gradients: list[Gradients],
        weights: list[float] | None = None,
    ) -> Gradients:
        """Aggregate multiple gradient updates."""
        pass


class FederatedAveraging(AggregationStrategy):
    """Standard federated averaging aggregation."""

    def aggregate(
        self,
        gradients: list[Gradients],
        weights: list[float] | None = None,
    ) -> Gradients:
        """Aggregate gradients using weighted averaging."""
        if not gradients:
            raise ValueError("No gradients to aggregate")

        if len(gradients) == 1:
            return gradients[0]

        if weights is None:
            weights = [1.0 / len(gradients)] * len(gradients)

        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        aggregated: Gradients = {}
        param_names = gradients[0].keys()

        for param_name in param_names:
            weighted_gradients = []
            for grad_dict, weight in zip(gradients, normalized_weights):
                if param_name in grad_dict:
                    weighted_gradients.append(grad_dict[param_name] * weight)

            aggregated[param_name] = sum(weighted_gradients)

        return aggregated


class TrimmedMean(AggregationStrategy):
    """Trimmed mean aggregation for Byzantine-robust training."""

    def __init__(self, trim_ratio: float = 0.1):
        if not 0 <= trim_ratio < 0.5:
            raise ValueError("trim_ratio must be in [0, 0.5)")
        self.trim_ratio = trim_ratio

    def aggregate(
        self,
        gradients: list[Gradients],
        weights: list[float] | None = None,
    ) -> Gradients:
        """Aggregate gradients using trimmed mean."""
        if len(gradients) < 3:
            return FederatedAveraging().aggregate(gradients, weights)

        aggregated: Gradients = {}
        param_names = gradients[0].keys()
        trim_count = int(len(gradients) * self.trim_ratio)

        for param_name in param_names:
            param_grads = [g[param_name].flatten() for g in gradients if param_name in g]

            if len(param_grads) < 3:
                aggregated[param_name] = torch.mean(torch.stack(param_grads)).reshape(
                    gradients[0][param_name].shape
                )
                continue

            stacked = torch.stack(param_grads)
            sorted_grads = torch.sort(stacked, dim=0)[0]

            if trim_count > 0:
                trimmed = sorted_grads[trim_count:-trim_count]
            else:
                trimmed = sorted_grads

            aggregated[param_name] = torch.mean(trimmed, dim=0).reshape(
                gradients[0][param_name].shape
            )

        return aggregated


class Krum(AggregationStrategy):
    """Krum aggregation for Byzantine-robust training."""

    def __init__(self, num_byzantine: int = 1):
        self.num_byzantine = num_byzantine

    def aggregate(
        self,
        gradients: list[Gradients],
        weights: list[float] | None = None,
    ) -> Gradients:
        """Aggregate using Krum selection."""
        if len(gradients) <= 2 * self.num_byzantine + 2:
            return FederatedAveraging().aggregate(gradients, weights)

        distances = self._compute_pairwise_distances(gradients)
        n = len(gradients)
        f = self.num_byzantine

        scores = []
        for i in range(n):
            sorted_distances = sorted(distances[i])
            score = sum(sorted_distances[: n - f - 2])
            scores.append(score)

        krum_index = scores.index(min(scores))
        return gradients[krum_index]

    def _compute_pairwise_distances(
        self,
        gradients: list[Gradients],
    ) -> list[list[float]]:
        """Compute pairwise L2 distances between gradient vectors."""
        n = len(gradients)
        distances: list[list[float]] = []

        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    row.append(0.0)
                    continue

                dist = 0.0
                param_names = gradients[i].keys()
                for param_name in param_names:
                    if param_name in gradients[i] and param_name in gradients[j]:
                        diff = gradients[i][param_name] - gradients[j][param_name]
                        dist += torch.norm(diff).item() ** 2
                row.append(dist**0.5)
            distances.append(row)

        return distances
