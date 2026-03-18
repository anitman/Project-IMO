"""Security for distributed training - poisoning detection and mitigation."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import TypeAlias

import torch

Gradients: TypeAlias = dict[str, torch.Tensor]


@dataclass
class SecurityAlert:
    """Security alert for training anomalies."""

    alert_type: str
    severity: str
    node_id: str
    message: str
    details: dict[str, float] = field(default_factory=dict)


class PoisoningDetector:
    """Detect gradient poisoning attacks."""

    def __init__(
        self,
        anomaly_threshold: float = 2.0,
        reputation_decay: float = 0.95,
    ):
        self.anomaly_threshold = anomaly_threshold
        self.reputation_decay = reputation_decay
        self.reputations: dict[str, float] = {}
        self.alerts: list[SecurityAlert] = []

    def analyze(
        self,
        node_id: str,
        gradients: Gradients,
        cluster_gradients: list[Gradients],
    ) -> SecurityAlert | None:
        """Analyze gradients for poisoning indicators."""
        if len(cluster_gradients) < 3:
            return None

        metrics = self._compute_metrics(gradients, cluster_gradients)

        if self._is_anomalous(metrics):
            self._update_reputation(node_id, decrease=True)
            alert = SecurityAlert(
                alert_type="gradient_poisoning",
                severity="high",
                node_id=node_id,
                message="Anomalous gradient detected",
                details=metrics,
            )
            self.alerts.append(alert)
            return alert

        self._update_reputation(node_id, decrease=False)
        return None

    def _compute_metrics(
        self,
        gradients: Gradients,
        cluster: list[Gradients],
    ) -> dict[str, float]:
        """Compute anomaly metrics."""
        metrics: dict[str, float] = {}

        all_norms = []
        for g in cluster:
            norm = self._compute_gradient_norm(g)
            all_norms.append(norm)

        if all_norms:
            mean_norm = statistics.mean(all_norms)
            std_norm = statistics.stdev(all_norms) if len(all_norms) > 1 else 0
            current_norm = self._compute_gradient_norm(gradients)

            metrics["gradient_norm"] = current_norm
            metrics["norm_z_score"] = (
                abs(current_norm - mean_norm) / std_norm if std_norm > 0 else 0
            )

        return metrics

    def _compute_gradient_norm(self, gradients: Gradients) -> float:
        """Compute L2 norm of gradient vector."""
        total = 0.0
        for tensor in gradients.values():
            total += torch.norm(tensor).item() ** 2
        return total**0.5

    def _is_anomalous(self, metrics: dict[str, float]) -> bool:
        """Check if metrics indicate anomaly."""
        z_score = metrics.get("norm_z_score", 0)
        return z_score > self.anomaly_threshold

    def _update_reputation(self, node_id: str, decrease: bool) -> None:
        """Update node reputation."""
        if node_id not in self.reputations:
            self.reputations[node_id] = 1.0

        if decrease:
            self.reputations[node_id] *= self.reputation_decay
        else:
            self.reputations[node_id] = min(
                1.0,
                self.reputations[node_id] * (1 + 0.01),
            )

    def get_reputation(self, node_id: str) -> float:
        """Get current reputation for a node."""
        return self.reputations.get(node_id, 1.0)


class ByzantineRobustAggregator:
    """Byzantine-robust gradient aggregation."""

    def __init__(self, method: str = "coordinate_wise_median"):
        if method not in {"trimmed_mean", "coordinate_wise_median", "krum"}:
            raise ValueError(f"Unknown method: {method}")
        self.method = method

    def aggregate(
        self,
        gradients_list: list[Gradients],
        trim_ratio: float = 0.1,
    ) -> Gradients:
        """Aggregate gradients with Byzantine resilience."""
        if self.method == "coordinate_wise_median":
            return self._coordinate_wise_median(gradients_list)
        elif self.method == "trimmed_mean":
            return self._trimmed_mean(gradients_list, trim_ratio)
        else:
            raise ValueError(f"Method not implemented: {self.method}")

    def _coordinate_wise_median(
        self,
        gradients_list: list[Gradients],
    ) -> Gradients:
        """Aggregate using coordinate-wise median."""
        if not gradients_list:
            raise ValueError("No gradients to aggregate")

        aggregated: Gradients = {}
        param_names = gradients_list[0].keys()

        for param_name in param_names:
            tensors = [g[param_name] for g in gradients_list if param_name in g]

            if len(tensors) == 1:
                aggregated[param_name] = tensors[0]
                continue

            stacked = torch.stack(tensors)
            median = torch.median(stacked, dim=0)[0]
            aggregated[param_name] = median

        return aggregated

    def _trimmed_mean(
        self,
        gradients_list: list[Gradients],
        trim_ratio: float,
    ) -> Gradients:
        """Aggregate using trimmed mean."""
        if not gradients_list:
            raise ValueError("No gradients to aggregate")

        n = len(gradients_list)
        trim_count = max(1, int(n * trim_ratio))

        if 2 * trim_count >= n:
            return self._coordinate_wise_median(gradients_list)

        aggregated: Gradients = {}
        param_names = gradients_list[0].keys()

        for param_name in param_names:
            tensors = [g[param_name] for g in gradients_list if param_name in g]

            if len(tensors) <= 2 * trim_count:
                aggregated[param_name] = torch.mean(torch.stack(tensors))
                continue

            stacked = torch.stack(tensors)
            sorted_tensor = torch.sort(stacked, dim=0)[0]
            trimmed = sorted_tensor[trim_count:-trim_count]
            aggregated[param_name] = torch.mean(trimmed, dim=0)

        return aggregated
