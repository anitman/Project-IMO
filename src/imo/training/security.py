"""Security for distributed training - poisoning detection and mitigation."""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import torch
import torch.nn as nn

Gradients: TypeAlias = dict[str, torch.Tensor]

logger = logging.getLogger(__name__)


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


class TrustedRootValidator:
    """FLTrust-style gradient validation using a small trusted root dataset.

    Instead of blindly rejecting statistical outliers (which kills diversity),
    this validator computes a "trusted gradient" from a small, curated root
    dataset on each step. Node gradients are then scored by cosine similarity
    to the trusted gradient, and aggregated with trust-weighted averaging.

    This means:
    - A node training on rare/diverse data that ALIGNS directionally with the
      trusted gradient is kept (diversity preserved).
    - A poisoned gradient that points in a fundamentally different direction
      is down-weighted to near zero.

    Reference: FLTrust (Cao et al., NDSS 2021).
    """

    def __init__(
        self,
        root_dataset: Any,
        model: nn.Module,
        loss_fn: Any,
        device: torch.device | str = "cpu",
        norm_clip: float = 10.0,
    ):
        self.root_dataset = root_dataset
        self.model = model
        self.loss_fn = loss_fn
        self.device = torch.device(device) if isinstance(device, str) else device
        self.norm_clip = norm_clip
        self._root_iter: Any = None

    def _get_root_batch(self) -> dict[str, torch.Tensor]:
        """Get next batch from root dataset (cycles)."""
        if self._root_iter is None:
            self._root_iter = iter(self.root_dataset)
        try:
            batch = next(self._root_iter)
        except StopIteration:
            self._root_iter = iter(self.root_dataset)
            batch = next(self._root_iter)
        if isinstance(batch, dict):
            return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
        result: dict[str, torch.Tensor] = dict(batch)
        return result

    def compute_trusted_gradient(self) -> Gradients:
        """Compute gradient on the trusted root dataset."""
        self.model.zero_grad()
        batch = self._get_root_batch()

        if self.loss_fn is not None:
            outputs = self.model(**batch)
            loss = self.loss_fn(outputs, batch)
        else:
            outputs = self.model(**batch)
            loss = outputs.loss

        loss.backward()

        trusted: Gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                trusted[name] = param.grad.detach().clone()

        self.model.zero_grad()
        return trusted

    def compute_trust_scores(
        self,
        trusted_gradient: Gradients,
        node_gradients: dict[str, Gradients],
    ) -> dict[str, float]:
        """Score each node's gradient by cosine similarity to trusted gradient.

        Returns a dict of node_id -> trust score in [0, 1].
        Negative similarities are clamped to 0 (adversarial direction).
        """
        scores: dict[str, float] = {}

        trusted_flat = self._flatten(trusted_gradient)
        trusted_norm = torch.norm(trusted_flat)

        if trusted_norm == 0:
            return {nid: 1.0 for nid in node_gradients}

        for node_id, grads in node_gradients.items():
            node_flat = self._flatten(grads)
            node_norm = torch.norm(node_flat)

            if node_norm == 0:
                scores[node_id] = 0.0
                continue

            cos_sim = torch.dot(trusted_flat, node_flat) / (trusted_norm * node_norm)
            # ReLU: clamp negative similarity to 0 (adversarial gradients get zero weight)
            scores[node_id] = max(0.0, cos_sim.item())

        return scores

    def trust_weighted_aggregate(
        self,
        trusted_gradient: Gradients,
        node_gradients: dict[str, Gradients],
    ) -> Gradients:
        """Aggregate node gradients weighted by trust scores.

        Each node gradient is normalized to the trusted gradient's norm
        (preventing magnitude attacks), then weighted by its trust score.
        """
        scores = self.compute_trust_scores(trusted_gradient, node_gradients)

        total_score = sum(scores.values())
        if total_score == 0:
            logger.warning("All node gradients have zero trust — falling back to trusted gradient")
            return trusted_gradient

        trusted_norm = torch.norm(self._flatten(trusted_gradient)).item()

        aggregated: Gradients = {}
        for param_name in trusted_gradient:
            weighted_sum = torch.zeros_like(trusted_gradient[param_name])

            for node_id, grads in node_gradients.items():
                if param_name not in grads:
                    continue

                score = scores[node_id]
                if score == 0:
                    continue

                # Normalize node gradient to trusted norm (prevent magnitude attacks)
                node_flat = self._flatten(grads)
                node_norm = torch.norm(node_flat).item()
                if node_norm > 0:
                    scale = min(trusted_norm / node_norm, self.norm_clip)
                else:
                    scale = 0.0

                weighted_sum += (score / total_score) * grads[param_name] * scale

            aggregated[param_name] = weighted_sum

        return aggregated

    def _flatten(self, gradients: Gradients) -> torch.Tensor:
        """Flatten all gradient tensors into a single vector."""
        tensors = [gradients[k].flatten() for k in sorted(gradients.keys())]
        if not tensors:
            return torch.tensor(0.0)
        return torch.cat(tensors)


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
