"""Gradient consistency verification and cheating detection."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TypeAlias

import torch

Gradients: TypeAlias = dict[str, torch.Tensor]


@dataclass
class VerificationResult:
    """Result of gradient verification."""

    node_id: str
    is_valid: bool
    cosine_similarity: float
    checksum: str
    message: str


class GradientVerifier:
    """Verify gradient consistency across nodes."""

    def __init__(self, similarity_threshold: float = 0.95):
        self.similarity_threshold = similarity_threshold

    def verify(
        self,
        node_id: str,
        gradients: Gradients,
        reference_gradients: Gradients | None = None,
    ) -> VerificationResult:
        """Verify gradient consistency."""
        checksum = self._compute_checksum(gradients)

        if reference_gradients is None:
            return VerificationResult(
                node_id=node_id,
                is_valid=True,
                cosine_similarity=1.0,
                checksum=checksum,
                message="No reference provided",
            )

        similarity = self._compute_cosine_similarity(gradients, reference_gradients)

        if similarity < self.similarity_threshold:
            return VerificationResult(
                node_id=node_id,
                is_valid=False,
                cosine_similarity=similarity,
                checksum=checksum,
                message=f"Low similarity: {similarity:.4f}",
            )

        return VerificationResult(
            node_id=node_id,
            is_valid=True,
            cosine_similarity=similarity,
            checksum=checksum,
            message="Gradient verified",
        )

    def verify_batch(
        self,
        node_id: str,
        gradients: Gradients,
        reference_gradients: Gradients,
    ) -> VerificationResult:
        """Verify a batch of gradients."""
        return self.verify(node_id, gradients, reference_gradients)

    def _compute_checksum(self, gradients: Gradients) -> str:
        """Compute checksum of gradients."""
        param_hashes = []
        for param_name in sorted(gradients.keys()):
            tensor = gradients[param_name]
            tensor_hash = hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()
            param_hashes.append(tensor_hash)

        combined = "|".join(param_hashes)
        return hashlib.sha256(combined.encode()).hexdigest()

    def _compute_cosine_similarity(
        self,
        gradients: Gradients,
        reference: Gradients,
    ) -> float:
        """Compute cosine similarity between gradient vectors."""
        total_dot = 0.0
        total_norm_grad = 0.0
        total_norm_ref = 0.0

        for param_name in gradients.keys():
            if param_name not in reference:
                continue

            grad = gradients[param_name].flatten()
            ref = reference[param_name].flatten()

            total_dot += torch.dot(grad, ref).item()
            total_norm_grad += torch.norm(grad).item() ** 2
            total_norm_ref += torch.norm(ref).item() ** 2

        if total_norm_grad == 0 or total_norm_ref == 0:
            return 0.0

        similarity = total_dot / ((total_norm_grad**0.5) * (total_norm_ref**0.5))
        return max(0.0, min(1.0, similarity))


class GradientAnomalyDetector:
    """Detect anomalous gradient updates."""

    def __init__(
        self,
        outlier_threshold: float = 3.0,
        min_samples: int = 10,
    ):
        self.outlier_threshold = outlier_threshold
        self.min_samples = min_samples
        self.history: dict[str, list[Gradients]] = {}

    def check(
        self,
        node_id: str,
        gradients: Gradients,
    ) -> bool:
        """Check if gradients are anomalous."""
        if node_id not in self.history:
            self.history[node_id] = []

        self.history[node_id].append(gradients)

        if len(self.history[node_id]) < self.min_samples:
            return True

        recent = self.history[node_id][-self.min_samples :]
        return not self._is_outlier(recent, gradients)

    def _is_outlier(
        self,
        history: list[Gradients],
        current: Gradients,
    ) -> bool:
        """Check if current gradients are statistical outliers."""
        param_stats: dict[str, tuple[float, float]] = {}

        for param_name in current.keys():
            values = [g[param_name].mean().item() for g in history if param_name in g]

            if len(values) < 2:
                continue

            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std = variance**0.5

            if std > 0:
                param_stats[param_name] = (mean, std)

        for param_name, (mean, std) in param_stats.items():
            if param_name not in current:
                continue

            current_value = current[param_name].mean().item()
            z_score = abs(current_value - mean) / std

            if z_score > self.outlier_threshold:
                return True

        return False
