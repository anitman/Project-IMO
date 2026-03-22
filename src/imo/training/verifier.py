"""Gradient consistency verification and cheating detection."""

from __future__ import annotations

import hashlib
import logging
import random
from dataclasses import dataclass
from typing import TypeAlias

import torch

Gradients: TypeAlias = dict[str, torch.Tensor]

logger = logging.getLogger(__name__)


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


@dataclass
class SpotCheckResult:
    """Result of a redundant computation spot-check."""

    batch_id: str
    nodes_checked: list[str]
    is_consistent: bool
    max_divergence: float
    flagged_nodes: list[str]
    message: str


class RedundantVerifier:
    """Folding@home-inspired redundant computation spot-checking.

    Randomly selects a fraction of training batches to be independently
    computed by multiple nodes. Results are compared — if a node's output
    diverges significantly from the majority, it is flagged as dishonest.

    This catches nodes that:
    - Submit fabricated gradients without actually computing them
    - Intentionally corrupt gradients (targeted poisoning)
    - Have hardware faults producing silent data corruption

    The key insight: you don't need to verify EVERY batch. Even a small
    spot-check rate (e.g. 5%) makes cheating risky, because a dishonest
    node never knows which batches are being verified.
    """

    def __init__(
        self,
        spot_check_rate: float = 0.05,
        divergence_threshold: float = 0.1,
        min_verifiers: int = 2,
        strike_limit: int = 3,
    ):
        if not 0 < spot_check_rate <= 1.0:
            raise ValueError("spot_check_rate must be in (0, 1]")
        self.spot_check_rate = spot_check_rate
        self.divergence_threshold = divergence_threshold
        self.min_verifiers = min_verifiers
        self.strike_limit = strike_limit
        self.strikes: dict[str, int] = {}
        self.check_history: list[SpotCheckResult] = []

    def should_spot_check(self, step: int) -> bool:
        """Decide whether this step should be spot-checked.

        Uses deterministic randomness seeded by step to ensure all nodes
        agree on which steps are checked without extra communication.
        """
        rng = random.Random(step * 31337)
        return rng.random() < self.spot_check_rate

    def select_verifiers(
        self,
        step: int,
        available_nodes: list[str],
        primary_node: str,
    ) -> list[str]:
        """Select which nodes should redundantly compute this batch.

        Excludes the primary node. Selection is deterministic from step
        so all participants agree without a coordination round.
        """
        candidates = [n for n in available_nodes if n != primary_node]
        if len(candidates) < self.min_verifiers:
            return candidates

        rng = random.Random(step * 31337 + 7)
        return rng.sample(candidates, min(self.min_verifiers, len(candidates)))

    def verify_results(
        self,
        batch_id: str,
        results: dict[str, Gradients],
    ) -> SpotCheckResult:
        """Compare redundantly computed results and flag divergent nodes.

        Uses majority-vote: compute pairwise cosine similarities, find the
        majority cluster, and flag nodes outside it.
        """
        node_ids = list(results.keys())

        if len(node_ids) < 2:
            return SpotCheckResult(
                batch_id=batch_id,
                nodes_checked=node_ids,
                is_consistent=True,
                max_divergence=0.0,
                flagged_nodes=[],
                message="Not enough nodes for comparison",
            )

        # Flatten each node's gradients
        flat: dict[str, torch.Tensor] = {}
        for nid, grads in results.items():
            tensors = [grads[k].flatten() for k in sorted(grads.keys())]
            flat[nid] = torch.cat(tensors) if tensors else torch.tensor(0.0)

        # Compute pairwise cosine similarities
        similarities: dict[tuple[str, str], float] = {}
        for i, nid_a in enumerate(node_ids):
            for nid_b in node_ids[i + 1:]:
                norm_a = torch.norm(flat[nid_a])
                norm_b = torch.norm(flat[nid_b])
                if norm_a == 0 or norm_b == 0:
                    sim = 0.0
                else:
                    sim = (torch.dot(flat[nid_a], flat[nid_b]) / (norm_a * norm_b)).item()
                similarities[(nid_a, nid_b)] = sim

        # Find majority cluster: node with highest average similarity to others
        avg_sim: dict[str, float] = {}
        for nid in node_ids:
            sims = []
            for (a, b), sim in similarities.items():
                if a == nid or b == nid:
                    sims.append(sim)
            avg_sim[nid] = sum(sims) / len(sims) if sims else 0.0

        # Flag nodes whose average similarity is below threshold
        flagged: list[str] = []
        max_divergence = 0.0

        median_sim = sorted(avg_sim.values())[len(avg_sim) // 2] if avg_sim else 1.0

        for nid, sim in avg_sim.items():
            divergence = max(0.0, median_sim - sim)
            max_divergence = max(max_divergence, divergence)
            if divergence > self.divergence_threshold:
                flagged.append(nid)
                self.strikes[nid] = self.strikes.get(nid, 0) + 1
                logger.warning(
                    "Spot-check: node %s diverged (avg_sim=%.4f, median=%.4f) — strike %d/%d",
                    nid, sim, median_sim, self.strikes[nid], self.strike_limit,
                )

        result = SpotCheckResult(
            batch_id=batch_id,
            nodes_checked=node_ids,
            is_consistent=len(flagged) == 0,
            max_divergence=max_divergence,
            flagged_nodes=flagged,
            message=f"Checked {len(node_ids)} nodes, flagged {len(flagged)}",
        )
        self.check_history.append(result)
        return result

    def is_banned(self, node_id: str) -> bool:
        """Check if a node has exceeded its strike limit."""
        return self.strikes.get(node_id, 0) >= self.strike_limit

    def get_strikes(self, node_id: str) -> int:
        """Get current strike count for a node."""
        return self.strikes.get(node_id, 0)
