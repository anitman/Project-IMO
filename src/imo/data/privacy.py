"""Differential privacy utilities for data contribution."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class PrivacyBudget:
    """Differential privacy budget (epsilon, delta)."""

    epsilon: float
    delta: float = 1e-5

    def __post_init__(self) -> None:
        if self.epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        if self.delta <= 0 or self.delta >= 1:
            raise ValueError("Delta must be in (0, 1)")


class DifferentialPrivacy:
    """Differential privacy mechanisms for gradient protection."""

    def __init__(self, epsilon: float, delta: float = 1e-5):
        self.budget = PrivacyBudget(epsilon=epsilon, delta=delta)

    def add_gaussian_noise(
        self,
        tensor: torch.Tensor,
        sensitivity: float,
    ) -> torch.Tensor:
        """Add Gaussian noise for (epsilon, delta)-DP."""
        sigma = (
            sensitivity * math.sqrt(2 * math.log(1.25 / self.budget.delta)) / self.budget.epsilon
        )

        noise = torch.randn_like(tensor) * sigma
        return tensor + noise

    def add_laplace_noise(
        self,
        tensor: torch.Tensor,
        sensitivity: float,
    ) -> torch.Tensor:
        """Add Laplace noise for epsilon-DP."""
        scale = sensitivity / self.budget.epsilon

        noise = torch.distributions.Laplace(0, scale).sample(tensor.shape)
        return tensor + noise

    def clip_gradients(
        self,
        gradients: list[torch.Tensor],
        max_norm: float,
    ) -> list[torch.Tensor]:
        """Clip gradients to bound sensitivity."""
        total_norm = torch.norm(torch.stack([torch.norm(g.detach()) for g in gradients]))

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            return [g * clip_coef for g in gradients]
        return gradients

    def compute_noise_multiplier(
        self,
        sampling_rate: float,
        num_steps: int,
        target_epsilon: float,
        target_delta: float,
    ) -> float:
        """Compute noise multiplier for targeted privacy budget."""
        from opacus.privacy_engine import PrivacyEngine

        privacy_engine = PrivacyEngine()
        noise_multiplier = privacy_engine.compute_noise_multiplier(
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            sampling_rate=sampling_rate,
            num_steps=num_steps,
        )
        return noise_multiplier


class SecureAggregation:
    """Secure aggregation using pairwise additive masking.

    Each pair of participants (i, j) shares a random seed. Participant i
    adds mask_ij and participant j subtracts the same mask_ij. When all
    masked gradients are summed, the masks cancel out, revealing only
    the aggregate — no individual gradient is ever exposed.

    This is the standard protocol from Bonawitz et al. (CCS 2017).
    """

    def __init__(self, num_participants: int):
        self.num_participants = num_participants

    def generate_pairwise_seeds(
        self,
        participant_id: int,
        num_participants: int,
    ) -> dict[int, int]:
        """Generate deterministic pairwise seeds.

        In a real deployment, these seeds would be established via
        Diffie-Hellman key agreement between each pair. Here we
        use a deterministic function of the pair for reproducibility.
        """
        seeds: dict[int, int] = {}
        for other_id in range(num_participants):
            if other_id == participant_id:
                continue
            # Deterministic seed from the ordered pair
            lo, hi = min(participant_id, other_id), max(participant_id, other_id)
            seeds[other_id] = hash((lo, hi)) & 0xFFFFFFFF
        return seeds

    def mask_gradient(
        self,
        tensor: torch.Tensor,
        participant_id: int,
    ) -> torch.Tensor:
        """Apply additive pairwise masks to a gradient tensor.

        For each pair (self, other):
          - If self < other: add the mask
          - If self > other: subtract the mask
        When all participants' masked tensors are summed, masks cancel.
        """
        masked = tensor.clone()
        seeds = self.generate_pairwise_seeds(participant_id, self.num_participants)

        for other_id, seed in seeds.items():
            gen = torch.Generator()
            gen.manual_seed(seed)
            mask = torch.randn(tensor.shape, generator=gen, dtype=tensor.dtype)

            if participant_id < other_id:
                masked = masked + mask
            else:
                masked = masked - mask

        return masked

    def aggregate_masked(
        self,
        masked_tensors: list[torch.Tensor],
    ) -> torch.Tensor:
        """Aggregate masked tensors — masks cancel, result is true sum.

        Requires ALL participants to contribute. If a participant drops,
        the protocol must be restarted with the remaining set.
        """
        if not masked_tensors:
            raise ValueError("No tensors to aggregate")
        if len(masked_tensors) != self.num_participants:
            raise ValueError(
                f"Expected {self.num_participants} tensors, got {len(masked_tensors)}. "
                "All participants must contribute for masks to cancel."
            )

        aggregated = torch.stack(masked_tensors).sum(dim=0)
        return aggregated
