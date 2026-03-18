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
    """Secure aggregation for privacy-preserving gradient updates."""

    def __init__(self, num_participants: int, modulus: int = 2**32):
        self.num_participants = num_participants
        self.modulus = modulus

    def encrypt(
        self,
        tensor: torch.Tensor,
        secret_key: int,
    ) -> torch.Tensor:
        """Encrypt tensor with secret key."""
        encrypted = (tensor * secret_key) % self.modulus
        return encrypted

    def decrypt(
        self,
        encrypted_tensor: torch.Tensor,
        secret_key: int,
    ) -> torch.Tensor:
        """Decrypt tensor with secret key."""
        inverse_key = pow(secret_key, -1, self.modulus)
        decrypted = (encrypted_tensor * inverse_key) % self.modulus
        return decrypted

    def aggregate_encrypted(
        self,
        encrypted_tensors: list[torch.Tensor],
    ) -> torch.Tensor:
        """Aggregate encrypted tensors without decryption."""
        if not encrypted_tensors:
            raise ValueError("No tensors to aggregate")

        aggregated = encrypted_tensors[0]
        for t in encrypted_tensors[1:]:
            aggregated = (aggregated + t) % self.modulus

        return aggregated
