"""Gradient compression and communication."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TypeAlias

import torch

Gradients: TypeAlias = dict[str, torch.Tensor]


@dataclass
class CompressedGradient:
    """Compressed gradient data."""

    indices: torch.Tensor
    values: torch.Tensor
    shape: tuple[int, ...]
    dtype: str


class CompressionMethod(ABC):
    """Base class for gradient compression."""

    @abstractmethod
    def compress(self, gradients: Gradients) -> dict[str, CompressedGradient]:
        """Compress gradients."""
        pass

    @abstractmethod
    def decompress(
        self,
        compressed: dict[str, CompressedGradient],
    ) -> Gradients:
        """Decompress gradients."""
        pass


class TopKCompression(CompressionMethod):
    """Top-K sparsification compression."""

    def __init__(self, sparsity: float = 0.01):
        if not 0 < sparsity <= 1:
            raise ValueError("sparsity must be in (0, 1]")
        self.sparsity = sparsity

    def compress(self, gradients: Gradients) -> dict[str, CompressedGradient]:
        """Compress using Top-K sparsification."""
        compressed: dict[str, CompressedGradient] = {}

        for param_name, tensor in gradients.items():
            flat = tensor.flatten()
            k = max(1, int(len(flat) * self.sparsity))

            _, indices = torch.topk(flat.abs(), k)
            values = flat[indices]

            compressed[param_name] = CompressedGradient(
                indices=indices,
                values=values,
                shape=tensor.shape,
                dtype=str(tensor.dtype),
            )

        return compressed

    def decompress(
        self,
        compressed: dict[str, CompressedGradient],
    ) -> Gradients:
        """Decompress Top-K gradients."""
        gradients: Gradients = {}

        for param_name, comp in compressed.items():
            dtype_str = comp.dtype.replace("torch.", "")
            tensor = torch.zeros(comp.shape, dtype=getattr(torch, dtype_str))
            flat = tensor.flatten()
            flat[comp.indices] = comp.values
            gradients[param_name] = flat.reshape(comp.shape)

        return gradients


class SignCompression(CompressionMethod):
    """Sign-based compression with magnitude estimation."""

    def compress(self, gradients: Gradients) -> dict[str, CompressedGradient]:
        """Compress using sign encoding."""
        compressed: dict[str, CompressedGradient] = {}

        for param_name, tensor in gradients.items():
            signs = (tensor >= 0).to(torch.int8)
            magnitude = tensor.abs().mean()

            compressed[param_name] = CompressedGradient(
                indices=signs,
                values=torch.tensor([magnitude.item()]),
                shape=tensor.shape,
                dtype="float32",
            )

        return compressed

    def decompress(
        self,
        compressed: dict[str, CompressedGradient],
    ) -> Gradients:
        """Decompress sign-based gradients."""
        gradients: Gradients = {}

        for param_name, comp in compressed.items():
            signs = comp.indices.to(torch.float32) * 2 - 1
            magnitude = comp.values[0]
            gradients[param_name] = signs * magnitude

        return gradients


class GradientCommunicator:
    """Handle gradient communication between nodes.

    All gradient data is transmitted over TLS-encrypted channels when a
    TLSTransport is provided. Without TLS, communication falls back to
    the DHT layer (which should itself use encrypted connections).
    """

    def __init__(
        self,
        compression_method: CompressionMethod | None = None,
        error_feedback: bool = True,
        dht: Any | None = None,
        require_tls: bool = True,
    ):
        self.compression = compression_method or TopKCompression()
        self.error_feedback = error_feedback
        self.residuals: dict[str, torch.Tensor] = {}
        self.dht = dht
        self.require_tls = require_tls
        self._tls_verified_peers: set[str] = set()

    def register_tls_peer(self, peer_id: str) -> None:
        """Mark a peer as having an established TLS channel."""
        self._tls_verified_peers.add(peer_id)

    def revoke_tls_peer(self, peer_id: str) -> None:
        """Remove a peer's TLS verification status."""
        self._tls_verified_peers.discard(peer_id)

    def _check_tls(self, peer_id: str) -> None:
        """Ensure peer has an established TLS channel if TLS is required."""
        if self.require_tls and peer_id not in self._tls_verified_peers:
            raise PermissionError(
                f"Peer {peer_id[:12]} has no verified TLS channel. "
                "Gradient exchange requires encrypted transport."
            )

    async def send_gradients(
        self,
        peer_id: str,
        gradients: Gradients,
    ) -> None:
        """Send compressed gradients to a peer over TLS."""
        self._check_tls(peer_id)
        compressed = self.compression.compress(gradients)

        if self.error_feedback:
            self._add_residuals(gradients, compressed)

        if self.dht:
            await self.dht.send(peer_id, "gradients", compressed)

    async def receive_gradients(self, peer_id: str) -> Gradients:
        """Receive and decompress gradients from a peer over TLS."""
        self._check_tls(peer_id)
        if self.dht:
            compressed = await self.dht.receive(peer_id, "gradients")
            return self.compression.decompress(compressed)
        return {}

    def _add_residuals(
        self,
        gradients: Gradients,
        compressed: dict[str, CompressedGradient],
    ) -> None:
        """Add compression error to residual buffer."""
        for param_name, tensor in gradients.items():
            if param_name not in self.residuals:
                self.residuals[param_name] = torch.zeros_like(tensor)

            decompressed = self.compression.decompress(
                {
                    param_name: compressed[param_name],
                }
            )
            error = tensor - decompressed[param_name]
            self.residuals[param_name] += error
