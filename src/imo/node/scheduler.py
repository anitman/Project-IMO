"""VRAM-aware task scheduling for heterogeneous clusters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

from imo.node.discovery import PeerInfo

LayerAssignment: TypeAlias = dict[str, list[int]]


@dataclass
class ClusterInfo:
    """Information about a cluster of nodes."""

    nodes: list[PeerInfo]
    total_vram: int
    model_layers: int
    model_params_billion: float = 0.0
    estimated_vram_per_layer_gb: float = 0.0


class VRAMScheduler:
    """Schedule model layers based on VRAM capacity.

    Uses bin-packing heuristic: sort peers by VRAM descending, assign
    layers proportionally. The last peer absorbs any rounding remainder.
    """

    def __init__(self, model_name: str, total_layers: int):
        self.model_name = model_name
        self.total_layers = total_layers

    def schedule(self, peers: list[PeerInfo]) -> LayerAssignment:
        """Assign layers to peers proportionally to VRAM."""
        if not peers:
            raise ValueError("No peers available")

        total_vram = sum(p.vram_gb for p in peers)
        if total_vram == 0:
            raise ValueError("No VRAM available")

        assignments: LayerAssignment = {}
        current_layer = 0

        sorted_peers = sorted(peers, key=lambda p: p.vram_gb, reverse=True)

        for i, peer in enumerate(sorted_peers):
            if i == len(sorted_peers) - 1:
                num_layers = self.total_layers - current_layer
            else:
                ratio = peer.vram_gb / total_vram
                num_layers = max(1, int(ratio * self.total_layers))
                num_layers = min(num_layers, self.total_layers - current_layer)

            if num_layers <= 0:
                assignments[peer.node_id] = []
                continue

            assignments[peer.node_id] = list(range(current_layer, current_layer + num_layers))
            current_layer += num_layers

        return assignments

    def reschedule(
        self,
        current_assignments: LayerAssignment,
        new_peers: list[PeerInfo],
        removed_peers: list[str],
    ) -> LayerAssignment:
        """Reschedule when cluster topology changes."""
        active_peers = [p for p in new_peers if p.node_id not in removed_peers]
        return self.schedule(active_peers)

    def estimate_vram_requirement(
        self,
        model_params_billion: float,
        dtype_bytes: int = 2,
        optimizer_multiplier: float = 3.0,
    ) -> float:
        """Estimate VRAM requirement in GB for training.

        A rough formula: params_B * dtype_bytes * optimizer_multiplier.
        For Adam with fp16: 7B model ~ 7 * 2 * 3 = 42 GB.
        """
        return model_params_billion * dtype_bytes * optimizer_multiplier


class GradientAggregator:
    """Assign gradient aggregation roles."""

    def __init__(self, num_aggregators: int = 2):
        self.num_aggregators = num_aggregators

    def select_aggregators(
        self,
        peers: list[PeerInfo],
    ) -> list[str]:
        """Select nodes to serve as gradient aggregators."""
        sorted_peers = sorted(
            peers,
            key=lambda p: (p.bandwidth_mbps, p.vram_gb),
            reverse=True,
        )
        return [p.node_id for p in sorted_peers[: self.num_aggregators]]


def create_cluster_info(
    peers: list[PeerInfo],
    model_layers: int,
) -> ClusterInfo:
    """Create cluster info from peer list."""
    total_vram = sum(p.vram_gb for p in peers)
    return ClusterInfo(
        nodes=peers,
        total_vram=total_vram,
        model_layers=model_layers,
    )
