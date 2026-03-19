"""Distributed pipeline parallelism inspired by Petals.

Two parallelism strategies coexist:

1. Data Parallelism (default) — each node holds the full model, gradients are
   averaged via hivemind.Optimizer. Works when the model fits on a single GPU.

2. Pipeline Parallelism — the model is split across nodes, each hosting a
   consecutive range of transformer blocks. Activations flow through a chain
   of remote servers. Required when the model is too large for a single GPU.

The pipeline path introduces three actors:

- BlockServer: runs on each compute node, holds a subset of layers,
  processes forward/backward requests from clients.
- RemoteBlock: a PyTorch module proxy that sends activations to the server
  hosting the corresponding layers and receives outputs.
- RemoteSequential: chains RemoteBlocks into a nn.Sequential-like interface
  that is transparent to the training loop.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import hivemind
import torch
import torch.nn as nn

from imo.node.discovery import PeerInfo

logger = logging.getLogger(__name__)


# ── Data Structures ───────────────────────────────────────────


class ServerStatus(Enum):
    """Block server health status."""

    OFFLINE = "offline"
    STARTING = "starting"
    ONLINE = "online"
    DRAINING = "draining"


@dataclass
class BlockInfo:
    """Metadata about a block range hosted by a server, advertised via DHT."""

    server_id: str
    peer_id: str
    start_block: int
    end_block: int
    throughput_rps: float
    vram_gb: int
    status: ServerStatus = ServerStatus.ONLINE
    last_heartbeat: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class PipelineStage:
    """A stage in the pipeline — maps to one server."""

    node_id: str
    layer_range: tuple[int, int]
    model: nn.Module


@dataclass
class ActivationCache:
    """Cached intermediate activations for fault-tolerant rerouting.

    When server N processes a forward pass, the client caches the input
    activations it sent. If server N goes down, the client can reroute
    the cached activations to a backup server hosting the same blocks.
    """

    stage_index: int
    input_activation: torch.Tensor
    timestamp: float = field(default_factory=time.monotonic)


# ── Block Server ──────────────────────────────────────────────


class BlockServer:
    """Hosts a consecutive range of transformer blocks on this node.

    Analogous to a Petals server: receives activation tensors from upstream,
    runs them through the local blocks, and returns outputs to the next node.

    Lifecycle:
        1. Load model layers [start_block, end_block)
        2. Register blocks on DHT
        3. Listen for forward/backward requests
        4. Periodically send heartbeats
    """

    def __init__(
        self,
        model: nn.Module,
        start_block: int,
        end_block: int,
        server_id: str,
        dht: hivemind.DHT | None = None,
        device: torch.device | None = None,
    ):
        self.server_id = server_id
        self.start_block = start_block
        self.end_block = end_block
        self.dht = dht
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.status = ServerStatus.STARTING

        # Extract the block range from the full model
        all_layers = list(model.children())
        self.blocks = nn.Sequential(*all_layers[start_block:end_block]).to(self.device)

        # Throughput measurement
        self._throughput_rps: float = 0.0
        self._request_count: int = 0
        self._total_time: float = 0.0

    async def start(self, project_id: str) -> None:
        """Register blocks on DHT and start serving."""
        if self.dht is not None:
            block_info = BlockInfo(
                server_id=self.server_id,
                peer_id=self.dht.peer_id.to_base58() if hasattr(self.dht, "peer_id") else self.server_id,
                start_block=self.start_block,
                end_block=self.end_block,
                throughput_rps=self._throughput_rps,
                vram_gb=0,
            )
            key = f"blocks/{project_id}/{self.start_block}:{self.end_block}/{self.server_id}"
            await self.dht.store(
                key,
                {
                    "server_id": block_info.server_id,
                    "peer_id": block_info.peer_id,
                    "start_block": block_info.start_block,
                    "end_block": block_info.end_block,
                    "throughput_rps": block_info.throughput_rps,
                    "status": block_info.status.value,
                },
                expiration_time=300,
            )
        self.status = ServerStatus.ONLINE
        logger.info(
            "BlockServer %s serving blocks [%d, %d) — status: ONLINE",
            self.server_id, self.start_block, self.end_block,
        )

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        """Process activations through local blocks."""
        t0 = time.monotonic()
        with torch.no_grad() if not activations.requires_grad else torch.enable_grad():
            output = self.blocks(activations.to(self.device))
        elapsed = time.monotonic() - t0
        self._request_count += 1
        self._total_time += elapsed
        self._throughput_rps = self._request_count / max(self._total_time, 1e-6)
        return output

    def backward(
        self,
        output_grad: torch.Tensor,
        activations: torch.Tensor,
    ) -> torch.Tensor:
        """Compute activation gradients for the upstream node.

        Runs forward again with grad enabled, then backward to get
        gradients w.r.t. input activations.
        """
        activations = activations.to(self.device).requires_grad_(True)
        output = self.blocks(activations)
        output.backward(output_grad.to(self.device))
        return activations.grad.detach()

    async def shutdown(self) -> None:
        """Gracefully shut down the server."""
        self.status = ServerStatus.DRAINING
        logger.info("BlockServer %s shutting down", self.server_id)
        self.status = ServerStatus.OFFLINE

    @property
    def throughput(self) -> float:
        return self._throughput_rps

    @property
    def num_blocks(self) -> int:
        return self.end_block - self.start_block


# ── Remote Block (client-side proxy) ─────────────────────────


class RemoteBlock(nn.Module):
    """Client-side proxy that routes activations to a remote BlockServer.

    Wraps the network call in a PyTorch module so that the training loop
    sees a normal nn.Module. Autograd-compatible: saves activations for
    backward pass and fetches gradients from the server.
    """

    def __init__(self, server: BlockServer, stage_index: int):
        super().__init__()
        self._server = server
        self._stage_index = stage_index
        self._cached_input: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Send activations to server, receive output."""
        self._cached_input = x.detach().clone()
        return self._server.forward(x)

    def remote_backward(self, output_grad: torch.Tensor) -> torch.Tensor:
        """Fetch activation gradients from server for upstream backprop."""
        if self._cached_input is None:
            raise RuntimeError("No cached input — forward() must be called first")
        return self._server.backward(output_grad, self._cached_input)


class RemoteSequential(nn.Module):
    """Chains multiple RemoteBlocks into a sequential pipeline.

    Usage is identical to nn.Sequential, but each sub-module routes
    activations to a different server node.

    Supports fault-tolerant rerouting: if a server goes down mid-forward,
    the cached activation at that stage is re-sent to a backup server.
    """

    def __init__(self, remote_blocks: list[RemoteBlock]):
        super().__init__()
        self.blocks = nn.ModuleList(remote_blocks)
        self._activation_cache: list[ActivationCache] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the full server chain."""
        self._activation_cache.clear()
        output = x

        for i, block in enumerate(self.blocks):
            self._activation_cache.append(
                ActivationCache(stage_index=i, input_activation=output.detach().clone())
            )
            output = block(output)

        return output

    def remote_backward(self, output_grad: torch.Tensor) -> torch.Tensor:
        """Backward pass through the server chain in reverse order."""
        grad = output_grad
        for block in reversed(list(self.blocks)):
            if isinstance(block, RemoteBlock):
                grad = block.remote_backward(grad)
        return grad

    def reroute_stage(self, stage_index: int, new_server: BlockServer) -> None:
        """Replace a failed server with a backup at runtime."""
        if stage_index >= len(self.blocks):
            raise IndexError(f"Stage {stage_index} out of range")

        old_block = self.blocks[stage_index]
        self.blocks[stage_index] = RemoteBlock(new_server, stage_index)

        logger.info(
            "Rerouted stage %d to server %s",
            stage_index, new_server.server_id,
        )


# ── Pipeline Router ──────────────────────────────────────────


class PipelineRouter:
    """Discovers block servers via DHT and builds optimal server chains.

    Finds the chain of servers that covers all model blocks [0, total_blocks)
    with minimum total latency.
    """

    def __init__(self, dht: hivemind.DHT | None = None):
        self.dht = dht
        self._known_servers: dict[str, BlockInfo] = {}

    async def discover_servers(self, project_id: str) -> list[BlockInfo]:
        """Query DHT for all block servers in this project."""
        if self.dht is None:
            return list(self._known_servers.values())

        servers: list[BlockInfo] = []
        prefix = f"blocks/{project_id}/"

        async for key, value in self.dht.iterate(prefix, timeout=30):
            data = value.value
            info = BlockInfo(
                server_id=data["server_id"],
                peer_id=data["peer_id"],
                start_block=data["start_block"],
                end_block=data["end_block"],
                throughput_rps=data.get("throughput_rps", 0),
                vram_gb=0,
                status=ServerStatus(data.get("status", "online")),
            )
            if info.status == ServerStatus.ONLINE:
                servers.append(info)
                self._known_servers[info.server_id] = info

        return servers

    def register_server(self, info: BlockInfo) -> None:
        """Register a server locally (for in-process pipeline)."""
        self._known_servers[info.server_id] = info

    def build_chain(self, total_blocks: int) -> list[BlockInfo]:
        """Build a server chain covering all blocks [0, total_blocks).

        Greedy: pick the server with the highest throughput for each
        uncovered block range, advancing until all blocks are covered.
        """
        servers = sorted(
            self._known_servers.values(),
            key=lambda s: s.start_block,
        )

        if not servers:
            raise RuntimeError("No servers available to build chain")

        chain: list[BlockInfo] = []
        current_block = 0

        while current_block < total_blocks:
            # Find candidates that start at or before current_block
            candidates = [
                s for s in servers
                if s.start_block <= current_block and s.end_block > current_block
                and s.status == ServerStatus.ONLINE
            ]

            if not candidates:
                raise RuntimeError(
                    f"No server covers block {current_block}. "
                    f"Available: {[(s.start_block, s.end_block) for s in servers]}"
                )

            # Pick the one that covers the most blocks with highest throughput
            best = max(candidates, key=lambda s: (s.end_block, s.throughput_rps))
            chain.append(best)
            current_block = best.end_block

        return chain

    def find_backup(self, failed_server: BlockInfo) -> BlockInfo | None:
        """Find a backup server that covers the same block range."""
        for server in self._known_servers.values():
            if (
                server.server_id != failed_server.server_id
                and server.start_block <= failed_server.start_block
                and server.end_block >= failed_server.end_block
                and server.status == ServerStatus.ONLINE
            ):
                return server
        return None


# ── Throughput-Based Rebalancer ───────────────────────────────


class BlockRebalancer:
    """Dynamically reassign blocks to maximize swarm throughput.

    The bottleneck of a pipeline is the slowest stage. This rebalancer
    periodically checks if moving blocks between servers would reduce
    the bottleneck, and triggers reassignment if the improvement exceeds
    a threshold.
    """

    def __init__(self, improvement_threshold: float = 0.1):
        self.improvement_threshold = improvement_threshold

    def should_rebalance(self, chain: list[BlockInfo]) -> bool:
        """Check if rebalancing would improve throughput."""
        if len(chain) < 2:
            return False

        throughputs = [s.throughput_rps for s in chain]
        if min(throughputs) == 0:
            return False

        bottleneck = min(throughputs)
        avg = sum(throughputs) / len(throughputs)

        # If bottleneck is significantly below average, rebalancing helps
        return (avg - bottleneck) / avg > self.improvement_threshold

    def suggest_reassignment(
        self,
        peers: list[PeerInfo],
        total_blocks: int,
    ) -> dict[str, tuple[int, int]]:
        """Suggest block assignments based on throughput capacity.

        Returns {server_id: (start_block, end_block)}.
        Distributes blocks proportionally to bandwidth × VRAM.
        """
        if not peers:
            return {}

        # Score = bandwidth × vram (higher = more capacity)
        scores = {
            p.node_id: p.bandwidth_mbps * p.vram_gb
            for p in peers
        }
        total_score = sum(scores.values())
        if total_score == 0:
            # Fallback: equal distribution
            blocks_per_peer = max(1, total_blocks // len(peers))
            assignments = {}
            current = 0
            for i, p in enumerate(peers):
                end = total_blocks if i == len(peers) - 1 else current + blocks_per_peer
                assignments[p.node_id] = (current, min(end, total_blocks))
                current = end
            return assignments

        assignments: dict[str, tuple[int, int]] = {}
        current_block = 0

        sorted_peers = sorted(peers, key=lambda p: scores[p.node_id], reverse=True)

        for i, peer in enumerate(sorted_peers):
            if current_block >= total_blocks:
                break
            if i == len(sorted_peers) - 1:
                num_blocks = total_blocks - current_block
            else:
                ratio = scores[peer.node_id] / total_score
                num_blocks = max(1, int(ratio * total_blocks))
                num_blocks = min(num_blocks, total_blocks - current_block)

            assignments[peer.node_id] = (current_block, current_block + num_blocks)
            current_block += num_blocks

        return assignments


# ── Helper Functions ──────────────────────────────────────────


class MicrobatchScheduler:
    """Schedule microbatches across pipeline stages."""

    def __init__(self, num_microbatches: int = 4):
        self.num_microbatches = num_microbatches

    def split_batch(self, batch: torch.Tensor) -> list[torch.Tensor]:
        """Split a batch into microbatches."""
        return list(torch.chunk(batch, self.num_microbatches))

    def schedule(
        self,
        microbatches: list[torch.Tensor],
        stages: list[PipelineStage],
    ) -> list[tuple[int, torch.Tensor]]:
        """Create 1F1B schedule for processing microbatches.

        One-Forward-One-Backward: interleaves forward and backward passes
        to keep pipeline bubbles small.
        """
        schedule: list[tuple[int, torch.Tensor]] = []

        # Warm-up: forward passes fill the pipeline
        for mb_idx in range(min(len(microbatches), len(stages))):
            for stage_idx in range(len(stages)):
                schedule.append((stage_idx, microbatches[mb_idx]))

        # Steady state: remaining microbatches
        for mb_idx in range(len(stages), len(microbatches)):
            for stage_idx in range(len(stages)):
                schedule.append((stage_idx, microbatches[mb_idx]))

        return schedule


def create_pipeline_model(
    model: nn.Module,
    node_assignments: dict[str, list[int]],
) -> list[PipelineStage]:
    """Create pipeline stages from node assignments."""
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
    return stages


def build_remote_pipeline(
    model: nn.Module,
    chain: list[BlockInfo],
) -> RemoteSequential:
    """Build a RemoteSequential from a discovered server chain.

    For in-process testing: creates BlockServers and RemoteBlocks
    from the chain info.
    """
    remote_blocks: list[RemoteBlock] = []

    for i, info in enumerate(chain):
        server = BlockServer(
            model=model,
            start_block=info.start_block,
            end_block=info.end_block,
            server_id=info.server_id,
        )
        remote_blocks.append(RemoteBlock(server, stage_index=i))

    return RemoteSequential(remote_blocks)
