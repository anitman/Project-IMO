"""Peer discovery via Hivemind DHT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import hivemind


@dataclass
class PeerInfo:
    """Information about a peer in the network."""

    node_id: str
    peer_id: str
    vram_gb: int
    bandwidth_mbps: float
    availability_window: float
    supported_dtypes: list[str]
    metadata: dict[str, Any]


class PeerDiscovery:
    """Discover and manage peers via Hivemind DHT."""

    def __init__(
        self,
        initial_peers: list[str] | None = None,
        node_id: str | None = None,
    ):
        self.initial_peers = initial_peers or []
        self.node_id = node_id
        self.dht = None
        self._running = False

    async def start(self) -> None:
        """Start the DHT node."""
        self.dht = await hivemind.DHT.create(
            initial_peers=self.initial_peers,
            start=True,
        )
        self._running = True

    async def stop(self) -> None:
        """Stop the DHT node."""
        self._running = False
        if self.dht:
            await self.dht.shutdown()

    async def advertise(self, info: PeerInfo) -> None:
        """Advertise peer information to the DHT."""
        if not self.dht:
            raise RuntimeError("DHT not started")

        key = f"peer/{info.node_id}"
        data = {
            "peer_id": info.peer_id,
            "vram_gb": info.vram_gb,
            "bandwidth_mbps": info.bandwidth_mbps,
            "availability_window": info.availability_window,
            "supported_dtypes": info.supported_dtypes,
        }

        await self.dht.store(key, data, expiration_time=3600)

    async def find_peer(self, node_id: str) -> PeerInfo | None:
        """Find a peer by node ID."""
        if not self.dht:
            raise RuntimeError("DHT not started")

        key = f"peer/{node_id}"
        result = await self.dht.get(key, timeout=30)

        if result is None:
            return None

        data = result.value
        return PeerInfo(
            node_id=node_id,
            peer_id=data["peer_id"],
            vram_gb=data["vram_gb"],
            bandwidth_mbps=data["bandwidth_mbps"],
            availability_window=data["availability_window"],
            supported_dtypes=data["supported_dtypes"],
            metadata={},
        )

    async def list_peers(self, limit: int = 100) -> list[PeerInfo]:
        """List available peers in the network."""
        if not self.dht:
            raise RuntimeError("DHT not started")

        peers: list[PeerInfo] = []
        prefix = "peer/"

        async for key, value in self.dht.iterate(prefix, timeout=60):
            if len(peers) >= limit:
                break

            data = value.value
            peer = PeerInfo(
                node_id=key.replace(prefix, ""),
                peer_id=data["peer_id"],
                vram_gb=data["vram_gb"],
                bandwidth_mbps=data["bandwidth_mbps"],
                availability_window=data["availability_window"],
                supported_dtypes=data["supported_dtypes"],
                metadata={},
            )
            peers.append(peer)

        return peers

    async def connect(self, peer_id: str) -> hivemind.MPRemote:
        """Connect to a peer."""
        return await hivemind.connect(peer_id)
