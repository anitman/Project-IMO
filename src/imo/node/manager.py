"""Node lifecycle manager — the single control plane for managed training nodes.

Orchestrates the full lifecycle:

    Recruitment → Registration → Authentication → TLS Channel → Training → Monitoring → Eviction

All nodes participating in a training project go through this manager.
No node can submit gradients, receive model updates, or access checkpoints
without being registered, authenticated, and holding a valid session.

Integrates:
- TrainingRoom: identity verification + admission control
- TLSTransport: encrypted communication channels
- Heartbeat monitoring: detect and evict dead nodes
- Security feedback: auto-ban nodes flagged by PoisoningDetector/RedundantVerifier
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from imo.node.auth import (
    AdmissionConfig,
    NodeIdentity,
    NodeRegistration,
    TrainingRoom,
)
from imo.node.transport import NodeCertificate, TLSConfig, TLSTransport

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Lifecycle status of a managed node."""

    RECRUITED = "recruited"          # Invited / discovered, not yet registered
    REGISTERED = "registered"        # Identity verified, admission approved
    AUTHENTICATED = "authenticated"  # Challenge-response passed, token issued
    CONNECTED = "connected"          # TLS channel established, ready to train
    TRAINING = "training"            # Actively participating in training steps
    SUSPENDED = "suspended"          # Temporarily paused (e.g. low bandwidth)
    EVICTED = "evicted"              # Removed for cause (poisoning, timeout, etc.)
    LEFT = "left"                    # Voluntarily departed


@dataclass
class ManagedNode:
    """Full state for a node under management."""

    node_id: str
    identity: NodeIdentity | None = None
    status: NodeStatus = NodeStatus.RECRUITED
    registration: NodeRegistration | None = None
    auth_token: str = ""
    tls_transport: TLSTransport | None = None
    certificate: NodeCertificate | None = None

    # Performance tracking
    steps_completed: int = 0
    total_compute_time: float = 0.0
    avg_gradient_norm: float = 0.0
    security_strikes: int = 0

    # Heartbeat
    last_heartbeat: float = 0.0
    last_gradient_at: float = 0.0

    # Timestamps
    recruited_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    connected_at: str = ""
    evicted_at: str = ""
    eviction_reason: str = ""


class NodeManager:
    """Central control plane for all nodes in a training project.

    This is the "admin dashboard" for the training swarm. It answers:
    - Who is in the room?
    - Are they authenticated?
    - Is their channel encrypted?
    - Are they behaving well?
    - Should they be kicked?

    Typical flow:
        manager = NodeManager(project_id="open-7b")

        # Recruit and onboard a node
        managed = manager.recruit(identity, vram_gb=24)
        managed = manager.authenticate(node_id)
        managed = manager.connect(node_id)

        # During training
        manager.record_gradient_submission(node_id, step=100)
        manager.heartbeat(node_id)

        # If poisoning detected
        manager.flag_security_issue(node_id, "gradient_poisoning")

        # Periodic maintenance
        await manager.evict_stale_nodes()
    """

    def __init__(
        self,
        project_id: str,
        admission_config: AdmissionConfig | None = None,
        tls_config: TLSConfig | None = None,
        heartbeat_timeout: float = 300.0,
        max_security_strikes: int = 3,
    ):
        self.project_id = project_id
        self.tls_config = tls_config or TLSConfig()
        self.heartbeat_timeout = heartbeat_timeout
        self.max_security_strikes = max_security_strikes

        # Core components
        self.room = TrainingRoom(
            project_id=project_id,
            config=admission_config,
        )

        # Managed nodes
        self._nodes: dict[str, ManagedNode] = {}

    # ------------------------------------------------------------------
    # Phase 1: Recruitment — discover or invite a node
    # ------------------------------------------------------------------

    def recruit(
        self,
        identity: NodeIdentity,
        vram_gb: int = 0,
        stake: float = 0.0,
    ) -> ManagedNode:
        """Recruit a node: verify identity, check admission, register.

        This is the entry point. A node cannot do anything until recruited.
        """
        node_id = identity.node_id

        if node_id in self._nodes:
            existing = self._nodes[node_id]
            if existing.status in (NodeStatus.EVICTED,):
                raise PermissionError(
                    f"Node {node_id[:12]} was evicted: {existing.eviction_reason}"
                )
            if existing.status not in (NodeStatus.LEFT,):
                raise ValueError(f"Node {node_id[:12]} already in room")

        # Register in TrainingRoom (enforces admission policy)
        registration = self.room.register(identity, vram_gb=vram_gb, stake=stake)

        # Generate TLS certificate from identity
        certificate = NodeCertificate(identity)
        tls_transport = TLSTransport(identity, self.tls_config)

        managed = ManagedNode(
            node_id=node_id,
            identity=identity,
            status=NodeStatus.REGISTERED,
            registration=registration,
            certificate=certificate,
            tls_transport=tls_transport,
        )
        self._nodes[node_id] = managed

        logger.info(
            "Node %s (%s) recruited into project %s [VRAM=%dGB]",
            node_id[:12], identity.node_name, self.project_id, vram_gb,
        )
        return managed

    # ------------------------------------------------------------------
    # Phase 2: Authentication — challenge-response to prove identity
    # ------------------------------------------------------------------

    def authenticate(self, node_id: str) -> ManagedNode:
        """Run challenge-response authentication for a registered node.

        Internally:
        1. Room creates a random nonce challenge
        2. Node signs the nonce with its Ed25519 private key
        3. Room verifies the signature
        4. Auth token issued on success
        """
        managed = self._get_managed(node_id)
        if managed.status not in (NodeStatus.REGISTERED, NodeStatus.AUTHENTICATED):
            raise RuntimeError(
                f"Node {node_id[:12]} in wrong state for auth: {managed.status.value}"
            )

        identity = managed.identity
        if identity is None:
            raise RuntimeError(f"Node {node_id[:12]} has no identity")

        # Challenge-response
        challenge = self.room.create_challenge(node_id)
        signature = identity.sign(challenge.nonce)
        token = self.room.verify_challenge(challenge.challenge_id, node_id, signature)

        managed.auth_token = token
        managed.status = NodeStatus.AUTHENTICATED

        logger.info("Node %s authenticated", node_id[:12])
        return managed

    # ------------------------------------------------------------------
    # Phase 3: Connection — establish TLS channel
    # ------------------------------------------------------------------

    def connect(self, node_id: str) -> ManagedNode:
        """Mark a node as connected (TLS channel established).

        In a real deployment, this is called after the TLS handshake
        completes. The node's certificate has been verified by TLSTransport,
        and a secure bidirectional channel is now open.
        """
        managed = self._get_managed(node_id)
        if managed.status != NodeStatus.AUTHENTICATED:
            raise RuntimeError(
                f"Node {node_id[:12]} must authenticate before connecting"
            )

        # Cross-register TLS certs with all existing connected nodes
        if managed.certificate is not None:
            for other_id, other in self._nodes.items():
                if (
                    other_id != node_id
                    and other.status in (NodeStatus.CONNECTED, NodeStatus.TRAINING)
                    and other.tls_transport is not None
                    and managed.tls_transport is not None
                ):
                    # Each side trusts the other's cert
                    other.tls_transport.add_trusted_peer(
                        node_id, managed.certificate.cert_pem
                    )
                    if other.certificate is not None:
                        managed.tls_transport.add_trusted_peer(
                            other_id, other.certificate.cert_pem
                        )

        managed.status = NodeStatus.CONNECTED
        managed.connected_at = datetime.now(timezone.utc).isoformat()
        managed.last_heartbeat = time.time()

        logger.info("Node %s connected via TLS", node_id[:12])
        return managed

    # ------------------------------------------------------------------
    # Phase 4: Training — node actively participates
    # ------------------------------------------------------------------

    def start_training(self, node_id: str) -> ManagedNode:
        """Mark a node as actively training."""
        managed = self._get_managed(node_id)
        if managed.status != NodeStatus.CONNECTED:
            raise RuntimeError(f"Node {node_id[:12]} not connected")
        managed.status = NodeStatus.TRAINING
        return managed

    def record_gradient_submission(
        self,
        node_id: str,
        step: int,
        gradient_norm: float = 0.0,
        compute_time: float = 0.0,
    ) -> None:
        """Record that a node submitted gradients for a training step."""
        managed = self._get_managed(node_id)
        if managed.status != NodeStatus.TRAINING:
            raise PermissionError(f"Node {node_id[:12]} not in training state")

        # Validate auth token is still good
        self.room.validate_token(managed.auth_token)

        managed.steps_completed = step
        managed.total_compute_time += compute_time
        managed.last_gradient_at = time.time()
        managed.last_heartbeat = time.time()

        # Running average of gradient norms (for anomaly baseline)
        if managed.avg_gradient_norm == 0:
            managed.avg_gradient_norm = gradient_norm
        else:
            managed.avg_gradient_norm = 0.9 * managed.avg_gradient_norm + 0.1 * gradient_norm

    def heartbeat(self, node_id: str) -> None:
        """Update heartbeat for a node."""
        managed = self._get_managed(node_id)
        managed.last_heartbeat = time.time()
        self.room.heartbeat(node_id)

    # ------------------------------------------------------------------
    # Security — flag issues, suspend, evict
    # ------------------------------------------------------------------

    def flag_security_issue(
        self,
        node_id: str,
        reason: str,
        auto_evict: bool = True,
    ) -> ManagedNode:
        """Flag a security issue for a node (from PoisoningDetector, etc.).

        Increments strike count. If max strikes reached and auto_evict is
        True, the node is evicted and banned.
        """
        managed = self._get_managed(node_id)
        managed.security_strikes += 1

        logger.warning(
            "Security flag for node %s: %s (strike %d/%d)",
            node_id[:12], reason, managed.security_strikes, self.max_security_strikes,
        )

        if auto_evict and managed.security_strikes >= self.max_security_strikes:
            return self.evict(node_id, reason=f"Max strikes reached: {reason}")

        return managed

    def suspend(self, node_id: str, reason: str = "") -> ManagedNode:
        """Temporarily suspend a node (can be resumed)."""
        managed = self._get_managed(node_id)
        managed.status = NodeStatus.SUSPENDED
        logger.info("Node %s suspended: %s", node_id[:12], reason)
        return managed

    def resume(self, node_id: str) -> ManagedNode:
        """Resume a suspended node."""
        managed = self._get_managed(node_id)
        if managed.status != NodeStatus.SUSPENDED:
            raise RuntimeError(f"Node {node_id[:12]} not suspended")
        managed.status = NodeStatus.CONNECTED
        managed.last_heartbeat = time.time()
        return managed

    def evict(self, node_id: str, reason: str = "") -> ManagedNode:
        """Evict a node permanently: ban from room, revoke TLS trust."""
        managed = self._get_managed(node_id)

        # Ban in TrainingRoom (revokes auth token)
        try:
            self.room.ban_node(node_id, reason=reason)
        except ValueError:
            pass  # Already removed

        # Remove TLS trust from all peers
        for other_id, other in self._nodes.items():
            if other_id != node_id and other.tls_transport is not None:
                other.tls_transport.remove_trusted_peer(node_id)

        managed.status = NodeStatus.EVICTED
        managed.evicted_at = datetime.now(timezone.utc).isoformat()
        managed.eviction_reason = reason

        logger.warning("Node %s evicted: %s", node_id[:12], reason)
        return managed

    def leave(self, node_id: str) -> None:
        """Handle voluntary departure of a node."""
        managed = self._get_managed(node_id)
        self.room.remove_node(node_id)

        for other_id, other in self._nodes.items():
            if other_id != node_id and other.tls_transport is not None:
                other.tls_transport.remove_trusted_peer(node_id)

        managed.status = NodeStatus.LEFT
        logger.info("Node %s left project %s", node_id[:12], self.project_id)

    # ------------------------------------------------------------------
    # Monitoring — heartbeat checks, stale eviction
    # ------------------------------------------------------------------

    async def evict_stale_nodes(self) -> list[str]:
        """Evict nodes that missed heartbeat deadline.

        Call this periodically (e.g. every 60s) to clean up dead nodes.
        """
        now = time.time()
        evicted: list[str] = []

        for node_id, managed in self._nodes.items():
            if managed.status not in (NodeStatus.CONNECTED, NodeStatus.TRAINING):
                continue

            elapsed = now - managed.last_heartbeat
            if managed.last_heartbeat > 0 and elapsed > self.heartbeat_timeout:
                self.evict(node_id, reason="Heartbeat timeout")
                evicted.append(node_id)

        if evicted:
            logger.info(
                "Evicted %d stale nodes from project %s", len(evicted), self.project_id,
            )

        return evicted

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_node(self, node_id: str) -> ManagedNode | None:
        """Get a managed node by ID."""
        return self._nodes.get(node_id)

    @property
    def active_nodes(self) -> list[ManagedNode]:
        """All nodes currently connected or training."""
        return [
            n for n in self._nodes.values()
            if n.status in (NodeStatus.CONNECTED, NodeStatus.TRAINING)
        ]

    @property
    def training_nodes(self) -> list[ManagedNode]:
        """Nodes actively training."""
        return [n for n in self._nodes.values() if n.status == NodeStatus.TRAINING]

    @property
    def node_count(self) -> int:
        """Number of active (connected + training) nodes."""
        return len(self.active_nodes)

    def get_tls_context_for(self, node_id: str, as_server: bool = True) -> Any:
        """Get the SSL context for a specific node.

        Used by the communication layer to establish encrypted channels.
        """
        managed = self._get_managed(node_id)
        if managed.tls_transport is None:
            raise RuntimeError(f"Node {node_id[:12]} has no TLS transport")

        if as_server:
            return managed.tls_transport.create_server_context()
        return managed.tls_transport.create_client_context()

    def get_all_node_ids(self) -> list[str]:
        """Get node IDs of all connected/training nodes."""
        return [n.node_id for n in self.active_nodes]

    def summary(self) -> dict[str, Any]:
        """Get a summary of the room state."""
        status_counts: dict[str, int] = {}
        for n in self._nodes.values():
            status_counts[n.status.value] = status_counts.get(n.status.value, 0) + 1

        return {
            "project_id": self.project_id,
            "total_nodes": len(self._nodes),
            "active_nodes": self.node_count,
            "training_nodes": len(self.training_nodes),
            "status_counts": status_counts,
            "admission_policy": self.room.config.policy.value,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_managed(self, node_id: str) -> ManagedNode:
        """Get a managed node, raising if not found."""
        managed = self._nodes.get(node_id)
        if managed is None:
            raise ValueError(f"Node {node_id[:12]} not found in project {self.project_id}")
        return managed
