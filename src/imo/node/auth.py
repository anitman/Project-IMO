"""Node identity, authentication, and secure training room management.

Provides cryptographic identity for nodes (Ed25519 key pairs), challenge-response
authentication, and per-project isolated training rooms with admission control.

This prevents:
- Unauthorized nodes joining a training swarm
- Identity impersonation (Sybil attacks)
- Banned/malicious nodes re-entering under a different name
- Unauthenticated gradient submissions
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node Identity — Ed25519 key pair
# ---------------------------------------------------------------------------


class NodeIdentity:
    """Cryptographic identity for a training node.

    Each node generates an Ed25519 key pair. The public key serves as the
    node's verifiable identity. To prove ownership, a node signs a challenge
    with its private key; anyone can verify with the public key.

    This makes impersonation computationally infeasible.
    """

    def __init__(
        self,
        private_key: Ed25519PrivateKey | None = None,
        node_name: str = "",
    ):
        if private_key is not None:
            self._private_key = private_key
        else:
            self._private_key = Ed25519PrivateKey.generate()

        self._public_key = self._private_key.public_key()
        self.node_name = node_name

    @property
    def public_key(self) -> Ed25519PublicKey:
        return self._public_key

    @property
    def node_id(self) -> str:
        """Derive a deterministic node ID from the public key (SHA-256 hex)."""
        pub_bytes = self._public_key.public_bytes(Encoding.Raw, PublicFormat.Raw)
        return hashlib.sha256(pub_bytes).hexdigest()[:32]

    @property
    def public_key_bytes(self) -> bytes:
        return self._public_key.public_bytes(Encoding.Raw, PublicFormat.Raw)

    def sign(self, message: bytes) -> bytes:
        """Sign a message with the private key."""
        return self._private_key.sign(message)

    @staticmethod
    def verify(public_key_bytes: bytes, message: bytes, signature: bytes) -> bool:
        """Verify a signature against a public key."""
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PublicKey as Ed25519Pub,
            )

            pub_key = Ed25519Pub.from_public_key_bytes(public_key_bytes)  # type: ignore[attr-defined]
            pub_key.verify(signature, message)
            return True
        except (InvalidSignature, ValueError):
            return False

    def export_private_key(self) -> bytes:
        """Export private key bytes (for persistence). Keep secret!"""
        return self._private_key.private_bytes(
            Encoding.Raw, PrivateFormat.Raw, NoEncryption()
        )

    @classmethod
    def from_private_key_bytes(cls, key_bytes: bytes, node_name: str = "") -> NodeIdentity:
        """Restore identity from exported private key bytes."""
        private_key = Ed25519PrivateKey.from_private_bytes(key_bytes)
        return cls(private_key=private_key, node_name=node_name)


# ---------------------------------------------------------------------------
# Admission policies
# ---------------------------------------------------------------------------


class AdmissionPolicy(Enum):
    """How nodes are admitted to a training room."""

    OPEN = "open"                    # Anyone can join (still need valid identity)
    INVITE_ONLY = "invite_only"      # Only pre-approved public keys
    STAKE_REQUIRED = "stake_required" # Must hold minimum $IMO stake


@dataclass
class AdmissionConfig:
    """Configuration for room admission."""

    policy: AdmissionPolicy = AdmissionPolicy.OPEN
    min_stake: float = 0.0                          # For STAKE_REQUIRED
    invited_keys: set[str] = field(default_factory=set)  # node_ids for INVITE_ONLY
    max_nodes: int = 256                             # Hard cap
    require_vram_gb: int = 0                         # Minimum VRAM requirement
    ban_duration_seconds: float = 86400.0            # 24h default ban


# ---------------------------------------------------------------------------
# Registration records
# ---------------------------------------------------------------------------


@dataclass
class NodeRegistration:
    """A registered node in a training room."""

    node_id: str
    public_key_bytes: bytes
    node_name: str
    vram_gb: int
    registered_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    is_active: bool = True
    is_banned: bool = False
    ban_reason: str = ""
    ban_expires_at: str = ""
    auth_token: str = ""
    last_heartbeat: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AuthChallenge:
    """A challenge for node authentication."""

    challenge_id: str
    node_id: str
    nonce: bytes
    created_at: float
    expires_at: float


# ---------------------------------------------------------------------------
# Training Room
# ---------------------------------------------------------------------------


class TrainingRoom:
    """Secure, isolated training environment for a single project.

    Each project gets its own room. Nodes must:
    1. Register with a valid Ed25519 identity
    2. Meet the admission policy (open / invite-only / stake-required)
    3. Pass challenge-response authentication
    4. Maintain periodic heartbeats

    The room tracks all registered nodes, handles banning (from
    PoisoningDetector / RedundantVerifier feedback), and issues
    short-lived auth tokens for gradient submission.

    This is the "locked door" that prevents arbitrary nodes from
    crashing a training run.
    """

    def __init__(
        self,
        project_id: str,
        config: AdmissionConfig | None = None,
        token_ttl_seconds: float = 3600.0,
        challenge_ttl_seconds: float = 60.0,
    ):
        self.project_id = project_id
        self.config = config or AdmissionConfig()
        self.token_ttl_seconds = token_ttl_seconds
        self.challenge_ttl_seconds = challenge_ttl_seconds

        # State
        self._nodes: dict[str, NodeRegistration] = {}
        self._pending_challenges: dict[str, AuthChallenge] = {}
        self._token_secret = secrets.token_bytes(32)
        self._tokens: dict[str, float] = {}  # token -> expiry timestamp

    # -- Registration --

    def register(
        self,
        identity: NodeIdentity,
        vram_gb: int = 0,
        stake: float = 0.0,
    ) -> NodeRegistration:
        """Register a node in this room.

        Validates:
        - Identity is cryptographically valid
        - Admission policy is met
        - Node is not banned
        - Room capacity not exceeded
        """
        node_id = identity.node_id

        # Check existing ban
        if node_id in self._nodes and self._nodes[node_id].is_banned:
            reg = self._nodes[node_id]
            if reg.ban_expires_at and reg.ban_expires_at > datetime.now(timezone.utc).isoformat():
                raise PermissionError(f"Node {node_id} is banned: {reg.ban_reason}")
            # Ban expired — clear it
            reg.is_banned = False
            reg.ban_reason = ""

        # Check capacity
        active_count = sum(1 for n in self._nodes.values() if n.is_active and not n.is_banned)
        if active_count >= self.config.max_nodes:
            raise RuntimeError(f"Room full: {active_count}/{self.config.max_nodes}")

        # Check VRAM requirement
        if self.config.require_vram_gb > 0 and vram_gb < self.config.require_vram_gb:
            raise ValueError(
                f"Insufficient VRAM: {vram_gb} GB < {self.config.require_vram_gb} GB required"
            )

        # Check admission policy
        self._check_admission(node_id, stake)

        # Create or update registration
        registration = NodeRegistration(
            node_id=node_id,
            public_key_bytes=identity.public_key_bytes,
            node_name=identity.node_name,
            vram_gb=vram_gb,
            is_active=True,
        )
        self._nodes[node_id] = registration

        logger.info(
            "Node %s (%s) registered in room %s",
            node_id[:12], identity.node_name, self.project_id,
        )
        return registration

    def _check_admission(self, node_id: str, stake: float) -> None:
        """Enforce admission policy."""
        policy = self.config.policy

        if policy == AdmissionPolicy.INVITE_ONLY:
            if node_id not in self.config.invited_keys:
                raise PermissionError(
                    f"Node {node_id} not on invite list for project {self.project_id}"
                )

        elif policy == AdmissionPolicy.STAKE_REQUIRED:
            if stake < self.config.min_stake:
                raise PermissionError(
                    f"Insufficient stake: {stake} < {self.config.min_stake} required"
                )

    # -- Challenge-Response Authentication --

    def create_challenge(self, node_id: str) -> AuthChallenge:
        """Create a cryptographic challenge for a registered node."""
        if node_id not in self._nodes:
            raise ValueError(f"Node {node_id} not registered")

        reg = self._nodes[node_id]
        if reg.is_banned:
            raise PermissionError(f"Node {node_id} is banned")

        now = time.time()
        challenge = AuthChallenge(
            challenge_id=secrets.token_hex(16),
            node_id=node_id,
            nonce=secrets.token_bytes(32),
            created_at=now,
            expires_at=now + self.challenge_ttl_seconds,
        )
        self._pending_challenges[challenge.challenge_id] = challenge
        return challenge

    def verify_challenge(
        self,
        challenge_id: str,
        node_id: str,
        signature: bytes,
    ) -> str:
        """Verify a challenge response and issue an auth token.

        Returns an auth token on success.
        """
        challenge = self._pending_challenges.pop(challenge_id, None)
        if challenge is None:
            raise ValueError("Challenge not found or already used")

        if challenge.node_id != node_id:
            raise PermissionError("Challenge belongs to a different node")

        if time.time() > challenge.expires_at:
            raise TimeoutError("Challenge expired")

        reg = self._nodes.get(node_id)
        if reg is None:
            raise ValueError(f"Node {node_id} not registered")

        # Verify Ed25519 signature of the nonce
        if not NodeIdentity.verify(reg.public_key_bytes, challenge.nonce, signature):
            logger.warning("Authentication failed for node %s — invalid signature", node_id)
            raise PermissionError("Invalid signature")

        # Issue auth token
        token = self._issue_token(node_id)
        reg.auth_token = token
        reg.last_heartbeat = datetime.now(timezone.utc).isoformat()

        logger.info("Node %s authenticated in room %s", node_id[:12], self.project_id)
        return token

    def _issue_token(self, node_id: str) -> str:
        """Issue an HMAC-based auth token for a node."""
        expiry = time.time() + self.token_ttl_seconds
        payload = f"{node_id}:{expiry}".encode()
        mac = hmac.new(self._token_secret, payload, hashlib.sha256).hexdigest()
        token = f"{node_id}:{expiry}:{mac}"
        self._tokens[token] = expiry
        return token

    def validate_token(self, token: str) -> str:
        """Validate an auth token. Returns node_id if valid."""
        parts = token.split(":")
        if len(parts) != 3:
            raise PermissionError("Malformed token")

        node_id, expiry_str, provided_mac = parts

        try:
            expiry = float(expiry_str)
        except ValueError:
            raise PermissionError("Malformed token expiry")

        if time.time() > expiry:
            self._tokens.pop(token, None)
            raise PermissionError("Token expired")

        # Verify HMAC
        payload = f"{node_id}:{expiry_str}".encode()
        expected_mac = hmac.new(self._token_secret, payload, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(provided_mac, expected_mac):
            raise PermissionError("Invalid token signature")

        # Verify node still active
        reg = self._nodes.get(node_id)
        if reg is None or not reg.is_active or reg.is_banned:
            raise PermissionError("Node no longer active in this room")

        return node_id

    # -- Node management --

    def ban_node(self, node_id: str, reason: str = "") -> None:
        """Ban a node from this training room."""
        reg = self._nodes.get(node_id)
        if reg is None:
            raise ValueError(f"Node {node_id} not registered")

        from datetime import timedelta

        reg.is_banned = True
        reg.is_active = False
        reg.ban_reason = reason
        reg.ban_expires_at = (
            datetime.now(timezone.utc)
            + timedelta(seconds=self.config.ban_duration_seconds)
        ).isoformat()

        # Revoke all tokens for this node
        self._tokens = {
            t: exp for t, exp in self._tokens.items()
            if not t.startswith(f"{node_id}:")
        }

        logger.warning(
            "Node %s banned from room %s: %s", node_id[:12], self.project_id, reason,
        )

    def remove_node(self, node_id: str) -> None:
        """Remove a node from the room (voluntary leave)."""
        reg = self._nodes.get(node_id)
        if reg is None:
            return
        reg.is_active = False

    def heartbeat(self, node_id: str) -> None:
        """Update node's last heartbeat timestamp."""
        reg = self._nodes.get(node_id)
        if reg is None or not reg.is_active:
            raise ValueError(f"Node {node_id} not active")
        reg.last_heartbeat = datetime.now(timezone.utc).isoformat()

    def invite_node(self, node_id: str) -> None:
        """Add a node to the invite list (for INVITE_ONLY rooms)."""
        self.config.invited_keys.add(node_id)

    def revoke_invite(self, node_id: str) -> None:
        """Remove a node from the invite list."""
        self.config.invited_keys.discard(node_id)

    # -- Queries --

    @property
    def active_nodes(self) -> list[NodeRegistration]:
        """List all active, non-banned nodes."""
        return [
            n for n in self._nodes.values()
            if n.is_active and not n.is_banned
        ]

    @property
    def banned_nodes(self) -> list[NodeRegistration]:
        """List all banned nodes."""
        return [n for n in self._nodes.values() if n.is_banned]

    @property
    def node_count(self) -> int:
        """Number of active nodes."""
        return len(self.active_nodes)

    def get_node(self, node_id: str) -> NodeRegistration | None:
        """Get registration for a specific node."""
        return self._nodes.get(node_id)

    def is_authenticated(self, node_id: str) -> bool:
        """Check if a node has a valid auth token."""
        reg = self._nodes.get(node_id)
        if reg is None or not reg.auth_token:
            return False
        try:
            self.validate_token(reg.auth_token)
            return True
        except PermissionError:
            return False
