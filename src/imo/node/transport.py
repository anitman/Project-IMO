"""TLS-encrypted transport for node-to-node gradient communication.

All gradient data between nodes is encrypted in transit using mutual TLS (mTLS).
Each node generates a self-signed TLS certificate from its Ed25519 identity,
and both sides verify each other's certificate during the handshake.

This prevents:
- Eavesdropping: gradients (which encode model weights) cannot be sniffed
- Man-in-the-middle: attacker cannot intercept and modify gradients in transit
- Impersonation at transport level: only nodes with valid keys can connect
"""

from __future__ import annotations

import json
import logging
import ssl
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.x509.oid import NameOID

from imo.node.auth import NodeIdentity

logger = logging.getLogger(__name__)

# Max message size: 256 MB (enough for large gradient payloads)
MAX_MESSAGE_SIZE = 256 * 1024 * 1024
# 4-byte length header
HEADER_SIZE = 4


@dataclass
class TLSConfig:
    """Configuration for TLS transport."""

    cert_dir: str = ".imo/certs"
    verify_peer: bool = True
    min_tls_version: str = "TLSv1.3"
    allowed_node_ids: set[str] | None = None  # None = accept any authenticated node


class NodeCertificate:
    """Generate and manage a node's TLS certificate from its Ed25519 identity.

    The certificate is self-signed using the node's Ed25519 private key.
    The Subject CN is set to the node_id so the peer can verify identity
    after the TLS handshake.
    """

    def __init__(self, identity: NodeIdentity, valid_days: int = 365):
        self.identity = identity
        self._cert, self._key = self._generate(valid_days)

    def _generate(
        self, valid_days: int
    ) -> tuple[x509.Certificate, Ed25519PrivateKey]:
        """Generate a self-signed X.509 certificate."""
        import datetime

        private_key = self.identity._private_key  # noqa: SLF001
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, self.identity.node_id),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "IMO Training Network"),
        ])

        now = datetime.datetime.now(datetime.timezone.utc)
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(self.identity.public_key)
            .serial_number(x509.random_serial_number())
            .not_valid_before(now)
            .not_valid_after(now + datetime.timedelta(days=valid_days))
            .sign(private_key, algorithm=None)
        )

        return cert, private_key

    @property
    def cert_pem(self) -> bytes:
        """Certificate in PEM format."""
        return self._cert.public_bytes(serialization.Encoding.PEM)

    @property
    def key_pem(self) -> bytes:
        """Private key in PEM format."""
        return self._key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )

    @property
    def node_id(self) -> str:
        return self.identity.node_id

    def save(self, cert_dir: str | Path) -> tuple[Path, Path]:
        """Save cert and key to disk."""
        path = Path(cert_dir)
        path.mkdir(parents=True, exist_ok=True)

        cert_path = path / f"{self.node_id}.crt"
        key_path = path / f"{self.node_id}.key"

        cert_path.write_bytes(self.cert_pem)
        key_path.write_bytes(self.key_pem)

        # Restrict key permissions
        key_path.chmod(0o600)

        return cert_path, key_path


class TLSTransport:
    """Encrypted transport layer for node-to-node communication.

    Provides mutual TLS: both connecting and accepting sides present
    certificates and verify each other. After handshake, the peer's
    node_id is extracted from their certificate's CN field.

    Usage:
        # Server side
        transport = TLSTransport(identity, config)
        ssl_ctx = transport.create_server_context()
        # ... use ssl_ctx with asyncio.start_tls_server()

        # Client side
        transport = TLSTransport(identity, config)
        ssl_ctx = transport.create_client_context(peer_cert_pem)
        # ... use ssl_ctx with asyncio.open_connection(ssl=ssl_ctx)
    """

    def __init__(
        self,
        identity: NodeIdentity,
        config: TLSConfig | None = None,
    ):
        self.identity = identity
        self.config = config or TLSConfig()
        self.certificate = NodeCertificate(identity)

        # Trusted peer certs: node_id -> PEM bytes
        self._trusted_certs: dict[str, bytes] = {}

    def add_trusted_peer(self, node_id: str, cert_pem: bytes) -> None:
        """Add a peer's certificate to the trust store."""
        self._trusted_certs[node_id] = cert_pem
        logger.debug("Added trusted peer cert: %s", node_id[:12])

    def remove_trusted_peer(self, node_id: str) -> None:
        """Remove a peer from the trust store."""
        self._trusted_certs.pop(node_id, None)

    def create_server_context(self) -> ssl.SSLContext:
        """Create an SSL context for the server (accepting connections).

        Requires client certificates (mutual TLS).
        """
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_3

        # Load our certificate and key
        self._load_identity_into_context(ctx)

        # Require client cert
        if self.config.verify_peer:
            ctx.verify_mode = ssl.CERT_REQUIRED
            self._load_trusted_certs(ctx)
        else:
            ctx.verify_mode = ssl.CERT_OPTIONAL

        return ctx

    def create_client_context(self, peer_cert_pem: bytes | None = None) -> ssl.SSLContext:
        """Create an SSL context for connecting to a peer.

        Args:
            peer_cert_pem: If provided, only trust this specific cert.
                          Otherwise trust all certs in the trust store.
        """
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_3
        ctx.check_hostname = False  # We verify node_id from CN instead

        # Load our certificate (for mTLS)
        self._load_identity_into_context(ctx)

        if self.config.verify_peer:
            ctx.verify_mode = ssl.CERT_REQUIRED
            if peer_cert_pem:
                ctx.load_verify_locations(cadata=peer_cert_pem.decode("ascii"))
            else:
                self._load_trusted_certs(ctx)
        else:
            ctx.verify_mode = ssl.CERT_NONE

        return ctx

    def verify_peer_identity(self, ssl_socket: ssl.SSLSocket) -> str:
        """Extract and verify peer's node_id from their TLS certificate.

        Call after handshake to confirm the peer is who they claim to be.
        Returns the peer's node_id.
        """
        peer_cert = ssl_socket.getpeercert()
        if peer_cert is None:
            raise PermissionError("Peer did not present a certificate")

        # Extract CN from subject
        subject_tuples: list[tuple[str, str]] = [
            item[0] for item in peer_cert.get("subject", ())
            if isinstance(item[0], tuple)
        ]
        subject = dict(subject_tuples)
        peer_node_id = subject.get("commonName", "")

        if not peer_node_id:
            raise PermissionError("Peer certificate missing CN (node_id)")

        # Check against allowed list if configured
        if (
            self.config.allowed_node_ids is not None
            and peer_node_id not in self.config.allowed_node_ids
        ):
            raise PermissionError(f"Node {peer_node_id} not in allowed list")

        return peer_node_id

    def _load_identity_into_context(self, ctx: ssl.SSLContext) -> None:
        """Load our cert + key into an SSL context via temp files."""
        import tempfile

        cert_path = Path(tempfile.mktemp(suffix=".crt"))
        key_path = Path(tempfile.mktemp(suffix=".key"))
        try:
            cert_path.write_bytes(self.certificate.cert_pem)
            key_path.write_bytes(self.certificate.key_pem)
            key_path.chmod(0o600)
            ctx.load_cert_chain(str(cert_path), str(key_path))
        finally:
            cert_path.unlink(missing_ok=True)
            key_path.unlink(missing_ok=True)

    def _load_trusted_certs(self, ctx: ssl.SSLContext) -> None:
        """Load all trusted peer certs into the context's CA store."""
        for node_id, cert_pem in self._trusted_certs.items():
            try:
                ctx.load_verify_locations(cadata=cert_pem.decode("ascii"))
            except ssl.SSLError:
                logger.warning("Failed to load cert for peer %s", node_id[:12])

    @staticmethod
    def serialize_message(data: dict[str, Any]) -> bytes:
        """Serialize data with a 4-byte length header for framed transport.

        Uses JSON for safe serialization — pickle is NOT used because it
        allows arbitrary code execution on deserialization (RCE).
        """
        payload = json.dumps(data, separators=(",", ":")).encode("utf-8")
        if len(payload) > MAX_MESSAGE_SIZE:
            raise ValueError(f"Message too large: {len(payload)} > {MAX_MESSAGE_SIZE}")
        header = struct.pack("!I", len(payload))
        return header + payload

    @staticmethod
    def deserialize_message(raw: bytes) -> dict[str, Any]:
        """Deserialize a framed message (header + payload).

        Uses JSON for safe deserialization — no arbitrary code execution.
        """
        if len(raw) < HEADER_SIZE:
            raise ValueError("Message too short")
        (length,) = struct.unpack("!I", raw[:HEADER_SIZE])
        if length > MAX_MESSAGE_SIZE:
            raise ValueError(f"Message too large: {length}")
        payload = raw[HEADER_SIZE : HEADER_SIZE + length]
        result: dict[str, Any] = json.loads(payload.decode("utf-8"))
        return result

    @staticmethod
    async def read_framed(reader: Any) -> bytes:
        """Read a length-framed message from an asyncio StreamReader."""
        header = await reader.readexactly(HEADER_SIZE)
        (length,) = struct.unpack("!I", header)
        if length > MAX_MESSAGE_SIZE:
            raise ValueError(f"Message too large: {length}")
        payload: bytes = await reader.readexactly(length)
        return payload

    @staticmethod
    async def write_framed(writer: Any, data: bytes) -> None:
        """Write a length-framed message to an asyncio StreamWriter."""
        header = struct.pack("!I", len(data))
        writer.write(header + data)
        await writer.drain()
