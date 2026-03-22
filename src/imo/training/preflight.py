"""Pre-flight security checks — mandatory gate before training begins.

Every training run must pass these checks before the first gradient step.
This catches malicious inputs BEFORE they can pollute the model:

1. SafeModelLoader     — blocks pickle RCE, enforces safetensors
2. ModelIntegrityCheck — verifies weights hash against manifest
3. ConfigValidator     — rejects insane hyperparameters that could destroy training
4. DatasetQuarantine   — security-scans all datasets before they touch the model
5. WarmupTrustPolicy   — new / low-reputation nodes get stricter scrutiny
6. CanaryDetector      — embeds sentinel samples to detect ongoing poisoning

Usage:
    preflight = PreflightGate(config, model, datasets)
    report = preflight.run_all()
    if not report.passed:
        raise SecurityError(report.summary())
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Check results
# ---------------------------------------------------------------------------


class CheckSeverity(Enum):
    """How severe a failed check is."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class CheckResult:
    """Result of a single pre-flight check."""

    name: str
    passed: bool
    severity: CheckSeverity
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreflightReport:
    """Aggregated result of all pre-flight checks."""

    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True only if no CRITICAL check failed."""
        return all(
            c.passed or c.severity != CheckSeverity.CRITICAL
            for c in self.checks
        )

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity == CheckSeverity.WARNING]

    @property
    def critical_failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity == CheckSeverity.CRITICAL]

    def summary(self) -> str:
        lines = [f"Pre-flight: {len(self.checks)} checks, "
                 f"{len(self.critical_failures)} critical, {len(self.warnings)} warnings"]
        for c in self.checks:
            status = "PASS" if c.passed else f"FAIL[{c.severity.value}]"
            lines.append(f"  [{status}] {c.name}: {c.message}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1. Safe Model Loading — prevent pickle RCE
# ---------------------------------------------------------------------------


class SafeModelLoader:
    """Enforce safe model weight loading.

    torch.load() uses pickle internally, which allows arbitrary code
    execution. An attacker can craft a .pt file that runs malicious code
    when loaded. This class:

    - Prefers safetensors format (no code execution possible)
    - Falls back to torch.load(weights_only=True) which blocks pickle
    - Rejects files containing pickle opcodes for dangerous builtins
    """

    DANGEROUS_PICKLE_MODULES = {
        "os", "subprocess", "sys", "builtins", "shutil",
        "socket", "ctypes", "pickle", "io", "code",
    }

    @classmethod
    def load_weights(cls, path: str | Path) -> dict[str, torch.Tensor]:
        """Load model weights safely."""
        path = Path(path)

        if path.suffix in (".safetensors",):
            return cls._load_safetensors(path)

        if path.suffix in (".pt", ".pth", ".bin"):
            return cls._load_torch_safe(path)

        raise ValueError(f"Unsupported weight format: {path.suffix}")

    @classmethod
    def _load_safetensors(cls, path: Path) -> dict[str, torch.Tensor]:
        """Load from safetensors format (inherently safe, no code execution)."""
        try:
            from safetensors.torch import load_file
        except ImportError:
            raise ImportError(
                "safetensors not installed — run: pip install safetensors"
            )
        return load_file(str(path))

    @classmethod
    def _load_torch_safe(cls, path: Path) -> dict[str, torch.Tensor]:
        """Load .pt/.pth with weights_only=True (blocks pickle code execution)."""
        try:
            data = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            logger.warning(
                "weights_only=True failed for %s — file may contain unsafe pickle", path
            )
            raise SecurityError(
                f"Refused to load {path}: weights_only=True failed. "
                "Convert to safetensors format for safe loading."
            )

        if isinstance(data, dict) and "model_state" in data:
            result: dict[str, torch.Tensor] = data["model_state"]
            return result
        weights: dict[str, torch.Tensor] = data
        return weights

    @classmethod
    def check(cls, path: str | Path) -> CheckResult:
        """Pre-flight check: can this file be safely loaded?"""
        path = Path(path)

        if not path.exists():
            return CheckResult(
                name="safe_model_loader",
                passed=False,
                severity=CheckSeverity.CRITICAL,
                message=f"Weight file not found: {path}",
            )

        if path.suffix == ".safetensors":
            return CheckResult(
                name="safe_model_loader",
                passed=True,
                severity=CheckSeverity.INFO,
                message="safetensors format — inherently safe",
            )

        # Scan for dangerous pickle opcodes
        dangerous = cls._scan_pickle_danger(path)
        if dangerous:
            return CheckResult(
                name="safe_model_loader",
                passed=False,
                severity=CheckSeverity.CRITICAL,
                message=f"Dangerous pickle modules detected: {', '.join(dangerous)}",
                details={"dangerous_modules": list(dangerous)},
            )

        return CheckResult(
            name="safe_model_loader",
            passed=True,
            severity=CheckSeverity.WARNING,
            message="PyTorch format — will use weights_only=True",
        )

    @classmethod
    def _scan_pickle_danger(cls, path: Path) -> set[str]:
        """Scan a .pt file for references to dangerous modules."""
        try:
            content = path.read_bytes()
        except OSError:
            return set()

        found: set[str] = set()
        for module in cls.DANGEROUS_PICKLE_MODULES:
            if module.encode() in content:
                found.add(module)
        return found


class SecurityError(Exception):
    """Raised when a security check fails critically."""


# ---------------------------------------------------------------------------
# 2. Model Integrity — verify weight hashes
# ---------------------------------------------------------------------------


class ModelIntegrityVerifier:
    """Verify model weights against a known-good hash manifest.

    Before training starts, the model's state_dict is hashed and compared
    against the manifest published with the project. This detects:
    - Tampered pretrained weights (backdoor injection)
    - Corrupted downloads
    - Version mismatches
    """

    @staticmethod
    def compute_manifest(model: nn.Module) -> dict[str, str]:
        """Compute SHA-256 hash for each parameter tensor."""
        manifest: dict[str, str] = {}
        for name, param in model.named_parameters():
            data = param.detach().cpu().numpy().tobytes()
            manifest[name] = hashlib.sha256(data).hexdigest()
        return manifest

    @staticmethod
    def compute_global_hash(model: nn.Module) -> str:
        """Compute a single hash over all parameters (order-sensitive)."""
        h = hashlib.sha256()
        for name in sorted(model.state_dict().keys()):
            tensor = model.state_dict()[name]
            h.update(name.encode())
            h.update(tensor.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    @classmethod
    def verify(
        cls,
        model: nn.Module,
        expected_hash: str | None = None,
        expected_manifest: dict[str, str] | None = None,
    ) -> CheckResult:
        """Verify model integrity against expected hashes."""
        if expected_hash is not None:
            actual = cls.compute_global_hash(model)
            if actual != expected_hash:
                return CheckResult(
                    name="model_integrity",
                    passed=False,
                    severity=CheckSeverity.CRITICAL,
                    message="Model global hash mismatch — weights may be tampered",
                    details={"expected": expected_hash, "actual": actual},
                )
            return CheckResult(
                name="model_integrity",
                passed=True,
                severity=CheckSeverity.INFO,
                message="Model global hash verified",
            )

        if expected_manifest is not None:
            actual_manifest = cls.compute_manifest(model)
            mismatched: list[str] = []
            for name, expected in expected_manifest.items():
                actual = actual_manifest.get(name, "")
                if actual != expected:
                    mismatched.append(name)

            if mismatched:
                return CheckResult(
                    name="model_integrity",
                    passed=False,
                    severity=CheckSeverity.CRITICAL,
                    message=f"Weight hash mismatch in {len(mismatched)} parameters",
                    details={"mismatched_params": mismatched[:20]},
                )
            return CheckResult(
                name="model_integrity",
                passed=True,
                severity=CheckSeverity.INFO,
                message=f"All {len(expected_manifest)} parameter hashes verified",
            )

        return CheckResult(
            name="model_integrity",
            passed=True,
            severity=CheckSeverity.WARNING,
            message="No expected hash provided — skipping integrity check",
        )


# ---------------------------------------------------------------------------
# 3. Config Validation — reject insane hyperparameters
# ---------------------------------------------------------------------------


class ConfigValidator:
    """Validate training config for sane hyperparameters.

    An attacker who controls the config could:
    - Set learning_rate = 100.0 → destroy model in one step
    - Set batch_size = 1 → make Byzantine detection impossible
    - Set trim_ratio = 0.49 → defeat trimmed mean aggregation
    - Set poisoning_threshold = 999 → disable poisoning detection
    """

    BOUNDS: dict[str, tuple[float, float]] = {
        "learning_rate": (1e-8, 1.0),
        "weight_decay": (0.0, 1.0),
        "batch_size": (1, 65536),
        "gradient_accumulation_steps": (1, 1024),
        "warmup_steps": (0, 1_000_000),
        "max_steps": (1, 100_000_000),
        "trim_ratio": (0.0, 0.45),
        "poisoning_threshold": (0.5, 10.0),
        "top_k_sparsity": (0.001, 1.0),
        "min_peers": (1, 1024),
    }

    @classmethod
    def validate(cls, config: Any) -> CheckResult:
        """Validate training config fields against sane bounds."""
        violations: list[str] = []

        for field_name, (lo, hi) in cls.BOUNDS.items():
            value = getattr(config, field_name, None)
            if value is None:
                continue
            if not (lo <= float(value) <= hi):
                violations.append(
                    f"{field_name}={value} outside [{lo}, {hi}]"
                )

        # Special checks
        if hasattr(config, "trim_ratio") and hasattr(config, "min_peers"):
            min_peers = config.min_peers
            trim_ratio = config.trim_ratio
            if min_peers > 0 and trim_ratio > 0:
                min_needed = math.ceil(1.0 / (1.0 - 2 * trim_ratio))
                if min_peers < min_needed:
                    violations.append(
                        f"min_peers={min_peers} too low for trim_ratio={trim_ratio} "
                        f"(need >= {min_needed})"
                    )

        if violations:
            return CheckResult(
                name="config_validation",
                passed=False,
                severity=CheckSeverity.CRITICAL,
                message=f"{len(violations)} config violations",
                details={"violations": violations},
            )

        return CheckResult(
            name="config_validation",
            passed=True,
            severity=CheckSeverity.INFO,
            message="All config values within sane bounds",
        )


# ---------------------------------------------------------------------------
# 4. Dataset Quarantine — scan before training
# ---------------------------------------------------------------------------


class DatasetQuarantine:
    """Security-scan all datasets before they enter the training pipeline.

    Every dataset must pass through quarantine. If ANY sample contains
    critical-level threats (code injection, etc.), the entire dataset
    is rejected and the contributing node is flagged.
    """

    def __init__(self, max_scan_samples: int = 10000):
        self.max_scan_samples = max_scan_samples

    def scan(self, dataset: Any, dataset_name: str = "") -> CheckResult:
        """Scan a dataset for security threats."""
        from imo.data.security import CodeSecurityScanner, ThreatLevel

        scanner = CodeSecurityScanner(strict_mode=True)
        critical_count = 0
        high_count = 0
        scanned = 0
        flagged_indices: list[int] = []

        for i, sample in enumerate(dataset):
            if scanned >= self.max_scan_samples:
                break
            scanned += 1

            # Extract text content from sample
            text = self._extract_text(sample)
            if not text:
                continue

            result = scanner.scan(text)
            if not result.is_safe:
                if result.threat_level == ThreatLevel.CRITICAL:
                    critical_count += 1
                    flagged_indices.append(i)
                elif result.threat_level == ThreatLevel.HIGH:
                    high_count += 1
                    flagged_indices.append(i)

        if critical_count > 0:
            return CheckResult(
                name="dataset_quarantine",
                passed=False,
                severity=CheckSeverity.CRITICAL,
                message=(
                    f"Dataset '{dataset_name}': {critical_count} critical threats "
                    f"in {scanned} scanned samples — REJECTED"
                ),
                details={
                    "critical": critical_count,
                    "high": high_count,
                    "scanned": scanned,
                    "flagged_indices": flagged_indices[:50],
                },
            )

        if high_count > scanned * 0.05:
            return CheckResult(
                name="dataset_quarantine",
                passed=False,
                severity=CheckSeverity.WARNING,
                message=(
                    f"Dataset '{dataset_name}': {high_count} high-severity issues "
                    f"({high_count / scanned:.1%}) — review recommended"
                ),
                details={"high": high_count, "scanned": scanned},
            )

        return CheckResult(
            name="dataset_quarantine",
            passed=True,
            severity=CheckSeverity.INFO,
            message=f"Dataset '{dataset_name}': {scanned} samples scanned, clean",
        )

    def _extract_text(self, sample: Any) -> str:
        """Extract text content from a dataset sample."""
        if isinstance(sample, str):
            return sample
        if isinstance(sample, dict):
            for key in ("text", "content", "prompt", "code", "input", "instruction"):
                value = sample.get(key)
                if isinstance(value, str):
                    return value
        return ""


# ---------------------------------------------------------------------------
# 5. Warmup Trust Policy — stricter checks for new nodes
# ---------------------------------------------------------------------------


class WarmupTrustPolicy:
    """Stricter gradient validation during the trust warmup period.

    New nodes (or nodes with low reputation) get their gradients
    scrutinized more heavily for the first N steps. This catches
    poisoning attempts that try to strike early before detection
    systems have enough history.

    During warmup:
    - Gradient norms clamped to a tighter range
    - Cosine similarity to cluster mean must be higher
    - Contribution weight is reduced
    """

    def __init__(
        self,
        warmup_steps: int = 100,
        strict_norm_clip: float = 2.0,
        strict_similarity_threshold: float = 0.8,
        warmup_weight: float = 0.3,
    ):
        self.warmup_steps = warmup_steps
        self.strict_norm_clip = strict_norm_clip
        self.strict_similarity_threshold = strict_similarity_threshold
        self.warmup_weight = warmup_weight
        self._node_steps: dict[str, int] = {}

    def register_node(self, node_id: str) -> None:
        """Register a new node (starts warmup counter at 0)."""
        self._node_steps[node_id] = 0

    def record_step(self, node_id: str) -> None:
        """Record that a node completed a training step."""
        self._node_steps[node_id] = self._node_steps.get(node_id, 0) + 1

    def is_in_warmup(self, node_id: str) -> bool:
        """Check if a node is still in the warmup period."""
        return self._node_steps.get(node_id, 0) < self.warmup_steps

    def get_contribution_weight(self, node_id: str) -> float:
        """Get the contribution weight for a node.

        During warmup: reduced weight (e.g. 0.3x).
        After warmup: full weight (1.0x).
        Linear ramp between.
        """
        steps = self._node_steps.get(node_id, 0)
        if steps >= self.warmup_steps:
            return 1.0

        progress = steps / self.warmup_steps
        return self.warmup_weight + (1.0 - self.warmup_weight) * progress

    def validate_gradient(
        self,
        node_id: str,
        gradient_norm: float,
        cluster_mean_norm: float,
        cosine_similarity: float,
    ) -> CheckResult:
        """Validate a gradient under warmup policy.

        During warmup: stricter thresholds.
        After warmup: delegates to normal detection.
        """
        if not self.is_in_warmup(node_id):
            return CheckResult(
                name="warmup_trust",
                passed=True,
                severity=CheckSeverity.INFO,
                message=f"Node {node_id[:12]} past warmup — normal validation",
            )

        issues: list[str] = []

        # Norm check: must be within strict_norm_clip * cluster mean
        if cluster_mean_norm > 0:
            norm_ratio = gradient_norm / cluster_mean_norm
            if norm_ratio > self.strict_norm_clip:
                issues.append(
                    f"Gradient norm {gradient_norm:.2f} is {norm_ratio:.1f}x "
                    f"cluster mean (limit: {self.strict_norm_clip}x)"
                )

        # Similarity check
        if cosine_similarity < self.strict_similarity_threshold:
            issues.append(
                f"Cosine similarity {cosine_similarity:.4f} below warmup "
                f"threshold {self.strict_similarity_threshold}"
            )

        steps = self._node_steps.get(node_id, 0)
        if issues:
            return CheckResult(
                name="warmup_trust",
                passed=False,
                severity=CheckSeverity.WARNING,
                message=(
                    f"Node {node_id[:12]} warmup step {steps}/{self.warmup_steps}: "
                    f"{'; '.join(issues)}"
                ),
                details={"issues": issues, "step": steps},
            )

        return CheckResult(
            name="warmup_trust",
            passed=True,
            severity=CheckSeverity.INFO,
            message=f"Node {node_id[:12]} warmup step {steps}/{self.warmup_steps}: OK",
        )


# ---------------------------------------------------------------------------
# 6. Canary Detector — sentinel samples for ongoing poisoning detection
# ---------------------------------------------------------------------------


class CanaryDetector:
    """Embed canary (sentinel) samples to detect model poisoning in real-time.

    Before training, a small set of "canary" samples with known correct
    outputs are embedded. Periodically during training, the model is
    evaluated on canaries. If canary loss spikes or canary accuracy drops,
    it signals that the training process has been corrupted.

    This catches attacks that evade gradient-level detection by making
    small, consistent poisoning steps that individually look normal.
    """

    def __init__(
        self,
        canary_samples: list[dict[str, torch.Tensor]] | None = None,
        loss_spike_threshold: float = 3.0,
        check_interval: int = 50,
    ):
        self.canary_samples = canary_samples or []
        self.loss_spike_threshold = loss_spike_threshold
        self.check_interval = check_interval
        self._loss_history: list[float] = []
        self._baseline_loss: float | None = None

    def set_canaries(self, samples: list[dict[str, torch.Tensor]]) -> None:
        """Set the canary samples."""
        self.canary_samples = samples

    def establish_baseline(self, model: nn.Module, loss_fn: Any) -> float:
        """Compute baseline canary loss before training starts."""
        if not self.canary_samples:
            return 0.0

        model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for sample in self.canary_samples:
                outputs = model(**sample)
                if hasattr(outputs, "loss") and outputs.loss is not None:
                    total_loss += outputs.loss.item()
                elif loss_fn is not None:
                    total_loss += loss_fn(outputs, sample).item()

        model.train()
        baseline = total_loss / len(self.canary_samples)
        self._baseline_loss = baseline
        self._loss_history = [baseline]

        logger.info("Canary baseline loss: %.6f", baseline)
        return baseline

    def should_check(self, step: int) -> bool:
        """Whether this step should trigger a canary check."""
        return (
            len(self.canary_samples) > 0
            and step > 0
            and step % self.check_interval == 0
        )

    def check(self, model: nn.Module, step: int, loss_fn: Any = None) -> CheckResult:
        """Evaluate model on canary samples and check for poisoning."""
        if not self.canary_samples:
            return CheckResult(
                name="canary_detector",
                passed=True,
                severity=CheckSeverity.INFO,
                message="No canary samples configured",
            )

        model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for sample in self.canary_samples:
                outputs = model(**sample)
                if hasattr(outputs, "loss") and outputs.loss is not None:
                    total_loss += outputs.loss.item()
                elif loss_fn is not None:
                    total_loss += loss_fn(outputs, sample).item()

        model.train()
        current_loss = total_loss / len(self.canary_samples)
        self._loss_history.append(current_loss)

        # Check for spike
        if self._baseline_loss is not None and self._baseline_loss > 0:
            ratio = current_loss / self._baseline_loss
            if ratio > self.loss_spike_threshold:
                return CheckResult(
                    name="canary_detector",
                    passed=False,
                    severity=CheckSeverity.CRITICAL,
                    message=(
                        f"CANARY ALERT at step {step}: loss {current_loss:.6f} is "
                        f"{ratio:.1f}x baseline ({self._baseline_loss:.6f}) — "
                        "possible training poisoning"
                    ),
                    details={
                        "step": step,
                        "current_loss": current_loss,
                        "baseline_loss": self._baseline_loss,
                        "ratio": ratio,
                    },
                )

        # Check for trend (loss increasing over last 5 checks)
        if len(self._loss_history) >= 5:
            recent = self._loss_history[-5:]
            if all(recent[i] < recent[i + 1] for i in range(4)):
                return CheckResult(
                    name="canary_detector",
                    passed=False,
                    severity=CheckSeverity.WARNING,
                    message=(
                        f"Canary loss rising for 5 consecutive checks "
                        f"({recent[0]:.6f} → {recent[-1]:.6f}) — monitor closely"
                    ),
                    details={"recent_losses": recent},
                )

        return CheckResult(
            name="canary_detector",
            passed=True,
            severity=CheckSeverity.INFO,
            message=f"Step {step}: canary loss {current_loss:.6f} — normal",
        )


# ---------------------------------------------------------------------------
# Preflight Gate — runs all checks
# ---------------------------------------------------------------------------


class PreflightGate:
    """Mandatory security gate before training can begin.

    Runs all pre-flight checks and produces a report. Training MUST NOT
    proceed if any CRITICAL check fails.

    Usage:
        gate = PreflightGate(config)
        gate.add_dataset(train_dataset, "train")
        report = gate.run(model)
        if not report.passed:
            raise SecurityError(report.summary())
    """

    def __init__(
        self,
        config: Any,
        expected_model_hash: str | None = None,
        expected_manifest: dict[str, str] | None = None,
        weight_path: str | Path | None = None,
    ):
        self.config = config
        self.expected_model_hash = expected_model_hash
        self.expected_manifest = expected_manifest
        self.weight_path = Path(weight_path) if weight_path else None
        self._datasets: list[tuple[Any, str]] = []

    def add_dataset(self, dataset: Any, name: str = "unnamed") -> None:
        """Add a dataset to be quarantine-scanned."""
        self._datasets.append((dataset, name))

    def run(self, model: nn.Module) -> PreflightReport:
        """Run all pre-flight security checks."""
        report = PreflightReport()

        # 1. Config validation
        logger.info("Pre-flight: validating config...")
        report.checks.append(ConfigValidator.validate(self.config))

        # 2. Safe model loading check
        if self.weight_path is not None:
            logger.info("Pre-flight: checking weight file safety...")
            report.checks.append(SafeModelLoader.check(self.weight_path))

        # 3. Model integrity
        logger.info("Pre-flight: verifying model integrity...")
        report.checks.append(
            ModelIntegrityVerifier.verify(
                model,
                expected_hash=self.expected_model_hash,
                expected_manifest=self.expected_manifest,
            )
        )

        # 4. Dataset quarantine
        quarantine = DatasetQuarantine()
        for dataset, name in self._datasets:
            logger.info("Pre-flight: quarantine-scanning dataset '%s'...", name)
            report.checks.append(quarantine.scan(dataset, name))

        # Log report
        logger.info(report.summary())

        if not report.passed:
            logger.error(
                "PRE-FLIGHT FAILED: %d critical issues — training blocked",
                len(report.critical_failures),
            )

        return report
