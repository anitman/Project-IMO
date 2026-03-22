"""Code security and injection detection for datasets."""

from __future__ import annotations

import ast
import json
import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Threat level for detected code patterns."""

    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityIssue:
    """A security issue detected in dataset content."""

    threat_level: ThreatLevel
    category: str
    message: str
    location: str
    snippet: str


@dataclass
class SecurityResult:
    """Result of security scanning."""

    is_safe: bool
    threat_level: ThreatLevel
    issues: list[SecurityIssue]
    scanned_samples: int

    def __bool__(self) -> bool:
        return self.is_safe


class CodeSecurityScanner:
    """Scan dataset content for code injection and malicious patterns."""

    DANGEROUS_PATTERNS = [
        (r"\b(eval|exec|compile|__import__)\s*\(", ThreatLevel.HIGH, "Dangerous function call"),
        (
            r"\b(subprocess|os\.system|os\.popen)\s*\(",
            ThreatLevel.CRITICAL,
            "System command execution",
        ),
        (r"\b(open|file)\s*\([^)]*['\"]w['\"]", ThreatLevel.HIGH, "File write operation"),
        (r"__\w+__", ThreatLevel.MEDIUM, "Dunder attribute access"),
        (r"\bimport\s+\w+", ThreatLevel.MEDIUM, "Import statement"),
        (r"\bfrom\s+\w+\s+import", ThreatLevel.MEDIUM, "From-import statement"),
        (r"\bsocket\.", ThreatLevel.CRITICAL, "Network socket access"),
        (r"\brequests\.(get|post|put|delete)", ThreatLevel.HIGH, "HTTP request"),
        (r"\b(base64|pickle|marshal)\.", ThreatLevel.HIGH, "Serialization/encoding"),
        (r"shell=True", ThreatLevel.CRITICAL, "Shell execution enabled"),
        (r"eval\(.*\)\s*\[\s*['\"]__", ThreatLevel.CRITICAL, "Attribute injection"),
        (r"globals\(\)|locals\(\)", ThreatLevel.HIGH, "Namespace access"),
        (r"setattr\(|getattr\(|delattr\(", ThreatLevel.HIGH, "Attribute manipulation"),
        (r"getattr\(.*\s*['\"]__", ThreatLevel.CRITICAL, "Dunder attribute access"),
    ]

    UNSAFE_IMPORTS = {
        "os",
        "sys",
        "subprocess",
        "socket",
        "ctypes",
        "shutil",
        "pickle",
        "marshal",
        "dbm",
        "sqlite3",
        "asyncore",
        "asynchat",
        "http.server",
        "xmlrpc",
        "ftplib",
        "poplib",
        "imaplib",
        "smtplib",
        "telnetlib",
    }

    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode

    def scan(self, text: str) -> SecurityResult:
        """Scan text content for security issues."""
        issues: list[SecurityIssue] = []

        self._scan_patterns(text, issues)
        self._scan_python_syntax(text, issues)
        self._scan_javascript(text, issues)
        self._scan_shell(text, issues)

        threat_level = self._compute_threat_level(issues)
        is_safe = threat_level in {ThreatLevel.SAFE, ThreatLevel.LOW}

        return SecurityResult(
            is_safe=is_safe,
            threat_level=threat_level,
            issues=issues,
            scanned_samples=1,
        )

    def scan_batch(self, texts: list[str]) -> list[SecurityResult]:
        """Scan multiple text samples."""
        return [self.scan(text) for text in texts]

    def scan_file(self, file_path: str | Path) -> SecurityResult:
        """Scan a file for security issues."""
        path = Path(file_path)

        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            return SecurityResult(
                is_safe=False,
                threat_level=ThreatLevel.CRITICAL,
                issues=[
                    SecurityIssue(
                        threat_level=ThreatLevel.CRITICAL,
                        category="file_error",
                        message=f"Failed to read file: {e}",
                        location=str(path),
                        snippet="",
                    )
                ],
                scanned_samples=0,
            )

        return self.scan(content)

    def _scan_patterns(self, text: str, issues: list[SecurityIssue]) -> None:
        """Scan for dangerous code patterns."""
        for pattern, threat_level, message in self.DANGEROUS_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 20)
                snippet = text[start:end].replace("\n", " ")

                issues.append(
                    SecurityIssue(
                        threat_level=threat_level,
                        category="pattern_match",
                        message=message,
                        location=f"position {match.start()}",
                        snippet=f"...{snippet}...",
                    )
                )

    def _scan_python_syntax(self, text: str, issues: list[SecurityIssue]) -> None:
        """Scan for Python code syntax that might indicate injection."""
        if not self._looks_like_python(text):
            return

        try:
            tree = ast.parse(text)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".")[0] in self.UNSAFE_IMPORTS:
                            issues.append(
                                SecurityIssue(
                                    threat_level=ThreatLevel.HIGH,
                                    category="unsafe_import",
                                    message=f"Unsafe import: {alias.name}",
                                    location=f"line {node.lineno}",
                                    snippet=f"import {alias.name}",
                                )
                            )

                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.split(".")[0] in self.UNSAFE_IMPORTS:
                        issues.append(
                            SecurityIssue(
                                threat_level=ThreatLevel.HIGH,
                                category="unsafe_import",
                                message=f"Unsafe from-import: {node.module}",
                                location=f"line {node.lineno}",
                                snippet=f"from {node.module} import ...",
                            )
                        )

        except SyntaxError:
            pass

    def _scan_javascript(self, text: str, issues: list[SecurityIssue]) -> None:
        """Scan for JavaScript code patterns."""
        js_patterns = [
            (r"\beval\s*\(", "JavaScript eval()"),
            (r"\bexec\s*\(", "JavaScript exec()"),
            (r"\brequire\s*\(", "Node.js require()"),
            (r"\bdocument\.(write|cookie)", "DOM manipulation"),
            (r"\bwindow\.(location|open)", "Window manipulation"),
            (r"\bfetch\s*\(|\ AXIOS\.", "Network request"),
        ]

        if not self._looks_like_javascript(text):
            return

        for pattern, message in js_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 20)
                snippet = text[start:end].replace("\n", " ")

                issues.append(
                    SecurityIssue(
                        threat_level=ThreatLevel.HIGH,
                        category="javascript_injection",
                        message=message,
                        location=f"position {match.start()}",
                        snippet=f"...{snippet}...",
                    )
                )

    def _scan_shell(self, text: str, issues: list[SecurityIssue]) -> None:
        """Scan for shell command patterns."""
        shell_patterns = [
            (r";\s*(rm|mv|cp|chmod|chown|wget|curl)\s", "Shell command injection"),
            (r"`[^`]+`", "Shell command substitution"),
            (r"\$\([^)]+\)", "Command substitution"),
            (r"\|\s*(nc|netcat|bash|sh)\s", "Pipe to shell"),
        ]

        for pattern, message in shell_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 20)
                snippet = text[start:end].replace("\n", " ")

                issues.append(
                    SecurityIssue(
                        threat_level=ThreatLevel.CRITICAL,
                        category="shell_injection",
                        message=message,
                        location=f"position {match.start()}",
                        snippet=f"...{snippet}...",
                    )
                )

    def _looks_like_python(self, text: str) -> bool:
        """Check if text looks like Python code."""
        python_indicators = [
            r"def\s+\w+\s*\(",
            r"class\s+\w+",
            r"^\s*(if|for|while|try|with)\s*",
            r"^\s*@\w+",
            r"->\s*[\w\[\],\s]+:\s*$",
        ]

        for pattern in python_indicators:
            if re.search(pattern, text, re.MULTILINE):
                return True
        return False

    def _looks_like_javascript(self, text: str) -> bool:
        """Check if text looks like JavaScript code."""
        js_indicators = [
            r"\b(function|const|let|var)\s+\w+",
            r"=>\s*[{\(]",
            r"\bconsole\.(log|error|warn)",
            r"module\.(exports|require)",
        ]

        for pattern in js_indicators:
            if re.search(pattern, text):
                return True
        return False

    def _compute_threat_level(self, issues: list[SecurityIssue]) -> ThreatLevel:
        """Compute overall threat level from issues."""
        if not issues:
            return ThreatLevel.SAFE

        level_order = {
            ThreatLevel.CRITICAL: 5,
            ThreatLevel.HIGH: 4,
            ThreatLevel.MEDIUM: 3,
            ThreatLevel.LOW: 2,
            ThreatLevel.SAFE: 1,
        }

        max_level = max(issues, key=lambda i: level_order[i.threat_level]).threat_level

        if self.strict_mode and max_level in {ThreatLevel.HIGH, ThreatLevel.CRITICAL}:
            return max_level

        return max_level


class SemgrepScanner:
    """AST-level code security scanning using Semgrep.

    Semgrep performs semantic analysis rather than regex matching, making it
    significantly harder to bypass with obfuscation tricks like:
    - String concatenation: exec("ev" + "al(...)")
    - Unicode escapes: \\u0065val()
    - Indirect calls: getattr(builtins, "eval")(...)
    - Alias tricks: e = eval; e(...)

    Semgrep understands the AST and dataflow, catching these patterns.

    Requires: pip install semgrep
    Reference: https://github.com/semgrep/semgrep
    """

    # Built-in rules targeting common injection patterns in dataset content
    BUILTIN_RULES: list[dict[str, object]] = [
        {
            "id": "imo-dangerous-exec",
            "pattern-either": [
                {"pattern": "exec(...)"},
                {"pattern": "eval(...)"},
                {"pattern": "compile(...)"},
                {"pattern": "__import__(...)"},
            ],
            "message": "Dangerous dynamic execution detected",
            "severity": "ERROR",
            "languages": ["python"],
        },
        {
            "id": "imo-system-command",
            "pattern-either": [
                {"pattern": "os.system(...)"},
                {"pattern": "os.popen(...)"},
                {"pattern": "subprocess.$FUNC(...)"},
            ],
            "message": "System command execution detected",
            "severity": "ERROR",
            "languages": ["python"],
        },
        {
            "id": "imo-network-access",
            "pattern-either": [
                {"pattern": "socket.socket(...)"},
                {"pattern": "requests.$FUNC(...)"},
                {"pattern": "urllib.request.urlopen(...)"},
                {"pattern": "httpx.$FUNC(...)"},
            ],
            "message": "Network access detected",
            "severity": "WARNING",
            "languages": ["python"],
        },
        {
            "id": "imo-unsafe-deserialization",
            "pattern-either": [
                {"pattern": "pickle.loads(...)"},
                {"pattern": "pickle.load(...)"},
                {"pattern": "marshal.loads(...)"},
                {"pattern": "yaml.load(..., Loader=yaml.FullLoader)"},
                {"pattern": "yaml.unsafe_load(...)"},
            ],
            "message": "Unsafe deserialization detected",
            "severity": "ERROR",
            "languages": ["python"],
        },
        {
            "id": "imo-attribute-injection",
            "patterns": [
                {"pattern": "getattr($OBJ, $ATTR, ...)"},
                {"metavariable-regex": {"metavariable": "$ATTR", "regex": ".*__.*"}},
            ],
            "message": "Dunder attribute access via getattr",
            "severity": "ERROR",
            "languages": ["python"],
        },
    ]

    def __init__(
        self,
        extra_rules_path: str | Path | None = None,
        use_registry: bool = False,
    ):
        self.extra_rules_path = Path(extra_rules_path) if extra_rules_path else None
        self.use_registry = use_registry

    def is_available(self) -> bool:
        """Check if semgrep is installed and accessible."""
        try:
            result = subprocess.run(
                ["semgrep", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def scan(self, text: str, language: str = "python") -> SecurityResult:
        """Scan text content using Semgrep AST analysis.

        Falls back to CodeSecurityScanner if Semgrep is not installed.
        """
        if not self.is_available():
            logger.warning("Semgrep not available, falling back to regex scanner")
            return CodeSecurityScanner(strict_mode=True).scan(text)

        return self._run_semgrep(text, language)

    def scan_file(self, file_path: str | Path) -> SecurityResult:
        """Scan a file using Semgrep."""
        path = Path(file_path)
        if not self.is_available():
            logger.warning("Semgrep not available, falling back to regex scanner")
            return CodeSecurityScanner(strict_mode=True).scan_file(path)

        return self._run_semgrep_on_path(path)

    def _run_semgrep(self, text: str, language: str) -> SecurityResult:
        """Run Semgrep on text content via a temp file."""
        suffix = {"python": ".py", "javascript": ".js", "bash": ".sh"}.get(language, ".py")

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=suffix, delete=False, encoding="utf-8"
        ) as f:
            f.write(text)
            tmp_path = Path(f.name)

        try:
            return self._run_semgrep_on_path(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def _run_semgrep_on_path(self, path: Path) -> SecurityResult:
        """Run Semgrep on a file path and parse results."""
        # Write built-in rules to temp config
        rules_config = {"rules": self.BUILTIN_RULES}

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as rf:
            import yaml  # type: ignore[import-untyped]

            yaml.dump(rules_config, rf, default_flow_style=False)
            rules_path = Path(rf.name)

        try:
            cmd = [
                "semgrep",
                "--config", str(rules_path),
                "--json",
                "--quiet",
                "--no-git-ignore",
                str(path),
            ]

            if self.extra_rules_path and self.extra_rules_path.exists():
                cmd.extend(["--config", str(self.extra_rules_path)])

            if self.use_registry:
                cmd.extend(["--config", "p/security-audit"])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )

            return self._parse_results(result.stdout)

        except subprocess.TimeoutExpired:
            return SecurityResult(
                is_safe=False,
                threat_level=ThreatLevel.MEDIUM,
                issues=[SecurityIssue(
                    threat_level=ThreatLevel.MEDIUM,
                    category="scanner_timeout",
                    message="Semgrep scan timed out — file may be too large",
                    location=str(path),
                    snippet="",
                )],
                scanned_samples=1,
            )
        finally:
            rules_path.unlink(missing_ok=True)

    def _parse_results(self, json_output: str) -> SecurityResult:
        """Parse Semgrep JSON output into SecurityResult."""
        issues: list[SecurityIssue] = []

        try:
            data = json.loads(json_output) if json_output.strip() else {}
        except json.JSONDecodeError:
            return SecurityResult(
                is_safe=True,
                threat_level=ThreatLevel.SAFE,
                issues=[],
                scanned_samples=1,
            )

        severity_map = {
            "ERROR": ThreatLevel.HIGH,
            "WARNING": ThreatLevel.MEDIUM,
            "INFO": ThreatLevel.LOW,
        }

        for finding in data.get("results", []):
            severity_str = finding.get("extra", {}).get("severity", "WARNING")
            threat = severity_map.get(severity_str, ThreatLevel.MEDIUM)

            lines = finding.get("extra", {}).get("lines", "")
            snippet = lines[:100] if isinstance(lines, str) else ""

            issues.append(SecurityIssue(
                threat_level=threat,
                category=f"semgrep:{finding.get('check_id', 'unknown')}",
                message=finding.get("extra", {}).get("message", "Security issue detected"),
                location=f"line {finding.get('start', {}).get('line', '?')}",
                snippet=snippet,
            ))

        if not issues:
            return SecurityResult(
                is_safe=True,
                threat_level=ThreatLevel.SAFE,
                issues=[],
                scanned_samples=1,
            )

        max_threat = max(issues, key=lambda i: {
            ThreatLevel.CRITICAL: 5, ThreatLevel.HIGH: 4,
            ThreatLevel.MEDIUM: 3, ThreatLevel.LOW: 2, ThreatLevel.SAFE: 1,
        }[i.threat_level]).threat_level

        return SecurityResult(
            is_safe=max_threat in {ThreatLevel.SAFE, ThreatLevel.LOW},
            threat_level=max_threat,
            issues=issues,
            scanned_samples=1,
        )
