"""Shared conservative sanitization for persisted operational metadata."""

from __future__ import annotations

import re
import shlex
import unicodedata
from collections.abc import Sequence

MAX_OPERATIONAL_TEXT = 512
REDACTED_OPERATIONAL_TEXT = "operational error details redacted"
REDACTED_OPERATIONAL_COMMAND = "operational command redacted"

_SENSITIVE_MARKER_RE = re.compile(
    r"(?i)(?:authorization|bearer|password|passwd|token|secret|credential|"
    r"api[_-]?key|access[_-]?key|client[_-]?secret|private[_-]?key|"
    r"ssh[_-]?password)"
)
_ENVIRONMENT_DUMP_RE = re.compile(
    r"(?i)(?:os\.environ|environment|environ|env)\s*[:=(]"
)
_ENVIRONMENT_ASSIGNMENT_RE = re.compile(
    r"(?:^|[\s,{])['\"]?[A-Z][A-Z0-9_]{1,}['\"]?\s*[:=]"
)
_URL_RE = re.compile(r"(?i)[A-Za-z][A-Za-z0-9+.-]*://")
_SSH_PASSWORD_RE = re.compile(r"(?i)[A-Za-z0-9_.-]+:[^\s]+@")
_REMOTE_TARGET_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9_.-])[A-Za-z0-9_.-]+@"
    r"(?:\[[0-9A-Fa-f:.]+\]|[A-Za-z0-9.-]+)(?::[0-9]{1,5})?"
    r"(?=$|[\s,;:)\]}])"
)
_WINDOWS_PATH_RE = re.compile(r"(?i)(?<![A-Za-z0-9_])[A-Z]:[\\/]")
_UNC_PATH_RE = re.compile(r"\\\\[^\\\s]+[\\/]")
_POSIX_PATH_RE = re.compile(r"(?<![A-Za-z0-9_])/(?!/)[^\s,;]+")
_SSH_COMMAND_RE = re.compile(
    r"(?i)(?:^|\s)(?:ssh|scp|sftp|plink|pscp)(?:\.exe)?(?:\s|$)"
)
_SSH_EXECUTABLE_RE = re.compile(
    r"(?i)(?:^|[\\/])(?:ssh|scp|sftp|plink|pscp)(?:\.exe)?$"
)
_TRACEBACK_RE = re.compile(r"(?i)traceback\s*\(most recent call last\)")
_PATH_OPTIONS = frozenset({"--corruption-dir", "--output-dir", "--config"})
_SECRET_OPTIONS = frozenset(
    {
        "--api-key",
        "--api_key",
        "--access-key",
        "--access_key",
        "--authorization",
        "--auth-token",
        "--auth_token",
        "--client-secret",
        "--client_secret",
        "--credential",
        "--credentials",
        "--key",
        "--passwd",
        "--password",
        "--password-file",
        "--password_file",
        "--private-key",
        "--private_key",
        "--secret",
        "--ssh-password",
        "--ssh_password",
        "--token",
    }
)
_UNSAFE_CATEGORIES = frozenset({"Cc", "Cf", "Cs", "Zl", "Zp"})


def contains_sensitive_operational_content(text: str) -> bool:
    """Return whether text could disclose credentials, commands, URLs, or paths."""
    return any(
        pattern.search(text) is not None
        for pattern in (
            _SENSITIVE_MARKER_RE,
            _ENVIRONMENT_DUMP_RE,
            _ENVIRONMENT_ASSIGNMENT_RE,
            _URL_RE,
            _SSH_PASSWORD_RE,
            _REMOTE_TARGET_RE,
            _WINDOWS_PATH_RE,
            _UNC_PATH_RE,
            _POSIX_PATH_RE,
            _SSH_COMMAND_RE,
            _TRACEBACK_RE,
        )
    )


def is_sanitized_operational_text(value: object) -> bool:
    """Return whether text is concise, control-free, and conservatively safe."""
    if not isinstance(value, str) or not value or value != value.strip():
        return False
    if len(value) > MAX_OPERATIONAL_TEXT:
        return False
    if any(
        unicodedata.category(character) in _UNSAFE_CATEGORIES for character in value
    ):
        return False
    return not contains_sensitive_operational_content(value)


def sanitize_operational_text(value: object) -> str:
    """Remove controls and replace any potentially sensitive message wholesale."""
    text = _normalize_operational_text(value)
    if contains_sensitive_operational_content(text):
        return REDACTED_OPERATIONAL_TEXT
    if not text:
        return "unspecified operational error"
    if len(text) > MAX_OPERATIONAL_TEXT:
        return text[: MAX_OPERATIONAL_TEXT - 3] + "..."
    return text


def _normalize_operational_text(value: object) -> str:
    text = "" if value is None else str(value)
    normalized = "".join(
        " " if unicodedata.category(character) in _UNSAFE_CATEGORIES else character
        for character in text
    )
    return " ".join(normalized.split())


def sanitize_launch_command(arguments: Sequence[str]) -> str:
    """Return a concise command without private paths, SSH, or credential values."""
    source = tuple(str(argument) for argument in arguments)
    if not source:
        return "latentguard evaluate-data"
    normalized_source = tuple(_normalize_operational_text(item) for item in source)
    if any(
        _SSH_COMMAND_RE.search(argument) is not None
        or _SSH_EXECUTABLE_RE.search(argument) is not None
        for argument in normalized_source
    ):
        return REDACTED_OPERATIONAL_COMMAND

    sanitized: list[str] = []
    redact_next_path = False
    redact_next_secret = False
    for raw in source:
        lowered = raw.lower()
        if redact_next_path:
            sanitized.append("<path>")
            redact_next_path = False
            continue
        if redact_next_secret:
            sanitized.append("[REDACTED]")
            redact_next_secret = False
            continue
        if lowered in _PATH_OPTIONS:
            sanitized.append(lowered)
            redact_next_path = True
            continue
        if lowered in _SECRET_OPTIONS:
            sanitized.append("<redacted-option>")
            redact_next_secret = True
            continue
        matched_path = next(
            (option for option in _PATH_OPTIONS if lowered.startswith(f"{option}=")),
            None,
        )
        if matched_path is not None:
            sanitized.append(f"{matched_path}=<path>")
            continue
        matched_secret = next(
            (option for option in _SECRET_OPTIONS if lowered.startswith(f"{option}=")),
            None,
        )
        if matched_secret is not None:
            sanitized.append("<redacted-option>=[REDACTED]")
            continue
        if lowered.startswith("--") and contains_sensitive_operational_content(raw):
            sanitized.append("<redacted-option>")
            redact_next_secret = True
            continue
        sanitized.append(sanitize_operational_text(raw))
    return " ".join(shlex.quote(item) for item in sanitized)


__all__ = [
    "MAX_OPERATIONAL_TEXT",
    "REDACTED_OPERATIONAL_COMMAND",
    "REDACTED_OPERATIONAL_TEXT",
    "contains_sensitive_operational_content",
    "is_sanitized_operational_text",
    "sanitize_launch_command",
    "sanitize_operational_text",
]
