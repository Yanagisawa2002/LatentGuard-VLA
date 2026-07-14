"""Safe, exact-revision synchronization of a remote execution checkout."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

Runner = Callable[..., subprocess.CompletedProcess[str]]

_FULL_SHA_RE = re.compile(r"[0-9a-fA-F]{40}\Z")
_HOST_RE = re.compile(r"[A-Za-z0-9_.@:-]+\Z")
_BRANCH_RE = re.compile(r"[A-Za-z0-9._/-]+\Z")
_SECRET_RE = re.compile(
    r"""(?i)((?<![A-Za-z0-9])(?:["'])?(?:[A-Za-z0-9]+[_-])*"""
    r"""(?:pass(?:word|wd|phrase)?|pwd|token|secret|credentials?|"""
    r"""(?:api|access|secret|ssh)[_-]?key|private[_ -]?key)(?:["'])?"""
    r"\s*[=:]\s*)[^\r\n]*"
)
_AUTHORIZATION_RE = re.compile(
    r"""(?i)((?:["'])?authorization(?:["'])?\s*[=:]\s*"""
    r"(?:bearer\s+)?)[^\r\n]*"
)
_URL_USERINFO_RE = re.compile(r"(?P<scheme>[A-Za-z][A-Za-z0-9+.-]*://)[^/\s]+@")
_SSH_PASSWORD_RE = re.compile(r"(?P<user>[A-Za-z0-9_.-]+):[^\s]+@")
_SUBPROCESS_TIMEOUT_SECONDS = 600


class RemoteSyncError(RuntimeError):
    """Raised when a local guard or remote synchronization operation fails."""

    def __init__(self, message: str, *, returncode: int | None = None) -> None:
        super().__init__(message)
        self.returncode = returncode


@dataclass(frozen=True, slots=True)
class RemoteSyncConfig:
    """Configuration for synchronizing one remote Git checkout."""

    host: str = field(repr=False)
    repo_dir: str = field(repr=False)
    branch: str
    expected_commit: str
    ssh_executable: str = field(default="ssh", repr=False)
    connect_timeout: int | None = None

    def __post_init__(self) -> None:
        host = _validated_config_text("host", self.host)
        repo_dir = _validated_config_text("repo_dir", self.repo_dir)
        branch = _validated_config_text("branch", self.branch)
        expected_commit = _validated_config_text(
            "expected_commit", self.expected_commit
        ).lower()
        ssh_executable = _validated_config_text("ssh_executable", self.ssh_executable)

        if not host or not _HOST_RE.fullmatch(host) or host.startswith("-"):
            raise ValueError(
                "RemoteSyncConfig.host must be a safe SSH host alias or user@host value"
            )
        if "@" in host:
            user, hostname = host.split("@", 1)
            if host.count("@") != 1 or not user or not hostname or ":" in user:
                raise ValueError(
                    "RemoteSyncConfig.host contains invalid or credential-shaped "
                    "SSH user information"
                )
        if not repo_dir:
            raise ValueError("RemoteSyncConfig.repo_dir must not be empty")
        if not _is_safe_branch(branch):
            raise ValueError("RemoteSyncConfig.branch is not a safe Git branch name")
        if not _FULL_SHA_RE.fullmatch(expected_commit):
            raise ValueError(
                "RemoteSyncConfig.expected_commit must be a full 40-character Git SHA"
            )
        if not ssh_executable:
            raise ValueError("RemoteSyncConfig.ssh_executable must not be empty")
        if self.connect_timeout is not None and (
            type(self.connect_timeout) is not int or self.connect_timeout <= 0
        ):
            raise ValueError(
                "RemoteSyncConfig.connect_timeout must be a positive integer"
            )

        object.__setattr__(self, "host", host)
        object.__setattr__(self, "repo_dir", repo_dir)
        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "expected_commit", expected_commit)
        object.__setattr__(self, "ssh_executable", ssh_executable)


@dataclass(frozen=True, slots=True)
class RemoteSyncResult:
    """Result and sanitized summary of a dry-run or completed synchronization."""

    config: RemoteSyncConfig = field(repr=False)
    command: tuple[str, ...] = field(repr=False)
    sanitized_plan: str = field(repr=False)
    returncode: int
    dry_run: bool

    def __str__(self) -> str:
        """Return only the user-facing sanitized synchronization summary."""

        return self.sanitized_plan


def resolve_remote_config(
    *,
    host: str | None,
    repo_dir: str | None,
    branch: str | None,
    expected_commit: str | None,
    ssh_executable: str | None,
    connect_timeout: int | None,
    environ: Mapping[str, str] | None = None,
) -> RemoteSyncConfig:
    """Resolve explicit remote configuration with environment-value fallbacks."""

    values = os.environ if environ is None else environ
    resolved_host = _argument_or_environment(host, values, "LATENTGUARD_REMOTE_HOST")
    resolved_repo = _argument_or_environment(
        repo_dir, values, "LATENTGUARD_REMOTE_REPO"
    )
    resolved_branch = _argument_or_environment(
        branch, values, "LATENTGUARD_REMOTE_BRANCH"
    )
    resolved_commit = _argument_or_environment(
        expected_commit, values, "LATENTGUARD_REMOTE_COMMIT"
    )
    resolved_ssh = _argument_or_environment(
        ssh_executable, values, "LATENTGUARD_REMOTE_SSH_EXECUTABLE"
    )

    missing = [
        name
        for name, value in (
            ("host", resolved_host),
            ("repo_dir", resolved_repo),
            ("branch", resolved_branch),
            ("expected_commit", resolved_commit),
        )
        if value is None or not value.strip()
    ]
    if missing:
        raise ValueError(
            "Missing remote synchronization configuration: " + ", ".join(missing)
        )

    timeout = connect_timeout
    if timeout is None:
        timeout_text = values.get("LATENTGUARD_REMOTE_CONNECT_TIMEOUT")
        if timeout_text:
            try:
                timeout = int(timeout_text)
            except ValueError as exc:
                raise ValueError(
                    "LATENTGUARD_REMOTE_CONNECT_TIMEOUT must be an integer"
                ) from exc

    return RemoteSyncConfig(
        host=resolved_host or "",
        repo_dir=resolved_repo or "",
        branch=resolved_branch or "",
        expected_commit=resolved_commit or "",
        ssh_executable=resolved_ssh or "ssh",
        connect_timeout=timeout,
    )


def build_remote_script(config: RemoteSyncConfig) -> str:
    """Build the fail-fast remote shell program with safely quoted configuration."""

    repo = shlex.quote(config.repo_dir)
    branch = shlex.quote(config.branch)
    expected_commit = shlex.quote(config.expected_commit)
    return "\n".join(
        (
            "set -eu",
            f"repo={repo}",
            f"branch={branch}",
            f"expected_commit={expected_commit}",
            'work_tree="$(git -C "$repo" rev-parse '
            '--is-inside-work-tree 2>/dev/null || true)"',
            'if [ ! -d "$repo" ] || [ "$work_tree" != "true" ]; then',
            '  echo "remote repository does not exist or is not a Git checkout" >&2',
            "  exit 40",
            "fi",
            "if ! command -v python3 >/dev/null 2>&1; then",
            '  echo "python3 is unavailable on the remote host" >&2',
            "  exit 43",
            "fi",
            'cd -- "$repo"',
            'if [ -n "$(git status --porcelain --untracked-files=no)" ]; then',
            '  echo "remote repository has tracked modifications" >&2',
            "  exit 41",
            "fi",
            "git fetch --prune origin",
            'git checkout "$branch"',
            'git pull --ff-only origin "$branch"',
            'actual_branch="$(git branch --show-current)"',
            'if [ "$actual_branch" != "$branch" ]; then',
            '  echo "remote branch does not match requested branch" >&2',
            "  exit 44",
            "fi",
            'actual_commit="$(git rev-parse HEAD)"',
            'if [ "$actual_commit" != "$expected_commit" ]; then',
            '  echo "remote HEAD does not match expected commit" >&2',
            "  exit 42",
            "fi",
            'if [ -n "$(git status --porcelain --untracked-files=no)" ]; then',
            '  echo "remote repository has tracked modifications after sync" >&2',
            "  exit 41",
            "fi",
        )
    )


def build_ssh_command(config: RemoteSyncConfig) -> tuple[str, ...]:
    """Construct a system-SSH argument vector without invoking a local shell."""

    command: list[str] = [config.ssh_executable, "-o", "BatchMode=yes"]
    if config.connect_timeout is not None:
        command.extend(("-o", f"ConnectTimeout={config.connect_timeout}"))
    command.extend((config.host, build_remote_script(config)))
    return tuple(command)


def sync_remote(
    config: RemoteSyncConfig,
    *,
    dry_run: bool = False,
    runner: Runner = subprocess.run,
) -> RemoteSyncResult:
    """Validate local Git state and synchronize a remote checkout to an exact SHA.

    Dry-run mode constructs and reports the operation without invoking any
    subprocess, so it requires neither a Git repository nor network access.
    """

    command = build_ssh_command(config)
    plan = _sanitized_plan(config, dry_run=dry_run)
    if dry_run:
        return RemoteSyncResult(
            config=config,
            command=command,
            sanitized_plan=plan,
            returncode=0,
            dry_run=True,
        )

    _validate_local_repository(config, runner)
    completed = _invoke(
        runner,
        command,
        purpose="remote synchronization",
        allow_nonzero=True,
    )
    if completed.returncode != 0:
        detail = _clean_process_detail(completed)
        message = "Remote synchronization failed"
        if completed.returncode == 41:
            message = "Remote synchronization refused tracked remote modifications"
        elif completed.returncode == 42:
            message = "Remote synchronization found an unexpected remote HEAD"
        elif completed.returncode == 43:
            message = "Remote synchronization requires remote python3"
        elif completed.returncode == 44:
            message = "Remote synchronization found an unexpected remote branch"
        if detail:
            message = f"{message}: {detail}"
        raise RemoteSyncError(message, returncode=completed.returncode)

    return RemoteSyncResult(
        config=config,
        command=command,
        sanitized_plan=_sanitized_plan(config, dry_run=False),
        returncode=completed.returncode,
        dry_run=False,
    )


def _validate_local_repository(config: RemoteSyncConfig, runner: Runner) -> None:
    status = _run_git(runner, ("status", "--porcelain"), "inspect local changes")
    if status:
        raise RemoteSyncError("Remote synchronization refused a dirty local repository")

    local_branch = _run_git(
        runner,
        ("branch", "--show-current"),
        "resolve the current local branch",
    )
    if local_branch != config.branch:
        raise RemoteSyncError(
            "Configured remote branch does not equal the current local branch"
        )

    upstream = _run_git(
        runner,
        ("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"),
        "resolve the local upstream",
        missing_upstream=True,
    )
    if not upstream:
        raise RemoteSyncError("Remote synchronization requires a local upstream branch")

    local_head = _run_git(runner, ("rev-parse", "HEAD"), "resolve local HEAD").lower()
    upstream_head = _run_git(
        runner, ("rev-parse", "@{upstream}"), "resolve upstream HEAD"
    ).lower()
    if not _FULL_SHA_RE.fullmatch(local_head):
        raise RemoteSyncError("Local Git returned an invalid HEAD revision")
    if local_head != upstream_head:
        raise RemoteSyncError("Local HEAD does not match its upstream revision")
    if config.expected_commit != local_head:
        raise RemoteSyncError("Expected commit does not equal local HEAD")


def _run_git(
    runner: Runner,
    arguments: Sequence[str],
    purpose: str,
    *,
    missing_upstream: bool = False,
) -> str:
    completed = _invoke(
        runner,
        ("git", *arguments),
        purpose=purpose,
        allow_nonzero=True,
    )
    if completed.returncode != 0:
        if missing_upstream:
            raise RemoteSyncError(
                "Remote synchronization requires a local upstream branch",
                returncode=completed.returncode,
            )
        detail = _clean_process_detail(completed)
        suffix = f": {detail}" if detail else ""
        raise RemoteSyncError(
            f"Could not {purpose}{suffix}", returncode=completed.returncode
        )
    return (completed.stdout or "").strip()


def _invoke(
    runner: Runner,
    command: Sequence[str],
    *,
    purpose: str,
    allow_nonzero: bool,
) -> subprocess.CompletedProcess[str]:
    try:
        completed = runner(
            list(command),
            check=False,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            text=True,
            shell=False,
            timeout=_SUBPROCESS_TIMEOUT_SECONDS,
        )
    except subprocess.CalledProcessError as exc:
        detail = _redact_text(_to_text(exc.stderr) or _to_text(exc.output))
        suffix = f": {detail}" if detail else ""
        raise RemoteSyncError(
            f"Could not {purpose}{suffix}", returncode=exc.returncode
        ) from None
    except subprocess.TimeoutExpired:
        raise RemoteSyncError(
            f"Could not {purpose}: operation exceeded "
            f"{_SUBPROCESS_TIMEOUT_SECONDS} seconds"
        ) from None
    except OSError as exc:
        raise RemoteSyncError(
            f"Could not {purpose}: {_redact_text(str(exc))}"
        ) from None

    if not isinstance(completed, subprocess.CompletedProcess):
        raise RemoteSyncError(
            f"Could not {purpose}: subprocess runner returned no result"
        )
    if completed.returncode != 0 and not allow_nonzero:
        raise RemoteSyncError(
            f"Could not {purpose}: {_clean_process_detail(completed)}",
            returncode=completed.returncode,
        )
    return completed


def _sanitized_plan(config: RemoteSyncConfig, *, dry_run: bool) -> str:
    mode = "dry-run" if dry_run else "complete"
    lines = (
        f"Remote sync {mode}",
        f"host: {_redact_text(config.host)}",
        f"repository: {_redact_text(config.repo_dir)}",
        f"branch: {config.branch}",
        f"expected commit: {config.expected_commit}",
        "local guards: clean tree, upstream present, exact HEAD parity",
        "remote steps: tracked-change check, fetch, checkout, ff-only pull, "
        "exact SHA check",
    )
    return "\n".join(lines)


def _argument_or_environment(
    argument: str | None, environ: Mapping[str, str], name: str
) -> str | None:
    return argument if argument is not None else environ.get(name)


def _require_single_line(field: str, value: str) -> None:
    if any(_is_unsafe_control(character) for character in value):
        raise ValueError(
            f"RemoteSyncConfig.{field} must not contain control or format characters"
        )


def _validated_config_text(field: str, value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(f"RemoteSyncConfig.{field} must be a string")
    _require_single_line(field, value)
    if value != value.strip():
        raise ValueError(
            f"RemoteSyncConfig.{field} must not have leading or trailing whitespace"
        )
    return value


def _is_safe_branch(branch: str) -> bool:
    if not branch or not _BRANCH_RE.fullmatch(branch):
        return False
    components = branch.split("/")
    return not (
        branch.startswith(("-", ".", "/"))
        or branch.endswith((".", "/", ".lock"))
        or ".." in branch
        or "//" in branch
        or "@{" in branch
        or any(
            component.startswith(".") or component.endswith(".lock")
            for component in components
        )
    )


def _clean_process_detail(completed: subprocess.CompletedProcess[str]) -> str:
    return _redact_text(_to_text(completed.stderr) or _to_text(completed.stdout))


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace").strip()
    return str(value).strip()


def _redact_text(value: str) -> str:
    value = "".join(
        character for character in value if not _is_unsafe_control(character)
    )
    redacted = _URL_USERINFO_RE.sub(r"\g<scheme>[REDACTED]@", value)
    redacted = _SSH_PASSWORD_RE.sub(r"\g<user>:[REDACTED]@", redacted)
    redacted = _AUTHORIZATION_RE.sub(r"\1[REDACTED]", redacted)
    return _SECRET_RE.sub(r"\1[REDACTED]", redacted)


def _is_unsafe_control(character: str) -> bool:
    return unicodedata.category(character) in {"Cc", "Cf", "Cs", "Zl", "Zp"}
