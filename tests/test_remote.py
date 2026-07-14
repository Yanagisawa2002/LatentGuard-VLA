"""CPU-only tests for safe remote synchronization planning and execution."""

from __future__ import annotations

import subprocess
import traceback
from collections.abc import Sequence
from typing import Any

import pytest

from latentguard.remote import (
    RemoteSyncConfig,
    RemoteSyncError,
    build_remote_script,
    build_ssh_command,
    resolve_remote_config,
    sync_remote,
)

SHA = "0123456789abcdef0123456789abcdef01234567"


def _config(**overrides: Any) -> RemoteSyncConfig:
    values: dict[str, Any] = {
        "host": "training-host",
        "repo_dir": "/srv/latentguard-vla",
        "branch": "codex/m0-data-contract",
        "expected_commit": SHA,
        "ssh_executable": "ssh-custom",
        "connect_timeout": 12,
    }
    values.update(overrides)
    return RemoteSyncConfig(**values)


def _completed(
    stdout: str = "", stderr: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=stderr)


class _QueueRunner:
    def __init__(
        self,
        responses: Sequence[
            subprocess.CompletedProcess[str]
            | subprocess.CalledProcessError
            | subprocess.TimeoutExpired
        ],
    ) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(
        self, command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append((command, kwargs))
        response = self.responses.pop(0)
        if isinstance(
            response, (subprocess.CalledProcessError, subprocess.TimeoutExpired)
        ):
            raise response
        return response


def _successful_local_responses() -> list[subprocess.CompletedProcess[str]]:
    return [
        _completed(),
        _completed("codex/m0-data-contract\n"),
        _completed("origin/codex/m0-data-contract\n"),
        _completed(f"{SHA}\n"),
        _completed(f"{SHA}\n"),
    ]


def test_resolve_remote_config_prefers_arguments_then_environment() -> None:
    config = resolve_remote_config(
        host="argument-host",
        repo_dir=None,
        branch=None,
        expected_commit=None,
        ssh_executable=None,
        connect_timeout=None,
        environ={
            "LATENTGUARD_REMOTE_HOST": "environment-host",
            "LATENTGUARD_REMOTE_REPO": "/srv/repository",
            "LATENTGUARD_REMOTE_BRANCH": "feature/m0",
            "LATENTGUARD_REMOTE_COMMIT": SHA.upper(),
            "LATENTGUARD_REMOTE_SSH_EXECUTABLE": "ssh-alt",
            "LATENTGUARD_REMOTE_CONNECT_TIMEOUT": "9",
        },
    )

    assert config.host == "argument-host"
    assert config.repo_dir == "/srv/repository"
    assert config.branch == "feature/m0"
    assert config.expected_commit == SHA
    assert config.ssh_executable == "ssh-alt"
    assert config.connect_timeout == 9


def test_resolve_remote_config_reports_missing_values() -> None:
    with pytest.raises(ValueError, match=r"host.*repo_dir.*branch.*expected_commit"):
        resolve_remote_config(
            host=None,
            repo_dir=None,
            branch=None,
            expected_commit=None,
            ssh_executable=None,
            connect_timeout=None,
            environ={},
        )


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"host": "-oProxyCommand=bad"}, "host"),
        ({"host": "user:credential-value@training-host"}, "host"),
        ({"host": "training-host\n"}, "host"),
        ({"repo_dir": "/srv/repo\nmalicious"}, "repo_dir"),
        ({"repo_dir": "/srv/repo\n"}, "repo_dir"),
        ({"repo_dir": "/srv/repo\x1b[2Jspoof"}, "repo_dir"),
        ({"repo_dir": "/srv/repo\tspoof"}, "repo_dir"),
        ({"repo_dir": "/srv/repo\u2028spoof"}, "repo_dir"),
        ({"repo_dir": "/srv/repo\u2029spoof"}, "repo_dir"),
        ({"branch": "../bad"}, "branch"),
        ({"branch": "feature/topic\u2028"}, "branch"),
        ({"branch": "feature/.hidden"}, "branch"),
        ({"branch": "feature.lock/topic"}, "branch"),
        ({"expected_commit": "abc"}, "expected_commit"),
        ({"expected_commit": SHA + "\n"}, "expected_commit"),
        ({"ssh_executable": "ssh\t"}, "ssh_executable"),
        ({"connect_timeout": 0}, "connect_timeout"),
        ({"connect_timeout": True}, "connect_timeout"),
        ({"connect_timeout": 1.5}, "connect_timeout"),
    ],
)
def test_remote_config_rejects_unsafe_values(
    overrides: dict[str, Any], field: str
) -> None:
    with pytest.raises(ValueError, match=field):
        _config(**overrides)


def test_dry_run_uses_no_subprocess_and_sanitizes_output() -> None:
    runner = _QueueRunner([])
    config = _config(repo_dir="/srv/api_key=do-not-print/repository")

    result = sync_remote(config, dry_run=True, runner=runner)

    assert result.returncode == 0
    assert result.dry_run is True
    assert runner.calls == []
    assert config.branch in str(result)
    assert config.expected_commit in str(result)
    assert "do-not-print" not in str(result)
    assert "do-not-print" not in repr(result)
    assert "do-not-print" not in repr(config)
    assert "[REDACTED]" in str(result)


@pytest.mark.parametrize(
    "repo_dir",
    [
        "/srv/password='alpha beta'/repository",
        '/srv/authorization="Bearer alpha beta"/repository',
        "/srv/passphrase=alpha beta/repository",
        "/srv/pwd='alpha beta'/repository",
        "/srv/pass=alpha beta/repository",
        "/srv/credentials=alpha beta/repository",
        "/srv/secret_key=alpha beta/repository",
        "/srv/secret_access_key=alpha beta/repository",
        "/srv/aws_secret_access_key=alpha beta/repository",
        "/srv/ssh_key=alpha beta/repository",
    ],
)
def test_dry_run_redacts_quoted_or_multiword_secrets(repo_dir: str) -> None:
    result = sync_remote(_config(repo_dir=repo_dir), dry_run=True)

    assert "alpha" not in str(result)
    assert "beta" not in str(result)
    assert "alpha" not in repr(result)
    assert "beta" not in repr(result)
    assert "[REDACTED]" in str(result)


def test_dry_run_does_not_over_redact_noncredential_key_suffixes() -> None:
    repo_dir = "/srv/compass=alpha beta/repository"

    result = sync_remote(_config(repo_dir=repo_dir), dry_run=True)

    assert repo_dir in str(result)


def test_ssh_command_is_an_argument_list_with_safe_remote_steps() -> None:
    config = _config(repo_dir="/srv/repo with spaces")

    command = build_ssh_command(config)
    script = build_remote_script(config)

    assert command == (
        "ssh-custom",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=12",
        "training-host",
        script,
    )
    assert 'git -C "$repo" rev-parse --is-inside-work-tree' in script
    assert '[ "$work_tree" != "true" ]' in script
    assert 'cd -- "$repo"' in script
    assert "git status --porcelain --untracked-files=no" in script
    assert "command -v python3" in script
    assert "git fetch --prune origin" in script
    assert 'git checkout "$branch"' in script
    assert 'git pull --ff-only origin "$branch"' in script
    assert 'actual_branch="$(git branch --show-current)"' in script
    assert 'if [ "$actual_branch" != "$branch" ]; then' in script
    assert 'actual_commit="$(git rev-parse HEAD)"' in script
    assert 'if [ "$actual_commit" != "$expected_commit" ]; then' in script
    assert script.count("git status --porcelain --untracked-files=no") == 2
    assert script.rindex("git status --porcelain --untracked-files=no") > script.index(
        'if [ "$actual_commit" != "$expected_commit" ]; then'
    )
    assert "exit 40" in script
    assert "exit 41" in script
    assert "exit 42" in script
    assert "exit 43" in script
    assert "exit 44" in script
    assert script.index('cd -- "$repo"') < script.index("git fetch --prune origin")
    assert script.index('actual_commit="$(git rev-parse HEAD)"') < script.index(
        'if [ "$actual_commit" != "$expected_commit" ]; then'
    )
    assert "git reset" not in script
    assert "git clean" not in script


def test_real_sync_checks_local_state_and_runs_ssh_without_a_shell() -> None:
    runner = _QueueRunner([*_successful_local_responses(), _completed("ok\n")])

    result = sync_remote(_config(), runner=runner)

    assert result.returncode == 0
    assert result.dry_run is False
    assert len(runner.calls) == 6
    assert runner.calls[0][0] == ["git", "status", "--porcelain"]
    assert runner.calls[-1][0][0] == "ssh-custom"
    assert all(call_kwargs["shell"] is False for _, call_kwargs in runner.calls)
    assert all(
        call_kwargs["stdin"] is subprocess.DEVNULL for _, call_kwargs in runner.calls
    )
    assert all(call_kwargs["timeout"] == 600 for _, call_kwargs in runner.calls)
    assert all(isinstance(command, list) for command, _ in runner.calls)


def test_dirty_local_repository_is_rejected_before_ssh() -> None:
    runner = _QueueRunner([_completed(" M src/latentguard/models.py\n")])

    with pytest.raises(RemoteSyncError, match="dirty local repository"):
        sync_remote(_config(), runner=runner)

    assert len(runner.calls) == 1


def test_missing_upstream_is_rejected_before_ssh() -> None:
    runner = _QueueRunner(
        [
            _completed(),
            _completed("codex/m0-data-contract\n"),
            _completed(stderr="no upstream", returncode=128),
        ]
    )

    with pytest.raises(RemoteSyncError, match="requires a local upstream"):
        sync_remote(_config(), runner=runner)

    assert len(runner.calls) == 3


def test_configured_branch_must_equal_current_local_branch() -> None:
    runner = _QueueRunner([_completed(), _completed("another-branch\n")])

    with pytest.raises(RemoteSyncError, match="current local branch"):
        sync_remote(_config(), runner=runner)

    assert len(runner.calls) == 2


def test_local_and_upstream_sha_mismatch_is_rejected() -> None:
    other_sha = "f" * 40
    runner = _QueueRunner(
        [
            _completed(),
            _completed("codex/m0-data-contract\n"),
            _completed("origin/feature\n"),
            _completed(f"{SHA}\n"),
            _completed(f"{other_sha}\n"),
        ]
    )

    with pytest.raises(RemoteSyncError, match="does not match its upstream"):
        sync_remote(_config(), runner=runner)

    assert len(runner.calls) == 5


def test_expected_sha_must_equal_local_head() -> None:
    runner = _QueueRunner(_successful_local_responses())

    with pytest.raises(RemoteSyncError, match="does not equal local HEAD"):
        sync_remote(_config(expected_commit="f" * 40), runner=runner)

    assert len(runner.calls) == 5


@pytest.mark.parametrize(
    ("returncode", "stderr", "message"),
    [
        (41, "remote repository has tracked modifications", "tracked remote"),
        (42, "remote HEAD does not match expected commit", "unexpected remote HEAD"),
        (43, "python3 is unavailable on the remote host", "requires remote python3"),
        (
            44,
            "remote branch does not match requested branch",
            "unexpected remote branch",
        ),
    ],
)
def test_remote_guard_failures_are_propagated(
    returncode: int, stderr: str, message: str
) -> None:
    runner = _QueueRunner(
        [
            *_successful_local_responses(),
            _completed(stderr=stderr, returncode=returncode),
        ]
    )

    with pytest.raises(RemoteSyncError, match=message) as error:
        sync_remote(_config(), runner=runner)

    assert error.value.returncode == returncode


def test_ssh_subprocess_exception_preserves_return_code_and_redacts_secrets() -> None:
    command_secret = "trace-command-secret"
    stderr_secret = "alpha beta"
    failure = subprocess.CalledProcessError(
        255,
        ["ssh", f"/srv/password={command_secret}"],
        stderr=(
            f'\x1b[2J connection failed {{"Authorization": "Bearer {stderr_secret}"}}'
        ),
    )
    runner = _QueueRunner([*_successful_local_responses(), failure])

    with pytest.raises(RemoteSyncError) as error:
        sync_remote(_config(), runner=runner)

    assert error.value.returncode == 255
    formatted = "".join(traceback.format_exception(error.value))
    assert command_secret not in str(error.value)
    assert command_secret not in formatted
    assert stderr_secret not in str(error.value)
    assert stderr_secret not in formatted
    assert "\x1b" not in str(error.value)
    assert "\x1b" not in formatted
    assert "[REDACTED]" in str(error.value)


def test_subprocess_error_redacts_quoted_fields_and_ambiguous_userinfo() -> None:
    failure = subprocess.CalledProcessError(
        255,
        ["ssh"],
        stderr=(
            "{'token': 'quoted secret'} "
            'user:ambiguous@userinfo@host password="final secret"'
        ),
    )
    runner = _QueueRunner([*_successful_local_responses(), failure])

    with pytest.raises(RemoteSyncError) as error:
        sync_remote(_config(), runner=runner)

    message = str(error.value)
    assert "quoted secret" not in message
    assert "ambiguous" not in message
    assert "userinfo" not in message
    assert "final secret" not in message
    assert "[REDACTED]" in message


def test_ssh_subprocess_timeout_is_wrapped() -> None:
    timeout = subprocess.TimeoutExpired(["ssh"], 600)
    runner = _QueueRunner([*_successful_local_responses(), timeout])

    with pytest.raises(RemoteSyncError, match="exceeded 600 seconds") as error:
        sync_remote(_config(), runner=runner)

    assert error.value.returncode is None
