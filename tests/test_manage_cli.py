import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANAGE = PROJECT_ROOT / "manage.sh"


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
        }
    )
    return subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_help_exits_zero_and_mentions_commands():
    proc = run(["bash", str(MANAGE), "--help"])
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "Commands (non-interactive):" in out
    assert "doctor" in out
    assert "dev ui" in out


def test_self_test_dry_runs_without_docker():
    # --self-test should not require docker and exit 0
    proc = run(["bash", str(MANAGE), "--self-test"])
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout + proc.stderr
    assert "Self-test complete" in out


def test_usage_mentions_new_commands():
    proc = run(["bash", str(MANAGE), "--help"])
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "start-all" in out
    assert "stop-all" in out
    assert "restart-all" in out
    assert "build-restart" in out


def test_new_commands_dry_run():
    env_cmd = ["bash", str(MANAGE), "--dry-run"]
    for args in (
        ["start-all"],
        ["stop-all"],
        ["restart-all"],
        ["build-restart", "all"],
    ):
        proc = run(env_cmd + args)
        assert proc.returncode == 0, proc.stderr
