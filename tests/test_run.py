import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from conftest import commit, git, write
from jev_commit import cli, jev, run, transports


def clean_response(questions, value=None):
    return {"model": "mock-kev", "answers": {name: {"type": "noul", "noul": value if value is not None else
            0.95 if name.startswith("message_") else 0.02} for name in questions},
            "usage": {"input_tokens": 12, "output_tokens": 0}}


@pytest.fixture
def staged(repo):
    commit(repo, "a.py", "x = 1\n", "add a")
    write(repo, "a.py", "x = 2\n")
    git(repo, "add", "a.py")
    return repo


@pytest.fixture
def mock_cli(monkeypatch):
    real_run = transports.subprocess.run
    calls = []

    def execute(argv, **kwargs):
        if argv[0] != "ollaya":
            return real_run(argv, **kwargs)
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=json.dumps(clean_response(json.loads(argv[-1]))).encode())

    monkeypatch.setattr(transports.subprocess, "run", execute)
    return calls


def read_run(repo):
    folders = list((repo / ".git" / "jev-commit-runs").iterdir())
    folder = folders[-1]
    return json.loads((folder / "summary.json").read_text()), folder


def check(repo, *args):
    return cli.main(["check", "--repo", str(repo), "-m", "fix: bump x", *args])


def test_one_run_preserves_head_index_and_worktree(staged, mock_cli):
    index = staged / ".git" / "index"
    before = hashlib.sha256(index.read_bytes()).hexdigest()
    head = git(staged, "rev-parse", "HEAD").stdout
    assert check(staged) == 0
    summary, folder = read_run(staged)
    assert summary["status"] == "ready"
    assert summary["checks"]["tests"]["status"] == "not-requested"
    assert summary["requests"] == {"started": 5, "completed": 5, "failed": 0, "cancelled": 0}
    assert not summary["git_writes"] and not summary["automatic_hosted_fallback"]
    assert hashlib.sha256(index.read_bytes()).hexdigest() == before
    assert git(staged, "rev-parse", "HEAD").stdout == head
    assert (staged / "a.py").read_text() == "x = 2\n"
    assert len((folder / "inference.jsonl").read_text().splitlines()) == 10


def test_empty_index_does_not_replay_last_commit(repo, mock_cli):
    commit(repo, "a.py", "x = 1\n", "add a")
    commit(repo, "b.py", "y = 2\n", "add b")
    assert check(repo) == run.HOLD
    assert not mock_cli
    assert read_run(repo)[0]["staged_files"] == []


@pytest.mark.parametrize("name", ["runs/frame.png", "roms/game.a26", "save.state", ".env", ".env.local"])
def test_playjev_forbidden_paths_block_before_any_inference(staged, mock_cli, name):
    write(staged, name, "private artifact\n")
    git(staged, "add", name)
    assert check(staged, "--profile", "playjev") == cli.BLOCK
    summary, folder = read_run(staged)
    assert name in summary["checks"]["forbidden_paths"]
    assert not mock_cli and not (folder / "inference.jsonl").exists()


def test_private_key_is_not_uploaded_or_saved(staged, mock_cli):
    write(staged, "a.py", "-----BEGIN OPENSSH PRIVATE KEY-----\nprivate-body\n")
    git(staged, "add", "a.py")
    assert check(staged) == cli.BLOCK
    summary, folder = read_run(staged)
    assert summary["belt_hits"][0]["precision"] == "high"
    assert "PRIVATE KEY" not in (folder / "summary.json").read_text()
    assert "private-body" not in (folder / "summary.json").read_text()
    assert not mock_cli


def test_known_key_in_message_blocks_and_never_gets_logged(staged, mock_cli, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "unique-hidden-value")
    assert cli.main(["check", "--repo", str(staged), "-m", "unique-hidden-value"]) == cli.BLOCK
    summary, folder = read_run(staged)
    assert summary["checks"]["known_credential_found"]
    assert "unique-hidden-value" not in (folder / "summary.json").read_text()
    assert not mock_cli


def test_whitespace_failure_skips_inference(staged, mock_cli):
    write(staged, "a.py", "x = 2  \n")
    git(staged, "add", "a.py")
    assert check(staged) == run.HOLD
    assert not mock_cli


@pytest.mark.parametrize("command, expected", [([sys.executable, "-c", "pass"], 0),
                                              ([sys.executable, "-c", "raise SystemExit(1)"], run.HOLD)])
def test_explicit_tests_own_pass_fail_gate(staged, mock_cli, command, expected):
    assert check(staged, "--test", *command) == expected
    summary, _ = read_run(staged)
    assert summary["checks"]["tests"]["returncode"] == (0 if expected == 0 else 1)
    assert bool(mock_cli) == (expected == 0)


def test_tests_on_unstaged_changes_cannot_authorize_staged_snapshot(staged, mock_cli):
    write(staged, "a.py", "x = 3\n")
    assert check(staged, "--test", sys.executable, "-c", "pass") == run.HOLD
    assert not mock_cli


def test_local_failure_is_hold_not_hook_fail_open(staged, mock_cli, monkeypatch):
    monkeypatch.setattr(transports.Transport, "_one", lambda *a, **k: (_ for _ in ()).throw(jev.JevError("allocation-limit")))
    assert check(staged) == run.HOLD
    assert read_run(staged)[0]["status"] == "hold"


def test_midrange_judgments_require_review(staged, mock_cli, monkeypatch):
    monkeypatch.setattr(transports.Transport, "_one", lambda self, state, questions, seconds:
                        {"answers": {name: 0.5 for name in questions}, "model": "mock", "usage": {}, "ms": 0})
    assert check(staged) == run.REVIEW
    assert read_run(staged)[0]["status"] == "review"


def test_staging_during_inference_rejects_stale_review(staged, mock_cli, monkeypatch):
    original = transports.Transport._one
    mutated = False

    def mutate(self, *args):
        nonlocal mutated
        result = original(self, *args)
        if not mutated:
            mutated = True
            write(staged, "a.py", "x = 3\n")
            git(staged, "add", "a.py")
        return result

    monkeypatch.setattr(transports.Transport, "_one", mutate)
    assert check(staged) == run.HOLD
    assert not read_run(staged)[0]["checks"]["snapshot_unchanged"]
    assert (staged / "a.py").read_text() == "x = 3\n", "must not roll back user's work"


def test_output_directory_is_never_overwritten(staged, tmp_path, mock_cli):
    out = tmp_path / "existing"
    out.mkdir()
    marker = out / "summary.json"
    marker.write_text("keep me")
    assert check(staged, "--out", str(out)) == cli.USAGE
    assert marker.read_text() == "keep me"
    assert not mock_cli


def test_playjev_profile_is_just_offline_unittests(tmp_path):
    python = write(tmp_path, ".venv/bin/python", "")
    assert run.test_command(tmp_path, "playjev", None) == [str(python), "-m", "unittest", "discover", "-s", "tests", "-q"]


def test_unexpected_runner_error_is_saved_as_hold(staged, mock_cli, monkeypatch):
    monkeypatch.setattr(run, "test_command", lambda *a: (_ for _ in ()).throw(RuntimeError("unexpected error")))
    assert check(staged) == run.HOLD
    assert read_run(staged)[0]["status"] == "hold"


def test_test_timeout_is_a_hold(staged, mock_cli):
    assert check(staged, "--test-timeout", "0.05", "--test", sys.executable, "-c", "import time; time.sleep(2)") == run.HOLD
    assert not mock_cli


def test_missing_router_key_is_saved_as_hold(staged, mock_cli, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert check(staged, "--provider", "openrouter") == run.HOLD
    summary, _ = read_run(staged)
    assert "OPENROUTER_API_KEY is required" in summary["inference_error"]
    assert summary["requests"]["failed"] == 1


def test_example_env_template_is_not_a_forbidden_artifact():
    assert run.forbidden_paths([".env.example"], "playjev") == []


def test_repo_subdirectory_resolves_whole_repository(staged, mock_cli):
    nested = staged / "src"
    nested.mkdir()
    assert check(nested) == 0
    assert read_run(staged)[0]["repo"] == str(staged)


def test_mixed_capture_cannot_upload_forbidden_artifacts(staged, mock_cli, monkeypatch):
    original = run.git.capture_staged

    def mixed(repo):
        capture = original(repo)
        capture["patch"] += "diff --git a/runs/secret.txt b/runs/secret.txt\n--- /dev/null\n+++ b/runs/secret.txt\n@@ -0,0 +1 @@\n+sensitive evidence\n"
        return capture

    monkeypatch.setattr(run.git, "capture_staged", mixed)
    assert check(staged, "--profile", "playjev") == cli.BLOCK
    assert not mock_cli
