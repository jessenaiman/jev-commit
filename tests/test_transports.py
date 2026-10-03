import json
import subprocess
from types import SimpleNamespace

import pytest

from jev_commit import cli, jev, transports
from jev_commit.questions import QUESTIONS


def response(questions, value=0.1):
    return {"model": "tested-model", "answers": {k: {"type": "noul", "noul": value} for k in questions},
            "usage": {"input_tokens": 10, "output_tokens": 0}}


def test_defaults_are_explicit_local():
    args = cli.parse_args(["msg"])
    assert args.provider == "ollaya"
    assert transports.Transport().model == "kev:0.8b"


def test_cli_argv_stdin_secret_isolation_and_serial_questions(monkeypatch, tmp_path):
    calls = []
    env = {"PATH": "/bin", "TYPESAFE_API_KEY": "private-one", "OPENROUTER_API_KEY": "private-two",
           "GITHUB_TOKEN": "private-three"}

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps(response(json.loads(argv[-1]))).encode())

    monkeypatch.setattr(transports.subprocess, "run", run)
    transport = transports.Transport(env=env, log=tmp_path / "inference.jsonl")
    out = transport.ask({"message": "fix: spaces; $(touch /evil)"}, QUESTIONS)
    assert len(calls) == len(QUESTIONS) == transport.completed
    for argv, kwargs in calls:
        assert argv[:5] == ["ollaya", "run", "kev:0.8b", "--format", "json"]
        assert len(json.loads(argv[-1])) == 1
        assert json.loads(kwargs["input"])["message"].endswith("$(touch /evil)")
        assert not any(k in kwargs["env"] for k in ("TYPESAFE_API_KEY", "OPENROUTER_API_KEY", "GITHUB_TOKEN"))
    assert out["usage"]["input_tokens"] == 50
    events = [json.loads(line) for line in transport.log.read_text().splitlines()]
    assert [r["event"] for r in events] == ["started", "completed"] * 5


@pytest.mark.parametrize("failure", [subprocess.TimeoutExpired("ollaya", 1), FileNotFoundError(),
                                    ValueError("bad output")])
def test_local_failures_do_not_call_http(monkeypatch, failure, tmp_path):
    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(transports.subprocess, "run", fail)
    monkeypatch.setattr(jev, "ask", lambda *a, **k: pytest.fail("hosted fallback"))
    t = transports.Transport(log=tmp_path / "requests.jsonl")
    with pytest.raises(jev.JevError, match="no provider fallback"):
        t.ask({}, {"q": {"type": "noul"}})
    assert (t.requests, t.completed, t.failed) == (1, 0, 1)
    assert json.loads(t.log.read_text().splitlines()[-1])["event"] == "failed"


@pytest.mark.parametrize("value", [True, -0.1, 1.1, float("nan"), float("inf"), "0.1"])
def test_invalid_probabilities_are_never_judgments(value):
    with pytest.raises(jev.JevError):
        transports.normalize(response({"q": {}}, value), {"q": {}})


def test_wrong_questions_truncation_and_types_are_rejected():
    for body in (response({"other": {}}), {**response({"q": {}}), "state_truncated": True},
                 {"answers": {"q": {"type": "choice", "noul": 0.1}}}):
        with pytest.raises(jev.JevError):
            transports.normalize(body, {"q": {}})


def test_openrouter_uses_its_key_and_endpoint_only_when_selected(monkeypatch):
    calls = []

    def ask(state, questions, **kwargs):
        calls.append(kwargs)
        return {"response": response(questions)}

    monkeypatch.setattr(jev, "ask", ask)
    env = {"OPENROUTER_API_KEY": "router-private", "TYPESAFE_API_KEY": "typesafe-private",
           "JEV_BASE_URL": "https://unwanted.example"}
    t = transports.Transport("openrouter", env=env)
    t.ask({}, QUESTIONS)
    assert len(calls) == 1
    assert calls[0]["model"] == "typesafe/jev-1.13"
    assert calls[0]["env"]["JEV_BASE_URL"] == "https://openrouter.ai/api"
    assert calls[0]["env"]["TYPESAFE_API_KEY"] == "router-private"


def test_missing_openrouter_key_does_not_use_typesafe_key(monkeypatch):
    monkeypatch.setattr(jev, "ask", lambda *a, **k: pytest.fail("wrong provider/key"))
    with pytest.raises(jev.JevError):
        transports.Transport("openrouter", env={"TYPESAFE_API_KEY": "private"}).ask({}, QUESTIONS)


def test_allocation_failure_is_classified_without_persisting_cli_output(monkeypatch, tmp_path):
    monkeypatch.setattr(transports.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=1, stdout=b"", stderr=b"Failed to allocate memory for buffer; sensitive-state"))
    t = transports.Transport(log=tmp_path / "requests.jsonl")
    with pytest.raises(jev.JevError, match="allocation-limit"):
        t.ask({}, QUESTIONS)
    assert "sensitive-state" not in t.log.read_text()


def test_nonfinite_optional_response_fields_are_rejected():
    body = response({"q": {}})
    body["usage"]["cost"] = float("nan")
    with pytest.raises(jev.JevError):
        transports.normalize(body, {"q": {}})


def test_cancelled_request_is_logged(monkeypatch, tmp_path):
    def cancel(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(transports.subprocess, "run", cancel)
    t = transports.Transport(log=tmp_path / "requests.jsonl")
    with pytest.raises(KeyboardInterrupt):
        t.ask({}, QUESTIONS)
    assert (t.cancelled, t.failed) == (1, 0)
    assert json.loads(t.log.read_text().splitlines()[-1])["event"] == "cancelled"


def test_known_credentials_are_redacted_recursively():
    assert transports.sanitize({"error": ["key=private"]}, {"OPENROUTER_API_KEY": "private"}) == {"error": ["key=[REDACTED]"]}
