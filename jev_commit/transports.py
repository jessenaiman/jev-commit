"""Explicit, provider-pinned judgments. Local means Ollaya CLI, never its HTTP API."""

import json
import math
import os
import subprocess
import time
from datetime import datetime, timezone

from jev_commit import git, jev

MODELS = {"ollaya": "kev:0.8b", "jev": "jev-1.13.0", "openrouter": "typesafe/jev-1.13"}


def positive_seconds(value):
    import argparse
    try:
        number = float(value)
        if math.isfinite(number) and number > 0:
            return number
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("must be a finite positive number")


def provider_args(parser, timeout=30):
    parser.add_argument("--provider", choices=tuple(MODELS), default="ollaya",
                        help="explicit transport; default: Ollaya CLI, no hosted fallback")
    parser.add_argument("--model", help="exact model ID (default depends on provider)")
    parser.add_argument("--timeout", type=positive_seconds, default=timeout,
                        help="total inference deadline in seconds")


def sanitize(value, env):
    """Remove known credential values from saved reports and service error text."""
    secrets = [env[name] for name in git.SECRET_ENV if name.endswith(("KEY", "TOKEN")) and env.get(name)]
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, dict):
        return {sanitize(k, env): sanitize(v, env) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v, env) for v in value]
    return value


def normalize(body, questions):
    if not isinstance(body, dict) or body.get("state_truncated"):
        raise jev.JevError("invalid response or truncated state")
    try:
        json.dumps(body, allow_nan=False)
    except (ValueError, TypeError) as err:
        raise jev.JevError("response contains non-JSON or nonfinite data") from err
    answers = body.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(questions):
        raise jev.JevError("response question set differs from request")
    values = {}
    for name, answer in answers.items():
        if not isinstance(answer, dict) or answer.get("type") != "noul":
            raise jev.JevError("expected a typed Noul answer")
        value = answer.get("noul")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise jev.JevError("Noul probability must be finite and between zero and one")
        values[name] = float(value)
    usage = body.get("usage", {})
    if not isinstance(usage, dict):
        raise jev.JevError("usage must be an object")
    for name in ("input_tokens", "output_tokens"):
        if name in usage and (type(usage[name]) is not int or usage[name] < 0):
            raise jev.JevError("invalid token usage")
    return values, usage


class Transport:
    def __init__(self, provider="ollaya", model=None, env=None, log=None):
        if provider not in MODELS:
            raise ValueError("unknown provider")
        self.provider = provider
        self.model = model or MODELS[provider]
        if not isinstance(self.model, str) or not self.model.strip() or self.model.startswith("-"):
            raise ValueError("invalid model ID")
        self.env = dict(os.environ if env is None else env)
        self.log = log
        self.requests = self.completed = self.failed = self.cancelled = 0

    def event(self, event, sequence, **fields):
        if self.log:
            row = {"event": event, "sequence": sequence, "at": datetime.now(timezone.utc).isoformat(),
                   "provider": self.provider, **fields}
            with self.log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(sanitize(row, self.env), allow_nan=False) + "\n")

    def _one(self, state, questions, seconds):
        sequence = self.requests
        self.requests += 1
        body = {"model": self.model, "state": state, "questions": questions}
        self.event("started", sequence, request=body)
        started = time.monotonic()
        try:
            if self.provider == "ollaya":
                # Small serial contexts match PlayJev's bounded local-memory contract.
                child_env = git.child_env(self.env)
                result = subprocess.run(
                    ["ollaya", "run", self.model, "--format", "json", "--state-json",
                     "--questions", json.dumps(questions)],
                    input=json.dumps(state).encode(), capture_output=True, env=child_env, timeout=seconds,
                )
                if result.returncode:
                    text = (result.stderr or b"").decode(errors="replace").lower()
                    kind = "allocation-limit" if "allocate memory" in text or "gpu allocation" in text else "unsupported-feature" if "unsupported" in text else "inference-error"
                    raise jev.JevError("Ollaya CLI %s (exit %d)" % (kind, result.returncode))
                response = json.loads(result.stdout)
            else:
                env = dict(self.env)
                if self.provider == "openrouter":
                    key = env.get("OPENROUTER_API_KEY")
                    if not key:
                        raise jev.JevError("OPENROUTER_API_KEY is required for the selected provider")
                    env.update(TYPESAFE_API_KEY=key, JEV_BASE_URL="https://openrouter.ai/api")
                elif not (env.get("TYPESAFE_API_KEY") or env.get("JEV_API_KEY")):
                    # A loopback fake is permitted for offline regression tests only.
                    from urllib.parse import urlsplit
                    if urlsplit(jev.base_url(env)).hostname not in jev.LOOPBACK:
                        raise jev.JevError("TypeSafe API key is required for the selected provider")
                response = jev.ask(state, questions, env=env, model=self.model, deadline_s=seconds)["response"]
            answers, usage = normalize(response, questions)
        except (Exception, KeyboardInterrupt) as err:
            if isinstance(err, KeyboardInterrupt):
                self.cancelled += 1
            else:
                self.failed += 1
            # Do not persist service/CLI text that may echo a sensitive request.
            safe_reasons = ("OPENROUTER_API_KEY is required for the selected provider",
                            "TypeSafe API key is required for the selected provider")
            reason = str(err) if isinstance(err, jev.JevError) and (self.provider == "ollaya" or str(err) in safe_reasons) else type(err).__name__
            self.event("cancelled" if isinstance(err, KeyboardInterrupt) else "failed", sequence, error=reason)
            if isinstance(err, KeyboardInterrupt):
                raise
            if isinstance(err, jev.TooBig):
                raise jev.TooBig("selected provider rejected the context size") from err
            raise jev.JevError("%s request failed (%s); no provider fallback" % (self.provider, reason)) from err
        self.completed += 1
        ms = round((time.monotonic() - started) * 1000)
        self.event("completed", sequence, response=response, ms=ms)
        return {"answers": answers, "usage": usage, "model": response.get("model", "unknown"), "ms": ms}

    def ask(self, state, questions, env=None, deadline_s=30):
        stop = time.monotonic() + deadline_s
        parts = [{key: question} for key, question in questions.items()] if self.provider == "ollaya" else [questions]
        combined = {"answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0}, "model": self.model, "ms": 0}
        for part in parts:
            left = stop - time.monotonic()
            if left <= 0:
                raise jev.JevError("inference deadline reached; no provider fallback")
            out = self._one(state, part, left)
            combined["answers"].update(out["answers"])
            for key in combined["usage"]:
                combined["usage"][key] += out["usage"].get(key, 0)
            combined["model"] = out["model"]
            combined["ms"] += out["ms"]
        return combined
