"""One bounded check-in run: snapshot -> checks -> typed judgments -> saved report.

This is not a Git writer, a code-correctness certificate, or a gameplay evaluator.
Unlike the optional hook, incomplete inference never becomes a successful check.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
from datetime import datetime, timezone
from uuid import uuid4

from jev_commit import belt, chunk, git, jev
from jev_commit.cli import BLOCK, USAGE, decide, judge, thresholds
from jev_commit.report import Report
from jev_commit.transports import Transport, provider_args, positive_seconds, sanitize

HOLD = 3
REVIEW = 4


def fingerprint(captured):
    return hashlib.sha256((captured["name_status"] + "\0" + captured["patch"]).encode()).hexdigest()


def forbidden_paths(paths, profile):
    result = []
    for name in paths:
        path = PurePosixPath(name)
        environment = path.name == ".env" or (path.name.startswith(".env.") and path.name not in (".env.example", ".env.sample"))
        artifact = profile == "playjev" and (set(path.parts) & {"runs", "node_modules", ".venv"} or
                    path.suffix.lower() in (".a26", ".bin", ".rom", ".state", ".webm", ".mp4"))
        if environment or artifact:
            result.append(name)
    return result


def test_command(repo, profile, override):
    if override:
        return override
    if profile == "playjev":
        python = repo / ".venv" / "bin" / "python"
        if not python.is_file():
            raise ValueError("PlayJev profile requires its .venv/bin/python (or explicit --test argv)")
        return [str(python), "-m", "unittest", "discover", "-s", "tests", "-q"]
    return None


def run_tests(argv, repo, env, seconds):
    if not argv:
        return {"status": "not-requested", "note": "No tests run; this is only a staged-message review."}
    try:
        process = subprocess.run(argv, cwd=repo, env=git.child_env(env), stdin=subprocess.DEVNULL,
                                 capture_output=True, timeout=seconds)
        return {"status": "passed" if process.returncode == 0 else "failed", "command": argv,
                "returncode": process.returncode}
    except (OSError, subprocess.TimeoutExpired) as err:
        return {"status": "failed", "command": argv, "error": type(err).__name__}


def execute(args):
    env = dict(os.environ)
    top, _, _ = git.run_git(["rev-parse", "--show-toplevel"], cwd=args.repo.resolve())
    repo = Path(top.strip()).resolve()
    captured = git.capture_staged(repo)
    snapshot_hash = fingerprint(captured)
    paths = git.parse_name_status(captured["name_status"])
    head, _, _ = git.run_git(["rev-parse", "--verify", "HEAD"], cwd=repo, check=False)
    branch, _, _ = git.run_git(["branch", "--show-current"], cwd=repo)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    if args.out:
        out = args.out.resolve()
    else:
        git_path, _, _ = git.run_git(["rev-parse", "--git-path", "jev-commit-runs"], cwd=repo)
        out = Path(git_path.strip())
        if not out.is_absolute():
            out = repo / out
        out = out / run_id
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    report = Report()
    transport = Transport(args.provider, args.model, env=env, log=out / "inference.jsonl")
    summary = {"schema": "jev-commit-run-v1", "run_id": run_id, "provider": args.provider,
               "model": transport.model, "profile": args.profile, "repo": str(repo),
               "head": head.strip() or None, "branch": branch.strip(), "staged_sha256": snapshot_hash,
               "staged_files": list(paths), "checks": {}, "status": "hold", "git_writes": False,
               "automatic_hosted_fallback": False,
               "note": "Judgments concern the staged message/diff, not gameplay success or code correctness. No commit/push authorization is inferred."}
    code = HOLD
    try:
        message = args.message if args.message is not None else args.message_file.read_text(encoding="utf-8")
        message = git.strip_message(message, git.comment_prefix(repo))
        prep = chunk.prepare(captured["patch"], paths)
        message_hunks = [{"path": "<commit-message>", "text": "\n".join("+" + line for line in message.splitlines())}]
        hits = belt.scan(prep["all_hunks"] + message_hunks)
        # Save metadata only, never the original matched line containing a credential.
        summary["belt_hits"] = [{k: hit[k] for k in ("kind", "path", "precision", "redacted")} for hit in hits]
        parsed_paths = {f["path"] for f in chunk.parse_patch(captured["patch"])}
        consistent = parsed_paths == set(paths)
        forbidden = forbidden_paths(set(paths) | parsed_paths, args.profile)
        secret_found = any(env.get(k) and env[k] in (message + captured["patch"]) for k in git.SECRET_ENV if k.endswith(("KEY", "TOKEN")))
        summary["checks"].update(nonempty_staged=bool(paths), message_present=bool(message),
                                 complete_capture=not captured["truncated"], forbidden_paths=forbidden,
                                 known_credential_found=bool(secret_found), path_table_consistent=consistent)
        report.line("jev-commit check · %s/%s · %d staged files" % (args.provider, transport.model, len(paths)))
        if forbidden or secret_found or belt.blocking(hits):
            summary["status"] = "blocked"
            code = BLOCK
        elif not paths or not message or captured["truncated"] or not consistent:
            summary["reason"] = "empty staged changes/message or incomplete capture"
        else:
            # No arbitrary commands are pulled from repo config; tests require a named profile or argv.
            whitespace, result, _ = git.run_git(["diff", "--cached", "--check"] + git.DIFF_FLAGS, cwd=repo, check=False)
            summary["checks"]["whitespace"] = {"passed": result == 0, "output": sanitize(whitespace, env)}
            argv = test_command(repo, args.profile, args.test)
            summary["checks"]["tests"] = run_tests(argv, repo, env, args.test_timeout)
            dirty, _, _ = git.run_git(["diff", "--name-only", "-z"] + git.DIFF_FLAGS, cwd=repo)
            untracked, _, _ = git.run_git(["ls-files", "--others", "--exclude-standard", "-z"], cwd=repo)
            summary["checks"]["working_tree"] = {"unstaged": dirty.split("\0")[:-1], "untracked": untracked.split("\0")[:-1],
                "note": "Tests execute the working tree, not an isolated staged snapshot."}
            changed_after_tests = fingerprint(git.capture_staged(repo)) != snapshot_hash
            summary["checks"]["staged_unchanged_after_tests"] = not changed_after_tests
            if result or summary["checks"]["tests"]["status"] == "failed" or changed_after_tests or (argv and (dirty or untracked)):
                summary["reason"] = "failed checks or tests cannot be bound to the staged snapshot; inference skipped"
            else:
                states = chunk.chunk_states(message, prep)
                summary["coverage"] = {"omitted": prep["omitted"], "more": prep["more"],
                                       "cut": any(chunk.CUT in repr(state) for state in states)}
                report.asking(args.provider + "/" + transport.model)
                try:
                    judged = judge(states, env, deadline_s=args.timeout, ask=transport.ask)
                finally:
                    report.stop_spinner()
                summary["judgments"] = {k: v for k, v in judged.items() if k != "error"}
                summary["inference_error"] = str(judged["error"]) if judged["error"] else None
                # Inference may run while the user stages more work: those replies must not pass.
                same = fingerprint(git.capture_staged(repo)) == snapshot_hash
                current_head, _, _ = git.run_git(["rev-parse", "--verify", "HEAD"], cwd=repo, check=False)
                summary["checks"]["snapshot_unchanged"] = same and current_head == head
                if argv:
                    dirty_after, _, _ = git.run_git(["diff", "--name-only", "-z"] + git.DIFF_FLAGS, cwd=repo)
                    untracked_after, _, _ = git.run_git(["ls-files", "--others", "--exclude-standard", "-z"], cwd=repo)
                    summary["checks"]["snapshot_unchanged"] &= not dirty_after and not untracked_after
                if judged["error"] or not summary["checks"]["snapshot_unchanged"]:
                    summary["reason"] = "incomplete inference or Git snapshot changed during the run"
                else:
                    _, findings, rows = decide(judged["answers"], hits, thresholds())
                    summary["findings"] = findings
                    for label, risk, status in rows:
                        report.check(label, risk, status)
                    uncertain = any(row[2] != "ok" for row in rows)
                    limited = any(summary["coverage"].values())
                    summary["status"] = "review" if findings or hits or uncertain or limited else "ready"
                    code = REVIEW if summary["status"] == "review" else 0
    except KeyboardInterrupt:
        summary["reason"] = "cancelled by user"
    except Exception as err:
        summary["reason"] = type(err).__name__ + ": " + sanitize(str(err), env)
    finally:
        summary["requests"] = {"started": transport.requests, "completed": transport.completed,
                               "failed": transport.failed, "cancelled": transport.cancelled}
        summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        # Keys and matched secret lines are never included in summary.json.
        (out / "summary.json").write_text(json.dumps(sanitize(summary, env), indent=2, allow_nan=False) + "\n", encoding="utf-8")
        report.verdict(summary["status"].upper() + " · no staging, commit or push performed",
                       "ok" if code == 0 else "flag" if code == BLOCK else "warn")
        report.line("Report: %s" % (out / "summary.json"))
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(prog="jev-commit check", description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("-m", "--message", help="proposed commit message")
    source.add_argument("--message-file", type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--profile", choices=("generic", "playjev"), default="generic")
    parser.add_argument("--out", type=Path, help="new local evidence directory (never overwritten)")
    parser.add_argument("--test-timeout", type=positive_seconds, default=120)
    provider_args(parser)
    parser.add_argument("--test", nargs=argparse.REMAINDER, help="explicit test argv, no shell; put last")
    args = parser.parse_args(argv)
    if args.test == []:
        parser.error("--test requires a command")
    try:
        return execute(args)
    except (OSError, ValueError, git.GitError) as err:
        sys.stderr.write("jev-commit check: %s\n" % err)
        return USAGE


if __name__ == "__main__":
    sys.exit(main())
