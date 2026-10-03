# Evidence-linked check-in runs

This is Jesse Naiman's fork of Valentyn Kit's MIT-licensed `jev-commit`, based on
upstream commit `311e163b8abb9c333132155bc2e0bbfac4f36283`. It reuses the existing
Git lockdown, diff chunking, secret belt, five typed Noul questions and display.
It does not vendor `jev-router` or configure global coding-agent models.

## Same shape as a gameplay run

```text
explicit start -> staged snapshot -> code-owned checks -> typed judgments
              -> freshness recheck -> ready / review / hold / blocked
              -> saved local evidence -> explicit human Git commands
```

Like PlayJev, inference interprets supplied evidence; code owns timing, safety,
execution and the final gate. A check is not gameplay: it cannot prove that Dig
Dug pumps, scores, terminal detection or repeatable controller runs work.

## Minimal workflow

1. Stage only intended source/docs/tests with explicit `git add <paths>`.
2. Run `jev-commit check --profile playjev -m "Describe those changes"` in PlayJev.
3. Inspect the verdict and the printed `summary.json` path. Fix review/hold results.
4. If satisfied and authorized, run Git's commit/push yourself. Re-run the check if
   HEAD, the index or tested working tree changes; the report is not a Git gate token.

The PlayJev profile runs `.venv/bin/python -m unittest discover -s tests -q`, checks
staged whitespace, and blocks staged run folders, ROMs, emulator states, recordings
and real `.env` paths. `.env.example`/`.env.sample` remain reviewable. Paths are
checked before capped model state is built; the belt scans all captured added hunks.

Generic checks run no tests unless requested:

```bash
jev-commit check -m "Describe changes" --test python -m pytest -q
jev-commit check --repo /path/to/PlayJev --profile playjev -m "Describe changes"
jev-commit check -m "Describe changes" --provider jev --model jev-1.13.0
jev-commit check -m "Describe changes" --provider openrouter --model typesafe/jev-1.13
```

`--test` takes argv (put it last); no shell expansion or repo-provided commands are
executed automatically. Named profiles/explicit test commands do execute project
code: only run them on trusted repositories. Test deadline defaults to 120 seconds;
inference deadline to 30 seconds. The optional hook defaults to an 8-second deadline.

## Provider and evidence contract

- Local is **Ollaya CLI only**, with one question per invocation for small local
  memory contexts. Local failure never triggers a hosted call.
- Direct Jev uses TypeSafe's `/v1/systemone`; OpenRouter explicitly uses
  `https://openrouter.ai/api/v1/systemone` and its own key. Model IDs may be pinned
  with `--model`; returned actual model IDs and usage are retained.
- `summary.json` binds provider, model, HEAD, branch, staged paths and SHA-256 of
  captured staged name/status + patch. `inference.jsonl` retains each logical
  provider request, response, latency and started/completed/failed/cancelled event.
  HTTP retries are internal to one logical request, not separate logged requests.
- Invalid/nonfinite/out-of-range Noul answers, unexpected question sets, truncated
  local state, incomplete inference and changed snapshots cannot pass the run.
- Known environment keys are redacted. Detected blocking secrets are not sent to
  inference; belt reports omit original matched lines. Reports may still contain
  private code or undetected secrets. Keep them local; thresholds are inherited
  defaults, **not calibrated accuracy claims for Kev or Atari**.
- Reports never overwrite an existing output directory. Default storage is under
  Git's local metadata, outside ordinary tracked source. No implicit amend replay.
- Tests run the working tree, not an isolated staged checkout. Their success is
  rejected for unstaged/untracked work, or a snapshot change during inference.
  Arbitrary tests can mutate files; this tool detects relevant changes but never
  rolls them back. A change after the last recheck still requires another run.
- `ready` has no power to grant consent, install hooks, run Git writes, claim a score,
  or complete gameplay. The optional hook deliberately fails open on API errors;
  the separate `check` runner deliberately returns hold instead.

## Verification

```bash
make test                    # mocks and loopback fake only; no inference service
python -m jev_commit.cli check --help
python -m compileall -q jev_commit
git diff --check
```

Regression coverage includes index/HEAD preservation, staged-only empty-index
behavior, secret/artifact exclusion before inference, test failure gates, snapshot
changes, timeout/allocation/malformed responses, cancellation, model/provider
pinning and OpenRouter key/URL mapping. External HTTP and real Ollaya execution are
forbidden by the offline test fixture. Live local smoke results are documented
separately; mocked tests do not establish model quality.

Local verification on 2026-10-03: **141 tests passed, one skipped** (no recorded
model-accuracy corpus). The pre-commit end-to-end test against a loopback fake
allowed an advisory mismatch and blocked a staged private key before inference.
An explicit live Ollaya smoke on one synthetic staged file completed all five
questions, with zero failed/cancelled requests and no hosted inference. Kev's
mid-range answers correctly produced **review (exit 4)**, not a fabricated pass.
Exact smoke evidence is local at `.git/local-smoke-review.json` and its referenced
scratch report; this verifies transport/reporting, not classifier accuracy.

Live API references reviewed 2026-10-03:

- <https://docs.typesafe.ai/api.md>
- <https://docs.typesafe.ai/primitives/noul.md>
- <https://docs.typesafe.ai/cookbooks/llm_guardrails.md>
- <https://openrouter.ai/docs/guides/community/typesafe-sdk>
