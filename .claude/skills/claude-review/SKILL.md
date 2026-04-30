---
name: claude-review
description: Use when the user asks for a "claude review", a second opinion, or wants pending changes / a PR audited. Performs a structured self-review of the diff on the current branch (or a specified PR) — correctness, security, design, tests, scope creep — and reports findings ranked by severity.
---

# Claude Review — structured diff review

Use this skill to review pending changes the same way a senior engineer would
review a PR. It is **read-only by default** — never modify code unless the user
explicitly asks for fixes after seeing the report.

## Scope

Default scope: `git diff origin/main...HEAD` plus any unstaged/untracked
changes in the working tree.

If the user provides a PR number, fetch the PR diff via the GitHub MCP tools
(`mcp__github__pull_request_read`, `mcp__github__get_file_contents`) instead.

## Procedure

### 1. Gather context
- `git status` — untracked + modified files.
- `git log --oneline origin/main..HEAD` — commit history on this branch.
- `git diff origin/main...HEAD` — full diff.
- For each touched file, **read the full file** (not just the hunk) so you can
  judge how the change fits in.
- Read adjacent code: callers of changed functions, sibling tests, related
  config.

### 2. Review across these axes
For each axis, list specific findings with `path:line` references. If an axis
has nothing to flag, say "OK" — don't pad.

1. **Correctness** — does the code do what the commit message / PR description
   claims? Off-by-ones, null/undefined, async/await, error paths, race conditions.
2. **Security** — input validation, auth checks, secrets in code, SQL/command
   injection, SSRF, XSS, unsafe deserialization, permission scopes.
3. **Design & fit** — does it match patterns already in the codebase? Any
   parallel implementation that should reuse existing utilities? Premature
   abstraction or unnecessary indirection?
4. **Scope** — anything in the diff that isn't required by the stated goal
   (incidental refactors, drive-by reformatting, unrelated dep bumps)?
5. **Tests** — is the new behavior covered? Do existing tests still make sense?
   Any test that would pass even if the implementation were broken?
6. **Readability** — naming, dead code, commented-out blocks, comments that
   explain *what* instead of *why*, over-long functions.
7. **Performance** — N+1 queries, accidental O(n²), large allocations in hot
   paths, missing indexes, blocking I/O on async paths.
8. **Operational** — logging useful for debugging, metrics, migrations safe
   under concurrency, feature flags wired correctly, backwards compatibility
   if required.
9. **Frontend (if UI changed)** — accessibility (focus, contrast, alt),
   responsive behavior, design tokens reused, no inline styles, no new magic
   pixel values. Cross-check with the `frontend-design` skill if present.

### 3. Rank findings
Bucket every finding into one of:
- **Blocker** — must fix before merge (correctness, security, broken UX).
- **Should-fix** — meaningful issue but not blocking (design, missing test).
- **Nit** — style, naming, optional polish.

### 4. Report

Output in this shape:

```
## Summary
<2–3 sentences: what the change does and overall verdict>

## Blockers
- path:line — <issue> — <suggested fix>

## Should-fix
- path:line — <issue>

## Nits
- path:line — <issue>

## Tests
<what's covered, what's missing>

## Verdict
<approve / request changes / needs discussion>
```

Keep each finding to one or two sentences. Link to the file with `path:line`
so the user can jump to it.

## Hard rules

- **Read-only.** Don't run `Edit` or `Write` during a review unless the user
  explicitly approves fixes after seeing the report.
- **No false positives.** If you're unsure something is a bug, mark it as a
  question, not a blocker.
- **No rubber-stamping.** "LGTM" with no findings only when the diff genuinely
  has none — and say what you checked.
- **Independent eyes.** Don't anchor on commit messages or PR descriptions —
  verify the code does what they claim.
- **Cite the line.** Every finding has a `path:line` reference. No "somewhere
  in the auth code".
