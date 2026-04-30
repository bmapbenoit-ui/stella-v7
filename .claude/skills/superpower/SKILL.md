---
name: superpower
description: Use when the user asks for "superpower mode" or when a task is non-trivial (touches >2 files, adds a new feature, refactors a subsystem, fixes a bug whose cause is unclear). Forces a structured loop — clarify, plan, execute in small verified steps — instead of jumping straight to code.
---

# Superpower — structured problem-solving loop

Activate this skill for any task that isn't a one-line change. It replaces
"start typing code immediately" with a short, disciplined loop that catches
mistakes before they ship.

## The loop

### 1. Clarify (≤ 2 minutes)
Before any tool call beyond reading files:
- Restate the goal in one sentence. If you can't, ask.
- List the **acceptance criteria** as bullets. What does "done" look like?
- Note any **constraints** the user mentioned (no new deps, must keep API stable,
  branch name, etc.).
- Identify **unknowns**. If an unknown blocks the plan, ask the user a single
  focused question via `AskUserQuestion`. Don't ask trivia.

### 2. Investigate (read before write)
- Use `Explore` / `Grep` / `Read` to map the relevant code.
- For each file you intend to change, read it fully — not just the snippet.
- Write down (in your reply, briefly): what currently happens, where it lives,
  what needs to change.

### 3. Plan
Output a numbered plan with:
- Files to touch and roughly what changes in each.
- Order of operations (tests first when possible).
- A risk/blast-radius note for anything destructive or shared.
- The verification step at the end (how you'll know it works).

If the plan is large or reversible-risk, surface it to the user before executing.

### 4. Execute in verified slices
- Make the smallest change that produces an observable result.
- Run the relevant check (typecheck, unit test, lint, manual verification) **after
  each slice**, not at the end.
- If a check fails, fix the root cause — don't loosen the check or skip the hook.
- Keep the working tree clean between slices: stageable, revertable.

### 5. Verify against acceptance criteria
- Walk every bullet from step 1 and confirm it's satisfied.
- For UI changes, exercise the feature in the browser — not just typecheck.
- For backend changes, run the affected tests and at least one integration path.

### 6. Report
- One or two sentences: what changed, what was verified, what's still open.
- Surface anything you noticed but didn't fix (with a one-line reason).

## Hard rules

- **Never** invent file paths, function names, or APIs — read first.
- **Never** add scope the user didn't ask for (no incidental refactors, no
  "while I'm here" cleanups, no new abstractions for hypothetical futures).
- **Never** mark a task done if a check is failing or skipped.
- **Never** use destructive git ops (`reset --hard`, `push --force`, `clean -fd`,
  branch deletion) without explicit user approval for that exact action.
- **Always** prefer editing existing files over creating new ones.
- **Always** stop and ask when an assumption could change the plan materially.

## Anti-patterns to refuse

- "Let me just try this and see" — investigate first.
- "I'll add a try/except to make the error go away" — find the cause.
- "I'll add a flag for backwards compatibility" — only if the user asked.
- "Let me also clean up this nearby code" — out of scope, separate task.
- Long internal monologue in user-facing text — keep updates terse.

## When to skip the loop

For a strictly local, obviously-correct change (typo, rename in one file,
adding a missing import the user pointed at), skip steps 1–3 and go straight to
execute + verify. The loop is a tool, not a ritual.
