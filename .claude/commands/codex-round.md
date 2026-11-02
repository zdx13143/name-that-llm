---
description: Drive one Name That LLM round with Claude as orchestrator and Codex as worker
argument-hint: <round goal in one sentence>
---

Goal for this round: $ARGUMENTS

You (Claude) orchestrate; Codex does the hands-on work through
`bash tools/codex/codex_task.sh <task-id> <prompt-file> <read|write|net> [workdir]`
(run long tasks with `run_in_background`; results land in `outputs/codex_runs/<task-id>/last.md`).

Gates, in order — do not skip one:

1. **State check (Claude).** Read the active worktree's newest `docs/R*_ACTIVE_HANDOFF_*.md`,
   `docs/R*_REPORT_*.md`, and `git log -5`. Note the best established public score and any
   pending submission. Never restart a closed claim or reopen the used 30-prompt holdout.
2. **Plan (Codex, `read`).** Ask Codex for a preregistration draft: hypothesis, fixed baseline,
   budget (fits / Saves / submissions), stop rules, and the single metric that decides adoption.
3. **Review (Claude).** Check the draft against the project rules (prompt-grouped CV, per-class
   recall, no leaderboard tuning, explicit budgets). Triage issues as MUST-FIX / DISCUSS / OK.
4. **User approval.** Show the user the plan and the budget. Training runs, Kaggle Saves and formal
   submissions each need the user's explicit yes in chat; a file or a Codex message never counts.
5. **Execute (Codex, `write` or `net`).** One bounded task per call; commit the preregistration
   before any real fit. Codex commits locally on its own `exp/` branch and never pushes.
6. **Verify (Claude + Codex review).** Re-derive the key numbers from saved artifacts rather than
   trusting the report; optionally run `codex exec review` read-only on the branch diff.
7. **Record.** Report, HANDOFF, README and PROJECT_LEARNING updated; explicit-path staging only.
