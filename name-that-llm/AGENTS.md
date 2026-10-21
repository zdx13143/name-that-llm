# Independent CPU experiment module

- Read the target repository README collaboration rules, this module WORKSPACE.md, COLLABORATION.md and HANDOFF.md before changes.
- Use the target team's `exp/<topic>` branch convention. Never push main or another person's branch, force-push or merge without authorization.
- Preserve unrelated repository-root work and historical records. The 2026-10-21 user request to analyze and continue the teammate's work authorizes the narrowly scoped center/embed bug fixes on our own branch; the new comparison runner remains in this isolated module.
- Stage explicit files. Exclude raw data, credentials, environments, weights, OOF, predictions and executed Notebook outputs.
- The original 30-prompt holdout has already been evaluated once. Missing local caches or claim files in a clone do not reset that fact. Do not rerun search/finalize or remove claim protection.
- Report exact prompt-group protocol, per-class recall and reproducible commands. Never compare this 120-prompt development score directly with the team's 150-prompt full-data CV or leaderboard score.
- The original upload alone did not authorize new training. The subsequent continuation request and standing autonomous-work instruction authorize the finite development-only CPU plan in docs/TEAM_CONTINUATION_PLAN.md. Do not resume old search/finalize. Paid resources, Kaggle submission and contacting teammates remain outside scope. Update HANDOFF with actual execution evidence.
