# Independent CPU experiment module

- Read the target repository README collaboration rules, this module WORKSPACE.md, COLLABORATION.md and HANDOFF.md before changes.
- Use the target team's `exp/<topic>` branch convention. Never push main or another person's branch, force-push or merge without authorization.
- Preserve the existing repository-root src/, scripts/, configs/, requirements and experiment records; this module is isolated.
- Stage explicit files. Exclude raw data, credentials, environments, weights, OOF, predictions and executed Notebook outputs.
- The original 30-prompt holdout has already been evaluated once. Missing local caches or claim files in a clone do not reset that fact. Do not rerun search/finalize or remove claim protection.
- Report exact prompt-group protocol, per-class recall and reproducible commands. Never compare this 120-prompt development score directly with the team's 150-prompt full-data CV or leaderboard score.
- This upload does not authorize new training, paid resources, Kaggle submission or contacting teammates. Update HANDOFF with actual execution evidence.
