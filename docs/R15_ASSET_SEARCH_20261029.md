# R15 continuation: read-only OOF search and admission decision

Observed 2026-10-29, approximately 04:50-05:15 UTC. Starting commit:
`81c64c797ef4a1cce2d40bb5abc43ce2365f1904`.

**Decision: no eligible complete-ensemble bundle found in the inspected public
and connected sources. Real R15 evaluation remains disabled.** There ARE
public component OOF and nested-validation assets. They must not be described
as absent, but none establishes the original 120-prompt / 2159-row R14
complete-ensemble comparison with all training dependencies excluded from
the validation fold and the protected 30 prompts.

This is a scoped search result, not proof that no private/unpublished bundle
exists anywhere. No author or teammate was contacted. No raw competition
data, prediction array, model weight or protected example was downloaded.
No actual OOF row was scored or remapped into a new split.

## Sources inspected and decisions

| Source | Direct observation | Admission result |
|---|---|---|
| [Teammate GitHub](https://github.com/zdx13143/name-that-llm) | Main still `dc62708e7116af472257d8f9159424fec2078f01`; inspected main/R11/R12 trees, branch listing, code search, open/closed issues and PRs, releases. R11/R12 trees contain 85/94 paths; issues/PRs/releases empty. No complete prediction payload or exclusion/coverage bundle found. | No eligible bundle. Repository validation source and public-score receipts are not OOF. |
| [Roman V32 output](https://www.kaggle.com/code/romantamrazov/name-that-llm-who-am-i/output?scriptVersionId=363222852) | 14 output files; `comp_logit.npy` is 344 B. `v31_diagnostics.json` reports 2699 trained rows, 3 tested rows, 37 markers and explicitly limits its fit to final submission inference, not OOF. | Saved visible-test run, not development OOF. |
| [Roman V32 input and comments](https://www.kaggle.com/code/romantamrazov/name-that-llm-who-am-i/input?scriptVersionId=363222852) | 31 inputs; core adapter/ATLAS/UltraBERT and training-kernel directories inspected. Five comments supply attribution links and general model discussion, no OOF bundle. | Inputs describe a production recipe, not fold exclusions. |
| [v5 LoRA adapters](https://www.kaggle.com/datasets/dhruvtibarewal/nlm-l24-adapters-v5) | V1, 2.59 GB, 54 files: 18 safetensors, 18 `.pt`, 18 JSON. Eighteen adapter directories, each with adapter config/weights/head. Data card has no description; provenance sources/methodology empty. | Fold suffixes do not bind to R15 folds. No OOF arrays or membership receipt in this inventory. |
| [ATLAS v1 package](https://www.kaggle.com/datasets/josephadamski91/ntl-atlas-qwen3-8b-v1) | V3, 462.8 MB, 15 files. Data card and CREDITS say five epochs on all 2699 official rows, with 18 source-specific adapters. `training-source-manifest.json` binds the official train SHA below. | Explicit full-data training violates R15 admission. |
| [UltraBERT](https://www.kaggle.com/datasets/pjmathematician/ultrabert-380m-ntl) | Five prompt GroupKFold fine-tunes, reported component OOF BA 0.408; 56 files, 3.82 GB, fold0..4 plus wheels. | Component checkpoints/score only; no proof of R15 split, protected-30 exclusion or complete R14 coverage. |
| [One Public LoRA24 Run](https://www.kaggle.com/datasets/dariushafshar/ntl-oof-logits-one-public-lora24-run) | Advertised `(2699,18)` float32 logits, `(2699,)` fold IDs, class order and row-alignment hash. Derived from S7 plain readout. Uses sorted prompt IDs shuffled with seed 0 and position modulo 5. | Actual component OOF, but different population/folds, no protected-30 exclusion, no two-stage/group heads, no complete R14 stack. Do not subset and rename it R15 OOF. |
| [S7 training output](https://www.kaggle.com/code/lucasboesen/name-that-llm-train-lora24-s7/output) | V1 / 358911769, 29 output files; `job_0`..`job_4`, `job_final`. Fold artifacts include `readouts.npz` and `valid_features.npz`; final bundle contains adapter/tokenizer/norms. | Useful source of component training mechanics, not full R14 OOF. Arrays were not opened. |
| [Name That LLM Artifacts](https://www.kaggle.com/datasets/lightningv08/ntl-olmo2-native24-final-inference-assets) | Current V17, 99.82 GB, 884 files. `name-that-llm-artifacts/ntl-oof-validation-cache` preserves 182 files; visible names include outer/inner DeBERTa arrays. The original `_source_metadata/ntl-oof-validation-cache.json` explicitly says public full-fit LoRA and UltraBERT have no verified matching OOF there. | Valuable partial nested-validation cache; missing required neural coverage and R15 identity/exclusions. Current V17 is not silently substituted for R14's pinned input version. |
| Same collection: `ntl-oof-q35-heads` | 28 files including fold plan, routes, nested report, source scripts and `oof_predictions.npz`. Only inventory/source metadata inspected. | Component validation assets, not evidence of complete R14/120-prompt compatibility. Fold IDs/arrays were not opened. |
| [Pierre final models/OOF](https://www.kaggle.com/datasets/pierrevanvolsem/name-that-llm-final-models) | Three OOF probability matrices for Qwen3.5-0.8B, Qwen3.5-4B and a linear member; 12 files, 1.75 GB; schema metadata shows 2699 IDs. | Different ensemble and full official population. No payload or row-level preview used for evaluation. |
| [Prism Feature Atlas](https://www.kaggle.com/datasets/verracodeguacas/name-that-llm-prism-feature-atlas) | Training feature cache and exploratory fixed-head diagnostics on reused prompts; explicitly not a complete hidden-test pipeline. No raw text/row IDs/ground truth in the release. | Does not supply R14 members or fold-isolated full-stack logits. |

Kaggle's single site search `name that llm oof` returned 16 datasets. Titles
were screened; the relevant OOF, source-collection and feature-cache cards
above were inspected. Unrelated datasets and generic external-passage sources
were not expanded. Public web indexing was incomplete: several exact Kaggle
pages failed through the web fetch tool, so normal authenticated browser reads
were used. No captcha or access restriction was bypassed.

The referenced [fivefold training source](https://www.kaggle.com/code/lightningv08/ntl-lora24-fivefold-training)
currently returns a page-not-found screen. S7's `worker.py` was visible in the
output inventory, but a source-only download attempt timed out and no file
was found in Downloads; it is not claimed as acquired or reviewed locally.

## Strongest exclusion evidence

ATLAS's published training-source manifest reports:

- `data/competition/train.csv`:
  `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`.
- `ntl_atlas/train.py`:
  `80440fa2589236b0cf6f1d3f9388f049941a4474050c55fe6729ee677a28ef87`.
- `configs/atlas-qwen3-8b-v1.json`:
  `bd0d037376f5b63f36953f84c92bcbbe29495a4a9e0c2a01c0708ac839847252`.

These are reported hashes, not locally downloaded-file verifications. The
train CSV hash matches the historical official dataset contract. This links
the public statement of full-data fitting to the known official population;
it is not a training-exclusion proof. The inspected `ntl_atlas` package
directory supplies `__init__.py`, `inference.py`, `model.py`; a hash referring
to `train.py` does not establish that the trainer itself is available.

## Additional provenance issue in frozen coefficients

The actual frozen source says the COMP `W` coefficients came from an L-BFGS
fit on prompt-grouped OOF (`reference_public660_v32.ipynb:976-980`). Gaussian
reduction/temperatures refer to a production fit using 135 training prompts
(`:986-991`). R15 outer training folds contain only 96 development prompts.
Their original fit scope therefore needs explicit audit; learned coefficients
must not be silently treated as task-independent constants.

Keeping these coefficients frozen allows only a conditional comparison of a
previously selected public recipe. It does not by itself establish that every
trained dependency is fold-isolated. Re-fitting learned coefficients inside
each outer training fold changes the upstream producer relative to the
currently frozen R15 definition. That would require a new producer protocol
before training, not a post-hoc relaxation of this admission check.

## Development-only supply remains a separate dependency

The old R2 `features.npz` exists (23,676,168 bytes), and its metadata schema
contains development row indices, IDs, groups, truth, labels and fold IDs.
The save function at `scripts/windows_modernbert_dev.py:255-259` does **not**
persist `texts`. Thus it cannot alone supply R15's `development.json`.
Its arrays were not opened. A development-only text export matching R15's
frozen text hash remains required; no raw `train.csv` reconstruction or old
30-prompt inspection was performed in this search.

## State and counters

The original protocol, replay code, tests and R14 frozen candidate remain
unchanged. No bundle was admitted, no producer review was fabricated, and
`real_evaluation_enabled` stays false. The prior 15/15 synthetic tests remain
historical engineering evidence; unchanged code was not retested.

This continuation: 0 model fits, 0 encoder forwards, 0 typography fits on
official data, 0 official-label scores, 0 protected-holdout accesses, 0 Kaggle
imports/Saves/submissions, 0 pushes, 0 new drafts/automations. No new public or
private result. Browser reads and metadata inspection are not prediction
validation. Historical counters and the fixed half-strength hypothesis remain
intact.

See [the minimal rebuild proposal](R15_REBUILD_PLAN_20261029.md) and the
[search receipt](../configs/r15_asset_search_receipt_20261029.json).
