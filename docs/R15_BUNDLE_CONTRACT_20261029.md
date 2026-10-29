# R15: minimum complete-ensemble development bundle

This document is a local asset specification, not a request sent to anyone.
No artifact described as missing below has been generated or downloaded.

## Cheapest route to a credible local comparison

Obtain existing, independently reviewable five-fold outputs from the complete
R14 upstream recipe. Each validation prompt must be absent from the training
of EVERY task-trained dependency. The original 30 protected prompts must also
be excluded. If those artifacts do not exist, producing them is a new
multi-model training project; do not silently launch it under the CPU replay
budget. The existing laptop's ModernBERT and R6-R10 component assets cannot
fill this gap. A public score or five checkpoint filenames is not a substitute.

The established split is recoverable. The prior Windows module contains
`scripts/restore_team_split.py` and `scripts/windows_modernbert_dev.py`; R2's
`outputs/windows_modernbert_dev_20261024_r2/contract.json` pins identities.
Do not make a new 120/30 split. The historic random-seeded split manifest is
not itself the later GroupKFold5 assignment. R15 pins the later fold hash.
This round read only code/contract hashes, not train.csv, split row lists,
old OOF arrays or protected examples.

## Files and their exact meaning

Place a future admitted bundle under `outputs/r15_inputs/` (ignored by Git):

| File | Required content |
|---|---|
| `development.json` | Only the original 2159 development rows: `row_index`, `ids`, `groups`, `truth` (integer class indices), `labels`, `texts`, `fold_id`; original order, 120 groups, 5 folds |
| `raw_logits.npy` | `(2159,18)` float32/float64, original-scale `_v38_before_hard`, assembled from independent validation folds |
| `r14_logits.npy` | Same shape/order; independently generated R14 full-strength post-typography outputs; typography fitted only on each outer training fold |
| `full_ensemble_oof_manifest.json` | Schema/identity/recipe/scale bindings, exact three file SHA256 values, five fold receipts and complete dependency inventory |
| `provenance_review.json` | Canonical manifest SHA256, conclusion `all_fold_exclusions_verified`, evidence-index SHA256; produced only after actual review |
| referenced producer evidence | Checkpoint/head hashes, training membership, external data overlap/filter scope, feature fitting scope, configuration/source hashes, actual runtime member and ATLAS coverage receipts |

One float64 prediction array contains 310,896 numeric bytes plus its NPY header;
two arrays contain about 0.593 MiB. Text/identities and provenance are extra.
There is no need to transfer all multi-billion-parameter weights just to replay
the postprocessor if valid predictions and provenance already exist.

Preserve `_v38_before_hard` directly before hard rules/typography. Do NOT use
`--save_lp`: that divides by `wtot` and changes confidence gating. The baseline
parity check rejects typical rescaling, but parity is not a proof of genuine
producer history. Both arms use the exact same upstream predictions and
ATLAS coverage. Recompute train priors and all train-derived transformations
within each outer fold. Never pool all five validation folds into one
transductive batch. No held-out labels or true validation prompt groups enter
upstream prediction; groups are used for splitting/evaluation only.

## Identity and metadata contract

`configs/r15_protocol_20261029.json` pins historical `row_identity_sha256`,
`fold_sha256` and `text_sha256`. Canonical hashing is UTF-8 JSON with sorted
keys, no spaces, `ensure_ascii=False`, `allow_nan=False`. Row identity hashes
the object containing exactly `row_index, ids, groups, truth, labels`; text
and folds hash their ordered lists. This matches the historical R2 code.

The development file preserves R2's original class order (`gpt-5.5` first),
with `truth` indexing that list. Prediction columns and manifest `labels`
preserve R14's alphabetic order (`claude-fable-5` first). They intentionally
differ. After checking the original development hashes, replay translates truth
indices by class name before fitting typography or computing metrics; it does
not reorder the export or overwrite its hash. The historical-to-prediction
index mapping is `[11,12,13,14,15,5,6,7,8,0,1,2,3,4,17,9,10,16]`.

Manifest top-level values:

- `schema`: `ntl-full-ensemble-oof-v1`.
- All four recipe/identity digest fields copied from the protocol; ordered
  labels exactly match the frozen R14 inference source.
- `replay_stage`: `_v38_before_hard_original_scale`.
- `protected_holdout_accesses`: integer 0, never boolean false.
- `transductive_batch_scope`: `one_outer_validation_fold`.
- `files`: exactly the three data/array filenames above, each mapped to its
  SHA256 over exact bytes.
- `folds`: five entries ordered 0..4. Each contains train/validation group
  hashes (sorted unique lists), validation ID hash (original row order),
  producer source/config, overlap audit and execution receipt hashes;
  `task_supervised_scope` and `prior_scope` both `outer_train_only`;
  `all_dependencies_enumerated: true`.

The `members` keys must equal `REQUIRED_ROLES` in
`experiments/r15_typography_replay.py`. Its 33 roles cover the five main LoRA
configurations, DeBERTa, both main Gaussian heads, TF-IDF/distortion/SVC,
main grouping, all configured COMP members/four Gaussian heads/family gates/
grouping, external-data selection, ATLAS and Sinkhorn prior. These are
dependency roles, not 33 trained models. Each role's provenance must enumerate
ALL underlying adapters, checkpoints, heads, fitting and filtering steps.

Each included role requires `asset_sha256` (nonempty digest list),
`provenance_sha256`, `status: included`, `supervision_scope` and
`official_fit_groups_sha256`. The latter must bind either the outer train
groups or the empty list for a verified `no_task_supervision` dependency.
Full-fit/unknown scope is rejected. A subset-trained dependency may be
represented by the conservative allowed outer-train envelope only if its
reviewed detailed receipt lists the actual subset; no validation/protected
prompt may be in either. This code-level contract alone cannot verify that
receipt's truth.

Only COMP token-attention roles may declare `upstream_optional_absent`, with
supporting provenance, matching the upstream recipe's explicit optional
policy. They may not silently disappear from the inventory. ATLAS records
`selected`, `scored`, `validation_rows` and `scored_ids_sha256`. Partial
coverage is permitted by the original recipe and is shared by both arms;
its budget/selection/coverage remain part of the reviewed producer evidence.

## What the delivered code does and does not admit

`python experiments/r15_typography_replay.py` is metadata readiness only.
It does not parse development records or prediction arrays, train anything,
or calculate accuracy. Even a complete-looking metadata bundle is explicitly
labelled with file hashes unchecked. The bounded loader independently checks
the actual bytes, shape/dtype before allocation, no-pickle loading and the
development/fold identities. The replay library checks independent R14
baseline parity before scoring.

`real_evaluation_enabled=false` and there is no CLI `--run`. Before using real
assets, a local admission must pin the reviewed bundle and producer evidence,
add a persistent single-use execution claim/supervisor for the already
preregistered <=120-second/2-GiB comparison, and keep prior counters intact.
Turning a boolean on is not sufficient. This bounded execution wrapper is
deliberately deferred while there is no real bundle to bind; it is an explicit
remaining dependency, not a completed full-ensemble evaluation runner.

Interpret a future gate as descriptive evidence on repeatedly used development
prompts. Neither accepted metadata nor synthetic parity proves predictive
improvement, hidden-test performance, private score, or causal contributions
of R14's joint upstream changes.
