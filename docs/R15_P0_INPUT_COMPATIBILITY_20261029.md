# P0: historical label compatibility and the remaining input decision

2026-10-29, continuing from `8707ff9e6e5fd1e8887d4e16aa6de8f065b79131`.
The previous goal turn made progress by acquiring source/config evidence. This
turn checks the remaining logged-in private-asset inventory, fixes a real
development-input incompatibility, and prepares a disabled fallback extraction
for a concrete permission decision. No real text, model or predictions ran.

## Private asset inventory

The logged-in [Your Work datasets page](https://www.kaggle.com/work/datasets)
showed six account datasets, with exactly three Name That LLM entries: R6, R7
and R10. Unrelated projects were not opened.

- [R10 V1](https://www.kaggle.com/datasets/zephyrpong/ntl-r10-assets-20261028)
  lists one file, `ntl_r10_payload.tgz.bin`, displayed as 211.26 MB.
- [R7 V1](https://www.kaggle.com/datasets/zephyrpong/ntl-r7-assets-20261026)
  lists `README.txt` and `ntl_r7_payload.tgz.bin`, displayed total 210.52 MB.
- R6's packaging was already examined in the preceding turn and was not repeated.

Historical `scripts/build_round10_assets.py:28-45` and
`scripts/build_round7_assets.py:35-48` copy only model artifacts, synthetic
compatibility fixtures, source, labels and external-source metadata. Their
`package_round*_runtime.py:44-52` packages those assets and a runtime. They do
not produce a standalone original development-text export. No archive, model
or array was downloaded, opened or unpickled. Remote filenames agree with the
packaging names; the remote archive bytes were not downloaded for comparison.

This closes the known account-private dataset lead. It does not establish that
no unshared artifact exists elsewhere. The original data-supply dependency
remains unresolved under the currently authorized no-protected-byte-read scope.

## Fixed: incompatible development and prediction label orders

R2's `source_snapshot/config/labels.json` starts with `gpt-5.5`. Its loader
encodes truth against that order, and its row-identity hash includes both
`truth` and `labels`. R15 correctly froze that R2 hash, but incorrectly also
required the development file to use R14's alphabetic prediction-column order,
which starts with `claude-fable-5`. A valid historical export could not satisfy
both checks. Earlier artificial fixtures used only the alphabetic order and
therefore failed to exercise this real interface mismatch.

The fix in [the replay library](../experiments/r15_typography_replay.py):

1. Keeps `DEVELOPMENT_LABELS` in the exact historical order, with untouched
   development rows/truth/labels and the original frozen identity hash.
2. Keeps prediction arrays and manifest `labels` in R14's `LABELS` order.
3. Converts truth indices by class name only after identity/fold validation and
   immediately before typography fitting and metrics. No export is reordered.

The mapping from historical to prediction indices is
`[11,12,13,14,15,5,6,7,8,0,1,2,3,4,17,9,10,16]`.
The [bundle contract](R15_BUNDLE_CONTRACT_20261029.md) now distinguishes these
orders. The nested fold planner uses groups/row positions and remains unchanged.

The two affected suites were rerun because the shared validation/replay code
changed: **28/28 PASS, 1.225 seconds**. New coverage checks all 18 mappings,
input immutability, rejection of reordered development exports, prediction
manifest column order and the truth passed to fold fitting. Upstream arithmetic
parity remains valid. No R14 suite or unchanged missing-input check was rerun.
Independent review found no material issue. The frozen protocol, strengths,
fold hashes, gate thresholds and registered candidate count did not change.

## Prepared fallback: pending permission, not executed

The [proposal](../configs/r15_raw_export_proposal_20261029.json) remains
`pending_user_boundary_amendment`, with no approval reference. Its
[implementation](../scripts/r15_prepare_raw_dev_export.py) stops at that gate
before source stat/open, claim creation or quarantine access. The existing
preisolated-source validator is not relaxed.

The exact proposed source is:

`D:/kaggle/name_that_llm/name-that-llm/data/raw/train.csv`

Known source identity:
`40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`.
Only filename/size metadata was inspected in this turn; the file was not
opened or hashed. If explicitly approved, the prepared operation would:

1. Create an exclusive attempt claim, then read at most 4 MiB of source bytes
   once and verify the frozen source SHA256.
2. Parse the same bytes twice in memory. The first pass reconstructs only
   identity metadata and verifies the exact historical split hash; the second
   retains only the original 2159 development rows. This **does parse the
   protected 540 records twice**, so it is not a zero-access operation.
3. Recreate the original five GroupKFold assignments, retaining historical
   label order, and verify the original row/text/fold hashes before any write.
4. Write only development data plus an honest extraction receipt into the new
   `outputs/r15_raw_export_quarantine/` directory. Existing outputs/attempts
   cannot be overwritten or silently retried. Protected texts are not printed,
   exported, trained on or scored; no all-row split manifest is persisted.

The receipt explicitly reports `raw_official_csv_opened=true` and
`protected_record_parse_events=1080`, with zero protected rows persisted and
zero holdout evaluations. It uses a distinct schema and is deliberately
incompatible with the old preisolated-only intake. Approval of extraction would
permit that one operation; subsequent source admission needs an explicit policy
amendment and actual review, and it would not resolve missing member trainers
or authorize model training/submission by itself.

The new exporter tests use generated artificial CSV bytes only:
**8/8 PASS, 0.316 seconds**. They check the pre-I/O pending gate, missing approval
reference, scope/path changes, source byte bounds/hash, original identity hashes,
multiline/quoted text preservation, excluded sentinel absence and no overwrite.
Independent review found no material issue. The checked-in entry point has not
performed an official-data invocation, and no real claim/quarantine exists.

## Current decision and evidence boundary

The permission needed is specifically **one local full-CSV read/parse for
development extraction**, because the user's current instruction says not to
touch the old 30-prompt holdout. Allowing low-load GPU use does not authorize
that read. No skill or tool approval rejection created this requirement.

Without that boundary amendment or a separately supplied preisolated export,
the data path remains blocked. The full score-improvement goal is unachieved;
the compatibility repair and artificial tests are engineering evidence only.
The cumulative preparation ceiling remains 4 CPU-hours, not a fresh budget.
Measured CPU, unchanged protocol hashes, source pointers, tests and zero real
execution counters are in [this turn's receipt](../configs/r15_p0c_receipt_20261029.json).
