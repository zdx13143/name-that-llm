# P0: approved one-time extraction and development-data admission

2026-10-29, continuing from `c262ba684b0f5b8eb52685586e668f53b7055333`.

The user explicitly answered “允许一次隔离导出及开发数据准入” after
disclosure that the operation reads the full CSV and parses the protected
records in memory. That exact operation is complete. The original 2159
development rows are admitted; model training and predictive evaluation
remain unexecuted. This is input preparation, with no score-improvement claim.

## Actual operation and protected-data accounting

The prepared exporter was unchanged from its reviewed SHA256
`567f1222b0bed6fd0db318414a064fd28edb73eac592f5d018243c9ed9be1903`.
The [proposal](../configs/r15_raw_export_proposal_20261029.json) now records
the explicit approval. Its exclusive attempt claim and quarantine output
prevent automatic repeat execution.

| Operation | Actual count |
|---|---:|
| Full official CSV read invocation | 1 |
| In-memory CSV parse passes | 2 |
| Protected records included per pass | 540, across 30 prompts |
| Protected-record parse events | 1080 |
| Protected rows written to quarantine or admitted data | 0 |
| Protected model fitting / scoring | 0 / 0 |
| Raw-source reads during independent review and admission | 0 |

Historical source SHA256 was checked during the single export. It was **not**
recomputed by opening the source again. The independent review inspected the
exporter, claim, runtime and receipt to assess that historical operation;
it directly recomputed the hashes and aggregate structure of the development
output. No raw texts, labels, IDs or prompt lists were displayed.

## Verified input identity and admission

The admitted file is local and Git-ignored:
`outputs/r15_dev_export/development.json` (1,714,928 bytes).
The immutable extraction copy remains in `outputs/r15_raw_export_quarantine`.

| Binding | SHA256 |
|---|---|
| Development file bytes | `1c9446efa6c8c477bd68448f1de212aadcc8dc90224b76728ec2f70e2ec2334f` |
| Original row identity | `eb94b93d3557fde2fbbfa4cd897d1ef25c2e1fe73ee01cf526a65d5c433dcd20` |
| Original texts | `ad192430092d19e4a44aa1ae263d051a987685dd1883fcdacab3f40a82cae5b9` |
| Original outer folds | `96040086312dca7ce1d3d14bb3ee59640870ef35edba42c746c7f84ae7b827da` |
| Generated nested plan, canonical JSON | `f4598325850bf2e1a24157b4b8e9b0032ae22afacce88ad86a65b9841ced13dc` |

The original row order and historical label order are unchanged. There are
2159 unique rows and 120 prompts. Outer validation counts are
`[432,432,432,432,431]`, each with 24 prompts and all 18 classes; training
uses 96 prompts. The plan contains five inner folds within each outer train,
25 inner folds total. No outer validation group enters inner fitting or
calibration. These are split plans, not trained models or OOF predictions.

[The separate admission entry point](../scripts/r15_admit_approved_export.py)
checks [the exact-source policy](../configs/r15_approved_export_policy_20261029.json)
and [independent review](../configs/r15_approved_export_review_20261029.json)
before reading the development payload. It then checks all frozen identities,
writes the admitted copy and sidecars without overwrite, and explicitly keeps
training/scoring disabled. The old preisolated-only intake remains unchanged;
this approved raw extraction is never labelled as a zero-access source.

The local admitted directory contains `development.json`, the unchanged
`extraction_receipt.json`, `admission_policy.json`, `admission_review.json`,
`nested_plan.json` and `admission_receipt.json`. Consumers must verify these
bindings and the frozen protocol before using the data. Use this directory
for subsequent separately preregistered work; never rerun the exporter.

## Verification and resources

The new admission suite initially passed 8/8 artificial tests. Independent
code review suggested explicitly enforcing the zero-reread policy field,
although the entry already had no raw-source read path. Root added that
guard and the fixed-directory check; the final changed suite passed **9/9**.
Both logs are retained. Existing unchanged suites were not rerun.

Independent output review verified actual hashes, counts, uniqueness and
prompt separation. Root's final verification checks the admitted copy,
sidecar bindings, exact outer/inner coverage and disabled execution flags.
The real admission invocation used a process audit hook that rejected raw
`train.csv` or extraction-claim opens; no such open was attempted.

The cumulative preparation ceiling remains **4 CPU-hours**. Known measured
Python CPU is itemized in [the execution receipt](../configs/r15_p0d_receipt_20261029.json);
browser, tool, Git and some review overhead are not a whole-host CPU meter.
No GPU job, new download, real model fit, encoder forward or label accuracy
evaluation occurred. No Git push, Kaggle import/Save/submission, new Public
Score or private result occurred.

## Remaining dependency and next executable step

The development-text dependency is resolved. P0 as a whole is still incomplete:

- v5, Nishant/main DeBERTa, public/Claude LoRA, UltraBERT and ATLAS need the
  remaining exact trainer/configuration and stage-membership evidence listed
  in [the source inventory](../configs/r15_p0_source_inventory_20261029.json).
- External inputs need fixed hashes, prompt lineage and per-stage exclusion
  receipts; a data-card statement or exact-text deduplication is insufficient.
- The nested producer needs the precise blend/Gaussian fitting objectives,
  constraints, selection rules and budget frozen before execution.

Next, complete one required member's source and data envelope, then adapt it
to the admitted outer/inner plan and measure its bounded resource needs. A
component check alone cannot establish full-ensemble improvement. Low-load
local GPU permission already exists, but no suitable full-producer training
release or credible local resource estimate is established. The original
R15 comparison remains disabled and unchanged.
