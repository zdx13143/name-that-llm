# R15 minimum complete-ensemble rebuild proposal

> Latest scope amendment: the user subsequently authorized Priority 0 local
> preparation within 4 CPU-hours, with zero GPU and the protected holdout still
> closed. [P0 results](R15_P0_REPORT_20261029.md) now include four acquired
> trainer sources, safe-export intake and a nested calibration design. Required
> dependencies remain missing, so no GPU pilot or real evaluation is released.
> The 155 final-target count below excludes inner calibration reconstruction;
> it is not a validated cost estimate for the new nested design.

Status: proposal only, not an executed experiment or authorization to start
GPU training. The current request authorizes a fixed replay if suitable
existing assets are found, and a rebuild plan otherwise. No eligible bundle
was found; the original R15 protocol stays frozen and disabled.

## Priority 0: resolve provenance and recipe feasibility before compute

Obtain the original training source/configuration for the v5 adapter families,
Nishant two-stage members, main DeBERTa, the four COMP LoRA members, UltraBERT,
ATLAS and any enabled token-attention heads. Include stage-one external-data
membership and all preprocessing/selection rules. An ATLAS training-source
hash is available but its trainer was not found in the inspected package;
the referenced lightningv08 fivefold source is currently unavailable.

Also recover the COMP blend fitting procedure and the Gaussian parameter
selection provenance. The frozen weights were learned from earlier OOF and
135-prompt fits. Do not declare them independent of the new outer validation
groups. There are two materially different scientific scopes:

1. Exact public coefficients held fixed: a conditional descriptive replay of
   an already selected recipe. It cannot be relabelled as a proof that every
   trained pipeline dependency excludes the evaluated prompts.
2. Strict complete-pipeline OOF: fit learned combination parameters within
   outer training, using inner OOF where required. This needs a revised,
   preregistered producer definition and extra inner training. Keep the sole
   final comparison at strength 1.0 versus 0.5, but identify this reconstructed
   producer separately from the exact R14 production recipe.

The existing R15 admission requires the strict exclusions. No scope change
is implemented in this plan. If source/provenance cannot support the chosen
producer definition, stop before model acquisition or training. No guessed
hyperparameters, omitted members or substituted weak SVC baseline.

Budget for this preparation: <=4 CPU-hours, <=100 MiB of source/config/metadata
downloads, 0 GPU jobs, 0 official-label evaluations. Bulk weights and the
99.82-GB public collection are not needed to answer the provenance questions.

## Priority 1: development-only data and fold contract

Require a standalone export containing only the original 2159 development
rows and texts, bound to the hashes in `r15_protocol_20261029.json`. The R2
cache supplies existing metadata but does not persist texts. Do not use it
as an excuse to reopen or rescore the old protected 30 prompts.

Keep 120 prompts and the existing five GroupKFold folds, so each outer fold
has 96 training / 24 validation prompts. Expected validation row counts are
431 or 432; training rows are 1727 or 1728. Verify the actual frozen fold hash
rather than creating a new assignment. A source whose 150-prompt fold plan
differs cannot be repaired merely by selecting 2159 prediction rows: its
training history remains unchanged.

Every task-supervised stage, external-data filter, vocabulary/projection,
readout, Gaussian fit, prior and learned gate/combination parameter must bind
to its allowed training envelope. Require per-source exclusions before an
external text enters training. When provenance cannot certify prompt
separation, reject the source; do not inspect protected texts to improvise a
new overlap filter. Generic task-independent base weights can be shared after
their provenance is verified. Task-fitted adapters/features cannot be reused
across an excluded validation boundary.

## Priority 2: preserve topology with a fold-safe training DAG

The following are required final checkpoint/bundle **targets**, not a count
of Python processes, optimizer runs or independent stage-one training jobs.
Shared safe base models and legitimate shared stage-one work can reduce
duplicate loading/work. Inner OOF calibration can add training beyond this
table.

| Family | Targets per outer fold | Five outer folds |
|---|---:|---:|
| q14v5 / q8v5 / ll8v5, six adapters per family | 18 | 90 |
| n14b + q25_14 | 2 | 10 |
| Main fine-tuned DeBERTa | 1 | 5 |
| Four COMP LoRA final bundles | 4 | 20 |
| COMP UltraBERT five-checkpoint average | 5 | 25 |
| ATLAS bundle, internally 18 source-specific adapters | 1 | 5 |
| **Required target total** | **31** | **155** |
| Optional token-attention heads, only if the reviewed runtime includes them | 0-2 | 0-10 |

Evidence: [`reference_public660_v32.ipynb`](../notebooks/reference_public660_v32.ipynb)
lines 69-73, 1004-1012, 1805-1825 and 1929-1934; ATLAS's 18-adapter count comes from its public data card, not
from the notebook's one-bundle API. Inference through all six v5 adapters or
all five UltraBERT checkpoints does not create OOF if any has trained on the
outer validation prompts. Their rebuilt versions must all exclude those
prompts. Internal folds must remain inside the outer-training envelope.

Additional per-outer-fold work: main TF-IDF, distortion and SVC; COMP
stylegram/LightGBM/plain and centred readouts/family gates; two main and four
COMP Gaussian fits; fold-specific priors, grouping and external-data
selection. Freeze non-learned settings before execution. Never pool five
outer-validation folds into one transductive inference batch. The same
upstream predictions and ATLAS selected/scored rows serve both comparison
arms.

Existing partial assets can contribute source code, schemas and reference
diagnostics after review. Their full-data feature banks and mismatched OOF
are not training inputs or baseline predictions for this protocol.

## Priority 3: measure before allocating the full rebuild budget

Live local snapshot on 2026-10-29:

- RTX 4060 Laptop GPU: 8188 MiB total, 3829 MiB free at observation.
- Host RAM: 31.73 GiB total, 10.43 GiB available.
- D: free space: 87.60 GiB.

This machine is suitable for the bounded CPU replay, source preparation and
metadata verification. It has not demonstrated capacity for the unchanged
14B/two-stage training workload. No large download, quantization substitution
or offload experiment is part of this plan's current execution.

Proposed rebuild capacity envelope: **2 GPUs with at least 48 GiB each,
128 GiB host RAM, 500 GiB free SSD, 16 CPU cores**. This is a planning reserve,
not a measured minimum. Two devices preserve the production code's two-device
execution structure. Do not silently change numerical precision to fit a
smaller GPU. Validate the exact producer and memory use in a bounded pilot.

| Phase | Proposed hard ceiling | Release / stop condition |
|---|---|---|
| Source/data/protocol preparation | 4 CPU-hours; 100 MiB source metadata; 0 GPU-hours | Stop if any required trainer, split, text export, exclusion record or learned-parameter scope is unresolved. |
| Representative pilot | **8 aggregate GPU-hours**; no accuracy-based selection | Cover the largest 14B training stage, ATLAS and an encoder/feature path. At most 200 optimizer steps per training probe, plus bounded inference profiling. No official validation metrics. Stop on OOM, nonfinite values, provenance failure or budget expiry. |
| Full reconstruction | **250 aggregate GPU-hours including the pilot**, 48 CPU-hours, 500 GiB SSD | This is a proposed ceiling, not a duration estimate or guarantee. Before launch, the pilot must support the DAG estimate plus 25% reserve within the remaining ceiling. Otherwise stop and re-budget; never remove members to force a fit. |
| Final producer audit and fixed replay | Original <=120 CPU seconds / <=2 GiB replay; 5 typography fits, 10 arm applications, one comparison | Requires admitted hashes, provenance and independent baseline parity. Use a persistent single-use claim and supervisor. |

Aggregate GPU-hours are the sum of allocated GPU time: two GPUs allocated
for four hours consume eight GPU-hours. No cloud price, paid purchase or
allocation is assumed or authorized here. The 250-hour ceiling is a
resource-control proposal, not a claim that a source-incomplete pipeline can
be rebuilt in that time.

Measured reference anchors only: public S7 V1 / 358911769 reports
2h54m45s on T4 x2 with six output job directories; V32 / 363222852 reports
2h44m11s on T4 x2. They are different jobs with already supplied assets, not
benchmarks for new 48-GiB hardware or complete fold reconstruction. Do not
multiply one public inference runtime by five and call it the training cost.

After the pilot, derive the compute budget from actual stages:

`H_projected = H_pilot + 1.25 * sum(remaining_stage_count * measured_stage_gpu_hours)`

Include stage-one pools, per-fold fine-tuning, required inner OOF, feature
extraction, classical fitting, five complete inference passes and output
verification. Deduplicate only stages whose weights and input membership are
provably identical. Count shared model downloads once; do not count final
checkpoint targets as automatically independent full-training runs.

## Priority 4: produce and admit the minimal replay bundle

Per fold, preserve original-scale `_v38_before_hard` and independently
generated full-strength R14-compatible final logits, original ID/label order,
all member/feature/checkpoint hashes, actual training exclusions, external
selection rules, prior/fit scope, and ATLAS selected/scored IDs and counts.
Export only development rows; no protected-30 or hidden-test records.

Two `(2159,18)` float64 prediction arrays contain 621,792 numeric bytes,
about 0.593 MiB before headers. Model weights need not accompany the replay
bundle if the producer audit is independently reviewable. Metadata/texts and
provenance are extra. The contract is in `R15_BUNDLE_CONTRACT_20261029.md`.

Admission must verify the actual bytes and producer evidence, freeze a new
execution record and enforce a single run. Then apply the original gate:
BA gain >=0.005, positive lower paired 95% bound, at least 4/5 folds improve,
no fold loss >0.01 and no class-recall loss >0.05. Failed gate stops the fixed
0.5 hypothesis. No retuning, push, Save or submission follows automatically.

## Present stop point

Priority 0 is not satisfied: complete OOF is missing, training/exclusion
provenance is incomplete, some trainer sources are unavailable, and the
learned-coefficient scope needs resolution. No GPU pilot, data extraction,
new training or real replay is launched. This proposal makes the dependency
and budget explicit; it is not a completed reconstruction.
