# R15 P0 continuation: export search and source provenance

Date: 2026-10-29. Base local commit: `b98698a5cb592b95fa0ed196d530a7596e8d6598`.
This continues the same cumulative 4 CPU-hour preparation budget. The user's
latest instruction allows a later local GPU experiment with modest load and
memory use, while retaining no push/Save/submission now and no old protected
30-prompt access. The earlier P0 report's GPU authorization statement is
historical; this report records the newer permission. No GPU work ran here.

**Outcome: more original configuration and source-attribution evidence is
archived, but the original 2159-row development text export and several exact
trainers remain unavailable. No data/training admission or predictive gain.**
The previous replay and intake implementations are unchanged and their tests
were not rerun. Newly acquired source cells received syntax parsing only.

## Development export search

The expanded filename/metadata search covered the project tree, `D:/Downloads`,
the user's Downloads/Documents/Desktop, related Drive file metadata and public
Kaggle owner/search metadata. The two competition ZIPs contain only raw train,
test and sample-submission members; only their member names/sizes were read.
No preexisting development-only export or certified byte-range index was found.
This does not exclude private assets or locations outside the searched scope.

The R6 private asset lead was resolved from the historical packaging source:
`name-that-llm/scripts/build_round6_assets.py` and `package_round6_runtime.py`
package encoders, heads, vocabulary/classifier state and runtime files, not
the original development texts. The historical style fitting code does not
persist the original training passages in that package. No model was unpickled.

Public Kaggle metadata returned no public datasets for `zephyrpong`; the only
visible `zdx13143` dataset was the unrelated, undescriptive `exctk2`. Searches
for 2159/120-prompt development exports and the frozen identity hashes yielded
no qualified candidate. No inference about unavailable private holdings follows.
The user also reported not knowing of a separately saved export.

The [existing intake command](../scripts/r15_validate_dev_export.py) remains
executable when a reviewed export exists. It was not rerun with the same missing
input, and no fake sidecars or new full-CSV extraction were created. If no
preisolated source can be supplied, the current data boundary prevents recovery
from the raw competition file. More training permission alone does not resolve
that input dependency. All 120/30 identities and historical counters are retained.

## Newly obtained original configuration

The [source archive](../reference_sources/r15_p0b/README.md) includes exact bytes
for five Nishant config files, three ATLAS metadata files, five source cells,
and a public data-card record. Every file is hash-bound in its source index.

### Nishant part two: correct dataset and member identities

The actual dataset is [nishantdahal/ntl-handwriting-blend V1](https://www.kaggle.com/datasets/nishantdahal/ntl-handwriting-blend),
whose title is **NTL two-stage LoRA part two**. Its 15-file listing is complete.
It contains checkpoints/configs and `pool_v4.parquet`, but no training program.

- `main.n14b`: Qwen3-14B-Base, keep 27; layers 2/4/6/9/13/17/22/27.
- `main.q25_14`: Qwen2.5-14B, keep 32; layers 3/5/8/11/16/21/27/32.
- Both saved configs specify max_len 447, EOS 151643, rank 16/alpha 32,
  LoRA dropout .05/head dropout .1, adapter/head LR 5e-5/5e-4, WD .01,
  warmup .1, epochs 2, six triplets per step, seed 42, gradient checkpointing.
  These are saved config fields, not proof of complete effective execution.
- `main.deb` is a **fully fine-tuned DeBERTa-v3-large**, according to this
  dataset's card. Its meta file provides max_len 384 and `mean_last` pooling,
  but no optimizer/scheduler configuration. It is distinct from the frozen
  encoder plus LR member `comp.debv3l` and the optional token-attention head.

The main/COMP distinction is independently grounded in frozen V32 notebook
lines 65-67 and 1932-1934 (main checkpoint) versus 1000-1003 (COMP frozen head).
An exploratory lead conflated these branches; that interpretation was rejected.

The saved LoRA adapter configs have `revision: null`; exact base revision and
stage-one initialization are not established by these files. Actual stage-one
and stage-two programs, membership and nested exclusion evidence remain missing.
The `train_info` timings/losses are upstream diagnostics, not local performance
or estimates for this laptop.

The newly downloaded [public-pool notebook V2 / 359887828](https://www.kaggle.com/code/nishantdahal/two-stage-lora-on-the-public-pool?scriptVersionId=359887828)
fits TF-IDF/SVC on official plus pool data (cell 03), then merges existing LoRA
weights and performs inference (cells 05/07). It contains no neural optimizer
or LoRA training loop. Its V1 checkpoint family differs from the part-two
members; it does not close their trainer dependency.

### ATLAS: detailed config acquired, implementation still unresolved

The [V3 checkpoint config](../reference_sources/r15_p0b/atlas/checkpoint.json)
pins Qwen3-8B-Base revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`, seed 17,
rank 16/alpha 32/dropout .05, LR 1e-4/WD .01, five epochs, effective/micro batch
128/4, max_len 512, clip 1 and warmup .05; fp16 base/fp32 adapter. The dataset
version was 3 before and after acquisition. It declares all 2699 official rows.

The exact training-source manifest names missing `ntl_atlas/train.py` hash
`80440fa2589236b0cf6f1d3f9388f049941a4474050c55fe6729ee677a28ef87` and config
hash `bd0d037376f5b63f36953f84c92bcbbe29495a4a9e0c2a01c0708ac839847252`.
The [method-reference commit](https://github.com/komseok0109/ATLAS/commit/93c1b15da6143161e222eecee7ebcdc3c60645a9)
contains an upstream trainer, but the Kaggle bundle declares an independent
PyTorch implementation. It cannot be substituted as the exact NTL trainer.

There is also a source-history discrepancy requiring explanation: the training
manifest names `ntl_atlas/model.py` hash `19d2a047...a94c4`, while the distributed
bundle names `3fb9394b...26fb`. This may reflect a later inference change; no
equivalence was established. Checkpoint and manifest bytes do match the bundle's
hashes. The standalone original config file was not recovered; embedded config
fields do not establish byte identity with its reported hash.

## External-data exclusion evidence and remaining trainers

The part-two data card identifies a 75,783-passage pool built from 25 datasets,
plus an additional `aswathvanan/ntl-proxy-passages` stage-one source for Qwen3.
It describes exact train-answer deduplication. Sources include same-prompt and
proxy generations. This attribution is useful, but it does not certify prompt
independence or exclusion of any R15 outer/inner validation prompt. No pool was
downloaded or inspected, and the presence of named sources is not proof that
specific held-out prompts overlap.

The [Claude LoRA card](https://www.kaggle.com/datasets/zbigniewkirchner/ntl-qwen3-8b-lora-claude-head)
says its 18-way heads use original data and its auxiliary Claude head also uses
Extra API Passages. Because the adapter is shared, a head-specific input rule
alone cannot establish isolation of the resulting features. The exact linked
[trainer](https://www.kaggle.com/code/zbigniewkirchner/ntl-lora24-claude-aux)
returns “We can't find that page.” Its final checkpoint is explicitly all-2699
training and is not eligible for R15 validation.

Public LoRA's linked fivefold trainer remains inaccessible. The v5 120k-pool
trainer/build procedure and UltraBERT fine-tuning source remain unlocated in the
bounded public search. No required member was silently dropped or replaced.

## Compute and executable next steps

A read-only snapshot showed RTX 4060 Laptop, 8188 MiB total, **4006 MiB already
used**, 5% utilization and 45 C. This is one observation, not a peak measurement.
No GPU context or model job was started. Future local work should use one task,
preserve other processes, reserve at least 2 GiB of device memory, and first
measure a short admitted workload. These are future design constraints, not an
implemented utilization limiter. No claim that the full 14B nested rebuild fits
this laptop or the old 250 GPU-hour proposal is supported.

1. Supply an already isolated 2159-row artifact with reviewed provenance; run
   `python scripts/r15_validate_dev_export.py --directory outputs/r15_dev_export`.
2. Obtain exact remaining trainers and row-level external membership, resolve
   the ATLAS source revision difference, and freeze the inner calibration
   procedure. Existing full-fit adapters remain reference assets only.
3. Once inputs are admitted, use the user's new local GPU permission for a
   preregistered small measurement within the memory/temperature/CPU envelope;
   release further work only if it is feasible. No permission re-request is
   needed merely because it uses the GPU, but missing data is still blocking.

The continuation receipt records measured preparation CPU, unmetered tool/UI
overhead, downloaded source/metadata bytes, all zero execution counters and
new-artifact checks. No previous synthetic test or no-input readiness check was
repeated. There is no new BA, Public Score or private result.
