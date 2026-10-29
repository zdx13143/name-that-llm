# P0: S7 fold-isolated source candidate and local memory bound

2026-10-29, continuing from `b2bbc7b9d6c146b6fc41599c7c7137ad00c89262`.
The preceding turn resolved the 2159-row development-text dependency. This
turn implements source preparation for the required `comp.lora24_seed7`
member; it does not start a training job or relax the complete-ensemble goal.

## Implemented source adaptation

[The builder](../scripts/build_r15_s7_adapter.py) accepts only the exact archived
Lucas Boesen S7 V1 / 358911769 source hash. It extracts the literal worker
without importing the training notebook, then applies individually checked
source changes. Attribution and Apache-2.0 licensing are preserved.

- Replace raw CSV and seed-0 fold discovery with the admitted development
  identities and fixed outer/inner plan. No original CSV is opened.
- Materialize fitting rows with labels; prediction rows contain only ID,
  prompt ID and text. The generated worker gives their unused truth positions
  a sentinel and computes no validation accuracy.
- Each outer fold has one final-style member on its 96 training prompts.
  Each of its five inner members trains only inside that outer train. There
  are **30 source job envelopes**, not 30 launched jobs or 30 ensemble members.
- Keep the original triplet sampler, representation, normalization, two-head
  loss, dropout, optimizer, schedule, precision, LoRA settings, batch/accumulation
  and epochs. Use the original final-style sampling seed `7 + 99` per envelope.
  The fivefold population and final-style routing identify a distinct nested
  producer, not exact reproduction of the original full-data job.
- Save adapter, normalization weights, heads and tokenizer for every isolated
  bundle. The original trainer only saved full bundles for its all-data final
  job, which could not supply this reconstruction.
- Keep `valid_features.npz` as a parity anchor for the first three fitting
  rows. Write held-out logits separately, without labels or accuracy metrics.

The frozen R14 consumer selects only S7's `job_final` bundle, uses its plain
readout at weight 0.113, and does not add its centered readout. The original
training objective still includes both heads and is preserved. Consumer
mapping is in `reference_public660_v32.ipynb:976-1011,1674-1725`; checkpoints,
norms, readouts and tokenizer must remain paired. Inner models are for inner
prediction/calibration, not an extra five-member ensemble.

## Actual preparation and tests

The builder read only the already admitted development output, checked its
approval/review/admission bindings, and recomputed the nested plan. It generated:

- `outputs/r15_s7_adapter/worker.py`: adapted source candidate;
- `outputs/r15_s7_adapter/preparation.json`: all 30 envelopes, source/data/fold
  hashes, fixed recipe and sampling counts, with training/scoring disabled.

Only the original NumPy sampler function was executed, once per planned
epoch/envelope, to check that each microbatch contains three fitting rows from
one prompt and stays inside the selected training range. No tokenizer, model,
torch, optimizer or encoder forward was run. Its planned counts total **7200
optimizer updates** across the 30 fits; this is not a duration measurement.

New tests: **7/7 PASS**, using artificial data and the fixed source. They cover
all outer/inner exclusions, prediction-label removal, canonical class mapping,
fit-only parity anchors, the unchanged sampler/training core, source drift,
disabled execution and the conservative memory calculation. An independent
read-only review found no material issue and confirmed the runtime limitation.

**The generated worker is not a runnable training release.** Its input entry
stops before torch imports. Base-file verification, a resource supervisor,
measured training budget and full nested-producer integration remain work to
complete when the necessary resources and remaining evidence exist. The
unimplemented release is explicit; compiling this source is not runtime proof.

## Verified base identity candidate and memory constraint

The bounded local inventory found a ModernBERT snapshot but no visible Qwen3
base, tokenizer or adapter under the project's model/cache locations or the
Hugging Face cache. Archives were not unpacked, so this is a scoped inventory.

Only **10,341 bytes** of official metadata/config/model card were downloaded
from [Qwen's repository](https://huggingface.co/Qwen/Qwen3-8B-Base).
The candidate revision is `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`.
This pins a generic base for the distinct reconstruction; it does not prove
the unspecified revision used in the historical Kaggle S7 run. No weights
or tokenizer payload were downloaded. See
[the metadata index](../reference_sources/r15_p0e/qwen3_8b_base/source_index.json).

The [pinned official config](https://huggingface.co/Qwen/Qwen3-8B-Base/blob/49e3418fbbbca6ecbdf9608b4d22e5a407081db4/config.json)
has hidden size 4096, intermediate size 12288, vocabulary 151936, 32 query
heads, 8 key/value heads and head dimension 128. For the trainer's 24 layers:

`parameters_floor = 151936*4096 + 24*(2*4096*(32*128) + 2*4096*(8*128) + 3*4096*12288)`

This gives **5,252,841,472 matrix parameters**, or **9.7841796875 GiB** at
FP16. It excludes norms, LoRA, heads, gradients, optimizer state, activations
and allocation overhead. It already exceeds this laptop's entire 8188 MiB
GPU memory. The current read-only snapshot showed 4153 MiB free, 3804 MiB
used and 9% GPU utilization; those usage numbers are only a point-in-time
observation. The full-GPU-resident original implementation cannot fit even
if the device were otherwise idle. No OOM probe or large download was needed.

## Remaining work and evidence boundary

S7's data routing and source adaptation are now prepared. Its actual runtime
and weights are still not admitted. The full pipeline also still lacks the
other exact trainer/stage data records, external prompt lineage/exclusions
and precise nested calibration specification listed in
[the source inventory](../configs/r15_p0_source_inventory_20261029.json).

The local low-load permission does not establish capacity for this unchanged
recipe. A different precision/offload implementation would need a separately
registered producer, compatibility evidence and measured resource gate; it
was not substituted here. This result changes the next compute decision:
do not attempt original S7 training on this laptop. Continue source/provenance
closure or obtain a suitable execution environment before releasing training.

The cumulative preparation ceiling remains 4 CPU-hours. No model training,
encoder forward, official-label accuracy evaluation, raw-source reread,
protected-row access, Git push, Kaggle Save/submission or new score occurred
in this turn. See [the execution receipt](../configs/r15_p0e_receipt_20261029.json).
