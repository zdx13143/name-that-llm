# P0 continuation: acquired configuration and provenance evidence

Local-only source/configuration evidence acquired on 2026-10-29. Nothing here
was imported or executed; no weights, predictions or passage data were fetched.
The [index](source_index.json) binds all 14 archived files by exact byte hashes.
Raw API acquisition receipts remain under `outputs/r15_p0b/` and are bound by
the [execution receipt](../../configs/r15_p0b_receipt_20261029.json).

- `atlas/`: original public checkpoint config, training-source manifest and
  bundle metadata from [Joseph Adamski's ATLAS dataset V3](https://www.kaggle.com/datasets/josephadamski91/ntl-atlas-qwen3-8b-v1).
  These are metadata, not the missing NTL training implementation. The method
  reference [ATLAS commit 93c1b15](https://github.com/komseok0109/ATLAS/commit/93c1b15da6143161e222eecee7ebcdc3c60645a9)
  is not an exact substitute for that independent NTL implementation.
- `nishant_part_two/`: five original JSON config/meta files from
  [Nishant Dahal's part-two assets V1](https://www.kaggle.com/datasets/nishantdahal/ntl-handwriting-blend).
  The data card declares CC BY-SA 4.0. No pool or checkpoint payload is included.
- `nishant_359887828/`: five unchanged source cells from
  [Two-stage LoRA on the public pool V2](https://www.kaggle.com/code/nishantdahal/two-stage-lora-on-the-public-pool?scriptVersionId=359887828).
  This fits a classical SVC and loads existing LoRA checkpoints for inference;
  it does not contain the LoRA training loop. Local reference copies only.
- `data_cards.json`: selected public metadata fields and descriptions, with
  original raw-response hashes; includes the exact Claude auxiliary trainer
  link and the part-two pool source attribution. These author statements are
  not independently verified row-level exclusion certificates.

See the [continuation report](../../docs/R15_P0_CONTINUATION_20261029.md) for
configuration details, the two ATLAS model-source hashes, and admission limits.
No required member or development export was newly admitted by this acquisition.
