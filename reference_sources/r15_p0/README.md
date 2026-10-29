# R15 priority-zero source archive

These four unmodified source snapshots were acquired through Kaggle's normal
Download control on 2026-10-29. They are reference text (`.py.txt`), not runnable
training entrypoints. No upstream script, worker, model or notebook was executed.
For notebooks, only code-cell source was retained; saved outputs were not
reviewed or copied into this archive. Do not use their historical score comments
as current evidence.

| Snapshot | Author / exact Kaggle version | R14 dependency |
|---|---|---|
| [s7_v1.py.txt](s7_v1.py.txt) | [Lucas Boesen, 358911769](https://www.kaggle.com/code/lucasboesen/name-that-llm-train-lora24-s7?scriptVersionId=358911769) | COMP `lora24_seed7` |
| [instyle_v2.py.txt](instyle_v2.py.txt) | [Travis Nguyen, 359652568](https://www.kaggle.com/code/vnton37nguyn/ntl-train-lora-qwen25-instyle?scriptVersionId=359652568), adapted from Lucas Boesen | COMP `lora20_q25_instyle` |
| [tokattn_v1.py.txt](tokattn_v1.py.txt) | [Lucas Boesen, 358928013](https://www.kaggle.com/code/lucasboesen/name-that-llm-train-tokattn?scriptVersionId=358928013), recipe attribution kirilltym | Optional `tokattn_q35` |
| [tokattn2_v1.py.txt](tokattn2_v1.py.txt) | [Lucas Boesen, 358948428](https://www.kaggle.com/code/lucasboesen/name-that-llm-train-tokattn2?scriptVersionId=358948428), recipe attribution kirilltym | Optional `tokattn_deb`; its extra q8l22 head is not active in R14 |

All four source pages identify Apache License 2.0. The complete license is
included as [LICENSE-2.0.txt](LICENSE-2.0.txt). Original author comments and
attributions are retained. This local archive adds filenames, this notice and
the index; the source contents themselves are unchanged.

[source_index.json](source_index.json) binds source bytes, original download
hashes and versions. Literal top-level constants were extracted using Python
AST/literal evaluation; this does not execute imports or workers. The tokattn2
backbone loop overrides its top-level LAYER/KEEP defaults, so those literal
defaults alone are not a complete effective configuration. Architecture,
loss, precision and schedule details remain in the source.

All four programs originally open full official `train.csv`; none may be run
as-is under R15. See [the P0 audit](../../docs/R15_P0_REPORT_20261029.md) for
scope defects, excluded external inputs and missing required trainers.
