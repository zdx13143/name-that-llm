# R18 active execution handoff — 2026-11-02

## CURRENT STATE — overrides the historical checkpoint below

Latest user asked to continue and submit a validated result. V1 failed in the
new fingerprint instrumentation. Root reproduced float32 index overflow using
four-byte CPU storage, then committed a disclosed, bounded engineering-retry
amendment BEFORE Save 2 (99d0414). Total two-Save limit did not increase.

**V2 / 364615652 / kernel 141979156 is COMPLETE. Real-model preflight PASSED.**
Saved 2026-11-02 01:07:05 Singapore. UI runtime252.5s; preflight body182.754s.
12 adapters completed,36 lifecycle events,finite2x18 outputs,matching family
fingerprints,all parameters cuda:0/1; released allocation8,519,680bytes/card.
Candidate notebooks/team_e659_r18_adapter_isolation_v2.ipynb SHA256:
867515c41bd5a7ceef51efb306ddc2235b3c4952ec915d26b41953f2cc20dc83.
V1 candidate and original runtime remain unchanged. V2 differs only cell4
fingerprint index construction/empty parameter guard. Nine boundary cases
passed locally; no real local models or data were used.

**DO NOT SUBMIT WITHOUT RESOLVING THE PENDING SCOPE CHOICE.** Actual saved inputs
differ from the frozen request in TWO external datasets:
- abdullah1afify/ntl-generations: requested4,actual9.
- lucasboesen/ntl-free-proxy-generations: requested11,actual12.
Fresh exact R16 V2 page still shows4/11. This is actual binding drift, not
merely generic latest-version display. CLI metadata version suffix acceptance
did NOT establish locking. Official docs only describe owner/slug inputs.
Other12datasets,7models,competition match. Eight kernel versions mapped using
explicit versioned request + returned slug + exact LastUpdated timestamp from
actual input UI against fixed upstream version pages; SDK lacks attached IDs.

Root asked user asynchronously: accept current V2 once as an exploratory
changed-data submission (same8h Python/8.3h total limit), or preserve frozen
versions and do not submit. No answer yet at this checkpoint. Their response
can resolve scope; do not treat waiting or silence as permission. Editor
edit/run/364615652 still Loading; AddInput/Import/Save menu disabled. No third
Save permitted by current amendment. There are ZERO formal submissions/new
scores. R16 .659 remains best established result. No new automation.

Artifact gate: `python scripts/verify_r18_v2_preflight.py` (system Python has
pandas; torch venv does not). Expected exit1, writes
configs/r18_v2_release_gate_20261102.json with both version mismatches and passed
model checks. Saved code readback all9cells exact; see V2 source receipt.
All downloaded small files were staged D:/Downloads then moved with SHA checks
into outputs/r18/v2_remote_evidence_20261102. No large output/model download.

If user explicitly accepts changed data, first update scope/report/receipt to
label the combined model-lifecycle + external-data experiment. Then submit
EXACT V2 once using official CLI:
`kaggle competitions submit name-that-llm -k zephyrpong/ntl-r18-adapter-isolation-20261101 -v 2 -f submission.parquet -m <clear description>`
CLI help confirmed these flags. Read/save returned ID, inspect submissions
page/API status, and do not confuse Save COMPLETE with formal score. Never
submit V1 or the local three-row placeholder as a standalone file.

Browser after reset: call cua.rewriteDocumentation(). Existing bindings:
r18Nov2Tab (1127720282) complete Logs, marked deliverable; r18SavedInputTab
(1127720297) V2 actual inputs; r18R16InputTab(1127720291) exact baseline inputs;
r18BindingEditor(1127720300) broken editor. r18V2InputEvidence/r18KernelPins hold
saved UI evidence, also retained on disk. No scouts running.

## HISTORICAL V1-running checkpoint — retained for provenance only

All status/next-action text below predates V1 failure and V2. Follow CURRENT
STATE above, not these obsolete execution steps.

Latest user instruction: continue making progress, submit a validated result to
Kaggle and show the score. Same R18 upload, Save preflight and one formal
submission are explicitly authorized. No Git push, new automation, new round,
local real-model forward/training/download, raw CSV or protected holdout access.

## Immediate state

- Worktree: D:/kaggle/name_that_llm/outputs/worktrees/team-e659-r18-20261101
- Branch: exp/team-e659-r18-20261101; previous committed HEAD bc46695.
- R18 candidate SHA256 unchanged:
  4c3eb9436f36af07aad55b92acaa9e1491aff3d352ee9fc78209c109851a735b.
- Local final 10/10 tests already passed. Do not rerun without source change.
- **R18 V1 / scriptVersionId 364611063 / kernel 141979156 is saved and running
  the REAL-ASSET PREFLIGHT. One Save used; zero formal submissions.**
- Saved at Singapore 2026-11-02 00:49:53. Actual page: GPU T4 x2.
- Last observed running 258.8 seconds. Log now says:
  `R18 starting r18_preflight.py ; remaining seconds 742`.
- Normal Save subprocess captures its output and prints it at finish/failure,
  so adapter logs do not stream while the child is active.
- URL: https://www.kaggle.com/code/zephyrpong/ntl-r18-adapter-isolation-20261101

## Gates and stop conditions

Read docs/R18_PLAN_20261101.md and current report/receipt. Save preflight total
900 seconds including initialization. Two artificial texts, both q14v5/q8v5
families, six adapters each, finite (2,18), identical sampled fresh-base
fingerprints, released old Torch allocations, all GPU parameters, T4 x2.
Max two Saves, second ONLY for a packaging error before real-model preflight.
**No retry after real model/runtime failure.** Formal submission exactly once
after all gates. Formal Python ceiling 8h, total envelope8.3h, Internet off.
Keep same30 inputs, coefficients and typography0.5. Missing main.ll8v5 and
comp.lora24_public remain explicit. Baseline R16 V2/363487968 Public .659.
No full isolated OOF or local accuracy claim. Public>.659 is the criterion.

## What unblocked uploading

Editor remains broken. Existing official OAuth credentials.json was overlooked
by earlier standard-config-only checks. The CLI refreshed it normally; the
first attempt failed only on workspace-external write permission. Retrying
official CLI status with require_escalated succeeded. No secrets read/printed
or copied. CLI2.2.4 is installed. Correct earlier 'CLI unconfigured' conclusion.

Official CLI `kernels pull owner/slug/2` returned403; it appends /2 to kernel
slug. Unversioned R16 pull succeeded, and all seven source cells exactly match
the frozen scored R16. This source is verified, not accepted on requested URL.

R16 external kernel metadata supplies actual Docker digest:
gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f
`machine_shape=NvidiaTeslaT4`, Private, GPU enabled, TPU disabled, Internet off.
The UI label says Latest Container Image, but actual saved digest matches R16.

All14 dataset version numbers were read from the exact R16V2 input page; all8
kernel versions bound to visible fork-version IDs and version numbers. Evidence:
outputs/r18/pinned_input_ui_20261102.json. Seven model exact /1 refs retained.
scripts/prepare_r18_cli_release.py packages immutable candidate + versioned
kernel-metadata.json in outputs/r18/cli_release. It verified baseline parity,
all30 refs, settings. configs/r18_cli_release_20261102.json records identity.

Executed once:
`kaggle kernels push -p outputs/r18/cli_release --accelerator NvidiaTeslaT4`
Result: Version1 successfully pushed. Log outputs/r18/cli_push_v1_20261102.log.
This used the existing notebook slug; no duplicate notebook.

## Saved source verification already done

CLI pulled R18 V1 into D:/Downloads then exact2 files moved into
outputs/r18/v1_cli_readback_20261102. R16 files similarly retained under
outputs/r18/r16_cli_readback_20261102. No model/large output download.
scripts/verify_r18_saved_source.py PASSED: all9 code cells exactly match frozen
candidate; Docker, Private, Internet off, machine shape match. Source slugs
match. CLI omits dataset/kernel version suffixes in readback metadata, so
**actual saved input UI version check is still required after preflight ends**.
While running, the /input page redirects to Logs.
Proof: configs/r18_v1_source_readback_20261102.json.

## Next actions

1. Monitor V1 to terminal. Preserve logs. Use max60-second waits and concise
   updates. If failed during real preflight, stop this round, no formal submit.
2. If successful, download ONLY small r18_preflight_receipt.json,
   r18_lifecycle.jsonl, r18_preflight.log, plus run log. `kaggle kernels output`
   supports `--file-pattern` regex. Do not download copied DeBERTa/weights.
   Stage D:/Downloads, then retain required files in project. CLI output version
   suffix has historical provenance problems; bind actual files to V1 and ensure
   no later save exists. Only V1 currently exists.
3. Verify actual12 adapters, finite outputs, matching fingerprints, memory
   release and times; source check already passed. Check saved input versions
   against baseline UI evidence. Then submit EXACT V1 once, via official CLI
   (inspect submit help for notebook flag) or normal page More options.
4. Record precise version/status/score; Pending/Running is not score. No
   automation is authorized. Preserve R16 .659 baseline and no private claim.
5. Update README/report/receipt, stage explicit files and small ignored evidence,
   commit locally; no push. Current new scripts/configs/evidence remain uncommitted.

## Browser continuation

Use existing Chrome browser id3 through cua_repl. After context transition call
`await cua.rewriteDocumentation()` before UI use. Existing CUA bindings:
- r18Nov2Tab: id1127720282, saved R18 Logs page.
- r18SavedInputTab: id1127720297, desired exactV1 input but currently Logs.
- r18R16InputTab: id1127720291, baseline input details.
- r18KernelPinTab: id1127720294, last inspected Instyle exact version.
- r18Nov2Fs: node:fs/promises for saving screenshots/states to evidence.

Current UI logs saved in outputs/r18/v1_running_logs_20261102.txt. For compact
progress use getAXState({emit:false,disableDiffing:true}) and filter log/status
lines, while retaining full state. Screenshot actual results and embed final.
There are no running scouts. Use fresh one-shot read-only scouts only if needed,
fork_turns=none; after dispatch wait until terminal, as user requires.
