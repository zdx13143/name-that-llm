#!/usr/bin/env bash
# Claude -> Codex delegation wrapper (pattern adapted from openai/codex-plugin-cc "rescue").
# Usage:
#   codex_task.sh <task-id> <prompt-file> [read|write|net] [workdir]
#   codex_task.sh --resume <task-id> <prompt-file> [read|write|net] [workdir]
# Modes:
#   read  -> read-only sandbox (review / diagnosis / planning)
#   write -> workspace-write, no network (code + local tests)
#   net   -> workspace-write + network (kaggle CLI, gh); never for training data pulls
# Outputs land in <repo>/outputs/codex_runs/<task-id>/ (git-ignored):
#   prompt.md, events.jsonl, last.md (final message), session_id, exit_code
set -uo pipefail

# The ~/.codex/.sandbox-bin copy lacks codex-code-mode-host.exe, so use the app-bundled CLI.
if [[ -z "${CODEX_BIN:-}" ]]; then
  pkg="$(powershell.exe -NoProfile -Command "(Get-AppxPackage OpenAI.Codex).InstallLocation" | tr -d '')"
  CODEX_BIN="$(cygpath -u "$pkg")/app/resources/codex.exe"
fi
CODEX="$CODEX_BIN"
REPO_ROOT="D:/kaggle/name_that_llm"

resume=0
if [[ "${1:-}" == "--resume" ]]; then resume=1; shift; fi
task_id="${1:?task id}"; prompt_file="${2:?prompt file}"; mode="${3:-read}"; workdir="${4:-$REPO_ROOT}"

run_dir="$REPO_ROOT/outputs/codex_runs/$task_id"
mkdir -p "$run_dir"
cp "$prompt_file" "$run_dir/prompt$( ((resume)) && echo "_resume_$(date +%H%M%S)").md"

case "$mode" in
  read)  sandbox=(-s read-only) ;;
  write) sandbox=(-s workspace-write) ;;
  net)   sandbox=(-s workspace-write -c sandbox_workspace_write.network_access=true) ;;
  *) echo "bad mode: $mode" >&2; exit 2 ;;
esac
# extra codex flags, e.g. CODEX_EXTRA="--add-dir C:/Users/admin/.kaggle"
read -r -a extra <<< "${CODEX_EXTRA:-}"
sandbox+=("${extra[@]}")

if ((resume)); then
  sid="$(cat "$run_dir/session_id" 2>/dev/null)"
  [[ -n "$sid" ]] || { echo "no session_id for $task_id" >&2; exit 2; }
  cmd=("$CODEX" exec "${sandbox[@]}" -C "$workdir" --json -o "$run_dir/last.md" resume "$sid" -)
else
  cmd=("$CODEX" exec "${sandbox[@]}" -C "$workdir" --json -o "$run_dir/last.md" -)
fi

echo "[codex_task] $task_id mode=$mode workdir=$workdir start $(date -Is)" | tee -a "$run_dir/run.log"
"${cmd[@]}" < "$prompt_file" >> "$run_dir/events.jsonl" 2>> "$run_dir/stderr.log"
code=$?
echo "$code" > "$run_dir/exit_code"
# first thread id in the event stream = session id for resume
grep -o '"thread_id":"[^"]*"' "$run_dir/events.jsonl" | head -1 | cut -d'"' -f4 > "$run_dir/session_id.tmp" \
  && [[ -s "$run_dir/session_id.tmp" ]] && mv "$run_dir/session_id.tmp" "$run_dir/session_id" || rm -f "$run_dir/session_id.tmp"
echo "[codex_task] $task_id exit=$code end $(date -Is)" | tee -a "$run_dir/run.log"
echo "----- last message -----"
cat "$run_dir/last.md" 2>/dev/null
exit "$code"
