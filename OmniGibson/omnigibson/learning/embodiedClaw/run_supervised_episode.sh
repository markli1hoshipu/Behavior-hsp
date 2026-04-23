#!/usr/bin/env bash
set -euo pipefail

# One-episode supervisor for embodiedClaw.
#
# You can configure this script either with environment variables or CLI flags.
# CLI flags win over environment defaults.
#
# Example:
# bash run_supervised_episode.sh --task-name picking_up_trash --task-id 1 --instance-id 0 --log-path /home/user/dataset/embodiedClaw_test --decision-mode image_state --pause-before false --spawn-master 1 --source-repo-root /home/user/codebase/B1K-DataGen/BEHAVIOR-1K --runtime-repo-root /home/user/codebase/BEHAVIOR-1K
#
# Required/important environment variables:
# - TASK_NAME: BEHAVIOR task name, default `picking_up_trash`
# - TASK_ID: numeric task id, default `1`
# - INSTANCE_ID: CSV index inside `eval_instance_ids`, default `0`
# - LOG_PATH: output directory passed to `run_agentic.py`
# - DECISION_MODE: `image_state` or `imageless_state`, default `image_state`
# - PAUSE_BEFORE: `true` or `false`, default `false`
#
# Launch mode:
# - SPAWN_MASTER=0: start services, wait for MCP, print handoff for a fresh
#   interactive master session
# - SPAWN_MASTER=1: start services, wait for MCP, then spawn the master with
#   `claude -p`
#
# Service control:
# - START_VLA=1: start the VLA server here
# - START_VLA=0: assume the VLA server is already running
# - KEEP_VLA_ON_EXIT=1: leave VLA running after this script exits
# - KEEP_VLA_ON_EXIT=0: kill VLA on exit if this script started it
#
# Optional overrides:
# - SOURCE_REPO_ROOT: source checkout containing `.mcp.json`, `mcp_call.py`, and skill files
# - RUNTIME_REPO_ROOT: runtime checkout used to launch `run_agentic.py`
#   In this repo, the common setup is:
#   SOURCE_REPO_ROOT=/home/user/codebase/B1K-DataGen/BEHAVIOR-1K
#   RUNTIME_REPO_ROOT=/home/user/codebase/BEHAVIOR-1K
# - MODEL_HOST / MODEL_PORT / MCP_HOST / MCP_PORT
# - DISPLAY_VALUE / HEADLESS
# - SIM_LOG / VLA_LOG / MASTER_LOG / DECISIONS_LOG
# - MASTER_MAX_TURNS
# - MASTER_EXIT_GRACE_SECONDS: how long to wait for Claude to exit after the
#   simulator has already exited, default `45`
# - VLA_CMD: full custom command for starting the VLA server

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_REPO_ROOT="${SOURCE_REPO_ROOT:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
RUNTIME_REPO_ROOT="${RUNTIME_REPO_ROOT:-/home/user/BEHAVIOR-1K}"

TASK_NAME="${TASK_NAME:-picking_up_trash}"
TASK_ID="${TASK_ID:-1}"
INSTANCE_ID="${INSTANCE_ID:-0}"
LOG_PATH="${LOG_PATH:-}"
DECISION_MODE="${DECISION_MODE:-image_state}"
PAUSE_BEFORE="${PAUSE_BEFORE:-false}"

START_VLA="${START_VLA:-1}"
SPAWN_MASTER="${SPAWN_MASTER:-0}"
KEEP_VLA_ON_EXIT="${KEEP_VLA_ON_EXIT:-1}"
HEADLESS="${HEADLESS:-false}"
DISPLAY_VALUE="${DISPLAY_VALUE:-:1}"

MODEL_HOST="${MODEL_HOST:-127.0.0.1}"
MODEL_PORT="${MODEL_PORT:-8000}"
MCP_HOST="${MCP_HOST:-127.0.0.1}"
MCP_PORT="${MCP_PORT:-8001}"

SIM_LOG="${SIM_LOG:-/tmp/embodiedclaw_sim.log}"
VLA_LOG="${VLA_LOG:-/tmp/embodiedclaw_vla.log}"
MASTER_LOG="${MASTER_LOG:-/tmp/embodiedclaw_master.log}"
DECISIONS_LOG="${DECISIONS_LOG:-}"
MASTER_MAX_TURNS="${MASTER_MAX_TURNS:-400}"
MASTER_EXIT_GRACE_SECONDS="${MASTER_EXIT_GRACE_SECONDS:-45}"
EPISODE_ISSUE_LOG="${EPISODE_ISSUE_LOG:-}"

VLA_CMD="${VLA_CMD:-}"

SIM_PID=""
VLA_PID=""
MASTER_PID=""
MASTER_PROMPT_FILE=""
MASTER_SYSTEM_PROMPT_FILE=""

usage() {
  cat <<EOF
Usage:
  bash run_supervised_episode.sh [options]

Options:
  --task-name NAME
  --task-id ID
  --instance-id IDX
  --log-path PATH
  --decision-mode MODE
  --pause-before true|false
  --spawn-master 0|1
  --start-vla 0|1
  --keep-vla-on-exit 0|1
  --headless true|false
  --display VALUE
  --source-repo-root PATH
  --runtime-repo-root PATH
  --model-host HOST
  --model-port PORT
  --mcp-host HOST
  --mcp-port PORT
  --sim-log PATH
  --vla-log PATH
  --master-log PATH
  --decisions-log PATH
  --master-max-turns N
  --vla-cmd CMD
  -h, --help

Example:
  bash run_supervised_episode.sh --task-name picking_up_trash --task-id 1 --instance-id 0 --log-path /home/user/dataset/embodiedClaw_test_gyj --decision-mode image_state --pause-before false --spawn-master 1 --source-repo-root /home/user/BEHAVIOR-1K --runtime-repo-root /home/user/BEHAVIOR-1K
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task-name) TASK_NAME="$2"; shift 2 ;;
    --task-id) TASK_ID="$2"; shift 2 ;;
    --instance-id) INSTANCE_ID="$2"; shift 2 ;;
    --log-path) LOG_PATH="$2"; shift 2 ;;
    --decision-mode) DECISION_MODE="$2"; shift 2 ;;
    --pause-before) PAUSE_BEFORE="$2"; shift 2 ;;
    --spawn-master) SPAWN_MASTER="$2"; shift 2 ;;
    --start-vla) START_VLA="$2"; shift 2 ;;
    --keep-vla-on-exit) KEEP_VLA_ON_EXIT="$2"; shift 2 ;;
    --headless) HEADLESS="$2"; shift 2 ;;
    --display) DISPLAY_VALUE="$2"; shift 2 ;;
    --source-repo-root) SOURCE_REPO_ROOT="$2"; shift 2 ;;
    --runtime-repo-root) RUNTIME_REPO_ROOT="$2"; shift 2 ;;
    --model-host) MODEL_HOST="$2"; shift 2 ;;
    --model-port) MODEL_PORT="$2"; shift 2 ;;
    --mcp-host) MCP_HOST="$2"; shift 2 ;;
    --mcp-port) MCP_PORT="$2"; shift 2 ;;
    --sim-log) SIM_LOG="$2"; shift 2 ;;
    --vla-log) VLA_LOG="$2"; shift 2 ;;
    --master-log) MASTER_LOG="$2"; shift 2 ;;
    --decisions-log) DECISIONS_LOG="$2"; shift 2 ;;
    --master-max-turns) MASTER_MAX_TURNS="$2"; shift 2 ;;
    --vla-cmd) VLA_CMD="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      printf 'Unknown option: %s\n\n' "$1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

# --- Derived defaults (must be after CLI parsing) ---
SKILL_ROOT="$SOURCE_REPO_ROOT/OmniGibson/omnigibson/learning/embodiedClaw/skills"
LOG_PATH="${LOG_PATH:-/tmp/embodiedclaw_${TASK_NAME}_${INSTANCE_ID}}"
DECISIONS_LOG="${DECISIONS_LOG:-$LOG_PATH/decisions.jsonl}"
EPISODE_ISSUE_LOG="${EPISODE_ISSUE_LOG:-$LOG_PATH/episode_issue.json}"
if [[ -z "$VLA_CMD" ]]; then
  VLA_CMD="cd /home/user/codebase/behavior-1k-solution && CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/serve_b1k.py --task-name=${TASK_NAME} --port=${MODEL_PORT} policy:checkpoint --policy.config pi_behavior_b1k_fast --policy.dir /home/user/.cache/huggingface/hub/models--IliaLarchenko--behavior_submission/snapshots/e2012d0a102e0d21fdcfb72009a87428917fb15c/checkpoint_2"
fi

log() {
  printf '[supervisor] %s\n' "$*"
}

tail_log_summary() {
  local log_path="$1"

  if [[ ! -f "$log_path" ]]; then
    printf 'log_missing:%s' "$log_path"
    return 0
  fi

  python3 - "$log_path" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
lines = [line.strip() for line in path.read_text(errors="replace").splitlines() if line.strip()]
snippet = " | ".join(lines[-5:]) if lines else f"log_empty:{path}"
print(snippet[:4000])
PY
}

record_episode_issue() {
  local category="$1"
  local reason="$2"
  local master_exit_code="${3:-}"
  local sim_exit_code="${4:-}"

  mkdir -p "$LOG_PATH"
  mkdir -p "$(dirname "$DECISIONS_LOG")"

  python3 - "$DECISIONS_LOG" "$EPISODE_ISSUE_LOG" "$category" "$reason" "$TASK_NAME" "$TASK_ID" "$INSTANCE_ID" "$master_exit_code" "$sim_exit_code" <<'PY'
import datetime as dt
import json
import os
import sys

decisions_log, issue_log, category, reason, task_name, task_id, instance_id, master_exit_code, sim_exit_code = sys.argv[1:]

payload = {
    "decision": "episode_error",
    "category": category,
    "reason": reason,
    "task_name": task_name,
    "task_id": int(task_id),
    "instance_id": int(instance_id),
    "master_exit_code": int(master_exit_code) if master_exit_code else None,
    "sim_exit_code": int(sim_exit_code) if sim_exit_code else None,
    "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
}

line = json.dumps(payload, ensure_ascii=True)

if decisions_log:
    os.makedirs(os.path.dirname(decisions_log), exist_ok=True)
    with open(decisions_log, "a", encoding="utf-8") as f:
        f.write(line + "\n")

if issue_log:
    os.makedirs(os.path.dirname(issue_log), exist_ok=True)
    with open(issue_log, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
        f.write("\n")
PY
}

record_master_failure() {
  local master_exit_code="$1"
  local category="master_process_error"
  local reason

  if [[ -f "$MASTER_LOG" ]] && grep -Eq 'API Error:|ValidationException:|Invalid `signature` in `thinking` block|InvokeModelWithResponseStream' "$MASTER_LOG"; then
    category="api_error"
  fi

  reason="$(tail_log_summary "$MASTER_LOG")"
  log "Recording $category for instance $INSTANCE_ID"
  record_episode_issue "$category" "$reason" "$master_exit_code" ""
}

record_sim_failure() {
  local sim_exit_code="$1"
  local reason

  reason="$(tail_log_summary "$SIM_LOG")"
  log "Recording sim_process_error for instance $INSTANCE_ID"
  record_episode_issue "sim_process_error" "$reason" "" "$sim_exit_code"
}

kill_process_group() {
  local pid="${1:-}"
  local i

  if [[ -z "$pid" ]]; then
    return 0
  fi

  kill -- "-$pid" >/dev/null 2>&1 || kill "$pid" >/dev/null 2>&1 || true
  for ((i=0; i<10; i+=1)); do
    if ! kill -0 "$pid" >/dev/null 2>&1 && ! pgrep -g "$pid" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done

  kill -9 -- "-$pid" >/dev/null 2>&1 || kill -9 "$pid" >/dev/null 2>&1 || true
}

model_server_is_reachable() {
  python3 - "$MODEL_HOST" "$MODEL_PORT" <<'EOF'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(1.0)
    sys.exit(0 if sock.connect_ex((host, port)) == 0 else 1)
EOF
}

start_vla() {
  if [[ "$START_VLA" != "1" ]]; then
    log "Skipping VLA startup because START_VLA=$START_VLA"
    return
  fi

  if model_server_is_reachable; then
    log "Model server already reachable at ${MODEL_HOST}:${MODEL_PORT}; skipping VLA startup"
    return
  fi

  log "Starting VLA server"
  setsid bash -lc "export PYTHONUNBUFFERED=1; $VLA_CMD" >"$VLA_LOG" 2>&1 &
  VLA_PID=$!
  log "VLA PID=$VLA_PID log=$VLA_LOG"
}

start_sim() {
  mkdir -p "$LOG_PATH"
  mkdir -p "$(dirname "$DECISIONS_LOG")"
  rm -f "$DECISIONS_LOG"
  log "Starting simulator + MCP for one episode job"
  setsid bash -lc "cd '$RUNTIME_REPO_ROOT' && export DISPLAY='$DISPLAY_VALUE' OMNI_KIT_ACCEPT_EULA=yes PYTHONUNBUFFERED=1 OMNIGIBSON_GPU_ID=0 && exec conda run --no-capture-output -n behavior python -u OmniGibson/omnigibson/learning/embodiedClaw/run_agentic.py policy=websocket task.name='$TASK_NAME' model.host='$MODEL_HOST' model.port='$MODEL_PORT' headless='$HEADLESS' log_path='$LOG_PATH' +mcp_host='$MCP_HOST' +mcp_port='$MCP_PORT' \"eval_instance_ids=[$INSTANCE_ID]\"" >"$SIM_LOG" 2>&1 &
  SIM_PID=$!
  log "SIM PID=$SIM_PID log=$SIM_LOG"
}

wait_for_mcp() {
  local attempts=60
  local sleep_s=5
  local i
  for ((i=1; i<=attempts; i+=1)); do
    if python3 "$SOURCE_REPO_ROOT/mcp_call.py" get_simulation_snapshot '{"include_images": false}' 20 --no-images >/dev/null 2>&1; then
      log "MCP readiness probe succeeded"
      return 0
    fi
    sleep "$sleep_s"
  done
  return 1
}

print_handoff() {
  cat <<EOF

EmbodiedClaw supervisor is ready.

Use a fresh agent session as the MCP-capable master with this handoff:
- task_name=$TASK_NAME
- task_id=$TASK_ID
- instance_id=$INSTANCE_ID
- decision_mode=$DECISION_MODE
- pause_before_decision=$PAUSE_BEFORE
- log_path=$LOG_PATH
- decisions_log=$DECISIONS_LOG
- mcp_url=http://$MCP_HOST:$MCP_PORT/sse

Interactive master prompt:
Use the embodiedclaw skill as the master agent for one episode.
Handoff:
- task_name=$TASK_NAME
- task_id=$TASK_ID
- instance_id=$INSTANCE_ID
- decision_mode=$DECISION_MODE
- pause_before_decision=$PAUSE_BEFORE
- log_path=$LOG_PATH
- decisions_log=$DECISIONS_LOG

Logs:
- VLA log: $VLA_LOG
- SIM log: $SIM_LOG
- Master log: $MASTER_LOG

Shutdown:
- kill $SIM_PID
EOF

  if [[ -n "$VLA_PID" && "$KEEP_VLA_ON_EXIT" != "1" ]]; then
    printf -- '- kill %s\n' "$VLA_PID"
  elif [[ -n "$VLA_PID" ]]; then
    printf -- '- VLA left running on purpose: PID=%s\n' "$VLA_PID"
  fi
}

spawn_master() {
  MASTER_PROMPT_FILE="$(mktemp /tmp/embodiedclaw_master_prompt.XXXXXX)"
  MASTER_SYSTEM_PROMPT_FILE="$(mktemp /tmp/embodiedclaw_master_system_prompt.XXXXXX)"

  cat >"$MASTER_PROMPT_FILE" <<EOF
Use the embodiedclaw skill as the master agent for one episode.
Handoff:
- task_name=$TASK_NAME
- task_id=$TASK_ID
- instance_id=$INSTANCE_ID
- decision_mode=$DECISION_MODE
- pause_before_decision=$PAUSE_BEFORE
- log_path=$LOG_PATH
- decisions_log=$DECISIONS_LOG
EOF

  cat "$SKILL_ROOT/SKILL.md" >"$MASTER_SYSTEM_PROMPT_FILE"

  log "Spawning master via claude -p"
  : >"$MASTER_LOG"
  setsid bash -lc "
    cd '$SOURCE_REPO_ROOT'
    export DECISIONS_LOG='$DECISIONS_LOG'
    export TASK_NAME='$TASK_NAME'
    export TASK_ID='$TASK_ID'
    export DECISION_MODE='$DECISION_MODE'
    export PAUSE_BEFORE='$PAUSE_BEFORE'
    exec claude -p \"\$(cat '$MASTER_PROMPT_FILE')\" \
      --append-system-prompt \"\$(cat '$MASTER_SYSTEM_PROMPT_FILE')\" \
      --mcp-config '$SOURCE_REPO_ROOT/.mcp.json' \
      --permission-mode bypassPermissions \
      --max-turns '$MASTER_MAX_TURNS'
  " >"$MASTER_LOG" 2>&1 &
  MASTER_PID=$!
  log "MASTER PID=$MASTER_PID log=$MASTER_LOG"
}

cleanup() {
  if [[ -n "$MASTER_PID" ]]; then
    kill_process_group "$MASTER_PID"
  fi
  if [[ -n "$SIM_PID" ]]; then
    kill_process_group "$SIM_PID"
  fi
  if [[ -n "$VLA_PID" && "$KEEP_VLA_ON_EXIT" != "1" ]]; then
    kill_process_group "$VLA_PID"
  fi
  rm -f "$MASTER_PROMPT_FILE" "$MASTER_SYSTEM_PROMPT_FILE"
}

wait_for_master_or_sim() {
  local sim_status=0
  local master_status=0
  local grace_remaining

  while true; do
    if [[ -n "$MASTER_PID" ]] && ! kill -0 "$MASTER_PID" >/dev/null 2>&1; then
      wait "$MASTER_PID" || master_status=$?
      MASTER_PID=""
      if [[ "$master_status" -ne 0 ]]; then
        record_master_failure "$master_status"
      fi
      return "$master_status"
    fi

    if [[ -n "$SIM_PID" ]] && ! kill -0 "$SIM_PID" >/dev/null 2>&1; then
      wait "$SIM_PID" || sim_status=$?
      SIM_PID=""
      if [[ "$sim_status" -ne 0 ]]; then
        record_sim_failure "$sim_status"
      fi

      if [[ -n "$MASTER_PID" ]] && kill -0 "$MASTER_PID" >/dev/null 2>&1; then
        log "Simulator exited with code $sim_status; waiting up to ${MASTER_EXIT_GRACE_SECONDS}s for the master to finish cleanly"
        for ((grace_remaining=MASTER_EXIT_GRACE_SECONDS; grace_remaining>0; grace_remaining-=1)); do
          if ! kill -0 "$MASTER_PID" >/dev/null 2>&1; then
            wait "$MASTER_PID" || master_status=$?
            MASTER_PID=""
            if [[ "$sim_status" -ne 0 && "$master_status" -ne 0 ]]; then
              record_master_failure "$master_status"
            fi
            return "$sim_status"
          fi
          sleep 1
        done

        log "Master still running after simulator shutdown; terminating PID=$MASTER_PID"
        kill_process_group "$MASTER_PID"
        wait "$MASTER_PID" || master_status=$?
        MASTER_PID=""
      fi

      return "$sim_status"
    fi

    sleep 2
  done
}

main() {
  if [[ "$SPAWN_MASTER" == "1" ]]; then
    trap cleanup EXIT INT TERM
  fi

  start_vla
  start_sim

  if ! wait_for_mcp; then
    log "MCP readiness probe failed. Check $SIM_LOG"
    exit 1
  fi

  if [[ "$SPAWN_MASTER" == "1" ]]; then
    spawn_master
    wait_for_master_or_sim
  else
    print_handoff
  fi
}

main "$@"
