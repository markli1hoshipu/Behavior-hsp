#!/usr/bin/env bash
set -euo pipefail

# Batch runner for embodiedClaw using the one-episode supervisor.
#
# You can configure this script either with environment variables or CLI flags.
# CLI flags win over environment defaults.
#
# Example:
# bash run_supervised_all_instances.sh --task-name picking_up_trash --task-id 1 --log-root /home/user/dataset/embodiedClaw_batch_picking_up_trash --decision-mode image_state --pause-before false --start-idx 0 --end-idx 199 --source-repo-root /home/user/codebase/B1K-DataGen/BEHAVIOR-1K --runtime-repo-root /home/user/codebase/BEHAVIOR-1K
#
# Important environment variables:
# - TASK_NAME: BEHAVIOR task name, default `picking_up_trash`
# - TASK_ID: numeric task id, default `1`
# - START_IDX / END_IDX: CSV index range to process
#   - if END_IDX is unset, it is derived from task metadata
# - LOG_ROOT: root directory for per-instance `log_path`
# - DECISION_MODE: `image_state` or `imageless_state`
# - PAUSE_BEFORE: `true` or `false`
#
# Defaults:
# - processes exactly one CSV index per run of `run_supervised_episode.sh`
# - starts VLA once and reuses it across instances
# - restarts simulator + MCP every instance
# - auto-spawns the master with `claude -p`
#
# Optional:
# - SOURCE_REPO_ROOT: source checkout containing `.mcp.json`, `mcp_call.py`, and skill files
# - RUNTIME_REPO_ROOT: runtime checkout used to launch `run_agentic.py`
#   In this repo, the common setup is:
#   SOURCE_REPO_ROOT=/home/user/codebase/B1K-DataGen/BEHAVIOR-1K
#   RUNTIME_REPO_ROOT=/home/user/codebase/BEHAVIOR-1K
# - EXCLUDE_INSTANCE_IDS: space-separated real instance ids to skip
# - START_VLA_ONCE=1 or 0
# - KEEP_VLA_ON_EXIT=1 or 0
# - SLEEP_BETWEEN_RUNS: seconds between instances, default `10`
# - HEADLESS / DISPLAY_VALUE / MODEL_HOST / MODEL_PORT / MCP_HOST / MCP_PORT
# - VLA_CMD: custom command for launching the VLA server

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_REPO_ROOT="${SOURCE_REPO_ROOT:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
RUNTIME_REPO_ROOT="${RUNTIME_REPO_ROOT:-/home/user/codebase/BEHAVIOR-1K}"
EMBODIEDCLAW_ROOT="$SOURCE_REPO_ROOT/OmniGibson/omnigibson/learning/embodiedClaw"
SUPERVISOR_SCRIPT="$EMBODIEDCLAW_ROOT/run_supervised_episode.sh"

TASK_NAME="${TASK_NAME:-picking_up_trash}"
TASK_ID="${TASK_ID:-1}"
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-}"
LOG_ROOT="${LOG_ROOT:-/home/user/dataset/embodiedClaw_batch_${TASK_NAME}}"
DECISION_MODE="${DECISION_MODE:-image_state}"
PAUSE_BEFORE="${PAUSE_BEFORE:-false}"

START_VLA_ONCE="${START_VLA_ONCE:-1}"
KEEP_VLA_ON_EXIT="${KEEP_VLA_ON_EXIT:-1}"
SLEEP_BETWEEN_RUNS="${SLEEP_BETWEEN_RUNS:-10}"

HEADLESS="${HEADLESS:-false}"
DISPLAY_VALUE="${DISPLAY_VALUE:-:1}"
MODEL_HOST="${MODEL_HOST:-127.0.0.1}"
MODEL_PORT="${MODEL_PORT:-8000}"
MCP_HOST="${MCP_HOST:-127.0.0.1}"
MCP_PORT="${MCP_PORT:-8001}"

VLA_LOG="${VLA_LOG:-/tmp/embodiedclaw_batch_vla.log}"
EXCLUDE_INSTANCE_IDS="${EXCLUDE_INSTANCE_IDS:-}"

VLA_CMD_DEFAULT="cd /home/user/codebase/behavior-1k-solution && CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/serve_b1k.py --task-name=${TASK_NAME} --port=${MODEL_PORT} policy:checkpoint --policy.config pi_behavior_b1k_fast --policy.dir /home/user/codebase/behavior_submission/checkpoint_2"
VLA_CMD="${VLA_CMD:-$VLA_CMD_DEFAULT}"

VLA_PID=""

usage() {
  cat <<EOF
Usage:
  bash run_supervised_all_instances.sh [options]

Options:
  --task-name NAME
  --task-id ID
  --start-idx IDX
  --end-idx IDX
  --log-root PATH
  --decision-mode MODE
  --pause-before true|false
  --exclude-instance-ids "ID ID ID"
  --start-vla-once 0|1
  --keep-vla-on-exit 0|1
  --sleep-between-runs SEC
  --headless true|false
  --display VALUE
  --source-repo-root PATH
  --runtime-repo-root PATH
  --model-host HOST
  --model-port PORT
  --mcp-host HOST
  --mcp-port PORT
  --vla-log PATH
  --vla-cmd CMD
  -h, --help

Example:
  bash run_supervised_all_instances.sh --task-name picking_up_trash --task-id 1 --log-root /home/user/dataset/embodiedClaw_batch_picking_up_trash --decision-mode image_state --pause-before false --start-idx 0 --end-idx 199 --source-repo-root /home/user/codebase/B1K-DataGen/BEHAVIOR-1K --runtime-repo-root /home/user/codebase/BEHAVIOR-1K
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task-name) TASK_NAME="$2"; shift 2 ;;
    --task-id) TASK_ID="$2"; shift 2 ;;
    --start-idx) START_IDX="$2"; shift 2 ;;
    --end-idx) END_IDX="$2"; shift 2 ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    --decision-mode) DECISION_MODE="$2"; shift 2 ;;
    --pause-before) PAUSE_BEFORE="$2"; shift 2 ;;
    --exclude-instance-ids) EXCLUDE_INSTANCE_IDS="$2"; shift 2 ;;
    --start-vla-once) START_VLA_ONCE="$2"; shift 2 ;;
    --keep-vla-on-exit) KEEP_VLA_ON_EXIT="$2"; shift 2 ;;
    --sleep-between-runs) SLEEP_BETWEEN_RUNS="$2"; shift 2 ;;
    --headless) HEADLESS="$2"; shift 2 ;;
    --display) DISPLAY_VALUE="$2"; shift 2 ;;
    --source-repo-root) SOURCE_REPO_ROOT="$2"; shift 2 ;;
    --runtime-repo-root) RUNTIME_REPO_ROOT="$2"; shift 2 ;;
    --model-host) MODEL_HOST="$2"; shift 2 ;;
    --model-port) MODEL_PORT="$2"; shift 2 ;;
    --mcp-host) MCP_HOST="$2"; shift 2 ;;
    --mcp-port) MCP_PORT="$2"; shift 2 ;;
    --vla-log) VLA_LOG="$2"; shift 2 ;;
    --vla-cmd) VLA_CMD="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      printf 'Unknown option: %s\n\n' "$1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

log() {
  printf '[batch] %s\n' "$*"
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

start_vla_once() {
  if [[ "$START_VLA_ONCE" != "1" ]]; then
    log "Skipping VLA startup because START_VLA_ONCE=$START_VLA_ONCE"
    return
  fi

  if model_server_is_reachable; then
    log "Model server already reachable at ${MODEL_HOST}:${MODEL_PORT}; skipping shared VLA startup"
    return
  fi

  log "Starting shared VLA server"
  setsid bash -lc "export PYTHONUNBUFFERED=1; $VLA_CMD" >"$VLA_LOG" 2>&1 &
  VLA_PID=$!
  log "VLA PID=$VLA_PID log=$VLA_LOG"
}

cleanup() {
  if [[ -n "$VLA_PID" && "$KEEP_VLA_ON_EXIT" != "1" ]]; then
    kill_process_group "$VLA_PID"
  fi
}

get_exclude_indices() {
  python3 <<EOF
import csv
import os

exclude_ids = "${EXCLUDE_INSTANCE_IDS}".split()
if not exclude_ids:
    print("")
    raise SystemExit(0)

task_id = int("${TASK_ID}")
csv_path = os.path.join(
    os.environ.get("OG_DATA_PATH", "/home/user/BEHAVIOR-1K/datasets"),
    "2025-challenge-task-instances", "metadata", "test_instances.csv",
)
with open(csv_path, "r") as f:
    lines = list(csv.reader(f))[1:]

test_instances = [int(x.strip()) for x in lines[task_id][2].strip().split(",")]
instance_to_idx = {inst: idx for idx, inst in enumerate(test_instances)}
indices = [instance_to_idx[int(inst_id)] for inst_id in exclude_ids if int(inst_id) in instance_to_idx]
print(" ".join(str(x) for x in sorted(indices)))
EOF
}

get_default_end_idx() {
  python3 <<EOF
import csv
import os

task_id = int("${TASK_ID}")
csv_path = os.path.join(
    os.environ.get("OG_DATA_PATH", "/home/user/BEHAVIOR-1K/datasets"),
    "2025-challenge-task-instances", "metadata", "test_instances.csv",
)
with open(csv_path, "r") as f:
    lines = list(csv.reader(f))[1:]

test_instances = [x.strip() for x in lines[task_id][2].strip().split(",") if x.strip()]
print(len(test_instances) - 1)
EOF
}

should_exclude() {
  local idx=$1
  for excluded in $EXCLUDE_INDICES; do
    if [[ "$idx" -eq "$excluded" ]]; then
      return 0
    fi
  done
  return 1
}

resolve_og_data_path() {
  if [[ -n "${OG_DATA_PATH:-}" && -d "${OG_DATA_PATH}" ]]; then
    printf '%s\n' "${OG_DATA_PATH}"
    return 0
  fi

  if [[ -d "$RUNTIME_REPO_ROOT/datasets" ]]; then
    printf '%s\n' "$RUNTIME_REPO_ROOT/datasets"
    return 0
  fi

  if [[ -d "$SOURCE_REPO_ROOT/datasets" ]]; then
    printf '%s\n' "$SOURCE_REPO_ROOT/datasets"
    return 0
  fi

  printf '%s\n' "/home/user/BEHAVIOR-1K/datasets"
}

run_instance() {
  local idx=$1
  local instance_log_root="$LOG_ROOT/instance_${idx}"
  local sim_log="$LOG_ROOT/logs/sim_${idx}.log"
  local master_log="$LOG_ROOT/logs/master_${idx}.log"
  local decisions_log="$instance_log_root/decisions.jsonl"
  local issue_log="$instance_log_root/episode_issue.json"

  mkdir -p "$instance_log_root" "$LOG_ROOT/logs"
  rm -f "$decisions_log"
  rm -f "$issue_log"

  log "Running CSV index $idx"
  SOURCE_REPO_ROOT="$SOURCE_REPO_ROOT" \
  RUNTIME_REPO_ROOT="$RUNTIME_REPO_ROOT" \
  START_VLA=0 \
  KEEP_VLA_ON_EXIT=1 \
  SPAWN_MASTER=1 \
  TASK_NAME="$TASK_NAME" \
  TASK_ID="$TASK_ID" \
  INSTANCE_ID="$idx" \
  LOG_PATH="$instance_log_root" \
  DECISION_MODE="$DECISION_MODE" \
  PAUSE_BEFORE="$PAUSE_BEFORE" \
  HEADLESS="$HEADLESS" \
  DISPLAY_VALUE="$DISPLAY_VALUE" \
  MODEL_HOST="$MODEL_HOST" \
  MODEL_PORT="$MODEL_PORT" \
  MCP_HOST="$MCP_HOST" \
  MCP_PORT="$MCP_PORT" \
  SIM_LOG="$sim_log" \
  MASTER_LOG="$master_log" \
  DECISIONS_LOG="$decisions_log" \
  EPISODE_ISSUE_LOG="$issue_log" \
  bash "$SUPERVISOR_SCRIPT"
}

main() {
  mkdir -p "$LOG_ROOT" "$LOG_ROOT/logs"
  trap cleanup EXIT INT TERM

  export OG_DATA_PATH
  OG_DATA_PATH="$(resolve_og_data_path)"
  log "Using OG_DATA_PATH=$OG_DATA_PATH"

  if [[ -z "$END_IDX" ]]; then
    END_IDX="$(get_default_end_idx)"
  fi

  EXCLUDE_INDICES="$(get_exclude_indices)"
  start_vla_once

  log "Processing CSV indices ${START_IDX} to ${END_IDX}"
  if [[ -n "$EXCLUDE_INDICES" ]]; then
    log "Excluding CSV indices: $EXCLUDE_INDICES"
  fi

  local i
  for ((i=START_IDX; i<=END_IDX; i+=1)); do
    if should_exclude "$i"; then
      log "Skipping CSV index $i"
      continue
    fi

    if run_instance "$i"; then
      log "Finished CSV index $i"
    else
      local rc=$?
      log "CSV index $i failed with exit code $rc; continuing to next instance"
    fi
    sleep "$SLEEP_BETWEEN_RUNS"
  done
}

main "$@"
