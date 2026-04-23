#!/usr/bin/env bash
set -euo pipefail
TASK_NAME="${1:?task name required}"
EVAL_INSTANCE_IDS="${EVAL_INSTANCE_IDS:-[0,1,2,3,4,5,6,7,8,9]}"
EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:-}"
EVAL_HEADLESS="${EVAL_HEADLESS:-false}"
EVAL_WRITE_VIDEO="${EVAL_WRITE_VIDEO:-true}"
cd /home/aaron/BEHAVIOR-1K
python OmniGibson/omnigibson/learning/eval.py \
  log_path="${LOG_ROOT:?}/${LOG_SUBDIR:?}" \
  policy=websocket \
  task.name="${TASK_NAME}" \
  env_wrapper._target_=omnigibson.learning.wrappers.RGBWrapper \
  eval_instance_ids="${EVAL_INSTANCE_IDS}" \
  headless="${EVAL_HEADLESS}" \
  write_video="${EVAL_WRITE_VIDEO}" \
  ${EXTRA_EVAL_ARGS}
