#!/usr/bin/env bash
set -e  # exit immediately on error

export CUDA_VISIBLE_DEVICES=0

TASK="picking_up_trash"
LOG_PATH="/home/user/Jan12_b1k_log"
OUTPUT_FOLDER="/home/user/Jan12_b1k_log/direct_output"

# CSV indices 420-519 correspond to instance IDs 701-800 (100 new instances), processed in batches of 1
START_IDX=330
END_IDX=419
BATCH_SIZE=1

echo "========================================"
echo "Data generation for task: ${TASK}"
echo "Instances: ${START_IDX} to ${END_IDX}"
echo "Batch size: ${BATCH_SIZE}"
echo "========================================"

for ((i=START_IDX; i<=END_IDX; i+=BATCH_SIZE)); do
  # Calculate end of this batch (don't exceed END_IDX)
  BATCH_END=$((i + BATCH_SIZE - 1))
  if [ $BATCH_END -gt $END_IDX ]; then
    BATCH_END=$END_IDX
  fi

  # Generate comma-separated list of instance IDs for this batch
  INSTANCE_IDS=$(seq -s, $i $BATCH_END)

  echo "========================================"
  echo "Running batch: instances ${i} to ${BATCH_END}"
  echo "eval_instance_ids=[${INSTANCE_IDS}]"
  echo "========================================"

  (
    # Fresh process -> OmniGibson fully relaunches each batch
    python3 /home/user/BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval_data_gen_par.py \
      log_path=${LOG_PATH} \
      policy=websocket \
      task.name=${TASK} \
      model.host=localhost \
      headless=false \
      "eval_instance_ids=[${INSTANCE_IDS}]" \
      +output_folder=${OUTPUT_FOLDER} \
      +record_rgb=true \
      +record_depth=true \
      +only_successes=true
  )

  echo "Finished batch: instances ${i} to ${BATCH_END}"

  # Wait for videos to finish encoding/saving before next batch
  echo "Waiting 10 seconds for videos to finish saving..."
  sleep 10
  echo
done

echo "All batches completed successfully for task: ${TASK}"
