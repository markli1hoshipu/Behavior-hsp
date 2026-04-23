#!/usr/bin/env bash
set -e  # exit immediately on error

export CUDA_VISIBLE_DEVICES=0

TASK="picking_up_trash"
LOG_PATH="/home/user/Jan12_b1k_log"
OUTPUT_FOLDER="/home/user/Jan12_b1k_log/direct_output"

# CSV indices 20-519 correspond to instance IDs 301-800, processed in batches of 1
START_IDX=170
END_IDX=519
BATCH_SIZE=1

# Already completed instance IDs (from existing parquet files in task-0001)
EXCLUDE_INSTANCE_IDS="301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 319 321 323 324 326 327 328 329 331 332 334 335 336 339 341 342 343 345 346 347 349 350 358 363 365 370 375 379 381 383 385 395 396 397 399 404 405 406 408 409 410 413 419 420 422 423 427 428 430 433 434 435 446 447 451 456 461 469 470 472 482 486 487 491 495 496 497 501 503 504 505 512 516 517 519 521 528 537 539 541 547 551 558 578 583 589 595 596 598 604 605 608 612 615 618 621 622 623 625 629 630 632 635 640 643 648 663 682 691 694 695 703 704 705 710 713 714 717 733 743 744 754 755 769 777 794"

# Convert instance IDs to CSV indices using Python
get_exclude_indices() {
  python3 << EOF
import csv
import os

# Load test instances for task 1 (picking_up_trash)
csv_path = os.path.join(
    os.environ.get('OG_DATA_PATH', '/home/user/BEHAVIOR-1K/datasets'),
    '2025-challenge-task-instances', 'metadata', 'test_instances.csv'
)
with open(csv_path, 'r') as f:
    lines = list(csv.reader(f))[1:]
test_instances = [int(x.strip()) for x in lines[1][2].strip().split(",")]

# Build instance_id -> csv_index mapping
instance_to_idx = {inst: idx for idx, inst in enumerate(test_instances)}

# Convert excluded instance IDs to CSV indices
exclude_ids = [int(x) for x in "${EXCLUDE_INSTANCE_IDS}".split()]
exclude_indices = [instance_to_idx[inst_id] for inst_id in exclude_ids if inst_id in instance_to_idx]
print(" ".join(str(x) for x in sorted(exclude_indices)))
EOF
}

# Compute exclusion indices once at startup
EXCLUDE_INDICES=$(get_exclude_indices)
EXCLUDE_COUNT=$(echo $EXCLUDE_INDICES | wc -w)
echo "========================================"
echo "Excluding $EXCLUDE_COUNT completed instances"
echo "Instance IDs: ${EXCLUDE_INSTANCE_IDS}"
echo "CSV indices:  ${EXCLUDE_INDICES}"
echo "========================================"

# Function to check if a CSV index should be excluded
should_exclude() {
  local idx=$1
  for excluded in $EXCLUDE_INDICES; do
    if [ "$idx" -eq "$excluded" ]; then
      return 0  # true, should exclude
    fi
  done
  return 1  # false, should not exclude
}

echo "========================================"
echo "Data generation (save all) for task: ${TASK}"
echo "Instances: ${START_IDX} to ${END_IDX}"
echo "Batch size: ${BATCH_SIZE}"
echo "========================================"

for ((i=START_IDX; i<=END_IDX; i+=BATCH_SIZE)); do
  # Skip if this CSV index is in the exclusion list (already completed)
  if should_exclude $i; then
    echo "Skipping CSV index $i (already completed)"
    continue
  fi

  # Calculate end of this batch (don't exceed END_IDX)
  BATCH_END=$((i + BATCH_SIZE - 1))
  if [ $BATCH_END -gt $END_IDX ]; then
    BATCH_END=$END_IDX
  fi

  # Generate comma-separated list of CSV indices for this batch
  INSTANCE_IDS=$(seq -s, $i $BATCH_END)

  echo "========================================"
  echo "Running batch: instances ${i} to ${BATCH_END}"
  echo "eval_instance_ids=[${INSTANCE_IDS}]"
  echo "========================================"

  (
    # Fresh process -> OmniGibson fully relaunches each batch
    python3 /home/user/BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval_data_gen_par_save_all.py \
      log_path=${LOG_PATH} \
      policy=websocket \
      task.name=${TASK} \
      model.host=localhost \
      headless=false \
      "eval_instance_ids=[${INSTANCE_IDS}]" \
      +output_folder=${OUTPUT_FOLDER} \
      +record_rgb=true \
      +record_depth=true \
      +only_successes=false
  )

  echo "Finished batch: instances ${i} to ${BATCH_END}"

  # Wait for videos to finish encoding/saving before next batch
  echo "Waiting 10 seconds for videos to finish saving..."
  sleep 10
  echo
done

echo "All batches completed successfully for task: ${TASK}"
