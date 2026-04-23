# GT Replay Decision Accuracy Test (MCP Mode)

You are testing the embodiedClaw decision module by observing a ground truth HDF5 replay through MCP.

## Context

The simulator is replaying a recorded episode frame-by-frame. At each observation point it pauses and waits for you to observe and decide. Your job is to determine when each subtask completes. Your decisions will be compared against ground truth annotation boundaries.

## Setup

Before starting, the replay runner must be running:

```bash
cd /home/user/codebase/BEHAVIOR-1K
DISPLAY=:1 OMNI_KIT_ACCEPT_EULA=yes conda run -n behavior python \
    OmniGibson/omnigibson/learning/embodiedClaw/gt_replay_runner.py \
    policy=local task.name=picking_up_trash headless=true \
    log_path=/tmp/gt_replay \
    +hdf5_path=/home/user/dataset/raw/task-0001/episode_00010010.hdf5 \
    +observation_interval=25
```

## Procedure

1. **Load annotations**: Call `load_task_annotations` with the task ID to get the subtask list and failure thresholds.

2. **At each observation point** (the sim pauses automatically):
   a. Call `get_simulation_snapshot` to see the current state (images, robot, objects, BDDL).
   b. Evaluate the subtask yourself: check BDDL predicates, object positions, robot state, and elapsed steps versus failure threshold from `get_task_memory`. Optionally call `get_demo_reference_frames` for visual comparison.
   c. If you determine the subtask succeeded, call `advance_to_next_subtask`.
   d. Call `resume_vla` to advance to the next observation point.

3. **Record your decisions**: For each subtask, note:
   - The frame at which you declared success
   - The ground truth end frame (from annotations)
   - The frame error

4. **After all subtasks**: Compile results and compute accuracy metrics.

## Decision Criteria

- **Success**: BDDL predicates for the subtask are satisfied, OR visual comparison with demo frames confirms completion.
- **Continue**: Subtask is still in progress — resume and check again at the next observation point.
- **Failure**: Elapsed steps exceed the failure threshold (2x max duration from annotations).

## Metrics

- `frame_error = agent_decision_frame - gt_end_frame`
- Report accuracy at thresholds: +/-50, +/-100, +/-200, +/-500 frames
- Target: >=80% accuracy at +/-100 frames
