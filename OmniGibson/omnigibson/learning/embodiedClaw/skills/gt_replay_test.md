---
name: gt_replay_test
description: GT Replay Decision Accuracy Test. Use when testing the embodiedClaw decision module by observing ground truth HDF5 replay through MCP. Measures subtask boundary detection accuracy against annotated ground truth.
---

# GT Replay Decision Accuracy Test (MCP Mode)

You are testing the embodiedClaw decision module by observing a ground truth HDF5 replay through MCP.

The simulator replays a recorded episode frame-by-frame. At each observation point it pauses and waits for you to observe and decide. Your job is to determine when each subtask completes. Your decisions will be compared against ground truth annotation boundaries.

See `references/gt_replay_protocol.md` for evaluation rules and output format.

## Setup

The replay runner must be running before starting:

```bash
cd /home/user/codebase/B1K-DataGen/BEHAVIOR-1K
DISPLAY=:1 OMNI_KIT_ACCEPT_EULA=yes conda run -n behavior python \
    OmniGibson/omnigibson/learning/embodiedClaw/gt_replay_runner.py \
    policy=local task.name=picking_up_trash headless=true \
    log_path=/tmp/gt_replay \
    +hdf5_path=/home/user/dataset/raw/task-0001/episode_00010010.hdf5 \
    +observation_interval=25
```

## Procedure

1. Call `load_task_annotations` with the task ID to get subtask list and failure thresholds.
2. At each observation point (sim pauses automatically):
   a. Call `get_simulation_snapshot` and check `world_predicates` (not legacy `bddl_state`), images, and robot state.
   b. Use `references/decision_rubric.md` to evaluate subtask completion. Optionally call `get_demo_reference_frames` for visual comparison.
   c. If subtask succeeded, call `advance_to_next_subtask`.
   d. Call `resume_vla` to advance to the next observation point.
3. Record decisions: agent frame, GT end frame, frame error per subtask.
4. After all subtasks: compile accuracy metrics.

## Decision Criteria

- **Success**: World predicates confirm subtask completion per decision rubric.
- **Continue**: Subtask in progress — resume and check at next observation point.
- **Failure**: Elapsed steps exceed failure threshold (2x max duration from annotations).

## Metrics

- `frame_error = agent_decision_frame - gt_end_frame`
- Accuracy thresholds: +/-50, +/-100, +/-200, +/-500 frames
- Target: >=80% at +/-100 frames
