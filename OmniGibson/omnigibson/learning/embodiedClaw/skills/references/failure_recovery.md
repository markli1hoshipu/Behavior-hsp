# Failure Recovery

This file defines the allowed control flow for success, retry, restore, and episode end.

## Pause/Resume Invariant

- Pause before collecting a stable decision snapshot.
- Resume only when the episode should keep running.
- Do not unconditionally resume in a `finally` block after a terminal decision.

## Success Branch

1. Call `pause_vla()`.
2. Optionally call `get_simulation_snapshot(include_images=true)` in
   `image_state`, or `get_simulation_snapshot(include_images=false)` in
   `imageless_state`, once more for stable final confirmation.
3. Record the current subtask label from the pre-advance state.
4. If another subtask exists, derive `next_subtask_id` and `next_language_instruction` before mutating task memory. Recommended method: use `current_subtask_idx + 1` as the positional index into the `subtask_list` returned by `load_task_annotations()`, read that next entry's `skill_idx`, then call `get_subtask_language_instruction(skill_idx=next_skill_idx)`.
5. Call `save_success_data(...)` with current `subtask_id`, optional `next_subtask_id`, and optional `next_language_instruction`.
6. Call `advance_to_next_subtask()`.
7. If it returns `episode_complete`, call `save_episode_data(success=true)` and stop.
8. Do not call `switch_vla()` again after `save_success_data(...)` unless you are intentionally reconnecting to another host or port.
9. Resume the simulator.

## Failure Branch

1. Call `pause_vla()`.
2. Optionally call `get_simulation_snapshot(include_images=true)` in
   `image_state`, or `get_simulation_snapshot(include_images=false)` in
   `imageless_state`, once more for stable failure confirmation.
3. Call `save_failure_data(subtask_id=...)`.
4. Call `record_failure()` to update task memory and decision history.
5. If either result indicates the episode should end, call `save_episode_data(success=false)` and stop.
6. Otherwise call `step_zero_actions(num_steps=10)` while still paused to stabilize the restored simulator state and refresh cached observations / images.
7. If you need a post-restore confirmation snapshot, call `get_simulation_snapshot(...)` only after that zero-action stabilization step.
8. Continue with the same current subtask after restore.
9. Do not call `reset_vla()` after `save_failure_data(...)` unless recovery skipped the restore path and you explicitly need a manual reset.
10. Do not blindly call `switch_vla()` again for the same subtask after a normal restore.
11. Resume the simulator.

## Retry Timing

On a normal non-terminal retry, `save_failure_data(...)` performs the restore
and `record_failure()` then re-bases `task_memory.subtask_start_step` to the
checkpoint step. That means elapsed-step math should restart from the restored
attempt once failure bookkeeping completes.

Remaining caveats:

- Treat `record_failure()` as authoritative for retry count.
- If restore is skipped or errors, post-retry elapsed-step checks may still be advisory.
- Treat predicate deltas as freshly rebaselined only after the next restored
  `get_simulation_snapshot()` call.
- Treat `step_zero_actions(num_steps=10)` as the standard way to refresh the
  paused snapshot after restore before doing image-based reasoning.
- Prefer visual and state-based evidence over timeout-only failure if restore did not complete cleanly.

## Branch Summary

- `continue`: resume
- `success` with remaining subtasks: save success, advance, resume
- `success` on final subtask: save episode, stop
- `failure` with retries left: save failure, record failure, step zero actions, resume
- `failure` with retries exhausted: save failure, record failure, save episode, stop
