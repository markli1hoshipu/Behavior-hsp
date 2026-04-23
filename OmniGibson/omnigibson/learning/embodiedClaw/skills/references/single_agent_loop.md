# Single-Agent Loop

Use this after supervisor handoff when one spawned master agent owns a single
episode. The supervisor has already started VLA + simulator + MCP.

This is the default workflow for `imageless_state`, and it is also acceptable
for simple `image_state` episodes when image load is modest.

## Phase 0: Preconditions

- the supervisor has already assigned one episode job
- MCP is reachable and a real tool call succeeds
- `task_name`, `task_id`, `decision_mode`, `pause_before_decision`, and `log_path` are known

Do not call `get_task_memory()` before `load_task_annotations()`.

## Phase 1: Initialize

1. Derive the numeric `task_id`.
2. Call `load_task_annotations(task_id=...)`.
3. Call `get_task_memory()`.
4. Read:
   - `current_subtask_idx`
   - `current_subtask`
   - `failure_threshold` (`2 * max_duration` across demo episodes for this subtask)
   - `retry_count`
5. Call `get_subtask_language_instruction()` for the current subtask.
6. Call `switch_vla(...)` once for the initial subtask.
7. Call `resume_vla()` to start the simulation. The robot begins moving and the agent is already observing from step 0.

## Phase 2: Observe-Act Loop

Use the run prompt's `Pause before decision: <true|false>` flag to choose the
decision cadence. Default to `false`.

- If `Pause before decision: true`, pause before each decision-agent call and
  keep the sim paused while the agent observes and decides.
- If `Pause before decision: false`, do not pause for routine `continue`
  checks. The VLA should run continuously while you monitor with asynchronous
  snapshots.
- Always be paused before success/failure persistence, checkpoint restore, or
  subtask transition. Resume only on non-terminal branches.

### Wait Cadence

- rough step cadence: check every 200-400 sim steps
- navigation-heavy subtasks: check roughly every 15-25s
- manipulation-heavy subtasks: check roughly every 25-50s
- shorten the interval as the elapsed ratio approaches timeout
- after a likely-complete observation, recheck sooner

Use:

- `snapshot.step_count`
- `task_memory.subtask_start_step`
- `task_memory.failure_threshold` (`2 * max_duration` from annotations for the current subtask)

to derive urgency.

### Observe

If `Pause before decision: true`, call `pause_vla()` before this observation.

1. Call `get_simulation_snapshot()`.
2. Resolve object names using `tool_contracts.md`.
3. Call `filter_information()` with normalized object-type names by default.
   Only narrow to an exact runtime object name when that refinement is
   actually necessary.
4. Inspect:
   - `world_predicates`
   - `robot_state.grasped_objects`
   - relevant `object_states`
   - current images as direct evidence for scene state, spatial relations, and contradiction checks

If repeated image-bearing observations start putting pressure on your prompt
context, stop using this single-master pattern and move that episode to the
fresh decision-worker pattern described in `SKILL.md`.

### Evaluate

Use `decision_rubric.md`.

Decide one of:

- `continue`
- `success`
- `failure`
- `needs_visual_comparison`

If the evidence is inconclusive and the subtask is sufficiently late, call
`get_demo_reference_frames(frame_types=["completion"], max_episodes=1)`.

Do not treat predicates as the only decision surface. Predicates, grasp state, pose, and current images should all inform the decision.
For decision-making, use three evidence surfaces together when available:
current absolute predicates/state/images, predicate deltas since subtask start,
and predicate deltas since the previous decision snapshot.
For interchangeable same-class objects, prefer count or relationship deltas over
exact annotation instance identity, and do not mark success from absolute count
alone when prior objects may already satisfy the relation.
Treat timeout as annotation-grounded: a subtask is over threshold only when `elapsed_steps > 2 * max_duration` for that `skill_idx`.

## Phase 3: Success Branch

Use this order to avoid advancing task memory before persistence succeeds.

1. Identify the current subtask label.
2. If another subtask exists, derive:
   - `next_subtask_id`
   - `next_language_instruction`
   Recommended method: use `current_subtask_idx + 1` as the positional index into the `subtask_list` returned by `load_task_annotations()`, read that next entry's `skill_idx`, then call `get_subtask_language_instruction(skill_idx=next_skill_idx)`.
3. Call `pause_vla()`.
4. Optionally call `get_simulation_snapshot()` once more for final confirmation.
5. Call `save_success_data(...)` with the current `subtask_id` and the next-subtask info if available.
6. Call `advance_to_next_subtask()`.
7. If it returns `episode_complete`, call `save_episode_data(success=true)` and stop.
8. Otherwise resume the simulator.

Do not immediately call `switch_vla()` again after `save_success_data(...)` unless you are intentionally reconnecting to another host or port.

## Phase 4: Failure Branch

Use this order so restore and negative-save happen before task-memory bookkeeping.

1. Identify the current subtask label.
2. Call `pause_vla()`.
3. Call `save_failure_data(subtask_id=...)`.
4. Call `record_failure()`.
5. If either result says the episode should end, call `save_episode_data(success=false)` and stop.
6. Otherwise call `step_zero_actions(num_steps=10)` while still paused so the restored simulator state is stabilized and the next snapshot has fresh observations.
7. Resume the simulator on the same subtask.

Do not call `reset_vla()` after `save_failure_data(...)` unless recovery explicitly skipped the restore path. The save tool already resets policy state after restore when it can.

Do not blindly call `switch_vla()` again for the same subtask after a normal restore.

## Terminal Rules

- final success: save episode and stop
- retries exhausted: save episode as failure and stop
- non-terminal continue: resume
- non-terminal retry: resume

## Operational Bias

- Prefer false-continue over false-success.
- Prefer agreement across predicates, grasp state, pose, and current images over any single weak cue.
- Keep pauses short.
- Never leave the simulator paused indefinitely on a non-terminal branch.
- Keep your own context compact; do not retain raw snapshot history when MCP task memory and predicate deltas already capture what changed.
