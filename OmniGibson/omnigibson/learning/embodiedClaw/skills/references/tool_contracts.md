# Tool Contracts

Use this file before calling MCP tools or deriving inputs for them.

## Required Initialization

- `load_task_annotations(task_id: int)` must run once before `get_task_memory()`.
- The `task_id` is numeric, for example `1`, not the string task name.
- If launch context only gives `task.name`, derive the id from
  `omnigibson/learning/utils/eval_utils.py:TASK_NAMES_TO_INDICES`.
- Do not guess `task_id` heuristically from the task name string.

## Snapshot Tools

- `get_simulation_snapshot(include_images: bool = true)` returns:
  `images`, `robot_state`, `object_states`, `world_predicates`,
  `predicate_deltas_since_subtask_start`,
  `predicate_deltas_since_previous_decision`, `predicate_delta_metadata`,
  `name_mapping`, `step_count`, `task_name`.
- When `include_images=false`, `images` is `{}` and the caller must treat the
  snapshot as a structured-state-only observation.
- `object_states` contains task-scope objects, not just the current subtask objects.
- `world_predicates` contains grounded boolean facts only. It does not include raw pose-like state values.
- `filter_information(object_names: List[str], include_images: bool = true)` operates on the most recent snapshot cached by `get_simulation_snapshot()`.
- `filter_information()` accepts either exact runtime names like `can_of_soda_114`
  or normalized type names like `can_of_soda`.
- `filter_information()` keeps `world_predicates` when any predicate argument matches one of the requested objects or the robot.
- `filter_information()` keeps predicate-delta entries using the same relevance
  rule as `world_predicates`. Deltas are additive evidence; they do not replace
  current `world_predicates`.
- `filter_information(..., include_images=false)` returns `images: {}` even when
  the cached snapshot had image payloads.

## Task Memory Fields

`get_task_memory()` currently returns:

- `task_name`
- `current_subtask_idx`
- `total_subtasks`
- `subtask_start_step`
- `current_subtask`
- `failure_threshold` (`2 * max_duration` across demo episodes for the current subtask)
- `retry_count`
- `episode_done`
- `episode_success`
- `num_decisions`
- `recent_decisions`
- `retry_counts`

Do not assume `elapsed_steps`, `progress_ratio`, or `status` exist in the payload.

## Derived Values

- `elapsed_steps = snapshot.step_count - task_memory.subtask_start_step`
- `over_threshold = elapsed_steps > task_memory.failure_threshold`
- `task_memory.failure_threshold` means `2 * max_duration` from the annotations for the current `skill_idx`
- `current_subtask = task_memory.current_subtask`
- `skill_idx = current_subtask.skill_idx`
- `skill_description = current_subtask.skill_description`
- `manipulating_objects = current_subtask.manipulating_object_ids`
- `object_groups = current_subtask.object_ids`
- `annotation_manipulating_objects = current_subtask.annotation_manipulating_object_ids`
- `annotation_object_groups = current_subtask.annotation_object_ids`

## Object Normalization Procedure

The annotation object structure is not the same as the runtime filtering surface.

1. Preserve `object_groups` as nested groups for decision logic.
2. Treat `object_groups` / `manipulating_objects` as rollout-facing normalized
   type names. Numeric annotation ids have already been stripped from these
   fields.
3. Flatten only when you need a coarse candidate list for filtering or messaging.
4. Use `filter_information()` with the normalized names by default.
5. If you later discover an exact runtime object name and need to narrow focus,
   you may pass that exact name to `filter_information()`.
6. Use `annotation_*` fields only for debugging or offline comparison against a
   demonstration; do not require them at rollout time.

## Stable Subtask Labels

`save_success_data()` and `save_failure_data()` take a free-form `subtask_id` string.
Use a stable label for the same subtask across retries within an episode.

Recommended format:

- `skill_{skill_idx}_{slug}`

Where `slug` is a short lowercase rendering of `skill_description`, for example
`skill_2_pick_up_from`.

## Control and Data Tools

- `pause_vla()` pauses the simulation loop and caches a stable snapshot.
- `resume_vla()` resumes the loop.
- `reset_vla()` clears local policy state. Use it after restore only if recovery did not already do so.
- `step_zero_actions(num_steps=10)` advances the paused simulator with zero actions, refreshes cached observations / images, and re-pauses. Use it after `save_failure_data(...)` on non-terminal retries before requesting a post-restore snapshot or resuming the subtask. It must be called while paused.
- `switch_vla(language_instruction, host=None, port=None)` updates the controller field and optionally reconnects the websocket client. Do not treat it as a verified remote instruction change.
- `save_success_data(...)` already switches VLA when `next_language_instruction` is provided. Do not double-call `switch_vla()` unless you are intentionally reconnecting.
- `record_failure()` is the MCP tool name. It updates task-memory retry state and, after a confirmed restored retry, re-bases `subtask_start_step` to the checkpoint step.
- `save_failure_data(...)` saves the negative segment, truncates buffers, restores the scene when possible, resets the policy after restore, and returns `restore_status` so callers can distinguish restored vs skipped/error paths.
