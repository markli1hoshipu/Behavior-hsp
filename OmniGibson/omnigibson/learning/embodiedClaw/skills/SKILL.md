---
name: embodiedclaw
description: Use when acting as the spawned MCP-capable master agent for one embodiedClaw collection episode. Covers in-episode initialization, observation, decisions, success/failure handling, optional image-state decision workers, and clean exit.
---

# EmbodiedClaw Master Skill

Use this skill only after the supervisor has already started the VLA server and
the simulator + MCP server, and has confirmed MCP readiness.

You are not the bootstrap session. You are the spawned MCP-capable master agent
for one episode.

## Role And Lifetime

- You own exactly one episode by default.
- You own episode-local MCP control, task progress, persistence, and final decisions.
- The supervisor owns process startup, restart policy, PID tracking, and emergency kill/restart.
- Optional subagents are helpers only. They do not own the episode.

## Preconditions

Before doing anything, require all of the following:

- MCP is live and usable
- `task_name` is known
- `task_id` is known
- `decision_mode` is known
- `pause_before_decision` is known
- `log_path` and `decisions_log` are known
- the supervisor has already assigned one episode job

If those preconditions are not met, stop and hand control back to the supervisor.

See `references/supervisor_handoff.md` for the outer-layer contract.

## Handoff Contract From Supervisor

Expect these fields on entry:

- `task_name`
- `task_id`
- `decision_mode`
- `pause_before_decision`
- `log_path`
- `decisions_log`
- optional `instance_id`
- optional `episode_output_dir`
- optional restart-budget metadata

## Read Order

- Read `references/runtime_alignment.md` first.
- Read `references/tool_contracts.md` before calling MCP tools.
- Read `references/single_agent_loop.md` for the default master-owned episode loop.
- Read `references/decision_rubric.md` before making any success/failure judgment.
- Read `references/failure_recovery.md` before handling success, retry, restore, or terminal failure.
- Read `references/multi_agent_protocol.md` only when using a dedicated perception helper.
- Read `references/perception_role.md` only when spawning a dedicated perception helper.
- Read `references/supervisor_handoff.md` only if the supervisor contract is unclear.
- Read `references/gt_replay_protocol.md` only for offline replay.

## Decision Modes

Choose exactly one mode for the episode:

- `image_state`:
  - current images are allowed
  - `get_demo_reference_frames()` is allowed when the rubric warrants it
  - a fresh disposable decision worker is recommended when image load or context pressure is high
- `imageless_state`:
  - call `get_simulation_snapshot(include_images=false)`
  - call `filter_information(..., include_images=false)`
  - do not inspect or cite images
  - do not call `get_demo_reference_frames()`
  - a separate decision worker is usually unnecessary

## Required MCP Tools

Control / progress tools:

- `load_task_annotations`
- `get_task_memory`
- `get_subtask_language_instruction`
- `switch_vla`
- `pause_vla`
- `resume_vla`
- `save_success_data`
- `save_failure_data`
- `save_episode_data`
- `advance_to_next_subtask`
- `record_failure`

Observation tools:

- `get_simulation_snapshot`
- `filter_information`
- `get_demo_reference_frames` only when the rubric says it is justified and the mode allows it

## Initialization Sequence

The sim starts paused by default. Do not move the robot until initialization is complete.

1. Confirm `task_id` is numeric and correct.
2. Call `load_task_annotations(task_id=...)`.
3. Call `get_task_memory()`.
4. Record:
   - `current_subtask_idx`
   - `current_subtask`
   - `failure_threshold`
   - `retry_count`
5. Call `get_subtask_language_instruction()` for the current subtask.
6. Call `switch_vla(...)` once for the initial subtask.
7. Call `resume_vla()` as the last initialization step.

## Context Policy

- Do not keep raw snapshot or image history in your prompt context.
- Treat MCP task memory and predicate deltas as the real long-horizon memory source.
- Keep only compact summaries in your own working context:
  - current subtask info
  - last few compact decisions
  - last recovery action
  - current episode metadata
- In `image_state`, prefer a fresh decision worker when repeated image-heavy observations would otherwise bloat the master context.

## Default Episode Loop

1. Derive urgency from:
   - `snapshot.step_count`
   - `task_memory.subtask_start_step`
   - `task_memory.failure_threshold`
   - subtask type
2. Wait coarsely at first, then tighten near timeout:
   - roughly 15-25s for navigation
   - roughly 25-50s for manipulation
   - shorter intervals as elapsed ratio approaches threshold
3. If `pause_before_decision=true`, call `pause_vla()` before the decision step.
4. Observe using:
   - `get_task_memory()`
   - `get_simulation_snapshot()`
   - `filter_information()`
5. Decide using `decision_rubric.md`.
6. Act on the result:
   - `continue`: resume if paused, then wait again
   - `success`: pause if needed, follow success recovery, save, advance, possibly end the episode
   - `failure`: pause if needed, follow failure recovery, save negative data, record failure, possibly end the episode
   - `needs_visual_comparison`: only in `image_state`; re-check with demo frames or a fresh image-state decision worker
7. Resume the simulator only on non-terminal branches.

## Optional Subagents

### Fresh Decision Worker

Use this primarily in `image_state` when:

- the episode is long
- image observations are frequent
- context pressure is growing
- you want image reasoning isolated from control-plane actions

Decision worker rules:

- Spawn only after MCP is already live.
- Give it MCP access explicitly with `--mcp-config`.
- It may use:
  - `get_task_memory`
  - `get_simulation_snapshot`
  - `filter_information`
  - `get_demo_reference_frames` only if mode and rubric allow it
- It must not call:
  - `pause_vla`
  - `resume_vla`
  - `switch_vla`
  - `save_*`
  - `advance_to_next_subtask`
  - `record_failure`
- It outputs one JSON decision and exits.
- You parse that JSON and remain the only owner of all control and persistence actions.

### Perception Worker

Use this only in `image_state` when you want a dedicated perception helper.
Follow `references/multi_agent_protocol.md` and `references/perception_role.md`.

## Success / Failure / Retry

Follow `references/failure_recovery.md`.

High-level rules:

- On success:
  - pause if needed
  - save success data
  - advance task memory
  - save the full episode on terminal success
- On failure:
  - pause if needed
  - save failure data
  - record failure
  - if retrying, call `step_zero_actions(num_steps=10)` while still paused before any post-restore snapshot or resume
  - save the full episode on terminal failure
- Do not call `switch_vla()` again after `save_success_data(...)` unless you are intentionally reconnecting.
- Do not call `reset_vla()` after `save_failure_data(...)` unless the normal restore path was skipped.

## Logging And Exit Contract

- Append one compact JSON line per decision to `decisions_log`.
- On success or failure actions, also log summarized MCP action results.
- Keep the final log compact and structured.
- Exit cleanly after:
  - `save_episode_data(success=true)` on terminal success
  - `save_episode_data(success=false)` on terminal failure
- Return control to the supervisor after the episode ends.

## Non-Negotiables

- Do not assume this session bootstrapped the simulator.
- Do not assume this session can hot-attach MCP after startup. You were spawned after MCP became live.
- Do not guess `task_id`.
- Do not call `get_task_memory()` before `load_task_annotations()`.
- Do not assume fields that `get_task_memory()` does not actually return.
- Do not pass nested `current_subtask.object_ids` directly into `filter_information()`.
- Use normalized rollout-facing object names for filtering and decision logic.
- Treat `annotation_*` fields as debug-only.
- Do not treat `switch_vla()` as proof that remote inference changed.
- Do not let the supervisor and master both issue routine control-plane MCP actions.
- Do not use generic predicate-count heuristics in place of grounded evidence.
