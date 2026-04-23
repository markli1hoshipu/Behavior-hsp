# Runtime Alignment

This skill package is intentionally stricter than the legacy markdown prompts because several older instructions do not match the current runtime.

## Current Gaps

- The simulator creates MCP during `run_agentic.py` startup, so the original
  parent session cannot be treated as a session that magically hot-attaches to
  MCP later. The active skill must be written from the spawned master-agent
  perspective, not the bootstrap perspective.
- `load_task_annotations(task_id)` requires a numeric task id. The MCP surface does not currently expose a separate helper that tells the agent what that id is. If launch context only gives `task.name`, derive the id from `omnigibson/learning/utils/eval_utils.py:TASK_NAMES_TO_INDICES`.
- `get_task_memory()` is unavailable until `load_task_annotations()` succeeds.
- `get_task_memory()` does not return `elapsed_steps`, `progress_ratio`, or a formal status field. Derive elapsed steps as `snapshot.step_count - subtask_start_step`.
- `current_subtask.object_ids` is nested. It now contains rollout-facing
  normalized type names, not ready-made flat arguments for `filter_information()`.
- Raw demonstration instance ids remain available under
  `current_subtask.annotation_object_ids` and
  `current_subtask.annotation_manipulating_object_ids` for debugging.
- The base config defaults to `policy=local`, but `LocalPolicy` returns zero actions unless another component injects a real policy. For live collection, prefer a websocket-backed VLA server and launch `run_agentic.py` with `policy=websocket`.
- `get_simulation_snapshot(include_images=true)` returns task-scope object poses, not a subtask-only object slice. Use filtering or explicit reasoning to narrow focus.
- `get_simulation_snapshot(include_images=false)` is the code-level state-only observation path for `imageless_state`; it returns `images: {}`.
- `filter_information()` accepts either exact runtime names or normalized
  object-type names. Pass `include_images=false` in `imageless_state`. Old
  prompt text that describes `bddl_state` as the main surface is obsolete.
- `world_predicates` is the current grounded boolean evidence surface. Old prompt text that treats `bddl_state` as the main completion signal is obsolete.
- `switch_vla()` stores `controller.language_instruction` and optionally reconnects the websocket client. It does not, by itself, guarantee that inference behavior changed on the remote policy.
- `save_success_data()` already switches VLA when `next_language_instruction` is provided. Agents should not blindly issue a second `switch_vla()` on the same branch.
- Multi-agent mode is a prompt convention over one shared MCP server. The perception role is not ACL-restricted in code.
- Claude Code Agent-tool subagents do not inherit MCP connections. Any role that needs MCP tools must be started as a separate `claude -p` process with `--mcp-config` after the MCP server is running.
- The most reliable current split is:
  - supervisor starts VLA + sim
  - fresh master is spawned after MCP is live
  - optional fresh decision worker is used only when image-heavy reasoning justifies it
- Retry bookkeeping is split between task memory and checkpoint manager. After a successful `save_failure_data(...)` restore, the subsequent `record_failure()` call re-bases `task_memory.subtask_start_step` to the checkpoint step. If restore is skipped or fails, elapsed-step math may remain stale.

## Skill Policy

- Prefer explicit caveats over convenient fiction. If a runtime helper does not exist, the skill should say so.
- Where the current MCP surface is too weak for a robust prompt-only solution, document the limitation and avoid inventing fake fields or guarantees.
- Keep the legacy markdown docs as historical guidance only. Treat this active bundle as the runtime-aligned version.
- Keep process lifecycle in supervisor scripts or docs, not in the master-agent skill body.
