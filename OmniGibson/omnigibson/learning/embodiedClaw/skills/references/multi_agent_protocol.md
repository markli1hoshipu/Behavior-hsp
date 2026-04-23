# Multi-Agent Protocol

Use this only when splitting work between a master agent and a perception agent
in `image_state` after supervisor handoff. Do not use this protocol in
`imageless_state`.

## Reality Check

- The supervisor is outside this protocol. It starts services and spawns the master.
- The master and perception agents talk to the same MCP server as separate `claude -p` processes.
- Each process must be spawned AFTER the MCP server is running to get native MCP tool access.
- Do NOT use the Agent tool for subagents that need MCP tools. Use `claude -p` via Bash.
- Read-only perception is enforced at the MCP-tool layer via `--allowedTools`. `Bash` is allowed only for shared-file transport. Use `--permission-mode bypassPermissions` for non-interactive auto-approval.
- The master agent owns all final decisions, persistence calls, and simulator control.
- Do not pause on every observation cycle. Pause only when a stable save or transition point is required.
- Agents communicate via shared files (e.g., `/tmp/perception_request.json`, `/tmp/perception_response.json`), not SendMessage.

## When To Use This

Use multi-agent mode only if:

- the decision mode is `image_state`
- the master wants to avoid reading base64 images directly
- a second agent can keep observations short and structured
- the added messaging overhead is smaller than the decision burden

If the perception agent is slow or unreliable, fall back to single-agent mode.

## Startup and Liveness

1. The supervisor brings up the VLA server and simulator/MCP server first.
2. The supervisor hands one episode to the master.
3. The master confirms the MCP endpoint is reachable.
4. The master spawns the perception agent as a `claude -p` process:
   ```bash
   claude -p "$(cat .../references/perception_role.md) Task: observe for <task_name>" \
     --mcp-config .mcp.json \
     --permission-mode bypassPermissions \
     --allowedTools "mcp__embodiedClaw__get_simulation_snapshot" "mcp__embodiedClaw__filter_information" "mcp__embodiedClaw__get_demo_reference_frames" "Read" "Bash" \
     --max-turns 100 \
     > /tmp/perception_agent.log 2>&1 &
   PERCEPTION_PID=$!
   ```
5. Check it started: `kill -0 $PERCEPTION_PID`. If dead, check log and retry once.
6. Write a test `CONTEXT` request to `/tmp/perception_request.json` and wait for a matching `CONTEXT_ACK` in `/tmp/perception_response.json`.
7. If the second attempt fails, continue in single-agent mode.

Track:

- `PERCEPTION_PID` — check liveness with `kill -0 $PERCEPTION_PID`
- Can `kill $PERCEPTION_PID` to terminate at any time
- whether the current subtask context has already been written

Do not leave the simulator paused while waiting for subagent file output.

## File Transport Rules

- The master writes one JSON object at a time to `/tmp/perception_request.json`.
- The perception agent writes one JSON object at a time to `/tmp/perception_response.json`.
- Every request must include a unique `request_id` string and an `action` field.
- Every response must echo the same `request_id`.
- To avoid partial reads, write to a temporary file such as `/tmp/perception_request.json.tmp`, then atomically rename it to `/tmp/perception_request.json`.
- Use file modification time plus `request_id` to detect new messages. Ignore stale responses whose `request_id` does not match the current request.
- If no matching response arrives within 30s, retry once. If the retry fails, fall back to single-agent mode.

## Role Split

### Master

- Owns the episode
- Owns `pause_vla`, `resume_vla`, `reset_vla`, `switch_vla`
- Owns `load_task_annotations`, `get_task_memory`, `advance_to_next_subtask`,
  `record_failure`
- Owns `save_success_data`, `save_failure_data`, `save_episode_data`
- Owns the final `continue`, `success`, `failure`, or `needs_visual_comparison` decision
- Owns fallback when the perception agent is missing, stale, or ambiguous

### Perception

- Reads `get_simulation_snapshot(include_images=true)`
- May read `filter_information(..., include_images=true)` after the master provides normalized target names
- May inspect demo reference frames only when the master explicitly asks
- Must not call control or persistence tools
- Must not decide episode transitions

## Observation Cadence

Do not observe at a fixed high frequency.

Use a coarse cadence first, then tighten near timeout:

- navigation-heavy subtasks: roughly every 15-25s
- manipulation-heavy subtasks: roughly every 25-50s
- once the attempt is near the threshold, shorten the interval
- after a likely-complete assessment, recheck quickly for confirmation

The master should derive urgency from:

- `elapsed_steps`
- `failure_threshold` (`2 * max_duration` across demo episodes for the current subtask)
- current subtask type
- whether the last observation was confident or unclear

## Canonical Message Flow

1. Master reads `get_task_memory()` and `get_simulation_snapshot(include_images=true)` when needed.
2. Master derives normalized target names using `tool_contracts.md`.
3. Master sends one `CONTEXT` message when the subtask changes or normalized targets change.
4. Master sends one `OBSERVE` request for the current cycle.
5. Perception replies with the exact response schema below.
6. Master applies `decision_rubric.md`.
7. Master performs recovery or advancement.
8. Master resumes only on non-terminal branches.

## Canonical `CONTEXT` Message

Send exactly these fields:

- `request_id`
- `action`: `CONTEXT`
- `subtask_idx`
- `skill_description`
- `skill_type`
- `object_groups`
- `manipulating_object_ids`
- `normalized_targets` (rollout-facing type names such as `can_of_soda`)
- `elapsed_steps`
- `failure_threshold` (`2 * max_duration` across demo episodes for the current subtask)
- `check_demo`

Perception must respond with a short acknowledgement:

- `request_id`
- `action`: `CONTEXT_ACK`
- `status`: `ok` or `needs_clarification`
- `subtask_idx`
- `summary`

For a liveness probe before real work, send the same schema with placeholder values for a known task or the current task after initialization.

## Canonical `OBSERVE` Request

Send exactly these fields:

- `request_id`
- `action`: `OBSERVE`
- `subtask_idx`
- `normalized_targets` (usually normalized type names; exact runtime names only when intentionally narrowed)
- `check_demo`
- `focus`

Where `focus` is a short note such as:

- `check placement relation`
- `check whether robot is grasping target`
- `check if state change is visible`

## Canonical `OBSERVE` Reply

Reply with exactly these fields:

- `request_id`
- `action`: `OBSERVE_RESULT`
- `status`: `ok` or `needs_clarification`
- `subtask_idx`
- `objects_checked`
- `summary`
- `grounded_evidence`
- `contradictions`
- `completion_signal`: `complete`, `incomplete`, or `unclear`
- `confidence`: `high`, `medium`, or `low`
- `recommended_next_step`

Keep the reply short and decision-facing. Do not emit a second schema.

## Canonical `SHUTDOWN` Request

Send:

- `request_id`
- `action`: `SHUTDOWN`

Perception should reply with:

- `request_id`
- `action`: `SHUTDOWN_ACK`
- `status`: `ok`

Then the perception process should exit.

## Master Decision Policy

- Treat the perception reply as evidence, not authority.
- Prefer false-continue over false-success.
- Use image observations, predicates, grasp state, and pose as joint evidence, not a predicates-only rule.
- If the reply is `needs_clarification`, either resend a tighter `focus` request or fall back to the master inspecting the snapshot directly.
- If the reply is `unclear` and the attempt is still early, continue.
- If the reply is `unclear` and the attempt is late, request demo comparison or fall back to a paused direct inspection.

## Pause Discipline

If the master pauses:

- use the paused window for save, restore, final confirmation, or handoff-critical state capture
- keep the paused period short
- always resume on non-terminal branches

Every `pause_vla` must be paired with `resume_vla` in a `finally` path unless the
episode is ending.

## Timeout and Fallback

- If perception does not acknowledge the context or reply to `OBSERVE`, retry once.
- If the second attempt fails, the master falls back to single-agent operation.
- If the perception reply is repeatedly low-signal, stop using multi-agent mode for
  that episode.
- The supervisor does not participate in this fallback except for process-level failure or restart.
