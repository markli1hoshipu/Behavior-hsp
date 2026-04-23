---
name: embodiedclaw
description: Use when operating the embodiedClaw MCP data-collection loop in OmniGibson. Provides runtime-aligned guidance for annotation loading, snapshot filtering, subtask completion decisions, multi-agent coordination, demo comparison, and failure recovery.
---

# EmbodiedClaw Agentic Collection

Operational skill package for the embodiedClaw MCP data-collection system. Matches the current MCP server and decision code implementation.

## Decision Modes

The orchestrator must choose exactly one mode for a collection run and pass it
to every fresh decision agent prompt as `Decision mode: <mode>`.

- `image_state` (default): use the current full observation surface. Decision
  agents call `get_simulation_snapshot(include_images=true)` and
  `filter_information(..., include_images=true)`, use structured simulator state
  plus current images, and may call `get_demo_reference_frames()` only when the
  rubric requests visual comparison.
- `imageless_state`: use only structured simulator state. Decision agents call
  `get_simulation_snapshot(include_images=false)` and
  `filter_information(..., include_images=false)`. They must not inspect image
  payloads, call `get_demo_reference_frames()`, or use a perception agent. The
  `images` field should be `{}` by construction; if a tool response contains
  image payloads anyway, ignore them and do not cite them as evidence.

Decision JSON must keep these channels separate: `state_evidence` contains
predicate, predicate delta, robot, object, timing, and threshold evidence;
`visual_evidence` contains image/demo evidence in `image_state` and exactly
`not_used_by_mode` in `imageless_state`.

The orchestrator may also set `Pause before decision: <true|false>` in the
per-decision prompt. Default to `false` unless the run prompt says otherwise.
This flag is owned by the orchestrator, not the decision agent:

- If `Pause before decision: true`, call `pause_vla()` before spawning the
  fresh decision agent. Keep the sim paused while the agent observes and
  decides.
- If `Pause before decision: false`, do not pause for routine `continue`
  checks. Use asynchronous snapshots and pause only for stable final
  confirmation, save, restore, or transition.
- For `continue` after a paused decision, call `resume_vla()` and wait for the
  next cadence interval.
- For `success` or `failure`, stay paused and follow `failure_recovery.md`.
  Resume only on non-terminal branches.

Snapshots include predicate deltas automatically:
`predicate_deltas_since_subtask_start` and
`predicate_deltas_since_previous_decision`. These track which predicates
changed value across decision cycles. The MCP server manages baselines
internally — they reset on subtask advance, failure restore, and episode
reset. Decision agents cite deltas in `state_evidence`. In `imageless_state`,
deltas are the primary directional evidence channel since images are
unavailable.

## Read Order

- Read `references/runtime_alignment.md` first. It records the prompt/runtime gaps that still exist in the current implementation.
- Read `references/tool_contracts.md` before calling MCP tools or deriving values.
- Read `references/single_agent_loop.md` for the default one-agent collection flow.
- Read `references/decision_rubric.md` when deciding `continue`, `success`, `failure`, or `needs_visual_comparison`.
- Read `references/multi_agent_protocol.md` only when using a master/perception split.
- Read `references/perception_role.md` only when spawning a dedicated perception subagent.
- Read `references/failure_recovery.md` before handling success, retry, restore, or episode termination.
- Read `references/gt_replay_protocol.md` only for offline replay evaluation.

## Startup

Use a 3-process layout: VLA policy server, simulator + MCP server, and orchestrator (this session). The sim starts PAUSED by default — the robot will not move until the orchestrator calls `resume_vla`.

1. Start the VLA policy server first (required for sim initialization):
   `cd /home/user/codebase/behavior-1k-solution && CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/serve_b1k.py --task-name=<task_name> --port=8000 policy:checkpoint --policy.config pi_behavior_b1k_fast --policy.dir /home/user/.cache/huggingface/hub/models--IliaLarchenko--behavior_submission/snapshots/e2012d0a102e0d21fdcfb72009a87428917fb15c/checkpoint_2`
2. Start simulator + MCP server (sim loop starts paused):
   `cd /home/user/BEHAVIOR-1K && DISPLAY=:0 OMNI_KIT_ACCEPT_EULA=yes conda run -n behavior python OmniGibson/omnigibson/learning/embodiedClaw/run_agentic.py policy=websocket task.name=<task_name> model.host=127.0.0.1 model.port=8000 headless=false log_path=<log_path>`
3. Confirm the MCP server is live. In `image_state`, call
   `get_simulation_snapshot(include_images=true)` and retry if images are
   missing or the call errors. In `imageless_state`, call
   `get_simulation_snapshot(include_images=false)` and require state fields but
   do not require images. Do not use `get_task_memory()` as a readiness check
   before `load_task_annotations()`.
4. Initialize while the robot is stationary:
   1. Derive the numeric `task_id` from launch context or `omnigibson/learning/utils/eval_utils.py:TASK_NAMES_TO_INDICES`.
   2. Call `load_task_annotations(task_id=...)`.
   3. Call `get_task_memory()`. Record `current_subtask_idx`, `current_subtask`, `failure_threshold`, `retry_count`.
   4. Call `get_subtask_language_instruction()` for the current subtask.
   5. Call `switch_vla(...)` with that language instruction.
5. Call `resume_vla` as the LAST step of initialization — the robot begins moving and the orchestrator starts the observe-act loop.

## Orchestration Gotchas

- The orchestrator (this session) has direct MCP access via `.mcp.json` in the project root. Call control-plane tools (`pause_vla`, `resume_vla`, `load_task_annotations`, `get_task_memory`, `switch_vla`, `save_success_data`, `save_failure_data`, `save_episode_data`, etc.) directly. Do NOT spawn `claude -p` subagents for these calls.
- `claude -p` is for subagents only (decision agents, perception agents in multi-agent mode). Subagents need `--mcp-config` explicitly because subprocesses do not inherit the parent session's MCP config.
- Do NOT use `--model` flag with `claude -p` — causes Bedrock "invalid beta flag" error. Omit it entirely.
- `claude -p` buffers ALL stdout until the process completes. Parse the JSON decision from stdout after the process exits.
- `resume_vla` sets a `threading.Event` that persists even after the agent process dies — the sim keeps running.
- Kill all agents and services: `~/kill_embodiedclaw.sh`

Do not assume `policy=local` is enough for real collection. In this repo it is a zero-action stub unless another component injects a live policy.

## Default Live Collection Loop

The orchestrator (this session) owns the loop. Decision agents are spawned fresh per cycle with clean context.

1. Obtain the numeric `task_id` from launch context or the static task-name mapping in `omnigibson/learning/utils/eval_utils.py`. Do not guess it heuristically.
2. Call `load_task_annotations(task_id=...)` once before relying on `get_task_memory()`.
3. Read the run prompt's `Pause before decision: <true|false>` flag. Default to `false`.
4. Wait based on subtask type or step cadence: roughly 200-400 sim steps, ~15-25s for navigation, or ~25-50s for manipulation. Shorten as elapsed ratio approaches timeout.
5. If `Pause before decision: true`, call `pause_vla()` now. If the flag is `false`, leave the sim running for routine checks.
6. Spawn a fresh decision agent:
   `claude -p "Observe and decide. Task: <task_name>. Decision mode: <image_state|imageless_state>. Pause before decision: <true|false>." --append-system-prompt "$(cat /home/user/BEHAVIOR-1K/OmniGibson/omnigibson/learning/embodiedClaw/skills/SKILL.md/decision_prompt.md)" --mcp-config /home/user/BEHAVIOR-1K/.mcp.json --permission-mode bypassPermissions --max-turns 20`
7. Parse the JSON decision from stdout. The decision agent outputs one JSON block and exits.
8. Act on the decision:
   - `continue`: if paused, call `resume_vla()`, then go to step 4.
   - `success`: if not already paused, call `pause_vla()` first, then follow `references/failure_recovery.md` Success Branch. On episode complete, call `save_episode_data(success=true)` and stop.
   - `failure`: if not already paused, call `pause_vla()` first, then follow `references/failure_recovery.md` Failure Branch. On retries exhausted, call `save_episode_data(success=false)` and stop.
   - `needs_visual_comparison`: if already paused, stay paused and re-spawn immediately; otherwise re-spawn with a shorter wait. The next decision agent will fetch demo frames itself.
9. Resume the simulator only on non-terminal branches.

## Single-Agent vs Multi-Agent

Default to single-agent mode (orchestrator + per-decision agents). Use multi-agent mode (`references/multi_agent_protocol.md`) only when:

- The orchestrator wants to offload perception to a dedicated agent
- A dedicated perception agent can keep observations short and structured
- The added messaging overhead is smaller than the decision burden on the orchestrator

If the perception agent is slow, unreliable, or dies twice, fall back to single-agent mode for the rest of the episode.

## Non-Negotiables

- Do not assume fields that are not actually returned by `get_task_memory()`.
- Do not pass nested `current_subtask.object_ids` directly into `filter_information()`.
- Use normalized `object_ids` / `manipulating_object_ids` for rollout decisions. Treat `annotation_*` fields as debug-only.
- Do not treat `switch_vla()` as proof that the active model instruction changed.
- Do not treat the perception role as sandboxed; the current MCP server does not enforce read-only access.
- Do not use generic "count the satisfied predicates" logic for all subtasks. Use the grounded evidence rubric.

## Decision Logging

After every decision agent completes, the orchestrator appends one JSON line to `/tmp/decisions.jsonl`. The decision agent outputs this JSON to stdout — the orchestrator parses and logs it.

```json
{"ts": "<ISO timestamp>", "decision_mode": "<image_state|imageless_state>", "pause_before_decision": <true|false>, "step": <step_count>, "subtask_idx": <idx>, "subtask": "<skill_description>", "elapsed": <elapsed_steps>, "threshold": <failure_threshold>, "decision": "<continue|success|failure|needs_visual_comparison>", "state_evidence": "<predicate/robot/object/timing evidence>", "visual_evidence": "<image/demo evidence|not_used_by_mode>", "evidence": "<1-2 sentence summary of why>"}
```

This is mandatory. The log is used for post-run analysis and debugging.
On success or failure actions, also log the MCP tool call results (summarized).
In multi-agent mode, only the orchestrator writes this log.
