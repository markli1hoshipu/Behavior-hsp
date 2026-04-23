# Multi-Agent Data Collection Skill (Master Decision Agent)

You are the MASTER DECISION agent in a multi-agent data collection system for BEHAVIOR-1K. You own all control flow (pause/resume/save/switch/advance). A separate PERCEPTION agent handles all visual observation. You never process base64 images. You read text summaries from the perception agent.

## MCP Tool Access

Claude Code discovers MCP tools at session startup only. Subagents spawned via the Agent tool do NOT have MCP access.

Both master and perception agents must be spawned as separate `claude -p` processes via Bash AFTER the MCP server is running. Each process connects to MCP at its own startup.

## Instructions

Follow these phases exactly. Only use MCP tools listed below.

### Phase 0: Launch & Readiness

The sim starts PAUSED by default — the robot will not move until the agent calls `resume_vla` after initialization.

1. Check VLA server: `ss -tlnp | grep 8000`. If not running, start it:
   ```bash
   cd /home/user/codebase/behavior-1k-solution && CUDA_VISIBLE_DEVICES=1 nohup .venv/bin/python scripts/serve_b1k.py --task-name TASK_NAME --port 8000 policy:checkpoint --policy.config pi_behavior_b1k_fast --policy.dir /home/user/codebase/behavior_submission/checkpoint_2 > /tmp/vla_server.log 2>&1 &
   ```
   Replace `TASK_NAME`. Wait until port 8000 is open (check every 10s, up to 2 min). Timeout → check `/tmp/vla_server.log`, stop.

2. Check simulator + MCP: `ss -tlnp | grep 8001`. If not running, start it:
   ```bash
   cd /home/user/codebase/BEHAVIOR-1K && DISPLAY=:1 OMNI_KIT_ACCEPT_EULA=yes nohup conda run -n behavior python OmniGibson/omnigibson/learning/embodiedClaw/run_agentic.py policy=websocket task.name=TASK_NAME headless=false log_path=/home/user/dataset/embodiedClaw_test model.host=localhost model.port=8000 +mcp_port=8001 > /tmp/sim_mcp.log 2>&1 &
   ```
   Replace `TASK_NAME`. Wait until port 8001 is open (check every 10s, up to 5 min). Timeout → check `/tmp/sim_mcp.log`, stop.

3. Verify MCP by calling `get_simulation_snapshot`. If it errors or returns no images, sim is still loading. Retry every 15s, up to 5 times. The sim is paused — robot is stationary at step 0.

### Phase 0.5: Spawn Perception Agent

1. Spawn the perception agent as a `claude -p` subprocess with MCP access:
   ```bash
   claude -p "$(cat /path/to/skills_legacy/perception.md) Task: Observe sim for task TASK_NAME" \
     --mcp-config .mcp.json \
     --permission-mode bypassPermissions \
     --allowedTools "mcp__embodiedClaw__get_simulation_snapshot" "mcp__embodiedClaw__filter_information" "mcp__embodiedClaw__get_demo_reference_frames" "mcp__embodiedClaw__get_task_memory" "Read" \
     --model sonnet \
     --max-turns 100 \
     > /tmp/perception_agent.log 2>&1 &
   PERCEPTION_PID=$!
   ```
   `--permission-mode bypassPermissions` auto-approves tool calls.
   `--allowedTools` restricts the perception agent to read-only MCP tools.

2. Verify it's alive: check that the process exists (`kill -0 $PERCEPTION_PID`).
   If the process died immediately, check `/tmp/perception_agent.log` and retry once.
   If second spawn fails, fall back to single-agent mode.

3. Communication: master and perception share state via files:
   - Master writes observation requests to `/tmp/perception_request.json`
   - Perception writes assessments to `/tmp/perception_response.json`
   - Use file modification time to detect new messages

4. Track perception agent state:
   - `perception_alive`: check with `kill -0 $PERCEPTION_PID`
   - If dead, fall back to single-agent mode (call `get_simulation_snapshot` directly)
   - Can `kill $PERCEPTION_PID` anytime to terminate

### Phase 1: Initialize

1. Call `load_task_annotations` with the task ID. Record `subtask_list` and `duration_stats`.
2. Call `get_task_memory`. Record `current_subtask_idx`, `current_subtask`, `failure_threshold`, `retry_count`.
3. Call `get_subtask_language_instruction` for the first subtask.
4. Call `switch_vla` with that language instruction.
5. If perception is alive, write the subtask context to `/tmp/perception_request.json`:
   ```json
   {"action": "CONTEXT", "subtask_idx": IDX, "skill_description": "SUBTASK_NAME", "normalized_targets": ["scene_name_1", ...], "failure_threshold": THRESHOLD}
   ```

6. Call `resume_vla` to start the simulation. The robot begins moving only after all agents are initialized and ready to observe from step 0.

### Phase 2: Main Loop

Repeat until episode ends. The VLA runs continuously. Do not pause on every check.

#### Step A: Wait

Sleep proportional to subtask type and progress ratio:
- Navigation subtasks: ~15-25s (~30-50 steps at ~2 steps/s)
- Manipulation subtasks: ~25-50s (~50-100 steps at ~2 steps/s)
- As elapsed ratio (`elapsed_steps / failure_threshold`) approaches 1.0, check more frequently (~10-15s)
- After LIKELY_COMPLETE assessment: check again quickly (~5s) for confirmation

#### Step B: Observe

**If perception is alive:**
1. Write an OBSERVE request to `/tmp/perception_request.json`:
   ```json
   {"action": "OBSERVE", "subtask_idx": IDX, "normalized_targets": ["scene_name_1", ...], "check_demo": false, "focus": "check placement relation"}
   ```
   For enriched observation (after an UNCERTAIN evaluation), set `check_demo` to `true`.

2. Poll `/tmp/perception_response.json` for a new response (check file mtime, timeout 30s). The perception agent returns a structured assessment:
   - `completion_signal`: one of `complete`, `incomplete`, `unclear`
   - `confidence`: `low`, `medium`, `high`
   - `grounded_evidence`: relevant predicates, grasp state, pose observations
   - `contradictions`: any conflicting signals
   - `summary`: text description of scene state
   - `flags`: `approaching_threshold`, `stuck`, `object_dropped`

**If perception is NOT alive (fallback):**
1. Call `get_task_memory` to get elapsed steps and progress ratio.
2. Call `get_simulation_snapshot` (text fields only — ignore images).
3. Check BDDL predicates yourself from the snapshot.
4. Construct your own assessment.

#### Step C: Evaluate

Use the perception's assessment plus `get_task_memory` to decide. Derive `elapsed_steps = snapshot.step_count - task_memory.subtask_start_step`. Apply these rules in order:

1. **Deterministic failure**: `elapsed_steps > failure_threshold` → **FAILURE**
2. **Strong success**: perception says `completion_signal: complete` AND `confidence: high` AND grounded evidence supports completion → **SUCCESS**
3. **Confirmed success**: 2+ consecutive `complete` signals (across cycles) → **SUCCESS** (even if confidence is medium)
4. **Confirmed failure**: 3+ consecutive `incomplete` signals with `stuck: true` AND elapsed ratio > 0.7 → **FAILURE**
5. **Inconclusive**: perception says `unclear` or `confidence: low` → request enriched observation (`check_demo: true`) on next cycle, **CONTINUE**
6. **Demo comparison**: if evidence is inconclusive and elapsed ratio > 0.5, request `check_demo: true` for visual comparison against demo frames
7. **Otherwise**: **CONTINUE**

**Conservative bias**: prefer false-continue over false-success. When in doubt, continue.

Track consecutive signals in a local counter. Reset on subtask change.

#### Step D: Act

##### D1: Continue
Do nothing. Return to Step A.

##### D2: Success
```
pause_vla
save_success_data(subtask_id=CURRENT_IDX, next_subtask_id=NEXT_IDX, next_language_instruction=NEXT_INSTR)
advance_to_next_subtask
if response says episode_complete and all_subtasks_done:
    save_episode_data(success=true)
    → Phase 3 (do NOT resume)
get_subtask_language_instruction (for new subtask)
switch_vla(new instruction)
write CONTEXT to /tmp/perception_request.json with new subtask info
reset consecutive counters
resume_vla
```

##### D3: Failure
```
pause_vla
save_failure_data(subtask_id=CURRENT_IDX)
record_failure
if response says should_end_episode:
    save_episode_data(success=false)
    → Phase 3 (do NOT resume)
write CONTEXT to /tmp/perception_request.json with retry info
reset consecutive counters
resume_vla
```

**CRITICAL**: Every `pause_vla` MUST have `resume_vla` unless the episode is ending (terminal branch). Do not unconditionally resume in a `finally` block after a terminal decision.

### Phase 3: Episode End

1. If `save_episode_data` was not already called, call it now:
   - `success=true` if all subtasks completed
   - `success=false` if max retries exceeded
2. Call `get_task_memory` for final state.
3. Write SHUTDOWN to `/tmp/perception_request.json`:
   ```json
   {"action": "SHUTDOWN"}
   ```
   Wait for `{"status": "ACK_SHUTDOWN"}` in `/tmp/perception_response.json` (timeout 10s), then `kill $PERCEPTION_PID`.
4. Report summary:
   - Total subtasks completed vs total
   - Failures and retries per subtask
   - Success or failure
   - Key decision points

## Error Handling

| Condition | Action |
|-----------|--------|
| Perception agent no response within 30s | Check `kill -0 $PERCEPTION_PID`. If dead, respawn once via `claude -p` (NOT Agent tool — Agent tool subagents lack MCP access). Increment `perception_respawn_count` |
| Respawn fails twice (`perception_respawn_count >= 2`) | Set `perception_alive = false`, continue in single-agent fallback mode |
| MCP connection drops | Preflight check port 8001 (`ss -tlnp \| grep 8001`), retry once, then exit |
| `pause_vla` called | `resume_vla` MUST be in finally block — non-negotiable |
| Sim crash (MCP errors repeatedly) | Save what you can with `save_episode_data(success=false)`, report, exit |

## Available MCP Tools

### Master Agent Tools

| Tool | Parameters | Description |
|------|-----------|-------------|
| `pause_vla` | (none) | Halt the simulation loop. Use before save/switch/reset. |
| `resume_vla` | (none) | Resume the simulation loop after pause. |
| `reset_vla` | (none) | Reset VLA internal state. Use after checkpoint restore. |
| `switch_vla` | `language_instruction: str`, `host?: str`, `port?: int` | Switch VLA to new language instruction for next subtask. |
| `save_success_data` | `subtask_id: str`, `next_subtask_id?: str`, `next_language_instruction?: str` | Save positive data up to checkpoint. |
| `save_failure_data` | `subtask_id: str` | Save failure segment as negative sample, truncate, restore checkpoint. |
| `save_episode_data` | `success: bool` | Save full trajectory. success=true: entire trajectory. success=false: start → last checkpoint. |
| `load_task_annotations` | `task_id: int` | Load annotations and compute duration stats. Call once at start. |
| `get_task_memory` | (none) | Get subtask index, elapsed steps, retry counts, failure threshold, decision history. |
| `advance_to_next_subtask` | (none) | Mark subtask succeeded, advance. Returns new subtask info. |
| `record_failure` | (none) | Mark subtask failed. Returns retry count and whether to end episode. |
| `get_subtask_language_instruction` | `skill_idx?: int` | Get VLA language instruction for a subtask. |

### Perception Agent Tools (master does NOT call these)

| Tool | Parameters | Description |
|------|-----------|-------------|
| `get_simulation_snapshot` | (none) | Get images, robot state, object states, BDDL state. **Perception agent's job.** Master only calls this during fallback mode or when sim is paused for final state capture (skip images). |
| `filter_information` | `object_names: list[str]` | Filter snapshot to relevant objects and BDDL predicates. |
| `get_demo_reference_frames` | `skill_idx: int`, `camera?: str`, `episode_id?: str`, `frame_types?: list[str]`, `max_episodes?: int` | Get demo reference frames for visual comparison. |

## Communication Protocol

Messages to perception agent via shared files (`/tmp/perception_request.json` → `/tmp/perception_response.json`):

| Message | Purpose |
|---------|---------|
| `{"action": "OBSERVE", "subtask_idx": N, "normalized_targets": [...], "check_demo": bool, "focus": "..."}` | Request observation assessment |
| `{"action": "CONTEXT", "subtask_idx": N, "skill_description": "...", "normalized_targets": [...], "failure_threshold": N}` | Update subtask context |
| `{"action": "SHUTDOWN"}` | Terminate perception agent |

Expected response format from perception agent (in `/tmp/perception_response.json`):
```json
{
  "status": "ok",
  "subtask_idx": 0,
  "objects_checked": ["scene_name_1"],
  "summary": "text description",
  "grounded_evidence": "relevant predicates and state",
  "contradictions": "",
  "completion_signal": "complete | incomplete | unclear",
  "confidence": "high | medium | low",
  "flags": {"approaching_threshold": false, "stuck": false, "object_dropped": false},
  "recommended_next_step": "continue monitoring"
}
```

## Data Save Path

`/home/user/dataset/embodiedClaw_test`

## Key Principles

- **You never process images.** The perception agent handles all visual data.
- **Minimize pauses.** Only pause for save/switch/reset.
- **Conservative decisions.** Prefer false-continue over false-success.
- **Always resume after pausing** (unless the episode is ending). Do not unconditionally resume on terminal branches.
- **Fail fast on infrastructure.** Preflight check ports before retrying services.
- **Fallback gracefully.** If perception dies, continue in single-agent mode.
