# Data Collection Skill (Legacy — for reference only)

Autonomous data collection orchestrator for BEHAVIOR-1K. This skill drives the full observe-decide-act loop using MCP tools connected to a live simulator. The VLA policy runs continuously, and this skill monitors progress asynchronously, only pausing when a critical action (save, switch, reset) is needed.

## MCP Tool Access

This skill requires MCP tools. Claude Code discovers MCP tools at session startup only.

If you are running as a subagent spawned via the Agent tool, you do NOT have MCP access. You must be spawned as a separate `claude -p` process after the MCP server is running:
```bash
claude -p "$(cat /path/to/this/skill.md) Task: <task_name>, task_id=<id>" \
  --mcp-config .mcp.json \
  --permission-mode bypassPermissions \
  --model opus --max-turns 50 \
  > /tmp/agent.log 2>&1 &
```

## Instructions

Follow these steps exactly. Do not skip steps. Do not improvise tool calls -- only use the MCP tools listed below.

### Phase 0: Launch & Readiness

The sim starts PAUSED by default — the robot will not move until you call `resume_vla` after initialization.

1. Check if VLA server is running: `ss -tlnp | grep 8000`. If not, start it:
   ```bash
   cd /home/user/codebase/behavior-1k-solution && CUDA_VISIBLE_DEVICES=1 nohup .venv/bin/python scripts/serve_b1k.py --task-name TASK_NAME --port 8000 policy:checkpoint --policy.config pi_behavior_b1k_fast --policy.dir /home/user/codebase/behavior_submission/checkpoint_2 > /tmp/vla_server.log 2>&1 &
   ```
   Replace `TASK_NAME`. Wait until port 8000 is open (check every 10s, up to 2 min). If timeout, stop — check `/tmp/vla_server.log`.

2. Check if simulator + MCP is running: `ss -tlnp | grep 8001`. If not, start it:
   ```bash
   cd /home/user/codebase/BEHAVIOR-1K && DISPLAY=:1 OMNI_KIT_ACCEPT_EULA=yes nohup conda run -n behavior python OmniGibson/omnigibson/learning/embodiedClaw/run_agentic.py policy=websocket task.name=TASK_NAME headless=false log_path=/home/user/dataset/embodiedClaw_test model.host=localhost model.port=8000 +mcp_port=8001 > /tmp/sim_mcp.log 2>&1 &
   ```
   Replace `TASK_NAME`. Wait until port 8001 is open (check every 10s, up to 5 min). If timeout, stop — check `/tmp/sim_mcp.log`.

3. Call `get_simulation_snapshot`. If it errors or returns no images, sim is still loading. Retry every 15s, up to 5 times. If all fail, stop.
4. Once you get a valid snapshot with images, proceed. The sim is paused — the robot is stationary at step 0.

### Phase 1: Initialize

1. Call `load_task_annotations` with the task ID to load annotation data and initialize task memory.
   - Record the `subtask_list` and `duration_stats` from the response.
   - Note the total number of subtasks.

2. Call `get_task_memory` to read the initial state.
   - Record `current_subtask_idx`, `current_subtask`, `failure_threshold` (`2 * max_duration` across demo episodes for this subtask), and `retry_count`.
   - The current subtask's `object_ids` tells you which objects to watch.

3. Call `get_subtask_language_instruction` to get the VLA instruction for the first subtask.

4. Call `switch_vla` with the language instruction for the first subtask so the VLA knows what to do.

5. Call `resume_vla` to start the simulation. The robot begins moving only after you are fully initialized and ready to observe from step 0.

### Phase 2: Main Loop

Repeat the following cycle until the episode ends. Each iteration corresponds to one observation-decision cycle.

**Do NOT pause on every check.** The VLA runs continuously. You observe asynchronously and only pause when you need to act.

#### Step A: Wait

Sleep for a period proportional to the expected subtask duration before checking. Use these guidelines:
- For navigation subtasks (`skill_type: "navigation"`): wait ~15-25s (~30-50 steps at ~2 steps/s)
- For manipulation subtasks (`skill_type: "uncoordinated"` or `"coordinated"`): wait ~25-50s (~50-100 steps at ~2 steps/s)
- As the subtask approaches the failure threshold, check more frequently (~10-15s)

Use `get_simulation_snapshot` to read `step_count`, and `get_task_memory` to read `subtask_start_step`. Elapsed steps = `step_count - subtask_start_step`.

#### Step B: Observe (NO pause)

1. Call `get_simulation_snapshot` to get the latest images, robot state, object states, and world predicates.
   - This reads from the latest cached observation WITHOUT pausing the simulation.
   - Images will be present (base64 PNGs from head, left_wrist, right_wrist cameras).

2. Call `filter_information` with the annotation object names for the current subtask. Flatten `object_ids` from `get_task_memory`'s `current_subtask` because it is nested lists, and extract the flat list of name strings.

3. Examine the filtered snapshot:
   - **World predicates**: Boolean facts about the relevant objects and robot (e.g., `Inside`, `OnTop`, `Open`). Each predicate has `args` with `scene_name` for identification.
   - **Grasping**: Check `robot_state["grasped_objects"]` (dict of arm → object name), NOT world predicates — there is no `IsGrasping` predicate.
   - **Robot state**: Where is the robot? What is it grasping?
   - **Object states**: Have objects moved to target locations? Use `name_mapping` to correlate BDDL scope names with annotation names.
   - **Images**: Analyze all three camera views (head, left_wrist, right_wrist). Images are a **primary decision signal**, not just confirmation. Look for object positions, contact, containment, robot gripper state, and scene layout. World predicates can be incomplete — images may reveal completion or failure that predicates miss.

#### Step C: Evaluate

Evaluate the subtask using **both** world predicates and image observations. Neither alone is sufficient — predicates can be incomplete and images can be ambiguous. Use all available signals:

1. **Image analysis** (primary): Examine all three camera views. What do you see? Is the object in the target location? Is the robot holding the right object? Does the scene look like the subtask is done? Compare against demo reference frames.
2. **World predicates**: From `world_predicates`, check relevant boolean facts. For placement: `Inside` or `OnTop` should be `true`. For pick: check `robot_state["grasped_objects"]` for the target object name. For navigation: check robot position vs object position.
3. **Demo comparison**: Call `get_demo_reference_frames` with the current `skill_idx` and compare completion frames against your current snapshot images. This is especially important when predicates are ambiguous or unavailable for the subtask type.
4. **Robot state**: Is the robot grasping the correct object? Is it in the expected area?
5. **Step budget**: Call `get_task_memory` and check elapsed steps vs failure threshold. Here `failure_threshold` means `2 * max_duration` across demo episodes for the current `skill_idx`.

**Do not rely on world predicates alone.** If images clearly show completion but predicates don't reflect it (or vice versa), weigh both signals and use demo comparison to break ties. Based on these signals, decide: **continue**, **success**, or **failure**.

#### Step D: Decide

Based on the evaluation, take ONE of the following actions:

##### D1: Continue (subtask in progress)

If the subtask is still in progress:
- Do nothing. Go back to Step A and wait before checking again.
- Optionally call `get_task_memory` to log progress.

##### D2: Success (subtask completed)

If the subtask is completed:

1. Call `pause_vla` to halt the simulation loop.
2. Call `get_simulation_snapshot` to capture the final state at the completion point.
3. Call `save_success_data` with:
   - `subtask_id`: the current subtask's `skill_idx` as a string
   - `next_subtask_id`: the next subtask's `skill_idx` as a string (if any)
   - `next_language_instruction`: the VLA instruction for the next subtask (if any)
   - This saves positive data, updates checkpoint, AND switches VLA to the next instruction automatically.
4. Call `advance_to_next_subtask` to update task memory.
5. If the response says `"episode_complete"` and `"all_subtasks_done": true`:
   - Call `save_episode_data` with `success=true` to save the entire trajectory as positive data.
   - Go to Phase 3.
6. Call `resume_vla` to continue the simulation.
7. Go back to Step A with the new subtask.

##### D3: Failure (subtask failed)

If the subtask failed:

1. Call `pause_vla` to halt the simulation loop.
2. Call `save_failure_data` with `subtask_id` set to the current subtask's `skill_idx` as a string.
   - This saves the failure segment (checkpoint -> failure point) as a negative sample, truncates the main recorder back to the checkpoint, and reverts the sim to the checkpoint.
3. Call `record_failure` to update task memory.
4. Check the response:
   - If `"should_end_episode": true` -- max retries exceeded. Call `save_episode_data` with `success=false` (saves start -> latest checkpoint as positive data). Go to Phase 3.
   - Otherwise, the checkpoint has been restored automatically. The VLA will retry from the last good state.
5. Call `resume_vla` to continue the simulation.
6. Go back to Step A.

### Phase 3: Episode End

When the episode ends (all subtasks completed, max retries exceeded, or sim terminated):

1. If `save_episode_data` was not already called in D2/D3, call it now:
   - `success=true` if all subtasks completed.
   - `success=false` if max retries exceeded (saves start -> last checkpoint as positive data; failure segments were already saved individually by `save_failure_data`).
2. Call `get_task_memory` to get the final state.
3. Report a summary:
   - Total subtasks completed vs. total
   - Number of failures and retries per subtask
   - Whether the episode was a success or failure
   - Key decision points and their outcomes

## Available MCP Tools

### Information Tools

| Tool | Parameters | Description |
|------|-----------|-------------|
| `get_simulation_snapshot` | (none) | Get images (base64 PNG), robot state, object states, world predicates (all boolean predicates for task objects + robot), name_mapping, step count. Works without pausing. |
| `filter_information` | `object_names: list[str]` | Filter snapshot to specified objects (annotation names like `trash_can_116`) and their predicates. Matches by `scene_name`, always includes robot predicates. |

### Control Tools

| Tool | Parameters | Description |
|------|-----------|-------------|
| `pause_vla` | (none) | Halt the simulation loop entirely. Use before save/switch/reset operations. |
| `resume_vla` | (none) | Resume the simulation loop after a pause. |
| `reset_vla` | (none) | Reset the VLA policy's internal state. Use after checkpoint restore. |
| `switch_vla` | `language_instruction: str`, `host?: str`, `port?: int` | Switch the VLA to a new language instruction. Use when transitioning between subtasks. |

### Data Tools

| Tool | Parameters | Description |
|------|-----------|-------------|
| `save_success_data` | `subtask_id: str`, `next_subtask_id?: str`, `next_language_instruction?: str` | Save positive data up to checkpoint (crash protection), update checkpoint. |
| `save_failure_data` | `subtask_id: str` | Save failure segment (checkpoint -> now) as negative sample, truncate recorder, restore checkpoint. |
| `save_episode_data` | `success: bool` | Save positive trajectory. success=true: entire trajectory. success=false: start -> last checkpoint. |

### Annotation & Decision Tools

| Tool | Parameters | Description |
|------|-----------|-------------|
| `load_task_annotations` | `task_id: int` | Load annotation files and compute duration stats. Call once at start. |
| `get_task_memory` | (none) | Get current task progress: subtask index, subtask_start_step, retry counts, failure threshold (`2 * max_duration` across demo episodes for the current subtask), recent decisions. Compute elapsed steps as `step_count - subtask_start_step`. |
| `advance_to_next_subtask` | (none) | Mark current subtask as succeeded and advance. Returns new subtask info and language instruction. |
| `record_failure` | (none) | Mark current subtask as failed. Returns retry count and whether episode should end. |
| `get_subtask_language_instruction` | `skill_idx: int = -1` | Get the VLA language instruction for a subtask. Pass -1 (default) for current subtask. |

### Demo Comparison Tools

| Tool | Parameters | Description |
|------|-----------|-------------|
| `get_demo_reference_frames` | `skill_idx: int`, `camera?: str`, `episode_id?: str`, `frame_types?: list[str]`, `max_episodes?: int` | Get base64-encoded JPEG reference frames from demo videos for visual comparison. Returns frames showing what the scene looks like at various points during the subtask (completion, pre_completion, intermediate). Default: head camera, up to 3 episodes. |

## Decision Rules

### When to Continue
- Elapsed steps < failure threshold
- World predicates for the subtask not yet satisfied
- Robot appears to be making progress (objects moving, robot not stuck)

### When to Save Success
- Images show the subtask is visually complete (object in target location, correct grasp, etc.)
- Relevant world predicates confirm completion (e.g., `Inside(obj, container)=true` for placement, `robot_state["grasped_objects"]` contains object for pick tasks)
- Current images match demo completion frames from `get_demo_reference_frames`
- **Either** strong image evidence **or** strong predicate evidence is sufficient if the other signal is unavailable, but prefer having both

### When to Save Failure
- Elapsed steps > failure threshold (compute from snapshot `step_count` minus `subtask_start_step` from `get_task_memory`)
- Robot is clearly stuck (same position for many steps, object dropped and not recovered)

### When to End Episode
- All subtasks completed successfully
- Same subtask failed 5 times (max retries exceeded)
- Simulation terminated or truncated externally

### Key Principles
- **Minimize pauses.** Only pause when you need to save data, switch subtasks, or reset. Every pause interrupts the VLA.
- **Trust the VLA.** Let it run. Check periodically, not continuously.
- **Use world predicates + vision together.** Neither is sufficient alone. World predicates are grounded boolean facts but can be incomplete. Images show the actual scene state but require interpretation. Always check both and use demo comparison when signals conflict.
- **Be conservative with failure calls.** Wait until the failure threshold is clearly exceeded. Premature failure wastes retries.
- **Always resume after pausing.** Every `pause_vla` must be followed by a `resume_vla` (unless the episode is ending).

## Decision Logging

After every observation-decision cycle, append one JSON line to `/tmp/decisions.jsonl`:

```json
{"ts": "<ISO timestamp>", "step": <step_count>, "subtask_idx": <idx>, "subtask": "<skill_description>", "elapsed": <elapsed_steps>, "threshold": <failure_threshold>, "decision": "<continue|success|failure>", "evidence": "<1-2 sentence summary of why>"}
```

This is mandatory. The log is used for post-run analysis and debugging.
On success or failure actions, also log the MCP tool call results (summarized).
