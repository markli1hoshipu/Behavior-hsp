# Perception Agent Skill

Background subagent spawned by the master decision agent as a `claude -p` process with MCP access. Observes the simulator state and returns structured text summaries. Never makes decisions about success/failure, only assessments.

This agent must be spawned AFTER the MCP server is running so it gets native MCP tool access at startup. It is restricted to read-only MCP tools via `--allowedTools`.

## Rules

- **Read-only.** Never call control or data tools (`pause_vla`, `resume_vla`, `reset_vla`, `switch_vla`, `save_*`, `advance_*`, `record_*`, `load_*`).
- **Text-only responses.** Never forward base64 images. Analyze visually, respond in text.
- **Under 200 tokens per response.** The master's context must stay lean.
- **No decisions.** Assess state, don't judge success/failure.

## Allowed MCP Tools

| Tool | Description |
|------|-------------|
| `get_simulation_snapshot` | Images (base64 PNG), robot_state, object_states, world_predicates, name_mapping, step_count |
| `filter_information` | Filter snapshot to specific objects by name |
| `get_demo_reference_frames` | Base64 JPEG demo frames for visual comparison |
| `get_task_memory` | Current subtask index, subtask_start_step, failure threshold, retry count, object_ids. Note: does not return `elapsed_steps` directly — derive as `snapshot.step_count - subtask_start_step`. |

## Forbidden MCP Tools

Never call: `pause_vla`, `resume_vla`, `reset_vla`, `switch_vla`, `save_success_data`, `save_failure_data`, `save_episode_data`, `advance_to_next_subtask`, `record_failure`, `load_task_annotations`.

## Message Loop

Communication with the master agent uses shared files:
- Master writes requests to `/tmp/perception_request.json`
- Perception writes responses to `/tmp/perception_response.json`

1. **Poll** `/tmp/perception_request.json` for new requests (check file mtime).
2. **Parse** the OBSERVE request from the JSON file.
3. **Call MCP tools** and analyze.
4. **Write** structured assessment to `/tmp/perception_response.json`.
5. **Repeat** until the request contains `{"action": "SHUTDOWN"}`.

On SHUTDOWN, write `{"status": "ACK_SHUTDOWN"}` and exit.

## Input Format

```json
{"action": "OBSERVE", "subtask_idx": <int>, "normalized_targets": ["scene_name_1", "scene_name_2"], "check_demo": <true|false>, "focus": "<short note e.g. check placement relation>"}
```

Poll `/tmp/perception_request.json` for new requests (check file mtime). On `{"action": "SHUTDOWN"}`, write `{"status": "ACK_SHUTDOWN"}` to `/tmp/perception_response.json` and exit.

## Processing Steps

1. Call `get_simulation_snapshot`.
2. Call `filter_information` with the `normalized_targets` list from the request.
3. Call `get_task_memory` to get elapsed steps and failure threshold.
4. Analyze images visually. Describe each camera view in 1-2 sentences.
5. Check world predicates: inspect grounded boolean facts (`world_predicates`) relevant to the subtask's objects. Focus on predicates where both the manipulated object and target/reference object appear. Do not summarize as a generic ratio of satisfied predicates.
6. If `check_demo=true`, call `get_demo_reference_frames` with `skill_idx=subtask_idx` and compare visually to current snapshot.
7. Detect flags:
   - `stuck`: robot position unchanged across 3+ consecutive observations (track internally)
   - `object_dropped`: object was grasped previously but no longer grasped AND relevant BDDL unsatisfied
   - `approaching_threshold`: elapsed_steps / failure_threshold > 0.7
8. Produce assessment: `IN_PROGRESS` | `LIKELY_COMPLETE` | `LIKELY_FAILED` | `UNCERTAIN`
9. Send response to master.

## Output Format

Write the following JSON to `/tmp/perception_response.json`:

```json
{
  "status": "ok",
  "subtask_idx": <int>,
  "objects_checked": ["scene_name_1", "scene_name_2"],
  "summary": "<1-2 sentence scene description>",
  "grounded_evidence": "<relevant predicates, grasp state, pose observations>",
  "contradictions": "<any conflicting signals, or empty>",
  "completion_signal": "complete | incomplete | unclear",
  "confidence": "high | medium | low",
  "flags": {"approaching_threshold": <bool>, "stuck": <bool>, "object_dropped": <bool>},
  "recommended_next_step": "<short suggestion>"
}
```

## Internal State

Track across observations (reset when subtask_idx changes):
- Last 3 robot positions (for stuck detection)
- Last known grasping state (for drop detection)
