# Supervisor Handoff

This document is for the outer supervisor layer, not the spawned master agent.

## Purpose

The supervisor owns process lifecycle. It starts and restarts services, waits
for MCP readiness, and hands a single episode job to a fresh master agent.

The supervisor must not act as a second orchestrator.

## Default Ownership Model

- Supervisor:
  - starts or reuses the VLA server
  - starts the simulator + MCP server
  - waits for SSE and MCP tool readiness
  - spawns the master
  - tracks PIDs and logs
  - kills and restarts on failure
- Master:
  - owns the full MCP control loop for one episode
  - owns final decisions, save/fail/advance, and logging
- Optional subagents:
  - spawned only by the master
  - never own process lifecycle

## Preconditions Before Spawning The Master

- VLA is reachable on the configured websocket host and port
- simulator + MCP server are running
- MCP SSE is reachable
- a real MCP tool call succeeds
- the episode job is known

Recommended readiness probe:

- `python mcp_call.py get_simulation_snapshot '{"include_images": false}' 20 --no-images`

## Required Handoff Fields

When spawning the master, provide these fields explicitly:

- `task_name`
- `task_id`
- `decision_mode`
- `pause_before_decision`
- `log_path`
- `episode_output_dir` if known
- `decisions_log`
- optional `instance_id`
- optional `restart_budget` metadata

## Default Episode Boundary

Default operating model:

- `1 episode = 1 master = 1 simulator+MCP process`

Allowed optimization mode:

- up to `5` episodes per simulator/master only if explicitly enabled

Do not let one master span multiple simulator restarts.

## Failure Triggers For Immediate Restart

- master crash or hang
- MCP disconnect
- readiness probe failure
- repeated decision-worker failures
- pause/resume inconsistency
- failed save or restore
- abnormal episode termination
- memory-pressure breach

## What The Supervisor Must Not Do

- routine `pause_vla` / `resume_vla`
- routine save/fail/advance calls
- final subtask decisions
- image-heavy reasoning

The supervisor may use direct SSE tooling only for readiness checks and
emergency recovery.
