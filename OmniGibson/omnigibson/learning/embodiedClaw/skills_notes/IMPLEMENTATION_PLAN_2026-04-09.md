# EmbodiedClaw Skill Rewrite Plan

Date: 2026-04-09

## Goals

- Rewrite the skill around a spawned MCP-capable master agent that owns one episode.
- Move bootstrap, restart, and process lifecycle instructions out of `SKILL.md`.
- Keep MCP as the simulator control surface.
- Make `image_state` support a fresh disposable decision worker.
- Add a practical bash supervisor entrypoint for one-episode runs.

## Architecture Decisions

- Default unit of work: `1 episode = 1 master = 1 simulator+MCP process`.
- Optional optimization: allow up to `5` episodes per simulator/master only behind explicit config, not as the documented default.
- The outer supervisor owns:
  - VLA startup
  - simulator + MCP startup
  - SSE readiness checks
  - master spawning
  - PID/log tracking
  - emergency restart
- The spawned master owns:
  - `load_task_annotations`
  - `get_task_memory`
  - `get_subtask_language_instruction`
  - `switch_vla`
  - `pause_vla` / `resume_vla`
  - `save_success_data` / `save_failure_data` / `save_episode_data`
  - `advance_to_next_subtask`
  - `record_failure`
  - decision logging
- Separate decision workers are optional:
  - `imageless_state`: no separate decision worker by default
  - `image_state`: fresh decision worker is recommended when image context pressure matters

## Files To Change

- Rewrite `skills/SKILL.md`
- Add `skills/references/supervisor_handoff.md`
- Update:
  - `skills/references/single_agent_loop.md`
  - `skills/references/multi_agent_protocol.md`
  - `skills/references/runtime_alignment.md`
  - `README.md`
- Add a supervisor launcher:
  - `OmniGibson/omnigibson/learning/embodiedClaw/run_supervised_episode.sh`

## Non-Goals For This Patch

- No simulator-core refactor
- No replacement of MCP with a new control stack
- No commitment to `orchestrator.py` as the primary runtime
- No broad cleanup of `skills_legacy/`

## Run Modes To Support

- Manual interactive mode:
  - supervisor script starts VLA + sim + MCP
  - user opens a fresh agent session and uses the skill as the master
- Automatic bash mode:
  - supervisor script starts VLA + sim + MCP
  - supervisor script optionally spawns the master via `claude -p`

## Acceptance Criteria

- `SKILL.md` is written from the master-agent perspective only
- Bootstrap responsibilities are documented outside the skill body
- The README tells the user exactly how to run one episode
- The new launcher starts one episode at a time and checks MCP readiness
- The docs explicitly state when a separate decision worker is recommended
