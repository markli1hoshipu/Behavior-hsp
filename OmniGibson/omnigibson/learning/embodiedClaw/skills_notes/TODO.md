# Skills TODO

## Agent Architecture Redesign

Restructure SKILL.md to use a multi-agent architecture with `claude -p` subprocesses.

### Key findings (2026-04-07)

- Claude Code MCP tools are discovered at session startup only
- Agent tool subagents do NOT inherit MCP connections
- `claude -p --mcp-config .mcp.json --permission-mode bypassPermissions` gives a subprocess native MCP tool access (verified working)
- Parent can `kill $PID` subprocesses at any time

### Proposed architecture

**Sim Manager** (Agent tool, no MCP):
- Starts/stops/restarts VLA, sim, MCP server
- Handles instance switching and crash recovery
- Returns PIDs when ports are up

**Decision Agent** (`claude -p`, has MCP tools):
- Spawned AFTER MCP server is running
- Owns the observe-decide-act collection loop
- No process management responsibility
- Killed and re-spawned on instance switch

**Monitor** (Agent tool, no MCP):
- Watches logs, ports, data output, process health

### Instance switch flow

1. Kill decision agent (`kill $DECISION_PID`)
2. Spawn new sim manager: "kill old sim, start new instance for task X"
3. Sim manager returns when ports up
4. Spawn new decision agent

### Files to update

- [x] `skills/SKILL.md` — add agent architecture, MCP access section, startup flow
- [x] `skills/references/multi_agent_protocol.md` — update for `claude -p` pattern
- [x] `skills/references/single_agent_loop.md` — ensure it works as standalone prompt for `claude -p`
- [ ] `skills_legacy/` — reference-only material; remove or archive outside the canonical skill path instead of continuing to update it

### Also update

- [x] `skills/references/perception_role.md` — file-based communication instead of SendMessage
- [x] `skills/references/runtime_alignment.md` — document `claude -p` as known runtime pattern

## Adaptive Simulation Speed

If the decision agent takes too long between consecutive decisions (e.g., 300+ frames pass between two observation-decision cycles), the agent should detect this and adjust the simulation speed. One approach: inject a short delay between sim steps (via an MCP tool or sim config) to slow the simulation down so the decision agent doesn't miss critical state changes like subtask completion boundaries.

This matters because:
- The agent processes images + predicates + reasoning per cycle, which can take 10-30s
- At ~10 Hz sim speed, that's 100-300 frames per decision
- If a subtask completes in a narrow window, the agent might miss it entirely
- Slowing the sim to match the agent's decision cadence prevents missed boundaries

Implementation ideas:
- Add a `set_sim_delay(seconds: float)` MCP tool that inserts a `time.sleep()` between env.step() calls
- The agent monitors `step_count` delta between consecutive observations
- If delta > threshold (e.g., 1000), call `set_sim_delay(0.1)` to slow down
- If delta < lower threshold (e.g., 50), call `set_sim_delay(0)` to speed back up
- Could also be done automatically in the sim loop based on MCP request frequency

### Decision Module Context clear and max_turns clear (naively implemented)
The decision module's decision process only depends on current state and observation. The history context is meaningless. Clear context and max_turns so that the everytime the decision is made from a fresh agent and it does not die.

### Pause VLA before decision?

### Decision Module needs some sort of memory (for example picking up soda_cans, it needs to know what has been changed since last time).

### For object with the same name, ignore ID

### step skipping (decision interval too long, 2 subtasks completed) (delta predicates may be incorrect )
{"decision_num":6,"step":1646,"elapsed":328,"threshold":2360,"subtask_idx":3,"subtask":"pick_up_can_of_soda","evidence":{"IsGrasping_robot_can_of_soda":false,"left_eef_near_can":true},"decision":"continue"}

{"decision_idx":7,"subtask_idx":3,"subtask":"pick_up_from can_of_soda","step":1923,"elapsed":605,"threshold":2360,"decision":"success","reason":"can_of_soda_113 Inside trash_can_116=true, off floor, effectively collected via scoop into held container","timestamp":"2026-04-09T05:46:51-04:00"}

{"decision_idx":8,"subtask_idx":4,"subtask":"place_in can_of_soda trash_can","step":2121,"elapsed":198,"threshold":1350,"decision":"success","reason":"Inside(can_of_soda_113,trash_can_116)=true, not grasped, placement satisfied","timestamp":"2026-04-09T05:48:20-04:00"}

- Instead of getting the delta between subtask start and last decision, can try to maintain a list of all changed predicates (with timestamp), and then give agent


### Failure detection: Right now is mainly from step wise. Could use help from Jinbang

### depth video saving too large

### sim spawn bug (shake after reset)