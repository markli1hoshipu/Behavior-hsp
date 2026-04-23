# EmbodiedClaw Compact Brief

- Write the skill for the spawned MCP-capable master, not the bootstrap session.
- Default ownership: supervisor -> one master per episode -> optional image-state decision worker.
- Keep MCP as the simulator control API.
- Keep process lifecycle in bash/script docs, not inside `SKILL.md`.
- `imageless_state`: master-only is acceptable.
- `image_state`: prefer a fresh stateless decision worker so the master does not accumulate raw image context.
- Use MCP task memory and predicate deltas as the real long-horizon memory source.
- Add a one-episode supervisor script that can:
  - start VLA
  - start sim + MCP
  - wait for real MCP readiness using `mcp_call.py`
  - either print a handoff for a fresh interactive master or auto-spawn one with `claude -p`
