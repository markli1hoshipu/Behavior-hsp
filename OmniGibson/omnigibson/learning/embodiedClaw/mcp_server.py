"""
MCP Tool Server for embodiedClaw -- bridges Claude Code <-> live simulator.

This server runs inside the simulator process, holds references to live
sim objects (env, robot, controller, checkpoint_mgr), and exposes every
tool as a named MCP tool.  Claude Code connects as a native MCP client.

Design: thin routing layer.  Each @mcp.tool() handler delegates to the
existing tool functions in tools/ -- it binds live objects and forwards
agent-supplied parameters.

Transport: the server supports both stdio and SSE transports.  The
``run()`` method blocks on stdio by default; ``run_sse()`` starts an
HTTP+SSE server (useful when the sim loop occupies the main thread and
the MCP server runs in a background thread).

Usage::

    from omnigibson.learning.embodiedClaw.mcp_server import EmbodiedClawMCPServer

    server = EmbodiedClawMCPServer(
        env=env,
        robot=robot,
        task_name="washing_dishes",
        controller=controller,
        checkpoint_mgr=checkpoint_mgr,
    )
    # Option A: run blocking on stdio (for subprocess-based MCP)
    server.run()

    # Option B: run in a background thread with SSE transport
    import threading
    t = threading.Thread(target=server.run_sse, daemon=True)
    t.start()
"""

import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from omnigibson.learning.embodiedClaw.tools.information_tools import (
    get_simulation_snapshot as _get_simulation_snapshot,
    filter_information as _filter_information,
    add_predicate_deltas as _add_predicate_deltas,
    strip_snapshot_images as _strip_snapshot_images,
)
from omnigibson.learning.embodiedClaw.tools.control_tools import (
    VLAPolicyController,
    pause_vla as _pause_vla,
    resume_vla as _resume_vla,
    reset_vla as _reset_vla,
    step_zero_actions as _step_zero_actions,
    switch_vla as _switch_vla,
)
from omnigibson.learning.embodiedClaw.tools.data_tools import (
    CheckpointManager,
    save_success_data as _save_success_data,
    save_failure_data as _save_failure_data,
    save_episode_data as _save_episode_data,
)
from omnigibson.learning.embodiedClaw.tools.annotation_tools import (
    load_task_annotations as _load_task_annotations,
    get_task_memory_summary as _get_task_memory_summary,
    advance_subtask as _advance_subtask,
    record_subtask_failure as _record_subtask_failure,
    rebase_retry_start_step as _rebase_retry_start_step,
    get_language_instruction as _get_language_instruction,
)
from omnigibson.learning.embodiedClaw.tools.demo_tools import (
    get_demo_reference_frames as _get_demo_reference_frames,
)
from omnigibson.learning.embodiedClaw.annotation_loader import (
    EpisodeAnnotation,
    SubtaskDurationStats,
)
from omnigibson.learning.embodiedClaw.demo_frame_extractor import (
    DemoFrameLibrary,
)
from omnigibson.learning.embodiedClaw.memory import TaskMemory

logger = logging.getLogger(__name__)

__all__ = ["EmbodiedClawMCPServer"]


class EmbodiedClawMCPServer:
    """MCP server that wraps embodiedClaw tools for agent access.

    The server captures references to live simulator objects at construction
    time.  Each registered MCP tool is a thin closure that binds these
    references and delegates to the corresponding function in ``tools/``.

    Only agent-facing parameters are exposed via MCP; internal objects
    (env, robot, controller, checkpoint_mgr) are bound internally and
    never serialised over the wire.

    Attributes:
        env: The live OmniGibson environment.
        robot: The live robot entity.
        task_name: Current BEHAVIOR task name.
        controller: The :class:`VLAPolicyController` wrapping the VLA policy.
        checkpoint_mgr: Optional :class:`CheckpointManager` for data saving.
    """

    def __init__(
        self,
        env: Any,
        robot: Any,
        task_name: str,
        controller: VLAPolicyController,
        checkpoint_mgr: Optional[CheckpointManager] = None,
        *,
        host: str = "127.0.0.1",
        port: int = 8001,
        dataset_root: str = "/home/user/dataset",
        task_id: int = 0,
        demo_frames_root: str = "/home/user/dataset/demo_frames",
    ) -> None:
        """
        Args:
            env: Live OmniGibson environment.
            robot: Live robot entity.
            task_name: Current task name.
            controller: VLAPolicyController instance.
            checkpoint_mgr: Optional CheckpointManager instance.
            host: Bind address for SSE transport (default ``127.0.0.1``).
            port: Port for SSE transport (default ``8001``).
            dataset_root: Root directory of the BEHAVIOR-1K dataset
                (default ``"/home/user/dataset"``).
            task_id: Numeric task identifier (e.g. 1 for picking_up_trash).
            demo_frames_root: Root directory for pre-extracted demo frames
                (default ``"/home/user/dataset/demo_frames"``).
        """
        self._env = env
        self._robot = robot
        self._task_name = task_name
        self._controller = controller
        self._checkpoint_mgr = checkpoint_mgr
        self._dataset_root = dataset_root
        self._task_id = task_id
        self._demo_frames_root = demo_frames_root

        # The most recent snapshot, cached so that filter_information can
        # operate on it without requiring the agent to pass a blob back.
        self._last_snapshot: Optional[Dict[str, Any]] = None
        self._subtask_start_snapshot: Optional[Dict[str, Any]] = None
        self._subtask_start_snapshot_idx: Optional[int] = None
        self._previous_decision_snapshot: Optional[Dict[str, Any]] = None
        self._reset_predicate_baselines_on_next_snapshot: bool = False
        self._pending_retry_rebase_step: Optional[int] = None

        # Lock to protect snapshot cache from concurrent tool calls.
        self._snapshot_lock = threading.Lock()

        # Decision module state
        self._task_memory: Optional[TaskMemory] = None
        self._all_annotations: Optional[List[EpisodeAnnotation]] = None
        self._duration_stats: Optional[Dict[int, SubtaskDurationStats]] = None

        # Demo frame library (pre-extracted reference frames)
        self._demo_library: Optional[DemoFrameLibrary] = None
        self._load_demo_library()

        self._mcp = FastMCP(
            "embodiedClaw",
            host=host,
            port=port,
        )
        self._register_tools()
        logger.info(
            "EmbodiedClawMCPServer initialised (task=%s, task_id=%d, host=%s, port=%d, dataset_root=%s, demo_library=%s)",
            task_name, task_id, host, port, dataset_root,
            "loaded" if self._demo_library and self._demo_library.num_entries > 0 else "not available",
        )

    # ------------------------------------------------------------------
    # Demo frame library
    # ------------------------------------------------------------------

    def _load_demo_library(self) -> None:
        """Try to load the pre-extracted demo frame library for the current task."""
        if self._task_id <= 0:
            return

        task_dir = os.path.join(
            self._demo_frames_root, f"task-{self._task_id:04d}"
        )
        if os.path.isfile(os.path.join(task_dir, "manifest.json")):
            try:
                self._demo_library = DemoFrameLibrary(task_dir)
                logger.info(
                    "Demo frame library loaded: %d entries for task %d",
                    self._demo_library.num_entries, self._task_id,
                )
            except Exception as e:
                logger.warning(
                    "Failed to load demo frame library from %s: %s",
                    task_dir, e,
                )
                self._demo_library = None
        else:
            logger.info(
                "No demo frame library found for task %d (checked %s)",
                self._task_id, task_dir,
            )

    # ------------------------------------------------------------------
    # Predicate-delta baselines
    # ------------------------------------------------------------------

    def _predicate_context_snapshot(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """Return the minimal snapshot fields needed for predicate diffs."""
        return {
            "world_predicates": snapshot.get("world_predicates", []),
            "step_count": snapshot.get("step_count", 0),
            "task_name": snapshot.get("task_name", self._task_name),
        }

    def _current_subtask_idx(self) -> Optional[int]:
        """Return the current task-memory subtask index, if available."""
        if self._task_memory is None or self._task_memory.episode_done:
            return None
        return self._task_memory.current_subtask_idx

    def _reset_predicate_baselines(self) -> None:
        """Clear predicate baselines so the next snapshot starts fresh."""
        self._subtask_start_snapshot = None
        self._subtask_start_snapshot_idx = None
        self._previous_decision_snapshot = None
        self._reset_predicate_baselines_on_next_snapshot = False

    def _set_predicate_baseline_from_last_snapshot(self) -> None:
        """Use the cached latest snapshot as the baseline for a new subtask."""
        current_idx = self._current_subtask_idx()
        if self._last_snapshot is None or current_idx is None:
            self._reset_predicate_baselines()
            return

        baseline = self._predicate_context_snapshot(self._last_snapshot)
        self._subtask_start_snapshot = baseline
        self._subtask_start_snapshot_idx = current_idx
        self._previous_decision_snapshot = baseline
        self._reset_predicate_baselines_on_next_snapshot = False

    def _snapshot_with_predicate_deltas(
        self,
        snapshot: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Attach predicate deltas and update baselines for the next decision."""
        current_idx = self._current_subtask_idx()
        current_context = self._predicate_context_snapshot(snapshot)

        if current_idx is None:
            enriched = _add_predicate_deltas(
                snapshot,
                subtask_start_snapshot=None,
                previous_decision_snapshot=None,
                current_subtask_idx=None,
            )
            self._reset_predicate_baselines()
            return enriched

        baseline_needs_reset = (
            self._reset_predicate_baselines_on_next_snapshot
            or self._subtask_start_snapshot is None
            or self._subtask_start_snapshot_idx != current_idx
        )
        if baseline_needs_reset:
            self._subtask_start_snapshot = current_context
            self._subtask_start_snapshot_idx = current_idx
            self._previous_decision_snapshot = None
            self._reset_predicate_baselines_on_next_snapshot = False

        enriched = _add_predicate_deltas(
            snapshot,
            subtask_start_snapshot=self._subtask_start_snapshot,
            previous_decision_snapshot=self._previous_decision_snapshot,
            current_subtask_idx=current_idx,
        )
        self._previous_decision_snapshot = current_context
        return enriched

    # ------------------------------------------------------------------
    # Tool registration
    # ------------------------------------------------------------------

    def _register_tools(self) -> None:
        """Register all embodiedClaw tools with the MCP server."""

        # Capture self for closures -- avoids binding issues with the
        # decorator which introspects function signatures.
        server = self

        # ==============================================================
        # Information tools
        # ==============================================================

        @server._mcp.tool(
            name="get_simulation_snapshot",
            description=(
                "Get a comprehensive snapshot of the current simulation state. "
                "Returns JSON with images (base64 PNGs unless "
                "include_images=false), robot_state, "
                "object_states, world_predicates (all boolean predicates for "
                "task objects + robot), name_mapping (BDDL scope name -> "
                "annotation scene name), step_count, task_name, and predicate "
                "deltas since subtask start and previous decision snapshot. "
                "When the simulation is paused, the snapshot includes camera "
                "images captured at pause time if include_images=true."
            ),
        )
        def get_simulation_snapshot(include_images: bool = True):
            """Get a comprehensive snapshot of the current simulation state.

            Args:
                include_images: If False, return ``images: {}`` and avoid
                    exposing cached image payloads to the caller.
            """
            # When paused, the controller has a cached snapshot that includes
            # camera images captured at pause time.  Use it if available.
            cached = server._controller.cached_snapshot
            if cached is not None:
                snapshot = cached
            else:
                # Not paused -- build a snapshot from the latest cached obs
                # so that images are available without pausing the sim loop.
                latest = server._controller.get_latest_snapshot(
                    server._env,
                    server._robot,
                    server._task_name,
                    include_images=include_images,
                )
                if latest is not None:
                    snapshot = latest
                else:
                    # No obs cached yet (sim hasn't stepped) -- fall back
                    # to a live snapshot without images.
                    snapshot = _get_simulation_snapshot(
                        server._env,
                        server._robot,
                        server._task_name,
                        include_images=include_images,
                    )

            if not include_images:
                snapshot = _strip_snapshot_images(snapshot)

            with server._snapshot_lock:
                snapshot = server._snapshot_with_predicate_deltas(snapshot)
                server._last_snapshot = snapshot

            return json.dumps(snapshot)

        @server._mcp.tool(
            name="filter_information",
            description=(
                "Filter the most recent simulation snapshot to include only "
                "information about the specified objects. Pass either exact "
                "annotation/runtime names (e.g. 'trash_can_116') or normalized "
                "type names (e.g. 'trash_can', 'can_of_soda'). Objects are "
                "matched by exact name first, then normalized base-name / "
                "canonical scope-name matching. World predicates are kept if "
                "any argument matches the specified objects or the robot. "
                "Predicate-delta fields are filtered with the same relevance "
                "rule. Robot state and images are included by default; pass "
                "include_images=false for state-only filtering."
            ),
        )
        def filter_information(object_names: List[str], include_images: bool = True):
            """Filter the most recent snapshot to specified objects.

            Args:
                object_names: List of exact or normalized object names to match.
                include_images: If False, return ``images: {}``.
            """
            with server._snapshot_lock:
                snapshot = server._last_snapshot

            if snapshot is None:
                return json.dumps({
                    "error": "No snapshot available. Call get_simulation_snapshot first."
                })

            filtered = _filter_information(
                snapshot,
                object_names,
                include_images=include_images,
            )
            return json.dumps(filtered)

        # ==============================================================
        # Control tools
        # ==============================================================

        @server._mcp.tool(
            name="pause_vla",
            description=(
                "Pause the simulation loop. The sim loop halts entirely -- "
                "it does NOT send zero actions while continuing to step. "
                "A snapshot of the state is cached at pause time. Call "
                "get_simulation_snapshot after pausing to observe the world."
            ),
        )
        def pause_vla():
            """Pause the simulation loop."""
            result = _pause_vla(server._controller)
            return json.dumps(result)

        @server._mcp.tool(
            name="resume_vla",
            description=(
                "Resume the simulation loop after a pause. The VLA policy "
                "continues executing from where it left off."
            ),
        )
        def resume_vla():
            """Resume the simulation loop."""
            result = _resume_vla(server._controller)
            return json.dumps(result)

        @server._mcp.tool(
            name="reset_vla",
            description=(
                "Reset the wrapped VLA policy. Clears any internal state "
                "accumulated during execution. Typically called after a "
                "checkpoint restore to avoid stale policy state."
            ),
        )
        def reset_vla():
            """Reset the wrapped VLA policy."""
            result = _reset_vla(server._controller)
            return json.dumps(result)

        @server._mcp.tool(
            name="step_zero_actions",
            description=(
                "While paused, advance the simulator for a small number of "
                "steps using zero actions instead of VLA actions, then "
                "re-pause with refreshed observations and images. Use this "
                "after a restore before calling get_simulation_snapshot or "
                "resuming the retried subtask."
            ),
        )
        def step_zero_actions(num_steps: int = 10):
            """Advance the paused simulator with zero actions, then re-pause."""
            result = _step_zero_actions(
                controller=server._controller,
                env=server._env,
                robot=server._robot,
                task_name=server._task_name,
                num_steps=num_steps,
            )
            return json.dumps(result)

        @server._mcp.tool(
            name="switch_vla",
            description=(
                "Switch the VLA policy to a new language instruction and "
                "optionally reconnect to a different VLA server. Use this "
                "when transitioning between subtasks that require different "
                "language instructions."
            ),
        )
        def switch_vla(
            language_instruction: str,
            host: Optional[str] = None,
            port: Optional[int] = None,
        ):
            """Switch the VLA to a new language instruction.

            Args:
                language_instruction: The new natural-language instruction.
                host: Optional websocket host to reconnect to.
                port: Optional websocket port to reconnect to.
            """
            result = _switch_vla(
                server._controller,
                language_instruction=language_instruction,
                host=host,
                port=port,
            )
            return json.dumps(result)

        # ==============================================================
        # Data tools
        # ==============================================================

        @server._mcp.tool(
            name="save_success_data",
            description=(
                "Handle subtask success: saves positive data up to the "
                "checkpoint as crash protection and updates the checkpoint "
                "to the current sim state (scene state + buffer index). "
                "Optionally switches the VLA to the next instruction. "
                "Returns checkpoint_step and checkpoint_buffer_idx."
            ),
        )
        def save_success_data(
            subtask_id: str,
            next_subtask_id: Optional[str] = None,
            next_language_instruction: Optional[str] = None,
        ):
            """Update checkpoint on subtask success (no data save yet).

            Args:
                subtask_id: Identifier for the subtask that just succeeded.
                next_subtask_id: Optional identifier for the upcoming subtask.
                next_language_instruction: Optional language instruction for
                    the next subtask.
            """
            if server._checkpoint_mgr is None:
                return json.dumps({
                    "error": "No CheckpointManager configured. Cannot save data."
                })

            result = _save_success_data(
                checkpoint_mgr=server._checkpoint_mgr,
                env=server._env,
                robot=server._robot,
                subtask_id=subtask_id,
                controller=server._controller,
                next_subtask_id=next_subtask_id,
                next_language_instruction=next_language_instruction,
            )
            return json.dumps(result)

        @server._mcp.tool(
            name="save_failure_data",
            description=(
                "Handle subtask failure: extracts the failure segment "
                "(checkpoint -> failure point) from the main recorder, "
                "saves it as a negative sample (parquet + video + HDF5 "
                "+ meta), truncates the main recorder back to the "
                "checkpoint, and reverts sim to the last checkpoint. "
                "If max retries (5) exceeded, flags episode for "
                "termination -- caller should then call "
                "save_episode_data. Returns retry_count, max_retries, "
                "and should_end_episode."
            ),
        )
        def save_failure_data(subtask_id: str):
            """Save failure segment and revert to checkpoint.

            Args:
                subtask_id: Identifier for the subtask that just failed.
            """
            if server._checkpoint_mgr is None:
                return json.dumps({
                    "error": "No CheckpointManager configured. Cannot save data."
                })

            try:
                result = _save_failure_data(
                    checkpoint_mgr=server._checkpoint_mgr,
                    env=server._env,
                    robot=server._robot,
                    subtask_id=subtask_id,
                    controller=server._controller,
                )
            except Exception as exc:
                logger.exception("save_failure_data: delegate crashed")
                result = {"error": f"save_failure_data crashed: {exc}"}
            if (
                result.get("status") == "failure_saved"
                and not result.get("should_end_episode", False)
            ):
                with server._snapshot_lock:
                    server._reset_predicate_baselines_on_next_snapshot = True
                    server._previous_decision_snapshot = None
                if result.get("restore_status") == "restored":
                    server._pending_retry_rebase_step = result.get("checkpoint_step")
                else:
                    server._pending_retry_rebase_step = None
            else:
                server._pending_retry_rebase_step = None
            return json.dumps(result)

        @server._mcp.tool(
            name="save_episode_data",
            description=(
                "Save the accumulated positive trajectory for the episode. "
                "Call when the episode ends: all subtasks done "
                "(success=true, saves entire trajectory) or max retries "
                "exceeded (success=false, saves start -> latest checkpoint "
                "as positive data). Writes parquet + video + HDF5 + meta. "
                "Returns status, total_failure_segments count, and end_idx."
            ),
        )
        def save_episode_data(success: bool):
            """Save positive trajectory and finalize the episode.

            Args:
                success: True if all subtasks completed, False if ending
                    due to max retries exceeded.
            """
            if server._checkpoint_mgr is None:
                return json.dumps({
                    "error": "No CheckpointManager configured. Cannot save data."
                })

            result = _save_episode_data(
                checkpoint_mgr=server._checkpoint_mgr,
                env=server._env,
                robot=server._robot,
                success=success,
            )
            return json.dumps(result)

        # ==============================================================
        # Annotation & Decision tools
        # ==============================================================

        @server._mcp.tool(
            name="load_task_annotations",
            description=(
                "Load annotation files for a task and compute subtask duration "
                "statistics. Must be called once at the start of data collection "
                "for a task. Returns the subtask list and failure thresholds. "
                "Also initialises the task memory for tracking progress."
            ),
        )
        def load_task_annotations(task_id: int):
            """Load annotations and initialise task memory.

            Args:
                task_id: Task number (e.g. 1 for picking_up_trash).
            """
            result = _load_task_annotations(
                task_id=task_id,
                dataset_root=server._dataset_root,
            )
            if result.get("status") == "loaded":
                # Store the annotation objects on the server
                server._all_annotations = result.pop("_annotations", None)
                server._duration_stats = result.pop("_duration_stats", None)
                # Initialise task memory with the first episode annotation
                server.reset_task_memory()
            else:
                # Remove internal keys even on error
                result.pop("_annotations", None)
                result.pop("_duration_stats", None)
            return json.dumps(result)

        @server._mcp.tool(
            name="get_task_memory",
            description=(
                "Get the current task progress: subtask index, subtask_start_step, "
                "retry counts, failure threshold, and recent decisions. To compute "
                "elapsed steps, subtract subtask_start_step from the snapshot's "
                "step_count."
            ),
        )
        def get_task_memory():
            """Return current task memory state."""
            result = _get_task_memory_summary(server._task_memory)
            return json.dumps(result)

        @server._mcp.tool(
            name="advance_to_next_subtask",
            description=(
                "Mark the current subtask as succeeded and advance to the next "
                "one. Returns the new subtask info and suggested VLA language "
                "instruction. Should be called after the agent determines the "
                "subtask has succeeded."
            ),
        )
        def advance_to_next_subtask():
            """Advance to the next subtask after success."""
            with server._snapshot_lock:
                snapshot = server._last_snapshot
            current_step = snapshot.get("step_count", 0) if snapshot else 0
            result = _advance_subtask(server._task_memory, current_step)
            with server._snapshot_lock:
                if result.get("status") == "advanced":
                    server._set_predicate_baseline_from_last_snapshot()
                elif result.get("status") == "episode_complete":
                    server._reset_predicate_baselines()
            return json.dumps(result)

        @server._mcp.tool(
            name="record_failure",
            description=(
                "Mark the current subtask as failed and prepare for retry. "
                "Increments the retry counter. Returns retry count and "
                "whether the episode should end (max retries exceeded)."
            ),
        )
        def record_failure():
            """Record a subtask failure."""
            with server._snapshot_lock:
                snapshot = server._last_snapshot
            current_step = snapshot.get("step_count", 0) if snapshot else 0
            result = _record_subtask_failure(server._task_memory, current_step=current_step)
            did_rebase = False
            if (
                result.get("status") == "failure_recorded"
                and not result.get("should_end_episode", False)
            ):
                with server._snapshot_lock:
                    server._reset_predicate_baselines_on_next_snapshot = True
                    server._previous_decision_snapshot = None
                if server._pending_retry_rebase_step is not None:
                    did_rebase = _rebase_retry_start_step(
                        server._task_memory,
                        checkpoint_step=server._pending_retry_rebase_step,
                        restore_status="restored",
                    )
                    result["memory_summary"] = _get_task_memory_summary(server._task_memory)
            server._pending_retry_rebase_step = None
            if did_rebase:
                result["subtask_start_step_rebased"] = True
            return json.dumps(result)

        @server._mcp.tool(
            name="get_subtask_language_instruction",
            description=(
                "Get the VLA language instruction for a specific subtask "
                "index. If no skill_idx is provided, returns the instruction "
                "for the current subtask."
            ),
        )
        def get_subtask_language_instruction(skill_idx: int = -1):
            """Get VLA language instruction for a subtask.

            Args:
                skill_idx: Skill index to look up. Use -1 for the current
                    subtask.
            """
            idx = skill_idx if skill_idx >= 0 else None
            result = _get_language_instruction(server._task_memory, idx)
            return json.dumps(result)

        # ==============================================================
        # Demo comparison tools
        # ==============================================================

        @server._mcp.tool(
            name="get_demo_reference_frames",
            description=(
                "Get reference frames from demo videos for the specified subtask. "
                "Returns base64-encoded JPEG images showing what the scene looks "
                "like at various points during the subtask in the demo data. Use "
                "these to visually compare with the current simulation snapshot "
                "and judge whether the subtask is progressing correctly or is "
                "completed. By default returns only the head camera completion "
                "frame from 1 episode. Pass frame_types and max_episodes "
                "explicitly to request more reference frames. Each frame includes "
                "metadata: frame_type (completion, pre_completion, "
                "intermediate_1, intermediate_2), "
                "relative_position (0.0-1.0 through subtask), and the "
                "episode_id it came from."
            ),
        )
        def get_demo_reference_frames(
            skill_idx: int,
            camera: str = "head",
            episode_id: Optional[str] = None,
            frame_types: Optional[List[str]] = None,
            max_episodes: int = 1,
        ):
            """Get demo reference frames for visual comparison.

            Args:
                skill_idx: Subtask index to get reference frames for.
                camera: Camera name ("head", "left_wrist", "right_wrist").
                    Default "head".
                episode_id: Optional specific episode ID. If None, returns
                    frames from up to max_episodes episodes.
                frame_types: Optional list of frame types to include.
                    Default returns only completion frames.
                max_episodes: Maximum number of episodes to return frames
                    from. Default 1.
            """
            result = _get_demo_reference_frames(
                library=server._demo_library,
                skill_idx=skill_idx,
                camera=camera,
                episode_id=episode_id,
                frame_types=frame_types,
                max_episodes=max_episodes,
            )
            return json.dumps(result)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def checkpoint_mgr(self) -> Optional[CheckpointManager]:
        """The active :class:`CheckpointManager` (may be ``None``)."""
        return self._checkpoint_mgr

    @checkpoint_mgr.setter
    def checkpoint_mgr(self, value: Optional[CheckpointManager]) -> None:
        """Update the checkpoint manager (typically once per episode)."""
        self._checkpoint_mgr = value

    @property
    def task_memory(self) -> Optional[TaskMemory]:
        """The active :class:`TaskMemory` (may be ``None``)."""
        return self._task_memory

    @task_memory.setter
    def task_memory(self, value: Optional[TaskMemory]) -> None:
        self._task_memory = value
        with self._snapshot_lock:
            self._reset_predicate_baselines()

    @property
    def mcp(self) -> FastMCP:
        """The underlying FastMCP server instance."""
        return self._mcp

    @property
    def last_snapshot(self) -> Optional[Dict[str, Any]]:
        """The most recently captured simulation snapshot (if any)."""
        with self._snapshot_lock:
            return self._last_snapshot

    # ------------------------------------------------------------------
    # Task memory lifecycle
    # ------------------------------------------------------------------

    def reset_task_memory(self) -> None:
        """Clear the task memory for a new episode.

        If annotations have been loaded, creates a fresh
        :class:`TaskMemory` using the first episode's annotation.
        """
        if self._all_annotations and self._duration_stats is not None:
            first_anno = self._all_annotations[0]
            self._task_memory = TaskMemory(
                task_name=self._task_name,
                episode_annotation=first_anno,
                duration_stats=self._duration_stats,
            )
            with self._snapshot_lock:
                self._reset_predicate_baselines()
            self._pending_retry_rebase_step = None
            logger.info("TaskMemory reset for new episode (task=%s)", self._task_name)
        else:
            self._task_memory = None
            with self._snapshot_lock:
                self._reset_predicate_baselines()
            self._pending_retry_rebase_step = None
            logger.info("TaskMemory cleared (no annotations loaded yet)")

    # ------------------------------------------------------------------
    # Running the server
    # ------------------------------------------------------------------

    def run(self, transport: str = "stdio") -> None:
        """Start the MCP server (blocking).

        Args:
            transport: Transport protocol -- ``"stdio"`` (default) or
                ``"sse"`` for HTTP+SSE.
        """
        logger.info("Starting MCP server with transport=%s", transport)
        self._mcp.run(transport=transport)

    def run_sse(self) -> None:
        """Start the MCP server using SSE transport (blocking).

        Convenience wrapper that manually creates the uvicorn server with
        ``ws="none"`` to avoid importing websocket protocols, which fail
        inside Isaac Sim's bundled Python environment.

        Intended to be called from a background thread::

            import threading
            t = threading.Thread(target=server.run_sse, daemon=True)
            t.start()
        """
        import asyncio
        import uvicorn

        starlette_app = self._mcp.sse_app()
        config = uvicorn.Config(
            starlette_app,
            host=self._mcp.settings.host,
            port=self._mcp.settings.port,
            log_level=self._mcp.settings.log_level.lower(),
            ws="none",  # Disable websocket protocol to avoid Isaac Sim import conflict
        )
        server = uvicorn.Server(config)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(server.serve())
        finally:
            loop.close()
