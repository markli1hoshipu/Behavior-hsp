"""
Data management tools for the agentic data collection system.

Provides subtask-level checkpointing and success/failure saving during
data collection episodes.  The ``CheckpointManager`` tracks scene state
snapshots so that the simulator can be rolled back after a failed subtask
attempt.

Saving logic:
- **Subtask success:** update checkpoint (record current sim state as
  restore point), do NOT save trajectory yet -- the positive segment
  keeps growing.
- **Subtask failure:** save checkpoint -> failure point as a negative
  sample (parquet + video + HDF5 + meta), revert sim to checkpoint,
  truncate the main recorder back to the checkpoint buffer index,
  increment retry counter.
- **5 consecutive failures (same subtask):** save start -> latest
  checkpoint as positive data (parquet + video + HDF5 + meta), all
  failure segments already saved individually, end episode.
- **Episode complete (all subtasks done):** save entire trajectory
  (start -> end) as positive data (parquet + video + HDF5 + meta).

Failure segments are extracted directly from the main recorder's
accumulated buffers via :meth:`DataRecorder.save_segment`, then the
buffers are truncated back to the checkpoint via
:meth:`DataRecorder.truncate_to`.  No separate failure recorders are
needed.

The "agent" is Claude Code (a separate process); the simulation runs in
its own process.  These tools bridge the two.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import omnigibson as og
from omnigibson.learning.embodiedClaw.data_recording.bddl_state_tracker import BDDLStateTracker
from omnigibson.learning.embodiedClaw.data_recording.data_saver import DataRecorder

logger = logging.getLogger(__name__)

# Number of physics steps used to stabilise the scene after restoring a
# checkpoint (matches the pattern in eval.py / eval_data_gen_par.py).
_STABILISATION_STEPS = 25


class CheckpointManager:
    """Manages subtask-level checkpointing during data collection episodes.

    A checkpoint captures the serialized simulator state at the moment a
    subtask succeeds.  If a subsequent subtask fails, the scene can be
    rolled back to that checkpoint so the agent can retry.  Each subtask
    is allowed up to ``MAX_RETRIES_PER_SUBTASK`` attempts before the
    episode is abandoned.

    The manager tracks a ``_checkpoint_buffer_idx`` -- the index into the
    main :class:`DataRecorder`'s accumulated buffers that corresponds to
    the checkpoint.  On failure, the segment from
    ``_checkpoint_buffer_idx`` to the current buffer length is saved as a
    negative sample, then the buffers are truncated.

    Attributes:
        MAX_RETRIES_PER_SUBTASK: Maximum retry attempts per subtask
            before the episode is ended.
        output_folder: Root output directory for recorded artefacts.
        task_name: Human-readable BEHAVIOR task name.
        task_id: Numeric task identifier.
        demo_id: Numeric demonstration identifier.
        data_recorder: Active :class:`DataRecorder` for the positive
            trajectory (runs from episode start to end).
        bddl_tracker: :class:`BDDLStateTracker` for the current episode.
    """

    MAX_RETRIES_PER_SUBTASK: int = 5

    def __init__(
        self,
        output_folder: str,
        task_name: str,
        task_id: int,
        demo_id: int,
        record_rgb: bool = True,
        record_depth: bool = True,
    ) -> None:
        self.output_folder = output_folder
        self.task_name = task_name
        self.task_id = task_id
        self.demo_id = demo_id
        self.record_rgb = record_rgb
        self.record_depth = record_depth

        # Checkpoint state
        self._checkpoint_step: int = 0
        self._checkpoint_buffer_idx: int = 0  # index into main recorder buffers
        self._checkpoint_scene_state: Optional[Dict[str, Any]] = None
        self._checkpoint_hdf5_state: Optional[Any] = None  # serialized state vector
        self._retry_counts: Dict[str, int] = {}
        self._failure_segment_id: int = 0
        self._should_end_episode: bool = False

        # References (set externally via setup)
        self.data_recorder: Optional[DataRecorder] = None
        self.bddl_tracker: Optional[BDDLStateTracker] = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, data_recorder: DataRecorder, bddl_tracker: BDDLStateTracker) -> None:
        """Store references to the active data recorder and BDDL tracker.

        Args:
            data_recorder: The :class:`DataRecorder` for the positive
                trajectory (runs for the entire episode).
            bddl_tracker: The :class:`BDDLStateTracker` for this episode.
        """
        self.data_recorder = data_recorder
        self.bddl_tracker = bddl_tracker
        logger.info(
            "CheckpointManager: setup complete (task=%s, task_id=%d, demo_id=%d)",
            self.task_name,
            self.task_id,
            self.demo_id,
        )

    # ------------------------------------------------------------------
    # Checkpoint capture
    # ------------------------------------------------------------------

    def capture_checkpoint(self, env: Any, step_idx: int) -> None:
        """Capture a scene-state checkpoint at the given step index.

        Stores both the dict-based scene state (for ``load_state``) and
        the serialized HDF5 state vector (for data recording), plus the
        current buffer index in the main recorder.

        Args:
            env: The OmniGibson environment instance.
            step_idx: The simulation step index to associate with this
                checkpoint.
        """
        self._checkpoint_step = step_idx
        self._checkpoint_scene_state = env.scene.dump_state(serialized=False)
        self._checkpoint_hdf5_state = og.sim.dump_state(serialized=True)
        # Record the current buffer length as the checkpoint position.
        if self.data_recorder is not None:
            self._checkpoint_buffer_idx = len(self.data_recorder._step_data)
        logger.info(
            "CheckpointManager: captured checkpoint at step %d (buffer_idx=%d)",
            step_idx, self._checkpoint_buffer_idx,
        )

    # ------------------------------------------------------------------
    # Retry bookkeeping
    # ------------------------------------------------------------------

    def get_retry_count(self, subtask_id: str) -> int:
        """Return the current retry count for a given subtask.

        Args:
            subtask_id: Identifier for the subtask.

        Returns:
            Number of retries recorded so far (0 if none).
        """
        return self._retry_counts.get(subtask_id, 0)

    @property
    def should_end_episode(self) -> bool:
        """``True`` if any subtask has exceeded the maximum retry limit."""
        return self._should_end_episode


# ======================================================================
# Segment output-folder helpers
# ======================================================================

def _failure_output_folder(checkpoint_mgr: CheckpointManager) -> str:
    """Build an output folder path for a failure segment."""
    return (
        f"{checkpoint_mgr.output_folder}"
        f"/failure_{checkpoint_mgr._failure_segment_id:04d}"
    )


# ======================================================================
# Stabilisation helper
# ======================================================================

def _stabilise_scene(env: Any) -> None:
    """Run physics steps with ``keep_still`` to let objects settle.

    This follows the established pattern from ``eval.py`` (lines 284-288)
    where 25 physics steps are executed while calling ``keep_still`` on
    every task-relevant entity to dampen jitter after a state restore.
    """
    for _ in range(_STABILISATION_STEPS):
        og.sim.step_physics()
        for entity in env.task.object_scope.values():
            if not entity.is_system and entity.exists:
                entity.keep_still()


# ======================================================================
# Tool functions
# ======================================================================

def save_success_data(
    checkpoint_mgr: CheckpointManager,
    env: Any,
    robot: Any,
    subtask_id: str,
    controller: Optional[Any] = None,
    next_subtask_id: Optional[str] = None,
    next_language_instruction: Optional[str] = None,
) -> Dict[str, Any]:
    """Handle subtask success: save positive data up to checkpoint, then update checkpoint.

    Saves the accumulated positive trajectory to disk as crash protection,
    then updates the checkpoint to the current sim step.

    Steps performed:

    1. Update the checkpoint to the current simulation step (captures
       both scene state for revert, HDF5 state for recording, and the
       buffer index in the main recorder).
    2. Reset the retry counter for the completed subtask.
    3. Optionally switch the VLA policy to the next language instruction.

    Args:
        checkpoint_mgr: The :class:`CheckpointManager` for this episode.
        env: The OmniGibson environment.
        robot: The robot entity in the scene.
        subtask_id: Identifier for the subtask that just succeeded.
        controller: Optional :class:`VLAPolicyController`.  When provided
            together with *next_language_instruction*, the VLA is switched
            to the new instruction via :func:`switch_vla`.
        next_subtask_id: Optional identifier for the upcoming subtask
            (informational; included in the returned status dict).
        next_language_instruction: Optional language instruction for the
            next subtask.  Requires *controller* to take effect.

    Returns:
        JSON-serializable status dict.
    """
    # 1. Update checkpoint to the current step.
    current_step: int = getattr(env, "_current_step", 0)
    checkpoint_mgr.capture_checkpoint(env, current_step)

    # 2. Save positive data up to checkpoint (crash protection).
    #    Write to temp dir first, then atomic swap to latest_checkpoint.
    #    This ensures the old checkpoint stays intact if the write fails.
    recorder = checkpoint_mgr.data_recorder
    if recorder is not None and checkpoint_mgr._checkpoint_buffer_idx > 0:
        import shutil

        success_base = os.path.join(checkpoint_mgr.output_folder, "success")
        tmp_dir = os.path.join(success_base, f".tmp_checkpoint_{subtask_id}")
        final_dir = os.path.join(success_base, "latest_checkpoint")

        # Clean any leftover temp dir from a previous failed attempt
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

        # Write to temp dir
        recorder.save_segment(
            output_folder=tmp_dir,
            start_idx=0,
            end_idx=checkpoint_mgr._checkpoint_buffer_idx,
            success=True,
            segment_tag=f"positive_checkpoint_{subtask_id}",
        )

        # Atomic swap: remove old checkpoint, rename temp to final
        if os.path.exists(final_dir):
            shutil.rmtree(final_dir)
        os.rename(tmp_dir, final_dir)

        logger.info("save_success_data: saved positive segment [0, %d) to %s",
                     checkpoint_mgr._checkpoint_buffer_idx, final_dir)

    # 3. Clear retry counter for this subtask (it succeeded).
    checkpoint_mgr._retry_counts.pop(subtask_id, None)

    # 4. Optionally switch the VLA policy for the next subtask.
    if controller is not None and next_language_instruction is not None:
        from omnigibson.learning.embodiedClaw.tools.control_tools import switch_vla

        switch_result = switch_vla(
            controller=controller,
            language_instruction=next_language_instruction,
        )
        logger.info(
            "save_success_data: switched VLA -> %s",
            switch_result,
        )

    result: Dict[str, Any] = {
        "status": "checkpoint_updated",
        "subtask_id": subtask_id,
        "next_subtask_id": next_subtask_id,
        "checkpoint_step": checkpoint_mgr._checkpoint_step,
        "checkpoint_buffer_idx": checkpoint_mgr._checkpoint_buffer_idx,
    }
    logger.info("save_success_data: %s", result)
    return result


def save_failure_data(
    checkpoint_mgr: CheckpointManager,
    env: Any,
    robot: Any,
    subtask_id: str,
    controller: Optional[Any] = None,
) -> Dict[str, Any]:
    """Handle subtask failure: save failure segment, revert to checkpoint.

    The failure segment is extracted directly from the main recorder's
    accumulated buffers (from ``_checkpoint_buffer_idx`` to the current
    buffer end), saved to a separate output folder, then the main
    recorder's buffers are truncated back to the checkpoint so the
    positive trajectory remains clean.

    Steps performed:

    1. Extract the segment ``[checkpoint_buffer_idx, current_end)`` from
       the main recorder and save it as a negative sample (parquet +
       video + HDF5 + meta).
    2. Truncate the main recorder back to ``checkpoint_buffer_idx``.
    3. Increment the retry counter for the subtask.
    4. If the retry limit is reached, flag the episode for termination
       (the caller should then invoke :func:`save_episode_data`).
    5. Otherwise, restore the scene to the last checkpoint and stabilise.

    Args:
        checkpoint_mgr: The :class:`CheckpointManager` for this episode.
        env: The OmniGibson environment.
        robot: The robot entity in the scene.
        subtask_id: Identifier for the subtask that just failed.
        controller: Optional :class:`VLAPolicyController`.  When
            provided, the VLA policy is reset after checkpoint restore.

    Returns:
        JSON-serializable status dict.
    """
    bddl_results = (
        checkpoint_mgr.bddl_tracker.get_results()
        if checkpoint_mgr.bddl_tracker is not None
        else None
    )

    recorder = checkpoint_mgr.data_recorder
    start_idx = checkpoint_mgr._checkpoint_buffer_idx
    end_idx = len(recorder._step_data) if recorder is not None else 0

    # 1. Save the failure segment from the main recorder's buffers.
    if recorder is not None and end_idx > start_idx:
        failure_folder = _failure_output_folder(checkpoint_mgr)
        segment_tag = f"failure_{checkpoint_mgr._failure_segment_id:04d}"
        recorder.save_segment(
            output_folder=failure_folder,
            start_idx=start_idx,
            end_idx=end_idx,
            success=False,
            segment_tag=segment_tag,
            bddl_transitions=bddl_results,
        )
        logger.info(
            "save_failure_data: saved failure segment %d [%d:%d) for subtask '%s'",
            checkpoint_mgr._failure_segment_id, start_idx, end_idx, subtask_id,
        )

    # 2. Truncate the main recorder back to the checkpoint.
    if recorder is not None:
        recorder.truncate_to(start_idx)

    checkpoint_mgr._failure_segment_id += 1

    # 3. Increment the retry counter.
    retry_count = checkpoint_mgr._retry_counts.get(subtask_id, 0) + 1
    checkpoint_mgr._retry_counts[subtask_id] = retry_count
    logger.info(
        "save_failure_data: subtask '%s' retry count = %d / %d",
        subtask_id,
        retry_count,
        CheckpointManager.MAX_RETRIES_PER_SUBTASK,
    )

    # 4. Check whether the episode should be terminated.
    if retry_count >= CheckpointManager.MAX_RETRIES_PER_SUBTASK:
        checkpoint_mgr._should_end_episode = True
        logger.warning(
            "save_failure_data: subtask '%s' exceeded max retries -- marking episode for termination",
            subtask_id,
        )

    # 5. Restore scene to the last checkpoint (if not ending the episode
    #    and there were actually steps to revert).
    has_steps_to_revert = end_idx > start_idx
    restore_status = "skipped_max_retries" if checkpoint_mgr._should_end_episode else "pending"
    if not checkpoint_mgr._should_end_episode and has_steps_to_revert:
        if checkpoint_mgr._checkpoint_scene_state is not None:
            if controller is not None and hasattr(controller, 'request_restore'):
                # Use main-thread handoff — Isaac Sim physics ops must
                # run on the main thread, not the MCP server thread.
                logger.info(
                    "save_failure_data: requesting main-thread restore to step %d",
                    checkpoint_mgr._checkpoint_step,
                )
                restore_result = controller.request_restore(
                    env=env,
                    scene_state=checkpoint_mgr._checkpoint_scene_state,
                    step_idx=checkpoint_mgr._checkpoint_step,
                )
                if restore_result.get("status") == "error":
                    restore_status = "restore_error"
                    logger.error(
                        "save_failure_data: main-thread restore failed: %s",
                        restore_result.get("error"),
                    )
                else:
                    restore_status = "restored"
                    logger.info("save_failure_data: scene restored via main-thread handoff")
            else:
                # Fallback: direct restore (only safe when called from main thread)
                logger.info(
                    "save_failure_data: restoring scene to checkpoint at step %d (direct)",
                    checkpoint_mgr._checkpoint_step,
                )
                env.scene.load_state(
                    checkpoint_mgr._checkpoint_scene_state,
                    serialized=False,
                )
                _stabilise_scene(env)
                if hasattr(env, "_current_step"):
                    env._current_step = int(checkpoint_mgr._checkpoint_step)
                if controller is not None:
                    from omnigibson.learning.embodiedClaw.tools.control_tools import reset_vla
                    reset_vla(controller)
                restore_status = "restored"
                logger.info("save_failure_data: scene stabilised after checkpoint restore")
        else:
            restore_status = "skipped_no_checkpoint"
            logger.warning(
                "save_failure_data: no checkpoint scene state available -- cannot restore"
            )
    elif not has_steps_to_revert:
        restore_status = "no_steps_to_revert"
        logger.info(
            "save_failure_data: no steps since checkpoint -- skipping scene restore"
        )

    result: Dict[str, Any] = {
        "status": "failure_saved",
        "subtask_id": subtask_id,
        "retry_count": retry_count,
        "max_retries": CheckpointManager.MAX_RETRIES_PER_SUBTASK,
        "should_end_episode": checkpoint_mgr._should_end_episode,
        "failure_segment_id": checkpoint_mgr._failure_segment_id,
        "checkpoint_step": checkpoint_mgr._checkpoint_step,
        "checkpoint_buffer_idx": checkpoint_mgr._checkpoint_buffer_idx,
        "restore_status": restore_status,
    }
    logger.info("save_failure_data: %s", result)
    return result


def save_episode_data(
    checkpoint_mgr: CheckpointManager,
    env: Any,
    robot: Any,
    success: bool,
) -> Dict[str, Any]:
    """Save the accumulated positive trajectory data for the episode.

    Called when the episode ends, either because all subtasks completed
    (``success=True``) or because a subtask exceeded max retries
    (``success=False``; saves start -> latest checkpoint as positive).

    When ``success=False`` (max retries exceeded), only the data from
    the episode start up to the latest checkpoint is saved as positive
    data.  The failed attempt beyond the checkpoint has already been
    saved as a separate failure segment by :func:`save_failure_data`.

    Steps performed:

    1. Determine the end index for the positive trajectory:
       - ``success=True``: save all accumulated data.
       - ``success=False``: save only up to ``_checkpoint_buffer_idx``.
    2. End the positive (main) data recorder, flushing parquet + video +
       HDF5 + metadata to disk.

    Args:
        checkpoint_mgr: The :class:`CheckpointManager` for this episode.
        env: The OmniGibson environment.
        robot: The robot entity in the scene.
        success: Whether the episode completed all subtasks.

    Returns:
        JSON-serializable status dict.
    """
    bddl_results = (
        checkpoint_mgr.bddl_tracker.get_results()
        if checkpoint_mgr.bddl_tracker is not None
        else None
    )

    saved = False

    # Determine end index: all data on success, up-to-checkpoint on failure.
    end_idx: Optional[int] = None
    if not success and checkpoint_mgr._checkpoint_buffer_idx > 0:
        end_idx = checkpoint_mgr._checkpoint_buffer_idx

    # 1. End the main (positive) recorder.
    if checkpoint_mgr.data_recorder is not None and checkpoint_mgr.data_recorder.is_recording:
        n_steps = end_idx if end_idx is not None else len(checkpoint_mgr.data_recorder._step_data)
        checkpoint_mgr.data_recorder.end_episode(
            success=success,
            env=env,
            bddl_transitions=bddl_results,
            end_idx=end_idx,
        )
        saved = True
        logger.info(
            "save_episode_data: saved positive trajectory (success=%s, steps=%d)",
            success, n_steps,
        )

    result: Dict[str, Any] = {
        "status": "episode_saved" if saved else "no_data_to_save",
        "success": success,
        "checkpoint_step": checkpoint_mgr._checkpoint_step,
        "checkpoint_buffer_idx": checkpoint_mgr._checkpoint_buffer_idx,
        "end_idx": end_idx,
        "total_failure_segments": checkpoint_mgr._failure_segment_id,
    }
    logger.info("save_episode_data: %s", result)
    return result
