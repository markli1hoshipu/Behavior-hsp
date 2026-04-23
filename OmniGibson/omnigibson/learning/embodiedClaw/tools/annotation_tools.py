"""
MCP-facing wrappers for annotation loading, memory queries, and decision logic.

These functions return JSON-serializable dicts and are called by the MCP
tool handlers in ``mcp_server.py``.  They bridge the pure-Python modules
(``annotation_loader``, ``memory``, ``decision_module``) with the MCP layer.

No simulator dependencies -- all functions operate on pre-loaded data structures.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from omnigibson.learning.embodiedClaw.annotation_loader import (
    EpisodeAnnotation,
    SubtaskAnnotation,
    SubtaskDurationStats,
    compute_subtask_duration_stats,
    generate_language_instruction,
    get_rollout_subtask_object_refs,
    load_all_annotations,
)
from omnigibson.learning.embodiedClaw.decision_module import (
    evaluate_subtask_status,
)
from omnigibson.learning.embodiedClaw.memory import TaskMemory

logger = logging.getLogger(__name__)

__all__ = [
    "load_task_annotations",
    "get_task_memory_summary",
    "evaluate_current_subtask",
    "advance_subtask",
    "record_subtask_failure",
    "rebase_retry_start_step",
    "get_language_instruction",
]


def load_task_annotations(
    task_id: int,
    dataset_root: str = "/home/user/dataset",
) -> Dict[str, Any]:
    """Load all annotations for a task and compute duration stats.

    Args:
        task_id: Task number (e.g. 1 for ``picking_up_trash``).
        dataset_root: Root of the dataset directory.

    Returns:
        Dict with:
            - ``"task_name"``: str
            - ``"num_episodes"``: int
            - ``"subtask_list"``: list of subtask summary dicts (from first episode)
            - ``"duration_stats"``: dict of per-``skill_idx`` stats
            - ``"annotations"``: list of :class:`EpisodeAnnotation` (not returned;
              stored internally)
            - ``"status"``: ``"loaded"``
    """
    task_dir = os.path.join(
        dataset_root, "annotations", f"task-{task_id:04d}"
    )
    annotations = load_all_annotations(task_dir)

    if not annotations:
        return {
            "status": "error",
            "error": f"No annotations found in {task_dir}",
        }

    # Compute duration stats across all episodes
    duration_stats = compute_subtask_duration_stats(annotations)

    # Use the first episode to build the canonical subtask list
    first_anno = annotations[0]
    subtask_list = []
    for subtask in first_anno.subtasks:
        stats = duration_stats.get(subtask.skill_idx)
        subtask_list.append({
            "skill_idx": subtask.skill_idx,
            "skill_description": subtask.skill_description,
            "skill_type": subtask.skill_type,
            "frame_duration": subtask.frame_duration,
            "failure_threshold": stats.failure_threshold if stats else None,
            **get_rollout_subtask_object_refs(subtask),
        })

    # Serialise duration stats
    stats_dict: Dict[str, Any] = {}
    for idx, stats in sorted(duration_stats.items()):
        stats_dict[str(idx)] = {
            "skill_idx": stats.skill_idx,
            "skill_description": stats.skill_description,
            "min_duration": stats.min_duration,
            "max_duration": stats.max_duration,
            "mean_duration": stats.mean_duration,
            "count": stats.count,
            "failure_threshold": stats.failure_threshold,
        }

    result: Dict[str, Any] = {
        "status": "loaded",
        "task_name": first_anno.task_name,
        "num_episodes": len(annotations),
        "subtask_list": subtask_list,
        "duration_stats": stats_dict,
    }

    # Store the raw objects so callers (MCP server) can access them
    result["_annotations"] = annotations
    result["_duration_stats"] = duration_stats

    return result


def get_task_memory_summary(memory: TaskMemory) -> Dict[str, Any]:
    """Return the current task memory state as a JSON-serializable dict.

    Args:
        memory: The active :class:`TaskMemory` instance.

    Returns:
        The dict produced by ``memory.to_dict()``, or an error dict if
        memory is ``None``.
    """
    if memory is None:
        return {"error": "Task memory not initialised. Call load_task_annotations first."}
    return memory.to_dict()


def evaluate_current_subtask(
    snapshot: Dict[str, Any],
    memory: TaskMemory,
    demo_reference_frames: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the decision logic on the current snapshot and memory.

    Args:
        snapshot: Full simulation snapshot from ``get_simulation_snapshot``.
        memory: The active :class:`TaskMemory` instance.
        demo_reference_frames: Optional dict returned by
            :func:`~omnigibson.learning.embodiedClaw.tools.demo_tools.get_demo_reference_frames`.
            When provided, enables demo-comparison evidence in the result.

    Returns:
        The evaluation dict from :func:`evaluate_subtask_status`, or an
        error dict if inputs are missing.
    """
    if memory is None:
        return {"error": "Task memory not initialised. Call load_task_annotations first."}
    if snapshot is None:
        return {"error": "No snapshot available. Call get_simulation_snapshot first."}

    current_step = snapshot.get("step_count", 0)
    return evaluate_subtask_status(
        snapshot, memory, current_step,
        demo_reference_frames=demo_reference_frames,
    )


def advance_subtask(
    memory: TaskMemory,
    current_step: int,
) -> Dict[str, Any]:
    """Mark the current subtask as succeeded and advance to the next one.

    Args:
        memory: The active :class:`TaskMemory` instance.
        current_step: The simulation step at which the subtask succeeded.

    Returns:
        Dict with the new subtask info, suggested language instruction,
        and updated memory summary.
    """
    if memory is None:
        return {"error": "Task memory not initialised."}

    # Record the success decision
    subtask = memory.get_current_subtask()
    if subtask is not None:
        memory.record_decision(
            decision="success",
            step=current_step,
            reason=f"Subtask {subtask.skill_idx} ('{subtask.skill_description}') succeeded.",
        )

    # Advance to next subtask
    next_subtask = memory.advance_subtask(current_step)

    if next_subtask is None:
        return {
            "status": "episode_complete",
            "all_subtasks_done": True,
            "memory_summary": memory.to_dict(),
        }

    # Generate language instruction for the new subtask
    instruction = generate_language_instruction(next_subtask)

    return {
        "status": "advanced",
        "new_subtask": {
            "skill_idx": next_subtask.skill_idx,
            "skill_description": next_subtask.skill_description,
            "skill_type": next_subtask.skill_type,
            "frame_duration": next_subtask.frame_duration,
            **get_rollout_subtask_object_refs(next_subtask),
        },
        "language_instruction": instruction,
        "failure_threshold": memory.get_failure_threshold(),
        "memory_summary": memory.to_dict(),
    }


def record_subtask_failure(
    memory: TaskMemory,
    current_step: int = 0,
) -> Dict[str, Any]:
    """Mark the current subtask as failed and prepare for retry.

    Args:
        memory: The active :class:`TaskMemory` instance.
        current_step: The simulation step at which the failure was detected.
            Defaults to 0 if not available.

    Returns:
        Dict with retry count, whether the episode should end, and
        updated memory summary.
    """
    if memory is None:
        return {"error": "Task memory not initialised."}

    subtask = memory.get_current_subtask()
    subtask_desc = (
        f"{subtask.skill_idx} ('{subtask.skill_description}')"
        if subtask is not None
        else "unknown"
    )

    # Record the failure decision
    memory.record_decision(
        decision="failure",
        step=current_step,
        reason=f"Subtask {subtask_desc} failed.",
    )

    # Increment retry count
    retry_count = memory.record_retry()
    max_retries = 5  # CheckpointManager.MAX_RETRIES_PER_SUBTASK

    should_end = retry_count >= max_retries
    if should_end:
        memory.episode_done = True
        memory.episode_success = False

    # The retry timer is rebased by the MCP save_failure_data wrapper after a
    # successful checkpoint restore. Do not mutate subtask_start_step here,
    # because this helper runs before the restore outcome is known.

    return {
        "status": "failure_recorded",
        "subtask_idx": memory.current_subtask_idx,
        "retry_count": retry_count,
        "max_retries": max_retries,
        "should_end_episode": should_end,
        "memory_summary": memory.to_dict(),
    }


def rebase_retry_start_step(
    memory: Optional[TaskMemory],
    checkpoint_step: Optional[int],
    restore_status: Optional[str],
) -> bool:
    """Rebase retry timing after a successful restore to checkpoint."""
    if memory is None or checkpoint_step is None:
        return False

    if restore_status != "restored":
        return False

    memory.subtask_start_step = int(checkpoint_step)
    return True


def get_language_instruction(
    memory: TaskMemory,
    skill_idx: Optional[int] = None,
) -> Dict[str, Any]:
    """Get the VLA language instruction for a subtask.

    If ``skill_idx`` is not provided, returns the instruction for the
    current subtask.

    Args:
        memory: The active :class:`TaskMemory` instance.
        skill_idx: Optional skill index to look up. Defaults to the
            current subtask.

    Returns:
        Dict with ``"instruction"`` (str) and subtask info.
    """
    if memory is None:
        return {"error": "Task memory not initialised."}

    if skill_idx is not None:
        subtask = memory.get_subtask_by_idx(skill_idx)
    else:
        subtask = memory.get_current_subtask()

    if subtask is None:
        return {
            "error": f"No subtask found for skill_idx={skill_idx}.",
        }

    instruction = generate_language_instruction(subtask)
    return {
        "instruction": instruction,
        "skill_idx": subtask.skill_idx,
        "skill_description": subtask.skill_description,
        **get_rollout_subtask_object_refs(subtask),
    }
