"""
In-memory task progress tracker for the embodiedClaw decision module.

Tracks which subtask the agent is working on, how many steps have elapsed,
retry counts, and a chronological decision history.  The ``TaskMemory``
object lives inside the MCP server process and is read/written by the
agent through MCP tools.

This module has no simulator dependencies.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from omnigibson.learning.embodiedClaw.annotation_loader import (
    EpisodeAnnotation,
    SubtaskAnnotation,
    SubtaskDurationStats,
    get_rollout_subtask_object_refs,
)

__all__ = ["SubtaskProgress", "TaskMemory"]


@dataclass
class SubtaskProgress:
    """Progress tracking for a single subtask attempt."""

    skill_idx: int
    skill_description: str
    started_at_step: int
    elapsed_steps: int = 0
    retry_count: int = 0
    max_retries: int = 5  # from CheckpointManager.MAX_RETRIES_PER_SUBTASK
    failure_threshold_steps: int = 0  # 2 * max_duration from annotations
    status: str = "in_progress"  # "in_progress" | "succeeded" | "failed" | "skipped"

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable summary."""
        return {
            "skill_idx": self.skill_idx,
            "skill_description": self.skill_description,
            "started_at_step": self.started_at_step,
            "elapsed_steps": self.elapsed_steps,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "failure_threshold_steps": self.failure_threshold_steps,
            "status": self.status,
        }


class TaskMemory:
    """Tracks task-level progress and decision history.

    This is an in-memory object that lives inside the MCP server process.
    The agent reads/updates it through MCP tools.

    Attributes:
        task_name: The name of the current task.
        annotation: The parsed episode annotation.
        duration_stats: Cross-episode subtask duration statistics.
        current_subtask_idx: Index of the current subtask.
        subtask_start_step: Simulation step at which the current subtask began.
        total_subtasks: Total number of subtasks in this episode.
        retry_counts: Mapping from skill_idx to retry count.
        decision_history: Chronological log of decisions.
        episode_done: Whether the episode has ended.
        episode_success: Whether the episode ended successfully.
    """

    def __init__(
        self,
        task_name: str,
        episode_annotation: EpisodeAnnotation,
        duration_stats: Dict[int, SubtaskDurationStats],
    ) -> None:
        self.task_name = task_name
        self.annotation = episode_annotation
        self.duration_stats = duration_stats

        self.current_subtask_idx: int = 0
        self.subtask_start_step: int = 0
        self.total_subtasks: int = len(episode_annotation.subtasks)
        self.retry_counts: Dict[int, int] = {}  # skill_idx -> count
        self.decision_history: List[Dict[str, Any]] = []  # chronological log
        self.episode_done: bool = False
        self.episode_success: bool = False

    # ------------------------------------------------------------------
    # Subtask access
    # ------------------------------------------------------------------

    def get_current_subtask(self) -> Optional[SubtaskAnnotation]:
        """Return the current subtask annotation, or ``None`` if all done.

        Returns:
            The :class:`SubtaskAnnotation` for the current subtask, or
            ``None`` if ``current_subtask_idx`` is out of range.
        """
        if self.current_subtask_idx >= self.total_subtasks:
            return None
        return self.annotation.subtasks[self.current_subtask_idx]

    def get_subtask_by_idx(self, skill_idx: int) -> Optional[SubtaskAnnotation]:
        """Return the subtask annotation for a given skill_idx.

        Args:
            skill_idx: The skill index to look up.

        Returns:
            The matching :class:`SubtaskAnnotation`, or ``None``.
        """
        for subtask in self.annotation.subtasks:
            if subtask.skill_idx == skill_idx:
                return subtask
        return None

    # ------------------------------------------------------------------
    # Duration / threshold helpers
    # ------------------------------------------------------------------

    def get_failure_threshold(self) -> int:
        """Return ``2 * max_duration`` for the current subtask's ``skill_idx``.

        Returns:
            The failure threshold in simulation steps.  Returns a large
            default (10000) if no stats are available.
        """
        subtask = self.get_current_subtask()
        if subtask is None:
            return 10000

        stats = self.duration_stats.get(subtask.skill_idx)
        if stats is not None:
            return stats.failure_threshold
        return 10000  # conservative default

    def get_elapsed_steps(self, current_step: int) -> int:
        """Return steps elapsed since the current subtask started.

        Args:
            current_step: The current simulation step count.

        Returns:
            Number of steps elapsed.
        """
        return max(0, current_step - self.subtask_start_step)

    # ------------------------------------------------------------------
    # Decision recording
    # ------------------------------------------------------------------

    def record_decision(
        self, decision: str, step: int, reason: str
    ) -> None:
        """Append a decision to the history.

        Args:
            decision: One of ``"continue"``, ``"success"``, ``"failure"``.
            step: The simulation step at which the decision was made.
            reason: A human-readable explanation.
        """
        entry = {
            "decision": decision,
            "step": step,
            "subtask_idx": self.current_subtask_idx,
            "reason": reason,
            "timestamp": time.time(),
        }
        self.decision_history.append(entry)

    # ------------------------------------------------------------------
    # Subtask advancement
    # ------------------------------------------------------------------

    def advance_subtask(self, current_step: int) -> Optional[SubtaskAnnotation]:
        """Move to the next subtask. Returns the new subtask or ``None`` if done.

        The current subtask is not explicitly marked as succeeded here;
        that should be done by the caller (via ``record_decision``).

        Args:
            current_step: The simulation step at which we advance.

        Returns:
            The new :class:`SubtaskAnnotation`, or ``None`` if all subtasks
            are complete.
        """
        self.current_subtask_idx += 1
        self.subtask_start_step = current_step

        if self.current_subtask_idx >= self.total_subtasks:
            self.episode_done = True
            self.episode_success = True
            return None

        return self.get_current_subtask()

    # ------------------------------------------------------------------
    # Retry tracking
    # ------------------------------------------------------------------

    def record_retry(self) -> int:
        """Increment retry count for the current subtask.

        Returns:
            The new retry count.
        """
        idx = self.current_subtask_idx
        current_count = self.retry_counts.get(idx, 0) + 1
        self.retry_counts[idx] = current_count
        return current_count

    def get_retry_count(self) -> int:
        """Return the retry count for the current subtask.

        Returns:
            Number of retries so far (0 if none).
        """
        return self.retry_counts.get(self.current_subtask_idx, 0)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable summary of the task memory state.

        Returns:
            Dict suitable for returning through MCP tools.
        """
        subtask = self.get_current_subtask()
        subtask_info: Optional[Dict[str, Any]] = None
        if subtask is not None:
            subtask_info = {
                "skill_idx": subtask.skill_idx,
                "skill_description": subtask.skill_description,
                "skill_type": subtask.skill_type,
                "frame_start": subtask.frame_start,
                "frame_end": subtask.frame_end,
                "frame_duration": subtask.frame_duration,
                **get_rollout_subtask_object_refs(subtask),
            }

        failure_threshold = self.get_failure_threshold()

        return {
            "task_name": self.task_name,
            "current_subtask_idx": self.current_subtask_idx,
            "total_subtasks": self.total_subtasks,
            "subtask_start_step": self.subtask_start_step,
            "current_subtask": subtask_info,
            "failure_threshold": failure_threshold,
            "retry_count": self.get_retry_count(),
            "episode_done": self.episode_done,
            "episode_success": self.episode_success,
            "num_decisions": len(self.decision_history),
            "recent_decisions": self.decision_history[-5:],  # last 5
            "retry_counts": dict(self.retry_counts),
        }
