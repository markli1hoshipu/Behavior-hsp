"""
Ground truth replay decision accuracy test for embodiedClaw.

This module implements section 3.6 step 2 of the requirements:

    Replay HDF5 episodes frame-by-frame through the simulator (no VLA).
    A test agent observes at regular intervals using ``evaluate_subtask``,
    and decides when each subtask completes. The agent's decision frame
    is compared against the ground truth ``frame_end`` from annotations.

    Metric: ``frame_error = agent_decision_frame - gt_end_frame``.
    Measure accuracy at +/-50, +/-100, +/-200, +/-500 frame thresholds.

This module provides:
  - ``SubtaskResult``: per-subtask decision accuracy record
  - ``EpisodeResult``: per-episode aggregation
  - ``compute_accuracy_metrics``: compute accuracy at thresholds
  - ``run_offline_accuracy_test``: run the full test without a simulator
    (uses pre-collected snapshots or annotation-only mode)
  - ``run_gt_replay_accuracy_test``: run with a live ``ReplayEvaluator``

No simulator dependencies in the data classes and metric functions.
The ``run_gt_replay_accuracy_test`` function requires the simulator.
"""

from __future__ import annotations

import json
import logging
import os
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from omnigibson.learning.embodiedClaw.annotation_loader import (
    EpisodeAnnotation,
    SubtaskAnnotation,
    SubtaskDurationStats,
    load_episode_annotation,
    load_all_annotations,
    compute_subtask_duration_stats,
)
from omnigibson.learning.embodiedClaw.decision_module import (
    evaluate_subtask_status,
)
from omnigibson.learning.embodiedClaw.memory import TaskMemory

logger = logging.getLogger("gt_replay_test")
logger.setLevel(logging.INFO)

# Accuracy thresholds (in frames)
ACCURACY_THRESHOLDS = [50, 100, 200, 500]


# ======================================================================
# Data structures
# ======================================================================


@dataclass
class SubtaskResult:
    """Per-subtask decision accuracy record."""

    episode_id: str
    skill_idx: int
    skill_description: str
    gt_start_frame: int
    gt_end_frame: int
    agent_decision_frame: Optional[int]  # None if agent never decided
    frame_error: Optional[int]  # agent_decision_frame - gt_end_frame
    decision_evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def within_50(self) -> bool:
        return self.frame_error is not None and abs(self.frame_error) <= 50

    @property
    def within_100(self) -> bool:
        return self.frame_error is not None and abs(self.frame_error) <= 100

    @property
    def within_200(self) -> bool:
        return self.frame_error is not None and abs(self.frame_error) <= 200

    @property
    def within_500(self) -> bool:
        return self.frame_error is not None and abs(self.frame_error) <= 500

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict."""
        return {
            "episode_id": self.episode_id,
            "skill_idx": self.skill_idx,
            "skill_description": self.skill_description,
            "gt_start_frame": self.gt_start_frame,
            "gt_end_frame": self.gt_end_frame,
            "agent_decision_frame": self.agent_decision_frame,
            "frame_error": self.frame_error,
            "within_50": self.within_50,
            "within_100": self.within_100,
            "within_200": self.within_200,
            "within_500": self.within_500,
            "decision_evidence": self.decision_evidence,
        }


@dataclass
class EpisodeResult:
    """Per-episode aggregation of subtask results."""

    episode_id: str
    task_name: str
    total_frames: int
    subtask_results: List[SubtaskResult] = field(default_factory=list)

    @property
    def num_subtasks(self) -> int:
        return len(self.subtask_results)

    @property
    def num_decided(self) -> int:
        """Number of subtasks where the agent made a decision."""
        return sum(
            1 for r in self.subtask_results if r.agent_decision_frame is not None
        )

    @property
    def num_missed(self) -> int:
        """Number of subtasks the agent never decided on."""
        return self.num_subtasks - self.num_decided

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable summary dict."""
        metrics = compute_accuracy_metrics(self.subtask_results)
        return {
            "episode_id": self.episode_id,
            "task_name": self.task_name,
            "total_frames": self.total_frames,
            "num_subtasks": self.num_subtasks,
            "num_decided": self.num_decided,
            "num_missed": self.num_missed,
            "accuracy": metrics,
            "subtask_results": [r.to_dict() for r in self.subtask_results],
        }


# ======================================================================
# Metrics
# ======================================================================


def compute_accuracy_metrics(
    results: List[SubtaskResult],
) -> Dict[str, Any]:
    """Compute accuracy at +/-50/100/200/500 frame thresholds.

    Args:
        results: List of subtask results (can span multiple episodes).

    Returns:
        Dict with:
            - ``"total_subtasks"``: int
            - ``"total_decided"``: int -- subtasks with a decision
            - ``"total_missed"``: int -- subtasks without a decision
            - ``"accuracy_at_50"``: float (0.0 to 1.0)
            - ``"accuracy_at_100"``: float
            - ``"accuracy_at_200"``: float
            - ``"accuracy_at_500"``: float
            - ``"mean_abs_error"``: float (mean of |frame_error|)
            - ``"median_abs_error"``: float
            - ``"mean_error"``: float (signed, shows bias)
            - ``"frame_errors"``: list of ints (for analysis)
    """
    if not results:
        return {
            "total_subtasks": 0,
            "total_decided": 0,
            "total_missed": 0,
            "accuracy_at_50": 0.0,
            "accuracy_at_100": 0.0,
            "accuracy_at_200": 0.0,
            "accuracy_at_500": 0.0,
            "mean_abs_error": 0.0,
            "median_abs_error": 0.0,
            "mean_error": 0.0,
            "frame_errors": [],
        }

    total = len(results)
    decided = [r for r in results if r.frame_error is not None]
    missed = total - len(decided)

    # Count within-threshold hits
    within_50 = sum(1 for r in decided if abs(r.frame_error) <= 50)
    within_100 = sum(1 for r in decided if abs(r.frame_error) <= 100)
    within_200 = sum(1 for r in decided if abs(r.frame_error) <= 200)
    within_500 = sum(1 for r in decided if abs(r.frame_error) <= 500)

    # Compute error statistics
    errors = [r.frame_error for r in decided]
    abs_errors = [abs(e) for e in errors]

    mean_abs = statistics.mean(abs_errors) if abs_errors else 0.0
    median_abs = statistics.median(abs_errors) if abs_errors else 0.0
    mean_signed = statistics.mean(errors) if errors else 0.0

    # Accuracy is fraction of total subtasks within threshold
    # (missed subtasks count as failures)
    return {
        "total_subtasks": total,
        "total_decided": len(decided),
        "total_missed": missed,
        "accuracy_at_50": within_50 / total if total > 0 else 0.0,
        "accuracy_at_100": within_100 / total if total > 0 else 0.0,
        "accuracy_at_200": within_200 / total if total > 0 else 0.0,
        "accuracy_at_500": within_500 / total if total > 0 else 0.0,
        "mean_abs_error": round(mean_abs, 1),
        "median_abs_error": round(median_abs, 1),
        "mean_error": round(mean_signed, 1),
        "frame_errors": errors,
    }


# ======================================================================
# Offline accuracy test (annotation-only, no simulator)
# ======================================================================


def run_offline_accuracy_test(
    annotation: EpisodeAnnotation,
    duration_stats: Dict[int, SubtaskDurationStats],
    episode_id: str,
    simulated_snapshots: Optional[List[Dict[str, Any]]] = None,
    observation_interval: int = 25,
) -> EpisodeResult:
    """Run a decision accuracy test using annotation data and optional
    pre-collected snapshots.

    This function simulates the agent's decision loop without a live
    simulator.  At each observation point, it calls
    ``evaluate_subtask_status`` with the snapshot (or a minimal mock
    snapshot if none is provided) and checks whether the decision module
    recommends "success".

    **Gap handling:** Annotation subtasks may have gaps between the end
    of one subtask and the start of the next (e.g., skill 0 ends at
    frame 325, skill 1 starts at frame 642).  During gaps, the agent
    should not make any decisions -- the test skips directly to the
    next subtask's start frame.

    Args:
        annotation: Parsed episode annotation.
        duration_stats: Cross-episode subtask duration statistics.
        episode_id: Episode identifier string.
        simulated_snapshots: Optional list of snapshots (one per
            observation point).  If None, uses minimal mock snapshots
            that only contain ``step_count``.
        observation_interval: Frames between observation points.

    Returns:
        :class:`EpisodeResult` with per-subtask decision records.
    """
    task_memory = TaskMemory(
        task_name=annotation.task_name,
        episode_annotation=annotation,
        duration_stats=duration_stats,
    )

    total_frames = annotation.task_duration
    result = EpisodeResult(
        episode_id=episode_id,
        task_name=annotation.task_name,
        total_frames=total_frames,
    )

    # Build snapshot lookup (frame -> snapshot)
    snapshot_lookup: Dict[int, Dict[str, Any]] = {}
    if simulated_snapshots:
        for snap in simulated_snapshots:
            frame = snap.get("step_count", 0)
            snapshot_lookup[frame] = snap

    subtask_idx = 0
    subtasks = annotation.subtasks

    for subtask in subtasks:
        # Set memory to track this subtask
        task_memory.current_subtask_idx = subtask_idx
        task_memory.subtask_start_step = subtask.frame_start

        agent_decision_frame = None
        decision_evidence: Dict[str, Any] = {}

        # Generate observation frames within this subtask's range.
        # Also check frames beyond gt_end (agent might decide late)
        # up to the failure threshold.
        check_start = subtask.frame_start
        failure_threshold = task_memory.get_failure_threshold()
        check_end = min(
            subtask.frame_start + failure_threshold,
            total_frames,
        )

        frame = check_start
        while frame <= check_end:
            # Round to nearest observation interval boundary
            obs_frame = (frame // observation_interval) * observation_interval
            if obs_frame < check_start:
                obs_frame += observation_interval

            # Get or create snapshot
            snapshot = snapshot_lookup.get(obs_frame)
            if snapshot is None:
                snapshot = _create_mock_snapshot(obs_frame)

            # Evaluate
            eval_result = evaluate_subtask_status(
                snapshot=snapshot,
                memory=task_memory,
                current_step=obs_frame,
            )

            recommendation = eval_result.get("recommendation", "continue")

            if recommendation == "success":
                agent_decision_frame = obs_frame
                decision_evidence = {
                    "recommendation": recommendation,
                    "reasoning": eval_result.get("reasoning", ""),
                    "bddl_evidence": eval_result.get("bddl_evidence", {}),
                }
                break
            elif recommendation == "failure":
                # Agent thinks subtask failed -- record as missed
                decision_evidence = {
                    "recommendation": recommendation,
                    "reasoning": eval_result.get("reasoning", ""),
                }
                break

            frame = obs_frame + observation_interval

        # Compute frame error
        frame_error = None
        if agent_decision_frame is not None:
            frame_error = agent_decision_frame - subtask.frame_end

        subtask_result = SubtaskResult(
            episode_id=episode_id,
            skill_idx=subtask.skill_idx,
            skill_description=subtask.skill_description,
            gt_start_frame=subtask.frame_start,
            gt_end_frame=subtask.frame_end,
            agent_decision_frame=agent_decision_frame,
            frame_error=frame_error,
            decision_evidence=decision_evidence,
        )
        result.subtask_results.append(subtask_result)

        # Advance memory regardless of decision outcome
        if agent_decision_frame is not None:
            task_memory.record_decision(
                decision="success",
                step=agent_decision_frame,
                reason=f"Offline test: subtask {subtask.skill_idx} decided at frame {agent_decision_frame}",
            )
        else:
            task_memory.record_decision(
                decision="missed",
                step=subtask.frame_end,
                reason=f"Offline test: subtask {subtask.skill_idx} not decided",
            )
        task_memory.advance_subtask(
            agent_decision_frame if agent_decision_frame is not None else subtask.frame_end
        )
        subtask_idx += 1

    return result


def _create_mock_snapshot(frame: int) -> Dict[str, Any]:
    """Create a minimal mock snapshot for offline testing.

    The mock snapshot has enough structure for ``evaluate_subtask_status``
    to run, but BDDL checks will always return False (no predicates
    satisfied).  This means offline-only tests measure only the failure
    threshold logic, not BDDL-based detection.

    Args:
        frame: The simulation frame for this snapshot.

    Returns:
        Minimal snapshot dict.
    """
    return {
        "step_count": frame,
        "task_name": "",
        "images": {},
        "robot_state": {
            "position": [0.0, 0.0, 0.0],
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "joint_positions": [],
            "eef_poses": {},
            "grasped_objects": {},
        },
        "object_states": {},
        "bddl_state": [],
    }


# ======================================================================
# Live accuracy test (requires simulator + ReplayEvaluator)
# ======================================================================


def run_gt_replay_accuracy_test(
    replay_evaluator,
    annotation: EpisodeAnnotation,
    duration_stats: Dict[int, SubtaskDurationStats],
    episode_id: str,
    observation_interval: int = 25,
) -> EpisodeResult:
    """Run a decision accuracy test using a live ReplayEvaluator.

    Replays the episode frame-by-frame, taking snapshots at the
    specified interval.  At each snapshot, evaluates subtask status
    and records when the agent would decide "success".

    **Gap handling:** Between subtask end and next subtask start, the
    replay fast-forwards (loads state but skips evaluation).

    Args:
        replay_evaluator: A ``ReplayEvaluator`` instance with the
            HDF5 episode loaded and initial state set.
        annotation: Parsed episode annotation for this episode.
        duration_stats: Cross-episode subtask duration statistics.
        episode_id: Episode identifier string.
        observation_interval: Frames between evaluation points.

    Returns:
        :class:`EpisodeResult` with per-subtask decision records.
    """
    task_memory = TaskMemory(
        task_name=annotation.task_name,
        episode_annotation=annotation,
        duration_stats=duration_stats,
    )

    total_frames = replay_evaluator.num_frames
    result = EpisodeResult(
        episode_id=episode_id,
        task_name=annotation.task_name,
        total_frames=total_frames,
    )

    subtasks = annotation.subtasks
    subtask_idx = 0

    for subtask in subtasks:
        task_memory.current_subtask_idx = subtask_idx
        task_memory.subtask_start_step = subtask.frame_start

        agent_decision_frame = None
        decision_evidence: Dict[str, Any] = {}

        # Check frames from subtask start up to failure threshold
        failure_threshold = task_memory.get_failure_threshold()
        check_end = min(
            subtask.frame_start + failure_threshold,
            total_frames,
        )

        frame = subtask.frame_start
        while frame <= check_end:
            # Align to observation interval
            obs_frame = (frame // observation_interval) * observation_interval
            if obs_frame < subtask.frame_start:
                obs_frame += observation_interval

            # Clamp to valid state range
            state_count = replay_evaluator.hdf5_data["state"].shape[0]
            state_idx = min(obs_frame, state_count - 1)

            # Load state and get snapshot
            replay_evaluator.load_state_at_frame(state_idx)
            snapshot = replay_evaluator.get_snapshot_at_current_frame()

            # Evaluate
            eval_result = evaluate_subtask_status(
                snapshot=snapshot,
                memory=task_memory,
                current_step=obs_frame,
            )

            recommendation = eval_result.get("recommendation", "continue")

            if recommendation == "success":
                agent_decision_frame = obs_frame
                decision_evidence = {
                    "recommendation": recommendation,
                    "reasoning": eval_result.get("reasoning", ""),
                    "bddl_evidence": eval_result.get("bddl_evidence", {}),
                }
                break
            elif recommendation == "failure":
                decision_evidence = {
                    "recommendation": recommendation,
                    "reasoning": eval_result.get("reasoning", ""),
                }
                break

            frame = obs_frame + observation_interval

        frame_error = None
        if agent_decision_frame is not None:
            frame_error = agent_decision_frame - subtask.frame_end

        subtask_result = SubtaskResult(
            episode_id=episode_id,
            skill_idx=subtask.skill_idx,
            skill_description=subtask.skill_description,
            gt_start_frame=subtask.frame_start,
            gt_end_frame=subtask.frame_end,
            agent_decision_frame=agent_decision_frame,
            frame_error=frame_error,
            decision_evidence=decision_evidence,
        )
        result.subtask_results.append(subtask_result)

        # Advance memory regardless of decision outcome
        if agent_decision_frame is not None:
            task_memory.record_decision(
                decision="success",
                step=agent_decision_frame,
                reason=f"GT replay: subtask {subtask.skill_idx} decided at frame {agent_decision_frame}",
            )
        else:
            task_memory.record_decision(
                decision="missed",
                step=subtask.frame_end,
                reason=f"GT replay: subtask {subtask.skill_idx} not decided",
            )
        task_memory.advance_subtask(
            agent_decision_frame if agent_decision_frame is not None else subtask.frame_end
        )
        subtask_idx += 1

    return result


# ======================================================================
# Multi-episode test runner
# ======================================================================


def run_multi_episode_test(
    dataset_root: str = "/home/user/dataset",
    task_id: int = 1,
    max_episodes: Optional[int] = None,
    observation_interval: int = 25,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the offline accuracy test across all episodes of a task.

    This function does NOT require the simulator. It loads annotations
    and uses mock snapshots (BDDL checks will always be negative).
    Useful for validating the test infrastructure before running with
    the live simulator.

    Args:
        dataset_root: Root of the BEHAVIOR-1K dataset.
        task_id: Numeric task ID (e.g. 1).
        max_episodes: Maximum number of episodes to test (None = all).
        observation_interval: Frames between evaluation points.
        output_path: Optional path to write JSON results.

    Returns:
        Dict with:
            - ``"task_id"``: int
            - ``"num_episodes"``: int
            - ``"episode_results"``: list of per-episode result dicts
            - ``"aggregate_accuracy"``: aggregate metrics across all episodes
    """
    annotation_dir = os.path.join(
        dataset_root, "annotations", f"task-{task_id:04d}"
    )
    annotations = load_all_annotations(annotation_dir)
    if not annotations:
        return {"error": f"No annotations found in {annotation_dir}"}

    duration_stats = compute_subtask_duration_stats(annotations)

    if max_episodes is not None:
        annotations = annotations[:max_episodes]

    # Get annotation filenames for episode ID extraction
    annotation_files = sorted(
        f for f in os.listdir(annotation_dir) if f.endswith(".json")
    )

    all_subtask_results: List[SubtaskResult] = []
    episode_results: List[Dict[str, Any]] = []

    for i, annotation in enumerate(annotations):
        if i < len(annotation_files):
            episode_id = annotation_files[i].replace("episode_", "").replace(".json", "")
        else:
            episode_id = f"episode_{i:08d}"

        logger.info(
            "Testing episode %s (%d/%d)...",
            episode_id, i + 1, len(annotations),
        )

        ep_result = run_offline_accuracy_test(
            annotation=annotation,
            duration_stats=duration_stats,
            episode_id=episode_id,
            observation_interval=observation_interval,
        )

        all_subtask_results.extend(ep_result.subtask_results)
        episode_results.append(ep_result.to_dict())

    # Aggregate metrics
    aggregate = compute_accuracy_metrics(all_subtask_results)

    output = {
        "task_id": task_id,
        "num_episodes": len(annotations),
        "observation_interval": observation_interval,
        "aggregate_accuracy": aggregate,
        "episode_results": episode_results,
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info("Results written to %s", output_path)

    return output


# ======================================================================
# CLI entry point (offline mode)
# ======================================================================


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Run GT replay decision accuracy test (offline mode)"
    )
    parser.add_argument(
        "--dataset-root",
        default="/home/user/dataset",
        help="Root of the BEHAVIOR-1K dataset",
    )
    parser.add_argument(
        "--task-id", type=int, default=1,
        help="Task ID (default: 1 = picking_up_trash)",
    )
    parser.add_argument(
        "--max-episodes", type=int, default=None,
        help="Max episodes to test (default: all)",
    )
    parser.add_argument(
        "--observation-interval", type=int, default=25,
        help="Frames between evaluation points (default: 25)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write JSON results",
    )
    args = parser.parse_args()

    results = run_multi_episode_test(
        dataset_root=args.dataset_root,
        task_id=args.task_id,
        max_episodes=args.max_episodes,
        observation_interval=args.observation_interval,
        output_path=args.output,
    )

    # Print summary
    agg = results.get("aggregate_accuracy", {})
    print("\n" + "=" * 60)
    print("  GT Replay Decision Accuracy Test -- Summary")
    print("=" * 60)
    print(f"  Task ID:              {results.get('task_id')}")
    print(f"  Episodes tested:      {results.get('num_episodes')}")
    print(f"  Total subtasks:       {agg.get('total_subtasks', 0)}")
    print(f"  Decided:              {agg.get('total_decided', 0)}")
    print(f"  Missed:               {agg.get('total_missed', 0)}")
    print(f"  Accuracy @ +/-50:     {agg.get('accuracy_at_50', 0):.1%}")
    print(f"  Accuracy @ +/-100:    {agg.get('accuracy_at_100', 0):.1%}")
    print(f"  Accuracy @ +/-200:    {agg.get('accuracy_at_200', 0):.1%}")
    print(f"  Accuracy @ +/-500:    {agg.get('accuracy_at_500', 0):.1%}")
    print(f"  Mean |error|:         {agg.get('mean_abs_error', 0):.1f} frames")
    print(f"  Median |error|:       {agg.get('median_abs_error', 0):.1f} frames")
    print(f"  Mean error (signed):  {agg.get('mean_error', 0):.1f} frames")
    print("=" * 60)
