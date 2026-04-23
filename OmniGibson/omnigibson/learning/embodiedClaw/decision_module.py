"""
Core decision logic for the embodiedClaw agentic data collection system.

Implements BDDL-based completion detection, failure threshold checking,
navigation distance checks, grasp detection, and placement detection.
All functions are pure Python operating on snapshot dicts and annotation data.

No simulator dependencies -- this module works with the structured dicts
produced by :func:`get_simulation_snapshot`.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

from omnigibson.learning.embodiedClaw.annotation_loader import (
    SubtaskAnnotation,
    get_rollout_subtask_object_refs,
)
from omnigibson.learning.embodiedClaw.memory import TaskMemory
from omnigibson.learning.embodiedClaw.object_matching import (
    arg_matches_query,
    build_match_query,
    entity_matches_query,
    name_matches_query,
)

logger = logging.getLogger(__name__)

__all__ = [
    "evaluate_subtask_status",
    "check_bddl_completion",
    "check_grasp_success",
    "check_placement_success",
    "check_navigation_success",
    "check_failure",
    "compare_with_demo",
]

# Distance threshold for navigation subtask completion (meters).
NAVIGATION_DISTANCE_THRESHOLD = 1.5


# ======================================================================
# Navigation check
# ======================================================================


def check_navigation_success(
    snapshot: Dict[str, Any],
    target_object_ids: List[str],
) -> Dict[str, Any]:
    """Check if the robot is close enough to the target object for navigation.

    Computes the Euclidean distance between the robot position and each
    target object position.  If any distance is below the threshold,
    the navigation is considered successful.

    Args:
        snapshot: Simulation snapshot from ``get_simulation_snapshot``.
        target_object_ids: Annotation object names (e.g. ``["trash_can_116"]``).

    Returns:
        Dict with keys ``"completed"`` (bool), ``"min_distance"`` (float or
        None), ``"threshold"`` (float), and ``"reasoning"`` (str).
    """
    robot_pos = snapshot.get("robot_state", {}).get("position")
    if robot_pos is None or len(robot_pos) < 3:
        return {
            "completed": False,
            "min_distance": None,
            "threshold": NAVIGATION_DISTANCE_THRESHOLD,
            "reasoning": "Robot position not available.",
        }

    object_states = snapshot.get("object_states", {})
    min_dist: Optional[float] = None
    closest_obj: Optional[str] = None

    target_query = build_match_query(
        target_object_ids,
        include_normalized=True,
        include_normalized_for_instance_specific=True,
    )

    for scope_name, obj_info in object_states.items():
        obj_pos = obj_info.get("position")
        if obj_pos is None or len(obj_pos) < 3:
            continue

        if not entity_matches_query(
            scope_name,
            obj_info.get("scene_name"),
            target_query,
        ):
            continue

        # Compute Euclidean distance (XY plane only -- height differences
        # between robot base and objects on surfaces shouldn't penalise)
        dx = robot_pos[0] - obj_pos[0]
        dy = robot_pos[1] - obj_pos[1]
        dist = math.sqrt(dx * dx + dy * dy)

        if min_dist is None or dist < min_dist:
            min_dist = dist
            closest_obj = scope_name

    if min_dist is None:
        return {
            "completed": False,
            "min_distance": None,
            "threshold": NAVIGATION_DISTANCE_THRESHOLD,
            "reasoning": "Target objects not found in snapshot object_states.",
        }

    completed = min_dist <= NAVIGATION_DISTANCE_THRESHOLD
    return {
        "completed": completed,
        "min_distance": round(min_dist, 3),
        "closest_object": closest_obj,
        "threshold": NAVIGATION_DISTANCE_THRESHOLD,
        "reasoning": (
            f"Distance to {closest_obj}: {min_dist:.3f}m "
            f"({'within' if completed else 'exceeds'} threshold "
            f"{NAVIGATION_DISTANCE_THRESHOLD}m)."
        ),
    }


# ======================================================================
# Grasp detection
# ======================================================================


def check_grasp_success(
    snapshot: Dict[str, Any],
    manipulating_objects: List[str],
) -> Dict[str, Any]:
    """Check if any of the manipulating objects are currently grasped.

    Looks at ``snapshot["robot_state"]["grasped_objects"]`` which maps
    arm names to grasped object names (or ``None``).

    Args:
        snapshot: Simulation snapshot.
        manipulating_objects: Annotation names of objects that should be
            grasped (e.g. ``["trash_can_116"]``).

    Returns:
        Dict with ``"completed"`` (bool), ``"grasped_objects"`` (dict),
        and ``"reasoning"`` (str).
    """
    grasped = snapshot.get("robot_state", {}).get("grasped_objects", {})
    target_query = build_match_query(
        manipulating_objects,
        include_normalized=True,
        include_normalized_for_instance_specific=True,
    )

    matched_arms: List[str] = []
    for arm, obj_name in grasped.items():
        if obj_name is None:
            continue
        if name_matches_query(obj_name, target_query):
            matched_arms.append(arm)

    completed = len(matched_arms) > 0
    return {
        "completed": completed,
        "grasped_objects": grasped,
        "matched_arms": matched_arms,
        "reasoning": (
            f"Grasped objects: {grasped}. "
            f"{'Target object grasped' if completed else 'Target object NOT grasped'} "
            f"(looking for {manipulating_objects})."
        ),
    }


# ======================================================================
# Placement detection
# ======================================================================

def check_placement_success(
    snapshot: Dict[str, Any],
    placed_object: str,
    target_container: str,
    placement_type: str = "Inside",
) -> Dict[str, Any]:
    """Check if the world predicate for placement is satisfied.

    Scans ``snapshot["world_predicates"]`` for a predicate whose name
    matches ``placement_type`` and whose args reference both
    ``placed_object`` and ``target_container``. Exact annotation-instance
    matches are preferred, but rollout-safe normalized type matches can also
    satisfy the check when the subtask created a new matching relation.

    Args:
        snapshot: Simulation snapshot (must contain ``world_predicates``).
        placed_object: Annotation name of the object being placed
            (e.g. ``"can_of_soda_114"``).
        target_container: Annotation name of the container/surface
            (e.g. ``"trash_can_116"``).
        placement_type: ``"Inside"`` or ``"OnTop"``.

    Returns:
        Dict with ``"completed"`` (bool), ``"relevant_predicates"`` (list),
        ``"still_grasping"`` (bool), and ``"reasoning"`` (str).
    """
    world_predicates = snapshot.get("world_predicates", [])
    predicate_deltas = snapshot.get("predicate_deltas_since_subtask_start", [])
    exact_placed_query = build_match_query([placed_object], include_normalized=False)
    exact_target_query = build_match_query([target_container], include_normalized=False)
    normalized_placed_query = build_match_query(
        [placed_object],
        include_normalized=True,
        include_normalized_for_instance_specific=True,
    )
    normalized_target_query = build_match_query(
        [target_container],
        include_normalized=True,
        include_normalized_for_instance_specific=True,
    )

    def _still_grasping_matched_object(
        successful_records: List[Dict[str, Any]],
        placed_query,
    ) -> bool:
        matched_scene_names = {
            str(arg.get("scene_name")).lower()
            for record in successful_records
            for arg in record.get("args", [])
            if arg.get("scene_name") is not None and arg_matches_query(arg, placed_query)
        }
        if matched_scene_names:
            return any(
                obj_name is not None and obj_name.lower() in matched_scene_names
                for obj_name in grasped.values()
            )
        return any(
            obj_name is not None and name_matches_query(obj_name, placed_query)
            for obj_name in grasped.values()
        )

    # Match predicates by: predicate name matches placement_type AND args
    # reference both the placed object and the target container. Prefer exact
    # annotation instance ids when available, but also allow rollout-safe
    # normalized type matches so repeated same-type subtasks are not locked to
    # the demonstrator's arbitrary instance ordering.
    exact_relevant_preds: List[Dict[str, Any]] = []
    normalized_relevant_preds: List[Dict[str, Any]] = []
    exact_true_preds: List[Dict[str, Any]] = []
    normalized_true_preds: List[Dict[str, Any]] = []

    for pred in world_predicates:
        pred_name = pred.get("predicate", "")
        if pred_name != placement_type:
            continue

        args = pred.get("args", [])
        if len(args) < 2:
            continue

        exact_has_placed = any(arg_matches_query(arg, exact_placed_query) for arg in args)
        exact_has_target = any(arg_matches_query(arg, exact_target_query) for arg in args)
        normalized_has_placed = any(
            arg_matches_query(arg, normalized_placed_query)
            for arg in args
        )
        normalized_has_target = any(
            arg_matches_query(arg, normalized_target_query)
            for arg in args
        )

        if exact_has_placed and exact_has_target:
            exact_relevant_preds.append(pred)
            if pred.get("value") is True:
                exact_true_preds.append(pred)
        elif normalized_has_placed and normalized_has_target:
            normalized_relevant_preds.append(pred)
            if pred.get("value") is True:
                normalized_true_preds.append(pred)

    exact_relevant_deltas: List[Dict[str, Any]] = []
    normalized_relevant_deltas: List[Dict[str, Any]] = []
    exact_true_deltas: List[Dict[str, Any]] = []
    normalized_true_deltas: List[Dict[str, Any]] = []

    for delta in predicate_deltas:
        if delta.get("predicate") != placement_type:
            continue

        args = delta.get("args", [])
        if len(args) < 2:
            continue

        exact_has_placed = any(arg_matches_query(arg, exact_placed_query) for arg in args)
        exact_has_target = any(arg_matches_query(arg, exact_target_query) for arg in args)
        normalized_has_placed = any(
            arg_matches_query(arg, normalized_placed_query)
            for arg in args
        )
        normalized_has_target = any(
            arg_matches_query(arg, normalized_target_query)
            for arg in args
        )
        became_true = (
            delta.get("change_type") in {"became_true", "added"}
            and delta.get("new_value") is True
        )

        if exact_has_placed and exact_has_target:
            exact_relevant_deltas.append(delta)
            if became_true:
                exact_true_deltas.append(delta)
        elif normalized_has_placed and normalized_has_target:
            normalized_relevant_deltas.append(delta)
            if became_true:
                normalized_true_deltas.append(delta)

    # Also check if the robot is no longer grasping the placed object
    grasped = snapshot.get("robot_state", {}).get("grasped_objects", {})
    exact_success_records = exact_true_preds + exact_true_deltas
    exact_still_grasping = _still_grasping_matched_object(
        exact_success_records,
        exact_placed_query,
    )
    normalized_delta_still_grasping = _still_grasping_matched_object(
        normalized_true_deltas,
        normalized_placed_query,
    )
    completed = False
    matching_strategy = "none"

    if exact_true_preds and not exact_still_grasping:
        completed = True
        matching_strategy = "exact_relation"
    elif exact_true_deltas and not exact_still_grasping:
        completed = True
        matching_strategy = "exact_delta"
    elif normalized_true_deltas and not normalized_delta_still_grasping:
        completed = True
        matching_strategy = "normalized_delta"

    if exact_true_preds or exact_true_deltas:
        still_grasping = exact_still_grasping
    elif normalized_true_deltas:
        still_grasping = normalized_delta_still_grasping
    else:
        still_grasping = exact_still_grasping or normalized_delta_still_grasping

    relevant_preds = exact_relevant_preds + normalized_relevant_preds
    relevant_deltas = exact_relevant_deltas + normalized_relevant_deltas

    return {
        "completed": completed,
        "relevant_predicates": relevant_preds,
        "relevant_deltas": relevant_deltas,
        "still_grasping": still_grasping,
        "matching_strategy": matching_strategy,
        "exact_true_relation_count": len(exact_true_preds),
        "normalized_true_relation_count": len(normalized_true_preds),
        "normalized_true_delta_count": len(normalized_true_deltas),
        "reasoning": (
            f"Placement check ({placement_type}): "
            f"placed_object={placed_object}, target={target_container}. "
            f"{'Relation satisfied' if completed else 'Relation NOT satisfied'}. "
            f"Still grasping: {still_grasping}. "
            f"strategy={matching_strategy}. "
            f"Found {len(relevant_preds)} relevant predicates and "
            f"{len(relevant_deltas)} relevant deltas."
        ),
    }


# ======================================================================
# Composite BDDL completion check
# ======================================================================


def check_bddl_completion(
    snapshot: Dict[str, Any],
    subtask: SubtaskAnnotation,
) -> Dict[str, Any]:
    """Check BDDL predicates relevant to the current subtask.

    Dispatches to the appropriate check function based on the subtask's
    ``skill_description``.

    Args:
        snapshot: Simulation snapshot.
        subtask: Current subtask annotation.

    Returns:
        Dict with ``"completed"`` (bool), subtask-specific evidence fields,
        ``"grasp_state"`` (dict of what robot is holding), and
        ``"reasoning"`` (str).
    """
    skill_desc = subtask.skill_description
    obj_group = subtask.object_ids[0] if subtask.object_ids else []
    manip_objs = subtask.manipulating_object_ids

    # Always include grasp state for reference
    grasped = snapshot.get("robot_state", {}).get("grasped_objects", {})

    if skill_desc == "move to":
        nav_result = check_navigation_success(snapshot, obj_group)
        return {
            "completed": nav_result["completed"],
            "check_type": "navigation",
            "navigation_evidence": nav_result,
            "grasp_state": grasped,
            "reasoning": nav_result["reasoning"],
        }

    elif skill_desc == "pick up from":
        grasp_result = check_grasp_success(snapshot, manip_objs)
        return {
            "completed": grasp_result["completed"],
            "check_type": "grasp",
            "grasp_evidence": grasp_result,
            "grasp_state": grasped,
            "reasoning": grasp_result["reasoning"],
        }

    elif skill_desc == "place in":
        if len(obj_group) >= 2:
            place_result = check_placement_success(
                snapshot, obj_group[0], obj_group[1], placement_type="Inside"
            )
        elif len(obj_group) == 1 and manip_objs:
            place_result = check_placement_success(
                snapshot, manip_objs[0], obj_group[0], placement_type="Inside"
            )
        else:
            place_result = {
                "completed": False,
                "relevant_predicates": [],
                "still_grasping": False,
                "reasoning": "Insufficient object info for placement check.",
            }
        return {
            "completed": place_result["completed"],
            "check_type": "placement_inside",
            "placement_evidence": place_result,
            "grasp_state": grasped,
            "reasoning": place_result["reasoning"],
        }

    elif skill_desc == "place on":
        if len(obj_group) >= 2:
            place_result = check_placement_success(
                snapshot, obj_group[0], obj_group[1], placement_type="OnTop"
            )
        elif len(obj_group) == 1 and manip_objs:
            place_result = check_placement_success(
                snapshot, manip_objs[0], obj_group[0], placement_type="OnTop"
            )
        else:
            place_result = {
                "completed": False,
                "relevant_predicates": [],
                "still_grasping": False,
                "reasoning": "Insufficient object info for placement check.",
            }
        return {
            "completed": place_result["completed"],
            "check_type": "placement_ontop",
            "placement_evidence": place_result,
            "grasp_state": grasped,
            "reasoning": place_result["reasoning"],
        }

    else:
        # Unknown skill type -- return inconclusive
        return {
            "completed": False,
            "check_type": "unknown",
            "grasp_state": grasped,
            "reasoning": f"Unknown skill description: '{skill_desc}'. Cannot evaluate.",
        }


# ======================================================================
# Failure detection
# ======================================================================


def check_failure(
    memory: TaskMemory,
    current_step: int,
) -> Dict[str, Any]:
    """Check if the current subtask has exceeded its failure threshold.

    The failure threshold is ``2 * max_duration`` across all demo episodes
    for this ``skill_idx``.

    Args:
        memory: Current :class:`TaskMemory`.
        current_step: Current simulation step.

    Returns:
        Dict with ``"is_failure"`` (bool), ``"elapsed_steps"`` (int),
        ``"failure_threshold"`` (int), and ``"ratio"`` (float).
    """
    elapsed = memory.get_elapsed_steps(current_step)
    threshold = memory.get_failure_threshold()

    ratio = elapsed / threshold if threshold > 0 else 0.0
    is_failure = elapsed > threshold

    return {
        "is_failure": is_failure,
        "elapsed_steps": elapsed,
        "failure_threshold": threshold,
        "ratio": round(ratio, 3),
    }


# ======================================================================
# Demo comparison
# ======================================================================


def compare_with_demo(
    snapshot: Dict[str, Any],
    subtask: SubtaskAnnotation,
    demo_reference_frames: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compare current state with demo reference frames.

    This function does NOT do image similarity scoring -- that is left to
    the agent's (Claude Code's) visual understanding.  Instead it packages
    the demo reference frame *metadata* alongside the current snapshot
    metadata so the agent can cross-reference them.  The actual base64
    images are returned by the ``get_demo_reference_frames`` MCP tool
    separately to avoid payload duplication.

    Args:
        snapshot: Simulation snapshot.
        subtask: Current subtask annotation.
        demo_reference_frames: Dict returned by
            :func:`~omnigibson.learning.embodiedClaw.tools.demo_tools.get_demo_reference_frames`
            (or ``None`` if demo frames are not available).

    Returns:
        Dict with:
            - ``"has_demo_frames"``: bool
            - ``"num_demo_frames"``: int
            - ``"demo_frames_summary"``: list of per-frame metadata
              (frame_type, relative_position, episode_id) WITHOUT
              the actual base64 images
            - ``"reasoning"``: str
    """
    if demo_reference_frames is None or demo_reference_frames.get("status") != "ok":
        return {
            "has_demo_frames": False,
            "num_demo_frames": 0,
            "demo_frames_summary": [],
            "reasoning": (
                "No demo reference frames available for this subtask. "
                "Visual comparison cannot be performed."
            ),
        }

    frames = demo_reference_frames.get("frames", [])
    num_frames = len(frames)

    # Build a lightweight summary (no base64 images)
    summary: List[Dict[str, Any]] = []
    for f in frames:
        summary.append({
            "episode_id": f.get("episode_id", ""),
            "frame_type": f.get("frame_type", ""),
            "frame_index": f.get("frame_index", 0),
            "relative_position": f.get("relative_position", 0.0),
            "camera": f.get("camera", ""),
        })

    # Count completion frames for reasoning
    completion_count = sum(1 for f in frames if f.get("frame_type") == "completion")
    episode_count = demo_reference_frames.get("num_episodes", 0)

    return {
        "has_demo_frames": True,
        "num_demo_frames": num_frames,
        "demo_frames_summary": summary,
        "reasoning": (
            f"Demo reference frames available: {num_frames} frames from "
            f"{episode_count} episodes ({completion_count} completion frames). "
            f"Use get_demo_reference_frames tool to visually compare the "
            f"current snapshot against demo completion states."
        ),
    }


# ======================================================================
# Main evaluation function
# ======================================================================


def evaluate_subtask_status(
    snapshot: Dict[str, Any],
    memory: TaskMemory,
    current_step: int,
    demo_reference_frames: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate whether the current subtask should continue, succeed, or fail.

    This function implements BOTH decision approaches (BDDL-based and
    demo-comparison) and returns evidence for each.  The agent (Claude
    Code) makes the final call using this evidence combined with its own
    visual understanding.

    Args:
        snapshot: Full simulation snapshot from ``get_simulation_snapshot``.
        memory: Current :class:`TaskMemory` state.
        current_step: Current simulation step count.
        demo_reference_frames: Optional dict returned by
            :func:`~omnigibson.learning.embodiedClaw.tools.demo_tools.get_demo_reference_frames`.
            When provided, enables demo-comparison evidence in the result.

    Returns:
        Dict with keys:
            - ``"recommendation"``: ``"continue"`` | ``"success"`` |
              ``"failure"`` | ``"needs_visual_comparison"``
            - ``"elapsed_steps"``: int
            - ``"failure_threshold"``: int
            - ``"over_threshold"``: bool
            - ``"bddl_evidence"``: dict (approach 1 results)
            - ``"demo_evidence"``: dict (approach 2 results)
            - ``"subtask_info"``: dict (current subtask details)
            - ``"memory_summary"``: dict (progress overview)
    """
    subtask = memory.get_current_subtask()

    # If all subtasks are done, the episode is complete
    if subtask is None:
        return {
            "recommendation": "success",
            "elapsed_steps": 0,
            "failure_threshold": 0,
            "over_threshold": False,
            "bddl_evidence": {},
            "demo_evidence": {},
            "subtask_info": None,
            "memory_summary": memory.to_dict(),
            "reasoning": "All subtasks completed. Episode is done.",
        }

    # Update elapsed steps
    elapsed = memory.get_elapsed_steps(current_step)

    # --- Approach 1: BDDL-based checks ---
    bddl_evidence = check_bddl_completion(snapshot, subtask)

    # --- Approach 2: Demo comparison ---
    demo_evidence = compare_with_demo(snapshot, subtask, demo_reference_frames)

    # --- Failure check ---
    failure_info = check_failure(memory, current_step)

    # --- Build recommendation ---
    recommendation: str
    reasoning: str

    has_demo = demo_evidence.get("has_demo_frames", False)

    if bddl_evidence.get("completed", False):
        recommendation = "success"
        reasoning = (
            f"BDDL evidence indicates subtask {subtask.skill_idx} "
            f"('{subtask.skill_description}') is complete. {bddl_evidence.get('reasoning', '')}"
        )
    elif failure_info["is_failure"]:
        recommendation = "failure"
        reasoning = (
            f"Subtask {subtask.skill_idx} ('{subtask.skill_description}') "
            f"has exceeded the failure threshold: {elapsed} steps elapsed "
            f"vs threshold of {failure_info['failure_threshold']} steps "
            f"(ratio: {failure_info['ratio']})."
        )
    elif (
        not bddl_evidence.get("completed", False)
        and has_demo
        and failure_info["ratio"] >= 0.5
    ):
        # BDDL is inconclusive, we're past halfway, and demo frames exist.
        # Signal that the agent should visually compare with demo frames.
        recommendation = "needs_visual_comparison"
        reasoning = (
            f"Subtask {subtask.skill_idx} ('{subtask.skill_description}') "
            f"is past halfway ({elapsed} steps, ratio: {failure_info['ratio']}). "
            f"BDDL is inconclusive. Demo reference frames are available -- "
            f"use get_demo_reference_frames to visually compare the current "
            f"snapshot with demo completion states."
        )
    else:
        recommendation = "continue"
        reasoning = (
            f"Subtask {subtask.skill_idx} ('{subtask.skill_description}') "
            f"is in progress. {elapsed} steps elapsed "
            f"(threshold: {failure_info['failure_threshold']}, "
            f"ratio: {failure_info['ratio']}). "
            f"BDDL: {bddl_evidence.get('reasoning', 'no evidence')}."
        )

    # Subtask info for the agent
    subtask_info = {
        "skill_idx": subtask.skill_idx,
        "skill_description": subtask.skill_description,
        "skill_type": subtask.skill_type,
        "frame_start": subtask.frame_start,
        "frame_end": subtask.frame_end,
        "expected_duration": subtask.frame_duration,
        **get_rollout_subtask_object_refs(subtask),
    }

    return {
        "recommendation": recommendation,
        "elapsed_steps": elapsed,
        "failure_threshold": failure_info["failure_threshold"],
        "over_threshold": failure_info["is_failure"],
        "bddl_evidence": bddl_evidence,
        "demo_evidence": demo_evidence,
        "subtask_info": subtask_info,
        "memory_summary": memory.to_dict(),
        "reasoning": reasoning,
    }
