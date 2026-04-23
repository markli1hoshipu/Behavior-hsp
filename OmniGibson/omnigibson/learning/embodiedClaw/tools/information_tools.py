"""
Information gathering and filtering tools for the agentic data collection system.

These functions are called by the agent (Claude Code, an external process) to observe
the simulator state and filter observations to relevant subsets.

The world predicate system dynamically enumerates ALL predicates the simulator
supports for task-scope objects + robot + particle systems.  No hardcoded predicate
lists — the object's ``.states`` dict declares what it supports at runtime.
"""

import base64
import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch as th

from omnigibson.learning.embodiedClaw.object_matching import (
    ObjectMatchQuery,
    arg_matches_query,
    build_match_query,
    entity_matches_query,
)
from omnigibson.learning.utils.eval_utils import ROBOT_CAMERA_NAMES
from omnigibson.object_states.object_state_base import (
    AbsoluteObjectState,
    RelativeObjectState,
)

logger = logging.getLogger(__name__)

PREDICATE_DELTA_FIELDS = (
    "predicate_deltas_since_subtask_start",
    "predicate_deltas_since_previous_decision",
)


# ------------------------------------------------------------------
# JSON conversion helper
# ------------------------------------------------------------------

def _to_json(value: Any) -> Any:
    """Convert a simulator value to a JSON-serializable Python object."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, th.Tensor):
        return value.cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set):
        # Set of objects — extract names, sorted for determinism
        names = []
        for item in value:
            if hasattr(item, "name"):
                names.append(str(item.name))
            else:
                names.append(str(item))
        return sorted(names)
    if isinstance(value, tuple):
        return [_to_json(v) for v in value]
    if isinstance(value, list):
        return [_to_json(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_json(v) for k, v in value.items()}
    # Fallback
    return str(value)


# ------------------------------------------------------------------
# Entity info helper
# ------------------------------------------------------------------

def _entity_info(scope_name: str, obj: Any) -> Dict[str, Any]:
    """Build an arg dict for a predicate entry."""
    scene_name = None
    if scope_name == "robot":
        scene_name = "robot"
    elif scope_name.startswith("system:"):
        scene_name = None
    else:
        try:
            if hasattr(obj, "name"):
                scene_name = str(obj.name)
        except Exception:
            pass
    return {"scope_name": scope_name, "scene_name": scene_name}


# ------------------------------------------------------------------
# World predicates
# ------------------------------------------------------------------

def _predicate_key(
    predicate: Dict[str, Any],
) -> Tuple[str, Tuple[Tuple[str, str], ...]]:
    """Return a stable identity key for a grounded predicate."""
    args = []
    for arg in predicate.get("args", []):
        args.append((
            str(arg.get("scope_name") or ""),
            str(arg.get("scene_name") or ""),
        ))
    return str(predicate.get("predicate", "")), tuple(args)


def _predicate_map(
    snapshot: Optional[Dict[str, Any]],
) -> Dict[Tuple[str, Tuple[Tuple[str, str], ...]], Dict[str, Any]]:
    """Index a snapshot's predicates by stable predicate identity."""
    if snapshot is None:
        return {}
    result = {}
    for predicate in snapshot.get("world_predicates", []):
        if "value" not in predicate:
            continue
        result[_predicate_key(predicate)] = predicate
    return result


def diff_world_predicates(
    baseline_snapshot: Optional[Dict[str, Any]],
    current_snapshot: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Return predicate value changes from ``baseline_snapshot`` to current.

    The diff is intentionally value-based and non-destructive: callers should
    keep current ``world_predicates`` and use this as additional evidence.
    """
    if baseline_snapshot is None:
        return []

    baseline_predicates = _predicate_map(baseline_snapshot)
    current_predicates = _predicate_map(current_snapshot)
    deltas: List[Dict[str, Any]] = []

    for key in sorted(set(baseline_predicates) | set(current_predicates)):
        old_predicate = baseline_predicates.get(key)
        new_predicate = current_predicates.get(key)
        old_value = old_predicate.get("value") if old_predicate is not None else None
        new_value = new_predicate.get("value") if new_predicate is not None else None

        if old_value == new_value:
            continue

        predicate = new_predicate if new_predicate is not None else old_predicate
        change_type = "changed"
        if old_predicate is None:
            change_type = "added"
        elif new_predicate is None:
            change_type = "removed"
        elif old_value is False and new_value is True:
            change_type = "became_true"
        elif old_value is True and new_value is False:
            change_type = "became_false"

        deltas.append({
            "predicate": predicate.get("predicate", ""),
            "args": predicate.get("args", []),
            "old_value": old_value,
            "new_value": new_value,
            "change_type": change_type,
        })

    return deltas


def add_predicate_deltas(
    snapshot: Dict[str, Any],
    *,
    subtask_start_snapshot: Optional[Dict[str, Any]] = None,
    previous_decision_snapshot: Optional[Dict[str, Any]] = None,
    current_subtask_idx: Optional[int] = None,
) -> Dict[str, Any]:
    """Attach predicate deltas to a snapshot without removing predicates."""
    enriched = dict(snapshot)
    enriched["predicate_deltas_since_subtask_start"] = diff_world_predicates(
        subtask_start_snapshot,
        snapshot,
    )
    enriched["predicate_deltas_since_previous_decision"] = diff_world_predicates(
        previous_decision_snapshot,
        snapshot,
    )
    enriched["predicate_delta_metadata"] = {
        "current_step": snapshot.get("step_count", 0),
        "current_subtask_idx": current_subtask_idx,
        "subtask_start_step": (
            subtask_start_snapshot.get("step_count")
            if subtask_start_snapshot is not None
            else None
        ),
        "previous_decision_step": (
            previous_decision_snapshot.get("step_count")
            if previous_decision_snapshot is not None
            else None
        ),
        "subtask_start_available": subtask_start_snapshot is not None,
        "previous_decision_available": previous_decision_snapshot is not None,
    }
    return enriched


def get_world_predicates(
    env: Any,
    robot: Any,
) -> List[Dict[str, Any]]:
    """Evaluate ALL predicates for task-scope objects, robot, and particle systems.

    Iterates every entity's ``.states`` dict, evaluates every unary and binary
    predicate it supports, and returns a flat list of predicate dicts.  No
    hardcoded predicate lists — the runtime declares what each entity supports.

    Args:
        env: The OmniGibson environment instance.
        robot: The robot entity in the scene.

    Returns:
        List of dicts, each with:
            - ``predicate``: state class name (e.g. ``"Inside"``, ``"Open"``)
            - ``args``: list of arg dicts with ``scope_name`` and ``scene_name``
            - ``value``: JSON-serializable evaluated value
    """
    # Collect all entities
    entities: Dict[str, Any] = {}  # scope_name -> unwrapped object

    # Task-scope objects
    try:
        if hasattr(env.task, "object_scope") and env.task.object_scope is not None:
            for scope_name, entity in env.task.object_scope.items():
                if entity is None:
                    continue
                if scope_name == "agent.n.01_1":
                    continue
                if entity.is_system or not entity.exists:
                    continue
                wrapped = entity.wrapped_obj
                if wrapped is not None:
                    entities[scope_name] = wrapped
    except Exception as e:
        logger.warning("get_world_predicates: failed to collect task objects: %s", e)

    # Robot
    entities["robot"] = robot

    # Particle systems (as potential binary predicate arguments)
    try:
        systems = getattr(env.scene, "active_systems", None)
        if systems is None:
            systems = getattr(env.scene, "systems", {})
        for sys_name, sys_obj in systems.items():
            entities[f"system:{sys_name}"] = sys_obj
    except Exception as e:
        logger.debug("get_world_predicates: no particle systems found: %s", e)

    # Evaluate all boolean predicates (skip non-predicate states like Pose, AABB)
    predicates: List[Dict[str, Any]] = []

    for name_a, obj_a in entities.items():
        if not hasattr(obj_a, "states"):
            continue

        for state_cls, state_inst in obj_a.states.items():
            state_name = state_cls.__name__

            if issubclass(state_cls, RelativeObjectState):
                # Binary predicate — evaluate against every other entity
                for name_b, obj_b in entities.items():
                    if name_a == name_b:
                        continue
                    try:
                        val = state_inst.get_value(obj_b)
                        if not isinstance(val, (bool, np.bool_)):
                            continue  # skip non-boolean results
                        predicates.append({
                            "predicate": state_name,
                            "args": [
                                _entity_info(name_a, obj_a),
                                _entity_info(name_b, obj_b),
                            ],
                            "value": bool(val),
                        })
                    except Exception:
                        pass  # incompatible pair — skip silently

            elif issubclass(state_cls, AbsoluteObjectState):
                # Unary predicate
                try:
                    val = state_inst.get_value()
                    if not isinstance(val, (bool, np.bool_)):
                        continue  # skip non-boolean (Pose, AABB, Temperature, etc.)
                    predicates.append({
                        "predicate": state_name,
                        "args": [_entity_info(name_a, obj_a)],
                        "value": bool(val),
                    })
                except Exception:
                    pass  # evaluation failed — skip silently

    return predicates


# ------------------------------------------------------------------
# Snapshot
# ------------------------------------------------------------------

def get_simulation_snapshot(
    env: Any,
    robot: Any,
    task_name: str,
    camera_names: Optional[Dict[str, str]] = None,
    obs: Optional[Dict[str, Any]] = None,
    include_images: bool = True,
) -> Dict[str, Any]:
    """Return a comprehensive snapshot of the current simulation state.

    Args:
        env: The OmniGibson environment instance.
        robot: The robot entity in the scene.
        task_name: Name of the current BEHAVIOR task.
        camera_names: Optional dict mapping short names to full sensor paths.
        obs: Optional pre-computed flattened observation dict.
        include_images: Whether to encode camera images from ``obs``.

    Returns:
        dict with keys:
            - ``images``: camera short-name -> base64-encoded PNG, or ``{}``
              when ``include_images`` is ``False``.
            - ``robot_state``: position, orientation, joints, eef_poses, grasped.
            - ``object_states``: scope_name -> {scene_name, position, orientation}.
            - ``world_predicates``: flat list of all evaluated predicates.
              The MCP layer may attach predicate-delta fields to this snapshot.
            - ``name_mapping``: scope_name -> scene_name for all task objects.
            - ``step_count``: current env step count.
            - ``task_name``: task name string.
    """
    if camera_names is None:
        camera_names = ROBOT_CAMERA_NAMES.get("R1Pro", {})

    # ------------------------------------------------------------------
    # Images
    # ------------------------------------------------------------------
    images: Dict[str, str] = {}
    if include_images and obs is not None:
        for short_name, full_camera_name in camera_names.items():
            rgb_key = f"{full_camera_name}::rgb"
            try:
                rgb_tensor = obs.get(rgb_key)
                if rgb_tensor is None:
                    continue
                if isinstance(rgb_tensor, th.Tensor):
                    rgb_np = rgb_tensor.cpu().numpy().astype(np.uint8)
                else:
                    rgb_np = np.asarray(rgb_tensor, dtype=np.uint8)
                success, buf = cv2.imencode(".png", cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))
                if success:
                    images[short_name] = base64.b64encode(buf.tobytes()).decode("ascii")
            except Exception as e:
                logger.warning("Failed to encode image for camera %s: %s", short_name, e)

    # ------------------------------------------------------------------
    # Robot state
    # ------------------------------------------------------------------
    robot_state: Dict[str, Any] = {}
    try:
        pos, quat = robot.get_position_orientation()
        robot_state["position"] = pos.cpu().tolist() if isinstance(pos, th.Tensor) else list(pos)
        robot_state["orientation"] = quat.cpu().tolist() if isinstance(quat, th.Tensor) else list(quat)
    except Exception as e:
        logger.warning("Failed to get robot position/orientation: %s", e)
        robot_state["position"] = None
        robot_state["orientation"] = None

    try:
        jpos = robot.get_joint_positions()
        robot_state["joint_positions"] = jpos.cpu().tolist() if isinstance(jpos, th.Tensor) else list(jpos)
    except Exception as e:
        logger.warning("Failed to get robot joint positions: %s", e)
        robot_state["joint_positions"] = None

    eef_poses: Dict[str, Dict[str, Any]] = {}
    try:
        for arm in robot.arm_names:
            try:
                eef_pos = robot.get_eef_position(arm=arm)
                eef_ori = robot.get_eef_orientation(arm=arm)
                eef_poses[arm] = {
                    "position": eef_pos.cpu().tolist() if isinstance(eef_pos, th.Tensor) else list(eef_pos),
                    "orientation": eef_ori.cpu().tolist() if isinstance(eef_ori, th.Tensor) else list(eef_ori),
                }
            except Exception as e:
                logger.warning("Failed to get EEF pose for arm %s: %s", arm, e)
                eef_poses[arm] = {"position": None, "orientation": None}
    except Exception as e:
        logger.warning("Failed to iterate arm names for EEF poses: %s", e)
    robot_state["eef_poses"] = eef_poses

    grasped_objects: Dict[str, Optional[str]] = {}
    try:
        for arm in robot.arm_names:
            obj_in_hand = robot._ag_obj_in_hand.get(arm)
            grasped_objects[arm] = obj_in_hand.name if obj_in_hand is not None else None
    except Exception as e:
        logger.warning("Failed to get grasped objects: %s", e)
    robot_state["grasped_objects"] = grasped_objects

    # ------------------------------------------------------------------
    # Object states (positions + scene_name, no inline predicates)
    # ------------------------------------------------------------------
    object_states: Dict[str, Dict[str, Any]] = {}
    name_mapping: Dict[str, str] = {}
    try:
        if hasattr(env.task, "object_scope") and env.task.object_scope is not None:
            for scope_name, entity in env.task.object_scope.items():
                if entity is None:
                    continue
                if scope_name == "agent.n.01_1":
                    continue
                if entity.is_system or not entity.exists:
                    continue

                obj_info: Dict[str, Any] = {}
                try:
                    wrapped = entity.wrapped_obj
                    if wrapped is not None and hasattr(wrapped, "name"):
                        obj_info["scene_name"] = str(wrapped.name)
                        name_mapping[scope_name] = str(wrapped.name)
                    else:
                        obj_info["scene_name"] = None
                except Exception:
                    obj_info["scene_name"] = None

                try:
                    wrapped = entity.wrapped_obj
                    if wrapped is not None:
                        o_pos, o_quat = wrapped.get_position_orientation()
                        obj_info["position"] = (
                            o_pos.cpu().tolist() if isinstance(o_pos, th.Tensor) else list(o_pos)
                        )
                        obj_info["orientation"] = (
                            o_quat.cpu().tolist() if isinstance(o_quat, th.Tensor) else list(o_quat)
                        )
                except Exception as e:
                    logger.debug("Could not get pose for %s: %s", scope_name, e)
                    obj_info["position"] = None
                    obj_info["orientation"] = None

                object_states[scope_name] = obj_info
    except Exception as e:
        logger.warning("Failed to collect object states: %s", e)

    # ------------------------------------------------------------------
    # World predicates (dynamic, full dump)
    # ------------------------------------------------------------------
    world_predicates = get_world_predicates(env, robot)

    # ------------------------------------------------------------------
    # Step count
    # ------------------------------------------------------------------
    step_count = 0
    try:
        if hasattr(env, "_current_step"):
            step_count = int(env._current_step)
    except Exception:
        pass

    return {
        "images": images,
        "robot_state": robot_state,
        "object_states": object_states,
        "world_predicates": world_predicates,
        "name_mapping": name_mapping,
        "step_count": step_count,
        "task_name": task_name,
    }


def strip_snapshot_images(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow snapshot copy with image payloads removed."""
    stripped = dict(snapshot)
    stripped["images"] = {}
    return stripped


def _predicate_mentions_targets(
    predicate: Dict[str, Any],
    query: ObjectMatchQuery,
) -> bool:
    """Return whether a predicate or delta arg references any target."""
    for arg in predicate.get("args", []):
        scene_name = arg.get("scene_name")
        scope_name = arg.get("scope_name")
        if scene_name == "robot" or scope_name == "robot":
            return True
        if arg_matches_query(arg, query):
            return True
    return False


def _filter_predicate_deltas(
    snapshot: Dict[str, Any],
    query: ObjectMatchQuery,
) -> Dict[str, List[Dict[str, Any]]]:
    """Filter any predicate-delta lists already attached to a snapshot."""
    filtered = {}
    for field_name in PREDICATE_DELTA_FIELDS:
        filtered[field_name] = [
            delta
            for delta in snapshot.get(field_name, [])
            if _predicate_mentions_targets(delta, query)
        ]
    return filtered


# ------------------------------------------------------------------
# Filtering
# ------------------------------------------------------------------

def filter_information(
    snapshot: Dict[str, Any],
    object_names: List[str],
    include_images: bool = True,
) -> Dict[str, Any]:
    """Filter a snapshot to only include information about specified objects.

    Matches ``object_names`` against runtime object references using a relaxed
    rollout-friendly rule:

    - exact scene names like ``"trash_can_116"``
    - stripped type names like ``"trash_can"``
    - canonicalized BDDL scope names like ``"trash_can.n.01_1"``

    This allows the skill sequence to work from annotation object types without
    depending on numeric instance ids during rollout. Predicates involving the
    robot are always preserved.

    Args:
        snapshot: A snapshot dict from :func:`get_simulation_snapshot`.
        object_names: Annotation-style object names to filter by.
        include_images: Whether to preserve image payloads in the filtered
            result.

    Returns:
        Filtered snapshot with same top-level keys. If predicate-delta fields
        are attached, they are filtered with the same relevance rule as
        ``world_predicates``.
    """
    if not object_names:
        return dict(snapshot) if include_images else strip_snapshot_images(snapshot)

    query = build_match_query(object_names)

    # Filter object_states by exact or normalized object-name match
    filtered_objects = {}
    for scope_name, state in snapshot.get("object_states", {}).items():
        scene_name = state.get("scene_name")
        if entity_matches_query(scope_name, scene_name, query):
            filtered_objects[scope_name] = state

    # Filter world_predicates with the same exact-or-normalized matching rule
    filtered_predicates = []
    for pred in snapshot.get("world_predicates", []):
        if _predicate_mentions_targets(pred, query):
            filtered_predicates.append(pred)

    filtered_deltas = _filter_predicate_deltas(snapshot, query)

    return {
        "images": snapshot.get("images", {}) if include_images else {},
        "robot_state": snapshot.get("robot_state", {}),
        "object_states": filtered_objects,
        "world_predicates": filtered_predicates,
        **filtered_deltas,
        "predicate_delta_metadata": snapshot.get("predicate_delta_metadata", {}),
        "name_mapping": snapshot.get("name_mapping", {}),
        "step_count": snapshot.get("step_count", 0),
        "task_name": snapshot.get("task_name", ""),
    }
