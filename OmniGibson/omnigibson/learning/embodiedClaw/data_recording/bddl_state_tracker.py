"""
BDDL predicate state tracker for BEHAVIOR-1K episodes.

Monitors goal conditions, binary/unary predicates, and grasp state across
task-relevant objects, recording every state transition with its timestep.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from omnigibson.object_states import (
    Cooked,
    Frozen,
    Inside,
    NextTo,
    OnTop,
    Open,
    ToggledOn,
    Touching,
    Under,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Predicate registries
# ---------------------------------------------------------------------------

BINARY_STATE_CLASSES: Dict[str, type] = {
    "Inside": Inside,
    "OnTop": OnTop,
    "NextTo": NextTo,
    "Touching": Touching,
    "Under": Under,
}

UNARY_STATE_CLASSES: Dict[str, type] = {
    "Open": Open,
    "ToggledOn": ToggledOn,
    "Cooked": Cooked,
    "Frozen": Frozen,
}


class BDDLStateTracker:
    """Tracks BDDL predicate state transitions during an episode.

    Records the exact timestep of every state change for goal conditions,
    grasping, pairwise spatial predicates, and unary object states.  All
    results are JSON-serializable so they can be written directly to disk.
    """

    def __init__(self) -> None:
        # Public accumulation buffers
        self.transitions: List[Dict[str, Any]] = []
        self.grasp_history: List[Dict[str, Any]] = []

        # Internal bookkeeping
        self._tracked_predicates: List[Dict[str, Any]] = []
        self._prev_states: Dict[str, Any] = {}
        self._goal_heads: list = []
        self._parsed_goal_conditions: List[Dict[str, Any]] = []
        self._task_objects: Dict[str, Any] = {}  # scope_name -> obj
        self._robot: Optional[Any] = None
        self._prev_grasp: Dict[str, Optional[str]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, env: Any, robot: Any) -> None:
        """Initialize tracking for a new episode.

        Args:
            env: The OmniGibson environment (must have ``env.task``).
            robot: The robot instance (must expose ``arm_names`` and
                ``_ag_obj_in_hand``).
        """
        # Reset all state
        self.transitions = []
        self.grasp_history = []
        self._tracked_predicates = []
        self._prev_states = {}
        self._goal_heads = []
        self._parsed_goal_conditions = []
        self._task_objects = {}
        self._robot = robot
        self._prev_grasp = {}

        task = env.task

        # 1. Parse goal conditions from ground_goal_state_options[0]
        self._init_goal_conditions(task)

        # 2. Collect task-relevant objects from object_scope
        self._init_task_objects(task)

        # 3. Initialize grasp tracking per arm
        self._init_grasp_tracking(robot)

        # 4. Initialize pairwise binary predicates between task objects
        self._init_binary_predicates()

        # 5. Initialize unary predicates on task objects
        self._init_unary_predicates()

        logger.info(
            "BDDLStateTracker: Tracking %d predicates "
            "(%d goal, %d task objects)",
            len(self._tracked_predicates),
            len(self._goal_heads),
            len(self._task_objects),
        )

    def step(self, env: Any, robot: Any, step_idx: int) -> None:
        """Re-evaluate all tracked predicates and record any changes.

        Args:
            env: The OmniGibson environment.
            robot: The robot instance.
            step_idx: The current simulation step index.
        """
        self._step_goal_conditions(step_idx)
        self._step_grasp(robot, step_idx)
        self._step_binary_predicates(step_idx)
        self._step_unary_predicates(step_idx)

    def get_results(self) -> Dict[str, Any]:
        """Return a JSON-serializable summary of all tracking data.

        Returns:
            dict with keys:
                - ``goal_conditions``: parsed goal predicates
                - ``transitions``: list of all predicate state transitions
                - ``final_state``: current value of every tracked predicate
                - ``grasp_history``: list of grasp state changes
        """
        # Build the final_state snapshot from current prev_states
        final_state: Dict[str, Any] = {}
        for pred_info in self._tracked_predicates:
            pred_id = pred_info["id"]
            final_state[pred_id] = self._prev_states.get(pred_id)

        return {
            "goal_conditions": self._parsed_goal_conditions,
            "transitions": self.transitions,
            "final_state": final_state,
            "grasp_history": self.grasp_history,
        }

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _init_goal_conditions(self, task: Any) -> None:
        """Parse goal conditions from the task's ground_goal_state_options."""
        try:
            if not (hasattr(task, "ground_goal_state_options") and task.ground_goal_state_options):
                return
            self._goal_heads = list(task.ground_goal_state_options[0])
            for i, head in enumerate(self._goal_heads):
                pred_id = f"goal_{i}"
                predicate_name = "unknown"
                args: List[str] = []
                try:
                    if isinstance(head.body, (list, tuple)) and len(head.body) > 0:
                        predicate_name = str(head.body[0])
                        args = [str(a) for a in head.body[1:]]
                except Exception:
                    pass

                self._tracked_predicates.append({
                    "id": pred_id,
                    "type": "goal_condition",
                    "predicate": predicate_name,
                    "args": args,
                })
                self._parsed_goal_conditions.append({
                    "index": i,
                    "predicate": predicate_name,
                    "args": args,
                })
                try:
                    self._prev_states[pred_id] = bool(head.evaluate())
                except Exception:
                    self._prev_states[pred_id] = False
        except Exception as exc:
            logger.warning("BDDLStateTracker: Failed to parse goal conditions: %s", exc)

    def _init_task_objects(self, task: Any) -> None:
        """Collect non-agent, non-system task objects from object_scope."""
        try:
            if not hasattr(task, "object_scope"):
                return
            for inst_name, entity in task.object_scope.items():
                if entity is None or entity.is_system or not entity.exists:
                    continue
                obj = entity.wrapped_obj
                if obj is not None and inst_name != "agent.n.01_1":
                    self._task_objects[inst_name] = obj
        except Exception as exc:
            logger.warning("BDDLStateTracker: Failed to collect task objects: %s", exc)

    def _init_grasp_tracking(self, robot: Any) -> None:
        """Set up grasp tracking for each robot arm."""
        try:
            for arm in robot.arm_names:
                pred_id = f"grasp_{arm}"
                self._tracked_predicates.append({
                    "id": pred_id,
                    "type": "grasp",
                    "arm": arm,
                })
                obj = robot._ag_obj_in_hand.get(arm)
                val = obj.name if obj is not None else None
                self._prev_states[pred_id] = val
                self._prev_grasp[arm] = val
        except Exception as exc:
            logger.warning("BDDLStateTracker: Failed to parse grasp states: %s", exc)

    def _init_binary_predicates(self) -> None:
        """Register pairwise binary predicates between all task objects."""
        for inst_a, obj_a in self._task_objects.items():
            for inst_b, obj_b in self._task_objects.items():
                if inst_a == inst_b:
                    continue
                for state_name, state_cls in BINARY_STATE_CLASSES.items():
                    if state_cls not in obj_a.states:
                        continue
                    pred_id = f"{state_name}_{inst_a}_{inst_b}"
                    self._tracked_predicates.append({
                        "id": pred_id,
                        "type": "binary_state",
                        "predicate": state_name,
                        "args": [inst_a, inst_b],
                    })
                    try:
                        self._prev_states[pred_id] = bool(
                            obj_a.states[state_cls].get_value(obj_b)
                        )
                    except Exception:
                        self._prev_states[pred_id] = False

    def _init_unary_predicates(self) -> None:
        """Register unary predicates for all task objects that support them."""
        for inst_name, obj in self._task_objects.items():
            for state_name, state_cls in UNARY_STATE_CLASSES.items():
                if state_cls not in obj.states:
                    continue
                pred_id = f"{state_name}_{inst_name}"
                self._tracked_predicates.append({
                    "id": pred_id,
                    "type": "unary_state",
                    "predicate": state_name,
                    "args": [inst_name],
                })
                try:
                    self._prev_states[pred_id] = bool(
                        obj.states[state_cls].get_value()
                    )
                except Exception:
                    self._prev_states[pred_id] = False

    # ------------------------------------------------------------------
    # Per-step evaluation helpers
    # ------------------------------------------------------------------

    def _step_goal_conditions(self, step_idx: int) -> None:
        """Check each goal-condition head for value changes."""
        for i, head in enumerate(self._goal_heads):
            pred_id = f"goal_{i}"
            try:
                new_val = bool(head.evaluate())
            except Exception:
                continue
            old_val = self._prev_states.get(pred_id)
            if new_val != old_val:
                self.transitions.append({
                    "step_idx": step_idx,
                    "predicate_id": pred_id,
                    "predicate_name": self._parsed_goal_conditions[i]["predicate"],
                    "args": self._parsed_goal_conditions[i]["args"],
                    "old_value": old_val,
                    "new_value": new_val,
                })
                self._prev_states[pred_id] = new_val

    def _step_grasp(self, robot: Any, step_idx: int) -> None:
        """Check for grasp state changes on each arm."""
        for arm in robot.arm_names:
            pred_id = f"grasp_{arm}"
            if pred_id not in self._prev_states:
                continue
            try:
                obj = robot._ag_obj_in_hand.get(arm)
                new_val: Optional[str] = obj.name if obj is not None else None
            except Exception:
                continue
            old_val = self._prev_states.get(pred_id)
            if new_val != old_val:
                change: Dict[str, Any] = {
                    "step_idx": step_idx,
                    "arm": arm,
                    "old_value": old_val,
                    "new_value": new_val,
                }
                self.grasp_history.append(change)
                self.transitions.append({
                    "step_idx": step_idx,
                    "predicate_id": pred_id,
                    "predicate_name": "Grasp",
                    "obj_a": new_val or old_val,
                    "old_value": old_val,
                    "new_value": new_val,
                })
                self._prev_states[pred_id] = new_val

    def _step_binary_predicates(self, step_idx: int) -> None:
        """Re-evaluate all registered binary predicates."""
        for pred_info in self._tracked_predicates:
            if pred_info["type"] != "binary_state":
                continue
            pred_id = pred_info["id"]
            inst_a, inst_b = pred_info["args"]
            state_name = pred_info["predicate"]
            state_cls = BINARY_STATE_CLASSES[state_name]
            obj_a = self._task_objects.get(inst_a)
            obj_b = self._task_objects.get(inst_b)
            if obj_a is None or obj_b is None:
                continue
            try:
                new_val = bool(obj_a.states[state_cls].get_value(obj_b))
            except Exception:
                continue
            old_val = self._prev_states.get(pred_id)
            if new_val != old_val:
                self.transitions.append({
                    "step_idx": step_idx,
                    "predicate_id": pred_id,
                    "predicate_name": state_name,
                    "obj_a": inst_a,
                    "obj_b": inst_b,
                    "old_value": old_val,
                    "new_value": new_val,
                })
                self._prev_states[pred_id] = new_val

    def _step_unary_predicates(self, step_idx: int) -> None:
        """Re-evaluate all registered unary predicates."""
        for pred_info in self._tracked_predicates:
            if pred_info["type"] != "unary_state":
                continue
            pred_id = pred_info["id"]
            inst_name = pred_info["args"][0]
            state_name = pred_info["predicate"]
            state_cls = UNARY_STATE_CLASSES[state_name]
            obj = self._task_objects.get(inst_name)
            if obj is None:
                continue
            try:
                new_val = bool(obj.states[state_cls].get_value())
            except Exception:
                continue
            old_val = self._prev_states.get(pred_id)
            if new_val != old_val:
                self.transitions.append({
                    "step_idx": step_idx,
                    "predicate_id": pred_id,
                    "predicate_name": state_name,
                    "obj_a": inst_name,
                    "old_value": old_val,
                    "new_value": new_val,
                })
                self._prev_states[pred_id] = new_val
