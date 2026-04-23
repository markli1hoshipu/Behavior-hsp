"""
gt_task_executor.py - Ground-Truth Task Executor

This module provides a ground-truth based task executor that replaces ML policies
for data generation. It uses simulator knowledge (known maps, object poses, task goals)
to generate realistic physics-based robot trajectories.

The key idea is to:
1. Parse BDDL goals from the task
2. Convert them into sequences of action primitives
3. Execute primitives using StarterSemanticActionPrimitives (physics-based)
4. Return actions step-by-step matching the policy.forward() interface
"""

import logging
import numpy as np
import torch as th
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple, Set

from omnigibson import object_states
from omnigibson.action_primitives.starter_semantic_action_primitives import (
    StarterSemanticActionPrimitives,
    StarterSemanticActionPrimitiveSet,
)
from omnigibson.action_primitives.symbolic_semantic_action_primitives import (
    SymbolicSemanticActionPrimitives,
    SymbolicSemanticActionPrimitiveSet,
)
from omnigibson.action_primitives.action_primitive_set_base import ActionPrimitiveError
from omnigibson.tasks.behavior_task import BehaviorTask
from omnigibson.robots import BaseRobot
from omnigibson.objects.object_base import BaseObject

# Create module logger
logger = logging.getLogger("gt_task_executor")
logger.setLevel(logging.INFO)


@dataclass
class PrimitiveSpec:
    """Specification for an action primitive to execute."""
    primitive_type: StarterSemanticActionPrimitiveSet
    target_object: Any  # BDDLEntity or BaseObject
    secondary_object: Optional[Any] = None  # For binary predicates like place_on_top
    is_symbolic: bool = False  # Use symbolic primitives instead of physics-based


class BDDLGoalParser:
    """
    Parses BDDL goal conditions and converts them into sequences of action primitives.

    Uses ground_goal_state_options which provides flattened, grounded versions of goals.
    Each element is a HEAD object containing an AtomicFormula or Negation.
    """

    # Map BDDL predicates to primitive types
    PREDICATE_TO_PRIMITIVE = {
        "ontop": StarterSemanticActionPrimitiveSet.PLACE_ON_TOP,
        "inside": StarterSemanticActionPrimitiveSet.PLACE_INSIDE,
        "open": StarterSemanticActionPrimitiveSet.OPEN,
        "toggled_on": StarterSemanticActionPrimitiveSet.TOGGLE_ON,
        "nextto": StarterSemanticActionPrimitiveSet.PLACE_ON_TOP,  # Place nearby
    }

    # Predicates that require symbolic execution (not physics-based)
    SYMBOLIC_PREDICATES = {
        "covered", "saturated", "cooked", "burnt", "frozen", "hot", "on_fire",
        "folded", "unfolded", "draped", "overlaid"
    }

    def __init__(self, task: BehaviorTask):
        """
        Initialize the BDDL goal parser.

        Args:
            task: The BehaviorTask containing goal conditions and object scope
        """
        self.task = task
        self.object_scope = task.object_scope
        self.ground_goal_state_options = task.ground_goal_state_options

        logger.info(f"BDDLGoalParser initialized for task: {task.activity_name}")
        logger.info(f"Object scope has {len(self.object_scope)} objects")
        logger.info(f"Found {len(self.ground_goal_state_options)} ground goal state options")

    def get_unsatisfied_conditions(self, grounding_idx: int = 0) -> List:
        """
        Get list of unsatisfied goal conditions from a specific grounding.

        Args:
            grounding_idx: Which ground_goal_state_option to use (default: first/shortest)

        Returns:
            List of unsatisfied HEAD conditions
        """
        if grounding_idx >= len(self.ground_goal_state_options):
            logger.warning(f"Grounding index {grounding_idx} out of range, using 0")
            grounding_idx = 0

        grounding = self.ground_goal_state_options[grounding_idx]
        unsatisfied = []

        for condition in grounding:
            if not condition.evaluate():
                unsatisfied.append(condition)

        logger.info(f"Found {len(unsatisfied)} unsatisfied conditions out of {len(grounding)}")
        return unsatisfied

    def _parse_condition(self, condition) -> Tuple[str, List[Any], bool]:
        """
        Parse a HEAD condition to extract predicate name, objects, and polarity.

        Args:
            condition: A HEAD object from ground_goal_state_options

        Returns:
            Tuple of (predicate_name, list_of_objects, is_positive)
        """
        # Get the body of the condition
        body = condition.body

        # Check if it's negated
        is_positive = True
        if body[0] == "not":
            is_positive = False
            body = body[1]

        # Extract predicate name (first element)
        predicate_name = body[0]

        # Extract object instances from terms
        objects = []
        for term in condition.terms:
            if term in condition.scope and condition.scope[term] is not None:
                obj = condition.scope[term]
                # Get the actual entity from BDDLEntity wrapper
                if hasattr(obj, 'entity'):
                    objects.append(obj.entity)
                else:
                    objects.append(obj)

        return predicate_name, objects, is_positive

    def _condition_to_primitives(self, condition) -> List[PrimitiveSpec]:
        """
        Convert a single BDDL condition to a list of primitive specifications.

        Args:
            condition: A HEAD condition to convert

        Returns:
            List of PrimitiveSpec objects to execute
        """
        predicate_name, objects, is_positive = self._parse_condition(condition)
        primitives = []

        logger.debug(f"Converting condition: {predicate_name}({[getattr(o, 'name', str(o)) for o in objects]}), positive={is_positive}")

        # Handle negated conditions (e.g., not stained)
        if not is_positive:
            # For negative conditions, we may need symbolic execution
            logger.info(f"Skipping negated condition: not {predicate_name}")
            return primitives

        # Check if this requires symbolic execution
        is_symbolic = predicate_name in self.SYMBOLIC_PREDICATES

        # Map predicate to primitive(s)
        if predicate_name == "ontop":
            # ontop(A, B): grasp A, navigate to B, place A on top of B
            if len(objects) >= 2:
                obj_to_move, target_surface = objects[0], objects[1]
                primitives.append(PrimitiveSpec(
                    primitive_type=StarterSemanticActionPrimitiveSet.GRASP,
                    target_object=obj_to_move
                ))
                primitives.append(PrimitiveSpec(
                    primitive_type=StarterSemanticActionPrimitiveSet.PLACE_ON_TOP,
                    target_object=target_surface
                ))

        elif predicate_name == "inside":
            # inside(A, B): optionally open B, grasp A, navigate to B, place A inside B
            if len(objects) >= 2:
                obj_to_move, container = objects[0], objects[1]

                # Check if container is openable and closed
                if hasattr(container, 'states') and object_states.Open in container.states:
                    if not container.states[object_states.Open].get_value():
                        primitives.append(PrimitiveSpec(
                            primitive_type=StarterSemanticActionPrimitiveSet.OPEN,
                            target_object=container
                        ))

                primitives.append(PrimitiveSpec(
                    primitive_type=StarterSemanticActionPrimitiveSet.GRASP,
                    target_object=obj_to_move
                ))
                primitives.append(PrimitiveSpec(
                    primitive_type=StarterSemanticActionPrimitiveSet.PLACE_INSIDE,
                    target_object=container
                ))

        elif predicate_name == "open":
            # open(A): navigate to A, open A
            if len(objects) >= 1:
                primitives.append(PrimitiveSpec(
                    primitive_type=StarterSemanticActionPrimitiveSet.OPEN,
                    target_object=objects[0]
                ))

        elif predicate_name == "toggled_on":
            # toggled_on(A): navigate to A, toggle A on
            if len(objects) >= 1:
                primitives.append(PrimitiveSpec(
                    primitive_type=StarterSemanticActionPrimitiveSet.TOGGLE_ON,
                    target_object=objects[0]
                ))

        elif predicate_name == "nextto":
            # nextto(A, B): grasp A, place A near B
            if len(objects) >= 2:
                obj_to_move, target = objects[0], objects[1]
                primitives.append(PrimitiveSpec(
                    primitive_type=StarterSemanticActionPrimitiveSet.GRASP,
                    target_object=obj_to_move
                ))
                # Place on a surface near the target
                primitives.append(PrimitiveSpec(
                    primitive_type=StarterSemanticActionPrimitiveSet.PLACE_ON_TOP,
                    target_object=target
                ))

        elif is_symbolic:
            # For symbolic predicates, log a warning
            logger.warning(f"Symbolic predicate '{predicate_name}' not yet implemented")

        else:
            logger.warning(f"Unknown predicate: {predicate_name}")

        return primitives

    def get_primitive_sequence(self, grounding_idx: int = 0) -> List[PrimitiveSpec]:
        """
        Convert all unsatisfied goals to an ordered sequence of primitives.

        Args:
            grounding_idx: Which ground_goal_state_option to use

        Returns:
            Ordered list of PrimitiveSpec objects to execute
        """
        unsatisfied = self.get_unsatisfied_conditions(grounding_idx)
        all_primitives = []

        for condition in unsatisfied:
            primitives = self._condition_to_primitives(condition)
            all_primitives.extend(primitives)

        # Resolve dependencies and optimize order
        optimized = self._resolve_dependencies(all_primitives)

        logger.info(f"Generated {len(optimized)} primitives from {len(unsatisfied)} conditions")
        return optimized

    def _resolve_dependencies(self, primitives: List[PrimitiveSpec]) -> List[PrimitiveSpec]:
        """
        Resolve dependencies between primitives and optimize execution order.

        Rules:
        1. Must grasp before place
        2. Must navigate before grasp (handled by primitives internally)
        3. Must open before placing inside
        4. Group operations by object location to minimize navigation

        Args:
            primitives: Unordered list of primitives

        Returns:
            Ordered list with dependencies resolved
        """
        # For now, return as-is since primitives are already generated in dependency order
        # TODO: Implement smarter ordering for efficiency (minimize navigation)
        return primitives


class GroundTruthTaskExecutor:
    """
    Ground-truth based task executor that replaces ML policies for data generation.

    Uses BDDLGoalParser to understand task goals and StarterSemanticActionPrimitives
    to execute physics-based robot motions. The interface matches policy.forward().
    """

    def __init__(
        self,
        env,
        robot: BaseRobot,
        task: BehaviorTask,
        use_symbolic_for_unsupported: bool = True,
        max_retries: int = 3,
    ):
        """
        Initialize the ground-truth task executor.

        Args:
            env: The OmniGibson environment
            robot: The robot instance
            task: The BehaviorTask containing goal conditions
            use_symbolic_for_unsupported: Use symbolic primitives for unsupported predicates
            max_retries: Maximum retries per primitive on failure
        """
        self.env = env
        self.robot = robot
        self.task = task
        self.use_symbolic_for_unsupported = use_symbolic_for_unsupported
        self.max_retries = max_retries

        # Initialize action primitives
        logger.info("Initializing StarterSemanticActionPrimitives...")
        self.primitives = StarterSemanticActionPrimitives(
            env=env,
            robot=robot,
            enable_head_tracking=True,
            task_relevant_objects_only=True,
        )

        # Optional symbolic primitives for unsupported operations
        if use_symbolic_for_unsupported:
            logger.info("Initializing SymbolicSemanticActionPrimitives for unsupported predicates...")
            self.symbolic_primitives = SymbolicSemanticActionPrimitives(env=env, robot=robot)
        else:
            self.symbolic_primitives = None

        # Initialize goal parser
        self.parser = BDDLGoalParser(task)

        # Execution state
        self.action_generator: Optional[Generator] = None
        self.current_primitive_idx = 0
        self.primitive_sequence: List[PrimitiveSpec] = []
        self.completed = False
        self._idle_action = None

        logger.info("GroundTruthTaskExecutor initialized successfully")

    def reset(self):
        """Reset the executor state for a new episode."""
        logger.info("Resetting GroundTruthTaskExecutor")
        self.action_generator = None
        self.current_primitive_idx = 0
        self.primitive_sequence = []
        self.completed = False
        self.primitives.reset() if hasattr(self.primitives, 'reset') else None

        # Re-parse goals
        self.primitive_sequence = self.parser.get_primitive_sequence()
        logger.info(f"Reset complete. {len(self.primitive_sequence)} primitives to execute")

    def _get_idle_action(self) -> th.Tensor:
        """Get an idle action (no movement)."""
        if self._idle_action is None:
            # Create action with same shape as robot action space
            action_dim = self.robot.action_dim
            self._idle_action = th.zeros(action_dim)
        return self._idle_action.clone()

    def _plan_and_execute(self) -> Generator[th.Tensor, None, None]:
        """
        Generator that plans the task and yields actions step by step.

        Yields:
            Action tensors for each simulation step
        """
        # Parse goals if not already done
        if not self.primitive_sequence:
            self.primitive_sequence = self.parser.get_primitive_sequence()

        logger.info(f"Starting execution of {len(self.primitive_sequence)} primitives")

        for idx, spec in enumerate(self.primitive_sequence):
            self.current_primitive_idx = idx
            logger.info(f"Executing primitive {idx + 1}/{len(self.primitive_sequence)}: "
                       f"{spec.primitive_type.name} on {getattr(spec.target_object, 'name', str(spec.target_object))}")

            # Choose which primitive set to use
            if spec.is_symbolic and self.symbolic_primitives:
                primitive_set = self.symbolic_primitives
                primitive_type = SymbolicSemanticActionPrimitiveSet[spec.primitive_type.name]
            else:
                primitive_set = self.primitives
                primitive_type = spec.primitive_type

            # Execute with retries
            success = False
            for attempt in range(self.max_retries):
                try:
                    # Get the target object (extract entity from BDDLEntity if needed)
                    target_obj = spec.target_object
                    if hasattr(target_obj, 'entity'):
                        target_obj = target_obj.entity

                    # Execute primitive
                    yield from primitive_set.apply_ref(primitive_type, target_obj)
                    success = True
                    logger.info(f"Primitive {idx + 1} completed successfully")
                    break

                except ActionPrimitiveError as e:
                    logger.warning(f"Primitive {idx + 1} failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                    if attempt < self.max_retries - 1:
                        # Yield a few idle actions before retry
                        for _ in range(10):
                            yield self._get_idle_action()

            if not success:
                logger.error(f"Primitive {idx + 1} failed after {self.max_retries} attempts, continuing...")

        self.completed = True
        logger.info("All primitives executed. Task execution complete.")

        # Yield idle actions indefinitely after completion
        while True:
            yield self._get_idle_action()

    def forward(self, obs: Dict[str, Any]) -> th.Tensor:
        """
        Get the next action for the current step.

        This matches the interface of ML policies: policy.forward(obs) -> action

        Args:
            obs: Current observation dictionary (not used in GT execution)

        Returns:
            Action tensor for the current step
        """
        # Initialize generator if needed
        if self.action_generator is None:
            self.reset()
            self.action_generator = self._plan_and_execute()

        # Get next action from generator
        try:
            action = next(self.action_generator)
            if action is None:
                action = self._get_idle_action()
            return action
        except StopIteration:
            self.completed = True
            return self._get_idle_action()

    @property
    def is_completed(self) -> bool:
        """Check if task execution is complete."""
        return self.completed

    def get_progress(self) -> Tuple[int, int]:
        """Get current progress (current_primitive, total_primitives)."""
        return self.current_primitive_idx, len(self.primitive_sequence)
