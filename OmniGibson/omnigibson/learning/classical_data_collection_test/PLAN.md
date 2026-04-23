# Ground-Truth Based Data Generation for BEHAVIOR-1K

## Overview

Replace the ML policy in `eval_data_gen_par.py` with a ground-truth task planner that uses simulator knowledge (known maps, object poses, task goals) to generate realistic physics-based robot trajectories.

## File Organization

All new/modified files go in a separate folder to preserve existing functionality:
```
OmniGibson/omnigibson/learning/classical_data_collection_test/
├── PLAN.md                     # This plan document
├── gt_task_executor.py         # NEW: Ground-truth task executor
├── eval_data_gen_gt.py         # COPY: Modified eval_data_gen_par.py
└── configs/
    └── gt_config.yaml          # COPY: Modified base_config.yaml
```

## Architecture

```
[BDDL Goals] --> [GroundTruthTaskExecutor] --> [StarterSemanticActionPrimitives] --> [Robot]
                        |                              |
                  BDDLGoalParser                  CuRobo Motion Planning
                        |                              |
                  DependencyResolver              Physics Simulation
```

## Key Components to Create

### 1. `GroundTruthTaskExecutor` (New Class)
**File**: `classical_data_collection_test/gt_task_executor.py`

Main controller that replaces ML policy. Interface matches `policy.forward()`:

```python
class GroundTruthTaskExecutor:
    def __init__(self, env, robot, task: BehaviorTask):
        self.primitives = StarterSemanticActionPrimitives(env, robot)
        self.parser = BDDLGoalParser(task)
        self.action_generator = None

    def forward(self, obs) -> np.ndarray:
        """Return action for current step"""
        if self.action_generator is None:
            self.action_generator = self._plan_and_execute()
        try:
            return next(self.action_generator)
        except StopIteration:
            return self._idle_action()

    def reset(self):
        self.action_generator = None
```

### 2. `BDDLGoalParser` (New Class)
**File**: `classical_data_collection_test/gt_task_executor.py`

Converts BDDL goals into primitive sequences:

```python
class BDDLGoalParser:
    def __init__(self, task: BehaviorTask):
        self.task = task
        self.object_scope = task.object_scope
        self.ground_goal_state_options = task.ground_goal_state_options

    def get_primitive_sequence(self) -> List[Tuple]:
        """Convert unsatisfied goals to ordered primitive sequence"""
        # 1. Get unsatisfied conditions from first grounding option
        # 2. Map each condition to primitive(s)
        # 3. Resolve dependencies (navigate before grasp, open before inside)
        # 4. Return ordered list of (primitive_type, target_obj, [secondary_obj])
```

### 3. Predicate-to-Primitive Mapping

| BDDL Predicate | Primitive Sequence |
|----------------|-------------------|
| `ontop(A, B)` | NAVIGATE_TO(A) → GRASP(A) → NAVIGATE_TO(B) → PLACE_ON_TOP(B) |
| `inside(A, B)` | [OPEN(B) if needed] → NAVIGATE_TO(A) → GRASP(A) → NAVIGATE_TO(B) → PLACE_INSIDE(B) |
| `open(A)` | NAVIGATE_TO(A) → OPEN(A) |
| `toggled_on(A)` | NAVIGATE_TO(A) → TOGGLE_ON(A) |
| `nextto(A, B)` | NAVIGATE_TO(A) → GRASP(A) → NAVIGATE_TO(B) → PLACE_ON_TOP(nearby surface) |

### 4. Dependency Resolution

Order primitives based on:
1. **Physical constraints**: Must grasp before place, navigate before grasp
2. **State prerequisites**: Open container before placing inside
3. **Resource constraints**: Release object before grasping another (single arm)
4. **Spatial efficiency**: Minimize navigation by grouping nearby operations

## Files to Create/Copy

### New Files
| File | Description |
|------|-------------|
| `classical_data_collection_test/gt_task_executor.py` | New ground-truth executor and parser |
| `classical_data_collection_test/PLAN.md` | Copy of this plan |

### Copied & Modified Files
| Original | Copy To | Changes |
|----------|---------|---------|
| `learning/eval_data_gen_par.py` | `classical_data_collection_test/eval_data_gen_gt.py` | Replace policy with GT executor |
| `learning/configs/base_config.yaml` | `classical_data_collection_test/configs/gt_config.yaml` | Add GT-specific config options |

## Modifications to `eval_data_gen_gt.py`

### Change 1: Import GT executor
```python
from omnigibson.learning.classical_data_collection_test.gt_task_executor import GroundTruthTaskExecutor
```

### Change 2: Replace policy loading (around line 497-505)
```python
def load_policy(self) -> Any:
    """Loads and returns the ground-truth task executor."""
    executor = GroundTruthTaskExecutor(self.env, self.robot, self.env.task)
    logger.info("Loaded GroundTruthTaskExecutor (no ML policy)")
    return executor
```

## Ground Truth Sources Used

| Information | Source | Location |
|-------------|--------|----------|
| Task goals | `task.ground_goal_state_options` | `behavior_task.py:328-330` |
| Object poses | `obj.get_position_orientation()` | `entity_prim.py` |
| Navigation paths | `scene.trav_map.get_shortest_path()` | `traversable_map.py:166-207` |
| Grasp poses | `get_grasp_poses_for_object_sticky()` | `grasping_planning_utils.py` |
| Object states | `obj.states[StateType].get_value()` | Various in `object_states/` |

## Implementation Phases

### Phase 1: Core Framework
- [ ] Create `classical_data_collection_test/` folder
- [ ] Create `gt_task_executor.py` with `GroundTruthTaskExecutor` and `BDDLGoalParser`
- [ ] Implement basic predicate mapping for `ontop`, `inside`, `open`, `toggled_on`
- [ ] Copy and modify `eval_data_gen_par.py` → `eval_data_gen_gt.py`

### Phase 2: Dependency Resolution
- [ ] Implement dependency graph for primitive ordering
- [ ] Handle implicit prerequisites (open before inside)
- [ ] Add dual-arm coordination for R1Pro

### Phase 3: Extended Predicates
- [ ] Add particle-based predicates (`covered`, `saturated`) using `SymbolicSemanticActionPrimitives`
- [ ] Add temperature predicates (`cooked`, `frozen`) with waiting logic
- [ ] Add special predicates (`attached`, `sliced`)

### Phase 4: Robustness
- [ ] Implement retry strategies on primitive failures
- [ ] Add re-planning when conditions change mid-execution
- [ ] Handle edge cases (unreachable objects, blocked paths)

## Critical Reference Files (Read-Only)

| File | Purpose |
|------|---------|
| `starter_semantic_action_primitives.py` | Physics-based primitives to use (`apply_ref()` at line 295) |
| `behavior_task.py` | Source of `ground_goal_state_options` (line 328-330) |
| `symbolic_semantic_action_primitives.py` | For state-change primitives (wipe, cut, soak) |
| `grasping_planning_utils.py` | Grasp pose generation |
| `traversable_map.py` | Navigation path planning |

## Verification Plan

1. **Unit tests**: Test `BDDLGoalParser` on simple tasks (e.g., `setting_up_candles`)
2. **Integration test**: Run GT executor on single task instance, verify task completion
3. **Data quality check**: Compare GT-generated trajectories with human demos for similarity
4. **Full benchmark**: Run on all BEHAVIOR-1K tasks, measure success rate vs human success rate

## Potential Challenges

| Challenge | Mitigation |
|-----------|------------|
| Complex goal conditions (forall/exists) | `ground_goal_state_options` already flattens these |
| Navigation failures | Use `TraversableMap` for valid paths; retry with different targets |
| Grasp failures | Multiple grasp pose sampling; retry with different poses |
| State changes during execution | Periodic re-evaluation; re-plan if needed |
| Particle/temperature states | Use `SymbolicSemanticActionPrimitives` for direct state changes |

## Usage

To run with ground-truth data collection:
```bash
cd OmniGibson/omnigibson/learning/classical_data_collection_test
python eval_data_gen_gt.py task.name=setting_up_candles
```
