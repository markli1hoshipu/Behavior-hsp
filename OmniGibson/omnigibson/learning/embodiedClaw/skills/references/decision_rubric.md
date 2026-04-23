# Decision Rubric

Use this rubric instead of vague rules like "count satisfied predicates."

## Decision Order

1. Read `get_task_memory()` and `get_simulation_snapshot()`.
2. Derive `elapsed_steps` from `step_count - subtask_start_step`.
3. Infer the intended end state from `current_subtask.skill_description`, `manipulating_object_ids`, and `object_ids`.
4. Inspect all available evidence involving those entities:
   `world_predicates`, `robot_state`, `object_states`, and current images.
   Use three evidence surfaces together when they are available:
   current absolute predicates/state/images, predicate deltas since subtask
   start, and predicate deltas since the previous decision snapshot.
5. If the combined evidence clearly supports completion and there is no strong contradiction, mark success.
6. If `elapsed_steps > failure_threshold`, mark failure. Here `failure_threshold` means `2 * max_duration` across demo episodes for the current `skill_idx`.
   Exception: if a retry restore was skipped or errored, apply the retry timing guidance in `failure_recovery.md`; elapsed-step checks may be advisory until timing is rebased.
7. If combined evidence is inconclusive, demo frames exist, and the attempt is at least halfway to timeout, request visual comparison.
8. Otherwise continue.

## Subtask Interpretation Procedure

Treat the subtask text as structured language. Do not wait for an exact hard-coded rubric entry.

Infer four things:

1. What action is being attempted.
2. Which object is being changed.
3. Which target object, surface, container, or reference object matters.
4. What should become true, and what should stop being true.

Examples:

- `pick up trash_can from floor`:
  trash can should be grasped; support by floor should no longer be the main state.
- `turn on radio`:
  radio should satisfy a toggle-on style predicate.
- `place soda can in trash can`:
  soda can should satisfy an inside-container relation.
- `place apple on table next to plate`:
  apple should satisfy both support and adjacency relations.

Do not invent predicate names. Choose only from grounded facts already present in `world_predicates`, plus direct robot/object state and current images from the snapshot.

## Evidence Model

Use all of these evidence sources together:

- Grounded multi-entity predicates in `world_predicates`.
  Examples: `Inside`, `OnTop`, `NextTo`.
- Grounded unary state predicates in `world_predicates`.
  Examples: `Open`, `ToggledOn`.
- `robot_state.grasped_objects` for acquisition or transfer subtasks.
- Object position and orientation from `object_states`.
- Current images for visual state confirmation, contradiction checks, and cases where predicates alone are incomplete.

Do not rely on predicates alone. Predicates, grasp state, pose, and images are all valid decision inputs.

Prefer evidence that directly matches the manipulated object and the target or
reference object. Strong agreement across multiple evidence sources is better than a
single weak cue.

## Grounded Evidence Rules

- Resolve annotation objects to runtime entities before making a decision.
- Inspect only predicates that mention the current manipulated object, current target, or the robot.
- Prefer predicates where both relevant entities appear in the same grounded fact.
- Treat single-object matches as weak evidence unless the subtask is clearly a unary
  state change.
- For interchangeable same-class objects such as multiple soda cans, prefer count
  or relationship deltas over exact annotation instance identity.
- Do not mark success from absolute count alone when prior objects may already
  satisfy the relation.
- Ignore unrelated `true` predicates.
- Do not summarize completion as a generic ratio of satisfied predicates.
- Treat image observations as first-class evidence, not just a fallback after
  predicate inspection.

## Common Inference Patterns

Use these as patterns, not a closed list:

- Navigation:
  success usually means the robot is near the relevant object or region.
- Acquisition:
  success usually means the manipulated object is grasped by the robot.
- Placement:
  success usually means a relation like `Inside` or `OnTop` is true, and the object is not still being held.
- Relative placement:
  success may require more than one relation, such as placement plus `NextTo`.
- Unary state change:
  success usually means a state like `Open` or `ToggledOn` has the correct value.
- Transfer or motion-heavy subtasks:
  use grasp state, grounded relations, pose/orientation, and current images together. Do not force a decision from one weak cue.
- Visually obvious but predicate-weak subtasks:
  use images as direct evidence, then check whether predicates and robot/object state support or contradict that reading.

## Contradiction Checks

Even when some evidence looks positive, treat the subtask as inconclusive if there is strong counter-evidence.

Common contradictions:

- object is still grasped after a supposed placement
- wrong object is grasped
- expected relation is absent for the intended object pair
- a state-change subtask still shows the opposite unary state
- pose changed, but no relevant target relation or control effect is visible
- the image evidence disagrees with the predicate or pose evidence

## Demo Comparison Trigger

Use `get_demo_reference_frames()` only when all three conditions hold:

- current predicates, robot/object state, and images are still inconclusive
- demo frames are available
- elapsed ratio is at least `0.5`

Default to completion-only comparison: request `frame_types=["completion"]` and
`max_episodes=1` unless the completion frame is ambiguous.

If visual comparison still does not support completion and the threshold is not yet exceeded, continue rather than forcing a success.
