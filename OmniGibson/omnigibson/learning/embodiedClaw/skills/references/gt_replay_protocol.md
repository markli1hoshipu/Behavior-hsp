# GT Replay Protocol

Use this file for offline replay evaluation only. Do not mix it with the live control loop.

## Goal

GT replay is for checking whether the decision rubric fires near the annotated subtask boundary. It is not a live failure handler.

## Evaluation Rules

- Measure whether the rubric predicts success near the annotated completion window.
- Report early, on-time, late, and missed detections separately.
- Treat timeout overruns as evaluation signals, not as live episode failures.
- Keep replay outputs diagnostic: boundary quality, ambiguity, and unsupported cases.

## Required Inputs

- replay frame or snapshot
- current subtask annotation
- expected completion frame range
- optional demo reference frames for ambiguous cases

## Minimal Replay Output

- `subtask_idx`
- `expected_boundary`
- `predicted_boundary`
- `classification`: `early`, `on_time`, `late`, `missed`, or `ambiguous`
- `reasoning`
