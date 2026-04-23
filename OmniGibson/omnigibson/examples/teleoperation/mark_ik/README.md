# mark-ik

Lightweight Python package wrapping an IK solver for the HSP R1Pro using `pytorch_kinematics`.

## Install (local project)

- Editable install from this repo:

```
pip install -e .
```

- Optional heavy deps:

```
pip install .[full]
```

This provides `torch` and `pytorch-kinematics`. You can also install them yourself per your environment.

## Usage

```
from mark_ik import solve_ik
import numpy as np

# Choose which chain to solve
mode = "left_torso"  # or "right_torso"

# Target pose
target_pos_m = np.array([0.14, 0.25, 0.44], dtype=np.float64)
target_quat_xyzw = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)

# Use packaged URDF by default (None), or pass a custom path
res = solve_ik(
    mode,
    target_pos_m,
    target_quat_xyzw,
    urdf_path=None,
)

print(res["q_sol"])  # 11 joints (4 torso + 7 arm)
```

## Notes

- The package includes `r1pro.urdf` and uses it automatically when `urdf_path=None`.
- For torch install guidance (CUDA vs CPU), follow the official PyTorch instructions if `pip install torch` fails.
- Exposed API:
  - `solve_ik(...)`
  - `mat_to_quat_xyzw(R)`


