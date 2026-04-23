"""
R1Pro whole-body IK teleoperation (base-frame EEF control) using an external solver in ~/mark_ik.

Keys (summary)
- 1: Base mode (Phase 1)
- 2: EEF mode  (Phase 2)
- 3/4: switch active arm (left/right)
- Base mode:
  - W/S: +X / -X base translation
  - A/D: +Y / -Y base translation
  - Q/E: +Yaw / -Yaw base rotation
- EEF mode:
  - Arrow keys: +/-X and +/-Y (EEF target, BASE frame)
  - P / ; : +/-Z (EEF target, BASE frame)
  - N/B: +/- roll, O/U: +/- pitch, V/C: +/- yaw (EEF target, BASE frame axes)
  - SPACE: reset EEF target to current pose
  - K: paste single EEF target JSON (BASE frame)
  - T: paste trajectory JSON (BASE frame)
- Head (optional, both modes):
  - PageUp/PageDown: head tilt (tries torso_joint4)
  - J/L: head yaw (tries torso_joint3)
- R: reload instance + cancel motion
- ESC: quit
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import datetime
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.macros import gm
from omnigibson.metrics import AgentMetric, TaskMetric
from omnigibson.learning.utils.eval_utils import TASK_NAMES_TO_INDICES
from omnigibson.utils.asset_utils import get_task_instance_path
from omnigibson.utils.python_utils import recursively_convert_to_torch
from omnigibson.utils.ui_utils import KeyboardEventHandler, clear_debug_drawing
from gello.robots.sim_robot.og_teleop_utils import (
    load_available_tasks,
    generate_robot_config,
)
from omnigibson.learning.utils.eval_utils import (
    PROPRIOCEPTION_INDICES,
    TASK_NAMES_TO_INDICES,
    generate_basic_environment_config as gen_env_cfg_eval,
)
from hydra.utils import instantiate
from omnigibson.envs.env_wrapper import EnvironmentWrapper


DATASET_RELATIVE_PATH = Path("datasets/2025-challenge-task-instances/metadata/test_instances.csv")


# ----------------- Utility: dataset table / task selection -----------------

def to_jsonable(x):
    if isinstance(x, th.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    return x


class JsonlWriter:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "w", buffering=1)  # line-buffered

    def write(self, obj: dict):
        self.f.write(json.dumps(to_jsonable(obj)) + "\n")

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass

def find_repo_root() -> Path:
    anchor = Path(__file__).resolve()
    for parent in anchor.parents:
        candidate = parent / DATASET_RELATIVE_PATH
        if candidate.exists():
            return parent
    raise FileNotFoundError("Could not locate test_instances.csv from repository root.")


def load_test_instance_table(csv_path: Path) -> OrderedDict[str, List[int]]:
    table: "OrderedDict[str, List[int]]" = OrderedDict()
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids = [int(x.strip()) for x in row["Public Test Instance IDs"].split(",") if x.strip()]
            table[row["Task"]] = ids
    return table


def choose_from_options(options, name: str) -> str:
    print(f"\nAvailable {name}s:")
    for i, k in enumerate(options):
        print(f"[{i+1}] {k}")
    try:
        sel = int(input(f"Choose a {name} [1-{len(options)}]: ")) - 1
    except Exception:
        sel = 0
    sel = max(0, min(sel, len(options) - 1))
    return list(options)[sel]


def select_task(task_instances: OrderedDict[str, List[int]], requested: Optional[str]) -> str:
    if requested and requested in task_instances:
        return requested
    return choose_from_options(task_instances.keys(), "task")


def select_instance(instance_ids: List[int], requested: Optional[int]) -> int:
    if requested is not None and requested in instance_ids:
        return requested
    opt = choose_from_options([str(x) for x in instance_ids], "instance id")
    return int(opt)


# ----------------- Env + Robot configs -----------------

def generate_basic_environment_config(task_name: str, scene_model: str) -> dict:
    return {
        "env": {"action_frequency": 30, "rendering_frequency": 30, "physics_frequency": 120},
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": scene_model,
            "include_robots": False,
            "load_room_types": None,
            "load_room_instances": None,
        },
        "task": {
            "type": "BehaviorTask",
            "activity_name": task_name,
            "activity_definition_id": 0,
            "activity_instance_id": 0,
            "online_object_sampling": False,
            "highlight_task_relevant_objects": False,
            "termination_config": {"max_steps": 5000},
            "reward_config": {"r_potential": 1.0},
            "include_obs": False,
        },
    }


def build_robot_config() -> dict:
    return {
        "type": "R1Pro",
        "obs_modalities": ["rgb", "depth_linear", "seg_semantic", "camera_params"],
        "action_type": "continuous",
        "action_normalize": True,
        "grasping_mode": "physical",
        "default_reset_mode": "tuck",
        "controller_config": {
            "base": {"name": "HolonomicBaseJointController"},
            "trunk": {
                "name": "JointController",
                "motor_type": "position",
                "command_output_limits": None,
                "use_delta_commands": True,
            },
            "arm_left": {
                "name": "JointController",
                "motor_type": "position",
                "command_output_limits": None,
                "use_delta_commands": True,
            },
            "arm_right": {
                "name": "JointController",
                "motor_type": "position",
                "command_output_limits": None,
                "use_delta_commands": True,
            },
            "gripper_left": {"name": "MultiFingerGripperController", "mode": "smooth"},
            "gripper_right": {"name": "MultiFingerGripperController", "mode": "smooth"},
        },
    }


def apply_task_instance_state(env: og.Environment, instance_id: int) -> None:
    robot = env.robots[0]
    scene_model = env.task.scene_name
    template_name = env.task.get_cached_activity_scene_filename(
        scene_model=scene_model,
        activity_name=env.task.activity_name,
        activity_definition_id=env.task.activity_definition_id,
        activity_instance_id=instance_id,
    )
    root = get_task_instance_path(scene_model)
    tro_path = (
        Path(root)
        / "json"
        / f"{scene_model}_task_{env.task.activity_name}_instances"
        / f"{template_name}-tro_state.json"
    )
    with tro_path.open("r") as f:
        tro_state = recursively_convert_to_torch(json.load(f))
    for key, state in tro_state.items():
        if key == "robot_poses":
            pose = state[robot.model_name][0]
            robot.set_position_orientation(pose["position"], pose["orientation"])
            env.scene.write_task_metadata(key=key, data=state)
        else:
            env.task.object_scope[key].load_state(state, serialized=False)
    for _ in range(25):
        og.sim.step_physics()
        for entity in env.task.object_scope.values():
            if not entity.is_system and entity.exists:
                entity.keep_still()
    env.scene.update_initial_file()


# ----------------- Pose helpers -----------------

def get_pose_in_base(robot, link_name: str) -> Tuple[th.Tensor, th.Tensor]:
    base = robot.links[robot.base_footprint_link_name]
    link = robot.links[link_name]
    base_pos, base_quat = base.get_position_orientation()
    ee_pos, ee_quat = link.get_position_orientation()
    rel_pos, rel_quat = T.relative_pose_transform(ee_pos, ee_quat, base_pos, base_quat)
    return rel_pos, rel_quat


# ----------------- IK Adapter (mark_ik) -----------------

class MarkIKAdapter:
    """Adapter for mark_ik solver.

    - mode="left_torso" or "right_torso"
    - input: base-frame EEF target pos[m], quat[xyzw] + initial joint guess
    - output: dict {joint_name: absolute_joint_position}
    """

    def __init__(self, robot):
        self.robot = robot
        try:
            from mark_ik import solve_ik  # type: ignore
        except Exception:
            try:
                from mark_ik.hsp_r1pro_iksolver import solve_ik  # type: ignore
            except Exception as e:
                solve_ik = None  # type: ignore
                print(f"[MarkIKAdapter] Could not import solve_ik from mark_ik: {e}")
        self.solve_ik = solve_ik

    def _desired_joint_names(self, arm: str) -> List[str]:
        if arm == "left":
            return [
                "torso_joint1",
                "torso_joint2",
                "torso_joint3",
                "torso_joint4",
                "left_arm_joint1",
                "left_arm_joint2",
                "left_arm_joint3",
                "left_arm_joint4",
                "left_arm_joint5",
                "left_arm_joint6",
                "left_arm_joint7",
            ]
        else:
            return [
                "torso_joint1",
                "torso_joint2",
                "torso_joint3",
                "torso_joint4",
                "right_arm_joint1",
                "right_arm_joint2",
                "right_arm_joint3",
                "right_arm_joint4",
                "right_arm_joint5",
                "right_arm_joint6",
                "right_arm_joint7",
            ]

    def _initial_guess(self, arm: str) -> List[float]:
        q = self.robot.get_joint_positions()
        names = list(self.robot.joints.keys())
        index = {n: i for i, n in enumerate(names)}
        guess = []
        for name in self._desired_joint_names(arm):
            if name in index:
                guess.append(float(q[index[name]]))
            else:
                guess.append(0.0)
        return guess

    def solve_to_joint_targets(
        self, arm: str, target_pose_base: Tuple[th.Tensor, th.Tensor]
    ) -> Optional[Dict[str, float]]:
        if self.solve_ik is None:
            return None

        pos_t, quat_t = target_pose_base
        pos = pos_t.detach().cpu().numpy().astype("float64")
        quat = quat_t.detach().cpu().numpy().astype("float64")  # xyzw
        mode = "left_torso" if arm == "left" else "right_torso"

        # Prefer env var; fallback to a common location under ~/mark_ik
        urdf_path = os.environ.get(
            "R1PRO_URDF",
            str(Path.home() / "mark_ik" / "src" / "mark_ik" / "r1pro.urdf"),
        )

        q0 = self._initial_guess(arm)

        try:
            result = self.solve_ik(
                mode=mode,
                target_pos_m=pos,
                target_quat_xyzw=quat,
                urdf_path=urdf_path,
                initial_guess=q0,
                verbose=False,
            )
        except Exception as e:
            print(f"[MarkIKAdapter] solve_ik failed: {e}")
            return None

        if not isinstance(result, dict) or "q_sol" not in result:
            return None

        q_sol = result["q_sol"]
        names = self._desired_joint_names(arm)
        return {n: float(q_sol[i]) for i, n in enumerate(names)}


# ----------------- Teleop core -----------------

class EEFBaseTeleop:
    """
    Two-phase controller:
    - mode="base": keyboard drives base [dx, dy, dyaw]
    - mode="eef": keyboard nudges EEF target OR you provide ee6d target(s) to follow via IK (torso + active arm)
    """

    def __init__(self, robot, env):
        self.robot = robot
        self.env = env

        # Phase mode
        self.mode = "base"  # "base" or "eef"

        # Active arm
        self.active_arm = "left"

        # Desired EEF pose per arm (BASE frame), init to current
        self.desired: Dict[str, Tuple[th.Tensor, th.Tensor]] = {}
        for arm in robot.arm_names:
            ee_name = robot.gripper_link_names[arm][0]
            self.desired[arm] = get_pose_in_base(robot, ee_name)

        # Step sizes
        self.pos_step = 0.025  # meters
        self.rot_step = th.deg2rad(th.tensor(5.0))  # radians

        # Base motion step sizes (deltas)
        self.base_lin_step = 0.20
        self.base_yaw_step = 0.35

        # Head control steps (optional)
        self.head_pitch_step = th.deg2rad(th.tensor(5.0))
        self.head_yaw_step = th.deg2rad(th.tensor(5.0))

        # Head bindings
        self._head_binding: Optional[Tuple[str, int, str]] = None
        self._head_nudge: float = 0.0
        self._head_yaw_binding: Optional[Tuple[str, int, str]] = None
        self._head_yaw_nudge: float = 0.0

        # IK adapter
        self.ik = MarkIKAdapter(robot)

        # Target tracking + IK throttling
        self._target_dirty = True
        self._last_solve_t = 0.0
        self._ik_min_period = 1.0 / 15.0  # 15 Hz max

        # Base delta accumulator (one sim step)
        self._base_nudge = th.zeros(3)

        # Cache last IK action so between solves we keep moving
        self._last_ik_action: Optional[th.Tensor] = None

        # Auto-drive-to-target state (single target)
        self._drive_active: bool = False
        self._drive_target: Optional[Tuple[th.Tensor, th.Tensor]] = None
        self._drive_label: str = ""
        self._drive_pos_thresh: float = 0.01  # m
        self._drive_ori_thresh: float = float(th.deg2rad(th.tensor(5.0)))  # rad
        self._drive_start_time: float = 0.0
        self._drive_timeout: float = 600.0  # s
        self._drive_last_improve_t: float = 0.0

        # Trajectory queue: list of (pos, quat, label)
        self._traj_active: bool = False
        self._traj_arm: str = "left"
        self._traj_queue: List[Tuple[th.Tensor, th.Tensor, str]] = []
        self._traj_pos_thresh: float = 0.01
        self._traj_ori_thresh: float = float(th.deg2rad(th.tensor(5.0)))

        # Build action mapping layout
        self.controller_info = {}
        idx = 0
        for name, controller in robot._controllers.items():
            self.controller_info[name] = {
                "start_idx": idx,
                "dofs": controller.dof_idx,
                "command_dim": controller.command_dim,
            }
            idx += controller.command_dim

        # Default gripper hold
        self._gripper_hold_level = 0.99
        self._gripper_hold_command = {}
        for arm in robot.arm_names:
            ctrl_name = f"gripper_{arm}"
            info = self.controller_info.get(ctrl_name)
            if info is None:
                continue
            dim = info["command_dim"]
            self._gripper_hold_command[arm] = th.full((dim,), self._gripper_hold_level, dtype=th.float32)

        # Register keys
        self._register_keys()

    # ----------------- Mode / Arm -----------------

    def _set_mode(self, mode: str):
        if mode not in ("base", "eef"):
            return
        self.mode = mode
        # Clear transient nudges so they don't leak between modes
        self._base_nudge = th.zeros(3)
        self._head_nudge = 0.0
        self._head_yaw_nudge = 0.0
        self._last_ik_action = None
        print(f"[Teleop] Mode set to: {mode}  (1=base move, 2=eef follow)")

    def _set_arm(self, arm: str):
        if arm in self.robot.arm_names:
            self.active_arm = arm
            print(f"[Teleop] Active arm set to: {arm}")

    # ----------------- Keyboard registration -----------------

    def _register_keys(self):
        # Mode switch
        KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.KEY_1, lambda: self._set_mode("base"))
        KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.KEY_2, lambda: self._set_mode("eef"))

        # Toggle active arm
        KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.KEY_3, lambda: self._set_arm("left"))
        KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.KEY_4, lambda: self._set_arm("right"))

        # EEF position nudges (BASE frame) - only meaningful in EEF mode
        KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.UP, lambda: self._bump_xyz(+self.pos_step, 0, 0))
        KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.DOWN, lambda: self._bump_xyz(-self.pos_step, 0, 0))
        KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.LEFT, lambda: self._bump_xyz(0, +self.pos_step, 0))
        KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.RIGHT, lambda: self._bump_xyz(0, -self.pos_step, 0))
        KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.P, lambda: self._bump_xyz(0, 0, +self.pos_step))
        KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.SEMICOLON, lambda: self._bump_xyz(0, 0, -self.pos_step))

        # EEF orientation nudges (BASE frame axes)
        KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.N, lambda: self._bump_rpy(+self.rot_step, 0, 0))
        KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.B, lambda: self._bump_rpy(-self.rot_step, 0, 0))
        KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.O, lambda: self._bump_rpy(0, +self.rot_step, 0))
        KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.U, lambda: self._bump_rpy(0, -self.rot_step, 0))
        KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.V, lambda: self._bump_rpy(0, 0, +self.rot_step))
        KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.C, lambda: self._bump_rpy(0, 0, -self.rot_step))

        # Reset EEF target
        KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.SPACE, self._reset_target_pose)

        # Paste single EEF pose JSON (BASE frame)
        KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.K, self._cli_eef_pose_json)

        # Paste trajectory JSON (BASE frame)
        KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.T, self._cli_traj_json)

        # Base nudges (Phase 1): W/S forward/back (x), A/D strafe (y), Q/E yaw
        KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.W, lambda: self._nudge_base(+self.base_lin_step, 0.0, 0.0))
        KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.S, lambda: self._nudge_base(-self.base_lin_step, 0.0, 0.0))
        KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.A, lambda: self._nudge_base(0.0, +self.base_lin_step, 0.0))
        KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.D, lambda: self._nudge_base(0.0, -self.base_lin_step, 0.0))
        KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.Q, lambda: self._nudge_base(0.0, 0.0, +self.base_yaw_step))
        KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.E, lambda: self._nudge_base(0.0, 0.0, -self.base_yaw_step))

        # Head controls (optional)
        KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.PAGE_UP, lambda: self._tilt_head(+float(self.head_pitch_step)))
        KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.PAGE_DOWN, lambda: self._tilt_head(-float(self.head_pitch_step)))
        KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.J, lambda: self._yaw_head(+float(self.head_yaw_step)))
        KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.L, lambda: self._yaw_head(-float(self.head_yaw_step)))

    # ----------------- EEF target nudges -----------------

    def _bump_xyz(self, dx: float, dy: float, dz: float):
        if self.mode != "eef":
            return
        pos, quat = self.desired[self.active_arm]
        self.desired[self.active_arm] = (pos + th.tensor([dx, dy, dz], dtype=pos.dtype, device=pos.device), quat)
        self._target_dirty = True
        self._drive_active = False
        self._traj_active = False

    def _bump_rpy(self, droll: th.Tensor, dpitch: th.Tensor, dyaw: th.Tensor):
        if self.mode != "eef":
            return
        pos, quat = self.desired[self.active_arm]
        rot = T.euler2quat(th.tensor([droll, dpitch, dyaw], dtype=quat.dtype, device=quat.device))
        new_quat = T.quat_multiply(quat, rot)
        self.desired[self.active_arm] = (pos, new_quat)
        self._target_dirty = True
        self._drive_active = False
        self._traj_active = False

    # ----------------- Base nudges -----------------

    def _nudge_base(self, dx: float, dy: float, drz: float):
        if self.mode != "base":
            return
        self._base_nudge += th.tensor([dx, dy, drz], dtype=self._base_nudge.dtype)

    # ----------------- Head binding helpers -----------------

    def _find_controller_for_joint_idx(self, joint_idx: int) -> Optional[Tuple[str, int]]:
        for comp, info in self.controller_info.items():
            dofs = info["dofs"].tolist()
            if joint_idx in dofs:
                return comp, dofs.index(joint_idx)
        return None

    def _bind_head_tilt_joint(self) -> Optional[Tuple[str, int, str]]:
        names = list(self.robot.joints.keys())
        lc = [n.lower() for n in names]
        candidates: List[int] = []

        def add(pred):
            for i, n in enumerate(lc):
                if pred(n):
                    candidates.append(i)

        if "torso_joint4" in names:
            candidates.append(names.index("torso_joint4"))
        add(lambda n: ("head" in n or "zed" in n or "camera" in n) and ("tilt" in n or "pitch" in n))
        if not candidates:
            add(lambda n: ("head" in n or "zed" in n or "camera" in n))
        if not candidates:
            for jn in ("torso_joint4", "torso_joint3", "torso_joint2"):
                if jn in names:
                    candidates.append(names.index(jn))
                    break

        for j_idx in candidates:
            res = self._find_controller_for_joint_idx(j_idx)
            if res is None:
                continue
            comp, local = res
            start = self.controller_info[comp]["start_idx"]
            joint_name = names[j_idx]
            print(f"[Teleop] Head tilt bound to joint '{joint_name}' via component '{comp}'.")
            return comp, start + local, joint_name

        print("[Teleop] Unable to bind head tilt joint.")
        return None

    def _tilt_head(self, dpitch: float):
        if self._head_binding is None:
            self._head_binding = self._bind_head_tilt_joint()
        if self._head_binding is None:
            return
        self._head_nudge += float(dpitch)

    def _bind_head_yaw_joint(self) -> Optional[Tuple[str, int, str]]:
        names = list(self.robot.joints.keys())
        lc = [n.lower() for n in names]
        candidates: List[int] = []

        if "torso_joint3" in names:
            candidates.append(names.index("torso_joint3"))
        for i, n in enumerate(lc):
            if ("head" in n or "zed" in n or "camera" in n) and ("yaw" in n or "pan" in n):
                candidates.append(i)
        if not candidates:
            for jn in ("torso_joint3", "torso_joint2", "torso_joint4"):
                if jn in names:
                    candidates.append(names.index(jn))
                    break

        for j_idx in candidates:
            res = self._find_controller_for_joint_idx(j_idx)
            if res is None:
                continue
            comp, local = res
            start = self.controller_info[comp]["start_idx"]
            joint_name = names[j_idx]
            print(f"[Teleop] Head yaw bound to joint '{joint_name}' via component '{comp}'.")
            return comp, start + local, joint_name

        print("[Teleop] Unable to bind head yaw joint.")
        return None

    def _yaw_head(self, dyaw: float):
        if self._head_yaw_binding is None:
            self._head_yaw_binding = self._bind_head_yaw_joint()
        if self._head_yaw_binding is None:
            return
        self._head_yaw_nudge += float(dyaw)

    # ----------------- Gripper defaults -----------------

    def _apply_default_gripper_commands(self, action: th.Tensor) -> None:
        for arm, cmd in self._gripper_hold_command.items():
            ctrl_name = f"gripper_{arm}"
            info = self.controller_info.get(ctrl_name)
            if info is None:
                continue
            start = info["start_idx"]
            end = start + info["command_dim"]
            if end <= action.shape[0]:
                action[start:end] = cmd.to(device=action.device, dtype=action.dtype)

    # ----------------- EEF pose IO -----------------

    def _reset_target_pose(self):
        if self.mode != "eef":
            return
        arm = self.active_arm
        ee_name = self.robot.gripper_link_names[arm][0]
        self.desired[arm] = get_pose_in_base(self.robot, ee_name)
        self._target_dirty = True
        self._drive_active = False
        self._traj_active = False
        self._last_ik_action = None
        try:
            clear_debug_drawing()
        except Exception:
            pass
        print("[Teleop] Reset EEF target to current pose.")

    def _get_eef_pose(self, arm: Optional[str] = None) -> Tuple[th.Tensor, th.Tensor]:
        arm = arm or self.active_arm
        ee_name = self.robot.gripper_link_names[arm][0]
        return get_pose_in_base(self.robot, ee_name)

    @staticmethod
    def _quat_angle_error(q1: th.Tensor, q2: th.Tensor) -> float:
        q1 = q1 / th.linalg.norm(q1)
        q2 = q2 / th.linalg.norm(q2)
        dot = th.clamp(th.abs(th.dot(q1, q2)), 0.0, 1.0)
        return float(2.0 * th.acos(dot).item())  # radians

    def _begin_auto_drive(self, target_pos: th.Tensor, target_quat: th.Tensor, label: str = "target"):
        self._drive_active = True
        self._drive_target = (target_pos.clone(), target_quat.clone())
        self._drive_label = label
        now = time.monotonic()
        self._drive_start_time = now
        self._drive_last_improve_t = now
        self._last_ik_action = None

    def _maybe_advance_traj(self):
        if not self._traj_active or not self._traj_queue:
            return
        arm = self._traj_arm
        tgt_pos, tgt_quat, wp_label = self._traj_queue[0]
        cur_pos, cur_quat = self._get_eef_pose(arm)
        pos_err = float(th.linalg.norm(cur_pos - tgt_pos).item())
        ori_err = self._quat_angle_error(cur_quat, tgt_quat)
        if pos_err < self._traj_pos_thresh and ori_err < self._traj_ori_thresh:
            self._traj_queue.pop(0)
            print(f"[Traj] Reached {wp_label}: pos_err={pos_err:.4f}m ori_err={np.degrees(ori_err):.1f}deg")
            if not self._traj_queue:
                print("[Traj] Done.")
                self._traj_active = False
                self._drive_active = False
                self._drive_target = None
                self._last_ik_action = None

    def set_eef_target_base(
        self,
        arm: str,
        pos_xyz,
        quat_xyzw,
        start_drive: bool = False,
        label: str = "ee6d",
    ):
        """Set desired EEF target in BASE frame (meters, quaternion xyzw)."""
        if arm not in self.robot.arm_names:
            print(f"[Teleop] Invalid arm: {arm}")
            return

        # convert inputs to tensors
        pos = th.tensor(pos_xyz, dtype=th.float32)
        quat = th.tensor(quat_xyzw, dtype=th.float32)

        # normalize quaternion (safety)
        quat = quat / th.linalg.norm(quat)

        # set desired target
        self.desired[arm] = (pos, quat)
        self._target_dirty = True
        self._traj_active = False

        # optionally start auto-drive toward target
        if start_drive:
            self._begin_auto_drive(pos, quat, label=label)
            
    def step(self) -> th.Tensor:
        """Compute action vector for the robot by solving IK for the active arm.

        Returns: action tensor of shape (action_dim,)
        """
        self._draw_grasp_visualization()
        # Apply pending single-shot action if present
        if self._pending_action is not None:
            act = self._pending_action
            self._pending_action = None
            return act
        action = th.zeros(self.robot.action_dim)
        #print("test1")
        #print(action.shape)
        self._apply_default_gripper_commands(action)
        if self._pending_sam6d_error is not None:
            print(f"[SAM-6D] Request failed: {self._pending_sam6d_error}")
            self._pending_sam6d_error = None
        if self._pending_contact_error is not None:
            print(f"[Contact] Request failed: {self._pending_contact_error}")
            self._pending_contact_error = None
        if self._pending_sam6d_pose_base is not None:
            arm_name, pos_target, quat_target, source = self._pending_sam6d_pose_base
            if arm_name != self.active_arm:
                self._set_arm(arm_name)
            self.desired[arm_name] = (pos_target.clone(), quat_target.clone())
            self._target_dirty = True
            self._set_grasp_viz_pose(arm_name, pos_target, quat_target, source)
            self._pending_drive_target = (pos_target.clone(), quat_target.clone(), source)
            self._pending_sam6d_pose_base = None
        if self._pending_contact_pose_base is not None:
            arm_name, pos_target, quat_target, source = self._pending_contact_pose_base
            if arm_name != self.active_arm:
                self._set_arm(arm_name)
            self.desired[arm_name] = (pos_target.clone(), quat_target.clone())
            self._target_dirty = True
            self._set_grasp_viz_pose(arm_name, pos_target, quat_target, source)
            self._pending_drive_target = (pos_target.clone(), quat_target.clone(), source)
            self._pending_contact_pose_base = None
            if self._contact_candidate_queue:
                self._pending_contact_pose_base = self._contact_candidate_queue.pop(0)
        arm = self.active_arm
        if self._pending_sam6d_pose is not None:
            cam_pos = self._pending_sam6d_pose.get("cam_pos")
            cam_quat = self._pending_sam6d_pose.get("cam_quat")
            cam_pose = None
            if cam_pos is not None and cam_quat is not None:
                cam_pose = (cam_pos, cam_quat)
            self._schedule_target_from_camera_Rt(
                self._pending_sam6d_pose["R"],
                self._pending_sam6d_pose["t"],
                arm=self.active_arm,
                source="sam6d",
                cam_pose_base=cam_pose,
            )
            self._pending_sam6d_pose = None
        if self._sam6d_capture_pending and not self._sam6d_inflight:
            self._start_sam6d_request()
        if self._contact_capture_pending and not self._contact_inflight:
            self._start_contact_request()

        # Apply any scheduled camera-frame target now (safe in main loop)
        if self._pending_cam_target is not None:
            self._apply_scheduled_cam_target()
            self._draw_grasp_visualization()
        # Start any deferred auto-drive after target application
        if self._pending_drive_target is not None and not self._drive_active:
            pos, quat, label = self._pending_drive_target
            self._begin_auto_drive(pos, quat, label=label)
            self._pending_drive_target = None
        # Apply base nudge if any, then clear accumulator
        if "base" in self.controller_info and th.any(self._base_nudge != 0):
            base_start = self.controller_info["base"]["start_idx"]
            action[base_start : base_start + 3] = self._base_nudge
            self._base_nudge = th.zeros(3)
        # Snapshot head nudges and bindings (we will apply them after IK writes to avoid overwrite)
        pitch_idx = None
        yaw_idx = None
        pitch_val = 0.0
        yaw_val = 0.0
        if self._head_binding is not None and self._head_nudge != 0.0:
            _, idx, _ = self._head_binding
            pitch_idx = idx
            pitch_val = float(self._head_nudge)
        if self._head_yaw_binding is not None and self._head_yaw_nudge != 0.0:
            _, idx, _ = self._head_yaw_binding
            yaw_idx = idx
            yaw_val = float(self._head_yaw_nudge)

        now = time.monotonic()
        time_since_last = now - self._last_solve_t
        if self._drive_active and time_since_last >= self._ik_min_period and not self._target_dirty:
            self._target_dirty = True

        need_ik = self._target_dirty and time_since_last >= self._ik_min_period

        if not need_ik:
            if pitch_idx is not None:
                action[pitch_idx] += pitch_val
            if yaw_idx is not None:
                action[yaw_idx] += yaw_val
            if self._drive_active and self._last_ik_action is not None:
                action += self._last_ik_action
            self._head_nudge = 0.0
            self._head_yaw_nudge = 0.0
            self._evaluate_drive_goal(now)
            return action

        targets = self.ik.solve_to_joint_targets(arm=arm, target_pose_base=self.desired[arm])
        #print(f"[IK] Solving for {arm} EEF target, joint targets: {targets}")
        if targets is None:
            return action

        # Build mapping from joint name -> action index for trunk + active arm
        name_to_joint_index = {name: i for i, name in enumerate(self.robot.joints.keys())}
        def indices_for(ctrl_name: str) -> th.Tensor:
            return self.controller_info[ctrl_name]["dofs"]

        # Current joint positions
        q = self.robot.get_joint_positions()

        # Apply targets to trunk and active arm
        ctrl_groups = ["trunk", f"arm_{arm}"]
        for grp in ctrl_groups:
            if grp not in self.controller_info:
                continue
            dof_idx = indices_for(grp)
            start = self.controller_info[grp]["start_idx"]
            # For each joint in this controller, see if target exists by joint name
            for local_i, j_idx in enumerate(dof_idx.tolist()):
                j_name = list(self.robot.joints.keys())[j_idx]
                if j_name in targets:
                    # convert absolute target to delta since we use delta joint controllers
                    dq = float(targets[j_name]) - float(q[j_idx])
                    # Trunk writes should be additive to allow manual head pitch/yaw overlays
                    if grp == "trunk":
                        action[start + local_i] += dq
                    else:
                        action[start + local_i] = dq
        # Apply head nudges on top of IK trunk commands
        if pitch_idx is not None:
            action[pitch_idx] += pitch_val
        if yaw_idx is not None:
            action[yaw_idx] += yaw_val
        # ee pose after IK solve
        cur_pos, cur_quat = self._get_eef_pose(arm)
        print(f"[EEF after IK] {arm}: pos={cur_pos.tolist()} quat(xyzw)={cur_quat.tolist()}")
        #pdb.set_trace()
        # Clear accumulators
        self._head_nudge = 0.0
        self._head_yaw_nudge = 0.0
        self._last_solve_t = now
        self._target_dirty = False
        self._last_ik_action = action.clone()
        self._evaluate_drive_goal(now, cur_pos, cur_quat)
        # After solving, if auto-drive is active, we'll continue issuing actions until we reach target
        return action


        

def sanitize_task_name(name: str) -> str:
    # evaluator task names often contain spaces
    return name.replace(" ", "_").replace("/", "_")

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default = 'turning_on_radio',
                        help="Behavior task name, e.g. 'clean a trumpet'")
    parser.add_argument("--instance_id", type=int, default = 0,
                        help="Task instance id")
    parser.add_argument("--out_dir", type=str, default = '/home/user/BEHAVIOR-1K/OmniGibson/omnigibson/examples/teleoperation/traj',
                        help="Output trajectory jsonl")
    args = parser.parse_args()


    task_safe = sanitize_task_name(args.task_name)
    os.makedirs(args.out_dir, exist_ok=True)

    out_jsonl_path = os.path.join(
        args.out_dir,
        f"{task_safe}_{args.instance_id}.jsonl",
    )

    print(f"[Teleop] Recording trajectory to: {out_jsonl_path}")
    writer = JsonlWriter(out_jsonl_path)

    # --------- Global flags ----------
    gm.HEADLESS = False
    gm.USE_GPU_DYNAMICS = False

    # --------- Environment config ----------
    available_tasks = load_available_tasks()
    assert args.task_name in available_tasks, f"Invalid task name: {args.task_name}"

    task_cfg = available_tasks[args.task_name][0]

    cfg = gen_env_cfg_eval(task_name=args.task_name, task_cfg=task_cfg)

    # make scene / speed settings similar to your teleop needs
    # (optional) override timeout
    cfg["task"]["termination_config"]["max_steps"] = 5000

    # robot config exactly like evaluator does
    cfg["robots"] = [
        generate_robot_config(task_name=args.task_name, task_cfg=task_cfg)
    ]

    # make sure obs modalities contain what your teleop needs (your teleop uses IK, not obs)
    # evaluator uses ["proprio","rgb"], keep consistent:
    cfg["robots"][0]["obs_modalities"] = ["proprio", "rgb"]
    cfg["robots"][0]["proprio_obs"] = list(PROPRIOCEPTION_INDICES["R1Pro"].keys())

    # IMPORTANT: controller config must match evaluator if you want replay to work
    # If your evaluator overrides controllers via config.robot.controllers, mirror that here if needed.
    # (If you don't have hydra in teleop, just keep defaults from generate_robot_config)


    # --------- Create environment ----------
    env = og.Environment(configs=cfg)
    
    
    from omegaconf import OmegaConf
    from hydra.utils import instantiate

    wrapper_cfg = OmegaConf.create({
        "_target_": "omnigibson.learning.wrappers.RGBLowResWrapper"
    })
    env = instantiate(wrapper_cfg, env=env)
    # ---- END WRAPPER CODE ----

    robot = env.robots[0]
    # --------- Create teleop controller ----------
    teleop = EEFBaseTeleop(robot=robot, env=env)

    print("\nPipeline running:")
    print("1 → base mode (drive robot)")
    print("2 → eef mode (IK hand control)")
    print("ESC → quit\n")

    # --------- Main simulation loop ----------
    try:
        while True:
            action = teleop.step()   # ← controller computes action
            env.step(action)         # ← environment executes it
    finally:
        og.shutdown()


if __name__ == "__main__":
    main()
