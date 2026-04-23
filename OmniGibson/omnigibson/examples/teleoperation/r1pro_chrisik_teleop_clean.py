"""
R1Pro whole-body IK teleoperation (base-frame dual EEF control) using an external solver in Chris's b1k package in
~/Documents/cmower/behavior1k

Uses the controller defined in the class
from b1k.ik import DualArmIKController

Dcoumentation: See README in https://github.com/cmower/behavior1k

"""

from __future__ import annotations

import pdb
import argparse
import csv
import io
from math import radians
import json
import os
import socket
import struct
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from b1k.planner.whole_body import WholeBodyPlanner
from b1k.ik import DualArmIKController, GlobalDualArmIK
from b1k.models.r1pro.constants import r1pro_init_joint_state
from utils_chris_planner import ChrisIKAdapter
import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency for debugging
    plt = None
import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.macros import gm
from omnigibson.utils.asset_utils import get_task_instance_path
from omnigibson.utils.constants import (
    semantic_class_id_to_name,
    semantic_class_name_to_id,
)
from omnigibson.utils.python_utils import recursively_convert_to_torch
from omnigibson.utils.ui_utils import (
    KeyboardEventHandler,
    draw_line,
    clear_debug_drawing,
)
from omnigibson.learning.utils.eval_utils import CAMERA_INTRINSICS, ROBOT_CAMERA_NAMES


DATASET_RELATIVE_PATH = Path(
    "datasets/2025-challenge-task-instances/metadata/test_instances.csv"
)


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
            ids = [
                int(x.strip())
                for x in row["Public Test Instance IDs"].split(",")
                if x.strip()
            ]
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


def select_task(
    task_instances: OrderedDict[str, List[int]], requested: Optional[str]
) -> str:
    if requested and requested in task_instances:
        return requested
    return choose_from_options(task_instances.keys(), "task")


def select_instance(instance_ids: List[int], requested: Optional[int]) -> int:
    if requested is not None and requested in instance_ids:
        return requested
    opt = choose_from_options([str(x) for x in instance_ids], "instance id")
    return int(opt)


def generate_basic_environment_config(task_name: str, scene_model: str) -> dict:
    return {
        "env": {
            "action_frequency": 30,
            "rendering_frequency": 30,
            "physics_frequency": 120,
        },
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
                "command_output_limits": "default",#None,
                "use_delta_commands": False,
            },
            "arm_left": {
                "name": "JointController",
                "motor_type": "position",
                "command_output_limits": "default",  # Scale commands to actual joint limits
                "use_delta_commands": False,
            },
            "arm_right": {
                "name": "JointController",
                "motor_type": "position",
                "command_output_limits": "default",  # Scale commands to actual joint limits
                "use_delta_commands": False,
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


def get_pose_in_base(robot, link_name: str) -> Tuple[th.Tensor, th.Tensor]:
    base = robot.links[robot.base_footprint_link_name]
    link = robot.links[link_name]
    base_pos, base_quat = base.get_position_orientation()
    ee_pos, ee_quat = link.get_position_orientation()
    rel_pos, rel_quat = T.relative_pose_transform(ee_pos, ee_quat, base_pos, base_quat)
    return rel_pos, rel_quat


# class ChrisIKAdapter:
#     """Adapter for Chris's IK solver. Wraps DualArmIKController"""

#     def __init__(self):
#         self.dt = 1.0 / 50.0
#         self.ik = DualArmIKController(self.dt)
#         self.qn = [r1pro_init_joint_state[n] for n in self.ik.joint_names]

#         # Planner setup
#         self.T_plan = 30
#         self.planner = WholeBodyPlanner(self.T_plan)
#         self.global_ik = GlobalDualArmIK()

#         print("[ChrisIKAdapter] Initialized")

#     def get_current_joint_state(self, joint_positions: th.Tensor, joint_names: list) -> list:
#         """Convert robot joint positions to IK joint state.

#         Args:
#             joint_positions: Tensor of current joint positions from robot
#             joint_names: List of joint names from robot

#         Returns:
#             List of joint positions in the order expected by the IK solver
#         """
#         index = {n: i for i, n in enumerate(joint_names)}
#         q_dict = {}
#         for n in self.ik.joint_names:
#             if n in index:
#                 q_dict[n] = float(joint_positions[index[n]])
#             else:
#                 q_dict[n] = 0.0
#         # breakpoint()
#         return [q_dict[n] for n in self.ik.joint_names]

#     def solve_to_joint_targets(
#         self,
#         current_joint_state: list,
#         target_pose_base_left: Tuple[th.Tensor, th.Tensor],
#         target_pose_base_right: Tuple[th.Tensor, th.Tensor],
#     ) -> Optional[Dict[str, float]]:
#         """Solve IK for dual-arm targets.

#         Args:
#             current_joint_state: Current joint state from get_current_joint_state
#             target_pose_base_left: Left arm target (position, quaternion)
#             target_pose_base_right: Right arm target (position, quaternion)

#         Returns:
#             Dictionary mapping joint names to target positions
#         """
#         def parse_target(target_pose_base):
#             pos_t, quat_t = target_pose_base
#             pos = pos_t.detach().cpu().numpy().astype("float64")
#             quat = quat_t.detach().cpu().numpy().astype("float64")
#             return pos, quat

#         pG_left, rG_left = parse_target(target_pose_base_left)
#         pG_right, rG_right = parse_target(target_pose_base_right)

#         config = {
#             "pG_left": pG_left,
#             "pG_right": pG_right,
#             "rG_left": rG_left,
#             "rG_right": rG_right,
#             "w_dq": 0.01,
#             "w_qn": 1e5,
#             "qn": self.qn,
#             "w_p": 1e8,
#             "w_r": 1e8,
#             "w_gaze": 1e6,
#             "q": current_joint_state,
#         }
#         self.ik.reset(config)
#         if self.ik.solve():
#             dq = self.ik.get_solution()
#         else:
#             print("[IK] Solver failed")
#             dq = np.zeros(self.ik.dof)
#         qsol = current_joint_state + self.dt * dq
#         return {n: qsol[i] for i, n in enumerate(self.ik.joint_names)}

#     def solve_dual_arm_trajectory(
#         self,
#         current_joint_positions: list,
#         left_goal: Tuple[th.Tensor, th.Tensor],
#         right_goal: Tuple[th.Tensor, th.Tensor],
#         duration: float = 30,
#         num_targets: int = 120,
#         normalize_actions: bool = True,
#         joint_names: list = None,
#         joint_limits: dict = None,
#         debug: bool = False,
#     ) -> Tuple[list, list]:
#         """Solve global IK and motion planning for dual-arm trajectory.

#         Args:
#             current_joint_positions: Current joint positions as a list (from get_current_joint_state)
#             left_goal: Tuple of (position, quaternion) for left end-effector goal
#             right_goal: Tuple of (position, quaternion) for right end-effector goal
#             base_action: Base action tensor to clone for each trajectory point
#             joint_names: List of robot joint names
#             joint_limits: Dictionary mapping joint names to (lower_limit, upper_limit) tuples
#             duration: Trajectory duration in seconds
#             num_targets: Number of trajectory waypoints to generate
#             normalize_actions: If True, normalize joint positions to [-1, 1] range (default True)
#             debug: If True, print debug information during solving

#         Returns:
#             Tuple of (actions_out, q_targets) where:
#                 - actions_out: List of action tensors for the trajectory
#                 - q_targets: List of joint target dictionaries
#         """
#         # Hardcoded controller info for R1Pro robot
#         controller_info = {
#             'base': {'start_idx': 0, 'dofs': np.array([0, 1, 5]), 'command_dim': 3},
#             'trunk': {'start_idx': 3, 'dofs': np.array([6, 7, 8, 9]), 'command_dim': 4},
#             'arm_left': {'start_idx': 7, 'dofs': np.array([10, 12, 14, 16, 18, 20, 22]), 'command_dim': 7},
#             'gripper_left': {'start_idx': 14, 'dofs': np.array([24, 25]), 'command_dim': 1},
#             'arm_right': {'start_idx': 15, 'dofs': np.array([11, 13, 15, 17, 19, 21, 23]), 'command_dim': 7},
#             'gripper_right': {'start_idx': 22, 'dofs': np.array([26, 27]), 'command_dim': 1}
#         }

#         joint_names = ['torso_joint1', 'torso_joint2', 'torso_joint3', 'torso_joint4', 'left_arm_joint1', 'right_arm_joint1', 'left_arm_joint2', 'right_arm_joint2', 'left_arm_joint3', 'right_arm_joint3', 'left_arm_joint4', 'right_arm_joint4', 'left_arm_joint5', 'right_arm_joint5', 'left_arm_joint6', 'right_arm_joint6', 'left_arm_joint7', 'right_arm_joint7', 'left_gripper_finger_joint1', 'left_gripper_finger_joint2', 'right_gripper_finger_joint1', 'right_gripper_finger_joint2']
#         joint_limits = {'torso_joint1': (-1.1344999074935913, 1.8325998783111572), 'torso_joint2': (-2.7924997806549072, 2.5306997299194336), 'torso_joint3': (-1.8325998783111572, 1.5707999467849731), 'torso_joint4': (-3.054299831390381, 3.054299831390381), 'left_arm_joint1': (-4.4506001472473145, 1.309000015258789), 'right_arm_joint1': (-4.4506001472473145, 1.309000015258789), 'left_arm_joint2': (-0.1745000034570694, 3.1415998935699463), 'right_arm_joint2': (-3.1415998935699463, 0.1745000034570694), 'left_arm_joint3': (-2.3561956882476807, 2.3561956882476807), 'right_arm_joint3': (-2.3561956882476807, 2.3561956882476807), 'left_arm_joint4': (-2.0943996906280518, 0.3490999639034271), 'right_arm_joint4': (-2.0943996906280518, 0.3490999639034271), 'left_arm_joint5': (-2.3561956882476807, 2.3561956882476807), 'right_arm_joint5': (-2.3561956882476807, 2.3561956882476807), 'left_arm_joint6': (-1.0471980571746826, 1.0471980571746826), 'right_arm_joint6': (-1.0471980571746826, 1.0471980571746826), 'left_arm_joint7': (-1.5707999467849731, 1.5707999467849731), 'right_arm_joint7': (-1.5707999467849731, 1.5707999467849731), 'left_gripper_finger_joint1': (0.0, 0.05000000074505806), 'left_gripper_finger_joint2': (0.0, 0.05000000074505806), 'right_gripper_finger_joint1': (0.0, 0.05000000074505806), 'right_gripper_finger_joint2': (0.0, 0.05000000074505806)}
        
#         if joint_names is not None:
#             joint_names = joint_names
#         if joint_limits is not None:
#             joint_limits = joint_limits

#         def parse_target(target_pose_base):
#             """Convert target pose tensors to numpy arrays."""
#             pos_t, quat_t = target_pose_base
#             pos = pos_t.detach().cpu().numpy().astype("float64")
#             quat = quat_t.detach().cpu().numpy().astype("float64")
#             return pos, quat

#         # Parse goal poses
#         pG_left, rG_left = parse_target(left_goal)
#         pG_right, rG_right = parse_target(right_goal)

#         # Use provided current joint state
#         qc = current_joint_positions

#         # Solve global IK
#         global_ik_config = {
#             "q0": qc,
#             "pG_left": pG_left,
#             "rG_left": rG_left,
#             "pG_right": pG_right,
#             "rG_right": rG_right,
#             "p_err_max": 0.000001,
#             "r_err_max": 0.0001,
#             "maintain_gaze": 10.0,
#         }

#         self.global_ik.reset(global_ik_config)
#         if self.global_ik.solve():
#             if debug:
#                 print("Global IK solved")
#         else:
#             if debug:
#                 print("Global IK failed")
#             raise RuntimeError("global ik failed!")

#         qsol = self.global_ik.get_solution()

#         # Verify solution with forward kinematics
#         if debug:
#             print("GOAL POSITION:", pG_left)
#             fk = self.ik.left_arm_chain.forward_kinematics(
#                 {n: qsol[n] for n in self.ik.left_arm_chain.get_joint_parameter_names()}
#             )
#             print("Reached POSITION", fk.translation().as_vector())

#         qF = [qsol[n] for n in self.planner.joint_names]
#         if debug:
#             print("qF", qsol)
#             print("++++++++++ SOLVED PLANNER 1 +++++++++++")

#         # Generate initial trajectory guess
#         t = np.linspace(0, duration, self.T_plan)
#         Q0 = np.linspace(qc, qF, self.T_plan)
#         dQ0 = np.gradient(Q0, t, axis=0)
#         ddQ0 = np.gradient(dQ0, t, axis=0)

#         # Configure and solve trajectory planner
#         planner_config = {
#             "Q0": Q0,
#             "dQ0": dQ0,
#             "ddQ0": ddQ0,
#             "duration": duration,
#             "q0": qc,
#             "qF": qF,
#             "w_dQ": 1.0,
#             "w_ddQ": 50.0,
#         }

#         self.planner.reset(planner_config)

#         if self.planner.solve():
#             if debug:
#                 print("[PLANNER] Solved successfully")
#         else:
#             if debug:
#                 print("planner failed!")
#             raise RuntimeError("planner failed")

#         Qsol, _, _ = self.planner.get_solution()

#         # Sample trajectory at desired resolution
#         t = np.linspace(0, duration, num_targets)
#         q_targets = [
#             {n: Qsol[n](t[i]) for n in self.planner.joint_names}
#             for i in range(num_targets)
#         ]
#         if debug:
#             print(q_targets[-1])
#             print("++++++++++ SOLVED PLANNER 2 +++++++++++")

#         # Convert joint targets to action sequence
#         ctrl_groups = ["trunk", "arm_left", "arm_right"]
#         actions_out = [th.zeros(23) for _ in range(num_targets)]

#         for i in range(num_targets):
#             q_targ = q_targets[i]
#             for grp in ctrl_groups:
#                 if grp not in controller_info:
#                     continue

#                 dof_idx = controller_info[grp]["dofs"]
#                 start = controller_info[grp]["start_idx"]
#                 for local_i, j_idx in enumerate(dof_idx.tolist()):
#                     j_name = joint_names[j_idx-6]
#                     # print(j_name)
#                     if j_name in q_targ:
#                         if normalize_actions:
#                             # Convert absolute joint position to normalized [-1, 1] command
#                             normalized_cmd = self.normalize_joint_position(
#                                 j_name, float(q_targ[j_name]), joint_limits
#                             )
#                             actions_out[i][start + local_i] = normalized_cmd
#                         else:
#                             # Use absolute joint position directly
#                             actions_out[i][start + local_i] = float(q_targ[j_name])

#         return actions_out, q_targets

#     def normalize_joint_position(self, joint_name: str, position: float, joint_limits: dict) -> float:
#         """Normalize an absolute joint position to [-1, 1] range based on joint limits.

#         Args:
#             joint_name: Name of the joint
#             position: Absolute joint position in radians
#             joint_limits: Dictionary mapping joint names to (lower_limit, upper_limit) tuples

#         Returns:
#             Normalized position in [-1, 1] range
#         """
#         if joint_name not in joint_limits:
#             return 0.0

#         lower, upper = joint_limits[joint_name]

#         # Map from [lower, upper] to [-1, 1]
#         midpoint = (upper + lower) / 2.0
#         range_half = (upper - lower) / 2.0

#         if range_half == 0:
#             return 0.0

#         normalized = (position - midpoint) / range_half
#         return float(np.clip(normalized, -1.0, 1.0))

#     def print_joint_limits(self, joint_names: list, joint_limits: dict):
#         """Print joint limits for all joints to diagnose limit issues.

#         Args:
#             joint_names: List of joint names
#             joint_limits: Dictionary mapping joint names to (lower_limit, upper_limit) tuples
#         """
#         print("\n[Joint Limits Diagnostic]:")
#         for i, joint_name in enumerate(joint_names):
#             if joint_name in joint_limits:
#                 lower, upper = joint_limits[joint_name]
#                 print(f"  Joint {i:2d} ({joint_name:30s}): lower={lower:+8.4f}, upper={upper:+8.4f}")
#             else:
#                 print(f"  Joint {i:2d} ({joint_name:30s}): No limit data")

#     def check_joints_reached(self, current_joints: th.Tensor, target_joints: th.Tensor, joint_names: list, joint_threshold: float = 0.01) -> bool:
#         """Check if all joints have reached their target positions.

#         Args:
#             current_joints: Current joint positions as a tensor
#             target_joints: Target joint positions as a tensor
#             joint_names: List of joint names
#             joint_threshold: Joint error threshold in radians (default ~0.57 degrees)

#         Returns:
#             True if all joints are within threshold of target positions, False otherwise
#         """
#         # Ensure tensors are on the same device
#         if target_joints.device != current_joints.device:
#             target_joints = target_joints.to(current_joints.device)

#         # Calculate per-joint errors
#         joint_errors = th.abs(target_joints - current_joints)

#         # Check if all joints are within threshold
#         all_reached = th.all(joint_errors < joint_threshold).item()

#         # Convert to numpy for printing
#         current_joints_np = current_joints.detach().cpu().numpy()
#         target_joints_np = target_joints.detach().cpu().numpy()
#         joint_errors_np = joint_errors.detach().cpu().numpy()

#         print(f"\n[Joint Check]:")
#         print(f"  Number of joints: {len(current_joints)}")

#         # Print all current and target joints
#         print(f"\n  Current joints: {current_joints_np}")
#         print(f"  Target joints:  {target_joints_np}")
#         print(f"  Joint errors:   {joint_errors_np}")

#         # Print details for joints that are not within threshold
#         joints_not_ok = []
#         for i in range(len(current_joints)):
#             joint_ok = joint_errors_np[i] < joint_threshold
#             if not joint_ok:
#                 joints_not_ok.append(i)
#                 joint_name = joint_names[i] if i < len(joint_names) else f"joint_{i}"
#                 print(f"  Joint {i} ({joint_name}): current={current_joints_np[i]:+.4f}, "
#                       f"target={target_joints_np[i]:+.4f}, "
#                       f"error={joint_errors_np[i]:.4f}rad (ok={joint_ok})")

#         # Print summary statistics
#         max_error_idx = th.argmax(joint_errors).item()
#         max_joint_name = joint_names[max_error_idx] if max_error_idx < len(joint_names) else f"joint_{max_error_idx}"
#         print(f"  Max error: {joint_errors_np[max_error_idx]:.4f}rad at joint {max_error_idx} ({max_joint_name})")
#         print(f"  Mean error: {joint_errors.mean().item():.4f}rad")

#         if all_reached:
#             print(f"  Status: ALL JOINTS REACHED")
#         else:
#             print(f"  Status: {len(joints_not_ok)} joint(s) NOT REACHED")

#         return all_reached

#     def check_pose_reached(self, current_poses: dict, desired_poses: dict, pos_threshold: float = 0.02, ori_threshold: float = 0.05) -> bool:
#         """Check if all arms have reached their desired poses.

#         Args:
#             current_poses: Dictionary mapping arm names to current (position, quaternion) tuples
#             desired_poses: Dictionary mapping arm names to desired (position, quaternion) tuples
#             pos_threshold: Position error threshold in meters (default 2cm)
#             ori_threshold: Orientation error threshold in radians (default ~2.86 degrees)

#         Returns:
#             True if all arms are within threshold of desired pose, False otherwise
#         """
#         all_reached = True
#         for arm_name in desired_poses.keys():
#             # Get current and desired poses
#             cur_pos, cur_quat = current_poses[arm_name]
#             des_pos, des_quat = desired_poses[arm_name]

#             # Calculate position error
#             pos_error = th.linalg.norm(des_pos - cur_pos).item()

#             # Calculate orientation error (quaternion distance)
#             # Use minimum of q and -q distance (equivalent orientations)
#             ori_error = min(
#                 th.linalg.norm(des_quat - cur_quat).item(),
#                 th.linalg.norm(des_quat + cur_quat).item()
#             )

#             # Check if within thresholds
#             pos_ok = pos_error < pos_threshold
#             ori_ok = ori_error < ori_threshold

#             # Convert to numpy for printing
#             cur_pos_np = cur_pos.detach().cpu().numpy()
#             cur_quat_np = cur_quat.detach().cpu().numpy()
#             des_pos_np = des_pos.detach().cpu().numpy()
#             des_quat_np = des_quat.detach().cpu().numpy()

#             print(f"\n[Pose Check] {arm_name.upper()}:")
#             print(f"  Current Pos:  [{cur_pos_np[0]:+.4f}, {cur_pos_np[1]:+.4f}, {cur_pos_np[2]:+.4f}]")
#             print(f"  Goal Pos:     [{des_pos_np[0]:+.4f}, {des_pos_np[1]:+.4f}, {des_pos_np[2]:+.4f}]")
#             print(f"  Pos Error:    {pos_error:.4f}m (ok={pos_ok})")
#             print(f"  Current Quat: [{cur_quat_np[0]:+.4f}, {cur_quat_np[1]:+.4f}, {cur_quat_np[2]:+.4f}, {cur_quat_np[3]:+.4f}]")
#             print(f"  Goal Quat:    [{des_quat_np[0]:+.4f}, {des_quat_np[1]:+.4f}, {des_quat_np[2]:+.4f}, {des_quat_np[3]:+.4f}]")
#             print(f"  Ori Error:    {ori_error:.4f}rad (ok={ori_ok})")

#             if not (pos_ok and ori_ok):
#                 all_reached = False
#                 print(f"  Status: NOT REACHED")
#             else:
#                 print(f"  Status: REACHED")

#         return all_reached


class GraspSolverSocketClient:
    """Thin TCP client for exchanging RGB-D captures with a remote grasp solver."""

    def __init__(self, host: str, port: int, timeout: float = 10.0):
        self.host = host
        self.port = port
        self.timeout = timeout

    def request_pose(
        self,
        header: Dict[str, object],
        arrays: Dict[str, np.ndarray],
        on_result,
        on_error,
    ) -> None:
        """Serialize payload and dispatch asynchronous request."""

        def _worker():
            try:
                blob = self._pack_arrays(arrays)
                header_with_sizes = dict(header)
                header_with_sizes["binary_size"] = len(blob)
                header_bytes = json.dumps(header_with_sizes).encode("utf-8")
                with socket.create_connection(
                    (self.host, self.port), timeout=self.timeout
                ) as sock:
                    self._send_blob(sock, header_bytes, blob)
                    response = self._receive_json(sock)
                on_result(response)
            except Exception as exc:
                on_error(exc)

        threading.Thread(target=_worker, daemon=True).start()

    @staticmethod
    def _pack_arrays(arrays: Dict[str, np.ndarray]) -> bytes:
        filtered = {k: v for k, v in arrays.items() if v is not None}
        with io.BytesIO() as buffer:
            np.savez_compressed(buffer, **filtered)
            return buffer.getvalue()

    @staticmethod
    def _send_blob(sock: socket.socket, header_bytes: bytes, blob: bytes) -> None:
        sock.sendall(struct.pack("!I", len(header_bytes)))
        sock.sendall(header_bytes)
        sock.sendall(struct.pack("!I", len(blob)))
        sock.sendall(blob)

    def _receive_json(self, sock: socket.socket) -> Dict[str, object]:
        length_bytes = self._recv_exact(sock, 4)
        if length_bytes is None:
            raise ConnectionError(
                "SAM-6D server closed connection before sending response length."
            )
        msg_len = struct.unpack("!I", length_bytes)[0]
        payload = self._recv_exact(sock, msg_len)
        if payload is None:
            raise ConnectionError(
                "SAM-6D server closed connection before sending response payload."
            )
        return json.loads(payload.decode("utf-8"))

    @staticmethod
    def _recv_exact(sock: socket.socket, nbytes: int) -> Optional[bytes]:
        data = bytearray()
        while len(data) < nbytes:
            chunk = sock.recv(nbytes - len(data))
            if not chunk:
                return None
            data.extend(chunk)
        return bytes(data)


# Backwards compatibility for existing scripts that still import SAM6DSocketClient
SAM6DSocketClient = GraspSolverSocketClient


class EEFBaseTeleop:
    """Manages key nudges of the active arm's EEF target pose (base frame) and calls whole-body IK."""

    def __init__(
        self,
        robot,
        env,
        sam6d_client: Optional[GraspSolverSocketClient] = None,
        contact_client: Optional[GraspSolverSocketClient] = None,
        contact_max_candidates: int = 5,
        sam6d_target_object: Optional[str] = None,
        sam6d_target_label: Optional[str] = None,
        sam6d_debug_plot: bool = False,
        grasp_viz: bool = True,
        grasp_provider: str = "sam6d",
    ):
        self.barge_door = False
        self.robot = robot
        self.env = env
        self.active_arm = "left"
        self.sam6d_client = sam6d_client
        self._sam6d_target_object = sam6d_target_object
        self._sam6d_target_label = (
            sam6d_target_label.lower() if sam6d_target_label else None
        )
        if sam6d_debug_plot and plt is None:
            print("[SAM-6D] Matplotlib unavailable; disabling debug plot.")
            sam6d_debug_plot = False
        self._sam6d_debug_plot = sam6d_debug_plot
        self._sam6d_intrinsics_warned = False
        self._sam6d_capture_pending = False
        self._sam6d_inflight = False
        self._pending_sam6d_pose: Optional[Dict[str, np.ndarray]] = None
        self._pending_sam6d_pose_base: Optional[
            Tuple[str, th.Tensor, th.Tensor, str]
        ] = None
        self._pending_sam6d_error: Optional[str] = None
        self._sam6d_last_cam_pose: Optional[Tuple[np.ndarray, np.ndarray, str]] = None
        self._sam6d_warned_label = False
        self._sam6d_warned_object = False
        self._sam6d_target_category: Optional[str] = None
        self._sam6d_target_model: Optional[str] = None
        self.contact_client = contact_client
        self._contact_max_candidates = max(1, int(contact_max_candidates))
        self._contact_capture_pending = False
        self._contact_inflight = False
        self._pending_contact_error: Optional[str] = None
        self._pending_contact_pose_base: Optional[
            Tuple[str, th.Tensor, th.Tensor, str]
        ] = None
        self._contact_candidate_queue: List[Tuple[str, th.Tensor, th.Tensor, str]] = []
        self._grasp_provider = grasp_provider.lower()
        # desired poses per arm, initialized to current EEF pose
        self.desired: Dict[str, Tuple[th.Tensor, th.Tensor]] = {}
        for arm in robot.arm_names:
            # Use gripper link to match the IK chain's end-effector convention
            ee_name = robot.gripper_link_names[arm][0]
            self.desired[arm] = get_pose_in_base(robot, ee_name)
        # Optional grasp visualization state
        self._grasp_viz_enabled = grasp_viz and not gm.HEADLESS
        self._grasp_viz_pose: Dict[str, Optional[Tuple[th.Tensor, th.Tensor]]] = {
            arm: None for arm in robot.arm_names
        }
        self._grasp_viz_source: Dict[str, str] = {arm: "" for arm in robot.arm_names}
        self._grasp_viz_line_scale = 0.18  # axis length in meters
        self._grasp_viz_line_size = 2.0
        self._grasp_viz_warned = False
        self._grasp_viz_drawn = False
        self._base_link_name = getattr(robot, "base_footprint_link_name", None)

        # step sizes
        self.pos_step = (
            0.025  # meters per repeat (slightly larger to reduce required IK steps)
        )
        self.rot_step = th.deg2rad(th.tensor(5.0))  # radians per repeat

        # Base motion step sizes (HolonomicBaseJointController expects [dx, dy, drz] deltas)
        self.base_lin_step = 0.20
        self.base_yaw_step = 0.35
        # Head (camera) control steps (radians)
        self.head_pitch_step = th.deg2rad(th.tensor(5.0))
        self.head_yaw_step = th.deg2rad(th.tensor(5.0))
        # Head bindings + per-step accumulators
        # Pitch prefers 'torso_joint4'; Yaw prefers 'torso_joint3'
        self._head_binding: Optional[Tuple[str, int, str]] = (
            None  # pitch: (component, action_index, joint_name)
        )
        self._head_nudge: float = 0.0
        self._head_yaw_binding: Optional[Tuple[str, int, str]] = None  # yaw
        self._head_yaw_nudge: float = 0.0

        self.ik = ChrisIKAdapter()
        # Optional one-shot action set from CLI target
        self._pending_action: Optional[th.Tensor] = None
        # Target change tracking and IK throttling
        self._target_dirty = True
        self._last_solve_t = 0.0
        self._ik_min_period = 1.0 / 15.0  # throttle to ~15 Hz max
        # Global planner call counter
        self._planner_call_count = 0
        # Store last desired pose to detect actual changes
        self._last_desired: Dict[str, Tuple[th.Tensor, th.Tensor]] = {
            arm: (self.desired[arm][0].clone(), self.desired[arm][1].clone())
            for arm in robot.arm_names
        }
        # Per-step base nudge accumulator
        self._base_nudge = th.zeros(3)

        # Auto-drive-to-target state
        self._drive_active: bool = False
        self._drive_target: Optional[Tuple[th.Tensor, th.Tensor]] = None
        self._drive_label: str = ""
        self._drive_pos_thresh: float = 0.01  # meters
        self._drive_ori_thresh: float = float(th.deg2rad(th.tensor(5.0)))  # radians
        self._drive_start_time: float = 0.0
        self._drive_last_report_t: float = 0.0
        self._drive_timeout: float = 600.0  # seconds before giving up
        self._drive_best_pos_err: float = 0.02
        self._drive_best_ori_err: float = 2
        self._drive_last_improve_t: float = 0.0

        # Pending camera-frame object target to apply in the main loop
        self._pending_cam_target: Optional[dict] = None
        # Pending auto-drive target to start once target has been applied
        self._pending_drive_target: Optional[Tuple[th.Tensor, th.Tensor, str]] = None
        # Optional correction to align camera-frame grasp orientation with IK/gripper frame, make ee gripper perpendicular to grasp point
        self._eef_rot_correction = th.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=th.float32
        )
        # Cache previous IK action to keep applying between solves
        self._last_ik_action: Optional[th.Tensor] = None

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
        self._gripper_hold_level = 0.99
        self._gripper_hold_command = {}
        for arm in robot.arm_names:
            ctrl_name = f"gripper_{arm}"
            info = self.controller_info.get(ctrl_name)
            if info is None:
                continue
            dim = info["command_dim"]
            self._gripper_hold_command[arm] = th.full(
                (dim,), self._gripper_hold_level, dtype=th.float32
            )

        # Register keyboard inputs
        self._register_keys()
        # Note: Avoid configuring camera / changing viewport at init to keep Replicator stable

    def _register_keys(self):
        # Toggle active arm
        KeyboardEventHandler.add_keyboard_callback(
            lazy.carb.input.KeyboardInput.KEY_3, lambda: self._set_arm("left")
        )
        KeyboardEventHandler.add_keyboard_callback(
            lazy.carb.input.KeyboardInput.KEY_4, lambda: self._set_arm("right")
        )

        # Position nudges (base frame)
        KeyboardEventHandler.add_keyboard_callback(
            lazy.carb.input.KeyboardInput.UP,
            lambda: self._bump_xyz(+self.pos_step, 0, 0),
        )
        KeyboardEventHandler.add_keyboard_callback(
            lazy.carb.input.KeyboardInput.DOWN,
            lambda: self._bump_xyz(-self.pos_step, 0, 0),
        )
        KeyboardEventHandler.add_keyboard_callback(
            lazy.carb.input.KeyboardInput.LEFT,
            lambda: self._bump_xyz(0, +self.pos_step, 0),
        )
        KeyboardEventHandler.add_keyboard_callback(
            lazy.carb.input.KeyboardInput.RIGHT,
            lambda: self._bump_xyz(0, -self.pos_step, 0),
        )
        KeyboardEventHandler.add_keyboard_callback(
            lazy.carb.input.KeyboardInput.P,
            lambda: self._bump_xyz(0, 0, +self.pos_step),
        )
        KeyboardEventHandler.add_keyboard_callback(
            lazy.carb.input.KeyboardInput.SEMICOLON,
            lambda: self._bump_xyz(0, 0, -self.pos_step),
        )

        # Orientation nudges (base frame axes: x,y,z)
        KeyboardEventHandler.add_keyboard_callback(
            lazy.carb.input.KeyboardInput.N,
            lambda: self._bump_rpy(+self.rot_step, 0, 0),
        )
        KeyboardEventHandler.add_keyboard_callback(
            lazy.carb.input.KeyboardInput.B,
            lambda: self._bump_rpy(-self.rot_step, 0, 0),
        )
        KeyboardEventHandler.add_keyboard_callback(
            lazy.carb.input.KeyboardInput.O,
            lambda: self._bump_rpy(0, +self.rot_step, 0),
        )
        KeyboardEventHandler.add_keyboard_callback(
            lazy.carb.input.KeyboardInput.U,
            lambda: self._bump_rpy(0, -self.rot_step, 0),
        )
        KeyboardEventHandler.add_keyboard_callback(
            lazy.carb.input.KeyboardInput.V,
            lambda: self._bump_rpy(0, 0, +self.rot_step),
        )
        KeyboardEventHandler.add_keyboard_callback(
            lazy.carb.input.KeyboardInput.C,
            lambda: self._bump_rpy(0, 0, -self.rot_step),
        )

        # Reset EEF target to current pose
        KeyboardEventHandler.add_keyboard_callback(
            lazy.carb.input.KeyboardInput.SPACE, self._reset_target_pose
        )
        # CLI input for explicit EEF pose/IK
        KeyboardEventHandler.add_keyboard_callback(
            lazy.carb.input.KeyboardInput.G, self._cli_goal_prompt
        )
        # Paste object pose as JSON with keys R (3x3) and t (3,) in CAMERA frame
        KeyboardEventHandler.add_keyboard_callback(
            lazy.carb.input.KeyboardInput.K, self._cli_object_pose_json
        )

        # Base nudges: W/S forward/back (x), A/D strafe left/right (y), Q/E rotate CCW/CW (rz)
        KeyboardEventHandler.add_keyboard_callback(
            lazy.carb.input.KeyboardInput.W,
            lambda: self._nudge_base(+self.base_lin_step, 0.0, 0.0),
        )
        KeyboardEventHandler.add_keyboard_callback(
            lazy.carb.input.KeyboardInput.S,
            lambda: self._nudge_base(-self.base_lin_step, 0.0, 0.0),
        )
        KeyboardEventHandler.add_keyboard_callback(
            lazy.carb.input.KeyboardInput.A,
            lambda: self._nudge_base(0.0, +self.base_lin_step, 0.0),
        )
        KeyboardEventHandler.add_keyboard_callback(
            lazy.carb.input.KeyboardInput.D,
            lambda: self._nudge_base(0.0, -self.base_lin_step, 0.0),
        )
        KeyboardEventHandler.add_keyboard_callback(
            lazy.carb.input.KeyboardInput.Q,
            lambda: self._nudge_base(0.0, 0.0, +self.base_yaw_step),
        )
        KeyboardEventHandler.add_keyboard_callback(
            lazy.carb.input.KeyboardInput.E,
            lambda: self._nudge_base(0.0, 0.0, -self.base_yaw_step),
        )
        # Head tilt (joint-based, pitch)
        KeyboardEventHandler.add_keyboard_callback(
            lazy.carb.input.KeyboardInput.PAGE_UP,
            lambda: self._tilt_head(+float(self.head_pitch_step)),
        )
        KeyboardEventHandler.add_keyboard_callback(
            lazy.carb.input.KeyboardInput.PAGE_DOWN,
            lambda: self._tilt_head(-float(self.head_pitch_step)),
        )
        # Head turn (joint-based, yaw) left / right
        # Use J (left) and L (right) to avoid overlap with EEF keys
        KeyboardEventHandler.add_keyboard_callback(
            lazy.carb.input.KeyboardInput.J,
            lambda: self._yaw_head(+float(self.head_yaw_step)),
        )
        KeyboardEventHandler.add_keyboard_callback(
            lazy.carb.input.KeyboardInput.L,
            lambda: self._yaw_head(-float(self.head_yaw_step)),
        )
        if self._grasp_provider == "sam6d" and self.sam6d_client is not None:
            KeyboardEventHandler.add_keyboard_callback(
                lazy.carb.input.KeyboardInput.H, self._trigger_sam6d_capture
            )
        elif self._grasp_provider == "contact" and self.contact_client is not None:
            KeyboardEventHandler.add_keyboard_callback(
                lazy.carb.input.KeyboardInput.H, self._trigger_contact_capture
            )

    def _set_arm(self, arm: str):
        if arm in self.robot.arm_names:
            self.active_arm = arm
            print(f"Active arm set to: {arm}")

    def _bump_xyz(self, dx: float, dy: float, dz: float):
        pos, quat = self.desired[self.active_arm]
        self.desired[self.active_arm] = (
            pos + th.tensor([dx, dy, dz], dtype=pos.dtype, device=pos.device),
            quat,
        )
        self._target_dirty = True

    def _bump_rpy(self, droll: th.Tensor, dpitch: th.Tensor, dyaw: th.Tensor):
        pos, quat = self.desired[self.active_arm]
        # Convert small RPY (base frame) to quaternion and post-multiply
        rot = T.euler2quat(
            th.tensor([droll, dpitch, dyaw], dtype=quat.dtype, device=quat.device)
        )
        new_quat = T.quat_multiply(quat, rot)
        self.desired[self.active_arm] = (pos, new_quat)
        self._target_dirty = True

    def _nudge_base(self, dx: float, dy: float, drz: float):
        # Accumulate base motion for a single simulation step
        self._base_nudge += th.tensor([dx, dy, drz])

    def _find_controller_for_joint_idx(
        self, joint_idx: int
    ) -> Optional[Tuple[str, int]]:
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

        # First, explicitly prefer torso_joint4 for pitch if available
        if "torso_joint4" in names:
            candidates.append(names.index("torso_joint4"))
        # Otherwise, search for a head/camera tilt/pitch joint
        add(
            lambda n: ("head" in n or "zed" in n or "camera" in n)
            and ("tilt" in n or "pitch" in n)
        )
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
            print(
                f"[Teleop] Head tilt bound to joint '{joint_name}' via component '{comp}'."
            )
            return comp, start + local, joint_name
        print(
            "[Teleop] Unable to bind a head tilt joint; verify robot has a head/torso tilt DOF."
        )
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
        # Explicitly prefer torso_joint3 for yaw if available
        if "torso_joint3" in names:
            candidates.append(names.index("torso_joint3"))
        # Otherwise, search for a head/camera yaw joint
        for i, n in enumerate(lc):
            if ("head" in n or "zed" in n or "camera" in n) and (
                "yaw" in n or "pan" in n
            ):
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
            print(
                f"[Teleop] Head yaw bound to joint '{joint_name}' via component '{comp}'."
            )
            return comp, start + local, joint_name
        print(
            "[Teleop] Unable to bind a head yaw joint; verify robot has a head/torso yaw DOF."
        )
        return None

    def _yaw_head(self, dyaw: float):
        if self._head_yaw_binding is None:
            self._head_yaw_binding = self._bind_head_yaw_joint()
        if self._head_yaw_binding is None:
            return
        self._head_yaw_nudge += float(dyaw)

    def _trigger_sam6d_capture(self):
        if self.sam6d_client is None:
            print("[SAM-6D] Client not configured. Set host / port to enable capture.")
            return
        if self._sam6d_inflight:
            print("[SAM-6D] Request already in flight; waiting for server response.")
            return
        self._sam6d_capture_pending = True
        print("[SAM-6D] Capture queued. Data will be sent on the next control tick.")

    def _trigger_contact_capture(self):
        if self.contact_client is None:
            print("[Contact] Client not configured. Set host / port to enable capture.")
            return
        if self._contact_inflight:
            print("[Contact] Request already in flight; waiting for server response.")
            return
        self._contact_capture_pending = True
        print("[Contact] Capture queued. Data will be sent on the next control tick.")

    def _apply_default_gripper_commands(self, action: th.Tensor) -> None:
        for arm, cmd in self._gripper_hold_command.items():
            ctrl_name = f"gripper_{arm}"
            info = self.controller_info.get(ctrl_name)
            if info is None:
                continue
            start = info["start_idx"]
            end = start + info["command_dim"]
            if end > action.shape[0]:
                continue
            action[start:end] = cmd.to(device=action.device, dtype=action.dtype)

    def _compute_intrinsics(
        self, sensor, sensor_name: Optional[str], rgb_shape: Tuple[int, int]
    ) -> np.ndarray:
        intrinsics = None
        if hasattr(sensor, "initialize_sensors"):
            try:
                sensor.initialize_sensors(names="camera_params")
            except Exception:
                pass
        # First try pulling from camera parameters with a few render updates if needed
        for attempt in range(6):
            try:
                params = sensor.camera_parameters
                proj = params.get("cameraProjection")
                resolution = params.get("renderProductResolution")
                if proj is not None and resolution is not None:
                    proj_np = (
                        proj.detach().cpu().numpy()
                        if hasattr(proj, "detach")
                        else np.asarray(proj)
                    )
                    res_np = (
                        resolution.detach().cpu().numpy()
                        if hasattr(resolution, "detach")
                        else np.asarray(resolution)
                    )
                    proj_np = proj_np.astype(np.float64).reshape(4, 4)
                    res_np = res_np.astype(np.float64)
                    width = float(res_np[0])
                    height = float(res_np[1])
                    fx = proj_np[0, 0] * width * 0.5
                    fy = proj_np[1, 1] * height * 0.5
                    cx = (1.0 - proj_np[0, 2]) * width * 0.5
                    cy = (1.0 - proj_np[1, 2]) * height * 0.5
                    intrinsics = np.array(
                        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                        dtype=np.float32,
                    )
            except Exception:
                intrinsics = None
            if (
                intrinsics is not None
                and np.all(np.isfinite(intrinsics))
                and np.any(intrinsics)
                and intrinsics[0, 0] > 0
                and intrinsics[1, 1] > 0
            ):
                break
            try:
                og.sim.render()
            except Exception:
                pass
        # Fallback to known calibration values scaled to current resolution
        if (
            intrinsics is None
            or not np.all(np.isfinite(intrinsics))
            or not np.any(intrinsics)
        ):
            camera_id = None
            for cid, name in ROBOT_CAMERA_NAMES.get("R1Pro", {}).items():
                if name == sensor_name:
                    camera_id = cid
                    break
            if camera_id:
                base_intr = CAMERA_INTRINSICS["R1Pro"][camera_id].astype(np.float32)
                intrinsics = base_intr.copy()
                height, width = rgb_shape
                base_cx = base_intr[0, 2]
                base_cy = base_intr[1, 2]
                base_width = base_cx * 2.0
                base_height = base_cy * 2.0
                if base_width > 0 and base_height > 0:
                    scale_x = width / base_width
                    scale_y = height / base_height
                    intrinsics[0, 0] *= scale_x
                    intrinsics[0, 2] *= scale_x
                    intrinsics[1, 1] *= scale_y
                    intrinsics[1, 2] *= scale_y
            if (
                intrinsics is None
                or not np.all(np.isfinite(intrinsics))
                or not np.any(intrinsics)
            ):
                intrinsics = np.eye(3, dtype=np.float32)
                if not self._sam6d_intrinsics_warned:
                    print(
                        "[SAM-6D] Falling back to identity intrinsics; camera parameters unavailable."
                    )
                    self._sam6d_intrinsics_warned = True
        return intrinsics

    def _ensure_target_label_from_object(self):
        if self._sam6d_target_label or not self._sam6d_target_object:
            return
        env = getattr(self, "env", None)
        if env is None:
            return
        obj = None
        self._sam6d_target_category = None
        self._sam6d_target_model = None
        try:
            obj = env.scene.object_registry("name", self._sam6d_target_object)
        except Exception:
            obj = None
        if obj is None:
            target_lower = self._sam6d_target_object.lower()
            for candidate in getattr(env.scene, "objects", []):
                name = getattr(candidate, "name", "")
                name_lower = name.lower()
                if name_lower == target_lower:
                    obj = candidate
                    break
            if obj is None:
                base = target_lower.split("_", 1)[0]
                candidates = []
                for candidate in getattr(env.scene, "objects", []):
                    name = getattr(candidate, "name", "")
                    name_lower = name.lower()
                    if name_lower.startswith(base):
                        candidates.append(candidate)
                if candidates:
                    obj = candidates[0]
        if obj is not None:
            category = getattr(obj, "category", None)
            model = getattr(obj, "model", None)
            if category and model:
                self._sam6d_target_category = str(category)
                self._sam6d_target_model = str(model)
                refined = f"{category}_{model}"
                if refined.lower() != self._sam6d_target_object.lower():
                    self._sam6d_target_object = refined
                    print(
                        f"[SAM-6D] Refined SAM-6D target object to '{self._sam6d_target_object}'."
                    )
        if obj is None:
            if not self._sam6d_warned_object:
                print(
                    f"[SAM-6D] Could not find object named '{self._sam6d_target_object}' in the scene."
                )
                self._sam6d_warned_object = True
            return
        category = getattr(obj, "category", None)
        if not category:
            if not self._sam6d_warned_object:
                print(
                    f"[SAM-6D] Object '{self._sam6d_target_object}' has no category metadata; "
                    "specify --sam6d-target-label explicitly."
                )
                self._sam6d_warned_object = True
            return
        self._sam6d_target_label = str(category).lower()
        print(
            f"[SAM-6D] Using semantic label '{self._sam6d_target_label}' derived from object '{self._sam6d_target_object}'."
        )

    @staticmethod
    def _tensor_to_numpy(tensor) -> np.ndarray:
        if isinstance(tensor, th.Tensor):
            return tensor.detach().cpu().numpy()
        return np.asarray(tensor)

    def _capture_head_data(self, label: str) -> Optional[Dict[str, object]]:
        sensor_name, sensor = self._get_head_sensor()
        if sensor is None:
            print(f"[{label}] No head sensor available; unable to capture.")
            return None
        try:
            obs, _ = sensor.get_obs()
        except Exception as exc:
            print(f"[{label}] Failed to fetch sensor observations: {exc}")
            return None
        required_modalities = ["rgb", "depth_linear", "seg_semantic"]
        missing = [mod for mod in required_modalities if mod not in obs]
        if missing:
            print(
                f"[{label}] Missing modalities {missing}. Update robot obs_modalities to include them."
            )
            return None
        rgb = self._tensor_to_numpy(obs["rgb"])
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        if np.issubdtype(rgb.dtype, np.floating):
            rgb = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            rgb = rgb.astype(np.uint8, copy=False)
        depth = self._tensor_to_numpy(obs["depth_linear"]).astype(
            np.float32, copy=False
        )
        semantic = self._tensor_to_numpy(obs["seg_semantic"]).astype(
            np.int32, copy=False
        )
        intrinsics = self._compute_intrinsics(
            sensor, sensor_name, (rgb.shape[0], rgb.shape[1])
        )
        cam_link = (
            self._camera_link_from_sensor_name(sensor_name)
            if sensor_name is not None
            else None
        )
        if cam_link is None:
            cam_link = "zed_link"
        cam_pos_base, cam_quat_base = get_pose_in_base(self.robot, cam_link)
        cam_pos_np = self._tensor_to_numpy(cam_pos_base).astype(np.float32, copy=False)
        cam_quat_np = self._tensor_to_numpy(cam_quat_base).astype(
            np.float32, copy=False
        )
        payload: Dict[str, object] = {
            "sensor_name": sensor_name,
            "sensor": sensor,
            "rgb": rgb,
            "depth": depth,
            "semantic": semantic,
            "intrinsics": intrinsics,
            "cam_link": cam_link,
            "cam_pos": cam_pos_np,
            "cam_quat": cam_quat_np,
        }
        return payload

    def _start_sam6d_request(self):
        self._sam6d_capture_pending = False
        if self.sam6d_client is None:
            return
        self._ensure_target_label_from_object()
        capture = self._capture_head_data("SAM-6D")
        if capture is None:
            return
        sensor_name = capture["sensor_name"]
        rgb = capture["rgb"]
        depth = capture["depth"]
        semantic = capture["semantic"]
        intrinsics = capture["intrinsics"]
        cam_link = capture["cam_link"]
        cam_pos_np = capture["cam_pos"]
        cam_quat_np = capture["cam_quat"]
        self._sam6d_last_cam_pose = (
            cam_pos_np.copy(),
            cam_quat_np.copy(),
            self.active_arm,
        )
        print(self._sam6d_last_cam_pose, "----")

        mask = None
        target_id = None
        if self._sam6d_target_label:
            lookup = semantic_class_name_to_id()
            lookup_lower = {k.lower(): v for k, v in lookup.items()}
            label_key = self._sam6d_target_label.lower()
            if label_key in lookup_lower:
                target_id = int(lookup_lower[label_key])
                mask = (semantic == target_id).astype(np.uint8, copy=False)
            elif not self._sam6d_warned_label:
                print(
                    f"[SAM-6D] Unknown semantic label '{self._sam6d_target_label}'. Skipping mask generation."
                )
                self._sam6d_warned_label = True

        if mask is None:
            print(
                "[SAM-6D] No mask available (target label / object unresolved or absent in segmentation); cancelling request."
            )
            return
        if self._sam6d_debug_plot:
            plt.figure("SAM-6D Semantic Mask")
            plt.clf()
            plt.imshow(mask, cmap="magma")
            plt.title(f"Target mask (id={target_id}, label={self._sam6d_target_label})")
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.draw()
            plt.pause(0.001)
        arrays: Dict[str, np.ndarray] = {
            "rgb": rgb,
            "depth": depth,
            "seg_semantic": semantic,
            "intrinsics": intrinsics,
            "cam_pos_base": cam_pos_np,
            "cam_quat_base": cam_quat_np,
        }
        print(f"intrinsics:\n{intrinsics}")
        if mask is not None:
            arrays["mask"] = mask
        header: Dict[str, object] = {
            "type": "sam6d_request",
            "timestamp": time.time(),
            "sensor_name": sensor_name,
            "camera_link": cam_link,
            "rgb_shape": list(rgb.shape),
            "depth_shape": list(depth.shape),
            "seg_shape": list(semantic.shape),
            "semantic_ids": [int(x) for x in np.unique(semantic)],
            "target_label": self._sam6d_target_label,
            "target_object": self._sam6d_target_object,
            "target_category": self._sam6d_target_category,
            "target_model": self._sam6d_target_model,
            "target_id": target_id,
            "active_arm": self.active_arm,
        }
        self._sam6d_inflight = True
        host = getattr(self.sam6d_client, "host", "server")
        port = getattr(self.sam6d_client, "port", "?")
        print(
            f"[SAM-6D] Sending capture to {host}:{port} (H={rgb.shape[0]}, W={rgb.shape[1]})."
        )
        self.sam6d_client.request_pose(
            header=header,
            arrays=arrays,
            on_result=self._handle_sam6d_response,
            on_error=self._handle_sam6d_error,
        )

    def _handle_sam6d_response(self, message: Dict[str, object]):
        self._sam6d_inflight = False
        if not isinstance(message, dict):
            self._pending_sam6d_error = f"Unexpected response type: {type(message)}"
            return

        pose_dict = message.get("pose")
        rot = message.get("grasp_rotation") or message.get("R")
        trans = message.get("grasp_translation_m") or message.get("t")
        quat = message.get("quaternion") or message.get("q")
        if pose_dict and isinstance(pose_dict, dict):
            rot = rot or pose_dict.get("grasp_rotation")
            trans = (
                trans
                or pose_dict.get("grasp_translation_m")
                or pose_dict.get("position")
            )
            quat = quat or pose_dict.get("quaternion")
        if rot is not None:
            R = np.asarray(rot, dtype=np.float32)
            if R.shape != (3, 3):
                self._pending_sam6d_error = (
                    f"SAM-6D rotation must be 3x3. Got shape {R.shape}."
                )
                return
        elif quat is not None:
            quat_np = np.asarray(quat, dtype=np.float32).reshape(-1)
            if quat_np.size != 4:
                self._pending_sam6d_error = "SAM-6D quaternion must have 4 elements."
                return
            R = (
                T.quat2mat(th.tensor(quat_np, dtype=th.float32))
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        else:
            self._pending_sam6d_error = "SAM-6D response missing rotation / quaternion."
            return
        if trans is None:
            self._pending_sam6d_error = "SAM-6D response missing translation."
            return
        t = np.asarray(trans, dtype=np.float32).reshape(-1)
        if t.size != 3:
            self._pending_sam6d_error = (
                f"SAM-6D translation must have 3 elements. Got {t}."
            )
            return
        # Ensure rotation matrix is orthonormal
        try:
            U, S, Vh = np.linalg.svd(R)
            R = U @ Vh
            if np.linalg.det(R) < 0:
                U[:, -1] *= -1.0
                R = U @ Vh
        except Exception:
            pass
        arm = str(message.get("active_arm") or self.active_arm)
        cam_pose_np: Optional[Tuple[np.ndarray, np.ndarray]] = None
        if self._sam6d_last_cam_pose is not None:
            cam_pose_np = (self._sam6d_last_cam_pose[0], self._sam6d_last_cam_pose[1])
        print(
            f"[SAM-6D] Received pose from server for arm '{arm}': R=\n{R}\nt={t.tolist()}"
        )
        print(
            f"[camera pose] pos={cam_pose_np[0].tolist() if cam_pose_np is not None else None}"
        )
        try:
            pos_base, quat_base = self._compute_base_pose_from_camera(
                arm=arm,
                R_cam_obj_np=R,
                t_cam_obj_np=t[:3],
                cam_pose_base=cam_pose_np,
            )
            print(
                f"[SAM-6D] Base pose ({arm}) pos={pos_base.tolist()} quat={quat_base.tolist()}"
            )
            self._pending_sam6d_pose_base = (
                arm,
                pos_base.detach().clone(),
                quat_base.detach().clone(),
                "sam6d",
            )
            self._set_grasp_viz_pose(arm, pos_base, quat_base, source="sam6d")
            print("[SAM-6D] Pose converted to base frame; scheduling IK target.")
        except Exception as exc:
            self._pending_sam6d_pose = {
                "R": R,
                "t": t[:3],
                "cam_pos": cam_pose_np[0] if cam_pose_np is not None else None,
                "cam_quat": cam_pose_np[1] if cam_pose_np is not None else None,
            }
            print(
                f"[SAM-6D] Deferred base-frame conversion ({exc}); will convert in main loop."
            )
        finally:
            self._sam6d_last_cam_pose = None

    def _handle_sam6d_error(self, exc: Exception):
        self._sam6d_inflight = False
        self._pending_sam6d_error = str(exc)

    def _start_contact_request(self):
        self._contact_capture_pending = False
        if self.contact_client is None:
            return
        self._ensure_target_label_from_object()
        capture = self._capture_head_data("Contact")
        if capture is None:
            return
        rgb = capture["rgb"]
        depth = capture["depth"]
        semantic = capture["semantic"]
        intrinsics = capture["intrinsics"]
        cam_link = capture["cam_link"]
        cam_pos_np = capture["cam_pos"]
        cam_quat_np = capture["cam_quat"]
        self._contact_candidate_queue.clear()

        mask = None
        target_id = None
        if self._sam6d_target_label:
            lookup = semantic_class_name_to_id()
            lookup_lower = {k.lower(): v for k, v in lookup.items()}
            label_key = self._sam6d_target_label.lower()
            if label_key in lookup_lower:
                target_id = int(lookup_lower[label_key])
                mask = (semantic == target_id).astype(np.uint8, copy=False)

        if mask is None or not np.any(mask):
            print(
                "[Contact] No target mask available (target label / object unresolved or absent); cancelling request."
            )
            return

        contact_seg = np.zeros_like(semantic, dtype=np.int32)
        if target_id is None:
            contact_seg[mask > 0] = 1
        else:
            contact_seg[mask > 0] = target_id

        arrays: Dict[str, np.ndarray] = {
            "rgb": rgb,
            "depth": depth,
            "seg_semantic": contact_seg,
            "intrinsics": intrinsics,
            "cam_pos_base": cam_pos_np,
            "cam_quat_base": cam_quat_np,
        }
        arrays["mask"] = mask
        header: Dict[str, object] = {
            "type": "contact_grasp_request",
            "timestamp": time.time(),
            "sensor_name": capture["sensor_name"],
            "camera_link": cam_link,
            "rgb_shape": list(rgb.shape),
            "depth_shape": list(depth.shape),
            "seg_shape": list(semantic.shape),
            "semantic_ids": [int(x) for x in np.unique(semantic)],
            "active_arm": self.active_arm,
            "max_results": self._contact_max_candidates,
        }
        self._contact_inflight = True
        host = getattr(self.contact_client, "host", "server")
        port = getattr(self.contact_client, "port", "?")
        print(
            f"[Contact] Sending capture to {host}:{port} (H={rgb.shape[0]}, W={rgb.shape[1]})."
        )
        self.contact_client.request_pose(
            header=header,
            arrays=arrays,
            on_result=self._handle_contact_response,
            on_error=self._handle_contact_error,
        )

    def _handle_contact_response(self, message: Dict[str, object]):
        self._contact_inflight = False
        if not isinstance(message, dict):
            self._pending_contact_error = f"Unexpected response type: {type(message)}"
            return
        poses = message.get("poses")
        if not poses and "pose" in message:
            poses = [message["pose"]]
        if not isinstance(poses, list) or not poses:
            self._pending_contact_error = "Contact-GraspNet returned no poses."
            return
        arm = str(message.get("active_arm") or self.active_arm)
        if arm not in self.desired:
            self._pending_contact_error = (
                f"Contact-GraspNet response for unknown arm '{arm}'."
            )
            return
        queue: List[Tuple[str, th.Tensor, th.Tensor, str]] = []
        dtype = self.desired[arm][0].dtype
        device = self.desired[arm][0].device
        for idx, pose in enumerate(poses[: self._contact_max_candidates]):
            if not isinstance(pose, dict):
                continue
            pos = np.asarray(pose.get("position"), dtype=np.float32).reshape(-1)
            quat = np.asarray(
                pose.get("quaternion") or pose.get("quat") or pose.get("q"),
                dtype=np.float32,
            ).reshape(-1)
            if pos.size != 3 or quat.size != 4:
                continue
            pos_t = th.tensor(pos, dtype=dtype, device=device)
            norm = float(np.linalg.norm(quat))
            if norm <= 1e-6:
                continue
            quat_t = th.tensor(quat / norm, dtype=dtype, device=device)
            queue.append((arm, pos_t, quat_t, "contact"))
            print(
                f"[Contact] Candidate {idx}: pos={pos_t.tolist()} quat={quat_t.tolist()} "
                f"score={pose.get('score')}"
            )
        if not queue:
            self._pending_contact_error = (
                "Contact-GraspNet returned no valid pose entries."
            )
            return
        self._contact_candidate_queue = queue[1:]
        self._pending_contact_pose_base = queue[0]
        print(f"[Contact] Received {len(queue)} grasp candidates.")

    def _handle_contact_error(self, exc: Exception):
        self._contact_inflight = False
        self._pending_contact_error = str(exc)

    def _reset_target_pose(self):
        ee_name = self.robot.gripper_link_names[self.active_arm][0]
        self.desired[self.active_arm] = get_pose_in_base(self.robot, ee_name)
        print("Reset target pose to current EEF pose (base frame).")
        self._target_dirty = True
        self._clear_grasp_viz_pose(arm=self.active_arm)

    def _get_eef_pose(self, arm: Optional[str] = None) -> Tuple[th.Tensor, th.Tensor]:
        arm = arm or self.active_arm
        ee_name = self.robot.gripper_link_names[arm][0]
        return get_pose_in_base(self.robot, ee_name)

    def cancel_auto_drive(self) -> None:
        """Stop any auto-drive behavior and clear associated state."""
        self._drive_active = False
        self._drive_target = None
        self._pending_drive_target = None
        self._last_ik_action = None
        self._pending_action = None
        self._sam6d_capture_pending = False
        self._sam6d_inflight = False
        self._pending_sam6d_error = None
        self._pending_sam6d_pose_base = None
        self._pending_sam6d_pose = None
        self._contact_capture_pending = False
        self._contact_inflight = False
        self._pending_contact_error = None
        self._pending_contact_pose_base = None
        self._contact_candidate_queue.clear()
        self._pending_cam_target = None
        self._clear_grasp_visualization()

    def sync_desired_to_current_pose(self) -> None:
        """Refresh desired poses to match the robot's current state."""
        for arm_name in self.robot.arm_names:
            self.desired[arm_name] = self._get_eef_pose(arm_name)
            # Also update last_desired to match
            self._last_desired[arm_name] = (
                self.desired[arm_name][0].clone(),
                self.desired[arm_name][1].clone()
            )
        self._target_dirty = False

    def _begin_auto_drive(
        self, target_pos: th.Tensor, target_quat: th.Tensor, label: str = "target"
    ):
        self._drive_active = True
        self._drive_target = (target_pos.clone(), target_quat.clone())
        self._drive_label = label
        self._drive_start_time = time.monotonic()
        self._drive_last_report_t = self._drive_start_time
        self._drive_last_improve_t = self._drive_start_time
        self._last_ik_action = None

    def _maybe_report_drive_progress(self):
        if not self._drive_active or self._drive_target is None:
            return
        now = time.monotonic()
        # Only print every 0.5s to avoid flooding
        if now - self._drive_last_report_t < 0.5:
            return
        self._drive_last_report_t = now
        cur_pos, cur_quat = self._get_eef_pose(self.active_arm)
        tgt_pos, tgt_quat = self._drive_target
        pos_err = th.linalg.norm(cur_pos - tgt_pos).item()
        ori_delta = th.mean(
            th.abs(T.quat2axisangle(cur_quat) - T.quat2axisangle(tgt_quat))
        )
        print(
            f"[Drive] {self._drive_label}: t={now - self._drive_start_time:.2f}s cur_rot={cur_quat.tolist()} tgt_rot={tgt_quat.tolist()}"
        )
        if isinstance(ori_delta, th.Tensor):
            if ori_delta.numel() == 1:
                ori_err = float(ori_delta.detach().cpu().item())
            else:
                ori_err = float(th.linalg.norm(ori_delta).detach().cpu().item())
        else:
            ori_err = float(ori_delta)
        elapsed = now - self._drive_start_time
        print(
            f"[Drive] {self._drive_label}: t={elapsed:.2f}s pos_err={pos_err:.4f}m ori_err={th.rad2deg(th.tensor(ori_err)).item():.2f}deg"
        )

    def _evaluate_drive_goal(
        self,
        now: float,
        cur_pos: Optional[th.Tensor] = None,
        cur_quat: Optional[th.Tensor] = None,
    ) -> None:
        if not self._drive_active or self._drive_target is None:
            return
        if cur_pos is None or cur_quat is None:
            cur_pos, cur_quat = self._get_eef_pose(self.active_arm)
        tgt_pos, tgt_quat = self._drive_target
        pos_err = th.linalg.norm(cur_pos - tgt_pos).item()
        try:
            ori_delta = th.mean(
                th.abs(T.quat2axisangle(cur_quat) - T.quat2axisangle(tgt_quat))
            )
        except Exception:
            ori_delta = th.linalg.norm(cur_quat - tgt_quat)
        if isinstance(ori_delta, th.Tensor):
            if ori_delta.numel() == 1:
                ori_err = float(ori_delta.detach().cpu().item())
            else:
                ori_err = float(th.linalg.norm(ori_delta).detach().cpu().item())
        else:
            ori_err = float(ori_delta)
        if pos_err < 0.05 or ori_err < 5.0:
            self._drive_best_pos_err = pos_err
            self._drive_best_ori_err = ori_err
            self._drive_last_improve_t = now
        pos_ok = pos_err < 0.05
        ori_ok = ori_err < 5.0
        stagnation = now - self._drive_last_improve_t
        print(
            f"[Drive in step] {self._drive_label}: pos_err={pos_err:.4f}m ori_err={th.rad2deg(th.tensor(ori_err)).item():.2f}deg stagnation={stagnation:.2f}s"
        )
        if pos_ok and ori_ok:
            print(
                f"[EEF exec] {self.active_arm}: pos={cur_pos.tolist()} quat(xyzw)={cur_quat.tolist()} (reached)"
            )
            self._drive_active = False
            self._drive_target = None
            self._last_ik_action = None
            self._target_dirty = False
            self._clear_grasp_visualization()
            return
        elapsed = now - self._drive_start_time
        if pos_ok and stagnation > 1.0:
            print(
                f"[Drive] Position reached but orientation stagnant for {stagnation:.2f}s; accepting pose."
            )
            self._drive_active = False
            self._drive_target = None
            self._last_ik_action = None
            self._target_dirty = False
            self._clear_grasp_visualization()
        elif elapsed > self._drive_timeout or stagnation > self._drive_timeout:
            print(
                f"[Drive] Timeout ({elapsed:.1f}s). Leaving auto mode (pos_err={pos_err:.4f} ori_err={th.rad2deg(th.tensor(ori_err)).item():.2f}deg)"
            )
            self._drive_active = False
            self._drive_target = None
            self._last_ik_action = None
            self._target_dirty = False
            self._clear_grasp_visualization()

    def _schedule_target_from_camera_Rt(
        self,
        R_cam_obj: np.ndarray,
        t_cam_obj: np.ndarray,
        arm: Optional[str] = None,
        approach_offset_m: float = 0.0,
        source: str = "camera",
        cam_pose_base: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        """Schedule application of camera-frame object pose in the main loop to avoid heavy work in callbacks."""
        arm = arm or self.active_arm
        cam_pose_payload: Optional[Tuple[np.ndarray, np.ndarray]] = None
        if cam_pose_base is not None:
            cam_pos_np = np.asarray(cam_pose_base[0], dtype=float).reshape(3)
            cam_quat_np = np.asarray(cam_pose_base[1], dtype=float).reshape(4)
            cam_pose_payload = (cam_pos_np.copy(), cam_quat_np.copy())
        self._pending_cam_target = dict(
            R=np.asarray(R_cam_obj),
            t=np.asarray(t_cam_obj),
            arm=arm,
            approach=approach_offset_m,
            source=source,
            cam_pose=cam_pose_payload,
        )

    def _apply_scheduled_cam_target(self):
        if self._pending_cam_target is None:
            return
        try:
            arm = self._pending_cam_target["arm"]
            R_cam_obj = self._pending_cam_target["R"]
            t_cam_obj = self._pending_cam_target["t"]
            approach_offset_m = float(self._pending_cam_target.get("approach", 0.0))
            source = str(self._pending_cam_target.get("source", "camera"))
            cam_pose_override = self._pending_cam_target.get("cam_pose")
            # Get camera pose in base
            if cam_pose_override is not None:
                dtype = self.desired[arm][0].dtype
                device = self.desired[arm][0].device
                cam_pos_base = th.tensor(
                    cam_pose_override[0], dtype=dtype, device=device
                )
                cam_quat_base = th.tensor(
                    cam_pose_override[1], dtype=dtype, device=device
                )
            else:
                name, sensor = self._get_head_sensor()
                cam_link = (
                    self._camera_link_from_sensor_name(name)
                    if name is not None
                    else None
                )
                if cam_link is None:
                    cam_link = "zed_link"
                cam_pos_base, cam_quat_base = get_pose_in_base(self.robot, cam_link)
            T_base_cam = T.pose2mat((cam_pos_base, cam_quat_base))
            # Sanitize R to SO(3)
            R = th.tensor(
                np.asarray(R_cam_obj),
                dtype=cam_pos_base.dtype,
                device=cam_pos_base.device,
            )
            U, S, Vh = th.linalg.svd(R)
            R_hat = U @ Vh
            if th.linalg.det(R_hat) < 0:
                U[:, -1] *= -1
                R_hat = U @ Vh
            # Optional approach offset along -Z of object frame
            t = th.tensor(
                np.asarray(t_cam_obj),
                dtype=cam_pos_base.dtype,
                device=cam_pos_base.device,
            )

            T_cam_obj = th.eye(4, dtype=R_hat.dtype, device=R_hat.device)
            T_cam_obj[:3, :3] = R_hat
            T_cam_obj[:3, 3] = t
            # Compose base_T_obj and extract pose
            T_base_obj = T_base_cam @ T_cam_obj
            pos, quat = T.mat2pose(T_base_obj)
            # Print initial EEF pose
            # init_pos, init_quat = self._get_eef_pose(arm)
            # print(f"[EEF init] {arm}: pos={init_pos.tolist()} quat(xyzw)={init_quat.tolist()}")
            # Set desired target
            self.desired[arm] = (pos, quat)
            self._target_dirty = True
            # print(f"[EEF set] {arm} target from camera R,t. pos={pos.tolist()} quat={quat.tolist()}")
            self._set_grasp_viz_pose(arm, pos, quat, source=source)
            try:
                if self._base_link_name and self._base_link_name in self.robot.links:
                    base_link = self.robot.links[self._base_link_name]
                    base_pos_world, base_quat_world = (
                        base_link.get_position_orientation()
                    )
                else:
                    base_pos_world, base_quat_world = (
                        self.robot.get_position_orientation()
                    )
                T_world_base = T.pose2mat((base_pos_world, base_quat_world))
                T_world_obj = T_world_base @ T_base_obj
                world_pos, world_quat = T.mat2pose(T_world_obj)
                print(
                    f"[Grasp viz] {arm} {source}: world pos={world_pos.tolist()} quat={world_quat.tolist()}"
                )
            except Exception as exc:
                if not self._grasp_viz_warned:
                    print(f"[Grasp viz] Failed to report world pose: {exc}")
                    self._grasp_viz_warned = True
            # Defer auto drive start to main loop after this update completes
            self._pending_drive_target = (pos.clone(), quat.clone(), "camera-object")
        except Exception as e:
            print(f"[Camera target] Failed to apply: {e}")
        finally:
            self._pending_cam_target = None

    def _set_grasp_viz_pose(
        self, arm: str, pos_base: th.Tensor, quat_base: th.Tensor, source: str = ""
    ) -> None:
        if not self._grasp_viz_enabled:
            return
        try:
            self._grasp_viz_pose[arm] = (
                pos_base.detach().clone(),
                quat_base.detach().clone(),
            )
            self._grasp_viz_source[arm] = source
        except Exception as exc:
            if not self._grasp_viz_warned:
                print(f"[Grasp viz] Failed to cache pose: {exc}")
                self._grasp_viz_warned = True

    def _compute_base_pose_from_camera(
        self,
        arm: str,
        R_cam_obj_np: np.ndarray,
        t_cam_obj_np: np.ndarray,
        cam_pose_base: Optional[Tuple[np.ndarray, np.ndarray]],
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Convert a camera-frame pose (R_cam_obj, t_cam_obj) into a base-frame pose using the provided or current camera pose.
        """
        # Ensure desired entry exists to derive dtype / device
        if arm not in self.desired:
            raise ValueError(f"Unknown arm '{arm}' for SAM-6D target conversion.")
        dtype = self.desired[arm][0].dtype
        device = self.desired[arm][0].device

        if cam_pose_base is not None:
            cam_pos_base = th.tensor(cam_pose_base[0], dtype=dtype, device=device)
            cam_quat_base = th.tensor(cam_pose_base[1], dtype=dtype, device=device)
        else:
            name, sensor = self._get_head_sensor()
            cam_link = (
                self._camera_link_from_sensor_name(name) if name is not None else None
            )
            if cam_link is None:
                cam_link = "zed_link"
            cam_pos_base, cam_quat_base = get_pose_in_base(self.robot, cam_link)
            cam_pos_base = cam_pos_base.to(dtype=dtype, device=device)
            cam_quat_base = cam_quat_base.to(dtype=dtype, device=device)

        R_cam_obj = th.tensor(R_cam_obj_np, dtype=dtype, device=device)
        t_cam_obj = th.tensor(t_cam_obj_np, dtype=dtype, device=device)

        T_base_cam = T.pose2mat((cam_pos_base, cam_quat_base))
        T_cam_obj = th.eye(4, dtype=dtype, device=device)
        T_cam_obj[:3, :3] = R_cam_obj
        T_cam_obj[:3, 3] = t_cam_obj
        T_base_obj = T_base_cam @ T_cam_obj

        pos_base_obj = T_base_obj[:3, 3]
        R_base_obj = T_base_obj[:3, :3]
        R_corr = self._eef_rot_correction.to(dtype=dtype, device=device)
        # R_base_obj = R_base_obj @ R_corr
        quat_base_obj = T.mat2quat(R_base_obj)
        return pos_base_obj, quat_base_obj

    def _clear_grasp_viz_pose(self, arm: Optional[str] = None) -> None:
        if not self._grasp_viz_enabled:
            return
        if arm is None:
            for k in self._grasp_viz_pose:
                self._grasp_viz_pose[k] = None
                self._grasp_viz_source[k] = ""
            if all(v is None for v in self._grasp_viz_pose.values()):
                self._clear_grasp_visualization()
        elif arm in self._grasp_viz_pose:
            self._grasp_viz_pose[arm] = None
            self._grasp_viz_source[arm] = ""
            if all(v is None for v in self._grasp_viz_pose.values()):
                self._clear_grasp_visualization()

    def _clear_grasp_visualization(self) -> None:
        if not self._grasp_viz_enabled or not self._grasp_viz_drawn:
            return
        try:
            clear_debug_drawing()
        except Exception as exc:
            if not self._grasp_viz_warned:
                print(f"[Grasp viz] Failed to clear debug drawing: {exc}")
                self._grasp_viz_warned = True
        finally:
            self._grasp_viz_drawn = False

    @staticmethod
    def _scale_axis_color(
        color: Tuple[float, float, float, float], active: bool
    ) -> Tuple[float, float, float, float]:
        if active:
            return color
        dim = 0.45
        return (color[0] * dim, color[1] * dim, color[2] * dim, color[3])

    def _draw_grasp_visualization(self) -> None:
        if not self._grasp_viz_enabled:
            return
        if not self._drive_active or self._drive_target is None:
            if self._grasp_viz_drawn:
                self._clear_grasp_visualization()
            self._grasp_viz_drawn = False
            return
        if all(pose is None for pose in self._grasp_viz_pose.values()):
            if self._grasp_viz_drawn:
                self._clear_grasp_visualization()
                self._grasp_viz_drawn = False
            return
        try:
            if self._grasp_viz_drawn:
                self._clear_grasp_visualization()
            if self._base_link_name and self._base_link_name in self.robot.links:
                base_link = self.robot.links[self._base_link_name]
                base_pos, base_quat = base_link.get_position_orientation()
            else:
                base_pos, base_quat = self.robot.get_position_orientation()
            base_T = T.pose2mat((base_pos, base_quat))
            axis_colors = (
                (1.0, 0.15, 0.15, 1.0),  # x: red
                (0.15, 1.0, 0.15, 1.0),  # y: green
                (0.15, 0.55, 1.0, 1.0),  # z: blue
            )
            drawn_any = False
            for arm, pose in self._grasp_viz_pose.items():
                if pose is None:
                    continue
                pos_base, quat_base = pose
                pos_base = pos_base.to(device=base_pos.device, dtype=base_pos.dtype)
                quat_base = quat_base.to(device=base_quat.device, dtype=base_quat.dtype)
                target_T = T.pose2mat((pos_base, quat_base))
                world_T = base_T @ target_T
                origin = self._tensor_to_numpy(world_T[:3, 3])
                axes = self._tensor_to_numpy(world_T[:3, :3])
                if origin is None or axes is None:
                    continue
                origin = origin.astype(np.float32)
                axes = axes.astype(np.float32)
                active = arm == self.active_arm
                for idx, color in enumerate(axis_colors):
                    direction = axes[:, idx]
                    end = origin + direction * self._grasp_viz_line_scale
                    draw_line(
                        origin.tolist(),
                        end.tolist(),
                        color=self._scale_axis_color(color, active),
                        size=(
                            self._grasp_viz_line_size
                            if active
                            else self._grasp_viz_line_size * 0.75
                        ),
                    )
                approach_dir = -axes[:, 2]
                approach_end = origin + approach_dir * (
                    self._grasp_viz_line_scale * 0.6
                )
                approach_color = (
                    (1.0, 0.8, 0.25, 1.0) if active else (0.6, 0.5, 0.25, 1.0)
                )
                draw_line(
                    origin.tolist(),
                    approach_end.tolist(),
                    color=approach_color,
                    size=(
                        self._grasp_viz_line_size
                        if active
                        else self._grasp_viz_line_size * 0.75
                    ),
                )
                drawn_any = True
            self._grasp_viz_drawn = drawn_any
        except Exception as exc:
            if not self._grasp_viz_warned:
                print(f"[Grasp viz] Visualization disabled: {exc}")
                self._grasp_viz_warned = True
            self._grasp_viz_enabled = False

    # ---------- Head camera helpers ----------
    def _get_head_sensor(self):
        # Heuristic: prefer sensor with 'zed' in name; else a camera not containing 'realsense'
        for name, sensor in self.robot.sensors.items():
            if hasattr(sensor, "get_obs") and ("zed" in name):
                return name, sensor
        for name, sensor in self.robot.sensors.items():
            if (
                hasattr(sensor, "get_obs")
                and ("Camera" in name)
                and ("realsense" not in name)
            ):
                return name, sensor
        # Fallback to any vision sensor
        for name, sensor in self.robot.sensors.items():
            if hasattr(sensor, "get_obs"):
                return name, sensor
        return None, None

    def _camera_link_from_sensor_name(self, sensor_name: str) -> Optional[str]:
        # Extract a link candidate like 'zed_link' or 'realsense_link' from sensor key
        parts = sensor_name.split(":")
        for p in parts:
            if p.endswith("_link"):
                return p
        # Common known
        if "zed" in sensor_name:
            return "zed_link"
        if "realsense" in sensor_name:
            return "realsense_link"
        return None

    def _print_current_pose_and_joints(self):
        arm = self.active_arm
        ee_name = self.robot.gripper_link_names[arm][0]
        pos, quat = get_pose_in_base(self.robot, ee_name)
        print(
            f"Current {arm} EEF (base): pos={pos.tolist()} quat(xyzw)={quat.tolist()}"
        )
        names = list(self.robot.joints.keys())
        q = self.robot.get_joint_positions()
        print("Current joints (trunk + arm):")
        for grp in ("trunk", f"arm_{arm}"):
            if grp not in self.controller_info:
                continue
            dof_idx = self.controller_info[grp]["dofs"].tolist()
            for j_idx in dof_idx:
                print(f"  {names[j_idx]}: {float(q[j_idx]):.6f}")
        return pos, quat

    def _build_action_from_targets(self, targets: Dict[str, float]) -> th.Tensor:
        action = th.zeros(self.robot.action_dim)
        names = list(self.robot.joints.keys())
        q = self.robot.get_joint_positions()
        for grp in ("trunk", f"arm_{self.active_arm}"):
            if grp not in self.controller_info:
                continue
            start = self.controller_info[grp]["start_idx"]
            dof_idx = self.controller_info[grp]["dofs"].tolist()
            for local_i, j_idx in enumerate(dof_idx):
                j_name = names[j_idx]
                if j_name in targets:
                    action[start + local_i] = float(targets[j_name]) - float(q[j_idx])
        return action

    def _cli_goal_prompt(self):
        print(
            "\n[CLI] Image dump / manual entry disabled. Use key K with JSON pose input."
        )

    def _cli_object_pose_json(self):
        """Prompt for object pose JSON with keys "R" (3x3) and "t" (3,) in CAMERA frame and set EEF target."""
        print(
            "\n[CLI] Paste object pose JSON with keys 'R' (3x3) and 't' (3,) in CAMERA frame."
        )
        try:
            s = input(
                'JSON {"R": [[...],[...],[...]], "t": [x,y,z]} (blank to cancel): '
            ).strip()
        except EOFError:
            s = ""
        if not s:
            print("Cancelled.")
            return
        try:
            obj = json.loads(s)
            R = np.asarray(obj.get("R", []), dtype=float)
            t = np.asarray(obj.get("t", []), dtype=float)
            if R.shape != (3, 3) or t.shape != (3,):
                print(
                    f"Invalid shapes: R{R.shape}, t{t.shape}. Expected (3,3) and (3,)."
                )
                return
            # Schedule application in the main loop to avoid heavy operations inside the key callback
            self._schedule_target_from_camera_Rt(
                R, t, arm=self.active_arm, approach_offset_m=0.0, source="cli"
            )
            print("Scheduled camera-object target; driving will begin next frame.")
        except Exception as e:
            print(f"JSON parse / transform error: {e}")

    def step(self) -> th.Tensor:
        """Compute action vector for the robot by solving IK for the active arm.

        Returns: action tensor of shape (action_dim,)
        """
        # print("[Test 3] EEFBaseTeleop.step() called")
        self._draw_grasp_visualization()

        if self._pending_action is not None:
            print("[Test 4] Returning pending action")
            act = self._pending_action
            print(f"[Test 5] Action shape: {act.shape}")
            self._pending_action = None
            return act

        # print("[Test 6] Initializing zero action")
        action = th.zeros(self.robot.action_dim)
        # breakpoint()
        
        # print(f"[Test 7] Action shape: {action.shape}")
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
            print("+++++++++++++++++++++++++++++++++++++++++")
            # breakpoint()  # SAM-6D response received, about to set desired pose
            # self.desired[arm_name] = (pos_target.clone(), quat_target.clone())
            try:

                from scipy.spatial.transform import Rotation as Rot

                # R1Pro gripper convention (in base frame):
                # [0,0,0,1] = identity = gripper points DOWN (default orientation) ✓
                # [0,1,0,0] = 180° Y-rotation = gripper points UP (flipped upside down)
                # The gripper's natural/rest orientation already faces downward!
                # quat_target_r = th.tensor([0.7071, 0, 0, 0.7071])
                # quat_target_l = th.tensor([-0.7071, 0, 0, 0.7071])
                quat_target = th.tensor([0,0,0,1])
                # quat_target = th.tensor([+0.0006, -0.0055, -0.0746, +0.9972])
                pos_target = th.tensor([0.35, 0.15,  1.0])
                pos_target_r = th.tensor([0.0, -0.5,  1.0])
                self.desired['left'] = (pos_target, quat_target)
                self.desired['right'] = (pos_target_r, quat_target)
                self.barge_door = True
                self._target_dirty = True
            except:
                import traceback as tb
                tb.print_exc()
            self._set_grasp_viz_pose(arm_name, pos_target, quat_target, source)
            self._pending_drive_target = (
                pos_target.clone(),
                quat_target.clone(),
                source,
            )
            self._pending_sam6d_pose_base = None
        if self._pending_contact_pose_base is not None:
            arm_name, pos_target, quat_target, source = self._pending_contact_pose_base
            if arm_name != self.active_arm:
                self._set_arm(arm_name)
            print("------------------------------------------")
            self.desired[arm_name] = (pos_target.clone(), quat_target.clone())
            # self.desired['left'] = (tensor([0.70,  0.20,  0.5]), tensor([ 0, 1, 0, 0]))
            # self.desired['right'] = (tensor([0.70,  -0.20,  0.5]), tensor([ 0, 1, 0, 0]))
            self._target_dirty = True
            self._set_grasp_viz_pose(arm_name, pos_target, quat_target, source)
            self._pending_drive_target = (
                pos_target.clone(),
                quat_target.clone(),
                source,
            )
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
        try:
            # Check if desired pose has actually changed for any arm
            pose_changed = False
            for arm in self.robot.arm_names:
                pos_cur, quat_cur = self.desired[arm]
                pos_last, quat_last = self._last_desired[arm]

                pos_diff = th.linalg.norm(pos_cur - pos_last).item()
                # print("POSITION DIFF", pos_diff)
                # Compare quaternions by checking if they're close or negated (equivalent rotations)
                # quat_diff = min(
                #     th.linalg.norm(quat_cur - quat_last).item(),
                #     th.linalg.norm(quat_cur + quat_last).item()
                # )

                if pos_diff > 0.01:# or quat_diff > 0.01:
                    pose_changed = True
                    break

            # Only set target_dirty if the pose actually changed
            if pose_changed and not self._target_dirty:
                self._target_dirty = True
                # print(f"[IK] Target pose changed - marking dirty (pos_diff={pos_diff:.6f}, quat_diff={quat_diff:.6f})")

            need_ik = self._target_dirty #and time_since_last >= self._ik_min_period
            if not need_ik:
                # No IK needed - just apply head adjustments if any and return zero action
                # action = self._last_ik_action.clone()
                if pitch_idx is not None:
                    action[pitch_idx] += pitch_val
                if yaw_idx is not None:
                    action[yaw_idx] += yaw_val
                self._head_nudge = 0.0
                self._head_yaw_nudge = 0.0
                # action += self._last_ik_action
                return action

            # Get current joint positions and robot info (skip first 6 base footprint joints)
            all_joint_positions = self.robot.get_joint_positions()
            all_joint_names = list(self.robot.joints.keys())

            joint_positions = all_joint_positions[6:]
            joint_names = all_joint_names[6:]
            joint_limits = {
                name: (float(joint.lower_limit), float(joint.upper_limit))
                for name, joint in list(self.robot.joints.items())[6:]
                if hasattr(joint, 'lower_limit') and hasattr(joint, 'upper_limit')
            }
            # breakpoint()
            # print("jointname", joint_names)
            # joint_names = ['torso_joint1', 'torso_joint2', 'torso_joint3', 'torso_joint4', 'left_arm_joint1', 'right_arm_joint1', 'left_arm_joint2', 'right_arm_joint2', 'left_arm_joint3', 'right_arm_joint3', 'left_arm_joint4', 'right_arm_joint4', 'left_arm_joint5', 'right_arm_joint5', 'left_arm_joint6', 'right_arm_joint6', 'left_arm_joint7', 'right_arm_joint7', 'left_gripper_finger_joint1', 'left_gripper_finger_joint2', 'right_gripper_finger_joint1', 'right_gripper_finger_joint2']
            # joint_limits = {{'torso_joint1': (-1.1344999074935913, 1.8325998783111572), 'torso_joint2': (-2.7924997806549072, 2.5306997299194336), 'torso_joint3': (-1.8325998783111572, 1.5707999467849731), 'torso_joint4': (-3.054299831390381, 3.054299831390381), 'left_arm_joint1': (-4.4506001472473145, 1.309000015258789), 'right_arm_joint1': (-4.4506001472473145, 1.309000015258789), 'left_arm_joint2': (-0.1745000034570694, 3.1415998935699463), 'right_arm_joint2': (-3.1415998935699463, 0.1745000034570694), 'left_arm_joint3': (-2.3561956882476807, 2.3561956882476807), 'right_arm_joint3': (-2.3561956882476807, 2.3561956882476807), 'left_arm_joint4': (-2.0943996906280518, 0.3490999639034271), 'right_arm_joint4': (-2.0943996906280518, 0.3490999639034271), 'left_arm_joint5': (-2.3561956882476807, 2.3561956882476807), 'right_arm_joint5': (-2.3561956882476807, 2.3561956882476807), 'left_arm_joint6': (-1.0471980571746826, 1.0471980571746826), 'right_arm_joint6': (-1.0471980571746826, 1.0471980571746826), 'left_arm_joint7': (-1.5707999467849731, 1.5707999467849731), 'right_arm_joint7': (-1.5707999467849731, 1.5707999467849731), 'left_gripper_finger_joint1': (0.0, 0.05000000074505806), 'left_gripper_finger_joint2': (0.0, 0.05000000074505806), 'right_gripper_finger_joint1': (0.0, 0.05000000074505806), 'right_gripper_finger_joint2': (0.0, 0.05000000074505806)}}
            # print()
            # print("limits", joint_limits)
            # print()
            # print(joint_positions)
            current_joint_positions = self.ik.get_current_joint_state(joint_positions, joint_names)
            actions_out = np.array([0, 0, 0] + current_joint_positions[:15] + [1] + current_joint_positions[15:] + [1])
            # actions_out = [np.zeros(23)]
            if self.barge_door == False:
                print("HI")
                quat_target = th.tensor([0,0,0,1])
                # quat_target = th.tensor([+0.0006, -0.0055, -0.0746, +0.9972])
                pos_target = th.tensor([0.1, 0.3,  1.0])
                pos_target_r = th.tensor([0.1, -0.3,  1.0])
                self.desired['left'] = (pos_target, quat_target)
                self.desired['right'] = (pos_target_r, quat_target)
                actions_out, q_targets = self.ik.solve_dual_arm_trajectory(
                    current_joint_positions=current_joint_positions,
                    left_goal=self.desired["left"],
                    right_goal=self.desired["right"],
                    # base_action=action,
                    # joint_names=joint_names,
                    # joint_limits=joint_limits,
                    duration=30.0,
                    num_targets=120,
                    debug = False
                )
            else:
                current_joint_positions_dict = {joint:current_joint_positions[i] for i, joint in enumerate(self.ik.ik.joint_names)}
                actions_out = self.ik.forward_barge_through_door('left', current_joint_positions_dict, current_joint_positions)
            # self._last_ik_action = actions_out[-1]

        except:
            import traceback as tb
            tb.print_exc()
        self._last_solve_t = now
        self._target_dirty = False
        for arm in self.robot.arm_names:
            self._last_desired[arm] = (
                self.desired[arm][0].clone(),
                self.desired[arm][1].clone()
            )
        
        return actions_out, None, None #self.ik.planner, q_targets


def focus_view_on_robot(env: og.Environment) -> None:
    if gm.HEADLESS:
        return
    robot = env.robots[0]
    pos, _ = robot.get_position_orientation()
    cam_pos = pos + th.tensor([2.5, -2.5, 1.8], dtype=pos.dtype, device=pos.device)
    cam_quat = th.tensor([0.415626, 0.215278, 0.304337, 0.828153])
    og.sim.viewer_camera.set_position_orientation(
        position=cam_pos, orientation=cam_quat
    )


def gently_tilt_head_down(
    robot, joint_name: str = "torso_joint4", offset_rad: float = -0.15
) -> None:
    """Apply a small absolute offset to the specified head/torso pitch joint to tilt the head down slightly.

    Uses immediate joint position set (drive=False) to avoid waiting for controller convergence.
    Silently no-ops if the joint name is not present or indices cannot be resolved.
    """
    try:
        names = list(robot.joints.keys())
        if joint_name not in names:
            return
        j_idx = names.index(joint_name)
        cur = robot.get_joint_positions()[j_idx]
        target = cur + th.tensor(offset_rad, dtype=cur.dtype, device=cur.device)
        robot.set_joint_positions(
            positions=target.unsqueeze(0), indices=th.tensor([j_idx]), drive=False
        )
    except Exception:
        # Best-effort; ignore if robot API differs
        pass


def fully_open_grippers(
    robot, margin_ratio: float = 1e-3, min_margin: float = 1e-4, verbose: bool = True
) -> None:
    """Drive all available gripper joints to their upper limits so fingers start fully open."""
    try:
        joint_positions = robot.get_joint_positions()
        joint_names = list(robot.joints.keys())
    except Exception:
        return
    device = joint_positions.device
    dtype = joint_positions.dtype
    any_opened = False
    for arm in getattr(robot, "arm_names", []):
        idx_tensor = getattr(robot, "gripper_control_idx", {}).get(arm)
        if idx_tensor is None:
            continue
        if not isinstance(idx_tensor, th.Tensor):
            idx_tensor = th.tensor(idx_tensor, dtype=th.int64, device=device)
        else:
            idx_tensor = idx_tensor.to(device=device, dtype=th.int64)
        if idx_tensor.numel() == 0:
            continue
        targets = []
        for j_idx in idx_tensor.tolist():
            joint_name = joint_names[j_idx] if j_idx < len(joint_names) else None
            target = float(joint_positions[j_idx])
            if joint_name in robot.joints:
                joint = robot.joints[joint_name]
                try:
                    upper = float(joint.upper_limit)
                    if np.isfinite(upper) and abs(upper) < 1e6:
                        margin = max(abs(upper) * margin_ratio, min_margin)
                        target = upper - margin if upper >= 0.0 else upper + margin
                except Exception:
                    pass
            targets.append(target)
        try:
            pos_tensor = th.tensor(targets, dtype=dtype, device=device)
            robot.set_joint_positions(
                positions=pos_tensor, indices=idx_tensor, drive=False
            )
            any_opened = True
        except Exception:
            continue
    if verbose and any_opened:
        print("[Teleop] Grippers opened to their joint limits.")


def main():
    parser = argparse.ArgumentParser(
        description="R1Pro whole-body IK teleop (base-frame EEF control)"
    )
    parser.add_argument("--task", type=str, help="BEHAVIOR task name")
    parser.add_argument("--instance-id", type=int, help="Public test instance id")
    parser.add_argument("--headless", action="store_true", help="Run without viewer")

    def _env_cast(key: str, cast, default):
        value = os.environ.get(key)
        if value is None:
            return default
        try:
            return cast(value)
        except Exception:
            return default

    parser.add_argument(
        "--sam6d-host",
        type=str,
        default=os.environ.get("SAM6D_HOST", "127.0.0.1"),
        help="SAM-6D server host (override with SAM6D_HOST).",
    )
    parser.add_argument(
        "--sam6d-port",
        type=int,
        default=_env_cast("SAM6D_PORT", int, 6000),
        help="SAM-6D server TCP port (override with SAM6D_PORT).",
    )
    parser.add_argument(
        "--sam6d-timeout",
        type=float,
        default=_env_cast("SAM6D_TIMEOUT", float, 10.0),
        help="Socket timeout in seconds when talking to SAM-6D.",
    )
    parser.add_argument(
        "--sam6d-target-label",
        type=str,
        default=os.environ.get("SAM6D_TARGET_LABEL"),
        help="Semantic label to mask for SAM-6D (override with SAM6D_TARGET_LABEL).",
    )
    parser.add_argument(
        "--sam6d-target-object",
        type=str,
        default=os.environ.get("SAM6D_TARGET_OBJECT"),
        help="BEHAVIOR object name to auto-derive the semantic label for SAM-6D.",
    )
    parser.add_argument(
        "--sam6d-debug-plot",
        action="store_true",
        help="Display a matplotlib window with the semantic mask sent to SAM-6D.",
    )
    parser.add_argument(
        "--sam6d-disable",
        action="store_true",
        help="Disable SAM-6D socket client even if host/port are specified.",
    )
    parser.add_argument(
        "--grasp-provider",
        type=str,
        choices=["sam6d", "contact"],
        default=os.environ.get("GRASP_PROVIDER", "sam6d").lower(),
        help="Select grasp backend: 'sam6d' (default) or 'contact' (Contact-GraspNet).",
    )
    parser.add_argument(
        "--no-grasp-viz",
        action="store_true",
        help="Disable debug visualization for SAM-6D / camera grasp targets.",
    )
    parser.add_argument(
        "--contact-host",
        type=str,
        default=os.environ.get("CONTACT_HOST", "127.0.0.1"),
        help="Contact-GraspNet server host (override with CONTACT_HOST).",
    )
    parser.add_argument(
        "--contact-port",
        type=int,
        default=_env_cast("CONTACT_PORT", int, 6500),
        help="Contact-GraspNet server TCP port (override with CONTACT_PORT).",
    )
    parser.add_argument(
        "--contact-timeout",
        type=float,
        default=_env_cast("CONTACT_TIMEOUT", float, 30.0),
        help="Socket timeout in seconds when talking to Contact-GraspNet.",
    )
    parser.add_argument(
        "--contact-disable",
        action="store_true",
        help="Disable Contact-GraspNet socket client even if host/port are specified.",
    )
    parser.add_argument(
        "--contact-max-grasps",
        type=int,
        default=_env_cast("CONTACT_MAX_GRASPS", int, 5),
        help="Maximum number of Contact-GraspNet candidates to queue automatically.",
    )
    args = parser.parse_args()

    gm.HEADLESS = args.headless
    gm.USE_GPU_DYNAMICS = False
    gm.ENABLE_FLATCACHE = True
    gm.ENABLE_TRANSITION_RULES = True

    root = find_repo_root()
    instances = load_test_instance_table(root / DATASET_RELATIVE_PATH)
    task = select_task(instances, args.task)
    idx = select_instance(instances[task], args.instance_id)

    # Infer scene_model from available_tasks.yaml if present
    # Fallback to a common house; user may edit if needed
    scene_model = "house_double_floor_lower"
    cfg = generate_basic_environment_config(task, scene_model)
    cfg["robots"] = [build_robot_config()]

    env = og.Environment(configs=cfg)
    robot = env.robots[0]

    grasp_provider = args.grasp_provider.lower()
    sam6d_client = None
    contact_client = None
    sam6d_host = (args.sam6d_host or "").strip()
    contact_host = (args.contact_host or "").strip()
    if grasp_provider == "sam6d":
        if args.sam6d_disable:
            print(
                "[Teleop] SAM-6D provider selected but SAM-6D client disabled; grasp capture will be unavailable."
            )
        elif sam6d_host:
            sam6d_client = GraspSolverSocketClient(
                host=sam6d_host,
                port=args.sam6d_port,
                timeout=args.sam6d_timeout,
            )
        else:
            print(
                "[Teleop] SAM-6D provider selected but no SAM-6D host specified; grasp capture will be unavailable."
            )
    elif grasp_provider == "contact":
        if args.contact_disable:
            print(
                "[Teleop] Contact-GraspNet provider selected but contact client disabled; grasp capture will be unavailable."
            )
        elif contact_host:
            contact_client = GraspSolverSocketClient(
                host=contact_host,
                port=args.contact_port,
                timeout=args.contact_timeout,
            )
        else:
            print(
                "[Teleop] Contact-GraspNet provider selected but no contact host specified; grasp capture will be unavailable."
            )
    sam6d_target_label = (
        args.sam6d_target_label.strip()
        if isinstance(args.sam6d_target_label, str)
        else None
    )
    sam6d_target_object = (
        args.sam6d_target_object.strip()
        if isinstance(args.sam6d_target_object, str)
        else None
    )
    sam6d_debug_plot = bool(args.sam6d_debug_plot)

    # Load instance state and reset
    apply_task_instance_state(env, idx)
    env.reset()
    focus_view_on_robot(env)
    # Pre-bias head: tilt down a bit using torso_joint4 if available
    gently_tilt_head_down(robot, joint_name="torso_joint4", offset_rad=-0.15)
    fully_open_grippers(robot)
    # Increase head camera resolution to 720x720 if supported
    for sensor in robot.sensors.values():
        if hasattr(sensor, "image_height") and hasattr(sensor, "image_width"):
            try:
                sensor.image_height = 720
                sensor.image_width = 720
            except Exception:
                pass

    # Register utility keys
    KeyboardEventHandler.add_keyboard_callback(
        lazy.carb.input.KeyboardInput.M,
        lambda: print("Sensor visualization is disabled to avoid Qt backend issues."),
    )
    KeyboardEventHandler.add_keyboard_callback(
        lazy.carb.input.KeyboardInput.ESCAPE,
        lambda: og.shutdown(),
    )

    teleop = EEFBaseTeleop(
        robot,
        env=env,
        sam6d_client=sam6d_client,
        contact_client=contact_client,
        contact_max_candidates=args.contact_max_grasps,
        sam6d_target_object=sam6d_target_object,
        sam6d_target_label=sam6d_target_label,
        sam6d_debug_plot=sam6d_debug_plot,
        grasp_viz=not args.no_grasp_viz,
        grasp_provider=grasp_provider,
    )

    def reload_instance():
        teleop.cancel_auto_drive()
        apply_task_instance_state(env, idx)
        robot.reset()
        teleop.sync_desired_to_current_pose()
        fully_open_grippers(robot, verbose=False)
        print("Reloaded instance, reset robot, and cancelled auto-drive.")

    KeyboardEventHandler.add_keyboard_callback(
        lazy.carb.input.KeyboardInput.R,
        reload_instance,
    )
    print("\nTeleop ready. Keys:")
    print("3/4: switch active arm (left/right)")
    print("Arrows: +/- X/Y (base frame), P/;: +/- Z  (EEF target)")
    print("N/B: +/- roll, O/U: +/- pitch, V/C: +/- yaw  (EEF target)")
    print("W/A/S/D: base translate +/-X, +/-Y   |   Q/E: base yaw +/-")
    print("PageUp/PageDown: head tilt up/down (torso_joint4)")
    print("J/L: head turn left/right (torso_joint3)")
    print("G: (disabled) use K with JSON object pose")
    print(
        "K: paste object R (3x3) and t (3,) in CAMERA frame (JSON) — gripper goes to object"
    )
    if grasp_provider == "sam6d" and teleop.sam6d_client is not None:
        print(
            "H: capture head RGB/depth/semantic, send to SAM-6D, auto-target when pose arrives"
        )
        if sam6d_target_object:
            print(f"   SAM-6D target object: {sam6d_target_object}")
        elif sam6d_target_label:
            print(f"   SAM-6D target label: {sam6d_target_label}")
    elif grasp_provider == "contact" and teleop.contact_client is not None:
        print(
            f"H: capture head RGB/depth/semantic, send to Contact-GraspNet, queue up to {args.contact_max_grasps} grasps"
        )
    print("SPACE: reset EEF target; R: reload + cancel auto-drive; ESC: quit")

    try:

        while True:
        # Given full joint state for r1prp in variable q: dict[str, float]
        # you can call the FK using following (similar for right arm)
        # q = q_targets[-1]
        # fk = planner.left_arm_chain.forward_kinematics({[q[n] for n in planner.left_arm_chain.get_joint_parameter_names()]})
        # tr = fk.translation().as_vector().toarray().flatten() # numpy.ndarray (3,)
        # ro = fk.rotation().as_quat().toarray().flatten() # numpy.ndarray (4,)
        




            # try:
                # print("[Test 1] Main loop iteration starting")
            action = teleop.step()

            # if type(action) == list:
            if type(action) == tuple:
                action, planner, q_targets = action
                act = None
                for i, act in enumerate(action):
                    # print("step", i, "action", np.rad2deg(act.detach().numpy()))
                    # for _ in range(15):
                    env.step(act)
                # for _ in range(500):
                #     env.step(act)

                # quat_target = th.tensor([0, 0, 0, 1])  # Identity = downward
                # # quat_target = th.tensor([+0.0006, -0.0055, -0.0746, +0.9972])
                # pos_target = th.tensor([0.0, 0.2,  0.50])
                # pos_target_r = th.tensor([0.8, -0.14,  0.7])
                # left_pos_actual, left_quat_actual = teleop._get_eef_pose(arm="left")
                # right_pos_actual, right_quat_actual = teleop._get_eef_pose(arm="right")
                # print(f"Left target: {pos_target}, actual: {left_pos_actual}")
                # print(f"Right target: {pos_target_r}, actual: {right_pos_actual}")
                # print(f"Left quat target: {quat_target}, actual: {left_quat_actual}")
                # print(f"Right quat target: {quat_target}, actual: {right_quat_actual}")
                # breakpoint()

                # Check if desired pose was reached after trajectory execution
                # for _ in range(100):
                #     env.step(act)
                # q = q_targets[-1]
                # breakpoint()
                # print()
                # print("LEFT ARM FORWARD KINEMATICS")
                # fk = planner.left_arm_chain.forward_kinematics({n: q[n] for n in planner.left_arm_chain.get_joint_parameter_names()})
                # tr = fk.translation().as_vector().toarray().flatten()  # numpy.ndarray (3,)
                # ro = fk.rotation().as_quat().toarray().flatten()  # numpy.ndarray (4,)
                # print(f"[FK Result] Translation: {tr}")
                # print(f"[FK Result] Rotation (quat): {ro}")
                # print()
                # print("RIGHT ARM FORWARD KINEMATICS")
                # fk = planner.right_arm_chain.forward_kinematics({n: q[n] for n in planner.right_arm_chain.get_joint_parameter_names()})
                # tr = fk.translation().as_vector().toarray().flatten()  # numpy.ndarray (3,)
                # ro = fk.rotation().as_quat().toarray().flatten()  # numpy.ndarray (4,)
                # print(f"[FK Result] Translation: {tr}")
                # print(f"[FK Result] Rotation (quat): {ro}")
                # print("\n[Main Loop] Trajectory execution complete, checking pose...")

                # Get current poses for checking
                # breakpoint()
                # current_poses = {}
                # for arm_name in teleop.robot.arm_names:
                #     breakpoint()
                #     ee_name = teleop.robot.gripper_link_names[arm_name][0]
                #     current_poses[arm_name] = get_pose_in_base(teleop.robot, ee_name)
                # breakpoint()

                # teleop.ik.check_pose_reached(current_poses, teleop.desired)

                # Check if joints reached their target positions
                # target_joints_dict = q_targets[-1]
                # # Convert dict to tensor in the same order as robot.get_joint_positions()
                # joint_names = list(teleop.robot.joints.keys())
                # current_joints = teleop.robot.get_joint_positions()
                # target_joints_tensor = current_joints.clone()
                # for joint_name, joint_value in target_joints_dict.items():
                #     joint_idx = joint_names.index(joint_name)
                #     target_joints_tensor[joint_idx] = float(joint_value)
                # teleop.ik.check_joints_reached(current_joints, target_joints_tensor, joint_names)
                # # teleop.ik.print_joint_limits(joint_names, joint_limits)
                # input()
            else:
                # breakpoint()
                env.step(action)
            # except Exception as e:
            #     import traceback as tb

            #     tb.print_exc()
            #     raise ValueError(f"IK solver exception: {e}")
    except:
        import traceback as tb
        tb.print_exc()
    finally:
        og.shutdown()


if __name__ == "__main__":
    main()
