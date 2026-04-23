"""
Interactive HDF5 Replay + Takeover Script

Replays an HDF5 trajectory, allows user to trigger takeover at any point,
syncs Joylo hardware to the takeover state, and records a new combined trajectory.

Usage:
    Terminal 1 (Simulator):
        python joylo/experiments/launch_takeover.py \
            --input_hdf5 /path/to/episode_00010020.hdf5

        Output will be saved as episode_00010020_takeover.hdf5 (or specify --output_hdf5)

    Terminal 2 (Joylo - run AFTER simulator shows "Press X to start"):
        python joylo/experiments/run_joylo.py --gello_model r1pro --joint_config_file joint_config_hw.yaml

Workflow:
    1. Script loads HDF5 and replays trajectory visually
    2. Press 't' during replay to trigger takeover at current step
    3. Simulator enters waiting state, Joylo hardware syncs automatically when connected
    4. Press 'X' (keyboard or JoyCon) when ready to start recording
    5. Teleoperate to complete the task
    6. Press 'Home' to reset and save, or Ctrl+C to stop
"""

import json
import os
import sys
import select
import termios
import tty
import time
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict

import h5py
import numpy as np
import torch as th
import tyro
import zmq

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.envs import DataCollectionWrapper
from omnigibson.utils.python_utils import h5py_group_to_torch, create_object_from_init_info
from omnigibson.robots import R1, R1Pro

# Reuse utilities from og_sim without modifying them
from gello.robots.sim_robot.og_teleop_cfg import (
    DEFAULT_RESET_DELTA_SPEED, DEFAULT_TRUNK_TRANSLATE, N_COOLDOWN_SECS,
    RESOLUTION, SIMPLIFIED_TRUNK_CONTROL, VIEWING_MODE, ViewingMode,
    INCLUDE_BASE_CONTACT_OBS, INCLUDE_TRUNK_CONTACT_OBS,
    INCLUDE_ARM_CONTACT_OBS, INCLUDE_FINGER_CONTACT_OBS,
    GHOST_APPEAR_THRESHOLD, GHOST_APPEAR_TIME, GHOST_UPDATE_FREQ,
)
import gello.robots.sim_robot.og_teleop_utils as utils


def print_color(*args, color=None, attrs=(), **kwargs):
    """Print with optional color."""
    try:
        import termcolor
        if len(args) > 0:
            args = tuple(termcolor.colored(arg, color=color, attrs=attrs) for arg in args)
    except ImportError:
        pass
    print(*args, **kwargs)


@dataclass
class Args:
    input_hdf5: str
    """Path to input HDF5 file to replay (e.g. episode_00010020.hdf5)"""

    output_hdf5: Optional[str] = None
    """Path to output HDF5 file. If not specified, appends '_takeover' to input filename."""

    robot_port: int = 6001
    hostname: str = "127.0.0.1"


class TakeoverServer:
    """
    Standalone server for HDF5 replay and takeover.

    Reuses the ZMQ protocol and observation format from og_sim.py
    so that the existing run_joylo.py client works without modification.
    """

    def __init__(
        self,
        input_hdf5_path: str,
        output_hdf5_path: str,
        host: str = "127.0.0.1",
        port: int = 6001,
    ):
        self.input_hdf5_path = input_hdf5_path
        self.output_hdf5_path = output_hdf5_path
        self.host = host
        self.port = port

        # Open input HDF5
        self.input_hdf5 = h5py.File(input_hdf5_path, "r")

        # Load config and create environment
        self._setup_environment()

        # Wrap with DataCollectionWrapper for recording
        self.env = DataCollectionWrapper(
            env=self.env,
            output_path=output_hdf5_path,
            viewport_camera_path=og.sim.viewer_camera.active_camera_path,
            only_successes=False,
            flush_every_n_traj=1,
            keep_checkpoint_rollback_data=True,
        )

        # Copy metadata from input to output HDF5
        self._copy_hdf5_metadata()

        # State variables (matching og_sim.py protocol)
        self._waiting_to_resume = True
        self._in_cooldown = False
        self._resume_cooldown_time = None
        self._current_trunk_translate = DEFAULT_TRUNK_TRANSLATE
        self._current_trunk_tilt_offset = 0.0
        self._current_trunk_tilt = 0.0
        self._reset_max_arm_delta = DEFAULT_RESET_DELTA_SPEED * (np.pi / 180) * og.sim.get_sim_step_dt()

        # Joint state/command (matching og_sim.py format)
        self._joint_state = self.robot.get_joint_positions()
        self._joint_cmd = self._init_joint_cmd_from_current_state()

        # Gripper state
        self._grasp_action = {arm: 1 for arm in self.robot.arm_names}
        self._gripper_action_signal_detectors = {
            arm: utils.SignalChangeDetector(debounce_time=0.5)
            for arm in self.robot.arm_names
        }

        # Button toggle state
        self._button_toggled_state = {
            "x": False, "y": False, "a": False, "b": False,
            "left": False, "right": False,
        }

        # Cache joint limits
        qpos_min, qpos_max = self.robot.joint_lower_limits, self.robot.joint_upper_limits
        self._trunk_tilt_limits = {
            "lower": qpos_min[self.robot.trunk_control_idx][2],
            "upper": qpos_max[self.robot.trunk_control_idx][2]
        }
        self._arm_joint_limits = {}
        for arm in self.robot.arm_names:
            self._arm_joint_limits[arm] = {
                "lower": qpos_min[self.robot.arm_control_idx[arm]],
                "upper": qpos_max[self.robot.arm_control_idx[arm]],
            }

        # Shoulder directions
        self._arm_shoulder_directions = {"left": -1.0, "right": 1.0}

        # Takeover state
        self._takeover_step = -1
        self._cached_episode_data = None
        self._frame_counter = 0

        # Setup keyboard handler
        self._setup_keyboard_handler()

    def _setup_environment(self):
        """Create environment from HDF5 config."""
        # Load config from HDF5
        config = json.loads(self.input_hdf5["data"].attrs["config"])
        scene_file = json.loads(self.input_hdf5["data"].attrs["scene_file"])

        # Apply OmniGibson macros
        utils.apply_omnigibson_macros()

        # Configure environment
        config["env"]["flatten_obs_space"] = True
        config["scene"]["scene_file"] = scene_file

        # Disable sampling/reset behaviors for BehaviorTask since we restore from HDF5
        if config.get("task", {}).get("type") == "BehaviorTask":
            config["task"]["online_object_sampling"] = False
            config["task"]["use_presampled_robot_pose"] = False

        # Don't add objects (they're in scene file)
        config["objects"] = []

        # Create environment
        self.env = og.Environment(configs=config)
        self.robot = self.env.robots[0]

        # Setup ghost robot (red virtual links showing commanded arm positions)
        self.ghost = utils.setup_ghost_robot(self.env.scene)
        og.sim.step()  # Initialize ghost robot
        self._ghost_appear_counter = {arm: 0 for arm in self.robot.arm_names}
        self.ghost_info = utils.setup_ghost_robot_info(self.ghost, self.robot)

        # Setup cameras and viewport (matching OGRobotServer._setup_teleop_support)
        self.camera_paths, self.viewports = utils.setup_cameras(
            self.robot,
            self.env.external_sensors,
            RESOLUTION,
        )

        # Apply optimized sim settings (async rendering, rate limiting, etc.)
        utils.optimize_sim_settings(vr_mode=(VIEWING_MODE == ViewingMode.VR))

        # Set ghost link masses to be uniform to avoid orthonormal errors
        with og.sim.stopped():
            for link in self.ghost.links.values():
                link.mass = 0.1

        print_color(f"Environment created with robot: {type(self.robot).__name__}", color="cyan")

    def _copy_hdf5_metadata(self):
        """Copy metadata from input HDF5 to output HDF5."""
        data_grp = self.env.hdf5_file.require_group("data")
        data_grp.attrs["scene_file"] = self.input_hdf5["data"].attrs["scene_file"]
        data_grp.attrs["config"] = self.input_hdf5["data"].attrs["config"]

    def _init_joint_cmd_from_current_state(self):
        """Initialize joint command dictionary from current robot state."""
        joint_cmd = {
            f"{arm}_arm": self._joint_state[self.robot.arm_control_idx[arm]].clone()
            for arm in self.robot.arm_names
        }
        for arm in self.robot.arm_names:
            # Gripper cmd is a signal value (1=open trigger, -1=close trigger), not joint position
            joint_cmd[f"{arm}_gripper"] = th.ones(len(self.robot.gripper_action_idx[arm]))
        joint_cmd["base"] = self._joint_state[self.robot.base_control_idx].clone()
        # Trunk command is a delta (joystick input), starts at zero (no movement)
        joint_cmd["trunk"] = th.zeros(2)

        # Button commands
        for btn in ["button_-", "button_+", "button_x", "button_y", "button_b",
                    "button_a", "button_capture", "button_home", "button_left", "button_right"]:
            joint_cmd[btn] = th.zeros(1)

        return joint_cmd

    def _setup_keyboard_handler(self):
        """Setup keyboard event handler for X key (resume)."""
        import omnigibson.lazy as lazy

        def keyboard_event_handler(event, *args, **kwargs):
            if event.type == lazy.carb.input.KeyboardEventType.KEY_PRESS:
                if event.input == lazy.carb.input.KeyboardInput.X:
                    self._resume_control()
                elif event.input == lazy.carb.input.KeyboardInput.ESCAPE:
                    self._stop_requested = True
            return True

        appwindow = lazy.omni.appwindow.get_default_app_window()
        input_interface = lazy.carb.input.acquire_input_interface()
        keyboard = appwindow.get_keyboard()
        self._sub_keyboard = input_interface.subscribe_to_keyboard_events(keyboard, keyboard_event_handler)
        self._stop_requested = False

    def _resume_control(self):
        """Resume control after waiting (triggered by X key/button)."""
        if self._waiting_to_resume:
            self._waiting_to_resume = False
            self._resume_cooldown_time = time.time() + N_COOLDOWN_SECS
            self._in_cooldown = True
            print_color("\n>>> Control resumed, entering cooldown... <<<", color="green", attrs=("bold",))

    # ========== HDF5 Replay Methods ==========

    def _setup_terminal_for_keypress(self):
        """Setup terminal for non-blocking keypress detection."""
        self._old_terminal_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

    def _restore_terminal(self):
        """Restore terminal to original settings."""
        if hasattr(self, '_old_terminal_settings'):
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_terminal_settings)

    def _check_takeover_key(self) -> bool:
        """Non-blocking check for 't' key press."""
        if select.select([sys.stdin], [], [], 0.0)[0]:
            key = sys.stdin.read(1)
            return key.lower() == 't'
        return False

    def _load_episode_data(self):
        """Load episode data from input HDF5."""
        data_grp = self.input_hdf5["data"]
        episode_key = "demo_0"

        if episode_key not in data_grp:
            raise ValueError(f"demo_0 not found in {self.input_hdf5_path}")

        traj_grp = data_grp[episode_key]

        self._cached_episode_data = {
            "transitions": json.loads(traj_grp.attrs.get("transitions", "{}")),
            "state": th.from_numpy(traj_grp["state"][:]),
            "state_size": th.from_numpy(traj_grp["state_size"][:]),
            "action": th.from_numpy(traj_grp["action"][:]),
        }

        if "init_metadata" in traj_grp:
            self._cached_episode_data["init_metadata"] = h5py_group_to_torch(traj_grp["init_metadata"])
        else:
            self._cached_episode_data["init_metadata"] = {}

        print_color(f"Loaded demo_0: {len(self._cached_episode_data['action'])} steps",
                   color="cyan", attrs=("bold",))

    def _apply_transitions(self, cur_transitions):
        """Apply object/system transitions at current step."""
        scene = og.sim.scenes[0]

        for add_sys_name in cur_transitions.get("systems", {}).get("add", []):
            scene.get_system(add_sys_name, force_init=True)

        for remove_sys_name in cur_transitions.get("systems", {}).get("remove", []):
            scene.clear_system(remove_sys_name)

        for remove_obj_name in cur_transitions.get("objects", {}).get("remove", []):
            obj = scene.object_registry("name", remove_obj_name)
            if obj is not None:
                scene.remove_object(obj)

        for j, add_obj_info in enumerate(cur_transitions.get("objects", {}).get("add", [])):
            obj = create_object_from_init_info(add_obj_info)
            scene.add_object(obj)
            obj.set_position(th.ones(3) * 100.0 + th.ones(3) * 5 * j)

        if cur_transitions.get("objects", {}).get("add") or cur_transitions.get("systems", {}).get("add"):
            og.sim.step()

    def interactive_replay(self) -> int:
        """
        Replay episode interactively. Press 't' to trigger takeover.

        Returns:
            Step number where takeover was triggered, or -1 if completed.
        """
        self._load_episode_data()

        state = self._cached_episode_data["state"]
        state_size = self._cached_episode_data["state_size"]
        action = self._cached_episode_data["action"]
        transitions = self._cached_episode_data["transitions"]

        # Restore scene
        scene_file = json.loads(self.input_hdf5["data"].attrs["scene_file"])
        self.env.scene.restore(scene_file, update_initial_file=True)

        # Load initial state
        og.sim.load_state(state[0, :int(state_size[0])], serialized=True)
        og.sim.step()

        self._setup_terminal_for_keypress()

        print_color("\n" + "=" * 60, color="cyan", attrs=("bold",))
        print_color("INTERACTIVE REPLAY MODE", color="cyan", attrs=("bold",))
        print_color("=" * 60, color="cyan", attrs=("bold",))
        print_color("Press 't' to trigger TAKEOVER at any point", color="yellow", attrs=("bold",))
        print_color("=" * 60 + "\n", color="cyan", attrs=("bold",))

        takeover_step = -1

        try:
            for i in range(len(action)):
                if self._check_takeover_key():
                    print_color(f"\n>>> TAKEOVER TRIGGERED at step {i} <<<", color="green", attrs=("bold",))
                    takeover_step = i
                    break

                if str(i) in transitions:
                    self._apply_transitions(transitions[str(i)])

                if i + 1 < len(state):
                    og.sim.load_state(state[i + 1, :int(state_size[i + 1])], serialized=True)

                og.sim.render()

                if i % 50 == 0:
                    print_color(f"\rStep {i}/{len(action)}... (press 't' to takeover)",
                               color="white", end="", flush=True)

        finally:
            self._restore_terminal()

        if takeover_step == -1:
            print_color("\nReplay completed without takeover", color="yellow")

        return takeover_step

    def prepare_takeover(self, takeover_step: int):
        """Copy pre-takeover data to DataCollectionWrapper."""
        if self._cached_episode_data is None:
            raise RuntimeError("Must call interactive_replay() first")

        state = self._cached_episode_data["state"]
        state_size = self._cached_episode_data["state_size"]
        action = self._cached_episode_data["action"]
        transitions = self._cached_episode_data["transitions"]

        # Clear and copy trajectory data
        self.env.current_traj_history = []
        self.env.max_state_size = 0
        self.env.current_transitions = {}

        # Copy initial state
        self.env.current_traj_history.append({
            "state": state[0, :int(state_size[0])].clone(),
            "state_size": int(state_size[0]),
        })
        self.env.max_state_size = int(state_size[0])

        # Copy steps up to takeover
        for i in range(takeover_step):
            self.env.current_traj_history.append({
                "action": action[i].clone(),
                "state": state[i + 1, :int(state_size[i + 1])].clone(),
                "state_size": int(state_size[i + 1]),
                "reward": 0.0,
                "terminated": False,
                "truncated": False,
            })
            self.env.max_state_size = max(self.env.max_state_size, int(state_size[i + 1]))

            if str(i) in transitions:
                self.env.current_transitions[i] = transitions[str(i)]

        self._takeover_step = takeover_step

        # Reset ghost and frame counters
        self._ghost_appear_counter = {arm: 0 for arm in self.robot.arm_names}
        self._frame_counter = 0

        # Sync all state from the actual robot after replay
        self._joint_state = self.robot.get_joint_positions()
        self._joint_cmd = self._init_joint_cmd_from_current_state()

        # Infer trunk state from actual trunk joint positions
        # (matching og_sim.py rollback logic at line 552-556)
        trunk_qpos = self._joint_state[self.robot.trunk_control_idx]
        self._current_trunk_translate = utils.infer_trunk_translate_from_torso_qpos(trunk_qpos)
        base_trunk_pos = utils.infer_torso_qpos_from_trunk_translate(self._current_trunk_translate)
        self._current_trunk_tilt = 0.0  # Always 0.0, matching original
        self._current_trunk_tilt_offset = float(trunk_qpos[2] - base_trunk_pos[2])

        # Read gripper state from the last replayed action
        # action[gripper_action_idx] is -1 (closed) or 1 (open)
        last_action = action[takeover_step - 1] if takeover_step > 0 else action[0]
        for arm in self.robot.arm_names:
            gripper_val = last_action[self.robot.gripper_action_idx[arm]].item()
            self._grasp_action[arm] = -1 if gripper_val < 0 else 1

        print_color(f"Prepared: copied {takeover_step + 1} states, {takeover_step} actions",
                   color="green", attrs=("bold",))
        print_color(f"Trunk translate: {self._current_trunk_translate:.2f}, "
                   f"Gripper: {self._grasp_action}",
                   color="cyan")

    # ========== ZMQ Protocol (matching og_sim.py) ==========

    def _update_observations(self) -> Dict:
        """Update observations (matching og_sim.py format for Joylo client)."""
        from omnigibson.utils.usd_utils import GripperRigidContactAPI

        joint_pos = self.robot.get_joint_positions()
        joint_vel = self.robot.get_joint_velocities()
        finger_impulses = GripperRigidContactAPI.get_all_impulses(self.env.scene.idx) if INCLUDE_FINGER_CONTACT_OBS else None

        obs = {
            "active_arm": "right",
            "in_cooldown": self._in_cooldown,
            "reset_joints": bool(self._joint_cmd["button_y"][0].item()),
            "waiting_to_resume": self._waiting_to_resume,
            "base_contact": any(len(link.contact_list()) > 0 for link in self.robot.non_floor_touching_base_links) if INCLUDE_BASE_CONTACT_OBS else False,
            "trunk_contact": any(len(link.contact_list()) > 0 for link in self.robot.trunk_links) if INCLUDE_TRUNK_CONTACT_OBS else False,
        }

        for i, arm in enumerate(self.robot.arm_names):
            arm_idx = self.robot.arm_control_idx[arm]
            obs[f"arm_{arm}_control_idx"] = arm_idx
            obs[f"arm_{arm}_joint_positions"] = joint_pos[arm_idx].clone()
            obs[f"arm_{arm}_joint_positions"][0] -= self._current_trunk_tilt * self._arm_shoulder_directions[arm]
            obs[f"arm_{arm}_joint_velocities"] = joint_vel[arm_idx]
            obs[f"arm_{arm}_gripper_positions"] = joint_pos[self.robot.gripper_control_idx[arm]]
            obs[f"arm_{arm}_ee_pos_quat"] = th.concatenate(self.robot.eef_links[arm].get_position_orientation())
            obs[f"arm_{arm}_contact"] = any(len(link.contact_list()) > 0 for link in self.robot.arm_links[arm]) if INCLUDE_ARM_CONTACT_OBS else False
            obs[f"arm_{arm}_finger_max_contact"] = th.max(th.sum(th.square(finger_impulses[:, 2*i:2*(i+1), :]), dim=-1)).item() if INCLUDE_FINGER_CONTACT_OBS else 0.0
            obs[f"{arm}_gripper"] = self._joint_cmd[f"{arm}_gripper"].item()

        self._obs = obs
        return obs

    def command_joint_state(self, joint_state: th.Tensor):
        """Process joint state command from Joylo (matching og_sim.py format)."""
        state = joint_state.clone()

        if isinstance(self.robot, R1Pro):
            # R1Pro: 7+7+3+2+1+1 + buttons
            dims = [("left_arm", 7), ("right_arm", 7), ("base", 3), ("trunk", 2),
                    ("left_gripper", 1), ("right_gripper", 1),
                    ("button_-", 1), ("button_+", 1), ("button_x", 1), ("button_y", 1),
                    ("button_b", 1), ("button_a", 1), ("button_capture", 1), ("button_home", 1),
                    ("button_left", 1), ("button_right", 1)]
        else:  # R1
            dims = [("left_arm", 6), ("right_arm", 6), ("base", 3), ("trunk", 2),
                    ("left_gripper", 1), ("right_gripper", 1),
                    ("button_-", 1), ("button_+", 1), ("button_x", 1), ("button_y", 1),
                    ("button_b", 1), ("button_a", 1), ("button_capture", 1), ("button_home", 1),
                    ("button_left", 1), ("button_right", 1)]

        start_idx = 0
        for component, dim in dims:
            if start_idx >= len(state):
                break
            self._joint_cmd[component] = state[start_idx:start_idx + dim]
            start_idx += dim

    def _process_button_inputs(self):
        """Process button inputs (X to resume, Home to reset)."""
        # X button: resume control
        button_x = self._joint_cmd["button_x"].item() != 0.0
        if button_x and not self._button_toggled_state["x"]:
            if self._waiting_to_resume:
                self._resume_control()
            else:
                # Record checkpoint
                self.env.update_checkpoint()
                print_color("\nCheckpoint recorded", color="cyan")
        self._button_toggled_state["x"] = button_x

        # Home button: save and exit
        if self._joint_cmd["button_home"].item() != 0.0:
            if not self._in_cooldown:
                print_color("\nHome pressed - saving and exiting...", color="yellow")
                self._save_on_exit = True
                self._stop_requested = True

    def get_action(self) -> th.Tensor:
        """Generate robot action from joint commands (matching og_sim.py)."""
        action = th.zeros(self.robot.action_dim)

        # Arms
        for arm in ["left", "right"]:
            arm_act = self._joint_cmd[f"{arm}_arm"].clone()
            arm_act = arm_act.clip(self._arm_joint_limits[arm]["lower"],
                                    self._arm_joint_limits[arm]["upper"])

            if self._in_cooldown:
                robot_pos = self.robot.get_joint_positions()[self.robot.arm_control_idx[arm]]
                delta = arm_act - robot_pos
                arm_act = robot_pos + delta.clip(-self._reset_max_arm_delta, self._reset_max_arm_delta)

            arm_act[0] += self._current_trunk_tilt * self._arm_shoulder_directions[arm]
            action[self.robot.arm_action_idx[arm]] = arm_act

        # Base
        action[self.robot.base_action_idx] = self._joint_cmd["base"].clone()

        # Grippers
        for arm in self.robot.arm_names:
            gripper_signal = self._joint_cmd[f"{arm}_gripper"].item()
            if self._gripper_action_signal_detectors[arm].process_sample(gripper_signal):
                self._grasp_action[arm] = -self._grasp_action[arm]
            action[self.robot.gripper_action_idx[arm]] = self._grasp_action[arm]

        # Trunk
        if SIMPLIFIED_TRUNK_CONTROL:
            self._current_trunk_translate = float(th.clamp(
                th.tensor(self._current_trunk_translate) -
                self._joint_cmd["trunk"][0].item() * og.sim.get_sim_step_dt(),
                0.0, 2.0
            ))
            trunk_action = utils.infer_torso_qpos_from_trunk_translate(self._current_trunk_translate)

            self._current_trunk_tilt_offset = float(th.clamp(
                th.tensor(self._current_trunk_tilt_offset) +
                self._joint_cmd["trunk"][1].item() * og.sim.get_sim_step_dt(),
                self._trunk_tilt_limits["lower"] - trunk_action[2],
                self._trunk_tilt_limits["upper"] - trunk_action[2]
            ))
            trunk_action[2] += self._current_trunk_tilt_offset
            action[self.robot.trunk_action_idx] = trunk_action

        # Update ghost robot (red virtual links showing commanded arm positions)
        self._frame_counter += 1
        if self._frame_counter % GHOST_UPDATE_FREQ == 0:
            self._ghost_appear_counter = utils.update_ghost_robot(
                self.ghost,
                self.robot,
                action,
                self._ghost_appear_counter,
                self.ghost_info,
            )

        return action

    # ========== ZMQ Server ==========

    def _start_zmq_server(self):
        """Start ZMQ server for Joylo client communication."""
        self._zmq_context = zmq.Context()
        self._zmq_socket = self._zmq_context.socket(zmq.REP)
        addr = f"tcp://{self.host}:{self.port}"
        self._zmq_socket.bind(addr)
        self._zmq_socket.setsockopt(zmq.RCVTIMEO, 50)  # 50ms timeout
        print_color(f"ZMQ server listening on {addr}", color="cyan")

    def _handle_zmq_request(self):
        """Handle one ZMQ request from Joylo client."""
        try:
            message = self._zmq_socket.recv()
            request = pickle.loads(message)

            method = request.get("method")
            args = request.get("args", {})

            if method == "num_dofs":
                result = self.robot.n_joints
            elif method == "get_joint_state":
                result = self._joint_state
            elif method == "command_joint_state":
                self.command_joint_state(**args)
                result = None
            elif method == "get_observations":
                result = self._obs
            else:
                result = {"error": f"Unknown method: {method}"}

            self._zmq_socket.send(pickle.dumps(result))

        except zmq.Again:
            pass  # Timeout, no request

    def _stop_zmq_server(self):
        """Stop ZMQ server."""
        if hasattr(self, '_zmq_socket'):
            self._zmq_socket.close()
        if hasattr(self, '_zmq_context'):
            self._zmq_context.term()

    # ========== Main Serve Loop ==========

    def serve(self):
        """Main serving loop."""
        # Phase 1: Interactive replay
        print_color("\nStarting interactive replay...\n", color="cyan", attrs=("bold",))
        takeover_step = self.interactive_replay()

        if takeover_step == -1:
            print_color("No takeover triggered. Exiting.", color="yellow")
            self.stop()
            return

        # Phase 2: Prepare takeover
        self.prepare_takeover(takeover_step)

        # Phase 3: Start ZMQ server and wait for Joylo
        self._start_zmq_server()

        print_color("\n" + "=" * 60, color="green", attrs=("bold",))
        print_color("TAKEOVER READY", color="green", attrs=("bold",))
        print_color("=" * 60, color="green", attrs=("bold",))
        print_color("", color="yellow")
        print_color("Now run Joylo in another terminal:", color="yellow")
        print_color("  python joylo/experiments/run_joylo.py --gello_model r1pro \\", color="white")
        print_color("         --joint_config_file joint_config_hw.yaml", color="white")
        print_color("", color="yellow")
        print_color("Joylo will sync to simulation state automatically.", color="yellow")
        print_color("Wait for motors to settle, then press X to start recording.", color="yellow")
        print_color("Press Home to save and exit.", color="yellow")
        print_color("=" * 60 + "\n", color="green", attrs=("bold",))

        # Phase 4: Teleoperation loop
        self._save_on_exit = False
        try:
            while not self._stop_requested:
                # Update observations
                self._update_observations()

                # Handle ZMQ requests
                self._handle_zmq_request()

                # Process button inputs
                self._process_button_inputs()

                # Update cooldown
                if not self._waiting_to_resume and self._in_cooldown:
                    self._in_cooldown = time.time() < self._resume_cooldown_time

                # Step or render
                if self._waiting_to_resume:
                    og.sim.render()
                    print_color(f"\rWaiting... Press X to start recording.{' ' * 20}",
                               color="yellow", end="", flush=True)
                else:
                    action = self.get_action()
                    self.env.step(action)

                    status = "Cooldown..." if self._in_cooldown else "Recording..."
                    steps = len(self.env.current_traj_history)
                    print_color(f"\r{status} Steps: {steps} (Home to save){' ' * 10}",
                               color="green", end="", flush=True)

                # Update joint state for observations
                self._joint_state = self.robot.get_joint_positions()

        except KeyboardInterrupt:
            print_color("\n\nInterrupted. NOT saving (use Home to save).", color="yellow")

        self.stop()

    def stop(self):
        """Stop server. Only save data if Home was pressed."""
        self._stop_zmq_server()

        if getattr(self, '_save_on_exit', False):
            if len(self.env.current_traj_history) > 0:
                self.env.flush_current_traj()
            self.env.save_data()
            print_color(f"\nSaved to: {self.output_hdf5_path}", color="green", attrs=("bold",))
        else:
            print_color("\nExited without saving.", color="yellow")

        if hasattr(self, 'input_hdf5'):
            self.input_hdf5.close()

        og.shutdown()


def main(args: Args):
    if not os.path.exists(args.input_hdf5):
        print_color(f"ERROR: Input HDF5 not found: {args.input_hdf5}", color="red")
        sys.exit(1)

    # Auto-generate output path if not specified
    if args.output_hdf5 is None:
        input_path = Path(args.input_hdf5)
        args.output_hdf5 = str(input_path.parent / f"{input_path.stem}_takeover{input_path.suffix}")

    output_dir = os.path.dirname(args.output_hdf5)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print_color("\n" + "=" * 60, color="magenta", attrs=("bold",))
    print_color("JOYLO TAKEOVER SYSTEM", color="magenta", attrs=("bold",))
    print_color("=" * 60, color="magenta", attrs=("bold",))
    print_color(f"Input:    {args.input_hdf5}", color="white")
    print_color(f"Output:   {args.output_hdf5}", color="white")
    print_color("=" * 60 + "\n", color="magenta", attrs=("bold",))

    server = TakeoverServer(
        input_hdf5_path=args.input_hdf5,
        output_hdf5_path=args.output_hdf5,
        host=args.hostname,
        port=args.robot_port,
    )

    server.serve()


if __name__ == "__main__":
    main(tyro.cli(Args))
