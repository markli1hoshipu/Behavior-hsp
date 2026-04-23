"""
Modified data_wrapper.py with InteractiveReplayTakeoverWrapper class.
This is a copy of the original data_wrapper.py with a new class added at the end.

Original file: OmniGibson/omnigibson/envs/data_wrapper.py
"""

import json
import os
import sys
import select
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import logging
import numpy as np

import h5py
import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.envs.env_wrapper import EnvironmentWrapper, create_wrapper
from omnigibson.macros import gm, macros
from omnigibson.objects.object_base import BaseObject
from omnigibson.sensors.vision_sensor import VisionSensor
from omnigibson.systems.macro_particle_system import MacroPhysicalParticleSystem
from omnigibson.utils.config_utils import TorchEncoder
from omnigibson.utils.data_utils import merge_scene_files
from omnigibson.utils.python_utils import create_object_from_init_info, h5py_group_to_torch
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.tasks.behavior_task import BehaviorTask
from omnigibson.controllers.controller_base import ControlType

# Import the original classes
from omnigibson.envs.data_wrapper import DataWrapper, DataCollectionWrapper, DataPlaybackWrapper

# Create module logger
log = create_module_logger(module_name=__name__)
log.setLevel(logging.INFO)


class InteractiveReplayTakeoverWrapper(DataCollectionWrapper):
    """
    Wrapper that:
    1. Replays HDF5 interactively (user watches simulation)
    2. Listens for keyboard interrupt to trigger takeover
    3. Copies pre-takeover data to new trajectory
    4. Continues recording post-takeover data

    This class extends DataCollectionWrapper to enable seamless transition from
    replay mode to live control mode while maintaining a continuous trajectory.
    """

    def __init__(
        self,
        env,
        output_path,
        input_path,
        viewport_camera_path="/World/viewer_camera",
        overwrite=True,
        only_successes=False,  # Default to False to save all data
        flush_every_n_traj=1,
        use_vr=False,
        obj_attr_keys=None,
        keep_checkpoint_rollback_data=False,
        enable_dump_filters=False,  # Disable by default for takeover (need observations)
    ):
        """
        Args:
            env (Environment): The environment to wrap
            output_path (str): path to store output hdf5 data file
            input_path (str): path to input hdf5 file to replay from
            viewport_camera_path (str): prim path to the camera for rendering
            overwrite (bool): If set, will overwrite any pre-existing data at output_path
            only_successes (bool): Whether to only save successful episodes (default False for takeover)
            flush_every_n_traj (int): How often to flush data to file
            use_vr (bool): Whether to use VR headset
            obj_attr_keys (None or list of str): Object attributes to cache
            keep_checkpoint_rollback_data (bool): Whether to record rollback data
            enable_dump_filters (bool): Whether to enable dump filters (disabled for takeover)
        """
        # Open input HDF5 file
        self.input_hdf5 = h5py.File(input_path, "r")
        self.input_path = input_path

        # Takeover state
        self.takeover_step = None
        self.in_takeover_mode = False
        self._cached_episode_data = None

        # Call parent init
        super().__init__(
            env=env,
            output_path=output_path,
            viewport_camera_path=viewport_camera_path,
            overwrite=overwrite,
            only_successes=only_successes,
            flush_every_n_traj=flush_every_n_traj,
            use_vr=use_vr,
            obj_attr_keys=obj_attr_keys,
            keep_checkpoint_rollback_data=keep_checkpoint_rollback_data,
            enable_dump_filters=enable_dump_filters,
        )

    def _setup_terminal_for_keypress(self):
        """Setup terminal for non-blocking keypress detection."""
        import termios
        import tty

        # Save original terminal settings
        self._old_settings = termios.tcgetattr(sys.stdin)
        # Set terminal to raw mode for character-by-character input
        tty.setcbreak(sys.stdin.fileno())

    def _restore_terminal(self):
        """Restore terminal to original settings."""
        import termios

        if hasattr(self, '_old_settings'):
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)

    def _check_takeover_key(self) -> bool:
        """
        Non-blocking check for takeover key press.

        Returns:
            bool: True if 't' key was pressed, False otherwise
        """
        # Check if input is available (non-blocking)
        if select.select([sys.stdin], [], [], 0.0)[0]:
            key = sys.stdin.read(1)
            return key.lower() == 't'
        return False

    def _apply_transitions(self, cur_transitions):
        """
        Apply object/system transitions at current step.

        Args:
            cur_transitions (dict): Dictionary containing systems and objects to add/remove
        """
        scene = og.sim.scenes[0]

        # Add systems
        for add_sys_name in cur_transitions["systems"]["add"]:
            scene.get_system(add_sys_name, force_init=True)

        # Remove systems
        for remove_sys_name in cur_transitions["systems"]["remove"]:
            scene.clear_system(remove_sys_name)

        # Remove objects
        for remove_obj_name in cur_transitions["objects"]["remove"]:
            obj = scene.object_registry("name", remove_obj_name)
            scene.remove_object(obj)

        # Add objects
        for j, add_obj_info in enumerate(cur_transitions["objects"]["add"]):
            obj = create_object_from_init_info(add_obj_info)
            scene.add_object(obj)
            # Temporarily move far away to avoid collisions
            obj.set_position(th.ones(3) * 100.0 + th.ones(3) * 5 * j)

        # Step physics to initialize new objects
        if cur_transitions["objects"]["add"] or cur_transitions["systems"]["add"]:
            og.sim.step()

    def interactive_replay(self, episode_id: int, n_render_iterations: int = 3) -> int:
        """
        Replay episode interactively. User presses 't' to trigger takeover.

        Args:
            episode_id (int): Episode ID to replay from input HDF5
            n_render_iterations (int): Number of render iterations per step

        Returns:
            int: Step number where takeover was triggered, or -1 if replay completed
        """
        data_grp = self.input_hdf5["data"]
        assert f"demo_{episode_id}" in data_grp, f"No valid episode with ID {episode_id} found!"
        traj_grp = data_grp[f"demo_{episode_id}"]

        # Load episode data
        try:
            transitions = json.loads(traj_grp.attrs["transitions"])
            traj_data = h5py_group_to_torch(traj_grp)
            init_metadata = traj_data.get("init_metadata", {})
            action = traj_data["action"]
            state = traj_data["state"]
            state_size = traj_data["state_size"]
        except KeyError as e:
            log.error(f"Error loading episode {episode_id}: {str(e)}")
            return -1

        # Cache episode data for prepare_takeover
        self._cached_episode_data = {
            "transitions": transitions,
            "init_metadata": init_metadata,
            "action": action,
            "state": state,
            "state_size": state_size,
        }

        # Restore scene from input HDF5
        scene_file = json.loads(self.input_hdf5["data"].attrs["scene_file"])
        self.scene.restore(scene_file, update_initial_file=True)

        # Update the output HDF5's scene_file and config to match input
        # This ensures the output HDF5 can be replayed correctly
        data_grp = self.hdf5_file.require_group("data")
        if "scene_file" in data_grp.attrs:
            del data_grp.attrs["scene_file"]
        if "config" in data_grp.attrs:
            del data_grp.attrs["config"]
        # Copy from input HDF5
        data_grp.attrs["scene_file"] = self.input_hdf5["data"].attrs["scene_file"]
        data_grp.attrs["config"] = self.input_hdf5["data"].attrs["config"]

        # Restore init_metadata (object attributes like scale, visibility)
        if init_metadata:
            with og.sim.stopped():
                for i, obj in enumerate(self.scene.objects):
                    for attr, vals in init_metadata.items():
                        if i < len(vals):
                            val = vals[i]
                            setattr(obj, attr, val.item() if val.ndim == 0 else val)

        # Load initial state
        og.sim.load_state(state[0, :int(state_size[0])], serialized=True)

        # Step once to render initial state
        self.env.step(action[0], n_render_iterations=n_render_iterations)

        # Setup terminal for keypress detection
        self._setup_terminal_for_keypress()

        print("\n" + "=" * 60)
        print("INTERACTIVE REPLAY MODE")
        print("=" * 60)
        print("Press 't' to trigger TAKEOVER at any point")
        print("Press 'q' to quit without takeover")
        print("=" * 60 + "\n")

        takeover_triggered = False
        takeover_step = -1

        try:
            for i in range(len(action)):
                # Check for keyboard input
                if self._check_takeover_key():
                    print(f"\n>>> TAKEOVER TRIGGERED at step {i} <<<")
                    takeover_step = i
                    takeover_triggered = True
                    break

                # Apply transitions at this step
                if str(i) in transitions:
                    self._apply_transitions(transitions[str(i)])
                    # Record transition for output HDF5
                    self.current_transitions[i] = transitions[str(i)]

                # Load state at step i+1 (state after action[i])
                if i + 1 < len(state):
                    og.sim.load_state(state[i + 1, :int(state_size[i + 1])], serialized=True)
                    self.env.step(action[i], n_render_iterations=n_render_iterations)

                # Progress indicator
                if i % 50 == 0:
                    print(f"\rStep {i}/{len(action)}... (press 't' to takeover)", end="", flush=True)

        finally:
            # Restore terminal settings
            self._restore_terminal()

        if not takeover_triggered:
            print("\nReplay completed without takeover")
            return -1

        return takeover_step

    def prepare_takeover(self, episode_id: int, takeover_step: int):
        """
        Prepare for takeover by copying replayed data to current trajectory.

        After this method, the wrapper is ready for live control - any subsequent
        calls to step() will record new data that continues from the takeover point.

        Args:
            episode_id (int): Episode ID that was being replayed
            takeover_step (int): Step number where takeover was triggered
        """
        if self._cached_episode_data is None:
            raise RuntimeError("Must call interactive_replay() before prepare_takeover()")

        state = self._cached_episode_data["state"]
        state_size = self._cached_episode_data["state_size"]
        action = self._cached_episode_data["action"]
        transitions = self._cached_episode_data["transitions"]
        init_metadata = self._cached_episode_data["init_metadata"]

        # Clear any existing trajectory data
        self.current_traj_history = []
        self.max_state_size = 0
        self.current_transitions = {}

        # Copy init_metadata from the input HDF5
        # This is needed for postprocess_traj_group() to save object attributes
        if init_metadata:
            self.init_metadata = {k: v.clone() if isinstance(v, th.Tensor) else v
                                  for k, v in init_metadata.items()}
        else:
            self.init_metadata = {}

        # Copy initial state (step 0, no action)
        # Clone the tensor to avoid modifying the cached data
        initial_state = state[0, :int(state_size[0])].clone()
        self.current_traj_history.append({
            "state": initial_state,
            "state_size": int(state_size[0]),
        })
        self.max_state_size = int(state_size[0])

        # Copy steps 1 to takeover_step (with actions)
        for i in range(takeover_step):
            step_state = state[i + 1, :int(state_size[i + 1])].clone()
            self.current_traj_history.append({
                "action": action[i].clone() if isinstance(action[i], th.Tensor) else action[i],
                "state": step_state,
                "state_size": int(state_size[i + 1]),
                "reward": 0.0,
                "terminated": False,
                "truncated": False,
            })
            self.max_state_size = max(self.max_state_size, int(state_size[i + 1]))

            # Copy transitions (deep copy to avoid modifying cached data)
            if str(i) in transitions:
                self.current_transitions[i] = deepcopy(transitions[str(i)])

        # Update step count
        self.step_count = takeover_step

        # Set takeover state
        self.takeover_step = takeover_step
        self.in_takeover_mode = True

        log.info(f"Prepared takeover: copied {takeover_step + 1} states, {takeover_step} actions")
        log.info(f"Trajectory history length: {len(self.current_traj_history)}")
        log.info(f"Max state size: {self.max_state_size}")

    def get_robot_joint_state_for_gello(self) -> np.ndarray:
        """
        Get robot joint positions for syncing joylo hardware.

        Returns:
            np.ndarray: Robot joint positions in radians
        """
        robot = self.env.robots[0]
        joint_positions = robot.get_joint_positions()
        return joint_positions.cpu().numpy()

    def get_current_observation(self) -> dict:
        """
        Get current observation from the environment.

        Returns:
            dict: Current observation dictionary
        """
        return self.env._get_obs()

    def close(self):
        """Close the wrapper and release resources."""
        if hasattr(self, 'input_hdf5') and self.input_hdf5 is not None:
            self.input_hdf5.close()
        super().save_data()

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, '_old_settings'):
            self._restore_terminal()
