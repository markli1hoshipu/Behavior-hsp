"""
Interactive Replay-to-Failure + Takeover Script

This script enables:
1. Interactive replay of an HDF5 trajectory while user watches simulation
2. User presses 't' to trigger takeover at any point (e.g., when they see failure)
3. Another controller (policy or joylo) takes over from that exact point
4. Records the complete trajectory (replay portion + takeover portion) to a new HDF5

Usage:
    # Policy takeover
    python interactive_takeover.py \
        --input_hdf5 /path/to/failed.hdf5 \
        --output_hdf5 /path/to/corrected.hdf5 \
        --task_name picking_up_trash \
        --episode_id 0 \
        --controller policy \
        --policy_config /path/to/policy.yaml

    # Joylo takeover (starts ZMQ server, user runs run_joylo.py separately)
    python interactive_takeover.py \
        --input_hdf5 /path/to/failed.hdf5 \
        --output_hdf5 /path/to/corrected.hdf5 \
        --task_name picking_up_trash \
        --episode_id 0 \
        --controller joylo \
        --robot_port 6001

Original file location: OmniGibson/omnigibson/learning/interactive_takeover.py
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional
import numpy as np

import h5py
import torch as th

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import omnigibson as og
from omnigibson.macros import gm

# Import the modified wrapper (from same directory for testing)
from data_wrapper_with_takeover import InteractiveReplayTakeoverWrapper


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive replay-to-failure + takeover")

    # Input/Output paths
    parser.add_argument("--input_hdf5", type=str, required=True,
                        help="Path to input HDF5 file to replay")
    parser.add_argument("--output_hdf5", type=str, required=True,
                        help="Path to output HDF5 file for combined trajectory")

    # Episode selection
    parser.add_argument("--episode_id", type=int, default=0,
                        help="Episode ID to replay from input HDF5")

    # Controller type
    parser.add_argument("--controller", type=str, choices=["policy", "joylo"], required=True,
                        help="Controller type for takeover")

    # Policy-specific args
    parser.add_argument("--policy_config", type=str, default=None,
                        help="Path to policy config YAML (required if controller=policy)")
    parser.add_argument("--policy_checkpoint", type=str, default=None,
                        help="Path to policy checkpoint (required if controller=policy)")

    # Joylo-specific args
    parser.add_argument("--robot_port", type=int, default=6001,
                        help="ZMQ port for robot server (used with controller=joylo)")
    parser.add_argument("--hostname", type=str, default="127.0.0.1",
                        help="Hostname for ZMQ server")

    # Rendering
    parser.add_argument("--n_render_iterations", type=int, default=3,
                        help="Number of render iterations per step")

    # Task info (optional, can be inferred from HDF5)
    parser.add_argument("--task_name", type=str, default=None,
                        help="Task name (optional, inferred from HDF5 if not provided)")

    return parser.parse_args()


def setup_environment_from_hdf5(input_hdf5_path: str):
    """
    Setup OmniGibson environment from HDF5 file config.

    Args:
        input_hdf5_path: Path to input HDF5 file

    Returns:
        og.Environment: Configured environment
    """
    # Read config from HDF5
    with h5py.File(input_hdf5_path, "r") as f:
        config = json.loads(f["data"].attrs["config"])
        scene_file = json.loads(f["data"].attrs["scene_file"])

    # Disable transition rules during replay/takeover
    gm.ENABLE_TRANSITION_RULES = False

    # Configure for interactive use
    config["env"]["flatten_obs_space"] = True

    # Set scene file
    config["scene"]["scene_file"] = scene_file

    # Disable online object sampling for BehaviorTask
    if config.get("task", {}).get("type") == "BehaviorTask":
        config["task"]["online_object_sampling"] = False
        config["task"]["use_presampled_robot_pose"] = False

    # Don't add additional objects (they're in scene file)
    config["objects"] = []

    # Create environment
    env = og.Environment(configs=config)

    return env


def run_policy_takeover(wrapper, policy, obs):
    """
    Run policy control after takeover.

    Args:
        wrapper: InteractiveReplayTakeoverWrapper instance
        policy: Policy object with forward(obs) method
        obs: Current observation dictionary
    """
    print("\n" + "=" * 60)
    print("POLICY TAKEOVER MODE")
    print("=" * 60)
    print("Policy is now controlling the robot...")
    print("Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    done = False
    step = 0

    try:
        while not done:
            # Get action from policy
            action = policy.forward(obs=obs)

            # Step environment (automatically records via DataCollectionWrapper)
            obs, reward, terminated, truncated, info = wrapper.step(action)
            done = terminated or truncated
            step += 1

            if step % 50 == 0:
                print(f"Policy step {step}...")

            # Check for success
            if terminated and wrapper.env.task.success:
                print(f"\n>>> TASK COMPLETED SUCCESSFULLY at step {step} <<<")

    except KeyboardInterrupt:
        print(f"\nPolicy control interrupted at step {step}")

    print(f"Policy took {step} steps after takeover")


def run_joylo_takeover(wrapper, robot_port: int, hostname: str):
    """
    Run joylo teleoperation after takeover.

    Starts a ZMQ server that the joylo client (run_joylo.py) can connect to.
    Uses pickle protocol to match the existing ZMQClientRobot implementation.

    Recording starts when user presses the 'A' button on the JoyCon controller.
    This allows the user to:
    1. Wait for hardware to sync to simulation state
    2. Adjust their grip on the joylo arms
    3. Press 'A' when ready to start recording

    Args:
        wrapper: InteractiveReplayTakeoverWrapper instance
        robot_port: ZMQ port for robot server
        hostname: Hostname for ZMQ server
    """
    import pickle
    import zmq

    # Get robot joint state for hardware sync
    robot_joints = wrapper.get_robot_joint_state_for_gello()

    print("\n" + "=" * 60)
    print("JOYLO TAKEOVER MODE")
    print("=" * 60)
    print(f"Starting ZMQ server on {hostname}:{robot_port}")
    print(f"\nRobot joint state for hardware sync:")
    print(f"  {robot_joints}")
    print("\nIn another terminal, run:")
    print(f"  python joylo/experiments/run_joylo.py --sync_to_sim --gello_model r1pro")
    print("\n" + "-" * 60)
    print("IMPORTANT: Recording will NOT start until you press the")
    print("'A' button on the RIGHT JoyCon controller.")
    print("-" * 60)
    print("\nPress Ctrl+C to stop and save data")
    print("=" * 60 + "\n")

    # Create a custom server that uses our wrapper's environment
    # This mimics the ZMQServerRobot protocol used by joylo
    class TakeoverRobotServer:
        """
        Custom robot server that wraps the takeover environment.
        Uses pickle protocol compatible with ZMQClientRobot.

        Protocol methods (from joylo/gello/zmq_core/robot_node.py):
        - num_dofs: returns int
        - get_joint_state: returns np.ndarray
        - command_joint_state: takes joint_state np.ndarray, returns None
        - get_observations: returns dict

        Recording is triggered by pressing the 'A' button on the JoyCon.
        The button state is transmitted as part of the joint_state array.
        """

        def __init__(self, wrapper, port, host):
            self.wrapper = wrapper
            self.env = wrapper.env
            self.robot = self.env.robots[0]
            self._joint_state = self.robot.get_joint_positions()

            # Recording state - starts disabled, enabled by A button press
            self._recording_enabled = False
            self._waiting_for_start_signal = True
            self._button_a_was_pressed = False  # For edge detection

            # Determine button_a index based on robot type
            # Joint state format for R1Pro (from og_sim.py):
            #   [7 left_arm, 7 right_arm, 3 base, 2 trunk, 1 left_gripper, 1 right_gripper,
            #    1 button_-, 1 button_+, 1 button_x, 1 button_y, 1 button_b, 1 button_a, ...]
            # R1Pro: 7+7+3+2+1+1+1+1+1+1+1 = 26 → button_a is at index 26
            # R1:    6+6+3+2+1+1+1+1+1+1+1 = 24 → button_a is at index 24
            from omnigibson.robots import R1Pro
            if isinstance(self.robot, R1Pro):
                self._button_a_index = 26
            else:
                self._button_a_index = 24

            print(f"Robot type: {type(self.robot).__name__}, button_a index: {self._button_a_index}")

            # ZMQ setup - REP socket for request-reply pattern
            self._context = zmq.Context()
            self._socket = self._context.socket(zmq.REP)
            addr = f"tcp://{host}:{port}"
            print(f"Takeover Robot Server binding to {addr}")
            self._socket.bind(addr)
            self._socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout

        def num_dofs(self) -> int:
            """Return number of DOFs."""
            return self.robot.n_joints

        def get_joint_state(self) -> np.ndarray:
            """Get current joint state."""
            return self._joint_state.cpu().numpy()

        def command_joint_state(self, joint_state: np.ndarray) -> None:
            """
            Command robot to joint state and step simulation.

            Recording only starts after the 'A' button is pressed on the JoyCon.
            Before that, we just render the simulation without recording actions.
            """
            # Check for A button press (edge detection: transition from not pressed to pressed)
            button_a_pressed = False
            if len(joint_state) > self._button_a_index:
                button_a_pressed = (joint_state[self._button_a_index] != 0)

            # Detect rising edge of A button to start recording
            if self._waiting_for_start_signal:
                if button_a_pressed and not self._button_a_was_pressed:
                    self._waiting_for_start_signal = False
                    self._recording_enabled = True
                    print("\n" + "=" * 50)
                    print(">>> 'A' BUTTON PRESSED - RECORDING STARTED <<<")
                    print("=" * 50 + "\n")

            self._button_a_was_pressed = button_a_pressed

            # If not recording yet, just render without recording
            if not self._recording_enabled:
                # Update joint state for get_joint_state/get_observations queries
                self._joint_state = self.robot.get_joint_positions()
                # Render to keep visualization alive
                og.sim.render()
                return None

            # Normal recording mode - step environment and record action
            action = th.tensor(joint_state, dtype=th.float32)
            obs, reward, terminated, truncated, info = self.wrapper.step(action)
            self._joint_state = self.robot.get_joint_positions()
            return None

        def get_observations(self) -> dict:
            """Get current observations."""
            obs = {
                "joint_positions": self._joint_state.cpu().numpy(),
                "joint_state": self._joint_state.cpu().numpy(),
            }
            return obs

        def serve(self):
            """Main server loop - handles ZMQ requests using pickle protocol."""
            step = 0  # Only counts recorded steps
            running = True

            print("\nWaiting for joylo client to connect...")
            print("Recording will start when you press 'A' on the RIGHT JoyCon.\n")

            try:
                while running:
                    try:
                        # Wait for request from client
                        message = self._socket.recv()
                        request = pickle.loads(message)

                        # Process request based on method
                        method = request.get("method")
                        args = request.get("args", {})

                        if method == "num_dofs":
                            result = self.num_dofs()
                        elif method == "get_joint_state":
                            result = self.get_joint_state()
                        elif method == "command_joint_state":
                            result = self.command_joint_state(**args)
                            # Only count steps when recording is enabled
                            if self._recording_enabled:
                                step += 1
                                if step % 50 == 0:
                                    print(f"Recording step {step}...")
                            elif step == 0:
                                # Print waiting message occasionally
                                pass
                        elif method == "get_observations":
                            result = self.get_observations()
                        else:
                            result = {"error": f"Invalid method: {method}"}
                            print(f"Unknown method: {method}")

                        # Send response using pickle
                        self._socket.send(pickle.dumps(result))

                    except zmq.Again:
                        # Timeout - just continue loop (allows checking for interrupts)
                        pass

            except KeyboardInterrupt:
                print(f"\nJoylo control interrupted at step {step}")

            finally:
                self._socket.close()
                self._context.term()

            print(f"Joylo recorded {step} steps after takeover")

    # Run server
    server = TakeoverRobotServer(wrapper, robot_port, hostname)
    server.serve()


def main():
    args = parse_args()

    # Validate args
    if args.controller == "policy":
        if args.policy_config is None or args.policy_checkpoint is None:
            print("ERROR: --policy_config and --policy_checkpoint required for policy controller")
            sys.exit(1)

    # Check input file exists
    if not os.path.exists(args.input_hdf5):
        print(f"ERROR: Input HDF5 file not found: {args.input_hdf5}")
        sys.exit(1)

    # Create output directory if needed
    output_dir = os.path.dirname(args.output_hdf5)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("INTERACTIVE TAKEOVER SYSTEM")
    print("=" * 60)
    print(f"Input HDF5:  {args.input_hdf5}")
    print(f"Output HDF5: {args.output_hdf5}")
    print(f"Episode ID:  {args.episode_id}")
    print(f"Controller:  {args.controller}")
    print("=" * 60 + "\n")

    # Setup environment
    print("Setting up environment from HDF5 config...")
    env = setup_environment_from_hdf5(args.input_hdf5)

    # Create wrapper
    print("Creating InteractiveReplayTakeoverWrapper...")
    wrapper = InteractiveReplayTakeoverWrapper(
        env=env,
        output_path=args.output_hdf5,
        input_path=args.input_hdf5,
        only_successes=False,  # Save all data regardless of success
        flush_every_n_traj=1,
    )

    # Interactive replay phase
    print("\nStarting interactive replay...")
    takeover_step = wrapper.interactive_replay(
        episode_id=args.episode_id,
        n_render_iterations=args.n_render_iterations
    )

    if takeover_step == -1:
        print("\nNo takeover triggered. Exiting without saving.")
        wrapper.close()
        return

    # Prepare takeover (copy pre-takeover data)
    print(f"\nPreparing takeover at step {takeover_step}...")
    wrapper.prepare_takeover(episode_id=args.episode_id, takeover_step=takeover_step)

    # Get current observation
    obs = wrapper.get_current_observation()

    # Controller takeover
    if args.controller == "policy":
        # Load policy
        print("\nLoading policy...")
        # TODO: Implement policy loading based on your policy framework
        # policy = load_policy(args.policy_config, args.policy_checkpoint)
        print("ERROR: Policy loading not yet implemented. Please implement load_policy().")
        print("For now, use --controller joylo for teleoperation takeover.")
        wrapper.close()
        return

    elif args.controller == "joylo":
        run_joylo_takeover(wrapper, args.robot_port, args.hostname)

    # Save data
    print("\nSaving trajectory data...")
    wrapper.close()

    print("\n" + "=" * 60)
    print("TAKEOVER COMPLETE")
    print("=" * 60)
    print(f"Combined trajectory saved to: {args.output_hdf5}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
