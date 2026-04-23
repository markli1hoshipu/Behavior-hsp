"""
Modified run_joylo.py with --sync_to_sim option for hardware sync before takeover.

This is a copy of the original joylo/experiments/run_joylo.py with additional
functionality to sync the physical joylo hardware to the simulation robot state
before starting teleoperation (for takeover scenarios).

Original file: joylo/experiments/run_joylo.py
"""

import glob
import time
import yaml
from dataclasses import dataclass
from typing import Optional, Tuple, Literal

import numpy as np
import tyro

# Note: These imports assume joylo is installed
# Adjust paths as needed for your setup
try:
    from gello.agents.agent import MultiControllerAgent
    from gello.agents.gello_agent import DynamixelRobotConfig, MotorFeedbackConfig
    from gello.agents.joycon_agent import JoyconAgent
    from gello.env import RobotEnv
    from gello.robots.robot import PrintRobot
    from gello.zmq_core.robot_node import ZMQClientRobot
    from gello import REPO_DIR
    from gello.agents.r1_gello_agent import R1GelloAgent
    from gello.agents.r1pro_gello_agent import R1ProGelloAgent
    JOYLO_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import joylo components: {e}")
    print("This script requires joylo to be installed.")
    JOYLO_AVAILABLE = False
    REPO_DIR = "."


def print_color(*args, color=None, attrs=(), **kwargs):
    try:
        import termcolor
        if len(args) > 0:
            args = tuple(termcolor.colored(arg, color=color, attrs=attrs) for arg in args)
    except ImportError:
        pass
    print(*args, **kwargs)


@dataclass
class Args:
    robot_port: int = 6001
    wrist_camera_port: int = 5000
    base_camera_port: int = 5001
    hostname: str = "127.0.0.1"
    robot_type: str = None  # only needed for quest agent or spacemouse agent
    hz: int = 100
    start_joints: Optional[Tuple[float, ...]] = None

    # Specify which robot to use: 'r1' or 'r1pro'
    gello_model: Literal["r1", "r1pro"] = "r1pro"

    # Joint config defaults will be set based on gello_model
    joint_config_file: Optional[str] = None

    gello_port: Optional[str] = None
    mock: bool = False
    damping_motor_kp: float = 0.3
    motor_feedback_type: str = "NONE"
    use_joycons: bool = True
    data_dir: str = "~/bc_data"
    verbose: bool = False

    # NEW: Sync to simulation state before starting teleoperation
    sync_to_sim: bool = False

    # NEW: Time to wait for motors to settle after sync (seconds)
    motor_settle_time: float = 5.0


def sync_hardware_to_simulation(arm_agent, robot_client, verbose: bool = False):
    """
    Sync the physical joylo hardware to match the simulation robot state.

    This is used in takeover scenarios where the simulation robot is at a specific
    pose (from replay) and we need the physical hardware to match before the user
    starts teleoperation.

    Args:
        arm_agent: R1GelloAgent or R1ProGelloAgent instance
        robot_client: ZMQClientRobot connected to simulation
        verbose: Whether to print detailed info

    Returns:
        bool: True if sync successful, False otherwise
    """
    print_color("\n" + "=" * 60, color="cyan", attrs=("bold",))
    print_color("HARDWARE SYNC MODE", color="cyan", attrs=("bold",))
    print_color("=" * 60, color="cyan", attrs=("bold",))

    try:
        # Query simulation for current robot joint state
        print_color("\nQuerying simulation for robot joint state...", color="yellow")
        obs = robot_client.get_obs()

        if "joint_positions" not in obs:
            print_color("ERROR: Simulation did not return joint_positions", color="red")
            return False

        sim_joint_state = np.array(obs["joint_positions"])
        if verbose:
            print_color(f"Simulation joint state: {sim_joint_state}", color="white")

        # Convert simulation joint state to gello joint space
        # The agent's _obs_joints_to_gello_joints method handles the conversion
        print_color("Converting to gello joint space...", color="yellow")

        # Create a mock observation dict for the conversion
        obs_dict = {"joint_positions": sim_joint_state}
        gello_target = arm_agent._obs_joints_to_gello_joints(obs_dict)

        if verbose:
            print_color(f"Gello target joints: {gello_target}", color="white")

        # Move hardware to target pose
        print_color("\nMoving hardware to match simulation state...", color="yellow")
        print_color("(This may take a few seconds)", color="white")

        # Use the agent's reset mechanism to move to target
        arm_agent.set_reset_qpos(gello_target)
        arm_agent.reset()

        # Verify the hardware reached the target
        time.sleep(0.5)  # Wait for motors to settle
        current_joints = arm_agent._robot.get_joint_state()

        # Calculate error
        error = np.abs(current_joints - gello_target)
        max_error = np.max(error)
        mean_error = np.mean(error)

        if verbose:
            print_color(f"Current hardware joints: {current_joints}", color="white")
            print_color(f"Max joint error: {np.rad2deg(max_error):.2f} degrees", color="white")
            print_color(f"Mean joint error: {np.rad2deg(mean_error):.2f} degrees", color="white")

        # Check if sync was successful (within 5 degrees)
        if max_error < np.deg2rad(5.0):
            print_color("\nHardware sync SUCCESSFUL!", color="green", attrs=("bold",))
            return True
        else:
            print_color(f"\nWARNING: Hardware sync may be inaccurate (max error: {np.rad2deg(max_error):.1f} deg)",
                       color="yellow", attrs=("bold",))
            return True  # Still return True to allow user to proceed

    except Exception as e:
        print_color(f"\nERROR during hardware sync: {str(e)}", color="red", attrs=("bold",))
        return False


def main(args):
    if not JOYLO_AVAILABLE:
        print("ERROR: Joylo components not available. Please install joylo.")
        return

    # Set default joint config file based on robot model if not explicitly provided
    if args.joint_config_file is None:
        if args.gello_model == "r1":
            args.joint_config_file = "joint_config_black_gello.yaml"
        elif args.gello_model == "r1pro":
            args.joint_config_file = "joint_config_panda_pro.yaml"
        else:
            raise ValueError(f"Unsupported gello model {args.gello_model}")

    if args.mock:
        robot_client = PrintRobot(8, dont_print=True)
        camera_clients = {}
    else:
        camera_clients = {
            # you can optionally add camera nodes here for imitation learning purposes
            # "wrist": ZMQClientCamera(port=args.wrist_camera_port, host=args.hostname),
            # "base": ZMQClientCamera(port=args.base_camera_port, host=args.hostname),
        }
        robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)

    active_arm = "right"
    env = RobotEnv(robot_client, control_rate_hz=args.hz, camera_dict=camera_clients, active_arm=active_arm)

    gello_port = args.gello_port
    if gello_port is None:
        import platform

        if platform.system().lower() == "linux":
            usb_ports = glob.glob("/dev/serial/by-id/*")
        elif platform.system().lower() == "darwin":
            usb_ports = glob.glob("/dev/cu.usbserial-*")
        else:
            raise ValueError(f"Unsupported platform {platform.system()}")
        print(f"Found {len(usb_ports)} ports")
        if len(usb_ports) > 0:
            gello_port = usb_ports[0]
            print(f"using port {gello_port}")
        else:
            raise ValueError(
                "No gello port found, please specify one or plug in gello"
            )

    # Read joint config from yaml
    # Update for the gello you are using
    with open(f"{REPO_DIR}/configs/{args.joint_config_file}", "r") as file:
        joint_config = yaml.load(file, Loader=yaml.SafeLoader)

    # Set appropriate number of joints based on robot model
    if args.gello_model == "r1":
        num_motors = 16
    elif args.gello_model == "r1pro":
        num_motors = 18
    else:
        raise ValueError(f"Unsupported gello model {args.gello_model}")

    dynamixel_config = DynamixelRobotConfig(
        joint_ids=tuple(np.arange(num_motors).tolist()),
        joint_offsets=[np.deg2rad(x) for x in joint_config['joints']['offsets']],
        joint_signs=joint_config['joints']['signs'],
        gripper_config=None,  # (8, 202, 152),
    )

    # Set appropriate default start_joints based on robot model
    start_joints = args.start_joints
    if start_joints is None:
        if args.gello_model == "r1":
            # Neutral pose for R1
            start_joints = np.array([
                np.pi/2, np.pi/2,
                np.pi, np.pi,
                -np.pi,
                0,
                0,
                0,
                -np.pi/2, -np.pi/2,
                np.pi, np.pi,
                -np.pi,
                0,
                0,
                0,
            ])
        elif args.gello_model == "r1pro":
            # TODO: determine this for R1Pro
            start_joints = np.zeros(num_motors)
        else:
            raise ValueError(f"Unsupported gello model {args.gello_model}")

    base_trunk_gripper_agent = None
    if args.use_joycons:
        base_trunk_gripper_agent = JoyconAgent(
            calibration_dir=f"{REPO_DIR}/configs",
            deadzone_threshold=0.2,
            max_translation=0.35,
            max_rotation=0.3,
            max_trunk_translate=0.1,
            max_trunk_tilt=0.05,
            enable_rumble=False,
        )

    # Create the appropriate agent based on the robot model
    if args.gello_model == "r1":
        arm_agent = R1GelloAgent(
            port=gello_port,
            dynamixel_config=dynamixel_config,
            start_joints=start_joints,
            default_joints=None,
            damping_motor_kp=args.damping_motor_kp,
            motor_feedback_type=MotorFeedbackConfig[args.motor_feedback_type],
            enable_locking_joints=True,
            joycon_agent=base_trunk_gripper_agent,
        )
    elif args.gello_model == "r1pro":
        arm_agent = R1ProGelloAgent(
            port=gello_port,
            dynamixel_config=dynamixel_config,
            start_joints=start_joints,
            default_joints=None,
            damping_motor_kp=args.damping_motor_kp,
            motor_feedback_type=MotorFeedbackConfig[args.motor_feedback_type],
            enable_locking_joints=True,
            joycon_agent=base_trunk_gripper_agent,
        )
    else:
        raise ValueError(f"Unsupported gello model {args.gello_model}")

    agent_parts = [arm_agent]
    if base_trunk_gripper_agent is not None:
        agent_parts.append(base_trunk_gripper_agent)

    # Create and start the agent
    agent = MultiControllerAgent(agents=agent_parts)
    agent.start()

    # NEW: If sync_to_sim is enabled, sync hardware before starting teleoperation
    if args.sync_to_sim:
        print_color("\nSync-to-sim mode enabled", color="cyan", attrs=("bold",))

        # Step 1: Perform hardware sync (moves motors to match simulation)
        sync_success = sync_hardware_to_simulation(
            arm_agent=arm_agent,
            robot_client=robot_client,
            verbose=args.verbose
        )

        if not sync_success:
            print_color("\nWARNING: Hardware sync may have failed!", color="red", attrs=("bold",))

        # Step 2: Wait for motors to physically settle
        print_color(f"\nWaiting {args.motor_settle_time} seconds for motors to settle...", color="yellow")
        time.sleep(args.motor_settle_time)

        # Step 3: Release motors so user can adjust their grip
        print_color("Releasing motors...", color="yellow")
        arm_agent.start()

        # Step 4: Notify user about the workflow
        print_color("\n" + "=" * 60, color="cyan", attrs=("bold",))
        print_color("Hardware synced and motors RELEASED.", color="green", attrs=("bold",))
        print_color("", color="yellow")
        print_color("You can now adjust your grip on the joylo arms.", color="yellow")
        print_color("", color="yellow")
        print_color(">>> Press 'A' button on RIGHT JoyCon to START RECORDING <<<", color="yellow", attrs=("bold",))
        print_color("", color="yellow")
        print_color("(Recording will NOT start until you press 'A')", color="white")
        print_color("=" * 60, color="cyan", attrs=("bold",))

    else:
        # Original behavior: go to canonical start position
        print("Going to start position")
        agent.reset()

    print_color("*" * 40, color="magenta", attrs=("bold",))
    print_color(f"\nWelcome to JoyLo ({args.gello_model.upper()})!\n", color="magenta", attrs=("bold",))
    print_color(f"{args.gello_model.upper()} Teleoperation Commands:\n", color="magenta", attrs=("bold",))
    print_color(f"\t ZL / ZR: Toggle grasping", color="magenta", attrs=("bold",))
    print_color(f"\t Left Joystick (not pressed): Translate the robot base", color="magenta", attrs=("bold",))
    print_color(f"\t Right Joystick: Rotate the robot base + tilt the trunk torso", color="magenta", attrs=("bold",))
    print_color(f"\t Up / Down Button: Raise / Lower the trunk torso", color="magenta", attrs=("bold",))

    # Lock behavior is slightly different between models
    if args.gello_model == "r1":
        print_color(f"\t L / R: Lock the lower two wrist joints while leaving the upper joints free", color="magenta", attrs=("bold",))
    elif args.gello_model == "r1pro":
        print_color(f"\t L / R: Lock the lower three wrist joints while leaving the upper joints free", color="magenta", attrs=("bold",))
    else:
        raise ValueError(f"Unsupported gello model {args.gello_model}")

    print_color(f"\t - / +: Lock the upper joints while leaving the lower wrist roll joint free.\n\t\tThe wrist pose will NOT be tracked while held.", color="magenta", attrs=("bold",))
    print_color(f"\t Y: Move the robot towards its reset pose", color="magenta", attrs=("bold",))
    print_color(f"\t B: Toggle camera", color="magenta", attrs=("bold",))
    print_color(f"\t Home: Reset the environment", color="magenta", attrs=("bold",))

    # Add A button instruction for sync mode
    if args.sync_to_sim:
        print_color(f"\t A: Start recording (REQUIRED in sync mode)", color="green", attrs=("bold",))

    obs = env.get_obs()

    if args.sync_to_sim:
        print_color("\nPress 'A' on RIGHT JoyCon to START RECORDING!", color="green", attrs=("bold",))
    else:
        print_color("\nStart!", color="green", attrs=("bold",))
    start_time = time.time()
    while True:
        num = time.time() - start_time
        message = f"\rTime passed: {round(num, 2)}          "
        print_color(
            message,
            color="white",
            attrs=("bold",),
            end="",
            flush=True,
        )
        action = agent.act(obs)
        obs = env.step(action)


if __name__ == "__main__":
    main(tyro.cli(Args))
