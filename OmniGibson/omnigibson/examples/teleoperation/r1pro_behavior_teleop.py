"""
Teleoperation entry point for the bi-arm R1Pro robot on BEHAVIOR-1K 2025 challenge test instances.

The script loads the requested BEHAVIOR task instance, initializes the environment with all scene
objects, and exposes keyboard teleoperation bindings via `KeyboardRobotController`.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch as th
import yaml

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.learning.utils.eval_utils import generate_basic_environment_config
from omnigibson.macros import gm
from omnigibson.utils.asset_utils import get_task_instance_path
from omnigibson.utils.python_utils import recursively_convert_to_torch
from omnigibson.utils.ui_utils import KeyboardRobotController, choose_from_options


DATASET_RELATIVE_PATH = Path("datasets/2025-challenge-task-instances/metadata/test_instances.csv")
AVAILABLE_TASKS_RELATIVE_PATH = Path("joylo/sampled_task/available_tasks.yaml")


def find_repo_root() -> Path:
    """
    Locate the repository root by searching upwards for the BEHAVIOR test instance manifest.
    """
    anchor = Path(__file__).resolve()
    for parent in anchor.parents:
        candidate = parent / DATASET_RELATIVE_PATH
        if candidate.exists():
            return parent
    raise FileNotFoundError(
        f"Could not locate {DATASET_RELATIVE_PATH!s} relative to {anchor}. "
        "Ensure the script is launched from within the BEHAVIOR-1K repository."
    )


def load_test_instance_table(csv_path: Path) -> OrderedDict[str, List[int]]:
    """
    Parse the public test instance manifest into a mapping of task name -> list of instance ids.
    """
    task_to_instances: OrderedDict[str, List[int]] = OrderedDict()
    with csv_path.open("r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            raw_ids = row["Public Test Instance IDs"].split(",")
            task_to_instances[row["Task"]] = [int(entry.strip()) for entry in raw_ids if entry.strip()]
    return task_to_instances


def load_available_task_configs(yaml_path: Path) -> Dict[str, Dict[int, dict]]:
    """
    Load the pre-sampled task metadata describing base scene / robot presets.
    """
    with yaml_path.open("r") as handle:
        data = yaml.safe_load(handle)
    return data if data is not None else {}


def choose_task(task_instances: OrderedDict[str, List[int]], requested: Optional[str]) -> str:
    """
    Resolve which BEHAVIOR activity to load.
    """
    if requested:
        requested = requested.strip()
        if requested not in task_instances:
            available = ", ".join(task_instances.keys())
            raise ValueError(f"Unknown task '{requested}'. Available tasks: {available}")
        return requested

    options = OrderedDict(
        (task, f"{len(instances)} test instances") for task, instances in task_instances.items()
    )
    return choose_from_options(options=options, name="task")


def choose_instance(instance_ids: List[int], requested: Optional[int]) -> int:
    """
    Resolve which public test instance ID to load for the selected task.
    """
    if requested is not None:
        if requested not in instance_ids:
            raise ValueError(
                f"Instance id {requested} not available for this task. "
                f"Valid ids: {', '.join(str(x) for x in instance_ids)}"
            )
        return requested

    # choose_from_options mutates order for lists by iterating, so we explicitly pass a copy.
    ordered_ids = list(instance_ids)
    return choose_from_options(options=ordered_ids, name="instance id")


def select_task_config(
    available_tasks: Dict[str, Dict[int, dict]],
    task_name: str,
    definition_id: Optional[int],
) -> dict:
    """
    Pick the scene/task configuration preset describing the scene model and default robot pose.
    """
    if task_name not in available_tasks:
        raise ValueError(f"No configuration available for task '{task_name}'.")
    task_configs = available_tasks[task_name]
    if definition_id is not None and definition_id in task_configs:
        return task_configs[definition_id]
    # Default to the first definition (keys are ints) if explicit id not provided.
    first_key = next(iter(task_configs))
    return task_configs[first_key]


def build_robot_controller_config() -> Dict[str, dict]:
    """
    Define controller names for R1Pro components suitable for keyboard teleoperation.
    """
    return {
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
        "gripper_left": {
            "name": "MultiFingerGripperController",
            "mode": "smooth",
        },
        "gripper_right": {
            "name": "MultiFingerGripperController",
            "mode": "smooth",
        },
    }


def build_robot_config() -> dict:
    """
    Assemble the robot configuration for the R1Pro.
    """
    return {
        "type": "R1Pro",
        "obs_modalities": ["rgb"],
        "action_type": "continuous",
        "action_normalize": True,
        "grasping_mode": "physical",
        "default_reset_mode": "tuck",
        "controller_config": build_robot_controller_config(),
    }


def configure_holonomic_base_shortcuts(controller: KeyboardRobotController) -> None:
    """
    Install intuitive WASDQE bindings for the holonomic base, so teleop does not rely on joint cycling.
    """
    base_info = controller.controller_info.get("base")
    if base_info is None or base_info["name"] != "HolonomicBaseJointController":
        return

    start_idx = base_info["start_idx"]
    controller.keypress_mapping[lazy.carb.input.KeyboardInput.W] = {"idx": start_idx + 0, "val": 0.25}
    controller.keypress_mapping[lazy.carb.input.KeyboardInput.S] = {"idx": start_idx + 0, "val": -0.25}
    controller.keypress_mapping[lazy.carb.input.KeyboardInput.A] = {"idx": start_idx + 1, "val": 0.25}
    controller.keypress_mapping[lazy.carb.input.KeyboardInput.D] = {"idx": start_idx + 1, "val": -0.25}
    controller.keypress_mapping[lazy.carb.input.KeyboardInput.Q] = {"idx": start_idx + 2, "val": 0.3}
    controller.keypress_mapping[lazy.carb.input.KeyboardInput.E] = {"idx": start_idx + 2, "val": -0.3}


def configure_trunk_shortcuts(
    controller: KeyboardRobotController,
) -> List[Tuple[lazy.carb.input.KeyboardInput, lazy.carb.input.KeyboardInput]]:
    """
    Bind direct joint commands for the four trunk joints. Returns the keys that were registered so we can
    document them for the user.
    """
    trunk_info = controller.controller_info.get("trunk")
    if trunk_info is None or trunk_info["name"] != "JointController":
        return []

    key_name_pairs = [
        ("KEY_7", "KEY_8"),
        ("KEY_9", "KEY_0"),
        ("MINUS", "EQUAL"),
        ("APOSTROPHE", "BACKSLASH"),
    ]

    candidate_pairs = []
    for inc_name, dec_name in key_name_pairs:
        inc_key = getattr(lazy.carb.input.KeyboardInput, inc_name, None)
        dec_key = getattr(lazy.carb.input.KeyboardInput, dec_name, None)
        if inc_key is not None and dec_key is not None:
            candidate_pairs.append((inc_key, dec_key))

    registered_pairs: List[Tuple[lazy.carb.input.KeyboardInput, lazy.carb.input.KeyboardInput]] = []
    used_keys = set(controller.keypress_mapping.keys())

    for inc_key, dec_key in candidate_pairs:
        if inc_key in used_keys or dec_key in used_keys:
            continue
        registered_pairs.append((inc_key, dec_key))
        used_keys.add(inc_key)
        used_keys.add(dec_key)
        if len(registered_pairs) >= trunk_info["command_dim"]:
            break

    if len(registered_pairs) < trunk_info["command_dim"]:
        return []

    for joint_offset, (inc_key, dec_key) in enumerate(registered_pairs):
        idx = trunk_info["start_idx"] + joint_offset
        controller.keypress_mapping[inc_key] = {"idx": idx, "val": 0.05}
        controller.keypress_mapping[dec_key] = {"idx": idx, "val": -0.05}

    return registered_pairs


def configure_arm_joint_shortcuts(controller: KeyboardRobotController, step: float = 0.05) -> None:
    """
    Bind direct joint commands for arm_left (7 DOF) and arm_right (7 DOF) in joint position delta mode.
    Left arm gets 7 explicit pairs; right arm gets 5 pairs (one DOF left for default joint-selection if needed).
    """
    def register_pairs(info_key: str, pairs: list, step_size: float):
        info = controller.controller_info.get(info_key)
        if not info or info.get("name") != "JointController":
            return []
        used = set(controller.keypress_mapping.keys())
        registered = []
        for i, (inc_name, dec_name) in enumerate(pairs):
            if i >= info["command_dim"]:
                break
            inc_key = getattr(lazy.carb.input.KeyboardInput, inc_name, None)
            dec_key = getattr(lazy.carb.input.KeyboardInput, dec_name, None)
            if inc_key is None or dec_key is None:
                continue
            if inc_key in used or dec_key in used:
                continue
            idx = info["start_idx"] + len(registered)
            controller.keypress_mapping[inc_key] = {"idx": idx, "val": +step_size}
            controller.keypress_mapping[dec_key] = {"idx": idx, "val": -step_size}
            used.add(inc_key); used.add(dec_key)
            registered.append((inc_key, dec_key))
        return registered

    # Left arm: map 6 joints; remaining joint accessible via default joint-select ([/], 1/2)
    left_pairs = [
        ("Y", "H"),
        ("U", "J"),
        ("I", "K"),
        ("O", "L"),
        ("P", "SEMICOLON"),
        ("COMMA", "PERIOD"),
    ]
    register_pairs("arm_left", left_pairs, step)

    # Right arm: map 4-5 joints; remaining joint(s) via default [ / ] + 1/2 selection
    right_pairs = [
        ("KEY_3", "KEY_4"),
        ("Z", "X"),
        ("C", "V"),
        ("B", "N"),
    ]
    register_pairs("arm_right", right_pairs, step)

def apply_task_instance_state(env: og.Environment, instance_id: int) -> None:
    """
    Load the serialized TRO (task-relevant object) state for the requested instance, update the
    environment, and persist it as the new reset baseline.
    """
    robot = env.robots[0]
    scene_model = env.task.scene_name
    template_name = env.task.get_cached_activity_scene_filename(
        scene_model=scene_model,
        activity_name=env.task.activity_name,
        activity_definition_id=env.task.activity_definition_id,
        activity_instance_id=instance_id,
    )
    task_root = get_task_instance_path(scene_model)
    if task_root is None:
        raise FileNotFoundError(f"No task data found for scene '{scene_model}'.")

    tro_dir = Path(task_root) / "json" / f"{scene_model}_task_{env.task.activity_name}_instances"
    tro_path = tro_dir / f"{template_name}-tro_state.json"
    if not tro_path.exists():
        raise FileNotFoundError(f"TRO state '{tro_path}' missing for instance {instance_id}.")

    with tro_path.open("r") as handle:
        tro_state = recursively_convert_to_torch(json.load(handle))

    env.task.activity_instance_id = instance_id
    for scope_name, state in tro_state.items():
        if scope_name == "robot_poses":
            robot_candidates = state.get(robot.model_name)
            if not robot_candidates:
                raise ValueError(f"No presampled pose for robot '{robot.model_name}' in {tro_path.name}.")
            pose = robot_candidates[0]
            robot.set_position_orientation(position=pose["position"], orientation=pose["orientation"])
            env.scene.write_task_metadata(key=scope_name, data=state)
        else:
            env.task.object_scope[scope_name].load_state(state, serialized=False)

    # Allow objects to settle before capturing the state as the new reset baseline.
    for _ in range(25):
        og.sim.step_physics()
        for entity in env.task.object_scope.values():
            if not entity.is_system and entity.exists:
                entity.keep_still()

    env.scene.update_initial_file()


def create_environment(
    task_name: str,
    task_cfg: dict,
    headless: bool,
) -> og.Environment:
    """
    Instantiate the OmniGibson environment configured for the requested task.
    """
    gm.HEADLESS = headless
    gm.USE_GPU_DYNAMICS = False
    gm.ENABLE_FLATCACHE = True
    gm.ENABLE_TRANSITION_RULES = True

    cfg = generate_basic_environment_config(task_name=task_name, task_cfg=task_cfg)
    cfg["scene"]["load_room_types"] = None
    cfg["scene"]["load_room_instances"] = None
    cfg["scene"]["include_robots"] = False
    cfg["scene"]["not_load_object_categories"] = None
    cfg["robots"] = [build_robot_config()]
    cfg["task"]["activity_definition_id"] = task_cfg.get("activity_definition_id", 0)
    cfg["task"]["activity_instance_id"] = 0
    cfg["task"]["online_object_sampling"] = False
    cfg["task"]["highlight_task_relevant_objects"] = False
    cfg["task"]["include_obs"] = False

    env = og.Environment(configs=cfg)
    return env


def focus_view_on_robot(env: og.Environment) -> None:
    """
    Place the viewer camera close to the robot for convenience.
    """
    if gm.HEADLESS:
        return

    robot = env.robots[0]
    position, _ = robot.get_position_orientation()
    camera_position = position + th.tensor(
        [2.5, -2.5, 1.8], dtype=position.dtype, device=position.device
    )

    # Use a fixed quaternion that looks roughly towards the origin; users can fine-tune manually.
    default_orientation = th.tensor([0.415626, 0.215278, 0.304337, 0.828153])
    og.sim.viewer_camera.set_position_orientation(position=camera_position, orientation=default_orientation)


def main():
    parser = argparse.ArgumentParser(description="Teleoperate the R1Pro robot on BEHAVIOR-1K test instances.")
    parser.add_argument("--task", type=str, help="Name of the BEHAVIOR activity to load.")
    parser.add_argument(
        "--instance-id",
        type=int,
        help="Public test instance id (as listed in test_instances.csv). If omitted, prompts interactively.",
    )
    parser.add_argument(
        "--definition-id",
        type=int,
        default=0,
        help="Activity definition id to use from the available_tasks.yaml presets.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Launch without the OmniGibson viewer.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Optional step cap for the teleop session (-1 runs indefinitely).",
    )
    args = parser.parse_args()

    repo_root = find_repo_root()
    test_instance_csv = repo_root / DATASET_RELATIVE_PATH
    available_tasks_yaml = repo_root / AVAILABLE_TASKS_RELATIVE_PATH

    task_instances = load_test_instance_table(test_instance_csv)
    available_tasks = load_available_task_configs(available_tasks_yaml)

    task_name = choose_task(task_instances, args.task)
    instance_id = choose_instance(task_instances[task_name], args.instance_id)
    task_cfg = select_task_config(available_tasks, task_name, args.definition_id)

    env = create_environment(task_name=task_name, task_cfg=task_cfg, headless=args.headless)

    robot = env.robots[0]
    robot.reload_controllers(controller_config=build_robot_controller_config())
    env.scene.update_initial_file()

    instance_ids = task_instances[task_name]
    current_index = instance_ids.index(instance_id)

    def load_instance(target_index: int) -> None:
        nonlocal current_index, instance_id
        if not instance_ids:
            raise RuntimeError(f"No instances registered for task '{task_name}'.")
        current_index = target_index % len(instance_ids)
        instance_id = instance_ids[current_index]
        print(
            f"\nLoading task '{task_name}' instance {instance_id} "
            f"({current_index + 1}/{len(instance_ids)})...\n"
        )
        apply_task_instance_state(env=env, instance_id=instance_id)
        env.reset()
        focus_view_on_robot(env)

    load_instance(current_index)

    def reload_current_instance():
        load_instance(current_index)
        robot.reset()

    teleop_controller = KeyboardRobotController(robot=robot)

    teleop_controller.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.R,
        description="Reload current instance and reset robot joints",
        callback_fn=reload_current_instance,
    )
    if len(instance_ids) > 1:
        teleop_controller.register_custom_keymapping(
            key=lazy.carb.input.KeyboardInput.PAGE_DOWN,
            description="Load previous instance",
            callback_fn=lambda: load_instance(current_index - 1),
        )
        teleop_controller.register_custom_keymapping(
            key=lazy.carb.input.KeyboardInput.PAGE_UP,
            description="Load next instance",
            callback_fn=lambda: load_instance(current_index + 1),
        )
    teleop_controller.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.M,
        description="(disabled) sensor visualization",
        callback_fn=lambda: print(
            "Sensor visualization is disabled in this teleop script to avoid Qt backend issues."
        ),
    )
    configure_holonomic_base_shortcuts(teleop_controller)
    trunk_bindings = configure_trunk_shortcuts(teleop_controller)
    configure_arm_joint_shortcuts(teleop_controller)
    print("Base Control")
    print("W / S\tmove forward / backward")
    print("A / D\tstrafe left / right")
    print("Q / E\tyaw counter-clockwise / clockwise")
    if trunk_bindings:
        print("Torso Control")

        def format_key(key: lazy.carb.input.KeyboardInput) -> str:
            raw = str(key).split(".")[-1]
            key_alias = {
                "KEY_7": "7",
                "KEY_8": "8",
                "KEY_9": "9",
                "KEY_0": "0",
                "MINUS": "-",
                "EQUAL": "=",
                "APOSTROPHE": "'",
                "BACKSLASH": "\\",
            }
            return key_alias.get(raw, raw)

        for joint_idx, (inc_key, dec_key) in enumerate(trunk_bindings):
            print(f"{format_key(inc_key)} / {format_key(dec_key)}\ttrunk joint {joint_idx} +/-")
    print("Left Arm Joint Control")
    print("Y/H  U/J  I/K  O/L  P/;  ,/.   (remaining via [ / ] + 1/2)")
    print("Right Arm Joint Control")
    print("3/4  Z/X  C/V  B/N   (remaining via [ / ] + 1/2)")
    print("Note: M key is overridden to avoid launching the Matplotlib-based sensor viewer.")
    print(
        f"\nReady for teleoperation on '{task_name}' instance {instance_id}. "
        "Press PageUp/PageDown to cycle instances, R to reload, and ESC to exit.\n"
    )

    try:
        step = 0
        max_steps = args.max_steps
        while max_steps < 0 or step < max_steps:
            action = teleop_controller.get_teleop_action()
            env.step(action)
            step += 1
    finally:
        og.shutdown()


if __name__ == "__main__":
    main()

