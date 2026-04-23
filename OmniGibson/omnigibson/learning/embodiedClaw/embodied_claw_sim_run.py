"""
Agentic data collection sim runner for BEHAVIOR-1K.

This is a patched copy of eval.py that serves as the simulation runner for
the agentic data collection system (embodiedClaw).  It keeps the same
environment/robot/policy/metrics setup but replaces the evaluation loop
with an agentic-aware loop that integrates:

  - VLAPolicyController  (pause / resume / switch)
  - DataRecorder          (per-step video + tabular recording)
  - BDDLStateTracker      (predicate transition tracking)
  - CheckpointManager     (rollback / checkpoint persistence)
"""

import csv
import hydra
import json
import logging
import numpy as np
import omnigibson as og
import omnigibson.utils.transform_utils as T
import os
import sys
import torch as th
import traceback
from gello.robots.sim_robot.og_teleop_utils import (
    augment_rooms,
    load_available_tasks,
    generate_robot_config,
    get_task_relevant_room_types,
)
from gello.robots.sim_robot.og_teleop_cfg import DISABLED_TRANSITION_RULES
from hydra.utils import instantiate
from inspect import getsourcefile
from omegaconf import DictConfig, OmegaConf
from omnigibson.envs.env_wrapper import EnvironmentWrapper
from omnigibson.learning.utils.config_utils import register_omegaconf_resolvers
from omnigibson.learning.utils.eval_utils import (
    ROBOT_CAMERA_NAMES,
    PROPRIOCEPTION_INDICES,
    generate_basic_environment_config,
    flatten_obs_dict,
    TASK_NAMES_TO_INDICES,
)
from omnigibson.macros import gm, create_module_macros, macros
from omnigibson.metrics import MetricBase, AgentMetric, TaskMetric
from omnigibson.robots import BaseRobot
from omnigibson.utils.asset_utils import get_task_instance_path
from omnigibson.utils.python_utils import recursively_convert_to_torch
from pathlib import Path
from signal import signal, SIGINT
from typing import Any, Tuple, List

# Agentic system imports
from omnigibson.learning.embodiedClaw.tools.control_tools import VLAPolicyController
from omnigibson.learning.embodiedClaw.data_recording.data_saver import DataRecorder
from omnigibson.learning.embodiedClaw.data_recording.bddl_state_tracker import BDDLStateTracker
from omnigibson.learning.embodiedClaw.tools.data_tools import CheckpointManager

m = create_module_macros(module_path=__file__)
m.NUM_EVAL_EPISODES = 1
m.NUM_TRAIN_INSTANCES = 200
m.NUM_EVAL_INSTANCES = 20

# set global variables to boost performance
gm.ENABLE_FLATCACHE = True
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = True

# Set grasp window to larger value to account for hard grasps
with macros.unlocked():
    macros.robots.manipulation_robot.GRASP_WINDOW = 0.75


# create module logger
logger = logging.getLogger("agentic_evaluator")
logger.setLevel(20)  # info


class AgenticEvaluator:
    """
    Agentic evaluator for running VLA policies with data collection in BEHAVIOR-1K.

    Extends the base Evaluator pattern from eval.py with:
      - VLAPolicyController for pause/resume semantics
      - DataRecorder for per-step observation and action recording
      - BDDLStateTracker for predicate transition tracking
      - CheckpointManager for rollback/checkpoint persistence
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

        # record total number and success number of trials and trial time
        self.n_trials = 0
        self.n_success_trials = 0
        self.total_time = 0
        self.robot_action = dict()
        self._step_count = 0

        self.env = self.load_env(env_wrapper=self.cfg.env_wrapper)
        self.policy = self.load_policy()
        self.robot = self.load_robot()
        self.metrics = self.load_metrics()

        # Agentic system components
        self.controller = VLAPolicyController(self.policy)
        self.data_recorder = None  # set per episode
        self.bddl_tracker = BDDLStateTracker()
        self.checkpoint_mgr = None  # set per episode

        self.reset()
        # manually reset environment episode number
        self.env._current_episode = 0

    def load_env(self, env_wrapper: DictConfig) -> EnvironmentWrapper:
        """
        Read the environment config file and create the environment.
        The config file is located in the configs/envs directory.
        """
        # Disable a subset of transition rules for data collection
        for rule in DISABLED_TRANSITION_RULES:
            rule.ENABLED = False
        # Load config file
        available_tasks = load_available_tasks()
        task_name = self.cfg.task.name
        assert task_name in available_tasks, f"Got invalid task name: {task_name}"
        # Now, get human stats of the task
        task_idx = TASK_NAMES_TO_INDICES[task_name]
        self.human_stats = {
            "length": [],
            "distance_traveled": [],
            "left_eef_displacement": [],
            "right_eef_displacement": [],
        }
        with open(os.path.join(gm.DATA_PATH, "2025-challenge-task-instances", "metadata", "episodes.jsonl"), "r") as f:
            episodes = [json.loads(line) for line in f]
        for episode in episodes:
            if episode["episode_index"] // 1e4 == task_idx:
                for k in self.human_stats.keys():
                    self.human_stats[k].append(episode[k])
        # take a mean
        for k in self.human_stats.keys():
            self.human_stats[k] = sum(self.human_stats[k]) / len(self.human_stats[k])

        # Load the seed instance by default
        task_cfg = available_tasks[task_name][0]
        robot_type = self.cfg.robot.type
        assert robot_type == "R1Pro", f"Got invalid robot type: {robot_type}, only R1Pro is supported."
        cfg = generate_basic_environment_config(task_name=task_name, task_cfg=task_cfg)
        if self.cfg.partial_scene_load:
            relevant_rooms = get_task_relevant_room_types(activity_name=task_name)
            relevant_rooms = augment_rooms(relevant_rooms, task_cfg["scene_model"], task_name)
            cfg["scene"]["load_room_types"] = relevant_rooms

        cfg["robots"] = [
            generate_robot_config(
                task_name=task_name,
                task_cfg=task_cfg,
            )
        ]
        # Update observation modalities (includes depth_linear for agentic recording)
        cfg["robots"][0]["obs_modalities"] = ["proprio", "rgb", "depth_linear"]
        cfg["robots"][0]["proprio_obs"] = list(PROPRIOCEPTION_INDICES["R1Pro"].keys())
        if self.cfg.robot.controllers is not None:
            cfg["robots"][0]["controller_config"].update(self.cfg.robot.controllers)
        if self.cfg.max_steps is None:
            logger.info(
                f"Setting timeout to be 2x the average length of human demos: {int(self.human_stats['length'] * 2)}"
            )
            cfg["task"]["termination_config"]["max_steps"] = int(self.human_stats["length"] * 2)
        else:
            logger.info(f"Setting timeout to be {self.cfg.max_steps} steps through config.")
            cfg["task"]["termination_config"]["max_steps"] = self.cfg.max_steps
        cfg["task"]["include_obs"] = False
        env = og.Environment(configs=cfg)
        # instantiate env wrapper
        env = instantiate(env_wrapper, env=env)
        return env

    def load_robot(self) -> BaseRobot:
        """
        Loads and returns the robot instance from the environment.
        Returns:
            BaseRobot: The robot instance loaded from the environment.
        """
        robot = self.env.scene.object_registry("name", "robot_r1")
        return robot

    def load_policy(self) -> Any:
        """
        Loads and returns the policy instance.
        """
        policy = instantiate(self.cfg.model)
        logger.info("")
        logger.info("=" * 50)
        logger.info(f"Loaded policy: {self.cfg.policy_name}")
        logger.info("=" * 50)
        logger.info("")
        return policy

    def load_metrics(self) -> List[MetricBase]:
        """
        Load agent and task metrics.
        """
        return [AgentMetric(self.human_stats), TaskMetric(self.human_stats)]

    def step(self) -> Tuple[bool, bool, dict]:
        """
        Performs a single step of the task with agentic data collection support.

        Returns:
            Tuple[bool, bool, dict]:
                - terminated (bool): Whether the episode has terminated.
                - truncated (bool): Whether the episode was truncated.
                - info (dict): Additional info from the environment step.

        Workflow:
            1. Check pause (caches snapshot and blocks if paused).
            2. Get action from policy via controller.
            3. Step environment.
            4. Preprocess observations.
            5. Update step count.
            6. Record data if active.
            7. Track BDDL state.
            8. Termination tracking.
            9. Update metrics.
        """
        # 1. Check pause (caches snapshot and blocks if paused)
        self.controller.check_and_pause(self.env, self.robot, self.cfg.task.name, obs=self.obs)

        # 2. Get action from policy via controller
        self.robot_action = self.controller.forward(obs=self.obs)

        # 3. Step environment
        obs, _, terminated, truncated, info = self.env.step(self.robot_action, n_render_iterations=1)

        # 4. Preprocess obs
        self.obs = self._preprocess_obs(obs)

        # 4a. Cache latest obs for async observation (MCP reads without pausing)
        self.controller.update_latest_obs(self.obs)

        # 5. Update step count
        self._step_count += 1

        # 6. Record data if active
        if self.data_recorder and self.data_recorder.is_recording:
            self.data_recorder.record_step(
                action=self.robot_action,
                proprio=self.obs.get("robot_r1::robot_r1::proprio", None),
                cam_rel_poses=self.obs.get("robot_r1::cam_rel_poses", None),
                task_info={"step": self._step_count, "terminated": terminated, "truncated": truncated},
                obs=self.obs,
            )
            # Record serialized sim state for HDF5 output
            self.data_recorder.record_state(og.sim.dump_state(serialized=True))

        # 7. Track BDDL state
        self.bddl_tracker.step(self.env, self.robot, self._step_count)

        # 8. Termination tracking (same as eval.py)
        if terminated or truncated:
            self.n_trials += 1
            if info["done"]["success"]:
                self.n_success_trials += 1

        # 9. Update metrics
        for metric in self.metrics:
            metric.step_callback(self.env)

        return terminated, truncated, info

    def load_task_instance(self, instance_id: int, test_hidden: bool = False) -> None:
        """
        Loads the configuration for a specific task instance.

        Args:
            instance_id (int): The ID of the task instance to load.
            test_hidden (bool): [Internal use only] Whether to load the hidden test instance.
        """
        scene_model = self.env.task.scene_name
        tro_filename = self.env.task.get_cached_activity_scene_filename(
            scene_model=scene_model,
            activity_name=self.env.task.activity_name,
            activity_definition_id=self.env.task.activity_definition_id,
            activity_instance_id=instance_id,
        )
        if test_hidden:
            tro_file_path = os.path.join(
                gm.DATA_PATH,
                "2025-challenge-test-instances",
                self.env.task.activity_name,
                f"{tro_filename}-tro_state.json",
            )
        else:
            tro_file_path = os.path.join(
                get_task_instance_path(scene_model),
                f"json/{scene_model}_task_{self.env.task.activity_name}_instances/{tro_filename}-tro_state.json",
            )
        with open(tro_file_path, "r") as f:
            tro_state = recursively_convert_to_torch(json.load(f))
        for tro_key, tro_state in tro_state.items():
            if tro_key == "robot_poses":
                presampled_robot_poses = tro_state
                robot_pos = presampled_robot_poses[self.robot.model_name][0]["position"]
                robot_quat = presampled_robot_poses[self.robot.model_name][0]["orientation"]
                self.robot.set_position_orientation(robot_pos, robot_quat)
                # Write robot poses to scene metadata
                self.env.scene.write_task_metadata(key=tro_key, data=tro_state)
            else:
                self.env.task.object_scope[tro_key].load_state(tro_state, serialized=False)

        # Try to ensure that all task-relevant objects are stable
        # They should already be stable from the sampled instance, but there is some issue where loading the state
        # causes some jitter (maybe for small mass / thin objects?)
        for _ in range(25):
            og.sim.step_physics()
            for entity in self.env.task.object_scope.values():
                if not entity.is_system and entity.exists:
                    entity.keep_still()

        self.env.scene.update_initial_file()
        self.env.scene.reset()

    def _preprocess_obs(self, obs: dict) -> dict:
        """
        Preprocess the observation dictionary before passing it to the policy.
        Args:
            obs (dict): The observation dictionary to preprocess.

        Returns:
            dict: The preprocessed observation dictionary.
        """
        obs = flatten_obs_dict(obs)
        base_pose = self.robot.get_position_orientation()
        cam_rel_poses = []
        # The first time we query for camera parameters, it will return all zeros
        # For this case, we use camera.get_position_orientation() instead.
        # The reason we are not using camera.get_position_orientation() by defualt is because it will always return the most recent camera poses
        # However, since og render is somewhat "async", it takes >= 3 render calls per step to actually get the up-to-date camera renderings
        # Since we are using n_render_iterations=1 for speed concern, we need the correct corresponding camera poses instead of the most update-to-date one.
        # Thus, we use camera parameters which are guaranteed to be in sync with the visual observations.
        for camera_name in ROBOT_CAMERA_NAMES["R1Pro"].values():
            camera = self.robot.sensors[camera_name.split("::")[1]]
            direct_cam_pose = camera.camera_parameters["cameraViewTransform"]
            if np.allclose(direct_cam_pose, np.zeros(16)):
                cam_rel_poses.append(
                    th.cat(T.relative_pose_transform(*(camera.get_position_orientation()), *base_pose))
                )
            else:
                cam_pose = T.mat2pose(th.tensor(np.linalg.inv(np.reshape(direct_cam_pose, [4, 4]).T), dtype=th.float32))
                cam_rel_poses.append(th.cat(T.relative_pose_transform(*cam_pose, *base_pose)))
        obs["robot_r1::cam_rel_poses"] = th.cat(cam_rel_poses, axis=-1)
        # append task id to obs
        obs["task_id"] = th.tensor([TASK_NAMES_TO_INDICES[self.cfg.task.name]], dtype=th.int64)
        return obs

    def reset(self) -> None:
        """
        Reset the environment, policy, and compute metrics.
        """
        self.obs = self._preprocess_obs(self.env.reset()[0])
        # run metric start callbacks
        for metric in self.metrics:
            metric.start_callback(self.env)
        self.controller.reset()
        self.n_success_trials, self.n_trials = 0, 0
        self._step_count = 0

    def __enter__(self):
        signal(SIGINT, self._sigint_handler)
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        logger.info("")
        logger.info("=" * 50)
        logger.info(f"Total success trials: {self.n_success_trials}")
        logger.info(f"Total trials: {self.n_trials}")
        if self.n_trials > 0:
            logger.info(f"Success rate: {self.n_success_trials / self.n_trials}")
        logger.info("=" * 50)
        logger.info("")
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, exc_tb)
        # Close data recorder if active
        if self.data_recorder and self.data_recorder.is_recording:
            try:
                self.data_recorder.end_episode(
                    success=False,
                    env=self.env,
                    bddl_transitions=self.bddl_tracker.get_results(),
                )
            except Exception:
                logger.exception("Failed to close data recorder during exit.")
        self.env.close()
        og.shutdown()

    def _sigint_handler(self, signal_received, frame):
        logger.warning("SIGINT or CTRL-C detected.\n")
        self.__exit__(None, None, None)
        sys.exit(0)


if __name__ == "__main__":
    register_omegaconf_resolvers()
    # open yaml from task path — use the same configs directory as eval.py
    configs_dir = f"{Path(getsourcefile(lambda: 0)).parents[0].parent}/configs"
    with hydra.initialize_config_dir(configs_dir, version_base="1.1"):
        config = hydra.compose("base_config.yaml", overrides=sys.argv[1:])
    OmegaConf.resolve(config)
    # set headless mode
    gm.HEADLESS = config.headless

    assert not (
        config.eval_on_train_instances and config.test_hidden
    ), "Cannot eval on train instances and test hidden instances simultaneously."
    if config.test_hidden:
        logger.info("You are evaluating on hidden test instances! This is for internal use only.")

    # get run instances (same logic as eval.py)
    if config.eval_on_train_instances:
        logger.info(
            "You are evaluating on training instances, set eval_on_train_instances to False for test instances."
        )
        task_idx = TASK_NAMES_TO_INDICES[config.task.name]
        with open(os.path.join(gm.DATA_PATH, "2025-challenge-task-instances", "metadata", "episodes.jsonl"), "r") as f:
            episodes = [json.loads(line) for line in f]
        instances_to_run = []
        for episode in episodes:
            if episode["episode_index"] // 1e4 == task_idx:
                instances_to_run.append(str(int((episode["episode_index"] // 10) % 1e3)))
        if config.eval_instance_ids:
            assert set(config.eval_instance_ids).issubset(
                set(range(m.NUM_TRAIN_INSTANCES))
            ), f"eval instance ids must be in range({m.NUM_TRAIN_INSTANCES})"
            instances_to_run = [instances_to_run[i] for i in config.eval_instance_ids]
    elif config.test_hidden:
        instances_to_run = (
            config.eval_instance_ids if config.eval_instance_ids is not None else set(range(m.NUM_EVAL_INSTANCES))
        )
        assert set(instances_to_run).issubset(
            set(range(m.NUM_EVAL_INSTANCES))
        ), f"eval instance ids must be in range({m.NUM_EVAL_INSTANCES})"
    else:
        instances_to_run = (
            config.eval_instance_ids if config.eval_instance_ids is not None else set(range(m.NUM_EVAL_INSTANCES))
        )
        assert set(instances_to_run).issubset(
            set(range(m.NUM_EVAL_INSTANCES))
        ), f"eval instance ids must be in range({m.NUM_EVAL_INSTANCES})"
        # load csv file
        task_instance_csv_path = os.path.join(
            gm.DATA_PATH, "2025-challenge-task-instances", "metadata", "test_instances.csv"
        )
        with open(task_instance_csv_path, "r") as f:
            lines = list(csv.reader(f))[1:]
        assert (
            lines[TASK_NAMES_TO_INDICES[config.task.name]][1] == config.task.name
        ), f"Task name from config {config.task.name} does not match task name from csv {lines[TASK_NAMES_TO_INDICES[config.task.name]][1]}"
        test_instances = lines[TASK_NAMES_TO_INDICES[config.task.name]][2].strip().split(",")
        instances_to_run = [int(test_instances[i]) for i in instances_to_run]

    # establish metrics
    metrics = {}
    metrics_path = Path(config.log_path).expanduser() / "metrics"
    metrics_path.mkdir(parents=True, exist_ok=True)

    # establish data output path
    data_output_path = Path(config.log_path).expanduser() / "agentic_data"
    data_output_path.mkdir(parents=True, exist_ok=True)

    task_id = TASK_NAMES_TO_INDICES[config.task.name]
    camera_names = ROBOT_CAMERA_NAMES["R1Pro"]

    with AgenticEvaluator(config) as evaluator:
        logger.info("Starting agentic data collection...")

        for idx in instances_to_run:
            evaluator.reset()
            evaluator.load_task_instance(idx, test_hidden=config.test_hidden)
            logger.info(f"Starting task instance {idx} for agentic data collection...")

            for epi in range(m.NUM_EVAL_EPISODES):
                evaluator.reset()
                done = False

                # Create DataRecorder for this episode
                episode_output = str(data_output_path / f"{config.task.name}_{idx}_{epi}")
                evaluator.data_recorder = DataRecorder(
                    output_folder=episode_output,
                    task_name=config.task.name,
                    task_id=task_id,
                    demo_id=epi,
                    camera_names=camera_names,
                    record_rgb=True,
                    record_depth=True,
                )

                # Create CheckpointManager for this episode
                evaluator.checkpoint_mgr = CheckpointManager(
                    output_folder=episode_output,
                    task_name=config.task.name,
                    task_id=task_id,
                    demo_id=epi,
                    record_rgb=True,
                    record_depth=True,
                )
                evaluator.checkpoint_mgr.setup(evaluator.data_recorder, evaluator.bddl_tracker)

                # Start BDDL tracker
                evaluator.bddl_tracker.start(evaluator.env, evaluator.robot)

                # Start recording
                evaluator.data_recorder.start_episode()

                # Record initial sim state BEFORE first env.step (N+1 states for N actions)
                evaluator.data_recorder.record_state(og.sim.dump_state(serialized=True))

                # Capture initial checkpoint
                evaluator.checkpoint_mgr.capture_checkpoint(evaluator.env, 0)

                # Run metric start callbacks
                for metric in evaluator.metrics:
                    metric.start_callback(evaluator.env)

                # Main simulation loop
                while not done:
                    terminated, truncated, info = evaluator.step()
                    if terminated or truncated:
                        done = True
                    if evaluator._step_count % 1000 == 0:
                        logger.info(f"Current step: {evaluator._step_count}")

                # Determine success
                success = evaluator.n_success_trials > 0

                # End recording with BDDL transitions
                bddl_results = evaluator.bddl_tracker.get_results()
                evaluator.data_recorder.end_episode(
                    success=success,
                    env=evaluator.env,
                    bddl_transitions=bddl_results,
                )

                # Run metric end callbacks
                for metric in evaluator.metrics:
                    metric.end_callback(evaluator.env)

                # Log results
                logger.info(f"Episode finished at step {evaluator._step_count}.")
                logger.info(f"Episode exit state: terminated={terminated}, truncated={truncated}")
                logger.info(f"Total trials: {evaluator.n_trials}")
                logger.info(f"Total success trials: {evaluator.n_success_trials}")

                # Gather metric results and write to file
                for metric in evaluator.metrics:
                    metrics.update(metric.gather_results())
                with open(metrics_path / f"{config.task.name}_{idx}_{epi}.json", "w") as f:
                    json.dump(metrics, f)

                # Save BDDL results alongside metrics
                bddl_output_path = metrics_path / f"{config.task.name}_{idx}_{epi}_bddl.json"
                with open(bddl_output_path, "w") as f:
                    json.dump(bddl_results, f, indent=2)
                logger.info(f"Saved BDDL results to {bddl_output_path}")

                # Clean up episode-level references
                evaluator.data_recorder = None
                evaluator.checkpoint_mgr = None
