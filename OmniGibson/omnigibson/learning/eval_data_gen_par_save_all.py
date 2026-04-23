"""
eval_data_gen_par.py - Parallel/Direct Data Generation

This script combines eval_data_gen.py and replay_obs.py to directly output raw data
(parquet files and videos) during policy evaluation, bypassing the intermediate HDF5 step.

This is much faster than the two-step process of:
1. eval_data_gen.py -> HDF5
2. replay_obs.py -> parquet + videos
"""

import csv
import cv2
import hydra
import json
import logging
import numpy as np
import omnigibson as og
import omnigibson.utils.transform_utils as T
import os
import pandas as pd
import shutil
import sys
import torch as th
import traceback
from av.container import Container
from av.stream import Stream
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
import h5py
from omnigibson.envs.env_wrapper import EnvironmentWrapper
from omnigibson.envs.data_wrapper import HDF5CollectionWrapper
from omnigibson.learning.utils.config_utils import register_omegaconf_resolvers
from omnigibson.learning.utils.dataset_utils import makedirs_with_mode
from omnigibson.learning.utils.eval_utils import (
    ROBOT_CAMERA_NAMES,
    PROPRIOCEPTION_INDICES,
    generate_basic_environment_config,
    flatten_obs_dict,
    TASK_NAMES_TO_INDICES,
    HEAD_RESOLUTION,
    WRIST_RESOLUTION,
)
from omnigibson.learning.utils.obs_utils import (
    create_video_writer,
    write_video,
)
from omnigibson.macros import gm, create_module_macros, macros
from omnigibson.metrics import MetricBase, AgentMetric, TaskMetric
from omnigibson.sensors.vision_sensor import VisionSensor
from omnigibson.robots import BaseRobot
from omnigibson.object_states import Inside, OnTop, NextTo, Touching, Under, Open, ToggledOn, Cooked, Frozen
from omnigibson.utils.asset_utils import get_task_instance_path
from omnigibson.utils.python_utils import recursively_convert_to_torch
from pathlib import Path
from signal import signal, SIGINT
from typing import Any, Tuple, List, Dict, Optional

class PolicyEvalDataCollectionWrapper(HDF5CollectionWrapper):
    """
    A modified HDF5CollectionWrapper that does NOT disable camera render products.
    This allows the policy to receive correct observations during evaluation
    while still recording state dumps for replay compatibility.
    """

    def _optimize_sim_for_data_collection(self, viewport_camera_path):
        """
        Override to NOT disable camera render products.
        We skip the sensor disabling so observations remain valid for policy evaluation.
        """
        og.sim.viewer_camera.active_camera_path = viewport_camera_path
        if self._enable_dump_filters:
            self.enable_dump_filters()


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
logger = logging.getLogger("evaluator_par")
logger.setLevel(20)  # info


class BDDLStateTracker:
    """
    Tracks BDDL predicate state transitions during an episode.

    Records the exact timestep of every state change for goal conditions,
    grasping, pairwise spatial predicates, and unary object states.
    """

    BINARY_STATE_CLASSES = {
        "Inside": Inside,
        "OnTop": OnTop,
        "NextTo": NextTo,
        "Touching": Touching,
        "Under": Under,
    }

    UNARY_STATE_CLASSES = {
        "Open": Open,
        "ToggledOn": ToggledOn,
        "Cooked": Cooked,
        "Frozen": Frozen,
    }

    def __init__(self):
        self.transitions = []
        self.tracked_predicates = []
        self.prev_states = {}
        self._goal_heads = []
        self._task_objects = []
        self._robot = None

    def start(self, env, robot):
        """Initialize tracking for a new episode."""
        self.transitions = []
        self.tracked_predicates = []
        self.prev_states = {}
        self._goal_heads = []
        self._task_objects = []
        self._robot = robot
        task = env.task

        # 1. Goal conditions from ground_goal_state_options[0]
        try:
            if hasattr(task, "ground_goal_state_options") and task.ground_goal_state_options:
                self._goal_heads = list(task.ground_goal_state_options[0])
                for i, head in enumerate(self._goal_heads):
                    pred_id = f"goal_{i}"
                    predicate_name = "unknown"
                    args = []
                    try:
                        if isinstance(head.body, (list, tuple)) and len(head.body) > 0:
                            predicate_name = str(head.body[0])
                            args = [str(a) for a in head.body[1:]]
                    except Exception:
                        pass
                    self.tracked_predicates.append({
                        "id": pred_id,
                        "type": "goal_condition",
                        "predicate": predicate_name,
                        "args": args,
                    })
                    try:
                        self.prev_states[pred_id] = bool(head.evaluate())
                    except Exception:
                        self.prev_states[pred_id] = False
        except Exception as e:
            logger.warning(f"BDDLStateTracker: Failed to parse goal conditions: {e}")

        # 2. Grasping per arm
        try:
            for arm in robot.arm_names:
                pred_id = f"grasp_{arm}"
                self.tracked_predicates.append({
                    "id": pred_id,
                    "type": "grasp",
                    "arm": arm,
                })
                obj = robot._ag_obj_in_hand.get(arm)
                self.prev_states[pred_id] = obj.name if obj is not None else None
        except Exception as e:
            logger.warning(f"BDDLStateTracker: Failed to parse grasp states: {e}")

        # 3. Collect task objects (non-system, existing entities)
        try:
            if hasattr(task, "object_scope"):
                for inst_name, entity in task.object_scope.items():
                    if entity is None or entity.is_system or not entity.exists:
                        continue
                    obj = entity.wrapped_obj
                    if obj is not None and inst_name != "agent.n.01_1":
                        self._task_objects.append((inst_name, obj))
        except Exception as e:
            logger.warning(f"BDDLStateTracker: Failed to collect task objects: {e}")

        # 4. Pairwise binary spatial predicates between task objects
        for inst_a, obj_a in self._task_objects:
            for inst_b, obj_b in self._task_objects:
                if inst_a == inst_b:
                    continue
                for state_name, state_cls in self.BINARY_STATE_CLASSES.items():
                    if state_cls not in obj_a.states:
                        continue
                    pred_id = f"{state_name}_{inst_a}_{inst_b}"
                    self.tracked_predicates.append({
                        "id": pred_id,
                        "type": "binary_state",
                        "predicate": state_name,
                        "args": [inst_a, inst_b],
                    })
                    try:
                        self.prev_states[pred_id] = bool(obj_a.states[state_cls].get_value(obj_b))
                    except Exception:
                        self.prev_states[pred_id] = False

        # 5. Unary predicates on task objects
        for inst_name, obj in self._task_objects:
            for state_name, state_cls in self.UNARY_STATE_CLASSES.items():
                if state_cls not in obj.states:
                    continue
                pred_id = f"{state_name}_{inst_name}"
                self.tracked_predicates.append({
                    "id": pred_id,
                    "type": "unary_state",
                    "predicate": state_name,
                    "args": [inst_name],
                })
                try:
                    self.prev_states[pred_id] = bool(obj.states[state_cls].get_value())
                except Exception:
                    self.prev_states[pred_id] = False

        logger.info(
            f"BDDLStateTracker: Tracking {len(self.tracked_predicates)} predicates "
            f"({len(self._goal_heads)} goal, {len(self._task_objects)} task objects)"
        )

    def step(self, env, robot, step_idx):
        """Check all tracked predicates for state changes after an env step."""
        # 1. Goal conditions
        for i, head in enumerate(self._goal_heads):
            pred_id = f"goal_{i}"
            try:
                new_val = bool(head.evaluate())
            except Exception:
                continue
            old_val = self.prev_states.get(pred_id)
            if new_val != old_val:
                self.transitions.append({
                    "timestep": step_idx,
                    "predicate_id": pred_id,
                    "old_value": old_val,
                    "new_value": new_val,
                })
                self.prev_states[pred_id] = new_val

        # 2. Grasping per arm
        for arm in robot.arm_names:
            pred_id = f"grasp_{arm}"
            if pred_id not in self.prev_states:
                continue
            try:
                obj = robot._ag_obj_in_hand.get(arm)
                new_val = obj.name if obj is not None else None
            except Exception:
                continue
            old_val = self.prev_states.get(pred_id)
            if new_val != old_val:
                self.transitions.append({
                    "timestep": step_idx,
                    "predicate_id": pred_id,
                    "old_value": old_val,
                    "new_value": new_val,
                })
                self.prev_states[pred_id] = new_val

        # 3. Binary spatial predicates
        for inst_a, obj_a in self._task_objects:
            for inst_b, obj_b in self._task_objects:
                if inst_a == inst_b:
                    continue
                for state_name, state_cls in self.BINARY_STATE_CLASSES.items():
                    pred_id = f"{state_name}_{inst_a}_{inst_b}"
                    if pred_id not in self.prev_states:
                        continue
                    try:
                        new_val = bool(obj_a.states[state_cls].get_value(obj_b))
                    except Exception:
                        continue
                    old_val = self.prev_states.get(pred_id)
                    if new_val != old_val:
                        self.transitions.append({
                            "timestep": step_idx,
                            "predicate_id": pred_id,
                            "old_value": old_val,
                            "new_value": new_val,
                        })
                        self.prev_states[pred_id] = new_val

        # 4. Unary predicates
        for inst_name, obj in self._task_objects:
            for state_name, state_cls in self.UNARY_STATE_CLASSES.items():
                pred_id = f"{state_name}_{inst_name}"
                if pred_id not in self.prev_states:
                    continue
                try:
                    new_val = bool(obj.states[state_cls].get_value())
                except Exception:
                    continue
                old_val = self.prev_states.get(pred_id)
                if new_val != old_val:
                    self.transitions.append({
                        "timestep": step_idx,
                        "predicate_id": pred_id,
                        "old_value": old_val,
                        "new_value": new_val,
                    })
                    self.prev_states[pred_id] = new_val

    def get_results(self, task_name, instance_id, demo_id, total_steps, success):
        """Return JSON-serializable dict of all tracking data."""
        return {
            "task_name": task_name,
            "instance_id": instance_id,
            "demo_id": demo_id,
            "total_steps": total_steps,
            "success": success,
            "tracked_predicates": self.tracked_predicates,
            "transitions": self.transitions,
        }


class DirectDataRecorder:
    """
    Records data directly to parquet and video files during evaluation,
    bypassing the intermediate HDF5 step.
    """

    def __init__(
        self,
        output_folder: str,
        task_name: str,
        task_id: int,
        demo_id: int,
        camera_names: Dict[str, str] = ROBOT_CAMERA_NAMES["R1Pro"],
        record_rgb: bool = True,
        record_depth: bool = True,
        only_successes: bool = False,
    ):
        self.output_folder = output_folder
        self.task_name = task_name
        self.task_id = task_id
        self.demo_id = demo_id
        self.camera_names = camera_names
        self.record_rgb = record_rgb
        self.record_depth = record_depth
        self.only_successes = only_successes

        # Data buffers for parquet
        self.actions = []
        self.proprios = []
        self.cam_rel_poses = []
        self.task_infos = []

        # Video writers
        self.video_writers: Dict[str, Tuple[Container, Stream]] = {}

        # Frame buffers for video (to batch write)
        self.rgb_buffers: Dict[str, List[np.ndarray]] = {}
        self.depth_buffers: Dict[str, List[np.ndarray]] = {}

        self.step_count = 0
        self.is_recording = False

        # Create output directories
        self._setup_directories()

    def _setup_directories(self):
        """Create output directories for parquet, videos, and trajectories under both success/ and failure/."""
        for outcome in ("success", "failure"):
            # Parquet directory
            parquet_dir = os.path.join(
                self.output_folder, outcome, "2025-challenge-demos", "data", f"task-{self.task_id:04d}"
            )
            makedirs_with_mode(parquet_dir)

            # Metadata directory
            meta_dir = os.path.join(
                self.output_folder, outcome, "2025-challenge-demos", "meta", "episodes", f"task-{self.task_id:04d}"
            )
            makedirs_with_mode(meta_dir)

            # Trajectories (HDF5) directory
            traj_dir = os.path.join(
                self.output_folder, outcome, "2025-challenge-demos", "trajectories", f"task-{self.task_id:04d}"
            )
            makedirs_with_mode(traj_dir)

            # Video directories
            if self.record_rgb or self.record_depth:
                for camera_id, camera_name in self.camera_names.items():
                    if self.record_rgb:
                        rgb_dir = os.path.join(
                            self.output_folder,
                            outcome,
                            "2025-challenge-demos",
                            "videos",
                            f"task-{self.task_id:04d}",
                            f"observation.images.rgb.{camera_id}",
                        )
                        makedirs_with_mode(rgb_dir)

                    if self.record_depth:
                        depth_dir = os.path.join(
                            self.output_folder,
                            outcome,
                            "2025-challenge-demos",
                            "videos",
                            f"task-{self.task_id:04d}",
                            f"observation.images.depth.{camera_id}",
                        )
                        makedirs_with_mode(depth_dir)

    def _get_outcome_folder(self, success: bool) -> str:
        """Return the outcome subfolder path ('success' or 'failure')."""
        return os.path.join(self.output_folder, "success" if success else "failure")

    def start_episode(self):
        """Start recording a new episode.

        Files are written to a _staging/ subfolder first, then moved to
        success/ or failure/ at the end of the episode.
        """
        self.actions = []
        self.proprios = []
        self.cam_rel_poses = []
        self.task_infos = []
        self.step_count = 0
        self.is_recording = True

        # Staging folder for in-progress episode
        self._staging_dir = os.path.join(self.output_folder, "_staging")
        self._staging_video_paths: List[str] = []

        # Initialize video writers (writing to staging area)
        for camera_id, camera_name in self.camera_names.items():
            resolution = HEAD_RESOLUTION if "zed" in camera_name else WRIST_RESOLUTION

            if self.record_rgb:
                rgb_dir = os.path.join(
                    self._staging_dir,
                    "2025-challenge-demos",
                    "videos",
                    f"task-{self.task_id:04d}",
                    f"observation.images.rgb.{camera_id}",
                )
                makedirs_with_mode(rgb_dir)
                rgb_path = os.path.join(rgb_dir, f"episode_{self.demo_id:08d}.mp4")
                self._staging_video_paths.append(rgb_path)
                self.video_writers[f"{camera_name}::rgb"] = create_video_writer(
                    fpath=rgb_path,
                    resolution=resolution,
                    codec_name="libx265",
                    pix_fmt="yuv420p",
                    stream_options={"x265-params": "log-level=none"},
                )
                self.rgb_buffers[camera_name] = []

            if self.record_depth:
                depth_dir = os.path.join(
                    self._staging_dir,
                    "2025-challenge-demos",
                    "videos",
                    f"task-{self.task_id:04d}",
                    f"observation.images.depth.{camera_id}",
                )
                makedirs_with_mode(depth_dir)
                depth_path = os.path.join(depth_dir, f"episode_{self.demo_id:08d}.mp4")
                self._staging_video_paths.append(depth_path)
                self.video_writers[f"{camera_name}::depth_linear"] = create_video_writer(
                    fpath=depth_path,
                    resolution=resolution,
                    codec_name="libx265",
                    pix_fmt="yuv420p10le",
                    stream_options={"x265-params": "lossless=1:log-level=none"},
                )
                self.depth_buffers[camera_name] = []

    def record_step(
        self,
        action: np.ndarray,
        proprio: np.ndarray,
        cam_rel_poses: np.ndarray,
        task_info: Optional[np.ndarray],
        obs: Dict[str, th.Tensor],
    ):
        """Record a single step of data."""
        if not self.is_recording:
            return

        # Debug: log obs keys on first step
        if self.step_count == 0:
            logger.info(f"Observation keys: {list(obs.keys())}")
            for camera_id, camera_name in self.camera_names.items():
                logger.info(f"Looking for camera {camera_id}: {camera_name}::rgb")

        # Record low-dim data
        self.actions.append(action.copy() if isinstance(action, np.ndarray) else action.cpu().numpy().copy())
        self.proprios.append(proprio.copy() if isinstance(proprio, np.ndarray) else proprio.cpu().numpy().copy())
        self.cam_rel_poses.append(
            cam_rel_poses.copy() if isinstance(cam_rel_poses, np.ndarray) else cam_rel_poses.cpu().numpy().copy()
        )
        if task_info is not None:
            self.task_infos.append(
                task_info.copy() if isinstance(task_info, np.ndarray) else task_info.cpu().numpy().copy()
            )

        # Record video frames
        for camera_id, camera_name in self.camera_names.items():
            if self.record_rgb and f"{camera_name}::rgb" in obs:
                rgb_frame = obs[f"{camera_name}::rgb"]
                if isinstance(rgb_frame, th.Tensor):
                    rgb_frame = rgb_frame.cpu().numpy()
                self.rgb_buffers[camera_name].append(rgb_frame[..., :3].astype(np.uint8))

            if self.record_depth and f"{camera_name}::depth_linear" in obs:
                depth_frame = obs[f"{camera_name}::depth_linear"]
                if isinstance(depth_frame, th.Tensor):
                    depth_frame = depth_frame.cpu().numpy()
                self.depth_buffers[camera_name].append(depth_frame)

        self.step_count += 1

        # Periodically flush video buffers to avoid memory issues
        if self.step_count % 500 == 0:
            self._flush_video_buffers()
            logger.info(f"Flushed video buffers at step {self.step_count}, rgb_buffer_lens: {[len(v) for v in self.rgb_buffers.values()]}")

    def _flush_video_buffers(self):
        """Flush video buffers to disk."""
        for camera_name in self.camera_names.values():
            if self.record_rgb and camera_name in self.rgb_buffers and len(self.rgb_buffers[camera_name]) > 0:
                rgb_data = np.stack(self.rgb_buffers[camera_name], axis=0)
                write_video(
                    rgb_data,
                    video_writer=self.video_writers[f"{camera_name}::rgb"],
                    batch_size=len(rgb_data),
                    mode="rgb",
                )
                self.rgb_buffers[camera_name] = []

            if self.record_depth and camera_name in self.depth_buffers and len(self.depth_buffers[camera_name]) > 0:
                depth_data = np.stack(self.depth_buffers[camera_name], axis=0)
                write_video(
                    depth_data,
                    video_writer=self.video_writers[f"{camera_name}::depth_linear"],
                    batch_size=len(depth_data),
                    mode="depth",
                )
                self.depth_buffers[camera_name] = []

    def end_episode(self, success: bool, env=None, bddl_transitions=None) -> bool:
        """
        End the current episode and save data.

        Data is saved to success/ or failure/ subfolder based on the outcome.
        When only_successes is True, only successful episodes are saved.
        When only_successes is False, all episodes (success and failure) are saved.

        Args:
            success: Whether the episode was successful.
            env: The OmniGibson environment, used to extract rich metadata (config, scene_file,
                 task_obs_keys, ins_id_mapping, unique_ins_ids per camera).
            bddl_transitions: Optional dict of BDDL state transitions to save as JSON.

        Returns:
            bool: Whether data was saved
        """
        if not self.is_recording:
            return False

        self.is_recording = False
        self._success = success

        # Check if we should save
        if self.only_successes and not success:
            logger.info(f"Episode failed, discarding data (only_successes={self.only_successes})")
            self._close_video_writers()
            self._cleanup_staging()
            return False

        # Flush remaining video buffers
        self._flush_video_buffers()

        # Close video writers
        self._close_video_writers()

        # Determine destination folder
        outcome_folder = self._get_outcome_folder(success)

        # Move staged video files to the outcome folder
        self._move_staged_videos(outcome_folder)

        # Save parquet directly to outcome folder
        self._save_parquet(outcome_folder)

        # Save metadata directly to outcome folder
        self._save_metadata(outcome_folder, success=success, env=env)

        # Save BDDL state transitions if provided
        if bddl_transitions is not None:
            meta_dir = os.path.join(
                outcome_folder, "2025-challenge-demos", "meta", "episodes", f"task-{self.task_id:04d}"
            )
            makedirs_with_mode(meta_dir)
            bddl_path = os.path.join(meta_dir, f"episode_{self.demo_id:08d}_bddl_transitions.json")
            with open(bddl_path, "w") as f:
                json.dump(bddl_transitions, f, indent=2)
            logger.info(f"Saved BDDL transitions to {bddl_path}")

        logger.info(f"Saved episode data to {outcome_folder} (success={success})")
        return True

    def _close_video_writers(self):
        """Close all video writers."""
        for key, (container, stream) in self.video_writers.items():
            try:
                # Flush any remaining packets
                for packet in stream.encode():
                    container.mux(packet)
                container.close()
            except Exception as e:
                logger.warning(f"Error closing video writer {key}: {e}")
        self.video_writers = {}

    def _cleanup_staging(self):
        """Remove staged video files for discarded episodes."""
        for path in getattr(self, "_staging_video_paths", []):
            if os.path.exists(path):
                os.remove(path)

    def _move_staged_videos(self, outcome_folder: str):
        """Move staged video files to the appropriate outcome folder."""
        for src_path in getattr(self, "_staging_video_paths", []):
            if not os.path.exists(src_path):
                continue
            # Compute the relative path from the staging dir
            rel_path = os.path.relpath(src_path, self._staging_dir)
            dst_path = os.path.join(outcome_folder, rel_path)
            makedirs_with_mode(os.path.dirname(dst_path))
            shutil.move(src_path, dst_path)

    def _save_parquet(self, outcome_folder: str):
        """Save low-dimensional data to parquet file."""
        if len(self.actions) == 0:
            logger.warning("No data to save to parquet")
            return

        T = len(self.actions)
        actions = np.array(self.actions, dtype=np.float32)
        proprios = np.array(self.proprios, dtype=np.float32)
        cam_rel_poses = np.array(self.cam_rel_poses, dtype=np.float32)

        data = {
            "index": np.arange(T, dtype=np.int64),
            "episode_index": np.zeros(T, dtype=np.int64) + self.demo_id,
            "task_index": np.zeros(T, dtype=np.int64) + self.task_id,
            "timestamp": np.arange(T, dtype=np.float64) / 30.0,  # 30 fps
            "observation.state": list(proprios),
            "observation.cam_rel_poses": list(cam_rel_poses),
            "action": list(actions),
        }

        if len(self.task_infos) > 0:
            task_infos = np.array(self.task_infos, dtype=np.float32)
            data["observation.task_info"] = list(task_infos)

        df = pd.DataFrame(data)
        parquet_dir = os.path.join(
            outcome_folder, "2025-challenge-demos", "data", f"task-{self.task_id:04d}"
        )
        makedirs_with_mode(parquet_dir)
        parquet_path = os.path.join(parquet_dir, f"episode_{self.demo_id:08d}.parquet")
        df.to_parquet(parquet_path, index=False)
        logger.info(f"Saved parquet to {parquet_path}")

    def _save_metadata(self, outcome_folder: str, success: bool = False, env=None):
        """Save metadata JSON file matching the format produced by replay_obs.py.

        When env is provided, the metadata includes:
          - config: full environment config (JSON string)
          - scene_file: saved scene state (JSON string)
          - n_episodes, n_steps, num_samples: episode statistics
          - robot_type: "R1Pro"
          - task_obs_keys: task observation key names
          - ins_id_mapping: JSON-serialized VisionSensor.INSTANCE_ID_REGISTRY
          - {camera_name}::unique_ins_ids: unique instance IDs per camera
          - success: whether the episode was successful
        """
        n_steps = len(self.actions)
        metadata = {
            "n_episodes": 1,
            "n_steps": n_steps,
            "num_samples": n_steps,
            "robot_type": "R1Pro",
            "success": success,
        }

        if env is not None:
            try:
                # Unwrap to get the base OG environment
                base_env = env
                while hasattr(base_env, "env"):
                    base_env = base_env.env

                # Config
                from copy import deepcopy
                config = deepcopy(base_env.config)
                metadata["config"] = json.dumps(config, default=str)

                # Scene file
                scene_file = base_env.scene.save()
                metadata["scene_file"] = json.dumps(scene_file, default=str)

                # Task observation keys
                if hasattr(base_env, "task") and hasattr(base_env.task, "low_dim_obs_keys"):
                    metadata["task_obs_keys"] = list(base_env.task.low_dim_obs_keys)

                # Instance ID mapping
                metadata["ins_id_mapping"] = json.dumps(VisionSensor.INSTANCE_ID_REGISTRY)

                # Unique instance IDs per camera
                camera_names_set = set(self.camera_names.values())
                for sensor_name in base_env.robots[0].sensors:
                    full_name = f"robot_r1::{sensor_name}"
                    if full_name in camera_names_set:
                        sensor = base_env.robots[0].sensors[sensor_name]
                        if hasattr(sensor, "get_obs") and callable(sensor.get_obs):
                            try:
                                obs = sensor.get_obs()
                                if "seg_instance_id" in obs:
                                    unique_ids = th.unique(obs["seg_instance_id"]).to(th.uint32).tolist()
                                    metadata[f"{full_name}::unique_ins_ids"] = unique_ids
                            except Exception:
                                pass
            except Exception as e:
                logger.warning(f"Could not extract rich metadata from env: {e}")

        meta_dir = os.path.join(
            outcome_folder, "2025-challenge-demos", "meta", "episodes", f"task-{self.task_id:04d}"
        )
        makedirs_with_mode(meta_dir)
        meta_path = os.path.join(meta_dir, f"episode_{self.demo_id:08d}.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Saved metadata to {meta_path}")


class EvaluatorWithDirectRecording:
    """
    Evaluator class that directly records data to parquet and video files
    during policy evaluation, bypassing the intermediate HDF5 step.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

        # record total number and success number of trials and trial time
        self.n_trials = 0
        self.n_success_trials = 0
        self.total_time = 0
        self.robot_action = dict()
        self.data_recorder: Optional[DirectDataRecorder] = None
        self.hdf5_recorder = None
        self.output_folder: Optional[str] = None
        self.bddl_tracker = BDDLStateTracker()

        self.env = self.load_env(env_wrapper=self.cfg.env_wrapper)
        self.policy = self.load_policy()
        self.robot = self.load_robot()
        self.metrics = self.load_metrics()

        self.reset()
        # manually reset environment episode number
        self.env._current_episode = 0
        self._video_writer = None

    def load_env(self, env_wrapper: DictConfig) -> EnvironmentWrapper:
        """
        Read the environment config file and create the environment.
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
        # Update observation modalities - include depth for recording
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
        # Wrap with DataCollectionWrapper for HDF5 recording if recording_path is specified
        record_path = getattr(self.cfg, "recording_path", None)
        if record_path is not None:
            record_path = Path(record_path).expanduser()
            only_successes = getattr(self.cfg, "only_successes", False)
            logger.info(f"Recording rollouts to HDF5: {record_path} (only_successes={only_successes})")
            env = PolicyEvalDataCollectionWrapper(
                env=env,
                output_path=str(record_path),
                only_successes=only_successes,
                flush_every_n_traj=1,
            )
            self.hdf5_recorder = env
        return env

    def load_robot(self) -> BaseRobot:
        """Loads and returns the robot instance from the environment."""
        robot = self.env.scene.object_registry("name", "robot_r1")
        return robot

    def load_policy(self) -> Any:
        """Loads and returns the policy instance."""
        policy = instantiate(self.cfg.model)
        logger.info("")
        logger.info("=" * 50)
        logger.info(f"Loaded policy: {self.cfg.policy_name}")
        logger.info("=" * 50)
        logger.info("")
        return policy

    def load_metrics(self) -> List[MetricBase]:
        """Load agent and task metrics."""
        return [AgentMetric(self.human_stats), TaskMetric(self.human_stats)]

    def setup_recorder(self, output_folder: str, demo_id: int, record_rgb: bool = True, record_depth: bool = True):
        """Setup the direct data recorder for the current episode."""
        task_name = self.cfg.task.name
        task_id = TASK_NAMES_TO_INDICES[task_name]
        self.data_recorder = DirectDataRecorder(
            output_folder=output_folder,
            task_name=task_name,
            task_id=task_id,
            demo_id=demo_id,
            camera_names=ROBOT_CAMERA_NAMES["R1Pro"],
            record_rgb=record_rgb,
            record_depth=record_depth,
            only_successes=getattr(self.cfg, "only_successes", False),
        )

    def step(self) -> Tuple[bool, bool]:
        """
        Performs a single step of the task by executing the policy, interacting with the environment,
        processing observations, updating metrics, and tracking trial success.
        """
        self.robot_action = self.policy.forward(obs=self.obs)

        obs, _, terminated, truncated, info = self.env.step(self.robot_action, n_render_iterations=1)

        # Flatten obs for recording (this gives us keys like "robot_r1::robot_r1:zed_link:Camera:0::rgb")
        flat_obs = flatten_obs_dict(obs)

        # process obs (adds cam_rel_poses, task_id)
        self.obs = self._preprocess_obs(obs)

        # Record data if recorder is active
        if self.data_recorder is not None and self.data_recorder.is_recording:
            # Extract proprio from preprocessed obs
            proprio = self.obs.get("robot_r1::proprio", None)
            if proprio is None:
                proprio = flat_obs.get("robot_r1::proprio", None)

            cam_rel_poses = self.obs.get("robot_r1::cam_rel_poses", None)
            task_info = self.obs.get("task::low_dim", flat_obs.get("task::low_dim", None))

            # Get action
            action = self.robot_action
            if isinstance(action, dict):
                # Flatten action dict if needed
                action = np.concatenate([v.cpu().numpy() if isinstance(v, th.Tensor) else v for v in action.values()])
            elif isinstance(action, th.Tensor):
                action = action.cpu().numpy()

            # Record step with flattened observations for videos
            self.data_recorder.record_step(
                action=action,
                proprio=proprio,
                cam_rel_poses=cam_rel_poses,
                task_info=task_info,
                obs=flat_obs,
            )

        # Track BDDL state transitions
        if self.data_recorder is not None:
            try:
                self.bddl_tracker.step(self.env, self.robot, self.data_recorder.step_count)
            except Exception as e:
                logger.debug(f"BDDL tracker step error: {e}")

        if terminated or truncated:
            self.n_trials += 1
            if info["done"]["success"]:
                self.n_success_trials += 1

        for metric in self.metrics:
            metric.step_callback(self.env)
        return terminated, truncated, info

    @property
    def video_writer(self) -> Tuple[Container, Stream]:
        """Returns the video writer for the current evaluation step."""
        return self._video_writer

    @video_writer.setter
    def video_writer(self, video_writer: Tuple[Container, Stream]) -> None:
        if self._video_writer is not None:
            (container, stream) = self._video_writer
            for packet in stream.encode():
                container.mux(packet)
            container.close()
        self._video_writer = video_writer

    def load_task_instance(self, instance_id: int, test_hidden: bool = False) -> None:
        """Loads the configuration for a specific task instance."""
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
                self.env.scene.write_task_metadata(key=tro_key, data=tro_state)
            else:
                self.env.task.object_scope[tro_key].load_state(tro_state, serialized=False)

        for _ in range(25):
            og.sim.step_physics()
            for entity in self.env.task.object_scope.values():
                if not entity.is_system and entity.exists:
                    entity.keep_still()

        self.env.scene.update_initial_file()
        self.env.scene.reset()

    def _preprocess_obs(self, obs: dict) -> dict:
        """Preprocess the observation dictionary before passing it to the policy."""
        obs = flatten_obs_dict(obs)
        base_pose = self.robot.get_position_orientation()
        cam_rel_poses = []
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
        obs["task_id"] = th.tensor([TASK_NAMES_TO_INDICES[self.cfg.task.name]], dtype=th.int64)
        return obs

    def _write_video(self) -> None:
        """Write the current robot observations to video (for preview video)."""
        if ROBOT_CAMERA_NAMES["R1Pro"]["head"] + "::rgb" not in self.obs:
            return
        left_wrist_rgb = cv2.resize(
            self.obs[ROBOT_CAMERA_NAMES["R1Pro"]["left_wrist"] + "::rgb"].numpy(),
            (224, 224),
        )
        right_wrist_rgb = cv2.resize(
            self.obs[ROBOT_CAMERA_NAMES["R1Pro"]["right_wrist"] + "::rgb"].numpy(),
            (224, 224),
        )
        head_rgb = cv2.resize(
            self.obs[ROBOT_CAMERA_NAMES["R1Pro"]["head"] + "::rgb"].numpy(),
            (448, 448),
        )
        write_video(
            np.expand_dims(np.hstack([np.vstack([left_wrist_rgb, right_wrist_rgb]), head_rgb]), 0),
            video_writer=self.video_writer,
            batch_size=1,
            mode="rgb",
        )

    def reset(self) -> None:
        """Reset the environment, policy, and compute metrics."""
        self.obs = self._preprocess_obs(self.env.reset()[0])
        for metric in self.metrics:
            metric.start_callback(self.env)
        self.policy.reset()
        self.n_success_trials, self.n_trials = 0, 0

    def start_episode_recording(self):
        """Start recording for the current episode."""
        if self.data_recorder is not None:
            self.data_recorder.start_episode()
        try:
            self.bddl_tracker.start(self.env, self.robot)
        except Exception as e:
            logger.warning(f"Failed to start BDDL state tracker: {e}")

    def end_episode_recording(self, success: bool) -> bool:
        """End recording and save data if successful."""
        if self.data_recorder is not None:
            bddl_results = None
            try:
                instance_id = (self.data_recorder.demo_id % 10000) // 10
                bddl_results = self.bddl_tracker.get_results(
                    task_name=self.cfg.task.name,
                    instance_id=instance_id,
                    demo_id=self.data_recorder.demo_id,
                    total_steps=self.data_recorder.step_count,
                    success=success,
                )
            except Exception as e:
                logger.warning(f"Failed to get BDDL tracking results: {e}")
            return self.data_recorder.end_episode(success, env=self.env, bddl_transitions=bddl_results)
        return False

    def _finalize_episode_recording(self, instance_id: int, episode_idx: int, demo_id: int, success: bool = False) -> None:
        """Flush the current HDF5 trajectory to disk and tag it with metadata."""
        if self.hdf5_recorder is None:
            return
        prev_traj_count = self.hdf5_recorder.traj_count
        self.hdf5_recorder.flush_current_traj()
        if self.hdf5_recorder.traj_count > prev_traj_count:
            demo_idx = self.hdf5_recorder.traj_count - 1
            demo_group = self.hdf5_recorder.hdf5_file["data"][f"demo_{demo_idx}"]
            self.hdf5_recorder.add_metadata(group=demo_group, name="task_name", data=self.cfg.task.name)
            self.hdf5_recorder.add_metadata(group=demo_group, name="instance_id", data=instance_id)
            self.hdf5_recorder.add_metadata(group=demo_group, name="episode_idx", data=episode_idx)
            self.hdf5_recorder.add_metadata(group=demo_group, name="demo_id", data=demo_id)
            self.hdf5_recorder.add_metadata(group=demo_group, name="success", data=success)

    def _finalize_data_collection(self, output_folder: str = None) -> None:
        """Flush any pending HDF5 data, close the recorder, and split into per-episode files."""
        if self.hdf5_recorder is None:
            return
        if len(self.hdf5_recorder.current_traj_history) > 0:
            self.hdf5_recorder.flush_current_traj()
        # Grab filename before save_data() closes the file
        combined_path = self.hdf5_recorder.hdf5_file.filename
        self.hdf5_recorder.save_data()

        # Split the combined HDF5 into per-episode files under success/ and failure/
        if os.path.exists(combined_path) and output_folder is not None:
            self._split_hdf5_by_outcome(combined_path, output_folder)

    def _split_hdf5_by_outcome(self, combined_path: str, output_folder: str) -> None:
        """
        Split a combined HDF5 file into per-episode HDF5 files under success/ and failure/,
        matching the same naming convention as parquet and video files:
          {output_folder}/{outcome}/2025-challenge-demos/trajectories/task-{task_id:04d}/episode_{demo_id:08d}.hdf5
        """
        try:
            with h5py.File(combined_path, "r") as src:
                if "data" not in src:
                    logger.warning("No data group found in HDF5 file, skipping split")
                    return

                success_count = 0
                failure_count = 0

                for demo_name in sorted(src["data"].keys()):
                    demo_group = src["data"][demo_name]
                    is_success = bool(demo_group.attrs.get("success", False))
                    demo_id = int(demo_group.attrs.get("demo_id", 0))
                    task_id = demo_id // 10000

                    outcome = "success" if is_success else "failure"
                    episode_dir = os.path.join(
                        output_folder, outcome, "2025-challenge-demos", "trajectories", f"task-{task_id:04d}"
                    )
                    os.makedirs(episode_dir, exist_ok=True)
                    episode_path = os.path.join(episode_dir, f"episode_{demo_id:08d}.hdf5")

                    with h5py.File(episode_path, "w") as dst:
                        # Copy top-level attributes from source
                        for attr_name, attr_val in src.attrs.items():
                            dst.attrs[attr_name] = attr_val
                        dst.create_group("data")
                        src["data"].copy(demo_name, dst["data"], name="demo_0")
                        dst["data"].attrs["num_demos"] = 1

                    if is_success:
                        success_count += 1
                    else:
                        failure_count += 1
                    logger.info(f"Saved HDF5 episode to {episode_path}")

                logger.info(f"Split HDF5: {success_count} success, {failure_count} failure episodes")

        except Exception as e:
            logger.error(f"Error splitting HDF5 file: {e}")
            logger.info(f"Combined HDF5 file is still available at: {combined_path}")

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
        self.video_writer = None
        self._finalize_data_collection(output_folder=self.output_folder)
        self.env.close()
        og.shutdown()

    def _sigint_handler(self, signal_received, frame):
        logger.warning("SIGINT or CTRL-C detected.\n")
        self.__exit__(None, None, None)
        sys.exit(0)


if __name__ == "__main__":
    register_omegaconf_resolvers()
    # open yaml from task path
    with hydra.initialize_config_dir(f"{Path(getsourcefile(lambda: 0)).parents[0]}/configs", version_base="1.1"):
        config = hydra.compose("base_config.yaml", overrides=sys.argv[1:])
    OmegaConf.resolve(config)
    # set headless mode
    gm.HEADLESS = config.headless
    # set video path
    if config.write_video:
        video_path = Path(config.log_path).expanduser() / "videos"
        video_path.mkdir(parents=True, exist_ok=True)

    # Set output folder for direct recording
    output_folder = getattr(config, "output_folder", None)
    if output_folder is None:
        output_folder = Path(config.log_path).expanduser() / "direct_output"
    else:
        output_folder = Path(output_folder).expanduser()
    output_folder.mkdir(parents=True, exist_ok=True)

    # Set default HDF5 recording path under output_folder/_staging/ (will be split into
    # per-episode files under success/ and failure/ at the end, matching parquet/video naming)
    task_id_for_path = TASK_NAMES_TO_INDICES[config.task.name]
    if getattr(config, "recording_path", None) is None:
        traj_dir = output_folder / "_staging" / "2025-challenge-demos" / "trajectories" / f"task-{task_id_for_path:04d}"
        traj_dir.mkdir(parents=True, exist_ok=True)
        config.recording_path = str(traj_dir / f"{config.task.name}.hdf5")
    else:
        traj_dir = Path(config.recording_path).expanduser().parent
        traj_dir.mkdir(parents=True, exist_ok=True)

    # Recording options
    record_rgb = getattr(config, "record_rgb", True)
    record_depth = getattr(config, "record_depth", True)

    assert not (
        config.eval_on_train_instances and config.test_hidden
    ), "Cannot eval on train instances and test hidden instances simultaneously."
    if config.test_hidden:
        logger.info("You are evaluating on hidden test instances! This is for internal use only.")

    # get run instances
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
        # load csv file first to determine number of available instances
        task_instance_csv_path = os.path.join(
            gm.DATA_PATH, "2025-challenge-task-instances", "metadata", "test_instances.csv"
        )
        with open(task_instance_csv_path, "r") as f:
            lines = list(csv.reader(f))[1:]
        assert (
            lines[TASK_NAMES_TO_INDICES[config.task.name]][1] == config.task.name
        ), f"Task name from config {config.task.name} does not match task name from csv"
        test_instances = lines[TASK_NAMES_TO_INDICES[config.task.name]][2].strip().split(",")
        num_available_instances = len(test_instances)

        instances_to_run = (
            config.eval_instance_ids if config.eval_instance_ids is not None else set(range(min(m.NUM_EVAL_INSTANCES, num_available_instances)))
        )
        assert set(instances_to_run).issubset(
            set(range(num_available_instances))
        ), f"eval instance ids must be in range({num_available_instances})"
        instances_to_run = [int(test_instances[i]) for i in instances_to_run]

    # establish metrics
    metrics = {}
    metrics_path = Path(config.log_path).expanduser() / "metrics"
    metrics_path.mkdir(parents=True, exist_ok=True)

    task_id = TASK_NAMES_TO_INDICES[config.task.name]

    with EvaluatorWithDirectRecording(config) as evaluator:
        evaluator.output_folder = str(output_folder)
        logger.info("Starting evaluation with direct data recording...")

        for idx in instances_to_run:
            evaluator.reset()
            evaluator.load_task_instance(idx, test_hidden=config.test_hidden)
            logger.info(f"Starting task instance {idx} for evaluation...")

            for epi in range(m.NUM_EVAL_EPISODES):
                evaluator.reset()
                done = False

                # Compute demo_id: task_id * 10000 + instance_id * 10 + episode_id
                demo_id = task_id * 10000 + int(idx) * 10 + epi

                # Setup recorder for this episode
                evaluator.setup_recorder(
                    output_folder=str(output_folder),
                    demo_id=demo_id,
                    record_rgb=record_rgb,
                    record_depth=record_depth,
                )
                evaluator.start_episode_recording()

                if config.write_video:
                    video_name = str(video_path) + f"/{config.task.name}_{idx}_{epi}.mp4"
                    evaluator.video_writer = create_video_writer(
                        fpath=video_name,
                        resolution=(448, 672),
                    )

                # run metric start callbacks
                for metric in evaluator.metrics:
                    metric.start_callback(evaluator.env)

                success = False
                while not done:
                    terminated, truncated, info = evaluator.step()
                    if terminated or truncated:
                        done = True
                        success = info["done"]["success"]
                    if config.write_video:
                        evaluator._write_video()
                    if evaluator.env._current_step % 1000 == 0:
                        logger.info(f"Current step: {evaluator.env._current_step}")

                # End recording (parquet + video)
                saved = evaluator.end_episode_recording(success)
                if saved:
                    logger.info(f"Data saved for demo_id={demo_id}")

                # Finalize HDF5 recording for this episode
                evaluator._finalize_episode_recording(instance_id=int(idx), episode_idx=epi, demo_id=demo_id, success=success)

                # run metric end callbacks
                for metric in evaluator.metrics:
                    metric.end_callback(evaluator.env)

                logger.info(f"Evaluation finished at step {evaluator.env._current_step}.")
                logger.info(f"Evaluation exit state: terminated={terminated}, truncated={truncated}, success={success}")
                logger.info(f"Total trials: {evaluator.n_trials}")
                logger.info(f"Total success trials: {evaluator.n_success_trials}")

                # gather metric results and write to file
                for metric in evaluator.metrics:
                    metrics.update(metric.gather_results())
                with open(metrics_path / f"{config.task.name}_{idx}_{epi}.json", "w") as f:
                    json.dump(metrics, f)

                # reset video writer
                if config.write_video:
                    evaluator.video_writer = None
                    logger.info(f"Saved preview video to {video_name}")
