"""
Ground truth HDF5 replay evaluator for embodiedClaw.

Replays recorded episodes through the simulator frame-by-frame using
``og.sim.load_state()``, pausing at configurable intervals for agent
observation via MCP.

This module provides:
  - ``DummyPolicy``: a no-op policy returning zero actions
  - ``ReplayEvaluator``: follows the ``AgenticEvaluator`` pattern for
    env/robot setup, but replays HDF5 states instead of using a VLA

Usage::

    cd /home/user/codebase/BEHAVIOR-1K
    DISPLAY=:1 OMNI_KIT_ACCEPT_EULA=yes conda run -n behavior python \\
        OmniGibson/omnigibson/learning/embodiedClaw/gt_replay_runner.py \\
        policy=local task.name=picking_up_trash headless=true \\
        log_path=/tmp/gt_replay \\
        +hdf5_path=/home/user/dataset/raw/task-0001/episode_00010010.hdf5 \\
        +observation_interval=25
"""

import h5py
import hydra
import json
import logging
import numpy as np
import omnigibson as og
import os
import re
import sys
import threading
import torch as th

from inspect import getsourcefile
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from omnigibson.learning.utils.config_utils import register_omegaconf_resolvers
from omnigibson.learning.utils.eval_utils import (
    ROBOT_CAMERA_NAMES,
    TASK_NAMES_TO_INDICES,
)
from omnigibson.macros import gm, create_module_macros, macros

from omnigibson.learning.embodiedClaw.annotation_loader import (
    EpisodeAnnotation,
    load_episode_annotation,
    load_all_annotations,
    compute_subtask_duration_stats,
)
from omnigibson.learning.embodiedClaw.mcp_server import EmbodiedClawMCPServer
from omnigibson.learning.embodiedClaw.tools.control_tools import VLAPolicyController
from omnigibson.learning.embodiedClaw.tools.information_tools import (
    get_simulation_snapshot,
)

m = create_module_macros(module_path=__file__)
m.NUM_EVAL_EPISODES = 1
m.NUM_TRAIN_INSTANCES = 200
m.NUM_EVAL_INSTANCES = 20

# Performance settings (matching AgenticEvaluator)
gm.ENABLE_FLATCACHE = True
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = True

with macros.unlocked():
    macros.robots.manipulation_robot.GRASP_WINDOW = 0.75

logger = logging.getLogger("gt_replay_runner")
logger.setLevel(logging.INFO)


# ======================================================================
# Dummy policy
# ======================================================================


class DummyPolicy:
    """No-op policy that returns zero actions.

    Used by ReplayEvaluator because the replay drives simulator state
    from HDF5 data — no VLA is needed.
    """

    def forward(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Return an empty action dict (never used)."""
        return {}

    def reset(self) -> None:
        """No-op reset."""
        pass


# ======================================================================
# HDF5 utilities
# ======================================================================


def extract_episode_id(hdf5_path: str) -> str:
    """Extract the 8-digit episode ID from an HDF5 filename.

    Args:
        hdf5_path: Path like ``".../episode_00010010.hdf5"``.

    Returns:
        The episode ID string, e.g. ``"00010010"``.

    Raises:
        ValueError: If the filename doesn't match the expected pattern.
    """
    basename = os.path.basename(hdf5_path)
    match = re.match(r"episode_(\d{8})\.hdf5", basename)
    if not match:
        raise ValueError(
            f"Cannot extract episode ID from filename: {basename}. "
            f"Expected format: episode_XXXXXXXX.hdf5"
        )
    return match.group(1)


def extract_task_id_from_episode(episode_id: str) -> int:
    """Extract the task index from an episode ID.

    Episode IDs encode the task as ``TTTT____`` where TTTT is the
    zero-padded task index (e.g. ``00010010`` -> task 1).

    Args:
        episode_id: 8-digit episode ID string.

    Returns:
        The integer task index.
    """
    return int(episode_id[:4])


def load_hdf5_episode(hdf5_path: str, demo_id: int = 0) -> Dict[str, Any]:
    """Load state and action arrays from an HDF5 episode file.

    Args:
        hdf5_path: Path to the HDF5 file.
        demo_id: Demo index within the file (default 0).

    Returns:
        Dict with keys:
            - ``"state"``: numpy array of shape (N+1, state_dim)
            - ``"state_size"``: numpy array of shape (N+1,)
            - ``"action"``: numpy array of shape (N, action_dim)
            - ``"num_frames"``: int — number of action frames (N)
            - ``"transitions"``: dict of transition info per frame
    """
    with h5py.File(hdf5_path, "r") as f:
        grp_name = f"data/demo_{demo_id}"
        if grp_name not in f:
            raise KeyError(f"No group '{grp_name}' in {hdf5_path}")
        grp = f[grp_name]

        state = np.array(grp["state"])
        state_size = np.array(grp["state_size"])
        action = np.array(grp["action"])

        # Transitions (object additions/removals at specific frames)
        transitions_str = grp.attrs.get("transitions", "{}")
        if isinstance(transitions_str, bytes):
            transitions_str = transitions_str.decode("utf-8")
        transitions = json.loads(transitions_str) if transitions_str else {}

    return {
        "state": state,
        "state_size": state_size,
        "action": action,
        "num_frames": action.shape[0],
        "transitions": transitions,
    }


def find_annotation_for_episode(
    episode_id: str,
    dataset_root: str = "/home/user/dataset",
) -> Optional[str]:
    """Find the annotation JSON path for a given episode ID.

    Args:
        episode_id: 8-digit episode ID string (e.g. ``"00010010"``).
        dataset_root: Root of the dataset directory.

    Returns:
        Absolute path to the annotation file, or None if not found.
    """
    task_id = extract_task_id_from_episode(episode_id)
    annotation_dir = os.path.join(
        dataset_root, "annotations", f"task-{task_id:04d}"
    )
    annotation_file = os.path.join(
        annotation_dir, f"episode_{episode_id}.json"
    )
    if os.path.isfile(annotation_file):
        return annotation_file
    return None


# ======================================================================
# ReplayEvaluator
# ======================================================================


class ReplayEvaluator:
    """Replays HDF5 episodes through the simulator for decision accuracy testing.

    Follows the ``AgenticEvaluator`` pattern from ``embodied_claw_sim_run.py``
    for environment/robot setup, but drives the simulator by loading HDF5
    states frame-by-frame instead of running a VLA policy.

    The evaluator:
      1. Sets up the OmniGibson environment identically to AgenticEvaluator.
      2. Loads HDF5 state data for the episode.
      3. Loads states at observation-interval frames (not every frame).
      4. At each observation point, builds a simulation snapshot.
      5. Optionally pauses for MCP-connected agents to observe.

    Attributes:
        cfg: Hydra configuration.
        env: The OmniGibson environment.
        robot: The robot entity.
        controller: VLAPolicyController wrapping a DummyPolicy.
        hdf5_path: Path to the HDF5 file being replayed.
        observation_interval: How often (in frames) to pause for observation.
    """

    def __init__(
        self,
        cfg: DictConfig,
        hdf5_path: str,
        observation_interval: int = 25,
    ) -> None:
        self.cfg = cfg
        self.hdf5_path = hdf5_path
        self.observation_interval = observation_interval

        # Extract episode info
        self.episode_id = extract_episode_id(hdf5_path)
        self.task_id = extract_task_id_from_episode(self.episode_id)

        # Load HDF5 data
        logger.info("Loading HDF5 data from %s", hdf5_path)
        self.hdf5_data = load_hdf5_episode(hdf5_path)
        self.num_frames = self.hdf5_data["num_frames"]
        logger.info(
            "Episode %s: %d frames, state shape %s",
            self.episode_id,
            self.num_frames,
            self.hdf5_data["state"].shape,
        )

        # Use AgenticEvaluator's env setup by importing its pattern
        # We replicate the critical setup code directly here to avoid
        # needing a real VLA policy.
        from omnigibson.learning.embodiedClaw.embodied_claw_sim_run import (
            AgenticEvaluator,
        )

        # Temporarily swap out load_policy to use DummyPolicy
        original_load_policy = AgenticEvaluator.load_policy

        def dummy_load_policy(self_eval):
            return DummyPolicy()

        AgenticEvaluator.load_policy = dummy_load_policy
        try:
            self._evaluator = AgenticEvaluator(cfg)
        finally:
            AgenticEvaluator.load_policy = original_load_policy

        self.env = self._evaluator.env
        self.robot = self._evaluator.robot
        self.controller = self._evaluator.controller

        # Current replay position
        self._current_frame = 0
        self._obs = self._evaluator.obs

    def load_task_instance_for_replay(self) -> None:
        """Load the task instance matching the HDF5 episode.

        Calls ``load_task_instance`` on the underlying evaluator and then
        restores the initial HDF5 state so the scene matches the recording.
        """
        # The evaluator's load_task_instance sets up the BDDL scope
        # We need the BDDL scope to be populated for snapshot generation
        # Use instance 0 for seed setup (the HDF5 state will override)
        self._evaluator.load_task_instance(0)

        # Now restore the initial HDF5 state
        state = self.hdf5_data["state"]
        state_size = self.hdf5_data["state_size"]
        logger.info("Loading initial HDF5 state (frame 0)...")
        og.sim.load_state(
            th.tensor(state[0, :int(state_size[0])], dtype=th.float32),
            serialized=True,
        )
        self._current_frame = 0

    def load_state_at_frame(self, frame_idx: int) -> None:
        """Load simulator state from the HDF5 at the specified frame.

        Args:
            frame_idx: The frame index to load (0-based). Frame 0 is the
                initial state; frames 1..N correspond to states after
                actions 0..N-1.

        Raises:
            IndexError: If frame_idx is out of range.
        """
        state = self.hdf5_data["state"]
        state_size = self.hdf5_data["state_size"]

        if frame_idx < 0 or frame_idx >= state.shape[0]:
            raise IndexError(
                f"Frame index {frame_idx} out of range "
                f"[0, {state.shape[0]})"
            )

        # Handle transitions at this frame (object additions/removals)
        transitions = self.hdf5_data["transitions"]
        frame_key = str(frame_idx)
        if frame_key in transitions:
            self._apply_transitions(transitions[frame_key])

        og.sim.load_state(
            th.tensor(
                state[frame_idx, :int(state_size[frame_idx])],
                dtype=th.float32,
            ),
            serialized=True,
        )
        self._current_frame = frame_idx

    def _apply_transitions(self, transition_info: Dict[str, Any]) -> None:
        """Apply object/system transitions from HDF5 metadata.

        Follows the same pattern as DataPlaybackWrapper.playback_episode().

        Args:
            transition_info: Dict with "systems" and "objects" keys describing
                what to add/remove.
        """
        try:
            from omnigibson.utils.python_utils import create_object_from_init_info

            scene = og.sim.scenes[0]

            # Systems
            for add_sys_name in transition_info.get("systems", {}).get("add", []):
                scene.get_system(add_sys_name, force_init=True)
            for remove_sys_name in transition_info.get("systems", {}).get("remove", []):
                scene.clear_system(remove_sys_name)

            # Objects
            for remove_obj_name in transition_info.get("objects", {}).get("remove", []):
                obj = scene.object_registry("name", remove_obj_name)
                scene.remove_object(obj)
            for j, add_obj_info in enumerate(
                transition_info.get("objects", {}).get("add", [])
            ):
                obj = create_object_from_init_info(add_obj_info)
                scene.add_object(obj)
                obj.set_position(th.ones(3) * 100.0 + th.ones(3) * 5 * j)

            og.sim.step()
        except Exception as e:
            logger.warning("Failed to apply transitions at frame: %s", e)

    def get_snapshot_at_current_frame(self) -> Dict[str, Any]:
        """Build a simulation snapshot at the current loaded frame.

        Returns:
            Snapshot dict from ``get_simulation_snapshot``.
        """
        # Step physics once to propagate the loaded state
        # (needed for sensor data, camera renders, etc.)
        action = self.hdf5_data["action"]
        if self._current_frame < action.shape[0]:
            a = th.tensor(action[self._current_frame], dtype=th.float32)
            obs, _, _, _, _ = self.env.step(a, n_render_iterations=1)
        else:
            # Past the last action — just step with zeros
            zero_action = th.zeros(action.shape[1], dtype=th.float32)
            obs, _, _, _, _ = self.env.step(zero_action, n_render_iterations=1)

        # Preprocess obs (same as AgenticEvaluator)
        obs = self._evaluator._preprocess_obs(obs)

        snapshot = get_simulation_snapshot(
            self.env,
            self.robot,
            self.cfg.task.name,
            obs=obs,
        )
        # Override step_count to reflect the replay frame
        snapshot["step_count"] = self._current_frame
        return snapshot

    def replay_with_observations(
        self,
        callback=None,
        mcp_mode: bool = False,
    ) -> List[Dict[str, Any]]:
        """Replay the episode, collecting snapshots at observation intervals.

        Args:
            callback: Optional callable ``(frame_idx, snapshot) -> None``
                invoked at each observation point.
            mcp_mode: If True, pause the controller at each observation
                point so an MCP-connected agent can observe.

        Returns:
            List of snapshot dicts, one per observation point.
        """
        snapshots = []

        # Determine observation frames
        obs_frames = list(
            range(0, self.num_frames + 1, self.observation_interval)
        )
        # Ensure last frame is included
        if obs_frames[-1] != self.num_frames:
            obs_frames.append(self.num_frames)

        logger.info(
            "Replaying episode %s: %d frames, %d observation points",
            self.episode_id,
            self.num_frames,
            len(obs_frames),
        )

        for i, frame_idx in enumerate(obs_frames):
            # Clamp to valid state index range
            state_idx = min(frame_idx, self.hdf5_data["state"].shape[0] - 1)

            logger.debug("Loading state at frame %d", state_idx)
            self.load_state_at_frame(state_idx)

            # Build snapshot
            snapshot = self.get_snapshot_at_current_frame()

            if callback is not None:
                callback(frame_idx, snapshot)

            snapshots.append(snapshot)

            if mcp_mode:
                # Pause so MCP agent can observe
                logger.info(
                    "Observation point %d/%d at frame %d — pausing for MCP agent",
                    i + 1, len(obs_frames), frame_idx,
                )
                self.controller._run_event.clear()
                # Cache the snapshot for MCP reads
                self.controller._cached_snapshot = snapshot
                # Block until the agent resumes
                self.controller._run_event.wait()
                self.controller._cached_snapshot = None

        logger.info("Replay complete. %d snapshots collected.", len(snapshots))
        return snapshots

    def close(self) -> None:
        """Clean up the environment and simulator."""
        try:
            self.env.close()
            og.shutdown()
        except Exception as e:
            logger.warning("Error during cleanup: %s", e)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close()
        if exc_type is not None:
            import traceback
            traceback.print_exception(exc_type, exc_value, exc_tb)


# ======================================================================
# Entry point (MCP mode)
# ======================================================================


def run_automated_accuracy_test(config: DictConfig) -> None:
    """Run the automated GT replay decision accuracy test (no MCP needed).

    Starts the simulator, loads the HDF5 episode, loads annotations,
    and runs ``run_gt_replay_accuracy_test`` to measure decision accuracy
    against ground truth subtask boundaries.

    Results are printed and saved to ``/tmp/gt_live_results.json``.
    """
    from omnigibson.learning.embodiedClaw.gt_replay_test import (
        run_gt_replay_accuracy_test,
        compute_accuracy_metrics,
    )

    hdf5_path = getattr(config, "hdf5_path", None)
    if hdf5_path is None:
        logger.error("Missing required arg: +hdf5_path=<path>")
        sys.exit(1)

    observation_interval = int(getattr(config, "observation_interval", 25))
    dataset_root = getattr(config, "dataset_root", None) or "/home/user/dataset"
    output_path = getattr(config, "output_path", None) or "/tmp/gt_live_results.json"

    with ReplayEvaluator(config, hdf5_path, observation_interval) as evaluator:
        # Load task instance and initial state
        evaluator.load_task_instance_for_replay()

        # Load annotation for this episode
        episode_id = evaluator.episode_id
        annotation_path = find_annotation_for_episode(episode_id, dataset_root)
        if annotation_path is None:
            logger.error("No annotation found for episode %s", episode_id)
            sys.exit(1)

        logger.info("Loading annotation from %s", annotation_path)
        annotation = load_episode_annotation(annotation_path)

        # Load duration stats from all annotations in the task
        task_id = extract_task_id_from_episode(episode_id)
        annotation_dir = os.path.join(
            dataset_root, "annotations", f"task-{task_id:04d}"
        )
        all_annotations = load_all_annotations(annotation_dir)
        duration_stats = compute_subtask_duration_stats(all_annotations)

        logger.info("")
        logger.info("=" * 60)
        logger.info("  GT Replay Automated Accuracy Test")
        logger.info("  Episode: %s (%d frames)", episode_id, evaluator.num_frames)
        logger.info("  Subtasks: %d", len(annotation.subtasks))
        logger.info("  Observation interval: %d frames", observation_interval)
        logger.info("=" * 60)
        logger.info("")

        # Run the accuracy test
        result = run_gt_replay_accuracy_test(
            replay_evaluator=evaluator,
            annotation=annotation,
            duration_stats=duration_stats,
            episode_id=episode_id,
            observation_interval=observation_interval,
        )

        # Save results
        output = result.to_dict()
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info("Results written to %s", output_path)

        # Print summary
        agg = output.get("accuracy", {})
        print("\n" + "=" * 60)
        print("  GT Replay Live Accuracy Test -- Results")
        print("=" * 60)
        print(f"  Episode:              {output.get('episode_id')}")
        print(f"  Task:                 {output.get('task_name')}")
        print(f"  Total frames:         {output.get('total_frames')}")
        print(f"  Total subtasks:       {agg.get('total_subtasks', 0)}")
        print(f"  Decided:              {agg.get('total_decided', 0)}")
        print(f"  Missed:               {agg.get('total_missed', 0)}")
        print(f"  Accuracy @ +/-50:     {agg.get('accuracy_at_50', 0):.1%}")
        print(f"  Accuracy @ +/-100:    {agg.get('accuracy_at_100', 0):.1%}")
        print(f"  Accuracy @ +/-200:    {agg.get('accuracy_at_200', 0):.1%}")
        print(f"  Accuracy @ +/-500:    {agg.get('accuracy_at_500', 0):.1%}")
        print(f"  Mean |error|:         {agg.get('mean_abs_error', 0):.1f} frames")
        print(f"  Median |error|:       {agg.get('median_abs_error', 0):.1f} frames")
        print(f"  Mean error (signed):  {agg.get('mean_error', 0):.1f} frames")
        print("=" * 60)

        # Print per-subtask details
        for sr in output.get("subtask_results", []):
            status = "DECIDED" if sr["agent_decision_frame"] is not None else "MISSED"
            err_str = f"error={sr['frame_error']}" if sr["frame_error"] is not None else "no decision"
            print(
                f"  Skill {sr['skill_idx']:2d}: {sr['skill_description']:<20s} "
                f"gt_end={sr['gt_end_frame']:5d}  "
                f"agent={str(sr['agent_decision_frame']):>5s}  "
                f"{err_str:>15s}  [{status}]"
            )
        print("=" * 60)


def main() -> None:
    """Entry point for running the GT replay.

    Supports two modes:
      - ``+mode=test``: Automated accuracy test (no MCP needed).
      - Default: MCP mode (pauses at observation points for agent).

    Starts the simulator, loads the HDF5 episode, and either runs the
    automated test or starts the MCP server for interactive observation.
    """
    register_omegaconf_resolvers()
    configs_dir = f"{Path(getsourcefile(lambda: 0)).parents[0].parent}/configs"
    with hydra.initialize_config_dir(configs_dir, version_base="1.1"):
        config = hydra.compose("base_config.yaml", overrides=sys.argv[1:])
    OmegaConf.resolve(config)

    gm.HEADLESS = config.headless

    # Check mode
    mode = getattr(config, "mode", None) or "mcp"

    if mode == "test":
        run_automated_accuracy_test(config)
        return

    # Parse extra args
    hdf5_path = getattr(config, "hdf5_path", None)
    if hdf5_path is None:
        logger.error("Missing required arg: +hdf5_path=<path>")
        sys.exit(1)

    observation_interval = int(getattr(config, "observation_interval", 25))

    # MCP server settings
    mcp_host = getattr(config, "mcp_host", None) or "127.0.0.1"
    mcp_port = getattr(config, "mcp_port", None) or 8000
    dataset_root = getattr(config, "dataset_root", None) or "/home/user/dataset"

    with ReplayEvaluator(config, hdf5_path, observation_interval) as evaluator:
        # Load task instance and initial state
        evaluator.load_task_instance_for_replay()

        task_id = TASK_NAMES_TO_INDICES[config.task.name]

        # Start MCP server
        mcp_server = EmbodiedClawMCPServer(
            env=evaluator.env,
            robot=evaluator.robot,
            task_name=config.task.name,
            controller=evaluator.controller,
            checkpoint_mgr=None,
            host=mcp_host,
            port=mcp_port,
            dataset_root=dataset_root,
            task_id=task_id,
        )

        mcp_thread = threading.Thread(
            target=mcp_server.run_sse,
            name="mcp-sse-server",
            daemon=True,
        )
        mcp_thread.start()

        logger.info("")
        logger.info("=" * 60)
        logger.info("  GT Replay MCP server at http://%s:%s/sse", mcp_host, mcp_port)
        logger.info("  Episode: %s (%d frames)", evaluator.episode_id, evaluator.num_frames)
        logger.info("  Observation interval: %d frames", observation_interval)
        logger.info("=" * 60)
        logger.info("")

        # Replay with MCP pausing
        evaluator.replay_with_observations(mcp_mode=True)

        logger.info("GT replay complete.")


if __name__ == "__main__":
    main()
