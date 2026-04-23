#!/usr/bin/env python3
"""
Comprehensive test suite for the embodiedClaw agentic data collection system.

Since OmniGibson, torch, cv2, av, etc. are not available in this environment,
we mock all external dependencies and test the logic in isolation.

Run with:
    python3 test_embodied_claw.py
"""

import importlib
import importlib.util
import json
import os
import sys
import tempfile
import threading
import time
import traceback
import types
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

# ======================================================================
# MOCK SETUP: Install mock modules before any embodiedClaw imports
# ======================================================================

_REPO_ROOT = "/home/user/codebase/B1K-DataGen/BEHAVIOR-1K/OmniGibson"
_ECLAW_ROOT = os.path.join(_REPO_ROOT, "omnigibson", "learning", "embodiedClaw")

# Add the repo to sys.path so that `omnigibson.*` resolves to real packages
# where possible, but we pre-install mocks for things that can't be imported.
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def install_mock_modules():
    """Install mock modules for all missing external dependencies."""

    # --- torch ---
    torch_mock = types.ModuleType("torch")
    torch_mock.__path__ = []

    class FakeTensor:
        """Minimal mock for torch.Tensor."""
        def __init__(self, data=None):
            self._data = data if data is not None else [0.0]

        def cpu(self):
            return self

        def numpy(self):
            import numpy as np
            return np.array(self._data)

        def tolist(self):
            return list(self._data)

        def __len__(self):
            return len(self._data)

    torch_mock.Tensor = FakeTensor
    torch_mock.tensor = lambda data, **kw: FakeTensor(data)
    torch_mock.cat = lambda tensors, **kw: FakeTensor([0.0] * sum(len(t) for t in tensors))
    torch_mock.float32 = "float32"
    torch_mock.int64 = "int64"

    sys.modules["torch"] = torch_mock

    # --- cv2 ---
    cv2_mock = types.ModuleType("cv2")
    cv2_mock.imencode = lambda fmt, img: (True, MagicMock(tobytes=lambda: b"\x89PNG"))
    cv2_mock.COLOR_RGB2BGR = 4
    cv2_mock.cvtColor = lambda img, code: img
    sys.modules["cv2"] = cv2_mock

    # --- av ---
    av_mock = types.ModuleType("av")
    av_mock.__path__ = []

    av_container = types.ModuleType("av.container")
    av_container.OutputContainer = type("OutputContainer", (), {})
    sys.modules["av.container"] = av_container

    av_stream = types.ModuleType("av.stream")
    av_stream.Stream = type("Stream", (), {})
    sys.modules["av.stream"] = av_stream

    sys.modules["av"] = av_mock

    # --- omnigibson top-level package ---
    # We need to install a *real-ish* omnigibson package that lets Python
    # resolve sub-packages from the file system, but also has the mock attrs
    # that modules expect (e.g., og.sim).
    #
    # Strategy: create omnigibson as a namespace-style module with __path__
    # pointing at the real omnigibson directory, so that sub-packages like
    # omnigibson.learning.embodiedClaw are found naturally, while also
    # installing mocks for modules that require Isaac Sim (object_states, etc.).

    og_pkg_dir = os.path.join(_REPO_ROOT, "omnigibson")

    og_mock = types.ModuleType("omnigibson")
    og_mock.__path__ = [og_pkg_dir]
    og_mock.__package__ = "omnigibson"
    og_mock.sim = MagicMock()
    sys.modules["omnigibson"] = og_mock

    # omnigibson.learning — real package path
    learning_dir = os.path.join(og_pkg_dir, "learning")
    learning_mock = types.ModuleType("omnigibson.learning")
    learning_mock.__path__ = [learning_dir]
    learning_mock.__package__ = "omnigibson.learning"
    sys.modules["omnigibson.learning"] = learning_mock

    # omnigibson.learning.embodiedClaw — real package path
    eclaw_mock = types.ModuleType("omnigibson.learning.embodiedClaw")
    eclaw_mock.__path__ = [_ECLAW_ROOT]
    eclaw_mock.__package__ = "omnigibson.learning.embodiedClaw"
    sys.modules["omnigibson.learning.embodiedClaw"] = eclaw_mock

    # omnigibson.learning.embodiedClaw.tools
    tools_dir = os.path.join(_ECLAW_ROOT, "tools")
    tools_mock = types.ModuleType("omnigibson.learning.embodiedClaw.tools")
    tools_mock.__path__ = [tools_dir]
    tools_mock.__package__ = "omnigibson.learning.embodiedClaw.tools"
    sys.modules["omnigibson.learning.embodiedClaw.tools"] = tools_mock

    # omnigibson.learning.embodiedClaw.data_recording
    dr_dir = os.path.join(_ECLAW_ROOT, "data_recording")
    dr_mock = types.ModuleType("omnigibson.learning.embodiedClaw.data_recording")
    dr_mock.__path__ = [dr_dir]
    dr_mock.__package__ = "omnigibson.learning.embodiedClaw.data_recording"
    sys.modules["omnigibson.learning.embodiedClaw.data_recording"] = dr_mock

    # omnigibson.learning.utils — mock with needed symbols
    utils_dir = os.path.join(learning_dir, "utils")
    lu_mock = types.ModuleType("omnigibson.learning.utils")
    lu_mock.__path__ = [utils_dir]
    lu_mock.__package__ = "omnigibson.learning.utils"
    sys.modules["omnigibson.learning.utils"] = lu_mock

    # --- omnigibson.object_states (mock with sentinel classes) ---
    obj_states = types.ModuleType("omnigibson.object_states")
    obj_states.__path__ = []
    for state_name in [
        "Cooked", "Frozen", "Inside", "NextTo", "OnTop",
        "Open", "ToggledOn", "Touching", "Under",
    ]:
        setattr(obj_states, state_name, type(state_name, (), {}))
    sys.modules["omnigibson.object_states"] = obj_states

    obj_state_base = types.ModuleType("omnigibson.object_states.object_state_base")
    obj_state_base.AbsoluteObjectState = type("AbsoluteObjectState", (), {})
    obj_state_base.RelativeObjectState = type("RelativeObjectState", (), {})
    sys.modules["omnigibson.object_states.object_state_base"] = obj_state_base

    # --- omnigibson.learning.utils.eval_utils ---
    eval_utils = types.ModuleType("omnigibson.learning.utils.eval_utils")
    eval_utils.ROBOT_CAMERA_NAMES = {
        "R1Pro": {
            "head": "robot_r1::robot_r1:zed_link:Camera:0",
            "left_wrist": "robot_r1::robot_r1:left_wrist_link:Camera:0",
            "right_wrist": "robot_r1::robot_r1:right_wrist_link:Camera:0",
        }
    }
    eval_utils.PROPRIOCEPTION_INDICES = {"R1Pro": {"joint_pos": 0}}
    eval_utils.HEAD_RESOLUTION = (480, 640)
    eval_utils.WRIST_RESOLUTION = (240, 320)
    eval_utils.TASK_NAMES_TO_INDICES = {"test_task": 0}
    eval_utils.generate_basic_environment_config = MagicMock()
    eval_utils.flatten_obs_dict = MagicMock(return_value={})
    sys.modules["omnigibson.learning.utils.eval_utils"] = eval_utils

    # --- omnigibson.learning.utils.dataset_utils ---
    dataset_utils = types.ModuleType("omnigibson.learning.utils.dataset_utils")
    dataset_utils.makedirs_with_mode = lambda path, **kw: os.makedirs(path, exist_ok=True)
    sys.modules["omnigibson.learning.utils.dataset_utils"] = dataset_utils

    # --- omnigibson.learning.utils.obs_utils ---
    obs_utils = types.ModuleType("omnigibson.learning.utils.obs_utils")
    def mock_create_video_writer(fpath, resolution, codec_name, rate, pix_fmt):
        container = MagicMock()
        stream = MagicMock()
        stream.encode.return_value = []
        return (container, stream)
    obs_utils.create_video_writer = mock_create_video_writer
    obs_utils.write_video = MagicMock()
    sys.modules["omnigibson.learning.utils.obs_utils"] = obs_utils

    # --- Other omnigibson sub-modules that might be imported ---
    for mod_name in [
        "omnigibson.learning.utils.config_utils",
        "omnigibson.envs",
        "omnigibson.envs.env_wrapper",
        "omnigibson.macros",
        "omnigibson.metrics",
        "omnigibson.robots",
        "omnigibson.utils",
        "omnigibson.utils.asset_utils",
        "omnigibson.utils.python_utils",
        "omnigibson.utils.transform_utils",
    ]:
        m = types.ModuleType(mod_name)
        m.__path__ = []
        sys.modules[mod_name] = m

    # Populate stubs
    env_wrapper = sys.modules["omnigibson.envs.env_wrapper"]
    env_wrapper.EnvironmentWrapper = type("EnvironmentWrapper", (), {})

    macros_mod = sys.modules["omnigibson.macros"]
    macros_mod.gm = MagicMock()
    macros_mod.create_module_macros = MagicMock(return_value=MagicMock())
    macros_mod.macros = MagicMock()

    metrics_mod = sys.modules["omnigibson.metrics"]
    metrics_mod.MetricBase = type("MetricBase", (), {})
    metrics_mod.AgentMetric = MagicMock()
    metrics_mod.TaskMetric = MagicMock()

    robots_mod = sys.modules["omnigibson.robots"]
    robots_mod.BaseRobot = type("BaseRobot", (), {})

    asset_utils = sys.modules["omnigibson.utils.asset_utils"]
    asset_utils.get_task_instance_path = MagicMock()

    python_utils = sys.modules["omnigibson.utils.python_utils"]
    python_utils.recursively_convert_to_torch = MagicMock()

    config_utils = sys.modules["omnigibson.learning.utils.config_utils"]
    config_utils.register_omegaconf_resolvers = MagicMock()

    # --- gello (for embodied_claw_sim_run.py) ---
    for mod_name in [
        "gello",
        "gello.robots",
        "gello.robots.sim_robot",
        "gello.robots.sim_robot.og_teleop_utils",
        "gello.robots.sim_robot.og_teleop_cfg",
    ]:
        m = types.ModuleType(mod_name)
        m.__path__ = []
        sys.modules[mod_name] = m

    gello_utils = sys.modules["gello.robots.sim_robot.og_teleop_utils"]
    gello_utils.augment_rooms = MagicMock()
    gello_utils.load_available_tasks = MagicMock()
    gello_utils.generate_robot_config = MagicMock()
    gello_utils.get_task_relevant_room_types = MagicMock()

    gello_cfg = sys.modules["gello.robots.sim_robot.og_teleop_cfg"]
    gello_cfg.DISABLED_TRANSITION_RULES = []

    # --- hydra / omegaconf ---
    for mod_name in ["hydra", "hydra.utils", "omegaconf"]:
        if mod_name not in sys.modules:
            m = types.ModuleType(mod_name)
            m.__path__ = []
            sys.modules[mod_name] = m

    hydra_mod = sys.modules["hydra"]
    hydra_mod.initialize_config_dir = MagicMock()
    hydra_mod.compose = MagicMock()

    hydra_utils = sys.modules["hydra.utils"]
    hydra_utils.instantiate = MagicMock()

    omegaconf = sys.modules["omegaconf"]
    omegaconf.DictConfig = dict
    omegaconf.OmegaConf = MagicMock()


# Install mocks before importing anything from embodiedClaw
install_mock_modules()


# ======================================================================
# Now import the actual modules under test
# ======================================================================

import numpy as np

from omnigibson.learning.embodiedClaw.annotation_loader import (
    EpisodeAnnotation,
    SubtaskAnnotation,
    SubtaskDurationStats,
    get_rollout_subtask_object_refs,
)
from omnigibson.learning.embodiedClaw.decision_module import (
    check_grasp_success,
    check_navigation_success,
    check_placement_success,
    evaluate_subtask_status,
)
from omnigibson.learning.embodiedClaw.memory import (
    TaskMemory,
)
from omnigibson.learning.embodiedClaw.object_matching import (
    arg_matches_query,
    build_match_query,
    entity_matches_query,
    name_matches_query,
)
from omnigibson.learning.embodiedClaw.tools.control_tools import (
    VLAPolicyController,
    pause_vla,
    resume_vla,
    reset_vla,
    switch_vla,
)
from omnigibson.learning.embodiedClaw.tools.annotation_tools import (
    rebase_retry_start_step,
)
from omnigibson.learning.embodiedClaw.tools.information_tools import (
    get_simulation_snapshot,
    filter_information,
)
from omnigibson.learning.embodiedClaw.data_recording.bddl_state_tracker import BDDLStateTracker
from omnigibson.learning.embodiedClaw.data_recording.data_saver import DataRecorder
from omnigibson.learning.embodiedClaw.tools.data_tools import (
    CheckpointManager,
    save_success_data,
    save_failure_data,
    _segment_output_folder,
    _create_segment_recorder,
    _stabilise_scene,
)


# ======================================================================
# Helper factory functions for mock objects
# ======================================================================

def make_mock_robot():
    """Create a mock robot with arm_names, eef, position, joint positions, grasp info."""
    robot = MagicMock()
    robot.arm_names = ["left", "right"]
    robot.get_position_orientation.return_value = (
        np.array([1.0, 2.0, 3.0]),
        np.array([0.0, 0.0, 0.0, 1.0]),
    )
    robot.get_joint_positions.return_value = np.array([0.1, 0.2, 0.3])
    robot.get_eef_position.return_value = np.array([0.5, 0.6, 0.7])
    robot.get_eef_orientation.return_value = np.array([0.0, 0.0, 0.0, 1.0])
    robot._ag_obj_in_hand = {"left": None, "right": None}
    return robot


def make_mock_env():
    """Create a mock env with task, object_scope, ground_goal_state_options."""
    env = MagicMock()
    env._current_step = 42

    # Goal heads
    head = MagicMock()
    head.body = ["OnTop", "apple.n.01_1", "table.n.01_1"]
    head.evaluate.return_value = False

    task = MagicMock()
    task.ground_goal_state_options = [[head]]

    # Object scope
    entity_apple = MagicMock()
    entity_apple.is_system = False
    entity_apple.exists = True
    entity_apple.wrapped_obj = MagicMock()
    entity_apple.wrapped_obj.get_position_orientation.return_value = (
        np.array([4.0, 5.0, 6.0]),
        np.array([0.0, 0.0, 0.0, 1.0]),
    )
    entity_apple.wrapped_obj.states = {}
    entity_apple.wrapped_obj.name = "apple_obj"

    entity_table = MagicMock()
    entity_table.is_system = False
    entity_table.exists = True
    entity_table.wrapped_obj = MagicMock()
    entity_table.wrapped_obj.get_position_orientation.return_value = (
        np.array([7.0, 8.0, 9.0]),
        np.array([0.0, 0.0, 0.0, 1.0]),
    )
    entity_table.wrapped_obj.states = {}
    entity_table.wrapped_obj.name = "table_obj"

    task.object_scope = {
        "apple.n.01_1": entity_apple,
        "table.n.01_1": entity_table,
        "agent.n.01_1": MagicMock(),  # should be skipped
    }

    env.task = task
    env.scene = MagicMock()
    env.scene.dump_state.return_value = {"fake": "state"}
    return env


def make_mock_policy():
    """Create a mock VLA policy."""
    policy = MagicMock()
    policy.forward.return_value = np.zeros(10)
    policy.reset.return_value = None
    return policy


# ======================================================================
# Test Results Accumulator
# ======================================================================

class TestResults:
    """Accumulator for test results."""
    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []

    def record_pass(self, name, detail=""):
        self.passed.append((name, detail))

    def record_fail(self, name, error, suggestion=""):
        self.failed.append((name, str(error), suggestion))

    def record_warning(self, name, detail):
        self.warnings.append((name, detail))

    def summary(self):
        total = len(self.passed) + len(self.failed)
        lines = []
        lines.append("")
        lines.append("=" * 70)
        lines.append(f"TEST RESULTS: {len(self.passed)}/{total} passed, {len(self.failed)} failed, {len(self.warnings)} warnings")
        lines.append("=" * 70)

        if self.passed:
            lines.append("")
            lines.append("PASSED:")
            for name, detail in self.passed:
                d = f" -- {detail}" if detail else ""
                lines.append(f"  [PASS] {name}{d}")

        if self.warnings:
            lines.append("")
            lines.append("WARNINGS:")
            for name, detail in self.warnings:
                lines.append(f"  [WARN] {name}: {detail}")

        if self.failed:
            lines.append("")
            lines.append("FAILURES:")
            for name, error, suggestion in self.failed:
                lines.append(f"  [FAIL] {name}")
                lines.append(f"         Error: {error}")
                if suggestion:
                    lines.append(f"         Fix: {suggestion}")

        lines.append("")
        return "\n".join(lines)


results = TestResults()


# ======================================================================
# Test 1: Import verification
# ======================================================================

def test_1_imports():
    """Test 1: All public symbols importable"""
    test_name = "Test 1: Import verification"
    try:
        assert callable(get_simulation_snapshot), "get_simulation_snapshot is not callable"
        assert callable(filter_information), "filter_information is not callable"
        assert isinstance(VLAPolicyController, type), "VLAPolicyController is not a class"
        assert callable(pause_vla), "pause_vla is not callable"
        assert callable(resume_vla), "resume_vla is not callable"
        assert callable(reset_vla), "reset_vla is not callable"
        assert callable(switch_vla), "switch_vla is not callable"
        assert isinstance(CheckpointManager, type), "CheckpointManager is not a class"
        assert callable(save_success_data), "save_success_data is not callable"
        assert callable(save_failure_data), "save_failure_data is not callable"
        assert isinstance(BDDLStateTracker, type), "BDDLStateTracker is not a class"
        assert isinstance(DataRecorder, type), "DataRecorder is not a class"
        results.record_pass(test_name, "All 12 public symbols imported successfully")
    except Exception as e:
        results.record_fail(test_name, e)


def test_1b_package_init_imports():
    """Test 1b: Package __init__.py re-exports"""
    test_name = "Test 1b: Package __init__.py re-exports"
    try:
        # Force reload the __init__.py
        init_path = os.path.join(_ECLAW_ROOT, "__init__.py")
        spec = importlib.util.spec_from_file_location(
            "omnigibson.learning.embodiedClaw",
            init_path,
            submodule_search_locations=[_ECLAW_ROOT],
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["omnigibson.learning.embodiedClaw"] = mod
        spec.loader.exec_module(mod)

        assert hasattr(mod, "get_simulation_snapshot")
        assert hasattr(mod, "filter_information")
        assert hasattr(mod, "VLAPolicyController")
        assert hasattr(mod, "pause_vla")
        assert hasattr(mod, "resume_vla")
        assert hasattr(mod, "reset_vla")
        assert hasattr(mod, "switch_vla")
        assert hasattr(mod, "CheckpointManager")
        assert hasattr(mod, "save_success_data")
        assert hasattr(mod, "save_failure_data")

        # Verify they are the same objects
        assert mod.get_simulation_snapshot is get_simulation_snapshot
        assert mod.VLAPolicyController is VLAPolicyController
        assert mod.CheckpointManager is CheckpointManager

        results.record_pass(test_name, "Package re-exports match direct imports")
    except Exception as e:
        results.record_fail(test_name, e, traceback.format_exc())


def test_1c_all_attribute():
    """Test 1c: __all__ attribute"""
    test_name = "Test 1c: __all__ attribute"
    try:
        mod = sys.modules["omnigibson.learning.embodiedClaw"]
        expected = {
            "get_simulation_snapshot", "filter_information",
            "VLAPolicyController", "pause_vla", "resume_vla", "reset_vla", "switch_vla",
            "CheckpointManager", "save_success_data", "save_failure_data",
        }
        actual = set(mod.__all__)
        assert actual == expected, f"__all__ mismatch: missing={expected - actual}, extra={actual - expected}"
        results.record_pass(test_name, f"__all__ has {len(expected)} expected symbols")
    except Exception as e:
        results.record_fail(test_name, e)


# ======================================================================
# Test 2: BDDLStateTracker
# ======================================================================

def test_2a_bddl_tracker_init():
    """Test 2a: BDDLStateTracker.__init__"""
    test_name = "Test 2a: BDDLStateTracker.__init__"
    try:
        tracker = BDDLStateTracker()
        assert tracker.transitions == []
        assert tracker.grasp_history == []
        assert tracker._tracked_predicates == []
        assert tracker._prev_states == {}
        results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e)


def test_2b_bddl_tracker_start():
    """Test 2b: BDDLStateTracker.start()"""
    test_name = "Test 2b: BDDLStateTracker.start()"
    try:
        tracker = BDDLStateTracker()
        env = make_mock_env()
        robot = make_mock_robot()
        tracker.start(env, robot)

        assert len(tracker._goal_heads) == 1, f"Expected 1 goal head, got {len(tracker._goal_heads)}"
        assert len(tracker._parsed_goal_conditions) == 1

        assert "apple.n.01_1" in tracker._task_objects
        assert "table.n.01_1" in tracker._task_objects
        assert "agent.n.01_1" not in tracker._task_objects

        grasp_preds = [p for p in tracker._tracked_predicates if p["type"] == "grasp"]
        assert len(grasp_preds) == 2, f"Expected 2 grasp predicates, got {len(grasp_preds)}"

        results.record_pass(test_name, f"{len(tracker._tracked_predicates)} predicates tracked")
    except Exception as e:
        results.record_fail(test_name, e, traceback.format_exc())


def test_2c_bddl_tracker_step():
    """Test 2c: BDDLStateTracker.step() transition detection"""
    test_name = "Test 2c: BDDLStateTracker.step()"
    try:
        tracker = BDDLStateTracker()
        env = make_mock_env()
        robot = make_mock_robot()
        tracker.start(env, robot)

        tracker.step(env, robot, step_idx=1)
        transitions_before = len(tracker.transitions)

        env.task.ground_goal_state_options[0][0].evaluate.return_value = True
        tracker.step(env, robot, step_idx=2)

        goal_transitions = [t for t in tracker.transitions if t["predicate_id"].startswith("goal_")]
        assert len(goal_transitions) >= 1, f"Expected goal transition, got {len(goal_transitions)}"
        assert goal_transitions[-1]["new_value"] is True
        assert goal_transitions[-1]["old_value"] is False
        assert goal_transitions[-1]["step_idx"] == 2

        results.record_pass(test_name, f"Detected {len(tracker.transitions)} transitions")
    except Exception as e:
        results.record_fail(test_name, e, traceback.format_exc())


def test_2d_bddl_tracker_grasp_change():
    """Test 2d: BDDLStateTracker grasp detection"""
    test_name = "Test 2d: BDDLStateTracker grasp detection"
    try:
        tracker = BDDLStateTracker()
        env = make_mock_env()
        robot = make_mock_robot()
        tracker.start(env, robot)

        grasped_obj = MagicMock()
        grasped_obj.name = "apple_obj"
        robot._ag_obj_in_hand["left"] = grasped_obj

        tracker.step(env, robot, step_idx=5)

        assert len(tracker.grasp_history) >= 1, "Expected grasp change to be recorded"
        last_grasp = tracker.grasp_history[-1]
        assert last_grasp["arm"] == "left"
        assert last_grasp["new_value"] == "apple_obj"
        assert last_grasp["old_value"] is None

        results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e, traceback.format_exc())


def test_2e_bddl_tracker_get_results():
    """Test 2e: BDDLStateTracker.get_results()"""
    test_name = "Test 2e: BDDLStateTracker.get_results()"
    try:
        tracker = BDDLStateTracker()
        env = make_mock_env()
        robot = make_mock_robot()
        tracker.start(env, robot)
        tracker.step(env, robot, step_idx=1)

        results_dict = tracker.get_results()
        assert "goal_conditions" in results_dict
        assert "transitions" in results_dict
        assert "final_state" in results_dict
        assert "grasp_history" in results_dict

        json.dumps(results_dict["goal_conditions"])
        json.dumps(results_dict["transitions"])
        json.dumps(results_dict["grasp_history"])

        results.record_pass(test_name, "All result fields present and JSON-serializable")
    except Exception as e:
        results.record_fail(test_name, e, traceback.format_exc())


def test_2f_bddl_tracker_no_goals():
    """Test 2f: BDDLStateTracker with no goal conditions"""
    test_name = "Test 2f: BDDLStateTracker with no goal conditions"
    try:
        tracker = BDDLStateTracker()
        env = make_mock_env()
        env.task.ground_goal_state_options = None
        robot = make_mock_robot()
        tracker.start(env, robot)

        assert len(tracker._goal_heads) == 0
        assert len(tracker._parsed_goal_conditions) == 0

        tracker.step(env, robot, step_idx=1)
        res = tracker.get_results()
        assert res["goal_conditions"] == []

        results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e, traceback.format_exc())


def test_2g_bddl_tracker_no_object_scope():
    """Test 2g: BDDLStateTracker with no object_scope"""
    test_name = "Test 2g: BDDLStateTracker with no object_scope"
    try:
        tracker = BDDLStateTracker()
        env = make_mock_env()
        del env.task.object_scope
        robot = make_mock_robot()
        tracker.start(env, robot)

        assert len(tracker._task_objects) == 0
        tracker.step(env, robot, step_idx=1)

        results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e, traceback.format_exc())


# ======================================================================
# Test 3: DataRecorder
# ======================================================================

def test_3a_data_recorder_init():
    """Test 3a: DataRecorder.__init__"""
    test_name = "Test 3a: DataRecorder.__init__"
    try:
        recorder = DataRecorder(
            output_folder="/tmp/test_recorder",
            task_name="test_task",
            task_id=1,
            demo_id=0,
            camera_names={"head": "robot_r1::robot_r1:zed_link:Camera:0"},
            record_rgb=True,
            record_depth=True,
        )
        assert recorder.output_folder == "/tmp/test_recorder"
        assert recorder.task_name == "test_task"
        assert recorder.is_recording is False
        assert recorder._step_count == 0
        results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e)


def test_3b_data_recorder_episode_tag():
    """Test 3b: DataRecorder._episode_tag"""
    test_name = "Test 3b: DataRecorder._episode_tag"
    try:
        recorder = DataRecorder(
            output_folder="/tmp/test",
            task_name="test",
            task_id=1,
            demo_id=2,
            camera_names={},
        )
        assert recorder._episode_tag == "episode_00010002", f"Got: {recorder._episode_tag}"

        recorder2 = DataRecorder(
            output_folder="/tmp/test",
            task_name="test",
            task_id=99,
            demo_id=88,
            camera_names={},
        )
        assert recorder2._episode_tag == "episode_00990088", f"Got: {recorder2._episode_tag}"

        results.record_pass(test_name, f"episode_tag = '{recorder._episode_tag}'")
    except Exception as e:
        results.record_fail(test_name, e)


def test_3c_data_recorder_resolution():
    """Test 3c: DataRecorder._resolution_for_camera"""
    test_name = "Test 3c: DataRecorder._resolution_for_camera"
    try:
        assert DataRecorder._resolution_for_camera("head") == (480, 640)
        assert DataRecorder._resolution_for_camera("left_wrist") == (240, 320)
        assert DataRecorder._resolution_for_camera("right_wrist") == (240, 320)
        assert DataRecorder._resolution_for_camera("anything_else") == (240, 320)
        results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e)


def test_3d_data_recorder_start_episode():
    """Test 3d: DataRecorder.start_episode()"""
    test_name = "Test 3d: DataRecorder.start_episode()"
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = DataRecorder(
                output_folder=tmpdir,
                task_name="test_task",
                task_id=1,
                demo_id=0,
                camera_names={"head": "robot_r1::robot_r1:zed_link:Camera:0"},
                record_rgb=True,
                record_depth=True,
            )
            recorder.start_episode()

            assert recorder.is_recording is True
            assert recorder._step_count == 0
            assert recorder._step_data == []

            assert os.path.isdir(f"{tmpdir}/data/"), "data/ directory not created"
            assert os.path.isdir(f"{tmpdir}/videos/"), "videos/ directory not created"
            assert os.path.isdir(f"{tmpdir}/meta/"), "meta/ directory not created"

            assert "head_rgb" in recorder._video_writers
            assert "head_depth" in recorder._video_writers

            results.record_pass(test_name, "Directories and video writers created")
    except Exception as e:
        results.record_fail(test_name, e, traceback.format_exc())


def test_3e_data_recorder_record_step():
    """Test 3e: DataRecorder.record_step()"""
    test_name = "Test 3e: DataRecorder.record_step()"
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = DataRecorder(
                output_folder=tmpdir,
                task_name="test_task",
                task_id=1,
                demo_id=0,
                camera_names={"head": "robot_r1::robot_r1:zed_link:Camera:0"},
                record_rgb=True,
                record_depth=False,
            )
            recorder.start_episode()

            for i in range(5):
                recorder.record_step(
                    action=np.zeros(10),
                    proprio=np.zeros(5),
                    cam_rel_poses=np.zeros(7),
                    task_info={"step": i},
                    obs={},
                )

            assert recorder._step_count == 5, f"Expected step_count=5, got {recorder._step_count}"
            assert len(recorder._step_data) == 5

            first_step = recorder._step_data[0]
            assert "step_idx" in first_step
            assert "action" in first_step
            assert "proprio" in first_step
            assert "cam_rel_poses" in first_step
            assert "task_info" in first_step
            assert first_step["step_idx"] == 0

            results.record_pass(test_name, f"Recorded {recorder._step_count} steps")
    except Exception as e:
        results.record_fail(test_name, e, traceback.format_exc())


def test_3f_data_recorder_record_step_not_recording():
    """Test 3f: DataRecorder.record_step() when not recording"""
    test_name = "Test 3f: DataRecorder.record_step() when not recording"
    try:
        recorder = DataRecorder(
            output_folder="/tmp/test",
            task_name="test",
            task_id=1,
            demo_id=0,
            camera_names={},
        )
        recorder.record_step(
            action=np.zeros(10),
            proprio=np.zeros(5),
            cam_rel_poses=np.zeros(7),
            task_info={},
            obs={},
        )
        assert recorder._step_count == 0
        assert len(recorder._step_data) == 0

        results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e)


def test_3g_data_recorder_end_episode():
    """Test 3g: DataRecorder.end_episode()"""
    test_name = "Test 3g: DataRecorder.end_episode()"
    try:
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = DataRecorder(
                output_folder=tmpdir,
                task_name="test_task",
                task_id=1,
                demo_id=0,
                camera_names={"head": "robot_r1::robot_r1:zed_link:Camera:0"},
                record_rgb=True,
                record_depth=False,
            )
            recorder.start_episode()
            for i in range(3):
                recorder.record_step(
                    action=np.zeros(10),
                    proprio=np.zeros(5),
                    cam_rel_poses=np.zeros(7),
                    task_info={"step": i},
                    obs={},
                )

            result = recorder.end_episode(
                success=True,
                env=MagicMock(),
                bddl_transitions={"test": "data"},
            )

            assert result is True
            assert recorder.is_recording is False

            parquet_path = f"{tmpdir}/data/episode_00010000.parquet"
            assert os.path.exists(parquet_path), f"Parquet file not found: {parquet_path}"
            df = pd.read_parquet(parquet_path)
            assert len(df) == 3, f"Expected 3 rows, got {len(df)}"

            meta_path = f"{tmpdir}/meta/episode_00010000.json"
            assert os.path.exists(meta_path), f"Metadata JSON not found: {meta_path}"
            with open(meta_path) as f:
                meta = json.load(f)
            assert meta["success"] is True
            assert meta["n_steps"] == 3
            assert meta["task_name"] == "test_task"

            bddl_path = f"{tmpdir}/meta/episode_00010000_bddl.json"
            assert os.path.exists(bddl_path), f"BDDL JSON not found: {bddl_path}"
            with open(bddl_path) as f:
                bddl_data = json.load(f)
            assert bddl_data == {"test": "data"}

            results.record_pass(test_name, "Parquet, metadata JSON, and BDDL JSON all written")
    except Exception as e:
        results.record_fail(test_name, e, traceback.format_exc())


def test_3h_data_recorder_end_episode_not_recording():
    """Test 3h: DataRecorder.end_episode() when not recording"""
    test_name = "Test 3h: DataRecorder.end_episode() when not recording"
    try:
        recorder = DataRecorder(
            output_folder="/tmp/test",
            task_name="test",
            task_id=1,
            demo_id=0,
            camera_names={},
        )
        result = recorder.end_episode(success=False)
        assert result is False
        results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e)


def test_3i_data_recorder_no_bddl_transitions():
    """Test 3i: DataRecorder.end_episode() without BDDL transitions"""
    test_name = "Test 3i: DataRecorder.end_episode() without BDDL transitions"
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = DataRecorder(
                output_folder=tmpdir,
                task_name="test_task",
                task_id=1,
                demo_id=0,
                camera_names={},
                record_rgb=False,
                record_depth=False,
            )
            recorder.start_episode()
            recorder.record_step(
                action=np.zeros(10),
                proprio=np.zeros(5),
                cam_rel_poses=np.zeros(7),
                task_info={},
                obs={},
            )
            result = recorder.end_episode(success=False, bddl_transitions=None)

            assert result is True
            bddl_path = f"{tmpdir}/meta/episode_00010000_bddl.json"
            assert not os.path.exists(bddl_path), "BDDL JSON should not exist when transitions=None"

            results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e, traceback.format_exc())


# ======================================================================
# Test 4: VLAPolicyController + pause/resume
# ======================================================================

def test_4a_controller_init():
    """Test 4a: VLAPolicyController.__init__"""
    test_name = "Test 4a: VLAPolicyController.__init__"
    try:
        policy = make_mock_policy()
        controller = VLAPolicyController(policy)
        assert controller._run_event.is_set(), "Controller should start in running state"
        assert controller.cached_snapshot is None
        assert controller.language_instruction is None
        results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e)


def test_4b_pause_resume():
    """Test 4b: pause_vla / resume_vla"""
    test_name = "Test 4b: pause_vla / resume_vla"
    try:
        policy = make_mock_policy()
        controller = VLAPolicyController(policy)

        result = pause_vla(controller)
        assert result == {"status": "paused"}
        assert not controller._run_event.is_set(), "Event should be cleared after pause"

        result = resume_vla(controller)
        assert result == {"status": "running"}
        assert controller._run_event.is_set(), "Event should be set after resume"

        results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e)


def test_4c_check_and_pause_running():
    """Test 4c: check_and_pause when running (no-op)"""
    test_name = "Test 4c: check_and_pause when running"
    try:
        policy = make_mock_policy()
        controller = VLAPolicyController(policy)
        env = make_mock_env()
        robot = make_mock_robot()

        start = time.time()
        controller.check_and_pause(env, robot, "test_task")
        elapsed = time.time() - start
        assert elapsed < 1.0, f"check_and_pause took {elapsed}s, expected < 1s"
        assert controller.cached_snapshot is None, "Snapshot should not be cached when running"

        results.record_pass(test_name, f"Returned in {elapsed:.3f}s")
    except Exception as e:
        results.record_fail(test_name, e)


def test_4d_check_and_pause_blocks():
    """Test 4d: check_and_pause blocks when paused"""
    test_name = "Test 4d: check_and_pause blocks when paused"
    try:
        policy = make_mock_policy()
        controller = VLAPolicyController(policy)
        env = make_mock_env()
        robot = make_mock_robot()

        pause_vla(controller)

        blocked = threading.Event()
        unblocked = threading.Event()

        def sim_loop():
            blocked.set()
            controller.check_and_pause(env, robot, "test_task")
            unblocked.set()

        t = threading.Thread(target=sim_loop, daemon=True)
        t.start()

        blocked.wait(timeout=2.0)
        time.sleep(0.3)

        assert not unblocked.is_set(), "Sim loop should be blocked"

        # Check that snapshot was cached
        time.sleep(0.3)
        assert controller.cached_snapshot is not None, "Snapshot should be cached while paused"

        resume_vla(controller)
        unblocked.wait(timeout=2.0)
        assert unblocked.is_set(), "Sim loop should have unblocked after resume"

        t.join(timeout=2.0)
        assert controller.cached_snapshot is None, "Snapshot should be cleared after resume"

        results.record_pass(test_name, "Sim loop blocked and unblocked correctly")
    except Exception as e:
        results.record_fail(test_name, e, traceback.format_exc())


def test_4e_forward_delegation():
    """Test 4e: VLAPolicyController.forward delegation"""
    test_name = "Test 4e: VLAPolicyController.forward delegation"
    try:
        policy = make_mock_policy()
        controller = VLAPolicyController(policy)
        obs = {"key": "value"}
        controller.forward(obs)
        policy.forward.assert_called_once_with(obs=obs)
        results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e)


def test_4f_reset_delegation():
    """Test 4f: VLAPolicyController.reset delegation"""
    test_name = "Test 4f: VLAPolicyController.reset delegation"
    try:
        policy = make_mock_policy()
        controller = VLAPolicyController(policy)
        controller.reset()
        policy.reset.assert_called_once()
        results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e)


def test_4g_language_instruction():
    """Test 4g: language_instruction property"""
    test_name = "Test 4g: language_instruction property"
    try:
        policy = make_mock_policy()
        controller = VLAPolicyController(policy)
        assert controller.language_instruction is None
        controller.language_instruction = "pick up the cup"
        assert controller.language_instruction == "pick up the cup"
        results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e)


# ======================================================================
# Test 5: information_tools
# ======================================================================

def test_5a_get_simulation_snapshot_no_obs():
    """Test 5a: get_simulation_snapshot (no obs)"""
    test_name = "Test 5a: get_simulation_snapshot (no obs)"
    try:
        env = make_mock_env()
        robot = make_mock_robot()
        snapshot = get_simulation_snapshot(env, robot, "test_task")

        assert isinstance(snapshot, dict)
        assert "images" in snapshot and snapshot["images"] == {}
        assert "robot_state" in snapshot
        assert "object_states" in snapshot
        assert "bddl_state" in snapshot
        assert "step_count" in snapshot
        assert "task_name" in snapshot
        assert snapshot["task_name"] == "test_task"
        assert snapshot["step_count"] == 42

        rs = snapshot["robot_state"]
        assert rs["position"] == [1.0, 2.0, 3.0]
        assert rs["orientation"] == [0.0, 0.0, 0.0, 1.0]
        assert rs["joint_positions"] == [0.1, 0.2, 0.3]
        assert "eef_poses" in rs
        assert "left" in rs["eef_poses"]
        assert "right" in rs["eef_poses"]
        assert rs["grasped_objects"]["left"] is None
        assert rs["grasped_objects"]["right"] is None

        assert "apple.n.01_1" in snapshot["object_states"]
        assert "table.n.01_1" in snapshot["object_states"]
        assert "agent.n.01_1" not in snapshot["object_states"]

        assert len(snapshot["bddl_state"]) == 1
        assert snapshot["bddl_state"][0]["current_value"] is False

        results.record_pass(test_name, "Snapshot has correct structure and values")
    except Exception as e:
        results.record_fail(test_name, e, traceback.format_exc())


def test_5b_get_simulation_snapshot_with_obs():
    """Test 5b: get_simulation_snapshot (with obs)"""
    test_name = "Test 5b: get_simulation_snapshot (with obs)"
    try:
        env = make_mock_env()
        robot = make_mock_robot()
        obs = {
            "robot_r1::robot_r1:zed_link:Camera:0::rgb": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        }
        snapshot = get_simulation_snapshot(env, robot, "test_task", obs=obs)

        assert "head" in snapshot["images"], f"Expected 'head' in images, got: {list(snapshot['images'].keys())}"
        assert len(snapshot["images"]["head"]) > 0

        results.record_pass(test_name, f"Image encoded: {len(snapshot['images']['head'])} chars")
    except Exception as e:
        results.record_fail(test_name, e, traceback.format_exc())


def test_5b2_get_simulation_snapshot_without_images():
    """Test 5b2: get_simulation_snapshot (include_images=False)"""
    test_name = "Test 5b2: get_simulation_snapshot (include_images=False)"
    try:
        env = make_mock_env()
        robot = make_mock_robot()
        obs = {
            "robot_r1::robot_r1:zed_link:Camera:0::rgb": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        }
        snapshot = get_simulation_snapshot(
            env,
            robot,
            "test_task",
            obs=obs,
            include_images=False,
        )

        assert snapshot["images"] == {}
        assert "robot_state" in snapshot
        assert "object_states" in snapshot
        assert "step_count" in snapshot

        results.record_pass(test_name, "Image payloads were omitted")
    except Exception as e:
        results.record_fail(test_name, e, traceback.format_exc())


def test_5c_get_simulation_snapshot_error_handling():
    """Test 5c: get_simulation_snapshot error handling"""
    test_name = "Test 5c: get_simulation_snapshot error handling"
    try:
        env = make_mock_env()
        robot = make_mock_robot()
        robot.get_position_orientation.side_effect = RuntimeError("sim crashed")
        robot.get_joint_positions.side_effect = RuntimeError("sim crashed")

        snapshot = get_simulation_snapshot(env, robot, "test_task")

        assert snapshot["robot_state"]["position"] is None
        assert snapshot["robot_state"]["orientation"] is None
        assert snapshot["robot_state"]["joint_positions"] is None

        results.record_pass(test_name, "Gracefully handled robot errors")
    except Exception as e:
        results.record_fail(test_name, e, traceback.format_exc())


def test_5d_filter_information_basic():
    """Test 5d: filter_information basic"""
    test_name = "Test 5d: filter_information basic"
    try:
        snapshot = {
            "images": {"head": "base64..."},
            "robot_state": {"position": [1, 2, 3]},
            "object_states": {
                "apple.n.01_1": {"position": [4, 5, 6]},
                "table.n.01_1": {"position": [7, 8, 9]},
                "chair.n.01_1": {"position": [10, 11, 12]},
            },
            "bddl_state": [
                {"predicate_str": "OnTop apple.n.01_1 table.n.01_1", "current_value": False},
                {"predicate_str": "Inside cup.n.01_1 cabinet.n.01_1", "current_value": True},
            ],
            "step_count": 10,
            "task_name": "test_task",
        }

        filtered = filter_information(snapshot, ["apple"])

        assert "apple.n.01_1" in filtered["object_states"]
        assert "table.n.01_1" not in filtered["object_states"]
        assert "chair.n.01_1" not in filtered["object_states"]

        assert len(filtered["bddl_state"]) == 1
        assert "apple" in filtered["bddl_state"][0]["predicate_str"]

        assert filtered["robot_state"] == snapshot["robot_state"]
        assert filtered["images"] == snapshot["images"]

        results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e)


def test_5e_filter_information_empty_filter():
    """Test 5e: filter_information empty filter"""
    test_name = "Test 5e: filter_information empty filter"
    try:
        snapshot = {
            "images": {},
            "robot_state": {},
            "object_states": {"a": 1, "b": 2},
            "bddl_state": [{"predicate_str": "test", "current_value": True}],
            "step_count": 0,
            "task_name": "",
        }
        filtered = filter_information(snapshot, [])
        assert filtered["object_states"] == snapshot["object_states"]
        assert filtered["bddl_state"] == snapshot["bddl_state"]

        results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e)


def test_5f_filter_information_multiple_names():
    """Test 5f: filter_information multiple names"""
    test_name = "Test 5f: filter_information multiple names"
    try:
        snapshot = {
            "images": {},
            "robot_state": {},
            "object_states": {
                "apple.n.01_1": {"pos": [1]},
                "table.n.01_1": {"pos": [2]},
                "chair.n.01_1": {"pos": [3]},
            },
            "bddl_state": [],
            "step_count": 0,
            "task_name": "",
        }
        filtered = filter_information(snapshot, ["apple", "chair"])
        assert "apple.n.01_1" in filtered["object_states"]
        assert "chair.n.01_1" in filtered["object_states"]
        assert "table.n.01_1" not in filtered["object_states"]

        results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e)


def test_5g_filter_information_without_images():
    """Test 5g: filter_information (include_images=False)"""
    test_name = "Test 5g: filter_information (include_images=False)"
    try:
        snapshot = {
            "images": {"head": "base64..."},
            "robot_state": {"position": [1, 2, 3]},
            "object_states": {
                "apple.n.01_1": {"scene_name": "apple_1", "position": [4, 5, 6]},
                "table.n.01_1": {"scene_name": "table_1", "position": [7, 8, 9]},
            },
            "world_predicates": [
                {"args": [{"scene_name": "apple_1"}, {"scene_name": "table_1"}]},
            ],
            "name_mapping": {"apple.n.01_1": "apple_1", "table.n.01_1": "table_1"},
            "step_count": 10,
            "task_name": "test_task",
        }

        filtered = filter_information(snapshot, ["apple_1"], include_images=False)
        assert filtered["images"] == {}
        assert "apple.n.01_1" in filtered["object_states"]
        assert "table.n.01_1" not in filtered["object_states"]
        assert len(filtered["world_predicates"]) == 1

        unfiltered = filter_information(snapshot, [], include_images=False)
        assert unfiltered["images"] == {}
        assert unfiltered["object_states"] == snapshot["object_states"]
        assert snapshot["images"] == {"head": "base64..."}

        results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e, traceback.format_exc())


def test_5h_filter_information_normalized_type_names():
    """Test 5h: filter_information normalized type names"""
    test_name = "Test 5h: filter_information normalized type names"
    try:
        snapshot = {
            "images": {},
            "robot_state": {"grasped_objects": {"left": None, "right": None}},
            "object_states": {
                "can__of__soda.n.01_1": {
                    "scene_name": "can_of_soda_113",
                    "position": [1, 1, 0],
                },
                "can__of__soda.n.01_2": {
                    "scene_name": "can_of_soda_114",
                    "position": [2, 2, 0],
                },
                "ashcan.n.01_1": {
                    "scene_name": "trash_can_116",
                    "position": [3, 3, 0],
                },
                "banana.n.01_1": {
                    "scene_name": "banana_1",
                    "position": [4, 4, 0],
                },
            },
            "world_predicates": [
                {
                    "predicate": "Inside",
                    "args": [
                        {
                            "scope_name": "can__of__soda.n.01_1",
                            "scene_name": "can_of_soda_113",
                        },
                        {
                            "scope_name": "ashcan.n.01_1",
                            "scene_name": "trash_can_116",
                        },
                    ],
                    "value": False,
                },
                {
                    "predicate": "Inside",
                    "args": [
                        {
                            "scope_name": "banana.n.01_1",
                            "scene_name": "banana_1",
                        },
                        {
                            "scope_name": "ashcan.n.01_1",
                            "scene_name": "trash_can_116",
                        },
                    ],
                    "value": False,
                },
            ],
            "predicate_deltas_since_subtask_start": [],
            "predicate_deltas_since_previous_decision": [],
            "name_mapping": {
                "can__of__soda.n.01_1": "can_of_soda_113",
                "can__of__soda.n.01_2": "can_of_soda_114",
                "ashcan.n.01_1": "trash_can_116",
                "banana.n.01_1": "banana_1",
            },
            "step_count": 12,
            "task_name": "picking_up_trash",
        }

        filtered = filter_information(
            snapshot,
            ["can_of_soda", "trash_can"],
            include_images=False,
        )

        assert filtered["images"] == {}
        assert "can__of__soda.n.01_1" in filtered["object_states"]
        assert "can__of__soda.n.01_2" in filtered["object_states"]
        assert "ashcan.n.01_1" in filtered["object_states"]
        assert "banana.n.01_1" not in filtered["object_states"]
        assert len(filtered["world_predicates"]) == 2

        exact_filtered = filter_information(
            snapshot,
            ["can_of_soda_114"],
            include_images=False,
        )
        assert set(exact_filtered["object_states"]) == {"can__of__soda.n.01_2"}

        results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e, traceback.format_exc())


def test_5i_rollout_subtask_object_refs_strip_ids():
    """Test 5i: rollout subtask object refs strip ids"""
    test_name = "Test 5i: rollout subtask object refs strip ids"
    try:
        subtask = SubtaskAnnotation(
            skill_idx=4,
            skill_description="place in",
            object_ids=[["can_of_soda_114", "trash_can_116"]],
            manipulating_object_ids=["can_of_soda_114"],
            frame_start=0,
            frame_end=10,
            frame_duration=10,
            skill_type="coordinated",
        )

        refs = get_rollout_subtask_object_refs(subtask)

        assert refs["object_ids"] == [["can_of_soda", "trash_can"]]
        assert refs["manipulating_object_ids"] == ["can_of_soda"]
        assert refs["annotation_object_ids"] == [["can_of_soda_114", "trash_can_116"]]
        assert refs["annotation_manipulating_object_ids"] == ["can_of_soda_114"]

        results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e, traceback.format_exc())


def test_5j_check_placement_success_normalized_delta():
    """Test 5j: check_placement_success normalized delta"""
    test_name = "Test 5j: check_placement_success normalized delta"
    try:
        matching_args = [
            {
                "scope_name": "can__of__soda.n.01_1",
                "scene_name": "can_of_soda_113",
            },
            {
                "scope_name": "ashcan.n.01_1",
                "scene_name": "trash_can_116",
            },
        ]
        snapshot = {
            "world_predicates": [
                {
                    "predicate": "Inside",
                    "args": matching_args,
                    "value": True,
                },
            ],
            "predicate_deltas_since_subtask_start": [
                {
                    "predicate": "Inside",
                    "args": matching_args,
                    "old_value": False,
                    "new_value": True,
                    "change_type": "became_true",
                },
            ],
            "robot_state": {"grasped_objects": {"left": None, "right": None}},
        }

        result = check_placement_success(
            snapshot,
            placed_object="can_of_soda_114",
            target_container="trash_can_116",
            placement_type="Inside",
        )

        assert result["completed"] is True
        assert result["matching_strategy"] == "normalized_delta"
        assert result["normalized_true_delta_count"] == 1

        results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e, traceback.format_exc())


def test_5k_check_placement_success_no_static_same_type_false_positive():
    """Test 5k: check_placement_success no static same-type false positive"""
    test_name = "Test 5k: check_placement_success no static same-type false positive"
    try:
        snapshot = {
            "world_predicates": [
                {
                    "predicate": "Inside",
                    "args": [
                        {
                            "scope_name": "can__of__soda.n.01_1",
                            "scene_name": "can_of_soda_113",
                        },
                        {
                            "scope_name": "ashcan.n.01_1",
                            "scene_name": "trash_can_116",
                        },
                    ],
                    "value": True,
                },
            ],
            "predicate_deltas_since_subtask_start": [],
            "robot_state": {"grasped_objects": {"left": None, "right": None}},
        }

        result = check_placement_success(
            snapshot,
            placed_object="can_of_soda_114",
            target_container="trash_can_116",
            placement_type="Inside",
        )

        assert result["completed"] is False
        assert result["matching_strategy"] == "none"

        results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e, traceback.format_exc())


def test_5l_object_matching_shared_query():
    """Test 5l: shared object matching query semantics"""
    test_name = "Test 5l: shared object matching query semantics"
    try:
        exact_query = build_match_query(["can_of_soda_114"])
        normalized_query = build_match_query(
            ["can_of_soda_114"],
            include_normalized=True,
            include_normalized_for_instance_specific=True,
        )
        trash_query = build_match_query(["trash_can"])
        cup_query = build_match_query(["cup"])

        assert name_matches_query("can_of_soda_114", exact_query) is True
        assert name_matches_query("can_of_soda_113", exact_query) is False
        assert name_matches_query("can_of_soda_113", normalized_query) is True
        assert entity_matches_query("ashcan.n.01_1", "trash_can_116", trash_query) is True
        assert arg_matches_query(
            {
                "scope_name": "ashcan.n.01_1",
                "scene_name": "trash_can_116",
            },
            trash_query,
        ) is True
        assert entity_matches_query("cupcake.n.01_1", "cupcake_1", cup_query) is False

        results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e, traceback.format_exc())


def test_5m_check_navigation_success_normalized_instance_target():
    """Test 5m: check_navigation_success normalized instance target"""
    test_name = "Test 5m: check_navigation_success normalized instance target"
    try:
        snapshot = {
            "robot_state": {"position": [0.0, 0.0, 0.0]},
            "object_states": {
                "can__of__soda.n.01_1": {
                    "scene_name": "can_of_soda_113",
                    "position": [0.5, 0.0, 0.0],
                },
                "can__of__soda.n.01_2": {
                    "scene_name": "can_of_soda_114",
                    "position": [3.0, 0.0, 0.0],
                },
            },
        }

        result = check_navigation_success(snapshot, ["can_of_soda_114"])

        assert result["completed"] is True
        assert result["closest_object"] == "can__of__soda.n.01_1"
        assert result["min_distance"] == 0.5

        results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e, traceback.format_exc())


def test_5n_check_grasp_success_normalized_instance_target():
    """Test 5n: check_grasp_success normalized instance target"""
    test_name = "Test 5n: check_grasp_success normalized instance target"
    try:
        snapshot = {
            "robot_state": {
                "grasped_objects": {
                    "left": "can_of_soda_113",
                    "right": None,
                },
            },
        }

        result = check_grasp_success(snapshot, ["can_of_soda_114"])

        assert result["completed"] is True
        assert result["matched_arms"] == ["left"]

        results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e, traceback.format_exc())


def test_5o_evaluate_subtask_status_rollout_safe_subtask_info():
    """Test 5o: evaluate_subtask_status rollout-safe subtask info"""
    test_name = "Test 5o: evaluate_subtask_status rollout-safe subtask info"
    try:
        subtask = SubtaskAnnotation(
            skill_idx=4,
            skill_description="place in",
            object_ids=[["can_of_soda_114", "trash_can_116"]],
            manipulating_object_ids=["can_of_soda_114"],
            frame_start=0,
            frame_end=10,
            frame_duration=10,
            skill_type="coordinated",
        )
        annotation = EpisodeAnnotation(
            task_name="picking_up_trash",
            task_duration=100,
            valid_start=0,
            valid_end=100,
            subtasks=[subtask],
            primitive_annotations=[],
        )
        duration_stats = {
            4: SubtaskDurationStats(
                skill_idx=4,
                skill_description="place in",
                min_duration=10,
                max_duration=10,
                mean_duration=10.0,
                count=1,
                failure_threshold=20,
            ),
        }
        memory = TaskMemory("picking_up_trash", annotation, duration_stats)
        snapshot = {
            "robot_state": {"grasped_objects": {"left": None, "right": None}},
            "world_predicates": [],
            "predicate_deltas_since_subtask_start": [],
            "step_count": 3,
        }

        result = evaluate_subtask_status(snapshot, memory, current_step=3)

        assert result["subtask_info"]["object_ids"] == [["can_of_soda", "trash_can"]]
        assert result["subtask_info"]["manipulating_object_ids"] == ["can_of_soda"]
        assert result["subtask_info"]["annotation_object_ids"] == [
            ["can_of_soda_114", "trash_can_116"]
        ]
        assert result["subtask_info"]["annotation_manipulating_object_ids"] == [
            "can_of_soda_114"
        ]

        results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e, traceback.format_exc())


def test_5p_check_placement_success_allows_other_same_type_grasped():
    """Test 5p: placement success allows another same-type object to be grasped"""
    test_name = "Test 5p: placement success allows another same-type object to be grasped"
    try:
        matching_args = [
            {
                "scope_name": "can__of__soda.n.01_1",
                "scene_name": "can_of_soda_113",
            },
            {
                "scope_name": "ashcan.n.01_1",
                "scene_name": "trash_can_116",
            },
        ]
        snapshot = {
            "world_predicates": [
                {
                    "predicate": "Inside",
                    "args": matching_args,
                    "value": True,
                },
            ],
            "predicate_deltas_since_subtask_start": [
                {
                    "predicate": "Inside",
                    "args": matching_args,
                    "old_value": False,
                    "new_value": True,
                    "change_type": "became_true",
                },
            ],
            "robot_state": {
                "grasped_objects": {
                    "left": "can_of_soda_114",
                    "right": None,
                },
            },
        }

        result = check_placement_success(
            snapshot,
            placed_object="can_of_soda_114",
            target_container="trash_can_116",
            placement_type="Inside",
        )

        assert result["completed"] is True
        assert result["matching_strategy"] == "normalized_delta"
        assert result["still_grasping"] is False

        results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e, traceback.format_exc())


def test_5q_filter_information_preserves_robot_predicates_and_deltas():
    """Test 5q: filter_information preserves robot predicates and deltas"""
    test_name = "Test 5q: filter_information preserves robot predicates and deltas"
    try:
        snapshot = {
            "images": {},
            "robot_state": {"grasped_objects": {"left": None, "right": None}},
            "object_states": {
                "can__of__soda.n.01_1": {
                    "scene_name": "can_of_soda_113",
                    "position": [1, 1, 0],
                },
                "banana.n.01_1": {
                    "scene_name": "banana_1",
                    "position": [4, 4, 0],
                },
            },
            "world_predicates": [
                {
                    "predicate": "InReachOfRobot",
                    "args": [
                        {"scope_name": "robot", "scene_name": "robot"},
                        {
                            "scope_name": "banana.n.01_1",
                            "scene_name": "banana_1",
                        },
                    ],
                    "value": False,
                },
                {
                    "predicate": "Inside",
                    "args": [
                        {
                            "scope_name": "can__of__soda.n.01_1",
                            "scene_name": "can_of_soda_113",
                        },
                        {
                            "scope_name": "banana.n.01_1",
                            "scene_name": "banana_1",
                        },
                    ],
                    "value": False,
                },
            ],
            "predicate_deltas_since_subtask_start": [
                {
                    "predicate": "InReachOfRobot",
                    "args": [
                        {"scope_name": "robot", "scene_name": "robot"},
                        {
                            "scope_name": "banana.n.01_1",
                            "scene_name": "banana_1",
                        },
                    ],
                    "old_value": False,
                    "new_value": True,
                    "change_type": "became_true",
                },
                {
                    "predicate": "Inside",
                    "args": [
                        {
                            "scope_name": "can__of__soda.n.01_1",
                            "scene_name": "can_of_soda_113",
                        },
                        {
                            "scope_name": "banana.n.01_1",
                            "scene_name": "banana_1",
                        },
                    ],
                    "old_value": False,
                    "new_value": True,
                    "change_type": "became_true",
                },
            ],
            "predicate_deltas_since_previous_decision": [
                {
                    "predicate": "InReachOfRobot",
                    "args": [
                        {"scope_name": "robot", "scene_name": "robot"},
                        {
                            "scope_name": "banana.n.01_1",
                            "scene_name": "banana_1",
                        },
                    ],
                    "old_value": False,
                    "new_value": True,
                    "change_type": "became_true",
                },
            ],
            "predicate_delta_metadata": {"current_subtask_idx": 0},
            "name_mapping": {},
            "step_count": 12,
            "task_name": "picking_up_trash",
        }

        filtered = filter_information(snapshot, ["can_of_soda"], include_images=False)

        assert set(filtered["object_states"]) == {"can__of__soda.n.01_1"}
        assert len(filtered["world_predicates"]) == 2
        assert len(filtered["predicate_deltas_since_subtask_start"]) == 2
        assert len(filtered["predicate_deltas_since_previous_decision"]) == 1
        assert filtered["predicate_deltas_since_previous_decision"][0]["args"][0]["scene_name"] == "robot"

        results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e, traceback.format_exc())


def test_5r_check_placement_success_exact_relation():
    """Test 5r: check_placement_success exact relation"""
    test_name = "Test 5r: check_placement_success exact relation"
    try:
        snapshot = {
            "world_predicates": [
                {
                    "predicate": "Inside",
                    "args": [
                        {
                            "scope_name": "can__of__soda.n.01_2",
                            "scene_name": "can_of_soda_114",
                        },
                        {
                            "scope_name": "ashcan.n.01_1",
                            "scene_name": "trash_can_116",
                        },
                    ],
                    "value": True,
                },
                {
                    "predicate": "Inside",
                    "args": [
                        {
                            "scope_name": "can__of__soda.n.01_1",
                            "scene_name": "can_of_soda_113",
                        },
                        {
                            "scope_name": "ashcan.n.01_1",
                            "scene_name": "trash_can_116",
                        },
                    ],
                    "value": True,
                },
            ],
            "predicate_deltas_since_subtask_start": [],
            "robot_state": {"grasped_objects": {"left": None, "right": None}},
        }

        result = check_placement_success(
            snapshot,
            placed_object="can_of_soda_114",
            target_container="trash_can_116",
            placement_type="Inside",
        )

        assert result["completed"] is True
        assert result["matching_strategy"] == "exact_relation"
        assert result["exact_true_relation_count"] == 1

        results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e, traceback.format_exc())


def test_5s_check_placement_success_exact_delta():
    """Test 5s: check_placement_success exact delta"""
    test_name = "Test 5s: check_placement_success exact delta"
    try:
        matching_args = [
            {
                "scope_name": "can__of__soda.n.01_2",
                "scene_name": "can_of_soda_114",
            },
            {
                "scope_name": "ashcan.n.01_1",
                "scene_name": "trash_can_116",
            },
        ]
        snapshot = {
            "world_predicates": [],
            "predicate_deltas_since_subtask_start": [
                {
                    "predicate": "Inside",
                    "args": matching_args,
                    "old_value": False,
                    "new_value": True,
                    "change_type": "became_true",
                },
            ],
            "robot_state": {"grasped_objects": {"left": None, "right": None}},
        }

        result = check_placement_success(
            snapshot,
            placed_object="can_of_soda_114",
            target_container="trash_can_116",
            placement_type="Inside",
        )

        assert result["completed"] is True
        assert result["matching_strategy"] == "exact_delta"

        results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e, traceback.format_exc())


def test_5t_check_placement_success_blocks_when_matched_object_still_grasped():
    """Test 5t: placement blocked when matched object is still grasped"""
    test_name = "Test 5t: placement blocked when matched object is still grasped"
    try:
        matching_args = [
            {
                "scope_name": "can__of__soda.n.01_1",
                "scene_name": "can_of_soda_113",
            },
            {
                "scope_name": "ashcan.n.01_1",
                "scene_name": "trash_can_116",
            },
        ]
        snapshot = {
            "world_predicates": [
                {
                    "predicate": "Inside",
                    "args": matching_args,
                    "value": True,
                },
            ],
            "predicate_deltas_since_subtask_start": [
                {
                    "predicate": "Inside",
                    "args": matching_args,
                    "old_value": False,
                    "new_value": True,
                    "change_type": "became_true",
                },
            ],
            "robot_state": {
                "grasped_objects": {
                    "left": "can_of_soda_113",
                    "right": None,
                },
            },
        }

        result = check_placement_success(
            snapshot,
            placed_object="can_of_soda_114",
            target_container="trash_can_116",
            placement_type="Inside",
        )

        assert result["completed"] is False
        assert result["matching_strategy"] == "none"
        assert result["still_grasping"] is True

        results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e, traceback.format_exc())


# ======================================================================
# Test 6: control_tools standalone functions
# ======================================================================

def test_6a_reset_vla():
    """Test 6a: reset_vla"""
    test_name = "Test 6a: reset_vla"
    try:
        policy = make_mock_policy()
        controller = VLAPolicyController(policy)
        result = reset_vla(controller)
        assert result == {"status": "reset"}
        policy.reset.assert_called_once()
        results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e)


def test_6b_switch_vla_instruction_only():
    """Test 6b: switch_vla (instruction only)"""
    test_name = "Test 6b: switch_vla (instruction only)"
    try:
        policy = make_mock_policy()
        controller = VLAPolicyController(policy)
        result = switch_vla(controller, "pick up the apple")
        assert result["status"] == "switched"
        assert result["language_instruction"] == "pick up the apple"
        assert controller.language_instruction == "pick up the apple"
        policy.update_host.assert_not_called()

        results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e)


def test_6c_switch_vla_with_host():
    """Test 6c: switch_vla (with host/port)"""
    test_name = "Test 6c: switch_vla (with host/port)"
    try:
        policy = make_mock_policy()
        controller = VLAPolicyController(policy)
        result = switch_vla(controller, "new instruction", host="10.0.0.1", port=8080)
        assert result["status"] == "switched"
        policy.update_host.assert_called_once_with("10.0.0.1", 8080)

        results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e)


def test_6d_switch_vla_host_without_port():
    """Test 6d: switch_vla (host without port)"""
    test_name = "Test 6d: switch_vla (host without port)"
    try:
        policy = make_mock_policy()
        controller = VLAPolicyController(policy)
        result = switch_vla(controller, "instruction", host="10.0.0.1", port=None)
        assert result["status"] == "switched"
        policy.update_host.assert_not_called()

        results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e)


# ======================================================================
# Test 7: data_tools
# ======================================================================

def test_7a_checkpoint_manager_init():
    """Test 7a: CheckpointManager.__init__"""
    test_name = "Test 7a: CheckpointManager.__init__"
    try:
        mgr = CheckpointManager(
            output_folder="/tmp/test_ckpt",
            task_name="test_task",
            task_id=1,
            demo_id=0,
        )
        assert mgr.output_folder == "/tmp/test_ckpt"
        assert mgr.task_name == "test_task"
        assert mgr.task_id == 1
        assert mgr.demo_id == 0
        assert mgr.record_rgb is True
        assert mgr.record_depth is True
        assert mgr._checkpoint_step == 0
        assert mgr._checkpoint_scene_state is None
        assert mgr._retry_counts == {}
        assert mgr._segment_id == 0
        assert mgr.should_end_episode is False
        assert mgr.data_recorder is None
        assert mgr.bddl_tracker is None

        results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e)


def test_7b_checkpoint_manager_setup():
    """Test 7b: CheckpointManager.setup()"""
    test_name = "Test 7b: CheckpointManager.setup()"
    try:
        mgr = CheckpointManager(
            output_folder="/tmp/test",
            task_name="test",
            task_id=1,
            demo_id=0,
        )
        recorder = MagicMock()
        tracker = MagicMock()
        mgr.setup(recorder, tracker)
        assert mgr.data_recorder is recorder
        assert mgr.bddl_tracker is tracker

        results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e)


def test_7c_checkpoint_manager_capture():
    """Test 7c: CheckpointManager.capture_checkpoint()"""
    test_name = "Test 7c: CheckpointManager.capture_checkpoint()"
    try:
        mgr = CheckpointManager(
            output_folder="/tmp/test",
            task_name="test",
            task_id=1,
            demo_id=0,
        )
        env = make_mock_env()
        env.scene.dump_state.return_value = {"objects": {"apple": "state"}}

        mgr.capture_checkpoint(env, step_idx=50)

        assert mgr._checkpoint_step == 50
        assert mgr._checkpoint_scene_state == {"objects": {"apple": "state"}}
        env.scene.dump_state.assert_called_once_with(serialized=False)

        results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e)


def test_7d_checkpoint_manager_retry_counting():
    """Test 7d: CheckpointManager retry counting"""
    test_name = "Test 7d: CheckpointManager retry counting"
    try:
        mgr = CheckpointManager(
            output_folder="/tmp/test",
            task_name="test",
            task_id=1,
            demo_id=0,
        )
        assert mgr.get_retry_count("subtask_1") == 0

        mgr._retry_counts["subtask_1"] = 3
        assert mgr.get_retry_count("subtask_1") == 3
        assert mgr.get_retry_count("subtask_2") == 0

        results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e)


def test_7e_should_end_episode_property():
    """Test 7e: CheckpointManager.should_end_episode"""
    test_name = "Test 7e: CheckpointManager.should_end_episode"
    try:
        mgr = CheckpointManager(
            output_folder="/tmp/test",
            task_name="test",
            task_id=1,
            demo_id=0,
        )
        assert mgr.should_end_episode is False
        mgr._should_end_episode = True
        assert mgr.should_end_episode is True

        results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e)


def test_7f_segment_output_folder():
    """Test 7f: _segment_output_folder"""
    test_name = "Test 7f: _segment_output_folder"
    try:
        mgr = CheckpointManager(
            output_folder="/data/output",
            task_name="test",
            task_id=1,
            demo_id=0,
        )
        mgr._segment_id = 0
        assert _segment_output_folder(mgr) == "/data/output/segment_0000"

        mgr._segment_id = 42
        assert _segment_output_folder(mgr) == "/data/output/segment_0042"

        results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e)


def test_7g_create_segment_recorder():
    """Test 7g: _create_segment_recorder"""
    test_name = "Test 7g: _create_segment_recorder"
    try:
        mgr = CheckpointManager(
            output_folder="/tmp/test_seg",
            task_name="test_task",
            task_id=5,
            demo_id=3,
            record_rgb=True,
            record_depth=False,
        )
        mgr._segment_id = 2

        recorder = _create_segment_recorder(mgr)
        assert isinstance(recorder, DataRecorder)
        assert recorder.output_folder == "/tmp/test_seg/segment_0002"
        assert recorder.task_name == "test_task"
        assert recorder.task_id == 5
        assert recorder.demo_id == 3
        assert recorder.record_rgb is True
        assert recorder.record_depth is False

        results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e)


def test_7h_save_success_data():
    """Test 7h: save_success_data"""
    test_name = "Test 7h: save_success_data"
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(
                output_folder=tmpdir,
                task_name="test_task",
                task_id=1,
                demo_id=0,
            )
            recorder = MagicMock()
            tracker = MagicMock()
            tracker.get_results.return_value = {"transitions": []}
            mgr.setup(recorder, tracker)

            env = make_mock_env()
            env._current_step = 100

            result = save_success_data(
                checkpoint_mgr=mgr,
                env=env,
                robot=make_mock_robot(),
                subtask_id="pick_apple",
            )

            assert result["status"] == "success_saved"
            assert result["subtask_id"] == "pick_apple"
            assert result["segment_id"] == 1
            assert mgr._segment_id == 1

            recorder.end_episode.assert_called_once()
            assert mgr._checkpoint_step == 100

            assert mgr.data_recorder is not None
            assert mgr.data_recorder is not recorder

            results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e, traceback.format_exc())


def test_7i_save_success_data_with_switch():
    """Test 7i: save_success_data with VLA switch"""
    test_name = "Test 7i: save_success_data with VLA switch"
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(
                output_folder=tmpdir,
                task_name="test_task",
                task_id=1,
                demo_id=0,
            )
            recorder = MagicMock()
            tracker = MagicMock()
            tracker.get_results.return_value = {}
            mgr.setup(recorder, tracker)

            env = make_mock_env()
            policy = make_mock_policy()
            controller = VLAPolicyController(policy)

            result = save_success_data(
                checkpoint_mgr=mgr,
                env=env,
                robot=make_mock_robot(),
                subtask_id="pick_apple",
                controller=controller,
                next_subtask_id="place_apple",
                next_language_instruction="place the apple on the table",
            )

            assert result["next_subtask_id"] == "place_apple"
            assert controller.language_instruction == "place the apple on the table"

            results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e, traceback.format_exc())


def test_7j_save_failure_data():
    """Test 7j: save_failure_data"""
    test_name = "Test 7j: save_failure_data"
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(
                output_folder=tmpdir,
                task_name="test_task",
                task_id=1,
                demo_id=0,
            )
            recorder = MagicMock()
            tracker = MagicMock()
            tracker.get_results.return_value = {}
            mgr.setup(recorder, tracker)

            env = make_mock_env()
            mgr._checkpoint_scene_state = {"objects": "state"}
            mgr._checkpoint_step = 50

            result = save_failure_data(
                checkpoint_mgr=mgr,
                env=env,
                robot=make_mock_robot(),
                subtask_id="pick_apple",
            )

            assert result["status"] == "failure_saved"
            assert result["subtask_id"] == "pick_apple"
            assert result["retry_count"] == 1
            assert result["should_end_episode"] is False
            assert result["restore_status"] == "restored"
            assert mgr._retry_counts["pick_apple"] == 1
            assert env._current_step == 50

            env.scene.load_state.assert_called_once_with(
                {"objects": "state"}, serialized=False
            )

            assert mgr._segment_id == 1

            results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e, traceback.format_exc())


def test_7k_save_failure_data_max_retries():
    """Test 7k: save_failure_data max retries"""
    test_name = "Test 7k: save_failure_data max retries"
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(
                output_folder=tmpdir,
                task_name="test_task",
                task_id=1,
                demo_id=0,
            )
            recorder = MagicMock()
            tracker = MagicMock()
            tracker.get_results.return_value = {}
            mgr.setup(recorder, tracker)

            env = make_mock_env()
            mgr._checkpoint_scene_state = {"objects": "state"}

            mgr._retry_counts["pick_apple"] = CheckpointManager.MAX_RETRIES_PER_SUBTASK - 1

            result = save_failure_data(
                checkpoint_mgr=mgr,
                env=env,
                robot=make_mock_robot(),
                subtask_id="pick_apple",
            )

            assert result["should_end_episode"] is True
            assert result["restore_status"] == "skipped_max_retries"
            assert mgr.should_end_episode is True
            assert result["retry_count"] == CheckpointManager.MAX_RETRIES_PER_SUBTASK

            env.scene.load_state.assert_not_called()

            results.record_pass(test_name, f"Max retries = {CheckpointManager.MAX_RETRIES_PER_SUBTASK}")
    except Exception as e:
        results.record_fail(test_name, e, traceback.format_exc())


def test_7l_save_failure_no_checkpoint():
    """Test 7l: save_failure_data (no checkpoint)"""
    test_name = "Test 7l: save_failure_data (no checkpoint)"
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(
                output_folder=tmpdir,
                task_name="test_task",
                task_id=1,
                demo_id=0,
            )
            recorder = MagicMock()
            tracker = MagicMock()
            tracker.get_results.return_value = {}
            mgr.setup(recorder, tracker)

            env = make_mock_env()

            result = save_failure_data(
                checkpoint_mgr=mgr,
                env=env,
                robot=make_mock_robot(),
                subtask_id="pick_apple",
            )

            assert result["status"] == "failure_saved"
            env.scene.load_state.assert_not_called()

            results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e, traceback.format_exc())


def test_7m_save_failure_with_controller_reset():
    """Test 7m: save_failure_data resets controller"""
    test_name = "Test 7m: save_failure_data resets controller"
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(
                output_folder=tmpdir,
                task_name="test_task",
                task_id=1,
                demo_id=0,
            )
            recorder = MagicMock()
            tracker = MagicMock()
            tracker.get_results.return_value = {}
            mgr.setup(recorder, tracker)

            env = make_mock_env()
            mgr._checkpoint_scene_state = {"objects": "state"}

            policy = make_mock_policy()
            controller = VLAPolicyController(policy)

            result = save_failure_data(
                checkpoint_mgr=mgr,
                env=env,
                robot=make_mock_robot(),
                subtask_id="pick_apple",
                controller=controller,
            )

            policy.reset.assert_called()

            results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e, traceback.format_exc())


def test_7n_save_failure_no_data_recorder():
    """Test 7n: save_failure_data (no data_recorder)"""
    test_name = "Test 7n: save_failure_data (no data_recorder)"
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(
                output_folder=tmpdir,
                task_name="test_task",
                task_id=1,
                demo_id=0,
            )
            env = make_mock_env()
            mgr._checkpoint_scene_state = {"objects": "state"}

            result = save_failure_data(
                checkpoint_mgr=mgr,
                env=env,
                robot=make_mock_robot(),
                subtask_id="pick_apple",
            )

            assert result["status"] == "failure_saved"
            assert result["retry_count"] == 1

            results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e, traceback.format_exc())


def test_7p_rebase_retry_start_step():
    """Test 7p: retry timing rebase after restore"""
    test_name = "Test 7p: retry timing rebase after restore"
    try:
        subtask = SubtaskAnnotation(
            skill_idx=4,
            skill_description="place in",
            object_ids=[["can_of_soda_114", "trash_can_116"]],
            manipulating_object_ids=["can_of_soda_114"],
            frame_start=0,
            frame_end=10,
            frame_duration=10,
            skill_type="coordinated",
        )
        annotation = EpisodeAnnotation(
            task_name="picking_up_trash",
            task_duration=100,
            valid_start=0,
            valid_end=100,
            subtasks=[subtask],
            primitive_annotations=[],
        )
        duration_stats = {
            4: SubtaskDurationStats(
                skill_idx=4,
                skill_description="place in",
                min_duration=10,
                max_duration=10,
                mean_duration=10.0,
                count=1,
                failure_threshold=20,
            ),
        }
        memory = TaskMemory("picking_up_trash", annotation, duration_stats)
        memory.subtask_start_step = 0

        rebased = rebase_retry_start_step(memory, 50, "restored")
        assert rebased is True
        assert memory.subtask_start_step == 50

        rebased = rebase_retry_start_step(memory, 80, "skipped_no_checkpoint")
        assert rebased is False
        assert memory.subtask_start_step == 50

        rebased = rebase_retry_start_step(memory, 50, "no_steps_to_revert")
        assert rebased is False
        assert memory.subtask_start_step == 50

        results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e, traceback.format_exc())


def test_7o_save_success_no_data_recorder():
    """Test 7o: save_success_data (no data_recorder)"""
    test_name = "Test 7o: save_success_data (no data_recorder)"
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(
                output_folder=tmpdir,
                task_name="test_task",
                task_id=1,
                demo_id=0,
            )
            env = make_mock_env()

            result = save_success_data(
                checkpoint_mgr=mgr,
                env=env,
                robot=make_mock_robot(),
                subtask_id="pick_apple",
            )

            assert result["status"] == "success_saved"

            results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e, traceback.format_exc())


# ======================================================================
# Test 8: Cross-module integration
# ======================================================================

def test_8a_sim_run_syntax():
    """Test 8a: embodied_claw_sim_run.py syntax check"""
    test_name = "Test 8a: embodied_claw_sim_run.py syntax check"
    try:
        import ast
        src_path = os.path.join(_ECLAW_ROOT, "embodied_claw_sim_run.py")
        with open(src_path, "r") as f:
            source = f.read()
        tree = ast.parse(source, filename=src_path)

        class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        assert "AgenticEvaluator" in class_names, f"AgenticEvaluator class not found; classes: {class_names}"

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "AgenticEvaluator":
                methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                for expected_method in ["__init__", "step", "reset", "load_env", "load_policy", "load_robot", "load_metrics"]:
                    assert expected_method in methods, f"Missing method {expected_method} in AgenticEvaluator; methods: {methods}"

        results.record_pass(test_name, f"Parsed successfully, found AgenticEvaluator with expected methods")
    except SyntaxError as e:
        results.record_fail(test_name, f"Syntax error: {e}",
                           "Fix the syntax error in embodied_claw_sim_run.py")
    except Exception as e:
        results.record_fail(test_name, e, traceback.format_exc())


def test_8b_sim_run_import():
    """Test 8b: embodied_claw_sim_run.py import"""
    test_name = "Test 8b: embodied_claw_sim_run.py import"
    try:
        src_path = os.path.join(_ECLAW_ROOT, "embodied_claw_sim_run.py")
        spec = importlib.util.spec_from_file_location(
            "omnigibson.learning.embodiedClaw.embodied_claw_sim_run",
            src_path,
        )
        macros_mod = sys.modules["omnigibson.macros"]
        mock_macros_result = MagicMock()
        mock_macros_result.NUM_EVAL_EPISODES = 1
        mock_macros_result.NUM_TRAIN_INSTANCES = 200
        mock_macros_result.NUM_EVAL_INSTANCES = 20
        macros_mod.create_module_macros.return_value = mock_macros_result
        macros_mod.gm.ENABLE_FLATCACHE = True
        macros_mod.gm.USE_GPU_DYNAMICS = False
        macros_mod.gm.ENABLE_TRANSITION_RULES = True

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        assert hasattr(module, "AgenticEvaluator")
        results.record_pass(test_name, "Module imported successfully with mocks")
    except Exception as e:
        results.record_fail(test_name, e, traceback.format_exc())


def test_8c_stabilise_scene():
    """Test 8c: _stabilise_scene"""
    test_name = "Test 8c: _stabilise_scene"
    try:
        env = make_mock_env()
        entity1 = MagicMock()
        entity1.is_system = False
        entity1.exists = True

        entity2 = MagicMock()
        entity2.is_system = True
        entity2.exists = True

        env.task.object_scope = {"obj1": entity1, "sys1": entity2}

        _stabilise_scene(env)

        assert entity1.keep_still.call_count == 25, f"Expected 25 keep_still calls, got {entity1.keep_still.call_count}"
        assert entity2.keep_still.call_count == 0

        results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e, traceback.format_exc())


def test_8d_max_retries_constant():
    """Test 8d: MAX_RETRIES_PER_SUBTASK constant"""
    test_name = "Test 8d: MAX_RETRIES_PER_SUBTASK constant"
    try:
        assert isinstance(CheckpointManager.MAX_RETRIES_PER_SUBTASK, int)
        assert CheckpointManager.MAX_RETRIES_PER_SUBTASK > 0
        assert CheckpointManager.MAX_RETRIES_PER_SUBTASK <= 100
        results.record_pass(test_name, f"MAX_RETRIES_PER_SUBTASK = {CheckpointManager.MAX_RETRIES_PER_SUBTASK}")
    except Exception as e:
        results.record_fail(test_name, e)


def test_8e_pause_halts_sim_not_zero_actions():
    """Test 8e: Pause halts sim loop (not zero actions)"""
    test_name = "Test 8e: Pause halts sim loop (not zero actions)"
    try:
        policy = make_mock_policy()
        controller = VLAPolicyController(policy)

        pause_vla(controller)

        env = make_mock_env()
        robot = make_mock_robot()

        started = threading.Event()
        finished = threading.Event()

        def sim_loop():
            started.set()
            controller.check_and_pause(env, robot, "test")
            finished.set()

        t = threading.Thread(target=sim_loop, daemon=True)
        t.start()
        started.wait(timeout=2.0)
        time.sleep(0.3)

        assert not finished.is_set(), (
            "CRITICAL: check_and_pause returned immediately while paused! "
            "The pause mechanism should BLOCK the sim loop, not allow it to continue."
        )

        resume_vla(controller)
        finished.wait(timeout=2.0)
        t.join(timeout=2.0)

        results.record_pass(test_name, "Pause blocks sim loop thread entirely")
    except Exception as e:
        results.record_fail(test_name, e, traceback.format_exc())


def test_8f_data_recorder_record_step_torch_tensor():
    """Test 8f: DataRecorder.record_step with FakeTensor"""
    test_name = "Test 8f: DataRecorder.record_step with FakeTensor"
    try:
        import torch as th

        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = DataRecorder(
                output_folder=tmpdir,
                task_name="test_task",
                task_id=1,
                demo_id=0,
                camera_names={},
                record_rgb=False,
                record_depth=False,
            )
            recorder.start_episode()

            action = th.Tensor([0.1, 0.2, 0.3])
            proprio = th.Tensor([1.0, 2.0])
            cam_rel = th.Tensor([0.0])

            recorder.record_step(
                action=action,
                proprio=proprio,
                cam_rel_poses=cam_rel,
                task_info={"step": 0},
                obs={},
            )

            assert recorder._step_count == 1
            step = recorder._step_data[0]
            assert step["action"] == [0.1, 0.2, 0.3]
            assert step["proprio"] == [1.0, 2.0]

            results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e, traceback.format_exc())


def test_8g_data_recorder_record_step_none_values():
    """Test 8g: DataRecorder.record_step with None values"""
    test_name = "Test 8g: DataRecorder.record_step with None values"
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = DataRecorder(
                output_folder=tmpdir,
                task_name="test_task",
                task_id=1,
                demo_id=0,
                camera_names={},
                record_rgb=False,
                record_depth=False,
            )
            recorder.start_episode()

            recorder.record_step(
                action=None,
                proprio=None,
                cam_rel_poses=None,
                task_info={"step": 0},
                obs={},
            )

            assert recorder._step_count == 1
            step = recorder._step_data[0]
            assert step["action"] == []
            assert step["proprio"] == []
            assert step["cam_rel_poses"] == []

            results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e, traceback.format_exc())


def test_8h_multiple_pause_resume_cycles():
    """Test 8h: Multiple pause/resume cycles"""
    test_name = "Test 8h: Multiple pause/resume cycles"
    try:
        policy = make_mock_policy()
        controller = VLAPolicyController(policy)

        for i in range(5):
            result = pause_vla(controller)
            assert result["status"] == "paused"
            assert not controller._run_event.is_set()

            result = resume_vla(controller)
            assert result["status"] == "running"
            assert controller._run_event.is_set()

        results.record_pass(test_name, "5 cycles completed without issues")
    except Exception as e:
        results.record_fail(test_name, e)


def test_8i_double_pause():
    """Test 8i: Double pause"""
    test_name = "Test 8i: Double pause"
    try:
        policy = make_mock_policy()
        controller = VLAPolicyController(policy)

        pause_vla(controller)
        pause_vla(controller)
        assert not controller._run_event.is_set()

        resume_vla(controller)
        assert controller._run_event.is_set()

        results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e)


def test_8j_double_resume():
    """Test 8j: Double resume"""
    test_name = "Test 8j: Double resume"
    try:
        policy = make_mock_policy()
        controller = VLAPolicyController(policy)

        resume_vla(controller)
        resume_vla(controller)
        assert controller._run_event.is_set()

        results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e)


def test_8k_save_failure_data_multiple_subtasks():
    """Test 8k: save_failure_data multiple subtasks"""
    test_name = "Test 8k: save_failure_data multiple subtasks"
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(
                output_folder=tmpdir,
                task_name="test_task",
                task_id=1,
                demo_id=0,
            )
            recorder = MagicMock()
            tracker = MagicMock()
            tracker.get_results.return_value = {}
            mgr.setup(recorder, tracker)

            env = make_mock_env()
            mgr._checkpoint_scene_state = {"objects": "state"}

            save_failure_data(mgr, env, make_mock_robot(), "subtask_1")
            save_failure_data(mgr, env, make_mock_robot(), "subtask_1")
            assert mgr.get_retry_count("subtask_1") == 2

            save_failure_data(mgr, env, make_mock_robot(), "subtask_2")
            assert mgr.get_retry_count("subtask_2") == 1
            assert mgr.get_retry_count("subtask_1") == 2

            results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e, traceback.format_exc())


def test_8l_snapshot_json_serializable():
    """Test 8l: Snapshot JSON-serializable"""
    test_name = "Test 8l: Snapshot JSON-serializable"
    try:
        env = make_mock_env()
        robot = make_mock_robot()
        snapshot = get_simulation_snapshot(env, robot, "test_task")

        json_str = json.dumps(snapshot)
        assert len(json_str) > 0

        results.record_pass(test_name, f"JSON output: {len(json_str)} chars")
    except (TypeError, ValueError) as e:
        results.record_fail(test_name, f"JSON serialization failed: {e}",
                           "Ensure all numpy arrays / torch tensors are converted to lists")
    except Exception as e:
        results.record_fail(test_name, e, traceback.format_exc())


def test_8m_filter_information_returns_new_dict():
    """Test 8m: filter_information returns new dict"""
    test_name = "Test 8m: filter_information returns new dict"
    try:
        snapshot = {
            "images": {},
            "robot_state": {},
            "object_states": {"apple.n.01_1": {}, "table.n.01_1": {}},
            "bddl_state": [],
            "step_count": 0,
            "task_name": "",
        }
        original_objects = dict(snapshot["object_states"])

        filtered = filter_information(snapshot, ["apple"])
        assert filtered is not snapshot
        assert snapshot["object_states"] == original_objects

        results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, e)


# ======================================================================
# Run all tests
# ======================================================================

def run_all_tests():
    """Execute all test functions and print the summary."""
    test_functions = [
        # Test 1: Import verification
        test_1_imports,
        test_1b_package_init_imports,
        test_1c_all_attribute,
        # Test 2: BDDLStateTracker
        test_2a_bddl_tracker_init,
        test_2b_bddl_tracker_start,
        test_2c_bddl_tracker_step,
        test_2d_bddl_tracker_grasp_change,
        test_2e_bddl_tracker_get_results,
        test_2f_bddl_tracker_no_goals,
        test_2g_bddl_tracker_no_object_scope,
        # Test 3: DataRecorder
        test_3a_data_recorder_init,
        test_3b_data_recorder_episode_tag,
        test_3c_data_recorder_resolution,
        test_3d_data_recorder_start_episode,
        test_3e_data_recorder_record_step,
        test_3f_data_recorder_record_step_not_recording,
        test_3g_data_recorder_end_episode,
        test_3h_data_recorder_end_episode_not_recording,
        test_3i_data_recorder_no_bddl_transitions,
        # Test 4: VLAPolicyController + pause/resume
        test_4a_controller_init,
        test_4b_pause_resume,
        test_4c_check_and_pause_running,
        test_4d_check_and_pause_blocks,
        test_4e_forward_delegation,
        test_4f_reset_delegation,
        test_4g_language_instruction,
        # Test 5: information_tools
        test_5a_get_simulation_snapshot_no_obs,
        test_5b_get_simulation_snapshot_with_obs,
        test_5b2_get_simulation_snapshot_without_images,
        test_5c_get_simulation_snapshot_error_handling,
        test_5d_filter_information_basic,
        test_5e_filter_information_empty_filter,
        test_5f_filter_information_multiple_names,
        test_5g_filter_information_without_images,
        test_5h_filter_information_normalized_type_names,
        test_5i_rollout_subtask_object_refs_strip_ids,
        test_5j_check_placement_success_normalized_delta,
        test_5k_check_placement_success_no_static_same_type_false_positive,
        test_5l_object_matching_shared_query,
        test_5m_check_navigation_success_normalized_instance_target,
        test_5n_check_grasp_success_normalized_instance_target,
        test_5o_evaluate_subtask_status_rollout_safe_subtask_info,
        test_5p_check_placement_success_allows_other_same_type_grasped,
        test_5q_filter_information_preserves_robot_predicates_and_deltas,
        test_5r_check_placement_success_exact_relation,
        test_5s_check_placement_success_exact_delta,
        test_5t_check_placement_success_blocks_when_matched_object_still_grasped,
        # Test 6: control_tools
        test_6a_reset_vla,
        test_6b_switch_vla_instruction_only,
        test_6c_switch_vla_with_host,
        test_6d_switch_vla_host_without_port,
        # Test 7: data_tools
        test_7a_checkpoint_manager_init,
        test_7b_checkpoint_manager_setup,
        test_7c_checkpoint_manager_capture,
        test_7d_checkpoint_manager_retry_counting,
        test_7e_should_end_episode_property,
        test_7f_segment_output_folder,
        test_7g_create_segment_recorder,
        test_7h_save_success_data,
        test_7i_save_success_data_with_switch,
        test_7j_save_failure_data,
        test_7k_save_failure_data_max_retries,
        test_7l_save_failure_no_checkpoint,
        test_7m_save_failure_with_controller_reset,
        test_7n_save_failure_no_data_recorder,
        test_7p_rebase_retry_start_step,
        test_7o_save_success_no_data_recorder,
        # Test 8: Cross-module integration
        test_8a_sim_run_syntax,
        test_8b_sim_run_import,
        test_8c_stabilise_scene,
        test_8d_max_retries_constant,
        test_8e_pause_halts_sim_not_zero_actions,
        test_8f_data_recorder_record_step_torch_tensor,
        test_8g_data_recorder_record_step_none_values,
        test_8h_multiple_pause_resume_cycles,
        test_8i_double_pause,
        test_8j_double_resume,
        test_8k_save_failure_data_multiple_subtasks,
        test_8l_snapshot_json_serializable,
        test_8m_filter_information_returns_new_dict,
    ]

    print(f"Running {len(test_functions)} tests...")
    print()

    for i, test_fn in enumerate(test_functions, 1):
        test_label = test_fn.__doc__ or test_fn.__name__
        try:
            test_fn()
            print(f"  [{i:2d}/{len(test_functions)}] PASS: {test_label}")
        except Exception as exc:
            results.record_fail(test_label, str(exc), traceback.format_exc())
            print(f"  [{i:2d}/{len(test_functions)}] FAIL: {test_label} -- {exc}")

    print(results.summary())
    return len(results.failed)


if __name__ == "__main__":
    n_failures = run_all_tests()
    sys.exit(n_failures)
