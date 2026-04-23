"""
embodiedClaw — Agentic data collection tools for BEHAVIOR-1K.

Re-exports the public API from subpackages so callers can do:

    from omnigibson.learning.embodiedClaw import (
        get_simulation_snapshot, filter_information,
        VLAPolicyController, pause_vla, resume_vla, reset_vla, switch_vla,
        CheckpointManager, save_success_data, save_failure_data, save_episode_data,
        EmbodiedClawMCPServer,
        # Stage 3: Decision Module
        SubtaskAnnotation, EpisodeAnnotation, SubtaskDurationStats,
        load_episode_annotation, load_all_annotations,
        compute_subtask_duration_stats,
        TaskMemory,
        evaluate_subtask_status,
        # GT Replay Testing
        DummyPolicy, ReplayEvaluator,
        SubtaskResult, EpisodeResult, compute_accuracy_metrics,
        run_offline_accuracy_test, run_gt_replay_accuracy_test,
        run_multi_episode_test,
    )
"""

# Information tools
from omnigibson.learning.embodiedClaw.tools.information_tools import (
    get_simulation_snapshot,
    filter_information,
)

# Control tools
from omnigibson.learning.embodiedClaw.tools.control_tools import (
    VLAPolicyController,
    pause_vla,
    resume_vla,
    reset_vla,
    switch_vla,
)

# Data tools
from omnigibson.learning.embodiedClaw.tools.data_tools import (
    CheckpointManager,
    save_success_data,
    save_failure_data,
    save_episode_data,
)

# MCP server
from omnigibson.learning.embodiedClaw.mcp_server import (
    EmbodiedClawMCPServer,
)

# Annotation loader (Stage 3)
from omnigibson.learning.embodiedClaw.annotation_loader import (
    SubtaskAnnotation,
    EpisodeAnnotation,
    SubtaskDurationStats,
    load_episode_annotation,
    load_all_annotations,
    compute_subtask_duration_stats,
)

# Memory (Stage 3)
from omnigibson.learning.embodiedClaw.memory import TaskMemory

# Decision module (Stage 3)
from omnigibson.learning.embodiedClaw.decision_module import (
    evaluate_subtask_status,
)

# Demo frame extraction (Stage 3.3)
from omnigibson.learning.embodiedClaw.demo_frame_extractor import (
    DemoFrameEntry,
    DemoFrameLibrary,
    build_frame_library,
)

# GT Replay runner (Stage 3.6)
from omnigibson.learning.embodiedClaw.gt_replay_runner import (
    DummyPolicy,
    ReplayEvaluator,
)

# GT Replay test (Stage 3.6)
from omnigibson.learning.embodiedClaw.gt_replay_test import (
    SubtaskResult,
    EpisodeResult,
    compute_accuracy_metrics,
    run_offline_accuracy_test,
    run_gt_replay_accuracy_test,
    run_multi_episode_test,
)

__all__ = [
    # Information
    "get_simulation_snapshot",
    "filter_information",
    # Control
    "VLAPolicyController",
    "pause_vla",
    "resume_vla",
    "reset_vla",
    "switch_vla",
    # Data
    "CheckpointManager",
    "save_success_data",
    "save_failure_data",
    "save_episode_data",
    # MCP server
    "EmbodiedClawMCPServer",
    # Annotation loader (Stage 3)
    "SubtaskAnnotation",
    "EpisodeAnnotation",
    "SubtaskDurationStats",
    "load_episode_annotation",
    "load_all_annotations",
    "compute_subtask_duration_stats",
    # Memory (Stage 3)
    "TaskMemory",
    # Decision module (Stage 3)
    "evaluate_subtask_status",
    # Demo frame extraction (Stage 3.3)
    "DemoFrameEntry",
    "DemoFrameLibrary",
    "build_frame_library",
    # GT Replay runner (Stage 3.6)
    "DummyPolicy",
    "ReplayEvaluator",
    # GT Replay test (Stage 3.6)
    "SubtaskResult",
    "EpisodeResult",
    "compute_accuracy_metrics",
    "run_offline_accuracy_test",
    "run_gt_replay_accuracy_test",
    "run_multi_episode_test",
]
