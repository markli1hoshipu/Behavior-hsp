"""
Entry point for the embodiedClaw agentic data collection system.

Starts the OmniGibson simulator and MCP tool server together.
The MCP server runs in a background thread (SSE transport); the sim loop
runs on the main thread.  Claude Code connects to the MCP server and
controls the simulation via tool calls (pause / resume / save / snapshot).

Usage:
    cd /home/user/codebase/BEHAVIOR-1K
    DISPLAY=:1 OMNI_KIT_ACCEPT_EULA=yes conda run -n behavior python \
        OmniGibson/omnigibson/learning/embodiedClaw/run_agentic.py \
        policy=websocket task.name=picking_up_trash headless=false \
        model.host=127.0.0.1 model.port=8000 \
        log_path=/home/user/dataset/embodiedClaw_test
"""

import csv
import hydra
import json
import logging
import omnigibson as og
import os
import signal
import sys
import threading

from inspect import getsourcefile
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from omnigibson.learning.utils.config_utils import register_omegaconf_resolvers
from omnigibson.learning.utils.eval_utils import (
    ROBOT_CAMERA_NAMES,
    TASK_NAMES_TO_INDICES,
)
from omnigibson.macros import gm, create_module_macros

from omnigibson.learning.embodiedClaw.embodied_claw_sim_run import AgenticEvaluator
from omnigibson.learning.embodiedClaw.mcp_server import EmbodiedClawMCPServer
from omnigibson.learning.embodiedClaw.data_recording.data_saver import DataRecorder
from omnigibson.learning.embodiedClaw.tools.data_tools import CheckpointManager

# ---------------------------------------------------------------------------
# Module-level constants (mirrors embodied_claw_sim_run.py)
# ---------------------------------------------------------------------------
def _macro_module_path() -> Path:
    # `embodiedClaw/` may be symlinked into a runtime checkout. When this file is
    # executed as a script, `__file__` can resolve to the source checkout instead
    # of the active `omnigibson` package root, which breaks create_module_macros().
    module_path = Path(__file__).resolve()
    module_relative_path = module_path.relative_to(module_path.parents[2])
    return Path(og.__file__).resolve().parent / module_relative_path


m = create_module_macros(module_path=_macro_module_path())
m.NUM_EVAL_EPISODES = 1
m.NUM_TRAIN_INSTANCES = 200
m.NUM_EVAL_INSTANCES = 20

logger = logging.getLogger("run_agentic")
logger.setLevel(logging.INFO)

# Default MCP server address
DEFAULT_MCP_HOST = "127.0.0.1"
DEFAULT_MCP_PORT = 8001


def _resolve_instances(config: DictConfig):
    """Resolve which task instances to run (same logic as embodied_claw_sim_run.py __main__)."""
    if config.eval_on_train_instances:
        logger.info(
            "Evaluating on training instances (set eval_on_train_instances=false for test instances)."
        )
        task_idx = TASK_NAMES_TO_INDICES[config.task.name]
        with open(
            os.path.join(gm.DATA_PATH, "2025-challenge-task-instances", "metadata", "episodes.jsonl"), "r"
        ) as f:
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
        task_instance_csv_path = os.path.join(
            gm.DATA_PATH, "2025-challenge-task-instances", "metadata", "test_instances.csv"
        )
        with open(task_instance_csv_path, "r") as f:
            lines = list(csv.reader(f))[1:]
        assert (
            lines[TASK_NAMES_TO_INDICES[config.task.name]][1] == config.task.name
        ), (
            f"Task name from config {config.task.name} does not match "
            f"task name from csv {lines[TASK_NAMES_TO_INDICES[config.task.name]][1]}"
        )
        test_instances = lines[TASK_NAMES_TO_INDICES[config.task.name]][2].strip().split(",")
        num_available_instances = len(test_instances)
        instances_to_run = (
            config.eval_instance_ids
            if config.eval_instance_ids is not None
            else set(range(min(m.NUM_EVAL_INSTANCES, num_available_instances)))
        )
        assert set(instances_to_run).issubset(
            set(range(num_available_instances))
        ), f"eval instance ids must be in range({num_available_instances})"
        instances_to_run = [int(test_instances[i]) for i in instances_to_run]

    return instances_to_run


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Parse config via Hydra (same pattern as embodied_claw_sim_run.py)
    # ------------------------------------------------------------------
    register_omegaconf_resolvers()
    configs_dir = f"{Path(getsourcefile(lambda: 0)).parents[0].parent}/configs"
    with hydra.initialize_config_dir(configs_dir, version_base="1.1"):
        config = hydra.compose("base_config.yaml", overrides=sys.argv[1:])
    OmegaConf.resolve(config)

    with gm.unlocked():
        gm.HEADLESS = config.headless
        if not gm.HEADLESS:
            # Avoid GUI extension startup crashes in windowless / remote shells.
            # Users can still explicitly pass headless=false when a real display is available.
            logger.warning("Forcing headless mode because the current environment has no default window support.")
            gm.HEADLESS = True

    assert not (
        config.eval_on_train_instances and config.test_hidden
    ), "Cannot eval on train instances and test hidden instances simultaneously."
    if config.test_hidden:
        logger.info("Evaluating on hidden test instances (internal use only).")

    # Resolve task instances
    instances_to_run = _resolve_instances(config)

    # Output directories
    metrics_path = Path(config.log_path).expanduser() / "metrics"
    metrics_path.mkdir(parents=True, exist_ok=True)
    data_output_path = Path(config.log_path).expanduser() / "agentic_data"
    data_output_path.mkdir(parents=True, exist_ok=True)

    task_id = TASK_NAMES_TO_INDICES[config.task.name]
    camera_names = ROBOT_CAMERA_NAMES["R1Pro"]

    # Read MCP host/port from config (fall back to defaults)
    mcp_host = getattr(config, "mcp_host", None) or DEFAULT_MCP_HOST
    mcp_port = getattr(config, "mcp_port", None) or DEFAULT_MCP_PORT

    # ------------------------------------------------------------------
    # 2. Create AgenticEvaluator (starts sim)
    # ------------------------------------------------------------------
    with AgenticEvaluator(config) as evaluator:

        # --------------------------------------------------------------
        # 3. Create MCP server (binds to live sim objects)
        # --------------------------------------------------------------
        # Read dataset root from config or use default
        dataset_root = getattr(config, "dataset_root", None) or "/home/user/dataset"

        mcp_server = EmbodiedClawMCPServer(
            env=evaluator.env,
            robot=evaluator.robot,
            task_name=config.task.name,
            controller=evaluator.controller,
            checkpoint_mgr=None,  # set per episode below
            host=mcp_host,
            port=mcp_port,
            dataset_root=dataset_root,
            task_id=task_id,
        )

        # --------------------------------------------------------------
        # 4. Start MCP server in background thread (SSE transport)
        # --------------------------------------------------------------
        mcp_thread = threading.Thread(
            target=mcp_server.run_sse,
            name="mcp-sse-server",
            daemon=True,
        )
        mcp_thread.start()

        logger.info("")
        logger.info("=" * 60)
        logger.info("  MCP server running at http://%s:%s/sse", mcp_host, mcp_port)
        logger.info("  Connect Claude Code with:")
        logger.info('    mcp_url = "http://%s:%s/sse"', mcp_host, mcp_port)
        logger.info("=" * 60)
        logger.info("")

        # --------------------------------------------------------------
        # 5. Graceful shutdown handler
        # --------------------------------------------------------------
        _original_sigint = signal.getsignal(signal.SIGINT)

        def _shutdown_handler(sig, frame):
            logger.warning("SIGINT received — shutting down MCP server and sim.")
            # The AgenticEvaluator context manager handles env.close / og.shutdown
            # via __exit__, so we just need to propagate the signal.
            if callable(_original_sigint) and _original_sigint is not signal.SIG_DFL:
                _original_sigint(sig, frame)
            else:
                sys.exit(0)

        signal.signal(signal.SIGINT, _shutdown_handler)

        # --------------------------------------------------------------
        # 6. Per-episode loop (driven by agent via MCP)
        # --------------------------------------------------------------
        logger.info("Starting agentic data collection...")

        for idx in instances_to_run:
            evaluator.reset()
            evaluator.load_task_instance(idx, test_hidden=config.test_hidden)
            logger.info("Loaded task instance %s.", idx)

            for epi in range(m.NUM_EVAL_EPISODES):
                evaluator.reset()
                done = False

                # -- episode-level setup --
                episode_output = str(data_output_path / f"{config.task.name}_{idx}_{epi}")

                # DataRecorder
                evaluator.data_recorder = DataRecorder(
                    output_folder=episode_output,
                    task_name=config.task.name,
                    task_id=task_id,
                    demo_id=epi,
                    camera_names=camera_names,
                    record_rgb=True,
                    record_depth=True,
                )

                # CheckpointManager
                evaluator.checkpoint_mgr = CheckpointManager(
                    output_folder=episode_output,
                    task_name=config.task.name,
                    task_id=task_id,
                    demo_id=epi,
                    record_rgb=True,
                    record_depth=True,
                )
                evaluator.checkpoint_mgr.setup(evaluator.data_recorder, evaluator.bddl_tracker)

                # Expose checkpoint manager to MCP server
                mcp_server.checkpoint_mgr = evaluator.checkpoint_mgr

                # Reset task memory for the new episode
                mcp_server.reset_task_memory()

                # Start BDDL tracker
                evaluator.bddl_tracker.start(evaluator.env, evaluator.robot)

                # Start recording and capture initial pre-action state (N+1 contract)
                evaluator.data_recorder.start_episode()
                evaluator.data_recorder.record_state(og.sim.dump_state(serialized=True))

                # Capture initial checkpoint
                evaluator.checkpoint_mgr.capture_checkpoint(evaluator.env, 0)

                # Run metric start callbacks
                for metric in evaluator.metrics:
                    metric.start_callback(evaluator.env)

                logger.info(
                    "Episode %d of instance %s started. Sim loop running — "
                    "agent controls via MCP tools.",
                    epi, idx,
                )

                # ----------------------------------------------------------
                # Main simulation loop (agent controls via MCP pause/resume)
                # ----------------------------------------------------------
                while not done:
                    terminated, truncated, info = evaluator.step()
                    if terminated or truncated:
                        done = True
                    if evaluator._step_count % 1000 == 0:
                        logger.info("Step %d", evaluator._step_count)

                # ----------------------------------------------------------
                # Episode teardown
                # ----------------------------------------------------------
                success = evaluator.n_success_trials > 0

                bddl_results = evaluator.bddl_tracker.get_results()
                evaluator.data_recorder.end_episode(
                    success=success,
                    env=evaluator.env,
                    bddl_transitions=bddl_results,
                )

                for metric in evaluator.metrics:
                    metric.end_callback(evaluator.env)

                logger.info("Episode finished at step %d.", evaluator._step_count)
                logger.info(
                    "Exit state: terminated=%s, truncated=%s, success=%s",
                    terminated, truncated, success,
                )

                # Gather and save metrics
                metrics = {}
                for metric in evaluator.metrics:
                    metrics.update(metric.gather_results())
                with open(metrics_path / f"{config.task.name}_{idx}_{epi}.json", "w") as f:
                    json.dump(metrics, f)

                # Save BDDL results
                bddl_output_path = metrics_path / f"{config.task.name}_{idx}_{epi}_bddl.json"
                with open(bddl_output_path, "w") as f:
                    json.dump(bddl_results, f, indent=2)
                logger.info("Saved BDDL results to %s", bddl_output_path)

                # Clean up episode-level references
                evaluator.data_recorder = None
                evaluator.checkpoint_mgr = None
                mcp_server.checkpoint_mgr = None

        logger.info("All instances complete.")


if __name__ == "__main__":
    main()
