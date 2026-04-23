"""
Control tools for the VLA policy in the agentic data collection system.

The "agent" is Claude Code (a separate process), NOT a Python thread.
The simulation runs in its own process. The pause mechanism:
  1. Stops the simulation loop from stepping entirely (does NOT send zero actions).
  2. Caches a snapshot of the state before blocking.
  3. Blocks the sim loop thread until resumed.
"""

import logging
import threading
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

__all__ = [
    "VLAPolicyController",
    "pause_vla",
    "resume_vla",
    "reset_vla",
    "step_zero_actions",
    "switch_vla",
]


class VLAPolicyController:
    """Wraps an existing VLA policy without modifying it.

    Uses a :class:`threading.Event` to implement pause/resume semantics.
    The event being *set* means the simulation loop is allowed to run;
    *cleared* means it must block.

    ``check_and_pause`` is intended to be called inside the simulation loop
    **before** each ``env.step()`` call.  When the controller is paused the
    sim loop blocks here — it never reaches ``env.step()``.
    """

    def __init__(self, policy: Any) -> None:
        self._policy = policy
        self._run_event = threading.Event()
        # Start paused — the sim loop blocks until the decision agent
        # calls resume_vla via MCP, ensuring the agent is initialized
        # and observing before the robot begins moving.
        self._cached_snapshot: Optional[Dict[str, Any]] = None
        self._language_instruction: Optional[str] = None

        # Latest obs/snapshot for async observation (no pause required).
        # The main sim loop writes _latest_obs on every step; the MCP
        # thread reads it to build snapshots on demand.
        self._latest_obs_lock = threading.Lock()
        self._latest_obs: Optional[Dict[str, Any]] = None
        self._latest_snapshot: Optional[Dict[str, Any]] = None

        # Main-thread restore handoff.  Isaac Sim physics operations
        # (load_state, step_physics) must run on the main thread.  The
        # MCP thread sets _pending_restore and wakes the main thread;
        # the main thread executes the restore and signals _restore_done.
        self._restore_lock = threading.Lock()
        self._pending_restore: Optional[Dict[str, Any]] = None
        self._restore_done = threading.Event()
        self._restore_result: Optional[Dict[str, Any]] = None

        # Main-thread paused-sim stabilization handoff. This advances the
        # simulator with zero actions for a few steps, refreshes cached
        # observations / images, then re-pauses so callers can inspect the
        # restored state before resuming.
        self._zero_step_lock = threading.Lock()
        self._pending_zero_steps: Optional[Dict[str, Any]] = None
        self._zero_step_done = threading.Event()
        self._zero_step_result: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Sim-loop hook
    # ------------------------------------------------------------------

    def check_and_pause(
        self, env: Any, robot: Any, task_name: str, obs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Called by the sim loop each iteration **before** stepping.

        * If paused: caches a simulation snapshot, then blocks until resumed.
        * If running: returns immediately (no-op).

        Args:
            env: The OmniGibson environment.
            robot: The robot entity.
            task_name: Current task name.
            obs: Optional flattened observation dict (passed through to
                ``get_simulation_snapshot`` so the cached snapshot includes
                camera images).
        """
        if not self._run_event.is_set():
            # Lazy import to avoid circular dependencies — information_tools
            # may import symbols from this module or from shared packages.
            from omnigibson.learning.embodiedClaw.tools.information_tools import (
                get_simulation_snapshot,
            )

            log.info("Simulation paused — caching snapshot and blocking sim loop.")
            self._cached_snapshot = get_simulation_snapshot(env, robot, task_name, obs=obs)
            self._run_event.wait()  # blocks until resume is called

            while True:
                handled_request = False

                # ------------------------------------------------------
                # Main-thread restore handoff: if the MCP thread requested
                # a scene restore while we were paused, execute it here.
                # ------------------------------------------------------
                with self._restore_lock:
                    restore_req = self._pending_restore
                    self._pending_restore = None

                if restore_req is not None:
                    import omnigibson as og

                    handled_request = True
                    log.info("Executing pending scene restore on main thread.")
                    try:
                        restore_req["env"].scene.load_state(
                            restore_req["scene_state"], serialized=False,
                        )
                        # Stabilise: 25 physics steps with keep_still (matches eval.py)
                        for _ in range(25):
                            og.sim.step_physics()
                            for entity in restore_req["env"].task.object_scope.values():
                                if not entity.is_system and entity.exists:
                                    entity.keep_still()
                        target_step = restore_req.get("step_idx")
                        if target_step is not None and hasattr(restore_req["env"], "_current_step"):
                            restore_req["env"]._current_step = int(target_step)
                        self.reset()  # clear stale policy state
                        with self._restore_lock:
                            self._restore_result = {"status": "restored"}
                        log.info("Scene restore completed on main thread.")
                    except Exception as exc:
                        log.exception("Scene restore failed on main thread.")
                        with self._restore_lock:
                            self._restore_result = {"status": "error", "error": str(exc)}

                    # Signal the MCP thread that restore is done.
                    self._restore_done.set()

                    # Re-pause so the agent can observe the restored state
                    # before the sim continues stepping.
                    self._run_event.clear()
                    self._cached_snapshot = get_simulation_snapshot(env, robot, task_name)
                    log.info("Re-paused after restore — waiting for agent to resume.")
                    self._run_event.wait()  # blocks until agent calls resume_vla
                    continue

                # ------------------------------------------------------
                # Paused-sim stabilization handoff: advance the sim a few
                # steps with zero actions after a restore or before a
                # post-pause inspection, then re-pause with a fresh cached
                # snapshot.
                # ------------------------------------------------------
                with self._zero_step_lock:
                    zero_step_req = self._pending_zero_steps
                    self._pending_zero_steps = None

                if zero_step_req is not None:
                    handled_request = True
                    log.info(
                        "Executing %d pending zero-action step(s) on main thread.",
                        zero_step_req["num_steps"],
                    )
                    try:
                        final_obs = None
                        final_step_count = int(getattr(env, "_current_step", 0))
                        for _ in range(zero_step_req["num_steps"]):
                            zero_action = _build_zero_action(robot)
                            raw_obs, _, _, _, _ = env.step(
                                zero_action,
                                n_render_iterations=1,
                            )
                            final_obs = _preprocess_obs_for_controller(
                                env=env,
                                robot=robot,
                                task_name=task_name,
                                obs=raw_obs,
                            )
                            self.update_latest_obs(final_obs)
                            final_step_count = int(getattr(env, "_current_step", final_step_count))

                        if final_obs is None:
                            final_snapshot = get_simulation_snapshot(env, robot, task_name)
                        else:
                            final_snapshot = get_simulation_snapshot(
                                env,
                                robot,
                                task_name,
                                obs=final_obs,
                            )

                        self._cached_snapshot = final_snapshot
                        with self._latest_obs_lock:
                            self._latest_snapshot = final_snapshot
                        with self._zero_step_lock:
                            self._zero_step_result = {
                                "status": "stepped_zero_actions",
                                "num_steps": zero_step_req["num_steps"],
                                "start_step_count": zero_step_req["start_step_count"],
                                "final_step_count": final_step_count,
                                "images_cached": bool(final_snapshot.get("images")),
                                "snapshot_ready": True,
                            }
                    except Exception as exc:
                        log.exception("Zero-action debug stepping failed on main thread.")
                        with self._zero_step_lock:
                            self._zero_step_result = {"status": "error", "error": str(exc)}

                    self._zero_step_done.set()
                    self._run_event.clear()
                    log.info("Re-paused after zero-action stepping — waiting for agent to resume.")
                    self._run_event.wait()  # blocks until agent calls resume_vla
                    continue

                if not handled_request:
                    break

            self._cached_snapshot = None
            log.info("Simulation resumed — sim loop unblocked.")

    # ------------------------------------------------------------------
    # Async observation (latest obs cache)
    # ------------------------------------------------------------------

    def request_restore(
        self,
        env: Any,
        scene_state: Dict[str, Any],
        step_idx: Optional[int] = None,
        timeout: float = 60.0,
    ) -> Dict[str, Any]:
        """Request a scene restore executed on the main sim thread.

        Called from the MCP thread while the sim is paused.  Wakes the
        main thread to perform ``env.scene.load_state`` + stabilisation,
        then re-pauses so the agent can observe the restored state.

        Args:
            env: The OmniGibson environment.
            scene_state: Scene state dict (from ``dump_state(serialized=False)``).
            step_idx: Optional checkpoint step to restore into ``env._current_step``.
            timeout: Max seconds to wait for the main thread.

        Returns:
            Status dict, e.g. ``{"status": "restored"}`` or ``{"status": "error", ...}``.
        """
        with self._restore_lock:
            self._restore_done.clear()
            self._restore_result = None
            self._pending_restore = {
                "env": env,
                "scene_state": scene_state,
                "step_idx": step_idx,
            }

        # Wake the main thread (blocked on _run_event.wait() inside check_and_pause)
        self._run_event.set()

        # Wait for the main thread to complete the restore
        if not self._restore_done.wait(timeout=timeout):
            log.error("request_restore: timed out after %ss", timeout)
            with self._restore_lock:
                self._pending_restore = None
                return {"status": "error", "error": "timeout waiting for main thread"}

        with self._restore_lock:
            result = self._restore_result
            self._restore_result = None

        return result or {"status": "error", "error": "no result"}

    def request_zero_action_steps(
        self,
        env: Any,
        robot: Any,
        task_name: str,
        num_steps: int,
        timeout: float = 60.0,
    ) -> Dict[str, Any]:
        """Request a short paused stabilization advance with zero actions.

        Called from the MCP thread while the sim is paused. Wakes the
        main thread, performs ``num_steps`` regular ``env.step`` calls
        with zero action, then re-pauses so the caller can inspect the
        resulting snapshot.

        This path is intended for restore stabilization and post-pause
        inspection. It does not route through the normal VLA policy
        action path and must be called while paused.
        """
        if num_steps <= 0:
            return {"status": "error", "error": "num_steps must be positive"}
        if self._run_event.is_set():
            return {
                "status": "error",
                "error": "step_zero_actions must be called while the simulator is paused",
            }

        with self._zero_step_lock:
            self._zero_step_done.clear()
            self._zero_step_result = None
            self._pending_zero_steps = {
                "num_steps": int(num_steps),
                "start_step_count": int(getattr(env, "_current_step", 0)),
            }

        self._run_event.set()

        if not self._zero_step_done.wait(timeout=timeout):
            log.error("request_zero_action_steps: timed out after %ss", timeout)
            with self._zero_step_lock:
                self._pending_zero_steps = None
                return {"status": "error", "error": "timeout waiting for main thread"}

        with self._zero_step_lock:
            result = self._zero_step_result
            self._zero_step_result = None

        return result or {"status": "error", "error": "no result"}

    # ------------------------------------------------------------------

    def update_latest_obs(self, obs: Dict[str, Any]) -> None:
        """Cache the latest preprocessed obs from the sim loop.

        Called by :meth:`AgenticEvaluator.step` after every env step so
        that the MCP server can build snapshots without pausing.

        Thread-safe: the main sim thread writes, MCP threads read.

        Args:
            obs: The preprocessed (flattened) observation dict.
        """
        with self._latest_obs_lock:
            self._latest_obs = obs
            # Invalidate any previously computed lazy snapshot so the
            # next read recomputes from the fresh obs.
            self._latest_snapshot = None

    def get_latest_obs(self) -> Optional[Dict[str, Any]]:
        """Return a reference to the latest cached obs (thread-safe read).

        Returns:
            The most recent obs dict, or *None* if no step has occurred yet.
        """
        with self._latest_obs_lock:
            return self._latest_obs

    def get_latest_snapshot(
        self,
        env: Any,
        robot: Any,
        task_name: str,
        include_images: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Return a snapshot built from the latest cached obs.

        Computes the snapshot lazily: if the cached snapshot is still
        valid (i.e. ``_latest_obs`` hasn't changed since last compute),
        returns the cached version.  Otherwise recomputes.

        Args:
            env: The OmniGibson environment (for BDDL state, object states).
            robot: The robot entity.
            task_name: Current task name.
            include_images: Whether the returned snapshot should include
                encoded image payloads.

        Returns:
            A snapshot dict identical in structure to
            :func:`get_simulation_snapshot`, or *None* if no obs is
            available yet.
        """
        cached_snapshot = None
        with self._latest_obs_lock:
            obs = self._latest_obs
            if obs is None:
                return None
            if self._latest_snapshot is not None:
                if include_images:
                    return self._latest_snapshot
                cached_snapshot = self._latest_snapshot

        if cached_snapshot is not None:
            from omnigibson.learning.embodiedClaw.tools.information_tools import (
                strip_snapshot_images,
            )
            return strip_snapshot_images(cached_snapshot)

        # Compute snapshot outside the lock (obs is an immutable
        # reference at this point; the sim may overwrite _latest_obs
        # concurrently, but we're reading a consistent dict).
        from omnigibson.learning.embodiedClaw.tools.information_tools import (
            get_simulation_snapshot,
        )
        snapshot = get_simulation_snapshot(
            env,
            robot,
            task_name,
            obs=obs,
            include_images=include_images,
        )

        if include_images:
            with self._latest_obs_lock:
                self._latest_snapshot = snapshot
        return snapshot

    # ------------------------------------------------------------------
    # Cached state (pause-time snapshot)
    # ------------------------------------------------------------------

    @property
    def cached_snapshot(self) -> Optional[Dict[str, Any]]:
        """Available while paused — captured at pause time."""
        return self._cached_snapshot

    # ------------------------------------------------------------------
    # Policy delegation
    # ------------------------------------------------------------------

    def forward(self, obs: Dict[str, Any]) -> Any:
        """Delegate to the wrapped policy's ``forward``."""
        return self._policy.forward(obs=obs)

    def reset(self) -> None:
        """Reset the wrapped policy."""
        self._policy.reset()

    # ------------------------------------------------------------------
    # Language instruction
    # ------------------------------------------------------------------

    @property
    def language_instruction(self) -> Optional[str]:
        return self._language_instruction

    @language_instruction.setter
    def language_instruction(self, value: str) -> None:
        self._language_instruction = value


def _build_zero_action(robot: Any) -> Any:
    """Return a zero action vector matching the robot's action dimension."""
    import torch as th

    return th.zeros(robot.action_dim, dtype=th.float32)


def _preprocess_obs_for_controller(
    env: Any,
    robot: Any,
    task_name: str,
    obs: Dict[str, Any],
) -> Dict[str, Any]:
    """Mirror AgenticEvaluator observation preprocessing for debug stepping."""
    import numpy as np
    import omnigibson.utils.transform_utils as T
    import torch as th

    from omnigibson.learning.utils.eval_utils import (
        ROBOT_CAMERA_NAMES,
        TASK_NAMES_TO_INDICES,
        flatten_obs_dict,
    )

    obs = flatten_obs_dict(obs)
    base_pose = robot.get_position_orientation()
    cam_rel_poses = []
    for camera_name in ROBOT_CAMERA_NAMES["R1Pro"].values():
        camera = robot.sensors[camera_name.split("::")[1]]
        direct_cam_pose = camera.camera_parameters["cameraViewTransform"]
        if np.allclose(direct_cam_pose, np.zeros(16)):
            cam_rel_poses.append(
                th.cat(T.relative_pose_transform(*(camera.get_position_orientation()), *base_pose))
            )
        else:
            cam_pose = T.mat2pose(
                th.tensor(
                    np.linalg.inv(np.reshape(direct_cam_pose, [4, 4]).T),
                    dtype=th.float32,
                )
            )
            cam_rel_poses.append(th.cat(T.relative_pose_transform(*cam_pose, *base_pose)))

    obs["robot_r1::cam_rel_poses"] = th.cat(cam_rel_poses, axis=-1)
    obs["task_id"] = th.tensor([TASK_NAMES_TO_INDICES[task_name]], dtype=th.int64)
    return obs


# ======================================================================
# Standalone tool functions (callable by the external agent)
# ======================================================================


def pause_vla(controller: VLAPolicyController) -> Dict[str, str]:
    """Pause the simulation loop.

    Clears the internal run-event so the sim loop will block on its next
    call to ``controller.check_and_pause()``.  The sim loop does **not**
    send zero actions — it halts entirely.

    Args:
        controller: The :class:`VLAPolicyController` managing the loop.

    Returns:
        Status dict, e.g. ``{"status": "paused"}``.
    """
    log.info("pause_vla: clearing run event — sim loop will block on next iteration.")
    controller._run_event.clear()
    return {"status": "paused"}


def resume_vla(controller: VLAPolicyController) -> Dict[str, str]:
    """Resume the simulation loop.

    Sets the internal run-event so the sim loop unblocks and continues
    stepping the environment.

    Args:
        controller: The :class:`VLAPolicyController` managing the loop.

    Returns:
        Status dict, e.g. ``{"status": "running"}``.
    """
    log.info("resume_vla: setting run event — sim loop will unblock.")
    controller._run_event.set()
    return {"status": "running"}


def reset_vla(controller: VLAPolicyController) -> Dict[str, str]:
    """Reset the wrapped VLA policy.

    Delegates to ``controller.reset()`` which in turn calls
    ``policy.reset()`` on the underlying policy object.

    Args:
        controller: The :class:`VLAPolicyController` managing the loop.

    Returns:
        Status dict, e.g. ``{"status": "reset"}``.
    """
    log.info("reset_vla: resetting the wrapped policy.")
    controller.reset()
    return {"status": "reset"}


def step_zero_actions(
    controller: VLAPolicyController,
    env: Any,
    robot: Any,
    task_name: str,
    num_steps: int = 10,
) -> Dict[str, Any]:
    """Advance the paused simulator for a few zero-action steps, then re-pause.

    Intended for post-restore stabilization and refreshed observation
    capture. After this returns, callers can immediately call
    ``get_simulation_snapshot`` and inspect the new pause-time snapshot.
    """
    log.info(
        "step_zero_actions: requesting %d zero-action step(s) while paused.",
        num_steps,
    )
    return controller.request_zero_action_steps(
        env=env,
        robot=robot,
        task_name=task_name,
        num_steps=num_steps,
    )


def switch_vla(
    controller: VLAPolicyController,
    language_instruction: str,
    host: Optional[str] = None,
    port: Optional[int] = None,
) -> Dict[str, str]:
    """Switch the VLA policy's language instruction (and optionally its host).

    Stores the new language instruction on the controller.  If *host* and
    *port* are provided, updates the underlying ``WebsocketPolicy`` connection
    via ``policy.update_host(host, port)``.

    Args:
        controller: The :class:`VLAPolicyController` managing the loop.
        language_instruction: The new natural-language instruction for the VLA.
        host: Optional websocket host to reconnect to.
        port: Optional websocket port to reconnect to.

    Returns:
        Status dict with the new instruction.
    """
    log.info("switch_vla: setting language instruction to %r", language_instruction)
    controller.language_instruction = language_instruction

    if host is not None and port is not None:
        log.info("switch_vla: updating policy host to %s:%s", host, port)
        controller._policy.update_host(host, port)

    return {
        "status": "switched",
        "language_instruction": language_instruction,
    }
