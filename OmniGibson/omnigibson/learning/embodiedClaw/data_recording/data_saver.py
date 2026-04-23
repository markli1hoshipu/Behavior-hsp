"""
Data recording module for the BEHAVIOR-1K agentic data collection system.

Handles per-step recording of actions, proprioception, camera poses, and image
frames (RGB / depth) during policy rollout, as well as per-episode metadata,
BDDL state transition persistence, and HDF5 sim-state recording.

The recorder supports checkpoint-based segmentation:
- ``save_segment(start, end, ...)`` writes a slice of accumulated data (parquet
  + HDF5 + meta) to a separate output folder.  Video for segments is written
  from cached raw frames.
- ``truncate_to(index)`` discards all data after the given buffer index.  This
  is used after reverting the simulator to a checkpoint so the positive
  trajectory only contains successful steps.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import av
import h5py
import numpy as np
import pandas as pd
import torch as th

from omnigibson.learning.utils.dataset_utils import makedirs_with_mode
from omnigibson.learning.utils.eval_utils import HEAD_RESOLUTION, WRIST_RESOLUTION
from omnigibson.learning.utils.obs_utils import create_video_writer, write_video

logger = logging.getLogger(__name__)


class DataRecorder:
    """Records demonstration data (actions, observations, videos, metadata) during policy rollout.

    Per-step data (actions, proprioception, camera relative poses, task info) is
    accumulated in memory and flushed to a Parquet file at the end of each episode.
    RGB and depth frames are streamed directly to H.265 / H.264 video files via PyAV.

    Additionally, raw video frames are cached in memory so that arbitrary
    segments can be extracted and saved to separate video files (used for
    failure segments).  Call :meth:`save_segment` to write a sub-range of
    the accumulated buffers to disk.

    Args:
        output_folder: Root directory for all recorded artefacts.
        task_name: Human-readable task name (e.g. ``"picking_up_trash"``).
        task_id: Numeric task identifier.
        demo_id: Numeric demonstration identifier.
        camera_names: Mapping from short camera name (e.g. ``"head"``) to full
            OmniGibson sensor path.
        record_rgb: Whether to record RGB video streams.
        record_depth: Whether to record 16-bit depth video streams.
    """

    def __init__(
        self,
        output_folder: str,
        task_name: str,
        task_id: int,
        demo_id: int,
        camera_names: Dict[str, str],
        record_rgb: bool = True,
        record_depth: bool = True,
    ) -> None:
        self.output_folder = output_folder
        self.task_name = task_name
        self.task_id = task_id
        self.demo_id = demo_id
        self.camera_names = camera_names
        self.record_rgb = record_rgb
        self.record_depth = record_depth

        self.is_recording: bool = False
        self._step_data: List[Dict[str, Any]] = []
        self._video_writers: Dict[str, Tuple[av.container.OutputContainer, av.stream.Stream]] = {}
        self._step_count: int = 0

        # HDF5 sim-state buffers (populated per step via record_state)
        self._state_vectors: List[np.ndarray] = []
        self._max_state_size: int = 0

        # Raw frame cache for segment extraction.  Each entry is a dict
        # mapping writer_key (e.g. "head_rgb") -> numpy array.
        self._frame_cache: List[Dict[str, np.ndarray]] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def _episode_tag(self) -> str:
        """Canonical episode file-name stem, e.g. ``episode_00010002``."""
        return f"episode_{self.task_id:04d}{self.demo_id:04d}"

    @staticmethod
    def _resolution_for_camera(short_name: str) -> Tuple[int, int]:
        """Return ``(height, width)`` for a camera given its short name."""
        if short_name == "head":
            return HEAD_RESOLUTION
        # All non-head cameras are treated as wrist cameras.
        return WRIST_RESOLUTION

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def start_episode(self) -> None:
        """Prepare directories and video writers for a new episode.

        Creates the directory tree under *output_folder* and opens one video
        file per camera per enabled modality (RGB / depth).
        """
        # Create output directories.
        data_dir = f"{self.output_folder}/data/"
        videos_dir = f"{self.output_folder}/videos/"
        meta_dir = f"{self.output_folder}/meta/"
        for d in (data_dir, videos_dir, meta_dir):
            makedirs_with_mode(d)

        # Reset per-episode state.
        self._step_data = []
        self._step_count = 0
        self._video_writers = {}
        self._state_vectors = []
        self._max_state_size = 0
        self._frame_cache = []

        # Open video writers for each camera / modality pair.
        for short_name, _full_name in self.camera_names.items():
            resolution = self._resolution_for_camera(short_name)

            if self.record_rgb:
                rgb_path = f"{videos_dir}/rgb_{short_name}.mp4"
                self._video_writers[f"{short_name}_rgb"] = create_video_writer(
                    fpath=rgb_path,
                    resolution=resolution,
                    codec_name="libx264",
                    rate=30,
                    pix_fmt="yuv420p",
                )

            if self.record_depth:
                depth_path = f"{videos_dir}/depth_{short_name}.mp4"
                self._video_writers[f"{short_name}_depth"] = create_video_writer(
                    fpath=depth_path,
                    resolution=resolution,
                    codec_name="libx264",
                    rate=30,
                    pix_fmt="yuv420p",
                )

        self.is_recording = True
        logger.info("DataRecorder: started episode %s", self._episode_tag)

    # ------------------------------------------------------------------
    # Per-step recording
    # ------------------------------------------------------------------

    def record_step(
        self,
        action: Any,
        proprio: Any,
        cam_rel_poses: Any,
        task_info: Dict[str, Any],
        obs: Dict[str, Any],
    ) -> None:
        """Record a single environment step.

        Args:
            action: Action tensor executed at this step.
            proprio: Proprioceptive state tensor.
            cam_rel_poses: Camera-relative-pose tensor.
            task_info: Arbitrary task metadata dict for this step.
            obs: Flat observation dictionary keyed by
                ``"<full_camera_name>::rgb"`` / ``"<full_camera_name>::depth_linear"``.
        """
        if not self.is_recording:
            return

        # ----- tabular data -----
        step_record: Dict[str, Any] = {
            "step_idx": self._step_count,
            "action": action.tolist() if isinstance(action, th.Tensor) else (list(action) if action is not None else []),
            "proprio": proprio.tolist() if isinstance(proprio, th.Tensor) else (list(proprio) if proprio is not None else []),
            "cam_rel_poses": cam_rel_poses.tolist() if isinstance(cam_rel_poses, th.Tensor) else (list(cam_rel_poses) if cam_rel_poses is not None else []),
            "task_info": json.dumps(task_info) if isinstance(task_info, dict) else str(task_info),
        }
        self._step_data.append(step_record)

        # ----- video frames -----
        cached_frames: Dict[str, np.ndarray] = {}
        for short_name, full_name in self.camera_names.items():
            # RGB
            if self.record_rgb:
                rgb_key = f"{full_name}::rgb"
                if rgb_key in obs:
                    frame = obs[rgb_key]
                    if isinstance(frame, th.Tensor):
                        frame = frame.cpu().numpy()
                    # write_video expects shape (N, H, W, C)
                    if frame.ndim == 3:
                        frame = np.expand_dims(frame, axis=0)
                    writer_key = f"{short_name}_rgb"
                    # Write to main streaming writer (best-effort; may be in
                    # EOF state after a segment save).  The frame cache is the
                    # authoritative source for segment & episode video output.
                    try:
                        write_video(frame, self._video_writers[writer_key], mode="rgb")
                    except Exception:
                        pass  # writer in bad state; frame already cached below
                    cached_frames[writer_key] = frame

            # Depth
            if self.record_depth:
                depth_key = f"{full_name}::depth_linear"
                if depth_key in obs:
                    frame = obs[depth_key]
                    if isinstance(frame, th.Tensor):
                        frame = frame.cpu().numpy()
                    # write_video expects shape (N, H, W) for depth
                    if frame.ndim == 2:
                        frame = np.expand_dims(frame, axis=0)
                    writer_key = f"{short_name}_depth"
                    try:
                        write_video(frame, self._video_writers[writer_key], mode="depth")
                    except Exception:
                        pass  # writer in bad state; frame already cached below
                    cached_frames[writer_key] = frame

        self._frame_cache.append(cached_frames)
        self._step_count += 1

    def record_state(self, state_vector: Any) -> None:
        """Record a serialized sim-state vector for HDF5 output.

        Should be called once per step, typically with the result of
        ``og.sim.dump_state(serialized=True)``.

        Args:
            state_vector: 1-D tensor or array of floats representing the
                full serialized simulator state.
        """
        if not self.is_recording:
            return
        if isinstance(state_vector, th.Tensor):
            arr = state_vector.cpu().numpy().astype(np.float32)
        else:
            arr = np.asarray(state_vector, dtype=np.float32)
        self._state_vectors.append(arr)
        if len(arr) > self._max_state_size:
            self._max_state_size = len(arr)

    # ------------------------------------------------------------------
    # Segment extraction & truncation
    # ------------------------------------------------------------------

    def save_segment(
        self,
        output_folder: str,
        start_idx: int,
        end_idx: int,
        success: bool,
        segment_tag: str = "segment",
        bddl_transitions: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Save a slice of accumulated data to a separate output folder.

        Writes parquet, HDF5 (if state vectors exist), metadata JSON,
        and video files from the cached raw frames for the range
        ``[start_idx, end_idx)``.

        This does NOT modify the recorder's internal buffers -- use
        :meth:`truncate_to` afterward if needed.

        Args:
            output_folder: Directory to write the segment artefacts.
            start_idx: Start buffer index (inclusive).
            end_idx: End buffer index (exclusive).
            success: Whether to label this segment as successful.
            segment_tag: File-name stem for the saved files.
            bddl_transitions: Optional BDDL transition data.

        Returns:
            ``True`` if data was written, ``False`` on empty range.
        """
        if start_idx >= end_idx or start_idx < 0:
            logger.warning(
                "save_segment: invalid range [%d, %d), skipping",
                start_idx, end_idx,
            )
            return False

        data_dir = f"{output_folder}/data/"
        videos_dir = f"{output_folder}/videos/"
        meta_dir = f"{output_folder}/meta/"
        for d in (data_dir, videos_dir, meta_dir):
            makedirs_with_mode(d)

        seg_step_data = self._step_data[start_idx:end_idx]
        # State vectors have N+1 entries (initial state + one per action).
        # For actions [start_idx, end_idx), we need states [start_idx, end_idx].
        seg_state_vectors = self._state_vectors[start_idx:end_idx + 1]
        seg_frames = self._frame_cache[start_idx:end_idx]

        # ----- Parquet -----
        if seg_step_data:
            parquet_path = f"{data_dir}/{segment_tag}.parquet"
            df = pd.DataFrame(seg_step_data)
            df.to_parquet(parquet_path, index=False)
            logger.info(
                "save_segment: saved %d steps to %s", len(df), parquet_path,
            )

        # ----- HDF5 sim state -----
        if seg_state_vectors:
            hdf5_path = f"{data_dir}/{segment_tag}.hdf5"
            max_sz = max(len(sv) for sv in seg_state_vectors)
            n_states = len(seg_state_vectors)
            padded = np.zeros((n_states, max_sz), dtype=np.float32)
            sizes = np.zeros(n_states, dtype=np.int64)
            for i, sv in enumerate(seg_state_vectors):
                padded[i, : len(sv)] = sv
                sizes[i] = len(sv)
            actions_list = [
                row.get("action", []) for row in seg_step_data
            ]
            with h5py.File(hdf5_path, "w") as hf:
                grp = hf.create_group("data/demo_0")
                grp.create_dataset("state", data=padded)
                grp.create_dataset("state_size", data=sizes)
                if actions_list:
                    action_arr = np.array(actions_list, dtype=np.float32)
                    grp.create_dataset("action", data=action_arr)
            logger.info(
                "save_segment: saved %d state vectors to %s (max_size=%d)",
                n_states, hdf5_path, max_sz,
            )

        # ----- Video from cached frames -----
        if seg_frames:
            # Determine which writer keys are present
            all_keys: set = set()
            for fc in seg_frames:
                all_keys.update(fc.keys())

            for wkey in sorted(all_keys):
                is_depth = wkey.endswith("_depth")
                short_name = wkey.rsplit("_", 1)[0]
                resolution = self._resolution_for_camera(short_name)
                if is_depth:
                    vpath = f"{videos_dir}/{wkey}.mp4"
                    vwriter = create_video_writer(
                        fpath=vpath,
                        resolution=resolution,
                        codec_name="libx264",
                        rate=30,
                        pix_fmt="yuv420p",
                    )
                    mode = "depth"
                else:
                    vpath = f"{videos_dir}/{wkey}.mp4"
                    vwriter = create_video_writer(
                        fpath=vpath, resolution=resolution,
                        codec_name="libx264", rate=30, pix_fmt="yuv420p",
                    )
                    mode = "rgb"

                container, stream = vwriter
                for fc in seg_frames:
                    if wkey in fc:
                        write_video(fc[wkey], vwriter, mode=mode)
                # Close writer
                try:
                    for packet in stream.encode():
                        container.mux(packet)
                    container.close()
                except Exception:
                    logger.exception(
                        "save_segment: error closing video writer '%s'", wkey,
                    )

        # ----- Metadata JSON -----
        metadata: Dict[str, Any] = {
            "task_name": self.task_name,
            "task_id": self.task_id,
            "demo_id": self.demo_id,
            "success": success,
            "n_steps": end_idx - start_idx,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        meta_path = f"{meta_dir}/{segment_tag}.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("save_segment: saved metadata to %s", meta_path)

        # ----- BDDL transitions (optional) -----
        if bddl_transitions is not None:
            bddl_path = f"{meta_dir}/{segment_tag}_bddl.json"
            with open(bddl_path, "w") as f:
                json.dump(bddl_transitions, f, indent=2)

        return True

    def truncate_to(self, index: int) -> None:
        """Discard all buffered data at and after *index*.

        Used after reverting the simulator to a checkpoint.  The main
        video writers cannot be un-written (the positive-trajectory video
        will contain extra frames from the failed attempt), but parquet,
        HDF5, and frame-cache data are cleanly truncated so subsequent
        saves are correct.

        Args:
            index: Buffer position to truncate to.  Data from
                ``[index, len)`` is discarded.
        """
        if index < 0 or index > len(self._step_data):
            logger.warning(
                "truncate_to: index %d out of range (buffer len=%d), clamping",
                index, len(self._step_data),
            )
            index = max(0, min(index, len(self._step_data)))

        n_removed = len(self._step_data) - index
        self._step_data = self._step_data[:index]
        # State vectors have N+1 entries (initial + one per action); keep index+1.
        self._state_vectors = self._state_vectors[:index + 1]
        self._frame_cache = self._frame_cache[:index]
        self._step_count = index

        # Recompute max state size from remaining vectors.
        if self._state_vectors:
            self._max_state_size = max(len(sv) for sv in self._state_vectors)
        else:
            self._max_state_size = 0

        logger.info(
            "truncate_to: discarded %d steps, buffer now has %d steps",
            n_removed, index,
        )

    # ------------------------------------------------------------------
    # Episode finalisation
    # ------------------------------------------------------------------

    def end_episode(
        self,
        success: bool,
        env: Optional[Any] = None,
        bddl_transitions: Optional[List[Dict[str, Any]]] = None,
        end_idx: Optional[int] = None,
    ) -> bool:
        """Finalise the current episode and flush all data to disk.

        Args:
            success: Whether the episode achieved the task goal.
            env: Optional environment reference (reserved for future metadata).
            bddl_transitions: Optional list of BDDL state-transition dicts to
                persist alongside the episode.
            end_idx: Optional buffer index.  When provided, only data in
                ``[0, end_idx)`` is saved (used for the max-retries case
                where the positive trajectory ends at the last checkpoint).
                Defaults to saving all accumulated data.

        Returns:
            ``True`` if data was successfully written, ``False`` if the recorder
            was not active.
        """
        if not self.is_recording:
            return False

        data_dir = f"{self.output_folder}/data/"
        meta_dir = f"{self.output_folder}/meta/"
        for d in (data_dir, meta_dir):
            makedirs_with_mode(d)

        # Determine the data range to save.
        # step_data has N entries (one per action).
        # state_vectors has N+1 entries (initial state + one per action).
        step_data = self._step_data[:end_idx] if end_idx is not None else self._step_data
        state_vectors = self._state_vectors[:end_idx + 1] if end_idx is not None else self._state_vectors

        # ----- Save step data as Parquet -----
        parquet_path = f"{data_dir}/{self._episode_tag}.parquet"
        df = pd.DataFrame(step_data)
        df.to_parquet(parquet_path, index=False)
        logger.info("DataRecorder: saved %d steps to %s", len(df), parquet_path)

        # ----- Close main streaming video writers (best-effort) -----
        # The main writers may be in a bad state (EOF) after segment saves,
        # so we close them silently and recreate fresh writers from the
        # frame cache below.
        for writer_key, (container, stream) in self._video_writers.items():
            try:
                for packet in stream.encode():
                    container.mux(packet)
                container.close()
            except Exception:
                # Writer is in bad state -- silently discard.
                try:
                    container.close()
                except Exception:
                    pass
        self._video_writers = {}

        # ----- Write episode video from frame cache (fresh writers) -----
        # Create new av containers for each camera/modality and write all
        # cached frames.  This avoids relying on the long-lived streaming
        # writers which can enter EOF state after segment saves.
        videos_dir = f"{self.output_folder}/videos/"
        frame_data = self._frame_cache[:end_idx] if end_idx is not None else self._frame_cache
        if frame_data:
            all_keys: set = set()
            for fc in frame_data:
                all_keys.update(fc.keys())

            for wkey in sorted(all_keys):
                is_depth = wkey.endswith("_depth")
                short_name = wkey.rsplit("_", 1)[0]
                resolution = self._resolution_for_camera(short_name)
                if is_depth:
                    # Match start_episode naming: depth_{short_name}.mp4
                    vpath = f"{videos_dir}/depth_{short_name}.mp4"
                    vwriter = create_video_writer(
                        fpath=vpath,
                        resolution=resolution,
                        codec_name="libx264",
                        rate=30,
                        pix_fmt="yuv420p",
                    )
                    mode = "depth"
                else:
                    # Match start_episode naming: rgb_{short_name}.mp4
                    vpath = f"{videos_dir}/rgb_{short_name}.mp4"
                    vwriter = create_video_writer(
                        fpath=vpath, resolution=resolution,
                        codec_name="libx264", rate=30, pix_fmt="yuv420p",
                    )
                    mode = "rgb"

                ep_container, ep_stream = vwriter
                for fc in frame_data:
                    if wkey in fc:
                        write_video(fc[wkey], vwriter, mode=mode)
                try:
                    for packet in ep_stream.encode():
                        ep_container.mux(packet)
                    ep_container.close()
                except Exception:
                    logger.exception(
                        "DataRecorder: error closing fresh video writer '%s'", wkey,
                    )

        # ----- Save HDF5 sim state -----
        if state_vectors:
            hdf5_path = f"{data_dir}/{self._episode_tag}.hdf5"
            n_states = len(state_vectors)
            max_sz = max(len(sv) for sv in state_vectors)
            # Pad each state vector to the maximum length (same approach as data_wrapper)
            padded = np.zeros((n_states, max_sz), dtype=np.float32)
            sizes = np.zeros(n_states, dtype=np.int64)
            for i, sv in enumerate(state_vectors):
                padded[i, : len(sv)] = sv
                sizes[i] = len(sv)
            # Extract actions from step_data for alignment
            actions_list = [
                row.get("action", [])
                for row in step_data
            ]
            with h5py.File(hdf5_path, "w") as hf:
                grp = hf.create_group("data/demo_0")
                grp.create_dataset("state", data=padded)
                grp.create_dataset("state_size", data=sizes)
                if actions_list:
                    action_arr = np.array(actions_list, dtype=np.float32)
                    grp.create_dataset("action", data=action_arr)
            logger.info(
                "DataRecorder: saved %d state vectors to %s (max_size=%d)",
                n_states, hdf5_path, max_sz,
            )

        # ----- Save metadata JSON -----
        n_saved = len(step_data)
        metadata: Dict[str, Any] = {
            "task_name": self.task_name,
            "task_id": self.task_id,
            "demo_id": self.demo_id,
            "success": success,
            "n_steps": n_saved,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        meta_path = f"{meta_dir}/{self._episode_tag}.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("DataRecorder: saved metadata to %s", meta_path)

        # ----- Save BDDL transitions JSON (optional) -----
        if bddl_transitions is not None:
            bddl_path = f"{meta_dir}/{self._episode_tag}_bddl.json"
            with open(bddl_path, "w") as f:
                json.dump(bddl_transitions, f, indent=2)
            logger.info("DataRecorder: saved BDDL transitions to %s", bddl_path)

        self.is_recording = False
        return True
