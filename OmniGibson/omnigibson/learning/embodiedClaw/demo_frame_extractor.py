"""
Demo frame extraction and library for visual comparison.

Extracts reference frames from recorded demo videos at subtask boundaries,
builds a pre-extracted library on disk (JPEG files + JSON manifest), and
provides a query interface for loading frames as base64-encoded images.

No simulator dependencies -- operates on video files and annotation JSONs.

Usage (batch extraction)::

    python -m omnigibson.learning.embodiedClaw.demo_frame_extractor \
        --dataset_root /home/user/dataset \
        --task_id 1 \
        --output_dir /home/user/dataset/demo_frames/task-0001

Usage (programmatic)::

    from omnigibson.learning.embodiedClaw.demo_frame_extractor import (
        DemoFrameLibrary, build_frame_library,
    )

    library = build_frame_library(
        task_id=1,
        dataset_root="/home/user/dataset",
        output_root="/home/user/dataset/demo_frames",
    )
    frames = library.get_frames_as_base64(skill_idx=4, camera="head", max_episodes=3)
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import av
import numpy as np
from PIL import Image

from omnigibson.learning.embodiedClaw.annotation_loader import (
    EpisodeAnnotation,
    SubtaskAnnotation,
    load_all_annotations,
    load_episode_annotation,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DemoFrameEntry",
    "DemoFrameLibrary",
    "extract_frames_from_video",
    "compute_extraction_targets",
    "build_frame_library",
]

# JPEG quality for saved reference frames (85 gives good fidelity at ~10-30 KB).
JPEG_QUALITY = 85

# Cameras to extract from.
CAMERAS = ["head", "left_wrist", "right_wrist"]

# Minimum subtask duration (in frames) to include intermediate frames.
# Subtasks shorter than this only get completion + pre_completion.
MIN_DURATION_FOR_INTERMEDIATES = 20


# ======================================================================
# Data structures
# ======================================================================


@dataclass
class DemoFrameEntry:
    """A single reference frame extracted from a demo video."""

    task_id: int
    episode_id: str  # e.g. "00010010"
    skill_idx: int
    camera: str  # "head" | "left_wrist" | "right_wrist"
    frame_type: str  # "completion" | "pre_completion" | "intermediate_1" | "intermediate_2"
    frame_index: int  # video frame index (== sim step)
    subtask_start_frame: int
    subtask_end_frame: int
    relative_position: float  # 0.0 = start, 1.0 = end of subtask
    file_path: str  # absolute path to the saved JPEG file


# ======================================================================
# Frame extraction from video
# ======================================================================


def extract_frames_from_video(
    video_path: str,
    frame_indices: List[int],
) -> Dict[int, np.ndarray]:
    """Extract specific frames from a video file using PyAV.

    Uses keyframe-based seeking for efficiency: sorts target indices,
    seeks to the nearest keyframe before each target, then decodes
    forward to the exact frame.

    Args:
        video_path: Absolute path to an MP4 video file.
        frame_indices: List of 0-based frame indices to extract.

    Returns:
        Dict mapping frame_index -> numpy RGB array (H, W, 3).

    Raises:
        FileNotFoundError: If the video file does not exist.
        RuntimeError: If frame extraction fails.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    sorted_indices = sorted(set(frame_indices))
    result: Dict[int, np.ndarray] = {}

    container = av.open(video_path)
    stream = container.streams.video[0]

    # Determine the PTS step per frame.
    # For 30fps video with time_base=1/15360, pts_step = 15360/30 = 512.
    # But we calculate dynamically in case videos differ.
    fps = float(stream.average_rate)
    time_base = float(stream.time_base)
    if fps > 0 and time_base > 0:
        pts_per_frame = int(round(1.0 / (fps * time_base)))
    else:
        pts_per_frame = 512  # fallback

    total_frames = stream.frames
    if total_frames <= 0:
        # Estimate from duration if stream.frames is not set
        if stream.duration and stream.duration > 0:
            total_frames = int(stream.duration / pts_per_frame)
        else:
            total_frames = 999999  # large default

    try:
        # Decode sequentially through sorted targets.
        # For each target, seek to a keyframe before it, then decode forward.
        current_frame_idx = -1

        for target_idx in sorted_indices:
            if target_idx < 0:
                logger.warning("Skipping negative frame index: %d", target_idx)
                continue

            # Clamp to valid range
            clamped_idx = min(target_idx, total_frames - 1) if total_frames > 0 else target_idx

            target_pts = clamped_idx * pts_per_frame

            # Seek to the target (PyAV seeks to the nearest keyframe before)
            container.seek(target_pts, stream=stream)

            # Decode forward until we reach or pass the target
            found = False
            for frame in container.decode(video=0):
                frame_pts = frame.pts if frame.pts is not None else 0
                decoded_idx = int(round(frame_pts / pts_per_frame))

                if decoded_idx >= clamped_idx:
                    rgb = frame.to_ndarray(format="rgb24")
                    result[target_idx] = rgb
                    current_frame_idx = decoded_idx
                    found = True
                    break

            if not found:
                logger.warning(
                    "Could not decode frame %d from %s (total_frames=%d)",
                    target_idx, video_path, total_frames,
                )
    finally:
        container.close()

    return result


# ======================================================================
# Extraction target computation
# ======================================================================


def compute_extraction_targets(
    subtask: SubtaskAnnotation,
) -> List[Tuple[int, str, float]]:
    """Compute which frame indices and types to extract for a subtask.

    For each subtask with ``frame_duration = [start, end]``, extracts:

    - **completion**: ``end`` (the final frame -- what success looks like)
    - **pre_completion**: 80% through the subtask
    - **intermediate_1**: 33% through (if duration >= 20)
    - **intermediate_2**: 67% through (if duration >= 20)

    Args:
        subtask: A :class:`SubtaskAnnotation`.

    Returns:
        List of ``(frame_index, frame_type, relative_position)`` tuples.
    """
    start = subtask.frame_start
    end = subtask.frame_end
    duration = end - start

    if duration <= 0:
        return [(end, "completion", 1.0)]

    targets: List[Tuple[int, str, float]] = []

    # Always include completion and pre-completion
    targets.append((end, "completion", 1.0))
    targets.append((start + int(0.80 * duration), "pre_completion", 0.8))

    # Include intermediates for longer subtasks
    if duration >= MIN_DURATION_FOR_INTERMEDIATES:
        targets.append((start + int(0.33 * duration), "intermediate_1", 0.333))
        targets.append((start + int(0.67 * duration), "intermediate_2", 0.667))

    return targets


# ======================================================================
# Frame library
# ======================================================================


class DemoFrameLibrary:
    """In-memory index of pre-extracted demo reference frames.

    Loads from a ``manifest.json`` on disk and provides query methods
    to retrieve frames by skill_idx, camera, episode, and frame type.
    """

    def __init__(self, root_dir: str) -> None:
        """Load the frame library from a manifest file.

        Args:
            root_dir: Directory containing ``manifest.json`` and frame
                image files (e.g. ``/home/user/dataset/demo_frames/task-0001``).
        """
        self.root_dir = root_dir
        self.manifest: Dict[str, Any] = {}
        self._entries: List[DemoFrameEntry] = []

        manifest_path = os.path.join(root_dir, "manifest.json")
        if os.path.isfile(manifest_path):
            with open(manifest_path, "r") as f:
                self.manifest = json.load(f)
            self._build_entries()
            logger.info(
                "DemoFrameLibrary loaded: %d entries from %s",
                len(self._entries), manifest_path,
            )
        else:
            logger.warning("No manifest found at %s", manifest_path)

    def _build_entries(self) -> None:
        """Parse the manifest into DemoFrameEntry objects."""
        task_id = self.manifest.get("task_id", 0)
        episodes = self.manifest.get("episodes", {})

        for ep_id, ep_data in episodes.items():
            subtasks = ep_data.get("subtasks", {})
            for skill_idx_str, skill_data in subtasks.items():
                for frame_info in skill_data.get("frames", []):
                    file_rel = frame_info.get("file", "")
                    file_abs = os.path.join(self.root_dir, file_rel)
                    entry = DemoFrameEntry(
                        task_id=task_id,
                        episode_id=ep_id,
                        skill_idx=int(skill_idx_str),
                        camera=frame_info.get("camera", ""),
                        frame_type=frame_info.get("frame_type", ""),
                        frame_index=frame_info.get("frame_index", 0),
                        subtask_start_frame=skill_data.get("frame_duration", [0, 0])[0],
                        subtask_end_frame=skill_data.get("frame_duration", [0, 0])[1],
                        relative_position=frame_info.get("relative_position", 0.0),
                        file_path=file_abs,
                    )
                    self._entries.append(entry)

    @property
    def num_entries(self) -> int:
        """Total number of frame entries in the library."""
        return len(self._entries)

    @property
    def episode_ids(self) -> List[str]:
        """List of episode IDs in the library."""
        return list(self.manifest.get("episodes", {}).keys())

    @property
    def task_id(self) -> int:
        """The task ID this library was built for."""
        return self.manifest.get("task_id", 0)

    def get_frames(
        self,
        skill_idx: int,
        camera: str = "head",
        episode_id: Optional[str] = None,
        frame_types: Optional[List[str]] = None,
        max_episodes: int = 3,
    ) -> List[DemoFrameEntry]:
        """Query the library for matching frame entries.

        Args:
            skill_idx: Subtask index to retrieve frames for.
            camera: Camera name (``"head"``, ``"left_wrist"``, ``"right_wrist"``).
            episode_id: Optional specific episode. If ``None``, samples
                from available episodes up to ``max_episodes``.
            frame_types: Optional list of frame types to include.
                Defaults to all types.
            max_episodes: Maximum number of episodes to return frames
                from when ``episode_id`` is ``None``.

        Returns:
            List of matching :class:`DemoFrameEntry` objects.
        """
        matches = [
            e for e in self._entries
            if e.skill_idx == skill_idx and e.camera == camera
        ]

        if episode_id is not None:
            matches = [e for e in matches if e.episode_id == episode_id]
        else:
            # Sample up to max_episodes distinct episodes
            seen_eps: List[str] = []
            for e in matches:
                if e.episode_id not in seen_eps:
                    seen_eps.append(e.episode_id)
            # Take evenly spaced episodes for variety
            if len(seen_eps) > max_episodes:
                step = len(seen_eps) / max_episodes
                selected = [seen_eps[int(i * step)] for i in range(max_episodes)]
            else:
                selected = seen_eps
            matches = [e for e in matches if e.episode_id in selected]

        if frame_types is not None:
            matches = [e for e in matches if e.frame_type in frame_types]

        return matches

    def get_frames_as_base64(
        self,
        skill_idx: int,
        camera: str = "head",
        episode_id: Optional[str] = None,
        frame_types: Optional[List[str]] = None,
        max_episodes: int = 3,
    ) -> Dict[str, Any]:
        """Get demo reference frames as base64-encoded JPEG strings.

        Same query interface as :meth:`get_frames`, but reads the image
        files from disk and returns base64-encoded data.

        Args:
            skill_idx: Subtask index.
            camera: Camera name.
            episode_id: Optional specific episode ID.
            frame_types: Optional frame type filter.
            max_episodes: Max episodes to sample.

        Returns:
            Dict with ``"status"``, ``"skill_idx"``, ``"num_episodes"``,
            ``"frames"`` (list of frame dicts with ``"image_base64"``).
        """
        entries = self.get_frames(
            skill_idx=skill_idx,
            camera=camera,
            episode_id=episode_id,
            frame_types=frame_types,
            max_episodes=max_episodes,
        )

        if not entries:
            # Look up skill description from manifest
            skill_desc = self._get_skill_description(skill_idx)
            return {
                "status": "no_frames",
                "skill_idx": skill_idx,
                "skill_description": skill_desc,
                "num_episodes": 0,
                "frames": [],
                "message": (
                    f"No demo frames found for skill_idx={skill_idx}, "
                    f"camera={camera}."
                ),
            }

        frames_list: List[Dict[str, Any]] = []
        episode_ids_seen: set = set()

        for entry in entries:
            if not os.path.isfile(entry.file_path):
                logger.warning("Frame file missing: %s", entry.file_path)
                continue

            with open(entry.file_path, "rb") as f:
                img_bytes = f.read()
            img_b64 = base64.b64encode(img_bytes).decode("ascii")

            frames_list.append({
                "episode_id": entry.episode_id,
                "camera": entry.camera,
                "frame_type": entry.frame_type,
                "frame_index": entry.frame_index,
                "relative_position": entry.relative_position,
                "image_base64": img_b64,
            })
            episode_ids_seen.add(entry.episode_id)

        skill_desc = self._get_skill_description(skill_idx)

        return {
            "status": "ok",
            "skill_idx": skill_idx,
            "skill_description": skill_desc,
            "num_episodes": len(episode_ids_seen),
            "frames": frames_list,
        }

    def _get_skill_description(self, skill_idx: int) -> str:
        """Look up the skill description for a skill_idx from the manifest."""
        episodes = self.manifest.get("episodes", {})
        for ep_data in episodes.values():
            subtasks = ep_data.get("subtasks", {})
            skill_data = subtasks.get(str(skill_idx))
            if skill_data:
                return skill_data.get("skill_description", "")
        return ""


# ======================================================================
# Batch extraction
# ======================================================================


def _save_frame_as_jpeg(
    rgb_array: np.ndarray,
    output_path: str,
    quality: int = JPEG_QUALITY,
) -> None:
    """Save an RGB numpy array as a JPEG file.

    Args:
        rgb_array: Numpy array of shape (H, W, 3), dtype uint8.
        output_path: Absolute path to write the JPEG file.
        quality: JPEG quality (1-100).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img = Image.fromarray(rgb_array)
    img.save(output_path, format="JPEG", quality=quality)


def _episode_id_from_filename(filename: str) -> str:
    """Extract the episode ID from a filename like 'episode_00010010.mp4'.

    Returns:
        The episode ID string (e.g. ``"00010010"``).
    """
    match = re.search(r"episode_(\d+)", filename)
    return match.group(1) if match else ""


def _find_episodes_with_videos(
    task_id: int,
    dataset_root: str,
) -> Dict[str, Dict[str, str]]:
    """Find episodes that have both annotations and videos.

    Returns:
        Dict mapping episode_id -> dict with keys ``"annotation_path"``
        and ``"video_paths"`` (dict of camera -> video path).
    """
    annotation_dir = os.path.join(
        dataset_root, "annotations", f"task-{task_id:04d}"
    )
    video_base = os.path.join(
        dataset_root, "videos", f"task-{task_id:04d}"
    )

    # Collect annotation files
    anno_episodes: Dict[str, str] = {}
    if os.path.isdir(annotation_dir):
        for fname in sorted(os.listdir(annotation_dir)):
            if fname.endswith(".json"):
                ep_id = _episode_id_from_filename(fname)
                if ep_id:
                    anno_episodes[ep_id] = os.path.join(annotation_dir, fname)

    # Collect video files per camera
    video_episodes: Dict[str, Dict[str, str]] = {}
    for camera in CAMERAS:
        cam_dir = os.path.join(video_base, f"observation.images.rgb.{camera}")
        if not os.path.isdir(cam_dir):
            continue
        for fname in sorted(os.listdir(cam_dir)):
            if fname.endswith(".mp4"):
                ep_id = _episode_id_from_filename(fname)
                if ep_id:
                    video_episodes.setdefault(ep_id, {})[camera] = os.path.join(
                        cam_dir, fname
                    )

    # Intersect: episodes with both annotation AND at least one video
    result: Dict[str, Dict[str, str]] = {}
    for ep_id in sorted(anno_episodes.keys()):
        if ep_id in video_episodes:
            result[ep_id] = {
                "annotation_path": anno_episodes[ep_id],
                "video_paths": video_episodes[ep_id],
            }

    return result


def extract_frames_for_episode(
    episode_id: str,
    annotation: EpisodeAnnotation,
    video_paths: Dict[str, str],
    task_id: int,
    output_dir: str,
) -> List[Dict[str, Any]]:
    """Extract reference frames for one episode across all cameras.

    Extracts 4 frames per subtask per camera (completion, pre_completion,
    intermediate_1, intermediate_2) and saves them as JPEG files.

    Args:
        episode_id: Episode identifier (e.g. ``"00010010"``).
        annotation: Parsed annotation for this episode.
        video_paths: Dict of camera name -> video file path.
        task_id: Task number.
        output_dir: Root output directory for this task's frames.

    Returns:
        List of frame metadata dicts for the manifest.
    """
    all_frame_entries: List[Dict[str, Any]] = []

    for camera, video_path in video_paths.items():
        # Collect all frame indices needed across all subtasks
        subtask_targets: Dict[int, List[Tuple[int, str, float]]] = {}
        all_indices: List[int] = []

        for subtask in annotation.subtasks:
            targets = compute_extraction_targets(subtask)
            subtask_targets[subtask.skill_idx] = targets
            all_indices.extend(idx for idx, _, _ in targets)

        # Deduplicate and sort indices for efficient extraction
        unique_indices = sorted(set(all_indices))

        if not unique_indices:
            continue

        # Extract all needed frames from this video in one pass
        try:
            extracted = extract_frames_from_video(video_path, unique_indices)
        except Exception as e:
            logger.error(
                "Failed to extract frames from %s: %s", video_path, e,
            )
            continue

        # Save each frame and build metadata entries
        for subtask in annotation.subtasks:
            targets = subtask_targets.get(subtask.skill_idx, [])

            for frame_idx, frame_type, rel_pos in targets:
                rgb = extracted.get(frame_idx)
                if rgb is None:
                    logger.warning(
                        "Frame %d not extracted from %s (ep=%s, skill=%d)",
                        frame_idx, video_path, episode_id, subtask.skill_idx,
                    )
                    continue

                # Build output path
                rel_file = os.path.join(
                    f"skill_{subtask.skill_idx:02d}",
                    f"ep_{episode_id}",
                    f"{camera}_{frame_idx:04d}_{frame_type}.jpg",
                )
                abs_path = os.path.join(output_dir, rel_file)

                _save_frame_as_jpeg(rgb, abs_path)

                all_frame_entries.append({
                    "skill_idx": subtask.skill_idx,
                    "episode_id": episode_id,
                    "camera": camera,
                    "frame_type": frame_type,
                    "frame_index": frame_idx,
                    "relative_position": round(rel_pos, 3),
                    "file": rel_file,
                    "skill_description": subtask.skill_description,
                    "frame_duration": [subtask.frame_start, subtask.frame_end],
                })

    return all_frame_entries


def build_frame_library(
    task_id: int,
    dataset_root: str = "/home/user/dataset",
    output_root: str = "/home/user/dataset/demo_frames",
) -> DemoFrameLibrary:
    """Build the frame library for a task by extracting frames from all episodes.

    Iterates over all episodes that have both annotations and videos,
    extracts reference frames at subtask boundaries, saves them as JPEG
    files on disk, and writes a ``manifest.json`` index.

    Args:
        task_id: Task number (e.g. 1 for ``picking_up_trash``).
        dataset_root: Root directory of the BEHAVIOR-1K dataset.
        output_root: Root directory for extracted frame output. The
            frames will be saved under ``{output_root}/task-{task_id:04d}/``.

    Returns:
        A :class:`DemoFrameLibrary` loaded from the newly-built manifest.
    """
    output_dir = os.path.join(output_root, f"task-{task_id:04d}")
    os.makedirs(output_dir, exist_ok=True)

    # Find episodes with both annotations and videos
    episodes = _find_episodes_with_videos(task_id, dataset_root)
    logger.info(
        "Found %d episodes with annotations + videos for task %d",
        len(episodes), task_id,
    )

    if not episodes:
        logger.warning("No episodes found. Creating empty manifest.")
        manifest: Dict[str, Any] = {
            "task_id": task_id,
            "task_name": "",
            "created_at": datetime.now().isoformat(),
            "cameras": CAMERAS,
            "episodes": {},
        }
        manifest_path = os.path.join(output_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        return DemoFrameLibrary(output_dir)

    # Build manifest
    task_name = ""
    manifest_episodes: Dict[str, Any] = {}

    for ep_id, ep_info in episodes.items():
        annotation_path = ep_info["annotation_path"]
        video_paths = ep_info["video_paths"]

        logger.info("Processing episode %s ...", ep_id)

        try:
            annotation = load_episode_annotation(annotation_path)
        except Exception as e:
            logger.error("Failed to load annotation for %s: %s", ep_id, e)
            continue

        if not task_name:
            task_name = annotation.task_name

        # Extract frames
        frame_entries = extract_frames_for_episode(
            episode_id=ep_id,
            annotation=annotation,
            video_paths=video_paths,
            task_id=task_id,
            output_dir=output_dir,
        )

        # Organize by subtask
        subtasks_dict: Dict[str, Any] = {}
        for entry in frame_entries:
            skill_key = str(entry["skill_idx"])
            if skill_key not in subtasks_dict:
                subtasks_dict[skill_key] = {
                    "skill_description": entry["skill_description"],
                    "frame_duration": entry["frame_duration"],
                    "frames": [],
                }
            subtasks_dict[skill_key]["frames"].append({
                "frame_type": entry["frame_type"],
                "frame_index": entry["frame_index"],
                "relative_position": entry["relative_position"],
                "camera": entry["camera"],
                "file": entry["file"],
            })

        manifest_episodes[ep_id] = {
            "task_duration": annotation.task_duration,
            "subtasks": subtasks_dict,
        }

        logger.info(
            "  Episode %s: extracted %d frames", ep_id, len(frame_entries),
        )

    manifest = {
        "task_id": task_id,
        "task_name": task_name,
        "created_at": datetime.now().isoformat(),
        "cameras": CAMERAS,
        "episodes": manifest_episodes,
    }

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        "Frame library built: %s (%d episodes)",
        manifest_path, len(manifest_episodes),
    )

    return DemoFrameLibrary(output_dir)


# ======================================================================
# CLI entry point
# ======================================================================


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Extract demo reference frames for visual comparison.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/home/user/dataset",
        help="Root directory of the BEHAVIOR-1K dataset.",
    )
    parser.add_argument(
        "--task_id",
        type=int,
        required=True,
        help="Task number (e.g. 1 for picking_up_trash).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for extracted frames. Defaults to "
        "{dataset_root}/demo_frames/task-{task_id:04d}.",
    )

    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(
            args.dataset_root, "demo_frames", f"task-{args.task_id:04d}"
        )

    # output_dir is used by the library, but build_frame_library expects
    # output_root (parent) so we derive it
    output_root = os.path.dirname(output_dir)

    library = build_frame_library(
        task_id=args.task_id,
        dataset_root=args.dataset_root,
        output_root=output_root,
    )

    print(f"\nDone. Library has {library.num_entries} entries.")
    print(f"Episodes: {library.episode_ids}")
    print(f"Manifest: {os.path.join(output_dir, 'manifest.json')}")
