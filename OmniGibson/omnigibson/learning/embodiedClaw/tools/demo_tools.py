"""
MCP-facing wrappers for the demo frame library.

Provides functions that query the :class:`DemoFrameLibrary` and return
JSON-serializable dicts suitable for MCP tool responses.  Analogous to
``annotation_tools.py`` for annotations.

No simulator dependencies -- operates on the pre-extracted frame library.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from omnigibson.learning.embodiedClaw.demo_frame_extractor import (
    DemoFrameLibrary,
)

logger = logging.getLogger(__name__)

DEFAULT_FRAME_TYPES = ("completion",)

__all__ = [
    "get_demo_reference_frames",
]


def get_demo_reference_frames(
    library: Optional[DemoFrameLibrary],
    skill_idx: int,
    camera: str = "head",
    episode_id: Optional[str] = None,
    frame_types: Optional[List[str]] = None,
    max_episodes: int = 1,
) -> Dict[str, Any]:
    """Get demo reference frames for visual comparison.

    Queries the :class:`DemoFrameLibrary`, loads images from disk, and
    returns base64-encoded JPEG data along with metadata.

    Args:
        library: The pre-loaded :class:`DemoFrameLibrary`. If ``None``,
            returns an error dict.
        skill_idx: Subtask index to get reference frames for.
        camera: Camera name (``"head"``, ``"left_wrist"``,
            ``"right_wrist"``). Default ``"head"``.
        episode_id: Optional specific episode ID. If ``None``, returns
            frames from up to ``max_episodes`` episodes.
        frame_types: Optional list of frame types to include.
            Default returns only ``["completion"]``.
        max_episodes: Maximum number of episodes to return frames from.
            Default 1.

    Returns:
        JSON-serializable dict with:
            - ``"status"``: ``"ok"`` | ``"no_frames"`` | ``"error"``
            - ``"skill_idx"``: int
            - ``"skill_description"``: str
            - ``"num_episodes"``: int
            - ``"frames"``: list of frame dicts, each containing
              ``"episode_id"``, ``"camera"``, ``"frame_type"``,
              ``"frame_index"``, ``"relative_position"``,
              ``"image_base64"``
    """
    if library is None:
        return {
            "status": "error",
            "error": (
                "Demo frame library not loaded. Run the extraction script "
                "first: python -m omnigibson.learning.embodiedClaw.demo_frame_extractor "
                "--task_id <ID> --dataset_root /home/user/dataset"
            ),
            "skill_idx": skill_idx,
            "frames": [],
        }

    if library.num_entries == 0:
        return {
            "status": "error",
            "error": "Demo frame library is empty (no frames extracted).",
            "skill_idx": skill_idx,
            "frames": [],
        }

    requested_frame_types = (
        list(frame_types) if frame_types is not None else list(DEFAULT_FRAME_TYPES)
    )

    result = library.get_frames_as_base64(
        skill_idx=skill_idx,
        camera=camera,
        episode_id=episode_id,
        frame_types=requested_frame_types,
        max_episodes=max_episodes,
    )

    return result
