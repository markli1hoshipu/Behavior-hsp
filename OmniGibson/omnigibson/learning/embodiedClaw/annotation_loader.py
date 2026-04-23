"""
Annotation loader for the BEHAVIOR-1K dataset.

Loads, parses, and indexes annotation JSON files; computes cross-episode
subtask duration statistics; and generates VLA language instructions from
subtask annotations.

This module has no simulator dependencies -- it operates on files and dicts.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "SubtaskAnnotation",
    "EpisodeAnnotation",
    "SubtaskDurationStats",
    "load_episode_annotation",
    "load_all_annotations",
    "compute_subtask_duration_stats",
    "generate_language_instruction",
    "get_rollout_subtask_object_refs",
    "normalize_annotation_names",
    "normalize_annotation_object_groups",
    "strip_annotation_id",
]


# ======================================================================
# Data structures
# ======================================================================


@dataclass
class SubtaskAnnotation:
    """Parsed representation of one ``skill_annotation`` entry."""

    skill_idx: int
    skill_description: str  # e.g. "move to"
    object_ids: List[List[str]]  # e.g. [["trash_can_116", "floors_zqjkvm_0"]]
    manipulating_object_ids: List[str]  # e.g. ["trash_can_116"]
    frame_start: int
    frame_end: int
    frame_duration: int  # frame_end - frame_start
    skill_type: str  # "navigation" | "uncoordinated" | "coordinated"
    memory_prefix: List[str] = field(default_factory=list)
    spatial_prefix: List[str] = field(default_factory=list)


@dataclass
class EpisodeAnnotation:
    """Full parsed annotation for one episode."""

    task_name: str
    task_duration: int
    valid_start: int
    valid_end: int
    subtasks: List[SubtaskAnnotation]
    primitive_annotations: List[Dict[str, Any]]  # raw primitive_annotation list


@dataclass
class SubtaskDurationStats:
    """Cross-episode duration statistics for a subtask at a given ``skill_idx``."""

    skill_idx: int
    skill_description: str
    min_duration: int
    max_duration: int
    mean_duration: float
    count: int
    failure_threshold: int  # 2 * max_duration


# ======================================================================
# Name utilities
# ======================================================================


def strip_annotation_id(name: str) -> str:
    """Strip the trailing ``_NNN`` numeric suffix from an annotation name.

    Examples::

        strip_annotation_id("trash_can_116") -> "trash_can"
        strip_annotation_id("can_of_soda_114") -> "can_of_soda"
        strip_annotation_id("floors_zqjkvm_0") -> "floors_zqjkvm"

    Rule: split on ``_``, remove the last segment if it is purely numeric.

    Args:
        name: An annotation object name (e.g. ``"trash_can_116"``).

    Returns:
        The name with the trailing numeric suffix removed.
    """
    parts = name.split("_")
    if len(parts) > 1 and parts[-1].isdigit():
        return "_".join(parts[:-1])
    return name


def _clean_name_for_display(name: str) -> str:
    """Strip numeric suffix and replace underscores with spaces.

    Used for generating human-readable language instructions.

    Args:
        name: An annotation object name.

    Returns:
        A cleaned, display-friendly name.
    """
    base = strip_annotation_id(name)
    return base.replace("_", " ")


def normalize_annotation_names(names: List[str]) -> List[str]:
    """Return rollout-safe object names without numeric instance suffixes.

    This preserves ordering and multiplicity from the annotation while removing
    episode-specific instance ids (e.g. ``can_of_soda_114`` -> ``can_of_soda``).
    """
    return [strip_annotation_id(name) for name in names]


def normalize_annotation_object_groups(object_groups: List[List[str]]) -> List[List[str]]:
    """Normalize every annotation object group for rollout-facing use."""
    return [normalize_annotation_names(group) for group in object_groups]


def get_rollout_subtask_object_refs(subtask: SubtaskAnnotation) -> Dict[str, Any]:
    """Return rollout-safe and raw object references for a subtask.

    Public MCP payloads should use the normalized fields by default so the
    skill sequence does not depend on annotation instance ids that are unknown
    at rollout time. The raw annotation ids are still exposed under explicit
    ``annotation_*`` keys for debugging and offline analysis.
    """
    return {
        "object_ids": normalize_annotation_object_groups(subtask.object_ids),
        "manipulating_object_ids": normalize_annotation_names(
            subtask.manipulating_object_ids
        ),
        "annotation_object_ids": subtask.object_ids,
        "annotation_manipulating_object_ids": subtask.manipulating_object_ids,
    }


# ======================================================================
# Annotation loading
# ======================================================================


def load_episode_annotation(annotation_path: str) -> EpisodeAnnotation:
    """Load and parse a single annotation JSON file.

    Args:
        annotation_path: Absolute path to an ``episode_XXXXXXXX.json`` file.

    Returns:
        Parsed :class:`EpisodeAnnotation`.

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If required fields are missing from the JSON.
    """
    with open(annotation_path, "r") as f:
        data = json.load(f)

    task_name: str = data["task_name"]
    meta = data["meta_data"]
    task_duration: int = int(meta["task_duration"])
    valid_start: int = int(meta["valid_duration"][0])
    valid_end: int = int(meta["valid_duration"][1])

    subtasks: List[SubtaskAnnotation] = []
    for entry in data.get("skill_annotation", []):
        frame_start = int(entry["frame_duration"][0])
        frame_end = int(entry["frame_duration"][1])

        # skill_description and skill_type are single-element lists
        skill_desc = entry["skill_description"][0] if entry["skill_description"] else ""
        s_type = entry["skill_type"][0] if entry["skill_type"] else ""

        subtask = SubtaskAnnotation(
            skill_idx=int(entry["skill_idx"]),
            skill_description=skill_desc,
            object_ids=entry.get("object_id", []),
            manipulating_object_ids=entry.get("manipulating_object_id", []),
            frame_start=frame_start,
            frame_end=frame_end,
            frame_duration=frame_end - frame_start,
            skill_type=s_type,
            memory_prefix=entry.get("memory_prefix", []),
            spatial_prefix=entry.get("spatial_prefix", []),
        )
        subtasks.append(subtask)

    primitive_annotations = data.get("primitive_annotation", [])

    return EpisodeAnnotation(
        task_name=task_name,
        task_duration=task_duration,
        valid_start=valid_start,
        valid_end=valid_end,
        subtasks=subtasks,
        primitive_annotations=primitive_annotations,
    )


def load_all_annotations(task_annotation_dir: str) -> List[EpisodeAnnotation]:
    """Load all episode annotations from a task directory.

    Args:
        task_annotation_dir: Path to the directory containing annotation
            JSON files (e.g. ``"/home/user/dataset/annotations/task-0001/"``).

    Returns:
        List of :class:`EpisodeAnnotation` objects, one per file, sorted by
        filename.
    """
    annotations: List[EpisodeAnnotation] = []
    if not os.path.isdir(task_annotation_dir):
        logger.warning(
            "Annotation directory does not exist: %s", task_annotation_dir
        )
        return annotations

    files = sorted(
        f for f in os.listdir(task_annotation_dir) if f.endswith(".json")
    )
    for fname in files:
        fpath = os.path.join(task_annotation_dir, fname)
        try:
            anno = load_episode_annotation(fpath)
            annotations.append(anno)
        except Exception as e:
            logger.warning("Failed to load annotation %s: %s", fpath, e)

    logger.info(
        "Loaded %d annotations from %s", len(annotations), task_annotation_dir
    )
    return annotations


# ======================================================================
# Duration statistics
# ======================================================================


def compute_subtask_duration_stats(
    annotations: List[EpisodeAnnotation],
) -> Dict[int, SubtaskDurationStats]:
    """Compute min/max/mean duration for each ``skill_idx`` across all episodes.

    The failure threshold is set to ``2 * max_duration``.

    Args:
        annotations: All episode annotations for a task.

    Returns:
        Dict mapping ``skill_idx`` to :class:`SubtaskDurationStats`.
    """
    # Collect durations per skill_idx
    durations: Dict[int, List[int]] = {}
    descriptions: Dict[int, str] = {}

    for anno in annotations:
        for subtask in anno.subtasks:
            idx = subtask.skill_idx
            durations.setdefault(idx, []).append(subtask.frame_duration)
            if idx not in descriptions:
                descriptions[idx] = subtask.skill_description

    stats: Dict[int, SubtaskDurationStats] = {}
    for idx, durs in sorted(durations.items()):
        min_d = min(durs)
        max_d = max(durs)
        mean_d = sum(durs) / len(durs)
        stats[idx] = SubtaskDurationStats(
            skill_idx=idx,
            skill_description=descriptions.get(idx, ""),
            min_duration=min_d,
            max_duration=max_d,
            mean_duration=round(mean_d, 1),
            count=len(durs),
            failure_threshold=2 * max_d,
        )

    return stats


# ======================================================================
# Language instruction generation
# ======================================================================


def generate_language_instruction(subtask: SubtaskAnnotation) -> str:
    """Generate a natural-language instruction for the VLA from a subtask.

    The instruction is composed from the skill description and target objects.
    Object names are cleaned: trailing numeric suffixes are stripped and
    underscores are replaced with spaces.

    Examples::

        "move to" + [["trash_can_116"]]
            -> "move to the trash can"
        "pick up from" + [["can_of_soda_114", "floors_ulujpr_0"]]
            -> "pick up the can of soda from the floor"
        "place in" + [["can_of_soda_114", "trash_can_116"]]
            -> "place the can of soda in the trash can"

    Args:
        subtask: A :class:`SubtaskAnnotation`.

    Returns:
        A human-readable language instruction string.
    """
    desc = subtask.skill_description  # e.g. "move to", "pick up from", "place in"

    # Flatten the first (and typically only) object_id group
    obj_group = subtask.object_ids[0] if subtask.object_ids else []

    # Clean the object names
    clean_names = [_clean_name_for_display(n) for n in obj_group]

    # Remove floor-like names for brevity in certain contexts
    def _is_floor(name: str) -> bool:
        return name.startswith("floor")

    if desc in ("move to",):
        # "move to the <target>"
        targets = [n for n in clean_names if not _is_floor(n)]
        if targets:
            return f"move to the {targets[0]}"
        elif clean_names:
            return f"move to the {clean_names[0]}"
        return "move to the target"

    elif desc in ("pick up from",):
        # "pick up the <manipulated> from the <surface>"
        manip = (
            _clean_name_for_display(subtask.manipulating_object_ids[0])
            if subtask.manipulating_object_ids
            else (clean_names[0] if clean_names else "object")
        )
        surface_names = [n for n in clean_names if n != manip and not _is_floor(n)]
        if surface_names:
            return f"pick up the {manip} from the {surface_names[0]}"
        return f"pick up the {manip}"

    elif desc in ("place in",):
        # "place the <manipulated> in the <container>"
        manip = (
            _clean_name_for_display(subtask.manipulating_object_ids[0])
            if subtask.manipulating_object_ids
            else (clean_names[0] if clean_names else "object")
        )
        containers = [n for n in clean_names if n != manip]
        if containers:
            return f"place the {manip} in the {containers[0]}"
        return f"place the {manip}"

    elif desc in ("place on",):
        # "place the <manipulated> on the <surface>"
        manip = (
            _clean_name_for_display(subtask.manipulating_object_ids[0])
            if subtask.manipulating_object_ids
            else (clean_names[0] if clean_names else "object")
        )
        surfaces = [n for n in clean_names if n != manip]
        if surfaces:
            return f"place the {manip} on the {surfaces[0]}"
        return f"place the {manip}"

    else:
        # Generic fallback
        obj_str = ", ".join(clean_names) if clean_names else "the target"
        return f"{desc} {obj_str}"
