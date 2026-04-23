"""
Shared object-name matching utilities for embodiedClaw runtime logic.

This module centralizes the matching semantics used by observation filtering
and decision logic so rollout-facing object references behave consistently
across the system.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, FrozenSet, Iterable, Optional

from omnigibson.learning.embodiedClaw.annotation_loader import strip_annotation_id

__all__ = [
    "ObjectMatchQuery",
    "arg_matches_query",
    "build_match_query",
    "canonical_name_matches",
    "canonicalize_object_name",
    "entity_matches_query",
    "is_instance_specific_name",
    "name_matches_query",
]


@dataclass(frozen=True)
class ObjectMatchQuery:
    """Compiled object-name query used for exact and normalized matching."""

    exact_names: FrozenSet[str]
    canonical_names: FrozenSet[str]


def canonicalize_object_name(name: Optional[str]) -> str:
    """Return a normalized object identifier for rollout-time matching."""
    if not name:
        return ""
    lowered = strip_annotation_id(str(name).lower())
    return re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")


def canonical_name_matches(candidate: str, target: str) -> bool:
    """Return whether two canonical names are compatible."""
    if not candidate or not target:
        return False
    if candidate == target:
        return True
    candidate_tokens = candidate.split("_")
    target_tokens = target.split("_")
    prefix_len = min(len(candidate_tokens), len(target_tokens))
    return candidate_tokens[:prefix_len] == target_tokens[:prefix_len]


def is_instance_specific_name(name: str) -> bool:
    """Return whether a name includes a trailing numeric instance suffix."""
    return bool(re.search(r"_\d+$", str(name).lower()))


def build_match_query(
    object_names: Iterable[str],
    *,
    include_normalized: bool = True,
    include_normalized_for_instance_specific: bool = False,
) -> ObjectMatchQuery:
    """Compile exact / normalized target-name sets for repeated matching.

    Args:
        object_names: Names to match against runtime scene/scope identifiers.
        include_normalized: Whether to include canonicalized normalized names.
        include_normalized_for_instance_specific: If ``False``, a query like
            ``can_of_soda_114`` stays exact-only. If ``True``, it also gains the
            normalized canonical target ``can_of_soda``.
    """
    exact_names = set()
    canonical_names = set()

    for name in object_names:
        if name is None:
            continue
        text = str(name).strip()
        if not text:
            continue
        lowered = text.lower()
        exact_names.add(lowered)

        if not include_normalized:
            continue
        if (
            is_instance_specific_name(text)
            and not include_normalized_for_instance_specific
        ):
            continue

        canonical = canonicalize_object_name(text)
        if canonical:
            canonical_names.add(canonical)

    return ObjectMatchQuery(
        exact_names=frozenset(exact_names),
        canonical_names=frozenset(canonical_names),
    )


def name_matches_query(name: Optional[str], query: ObjectMatchQuery) -> bool:
    """Return whether a single runtime name matches a compiled query."""
    if not name:
        return False

    lowered = str(name).lower()
    if lowered in query.exact_names:
        return True

    candidate = canonicalize_object_name(lowered)
    return any(
        canonical_name_matches(candidate, target)
        for target in query.canonical_names
    )


def entity_matches_query(
    scope_name: Optional[str],
    scene_name: Optional[str],
    query: ObjectMatchQuery,
) -> bool:
    """Return whether a runtime entity matches a compiled query."""
    return (
        name_matches_query(scene_name, query)
        or name_matches_query(scope_name, query)
    )


def arg_matches_query(arg: dict[str, Any], query: ObjectMatchQuery) -> bool:
    """Return whether a predicate arg matches a compiled query."""
    return entity_matches_query(
        arg.get("scope_name"),
        arg.get("scene_name"),
        query,
    )
