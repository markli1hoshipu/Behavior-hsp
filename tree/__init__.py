"""Minimal subset of the ``tree`` package used by OmniGibson.

This keeps the eval environment self-contained when ``tree`` is not installed in
the Isaac Sim conda env.
"""

from __future__ import annotations

from collections.abc import Mapping


def _is_namedtuple_instance(value):
    return isinstance(value, tuple) and hasattr(value, "_fields")


def _is_sequence(value):
    return isinstance(value, (list, tuple)) and not _is_namedtuple_instance(value)


def _assert_same_structure(reference, other):
    if isinstance(reference, Mapping):
        if not isinstance(other, Mapping) or set(reference.keys()) != set(other.keys()):
            raise TypeError("Mismatched mapping structure")
        for key in reference:
            _assert_same_structure(reference[key], other[key])
        return

    if _is_namedtuple_instance(reference):
        if not _is_namedtuple_instance(other) or type(reference) is not type(other):
            raise TypeError("Mismatched namedtuple structure")
        for left, right in zip(reference, other):
            _assert_same_structure(left, right)
        return

    if _is_sequence(reference):
        if not _is_sequence(other) or len(reference) != len(other) or type(reference) is not type(other):
            raise TypeError("Mismatched sequence structure")
        for left, right in zip(reference, other):
            _assert_same_structure(left, right)


def _map_structure(fn, path, first, *rest, with_path):
    for other in rest:
        _assert_same_structure(first, other)

    if isinstance(first, Mapping):
        return first.__class__(
            {
                key: _map_structure(
                    fn,
                    path + (key,),
                    first[key],
                    *(other[key] for other in rest),
                    with_path=with_path,
                )
                for key in first
            }
        )

    if _is_namedtuple_instance(first):
        values = [
            _map_structure(
                fn,
                path + (idx,),
                first[idx],
                *(other[idx] for other in rest),
                with_path=with_path,
            )
            for idx in range(len(first))
        ]
        return type(first)(*values)

    if _is_sequence(first):
        values = [
            _map_structure(
                fn,
                path + (idx,),
                first[idx],
                *(other[idx] for other in rest),
                with_path=with_path,
            )
            for idx in range(len(first))
        ]
        return type(first)(values)

    if with_path:
        return fn(path, first, *rest)
    return fn(first, *rest)


def map_structure(func, *structures):
    if not structures:
        raise TypeError("map_structure requires at least one structure")
    return _map_structure(func, (), structures[0], *structures[1:], with_path=False)


def map_structure_with_path(func, *structures):
    if not structures:
        raise TypeError("map_structure_with_path requires at least one structure")
    return _map_structure(func, (), structures[0], *structures[1:], with_path=True)


def flatten(structure):
    leaves = []

    def _flatten(value):
        if isinstance(value, Mapping):
            for key in value:
                _flatten(value[key])
            return
        if _is_namedtuple_instance(value) or _is_sequence(value):
            for item in value:
                _flatten(item)
            return
        leaves.append(value)

    _flatten(structure)
    return leaves
