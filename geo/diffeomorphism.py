"""Protocols for maps, charts, and diffeomorphisms.

This module defines structural typing contracts for the basic mapping objects
used in differential geometry code. Smoothness and domain restrictions are
semantic requirements of these protocols and are not mechanically verified by
Python's type system.
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable


Source = TypeVar("Source")
Target = TypeVar("Target")


@runtime_checkable
class Map(Protocol[Source, Target]):
    """Protocol for a map between two spaces."""

    def __call__(self, point: Source) -> Target:
        """Apply the map."""


@runtime_checkable
class InvertibleMap(Map[Source, Target], Protocol[Source, Target]):
    """Protocol for a map equipped with an inverse operation."""

    def inverse(self, point: Target) -> Source:
        """Apply the inverse map."""


@runtime_checkable
class Diffeomorphism(InvertibleMap[Source, Target], Protocol[Source, Target]):
    """Protocol for a bijective smooth map with a smooth inverse.

    The protocol is intentionally minimal:

    - ``__call__`` applies the forward map;
    - ``inverse`` applies the inverse map to a target-space point.

    In mathematical code, satisfying this protocol means the implementing
    object is expected to represent a diffeomorphism between the source and
    target spaces.
    """

@runtime_checkable
class Chart(Diffeomorphism[Source, Target], Protocol[Source, Target]):
    """Protocol for a local coordinate chart.

    Semantically, the source is a geometric neighborhood and the target is a
    coordinate neighborhood in a Euclidean space. The operational contract is
    the same as for a diffeomorphism: forward map plus inverse map.
    """


__all__ = ["Map", "InvertibleMap", "Diffeomorphism", "Chart"]
