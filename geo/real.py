"""Real-line aliases for the current scalar model."""

# pylint: disable=invalid-name

from .line import (
    ALL_REALS_INTERVAL,
    EMPTY_INTERVAL,
    Point,
    Set,
)

real = Point
realset = Set
EMPTY_REAL_INTERVAL = EMPTY_INTERVAL

__all__ = [
    "real",
    "realset",
    "EMPTY_REAL_INTERVAL",
    "ALL_REALS_INTERVAL",
]
