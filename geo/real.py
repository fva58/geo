"""Real-number aliases for the current float-based model."""

# pylint: disable=invalid-name

from .floatset import ( FloatSet as realset ,
                        EMPTY_FLOAT_INTERVAL as EMPTY_REAL_INTERVAL ,
                        FULL_FLOAT_INTERVAL as ALL_REALS_INTERVAL )

real = float

__all__ = [
    "real",
    "realset",
    "EMPTY_REAL_INTERVAL",
    "ALL_REALS_INTERVAL",
]
