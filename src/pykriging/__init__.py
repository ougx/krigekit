"""
pykriging
=========
Python wrapper for a Fortran kriging and SGSIM engine.

The shared library (libkriging.so / kriging.dll) must be compiled from the
Fortran sources in the ``fortran/`` directory and placed in this package
directory before use.  See the README for build instructions.

Public API
----------
Classes
    Kriging                 — full control over the kriging workflow

Convenience functions
    ordinary_kriging        — one-shot point kriging
    cokriging               — one-shot cokriging
    sequential_gaussian_simulation — one-shot SGSIM
"""

from pykriging.kriging import (   # noqa: F401
    Kriging,
    ordinary_kriging,
    cokriging,
    sequential_gaussian_simulation,
)

from pykriging.kriging_st import (   # noqa: F401
    SpaceTimeKriging,
    spacetime_kriging,
    spacetime_cokriging,
)

# from pykriging.meuk import MEUK                    # noqa: F401
# from pykriging.meuk_fortran import MEUKFortran     # noqa: F401

__version__ = "0.1.0"
__all__ = [
    "Kriging",
    "ordinary_kriging",
    "cokriging",
    "sequential_gaussian_simulation",
    "SpaceTimeKriging",
    "spacetime_kriging",
    "spacetime_cokriging",
    "MEUK",
    "MEUKFortran",
]
