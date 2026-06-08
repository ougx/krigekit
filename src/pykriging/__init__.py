"""
pykriging
=========
Python wrapper for a Fortran kriging and SGSIM engine.

The shared library (libkriging.so / kriging.dll) must be compiled from the
Fortran sources in the ``fortran/`` directory and placed in this package
directory before use.  See the README for build instructions.

Public API
----------

    Spatial usage::

        k = Kriging(ndim=2, nvar=1)

    Space-time usage::

        k = SpaceTimeKriging(nvar=1)
        k.set_st_model(...)

Classes (returned by Kriging factory)
    Kriging             — spatial kriging / co-kriging / SGSIM
    SpaceTimeKriging    — 3-D + time kriging / SGSIM

Convenience functions
    ordinary_kriging            — one-shot point kriging
    cokriging                   — one-shot co-kriging
    sequential_gaussian_simulation — one-shot SGSIM
    spacetime_kriging           — one-shot ST kriging
    spacetime_cokriging         — one-shot ST co-kriging
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

from pykriging.kriging_indicator import (   # noqa: F401
    IndicatorKriging,
)

from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("pykriging")
except PackageNotFoundError:
    __version__ = "unknown"
__all__ = [
    # concrete classes (for isinstance checks and direct use)
    "Kriging",
    "SpaceTimeKriging",
    "IndicatorKriging",
    # convenience functions
    "ordinary_kriging",
    "cokriging",
    "sequential_gaussian_simulation",
    "spacetime_kriging",
    "spacetime_cokriging",
]
