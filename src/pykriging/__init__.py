"""
pykriging
=========
Python wrapper for a Fortran kriging and SGSIM engine.

The shared library (libkriging.so / kriging.dll) must be compiled from the
Fortran sources in the ``fortran/`` directory and placed in this package
directory before use.  See the README for build instructions.

Public API
----------
Entry point
    Kriging(st=False, **kwargs)
        Factory that returns a :class:`_SpatialKriging` when ``st=False``
        (the default) or a :class:`SpaceTimeKriging` when ``st=True``.

        Spatial usage::

            k = Kriging(ndim=2, nvar=1)

        Space-time usage::

            k = Kriging(st=True, nvar=1)
            k.set_st_model(...)

Classes (returned by Kriging factory)
    _SpatialKriging     — spatial kriging / co-kriging / SGSIM
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
    _SpatialKriging,
    ordinary_kriging,
    cokriging,
    sequential_gaussian_simulation,
)

from pykriging.kriging_st import (   # noqa: F401
    SpaceTimeKriging,
    spacetime_kriging,
    spacetime_cokriging,
)

__version__ = "0.1.0"
__all__ = [
    # unified entry point
    "Kriging",
    # concrete classes (for isinstance checks and direct use)
    "_SpatialKriging",
    "SpaceTimeKriging",
    # convenience functions
    "ordinary_kriging",
    "cokriging",
    "sequential_gaussian_simulation",
    "spacetime_kriging",
    "spacetime_cokriging",
]
