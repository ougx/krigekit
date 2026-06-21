"""
kriging_indicator.py
====================
Python wrapper for Multiple Indicator Kriging (MIK) and Sequential Indicator
Simulation (SIS) via the Fortran t_kriging_indicator type.

Each of the K categories (or threshold classes) is treated as one indicator
variable (ivar = 1..K).  Estimation produces K probability values per block;
simulation produces a single drawn category (encoded as a one-hot binary vector).

Indicator encoding and the K x K coregionalization are built with
:class:`~krigekit.IndicatorVariogramSystem` and transferred with its
``apply()``; this wrapper only allocates the engine, solves, and post-processes
probabilities.

Typical workflow — estimation
------------------------------
>>> system = IndicatorVariogramSystem(categories=[1, 2, 3])
>>> system.set_categorical_obs(obs_coord, obs_cat, nmax=20)
>>> system.set_indicator_vgm(vtype="sph", a_major=1000,
...                          sill_strategy="theoretical", cross_strategy="closure")
>>> ik = IndicatorKriging(ncat=3, ndim=2)
>>> system.apply(ik)                # transfers indicators + K x K structures
>>> ik.set_grid(coord=grid_coord)
>>> for k in range(1, 4): ik.set_search(ivar=k)
>>> ik.solve()
>>> probs, var = ik.get_results()   # probs.shape == (ngrid, ncat)
>>> del ik

Typical workflow — SIS
-----------------------
>>> system = IndicatorVariogramSystem(categories=[1, 2, 3])
>>> system.set_categorical_obs(obs_coord, obs_cat, nmax=20)
>>> system.set_indicator_vgm(vtype="sph", sill=0.2, a_major=500,
...                          sill_strategy="uniform", cross_strategy="uniform")
>>> ik = IndicatorKriging(ncat=3, ndim=2, nsim=100)
>>> system.apply(ik)
>>> ik.set_grid(coord=grid_coord)
>>> ik.set_sim()    # generates U(0,1) draws in Python and passes to Fortran
>>> for k in range(1, 4): ik.set_search(ivar=k)
>>> ik.solve()
>>> sims, _ = ik.get_results()     # sims.shape == (ngrid, ncat, nsim)
...                                 # each [:, :, i] is a one-hot encoded realisation
"""

import ctypes
import random
import sys
from typing import Optional

import numpy as np

from .kriging import (
    Kriging,
    _krige_ind_create,
    _krige_ind_set_ncat,
    _krige_initialize,
    _h, _c_int, _c_double, _farray, _dptr,
)


class IndicatorKriging(Kriging):
    """
    Multiple Indicator Kriging and Sequential Indicator Simulation,
    with optional secondary co-variate support.

    Extends :class:`Kriging` — all setup, solve, and results methods are
    inherited unchanged.  The differences are:

    * The Fortran object is a ``t_kriging_indicator`` (created via
      ``krige_ind_create``), which overrides ``prepare``, ``sim_draw``, and
      ``post_solve`` to implement indicator-specific behaviour.
    * ``ncat`` names the K indicator categories.  ``nvar`` defaults to
      ``ncat`` (pure MIS) but can be set larger to add secondary continuous
      co-variates for co-kriging MIS (see below).
    * Indicator encoding and the K x K coregionalization are built with
      :class:`~krigekit.IndicatorVariogramSystem` and transferred via its
      ``apply()``; this engine wrapper no longer owns that construction.

    Parameters
    ----------
    ncat : int
        Number of categories K.  Indicator variables occupy ivar = 1..ncat.
    nvar : int, optional
        Total number of co-kriging variables.  Defaults to ``ncat`` (pure MIS).
        Set ``nvar = ncat + M`` to include M secondary continuous variables
        (ivar = ncat+1 .. nvar).  Secondary variables contribute to the kriging
        weights but are excluded from the CDF draw and probability normalisation.
    ndim : int
        Number of spatial dimensions (2 or 3).
    nsim : int
        0 = estimation (returns probabilities); >0 = SIS (returns one-hot draws).

    Other Parameters
    ----------------
    **kwargs
        All other keyword arguments are passed through to :class:`Kriging`.

    Notes
    -----
    For SIS (``nsim > 0``), call :meth:`set_sim` after :meth:`set_grid`.

    Co-kriging MIS example (K=3 categories + 1 secondary variable)::

        system = IndicatorVariogramSystem(categories=[1, 2, 3])
        system.set_categorical_obs(coord, cats, nmax=20)
        system.set_indicator_vgm(vtype="sph", a_major=1000,
                                 sill_strategy="theoretical",
                                 cross_strategy="closure")
        ik = IndicatorKriging(ncat=3, nvar=4, ndim=2)
        system.apply(ik)                            # indicator block (ivar 1..3)
        ik.set_obs(ivar=4, coord=sec_coord, value=sec_val)   # secondary
        ik.set_vgm(ivar=4, jvar=4, ...)             # secondary auto/cross models
        ik.set_grid(coord=grid_coord)
        for k in range(1, 5):
            ik.set_search(ivar=k)
        ik.solve()
        probs, var = ik.get_results()   # shape (ngrid, 3) — secondary excluded
    """

    def __init__(
        self,
        ncat: int,
        nvar: Optional[int] = None,
        ndim: int = 2,
        ndrift: int = 0,
        unbias: int = 1,
        nsim: int = 0,
        anisotropic_search: bool = False,
        weight_correction: bool = False,
        use_old_weight: bool = False,
        store_weight: bool = False,
        cross_validation: bool = False,
        write_mat: bool = False,
        neglect_error: bool = True,
        varying_vgm: bool = False,
        std_ck: bool = False,
        verbose: bool = False,
        pf_cache: bool = False,
        weight_file: str = "",
        bounds: Optional[tuple] = None,
        seed: Optional[int] = None,
    ):
        nvar = nvar if nvar is not None else ncat
        if nvar < ncat:
            raise ValueError(f"nvar ({nvar}) must be >= ncat ({ncat})")

        # Allocate the indicator Fortran object (t_kriging_indicator).
        # We do NOT call super().__init__() because that would allocate a
        # plain t_kriging via krige_create; we need krige_ind_create instead.
        _h_tmp = ctypes.c_int64(0)
        _krige_ind_create(ctypes.byref(_h_tmp))
        self._handle: int = _h_tmp.value

        _huge = sys.float_info.max * 1e3
        c_bounds = _farray(bounds if bounds is not None else [-_huge, _huge])
        seed = seed or random.randint(0, 2**32 - 1)
        random.seed(seed)

        _krige_initialize(
            _h(self._handle),
            _c_int(ndim),
            _c_int(nvar),       # total co-kriging variables (>= ncat)
            _c_int(ndrift),
            _c_int(unbias),
            _c_int(nsim),
            _c_int(int(anisotropic_search)),
            _c_int(int(weight_correction)),
            _c_int(int(use_old_weight)),
            _c_int(int(store_weight)),
            _c_int(int(cross_validation)),
            _c_int(int(write_mat)),
            _c_int(int(neglect_error)),
            _c_int(int(varying_vgm)),
            _c_int(int(std_ck)),
            _c_int(int(verbose)),
            _c_int(int(pf_cache)),
            weight_file.encode("utf-8") if weight_file else b"",
            _dptr(c_bounds),
            _c_int(seed),
        )

        # Store ncat in the Fortran object only when ncat < nvar;
        # when ncat == nvar the Fortran default (ncat=0 → use nvar) is correct.
        if ncat < nvar:
            _krige_ind_set_ncat(_h(self._handle), _c_int(ncat))

        self.ncat   = ncat
        self.ndim   = ndim
        self.nvar   = nvar
        self.ndrift = ndrift
        self.nsim   = nsim
        self.verbose = verbose
        self.unbias = unbias
        self.anisotropic_search = anisotropic_search
        self.weight_correction  = weight_correction
        self.use_old_weight     = use_old_weight
        self.store_weight       = store_weight
        self.cross_validation   = cross_validation
        self.write_mat          = write_mat
        self.varying_vgm        = varying_vgm
        self.std_ck             = std_ck
        self.pf_cache           = pf_cache
        self.weight_file        = weight_file
        self.bounds             = c_bounds
        self.seed               = seed

        self._nblock   = 0
        self._nobs     = np.zeros(self.nvar, dtype=np.uint32)
        self._nmax     = np.zeros(self.nvar, dtype=np.uint32)
        self._set_search   = [False] * self.nvar
        self._set_sim      = False
        self._nobsdrift    = np.zeros(self.nvar, dtype=np.uint32)
        self._nvgm_struct  = np.zeros([self.nvar, self.nvar], dtype=np.uint32)

    def get_results(self, copy: bool = False, squeeze: bool = True):
        """Return indicator results, excluding secondary covariate channels.

        The kriging engine stores all ``nvar`` estimates internally because
        secondary variables participate in the cokriging system.  Public
        indicator results contain only the first ``ncat`` variables, matching
        the probability or one-hot category array documented by this class.
        """
        estimate, variance = super().get_results(copy=False, squeeze=squeeze)
        estimate = np.asarray(estimate)
        variance = np.asarray(variance)

        if estimate.ndim >= 2:
            estimate = estimate[:, :self.ncat, ...]
        if variance.ndim >= 3:
            variance = variance[:, :self.ncat, :self.ncat]

        if copy:
            estimate = np.array(estimate, order="C", copy=True)
            variance = np.array(variance, order="C", copy=True)
        return estimate, variance

    # ------------------------------------------------------------------
    def set_sim(
        self,
        randpath: Optional[np.ndarray] = None,
        sample: Optional[np.ndarray] = None,
    ):
        """
        Set up Sequential Indicator Simulation parameters.

        Overrides :meth:`~krigekit.Kriging.set_sim`.  When ``sample`` is
        ``None``, delegates to the parent with no sample so that the Fortran
        ``set_sim_indicator`` override generates U(0, 1) draws directly;
        no sample array is created in Python.  When ``sample`` is supplied,
        validates that every value lies in [0, 1] then delegates to the
        parent.

        Parameters
        ----------
        randpath : ndarray of int, shape (nblocks,), optional
            Random visiting order (1-based).  Generated with a random
            permutation if omitted.
        sample : ndarray, shape (nblocks, nvar, nsim), optional
            Pre-drawn U(0, 1) samples.  Every value must lie in [0, 1].
            When ``None``, Fortran generates U(0, 1) via ``set_sim_indicator``.
        """
        assert self.nsim > 0, "nsim must be > 0 when calling set_sim()"
        if sample is not None:
            s = np.asarray(sample, dtype=np.float64)
            lo, hi = float(s.min()), float(s.max())
            if lo < 0.0 or hi > 1.0:
                raise ValueError(
                    f"sample values must be in [0, 1] for indicator simulation; "
                    f"got range [{lo:.4g}, {hi:.4g}]"
                )
            super().set_sim(randpath=randpath, sample=s)
        else:
            super().set_sim(randpath=randpath)

