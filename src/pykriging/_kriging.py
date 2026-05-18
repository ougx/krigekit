"""
kriging.py
==========
Python wrapper for the Fortran kriging module via ISO C Binding.

Build the shared library first:
    gfortran -O2 -fPIC -fdefault-real-8 -fopenmp -shared \\
        common.f90 utils.F90 rotation.f90 variogram.f90 \\
        kriging.F90 kriging_capi.f90 \\
        -o libkriging.so

Then use this module:
    from kriging import Kriging
    import numpy as np

    k = Kriging(ndim=2, nvar=1)
    k.set_obs(ivar=1, coord=coord, value=value, nmax=20)
    k.set_vgm(ivar=1, jvar=1, spec="sph 0 1000 500 1000 500 0 0 0")
    k.set_grid(coord=grid_coord)
    k.set_search(ivar=1)
    k.solve()
    est, var = k.get_results()
"""

import ctypes
import os
import numpy as np
from typing import Optional

# ---------------------------------------------------------------------------
# Load the shared library (platform-aware)
# ---------------------------------------------------------------------------
def _load_lib():
    base = os.path.dirname(__file__)
    import sys as _sys
    if _sys.platform == "win32":
        names = ["kriging.dll"]
    elif _sys.platform == "darwin":
        names = ["libkriging.dylib"]
    else:
        names = ["libkriging.so"]
    for name in names:
        path = os.path.join(base, name)
        if os.path.exists(path):
            return ctypes.CDLL(path)
    raise FileNotFoundError(
        f"Compiled Fortran library not found in {base!r}.\n"
        "Build it first — see README.md for instructions."
    )

_lib = _load_lib()

# ---------------------------------------------------------------------------
# Declare argument and return types for every C-binding entry point
# ---------------------------------------------------------------------------
_c_int    = ctypes.c_int
_c_double = ctypes.c_double
_c_intptr = ctypes.c_int64   # integer(c_intptr_t) — handle
_c_char_p = ctypes.c_char_p
_ptr_int  = ctypes.POINTER(ctypes.c_int)
_ptr_dbl  = ctypes.POINTER(ctypes.c_double)

def _cfun(name, argtypes, restype=None):
    fn = getattr(_lib, name)
    fn.argtypes = argtypes
    fn.restype  = restype
    return fn

_ptr_int64 = ctypes.POINTER(ctypes.c_int64)
_krige_create      = _cfun("krige_create",      [_ptr_int64])
_krige_destroy     = _cfun("krige_destroy",     [_ptr_int64])
_krige_initialize  = _cfun("krige_initialize",  [
    ctypes.c_int64,                              # handle
    _c_int, _c_int, _c_int, _c_int, _c_int,     # ndim, nvar, ndrift, unbias, nsim
    _c_int, _c_int, _c_int, _c_int, _c_int, _c_int, _c_int,  # flags (6 booleans as int)
    _c_char_p,                                   # weight_file
    _ptr_dbl,                                    # bounds[2]
    _c_double,                                   # sk_mean
])
_krige_set_obs     = _cfun("krige_set_obs", [
    ctypes.c_int64,                              # handle
    _c_int, _c_int, _c_int,                      # ivar, nobs, ndim_c
    _ptr_dbl, _ptr_dbl, _ptr_dbl,                # coord, value, variance
    _c_int, _c_double,                           # nmax, maxdist
])
_krige_set_obs_drift = _cfun("krige_set_obs_drift", [
    ctypes.c_int64,                              # handle
    _c_int, _c_int, _c_int,                      # ivar, ndrift_c, nobs
    _ptr_dbl,                                    # drift[ndrift_c, nobs]
])
_krige_set_vgm     = _cfun("krige_set_vgm",  [ctypes.c_int64, _c_int, _c_int, _c_char_p])
_krige_set_grid    = _cfun("krige_set_grid", [
    ctypes.c_int64,                              # handle
    _c_int, _c_int, _ptr_dbl,                   # ngrid, ndim_c, coord
    _ptr_dbl, _ptr_dbl,                          # rangescale, localnugget
])
_krige_set_grid_block = _cfun("krige_set_grid_block", [
    ctypes.c_int64,                              # handle
    _c_int,                                      # block_type
    _c_int, _c_int, _ptr_dbl,                   # ngrid, ndim_c, coord
    _c_int, _ptr_int,                            # nblock, nblockpnt
    _ptr_dbl,                                    # pointweight[sum(nblockpnt)] — no npw
    _ptr_dbl, _ptr_dbl,                          # rangescale, localnugget
])
_krige_set_grid_cv = _cfun("krige_set_grid_cv", [ctypes.c_int64])
_krige_set_grid_drift = _cfun("krige_set_grid_drift", [
    ctypes.c_int64,                              # handle
    _c_int, _c_int,                              # ndrift_c, nblocks
    _ptr_dbl,                                    # drift[ndrift_c, nblocks]
])
_krige_set_sim     = _cfun("krige_set_sim", [
    ctypes.c_int64,                              # handle
    _c_int, _ptr_int,                            # nblocks, randpath[nblocks]
    _c_int, _ptr_dbl,                            # nsim_c, sample[nsim_c, nblocks]
])
_krige_set_search  = _cfun("krige_set_search", [
    ctypes.c_int64, _c_int,                      # handle, ivar
    _c_double, _c_double, _c_double, _c_double, _c_double,  # anis1, anis2, az, dip, plunge
])
_krige_solve       = _cfun("krige_solve",       [ctypes.c_int64])
_krige_get_nblocks = _cfun("krige_get_nblocks", [ctypes.c_int64, _ptr_int])
_krige_get_nsim    = _cfun("krige_get_nsim",    [ctypes.c_int64, _ptr_int])
_krige_get_estimate= _cfun("krige_get_estimate",[ctypes.c_int64, _c_int, _c_int, _ptr_dbl])
_krige_get_variance= _cfun("krige_get_variance",[ctypes.c_int64, _c_int, _ptr_dbl])


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _farray(a, dtype=np.float64):
    """Return a Fortran-contiguous array of the given dtype."""
    return np.asfortranarray(a, dtype=dtype)

def _coord_to_fortran(coord: np.ndarray) -> np.ndarray:
    """
    Convert coordinates from Python convention (nobs, ndim)
    to Fortran convention (ndim, nobs), column-major.

    The user always passes (nobs, ndim) — rows are points, columns are
    spatial dimensions, matching NumPy/pandas/scikit-learn convention.
    This function transposes to (ndim, nobs) and ensures Fortran memory order
    before the array is handed to the Fortran library.

    Note: if nobs == ndim the transpose is ambiguous. Use the shape assertion
    in each calling method (coord.shape[1] == self.ndim) to catch this.
    """
    a = np.asarray(coord, dtype=np.float64)
    if a.ndim == 1:
        # single point shape (ndim,) -> (ndim, 1)
        return np.asfortranarray(a.reshape(-1, 1))
    # (nobs, ndim) -> transpose -> (ndim, nobs), then make Fortran-contiguous
    return np.asfortranarray(a.T)

def _drift_to_fortran(drift: np.ndarray) -> np.ndarray:
    """
    Convert drift from Python convention (nobs, ndrift)
    to Fortran convention (ndrift, nobs), column-major.
    """
    return np.asfortranarray(np.asarray(drift, dtype=np.float64).T)

def _dptr(a):
    """ctypes pointer to a numpy float64 array."""
    return a.ctypes.data_as(_ptr_dbl)

def _iptr(a):
    """ctypes pointer to a numpy int32 array."""
    return a.ctypes.data_as(_ptr_int)

def _h(handle: int) -> ctypes.c_int64:
    """Wrap a plain-int handle as ctypes.c_int64 for every Fortran call.

    Storing the handle as a plain int and wrapping fresh at each call site
    avoids the OSError / access-violation that can occur when a ctypes
    object is passed where ctypes expects to auto-convert an integer value.
    """
    return ctypes.c_int64(handle)


# ---------------------------------------------------------------------------
# Main Python class
# ---------------------------------------------------------------------------

class Kriging:
    """
    Python interface to the Fortran t_kriging kriging/simulation engine.

    Array convention
    ----------------
    All coordinate arrays use **(nobs, ndim)** shape — rows are points,
    columns are spatial dimensions. This matches NumPy, pandas, and
    scikit-learn conventions. The wrapper transparently transposes to
    Fortran's (ndim, nobs) before calling the library.

    Typical workflow
    ----------------
    >>> k = Kriging(ndim=2, nvar=1)
    >>> k.set_obs(ivar=1, coord=obs_coord, value=obs_value, nmax=20)
    >>> k.set_vgm(ivar=1, jvar=1, spec="sph 100 900 500 1000 500 0 0 0")
    >>> k.set_grid(coord=grid_coord)
    >>> k.set_search(ivar=1)
    >>> k.solve()
    >>> estimate, variance = k.get_results()

    For sequential Gaussian simulation add ``nsim=N`` to the constructor and
    call :meth:`set_sim` after :meth:`set_grid`.
    """

    def __init__(
        self,
        ndim: int = 2,
        nvar: int = 1,
        ndrift: int = 0,
        unbias: int = 1,
        nsim: int = 0,
        anisotropic_search: bool = False,
        weight_correction: bool = False,
        use_old_weight: bool = False,
        store_weight: bool = False,
        cross_validation: bool = False,
        write_mat: bool = False,
        verbose: bool = False,
        weight_file: str = "",
        bounds: Optional[tuple] = None,
        sk_mean: float = 0.0,
    ):
        """
        Parameters
        ----------
        ndim : int
            Number of spatial dimensions (2 or 3).
        nvar : int
            Number of variables (1 for ordinary/simple kriging, >1 for cokriging).
        ndrift : int
            Number of external drift functions (0 = no drift).
        unbias : int
            1 = ordinary kriging (sum-of-weights = 1 constraint);
            0 = simple kriging (no constraint, uses sk_mean).
        nsim : int
            Number of simulations. 0 = kriging only; >0 = SGSIM.
        anisotropic_search : bool
            Use anisotropic search ellipse for neighbour lookup.
        weight_correction : bool
            Force kriging weights to be non-negative and sum to 1.
        use_old_weight : bool
            Read pre-computed weights from ``weight_file`` instead of solving.
        store_weight : bool
            Write computed weights to ``weight_file`` (skips estimate_block).
        cross_validation : bool
            Leave-one-out cross-validation mode.
        write_mat : bool
            Write matrix for debugging.
        verbose : bool
            Print progress messages.
        weight_file : str
            Path to the weight file (required when use_old_weight or store_weight).
        bounds : tuple(float, float) or None
            (lower, upper) clipping bounds for the estimate.
            None means no clipping (uses Fortran defaults: [-huge, +huge]).
        sk_mean : float
            Global mean for simple kriging (unbias=0). Default 0.0.
        """
        # Allocate Fortran object.
        # Store the handle as a plain Python int so every call site wraps
        # it fresh with ctypes.c_int64(self._handle).  Passing a ctypes
        # object directly can cause an OSError / access-violation because
        # ctypes may pass the object pointer instead of its integer value.
        _h_tmp = ctypes.c_int64(0)
        _krige_create(ctypes.byref(_h_tmp))
        self._handle: int = _h_tmp.value

        # build bounds array: Fortran default is [-huge, +huge]; replicate that here
        import sys
        _huge = sys.float_info.max
        c_bounds = _farray(bounds if bounds is not None else [-_huge, _huge])

        _krige_initialize(_h(self._handle),
            _c_int(ndim),
            _c_int(nvar),
            _c_int(ndrift),
            _c_int(unbias),
            _c_int(nsim),
            _c_int(int(anisotropic_search)),
            _c_int(int(weight_correction)),
            _c_int(int(use_old_weight)),
            _c_int(int(store_weight)),
            _c_int(int(cross_validation)),
            _c_int(int(write_mat)),
            _c_int(int(verbose)),
            weight_file.encode("utf-8") if weight_file else b"",
            _dptr(c_bounds),
            _c_double(sk_mean),
        )

        # store for convenience
        self.ndim   = ndim
        self.nvar   = nvar
        self.ndrift = ndrift
        self.nsim   = nsim

    # ------------------------------------------------------------------
    def set_obs(
        self,
        ivar: int,
        coord: np.ndarray,
        value: np.ndarray,
        variance: Optional[np.ndarray] = None,
        nmax: Optional[int] = None,
        maxdist: Optional[float] = None,
    ):
        """
        Set observations for variable ``ivar``.

        Drift values are set separately via :meth:`set_obs_drift` after this
        call, when ``ndrift > 0``.

        Parameters
        ----------
        ivar : int
            Variable index, 1-based.
        coord : ndarray, shape **(nobs, ndim)**
            Observation coordinates. Rows are points, columns are spatial
            dimensions — standard Python/NumPy convention. The wrapper
            transposes to Fortran's (ndim, nobs) internally.
        value : ndarray, shape (nobs,)
            Observed values.
        variance : ndarray, shape (nobs,), optional
            Per-observation measurement error variance added to the diagonal
            of the covariance matrix. Defaults to zeros (no measurement error).
        nmax : int, optional
            Maximum number of neighbours. Default: use all observations.
        maxdist : float, optional
            Maximum search distance. Default: unlimited.
        """
        import sys
        assert coord.shape[1] == self.ndim, (
            f"coord should be (nobs, ndim={self.ndim}), got {coord.shape}. "
            "Rows are points, columns are spatial dimensions.")

        coord_f  = _coord_to_fortran(coord)        # (nobs, ndim) -> (ndim, nobs) F-order
        value_f  = _farray(value.ravel())           # ensure 1-D
        nobs     = coord_f.shape[1]
        ndim_c   = coord_f.shape[0]

        # variance: always passed; default to zeros
        var_f = _farray(variance) if variance is not None else _farray(np.zeros(nobs))

        # nmax/maxdist: pass huge values when not specified (Fortran treats as "unlimited")
        c_nmax    = _c_int(nmax    if nmax    is not None else np.iinfo(np.int32).max)
        c_maxdist = _c_double(maxdist if maxdist is not None else sys.float_info.max)

        _krige_set_obs(_h(self._handle),
            _c_int(ivar), _c_int(nobs), _c_int(ndim_c),
            _dptr(coord_f), _dptr(value_f), _dptr(var_f),
            c_nmax, c_maxdist,
        )

    # ------------------------------------------------------------------
    def set_obs_drift(self, ivar: int, drift: np.ndarray):
        """
        Set external drift values at observation locations for variable ``ivar``.

        Call after :meth:`set_obs` for the same ``ivar``.
        Only needed when ``ndrift > 0`` was passed to the constructor.

        Parameters
        ----------
        ivar : int
            Variable index, 1-based.
        drift : ndarray, shape **(nobs, ndrift)**
            Drift values. Rows are observations, columns are drift functions.
            Transposed to (ndrift, nobs) internally before calling Fortran.
        """
        drift_f  = _drift_to_fortran(drift)   # (nobs, ndrift) -> (ndrift, nobs)
        ndrift_c = drift_f.shape[0]
        nobs     = drift_f.shape[1]
        assert ndrift_c == self.ndrift, (
            f"drift has {ndrift_c} column(s) but ndrift={self.ndrift} was declared. "
            "drift must be shape (nobs, ndrift)."
        )
        _krige_set_obs_drift(_h(self._handle),
            _c_int(ivar), _c_int(ndrift_c), _c_int(nobs),
            _dptr(drift_f),
        )

    # ------------------------------------------------------------------
    def set_vgm(self, ivar: int, jvar: int, spec: str):
        """
        Add a nested variogram structure for the (ivar, jvar) pair.

        Parameters
        ----------
        ivar, jvar : int
            Variable indices (1-based). Use ivar=jvar for auto-variograms,
            ivar≠jvar for cross-variograms. The LMC constraint
            b12² ≤ b11 × b22 must be satisfied for each nested structure.
        spec : str
            Space-separated variogram specification:
            ``"vtype nugget sill a_minor1 a_major a_minor2 azimuth dip plunge"``

            vtype is one of: sph, exp, gau, pow, lin, hol, bsq, cir.

        Example
        -------
        >>> k.set_vgm(1, 1, "sph 100.0 900.0 500.0 1000.0 500.0 0.0 0.0 0.0")
        """
        _krige_set_vgm(_h(self._handle),
            _c_int(ivar), _c_int(jvar),
            spec.encode("utf-8"),
        )

    # ------------------------------------------------------------------
    def set_grid(
        self,
        coord: np.ndarray,
        rangescale: Optional[np.ndarray] = None,
        localnugget: Optional[np.ndarray] = None,
    ):
        """
        Set the estimation grid for **point kriging** (one node per block).

        For block kriging use :meth:`set_grid_block`.
        For cross-validation use :meth:`set_grid_cv`.
        Drift is set separately via :meth:`set_grid_drift` when ``ndrift > 0``.

        Parameters
        ----------
        coord : ndarray, shape **(ngrid, ndim)**
            Grid coordinates. Rows are grid nodes, columns are spatial dimensions.
        rangescale : ndarray, shape (ngrid,), optional
            Per-block variogram range scaling factor. Values > 1 increase the
            effective range, useful to account for data sparsity.
            Default: 1.0 for all blocks.
        localnugget : ndarray, shape (ngrid,), optional
            Additional nugget added per block to model local uncertainty.
            Default: 0.0 for all blocks.
        """
        assert coord.shape[1] == self.ndim, (
            f"coord should be (ngrid, ndim={self.ndim}), got {coord.shape}. "
            "Rows are points, columns are spatial dimensions.")

        coord_f = _coord_to_fortran(coord)   # (ngrid, ndim) -> (ndim, ngrid)
        ngrid   = coord_f.shape[1]
        ndim_c  = coord_f.shape[0]

        rs_f = (_farray(rangescale)
                if rangescale  is not None else _farray(np.ones(ngrid)))
        ln_f = (_farray(localnugget)
                if localnugget is not None else _farray(np.zeros(ngrid)))

        _krige_set_grid(_h(self._handle),
            _c_int(ngrid), _c_int(ndim_c), _dptr(coord_f),
            _dptr(rs_f), _dptr(ln_f),
        )

    # ------------------------------------------------------------------
    def set_grid_block(
        self,
        coord: np.ndarray,
        block_type: int,
        nblockpnt: np.ndarray,
        pointweight: Optional[np.ndarray] = None,
        rangescale: Optional[np.ndarray] = None,
        localnugget: Optional[np.ndarray] = None,
    ):
        """
        Set the estimation grid for **block kriging**.

        Drift is set separately via :meth:`set_grid_drift` when ``ndrift > 0``.

        Parameters
        ----------
        coord : ndarray, shape **(ngrid, ndim)**
            Sub-node coordinates across all blocks (total ngrid = sum(nblockpnt)).
        block_type : int
            -4 = Gaussian quadrature nodes (auto-generated);
            >0 = user-supplied sub-nodes (coord contains sub-node positions).
        nblockpnt : ndarray of int, shape (nblock,)
            Number of sub-nodes per block.
        pointweight : ndarray, shape (sum(nblockpnt),), optional
            Weight of each sub-node. Uniform weights (1/nblockpnt) used if omitted.
        rangescale : ndarray, shape (nblock,), optional
            Per-block variogram range scaling. Default: 1.0.
        localnugget : ndarray, shape (nblock,), optional
            Per-block additional nugget. Default: 0.0.
        """
        assert coord.shape[1] == self.ndim, (
            f"coord should be (ngrid, ndim={self.ndim}), got {coord.shape}.")

        coord_f = _coord_to_fortran(coord)
        ngrid   = coord_f.shape[1]
        ndim_c  = coord_f.shape[0]
        nbp_f   = np.ascontiguousarray(nblockpnt, dtype=np.int32)
        nblock  = len(nbp_f)

        if pointweight is not None:
            pw_f = _farray(pointweight)
        else:
            # uniform weights: 1/nblockpnt for each sub-node
            pw_f = _farray(np.repeat(1.0 / nbp_f, nbp_f))

        rs_f = (_farray(rangescale)
                if rangescale  is not None else _farray(np.ones(nblock)))
        ln_f = (_farray(localnugget)
                if localnugget is not None else _farray(np.zeros(nblock)))

        _krige_set_grid_block(_h(self._handle),
            _c_int(block_type),
            _c_int(ngrid), _c_int(ndim_c), _dptr(coord_f),
            _c_int(nblock), _iptr(nbp_f),
            _dptr(pw_f),                   # Fortran derives length via sum(nblockpnt)
            _dptr(rs_f), _dptr(ln_f),
        )

    # ------------------------------------------------------------------
    def set_grid_cv(self):
        """
        Set up the grid for **cross-validation** mode.

        No coordinate argument is needed — Fortran derives the grid from the
        observation coordinates automatically.  Call instead of :meth:`set_grid`
        when ``cross_validation=True`` was passed to the constructor.
        """
        _krige_set_grid_cv(_h(self._handle))

    # ------------------------------------------------------------------
    def set_grid_drift(self, drift: np.ndarray):
        """
        Set external drift values at grid/block locations.

        Call after :meth:`set_grid`, :meth:`set_grid_block`, or
        :meth:`set_grid_cv`. Only needed when ``ndrift > 0``.

        Parameters
        ----------
        drift : ndarray, shape **(nblocks, ndrift)**
            Drift values. Rows are blocks, columns are drift functions.
            Note: use **nblocks** (number of blocks), not ngrid (number of
            sub-nodes), even for block kriging.
            Transposed to (ndrift, nblocks) internally before calling Fortran.
        """
        drift_f  = _drift_to_fortran(drift)   # (nblocks, ndrift) -> (ndrift, nblocks)
        ndrift_c = drift_f.shape[0]
        nblocks  = drift_f.shape[1]
        _krige_set_grid_drift(_h(self._handle),
            _c_int(ndrift_c), _c_int(nblocks),
            _dptr(drift_f),
        )

    # ------------------------------------------------------------------
    def set_sim(
        self,
        randpath: Optional[np.ndarray] = None,
        sample: Optional[np.ndarray] = None,
    ):
        """
        Set up Sequential Gaussian Simulation parameters.

        Call after :meth:`set_grid` and before :meth:`set_search`.
        Only needed when ``nsim > 0``.

        Parameters
        ----------
        randpath : ndarray of int, shape (nblocks,), optional
            Random visiting order for the block loop.
            Generated with a random permutation if omitted.
        sample : ndarray, shape (nsim, nblocks), optional
            Pre-drawn standard-normal samples used to add simulated variability.
            Drawn from N(0,1) if omitted.
        """
        # Python generates defaults so Fortran always receives concrete arrays.
        # We need the block count; retrieve it from the Fortran object.
        nb = _c_int(0)
        _krige_get_nblocks(_h(self._handle), ctypes.byref(nb))
        nblocks = nb.value

        if randpath is not None:
            rp_f = np.ascontiguousarray(randpath, dtype=np.int32)
        else:
            # random permutation of 1..nblocks (1-based for Fortran)
            rp_f = np.ascontiguousarray(
                np.random.permutation(nblocks) + 1, dtype=np.int32)

        if sample is not None:
            s_f    = _farray(sample)
            nsim_c = s_f.shape[0]
            n_s    = s_f.shape[1]
        else:
            nsim_c = self.nsim
            n_s    = nblocks
            s_f    = _farray(np.random.standard_normal((nsim_c, n_s)))

        _krige_set_sim(_h(self._handle),
            _c_int(nblocks), _iptr(rp_f),         # nblocks covers both randpath and sample
            _c_int(nsim_c), _dptr(s_f),
        )

    # ------------------------------------------------------------------
    def set_search(
        self,
        ivar: int = 1,
        anis1: float = 1.0,
        anis2: float = 1.0,
        azimuth: float = 0.0,
        dip: float = 0.0,
        plunge: float = 0.0,
    ):
        """
        Build the KD-tree and configure the search ellipse for variable ``ivar``.
        Call once per variable after :meth:`set_obs` (and :meth:`set_sim` for SGSIM).

        Parameters
        ----------
        ivar : int
            Variable index (1-based).
        anis1 : float
            Horizontal anisotropy ratio (minor / major range). 1.0 = isotropic.
        anis2 : float
            Vertical anisotropy ratio (vertical / major range). 1.0 = isotropic.
        azimuth : float
            Azimuth of the major axis (degrees, clockwise from North).
        dip : float
            Dip angle (degrees, positive downward).
        plunge : float
            Plunge angle (degrees).
        """
        _krige_set_search(_h(self._handle),
            _c_int(ivar),
            _c_double(anis1), _c_double(anis2),
            _c_double(azimuth), _c_double(dip), _c_double(plunge),
        )

    # ------------------------------------------------------------------
    def solve(self):
        """
        Run the kriging or SGSIM loop over all blocks.
        Calls prepare(), then the parallel block loop internally.
        """
        _krige_solve(_h(self._handle))

    # ------------------------------------------------------------------
    def get_results(self):
        """
        Retrieve the kriging estimates and variances after :meth:`solve`.

        Returns
        -------
        estimate : ndarray
            Shape **(ngrid,)** for ordinary kriging; shape
            **(nsim, ngrid)** for SGSIM.
        variance : ndarray, shape (nblocks,)
            Kriging variance at each block.

        Example
        -------
        >>> est, var = k.get_results()
        >>> kriging_estimate = est[0]          # shape (nblocks,)
        >>> sim_realisation1 = est[0]          # same for nsim=1
        """
        n_blocks = _c_int(0)
        n_sim    = _c_int(0)
        _krige_get_nblocks(_h(self._handle), ctypes.byref(n_blocks))
        _krige_get_nsim(_h(self._handle), ctypes.byref(n_sim))

        nb = n_blocks.value
        ns = n_sim.value

        estimate = _farray(np.empty((ns, nb), dtype=np.float64))
        variance = _farray(np.empty(nb,       dtype=np.float64))

        _krige_get_estimate(_h(self._handle), _c_int(ns), _c_int(nb), _dptr(estimate))
        _krige_get_variance(_h(self._handle), _c_int(nb),              _dptr(variance))

        # Return (ngrid,) for kriging (ns==1) or (nsim, ngrid) for SGSIM
        if ns == 1:
            return estimate[0].copy(), variance
        return estimate.copy(), variance

    # ------------------------------------------------------------------
    def __del__(self):
        if self._handle != 0:
            _tmp = ctypes.c_int64(self._handle)
            _krige_destroy(ctypes.byref(_tmp))
            self._handle = 0

    # ------------------------------------------------------------------
    def __repr__(self):
        return (f"Kriging(ndim={self.ndim}, nvar={self.nvar}, "
                f"ndrift={self.ndrift}, nsim={self.nsim})")


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def ordinary_kriging(
    obs_coord: np.ndarray,
    obs_value: np.ndarray,
    grid_coord: np.ndarray,
    variogram_spec: str,
    nmax: int = 20,
    maxdist: Optional[float] = None,
    search_anis1: float = 1.0,
    search_anis2: float = 1.0,
    search_azimuth: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One-shot ordinary kriging with a single isotropic (or anisotropic) variogram.

    Parameters
    ----------
    obs_coord : ndarray, shape **(nobs, ndim)**
        Observation coordinates. Rows are points, columns are spatial dimensions.
    obs_value : ndarray, shape (nobs,)
        Observation values.
    grid_coord : ndarray, shape **(ngrid, ndim)**
        Grid coordinates to estimate.
    variogram_spec : str
        Variogram specification string passed to :meth:`Kriging.set_vgm`.
    nmax : int
        Maximum number of neighbours.
    maxdist : float, optional
        Maximum search distance.
    search_anis1, search_anis2 : float
        Anisotropy ratios for search ellipse (1.0 = isotropic).
    search_azimuth : float
        Azimuth of search ellipse major axis (degrees from North).

    Returns
    -------
    estimate : ndarray, shape (ngrid,)
    variance : ndarray, shape (ngrid,)

    Example
    -------
    >>> est, var = ordinary_kriging(
    ...     obs_coord, obs_value, grid_coord,
    ...     variogram_spec="sph 100 900 500 1000 500 0 0 0",
    ...     nmax=20)
    """
    assert obs_coord.ndim == 2 and obs_coord.shape[0] >= obs_coord.shape[1], (
        f"obs_coord should be (nobs, ndim) with nobs >= ndim, got shape {obs_coord.shape}. "
        "Rows are points, columns are spatial dimensions."
    )
    assert obs_coord.shape[0] == obs_value.shape[0], (
        f"obs_coord has {obs_coord.shape[0]} rows but obs_value has {obs_value.shape[0]} elements."
    )
    ndim = obs_coord.shape[1]   # (nobs, ndim) -> ndim is axis 1
    k = Kriging(ndim=ndim, nvar=1)
    k.set_obs(ivar=1, coord=obs_coord, value=obs_value,
              nmax=nmax, maxdist=maxdist)
    k.set_vgm(ivar=1, jvar=1, spec=variogram_spec)
    k.set_grid(coord=grid_coord)
    k.set_search(ivar=1, anis1=search_anis1, anis2=search_anis2,
                 azimuth=search_azimuth)
    k.solve()
    est, var = k.get_results()   # est is already (ngrid,) for kriging
    return est, var


def cokriging(
    obs_coords: list[np.ndarray],
    obs_values: list[np.ndarray],
    grid_coord: np.ndarray,
    variogram_specs: dict,
    nmax: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One-shot ordinary co-kriging with multiple variables.

    Parameters
    ----------
    obs_coords : list of ndarray, each shape **(nobs_i, ndim)**
        Observation coordinates per variable. Rows are points.
    obs_values : list of ndarray, each shape (nobs_i,)
        Observation values per variable.
    grid_coord : ndarray, shape **(ngrid, ndim)**
        Grid coordinates.
    variogram_specs : dict
        Mapping (ivar, jvar) -> spec string. Both (i,j) and (j,i) can be
        provided; if only (i,j) is given, (j,i) will mirror it automatically
        (handled inside Fortran set_vgm).
    nmax : int
        Maximum neighbours per variable.

    Returns
    -------
    estimate : ndarray, shape (ngrid,)
    variance : ndarray, shape (ngrid,)

    Example
    -------
    >>> est, var = cokriging(
    ...     obs_coords=[coord1, coord2],
    ...     obs_values=[val1, val2],
    ...     grid_coord=grid,
    ...     variogram_specs={
    ...         (1,1): "sph 100 900 500 1000 500 0 0 0",
    ...         (2,2): "sph  50 450 500 1000 500 0 0 0",
    ...         (1,2): "sph   0 600 500 1000 500 0 0 0",
    ...     })
    """
    nvar = len(obs_coords)
    ndim = obs_coords[0].shape[1]   # (nobs, ndim) -> ndim is axis 1
    k = Kriging(ndim=ndim, nvar=nvar)

    for i, (coord, value) in enumerate(zip(obs_coords, obs_values), start=1):
        k.set_obs(ivar=i, coord=coord, value=value, nmax=nmax)

    for (iv, jv), spec in variogram_specs.items():
        k.set_vgm(ivar=iv, jvar=jv, spec=spec)

    k.set_grid(coord=grid_coord)

    for i in range(1, nvar + 1):
        k.set_search(ivar=i)

    k.solve()
    est, var = k.get_results()   # est is already (ngrid,) for kriging
    return est, var


def sequential_gaussian_simulation(
    obs_coord: np.ndarray,
    obs_value: np.ndarray,
    grid_coord: np.ndarray,
    variogram_spec: str,
    nsim: int,
    nmax: int = 20,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Sequential Gaussian Simulation.

    Parameters
    ----------
    obs_coord : ndarray, shape **(nobs, ndim)**
        Observation coordinates. Rows are points, columns are spatial dimensions.
    obs_value : ndarray, shape (nobs,)
        Observation values.
    grid_coord : ndarray, shape **(ngrid, ndim)**
        Grid coordinates.
    variogram_spec : str
    nsim : int
        Number of realisations.
    nmax : int
        Maximum neighbours (includes previously simulated nodes).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    simulations : ndarray, shape (nsim, ngrid)
        Each row is one realisation in the original (non-randomised) block order.
    """
    if seed is not None:
        np.random.seed(seed)

    ndim  = obs_coord.shape[1]   # (nobs, ndim) -> ndim is axis 1
    ngrid = grid_coord.shape[0]  # (ngrid, ndim) -> ngrid is axis 0

    k = Kriging(ndim=ndim, nvar=1, nsim=nsim)
    k.set_obs(ivar=1, coord=obs_coord, value=obs_value, nmax=nmax)
    k.set_vgm(ivar=1, jvar=1, spec=variogram_spec)
    k.set_grid(coord=grid_coord)
    # set_sim with no args: Python generates random path and N(0,1) samples
    k.set_sim()
    k.set_search(ivar=1)
    k.solve()

    sims, _ = k.get_results()   # shape (nsim, ngrid)
    return sims
