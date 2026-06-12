"""
_kriging_st.py
==============
Python wrapper for the space-time kriging C API (krige_st_* entry points).

Mirrors the structure of _kriging.py but exposes:
  - SpaceTimeKriging class  — full control over the ST kriging workflow
  - spacetime_kriging()     — one-shot ordinary ST kriging
  - spacetime_cokriging()   — one-shot ordinary ST co-kriging

Coordinate convention (same as base):
  All spatial coord arrays are (nobs, 3) — rows are points, columns are x, y, z.
  Time arrays are 1-D, shape (nobs,), in any consistent unit (e.g. decimal years).

Variogram spec formats:
  Spatial  (9 values): "vtype nugget sill a_major a_minor1 a_minor2 azimuth dip plunge"
  Temporal (4 values): "vtype nugget sill at_k"

ST model parameters (set once via set_st_model):
  model     : 'sum_metric' or 'product_sum'
  transform : 'nug', 'sph', 'exp', 'gau', 'pow', 'bsq', 'cir', or 'lin'
  at        : joint temporal scale (same time units as input)
  time_nugget, time_sill
            : f(dt) scale for the joint temporal distance
  k_ps      : product-sum coefficient k (only for model='product_sum')
"""

import ctypes
import sys
import os
import numpy as np
from typing import Optional, Union

# ---------------------------------------------------------------------------
# Intel OpenMP runtime guards — see _kriging.py for full explanation.
# Must be set before the first import of krigekit in a fresh process.
# ---------------------------------------------------------------------------
if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("KMP_STACKSIZE", "64m")

# ---------------------------------------------------------------------------
# Load the shared library (same library as the base kriging module)
# ---------------------------------------------------------------------------
def _load_lib():
    base = os.path.dirname(__file__)
    if sys.platform == "win32":
        names = ["kriging.dll"]
        # Prepend the package directory to PATH so any Intel runtime DLLs
        # placed alongside kriging.dll (libcaf_ifx.dll, libiomp5md.dll, …)
        # are found by Windows during runtime dynamic loads.
        os.environ['PATH'] = base + os.pathsep + os.environ.get('PATH', '')
    elif sys.platform == "darwin":
        names = ["libkriging.dylib"]
    else:
        names = ["libkriging.so"]
    for name in names:
        path = os.path.join(base, name)
        if os.path.exists(path):
            return ctypes.CDLL(path, winmode=0) if sys.platform == "win32" \
                   else ctypes.CDLL(path)
    raise FileNotFoundError(
        f"Compiled Fortran library not found in {base!r}.\n"
        "Build it first — see README.md for instructions."
    )

_lib = _load_lib()

# ---------------------------------------------------------------------------
# ctypes helpers
# ---------------------------------------------------------------------------
_c_int    = ctypes.c_int
_c_double = ctypes.c_double
_ptr_char = ctypes.POINTER(ctypes.c_char)
_ptr_dbl  = ctypes.POINTER(ctypes.c_double)
_ptr_int  = ctypes.POINTER(ctypes.c_int)
_ptr_int64 = ctypes.POINTER(ctypes.c_int64)
_ptr_void = ctypes.c_void_p

def _cfun(name, argtypes, restype=None):
    fn = getattr(_lib, name)
    fn.argtypes = argtypes
    fn.restype  = restype
    return fn

def _status_cfun(name, argtypes):
    """Wrap an ST C API function that returns ierr.

    The DLL records the detailed Fortran error message separately.  This helper
    checks ierr after every call and raises RuntimeError in Python instead of
    letting the wrapper proceed with an invalid Fortran object state.
    """
    fn = _cfun(name, argtypes, _c_int)

    def checked(*args):
        _check(fn(*args), name)

    checked.__name__ = name
    checked._cfunc = fn
    return checked

# ---------------------------------------------------------------------------
# Declare all krige_st_* C entry points
# ---------------------------------------------------------------------------
_st_create     = _status_cfun("krige_st_create",     [_ptr_int64])
_st_destroy    = _status_cfun("krige_st_destroy",    [_ptr_int64])
_st_initialize = _status_cfun("krige_st_initialize", [
    ctypes.c_int64,
    _c_int, _c_int, _c_int, _c_int,        # nvar ndrift unbias nsim
    _c_int, _c_int, _c_int, _c_int,         # aniso_search weight_corr use_old store_weight
    _c_int, _c_int, _c_int, _c_int,        # cross_val write_mat neglect_err verbose
    ctypes.c_char_p,                        # weight_file
    _ptr_dbl,                               # bounds[2]
    _c_int,                                 # seed
])
_st_set_st_model = _status_cfun("krige_st_set_st_model", [
    ctypes.c_int64, ctypes.c_char_p, ctypes.c_char_p,
    _c_double, _c_double, _c_double, _c_double,
])
_st_set_obs = _status_cfun("krige_st_set_obs", [
    ctypes.c_int64,
    _c_int, _c_int, _c_int,        # ivar, nobs, ndim
    _ptr_dbl, _ptr_dbl, _ptr_dbl,  # coord[ndim+1,nobs], value[nobs], variance[nobs]
    _c_int, _c_double, _c_double,  # nmax, maxdist, sk_mean
])
# --- Shared with spatial kriging (implemented in kriging_capi_common) ---
_st_update_obs_value = _status_cfun("krige_update_obs_value", [
    ctypes.c_int64, _c_int, _c_int, _ptr_dbl,
])
_st_set_obs_drift = _status_cfun("krige_set_obs_drift", [
    ctypes.c_int64, _c_int, _c_int, _c_int, _ptr_dbl,
])
_st_set_grad = _status_cfun("krige_st_set_grad", [
    ctypes.c_int64,
    _c_int, _c_int, _c_int,              # ivar, ngrad, ndim
    _ptr_dbl, _ptr_dbl,                  # coord1[ndim+1,ngrad], coord2[ndim+1,ngrad]
    _ptr_dbl, _ptr_dbl,                  # grad_val[ngrad], variance[ngrad]
    _c_int, _ptr_dbl,                    # ndrift_c, drift_ext[ndrift,ngrad]
])
_st_set_vgm = _status_cfun("krige_st_set_vgm", [
    ctypes.c_int64, _c_int, _c_int, ctypes.c_char_p,
    _c_double, _c_double, _c_double, _c_double, _c_double,
    _c_double, _c_double, _c_double,
])
_st_set_vgm_temporal = _status_cfun("krige_st_set_vgm_temporal", [
    ctypes.c_int64, _c_int, _c_int, ctypes.c_char_p,
    _c_double, _c_double, _c_double,
])
_st_set_vgm_joint_sills = _status_cfun("krige_st_set_vgm_joint_sills", [
    ctypes.c_int64, _c_int, _c_int, _c_int, _ptr_dbl,
])
_st_set_grid = _status_cfun("krige_st_set_grid", [
    ctypes.c_int64, _c_int, _ptr_dbl, _ptr_dbl, _ptr_dbl, _ptr_dbl,
])
_st_set_grid_block = _status_cfun("krige_st_set_grid_block", [
    ctypes.c_int64, _c_int,          # nblocks
    _ptr_dbl, _ptr_dbl,              # coord[3,nblocks], time[nblocks]
    _ptr_int, _c_int,                # nblockpnt[nblocks], npnts_total
    _ptr_dbl, _ptr_dbl, _ptr_dbl,   # blockcoord, blocktime, pointweight
    _ptr_dbl, _ptr_dbl,              # rangescale, localnugget
])
_st_set_grid_cv    = _status_cfun("krige_set_grid_cv",    [ctypes.c_int64])
_st_set_grid_drift = _status_cfun("krige_st_set_grid_drift", [
    ctypes.c_int64, _c_int, _c_int, _ptr_dbl,
])
_st_set_sim = _status_cfun("krige_st_set_sim", [
    ctypes.c_int64, _c_int, ctypes.c_void_p, _c_int, ctypes.c_void_p,
])
_st_set_search = _status_cfun("krige_st_set_search", [
    ctypes.c_int64, _c_int,
    _c_double, _c_double, _c_double, _c_double, _c_double, _c_double,
    _c_int,
])
_st_prepare       = _status_cfun("krige_prepare",        [ctypes.c_int64])
_st_reset_vgm     = _status_cfun("krige_st_reset_vgm",  [ctypes.c_int64, _c_int, _c_int])
_st_solve         = _status_cfun("krige_solve",          [ctypes.c_int64, _c_int, _c_int])
_st_get_nblocks   = _status_cfun("krige_get_nblocks",   [ctypes.c_int64, _ptr_int])
_st_get_nsim      = _status_cfun("krige_get_nsim",      [ctypes.c_int64, _ptr_int])
_st_get_block_coord = _status_cfun("krige_get_block_coord", [
    ctypes.c_int64, _c_int, _c_int, _ptr_dbl,
])
_st_get_estimate     = _status_cfun("krige_get_estimate",      [ctypes.c_int64, _c_int, _c_int, _ptr_dbl])
_st_get_estimate_all = _status_cfun("krige_get_estimate_all",  [ctypes.c_int64, _c_int, _c_int, _c_int, _ptr_dbl])
_st_get_variance     = _status_cfun("krige_get_variance",      [ctypes.c_int64, _c_int, _ptr_dbl])
_st_get_variance_all = _status_cfun("krige_get_variance_all",  [ctypes.c_int64, _c_int, _c_int, _ptr_dbl])
_st_to_str           = _cfun("krige_to_str",                   [ctypes.c_int64], _ptr_void)
_st_get_factor_info  = _status_cfun("krige_get_factor_info",   [
    ctypes.c_int64, _ptr_int, _ptr_int, _ptr_int,
])
_st_get_factor_matrices = _status_cfun("krige_get_factor_matrices", [
    ctypes.c_int64, _c_int, _c_int, _ptr_dbl, _ptr_dbl, _ptr_dbl,
])
_st_get_factor_system = _status_cfun("krige_get_factor_system", [
    ctypes.c_int64, _c_int, _c_int, _c_int, _ptr_dbl, _ptr_dbl,
])
_st_free_weight_store  = _status_cfun("krige_free_weight_store",  [ctypes.c_int64])
_st_get_weight_dims    = _status_cfun("krige_get_weight_dims",    [ctypes.c_int64, _ptr_int, _ptr_int, _ptr_int])
_st_get_weight_nnear   = _status_cfun("krige_get_weight_nnear",   [ctypes.c_int64, _c_int, _c_int, _ptr_int])
_st_get_weight_inear   = _status_cfun("krige_get_weight_inear",   [ctypes.c_int64, _c_int, _c_int, _c_int, _ptr_int])
_st_get_weight_data    = _status_cfun("krige_get_weight_data",    [ctypes.c_int64, _c_int, _c_int, _c_int, _c_int, _ptr_dbl])
_st_get_weight_var     = _status_cfun("krige_get_weight_var",     [ctypes.c_int64, _c_int, _c_int, _ptr_dbl])
_st_set_weights        = _status_cfun("krige_set_weights",        [
    ctypes.c_int64, _c_int, _c_int, _c_int, _c_int,
    _ptr_int, _ptr_int, _ptr_dbl, _ptr_int, _ptr_dbl,
])
_st_get_max_threads  = _cfun("krige_st_get_max_threads", [_ptr_int])
_st_get_num_threads  = _cfun("krige_st_get_num_threads", [_ptr_int])
# krige_solver_stats(handle, out[3])  — shared with spatial kriging (in kriging_capi_common)
# out[0]=n_chol_ok  out[1]=n_ssytrf_fact  out[2]=n_ssytrf_reuse
_st_solver_stats     = _cfun("krige_solver_stats",
                              [ctypes.c_int64, ctypes.POINTER(ctypes.c_int)])
_get_last_error      = _cfun("krige_get_last_error", [_ptr_char, _c_int], _c_int)

# ---------------------------------------------------------------------------
# Helpers (same pattern as _kriging.py)
# ---------------------------------------------------------------------------

# Legacy aliases used by older user code — map to the canonical 3-char prefix
# that vtype_from_str recognises.
_VTYPE_ALIASES = {'linear': 'lin', 'bounded': 'exp', 'power': 'pow'}

def _normalize_vtype(s: str) -> str:
    """Normalise a variogram type name to the 3-char code expected by Fortran."""
    return _VTYPE_ALIASES.get(s.lower(), s.lower())

def _farray(a, dtype=np.float64):
    return np.asfortranarray(a, dtype=dtype)

def _fempty(shape, dtype=np.float64):
    return np.empty(shape, dtype=dtype, order="F")

def _coordS_to_fortran(coord: np.ndarray) -> np.ndarray:
    """(nobs, ndim) → Fortran (ndim, nobs); ndim must be 1, 2, or 3 (spatial only)."""
    a = np.asarray(coord, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] not in (1, 2, 3):
        raise ValueError(
            f"coord must be (nobs, ndim) with ndim=1, 2, or 3, got {a.shape}"
        )
    return np.asfortranarray(a.T)

def _coordST_to_fortran(coord: np.ndarray) -> np.ndarray:
    """(n, ndim+1) → Fortran (ndim+1, n); ndim must be 1, 2, or 3, last column is time."""
    a = np.asarray(coord, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] not in (2, 3, 4):
        raise ValueError(
            f"coord must be (n, ndim+1) with ndim=1, 2, or 3 [x[,y[,z]],t], got {a.shape}"
        )
    return np.asfortranarray(a.T)

def _dptr(a):
    return a.ctypes.data_as(_ptr_dbl)

def _iptr(a):
    return a.ctypes.data_as(_ptr_int)

def _h(handle: int) -> ctypes.c_int64:
    return ctypes.c_int64(handle)

def _last_error() -> str:
    """Return the last Fortran error message recorded by kriging.dll."""
    buf = ctypes.create_string_buffer(4096)
    _get_last_error(buf, _c_int(len(buf)))
    return buf.value.decode("utf-8", errors="replace").strip()

def _check(ierr: int, call_name: str) -> None:
    """Raise a Python exception when a Fortran C API call reports failure."""
    if int(ierr) != 0:
        msg = _last_error() or f"{call_name} failed with ierr={int(ierr)}"
        raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# SpaceTimeKriging class
# ---------------------------------------------------------------------------
class SpaceTimeKriging:
    """
    Python interface to the Fortran t_kriging_st space-time kriging engine.

    Supports 3D spatial + 1D temporal data, sum-metric and product-sum
    covariance models, ordinary/simple kriging, co-kriging, ST gradient
    constraints, and SGSIM
    (primary variable only, conditioned on secondary observations).

    Coordinate convention
    ---------------------
    Observation coord arrays use **(nobs, ndim+1)** shape — the first ``ndim``
    columns are spatial (x, y [, z]) and the last column is time.  ``ndim`` may
    be 2 (x, y, t) or 3 (x, y, z, t) and is inferred from the first
    :meth:`set_obs` call.  :meth:`set_grid` accepts either the same combined
    ``(ngrid, ndim+1)`` format or split ``(ngrid, ndim)`` + ``time`` arrays.

    Typical workflow (single variable, sum-metric)
    -----------------------------------------------
    >>> k = SpaceTimeKriging(nvar=1)
    >>> k.set_st_model(model='sum_metric', transform='bounded', at=5.0)
    >>> k.set_obs(ivar=1, coord=obs_coord_st, value=obs_value,
    ...           nmax=30, maxdist=5000)
    >>> k.set_vgm(ivar=1, jvar=1, vtype="sph", nugget=0, sill=0.8, a_major=1000, a_minor1=500, a_minor2=200)
    >>> k.set_vgm_temporal(ivar=1, jvar=1, spec="exp 0 0.6 10.0")
    >>> k.set_vgm_joint_sills(ivar=1, jvar=1, sills=[0.4])
    >>> k.set_grid(coord=grid_coord, time=grid_time)
    >>> k.set_search(ivar=1)
    >>> k.solve()
    >>> estimate, variance = k.get_results()
    >>> del k
    """

    def __init__(
        self,
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
        neglect_error: bool = True,
        verbose: bool = False,
        weight_file: str = "",
        bounds: Optional[tuple] = None,
        seed: Optional[int] = None,
    ):
        import random as _random
        _h_tmp = ctypes.c_int64(0)
        _st_create(ctypes.byref(_h_tmp))
        self._handle: int = _h_tmp.value

        import sys as _sys
        _huge = _sys.float_info.max
        c_bounds = _farray(bounds if bounds is not None else [-_huge, _huge])
        seed = seed or _random.randint(0, 2**31 - 1)

        _st_initialize(
            _h(self._handle),
            _c_int(nvar), _c_int(ndrift), _c_int(unbias), _c_int(nsim),
            _c_int(int(anisotropic_search)), _c_int(int(weight_correction)),
            _c_int(int(use_old_weight)),     _c_int(int(store_weight)),
            _c_int(int(cross_validation)),   _c_int(int(write_mat)),
            _c_int(int(neglect_error)),      _c_int(int(verbose)),
            weight_file.encode("utf-8") if weight_file else b"",
            _dptr(c_bounds),
            _c_int(seed),
        )

        self.nvar   = nvar
        self.ndrift = ndrift
        self.nsim   = nsim
        self.verbose = verbose
        self.ndim   = None   # set on first set_obs call (2 or 3)
        self._nobs = [0 for _ in range(nvar)]

    # ------------------------------------------------------------------
    def set_st_model(
        self,
        model: str = "sum_metric",
        transform: str = "linear",
        at: float = 1.0,
        time_nugget: float = 0.0,
        time_sill: float = 1.0,
        k_ps: float = 0.0,
    ):
        r"""
        Set global space-time model parameters.  Must be called before set_vgm.

        Parameters
        ----------
        model : str
            ``'sum_metric'`` or ``'product_sum'``.
        transform : str
            Variogram type used for f(dt): ``'nug'`` | ``'sph'`` | ``'exp'`` |
            ``'gau'`` | ``'pow'`` | ``'bsq'`` | ``'cir'`` | ``'lin'``.
            Aliases: ``'linear'`` → ``'lin'``, ``'bounded'`` → ``'exp'``,
            ``'power'`` → ``'pow'``.
        at : float
            Joint temporal scale (same time units as observations).
        time_nugget : float
            Nugget jump in f(dt) for the sum-metric model (applied for dt ≠ 0).
        time_sill : float
            Upper scale in f(dt): ``f(dt) = time_nugget + time_sill *
            (1 - corefunc(|dt| / at))`` for dt ≠ 0; f(0) = 0 always.
        k_ps : float
            Product-sum coefficient k (``model='product_sum'`` only).
        """
        model = model.lower()
        assert model in ["sum_metric", "product_sum"], f"model must be 'sum_metric' or 'product_sum'"
        _st_set_st_model(_h(self._handle),
            str(model).encode(), _normalize_vtype(str(transform)).encode(), _c_double(at),
            _c_double(time_nugget), _c_double(time_sill), _c_double(k_ps))

    # ------------------------------------------------------------------
    def set_obs(
        self,
        ivar: int,
        coord: np.ndarray,
        value: np.ndarray,
        time: Optional[np.ndarray] = None,
        variance: Optional[np.ndarray] = None,
        nmax: Optional[int] = None,
        maxdist: Optional[float] = None,
        sk_mean: float = 0.0,
    ):
        """
        Load observations for variable ivar.
        Duplicate checks include all coordinate columns including time, so repeated
        spatial locations are allowed at different times.

        Accepts two coordinate formats — pick whichever matches your workflow:

        **Combined format** (default)::

            coord : (nobs, ndim+1) — first ndim columns are spatial (x[,y[,z]]),
                    last column is time; ndim must be 1, 2, or 3.
            time  : omitted (None)

        **Split format** (explicit time array)::

            coord : (nobs, ndim)  spatial coordinates only
            time  : (nobs,)       observation times

        ``ndim`` is inferred from the first :meth:`set_obs` call and must be
        consistent across all subsequent calls on the same object.

        Other parameters
        ----------------
        value    : (nobs,)   observed values
        variance : (nobs,)   measurement error variance (default: zeros)
        nmax     : max neighbours
        maxdist  : max search radius in km-equivalent space (same units as h_ST)
        sk_mean  : global mean for simple kriging (unbias=0); default 0
        """
        import sys as _sys
        a = np.asarray(coord, dtype=np.float64)
        if a.ndim != 2:
            raise ValueError(f"coord must be 2-D, got shape {a.shape}")

        if time is None:
            # Combined format — last column is time
            if a.shape[1] not in (2, 3, 4):
                raise ValueError(
                    f"Combined coord must be (nobs, ndim+1) with ndim=1,2,3, got {a.shape}"
                )
            spatial  = a[:, :-1]
            time_arr = a[:, -1]
        else:
            # Split format — spatial coord + separate time array
            if a.shape[1] not in (1, 2, 3):
                raise ValueError(
                    f"Split coord must be (nobs, ndim) with ndim=1,2,3, got {a.shape}"
                )
            spatial  = a
            time_arr = np.asarray(time, dtype=np.float64).ravel()
            if len(time_arr) != len(spatial):
                raise ValueError(
                    f"time length ({len(time_arr)}) != nobs ({len(spatial)})"
                )

        ndim = spatial.shape[1]
        if self.ndim is None:
            self.ndim = ndim
        elif self.ndim != ndim:
            raise ValueError(
                f"ndim mismatch: object was initialised with ndim={self.ndim} "
                f"but this coord implies ndim={ndim}"
            )

        # Assemble (ndim+1, nobs) Fortran array: spatial rows then time row
        nobs    = len(spatial)
        coord_f = np.empty((ndim + 1, nobs), dtype=np.float64, order="F")
        coord_f[:ndim, :] = spatial.T
        coord_f[ndim,  :] = time_arr

        if len(value) != nobs:
            raise ValueError(f"value length ({len(value)}) != nobs ({nobs})")

        value_f = _farray(np.asarray(value).ravel())
        var_f   = _farray(variance) if variance is not None else _farray(np.zeros(nobs))

        c_nmax    = _c_int(nmax if nmax is not None else np.iinfo(np.int32).max)
        c_maxdist = _c_double(maxdist if maxdist is not None else _sys.float_info.max)

        _st_set_obs(_h(self._handle),
            _c_int(ivar), _c_int(nobs), _c_int(ndim),
            _dptr(coord_f), _dptr(value_f), _dptr(var_f),
            c_nmax, c_maxdist, _c_double(sk_mean))
        self._nobs[ivar - 1] = nobs

    # ------------------------------------------------------------------
    def update_obs_value(self, ivar: int, value: np.ndarray):
        """
        Replace observation values for variable ``ivar`` in-place.

        Coordinates and the KD-tree are unchanged.  After solving once with
        stored weights, call this method with new observed values and solve
        again to reuse the existing neighbourhoods and weights.
        """
        if ivar < 1 or ivar > self.nvar:
            raise ValueError(f"ivar must be in [1, {self.nvar}], got {ivar}")
        nobs = self._nobs[ivar - 1]
        value_f = _farray(np.asarray(value, dtype=np.float64).ravel())
        if value_f.size != nobs:
            raise ValueError(
                f"value length ({value_f.size}) must match nobs ({nobs}) "
                f"set by the previous set_obs call for ivar={ivar}"
            )
        _st_update_obs_value(_h(self._handle),
            _c_int(ivar), _c_int(nobs), _dptr(value_f))

    # ------------------------------------------------------------------
    def set_obs_drift(self, ivar: int, drift: np.ndarray):
        """
        Set external drift values at observations for variable ivar.
        drift shape: (nobs, ndrift) — transposed internally.
        """
        drift_f = _farray(np.asarray(drift, dtype=np.float64).T)  # (ndrift, nobs)
        _st_set_obs_drift(_h(self._handle),
            _c_int(ivar), _c_int(drift_f.shape[0]), _c_int(drift_f.shape[1]),
            _dptr(drift_f))

    # ------------------------------------------------------------------
    def set_grad(
        self,
        coord1: np.ndarray,
        coord2: np.ndarray,
        grad_value: np.ndarray,
        ivar: int = 1,
        variance: Optional[np.ndarray] = None,
        drift_ext: Optional[np.ndarray] = None,
    ):
        """
        Set time-aware ST gradient observation pairs.

        ``coord1`` and ``coord2`` must both have shape ``(ngrad, ndim+1)`` with
        columns ``x, y [, z], t`` (same ``ndim`` as :meth:`set_obs`).  The
        constraint is ``Z(coord1[i]) - Z(coord2[i]) = grad_value[i]``.  Because
        time is part of each endpoint coordinate, targets at other times are
        penalized by the temporal covariance model.
        """
        if ivar < 1 or ivar > self.nvar:
            raise ValueError(f"ivar must be in [1, {self.nvar}], got {ivar}")

        c1_f = _coordST_to_fortran(coord1)
        c2_f = _coordST_to_fortran(coord2)
        if c1_f.shape != c2_f.shape:
            raise ValueError(f"coord1 and coord2 shapes differ: {c1_f.T.shape} vs {c2_f.T.shape}")

        ndim_st = c1_f.shape[0]   # ndim+1 rows in the Fortran layout
        ngrad = c1_f.shape[1]
        if ngrad == 0:
            c1_arg = np.zeros((ndim_st, 1), dtype=np.float64, order="F")
            c2_arg = np.zeros((ndim_st, 1), dtype=np.float64, order="F")
            gval = _farray(np.zeros(1, dtype=np.float64))
            gvar = _farray(np.zeros(1, dtype=np.float64))
            de_f = np.zeros((1, 1), dtype=np.float64, order="F")
            ndrift_c = 0
        else:
            c1_arg = c1_f
            c2_arg = c2_f
            gval = _farray(np.asarray(grad_value, dtype=np.float64).ravel())
            if gval.size != ngrad:
                raise ValueError(f"grad_value length ({gval.size}) must match ngrad ({ngrad})")

            if variance is not None:
                gvar = _farray(np.asarray(variance, dtype=np.float64).ravel())
                if gvar.size != ngrad:
                    raise ValueError(f"variance length ({gvar.size}) must match ngrad ({ngrad})")
            else:
                gvar = _farray(np.zeros(ngrad, dtype=np.float64))

            if drift_ext is not None:
                de = np.asarray(drift_ext, dtype=np.float64)
                if de.ndim == 1:
                    de = de.reshape(ngrad, 1)
                if de.ndim != 2 or de.shape[0] != ngrad:
                    raise ValueError(
                        f"drift_ext must be (ngrad, ndrift), got {de.shape} for ngrad={ngrad}"
                    )
                de_f = np.asfortranarray(de.T)
                ndrift_c = de_f.shape[0]
            else:
                de_f = np.zeros((1, max(ngrad, 1)), dtype=np.float64, order="F")
                ndrift_c = 0

        _st_set_grad(
            _h(self._handle),
            _c_int(ivar), _c_int(ngrad), _c_int(ndim_st - 1),
            _dptr(c1_arg), _dptr(c2_arg),
            _dptr(gval), _dptr(gvar),
            _c_int(ndrift_c), _dptr(de_f),
        )

    # ------------------------------------------------------------------
    def set_vgm(
        self, ivar: int, jvar: int, vtype: str,
        nugget: float = 0.0, sill: float = 1.0,
        a_major: float = 1.0,
        a_minor1: Optional[float] = None,
        a_minor2: Optional[float] = None,
        azimuth: float = 0.0, dip: float = 0.0, plunge: float = 0.0,
    ):
        """
        Add one spatial nested structure to vgm(ivar, jvar).
        Call multiple times for nested models.

        Same parameters as :meth:`Kriging.set_vgm` — see that docstring.
        """
        if a_minor1 is None:
            a_minor1 = a_major
        if a_minor2 is None:
            a_minor2 = a_minor1
        _st_set_vgm(_h(self._handle), _c_int(ivar), _c_int(jvar),
                    str(vtype).encode(),
                    _c_double(nugget), _c_double(sill), _c_double(a_major),
                    _c_double(a_minor1), _c_double(a_minor2),
                    _c_double(azimuth), _c_double(dip), _c_double(plunge))

    # ------------------------------------------------------------------
    def set_vgm_temporal(
        self, ivar: int, jvar: int, vtype: str,
        nugget: float = 0.0, sill: float = 1.0, at_k: float = 1.0,
    ):
        """
        Add one temporal nested structure to vgm(ivar, jvar).
        Call multiple times for nested models.

        vtype  : variogram type (e.g. 'sph', 'exp', 'gau')
        nugget : nugget contribution of this structure
        sill   : partial sill of this structure
        at_k   : temporal practical range (same time units as observations)
        """
        _st_set_vgm_temporal(_h(self._handle), _c_int(ivar), _c_int(jvar),
                              str(vtype).encode(),
                              _c_double(nugget), _c_double(sill), _c_double(at_k))

    # ------------------------------------------------------------------
    def set_vgm_joint_sills(self, ivar: int, jvar: int, *sills: float):
        """
        Set joint sills for the sum-metric model.

        Pass one float per spatial nested structure of vgm(ivar, jvar).
        Must be called after all set_vgm() calls for (ivar, jvar).

        Example:
            k.set_vgm_joint_sills(1, 1, 0.05, 0.07)
        """
        arr = _farray(np.asarray(sills, dtype=np.float64))
        _st_set_vgm_joint_sills(_h(self._handle),
            _c_int(ivar), _c_int(jvar), _c_int(len(sills)), _dptr(arr))

    # ------------------------------------------------------------------
    def set_grid(
        self,
        coord: np.ndarray,
        time: Optional[np.ndarray] = None,
        rangescale: Optional[np.ndarray] = None,
        localnugget: Optional[np.ndarray] = None,
    ):
        """
        Set point estimation targets.

        Accepts two coordinate formats — pick whichever matches your workflow:

        **Combined format** (consistent with :meth:`set_obs`)::

            coord : (ngrid, ndim+1) — first ndim columns are spatial (x[,y[,z]]),
                    last column is time; ndim must be 1, 2, or 3.
            time  : omitted (None)

        **Split format** (explicit time array)::

            coord : (ngrid, ndim)  spatial coordinates only
            time  : (ngrid,)       prediction times

        Other parameters
        ----------------
        rangescale  : (ngrid,) local range scale factors (default: ones)
        localnugget : (ngrid,) local nugget additions   (default: zeros)
        """
        a = np.asarray(coord, dtype=np.float64)
        if a.ndim != 2:
            raise ValueError(f"coord must be 2-D, got shape {a.shape}")

        if time is None:
            # Combined format — last column is time (same as set_obs)
            if a.shape[1] not in (2, 3, 4):
                raise ValueError(
                    f"Combined coord must be (ngrid, ndim+1) with ndim=1,2,3, got {a.shape}"
                )
            spatial  = a[:, :-1]
            time_arr = a[:, -1]
        else:
            # Split format — spatial coord + separate time array
            if a.shape[1] not in (1, 2, 3):
                raise ValueError(
                    f"Split coord must be (ngrid, ndim) with ndim=1,2,3, got {a.shape}"
                )
            spatial  = a
            time_arr = np.asarray(time, dtype=np.float64).ravel()
            if len(time_arr) != len(spatial):
                raise ValueError(
                    f"time length ({len(time_arr)}) != ngrid ({len(spatial)})"
                )

        coord_f = _coordS_to_fortran(spatial)
        ngrid   = coord_f.shape[1]
        time_f  = _farray(time_arr)
        rs_f    = _farray(rangescale  if rangescale  is not None else np.ones(ngrid))
        ln_f    = _farray(localnugget if localnugget is not None else np.zeros(ngrid))

        _st_set_grid(_h(self._handle), _c_int(ngrid),
                     _dptr(coord_f), _dptr(time_f), _dptr(rs_f), _dptr(ln_f))

    # ------------------------------------------------------------------
    def set_grid_cv(self):
        """Cross-validation mode: predict at observation locations."""
        _st_set_grid_cv(_h(self._handle))

    # ------------------------------------------------------------------
    def set_grid_drift(self, drift: np.ndarray):
        """
        Drift values at estimation grid.
        drift shape: (ngrid, ndrift).
        """
        drift_f = _farray(np.asarray(drift, dtype=np.float64).T)  # (ndrift, ngrid)
        _st_set_grid_drift(_h(self._handle),
            _c_int(drift_f.shape[0]), _c_int(drift_f.shape[1]), _dptr(drift_f))

    # ------------------------------------------------------------------
    def set_sim(
        self,
        randpath: Optional[np.ndarray] = None,
        sample: Optional[np.ndarray] = None,
    ):
        """
        Prepare SGSIM random path and pre-drawn N(0,1) samples.
        Call after set_grid() and set_obs() but before set_search().
        When randpath/sample are None, Fortran generates them internally.
        """
        rp_ptr = ctypes.c_void_p(0)
        s_ptr  = ctypes.c_void_p(0)
        nblocks = 0
        nsim_c  = 0

        if randpath is not None:
            nblocks = self._get_nblocks_raw()
            if nblocks == 0:
                raise RuntimeError("set_grid must be called before set_sim")
            rp_f = np.ascontiguousarray(
                np.asarray(randpath, dtype=np.int32).ravel(), dtype=np.int32)
            if rp_f.size != nblocks:
                raise ValueError(
                    f"randpath length ({rp_f.size}) must match nblocks ({nblocks})")
            if not np.array_equal(np.sort(rp_f),
                                  np.arange(1, nblocks + 1, dtype=np.int32)):
                raise ValueError("randpath must be a 1-based permutation of 1..nblocks")
            rp_ptr = ctypes.c_void_p(rp_f.ctypes.data)

        if sample is not None:
            samp_f = _farray(np.asarray(sample, dtype=np.float64))
            if samp_f.ndim != 2:
                raise ValueError("sample must be 2-D (nsim, nblocks)")
            nsim_c, n_s = samp_f.shape[0], samp_f.shape[1]
            if nblocks == 0:
                nblocks = n_s
            if nsim_c != self.nsim or n_s != nblocks:
                raise ValueError(
                    f"sample shape ({nsim_c}, {n_s}) must be ({self.nsim}, {nblocks})")
            s_ptr = ctypes.c_void_p(samp_f.ctypes.data)

        _st_set_sim(_h(self._handle), _c_int(nblocks),
                    rp_ptr, _c_int(nsim_c), s_ptr)

    # ------------------------------------------------------------------
    def set_search(
        self,
        ivar: int,
        time_at: float = 1.0,
        anis1: float = 1.0,
        anis2: float = 1.0,
        azimuth: float = 0.0,
        dip: float = 0.0,
        plunge: float = 0.0,
        sector_search: bool = False,
    ):
        """
        Build the space-time KD-tree and configure the search ellipse for variable ``ivar``.

        Call after :meth:`set_obs` (and after :meth:`set_sim` for ivar=1 in SGSIM).

        Parameters
        ----------
        ivar : int
            Variable index (1-based).
        time_at : float
            Temporal scale factor (default 1.0) to convert the time axis into
            km-equivalent search units: the search-tree time coordinate is
            ``t * time_at``. This ensures that L2 distance in the 4D search
            space matches the sum-metric space-time distance:
            ``h_ST = sqrt(h_S^2 + (time_at * dt)^2)``. Normally, you should pass
            the same value as ``at`` in :meth:`set_st_model`.
        anis1 : float
            Spatial minor/major anisotropy ratio (default 1.0).
        anis2 : float
            Spatial vertical/major anisotropy ratio (default 1.0).
        azimuth : float
            Azimuth of the spatial major axis in degrees (default 0.0, clockwise from North).
        dip : float
            Dip angle of the spatial major axis in degrees (default 0.0, positive downward).
        plunge : float
            Plunge angle of the spatial major axis in degrees (default 0.0).
        sector_search : bool
            Enable sector (octant) search limiting candidates per sector.
            If ``True``, candidate neighbours are partitioned into 8 spatial
            octants centered on the prediction location. At most ``nmax``
            (from :meth:`set_obs`) candidates are selected per octant.
            This ensures a balanced spatial distribution of neighbours and prevents
            clustering artifacts. The maximum total neighbours selected is ``8 * nmax``.
            If search anisotropy is enabled, spatial coordinates are rotated/scaled
            according to the anisotropy parameters before sector assignment.
        """
        _st_set_search(_h(self._handle), _c_int(ivar),
                       _c_double(time_at),
                       _c_double(anis1), _c_double(anis2),
                       _c_double(azimuth), _c_double(dip), _c_double(plunge),
                       _c_int(int(sector_search)))

    # ------------------------------------------------------------------
    def solve(self, nthread: int = 0, ncache: Optional[int] = None):
        """Run the ST kriging or SGSIM loop.

        Parameters
        ----------
        nthread : int, optional
            Maximum number of OpenMP threads.  0 (default) lets the OpenMP
            runtime choose (respects ``OMP_NUM_THREADS``).
        ncache : int, optional
            Number of per-thread multi-slot hcache entries for this solve.
            ``None`` keeps the compiled/object default, ``0`` disables hcache,
            and ``1`` builds a one-slot hcache for overhead comparisons.
        """
        ncache_c = -1 if ncache is None else int(ncache)
        _st_solve(_h(self._handle), _c_int(nthread), _c_int(ncache_c))

    # ------------------------------------------------------------------
    def get_factor(self) -> dict:
        """Return cached factor matrices and the assembled linear system."""
        c_npp = ctypes.c_int(0)
        c_p = ctypes.c_int(0)
        c_valid = ctypes.c_int(0)
        _st_get_factor_info(
            _h(self._handle),
            ctypes.byref(c_npp),
            ctypes.byref(c_p),
            ctypes.byref(c_valid),
        )
        valid = bool(c_valid.value)
        npp = c_npp.value
        p = c_p.value

        if not valid:
            return dict(valid=False, npp=0, p=0,
                        L=None, kinv_drift=None, schur=None,
                        matA=None, rhsB=None)

        pg = max(1, p)
        matsize = npp + p
        L_buf = _fempty((npp, npp), dtype=np.float64)
        kinv_buf = _fempty((npp, pg), dtype=np.float64)
        schur_buf = _fempty((pg, pg), dtype=np.float64)
        matA_buf = _fempty((matsize, matsize), dtype=np.float64)
        rhsB_buf = _fempty((self.nvar, matsize), dtype=np.float64)

        _st_get_factor_matrices(
            _h(self._handle),
            _c_int(npp), _c_int(p),
            _dptr(L_buf), _dptr(kinv_buf), _dptr(schur_buf),
        )
        _st_get_factor_system(
            _h(self._handle),
            _c_int(npp), _c_int(p), _c_int(self.nvar),
            _dptr(matA_buf), _dptr(rhsB_buf),
        )

        return dict(
            valid=True, npp=npp, p=p,
            L=np.asarray(L_buf, order="C"),
            kinv_drift=np.asarray(kinv_buf, order="C"),
            schur=np.asarray(schur_buf, order="C"),
            matA=np.asarray(matA_buf, order="C"),
            rhsB=np.asarray(rhsB_buf, order="C"),
        )

    # ------------------------------------------------------------------
    @property
    def solver_stats(self) -> dict:
        """Solver statistics from the most recent :meth:`solve` call.

        Returns a dict with three integer counts that are reset to zero at the
        start of every ``solve()``:

        ``chol_ok``
            Blocks solved by Cholesky factorization (either a fresh factorize
            or a cache hit that reused a previously computed Cholesky factor).
        ``ssytrf_fact``
            Number of SSYTRF (Bunch-Kaufman LDL^T) factorizations performed.
            Each one is O(n³) but occurs only once per unique neighbourhood.
            A non-zero value means Cholesky failed for at least one
            neighbourhood; a value equal to 1 with global search means the
            factorization was done once and cached for all blocks.
        ``ssytrf_reuse``
            Blocks solved by a *cached* SSYTRF factorization using SSYTRS,
            which is O(n²).  When this is large relative to ``ssytrf_fact``
            the SSYTRF caching is working effectively.

        Example — global neighbourhood, Cholesky fails, 10 000 grid blocks::

            k.solve()
            s = k.solver_stats
            # Expected: chol_ok=0, ssytrf_fact=1, ssytrf_reuse=9999
        """
        buf = (ctypes.c_int * 3)(0, 0, 0)
        _st_solver_stats(_h(self._handle), buf)
        return {
            "chol_ok":      buf[0],
            "ssytrf_fact":  buf[1],
            "ssytrf_reuse": buf[2],
        }

    # ------------------------------------------------------------------
    def get_results(self, copy: bool = False, squeeze: bool = True) -> "tuple[np.ndarray, np.ndarray]":
        """
        Retrieve kriging estimate and variance.

        Parameters
        ----------
        copy : bool, default False
            If True, return C-contiguous copies for downstream NumPy/Pandas use.
            If False, return views / Fortran-order arrays when possible.
        squeeze : bool, default True
            If True, return a 1-D estimate when ``nsim == 1``.

        Returns
        -------
        estimate : ndarray, shape (ngrid,) when ``nsim == 1 and squeeze``;
            otherwise shape (ngrid, nsim)  [block first]
        variance : ndarray, shape (ngrid,)
        """
        nb = ctypes.c_int(0)
        ns = ctypes.c_int(0)
        _st_get_nblocks(_h(self._handle), ctypes.byref(nb))
        _st_get_nsim   (_h(self._handle), ctypes.byref(ns))
        nb, ns = nb.value, ns.value
        ns = max(ns, 1)  # plain kriging returns nsim=0; treat as 1 sim

        # Fortran fills out(nblocks, nsim_c) with block index first (see
        # krige_get_estimate in kriging_capi_common.F90).  Python receives a
        # Fortran-order (nb, ns) array where estimate[block, sim].
        estimate = _fempty((nb, ns), dtype=np.float64)
        variance = _fempty(nb, dtype=np.float64)
        _st_get_estimate(_h(self._handle), _c_int(ns), _c_int(nb), _dptr(estimate))
        _st_get_variance(_h(self._handle), _c_int(nb),              _dptr(variance))

        if squeeze and ns == 1:
            est = estimate[:, 0]   # shape (nb,) — all blocks, first sim
        else:
            est = estimate         # shape (nb, ns)

        if copy:
            est = np.array(est, order="C", copy=True)
            variance = np.array(variance, order="C", copy=True)

        return est, variance

    # ------------------------------------------------------------------
    def _get_nblocks_raw(self) -> int:
        n = ctypes.c_int(0)
        _st_get_nblocks(_h(self._handle), ctypes.byref(n))
        return n.value

    # ------------------------------------------------------------------
    def __del__(self):
        if self._handle != 0:
            _tmp = ctypes.c_int64(self._handle)
            try:
                _st_destroy(ctypes.byref(_tmp))
            except Exception:
                pass
            self._handle = 0

    def __repr__(self):
        ptr = _st_to_str(_h(self._handle))
        if ptr:
            return ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8", errors="ignore")
        return f"<SpaceTimeKriging nvar={self.nvar} at {self._handle}>"


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def spacetime_kriging(
    obs_coord: np.ndarray,
    obs_value: np.ndarray,
    grid_coord: np.ndarray,
    grid_time: np.ndarray,
    spatial_spec: "dict | list[dict]",
    temporal_spec: "dict | list[dict]",
    joint_sills: "list[float]",
    model: str = "sum_metric",
    transform: str = "linear",
    at: float = 1.0,
    time_nugget: float = 0.0,
    time_sill: float = 1.0,
    nmax: int = 20,
    maxdist: Optional[float] = None,
    search_anis1: float = 1.0,
    search_anis2: float = 1.0,
    search_azimuth: float = 0.0,
    k_ps: float = 0.0,
    nthread: int = 0,
    ncache: Optional[int] = None,
) -> "tuple[np.ndarray, np.ndarray]":
    """
    One-shot ordinary space-time kriging (single variable).

    Parameters
    ----------
    obs_coord    : (nobs, ndim+1)  observation coordinates — first ndim cols spatial, last col time
    obs_value    : (nobs,)         observed values
    grid_coord   : (ngrid, ndim)   prediction spatial coordinates
    grid_time    : (ngrid,)    prediction times
    spatial_spec : dict or list[dict]  spatial variogram structure(s)
    temporal_spec: dict or list[dict]  temporal variogram structure(s)
    joint_sills  : list[float]         joint sills (sum-metric only)
    model        : 'sum_metric' or 'product_sum'
    transform    : 'nug', 'sph', 'exp', 'gau', 'pow', 'bsq', 'cir', or 'lin'
    at           : joint temporal scale (also used as ``time_at`` for the KD-tree)
    time_nugget, time_sill : temporal variogram nugget/sill for ``set_vgm_temporal``
    nmax         : max neighbours
    maxdist      : max search radius in km-equivalent space (h_ST units)
    nthread      : max OMP threads for this call (0 = OMP default)
    ncache       : per-thread hcache slots for this solve; None uses default

    Returns
    -------
    estimate : (ngrid,)
    variance : (ngrid,)
    """
    k = SpaceTimeKriging(nvar=1)
    k.set_st_model(model=model, transform=transform, at=at,
                   time_nugget=time_nugget, time_sill=time_sill, k_ps=k_ps)
    k.set_obs(ivar=1, coord=obs_coord, value=obs_value,
              nmax=nmax, maxdist=maxdist)
    for spec in ([spatial_spec] if isinstance(spatial_spec, dict) else list(spatial_spec)):
        k.set_vgm(1, 1, **spec)
    for spec in ([temporal_spec] if isinstance(temporal_spec, dict) else list(temporal_spec)):
        k.set_vgm_temporal(1, 1, **spec)
    if model == "sum_metric":
        k.set_vgm_joint_sills(1, 1, *joint_sills)
    k.set_grid(coord=grid_coord, time=grid_time)
    k.set_search(ivar=1, time_at=at,
                 anis1=search_anis1, anis2=search_anis2, azimuth=search_azimuth)
    k.solve(nthread=nthread, ncache=ncache)
    return k.get_results()


def spacetime_cokriging(
    obs_coords: "list[np.ndarray]",
    obs_values: "list[np.ndarray]",
    grid_coord: np.ndarray,
    grid_time:  np.ndarray,
    spatial_specs: dict,
    temporal_specs: dict,
    joint_sills: dict,
    model: str = "sum_metric",
    transform: str = "linear",
    at: float = 1.0,
    time_nugget: float = 0.0,
    time_sill: float = 1.0,
    nmax: int = 20,
    maxdist: Optional[float] = None,
    nthread: int = 0,
    ncache: Optional[int] = None,
) -> "tuple[np.ndarray, np.ndarray]":
    """
    One-shot ordinary space-time co-kriging.

    Parameters
    ----------
    obs_coords   : list of (nobs_i, ndim+1) arrays, one per variable — first ndim cols spatial, last col time
    obs_values   : list of (nobs_i,)   arrays
    grid_coord   : (ngrid, ndim)
    grid_time    : (ngrid,)
    spatial_specs : dict (ivar,jvar) -> dict or list[dict]
    temporal_specs: dict (ivar,jvar) -> dict or list[dict]
    joint_sills  : dict (ivar,jvar) -> list[float]
    nthread      : max OMP threads for this call (0 = OMP default)
    ncache       : per-thread hcache slots for this solve; None uses default

    Returns
    -------
    estimate : (ngrid,)
    variance : (ngrid,)
    """
    nvar = len(obs_coords)
    k = SpaceTimeKriging(nvar=nvar)
    k.set_st_model(model=model, transform=transform, at=at,
                   time_nugget=time_nugget, time_sill=time_sill)

    for i, (coord, value) in enumerate(zip(obs_coords, obs_values), start=1):
        k.set_obs(ivar=i, coord=coord, value=value,
                  nmax=nmax, maxdist=maxdist)

    for (iv, jv), spec in spatial_specs.items():
        for s in ([spec] if isinstance(spec, dict) else list(spec)):
            k.set_vgm(iv, jv, **s)

    for (iv, jv), spec in temporal_specs.items():
        for s in ([spec] if isinstance(spec, dict) else list(spec)):
            k.set_vgm_temporal(iv, jv, **s)

    if model == "sum_metric":
        for (iv, jv), sills in joint_sills.items():
            k.set_vgm_joint_sills(iv, jv, *sills)

    k.set_grid(coord=grid_coord, time=grid_time)

    for i in range(1, nvar + 1):
        k.set_search(ivar=i, time_at=at)

    k.solve(nthread=nthread, ncache=ncache)
    return k.get_results()
