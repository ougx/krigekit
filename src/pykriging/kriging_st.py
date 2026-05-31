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
  transform : 'linear', 'bounded', or 'power'
  at        : joint temporal scale (same time units as input)
  alpha     : power exponent (only for transform='power')
  k_ps      : product-sum coefficient k (only for model='product_sum')
"""

import ctypes
import sys
import os
import numpy as np
from typing import Optional, Union

# ---------------------------------------------------------------------------
# Intel OpenMP runtime guards — see _kriging.py for full explanation.
# Must be set before the first import of pykriging in a fresh process.
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
    _c_int, _c_int, _c_int, _c_int,        # aniso_search weight_corr use_old store_weight
    _c_int, _c_int, _c_int, _c_int,        # cross_val write_mat neglect_err verbose
    ctypes.c_char_p,                        # weight_file
    _ptr_dbl,                               # bounds[2]
    _c_double,                              # sk_mean
    _c_int,                                 # seed
])
_st_set_st_model = _status_cfun("krige_st_set_st_model", [
    ctypes.c_int64, _c_int, _c_int, _c_double, _c_double, _c_double,
])
_st_set_obs = _status_cfun("krige_st_set_obs", [
    ctypes.c_int64,
    _c_int, _c_int,          # ivar, nobs
    _ptr_dbl, _ptr_dbl,      # coord[3,nobs], value[nobs]
    _ptr_dbl, _ptr_dbl,      # time[nobs], variance[nobs]
    _c_int, _c_double, _c_double,  # nmax, maxdist, maxtlag
])
_st_set_obs_drift = _status_cfun("krige_st_set_obs_drift", [
    ctypes.c_int64, _c_int, _c_int, _c_int, _ptr_dbl,
])
_st_set_vgm = _status_cfun("krige_st_set_vgm", [
    ctypes.c_int64, _c_int, _c_int, ctypes.c_char_p,
])
_st_set_vgm_temporal = _status_cfun("krige_st_set_vgm_temporal", [
    ctypes.c_int64, _c_int, _c_int, ctypes.c_char_p,
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
_st_set_grid_cv    = _status_cfun("krige_st_set_grid_cv",    [ctypes.c_int64])
_st_set_grid_drift = _status_cfun("krige_st_set_grid_drift", [
    ctypes.c_int64, _c_int, _c_int, _ptr_dbl,
])
_st_set_sim = _status_cfun("krige_st_set_sim", [
    ctypes.c_int64, _c_int, _ptr_int, _c_int, _ptr_dbl,
])
_st_set_search = _status_cfun("krige_st_set_search", [
    ctypes.c_int64, _c_int,
    _c_double, _c_double, _c_double, _c_double, _c_double,
])
_st_solve         = _status_cfun("krige_st_solve",         [ctypes.c_int64])
_st_get_nblocks   = _status_cfun("krige_st_get_nblocks",   [ctypes.c_int64, _ptr_int])
_st_get_nsim      = _status_cfun("krige_st_get_nsim",      [ctypes.c_int64, _ptr_int])
_st_get_estimate  = _status_cfun("krige_st_get_estimate",  [ctypes.c_int64, _c_int, _c_int, _ptr_dbl])
_st_get_variance  = _status_cfun("krige_st_get_variance",  [ctypes.c_int64, _c_int, _ptr_dbl])
_get_last_error   = _cfun("krige_get_last_error", [_ptr_char, _c_int], _c_int)

# ---------------------------------------------------------------------------
# String constants for the Python API
# ---------------------------------------------------------------------------
_MODEL_MAP = {"sum_metric": 0, "product_sum": 1}
_TRANSFORM_MAP = {"linear": 0, "bounded": 1, "power": 2}

# ---------------------------------------------------------------------------
# Helpers (same pattern as _kriging.py)
# ---------------------------------------------------------------------------
def _farray(a, dtype=np.float64):
    return np.asfortranarray(a, dtype=dtype)

def _fempty(shape, dtype=np.float64):
    return np.empty(shape, dtype=dtype, order="F")

def _coord3_to_fortran(coord: np.ndarray) -> np.ndarray:
    """(nobs, 3) → Fortran (3, nobs)."""
    a = np.asarray(coord, dtype=np.float64)
    assert a.ndim == 2 and a.shape[1] == 3, \
        f"coord must be (nobs, 3), got {a.shape}"
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
    covariance models, ordinary/simple kriging, co-kriging, and SGSIM
    (primary variable only, conditioned on secondary observations).

    Coordinate convention
    ---------------------
    All spatial coord arrays use **(nobs, 3)** shape — rows are points,
    columns are x, y, z.  Time arrays are 1-D, shape (nobs,).

    Typical workflow (single variable, sum-metric)
    -----------------------------------------------
    >>> k = SpaceTimeKriging(nvar=1)
    >>> k.set_st_model(model='sum_metric', transform='bounded', at=5.0)
    >>> k.set_obs(ivar=1, coord=obs_coord, value=obs_value, time=obs_time,
    ...           nmax=30, maxdist=5000, maxtlag=20.0)
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
        sk_mean: float = 0.0,
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
            _c_double(sk_mean),
            _c_int(seed),
        )

        self.nvar   = nvar
        self.ndrift = ndrift
        self.nsim   = nsim
        self.verbose = verbose

    # ------------------------------------------------------------------
    def set_st_model(
        self,
        model: str = "sum_metric",
        transform: str = "linear",
        at: float = 1.0,
        alpha: float = 1.0,
        k_ps: float = 0.0,
    ):
        """
        Set global space-time model parameters.  Must be called before set_vgm.

        Parameters
        ----------
        model     : 'sum_metric' or 'product_sum'
        transform : 'linear' | 'bounded' | 'power'
                    Controls f(dt) used in the joint ST distance:
                      linear  → dw = \|dt\| / at
                      bounded → dw = 1 - exp(-\|dt\| / at)
                      power   → dw = (\|dt\| / at)^alpha
        at        : joint temporal scale (same time units as observations)
        alpha     : power exponent (transform='power' only)
        k_ps      : product-sum coefficient k (model='product_sum' only)
        """
        m = _MODEL_MAP.get(model)
        if m is None:
            raise ValueError(f"model must be 'sum_metric' or 'product_sum', got {model!r}")
        t = _TRANSFORM_MAP.get(transform)
        if t is None:
            raise ValueError(f"transform must be 'linear', 'bounded', or 'power', got {transform!r}")
        _st_set_st_model(_h(self._handle),
            _c_int(m), _c_int(t), _c_double(at), _c_double(alpha), _c_double(k_ps))

    # ------------------------------------------------------------------
    def set_obs(
        self,
        ivar: int,
        coord: np.ndarray,
        value: np.ndarray,
        time: np.ndarray,
        variance: Optional[np.ndarray] = None,
        nmax: Optional[int] = None,
        maxdist: Optional[float] = None,
        maxtlag: Optional[float] = None,
    ):
        """
        Load observations for variable ivar.

        Parameters
        ----------
        ivar     : variable index, 1-based
        coord    : (nobs, 3) spatial coordinates
        value    : (nobs,)   observed values
        time     : (nobs,)   observation times
        variance : (nobs,)   measurement error variance (default: zeros)
        nmax     : max spatial neighbours
        maxdist  : max spatial search distance
        maxtlag  : max temporal lag (same units as time)
        """
        import sys as _sys
        coord_f = _coord3_to_fortran(coord)
        nobs    = coord_f.shape[1]
        assert len(value) == nobs, "value length != nobs"
        assert len(time)  == nobs, "time length != nobs"

        value_f = _farray(np.asarray(value).ravel())
        time_f  = _farray(np.asarray(time).ravel())
        var_f   = _farray(variance) if variance is not None else _farray(np.zeros(nobs))

        c_nmax    = _c_int(nmax    if nmax    is not None else np.iinfo(np.int32).max)
        c_maxdist = _c_double(maxdist if maxdist is not None else _sys.float_info.max)
        c_maxtlag = _c_double(maxtlag if maxtlag is not None else _sys.float_info.max)

        _st_set_obs(_h(self._handle),
            _c_int(ivar), _c_int(nobs),
            _dptr(coord_f), _dptr(value_f), _dptr(time_f), _dptr(var_f),
            c_nmax, c_maxdist, c_maxtlag)

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
        spec = (f"{vtype} {nugget} {sill} {a_major} {a_minor1} {a_minor2}"
                f" {azimuth} {dip} {plunge}")
        _st_set_vgm(_h(self._handle), _c_int(ivar), _c_int(jvar), spec.encode("utf-8"))

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
        spec = f"{vtype} {nugget} {sill} {at_k}"
        _st_set_vgm_temporal(_h(self._handle), _c_int(ivar), _c_int(jvar),
                              spec.encode("utf-8"))

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
        time: np.ndarray,
        rangescale: Optional[np.ndarray] = None,
        localnugget: Optional[np.ndarray] = None,
    ):
        """
        Set point estimation targets.

        coord : (ngrid, 3) spatial coordinates
        time  : (ngrid,)   prediction times
        """
        coord_f = _coord3_to_fortran(coord)
        ngrid   = coord_f.shape[1]
        assert len(time) == ngrid, "time length != ngrid"

        time_f  = _farray(np.asarray(time).ravel())
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
        If randpath/sample are None, they are generated internally.
        """
        import random as _random
        nb = self._get_nblocks_raw()
        if nb == 0:
            raise RuntimeError("set_grid must be called before set_sim")

        if randpath is None:
            rp = np.arange(1, nb + 1, dtype=np.int32)
            _random.shuffle(rp)
        else:
            rp = np.asarray(randpath, dtype=np.int32)

        if sample is None:
            rng = np.random.default_rng()
            samp = rng.standard_normal((self.nsim, nb)).astype(np.float64, order='F')
        else:
            samp = _farray(sample)

        rp_f   = np.asfortranarray(rp, dtype=np.int32)
        samp_f = _farray(samp)
        _st_set_sim(_h(self._handle), _c_int(nb),
                    _iptr(rp_f), _c_int(self.nsim), _dptr(samp_f))

    # ------------------------------------------------------------------
    def set_search(
        self,
        ivar: int,
        anis1: float = 1.0,
        anis2: float = 1.0,
        azimuth: float = 0.0,
        dip: float = 0.0,
        plunge: float = 0.0,
    ):
        """
        Build spatial KD-tree for variable ivar.
        Call after set_obs (and after set_sim for ivar=1 in SGSIM).
        """
        _st_set_search(_h(self._handle), _c_int(ivar),
                       _c_double(anis1), _c_double(anis2),
                       _c_double(azimuth), _c_double(dip), _c_double(plunge))

    # ------------------------------------------------------------------
    def solve(self):
        """Run the ST kriging or SGSIM loop."""
        _st_solve(_h(self._handle))

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
            otherwise shape (nsim, ngrid)
        variance : ndarray, shape (ngrid,)
        """
        nb = ctypes.c_int(0)
        ns = ctypes.c_int(0)
        _st_get_nblocks(_h(self._handle), ctypes.byref(nb))
        _st_get_nsim   (_h(self._handle), ctypes.byref(ns))
        nb, ns = nb.value, ns.value

        estimate = _fempty((ns, nb), dtype=np.float64)
        variance = _fempty(nb, dtype=np.float64)
        _st_get_estimate(_h(self._handle), _c_int(ns), _c_int(nb), _dptr(estimate))
        _st_get_variance(_h(self._handle), _c_int(nb),              _dptr(variance))

        if squeeze and ns == 1:
            est = estimate[0]
        else:
            est = estimate

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
        return f"SpaceTimeKriging(nvar={self.nvar}, ndrift={self.ndrift}, nsim={self.nsim})"


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def spacetime_kriging(
    obs_coord: np.ndarray,
    obs_value: np.ndarray,
    obs_time: np.ndarray,
    grid_coord: np.ndarray,
    grid_time: np.ndarray,
    spatial_spec: "dict | list[dict]",
    temporal_spec: "dict | list[dict]",
    joint_sills: "list[float]",
    model: str = "sum_metric",
    transform: str = "linear",
    at: float = 1.0,
    alpha: float = 1.0,
    nmax: int = 20,
    maxdist: Optional[float] = None,
    maxtlag: Optional[float] = None,
    search_anis1: float = 1.0,
    search_anis2: float = 1.0,
    search_azimuth: float = 0.0,
    k_ps: float = 0.0,
) -> "tuple[np.ndarray, np.ndarray]":
    """
    One-shot ordinary space-time kriging (single variable).

    Parameters
    ----------
    obs_coord    : (nobs, 3)   observation spatial coordinates
    obs_value    : (nobs,)     observed values
    obs_time     : (nobs,)     observation times
    grid_coord   : (ngrid, 3)  prediction spatial coordinates
    grid_time    : (ngrid,)    prediction times
    spatial_spec : dict or list[dict]  spatial variogram structure(s)
    temporal_spec: dict or list[dict]  temporal variogram structure(s)
    joint_sills  : list[float]         joint sills (sum-metric only)
    model        : 'sum_metric' or 'product_sum'
    transform    : 'linear', 'bounded', or 'power'
    at           : joint temporal scale
    alpha        : power exponent (transform='power')
    nmax         : max spatial neighbours
    maxdist      : max spatial search distance
    maxtlag      : max temporal lag

    Returns
    -------
    estimate : (ngrid,)
    variance : (ngrid,)
    """
    k = SpaceTimeKriging(nvar=1)
    k.set_st_model(model=model, transform=transform, at=at, alpha=alpha, k_ps=k_ps)
    k.set_obs(ivar=1, coord=obs_coord, value=obs_value, time=obs_time,
              nmax=nmax, maxdist=maxdist, maxtlag=maxtlag)
    for spec in ([spatial_spec] if isinstance(spatial_spec, dict) else list(spatial_spec)):
        k.set_vgm(1, 1, **spec)
    for spec in ([temporal_spec] if isinstance(temporal_spec, dict) else list(temporal_spec)):
        k.set_vgm_temporal(1, 1, **spec)
    if model == "sum_metric":
        k.set_vgm_joint_sills(1, 1, *joint_sills)
    k.set_grid(coord=grid_coord, time=grid_time)
    k.set_search(ivar=1, anis1=search_anis1, anis2=search_anis2, azimuth=search_azimuth)
    k.solve()
    return k.get_results()


def spacetime_cokriging(
    obs_coords: "list[np.ndarray]",
    obs_values: "list[np.ndarray]",
    obs_times:  "list[np.ndarray]",
    grid_coord: np.ndarray,
    grid_time:  np.ndarray,
    spatial_specs: dict,
    temporal_specs: dict,
    joint_sills: dict,
    model: str = "sum_metric",
    transform: str = "linear",
    at: float = 1.0,
    alpha: float = 1.0,
    nmax: int = 20,
    maxdist: Optional[float] = None,
    maxtlag: Optional[float] = None,
) -> "tuple[np.ndarray, np.ndarray]":
    """
    One-shot ordinary space-time co-kriging.

    Parameters
    ----------
    obs_coords   : list of (nobs_i, 3) arrays, one per variable
    obs_values   : list of (nobs_i,)   arrays
    obs_times    : list of (nobs_i,)   arrays
    grid_coord   : (ngrid, 3)
    grid_time    : (ngrid,)
    spatial_specs : dict (ivar,jvar) -> dict or list[dict]
    temporal_specs: dict (ivar,jvar) -> dict or list[dict]
    joint_sills  : dict (ivar,jvar) -> list[float]

    Returns
    -------
    estimate : (ngrid,)
    variance : (ngrid,)
    """
    nvar = len(obs_coords)
    k = SpaceTimeKriging(nvar=nvar)
    k.set_st_model(model=model, transform=transform, at=at, alpha=alpha)

    for i, (coord, value, time) in enumerate(zip(obs_coords, obs_values, obs_times), start=1):
        k.set_obs(ivar=i, coord=coord, value=value, time=time,
                  nmax=nmax, maxdist=maxdist, maxtlag=maxtlag)

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
        k.set_search(ivar=i)

    k.solve()
    return k.get_results()
