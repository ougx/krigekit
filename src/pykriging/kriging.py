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
    from pykriging import Kriging
    import numpy as np

    # Spatial ordinary kriging
    k = Kriging(ndim=2, nvar=1)
    k.set_obs(ivar=1, coord=coord, value=value, nmax=20)
    k.set_grid(coord=grid_coord)
    k.set_vgm(ivar=1, jvar=1, vtype="sph", nugget=0, sill=1.0, a_major=1000, a_minor1=500, a_minor2=500)
    k.set_search(ivar=1)
    k.solve()
    est, var = k.get_results()
    del k    # release memory

    # Space-time kriging — same entry point, st=True
    k = Kriging(st=True, nvar=1)
    k.set_st_model(model='sum_metric', transform='bounded', at=5.0)
    ...

Kriging is a factory function that returns either a _SpatialKriging or
SpaceTimeKriging instance depending on the ``st`` keyword.
"""

import ctypes
import os
import numpy as np
from typing import Optional
import random

# ---------------------------------------------------------------------------
# Intel OpenMP runtime guards (Windows + ifx/ifort builds)
#
# KMP_DUPLICATE_LIB_OK=TRUE  — suppresses the crash that occurs when two
#   OpenMP runtimes (e.g. Intel libiomp5md.dll and GNU libgomp.dll from
#   a pip-installed numpy/scipy) are both loaded into the same process.
#   Without this, the first !$OMP PARALLEL region triggers an access
#   violation that cascades across all OpenMP threads.
#
# KMP_STACKSIZE — each Intel OpenMP worker thread gets its own stack.
#   Default is 4 MB on Windows.  The largest automatic array in the hot
#   path is L(nmax, nmax) in kriging_solve: L(1000,1000) ≈ 4 MB, which
#   would overflow the 4 MB default.  Setting 64 MB is safe for any
#   realistic nmax.  Users can override this via the environment variable
#   before importing pykriging.
# ---------------------------------------------------------------------------
if os.name == "nt":  # Windows only
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("KMP_STACKSIZE", "64m")

# ---------------------------------------------------------------------------
# Load the shared library (platform-aware)
# ---------------------------------------------------------------------------
def _load_lib():
    base = os.path.dirname(__file__)
    import sys as _sys
    if _sys.platform == "win32":
        names = ["kriging.dll"]
        # Prepend the package directory to PATH so that any Intel runtime DLLs
        # placed alongside kriging.dll (e.g. libcaf_ifx.dll, libiomp5md.dll)
        # are found by Windows when they are dynamically loaded at runtime.
        # This is needed because LoadLibraryW (used by libiomp5md.dll to load
        # libcaf_ifx.dll at runtime) searches PATH, not the DLL's own directory.
        os.environ['PATH'] = base + os.pathsep + os.environ.get('PATH', '')
    elif _sys.platform == "darwin":
        names = ["libkriging.dylib"]
    else:
        names = ["libkriging.so"]
    for name in names:
        path = os.path.join(base, name)
        if os.path.exists(path):
            return ctypes.CDLL(path, winmode=0)
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
_c_char_p = ctypes.c_char_p
_ptr_void = ctypes.c_void_p
_ptr_char = ctypes.POINTER(ctypes.c_char)
_ptr_int  = ctypes.POINTER(ctypes.c_int)
_ptr_dbl  = ctypes.POINTER(ctypes.c_double)

def _cfun(name, argtypes, restype=None):
    fn = getattr(_lib, name)
    fn.argtypes = argtypes
    fn.restype  = restype
    return fn

def _status_cfun(name, argtypes):
    """Wrap a kriging C API function that returns ierr.

    The Fortran side records the detailed message in kriging_err; this wrapper
    turns any non-zero ierr into a Python RuntimeError so ctypes callers do not
    continue after a failed Fortran setup or solve call.
    """
    fn = _cfun(name, argtypes, _c_int)

    def checked(*args):
        _check(fn(*args), name)

    checked.__name__ = name
    checked._cfunc = fn
    return checked

def _optional_status_cfun(name, argtypes):
    """Like _status_cfun but tolerates a missing DLL symbol.

    If the symbol is absent (e.g. the library was compiled before this feature
    was added), returns a stub that raises a clear RuntimeError when called,
    instead of crashing the whole module at import time.
    """
    try:
        return _status_cfun(name, argtypes)
    except AttributeError:
        def _stub(*_args, **_kwargs):
            raise RuntimeError(
                f"'{name}' was not found in the compiled library.  "
                "Recompile the Fortran library to enable the weight-store API."
            )
        _stub.__name__ = name
        return _stub

_ptr_int64 = ctypes.POINTER(ctypes.c_int64)
_krige_create      = _status_cfun("krige_create",      [_ptr_int64])
_krige_destroy     = _status_cfun("krige_destroy",     [_ptr_int64])
_krige_initialize  = _status_cfun("krige_initialize",  [
    ctypes.c_int64,                              # handle
    _c_int, _c_int, _c_int, _c_int, _c_int,     # ndim, nvar, ndrift, unbias, nsim
    # flags: anisotropic_search, weight_correction, use_old_weight, store_weight,
    #        cross_validation, write_mat, neglect_error, varying_vgm, std_ck, verbose  (10 booleans as int)
    _c_int, _c_int, _c_int, _c_int, _c_int, _c_int, _c_int, _c_int, _c_int, _c_int,
    _c_int,                                      # pf_cache
    _c_char_p,                                   # weight_file
    _ptr_dbl,                                    # bounds[2]
    _c_int,                                      # seed
])
_krige_set_obs     = _status_cfun("krige_set_obs", [
    ctypes.c_int64,                              # handle
    _c_int, _c_int, _c_int,                      # ivar, nobs, ndim_c
    _ptr_dbl, _ptr_dbl, _ptr_dbl,                # coord, value, variance
    _c_int, _c_double, _c_double,                # nmax, maxdist, sk_mean
])
_krige_set_obs_drift = _status_cfun("krige_set_obs_drift", [
    ctypes.c_int64,                              # handle
    _c_int, _c_int, _c_int,                      # ivar, ndrift_c, nobs
    _ptr_dbl,                                    # drift[ndrift_c, nobs]
])
_krige_update_obs_value = _status_cfun("krige_update_obs_value", [
    ctypes.c_int64,                              # handle
    _c_int, _c_int,                              # ivar, nobs
    _ptr_dbl,                                    # value[nobs]
])
_krige_reset_vgm   = _status_cfun("krige_reset_vgm", [
    ctypes.c_int64,                              # handle
    _c_int, _c_int,                              # ivar, jvar
])
_krige_set_vgm     = _status_cfun("krige_set_vgm",  [
    ctypes.c_int64,                              # handle
    _c_int, _c_int,                              # ivar, jvar
    _c_char_p,                                   # vtype (null-terminated)
    _c_double, _c_double,                        # nugget, sill
    _c_double, _c_double, _c_double,             # a_major, a_minor1, a_minor2
    _c_double, _c_double, _c_double,             # azimuth, dip, plunge
])
_krige_set_vgm_block = _status_cfun("krige_set_vgm_block", [
    ctypes.c_int64,                              # handle
    _c_int, _c_int, _c_int,                      # ivar, jvar, ib
    _c_char_p,                                   # vtype (null-terminated)
    _c_double, _c_double,                        # nugget, sill
    _c_double, _c_double, _c_double,             # a_major, a_minor1, a_minor2
    _c_double, _c_double, _c_double,             # azimuth, dip, plunge
])
_krige_set_grid    = _status_cfun("krige_set_grid", [
    ctypes.c_int64,                              # handle
    _c_int, _c_int, _ptr_dbl,                   # ngrid, ndim_c, coord
    _ptr_dbl, _ptr_dbl,                          # rangescale, localnugget
])
_krige_set_grid_block = _status_cfun("krige_set_grid_block", [
    ctypes.c_int64,                              # handle
    _c_int,                                      # block_type
    _c_int, _c_int, _ptr_dbl,                   # ngrid, ndim_c, coord
    _c_int, _ptr_int,                            # nblock, nblockpnt
    _ptr_dbl,                                    # pointweight[sum(nblockpnt)] — no npw
    _ptr_dbl,                                    # blocksize
    _ptr_dbl, _ptr_dbl,                          # rangescale, localnugget
])
_krige_set_grid_cv = _status_cfun("krige_set_grid_cv", [ctypes.c_int64])
_krige_set_grid_drift = _status_cfun("krige_set_grid_drift", [
    ctypes.c_int64,                              # handle
    _c_int,                                      # ivar (1-based target variable, or < 0 = broadcast)
    _c_int, _c_int,                              # ndrift_c, nblocks
    _ptr_dbl,                                    # drift[ndrift_c, nblocks]
])
_krige_set_sim     = _status_cfun("krige_set_sim", [
    ctypes.c_int64,                              # handle
    _c_int, _ptr_int,                            # nblocks, randpath[nblocks]
    _c_int, _c_int, _ptr_dbl,                    # nsim_c, nvar_c, sample[nsim_c, nvar_c, nblocks]
])
_krige_set_search  = _status_cfun("krige_set_search", [
    ctypes.c_int64, _c_int,                      # handle, ivar
    _c_double, _c_double, _c_double, _c_double, _c_double,  # anis1, anis2, az, dip, plunge
])
_krige_set_grad    = _status_cfun("krige_set_grad", [
    ctypes.c_int64,                              # handle
    _c_int, _c_int, _c_int,                      # ivar, ngrad, ndim_c
    _ptr_dbl, _ptr_dbl,                          # coord1[ndim,ngrad], coord2[ndim,ngrad]
    _ptr_dbl,                                    # grad_val[ngrad]
    _ptr_dbl,                                    # variance[ngrad]
    _c_int, _ptr_dbl,                            # ndrift_c, drift_ext[ndrift,ngrad]
])
_krige_solve       = _status_cfun("krige_solve",       [ctypes.c_int64, ctypes.c_int, ctypes.c_int])
# _krige_print       = _cfun("krige_print",       [ctypes.c_int64])
_krige_get_nblocks     = _status_cfun("krige_get_nblocks",     [ctypes.c_int64, _ptr_int])
_krige_get_nsim        = _status_cfun("krige_get_nsim",        [ctypes.c_int64, _ptr_int])
_krige_get_block_coord = _status_cfun("krige_get_block_coord", [ctypes.c_int64, _c_int, _c_int, _ptr_dbl])
_krige_get_estimate    = _status_cfun("krige_get_estimate",    [ctypes.c_int64, _c_int, _c_int, _ptr_dbl])
_krige_get_estimate_all= _status_cfun("krige_get_estimate_all",[ctypes.c_int64, _c_int, _c_int, _c_int, _ptr_dbl])
_krige_get_variance    = _status_cfun("krige_get_variance",    [ctypes.c_int64, _c_int, _ptr_dbl])
_krige_get_variance_all= _status_cfun("krige_get_variance_all",[ctypes.c_int64, _c_int, _c_int, _ptr_dbl])
_krige_get_factor_info    = _status_cfun("krige_get_factor_info",
    [ctypes.c_int64, _ptr_int, _ptr_int, _ptr_int])
_krige_get_factor_matrices= _status_cfun("krige_get_factor_matrices",
    [ctypes.c_int64, _c_int, _c_int, _ptr_dbl, _ptr_dbl, _ptr_dbl])
_krige_get_factor_system= _status_cfun("krige_get_factor_system",
    [ctypes.c_int64, _c_int, _c_int, _c_int, _ptr_dbl, _ptr_dbl])

# _krige_alloc_weight_store = _optional_status_cfun("krige_alloc_weight_store", [ctypes.c_int64])
_krige_free_weight_store  = _optional_status_cfun("krige_free_weight_store",  [ctypes.c_int64])
_krige_get_weight_dims    = _optional_status_cfun("krige_get_weight_dims",
    [ctypes.c_int64, _ptr_int, _ptr_int, _ptr_int])   # nm, ng, nb
_krige_get_weight_nnear   = _optional_status_cfun("krige_get_weight_nnear",
    [ctypes.c_int64, _c_int, _c_int, _ptr_int])
_krige_get_weight_inear   = _optional_status_cfun("krige_get_weight_inear",
    [ctypes.c_int64, _c_int, _c_int, _c_int, _ptr_int])
_krige_get_weight_data    = _optional_status_cfun("krige_get_weight_data",
    [ctypes.c_int64, _c_int, _c_int, _c_int, _c_int, _ptr_dbl])
_krige_get_weight_var     = _optional_status_cfun("krige_get_weight_var",
    [ctypes.c_int64, _c_int, _c_int, _ptr_dbl])
_krige_set_weights        = _optional_status_cfun("krige_set_weights",
    [ctypes.c_int64, _c_int, _c_int, _c_int, _c_int,
     _ptr_int, _ptr_int, _ptr_dbl, _ptr_int, _ptr_dbl])
     # handle, nmax, ngroups, nvar, nblock, nnear, inear, weight, order, var

_krige_get_last_error = _cfun("krige_get_last_error", [_ptr_char, _c_int], _c_int)

_krige_to_str      = _cfun("krige_to_str"   , [ctypes.c_int64], _ptr_void)

_krige_get_max_threads = _cfun("krige_get_max_threads", [_ptr_int])
_krige_get_num_threads = _cfun("krige_get_num_threads", [_ptr_int])

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# OpenMP diagnostics
# ---------------------------------------------------------------------------

def omp_info() -> dict:
    """
    Return a dict with OpenMP thread counts as seen by the Fortran runtime.

    Keys
    ----
    max_threads : int
        Value of omp_get_max_threads() — the number of threads that will be
        used in the next parallel region (respects OMP_NUM_THREADS and any
        omp_set_num_threads() calls).  Returns 1 when OpenMP is not compiled in.
    num_threads : int
        Value of omp_get_num_threads() measured inside an actual parallel
        region — the number of threads that are *actually* running.
        Returns 1 when OpenMP is not compiled in.
    openmp : bool
        True when the library was compiled with OpenMP support.

    Example
    -------
    >>> import os; os.environ["OMP_NUM_THREADS"] = "4"
    >>> from _kriging import omp_info
    >>> omp_info()
    {'max_threads': 4, 'num_threads': 4, 'openmp': True}
    """
    max_t = ctypes.c_int(0)
    num_t = ctypes.c_int(0)
    _krige_get_max_threads(ctypes.byref(max_t))
    _krige_get_num_threads(ctypes.byref(num_t))
    return {
        "max_threads": max_t.value,
        "num_threads": num_t.value,
        "openmp": max_t.value > 1 or num_t.value > 1,
    }

def get_omp_info():
    omp = omp_info()
    print(f"OpenMP max_threads={omp['max_threads']}  actual threads={omp['num_threads']}  OpenMP {'On' if omp['openmp'] else 'Off'}")

def _farray(a, dtype=np.float64):
    """Return a Fortran-contiguous array of the given dtype."""
    return np.asfortranarray(a, dtype=dtype)

def _fempty(shape, dtype=np.float64):
    """Allocate a Fortran-contiguous output array directly."""
    return np.empty(shape, dtype=dtype, order="F")

def _coord_to_fortran(coord: np.ndarray) -> np.ndarray:
    """
    Convert coordinates from Python convention (nobs, ndim)
    to Fortran convention (ndim, nobs), column-major.

    The user always passes (nobs, ndim) — rows are points, columns are
    spatial dimensions, matching NumPy/pandas/scikit-learn convention.
    This function transposes to (ndim, nobs) and ensures Fortran memory order
    before the array is handed to the Fortran library.

    Fortran receives the transposed array and validates the resulting
    (ndim, nobs) shape, returning ierr instead of relying on Python asserts.
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

def _last_error() -> str:
    """Return the last Fortran error message recorded by kriging.dll."""
    buf = ctypes.create_string_buffer(4096)
    _krige_get_last_error(buf, _c_int(len(buf)))
    return buf.value.decode("utf-8", errors="replace").strip()

def _check(ierr: int, call_name: str) -> None:
    """Raise a Python exception when a Fortran C API call returns an error."""
    if int(ierr) != 0:
        msg = _last_error() or f"{call_name} failed with ierr={int(ierr)}"
        raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Main Python class
# ---------------------------------------------------------------------------

class _SpatialKriging:
    """
    Python interface to the Fortran t_kriging spatial kriging/simulation engine.

    Do not instantiate directly — use the :func:`Kriging` factory instead::

        k = Kriging(ndim=2, nvar=1)           # spatial (default)
        k = Kriging(st=True, nvar=1)          # space-time

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
    >>> k.set_grid(coord=grid_coord)
    >>> k.set_vgm(ivar=1, jvar=1, vtype="sph", nugget=0, sill=1.0, a_major=1000, a_minor1=500, a_minor2=50)
    >>> k.set_search(ivar=1)
    >>> k.solve()
    >>> estimate, variance = k.get_results()
    >>> del k    # release memory

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
        neglect_error: bool = True,
        varying_vgm: bool = False,
        std_ck: bool = False,
        verbose: bool = False,
        pf_cache: bool = False,
        weight_file: str = "",
        bounds: Optional[tuple] = None,
        seed: Optional[int] = None,
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
            Write computed weights to ``weight_file`` while also estimating blocks.
        cross_validation : bool
            Leave-one-out cross-validation mode.
        write_mat : bool
            Write matrix for debugging.
        neglect_error : bool
            Ignore solver errors and set failed block to NaN instead of aborting.
        varying_vgm : bool
            Use a different variogram per estimation block (spatially varying
            anisotropy).  When True, call :meth:`set_vgm_block` for each block
            after :meth:`set_grid`.  Defaults to False (single global model).
        std_ck : bool
            Co-kriging unbiasedness formulation (only relevant when ``nvar > 1``
            and ``unbias=1``).

            * ``False`` (default) — Isaaks & Srivastava: single combined
              constraint (Σw₁ + Σw₂ = 1) plus a local-mean correction applied
              post-solve.  Matches the GSLIB/legacy behaviour.
            * ``True`` — standard cokriging: separate per-variable constraints
              (Σwᵢ = 1 for own variable, Σwⱼ = 0 for others), equivalent to
              the gstat/ISATIS formulation.  Use this to match gstat output.
        verbose : bool
            Print progress messages.
        pf_cache : bool
            Enable the persistent between-solve factorization cache.  When
            ``True``, the Cholesky factor of K is stored after the first
            :meth:`solve` and reused on subsequent calls when the neighbour
            set and variogram are unchanged (speeds up repeated solves on the
            same observation grid).  Defaults to ``False``; enable only when
            you plan to call :meth:`solve` multiple times and need the speedup.
        weight_file : str
            Path to the weight file (required when use_old_weight or store_weight).
        bounds : tuple(float, float) or None
            (lower, upper) clipping bounds for the estimate.
            None means no clipping (uses Fortran defaults: [-huge, +huge]).
        seed : int, optional
            Random seed.
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
        _huge = sys.float_info.max * 1e3
        c_bounds = _farray(bounds if bounds is not None else [-_huge, _huge])
        seed = seed or random.randint(0, 2**32-1)
        # set random seed in python
        random.seed(seed)
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
            _c_int(int(neglect_error)),
            _c_int(int(varying_vgm)),
            _c_int(int(std_ck)),
            _c_int(int(verbose)),
            _c_int(int(pf_cache)),
            weight_file.encode("utf-8") if weight_file else b"",
            _dptr(c_bounds),
            _c_int(seed),
        )

        # store for convenience
        self.ndim   = ndim
        self.nvar   = nvar
        self.ndrift = ndrift
        self.nsim   = nsim
        self.verbose = verbose

        self.unbias = unbias
        self.anisotropic_search = anisotropic_search
        self.weight_correction = weight_correction
        self.use_old_weight = use_old_weight
        self.store_weight = store_weight
        self.cross_validation = cross_validation
        self.write_mat = write_mat
        self.varying_vgm = varying_vgm
        self.std_ck = std_ck
        self.pf_cache = pf_cache
        self.weight_file = weight_file
        self.bounds = c_bounds
        self.seed = seed

        #-- Sanity checks: mutually exclusive flag combinations
        # use_old_weight + weight_file="" is valid: signals the in-memory wstore path.
        # Call set_weights() before solve() to populate the store in that case.
        if (self.store_weight and self.weight_file == b""):
            raise ValueError('store_weight requires weight_file to be specified')
        if (self.store_weight and self.use_old_weight):
            raise ValueError('store_weight and use_old_weight are mutually exclusive')
        if (self.cross_validation and self.nsim > 0):
            raise ValueError('nsim>0 and cross_validation are mutually exclusive')

        # -- size tracking
        self._nblock = 0
        self._nobs = np.zeros(self.nvar, dtype=np.uint32)
        self._nmax = np.zeros(self.nvar, dtype=np.uint32)
        self._set_search = [False,] * self.nvar
        self._set_sim    = False
        self._nobsdrift = np.zeros(self.nvar, dtype=np.uint32)
        self._nvgm_struct = np.zeros([self.nvar, self.nvar], dtype=np.uint32) # does not fully track nvgm_struct with varying vgm mode
        # Python-side default for ngroups (= ngroups_base, no grad).
        # get_weights() calls krige_get_weight_dims to read the actual value
        # from the weight store after solve(), so this only affects set_weights().
        self._ngroups = self.nvar if nsim == 0 else self.nvar * 2
    # ------------------------------------------------------------------
    def set_obs(
        self,
        ivar: int,
        coord: np.ndarray,
        value: np.ndarray,
        variance: Optional[np.ndarray] = None,
        nmax: Optional[int] = None,
        maxdist: Optional[float] = None,
        sk_mean: float = 0.0,
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
        sk_mean : float
            Global mean for simple kriging (unbias=0). Default 0.0.
        """
        import sys
        coord_f  = _coord_to_fortran(coord)        # (nobs, ndim) -> (ndim, nobs) F-order
        value_f  = _farray(np.asarray(value, dtype=np.float64).ravel())
        nobs     = coord_f.shape[1]
        ndim_c   = coord_f.shape[0]

        # The C API receives value(nobs) and variance(nobs) raw pointers; check
        # lengths here so ctypes never lets Fortran read past a NumPy buffer.
        if value_f.size != nobs:
            raise ValueError(
                f"value length ({value_f.size}) must match nobs ({nobs})")
        if variance is not None:
            var_f = _farray(np.asarray(variance, dtype=np.float64).ravel())
            if var_f.size != nobs:
                raise ValueError(
                    f"variance length ({var_f.size}) must match nobs ({nobs})")
        else:
            var_f = _farray(np.zeros(nobs))

        # nmax/maxdist: pass huge values when not specified (Fortran treats as "unlimited")
        c_nmax    = _c_int(nmax    if nmax    is not None else np.iinfo(np.int32).max)
        c_maxdist = _c_double(maxdist if maxdist is not None else sys.float_info.max)
        c_sk_mean = _c_double(sk_mean)
        _krige_set_obs(_h(self._handle),
            _c_int(ivar), _c_int(nobs), _c_int(ndim_c),
            _dptr(coord_f), _dptr(value_f), _dptr(var_f),
            c_nmax, c_maxdist, c_sk_mean
        )
        self._nobs[ivar-1] = nobs
        self._nmax[ivar-1] = min(nobs, nmax) if nmax is not None else nobs + (self._nblock if self.nsim>0 else 0)

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
        _krige_set_obs_drift(_h(self._handle),
            _c_int(ivar), _c_int(ndrift_c), _c_int(nobs),
            _dptr(drift_f),
        )

    # ------------------------------------------------------------------
    def update_obs_value(self, ivar: int, value: np.ndarray):
        """
        Replace observation values for variable ``ivar`` in-place.

        Coordinates and the kd-tree are unchanged.  The primary use case
        is weight reuse: after solving once (with ``store_weight=True`` or
        ``use_old_weight=True``), call this method with new observed values
        at the same locations and call :meth:`solve` again to get updated
        estimates without recomputing search neighbourhoods or the LHS
        factorization.

        Parameters
        ----------
        ivar : int
            Variable index, 1-based.
        value : ndarray, shape (nobs,)
            New observed values.  Length must match the ``nobs`` passed to
            the previous :meth:`set_obs` call for this variable.
        """
        nobs    = self._nobs[ivar - 1]
        value_f = _farray(np.asarray(value, dtype=np.float64).ravel())
        if value_f.size != nobs:
            raise ValueError(
                f"value length ({value_f.size}) must match nobs ({nobs}) "
                f"set by the previous set_obs call for ivar={ivar}"
            )
        _krige_update_obs_value(_h(self._handle),
            _c_int(ivar), _c_int(nobs),
            _dptr(value_f),
        )

    # ------------------------------------------------------------------
    def set_vgm(
        self, ivar: int, jvar: int, vtype: str,
        nugget: float = 0.0, sill: float = 1.0,
        a_major: float = 1.0,
        a_minor1: Optional[float] = None,
        a_minor2: Optional[float] = None,
        azimuth: float = 0.0, dip: float = 0.0, plunge: float = 0.0,
        append: bool = True,
    ):
        """
        Add one nested variogram structure for the (ivar, jvar) pair.
        Call multiple times with ``append=True`` to build a nested
        (multi-structure) model.  Pass ``append=False`` to clear any
        previously set structures for the pair before adding this one
        (useful when reusing a Kriging object with a different variogram).

        Parameters
        ----------
        ivar, jvar : int
            Variable indices (1-based). Use ivar=jvar for auto-variograms,
            ivar≠jvar for cross-variograms. The LMC constraint
            b12² ≤ b11 × b22 must be satisfied for each nested structure.
        vtype : str
            Variogram type: one of ``sph``, ``exp``, ``gau``, ``pow``,
            ``lin``, ``hol``, ``bsq``, ``cir``, ``nug``.
        nugget : float
            Nugget contribution of this structure (default 0).
        sill : float
            Partial sill of this structure (default 1).
        a_major : float
            Range along the major axis (default 1).
        a_minor1 : float, optional
            Range along the first minor axis. Defaults to ``a_major``
            (isotropic in the horizontal plane).
        a_minor2 : float, optional
            Range along the second minor axis. Defaults to ``a_minor1``.
        azimuth, dip, plunge : float
            Rotation angles in degrees (default 0).

        Example
        -------
        >>> k.set_vgm(1, 1, vtype="sph", nugget=0.0, sill=1.0, a_major=500.0)
        >>> k.set_vgm(1, 1, vtype="nug", nugget=0.1, sill=0.0, a_major=1.0)
        >>> k.set_vgm(1, 1, vtype="sph", nugget=0.0, sill=0.9, a_major=500.0)
        """
        if a_minor1 is None:
            a_minor1 = a_major
        if a_minor2 is None:
            a_minor2 = a_minor1
        if not append and self._nvgm_struct[ivar-1, jvar-1] > 0:
            _krige_reset_vgm(_h(self._handle), _c_int(ivar), _c_int(jvar))
            self._nvgm_struct[ivar-1, jvar-1] = 0
            if ivar != jvar:
                self._nvgm_struct[jvar-1, ivar-1] = 0
        if (jvar<ivar):
            it = ivar
            jvar=ivar
            ivar=it
        _krige_set_vgm(_h(self._handle),
            _c_int(ivar), _c_int(jvar),
            vtype.lower()[:3].encode("utf-8"),
            nugget, sill, a_major, a_minor1, a_minor2,
            azimuth, dip, plunge,
        )
        self._nvgm_struct[ivar-1, jvar-1] += 1
        if (ivar!=jvar):
            self._nvgm_struct[jvar-1, ivar-1] += 1

    # ------------------------------------------------------------------
    def set_vgm_block(
        self, ib: int, ivar: int, jvar: int, vtype: str,
        nugget: float = 0.0, sill: float = 1.0,
        a_major: float = 1.0,
        a_minor1: Optional[float] = None,
        a_minor2: Optional[float] = None,
        azimuth: float = 0.0, dip: float = 0.0, plunge: float = 0.0,
    ):
        """
        Add one nested variogram structure for a *specific block* ``ib``.

        Requires ``varying_vgm=True`` in the constructor and :meth:`set_grid`
        to have been called first (because the number of blocks must be known
        before the per-block variogram array can be allocated in Fortran).

        Call multiple times for the same ``ib`` to build a nested model.

        Parameters
        ----------
        ib : int
            Block index (1-based).
        ivar, jvar : int
            Variable indices (1-based).
        vtype : str
            Variogram type: ``sph``, ``exp``, ``gau``, ``pow``, ``lin``,
            ``hol``, ``bsq``, ``cir``, or ``nug``.
        nugget : float
            Nugget contribution (default 0).
        sill : float
            Partial sill (default 1).
        a_major : float
            Range along the major axis (default 1).
        a_minor1 : float, optional
            First minor-axis range (defaults to ``a_major``).
        a_minor2 : float, optional
            Second minor-axis range (defaults to ``a_minor1``).
        azimuth, dip, plunge : float
            Rotation angles in degrees (default 0).
        """
        assert self.varying_vgm, "set_vgm_block requires varying_vgm=True"
        if a_minor1 is None:
            a_minor1 = a_major
        if a_minor2 is None:
            a_minor2 = a_minor1
        _krige_set_vgm_block(_h(self._handle),
            _c_int(ivar), _c_int(jvar), _c_int(ib),
            vtype.encode("utf-8"),
            nugget, sill, a_major, a_minor1, a_minor2,
            azimuth, dip, plunge,
        )

    # ------------------------------------------------------------------
    def set_grid(
        self,
        coord: Optional[np.ndarray] = None,
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
        if coord is None:
            self.set_grid_cv()
            return

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
        self._nblock = ngrid

    # ------------------------------------------------------------------
    def set_grid_block(
        self,
        coord: np.ndarray,
        block_type: int,
        nblockpnt: np.ndarray,
        pointweight: Optional[np.ndarray] = None,
        blocksize: Optional[np.ndarray] = None,
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
        blocksize : ndarray, shape (nblock,ndim), optional
            Block size in each dimension when block_type == -4.
        rangescale : ndarray, shape (nblock,), optional
            Per-block variogram range scaling. Default: 1.0.
        localnugget : ndarray, shape (nblock,), optional
            Per-block additional nugget. Default: 0.0.
        """
        coord_f = _coord_to_fortran(coord)
        ngrid   = coord_f.shape[1]
        ndim_c  = coord_f.shape[0]

        if block_type == -4:
            nblock = ngrid
            assert blocksize is not None, (
                "blocksize must be specified for Gaussian quadrature blocks.")
            if blocksize.ndim == 1:
                # broadcasts the 1-D blocksize vector into a (ndim, nblock) matrix
                blocksize = np.tile(blocksize, (nblock, 1))
            else:
                assert len(blocksize) == nblock and len(blocksize[0]) == self.ndim, (
                    f"blocksize should be (nblock={nblock}, ndim={self.ndim})")
            blocksize_f = _coord_to_fortran(blocksize)
            nbp_f   = np.ascontiguousarray(np.ones(nblock, dtype=np.int32))
            pw_f = _farray(np.ones(nblock))
        else:
            nbp_f   = np.ascontiguousarray(nblockpnt, dtype=np.int32)
            nblock  = len(nblockpnt)
            npoint  = int(np.sum(nbp_f))
            # Fortran derives the pointweight length from sum(nblockpnt) and
            # reads coord(:,1:sum(nblockpnt)); reject inconsistent block maps
            # before a raw pointer can be indexed out of bounds.
            if np.any(nbp_f <= 0):
                raise ValueError("nblockpnt must contain positive counts")
            if npoint != ngrid:
                raise ValueError(
                    f"sum(nblockpnt) ({npoint}) must match coord rows ({ngrid})")
            blocksize_f = _coord_to_fortran(np.zeros((nblock, self.ndim)))
            if pointweight is not None:
                if len(pointweight) != npoint:
                    raise ValueError(
                        f"pointweight length ({len(pointweight)}) must match "
                        f"sum(nblockpnt) ({npoint})")
                pw_f = _farray(pointweight)
            else:
                # uniform weights: 1/nblockpnt for each sub-node
                pw_f = _farray(np.repeat(1.0 / nbp_f, nbp_f))
        rs_f = (_farray(rangescale)
                if rangescale  is not None else _farray(np.ones(nblock)))
        ln_f = (_farray(localnugget)
                if localnugget is not None else _farray(np.zeros(nblock)))
        assert len(rs_f) == nblock, (
            f"rangescale should be (nblock={nblock})")
        assert len(ln_f) == nblock, (
            f"localnugget should be (nblock={nblock})")

        _krige_set_grid_block(_h(self._handle),
            _c_int(block_type),
            _c_int(ngrid), _c_int(ndim_c), _dptr(coord_f),
            _c_int(nblock), _iptr(nbp_f),
            _dptr(pw_f),                   # Fortran derives length via sum(nblockpnt)
            _dptr(blocksize_f),            # blocksize_f is (nblock, ndim)
            _dptr(rs_f), _dptr(ln_f),
        )
        self._nblock = nblock

    # ------------------------------------------------------------------
    def set_grid_cv(self):
        """
        Set up the grid for **cross-validation** mode.

        No coordinate argument is needed — Fortran derives the grid from the
        observation coordinates automatically.  Call instead of :meth:`set_grid`
        when ``cross_validation=True`` was passed to the constructor.
        """
        _krige_set_grid_cv(_h(self._handle))
        self._nblock = self._nobs[0]

    # ------------------------------------------------------------------
    def set_grid_drift(self, drift: np.ndarray, ivar: Optional[int] = None):
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
        ivar : int, optional
            Target-variable index (1-based) whose RHS receives this drift.
            ``None`` (default) broadcasts the same drift to **all** target
            variables — the usual case when external drift is independent of
            which variable is being estimated.

        Note
        ----
        ``ivar`` here refers to the **target** variable (which variable's
        estimate uses this drift in its RHS), not the source variable.
        This is the opposite end from :meth:`set_obs_drift`, whose ``ivar``
        identifies the **source** variable (whose observations form the
        F-matrix column).
        """
        drift_f  = _drift_to_fortran(drift)   # (nblocks, ndrift) -> (ndrift, nblocks)
        ndrift_c = drift_f.shape[0]
        nblocks  = drift_f.shape[1]
        _krige_set_grid_drift(_h(self._handle),
            _c_int(ivar if ivar is not None else -1),
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
        sample : ndarray, shape (nblocks, nvar, nsim), optional
            Pre-drawn standard-normal samples used to add simulated variability.
            Drawn from N(0,1) if omitted.
        """
        # Python generates defaults so Fortran always receives concrete arrays.
        # We need the block count; retrieve it from the Fortran object.
        assert self.nsim > 0, ("nsim must be > 0 when setting SGSIM parameters.")
        nb = _c_int(0)
        _krige_get_nblocks(_h(self._handle), ctypes.byref(nb))
        nblocks = nb.value
        rng = np.random.default_rng(self.seed)
        if randpath is not None:
            rp_f = np.ascontiguousarray(
                np.asarray(randpath, dtype=np.int32).ravel(), dtype=np.int32)
            # randpath is consumed as a 1-based Fortran permutation of blocks.
            # Validate both length and membership before exposing the buffer.
            if rp_f.size != nblocks:
                raise ValueError(
                    f"randpath length ({rp_f.size}) must match nblocks ({nblocks})")
            expected_path = np.arange(1, nblocks + 1, dtype=np.int32)
            if not np.array_equal(np.sort(rp_f), expected_path):
                raise ValueError("randpath must be a 1-based permutation of 1..nblocks")
        else:
            # random permutation of 1..nblocks (1-based for Fortran)
            rp_f = np.ascontiguousarray(
                rng.permutation(nblocks) + 1, dtype=np.int32)

        if sample is not None:
            sample_a = np.asarray(sample, dtype=np.float64)
            if sample_a.ndim == 1:
                # (nblocks) shorthand when nvar==1
                sample_a = sample_a[:, np.newaxis, np.newaxis]  # → (nblocks, 1, 1, )
            elif sample_a.ndim == 2:
                # (nsim, nblocks) shorthand when nvar==1
                sample_a = sample_a[:, np.newaxis, :]  # → (nblocks, 1 ,nsim)
            elif sample_a.ndim != 3:
                raise ValueError(
                    f"sample must be 2-D (nblocks, nsim) or 3-D (nblocks, nvar, nsim), "
                    f"got shape {sample_a.shape}")
            s_f    = _drift_to_fortran(sample_a)
            nsim_c = s_f.shape[0]
            nvar_c = s_f.shape[1]
            n_s    = s_f.shape[2]
            if nsim_c != self.nsim or nvar_c != self.nvar or n_s != nblocks:
                raise ValueError(
                    f"sample shape ({nsim_c}, {nvar_c}, {n_s}) must be "
                    f"({self.nsim}, {self.nvar}, {nblocks})")
        else:
            nsim_c = self.nsim
            nvar_c = self.nvar
            n_s    = nblocks
            s_f    = _farray(rng.standard_normal((nsim_c, nvar_c, n_s)))

        _krige_set_sim(_h(self._handle),
            _c_int(nblocks), _iptr(rp_f),         # nblocks covers both randpath and sample
            _c_int(nsim_c), _c_int(nvar_c), _dptr(s_f),
        )
        self._set_sim = True

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
        self._set_search[ivar-1] = True

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
        Set gradient observation pairs (Delhomme 1979: "Kriging in hydrology").

        Each pair ``(coord1[i], coord2[i])`` approximates the directional
        gradient at a boundary as a finite difference.  The constraint
        ``Z(xs1) - Z(xs2) = grad_value[i]`` is enforced as a hard equality.
        For a no-flow (zero normal gradient) boundary use ``grad_value = 0``.

        Call :meth:`set_grad` **after** :meth:`set_search` and **before**
        :meth:`solve`.

        Parameters
        ----------
        coord1 : ndarray, shape **(ngrad, ndim)**
            Positive-side virtual node coordinates.
        coord2 : ndarray, shape **(ngrad, ndim)**
            Negative-side virtual node coordinates.
        grad_value : ndarray, shape **(ngrad,)**
            Known gradient values.  Use 0 for no-flow boundaries.
        ivar : int, default 1
            Variable index (1-based) the gradient pairs constrain.
            For cokriging, specifies which variable's gradient is observed.
        variance : ndarray, shape **(ngrad,)**, optional
            Gradient observation variance (default 0 = exact constraint).
            A non-zero value relaxes the constraint, analogous to obs nugget.
        drift_ext : ndarray, shape **(ngrad, ndrift)**, optional
            External drift differences ``f_ext(xs1) - f_ext(xs2)`` for each
            pair.  Required when ``ndrift > 0``; omit for ordinary kriging.
        """
        # _coord_to_fortran: (ngrad, ndim) → (ndim, ngrad), Fortran-contiguous
        c1_f   = _coord_to_fortran(coord1)
        c2_f   = _coord_to_fortran(coord2)
        ngrad  = c1_f.shape[1]
        ndim_c = c1_f.shape[0]
        gval   = _farray(np.asarray(grad_value, dtype=np.float64).ravel())

        if variance is not None:
            gvar = _farray(np.asarray(variance, dtype=np.float64).ravel())
        else:
            gvar = _farray(np.zeros(max(ngrad, 1), dtype=np.float64))

        if drift_ext is not None:
            # _drift_to_fortran: (ngrad, ndrift) → (ndrift, ngrad), Fortran-contiguous
            de_f     = _drift_to_fortran(np.atleast_2d(drift_ext))
            ndrift_c = de_f.shape[0]
        else:
            de_f     = np.zeros((1, max(ngrad, 1)), dtype=np.float64, order='F')
            ndrift_c = 0

        _krige_set_grad(_h(self._handle),
            _c_int(ivar), _c_int(ngrad), _c_int(ndim_c),
            _dptr(c1_f), _dptr(c2_f),
            _dptr(gval),
            _dptr(gvar),
            _c_int(ndrift_c), _dptr(de_f),
        )

    # ------------------------------------------------------------------
    def solve(self, nthread: int = 0, ncache: Optional[int] = None):
        """
        Run the kriging or SGSIM loop over all blocks.
        Calls prepare(), then the parallel block loop internally.

        Parameters
        ----------
        nthread : int, optional
            Number of OpenMP threads to use for this call.
            ``0`` (default) leaves the OMP runtime setting unchanged.
            ``1`` forces single-threaded execution (useful for reproducible
            results or when calling :meth:`solve` from inside another
            parallel region).
        ncache : int, optional
            Number of per-thread multi-slot hcache entries to use for this
            solve call.  ``None`` keeps the compiled/object default, ``0``
            disables the hcache, and ``1`` gives a one-slot hcache for
            cache-overhead comparisons.  The single-entry ``ctx%cache`` and
            optional persistent factor cache are unaffected.
        """
        if self.verbose:
            get_omp_info()
        ncache_c = -1 if ncache is None else int(ncache)
        _krige_solve(_h(self._handle), ctypes.c_int(nthread), ctypes.c_int(ncache_c))

    # ------------------------------------------------------------------
    def get_results(self, copy: bool = False, squeeze: bool = True):
        """
        Retrieve the kriging estimates and variances after :meth:`solve`.

        Fortran fills ``estimate(nsim, nblocks)`` directly into a
        Fortran-contiguous Python-owned buffer.

        Parameters
        ----------
        copy : bool, default False
            If True, return C-contiguous copies for downstream NumPy/Pandas use.
            If False, return views / Fortran-order arrays when possible.
        squeeze : bool, default True
            If True, return a 1-D estimate when ``nsim == 1``.

        Returns
        -------
        estimate : ndarray
            **(ngrid, nvar, nsim)**.
            Shape **(ngrid,)** when ``(nsim == 1 or ==1) and squeeze``; otherwise shape
            **(ngrid, nvar, nsim)**.
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
        nv = self.nvar
        ns = max(1, n_sim.value)

        estimate = _fempty((nb, nv, ns), dtype=np.float64)   # (nblock, nvar, nsim)
        variance = _fempty((nb, nv, nv), dtype=np.float64)   # (nblock, nvar, nvar)

        _krige_get_estimate_all(_h(self._handle), _c_int(nb), _c_int(nv), _c_int(ns), _dptr(estimate))
        _krige_get_variance_all(_h(self._handle), _c_int(nb), _c_int(nv), _dptr(variance))

        if squeeze:
            if self.nvar == 1:
                estimate = estimate[:, 0]     # (nb, 1, ns) → (nb, ns)
                variance = variance[:, 0, 0]  # (nb, 1, 1) → (nb,)
            if ns <= 1:
                estimate = estimate[..., 0]   # (..., 1)   → (...) — remove last dim
        est = estimate


        if copy:
            est = np.array(est, order="C", copy=True)
            variance = np.array(variance, order="C", copy=True)

        return est, variance

    def get_result_array(self) -> np.ndarray:
        """
        Return all results as a NumPy structured (record) array — one row per block.

        The array contains block centroid coordinates alongside every estimate,
        simulation realization, and variance produced by the last :meth:`solve`.

        Fields
        ------
        Coordinates (always present):
          ``x``, ``y`` [, ``z``]  — block centroid coordinates.

        Estimates / simulations:
          Kriging (``nsim == 0``), ``nvar == 1``:  ``estimate``
          Kriging (``nsim == 0``), ``nvar > 1``:   ``est_v1``, ``est_v2``, …
          SGSIM  (``nsim > 0``),  ``nvar == 1``:  ``sim_1``, ``sim_2``, …, ``sim_{nsim}``
          SGSIM  (``nsim > 0``),  ``nvar > 1``:   ``v1_s1``, ``v1_s2``, …, ``v{nvar}_s{nsim}``

        Variances (always present, diagonal of conditional covariance):
          ``nvar == 1``:  ``variance``
          ``nvar > 1``:   ``var_v1``, ``var_v2``, …

        Returns
        -------
        np.ndarray with named fields (structured array / record array)
            Shape ``(nblocks,)``.  Access a column with ``arr['estimate']``,
            convert to a plain 2-D array with ``np.column_stack([arr[f] for f in arr.dtype.names])``.

        Example
        -------
        >>> k.solve()
        >>> ra = k.get_result_array()
        >>> ra.dtype.names
        ('x', 'y', 'estimate', 'variance')
        >>> ra['estimate']          # 1-D array, shape (nblocks,)
        >>> import pandas as pd
        >>> df = pd.DataFrame(ra)   # direct conversion to DataFrame
        """
        nb_c = ctypes.c_int(0)
        ns_c = ctypes.c_int(0)
        _krige_get_nblocks(_h(self._handle), ctypes.byref(nb_c))
        _krige_get_nsim   (_h(self._handle), ctypes.byref(ns_c))

        nblocks = nb_c.value
        nsim    = ns_c.value          # 0 for plain kriging
        ns      = max(1, nsim)        # always at least 1 for Fortran call
        nv      = self.nvar
        nd      = self.ndim

        # --- coordinates: Fortran fills (ndim, nblocks), Python reads (nblocks, ndim) ---
        coord_f = _fempty((nd, nblocks))
        _krige_get_block_coord(_h(self._handle), _c_int(nd), _c_int(nblocks), _dptr(coord_f))
        coord = np.ascontiguousarray(coord_f.T)        # (nblocks, ndim)

        # --- estimates / simulations: (nblocks, nvar, ns) Fortran-order ---
        est_f = _fempty((nblocks, nv, ns))
        _krige_get_estimate_all(_h(self._handle), _c_int(nblocks), _c_int(nv), _c_int(ns), _dptr(est_f))
        est = np.ascontiguousarray(est_f)              # (nblocks, nvar, ns)

        # --- variance: (nblocks, nvar, nvar), take diagonal → (nblocks, nvar) ---
        var_f = _fempty((nblocks, nv, nv))
        _krige_get_variance_all(_h(self._handle), _c_int(nblocks), _c_int(nv), _dptr(var_f))
        var_arr = np.ascontiguousarray(var_f)          # (nblocks, nvar, nvar)
        var_diag = np.stack([var_arr[:, i, i] for i in range(nv)], axis=1)  # (nblocks, nvar)

        # --- build dtype ---
        coord_names = ['x', 'y', 'z'][:nd]
        dtype_fields = [(name, np.float64) for name in coord_names]

        is_sgsim = nsim > 0
        if is_sgsim:
            if nv == 1:
                sim_fields = [f'sim_{i+1}' for i in range(nsim)]
            else:
                sim_fields = [f'v{iv+1}_s{isim+1}' for iv in range(nv) for isim in range(nsim)]
        else:
            if nv == 1:
                sim_fields = ['estimate']
            else:
                sim_fields = [f'est_v{iv+1}' for iv in range(nv)]
        dtype_fields += [(name, np.float64) for name in sim_fields]

        if nv == 1:
            var_fields = ['variance']
        else:
            var_fields = [f'var_v{iv+1}' for iv in range(nv)]
        dtype_fields += [(name, np.float64) for name in var_fields]

        # --- populate record array ---
        out = np.empty(nblocks, dtype=dtype_fields)

        for i, name in enumerate(coord_names):
            out[name] = coord[:, i]

        if is_sgsim:
            if nv == 1:
                for isim in range(nsim):
                    out[f'sim_{isim+1}'] = est[:, 0, isim]
            else:
                for iv in range(nv):
                    for isim in range(nsim):
                        out[f'v{iv+1}_s{isim+1}'] = est[:, iv, isim]
        else:
            if nv == 1:
                out['estimate'] = est[:, 0, 0]
            else:
                for iv in range(nv):
                    out[f'est_v{iv+1}'] = est[:, iv, 0]

        if nv == 1:
            out['variance'] = var_diag[:, 0]
        else:
            for iv in range(nv):
                out[f'var_v{iv+1}'] = var_diag[:, iv]

        return out

    def get_factor(self) -> dict:
        """Return the persistent LHS factorization cached after the last :meth:`solve`.

        The Cholesky factorization of the kriging covariance matrix K and the
        related Schur-complement matrices are computed once per solve() call (or
        reused across blocks with the same neighbour set).  Starting from the
        second solve() call on unchanged observations and variogram, the cached
        factors allow the Fortran engine to skip ``kriging_setup`` entirely.

        This method exposes those matrices and the assembled linear system so
        that users can inspect or verify the factorization inputs.

        Returns
        -------
        dict with keys:

        ``valid`` : bool
            ``True`` if a persistent factor exists (i.e. at least one solve()
            has been completed and observations/variogram have not changed since).
        ``npp`` : int
            Number of neighbours in the LHS matrix (size of K).
        ``p`` : int
            Number of drift + unbiasedness columns (size of the Schur complement).
        ``L`` : ndarray, shape (npp, npp)
            Lower-triangular Cholesky factor of K  (stored column-major by Fortran,
            returned as a C-contiguous array).
        ``kinv_drift`` : ndarray, shape (npp, max(1, p))
            K⁻¹ F  (K inverse applied to the drift matrix F).
        ``schur`` : ndarray, shape (max(1, p), max(1, p))
            Cholesky factor of the Schur complement  F' K⁻¹ F.

        ``matA`` : ndarray, shape (npp + p, npp + p)
            Assembled linear-system LHS before factorization.
        ``rhsB`` : ndarray, shape (nvar, npp + p)
            Assembled linear-system RHS before solving.

        Example
        -------
        >>> k = Kriging(ndim=2, nvar=1, ndrift=1, unbias=0)
        >>> # ... set_obs, set_vgm, set_grid, set_obs_drift, set_grid_drift ...
        >>> k.solve()
        >>> f = k.get_factor()
        >>> if f['valid']:
        ...     L = f['L']            # Cholesky factor of covariance matrix
        ...     kinv = f['kinv_drift']  # K^{-1} F
        ...     schur = f['schur']    # Cholesky of Schur complement

        Notes
        -----
        The factor is invalidated (``valid = False``) whenever :meth:`set_obs`
        or :meth:`set_vgm` is called.  Call :meth:`solve` again to repopulate.
        """
        c_npp   = ctypes.c_int(0)
        c_p     = ctypes.c_int(0)
        c_valid = ctypes.c_int(0)
        _krige_get_factor_info(_h(self._handle),
                               ctypes.byref(c_npp),
                               ctypes.byref(c_p),
                               ctypes.byref(c_valid))
        valid = bool(c_valid.value)
        npp   = c_npp.value
        p     = c_p.value

        if not valid:
            return dict(valid=False, npp=0, p=0,
                        L=None, kinv_drift=None, schur=None,
                        matA=None, rhsB=None)

        pg = max(1, p)
        matsize = npp + p
        L_buf     = _fempty((npp, npp), dtype=np.float64)
        kinv_buf  = _fempty((npp, pg),  dtype=np.float64)
        schur_buf = _fempty((pg,  pg),  dtype=np.float64)
        matA_buf  = _fempty((matsize, matsize), dtype=np.float64)
        rhsB_buf  = _fempty((self.nvar, matsize), dtype=np.float64)

        _krige_get_factor_matrices(
            _h(self._handle),
            _c_int(npp), _c_int(p),
            _dptr(L_buf), _dptr(kinv_buf), _dptr(schur_buf),
        )
        _krige_get_factor_system(
            _h(self._handle),
            _c_int(npp), _c_int(p), _c_int(self.nvar),
            _dptr(matA_buf), _dptr(rhsB_buf),
        )

        return dict(
            valid=True, npp=npp, p=p,
            L         = np.asarray(L_buf,     order='C'),
            kinv_drift= np.asarray(kinv_buf,  order='C'),
            schur     = np.asarray(schur_buf, order='C'),
            matA      = np.asarray(matA_buf,  order='C'),
            rhsB      = np.asarray(rhsB_buf,  order='C'),
        )

    def get_estimate_all(self, copy: bool = False):
        """Return multivariable estimates / simulations for all variables.

        Populated when ``nvar > 1``.  For co-kriging without simulation,
        the leading dimension is 1.

        Parameters
        ----------
        copy : bool, default False
            If True, return a C-contiguous copy. If False, return the
            Fortran-contiguous output buffer filled by the Fortran core.

        Returns
        -------
        np.ndarray, shape (nblock, nvar, max(nsim, 1))
            Values of all variables.  ``out[ib, kvar, isim]`` is the value at
            block ``ib+1`` for variable ``kvar+1`` in realization ``isim+1``.
        """
        if self.nvar <= 1:
            raise RuntimeError("get_estimate_all is only available for multivariable kriging/simulation (nvar > 1)")

        n_blocks = ctypes.c_int(0)
        n_sim    = ctypes.c_int(0)
        _krige_get_nblocks(_h(self._handle), ctypes.byref(n_blocks))
        _krige_get_nsim   (_h(self._handle), ctypes.byref(n_sim))

        nb = n_blocks.value
        ns = max(1, n_sim.value)
        nv = self.nvar

        out = _fempty((nb, nv, ns), dtype=np.float64)
        _krige_get_estimate_all(_h(self._handle), _c_int(nb), _c_int(nv), _c_int(ns), _dptr(out))

        if copy:
            return np.array(out, order="C", copy=True)
        return out

    def get_variance_all(self, copy: bool = False):
        """Return the conditional covariance matrix for all variables.

        Returns
        -------
        np.ndarray, shape (nblock, nvar, nvar)
            Conditional covariance matrix at each block.  The diagonal contains
            each variable's kriging variance, and ``out[:, 0, 0]`` matches the
            variance returned by :meth:`get_results`.
        """
        n_blocks = ctypes.c_int(0)
        _krige_get_nblocks(_h(self._handle), ctypes.byref(n_blocks))

        nb = n_blocks.value
        nv = self.nvar
        out = _fempty((nb, nv, nv), dtype=np.float64)
        _krige_get_variance_all(_h(self._handle), _c_int(nb), _c_int(nv), _dptr(out))

        if copy:
            return np.array(out, order="C", copy=True)
        return out

    # ------------------------------------------------------------------
    def __del__(self):
        if self._handle != 0:
            _tmp = ctypes.c_int64(self._handle)
            try:
                _krige_destroy(ctypes.byref(_tmp))
            except Exception:
                pass
            self._handle = 0

    # ------------------------------------------------------------------
    def free_weight_store(self):
        """Release the in-memory weight store, freeing its memory."""
        _krige_free_weight_store(_h(self._handle))

    # ------------------------------------------------------------------
    def set_weights(self, weights: dict) -> None:
        """Load kriging weights into the in-memory store so solve() reuses them.

        Activated when ``use_old_weight=True`` and no ``weight_file`` is given.
        :meth:`solve` then applies the supplied neighbour indices and kriging
        weights directly — skipping the kriging-system solve — and restores
        the stored variance.  This is the in-memory equivalent of the
        ``use_old_weight=True`` + factor-file workflow.

        Typical workflow
        ----------------
        >>> # First run: solve and capture weights + variance
        >>> k1 = Kriging(ndim=2, nvar=1, store_weight=True)
        >>> k1.set_obs(...); k1.set_grid(...); k1.set_vgm(...); k1.set_search(...)
        >>> k1.solve()
        >>> w = k1.get_weights()           # {'nnear', 'inear', 'weight', 'variance'}
        >>>
        >>> # Second run: same grid/vgm, new obs values, reuse weights
        >>> k2 = Kriging(ndim=2, nvar=1, use_old_weight=True)   # no weight_file
        >>> k2.set_obs(...new_values...)
        >>> k2.set_grid(...); k2.set_vgm(...); k2.set_search(...)
        >>> k2.set_weights(w)              # populate the in-memory store
        >>> k2.solve()                     # fast: skips kriging system solve
        >>> est, var = k2.get_results()

        Parameters
        ----------
        weights : dict
            Dict as returned by :meth:`get_weights`, with keys:

            ``nnear`` : ndarray (nblock, ngroups), int32
                Number of active neighbours per block and group.
            ``inear`` : ndarray (nblock, ngroups, nmax), int32
                1-based neighbour indices.
            ``weight`` : ndarray (nblock, nvar, ngroups, nmax), float64
                Kriging weights.  For ``nvar==1`` the array may be 3-D
                ``(nblock, ngroups, nmax)`` (the shape returned by
                :meth:`get_weights`).
            ``order`` : ndarray (nblock,), optional
                Random visiting order for SGSIM.  Defaults to None.
            ``variance`` : ndarray (nblock,) or (nblock, nvar, nvar), optional
                Per-block conditional variance.  Included automatically
                when the dict was produced by :meth:`get_weights`.
                Defaults to zeros if absent.

        Notes
        -----
        Call after :meth:`set_obs`, :meth:`set_grid`, :meth:`set_vgm`, and
        :meth:`set_search`.  The Fortran-side ``use_old_weight`` flag is set
        automatically; you may also pass ``use_old_weight=True`` to the
        constructor to declare intent explicitly.
        """
        if self.store_weight:
            raise ValueError(
                "set_weights() is incompatible with store_weight=True. "
                "Construct Kriging without store_weight=True when reusing weights."
            )

        nnear  = np.ascontiguousarray(weights["nnear"],  dtype=np.int32)
        inear  = np.ascontiguousarray(weights["inear"],  dtype=np.int32)
        weight = np.asarray(weights["weight"],           dtype=np.float64)

        if nnear.ndim != 2:
            raise ValueError("nnear must be 2-D (nblock, ngroups)")
        if inear.ndim != 3:
            raise ValueError("inear must be 3-D (nblock, ngroups, nmax)")

        nb, ng = nnear.shape
        nm     = inear.shape[2]
        nv     = self.nvar

        # order: 1-based block path indices (nblock,).
        # Use the stored value when present; default to sequential (plain kriging).
        order_raw = weights.get("order", None)
        if order_raw is None or self.nsim == 0:
            order_f = np.ascontiguousarray(np.arange(1, nb + 1, dtype=np.int32))
        else:
            order_f = np.ascontiguousarray(order_raw, dtype=np.int32)

        # Transpose nnear/inear to Fortran column-major order.
        nnear_f = np.asfortranarray(nnear.T)          # (ng, nb)
        inear_f = np.asfortranarray(inear.T)           # (nm, ng, nb)

        # weight may be:
        #   3-D (nb, ng, nm)       — nvar=1 shape from get_weights()
        #   4-D (nb, nv, ng, nm)   — general shape (nv first after nb)
        # Fortran wstore layout: weight(nm, ng, nv, nb)
        if weight.ndim == 3:
            # (nb, ng, nm) → (nm, ng, nb) → insert nvar axis → (nm, ng, nv=1, nb)
            w3 = np.asfortranarray(weight.T)           # (nm, ng, nb)
            weight_f = np.asfortranarray(w3[:, :, np.newaxis, :])  # (nm, ng, 1, nb)
        elif weight.ndim == 4:
            # (nb, nv, ng, nm) → (nm, ng, nv, nb)
            weight_f = np.asfortranarray(np.transpose(weight, (3, 2, 1, 0)))
        else:
            raise ValueError(
                "weight must be 3-D (nblock, ngroups, nmax) or "
                "4-D (nblock, nvar, ngroups, nmax)"
            )

        # Variance: (nb,) for nvar=1 or (nb, nv, nv) for general.
        # Fortran layout: var(nv, nv, nb).
        if "variance" in weights and weights["variance"] is not None:
            var = np.asarray(weights["variance"], dtype=np.float64)
            if var.ndim == 1:
                var = var[:, np.newaxis, np.newaxis]   # (nb,) → (nb, 1, 1)
            # (nb, nv, nv) → (nv, nv, nb)
            var_f = np.asfortranarray(np.transpose(var, (1, 2, 0)))
        else:
            var_f = np.asfortranarray(np.zeros((nv, nv, nb), dtype=np.float64))

        _krige_set_weights(
            _h(self._handle),
            _c_int(nm), _c_int(ng), _c_int(nv), _c_int(nb),
            nnear_f.ctypes.data_as(_ptr_int),
            inear_f.ctypes.data_as(_ptr_int),
            weight_f.ctypes.data_as(_ptr_dbl),
            order_f.ctypes.data_as(_ptr_int),
            var_f.ctypes.data_as(_ptr_dbl),
        )
        self.use_old_weight = True

    # ------------------------------------------------------------------
    def get_weights(self) -> dict:
        """Return the stored kriging weights and neighbour indices.

        :meth:`alloc_weight_store` must have been called before
        :meth:`solve`.

        Returns
        -------
        dict with keys:

        ``nnear`` : ndarray, shape ``(nblock, ngroups)``, dtype int32
            Number of active neighbours for each block and group.
            ngroups = ngroups_base when set_grad has not been called, or
            ngroups_base + nvar when gradient data is present.
            ngroups_base = nvar (kriging) or 2*nvar (SGSIM).
            Group layout:
              indices 0..nvar-1          — obs groups (variable 1..nvar)
              indices nvar..2*nvar-1     — sim groups (SGSIM only)
              indices ngroups_base..ngroups-1 — grad groups (present only when set_grad called)

        ``inear`` : ndarray, shape ``(nblock, ngroups, nmax)``, dtype int32
            1-based neighbour indices.  Entries beyond ``nnear[ib, ig]``
            are zero.

        ``weight`` : ndarray, shape ``(nblock, nvar, ngroups, nmax)``, dtype float64
            Kriging weights.  Entries beyond ``nnear[ib, ig]`` are zero.
            Shape is ``(nblock, ngroups, nmax)`` when ``nvar == 1``.

        ``variance`` : ndarray, shape ``(nblock,)`` for ``nvar==1``, else ``(nblock, nvar, nvar)``
            Per-block conditional kriging variance stored alongside the
            weights.  Present only when the compiled library supports the
            variance store (i.e. built with the current source).  Pass this
            dict directly to :meth:`set_weights` to get a full round-trip.
        """

        # Query actual weight-store dimensions from Fortran.
        # ngroups = ngroups_base (no grad) or ngroups_base+nvar (grad present).
        _nm = np.zeros(1, dtype=np.int32)
        _ng = np.zeros(1, dtype=np.int32)
        _nb = np.zeros(1, dtype=np.int32)
        _krige_get_weight_dims(
            _h(self._handle),
            _nm.ctypes.data_as(_ptr_int),
            _ng.ctypes.data_as(_ptr_int),
            _nb.ctypes.data_as(_ptr_int),
        )
        nb, ng, nm, nv = int(_nb[0]), int(_ng[0]), int(_nm[0]), self.nvar

        # Allocate Fortran-order buffers matching the CAPI layout
        nnear_f  = np.zeros((ng, nb),     dtype=np.int32,   order='F')
        inear_f  = np.zeros((nm, ng, nb), dtype=np.int32,   order='F')
        weight_f = np.zeros((nm, ng, nv, nb), dtype=np.float64, order='F')

        _krige_get_weight_nnear(
            _h(self._handle), _c_int(ng), _c_int(nb),
            nnear_f.ctypes.data_as(_ptr_int),
        )
        _krige_get_weight_inear(
            _h(self._handle), _c_int(nm), _c_int(ng), _c_int(nb),
            inear_f.ctypes.data_as(_ptr_int),
        )
        _krige_get_weight_data(
            _h(self._handle), _c_int(nm), _c_int(ng), _c_int(nv), _c_int(nb),
            weight_f.ctypes.data_as(_ptr_dbl),
        )

        # Transpose to Python-friendly (block-major) layout:
        #   nnear_f  (ng, nb)         → .T → (nb, ng)
        #   inear_f  (nm, ng, nb)     → .T → (nb, ng, nm)
        #   weight_f (nm, ng, nv, nb) → .T → (nb, nv, ng, nm)
        #     squeeze nvar axis when nvar==1 → (nb, ng, nm)
        weight_out = np.ascontiguousarray(weight_f.T)  # (nb, nv, ng, nm)
        if nv == 1:
            weight_out = weight_out[:, 0, :, :]        # (nb, ng, nm)
        out = {
            "nnear":  np.ascontiguousarray(nnear_f.T),
            "inear":  np.ascontiguousarray(inear_f.T),
            "weight": weight_out,
        }

        # Retrieve per-block conditional variance from the weight store.
        var_f = np.zeros((nv, nv, nb), dtype=np.float64, order='F')
        try:
            _krige_get_weight_var(
                _h(self._handle), _c_int(nv), _c_int(nb),
                var_f.ctypes.data_as(_ptr_dbl),
            )
            # var_f (nv, nv, nb) → .T → (nb, nv, nv); squeeze for nvar=1
            var_out = np.ascontiguousarray(var_f.T)     # (nb, nv, nv)
            if nv == 1:
                var_out = var_out[:, 0, 0]              # (nb,)
            out["variance"] = var_out
        except RuntimeError:
            pass  # old library without krige_get_weight_var — omit variance key

        return out

    # ------------------------------------------------------------------
    def get_info(self):
        ptr = _krige_to_str(_h(self._handle))
        if not ptr:
            return ""
        return ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8", errors="ignore")

    # ------------------------------------------------------------------
    def __repr__(self):
        return f"<_SpatialKriging ndim={self.ndim} nvar={self.nvar} at {self._handle}>"

    # ------------------------------------------------------------------
    def __str__(self):
        return self.get_info()

# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

# Parameters that exist only in _SpatialKriging and have no ST equivalent.
_SPATIAL_ONLY_PARAMS = {"ndim", "varying_vgm", "std_ck", "pf_cache"}
_SPATIAL_ONLY_DEFAULTS = {"ndim": 2, "varying_vgm": False, "std_ck": False, "pf_cache": False}


def Kriging(st: bool = False, **kwargs):
    """
    Create a kriging object — spatial or space-time.

    This is the primary entry point for both backends.  Pass ``st=True`` to
    obtain a :class:`SpaceTimeKriging` instance; omit it (or ``st=False``) for
    ordinary spatial kriging (:class:`_SpatialKriging`).

    Parameters
    ----------
    st : bool, default False
        ``False`` — spatial kriging (:class:`_SpatialKriging`).
        ``True``  — space-time kriging (:class:`SpaceTimeKriging`).
    **kwargs
        Forwarded verbatim to the chosen constructor.

        Spatial-only parameters (``ndim``, ``varying_vgm``, ``std_ck``,
        ``pf_cache``) are silently removed when ``st=True``; a
        :class:`UserWarning` is emitted for any that were supplied with a
        non-default value so they are not silently ignored.

    Returns
    -------
    _SpatialKriging
        When ``st=False``.  Full set of methods: :meth:`~_SpatialKriging.set_obs`,
        :meth:`~_SpatialKriging.set_vgm`, :meth:`~_SpatialKriging.set_grid`,
        :meth:`~_SpatialKriging.set_search`, :meth:`~_SpatialKriging.solve`,
        :meth:`~_SpatialKriging.get_results`, and more.
    SpaceTimeKriging
        When ``st=True``.  Same core methods plus
        :meth:`~SpaceTimeKriging.set_st_model`,
        :meth:`~SpaceTimeKriging.set_vgm_temporal`, and
        :meth:`~SpaceTimeKriging.set_vgm_joint_sills`.
        :meth:`~SpaceTimeKriging.set_grid` requires an extra ``time=`` argument.
        :meth:`~SpaceTimeKriging.set_search` requires extra ``time_*`` arguments.

    Notes
    -----
    *Factory, not a class*: ``Kriging`` is a function that returns one of two
    concrete types.  Use ``isinstance(k, _SpatialKriging)`` or
    ``isinstance(k, SpaceTimeKriging)`` for type checks.

    *Output shape*: :meth:`get_results` shapes are consistent within each
    backend but differ when ``nsim > 1``:

    * spatial  — estimate ``(nblocks, nvar, nsim)`` squeezed to ``(nblocks,)``
      for ``nvar=1, nsim=1``.
    * ST       — estimate ``(nsim, nblocks)`` squeezed to ``(nblocks,)`` for
      ``nsim=1``.

    Examples
    --------
    Spatial ordinary kriging::

        k = Kriging(ndim=2, nvar=1)
        k.set_obs(ivar=1, coord=coord, value=value, nmax=20)
        k.set_grid(coord=grid_coord)
        k.set_vgm(ivar=1, jvar=1, vtype="sph", sill=1.0, a_major=500)
        k.set_search(ivar=1)
        k.solve()
        est, var = k.get_results()

    Space-time ordinary kriging::

        k = Kriging(st=True, nvar=1)
        k.set_st_model(model='sum_metric', transform='bounded', at=5.0)
        k.set_obs(ivar=1, coord=obs_coord4, value=obs_value, nmax=30)
        k.set_vgm(ivar=1, jvar=1, vtype="sph", sill=0.8, a_major=1000)
        k.set_vgm_temporal(ivar=1, jvar=1, vtype="exp", sill=0.6, at_k=10.0)
        k.set_vgm_joint_sills(1, 1, 0.4)
        k.set_grid(coord=grid_coord, time=grid_time)
        k.set_search(ivar=1)
        k.solve()
        est, var = k.get_results()
    """
    if st:
        # Warn about any spatial-only params passed with a non-default value
        non_default = {
            k: v for k, v in kwargs.items()
            if k in _SPATIAL_ONLY_PARAMS and v != _SPATIAL_ONLY_DEFAULTS.get(k)
        }
        if non_default:
            import warnings
            warnings.warn(
                "Kriging(st=True) ignores spatial-only parameters: "
                + ", ".join(f"{k}={v!r}" for k, v in non_default.items()),
                UserWarning,
                stacklevel=2,
            )
        for key in _SPATIAL_ONLY_PARAMS:
            kwargs.pop(key, None)
        try:
            from pykriging.kriging_st import SpaceTimeKriging
        except ImportError:
            from kriging_st import SpaceTimeKriging  # type: ignore[no-redef]
        return SpaceTimeKriging(**kwargs)
    else:
        return _SpatialKriging(**kwargs)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def ordinary_kriging(
    obs_coord: np.ndarray,
    obs_value: np.ndarray,
    grid_coord: np.ndarray,
    vgm_spec: "dict | list[dict]",
    nmax: Optional[int] = None,
    maxdist: Optional[float] = None,
    search_anis1: float = 1.0,
    search_anis2: float = 1.0,
    search_azimuth: float = 0.0,
    rangescale: Optional[float] = None,
    localnugget: Optional[float] = None,
    nthread=0,
    ncache: Optional[int] = None,
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
    vgm_spec : dict or list of dict
        One variogram structure dict, or a list of dicts for nested models.
        Each dict is passed as keyword arguments to :meth:`Kriging.set_vgm`
        (keys: ``vtype``, ``nugget``, ``sill``, ``a_major``, and optionally
        ``a_minor1``, ``a_minor2``, ``azimuth``, ``dip``, ``plunge``).
    nmax : int
        Maximum number of neighbours.
    maxdist : float, optional
        Maximum search distance.
    search_anis1, search_anis2 : float
        Anisotropy ratios for search ellipse (1.0 = isotropic).
    search_azimuth : float
        Azimuth of search ellipse major axis (degrees from North).
    nthread: int
        max OMP threads for this call (0 or absent = OMP default)
    ncache : int, optional
        Per-thread hcache slots for this solve. None uses the default.

    Returns
    -------
    estimate : ndarray, shape (ngrid,)
    variance : ndarray, shape (ngrid,)

    Example
    -------
    >>> est, var = ordinary_kriging(
    ...     obs_coord, obs_value, grid_coord,
    ...     vgm_spec=dict(vtype="sph", nugget=100, sill=900, a_major=1000, a_minor1=500),
    ...     nmax=20)
    """
    assert obs_coord.shape[0] == obs_value.shape[0], (
        f"obs_coord has {obs_coord.shape[0]} rows but obs_value has {obs_value.shape[0]} elements."
    )
    if nmax is None:
        nmax = len(obs_coord) + len(grid_coord)
    ndim = obs_coord.shape[1]   # (nobs, ndim) -> ndim is axis 1
    k = _SpatialKriging(ndim=ndim, nvar=1)
    k.set_obs(ivar=1, coord=obs_coord, value=obs_value,
              nmax=nmax, maxdist=maxdist)
    k.set_grid(coord=grid_coord, rangescale=rangescale, localnugget=localnugget)
    for spec in ([vgm_spec] if isinstance(vgm_spec, dict) else list(vgm_spec)):
        k.set_vgm(ivar=1, jvar=1, **spec)
    k.set_search(ivar=1, anis1=search_anis1, anis2=search_anis2,
                 azimuth=search_azimuth)
    k.solve(nthread=nthread, ncache=ncache)
    est, var = k.get_results()   # est is already (ngrid,) for kriging
    return est, var


def cokriging(
    obs_coords: list[np.ndarray],
    obs_values: list[np.ndarray],
    grid_coord: np.ndarray,
    vgm_spec: dict,
    nmax: Optional[int] = None,
    rangescale: Optional[float] = None,
    localnugget: Optional[float] = None,
    nthread: int = 0,
    ncache: Optional[int] = None,
    std_ck: bool = False,
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
    vgm_spec : dict
        Mapping ``(ivar, jvar)`` to a variogram dict or list of dicts.
        Each dict is passed as keyword arguments to :meth:`Kriging.set_vgm`.
        Both (i,j) and (j,i) can be provided; if only (i,j) is given,
        (j,i) will mirror it automatically (handled inside Fortran set_vgm).
    nmax : int
        Maximum neighbours per variable.
    nthread: int
        max OMP threads for this call (0 or absent = OMP default)
    ncache : int, optional
        Per-thread hcache slots for this solve. None uses the default.
    std_ck: bool
        Use standard Ordinary Kriging.

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
    ...     vgm_spec={
    ...         (1,1): dict(vtype="sph", nugget=100, sill=900, a_major=1000, a_minor1=500),
    ...         (2,2): dict(vtype="sph", nugget=50,  sill=450, a_major=1000, a_minor1=500),
    ...         (1,2): dict(vtype="sph", nugget=0,   sill=600, a_major=1000, a_minor1=500),
    ...     })
    """
    nvar = len(obs_coords)
    ndim = obs_coords[0].shape[1]   # (nobs, ndim) -> ndim is axis 1
    if nmax is None:
        nmax = max([len(obs_coord) for obs_coord in obs_coords]) + len(grid_coord)
    k = _SpatialKriging(ndim=ndim, nvar=nvar, std_ck=std_ck)

    for i, (coord, value) in enumerate(zip(obs_coords, obs_values), start=1):
        k.set_obs(ivar=i, coord=coord, value=value, nmax=nmax)

    k.set_grid(coord=grid_coord, rangescale=rangescale, localnugget=localnugget)

    for (iv, jv), spec in vgm_spec.items():
        for s in ([spec] if isinstance(spec, dict) else list(spec)):
            k.set_vgm(ivar=iv, jvar=jv, **s)

    for i in range(1, nvar + 1):
        k.set_search(ivar=i)

    k.solve(nthread=nthread, ncache=ncache)
    est, var = k.get_results()   # est is already (ngrid,) for kriging
    return est, var


def sequential_gaussian_simulation(
    obs_coord: np.ndarray,
    obs_value: np.ndarray,
    grid_coord: np.ndarray,
    vgm_spec: str,
    nsim: int,
    nmax: Optional[int] = None,
    randpath: Optional[np.ndarray] = None,
    sample: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    rangescale: Optional[float] = None,
    localnugget: Optional[float] = None,
    nthread: int = 0,
    ncache: Optional[int] = None,
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
    vgm_spec : dict or list of dict
        One or more nested variogram structure dicts, each passed as keyword
        arguments to :meth:`Kriging.set_vgm`.
    nsim : int
        Number of realisations.
    nmax : int
        Maximum neighbours (includes previously simulated nodes).
    seed : int, optional
        Random seed for reproducibility.
    nthread: int
        max OMP threads for this call (0 or absent = OMP default)
    ncache : int, optional
        Per-thread hcache slots for this solve. None uses the default.

    Returns
    -------
    simulations : ndarray, shape (nsim, ngrid)
        Each row is one realisation in the original (non-randomised) block order.
    """

    ndim = obs_coord.shape[1]   # (nobs, ndim) -> ndim is axis 1
    if nmax is None:
        nmax = len(obs_coord) + len(grid_coord)

    k = _SpatialKriging(ndim=ndim, nvar=1, nsim=nsim, seed=seed)
    k.set_obs(ivar=1, coord=obs_coord, value=obs_value, nmax=nmax)
    k.set_grid(coord=grid_coord, rangescale=rangescale, localnugget=localnugget)
    for spec in ([vgm_spec] if isinstance(vgm_spec, dict) else list(vgm_spec)):
        k.set_vgm(ivar=1, jvar=1, **spec)
    # set_sim with no args: Python generates random path and N(0,1) samples
    k.set_sim(randpath, sample)
    k.set_search(ivar=1)
    k.solve(nthread=nthread, ncache=ncache)

    sims, _ = k.get_results()   # shape (nsim, ngrid)
    return sims
