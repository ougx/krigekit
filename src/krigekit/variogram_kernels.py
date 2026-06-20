
"""Engine-compatible variogram kernels and component definitions."""


import numpy as np


# ---------------------------------------------------------------------------
# 1. Theoretical variogram / covariance models
# ---------------------------------------------------------------------------
# Each model is expressed for a *normalised* lag ``hr = h / range`` and a unit
# sill.  ``_covfunc`` mirrors the Fortran engine's ``corefunc_fn`` exactly for
# the public krigekit model codes; ``_vgmfunc`` is derived as ``1 - rho``.

def _corr_nug(hr):
    """Unit nugget correlation: one at zero lag and zero otherwise."""
    return np.where(np.asarray(hr) <= 0.0, 1.0, 0.0)


def _corr_sph(hr):
    """Unit spherical correlation with practical range at ``hr == 1``."""
    return np.where(hr < 1.0, 1.0 - 1.5 * hr + 0.5 * hr ** 3, 0.0)


def _corr_pow(hr, power=1.5):
    """Unit power correlation used by the engine, clamped beyond range."""
    return np.where(hr < 1.0, 1.0 - hr ** power, 0.0)


def _corr_bsq(hr):
    """Unit bi-square correlation, compactly supported at ``hr == 1``."""
    return np.where(hr < 1.0, (1.0 - hr ** 2) ** 2, 0.0)


def _corr_cir(hr):
    """Unit circular correlation, compactly supported at ``hr == 1``."""
    rc = np.clip(hr, 0.0, 1.0)
    inside = 1.0 - (2.0 * rc * np.sqrt(1.0 - rc ** 2) + 2.0 * np.arcsin(rc)) / np.pi
    return np.where(hr < 1.0, inside, 0.0)


_covfunc = dict(
    nug=_corr_nug,
    sph=_corr_sph,
    exp=lambda hr: np.exp(-3.0 * hr),
    gau=lambda hr: np.exp(-3.0625 * hr * hr),
    hol=lambda hr: np.cos(np.pi * hr),
    pow=_corr_pow,
    bsq=_corr_bsq,
    cir=_corr_cir,
    lin=lambda hr: np.where(hr < 1.0, 1.0 - hr, 0.0),
    cyc=lambda hr: np.exp(-2.0 * np.sin(np.pi * hr) ** 2),
    dco=lambda hr: np.exp(-3.0 * hr) * np.cos(np.pi * hr),
)

_vgmfunc = {name: (lambda hr, f=f: 1.0 - f(hr)) for name, f in _covfunc.items()}

# Models with an analytic non-zero/oscillating tail beyond hr = 1.
_ANALYTIC_TAIL = {"exp", "gau", "hol", "cyc", "dco"}

# Map common long names / aliases onto the 3-letter canonical keys.
_MODEL_ALIASES = {
    "nugget": "nug",
    "linear": "lin",
    "exponential": "exp",
    "gaussian": "gau",
    "spherical": "sph",
    "bi-square": "bsq",
    "bisquare": "bsq",
    "circular": "cir",
    "hole": "hol",
    "hole_effect": "hol",
    "periodic": "cyc",
    "cycle": "cyc",
    "damped_cosine": "dco",
    "power": "pow",
}


def resolve_model(name):
    """Return the 3-letter canonical model key for *name*.

    Accepts canonical keys (``"exp"``), full names (``"exponential"``) or any
    unambiguous prefix.  Raises ``KeyError`` with a helpful message otherwise.
    """
    key = str(name).strip().lower()
    if key in _vgmfunc:
        return key
    if key in _MODEL_ALIASES:
        return _MODEL_ALIASES[key]
    short = key[:3]
    if short in _vgmfunc:
        return short
    raise KeyError(
        f"Unknown variogram model {name!r}; valid models: {sorted(_vgmfunc)}"
    )


def calc_cov(vtype, d, psill=1.0, rng=1.0):
    """Evaluate an engine-compatible covariance model.

    Parameters
    ----------
    vtype : str
        Variogram/covariance model code or alias accepted by
        :func:`resolve_model`.
    d : array-like
        Lag distance(s) in the same units as ``rng``.
    psill : float, optional
        Partial sill multiplier for the unit correlation function.
    rng : float, optional
        Practical range / period parameter.  Must be positive.

    Returns
    -------
    numpy.ndarray or scalar-like
        ``psill * rho(d / rng)`` using the same model shapes as the Fortran
        engine.  Nugget handling here mirrors a standalone model evaluation:
        ``"nug"`` is one at zero lag and zero otherwise.  In kriging matrix
        assembly, per-structure nugget terms are added on the diagonal by the
        engine.
    """
    vt = resolve_model(vtype)
    if rng <= 0:
        raise ValueError("rng must be positive")
    hr = np.asarray(d, dtype=float) / rng
    if vt not in _ANALYTIC_TAIL:
        hr = np.minimum(1.0, hr)
    return psill * _covfunc[vt](hr)


def calc_vgm(vtype, d, psill=1.0, rng=1.0, nugget=0.0):
    """Evaluate a semivariogram model.

    The returned value is ``psill * (1 - rho(d / rng)) + nugget``.  This helper
    is intended for experimental variogram fitting and plotting, so the nugget
    is added uniformly to the curve.  Kriging matrix assembly handles nugget
    terms separately through :meth:`krigekit.Kriging.set_vgm`.
    """
    return psill * (1.0 - calc_cov(vtype, d, 1.0, rng)) + nugget


def vgmfunc(models, h, *params):
    """Evaluate a (possibly nested) variogram model at lags *h*.

    Parameters
    ----------
    models : sequence of str
        One model name per nested structure.
    *params :
        Flattened ``(sill, range)`` pairs, one pair per model, optionally
        followed by a single trailing nugget when an odd number of values is
        given:  ``sill0, range0, sill1, range1, ..., [nugget]``.

    Returns
    -------
    numpy.ndarray
        Sum of the requested nested variogram structures at the supplied lags.
    """
    has_nugget = len(params) % 2 != 0
    total = params[-1] if has_nugget else 0.0
    for im, m in enumerate(models):
        total = total + calc_vgm(m, h, params[im * 2], params[im * 2 + 1])
    return total


class vgm:
    """Backwards-compatible namespace for theoretical model functions."""

    models = tuple(_vgmfunc)
    vgmfunc = staticmethod(_vgmfunc.get)
    covfunc = staticmethod(_covfunc.get)
    calc_cov = staticmethod(calc_cov)
    calc_vgm = staticmethod(calc_vgm)
    resolve_model = staticmethod(resolve_model)


