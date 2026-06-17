# -*- coding: utf-8 -*-
"""
Variogram analysis for 1D / 2D / 3D (and space-time) geostatistics.

The module is organised in clearly separated sections:

1. Theoretical models        -- normalised model functions, ``calc_cov`` / ``calc_vgm``
2. Distance helpers          -- Euclidean and great-circle distances
3. Empirical variogram       -- ``raw_vgm`` (variogram cloud) and ``cross_vgm``
4. Binning / averaging       -- ``avg_vgm`` with optional robust (Cressie) estimator
5. Anisotropy & directions   -- ``estimate_aniso_angle``, ``directional_vgm`` (3D aware)
6. Model fitting             -- ``vgmfunc``, ``fit_vgm``
7. Plotting                  -- variogram maps (2D / polar / 3D) and model plots

Conventions
-----------
* A *variogram cloud* is a :class:`pandas.DataFrame` with one row per pair of
  points and the following columns:

  - ``index0``: index of the first point of the pair
  - ``index1``: index of the second point of the pair
  - ``distance``: Euclidean (spatial) lag ``|h|``
  - ``variogram``: semivariance of the pair,
    ``0.5 * (z_i - z_j) ** 2``
  - ``d0..d{k}``: signed lag components (``coord1 - coord0``) per dimension
  - ``dh_hori``: horizontal lag ``sqrt(d0**2 + d1**2)`` (>= 2D)
  - ``angle_h``: horizontal azimuth in degrees, clockwise from +Y (North)
  - ``angle_v``: vertical dip in degrees, positive downward (3D only)
  - ``time_lag``: absolute time lag (only when ``times`` is supplied)

* **Azimuth** is measured clockwise from the positive Y axis (compass / North),
  so ``0`` = +Y, ``90`` = +X.  **Dip** is the angle below the horizontal plane
  (positive down), matching ``(azimuth, dip, plunge)`` as scipy ``"zxy"`` Euler
  angles.

@author: michaelou (refactored)
"""

import sys
import math
import random
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import Normalize
from scipy.optimize import curve_fit, least_squares
from scipy.spatial.transform import Rotation


@dataclass
class _VgmComponent:
    """One engine-compatible variogram structure."""

    vtype: str
    nugget: float = 0.0
    sill: float = 1.0
    a_major: float = 1.0
    a_minor1: float = 1.0
    a_minor2: float = 1.0
    azimuth: float = 0.0
    dip: float = 0.0
    plunge: float = 0.0
    product: bool = False


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


def _normalise_model_specs(models):
    """Return a list of variogram specs from strings, dicts, or a model object."""
    if isinstance(models, VariogramModel):
        return models.to_kriging_specs()
    if isinstance(models, dict):
        return [dict(models)]

    specs = []
    for item in models:
        if isinstance(item, str):
            specs.append({"vtype": item})
        else:
            specs.append(dict(item))
    return specs


def _uses_model_template(models):
    """True when ``models`` carries full structure specs, not only names."""
    if isinstance(models, (VariogramModel, dict)):
        return True
    return any(not isinstance(m, str) for m in models)


def _model_from_params(models, params):
    """Build a :class:`VariogramModel` from a structure template and flat params."""
    specs = _normalise_model_specs(models)
    params = tuple(params)
    has_nugget = len(params) % 2 != 0
    if len(params) // 2 != len(specs):
        raise ValueError("params must contain one (sill, range) pair per model")

    out = VariogramModel()
    for i, spec in enumerate(specs):
        spec = dict(spec)
        spec["sill"] = params[2 * i]
        spec["a_major"] = params[2 * i + 1]
        spec.setdefault("append", i > 0)
        if has_nugget and i == 0:
            spec["nugget"] = params[-1]
        else:
            spec.setdefault("nugget", 0.0)
        out.set_vgm(**spec)
    return out


def _vgmfunc_from_model_specs(models, h, *params):
    """Evaluate variogram specs with nested/product semantics."""
    return _model_from_params(models, params).variogram(h)


class VariogramModel:
    """Python-side variogram model with ``Kriging.set_vgm``-style specs.

    The class is useful for fitting, plotting, and validating model choices
    before replaying the same structures into the Fortran solver.  It supports
    additive nested structures and product structures using the same convention
    as :meth:`krigekit.Kriging.set_vgm`: a structure with ``product=True`` is
    multiplied with the immediately preceding structure in covariance space.

    Notes
    -----
    ``covariance(h)`` and ``variogram(h)`` evaluate lag distances directly.
    Use :meth:`calc_covariance` and :meth:`calc_variogram` to evaluate between
    coordinates with each structure's anisotropy parameters applied.
    """

    def __init__(self, structures=None):
        """Create an empty model or load a sequence of ``set_vgm`` specs."""
        self.structures = []
        self.obs_coord = None
        self.obs_value = None
        self.obs_time = None
        self._raw = None
        self._avg = None
        self._dir = None
        self._params = None
        self._pcov = None
        self._fitted_model = None
        if structures is not None:
            for i, spec in enumerate(structures):
                spec = dict(spec)
                spec.setdefault("append", i > 0)
                self.set_vgm(**spec)

    @property
    def raw_variogram_(self):
        """Raw empirical variogram cloud cached by :meth:`calc_experimental`."""
        return self._raw

    @raw_variogram_.setter
    def raw_variogram_(self, value):
        """Set the cached raw empirical variogram cloud."""
        self._raw = value

    @property
    def avg_variogram_(self):
        """Averaged variogram table cached by :meth:`calc_average`."""
        return self._avg

    @avg_variogram_.setter
    def avg_variogram_(self, value):
        """Set the cached averaged variogram table."""
        self._avg = value

    @property
    def directional_variogram_(self):
        """Directional averaged variogram cached by :meth:`calc_directional_average`."""
        return self._dir

    @directional_variogram_.setter
    def directional_variogram_(self, value):
        """Set the cached directional averaged variogram table."""
        self._dir = value

    @property
    def params_(self):
        """Fitted flat parameter vector, or ``None`` before :meth:`fit`."""
        return self._params

    @property
    def pcov_(self):
        """Approximate fitted parameter covariance, or ``None`` before fitting."""
        return self._pcov

    @property
    def fitted_model_(self):
        """Most recent fitted model returned by :meth:`fit`."""
        return self._fitted_model

    def _clear_fit_state(self):
        """Clear cached fit outputs after model inputs or template change."""
        self._params = None
        self._pcov = None
        self._fitted_model = None

    def set_obs(self, coord, value, times=None):
        """Store observations for empirical variogram calculation.

        Parameters
        ----------
        coord : array-like, shape (n, dim)
            Observation coordinates.  One-dimensional coordinate vectors are
            reshaped to ``(n, 1)``.
        value : array-like, shape (n,)
            Observation values for the direct variogram.
        times : array-like, optional
            Optional per-observation time stamps for space-time variograms.

        Returns
        -------
        VariogramModel
            ``self``, so calls can be chained.
        """
        coord = np.asarray(coord, dtype=float)
        value = np.asarray(value, dtype=float).reshape(-1)
        if coord.ndim == 1:
            coord = coord.reshape(-1, 1)
        if coord.ndim != 2:
            raise ValueError("coord must have shape (n, dim)")
        if coord.shape[0] != len(value):
            raise ValueError("coord and value must have matching lengths")
        if coord.shape[1] not in (1, 2, 3):
            raise ValueError("coordinates must be 1D, 2D or 3D")
        if times is not None:
            times = np.asarray(times, dtype=float).reshape(-1)
            if len(times) != len(value):
                raise ValueError("times and value must have matching lengths")

        self.obs_coord = coord
        self.obs_value = value
        self.obs_time = times
        self._raw = None
        self._avg = None
        self._dir = None
        self._clear_fit_state()
        return self

    def experimental(self, store: bool = True, **kwargs):
        """Compute the raw empirical variogram cloud from stored observations.

        Extra keyword arguments are forwarded to :func:`raw_vgm`.  If
        ``times`` is not supplied explicitly, the time vector passed to
        :meth:`set_obs` is used.
        """
        if self.obs_coord is None or self.obs_value is None:
            raise RuntimeError("call set_obs() before experimental()")
        kwargs.setdefault("times", self.obs_time)
        cloud = raw_vgm(self.obs_coord, self.obs_value, **kwargs)
        if store:
            self._raw = cloud
            self._avg = None
            self._dir = None
        return cloud

    def calc_experimental(self, store: bool = True, **kwargs):
        """Compute the raw empirical variogram cloud from stored observations.

        This is the preferred verb-style alias for :meth:`experimental`.
        """
        return self.experimental(store=store, **kwargs)

    def calc_empirical(self, store: bool = True, **kwargs):
        """Alias for :meth:`calc_experimental`."""
        return self.calc_experimental(store=store, **kwargs)

    def average(self, rawvgm=None, store: bool = True, raw_kwargs=None, **kwargs):
        """Average a variogram cloud into lag bins.

        Parameters
        ----------
        rawvgm : pandas.DataFrame, optional
            Existing raw variogram cloud.  If omitted, the cached cloud from a
            previous :meth:`experimental` call is used; if no cache exists, a
            new cloud is computed from stored observations.
        store : bool, optional
            Store the averaged table as ``avg_variogram_``.
        raw_kwargs : dict, optional
            Keyword arguments passed to :meth:`experimental` if a new raw cloud
            must be computed.
        **kwargs
            Forwarded to :func:`avg_vgm`.
        """
        if rawvgm is None:
            if self._raw is None:
                rawvgm = self.experimental(**(raw_kwargs or {}))
            else:
                rawvgm = self._raw
        avg = avg_vgm(rawvgm, **kwargs)
        if store:
            self._avg = avg
        return avg

    def calc_average(self, rawvgm=None, store: bool = True, raw_kwargs=None, **kwargs):
        """Average a variogram cloud into lag bins.

        This is the preferred verb-style alias for :meth:`average`.
        """
        return self.average(rawvgm=rawvgm, store=store, raw_kwargs=raw_kwargs, **kwargs)

    @staticmethod
    def _raw_dimension(rawvgm):
        """Infer spatial dimension from ``d0``, ``d1`` and ``d2`` columns."""
        return sum(f"d{k}" in rawvgm.columns for k in range(3))

    def _common_orientation(self):
        """Return the shared ``(azimuth, dip, plunge)`` for all structures."""
        if not self.structures:
            raise RuntimeError("call set_vgm() before directional fitting")
        ref = self.structures[0]
        values = (ref.azimuth, ref.dip, ref.plunge)
        for comp in self.structures[1:]:
            if not np.allclose(values, (comp.azimuth, comp.dip, comp.plunge)):
                raise ValueError(
                    "directional range fitting requires all structures to share "
                    "azimuth, dip and plunge"
                )
        return values

    def _principal_directions(self, dim: int, include_minor2: bool = None):
        """Return fixed major/minor unit directions from the model orientation."""
        if dim not in (2, 3):
            raise ValueError("directional anisotropy fitting requires 2D or 3D data")
        azimuth, dip, plunge = self._common_orientation()
        if dim == 2:
            # 2D uses azimuth only; take the XY parts of the major/minor1 axes.
            rot = _engine_rotation(azimuth, 0.0, 0.0).as_matrix()
            directions = np.array([rot[1, :2], rot[0, :2]])
            names = ["major", "minor1"]
        else:
            directions = rotation_matrix_3d(azimuth, dip, plunge).T
            names = ["major", "minor1", "minor2"]
            if include_minor2 is False:
                directions = directions[:2]
                names = names[:2]
        return directions, names

    def calc_directional_average(
        self,
        rawvgm=None,
        store: bool = True,
        raw_kwargs=None,
        h_width=None,
        h_bins=15,
        cutoff=None,
        bandwidth=None,
        angle_tol=22.5,
        robust=False,
        include_minor2: bool = None,
        **kwargs,
    ):
        """Average the raw cloud along the model's fixed anisotropy axes.

        The model orientation is taken from the stored structures.  For 2D
        data, the output contains major and ``minor1`` directions; for 3D data,
        ``minor2`` is included unless ``include_minor2=False``.
        """
        if rawvgm is None:
            if self._raw is None:
                rawvgm = self.experimental(**(raw_kwargs or {}))
            else:
                rawvgm = self._raw
        if cutoff is not None:
            rawvgm = rawvgm.loc[rawvgm["distance"] <= cutoff]
        dim = self._raw_dimension(rawvgm)
        directions, names = self._principal_directions(dim, include_minor2=include_minor2)
        avg = directional_vgm(
            rawvgm,
            directions,
            h_width=h_width,
            h_bins=h_bins,
            bandwidth=bandwidth,
            angle_tol=angle_tol,
            robust=robust,
            dim=dim,
            **kwargs,
        )
        if len(avg):
            avg = avg.copy()
            avg["axis"] = pd.Categorical(
                [names[int(i)] for i in avg["direction"]],
                categories=names,
                ordered=True,
            )
        if store:
            self._dir = avg
        return avg

    def _default_fit_p0(self, fit_nugget: bool = True):
        """Build default flat fit parameters from current structure values."""
        if not self.structures:
            raise RuntimeError("call set_vgm() before fit() or pass p0")
        params = []
        for comp in self.structures:
            params.extend([comp.sill, comp.a_major])
        if fit_nugget:
            params.append(self.structures[0].nugget)
        return tuple(params)

    def calc_params(self, fit_nugget: bool = True):
        """Return the current flat ``(sill, range, ..., [nugget])`` vector."""
        return np.asarray(self._default_fit_p0(fit_nugget=fit_nugget), dtype=float)

    def _store_manual_params(self, fit_nugget: bool = True):
        """Record current model parameters after a manual adjustment."""
        self._params = self.calc_params(fit_nugget=fit_nugget)
        self._pcov = None
        self._fitted_model = self

    def set_params(
        self,
        params=None,
        *,
        sills=None,
        ranges=None,
        sill=None,
        a_major=None,
        range_=None,
        nugget=None,
        fit_nugget: bool = True,
    ):
        """Manually update fitted variogram parameters.

        ``params`` uses the same flat convention returned by :meth:`fit`:
        ``sill0, range0, sill1, range1, ..., [nugget]``.  For a single
        structure, the singular keywords ``sill``, ``a_major``/``range_`` and
        ``nugget`` are convenient alternatives.  ``sills`` and ``ranges`` may
        be sequences with one value per stored structure.

        Returns
        -------
        VariogramModel
            ``self``, so manual adjustments can be chained before plotting or
            applying to kriging.
        """
        if not self.structures:
            raise RuntimeError("call set_vgm() before set_params()")
        nstruct = len(self.structures)
        include_nugget = bool(fit_nugget)

        if params is not None:
            flat = np.asarray(params, dtype=float).reshape(-1)
            if len(flat) not in (2 * nstruct, 2 * nstruct + 1):
                raise ValueError(
                    "params must contain one (sill, range) pair per structure, "
                    "optionally followed by one trailing nugget"
                )
            for i, comp in enumerate(self.structures):
                comp.sill = float(flat[2 * i])
                comp.a_major = float(flat[2 * i + 1])
            if len(flat) == 2 * nstruct + 1:
                self.structures[0].nugget = float(flat[-1])
                include_nugget = True
            else:
                include_nugget = False

        if sills is not None:
            sills = np.asarray(sills, dtype=float).reshape(-1)
            if len(sills) != nstruct:
                raise ValueError("sills must have one value per structure")
            for comp, value in zip(self.structures, sills):
                comp.sill = float(value)

        if ranges is not None:
            ranges = np.asarray(ranges, dtype=float).reshape(-1)
            if len(ranges) != nstruct:
                raise ValueError("ranges must have one value per structure")
            if np.any(ranges <= 0.0):
                raise ValueError("ranges must be positive")
            for comp, value in zip(self.structures, ranges):
                comp.a_major = float(value)

        if sill is not None:
            if nstruct != 1:
                raise ValueError("use sills=... when the model has multiple structures")
            self.structures[0].sill = float(sill)

        new_range = a_major if a_major is not None else range_
        if new_range is not None:
            if nstruct != 1:
                raise ValueError("use ranges=... when the model has multiple structures")
            if new_range <= 0.0:
                raise ValueError("a_major/range_ must be positive")
            self.structures[0].a_major = float(new_range)

        for comp in self.structures:
            if comp.a_major <= 0.0:
                raise ValueError("ranges must be positive")

        if nugget is not None:
            self.structures[0].nugget = float(nugget)
            include_nugget = True

        self._store_manual_params(fit_nugget=include_nugget)
        return self

    def fit(
        self,
        avgvgm=None,
        p0=None,
        x_col=("distance", "mean"),
        y_col=("variogram", "mean"),
        sigma_col=None,
        weight_col=None,
        weights=None,
        bounds=None,
        inplace: bool = False,
        return_params: bool = False,
        makeplot: bool = False,
        fit_nugget: bool = True,
        raw_kwargs=None,
        avg_kwargs=None,
        **kwargs,
    ):
        """Fit this model template to an averaged variogram.

        If ``avgvgm`` is omitted, the method uses ``avg_variogram_`` when
        available, otherwise computes ``experimental()`` and ``average()`` from
        observations supplied by :meth:`set_obs`.

        ``sigma_col`` supplies SciPy-style standard deviations.  ``weights``
        or ``weight_col`` supply weighted least-squares weights, where larger
        values carry more influence.  Other parameters, except ``inplace``,
        ``return_params``, ``fit_nugget``, ``raw_kwargs`` and ``avg_kwargs``,
        are forwarded to :func:`fit_vgm`.

        Returns
        -------
        tuple
            By default ``(fitted_model, covariance)``.  If ``return_params`` is
            true, the fitted parameter vector is appended.  If ``makeplot`` is
            true, the Matplotlib axis is appended.
        """
        if avgvgm is None:
            if self._avg is None:
                avgvgm = self.average(raw_kwargs=raw_kwargs, **(avg_kwargs or {}))
            else:
                avgvgm = self._avg
        if p0 is None:
            p0 = self._default_fit_p0(fit_nugget=fit_nugget)

        result = fit_vgm(
            avgvgm,
            x_col=x_col,
            y_col=y_col,
            sigma_col=sigma_col,
            weight_col=weight_col,
            weights=weights,
            models=self,
            p0=p0,
            bounds=bounds,
            return_model=True,
            makeplot=makeplot,
            **kwargs,
        )
        params, cov, fitted = result[:3]
        extras = result[3:]
        self._params = np.asarray(params, dtype=float)
        self._pcov = cov
        self._fitted_model = fitted
        if inplace:
            self.structures = [_VgmComponent(**asdict(comp)) for comp in fitted.structures]
            fitted = self
            self._fitted_model = self
        else:
            fitted._raw = self._raw
            fitted._avg = self._avg
            fitted._params = np.asarray(params, dtype=float)
            fitted._pcov = cov
            fitted._fitted_model = fitted

        out = [fitted, cov]
        if return_params:
            out.append(params)
        out.extend(extras)
        return tuple(out)

    def _default_anisotropic_fit_p0(self, include_minor2: bool, fit_nugget: bool = True):
        """Build default ``(sill, major, minor1[, minor2], ..., [nugget])`` params."""
        if not self.structures:
            raise RuntimeError("call set_vgm() before fit_anisotropy() or pass p0")
        params = []
        for comp in self.structures:
            params.extend([comp.sill, comp.a_major, comp.a_minor1])
            if include_minor2:
                params.append(comp.a_minor2)
        if fit_nugget:
            params.append(self.structures[0].nugget)
        return tuple(params)

    def calc_anisotropic_params(self, include_minor2: bool = False,
                                fit_nugget: bool = True):
        """Return the current anisotropic flat parameter vector."""
        return np.asarray(
            self._default_anisotropic_fit_p0(
                include_minor2=include_minor2,
                fit_nugget=fit_nugget,
            ),
            dtype=float,
        )

    def set_anisotropic_params(self, params, include_minor2: bool = False,
                               fit_nugget: bool = True):
        """Manually update sills and anisotropic ranges from a flat vector.

        ``params`` follows the :meth:`fit_anisotropy` convention:
        ``sill, a_major, a_minor1`` for each structure, optionally ``a_minor2``,
        followed by a trailing nugget when ``fit_nugget=True``.
        """
        fitted = self._model_from_anisotropic_params(
            params,
            include_minor2=include_minor2,
            fit_nugget=fit_nugget,
        )
        self.structures = [_VgmComponent(**asdict(comp)) for comp in fitted.structures]
        self._params = np.asarray(params, dtype=float)
        self._pcov = None
        self._fitted_model = self
        return self

    def _model_from_anisotropic_params(self, params, include_minor2: bool,
                                       fit_nugget: bool = True):
        """Build a model from anisotropic flat fit parameters."""
        params = np.asarray(params, dtype=float).reshape(-1)
        nper = 4 if include_minor2 else 3
        expected = nper * len(self.structures) + (1 if fit_nugget else 0)
        if len(params) != expected:
            raise ValueError("anisotropic parameter vector has the wrong length")

        out = VariogramModel()
        for i, template in enumerate(self.structures):
            offset = nper * i
            spec = asdict(template)
            spec["sill"] = float(params[offset])
            spec["a_major"] = float(params[offset + 1])
            spec["a_minor1"] = float(params[offset + 2])
            if include_minor2:
                spec["a_minor2"] = float(params[offset + 3])
            else:
                spec["a_minor2"] = template.a_minor2
            spec["append"] = i > 0
            if fit_nugget and i == 0:
                spec["nugget"] = float(params[-1])
            elif i > 0:
                spec["nugget"] = template.nugget
            out.set_vgm(**spec)
        return out

    @staticmethod
    def _default_anisotropic_bounds(p0, include_minor2: bool, nstruct: int,
                                    fit_nugget: bool):
        """Return loose bounds that keep fitted ranges positive."""
        lower = np.full(len(p0), -np.inf, dtype=float)
        upper = np.full(len(p0), np.inf, dtype=float)
        nper = 4 if include_minor2 else 3
        for i in range(nstruct):
            lower[nper * i + 1] = np.finfo(float).eps
            lower[nper * i + 2] = np.finfo(float).eps
            if include_minor2:
                lower[nper * i + 3] = np.finfo(float).eps
        if fit_nugget:
            lower[-1] = 0.0
        return lower, upper

    def fit_anisotropy(
        self,
        directional=None,
        p0=None,
        axis_col="direction",
        x_col="lag",
        y_col="variogram",
        sigma_col=None,
        weight_col=None,
        weights=None,
        bounds=None,
        inplace: bool = False,
        return_params: bool = False,
        makeplot: bool = False,
        fit_nugget: bool = True,
        include_minor2: bool = None,
        raw_kwargs=None,
        directional_kwargs=None,
        maxfev=9999,
        ax=None,
        xlabel="Lag",
        ylabel="Semivariogram",
        **kwargs,
    ):
        """Fit sills and anisotropic ranges with fixed orientation.

        The fitted parameter vector is ``sill, a_major, a_minor1`` for each
        structure, optionally ``a_minor2`` for 3D fits, followed by a trailing
        nugget when ``fit_nugget=True``.  Directional data can be supplied
        directly, or it is computed with :meth:`calc_directional_average` using
        the current ``azimuth``, ``dip`` and ``plunge`` as fixed axes.
        """
        if directional is None:
            if self._dir is None:
                directional = self.calc_directional_average(
                    raw_kwargs=raw_kwargs,
                    **(directional_kwargs or {}),
                )
            else:
                directional = self._dir
        directional = directional.copy()
        if len(directional) == 0:
            raise ValueError("directional variogram table is empty")

        dir_index = np.asarray(directional[axis_col], dtype=int)
        if include_minor2 is None:
            include_minor2 = bool(np.nanmax(dir_index) >= 2)
        dim = 3 if include_minor2 else 2
        directions, _ = self._principal_directions(dim, include_minor2=include_minor2)
        if np.any((dir_index < 0) | (dir_index >= len(directions))):
            raise ValueError("direction index is outside the available model axes")

        x = np.asarray(directional[x_col], dtype=float)
        y = np.asarray(directional[y_col], dtype=float)
        finite = np.isfinite(dir_index) & np.isfinite(x) & np.isfinite(y)
        directional = directional.loc[finite].copy()
        dir_index = dir_index[finite]
        x = x[finite]
        y = y[finite]
        if len(y) == 0:
            raise ValueError("directional variogram table has no finite values")

        if p0 is None:
            p0 = self._default_anisotropic_fit_p0(
                include_minor2=include_minor2,
                fit_nugget=fit_nugget,
            )
        p0 = np.asarray(p0, dtype=float)
        if bounds is None:
            bounds = self._default_anisotropic_bounds(
                p0,
                include_minor2=include_minor2,
                nstruct=len(self.structures),
                fit_nugget=fit_nugget,
            )

        sigma = _fit_sigma(
            directional,
            x_col,
            sigma_col=sigma_col,
            weight_col=weight_col,
            weights=weights,
        )
        xdata = np.vstack([dir_index, x])

        def model(xdata, *params):
            """Curve-fit callback for fixed-axis anisotropic ranges."""
            axis = xdata[0].astype(int)
            lag = xdata[1]
            candidate = self._model_from_anisotropic_params(
                params,
                include_minor2=include_minor2,
                fit_nugget=fit_nugget,
            )
            coord0 = np.zeros((len(lag), dim), dtype=float)
            coord1 = directions[axis] * lag[:, None]
            return candidate.calc_variogram(coord0, coord1)

        fit_kwargs = dict(p0=p0, sigma=sigma, bounds=bounds, maxfev=maxfev)
        fit_kwargs.update(kwargs)
        params, cov = curve_fit(model, xdata=xdata, ydata=y, **fit_kwargs)
        fitted = self._model_from_anisotropic_params(
            params,
            include_minor2=include_minor2,
            fit_nugget=fit_nugget,
        )

        self._params = np.asarray(params, dtype=float)
        self._pcov = cov
        self._fitted_model = fitted
        if inplace:
            self.structures = [_VgmComponent(**asdict(comp)) for comp in fitted.structures]
            fitted = self
            self._fitted_model = self
        else:
            fitted._raw = self._raw
            fitted._avg = self._avg
            fitted._dir = self._dir
            fitted._params = np.asarray(params, dtype=float)
            fitted._pcov = cov
            fitted._fitted_model = fitted

        out = [fitted, cov]
        if return_params:
            out.append(params)
        if makeplot:
            if ax is None:
                _, ax = plt.subplots(figsize=(8, 5))
            for axis in np.unique(dir_index):
                mask = dir_index == axis
                ax.plot(x[mask], y[mask], marker="o", linestyle="none",
                        label=f"direction {axis:g}")
                xx = np.linspace(0.0, np.nanmax(x[mask]) * 1.1, 200)
                xxdata = np.vstack([np.full_like(xx, axis), xx])
                ax.plot(xx, model(xxdata, *params))
            ax.set(xlabel=xlabel, ylabel=ylabel)
            ax.legend()
            out.append(ax)
        return tuple(out)

    def set_vgm(
        self,
        vtype: str,
        nugget: float = 0.0,
        sill: float = 1.0,
        a_major: float = 1.0,
        a_minor1: float = None,
        a_minor2: float = None,
        azimuth: float = 0.0,
        dip: float = 0.0,
        plunge: float = 0.0,
        append: bool = True,
        product: bool = False,
    ):
        """Add one nested variogram structure.

        Parameters mirror :meth:`krigekit.Kriging.set_vgm`, except ``ivar`` and
        ``jvar`` are omitted because this object represents one variable-pair
        model.  Pass ``append=False`` to clear existing structures before
        adding the new one.  Pass ``product=True`` to multiply this structure
        with the immediately preceding structure in covariance space.

        Returns
        -------
        VariogramModel
            ``self``, so calls can be chained.
        """
        if a_minor1 is None:
            a_minor1 = a_major
        if a_minor2 is None:
            a_minor2 = a_minor1
        if a_major <= 0.0 or a_minor1 <= 0.0 or a_minor2 <= 0.0:
            raise ValueError("a_major, a_minor1 and a_minor2 must be positive")
        if not append:
            self.structures.clear()

        self.structures.append(_VgmComponent(
            vtype=resolve_model(vtype),
            nugget=float(nugget),
            sill=float(sill),
            a_major=float(a_major),
            a_minor1=float(a_minor1),
            a_minor2=float(a_minor2),
            azimuth=float(azimuth),
            dip=float(dip),
            plunge=float(plunge),
            product=bool(product),
        ))
        self._clear_fit_state()
        return self

    def set_structure_params(self, index: int = 0, **params):
        """Manually update fields on one stored variogram structure.

        Parameters
        ----------
        index : int, optional
            Zero-based structure index.
        **params
            Any :class:`_VgmComponent` field except ``append``.  Use this for
            edits that do not fit in the flat ``set_params`` vector, such as
            ``a_minor1``, ``azimuth``, ``dip`` or ``product``.
        """
        if not self.structures:
            raise RuntimeError("call set_vgm() before set_structure_params()")
        if not 0 <= index < len(self.structures):
            raise IndexError("structure index out of range")
        allowed = set(_VgmComponent.__dataclass_fields__)
        unknown = set(params) - allowed
        if unknown:
            raise TypeError(f"unknown structure parameter(s): {sorted(unknown)}")

        comp = self.structures[index]
        for key, value in params.items():
            if key == "vtype":
                value = resolve_model(value)
            elif key in {"nugget", "sill", "a_major", "a_minor1", "a_minor2",
                         "azimuth", "dip", "plunge"}:
                value = float(value)
            elif key == "product":
                value = bool(value)
            setattr(comp, key, value)

        if comp.a_major <= 0.0 or comp.a_minor1 <= 0.0 or comp.a_minor2 <= 0.0:
            raise ValueError("a_major, a_minor1 and a_minor2 must be positive")
        self._store_manual_params(fit_nugget=True)
        return self

    def set_anisotropy(
        self,
        structures=None,
        *,
        a_minor1=None,
        a_minor2=None,
        ratio_minor1=None,
        ratio_minor2=None,
        anis1=None,
        anis2=None,
        azimuth=None,
        dip=None,
        plunge=None,
    ):
        """Apply anisotropy parameters to one or more structures.

        Parameters
        ----------
        structures : int or sequence of int, optional
            Zero-based structure indices to update.  Defaults to all stored
            structures.
        a_minor1, a_minor2 : float or sequence, optional
            Absolute minor-axis ranges.  A sequence must have one value per
            selected structure.
        ratio_minor1, ratio_minor2 : float or sequence, optional
            Minor/major range ratios.  ``anis1`` and ``anis2`` are accepted as
            aliases for compatibility with the kriging search terminology.
        azimuth, dip, plunge : float, optional
            Rotation angles in degrees.

        Returns
        -------
        VariogramModel
            ``self``.
        """
        if not self.structures:
            raise RuntimeError("call set_vgm() before set_anisotropy()")
        if anis1 is not None:
            if ratio_minor1 is not None:
                raise ValueError("pass either ratio_minor1 or anis1, not both")
            ratio_minor1 = anis1
        if anis2 is not None:
            if ratio_minor2 is not None:
                raise ValueError("pass either ratio_minor2 or anis2, not both")
            ratio_minor2 = anis2
        if a_minor1 is not None and ratio_minor1 is not None:
            raise ValueError("pass either a_minor1 or ratio_minor1, not both")
        if a_minor2 is not None and ratio_minor2 is not None:
            raise ValueError("pass either a_minor2 or ratio_minor2, not both")

        if structures is None:
            indices = list(range(len(self.structures)))
        elif np.isscalar(structures):
            indices = [int(structures)]
        else:
            indices = [int(i) for i in structures]
        for index in indices:
            if not 0 <= index < len(self.structures):
                raise IndexError("structure index out of range")

        def _values(value, name):
            """Broadcast one anisotropy value or validate a selected sequence."""
            if value is None:
                return [None] * len(indices)
            arr = np.asarray(value, dtype=float)
            if arr.ndim == 0:
                return [float(arr)] * len(indices)
            flat = arr.reshape(-1)
            if len(flat) != len(indices):
                raise ValueError(f"{name} must be scalar or match selected structures")
            return [float(v) for v in flat]

        vals = {
            "a_minor1": _values(a_minor1, "a_minor1"),
            "a_minor2": _values(a_minor2, "a_minor2"),
            "ratio_minor1": _values(ratio_minor1, "ratio_minor1"),
            "ratio_minor2": _values(ratio_minor2, "ratio_minor2"),
        }

        for k, index in enumerate(indices):
            comp = self.structures[index]
            if vals["a_minor1"][k] is not None:
                comp.a_minor1 = vals["a_minor1"][k]
            if vals["ratio_minor1"][k] is not None:
                comp.a_minor1 = comp.a_major * vals["ratio_minor1"][k]
            if vals["a_minor2"][k] is not None:
                comp.a_minor2 = vals["a_minor2"][k]
            if vals["ratio_minor2"][k] is not None:
                comp.a_minor2 = comp.a_major * vals["ratio_minor2"][k]
            if azimuth is not None:
                comp.azimuth = float(azimuth)
            if dip is not None:
                comp.dip = float(dip)
            if plunge is not None:
                comp.plunge = float(plunge)
            if comp.a_minor1 <= 0.0 or comp.a_minor2 <= 0.0:
                raise ValueError("minor-axis ranges must be positive")

        self._store_manual_params(fit_nugget=True)
        return self

    def _component_covariance(self, comp, h):
        """Evaluate one structure's covariance contribution at lag ``h``."""
        h = np.asarray(h, dtype=float)
        base = calc_cov(comp.vtype, h, psill=comp.sill, rng=comp.a_major)
        return np.where(h <= 0.0, comp.sill + comp.nugget, base)

    @staticmethod
    def _coordinate_lags(coord0, coord1, pairwise: bool = False):
        """Return lag vectors ``coord1 - coord0`` for row-wise or pairwise use."""
        coord0 = np.asarray(coord0, dtype=float)
        coord1 = np.asarray(coord1, dtype=float)
        if coord0.ndim == 1:
            coord0 = coord0.reshape(1, -1)
        if coord1.ndim == 1:
            coord1 = coord1.reshape(1, -1)
        if coord0.ndim != 2 or coord1.ndim != 2:
            raise ValueError("coordinates must have shape (n, dim) or (dim,)")
        if coord0.shape[1] != coord1.shape[1]:
            raise ValueError("coordinate arrays must share the same dimensionality")
        if coord0.shape[1] not in (1, 2, 3):
            raise ValueError("coordinates must be 1D, 2D or 3D")
        if pairwise:
            return coord1[None, :, :] - coord0[:, None, :]
        if coord0.shape[0] == coord1.shape[0]:
            return coord1 - coord0
        if coord0.shape[0] == 1:
            return coord1 - coord0[0]
        if coord1.shape[0] == 1:
            return coord1[0] - coord0
        raise ValueError(
            "coordinate arrays must have matching lengths, or one array must "
            "contain a single coordinate; pass pairwise=True for an n x m matrix"
        )

    @staticmethod
    def _anisotropic_hr(comp, lag):
        """Return reduced anisotropic lag ``h/a`` for one structure.

        Lags are rotated with the engine-consistent :func:`_engine_rotation`
        (2D lags are embedded in 3D), then scaled by the per-axis ranges of the
        model frame ``(x=a_minor1, y=a_major, z=a_minor2)``.
        """
        lag = np.asarray(lag, dtype=float)
        dim = lag.shape[-1]
        if dim == 1:
            return np.abs(lag[..., 0]) / comp.a_major
        flat = lag.reshape(-1, dim)
        if dim == 2:
            flat = np.column_stack([flat, np.zeros(len(flat))])
        rotated = _engine_rotation(comp.azimuth, comp.dip, comp.plunge).apply(flat)
        ranges = np.array([comp.a_minor1, comp.a_major, comp.a_minor2], dtype=float)
        hr = np.sqrt(np.sum((rotated / ranges) ** 2, axis=-1))
        return hr.reshape(lag.shape[:-1])

    def _component_covariance_between(self, comp, lag):
        """Evaluate one structure's covariance contribution for lag vectors."""
        hr = self._anisotropic_hr(comp, lag)
        base = comp.sill * _covfunc[comp.vtype](
            np.minimum(1.0, hr) if comp.vtype not in _ANALYTIC_TAIL else hr
        )
        return np.where(hr <= 0.0, comp.sill + comp.nugget, base)

    def covariance(self, h):
        """Evaluate the nested/product covariance model at lag distance ``h``.

        Product groups are evaluated exactly like the Fortran engine: start
        with one structure, multiply by each immediately following structure
        whose ``product`` flag is true, then add the group to the total.
        """
        if not self.structures:
            return np.zeros_like(np.asarray(h, dtype=float))

        h = np.asarray(h, dtype=float)
        total = np.zeros_like(h, dtype=float)
        i = 0
        while i < len(self.structures):
            group = self._component_covariance(self.structures[i], h)
            i += 1
            while i < len(self.structures) and self.structures[i].product:
                group = group * self._component_covariance(self.structures[i], h)
                i += 1
            total = total + group
        return total

    @property
    def cov0(self):
        """Covariance at zero lag, including nugget and product groups."""
        return self.covariance(0.0)

    def variogram(self, h):
        """Evaluate the semivariogram ``gamma(h) = C(0) - C(h)``."""
        return self.cov0 - self.covariance(h)

    def calc_covariance(self, coord0, coord1, pairwise: bool = False):
        """Evaluate covariance between coordinates, applying anisotropy.

        Parameters
        ----------
        coord0, coord1 : array-like
            Coordinates with shape ``(dim,)`` or ``(n, dim)``.  By default,
            arrays with matching lengths are compared row-wise; if one side has
            one coordinate it is broadcast against the other side.
        pairwise : bool, optional
            If true, return the full ``(n0, n1)`` covariance matrix.

        Returns
        -------
        numpy.ndarray or scalar-like
            Covariance value(s) from the nested/product model.
        """
        lag = self._coordinate_lags(coord0, coord1, pairwise=pairwise)
        if not self.structures:
            return np.zeros(lag.shape[:-1], dtype=float)

        total = np.zeros(lag.shape[:-1], dtype=float)
        i = 0
        while i < len(self.structures):
            group = self._component_covariance_between(self.structures[i], lag)
            i += 1
            while i < len(self.structures) and self.structures[i].product:
                group = group * self._component_covariance_between(self.structures[i], lag)
                i += 1
            total = total + group
        total = np.asarray(total)
        return total if pairwise else total.squeeze()

    def calc_variogram(self, coord0, coord1, pairwise: bool = False):
        """Evaluate semivariogram values between coordinates with anisotropy."""
        gamma = np.asarray(self.cov0 - self.calc_covariance(
            coord0, coord1, pairwise=pairwise))
        return gamma if pairwise else gamma.squeeze()

    def plot(
        self,
        avgvgm=None,
        ax=None,
        x_col=("distance", "mean"),
        y_col=("variogram", "mean"),
        h=None,
        plot_data: bool = True,
        plot_model: bool = True,
        annotate: bool = True,
        plotkws_data=None,
        plotkws_model=None,
        xlabel="Lag",
        ylabel="Semivariogram",
    ):
        """Plot cached/explicit averaged data and the current model curve.

        If ``avgvgm`` is omitted, the cached ``_avg`` table from
        :meth:`calc_average` is used when available.  The curve is evaluated
        with :meth:`variogram`, so it represents the isotropic lag-distance
        model.  Use :meth:`calc_variogram` for anisotropy-aware coordinate
        evaluation.
        """
        avgvgm = self._avg if avgvgm is None else avgvgm
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))
        plotkws_data = plotkws_data or dict(color="darkgrey", marker=".", linewidth=0.5)
        plotkws_model = plotkws_model or dict(color="r", linewidth=1.5)

        xmax = None
        if plot_data:
            if avgvgm is None:
                raise RuntimeError("call calc_average() first or pass avgvgm")
            x = avgvgm.loc[:, x_col]
            y = avgvgm.loc[:, y_col]
            ax.plot(x, y, **plotkws_data)
            xmax = float(np.nanmax(x))
        if plot_model:
            if h is None:
                if xmax is None:
                    if self.structures:
                        xmax = max(comp.a_major for comp in self.structures)
                    else:
                        xmax = 1.0
                h = np.linspace(0.0, xmax * 1.1, 200)
            ax.plot(h, self.variogram(h), **plotkws_model)
        if annotate and self.structures:
            ms = "Model: " + "\t".join(comp.vtype.capitalize() for comp in self.structures)
            ss = "\nSill : " + "\t".join(f"{comp.sill:.5g}" for comp in self.structures)
            rr = "\nRange: " + "\t".join(f"{comp.a_major:.6g}" for comp in self.structures)
            nn = f"\nNugget: {self.structures[0].nugget:.5g}"
            ax.text(0.95, 0.05, ms + ss + rr + nn, ha="right", va="bottom",
                    transform=ax.transAxes)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        return ax

    def plot_map(
        self,
        rawvgm=None,
        ax=None,
        angle_aniso="model",
        ellipse_aniso="model",
        estimate: bool = False,
        raw_kwargs=None,
        **kwargs,
    ):
        """Plot a 2D variogram map from the cached raw cloud.

        Parameters
        ----------
        rawvgm : pandas.DataFrame, optional
            Raw variogram cloud.  If omitted, the cached ``_raw`` table is used,
            or computed from stored observations.
        angle_aniso : {"model", "estimate", None} or float, optional
            Angle overlay for maximum continuity.  ``"model"`` uses the first
            structure's azimuth, ``"estimate"`` estimates the angle from the raw
            cloud, and a float uses that azimuth directly.
        ellipse_aniso : {"model", None} or tuple, optional
            Ellipse overlay.  ``"model"`` uses the first structure's major and
            first minor range.
        estimate : bool, optional
            Shorthand for ``angle_aniso="estimate"``.
        raw_kwargs : dict, optional
            Keyword arguments passed to :meth:`experimental` if a new raw cloud
            must be computed.
        **kwargs
            Forwarded to :func:`plot_vgm_map`.

        Returns
        -------
        matplotlib.axes.Axes
            Axis containing the variogram map.
        """
        if rawvgm is None:
            if self._raw is None:
                rawvgm = self.experimental(**(raw_kwargs or {}))
            else:
                rawvgm = self._raw
        if self._raw_dimension(rawvgm) < 2:
            raise ValueError("plot_map() requires a 2D variogram cloud")

        if estimate:
            angle_aniso = "estimate"
        if angle_aniso == "model":
            angle = self.structures[0].azimuth if self.structures else None
        elif angle_aniso == "estimate":
            angle = estimate_aniso_angle(rawvgm, dim3d=False)[0][0]
        else:
            angle = angle_aniso

        if ellipse_aniso == "model":
            if self.structures:
                comp = self.structures[0]
                ellipse = (2.0 * comp.a_major, 2.0 * comp.a_minor1)
            else:
                ellipse = None
        else:
            ellipse = ellipse_aniso

        return plot_vgm_map(
            rawvgm,
            ax=ax,
            angle_aniso=angle,
            ellipse_aniso=ellipse,
            **kwargs,
        )

    def plot_map3d(
        self,
        rawvgm=None,
        ax=None,
        angle_aniso="model",
        estimate: bool = False,
        raw_kwargs=None,
        **kwargs,
    ):
        """Plot a 3D variogram map as orthogonal fence sections.

        Calls :func:`plot_vgm_map3d`.  By default (``rotate_fences=False``)
        fences align with the world X/Y/Z axes:

        - **Fence A** — horizontal XY plane (azimuth pattern).
        - **Fence B** (``n_fences ≥ 2``) — vertical XZ East–West section (dip).
        - **Fence C** (``n_fences ≥ 3``) — vertical YZ North–South section.

        A red line is projected onto each fence showing the major axis
        direction so the fitted orientation can be compared with the empirical
        map.  Pass ``rotate_fences=True`` to rotate the fences to the model's
        principal planes instead.

        Parameters
        ----------
        rawvgm : pandas.DataFrame, optional
            Raw 3D variogram cloud.  If omitted the cached cloud is used, or
            computed from stored observations.
        angle_aniso : {"model", "estimate", None} or float or tuple, optional
            Model orientation.  ``"model"`` reads ``(azimuth, dip, plunge)``
            from the first fitted structure; ``"estimate"`` estimates the
            major direction from the raw cloud; a float is azimuth only; a
            tuple is ``(azimuth[, dip[, plunge]])``.
        estimate : bool, optional
            Shorthand for ``angle_aniso="estimate"``.
        raw_kwargs : dict, optional
            Forwarded to :meth:`calc_experimental` when the raw cloud must be
            computed.
        **kwargs
            Forwarded to :func:`plot_vgm_map3d` (e.g. ``n_fences``,
            ``rotate_fences``, ``dx``, ``dz``, ``cutoff``, ``vmax``,
            ``fill_nan``).

        Returns
        -------
        matplotlib.axes.Axes
            3D axis containing the variogram map.
        """
        if rawvgm is None:
            if self._raw is None:
                rawvgm = self.calc_experimental(**(raw_kwargs or {}))
            else:
                rawvgm = self._raw
        if self._raw_dimension(rawvgm) < 3:
            raise ValueError("plot_map3d() requires a 3D variogram cloud")

        if estimate:
            angle_aniso = "estimate"
        if angle_aniso == "model":
            if self.structures:
                comp = self.structures[0]
                angle = (comp.azimuth, comp.dip, comp.plunge)
            else:
                angle = None
        elif angle_aniso == "estimate":
            angle = estimate_aniso_angle(rawvgm, dim3d=True)[0]
        else:
            angle = angle_aniso

        return plot_vgm_map3d(
            rawvgm,
            ax=ax,
            angle_aniso=angle,
            **kwargs,
        )

    def to_kriging_specs(self, replace: bool = False):
        """Return structures as dictionaries accepted by ``Kriging.set_vgm``.

        Parameters
        ----------
        replace : bool, optional
            If true, the first returned spec has ``append=False`` and later
            specs have ``append=True``.  This is convenient when applying a
            complete model to a reused :class:`krigekit.Kriging` object.
        """
        specs = []
        for i, comp in enumerate(self.structures):
            spec = asdict(comp)
            spec["append"] = not (replace and i == 0)
            specs.append(spec)
        return specs

    def apply_to(self, kriging, ivar: int, jvar: int, replace: bool = True):
        """Apply this model to a :class:`krigekit.Kriging` object.

        The first structure clears any existing model for ``(ivar, jvar)`` when
        ``replace=True``.  Set ``replace=False`` to append all structures to an
        existing model.
        """
        for spec in self.to_kriging_specs(replace=replace):
            kriging.set_vgm(ivar=ivar, jvar=jvar, **spec)
        return kriging

    def __len__(self):
        """Return the number of stored structures."""
        return len(self.structures)

    def __repr__(self):
        """Return a compact debugging representation."""
        return f"VariogramModel(nstruct={len(self.structures)})"


class VariogramSystem:
    """Multivariable variogram system for cokriging workflows.

    The system stores observations and variogram models by 1-based variable
    pair ``(ivar, jvar)``.  Each pair model is a :class:`VariogramModel`, while
    :meth:`fit_lmc` fits all requested pairs together with positive-semidefinite
    coregionalization matrices.
    """

    def __init__(self, nvar=None):
        """Create an empty multivariable variogram system."""
        self.nvar = int(nvar) if nvar is not None else None
        if self.nvar is not None and self.nvar < 1:
            raise ValueError("nvar must be positive")
        self.observations = {}
        self.models = {}
        self.raw_variograms_ = {}
        self.avg_variograms_ = {}
        self.fit_result_ = None

    def _check_ivar(self, ivar):
        """Validate and register a 1-based variable index."""
        ivar = int(ivar)
        if ivar < 1:
            raise ValueError("ivar must be a positive 1-based index")
        if self.nvar is None:
            self.nvar = ivar
        elif ivar > self.nvar:
            raise ValueError(f"ivar={ivar} exceeds nvar={self.nvar}")
        return ivar

    def _pair_key(self, ivar, jvar=None):
        """Return the canonical sorted key for a variable pair."""
        ivar = self._check_ivar(ivar)
        jvar = ivar if jvar is None else self._check_ivar(jvar)
        return (ivar, jvar) if ivar <= jvar else (jvar, ivar)

    def _get_model(self, ivar, jvar=None, create=True):
        """Return the pair model, optionally creating an empty one."""
        key = self._pair_key(ivar, jvar)
        if key not in self.models:
            if not create:
                raise KeyError(f"no variogram model has been set for pair {key}")
            self.models[key] = VariogramModel()
        return self.models[key]

    def set_obs(self, ivar, coord, value, times=None):
        """Store observations for variable ``ivar``.

        Parameters mirror :meth:`VariogramModel.set_obs`, with the additional
        1-based variable index used by :class:`krigekit.Kriging`.
        """
        ivar = self._check_ivar(ivar)
        model = VariogramModel().set_obs(coord, value, times=times)
        self.observations[ivar] = {
            "coord": model.obs_coord,
            "value": model.obs_value,
            "times": model.obs_time,
        }
        for key in list(self.raw_variograms_):
            if ivar in key:
                self.raw_variograms_.pop(key, None)
                self.avg_variograms_.pop(key, None)
        return self

    def set_vgm(self, ivar, jvar, vtype, **kwargs):
        """Add one nested structure to the model for ``(ivar, jvar)``."""
        self._get_model(ivar, jvar).set_vgm(vtype=vtype, **kwargs)
        return self

    def set_raw_vgm(self, ivar, jvar, rawvgm):
        """Store an externally computed raw variogram cloud for a pair."""
        key = self._pair_key(ivar, jvar)
        self.raw_variograms_[key] = rawvgm
        self.avg_variograms_.pop(key, None)
        return self

    def set_avg_vgm(self, ivar, jvar, avgvgm):
        """Store an externally computed averaged variogram for a pair."""
        self.avg_variograms_[self._pair_key(ivar, jvar)] = avgvgm
        return self

    def _require_obs(self, ivar):
        """Return stored observations for a variable or raise a clear error."""
        ivar = self._check_ivar(ivar)
        if ivar not in self.observations:
            raise RuntimeError(f"call set_obs(ivar={ivar}, ...) first")
        return self.observations[ivar]

    @staticmethod
    def _same_obs_grid(obs_i, obs_j):
        """Return true when two variables are collocated in space and time."""
        same_coord = (
            obs_i["coord"].shape == obs_j["coord"].shape
            and np.allclose(obs_i["coord"], obs_j["coord"])
        )
        if not same_coord:
            return False
        ti, tj = obs_i["times"], obs_j["times"]
        if ti is None or tj is None:
            return ti is None and tj is None
        return ti.shape == tj.shape and np.allclose(ti, tj)

    def calc_experimental(
        self,
        ivar,
        jvar=None,
        cross="auto",
        store=True,
        **kwargs,
    ):
        """Compute a raw empirical variogram cloud for one variable pair.

        Direct pairs use :func:`raw_vgm`.  Cross pairs use the LMC
        cross-variogram estimator :func:`raw_cross_vgm` when the observations
        are collocated.  Set ``cross="pseudo"`` to force :func:`cross_vgm`, or
        ``cross="lmc"`` to require collocated observations.
        """
        key = self._pair_key(ivar, jvar)
        obs_i = self._require_obs(key[0])
        obs_j = self._require_obs(key[1])
        cross = str(cross).lower()
        if cross not in ("auto", "lmc", "pseudo"):
            raise ValueError("cross must be 'auto', 'lmc', or 'pseudo'")

        if key[0] == key[1]:
            kwargs.setdefault("times", obs_i["times"])
            cloud = raw_vgm(obs_i["coord"], obs_i["value"], **kwargs)
        else:
            if cross != "pseudo" and self._same_obs_grid(obs_i, obs_j):
                kwargs.setdefault("times", obs_i["times"])
                cloud = raw_cross_vgm(
                    obs_i["coord"],
                    obs_i["value"],
                    obs_j["value"],
                    **kwargs,
                )
            elif cross == "lmc":
                raise ValueError(
                    "LMC cross-variogram fitting requires collocated "
                    "coordinates and matching times for the variable pair"
                )
            else:
                kwargs.setdefault("timesA", obs_i["times"])
                kwargs.setdefault("timesB", obs_j["times"])
                cloud = cross_vgm(
                    obs_i["coord"],
                    obs_i["value"],
                    obs_j["coord"],
                    obs_j["value"],
                    **kwargs,
                )

        if store:
            self.raw_variograms_[key] = cloud
        return cloud

    def calc_empirical(self, *args, **kwargs):
        """Alias for :meth:`calc_experimental`."""
        return self.calc_experimental(*args, **kwargs)

    def calc_average(
        self,
        ivar=None,
        jvar=None,
        rawvgm=None,
        store=True,
        raw_kwargs=None,
        **kwargs,
    ):
        """Average one pair, or all cached raw variograms when no pair is given."""
        if ivar is None:
            out = {}
            keys = sorted(self.raw_variograms_)
            if not keys:
                nvar = self.nvar or 0
                keys = [(i, j) for i in range(1, nvar + 1)
                        for j in range(i, nvar + 1)
                        if i in self.observations and j in self.observations]
            for key in keys:
                out[key] = self.calc_average(
                    key[0], key[1], store=store,
                    raw_kwargs=raw_kwargs, **kwargs)
            return out

        key = self._pair_key(ivar, jvar)
        if rawvgm is None:
            if key not in self.raw_variograms_:
                rawvgm = self.calc_experimental(
                    key[0], key[1], **(raw_kwargs or {}))
            else:
                rawvgm = self.raw_variograms_[key]
        avg = avg_vgm(rawvgm, **kwargs)
        if store:
            self.avg_variograms_[key] = avg
        return avg

    def fit_pair(self, ivar, jvar=None, avgvgm=None, inplace=True, **kwargs):
        """Fit one variable-pair model independently.

        This is convenient for direct variograms, but :meth:`fit_lmc` is the
        safer choice for cokriging because it constrains cross-pair sills.
        """
        key = self._pair_key(ivar, jvar)
        model = self._get_model(*key, create=False)
        if avgvgm is None:
            avgvgm = self.avg_variograms_.get(key)
        if avgvgm is None:
            avgvgm = self.calc_average(*key)
        fitted, *rest = model.fit(avgvgm, inplace=inplace, **kwargs)
        if inplace:
            self.models[key] = fitted
        return (fitted, *rest)

    def _template_components(self):
        """Return the shared LMC structure template from existing models."""
        if not self.models:
            raise RuntimeError("call set_vgm() before fit_lmc()")
        preferred = self.models.get((1, 1))
        template_model = preferred if preferred and len(preferred) else None
        if template_model is None:
            template_model = next((m for m in self.models.values() if len(m)), None)
        if template_model is None:
            raise RuntimeError("call set_vgm() before fit_lmc()")
        if any(comp.product for comp in template_model.structures):
            raise NotImplementedError("fit_lmc() currently supports additive LMC structures only")
        return [_VgmComponent(**asdict(comp)) for comp in template_model.structures]

    def _validate_lmc_templates(self, template):
        """Check pair models are compatible with the shared LMC template."""
        for key, model in self.models.items():
            if len(model.structures) != len(template):
                raise ValueError(f"pair {key} does not match the LMC structure count")
            for comp, tmpl in zip(model.structures, template):
                if comp.product or comp.vtype != tmpl.vtype:
                    raise ValueError(
                        f"pair {key} must use the same additive model types "
                        "as the LMC template"
                    )

    @staticmethod
    def _nearest_psd_factor(matrix):
        """Return a lower factor for a numerically PSD version of ``matrix``."""
        matrix = 0.5 * (matrix + matrix.T)
        evals, evecs = np.linalg.eigh(matrix)
        evals = np.clip(evals, 0.0, None)
        psd = (evecs * evals) @ evecs.T
        psd = 0.5 * (psd + psd.T)
        jitter = max(np.trace(psd), 1.0) * 1e-12
        return np.linalg.cholesky(psd + np.eye(psd.shape[0]) * jitter)

    def _initial_lmc_matrices(self, template, fit_nugget):
        """Build initial nugget and partial-sill matrices from pair models."""
        nvar = self.nvar or max(max(k) for k in self.models)
        mats = [np.zeros((nvar, nvar), dtype=float) for _ in template]
        nugget = np.zeros((nvar, nvar), dtype=float)
        for i in range(1, nvar + 1):
            for j in range(i, nvar + 1):
                model = self.models.get((i, j))
                if model is None:
                    continue
                for k, comp in enumerate(model.structures):
                    mats[k][i - 1, j - 1] = mats[k][j - 1, i - 1] = comp.sill
                if fit_nugget and model.structures:
                    nugget[i - 1, j - 1] = nugget[j - 1, i - 1] = model.structures[0].nugget
        return nugget, mats

    def _default_lmc_pairs(self):
        """Return variable pairs to include in LMC fitting."""
        if self.avg_variograms_:
            return sorted(self.avg_variograms_)
        nvar = self.nvar or 0
        return [(i, j) for i in range(1, nvar + 1)
                for j in range(i, nvar + 1)
                if i in self.observations and j in self.observations]

    def fit_lmc(
        self,
        pairs=None,
        x_col=("distance", "mean"),
        y_col=("variogram", "mean"),
        sigma_col=None,
        weight_col=None,
        fit_ranges=True,
        fit_nugget=True,
        inplace=False,
        raw_kwargs=None,
        avg_kwargs=None,
        max_nfev=20000,
        **kwargs,
    ):
        """Fit an additive linear model of coregionalization.

        The sill matrix for each nested structure is parameterized as
        ``L @ L.T`` during optimization, so every fitted coregionalization
        matrix is positive semidefinite.  Ranges are shared across all pairs.
        Use ``sigma_col`` for uncertainty-style residual scaling or
        ``weight_col`` for weighted least squares.
        """
        if sigma_col is not None and weight_col is not None:
            raise ValueError("pass either sigma_col or weight_col, not both")
        template = self._template_components()
        self._validate_lmc_templates(template)
        pairs = self._default_lmc_pairs() if pairs is None else [self._pair_key(*p) for p in pairs]
        if not pairs:
            raise RuntimeError("no averaged variograms or observations are available for fit_lmc()")

        avg_tables = {}
        for key in pairs:
            avg = self.avg_variograms_.get(key)
            if avg is None:
                avg = self.calc_average(
                    key[0], key[1],
                    raw_kwargs=raw_kwargs,
                    **(avg_kwargs or {}),
                )
            avg_tables[key] = avg

        nvar = self.nvar or max(max(k) for k in pairs)
        tri = np.tril_indices(nvar)
        ntri = len(tri[0])
        init_nugget, init_mats = self._initial_lmc_matrices(template, fit_nugget)
        init_ranges = np.array([comp.a_major for comp in template], dtype=float)

        p0 = []
        if fit_nugget:
            p0.extend(self._nearest_psd_factor(init_nugget)[tri])
        for mat in init_mats:
            p0.extend(self._nearest_psd_factor(mat)[tri])
        if fit_ranges:
            p0.extend(np.log(init_ranges))
        p0 = np.asarray(p0, dtype=float)

        def unpack(params):
            """Unpack optimizer parameters into nugget, sill matrices and ranges."""
            offset = 0
            nugget = np.zeros((nvar, nvar), dtype=float)
            if fit_nugget:
                lmat = np.zeros((nvar, nvar), dtype=float)
                lmat[tri] = params[offset:offset + ntri]
                nugget = lmat @ lmat.T
                offset += ntri
            mats = []
            for _ in template:
                lmat = np.zeros((nvar, nvar), dtype=float)
                lmat[tri] = params[offset:offset + ntri]
                mats.append(lmat @ lmat.T)
                offset += ntri
            ranges = init_ranges.copy()
            if fit_ranges:
                ranges = np.exp(params[offset:offset + len(template)])
            return nugget, mats, ranges

        def residual(params):
            """Return concatenated weighted residuals for all fitted pairs."""
            nugget, mats, ranges = unpack(params)
            pieces = []
            for key, avg in avg_tables.items():
                i, j = key[0] - 1, key[1] - 1
                h = np.asarray(avg.loc[:, x_col], dtype=float)
                pred = np.where(h <= 0.0, 0.0, nugget[i, j])
                for k, comp in enumerate(template):
                    pred = pred + calc_vgm(
                        comp.vtype, h,
                        psill=mats[k][i, j],
                        rng=ranges[k],
                    )
                obs = np.asarray(avg.loc[:, y_col], dtype=float)
                r = pred - obs
                if sigma_col is not None and sigma_col in avg.columns:
                    sigma = np.asarray(avg.loc[:, sigma_col], dtype=float)
                    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
                        raise ValueError("sigma values must be finite and positive")
                    r = r / sigma
                elif weight_col is not None:
                    if weight_col not in avg.columns:
                        raise KeyError(f"weight_col {weight_col!r} is not present in avgvgm")
                    w = np.asarray(avg.loc[:, weight_col], dtype=float)
                    if np.any(~np.isfinite(w)) or np.any(w <= 0.0):
                        raise ValueError("weights must be finite and positive")
                    r = r * np.sqrt(w)
                pieces.append(r)
            return np.concatenate(pieces)

        result = least_squares(residual, p0, max_nfev=max_nfev, **kwargs)
        nugget, mats, ranges = unpack(result.x)
        fitted = self._build_lmc_system(template, nugget, mats, ranges)
        fitted.observations = dict(self.observations)
        fitted.raw_variograms_ = dict(self.raw_variograms_)
        fitted.avg_variograms_ = dict(self.avg_variograms_)
        fitted.fit_result_ = result
        if inplace:
            self.models = fitted.models
            self.fit_result_ = result
            fitted = self
        return fitted, result

    def _build_lmc_system(self, template, nugget, mats, ranges):
        """Build a fitted :class:`VariogramSystem` from LMC matrices."""
        nvar = self.nvar or nugget.shape[0]
        fitted = VariogramSystem(nvar=nvar)
        for i in range(1, nvar + 1):
            for j in range(i, nvar + 1):
                model = VariogramModel()
                for k, comp in enumerate(template):
                    spec = asdict(comp)
                    ratio = ranges[k] / comp.a_major
                    spec["sill"] = mats[k][i - 1, j - 1]
                    spec["a_major"] = ranges[k]
                    spec["a_minor1"] = comp.a_minor1 * ratio
                    spec["a_minor2"] = comp.a_minor2 * ratio
                    spec["nugget"] = nugget[i - 1, j - 1] if k == 0 else 0.0
                    spec["append"] = k > 0
                    model.set_vgm(**spec)
                fitted.models[(i, j)] = model
        return fitted

    def get_lmc_matrices(self, include_nugget=True):
        """Return fitted/coregionalization matrices from current pair models."""
        template = self._template_components()
        nugget, mats = self._initial_lmc_matrices(template, include_nugget)
        return (nugget, mats) if include_nugget else mats

    def apply_to(self, kriging, replace=True, pairs=None):
        """Apply all pair models to a :class:`krigekit.Kriging` object."""
        pairs = sorted(self.models) if pairs is None else [self._pair_key(*p) for p in pairs]
        for key in pairs:
            self.models[key].apply_to(kriging, key[0], key[1], replace=replace)
        return kriging

    def __repr__(self):
        """Return a compact debugging representation."""
        return f"VariogramSystem(nvar={self.nvar}, npairs={len(self.models)})"


# ---------------------------------------------------------------------------
# 2. Distance helpers
# ---------------------------------------------------------------------------
def _great_circle_dist(coord1, coord2, earth_r=6371.2):
    """Great-circle distance for ``(lon, lat)`` coordinates (degrees)."""
    coord1 = np.radians(coord1)
    coord2 = np.radians(coord2)
    lon1, lat1 = coord1[:, 0], coord1[:, 1]
    lon2, lat2 = coord2[:, 0], coord2[:, 1]
    res = (np.sin(lat1) * np.sin(lat2)
           + np.cos(lat1) * np.cos(lat2) * np.cos(lon1 - lon2))
    return earth_r * np.arccos(np.clip(res, -1.0, 1.0))


class vgm:
    """Namespace exposing the theoretical model functions.

    Kept for backwards compatibility (it used to be an empty placeholder) and
    as a convenient handle, e.g. ``vgm.calc_vgm("exp", d, sill, rng)``.
    """

    models = tuple(_vgmfunc)
    vgmfunc = staticmethod(_vgmfunc.get)
    covfunc = staticmethod(_covfunc.get)
    calc_cov = staticmethod(calc_cov)
    calc_vgm = staticmethod(calc_vgm)
    resolve_model = staticmethod(resolve_model)


# ---------------------------------------------------------------------------
# 3. Empirical variogram (variogram cloud)
# ---------------------------------------------------------------------------
def _choose_data(coords, vals, nmax, times=None):
    """Optionally draw a random subset of at most *nmax* points.

    Returns ``(selected_indices, coords, vals, times)`` where the arrays are
    restricted to the selection (and the indices map back to the originals).
    """
    n = len(coords)
    if nmax is None or nmax >= n:
        return np.arange(n), coords, vals, times

    selected = np.sort(random.sample(range(n), nmax))
    times = np.asarray(times)[selected] if times is not None else None
    return selected, coords[selected], vals[selected], times


def _pair_lags(coords0, coords1, val0, val1, cutoff,
               time0, time1, time_cutoff, calc_angle,
               valB0=None, valB1=None, great_circle=False):
    """Compute lags and (cross-)semivariances between one point and a block.

    ``coords0`` holds a single point (shape ``(1, dim)``); ``coords1`` holds the
    candidate partners.  Returns a dict of the per-pair arrays already reduced
    to the pairs that pass the spatial (and optional temporal) cutoff, plus the
    boolean ``mask`` over ``coords1``.

    When a second variable (``valB0`` for the anchor point, ``valB1`` for the
    block) is supplied, the ``variogram`` column holds the **cross**-semivariance
    ``0.5 * (A_i - A_j) * (B_i - B_j)`` instead of the direct semivariance.

    With ``great_circle`` (2D ``(lon, lat)`` coordinates in degrees) the spatial
    lag ``distance`` is the great-circle distance; the signed components and
    azimuth are still derived from the planar degree differences.
    """
    dh = coords1 - coords0                       # signed lag components
    if great_circle:
        hlag = _great_circle_dist(np.broadcast_to(coords0, coords1.shape), coords1)
    else:
        hlag = np.linalg.norm(dh, axis=1)        # |h|

    mask = hlag <= cutoff
    if time0 is not None:
        tlag = np.abs(time1 - time0)
        mask &= tlag <= time_cutoff

    out = {"mask": mask}
    if not np.any(mask):
        return out

    dim = coords1.shape[1]
    dh = dh[mask]
    out["distance"] = hlag[mask]
    dA = val1[mask] - val0
    if valB0 is None:
        out["variogram"] = 0.5 * dA ** 2
    else:
        out["variogram"] = 0.5 * dA * (valB1[mask] - valB0)
    out["dh"] = dh
    if time0 is not None:
        out["time_lag"] = np.abs(time1 - time0)[mask]

    if dim >= 2:
        dx, dy = dh[:, 0], dh[:, 1]
        out["dh_hori"] = np.hypot(dx, dy)
        if calc_angle:
            # azimuth clockwise from +Y (North): atan2(dx, dy)
            out["angle_h"] = np.degrees(np.arctan2(dx, dy)) % 360.0
            if dim == 3:
                # dip below horizontal plane, positive downward (matches the
                # model ``dip`` convention and scipy 'zxy' Euler angles)
                out["angle_v"] = np.degrees(np.arctan2(-dh[:, 2], out["dh_hori"]))
    return out


def _build_cloud(dim, idx0, idx1, records, i0s, i1s):
    """Assemble the variogram-cloud DataFrame from accumulated per-block lists."""
    if not records["distance"]:
        cols = ["index0", "index1", "distance", "variogram"] + [f"d{k}" for k in range(dim)]
        return pd.DataFrame(columns=cols)

    out = {
        "index0": idx0[np.concatenate(i0s)],
        "index1": idx1[np.concatenate(i1s)],
        "distance": np.concatenate(records["distance"]),
        "variogram": np.concatenate(records["variogram"]),
    }
    dh = np.concatenate(records["dh"], axis=0)
    for k in range(dim):
        out[f"d{k}"] = dh[:, k]
    for key in ("time_lag", "dh_hori", "angle_h", "angle_v"):
        if records[key]:
            out[key] = np.concatenate(records[key])
    return pd.DataFrame(out)


def _empty_records():
    """Create empty list accumulators for variogram-cloud columns."""
    return {k: [] for k in
            ("distance", "variogram", "dh", "time_lag", "dh_hori", "angle_h", "angle_v")}


def _accumulate(records, pair):
    """Append one block of pair arrays into the variogram-cloud accumulators."""
    for key in records:
        if key in pair:
            records[key].append(pair[key])


def _progress(i, total, verbose):
    """Print an in-place integer progress percentage when ``verbose`` is true."""
    if verbose and total > 0 and i % max(total // 100, 1) == 0:
        sys.stdout.write(f"\rProgress: {int(i / total * 100)}%")
        sys.stdout.flush()


def _axis_angle_diff(angle, target):
    """Smallest angular difference in degrees between two undirected axes."""
    return np.abs((np.asarray(angle) - target + 90.0) % 180.0 - 90.0)


def _circular_angle_diff(angle, target):
    """Smallest angular difference in degrees between two directed bearings."""
    return np.abs((np.asarray(angle) - target + 180.0) % 360.0 - 180.0)


def _axis_dip_mask(angle_h, angle_v, target_h, target_v, h_tol, v_tol):
    """Mask for an undirected 3D variogram axis.

    The same axis can be represented as ``(azimuth, dip)`` or as the reversed
    lag vector ``(azimuth + 180, -dip)``.  Test both directed representations so
    vertical directions are not mismatched when horizontal azimuth is flipped.
    """
    angle_h = np.asarray(angle_h)
    angle_v = np.asarray(angle_v)
    same = (
        (_circular_angle_diff(angle_h, target_h) <= h_tol)
        & (np.abs(angle_v - target_v) <= v_tol)
    )
    flipped = (
        (_circular_angle_diff(angle_h, target_h + 180.0) <= h_tol)
        & (np.abs(angle_v + target_v) <= v_tol)
    )
    return same | flipped


def raw_vgm(coords, vals, cutoff=np.inf, times=None, t_cutoff=np.inf,
            calc_angle=False, maxobs=None, seed=None, verbose=True,
            great_circle=False):
    """Compute the raw empirical variogram cloud (all admissible pairs).

    Parameters
    ----------
    coords : array-like, shape (n, dim)
        Sample coordinates; ``dim`` may be 1, 2 or 3.
    vals : array-like, shape (n,)
        Sample values.
    cutoff : float, optional
        Maximum spatial lag ``|h|`` to keep (default: no cutoff).
    times : array-like, optional
        Per-sample time stamps for a space-time variogram.
    t_cutoff : float, optional
        Maximum absolute time lag to keep.
    calc_angle : bool, optional
        Also compute ``angle_h`` (and ``angle_v`` in 3D) for each pair.
    maxobs : int, optional
        Randomly subsample to at most ``maxobs`` points before pairing.
    seed : int, optional
        Random seed for the subsampling.
    verbose : bool, optional
        Print a progress indicator.
    great_circle : bool, optional
        Treat 2D coordinates as ``(lon, lat)`` in degrees and use the
        great-circle distance (km) for the ``distance`` lag and ``cutoff``.

    Returns
    -------
    pandas.DataFrame
        The variogram cloud (see module docstring for the columns).
    """
    coords = np.asarray(coords, dtype=float)
    vals = np.asarray(vals, dtype=float).reshape(-1)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 1)
    n, dim = coords.shape
    if len(vals) != n:
        raise ValueError("coords and vals must have matching lengths")
    if dim not in (1, 2, 3):
        raise ValueError("coordinates must be 1D, 2D or 3D")
    if great_circle and dim != 2:
        raise ValueError("great_circle requires 2D (lon, lat) coordinates")

    if seed is not None:
        random.seed(seed)
    selected, coords, vals, times = _choose_data(coords, vals, maxobs, times)
    n = len(selected)
    if verbose:
        print(f"{n} points, dim={dim}")
    has_time = times is not None

    records = _empty_records()
    i0s, i1s = [], []
    for i in range(n - 1):
        pair = _pair_lags(
            coords[i:i + 1], coords[i + 1:], vals[i], vals[i + 1:], cutoff,
            times[i] if has_time else None, times[i + 1:] if has_time else None,
            t_cutoff, calc_angle, great_circle=great_circle)
        mask = pair["mask"]
        if np.any(mask):
            i0s.append(np.full(np.count_nonzero(mask), i))
            i1s.append(np.arange(i + 1, n)[mask])
            _accumulate(records, pair)
        _progress(i, n - 2, verbose)

    if verbose:
        print("\rProgress: 100%")
    return _build_cloud(dim, selected, selected, records, i0s, i1s)


def raw_cross_vgm(coords, valsA, valsB, cutoff=np.inf, times=None, t_cutoff=np.inf,
                  calc_angle=False, maxobs=None, seed=None, verbose=True):
    """Traditional (LMC) cross-variogram cloud on **collocated** data.

    Both variables are measured at the same ``coords`` (isotopic data).  Pairs
    are formed *within* the single point set and the cross-semivariance::

        gamma_AB(h) = 0.5 * (A_i - A_j) * (B_i - B_j)

    is stored in the ``variogram`` column.  Unlike the pseudo estimator in
    :func:`cross_vgm`, this is symmetric in A/B, its sill is the LMC cross-sill
    you can feed into cokriging, and it costs O(n**2 / 2) rather than O(nA*nB).

    The returned cloud has exactly the same columns as :func:`raw_vgm`, so it
    flows straight into :func:`avg_vgm`, :func:`estimate_aniso_angle`,
    :func:`directional_vgm` and the variogram-map plots.  Calling it with
    ``valsB is valsA`` reproduces the direct variogram of A.

    For *heterotopic* data (A and B at different locations) the true
    cross-variogram is not computable; use :func:`cross_vgm` (pseudo estimator)
    or restrict to a collocated subset before calling this function.
    """
    coords = np.asarray(coords, dtype=float)
    valsA = np.asarray(valsA, dtype=float).reshape(-1)
    valsB = np.asarray(valsB, dtype=float).reshape(-1)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 1)
    n, dim = coords.shape
    if not (len(valsA) == n and len(valsB) == n):
        raise ValueError("coords, valsA and valsB must have matching lengths")
    if dim not in (1, 2, 3):
        raise ValueError("coordinates must be 1D, 2D or 3D")

    if seed is not None:
        random.seed(seed)
    selected, coords, _, times = _choose_data(coords, valsA, maxobs, times)
    valsA = valsA[selected]
    valsB = valsB[selected]
    n = len(selected)
    if verbose:
        print(f"{n} collocated points, dim={dim}")
    has_time = times is not None

    records = _empty_records()
    i0s, i1s = [], []
    for i in range(n - 1):
        pair = _pair_lags(
            coords[i:i + 1], coords[i + 1:], valsA[i], valsA[i + 1:], cutoff,
            times[i] if has_time else None, times[i + 1:] if has_time else None,
            t_cutoff, calc_angle, valB0=valsB[i], valB1=valsB[i + 1:])
        mask = pair["mask"]
        if np.any(mask):
            i0s.append(np.full(np.count_nonzero(mask), i))
            i1s.append(np.arange(i + 1, n)[mask])
            _accumulate(records, pair)
        _progress(i, n - 2, verbose)

    if verbose:
        print("\rProgress: 100%")
    return _build_cloud(dim, selected, selected, records, i0s, i1s)


def cross_vgm(coordsA, valsA, coordsB, valsB, cutoff=np.inf, residual=True,
              timesA=None, timesB=None, t_cutoff=np.inf, calc_angle=False,
              maxobs=None, maxobsA=None, maxobsB=None, seed=None, verbose=True):
    """Cross-variogram cloud between two co-located variables A and B.

    Pairs are formed across the two data sets (every A vs every B).  When
    ``residual`` is True the means are removed first so the result is a
    cross-covariance-style estimator.

    .. warning::
        This is the **pseudo** cross-variogram ``0.5 * E[(A(x) - B(x+h))**2]``,
        not the linear-model-of-coregionalisation (LMC) cross-variogram
        ``0.5 * E[(A(x) - A(x+h)) * (B(x) - B(x+h))]``.  Its sill mixes
        ``var(A) + var(B) - 2*cov(A, B)``, so do **not** read an LMC cross-sill
        directly off it.  For heterotopic data (A and B at different locations)
        the true cross-variogram is not computable and this pseudo form is a
        reasonable substitute; for a cokriging cross-sill, derive
        ``cov(A, B)`` from collocated hard-data pairs instead.
    """
    coordsA = np.asarray(coordsA, dtype=float)
    coordsB = np.asarray(coordsB, dtype=float)
    valsA = np.asarray(valsA, dtype=float).reshape(-1)
    valsB = np.asarray(valsB, dtype=float).reshape(-1)
    if coordsA.ndim == 1:
        coordsA = coordsA.reshape(-1, 1)
    if coordsB.ndim == 1:
        coordsB = coordsB.reshape(-1, 1)

    nA, dA = coordsA.shape
    nB, dB = coordsB.shape
    if not (nA == len(valsA) and nB == len(valsB)):
        raise ValueError("coords and vals must have matching lengths")
    if dA != dB:
        raise ValueError("A and B must share the same dimensionality")
    dim = dA
    if dim not in (1, 2, 3):
        raise ValueError("coordinates must be 1D, 2D or 3D")

    if maxobs is not None:
        maxobsA = maxobsA or min(maxobs, nA)
        maxobsB = maxobsB or min(maxobs, nB)
    if maxobsA is not None and maxobsB is None:
        maxobsB = min(maxobsA, nB)

    if seed is not None:
        random.seed(seed)
    selA, coordsA, valsA, timesA = _choose_data(coordsA, valsA, maxobsA, timesA)
    selB, coordsB, valsB, timesB = _choose_data(coordsB, valsB, maxobsB, timesB)
    nA = len(selA)

    if residual:
        valsA = valsA - np.mean(valsA)
        valsB = valsB - np.mean(valsB)

    has_time = timesA is not None and timesB is not None

    records = _empty_records()
    i0s, i1s = [], []
    for i in range(nA):
        pair = _pair_lags(
            coordsA[i:i + 1], coordsB, valsA[i], valsB, cutoff,
            timesA[i] if has_time else None, timesB if has_time else None,
            t_cutoff, calc_angle)
        mask = pair["mask"]
        if np.any(mask):
            i0s.append(np.full(np.count_nonzero(mask), i))
            i1s.append(np.arange(len(selB))[mask])
            _accumulate(records, pair)
        _progress(i, nA - 1, verbose)

    if verbose:
        print("\rProgress: 100%")
    return _build_cloud(dim, selA, selB, records, i0s, i1s)


# ---------------------------------------------------------------------------
# 4. Binning / averaging
# ---------------------------------------------------------------------------
def _cressie_gamma(semivariances):
    """Robust Cressie--Hawkins variogram estimate from per-pair semivariances.

    From semivariance ``g = 0.5 (z_i - z_j)^2`` we recover ``|z_i - z_j|^0.5 =
    (2 g)^0.25``.  The robust estimator is::

        2*gamma = mean(|dz|^0.5)^4 / (0.457 + 0.494/N + 0.045/N^2)
    """
    g = np.asarray(semivariances, dtype=float)
    n = g.size
    if n == 0:
        return np.nan
    term = np.mean((2.0 * g) ** 0.25) ** 4
    return 0.5 * term / (0.457 + 0.494 / n + 0.045 / n ** 2)


def avg_vgm(rawvgm, h_col="distance", t_col=None, cutoff=None, t_cutoff=None,
            h_width=None, t_width=None, h_bins=None, t_bins=None,
            tor_hori=None, tor_vert=None, angleh=None, anglev=None,
            angleh_tor=15, anglev_tor=10, robust=False, vgm_col="variogram"):
    """Bin a variogram cloud and average it.

    Supports distance (and optional time) binning, directional and bandwidth
    filtering, and either the classic (Matheron) mean or the robust
    Cressie--Hawkins estimator (``robust=True``).

    Directional filtering follows variogram-axis symmetry.  If only
    ``angleh`` is supplied, azimuths ``angleh`` and ``angleh + 180`` are treated
    as the same horizontal axis.  If both ``angleh`` and ``anglev`` are
    supplied, the same 3D axis is represented by ``(angleh, anglev)`` and the
    reversed lag vector ``(angleh + 180, -anglev)``.

    Returns
    -------
    pandas.DataFrame
        Grouped variogram table whose columns are a ``(variable, statistic)``
        MultiIndex with statistics ``mean``, ``std`` and ``count``.  The
        averaged semivariogram is available as ``result[(vgm_col, "mean")]``.
    """
    df = rawvgm
    if cutoff is not None:
        df = df.loc[df[h_col] <= cutoff]
    if t_cutoff is not None and t_col is not None:
        df = df.loc[df[t_col] <= t_cutoff]
    if tor_hori is not None:
        df = df.loc[df["dh_hori"] <= tor_hori]
    if tor_vert is not None:
        df = df.loc[df["d2"].abs() <= tor_vert]
    if angleh is not None and anglev is not None:
        mask = _axis_dip_mask(
            df["angle_h"], df["angle_v"],
            angleh, anglev, angleh_tor, anglev_tor,
        )
        df = df.loc[mask]
    elif angleh is not None:
        # Keep pairs whose azimuth is within +/-angleh_tor of angleh or its
        # 180-degree opposite; without a dip constraint this is an undirected
        # horizontal axis.
        diff = _axis_angle_diff(df["angle_h"], angleh)
        df = df.loc[diff <= angleh_tor]
    elif anglev is not None:
        diff = df["angle_v"] - anglev
        df = df.loc[(diff >= -anglev_tor) & (diff < anglev_tor)]

    df = df.copy()
    indices = []
    if h_col in df.columns:
        if h_bins is None and h_width is None:
            h_bins = 15
        if isinstance(h_bins, int) and h_width is None:
            hmax = df[h_col].max()
            if not np.isfinite(hmax) or hmax <= 0:
                raise ValueError("cannot bin a variogram cloud with no positive lags")
            h_width = hmax / h_bins
        if h_width is not None:
            df["hindex"] = (df[h_col] // h_width).astype(int)
        else:
            df["hindex"] = np.searchsorted(h_bins, df[h_col])
        indices.append("hindex")

    if t_col is not None and t_col in df.columns:
        if t_bins is None and t_width is None:
            t_bins = 15
        if isinstance(t_bins, int) and t_width is None:
            t_width = df[t_col].max() / t_bins
        if t_width is not None:
            df["tindex"] = (df[t_col] // t_width).astype(int)
        else:
            df["tindex"] = np.searchsorted(t_bins, df[t_col])
        indices.append("tindex")

    grouped = df.groupby(indices)
    out = grouped.agg(["mean", "std", "count"])
    if robust:
        out[(vgm_col, "mean")] = grouped[vgm_col].apply(_cressie_gamma).values
    return out


# ---------------------------------------------------------------------------
# 5. Anisotropy and directional analysis
# ---------------------------------------------------------------------------
def distance_pnt_line(dx, dy, azimuth):
    """Perpendicular distance from offset vectors ``(dx, dy)`` to a line through
    the origin with the given *azimuth* (degrees, clockwise from +Y)."""
    theta = np.deg2rad(azimuth)
    nx, ny = np.cos(theta), -np.sin(theta)   # unit normal to the line
    return np.abs(dx * nx + dy * ny)


def filter_vgm(dx, dy, azimuth, azimuth_anis, bandwidth=None, angle_tol=10):
    """Boolean mask selecting pairs aligned with *azimuth_anis*.

    This is a 2D/horizontal-axis helper.  A pair is kept when its azimuth is
    within ``angle_tol`` of the target axis, treating ``azimuth_anis`` and
    ``azimuth_anis + 180`` as equivalent.  If ``bandwidth`` is given, the pair
    must also fall within ``bandwidth / 2`` of the directional line.

    For 3D azimuth+dip filtering, use :func:`avg_vgm`, which flips the sign of
    dip for the reversed azimuth.
    """
    ang_diff = _axis_angle_diff(azimuth, azimuth_anis)
    mask = ang_diff < angle_tol
    if bandwidth is not None:
        mask = mask & (distance_pnt_line(dx, dy, azimuth_anis) < bandwidth / 2.0)
    return mask


def _canonical_axis_sign(vector, dim3d):
    """Fix the arbitrary sign of an eigenvector for reproducible angles.

    ``numpy.linalg.eigh`` returns eigenvectors with an arbitrary sign, which
    would flip the reported dip (and, in 2D, the azimuth by 180 degrees) from
    run to run on identical data.  An anisotropy axis is undirected, so we pick a
    deterministic orientation: the vertical component non-positive in 3D (so a
    typical down-tilted axis reports a positive, down-dip), the +Y (northing)
    component non-negative in 2D.
    """
    vector = np.asarray(vector, dtype=float)
    if dim3d:
        if vector[2] > 0:
            vector = -vector
    elif vector[1] < 0:
        vector = -vector
    return vector


def estimate_aniso_angle(rawvgm, x="d0", y="d1", dist="distance", vgm="variogram",
                         r_max=None, z="d2", dim3d=False, robust=False,
                         get_eigens=False, dx=None, dy=None, dz=None):
    """Estimate anisotropy directions from a variogram cloud via weighted PCA.

    Lag coordinates near the origin are binned, weighted inversely by the
    (normalised) variogram value -- low variogram means strong continuity --
    and an eigen-decomposition of the weighted covariance yields the principal
    axes.

    Returns
    -------
    If ``get_eigens`` is True: ``(eigvals, eigvecs)`` sorted descending.
    If ``dim3d`` is False: ``((azimuth_deg,), axis_lengths)``.
    If ``dim3d`` is True:  ``((azimuth_deg, dip_deg), axis_lengths)``.

    ``azimuth_deg`` is the compass azimuth (0 = North, 90 = East) of the major
    (maximum-continuity) axis and ``dip_deg`` its tilt below the horizontal
    plane, **positive downward** -- the same convention as the model ``dip``
    field and the Fortran engine, so the returned pair can be fed straight into
    :meth:`VariogramModel.set_vgm` as ``azimuth``/``dip``.
    """
    if r_max is None:
        r_max = np.quantile(rawvgm[dist], 0.25)
    df = rawvgm[rawvgm[dist] < r_max].copy()

    if dx is None:
        dx = abs(np.quantile(df[x], 0.9) / 20)
    if dy is None:
        dy = dx
    if dz is None and dim3d:
        dz = abs(np.quantile(df[z], 0.9) / 20)

    df["ix"] = (df[x] // dx).astype(int) * dx
    df["iy"] = (df[y] // dy).astype(int) * dy
    if dim3d:
        df["iz"] = (df[z] // dz).astype(int) * dz

    dims = ["ix", "iy", "iz"] if dim3d else ["ix", "iy"]
    binned = df.groupby(dims, as_index=False)[vgm].mean()

    g = binned[vgm].values
    gmin, gmax = np.nanmin(g), np.nanmax(g)
    if gmax == gmin:
        w = np.ones_like(g)
    else:
        w = np.maximum(1.0 - (g - gmin) / (gmax - gmin), 1e-6)
    if robust:
        w = np.clip(w, *np.quantile(w, [0.05, 0.95]))

    cov = np.cov(binned[dims].values.T, aweights=w)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    if get_eigens:
        return eigvals, eigvecs

    v = _canonical_axis_sign(eigvecs[:, 0], dim3d)
    azimuth = (90 - np.degrees(np.arctan2(v[1], v[0]))) % 360
    if dim3d:
        # tilt below horizontal, positive down (matches the model ``dip`` field)
        dip = np.degrees(-np.arctan2(v[2], np.hypot(v[0], v[1])))
        return (azimuth, dip), 2 * np.sqrt(eigvals)
    return (azimuth,), 2 * np.sqrt(eigvals)


def estimate_angle_angular_profile(rawvgm, angle="angle_h", dist="distance",
                                   r_profile=None, da=10, a_tol=2.5, close=False,
                                   vgm="variogram", dim3d=False):
    """Mean variogram as a function of azimuth, within a near-origin radius.

    Sweeps the azimuth in steps of *da* degrees (overlapping by ``a_tol``) and
    returns the per-sector mean of ``angle``, ``dist`` and ``vgm``.
    """
    if r_profile is None:
        r_profile = np.quantile(rawvgm[dist], 0.25)
    df = rawvgm[rawvgm[dist] <= r_profile].copy()

    profile = []
    for i in np.arange(0, 360, da):
        center = i + da / 2.0
        sel = (df[angle] - center).abs() < da / 2.0 + a_tol
        if sel.any():
            profile.append(df.loc[sel, [angle, dist, vgm]].mean())
    if close and profile:
        profile.append(profile[0])
    return pd.DataFrame(profile)


def _engine_rotation(azimuth=0.0, dip=0.0, plunge=0.0):
    """Return the :class:`scipy.spatial.transform.Rotation` used everywhere.

    Applying it to a data-space lag yields coordinates in the model frame whose
    axes are ``(x = minor1, y = major, z = minor2)``.  ``(azimuth, dip, plunge)``
    are exactly scipy's extrinsic ``"zxy"`` Euler angles, which is also the
    rotation built by the Fortran engine's ``calc_rotmat``
    (``Ry(plunge) * Rx(dip) * Rz(azimuth)``), so the Python preview matches how
    the solver kriges the model.  ``dip`` is positive **down** (below horizontal).
    """
    return Rotation.from_euler("zxy", [azimuth, dip, plunge], degrees=True)


def azimuth_dip_to_vector(azimuth, dip=0.0):
    """Unit vector for an *azimuth* (deg, clockwise from +Y/North) and *dip*
    (deg below horizontal, positive down).  Returns a length-3 array
    ``[x, y, z]``."""
    # The major axis is the data-space direction that maps onto model +Y.
    return _engine_rotation(azimuth, dip, 0.0).as_matrix()[1]


def rotation_matrix_3d(azimuth=0.0, dip=0.0, rake=0.0):
    """Right-handed 3D rotation whose columns are the principal axes.

    The major axis points along ``azimuth``/``dip``; *rake* (the model
    ``plunge``) rotates the minor axes about the major axis.  Useful for
    building anisotropic search/model coordinate frames or directional axis
    sets for :func:`directional_vgm`.

    The axes are taken from the engine-consistent :func:`_engine_rotation`: its
    rows are the data-space principal directions in model-axis order
    ``(minor1, major, minor2)``, which are reordered to ``(major, minor1,
    minor2)`` columns here, flipping ``minor2`` to keep the frame right-handed.
    """
    R = _engine_rotation(azimuth, dip, rake).as_matrix()
    return np.column_stack([R[1], R[0], -R[2]])


def directional_vgm(rawvgm, directions, h_width=None, h_bins=15, bandwidth=None,
                    angle_tol=22.5, dist="distance", vgm="variogram",
                    robust=False, dim=None):
    """Experimental directional variograms along arbitrary (2D/3D) axes.

    For each direction the pairs whose lag vector falls within ``angle_tol`` of
    that axis (and within the perpendicular ``bandwidth``) are selected and
    binned by lag distance -- the multi-dimensional analogue of the gstools
    ``vario_estimate(direction=...)`` workflow.

    Parameters
    ----------
    rawvgm : DataFrame
        Variogram cloud with ``d0..d{dim-1}`` lag components.
    directions : array-like, shape (n_dir, dim)
        Direction vectors (need not be normalised).  Use
        :func:`rotation_matrix_3d` to obtain a rotated orthonormal axis set.
    h_width : float, optional
        Fixed bin width applied to every direction.  When ``None`` (default)
        the width is computed **per direction** as ``max_proj / h_bins``,
        where ``max_proj`` is the largest projected lag along that axis.
        This gives equal resolution across all axes regardless of range, which
        matters for strongly anisotropic data where a single cutoff-derived
        width would leave the short-range axes with very few bins.
    h_bins : int
        Number of bins per direction when ``h_width`` is ``None``.
    bandwidth : float, optional
        Maximum perpendicular distance from the directional line.
    angle_tol : float, optional
        Angular tolerance (degrees) between a lag and the direction.

    Returns
    -------
    DataFrame
        Long format with columns ``direction`` (axis index), ``lag``
        (mean **projected** lag in the bin), ``variogram`` and ``count``.
    """
    directions = np.atleast_2d(np.asarray(directions, dtype=float))
    n_dir, ddim = directions.shape
    if dim is None:
        dim = ddim
    comp_cols = [f"d{k}" for k in range(dim)]
    dh = rawvgm[comp_cols].values
    h = rawvgm[dist].values
    g = rawvgm[vgm].values

    fixed_h_width = h_width  # None → compute per-direction from projected lag
    cos_tol = np.cos(np.deg2rad(angle_tol))

    rows = []
    safe_h = np.where(h == 0, np.nan, h)
    for j in range(n_dir):
        u = directions[j] / np.linalg.norm(directions[j])
        proj_abs = np.abs(dh @ u)                  # projected lag along this axis
        cos_ang = proj_abs / safe_h                # |cos(angle to axis)|
        mask = cos_ang >= cos_tol
        if bandwidth is not None:
            perp = np.sqrt(np.maximum(h ** 2 - proj_abs ** 2, 0.0))
            mask &= perp <= bandwidth
        if not np.any(mask):
            continue
        hh, gg = proj_abs[mask], g[mask]          # project onto axis; calibrates per direction
        if fixed_h_width is None:
            hmax_j = np.nanmax(hh)
            if not np.isfinite(hmax_j) or hmax_j <= 0:
                continue
            this_h_width = hmax_j / h_bins
        else:
            this_h_width = fixed_h_width
        bins = (hh // this_h_width).astype(int)
        sub = pd.DataFrame({"bin": bins, "lag": hh, vgm: gg})
        grouped = sub.groupby("bin")
        agg = grouped.agg(lag=("lag", "mean"), count=(vgm, "count"))
        agg[vgm] = (grouped[vgm].apply(_cressie_gamma) if robust
                    else grouped[vgm].mean())
        agg["direction"] = j
        rows.append(agg.reset_index(drop=True))

    if not rows:
        return pd.DataFrame(columns=["direction", "lag", vgm, "count"])
    return pd.concat(rows, ignore_index=True)[["direction", "lag", vgm, "count"]]


# ---------------------------------------------------------------------------
# 6. Model fitting
# ---------------------------------------------------------------------------
def plot_vgm(avgvgm, x_col=("distance", "mean"), y_col=("variogram", "mean"),
             models=("exponential",), parameters=(1, 100, 0),
             plot_data=True, plot_model=True, annotate=True, ax=None,
             plotkws_data=None, plotkws_model=None,
             xlabel="Lag", ylabel="Semivariogram"):
    """Plot an averaged variogram and/or a fitted model curve."""
    plotkws_data = plotkws_data or dict(color="darkgrey", marker=".", linewidth=0.5)
    plotkws_model = plotkws_model or dict(color="r", linewidth=1.5)
    has_nugget = len(parameters) % 2 != 0
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    x = avgvgm.loc[:, x_col]
    if plot_data:
        ax.plot(x, avgvgm.loc[:, y_col], **plotkws_data)
    if plot_model:
        xhat = np.linspace(0, x.max() * 1.1, 200)
        if _uses_model_template(models):
            yhat = _vgmfunc_from_model_specs(models, xhat, *parameters)
        else:
            yhat = vgmfunc(models, xhat, *parameters)
        ax.plot(xhat, yhat, **plotkws_model)
    if annotate and plot_model:
        specs = _normalise_model_specs(models)
        ms = "Model: " + "\t".join(resolve_model(m["vtype"])[:3].capitalize() for m in specs)
        ss = "\nSill : " + "\t".join(f"{parameters[i * 2]:.5g}" for i in range(len(specs)))
        rr = "\nRange: " + "\t".join(f"{parameters[i * 2 + 1]:.6g}" for i in range(len(specs)))
        nn = f"\nNugget: {parameters[-1]:.5g}" if has_nugget else ""
        ax.text(0.95, 0.05, ms + ss + rr + nn, ha="right", va="bottom",
                transform=ax.transAxes)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    return ax


def _fit_sigma(avgvgm, x_col, sigma_col=None, weight_col=None, weights=None):
    """Return SciPy ``sigma`` from either uncertainty or weight inputs."""
    if sigma_col is not None and (weight_col is not None or weights is not None):
        raise ValueError("pass either sigma_col or weights/weight_col, not both")
    n = len(avgvgm.loc[:, x_col])
    if sigma_col is not None:
        if sigma_col not in avgvgm.columns:
            raise KeyError(f"sigma_col {sigma_col!r} is not present in avgvgm")
        sigma = np.asarray(avgvgm.loc[:, sigma_col], dtype=float)
        if sigma.shape[0] != n:
            raise ValueError("sigma_col length does not match fitted data")
        if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
            raise ValueError("sigma values must be finite and positive")
        return sigma

    if weight_col is not None:
        if weights is not None:
            raise ValueError("pass either weights or weight_col, not both")
        if weight_col not in avgvgm.columns:
            raise KeyError(f"weight_col {weight_col!r} is not present in avgvgm")
        weights = avgvgm.loc[:, weight_col]
    if weights is None:
        return None

    weights = np.asarray(weights, dtype=float).reshape(-1)
    if weights.shape[0] != n:
        raise ValueError("weights length does not match fitted data")
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("weights must be finite and positive")
    return 1.0 / np.sqrt(weights)


def fit_vgm(avgvgm, x_col=("distance", "mean"), y_col=("variogram", "mean"),
            sigma_col=None, weight_col=None, weights=None,
            models=("exponential",), p0=(), makeplot=False,
            maxfev=9999, ax=None, xlabel="Lag", ylabel="Semivariogram",
            bounds=None, return_model=False):
    """Least-squares fit of a (nested) variogram model to averaged data.

    ``models`` may be a sequence of model names, a sequence of
    ``VariogramModel.set_vgm``-style dictionaries, or a :class:`VariogramModel`
    template.  String-only models preserve the legacy additive fitting path.
    Dictionary/model templates also preserve ``product=True`` flags, so product
    structures can be fitted and returned as a Python-side model.

    ``p0`` is the initial guess as ``sill0, range0, ..., [nugget]``.
    ``sigma_col`` gives SciPy-style observation standard deviations.  Use
    ``weights`` or ``weight_col`` for weighted least squares, where larger
    values carry more influence; internally these are converted to
    ``sigma = 1 / sqrt(weight)``.  Returns ``(params, covariance)`` by default;
    when ``return_model=True`` the fitted :class:`VariogramModel` is appended
    to the return tuple.  When ``makeplot`` is true, the Matplotlib axis is
    also appended.
    """
    use_template = _uses_model_template(models)

    def model(h, *p):
        """Curve-fit callback evaluating the requested nested variogram."""
        if use_template:
            return _vgmfunc_from_model_specs(models, h, *p)
        return vgmfunc(models, h, *p)

    sigma = _fit_sigma(avgvgm, x_col, sigma_col=sigma_col,
                       weight_col=weight_col, weights=weights)
    kwargs = dict(p0=p0, sigma=sigma, maxfev=maxfev)
    if bounds is not None:
        kwargs["bounds"] = bounds
    p, cov = curve_fit(model, xdata=avgvgm[x_col], ydata=avgvgm[y_col], **kwargs)

    result = [p, cov]
    if return_model:
        result.append(_model_from_params(models, p))
    if makeplot:
        ax = plot_vgm(avgvgm, x_col, y_col, models, p, ax=ax,
                      xlabel=xlabel, ylabel=ylabel)
        result.append(ax)
    return tuple(result)


# ---------------------------------------------------------------------------
# 7. Plotting -- variogram maps
# ---------------------------------------------------------------------------
def _grid_average(df, xcol, ycol, dx, dy, vgm_col, mirror=True):
    """Bin pairs onto a regular ``(x, y)`` grid and return ``(xx, yy, vv)``.

    When *mirror* is True the cloud is reflected through the origin first so the
    map is point-symmetric (as a variogram map should be).
    """
    rr = df[[xcol, ycol, vgm_col]].copy()
    if mirror:
        mir = rr.copy()
        mir[xcol] *= -1
        mir[ycol] *= -1
        rr = pd.concat([rr, mir], ignore_index=True)

    rr["ix"] = (rr[xcol] // dx).astype(int)
    rr["iy"] = (rr[ycol] // dy).astype(int)
    vmap = rr.groupby(["iy", "ix"], as_index=False)[vgm_col].mean()

    ix0, ix1 = vmap["ix"].min(), vmap["ix"].max()
    iy0, iy1 = vmap["iy"].min(), vmap["iy"].max()
    vv = np.full([iy1 - iy0 + 1, ix1 - ix0 + 1], np.nan)
    vv[vmap.iy - iy0, vmap.ix - ix0] = vmap[vgm_col]

    xx, yy = np.meshgrid(np.arange(ix0, ix1 + 2), np.arange(iy0, iy1 + 2))
    return dx * xx, dy * yy, vv


def plot_vgm_map(rawvgm, x="d0", y="d1", dist="distance", dx=None, dy=None,
                 cutoff=None, ax=None, angle_aniso=None, ellipse_aniso=None,
                 r_profile=None, angle=None, cmap="viridis_r", vmin=None,
                 vmax=None, vgm="variogram",
                 kws_cbar=None, title="Variogram Map", show_cbar=True,
                 show_title=True):
    """2D variogram map (pcolormesh) with optional anisotropy overlays."""
    kws_cbar = kws_cbar or {"label": "Variogram", "pad": 0.01}
    if dx is None:
        dx = np.quantile(rawvgm[x], 0.9) / 20
    if dy is None:
        dy = dx

    r0 = rawvgm.copy()
    if cutoff is not None:
        r0 = r0[r0[dist] <= cutoff].copy()

    rr = r0.copy()
    rr[x] *= -1
    rr[y] *= -1
    if angle:
        rr[angle] = (rr[angle] + 180) % 360
    rr = pd.concat([rr, r0], ignore_index=True)

    xx, yy, vv = _grid_average(r0, x, y, dx, dy, vgm, mirror=True)
    if ax is None:
        nx, ny = vv.shape[1], vv.shape[0]
        _, ax = plt.subplots(figsize=(8, 8 / 1.2 / nx * ny))

    finite = vv[np.isfinite(vv)]
    cm = ax.pcolormesh(xx, yy, vv, cmap=cmap,
                       vmin=vmin if vmin is not None else np.quantile(finite, 0.05),
                       vmax=vmax if vmax is not None else np.quantile(finite, 0.95),
                       zorder=10)
    ax.axline((0, 0), (0, 1), color="w", linewidth=0.5, zorder=11)
    ax.axline((0, 0), (1, 0), color="w", linewidth=0.5, zorder=11)

    if angle_aniso is not None:
        rm = r0[dist].max() * 1.1
        xm = rm * np.sin(np.deg2rad(angle_aniso))
        ym = rm * np.cos(np.deg2rad(angle_aniso))
        ax.plot((xm, -xm), (ym, -ym), color="r", zorder=20)
        extra = f"Angle of maximum continuity: {angle_aniso:.1f}$^o$"
        title = f"{title}\n{extra}" if title else extra
    if ellipse_aniso is not None and angle_aniso is not None:
        ax.add_patch(Ellipse((0, 0), width=ellipse_aniso[0], height=ellipse_aniso[1],
                             angle=90 - angle_aniso, edgecolor="m",
                             facecolor="none", lw=2, zorder=30))
    if angle is not None and r_profile is not None:
        prof = estimate_angle_angular_profile(rr, angle, dist, r_profile,
                                              da=10, a_tol=2.5, close=True)
        scale = prof[dist].mean() / prof[vgm].mean()
        ax.plot(scale * prof[vgm] * np.sin(np.deg2rad(prof[angle])),
                scale * prof[vgm] * np.cos(np.deg2rad(prof[angle])),
                color="m", label="Mean variogram\n(angular profile)", zorder=25)
        ax.legend(loc="upper left").set_zorder(50)
    if show_cbar:
        ax.figure.colorbar(cm, **kws_cbar)
    ax.set(aspect=1.0)
    if cutoff is not None:
        ax.set(xlim=(-cutoff, cutoff), ylim=(-cutoff, cutoff))
    if show_title:
        ax.set_title(title, fontsize="large")
    return ax


def plot_vgm_map_polar(rawvgm, angle="angle_h", dist="distance", da=None, dr=None,
                       cutoff=None, angle_aniso=None, r_profile=None,
                       cmap="viridis", vmin=None, vmax=None, vgm="variogram",
                       kws_cbar=None):
    """Variogram map in polar coordinates (azimuth vs lag distance)."""
    kws_cbar = kws_cbar or {"label": "Variogram", "pad": 0.1}
    if dr is None:
        dr = np.quantile(rawvgm[dist], 0.9) / 20
    if da is None:
        da = 10

    r0 = rawvgm[rawvgm[dist] <= cutoff].copy() if cutoff is not None else rawvgm.copy()
    r1 = r0.copy()
    r1[angle] = (r1[angle] + 180) % 360
    rr = pd.concat([r0, r1], ignore_index=True)

    rr["ia"] = (rr[angle] // da).astype(int)
    rr["ir"] = (rr[dist] // dr).astype(int)
    vmap = rr.groupby(["ir", "ia"], as_index=False)[vgm].mean()

    ia0, ia1 = vmap["ia"].min(), vmap["ia"].max()
    ir0, ir1 = vmap["ir"].min(), vmap["ir"].max()
    vv = np.full([ir1 - ir0 + 1, ia1 - ia0 + 1], np.nan)
    vv[vmap.ir - ir0, vmap.ia - ia0] = vmap[vgm]

    xx, yy = np.meshgrid(np.arange(ia0, ia1 + 2), np.arange(ir0, ir1 + 2))
    xx = np.deg2rad(da * xx)
    yy = dr * yy

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, polar=True)
    pcm = ax.pcolormesh(xx, yy, vv, cmap=cmap, shading="auto",
                        vmin=vmin, vmax=vmax, zorder=0)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.figure.colorbar(pcm, **kws_cbar)

    title = "Variogram Map (Polar, North=0$^o$)"
    if angle_aniso:
        ax.axvline(np.deg2rad(angle_aniso), color="r", zorder=20)
        ax.axvline(np.deg2rad(angle_aniso + 180), color="r", zorder=20)
        title += f"\nAngle of maximum continuity: {angle_aniso:.1f}$^o$"
    if r_profile is not None:
        prof = estimate_angle_angular_profile(rr, angle, dist, r_profile,
                                              da=10, a_tol=2.5, close=True)
        ax.plot(np.deg2rad(prof[angle]), prof[dist], color="m",
                label="Mean variogram\n(angular profile)")
        ax.legend(loc=(-0.07, -0.07))
    ax.set_title(title, fontsize="medium")
    return ax


def plot_vgm_anisotropy3d(eigenvalues, eigenvectors, ax=None):
    """Plot the three principal anisotropy axes as 3D arrows scaled by their
    eigenvalues, annotating the azimuth and dip of the major axis."""
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

    total = np.sqrt((eigenvalues ** 2).sum())
    for k, color in enumerate("rgb"):
        v = eigenvectors[:, k]
        ax.quiver(0, 0, 0, v[0], v[1], v[2], color=color,
                  length=eigenvalues[k] / total, arrow_length_ratio=0.03)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=25, azim=65)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])

    v = _canonical_axis_sign(eigenvectors[:, 0], dim3d=True)
    azimuth = (90 - np.degrees(np.arctan2(v[1], v[0]))) % 360
    # tilt below horizontal, positive down (matches the model ``dip`` field)
    dip = np.degrees(-np.arctan2(v[2], np.hypot(v[0], v[1])))
    ax.set_title(f"Anisotropy angles\nAzimuth: {azimuth:.1f}$^o$; Dip: {dip:.1f}$^o$")
    return ax


def _face_rgba(vv, cmap_obj, norm):
    """RGBA array for plot_surface facecolors; NaN bins are fully transparent."""
    rgba = cmap_obj(norm(np.where(np.isfinite(vv), vv, 0.0)))
    rgba[~np.isfinite(vv), 3] = 0.0
    return rgba


def _fill_nan_nearest(vv, support_mask=None):
    """Replace NaN cells with their nearest valid neighbour (display only).

    When ``support_mask`` is supplied, only NaN cells inside that mask are
    filled.  Cells outside the mask remain NaN, which prevents display-only
    filling from extrapolating beyond the lag-distance support.
    """
    valid = np.isfinite(vv)
    if valid.all() or not valid.any():
        return vv
    if support_mask is None:
        support_mask = np.ones_like(valid, dtype=bool)
    else:
        support_mask = np.asarray(support_mask, dtype=bool)
        if support_mask.shape != valid.shape:
            raise ValueError("support_mask shape must match vv")
    fillable = support_mask & ~valid
    if not fillable.any():
        return vv
    from scipy.ndimage import distance_transform_edt
    _, idx = distance_transform_edt(~valid, return_indices=True)
    out = vv.copy()
    out[fillable] = vv[tuple(idx[:, fillable])]
    return out


def _cell_center_radius(xx, yy, zz):
    """Euclidean radius at each surface cell centre."""
    cx = 0.25 * (xx[:-1, :-1] + xx[1:, :-1] + xx[1:, 1:] + xx[:-1, 1:])
    cy = 0.25 * (yy[:-1, :-1] + yy[1:, :-1] + yy[1:, 1:] + yy[:-1, 1:])
    cz = 0.25 * (zz[:-1, :-1] + zz[1:, :-1] + zz[1:, 1:] + zz[:-1, 1:])
    return np.sqrt(cx * cx + cy * cy + cz * cz)


def _axis_bin(u, dx, dz):
    """Return dz for axes more than 50% vertical, dx otherwise."""
    return dz if abs(float(u[2])) > 0.5 else dx


def _rotated_fence(df, u1, u2, u_normal, d1, d2, bandwidth, vgm_col="variogram"):
    """Bin pairs near the plane spanned by ``u1 × u2`` into a 2-D grid.

    Parameters
    ----------
    u1, u2 : array-like, shape (3,)
        Unit vectors in the fence plane.
    u_normal : array-like, shape (3,)
        Fence normal; pairs with ``|dh @ u_normal| > bandwidth`` are excluded.
    d1, d2 : float
        Bin sizes along u1 and u2.
    bandwidth : float
        Half-thickness slab for pair selection.

    Returns
    -------
    tuple ``(X, Y, Z, vv)`` or None
        3-D mesh coordinates and binned variogram values; None if empty.
    """
    u1 = np.asarray(u1, dtype=float); u1 /= np.linalg.norm(u1)
    u2 = np.asarray(u2, dtype=float); u2 /= np.linalg.norm(u2)
    u_normal = np.asarray(u_normal, dtype=float); u_normal /= np.linalg.norm(u_normal)
    dh = df[["d0", "d1", "d2"]].values
    mask = np.abs(dh @ u_normal) <= bandwidth
    if not mask.any():
        return None
    r1 = (dh @ u1)[mask]
    r2 = (dh @ u2)[mask]
    sub = pd.DataFrame({"_r1": r1, "_r2": r2,
                        vgm_col: df[vgm_col].values[mask]})
    rr, ss, vv = _grid_average(sub, "_r1", "_r2", d1, d2, vgm_col, mirror=True)
    return (rr * u1[0] + ss * u2[0],
            rr * u1[1] + ss * u2[1],
            rr * u1[2] + ss * u2[2],
            vv)


def plot_vgm_map3d(rawvgm, x="d0", y="d1", z="d2", dist="distance",
                   dx=None, dy=None, dz=None, cutoff=None, ax=None,
                   angle_aniso=None, rotate_fences=False,
                   vgm="variogram", cmap="viridis_r", vmin=None, vmax=None,
                   bandwidth_factor=2.0, n_fences=2, fill_nan=False,
                   kws_cbar=None, title="3D Variogram Map",
                   show_cbar=True, show_title=True):
    """3D variogram map shown as orthogonal fence sections.

    Up to three mutually orthogonal fences are drawn.  By default they are
    aligned with the world coordinate axes (XY, XZ, YZ), which makes the
    anisotropy angles easy to read off the plot.  Set ``rotate_fences=True``
    to align the fences with the fitted model axes instead.

    World-axis-aligned fences (``rotate_fences=False``, default):

    * **Fence A** (always) — horizontal XY plane.
    * **Fence B** (``n_fences >= 2``) — vertical XZ (East–West) plane.
    * **Fence C** (``n_fences >= 3``) — vertical YZ (North–South) plane.

    Model-axis-aligned fences (``rotate_fences=True``):

    * **Fence A** — minor1 × minor2 plane (normal = major axis).
    * **Fence B** (``n_fences >= 2``) — major × minor2 plane (dip section).
    * **Fence C** (``n_fences >= 3``) — major × minor1 plane (plunge section).

    Parameters
    ----------
    dx, dy : float, optional
        Reference horizontal bin size.  Defaults to ``cutoff / 10``.  ``dy``
        is accepted for API compatibility but is ignored; use ``dx`` only.
    dz : float, optional
        Reference vertical bin size.  Defaults to the 90th-percentile of
        ``|d2|`` divided by 10, which naturally tracks the (typically finer)
        vertical sampling scale.
    angle_aniso : None, float, or tuple, optional
        Model orientation.  A scalar is interpreted as azimuth (degrees);
        a 2-tuple as ``(azimuth, dip)``; a 3-tuple as
        ``(azimuth, dip, plunge)``.  Only used when ``rotate_fences=True``
        (or to label the title).  ``None`` estimates the horizontal azimuth
        from the cloud with :func:`estimate_aniso_angle`.
    rotate_fences : bool, optional
        If ``False`` (default) fences align with the world X/Y/Z axes.
        If ``True`` fences are rotated to the model anisotropy axes defined
        by ``angle_aniso``.
    bandwidth_factor : float, optional
        Each fence selects pairs within ``bandwidth_factor`` × (normal-axis
        bin size) of the fence plane.  Default 2.
    n_fences : {1, 2, 3}, optional
        Number of orthogonal fences to draw (A only, A+B, or A+B+C).
        Default 2.  Three fences can be informative but crowded.
    fill_nan : bool, optional
        If ``True``, empty bins are filled with their nearest valid neighbour
        before rendering, but only inside the cutoff/max-lag radius.  Useful
        when data are sparse and bins patchy.  Default ``False``.
    """
    kws_cbar = kws_cbar or {"label": "Variogram", "pad": 0.05}

    df = rawvgm[rawvgm[dist] <= cutoff].copy() if cutoff is not None else rawvgm.copy()

    # Horizontal bin size: tied to the search radius.
    if dx is None:
        hmax = cutoff if cutoff is not None else float(rawvgm[dist].quantile(0.9))
        dx = hmax / 10
    # Vertical bin size: tied to the actual vertical lag spread (typically finer).
    if dz is None:
        dz = max(abs(float(df[z].quantile(0.9))), dx * 0.1) / 10
    fill_radius = float(cutoff) if cutoff is not None else float(df[dist].max())

    # --- Parse model orientation ------------------------------------------
    if angle_aniso is None:
        try:
            ang = estimate_aniso_angle(df, x, y, dist, vgm, dim3d=False)[0]
        except Exception:
            ang = None
        azimuth = float(ang[0]) if ang is not None else 0.0
        dip     = float(ang[1]) if ang is not None and len(ang) > 1 else 0.0
        plunge  = 0.0
        _angles_known = ang is not None
    elif np.ndim(angle_aniso) == 0:
        azimuth, dip, plunge = float(angle_aniso), 0.0, 0.0
        _angles_known = True
    else:
        a = tuple(angle_aniso)
        azimuth = float(a[0])
        dip     = float(a[1]) if len(a) > 1 else 0.0
        plunge  = float(a[2]) if len(a) > 2 else 0.0
        _angles_known = True

    # --- Step 1: compute fence grids (no drawing yet) ---------------------
    if rotate_fences:
        # Model-axis-aligned fences — rotated to fitted anisotropy orientation.
        R = rotation_matrix_3d(azimuth, dip, plunge)
        u_major, u_minor1, u_minor2 = R[:, 0], R[:, 1], R[:, 2]
        d_maj  = _axis_bin(u_major,  dx, dz)
        d_min1 = _axis_bin(u_minor1, dx, dz)
        d_min2 = _axis_bin(u_minor2, dx, dz)
        # Fence A: minor1–minor2 plane, normal = major
        fence_A = _rotated_fence(df, u_minor1, u_minor2, u_major,
                                 d_min1, d_min2,
                                 bandwidth=d_maj * bandwidth_factor, vgm_col=vgm)
        # Fence B: major–minor2 plane (dip section), normal = minor1
        fence_B = None
        if n_fences >= 2:
            fence_B = _rotated_fence(df, u_major, u_minor2, u_minor1,
                                     d_maj, d_min2,
                                     bandwidth=d_min1 * bandwidth_factor, vgm_col=vgm)
        # Fence C: major–minor1 plane (plunge section), normal = minor2
        fence_C = None
        if n_fences >= 3:
            fence_C = _rotated_fence(df, u_major, u_minor1, u_minor2,
                                     d_maj, d_min1,
                                     bandwidth=d_min2 * bandwidth_factor, vgm_col=vgm)
    else:
        # World-axis-aligned fences — easy to relate to map coordinates.
        _X = np.array([1., 0., 0.])
        _Y = np.array([0., 1., 0.])
        _Z = np.array([0., 0., 1.])
        # Fence A: horizontal XY plane
        fence_A = _rotated_fence(df, _X, _Y, _Z, dx, dx,
                                 bandwidth=dz * bandwidth_factor, vgm_col=vgm)
        # Fence B: vertical XZ plane (East–West section)
        fence_B = None
        if n_fences >= 2:
            fence_B = _rotated_fence(df, _X, _Z, _Y, dx, dz,
                                     bandwidth=dx * bandwidth_factor, vgm_col=vgm)
        # Fence C: vertical YZ plane (North–South section)
        fence_C = None
        if n_fences >= 3:
            fence_C = _rotated_fence(df, _Y, _Z, _X, dx, dz,
                                     bandwidth=dx * bandwidth_factor, vgm_col=vgm)

    # --- Step 2: colour scale from averaged values, not raw pairs ----------
    cmap_obj = plt.get_cmap(cmap)
    all_avg = np.concatenate([
        s[3][np.isfinite(s[3])].ravel()
        for s in (fence_A, fence_B, fence_C) if s is not None
    ])
    if vmin is None:
        vmin = float(np.nanmin(all_avg)) if len(all_avg) else 0.0
    if vmax is None:
        vmax = float(np.nanquantile(all_avg, 0.95)) if len(all_avg) else 1.0
    norm = Normalize(vmin, vmax)

    # --- Step 3: draw -------------------------------------------------------
    # Merge all fence faces into a single Poly3DCollection so matplotlib
    # depth-sorts all polygons together — fixes z-order artefacts when rotating.
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if ax is None:
        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection="3d")

    alphas = [1.0, 0.85, 0.70]
    all_verts, all_fc = [], []
    all_xx, all_yy, all_zz = [], [], []

    for surf, alpha in zip((fence_A, fence_B, fence_C), alphas):
        if surf is None:
            continue
        xx, yy, zz, vv = surf
        all_xx.append(xx.ravel())
        all_yy.append(yy.ravel())
        all_zz.append(zz.ravel())
        if fill_nan:
            support = _cell_center_radius(xx, yy, zz) <= fill_radius
            vv = _fill_nan_nearest(vv, support_mask=support)
        fc = _face_rgba(vv, cmap_obj, norm)  # (R, C, 4)
        nr, nc = vv.shape
        for i in range(nr):
            for j in range(nc):
                if not np.isfinite(vv[i, j]):
                    continue
                all_verts.append([
                    (xx[i,   j],   yy[i,   j],   zz[i,   j]),
                    (xx[i+1, j],   yy[i+1, j],   zz[i+1, j]),
                    (xx[i+1, j+1], yy[i+1, j+1], zz[i+1, j+1]),
                    (xx[i,   j+1], yy[i,   j+1], zz[i,   j+1]),
                ])
                c = fc[i, j].copy()
                c[3] = alpha
                all_fc.append(c)

    # Poly3DCollection doesn't auto-scale the axes; compute limits from the grids.
    xs = np.concatenate(all_xx) if all_xx else np.array([-dx, dx])
    ys = np.concatenate(all_yy) if all_yy else np.array([-dx, dx])
    zs = np.concatenate(all_zz) if all_zz else np.array([-dz, dz])

    if all_verts:
        coll = Poly3DCollection(all_verts, facecolors=all_fc,
                                edgecolors=all_fc, linewidths=0.3,
                                antialiased=False)
        ax.add_collection3d(coll)
        ax.set_xlim(xs.min(), xs.max())
        ax.set_ylim(ys.min(), ys.max())
        ax.set_zlim(zs.min(), zs.max())

    # --- Anisotropy axis lines projected onto each fence plane --------------
    # Each line lies ON its fence (z=0 for XY, y=0 for XZ, x=0 for YZ) so
    # you can read the azimuth directly off the horizontal fence and the dip
    # off the vertical fences.
    if not rotate_fences and _angles_known:
        R = rotation_matrix_3d(azimuth, dip, plunge)
        u_major, u_minor1, u_minor2 = R[:, 0], R[:, 1], R[:, 2]
        rm = max(float(np.abs(xs).max()), float(np.abs(ys).max()),
                 float(np.abs(zs).max()))

        def _proj_seg(u, drop_idx):
            """Project unit vector u onto the plane normal to axis drop_idx.
            Returns plot (xs, ys, zs) tuple, or None if projection is degenerate."""
            p = np.array(u, dtype=float)
            p[drop_idx] = 0.0
            n = np.linalg.norm(p)
            if n < 0.15:   # axis nearly perpendicular to this fence — skip
                return None
            p = p / n * rm
            return ([-p[0], p[0]], [-p[1], p[1]], [-p[2], p[2]])

        _az_lbl = f"az={azimuth:.0f}°, dip={dip:.0f}°"
        if n_fences >= 3 or abs(plunge) > 0.1:
            _az_lbl += f", pl={plunge:.0f}°"

        shown = set()

        def _draw_proj(u, drop_idx, color, label):
            seg = _proj_seg(u, drop_idx)
            if seg is None:
                return
            kw = dict(color=color, linewidth=1.5, zorder=200)
            if label and label not in shown:
                kw["label"] = label
                shown.add(label)
            ax.plot(*seg, **kw)

        # Project major axis onto each fence plane to show azimuth (XY) and dip (XZ/YZ).
        _draw_proj(u_major, 2, "r", f"major ({_az_lbl})")   # Fence A: XY
        if n_fences >= 2:
            _draw_proj(u_major, 1, "r", f"major ({_az_lbl})")  # Fence B: XZ
        if n_fences >= 3:
            _draw_proj(u_major, 0, "r", f"major ({_az_lbl})")  # Fence C: YZ

        if shown:
            ax.legend(fontsize=8, loc="upper right")

    if show_cbar:
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        mappable.set_array([])
        ax.figure.colorbar(mappable, ax=ax, **kws_cbar)

    ax.set_xlabel("X lag")
    ax.set_ylabel("Y lag")
    ax.set_zlabel("Z lag")
    if show_title and title:
        if rotate_fences:
            extra = f"Azimuth {azimuth:.1f}°  Dip {dip:.1f}°  Plunge {plunge:.1f}°"
            ax.set_title(f"{title}\n{extra}", fontsize="large")
        else:
            ax.set_title(title, fontsize="large")
    return ax


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    hh = np.linspace(0, 1.5, 151)

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, f in _vgmfunc.items():
        ax.plot(hh, f(hh), label=name)
    ax.legend()
    ax.set_title("Normalised variogram models")

    fig, ax = plt.subplots(figsize=(10, 6))
    for name in _vgmfunc:
        ax.plot(hh, calc_cov(name, hh), label=name)
    ax.legend()
    ax.set_title("Normalised covariance models")
    plt.show()
