"""Indicator variogram system: categorical encoding, closure-aware LMC fitting.

An :class:`IndicatorVariogramSystem` is a :class:`VariogramSystem` specialised
for mutually exclusive, exhaustive categorical indicators ``I_1 + ... + I_K = 1``.
It owns the *construction* of indicator variograms in Python -- encoding raw
categories into ``K`` binary variables, building the ``K x K`` coregionalization
from category proportions, and fitting a coregionalization that is both positive
semidefinite and closed (rows summing to zero).  The Fortran ``IndicatorKriging``
wrapper still executes the kriging and post-processes probabilities; this system
configures it via :meth:`apply`.
"""

import numpy as np
from scipy.optimize import least_squares

from .variogram_fitting import FitResult
from .variogram_kernels import calc_vgm
from .variogram_system import VariogramSystem


# ---------------------------------------------------------------------------
# Pure helpers (useful independently of the system object)
# ---------------------------------------------------------------------------
def encode_indicator_matrix(values, categories):
    """Return the ``(nobs, K)`` one-hot indicator matrix for raw ``values``.

    ``categories`` is the ordered list of ``K`` labels; column ``k`` is the
    binary indicator ``values == categories[k]`` as floats.
    """
    values = np.asarray(values)
    cats = list(categories)
    out = np.zeros((values.shape[0], len(cats)), dtype=float)
    for k, label in enumerate(cats):
        out[:, k] = (values == label)
    return out


def indicator_covariance(proportions):
    """Return the closed zero-lag indicator covariance ``diag(p) - p p^T``.

    For a probability vector ``p`` summing to one this is positive semidefinite
    with row sums of zero, giving auto variances ``p_k (1 - p_k)`` and cross
    covariances ``-p_k p_l``.
    """
    p = np.asarray(proportions, dtype=float).reshape(-1)
    return np.diag(p) - np.outer(p, p)


def contrast_basis(ncat):
    """Return an orthonormal ``(K, K-1)`` basis ``Q`` of the contrast space.

    Columns of ``Q`` are orthonormal and orthogonal to the all-ones vector, so
    ``B = Q L L^T Q^T`` is positive semidefinite and closed (``B 1 = 0``) for any
    lower-triangular ``L``.
    """
    if ncat < 2:
        raise ValueError("contrast_basis requires at least 2 categories")
    centering = np.eye(ncat) - np.ones((ncat, ncat)) / ncat
    u, _, _ = np.linalg.svd(centering)
    return u[:, : ncat - 1]


def _psd_factor(matrix):
    """Return a lower Cholesky factor of the nearest PSD matrix to ``matrix``."""
    matrix = 0.5 * (matrix + matrix.T)
    evals, evecs = np.linalg.eigh(matrix)
    evals = np.clip(evals, 0.0, None)
    psd = (evecs * evals) @ evecs.T
    psd = 0.5 * (psd + psd.T)
    jitter = max(np.trace(psd), 1.0) * 1e-12
    return np.linalg.cholesky(psd + np.eye(psd.shape[0]) * jitter)


# ---------------------------------------------------------------------------
class IndicatorVariogramSystem(VariogramSystem):
    """Variogram system for ``K`` mutually exclusive categorical indicators."""

    def __init__(self, categories=None, ncat=None):
        """Create an indicator system from category labels or a count.

        Provide ``categories`` (an ordered label list) or ``ncat``.  The system
        has exactly ``K`` variables (``ivar = 1..K``); secondary co-variates are
        out of scope here -- configure those on the kriging object directly.
        """
        if categories is not None:
            categories = tuple(categories)
            if ncat is not None and ncat != len(categories):
                raise ValueError("ncat must equal len(categories)")
            n = len(categories)
        elif ncat is not None:
            n = int(ncat)
        else:
            raise ValueError("provide categories or ncat")
        if n < 2:
            raise ValueError("indicator systems need at least 2 categories")
        super().__init__(nvar=n)
        self.categories = categories
        self.proportions = None

    @property
    def ncat(self):
        """Number of categories ``K`` (always the system's variable count)."""
        return len(self.categories) if self.categories is not None else self.nvar

    def _new_system(self, nvar):
        """Factory used by LMC fitting to keep the indicator type and metadata."""
        out = IndicatorVariogramSystem(ncat=nvar)
        out.categories = self.categories
        out.proportions = (None if self.proportions is None
                           else np.array(self.proportions, dtype=float))
        return out

    # -- observations -------------------------------------------------------
    def encode_indicators(self, categories):
        """Return the one-hot indicator matrix for ``categories`` using labels."""
        if self.categories is None:
            raise RuntimeError("set categories before encode_indicators()")
        return encode_indicator_matrix(categories, self.categories)

    def set_categorical_obs(self, coord, categories, *, category_labels=None,
                            nmax=32, maxdist=None, variance=None):
        """Encode raw categories into ``K`` indicator datasets and store them.

        ``category_labels`` (or the constructor ``categories``) fixes the label
        order; otherwise sorted unique values are used.  Proportions are
        computed from the encoded indicators.
        """
        cats = np.asarray(categories)
        expected = self.ncat
        if category_labels is not None:
            labels = list(category_labels)
        elif self.categories is not None:
            labels = list(self.categories)
        else:
            labels = sorted(np.unique(cats).tolist())
        if len(labels) != expected:
            raise ValueError(
                f"len(category_labels)={len(labels)} must equal ncat={expected}")
        if self.categories is None:
            self.categories = tuple(labels)
        matrix = encode_indicator_matrix(cats, labels)
        var_arr = None if variance is None else np.asarray(variance, dtype=float)
        for k in range(1, self.ncat + 1):
            self.set_obs(k, coord, matrix[:, k - 1], variance=var_arr,
                        nmax=nmax, maxdist=maxdist)
        self.calc_proportions()
        return self

    def calc_proportions(self):
        """Compute and store category proportions from configured indicators."""
        p = np.empty(self.ncat, dtype=float)
        for k in range(1, self.ncat + 1):
            obs = self.obs.get(k, create=False)
            if obs is None or not obs.configured:
                raise RuntimeError(
                    "configure categorical observations before calc_proportions()")
            p[k - 1] = float(np.mean(obs.value))
        self.proportions = p
        return p

    def initial_indicator_covariance(self):
        """Return the closed zero-lag covariance ``diag(p) - p p^T``."""
        if self.proportions is None:
            self.calc_proportions()
        return indicator_covariance(self.proportions)

    # -- model construction -------------------------------------------------
    def _resolve_proportions(self, proportions):
        """Return a proportion vector from the argument, state, or observations."""
        if proportions is not None:
            p = np.asarray(proportions, dtype=float).reshape(-1)
        elif self.proportions is not None:
            p = np.asarray(self.proportions, dtype=float).reshape(-1)
        else:
            p = self.calc_proportions()
        if len(p) != self.ncat:
            raise ValueError(f"proportions must have length ncat={self.ncat}")
        return p

    def set_indicator_vgm(self, vtype="sph", nugget=0.0, sill=1.0, a_major=1.0,
                          a_minor1=None, a_minor2=None, azimuth=0.0, dip=0.0,
                          plunge=0.0, sill_strategy="theoretical",
                          cross_strategy="closure", proportions=None):
        """Configure all ``K^2`` indicator pairs with one shared model shape.

        ``sill_strategy`` sets the auto (diagonal) partial sills:

        - ``"theoretical"`` -- ``p_k (1 - p_k)`` from category proportions;
        - ``"uniform"`` -- the given ``sill`` for every category.

        ``cross_strategy`` sets the off-diagonal partial sills:

        - ``"closure"`` *(recommended)* -- ``-p_k p_l``, the closed
          indicator covariance off-diagonal;
        - ``"independent"`` -- zero (no co-kriging);
        - ``"proportional"`` -- ``sqrt(s_k s_l)`` from the auto sills;
        - ``"uniform"`` -- the given ``sill``.

        Legacy configurations map directly: old ``cross="same"`` is
        ``sill_strategy="uniform", cross_strategy="uniform"``; old
        ``cross="independent"``/``"proportional"`` pair with
        ``sill_strategy="theoretical"``.
        """
        if sill_strategy not in ("theoretical", "uniform"):
            raise ValueError("sill_strategy must be 'theoretical' or 'uniform'")
        if cross_strategy not in ("closure", "independent", "proportional", "uniform"):
            raise ValueError(
                "cross_strategy must be 'closure', 'independent', "
                "'proportional', or 'uniform'")

        needs_p = sill_strategy == "theoretical" or cross_strategy == "closure"
        p = self._resolve_proportions(proportions) if needs_p else None

        if sill_strategy == "theoretical":
            auto_sills = p * (1.0 - p)
        else:
            auto_sills = np.full(self.ncat, float(sill))

        shape = dict(vtype=vtype, a_major=a_major, a_minor1=a_minor1,
                     a_minor2=a_minor2, azimuth=azimuth, dip=dip, plunge=plunge)
        for iv in range(1, self.ncat + 1):
            for jv in range(iv, self.ncat + 1):
                if iv == jv:
                    pair_sill = float(auto_sills[iv - 1])
                    pair_nugget = float(nugget)
                elif cross_strategy == "independent":
                    pair_sill = 0.0
                    pair_nugget = 0.0
                elif cross_strategy == "closure":
                    pair_sill = float(-p[iv - 1] * p[jv - 1])
                    pair_nugget = float(nugget)
                elif cross_strategy == "proportional":
                    pair_sill = float(np.sqrt(auto_sills[iv - 1] * auto_sills[jv - 1]))
                    pair_nugget = float(nugget)
                else:  # "uniform"
                    pair_sill = float(sill)
                    pair_nugget = float(nugget)
                self.vgm[iv, jv].set_vgm(
                    nugget=pair_nugget, sill=pair_sill, append=False, **shape)
        return self

    # -- closure-aware fitting ---------------------------------------------
    def fit_indicator_lmc(self, pairs=None, shared_ranges=True,
                          x_col=("distance", "mean"), y_col=("variogram", "mean"),
                          sigma_col=None, weight_col=None, fit_nugget=True,
                          inplace=False, raw_kwargs=None, avg_kwargs=None,
                          max_nfev=20000, **kwargs):
        """Fit a closed, PSD linear model of coregionalization for indicators.

        Each nested coregionalization matrix is parameterized as
        ``B = Q L L^T Q^T`` with ``Q`` the contrast basis, guaranteeing positive
        semidefiniteness *and* closure (``B 1 = 0``) at every fitted lag.  Ranges
        are shared across pairs when ``shared_ranges=True``.  Returns a
        :class:`FitResult` whose ``target`` is the fitted indicator system.
        """
        if sigma_col is not None and weight_col is not None:
            raise ValueError("pass either sigma_col or weight_col, not both")
        ncat = self.ncat
        Q = contrast_basis(ncat)
        r = ncat - 1
        tri = np.tril_indices(r)
        ntri = len(tri[0])

        template = self._template_components()
        self._validate_lmc_templates(template)
        pairs = (self._default_lmc_pairs() if pairs is None
                 else [self._pair_key(*p) for p in pairs])
        if not pairs:
            raise RuntimeError("no averaged variograms or observations for fit_indicator_lmc()")

        avg_tables = {}
        for key in pairs:
            avg = self.avg_variograms_.get(key)
            if avg is None:
                avg = self.calc_average(key[0], key[1], raw_kwargs=raw_kwargs,
                                        **(avg_kwargs or {}))
            avg_tables[key] = avg

        init_nugget, init_mats = self._initial_lmc_matrices(template, fit_nugget)
        init_ranges = np.array([comp.a_major for comp in template], dtype=float)

        p0 = []
        if fit_nugget:
            p0.extend(_psd_factor(Q.T @ init_nugget @ Q)[tri])
        for mat in init_mats:
            p0.extend(_psd_factor(Q.T @ mat @ Q)[tri])
        if shared_ranges:
            p0.extend(np.log(init_ranges))
        p0 = np.asarray(p0, dtype=float)

        def unpack(params):
            """Unpack optimizer parameters into closed nugget/sill matrices."""
            offset = 0
            nugget = np.zeros((ncat, ncat), dtype=float)
            if fit_nugget:
                lmat = np.zeros((r, r), dtype=float)
                lmat[tri] = params[offset:offset + ntri]
                nugget = Q @ (lmat @ lmat.T) @ Q.T
                offset += ntri
            mats = []
            for _ in template:
                lmat = np.zeros((r, r), dtype=float)
                lmat[tri] = params[offset:offset + ntri]
                mats.append(Q @ (lmat @ lmat.T) @ Q.T)
                offset += ntri
            ranges = init_ranges.copy()
            if shared_ranges:
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
                    pred = pred + calc_vgm(comp.vtype, h, psill=mats[k][i, j],
                                           rng=ranges[k])
                obs = np.asarray(avg.loc[:, y_col], dtype=float)
                res = pred - obs
                if sigma_col is not None and sigma_col in avg.columns:
                    sigma = np.asarray(avg.loc[:, sigma_col], dtype=float)
                    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
                        raise ValueError("sigma values must be finite and positive")
                    res = res / sigma
                elif weight_col is not None:
                    if weight_col not in avg.columns:
                        raise KeyError(f"weight_col {weight_col!r} is not present in avgvgm")
                    w = np.asarray(avg.loc[:, weight_col], dtype=float)
                    if np.any(~np.isfinite(w)) or np.any(w <= 0.0):
                        raise ValueError("weights must be finite and positive")
                    res = res * np.sqrt(w)
                pieces.append(res)
            return np.concatenate(pieces)

        result = least_squares(residual, p0, max_nfev=max_nfev, **kwargs)
        nugget, mats, ranges = unpack(result.x)
        fitted = self._build_lmc_system(template, nugget, mats, ranges)
        fitted.observations = dict(self.observations)
        fitted.raw_variograms_ = dict(self.raw_variograms_)
        fitted.avg_variograms_ = dict(self.avg_variograms_)
        fitted.fit_result_ = result
        if inplace:
            self._structures = fitted._structures
            self.fit_result_ = result
            fitted = self
        return FitResult(target=fitted, optimizer=result)

    def validate_closure(self, tol=1.0e-6, include_nugget=True):
        """Check that every fitted coregionalization matrix is closed.

        Raises if any matrix (sill structures, plus the nugget when
        ``include_nugget``) has a row sum exceeding ``tol`` in magnitude.
        """
        nugget, mats = self.get_lmc_matrices(include_nugget=True)
        matrices = ([nugget] if include_nugget else []) + list(mats)
        for matrix in matrices:
            worst = float(np.max(np.abs(matrix.sum(axis=1))))
            if worst > tol:
                raise ValueError(
                    f"coregionalization is not closed: max |row sum| = {worst:.3g}")
        return self

    # -- fitting facade and transfer ---------------------------------------
    def fit(self, *, method="closure", **kwargs):
        """Fit the indicator system; ``method="closure"`` is the default.

        ``"closure"`` runs :meth:`fit_indicator_lmc`; ``"lmc"`` and ``"pair"``
        fall through to the unconstrained :class:`VariogramSystem` facade.
        """
        method = str(method).lower()
        if method == "closure":
            return self.fit_indicator_lmc(**kwargs)
        return super().fit(method=method, **kwargs)

    def apply(self, kriging, observations=True, variograms=True, replace=True,
              pairs=None):
        """Transfer indicators and structures, checking ``ncat`` agreement."""
        target_ncat = getattr(kriging, "ncat", None)
        if target_ncat is not None and target_ncat != self.ncat:
            raise ValueError(
                f"target ncat={target_ncat} does not match system ncat={self.ncat}")
        return super().apply(kriging, observations=observations,
                             variograms=variograms, replace=replace, pairs=pairs)

    def __repr__(self):
        """Return a compact debugging representation."""
        return (f"IndicatorVariogramSystem(ncat={self.ncat}, "
                f"npairs={len(self._structures)})")


__all__ = [
    "IndicatorVariogramSystem",
    "encode_indicator_matrix",
    "indicator_covariance",
    "contrast_basis",
]
