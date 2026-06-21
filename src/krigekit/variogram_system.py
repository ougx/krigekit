"""Multivariable variogram systems and LMC fitting."""


import numpy as np
from scipy.optimize import least_squares

from .variogram_empirical import avg_vgm, cross_vgm, raw_cross_vgm, raw_vgm
from .variogram_accessors import _VgmAccessor
from .variogram_component import VgmComponent
from .variogram_kernels import calc_vgm
from .variogram_model import VariogramModel


class VariogramSystem:
    """Multivariable variogram system for cokriging workflows.

    The system stores observations and variogram models by 1-based variable
    pair ``(ivar, jvar)``.  Each pair model is a :class:`VariogramModel`, while
    :meth:`fit_lmc` fits all requested pairs together with positive-semidefinite
    coregionalization matrices.
    """

    _INDEX_HELP = (
        "variables are 1-based: the first variable is vgm[1, 1] / obs[1], not 0"
    )

    def __init__(self, nvar=None):
        """Create an empty multivariable variogram system.

        ``nvar=None`` selects dynamic mode: the variable count grows to the
        largest referenced 1-based index.  An explicit ``nvar`` is a strict
        upper bound and access beyond it raises.
        """
        self.nvar = int(nvar) if nvar is not None else None
        if self.nvar is not None and self.nvar < 1:
            raise ValueError("nvar must be positive")
        self._dynamic = nvar is None
        self.observations = {}
        self.models = {}
        self.raw_variograms_ = {}
        self.avg_variograms_ = {}
        self.fit_result_ = None
        self.vgm = _VgmAccessor(
            key=self._pair_key,
            ensure=self._vgm_ensure,
            peek=self._vgm_peek,
            assign=self._vgm_assign,
            drop=self._vgm_drop,
            materialized=self._vgm_materialized,
        )

    def _check_ivar(self, ivar):
        """Validate a 1-based variable index, growing ``nvar`` in dynamic mode."""
        if isinstance(ivar, bool) or not isinstance(ivar, (int, np.integer)):
            raise TypeError(self._INDEX_HELP)
        ivar = int(ivar)
        if ivar < 1:
            raise ValueError(self._INDEX_HELP)
        if self._dynamic:
            if self.nvar is None or ivar > self.nvar:
                self.nvar = ivar
        elif ivar > self.nvar:
            raise ValueError(f"ivar={ivar} exceeds nvar={self.nvar}: {self._INDEX_HELP}")
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

    # -- storage callbacks used by the ``vgm`` accessor --------------------
    def _vgm_ensure(self, key):
        """Return the structure for a canonical pair, creating it if needed."""
        if key not in self.models:
            self.models[key] = VariogramModel()
        return self.models[key].structure

    def _vgm_peek(self, key):
        """Return the structure for a canonical pair, or ``None``."""
        model = self.models.get(key)
        return None if model is None else model.structure

    def _vgm_assign(self, key, structure):
        """Replace the structure stored for a canonical pair."""
        model = self.models.get(key)
        if model is None:
            model = VariogramModel()
            self.models[key] = model
        model.structure = structure

    def _vgm_drop(self, key):
        """Remove a materialized pair entry."""
        self.models.pop(key, None)

    def _vgm_materialized(self):
        """Return ``(key, structure)`` for every materialized pair."""
        return [(key, model.structure) for key, model in self.models.items()]

    def set_obs(self, ivar, coord, value, times=None):
        """Store observations for variable ``ivar``.

        Parameters mirror :meth:`VariogramModel.set_obs`, with the additional
        1-based variable index used by :class:`krigekit.Kriging`.
        """
        ivar = self._check_ivar(ivar)
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
        self.observations[ivar] = {
            "coord": coord,
            "value": value,
            "times": times,
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
        if any(comp.product for comp in template_model.structure.components):
            raise NotImplementedError("fit_lmc() currently supports additive LMC structures only")
        return [comp.copy() for comp in template_model.structure.components]

    def _validate_lmc_templates(self, template):
        """Check pair models are compatible with the shared LMC template."""
        for key, model in self.models.items():
            if len(model.structure.components) != len(template):
                raise ValueError(f"pair {key} does not match the LMC structure count")
            for comp, tmpl in zip(model.structure.components, template):
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
                for k, comp in enumerate(model.structure.components):
                    mats[k][i - 1, j - 1] = mats[k][j - 1, i - 1] = comp.sill
                if fit_nugget and model.structure.components:
                    nugget[i - 1, j - 1] = nugget[j - 1, i - 1] = model.structure.components[0].nugget
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
                    spec = comp.to_flat_dict()
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

    def set_markov_cross(self, primary, secondary, corr=None,
                         structure="secondary", cross_nugget=0.0):
        """Build the ``(primary, secondary)`` cross variogram by the Markov
        Model 1 (MM1) collocated-cokriging assumption.

        For a sparsely sampled primary and a densely sampled secondary the
        cross-covariance cannot be fit from the primary's (often
        nugget-dominated) structure -- a joint :meth:`fit_lmc` would drive it to
        zero.  MM1 instead transfers it through the collocated correlation: the
        cross adopts the nested structure of one variable
        (``structure="secondary"`` by default -- the dense covariate that
        carries the spatial continuity), and each cross partial sill is

        .. math::  b_{ps}^{(k)} = \\rho \\, \\sqrt{b_{pp}^{(k)} \\, b_{ss}^{(k)}}

        which is positive-semidefinite per structure for ``|rho| <= 1``, so the
        coregionalization is valid by construction (no clamping needed).  This
        is the appropriate model for sparse-hard + dense-soft cokriging
        (Almeida & Journel, 1994; Goovaerts, 1997).  Markov Model 2 is not yet
        implemented.

        Parameters
        ----------
        primary, secondary : int
            1-based indices; both auto-models must already be set via
            :meth:`set_vgm` and share the same nested-structure count.
        corr : float, optional
            Collocated cross-correlation in ``[-1, 1]``.  If ``None`` it is
            estimated from the collocated observations of the two variables
            (which must share coordinates; otherwise pass ``corr`` explicitly).
        structure : {"secondary", "primary"}
            Which variable's structure shapes/ranges the cross adopts.
        cross_nugget : float
            Cross nugget partial sill (default 0).
        """
        if structure not in ("secondary", "primary"):
            raise ValueError("structure must be 'secondary' or 'primary'")
        pi = self._check_ivar(primary)
        si = self._check_ivar(secondary)
        if pi == si:
            raise ValueError("primary and secondary must be different variables")
        model_p = self._get_model(pi, pi, create=False)
        model_s = self._get_model(si, si, create=False)
        s_p = [comp.sill for comp in model_p.structure.components]
        s_s = [comp.sill for comp in model_s.structure.components]
        if not s_p or not s_s:
            raise RuntimeError("set auto-models for both variables before set_markov_cross()")
        if len(s_p) != len(s_s):
            raise ValueError(
                "primary and secondary auto-models must share the same number of "
                f"nested structures (got {len(s_p)} and {len(s_s)})")

        if corr is None:
            obs_p = self._require_obs(pi)
            obs_s = self._require_obs(si)
            if not self._same_obs_grid(obs_p, obs_s):
                raise ValueError(
                    "cannot estimate corr: the two variables are not collocated; "
                    "pass corr= explicitly")
            corr = float(np.corrcoef(obs_p["value"], obs_s["value"])[0, 1])
        corr = float(np.clip(corr, -1.0, 1.0))

        base = model_s if structure == "secondary" else model_p
        cross = VariogramModel()
        for k, comp in enumerate(base.structure.components):
            cross.set_vgm(
                vtype=comp.vtype,
                nugget=cross_nugget if k == 0 else 0.0,
                sill=corr * float(np.sqrt(s_p[k] * s_s[k])),
                a_major=comp.a_major,
                a_minor1=comp.a_minor1,
                a_minor2=comp.a_minor2,
                azimuth=comp.azimuth,
                dip=comp.dip,
                plunge=comp.plunge,
                product=comp.product,
                append=k > 0,
            )
        self.models[self._pair_key(pi, si)] = cross
        self.markov_corr_ = getattr(self, "markov_corr_", {})
        self.markov_corr_[self._pair_key(pi, si)] = corr
        return self

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
