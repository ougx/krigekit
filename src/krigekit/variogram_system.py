"""Multivariable variogram systems and LMC fitting."""


import numpy as np
from scipy.optimize import least_squares

from .variogram_empirical import avg_vgm, cross_vgm, raw_cross_vgm, raw_vgm
from .variogram_accessors import _ObservationAccessor, _VgmAccessor
from .variogram_component import VgmComponent
from .variogram_fitting import FitResult
from .variogram_kernels import calc_vgm
from .variogram_observation import ObservationSet
from .variogram_structure import VgmStructure


class VariogramSystem:
    """Multivariable variogram system for cokriging workflows.

    The system stores observations and variogram structures by 1-based variable
    pair ``(ivar, jvar)``.  Each pair is a :class:`VgmStructure`, while
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
        self._structures = {}
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
        self.obs = _ObservationAccessor(
            index=self._check_ivar,
            ensure=self._obs_ensure,
            peek=self._obs_peek,
            drop=self._obs_drop,
            materialized=self._obs_materialized,
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

    # -- storage callbacks used by the ``vgm`` accessor --------------------
    def _vgm_ensure(self, key):
        """Return the structure for a canonical pair, creating it if needed."""
        if key not in self._structures:
            self._structures[key] = VgmStructure()
        return self._structures[key]

    def _vgm_peek(self, key):
        """Return the structure for a canonical pair, or ``None``."""
        return self._structures.get(key)

    def _vgm_assign(self, key, structure):
        """Replace the structure stored for a canonical pair."""
        self._structures[key] = structure

    def _vgm_drop(self, key):
        """Remove a materialized pair entry."""
        self._structures.pop(key, None)

    def _vgm_materialized(self):
        """Return ``(key, structure)`` for every materialized pair."""
        return list(self._structures.items())

    # -- storage callbacks used by the ``obs`` accessor --------------------
    def _obs_ensure(self, ivar):
        """Return the observation set for a variable, creating it if needed."""
        if ivar not in self.observations:
            self.observations[ivar] = ObservationSet()
        return self.observations[ivar]

    def _obs_peek(self, ivar):
        """Return the observation set for a variable, or ``None``."""
        return self.observations.get(ivar)

    def _obs_drop(self, ivar):
        """Remove a materialized observation entry."""
        self.observations.pop(ivar, None)

    def _obs_materialized(self):
        """Return ``(ivar, set)`` for every materialized observation."""
        return list(self.observations.items())

    def _invalidate_pair_caches(self, ivar):
        """Drop cached raw/averaged variograms for any pair containing ``ivar``."""
        for key in list(self.raw_variograms_):
            if ivar in key:
                self.raw_variograms_.pop(key, None)
                self.avg_variograms_.pop(key, None)

    def set_obs(self, ivar, coord, value, times=None, variance=None,
                nmax=None, maxdist=None, sk_mean=None, drift=None):
        """Store observations for variable ``ivar`` (ergonomic wrapper).

        Delegates to ``self.obs[ivar].set(...)`` and invalidates cached
        empirical variograms for any pair containing ``ivar``.
        """
        ivar = self._check_ivar(ivar)
        self.obs[ivar].set(coord, value, times=times, variance=variance,
                           nmax=nmax, maxdist=maxdist, sk_mean=sk_mean,
                           drift=drift)
        self._invalidate_pair_caches(ivar)
        return self

    def set_search(self, ivar, *, nmax=None, maxdist=None, anis1=1.0, anis2=1.0,
                   azimuth=0.0, dip=0.0, plunge=0.0, sector_search=False,
                   time_at=None):
        """Configure per-variable neighbour search, transferred by ``apply()``.

        Search settings are stored on the variable's :class:`ObservationSet`;
        the Fortran engine performs the actual neighbour selection.
        """
        ivar = self._check_ivar(ivar)
        self.obs[ivar].set_search(nmax=nmax, maxdist=maxdist, anis1=anis1,
                                 anis2=anis2, azimuth=azimuth, dip=dip,
                                 plunge=plunge, sector_search=sector_search,
                                 time_at=time_at)
        return self

    def set_vgm(self, ivar, jvar, vtype, **kwargs):
        """Add one nested structure to the model for ``(ivar, jvar)``."""
        self.vgm[ivar, jvar].set_vgm(vtype=vtype, **kwargs)
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
        """Return the configured observation set for a variable, or raise."""
        ivar = self._check_ivar(ivar)
        obs = self.obs.get(ivar, create=False)
        if obs is None or not obs.configured:
            raise RuntimeError(f"call set_obs(ivar={ivar}, ...) first")
        return obs

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
            kwargs.setdefault("times", obs_i.times)
            cloud = raw_vgm(obs_i.coord, obs_i.value, **kwargs)
        else:
            if cross != "pseudo" and obs_i.is_collocated_with(obs_j):
                kwargs.setdefault("times", obs_i.times)
                cloud = raw_cross_vgm(
                    obs_i.coord,
                    obs_i.value,
                    obs_j.value,
                    **kwargs,
                )
            elif cross == "lmc":
                raise ValueError(
                    "LMC cross-variogram fitting requires collocated "
                    "coordinates and matching times for the variable pair"
                )
            else:
                kwargs.setdefault("timesA", obs_i.times)
                kwargs.setdefault("timesB", obs_j.times)
                cloud = cross_vgm(
                    obs_i.coord,
                    obs_i.value,
                    obs_j.coord,
                    obs_j.value,
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
                configured = set(self.obs.configured_indices())
                keys = [(i, j) for i in range(1, nvar + 1)
                        for j in range(i, nvar + 1)
                        if i in configured and j in configured]
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
        structure = self.vgm.get(key[0], key[1], create=False)
        if structure is None or not structure.ncomponent:
            raise KeyError(f"no variogram model has been set for pair {key}")
        if avgvgm is None:
            avgvgm = self.avg_variograms_.get(key)
        if avgvgm is None:
            avgvgm = self.calc_average(*key)
        return structure.fit(avgvgm, inplace=inplace, **kwargs)

    def fit(self, *, method="lmc", ivar=None, jvar=None, pairs=None, **kwargs):
        """Fit the system, returning a :class:`FitResult` (method facade).

        ``method="lmc"`` (default) fits a joint linear model of coregionalization
        via :meth:`fit_lmc`; ``method="pair"`` fits one variable pair
        independently via :meth:`fit_pair` and requires ``ivar`` (with optional
        ``jvar``).  Subclasses extend this facade -- ``IndicatorVariogramSystem``
        adds ``method="closure"``.
        """
        method = str(method).lower()
        if method == "lmc":
            return self.fit_lmc(pairs=pairs, **kwargs)
        if method == "pair":
            if ivar is None:
                raise ValueError("method='pair' requires ivar (and optional jvar)")
            return self.fit_pair(ivar, jvar, **kwargs)
        raise ValueError("method must be 'lmc' or 'pair'")

    def _template_components(self):
        """Return the shared LMC structure template from configured pairs."""
        configured = self.vgm.configured_items()
        if not configured:
            raise RuntimeError("call set_vgm() before fit_lmc()")
        preferred = self.vgm.get(1, 1, create=False)
        template = preferred if (preferred is not None and preferred.ncomponent) else None
        if template is None:
            template = next((s for _, s in configured), None)
        if any(comp.product for comp in template.components):
            raise NotImplementedError("fit_lmc() currently supports additive LMC structures only")
        return [comp.copy() for comp in template.components]

    def _validate_lmc_templates(self, template):
        """Check configured pairs are compatible with the shared LMC template."""
        for key, structure in self.vgm.configured_items():
            if len(structure.components) != len(template):
                raise ValueError(f"pair {key} does not match the LMC structure count")
            for comp, tmpl in zip(structure.components, template):
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
        """Build initial nugget and partial-sill matrices from configured pairs."""
        nvar = self.nvar or max((max(k) for k in self._structures), default=0)
        mats = [np.zeros((nvar, nvar), dtype=float) for _ in template]
        nugget = np.zeros((nvar, nvar), dtype=float)
        for (i, j), structure in self.vgm.configured_items():
            for k, comp in enumerate(structure.components):
                mats[k][i - 1, j - 1] = mats[k][j - 1, i - 1] = comp.sill
            if fit_nugget and structure.components:
                nugget[i - 1, j - 1] = nugget[j - 1, i - 1] = structure.components[0].nugget
        return nugget, mats

    def _default_lmc_pairs(self):
        """Return variable pairs to include in LMC fitting."""
        if self.avg_variograms_:
            return sorted(self.avg_variograms_)
        nvar = self.nvar or 0
        configured = set(self.obs.configured_indices())
        return [(i, j) for i in range(1, nvar + 1)
                for j in range(i, nvar + 1)
                if i in configured and j in configured]

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
            self._structures = fitted._structures
            self.fit_result_ = result
            fitted = self
        return FitResult(target=fitted, optimizer=result)

    def _build_lmc_system(self, template, nugget, mats, ranges):
        """Build a fitted :class:`VariogramSystem` from LMC matrices."""
        nvar = self.nvar or nugget.shape[0]
        fitted = VariogramSystem(nvar=nvar)
        for i in range(1, nvar + 1):
            for j in range(i, nvar + 1):
                structure = fitted.vgm[i, j]
                for k, comp in enumerate(template):
                    spec = comp.to_flat_dict()
                    ratio = ranges[k] / comp.a_major
                    spec["sill"] = mats[k][i - 1, j - 1]
                    spec["a_major"] = ranges[k]
                    spec["a_minor1"] = comp.a_minor1 * ratio
                    spec["a_minor2"] = comp.a_minor2 * ratio
                    spec["nugget"] = nugget[i - 1, j - 1] if k == 0 else 0.0
                    spec["append"] = k > 0
                    structure.set_vgm(**spec)
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
        structure_p = self.vgm.get(pi, pi, create=False)
        structure_s = self.vgm.get(si, si, create=False)
        s_p = [comp.sill for comp in structure_p.components] if structure_p else []
        s_s = [comp.sill for comp in structure_s.components] if structure_s else []
        if not s_p or not s_s:
            raise RuntimeError("set auto-models for both variables before set_markov_cross()")
        if len(s_p) != len(s_s):
            raise ValueError(
                "primary and secondary auto-models must share the same number of "
                f"nested structures (got {len(s_p)} and {len(s_s)})")

        if corr is None:
            obs_p = self._require_obs(pi)
            obs_s = self._require_obs(si)
            if not obs_p.is_collocated_with(obs_s):
                raise ValueError(
                    "cannot estimate corr: the two variables are not collocated; "
                    "pass corr= explicitly")
            corr = float(np.corrcoef(obs_p.value, obs_s.value)[0, 1])
        corr = float(np.clip(corr, -1.0, 1.0))

        base = structure_s if structure == "secondary" else structure_p
        cross = VgmStructure()
        for k, comp in enumerate(base.components):
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
        self.vgm[pi, si] = cross
        self.markov_corr_ = getattr(self, "markov_corr_", {})
        self.markov_corr_[self._pair_key(pi, si)] = corr
        return self

    def _selected_pairs(self, pairs):
        """Return configured ``(key, structure)`` items, optionally filtered."""
        items = self.vgm.configured_items()
        if pairs is None:
            return items
        keys = {self._pair_key(*p) for p in pairs}
        return [(k, s) for k, s in items if k in keys]

    def validate_for(self, kriging, observations=True, variograms=True, pairs=None):
        """Check that this system can be applied to ``kriging`` without leaving
        it partially configured.  Raises on the first problem and mutates
        nothing; this is the validate-first half of :meth:`apply`.
        """
        nvar, ndim = kriging.nvar, kriging.ndim
        if self.nvar is not None and self.nvar > nvar:
            raise ValueError(f"system nvar={self.nvar} exceeds target nvar={nvar}")

        if observations:
            for ivar, obs in self.obs.configured_items():
                if ivar > nvar:
                    raise ValueError(f"observation variable {ivar} exceeds target nvar={nvar}")
                obs.validate(ndim=ndim)
                if obs.times is not None:
                    raise ValueError(
                        "space-time observations require a space-time system "
                        "(not yet supported by apply())")
                if obs.ndrift and obs.ndrift != kriging.ndrift:
                    raise ValueError(
                        f"variable {ivar} has {obs.ndrift} drift column(s) but "
                        f"target ndrift={kriging.ndrift}")

        if variograms:
            all_pairs = set(self.vgm.configured_pairs())
            for (i, j), structure in self._selected_pairs(pairs):
                if i > nvar or j > nvar:
                    raise ValueError(f"variogram pair {(i, j)} exceeds target nvar={nvar}")
                structure.validate()
                if i != j:
                    missing = [a for a in ((i, i), (j, j)) if a not in all_pairs]
                    if missing:
                        raise ValueError(
                            f"cross pair {(i, j)} requires auto-variogram(s) {missing}")
        return self

    def apply_observations(self, kriging):
        """Transfer configured observations (and drift) in ascending order."""
        for ivar, obs in self.obs.configured_items():
            obs.apply_to(kriging, ivar)
        return kriging

    def apply_variograms(self, kriging, replace=True, pairs=None):
        """Transfer configured pair structures in canonical upper-triangle order."""
        for (i, j), structure in self._selected_pairs(pairs):
            structure.apply_to(kriging, i, j, replace=replace)
        return kriging

    def apply(self, kriging, observations=True, variograms=True, replace=True,
              pairs=None):
        """Validate, then transfer observations and/or variograms to ``kriging``.

        Validation (:meth:`validate_for`) completes before any mutation, so a
        failure leaves ``kriging`` unchanged.  Observations are applied in
        ascending variable order (drift immediately after each base
        observation), then configured variograms in canonical upper-triangle
        order.
        """
        self.validate_for(kriging, observations=observations,
                           variograms=variograms, pairs=pairs)
        if observations:
            self.apply_observations(kriging)
        if variograms:
            self.apply_variograms(kriging, replace=replace, pairs=pairs)
        return kriging

    def __repr__(self):
        """Return a compact debugging representation."""
        return f"VariogramSystem(nvar={self.nvar}, npairs={len(self._structures)})"


# ---------------------------------------------------------------------------
# 2. Distance helpers
# ---------------------------------------------------------------------------
