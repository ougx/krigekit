"""Composed space-time variogram model and coupling fit workflows."""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize

from .variogram_base import _VariogramModelBase
from .variogram_geometry import calc_anisotropic_lag, calc_lag_vectors
from .variogram_kernels import calc_vgm, resolve_model
from .variogram_model import VariogramModel


class SpaceTimeVariogramModel(_VariogramModelBase):
    """Joint model composed from spatial and temporal marginal models.

    This mirrors the Fortran ``vgm_struct_st`` layout: ``spatial`` corresponds
    to ``cs``, ``temporal`` corresponds to ``ct``, and this object owns only
    full space-time observations, coupling parameters, and transfer metadata.
    """

    def __init__(self, spatial=None, temporal=None):
        """Create a space-time model from optional marginal models."""
        super().__init__()
        self.spatial = spatial if spatial is not None else VariogramModel()
        self.temporal = temporal if temporal is not None else VariogramModel()
        if not isinstance(self.spatial, VariogramModel):
            raise TypeError("spatial must be a VariogramModel")
        if not isinstance(self.temporal, VariogramModel):
            raise TypeError("temporal must be a VariogramModel")
        self.spacetime_params_ = None
        self.spacetime_fit_result_ = None
        self.spacetime_fit_results_ = None
        self.spacetime_spatial_vtype_ = None
        self.spacetime_temporal_vtype_ = None
        self.sum_metric_params_ = None
        self.sum_metric_fit_result_ = None
        self.sum_metric_spatial_model_ = None
        self.sum_metric_temporal_model_ = None
        self.sum_metric_transform_ = None
        self.sum_metric_time_nugget_ = 0.0
        self.sum_metric_time_sill_ = 1.0
        self.spacetime_anisotropy_ = {
            "anis1": 1.0,
            "anis2": 1.0,
            "azimuth": 0.0,
            "dip": 0.0,
            "plunge": 0.0,
        }

    def _clear_fit_state(self):
        """Clear coupling fits while retaining observations and marginals."""
        super()._clear_fit_state()
        self.spacetime_params_ = None
        self.spacetime_fit_result_ = None
        self.spacetime_fit_results_ = None
        self.sum_metric_params_ = None
        self.sum_metric_fit_result_ = None

    def set_vgm(self, *args, **kwargs):
        """Add a structure to the spatial marginal and return ``self``."""
        self.spatial.set_vgm(*args, **kwargs)
        self._clear_fit_state()
        return self

    def set_vgm_temporal(
        self,
        vtype,
        nugget=0.0,
        sill=1.0,
        at_k=1.0,
        product=False,
        **kwargs,
    ):
        """Add a structure to the temporal marginal and return ``self``."""
        self.temporal.set_vgm(
            vtype=vtype,
            nugget=nugget,
            sill=sill,
            a_major=at_k,
            product=product,
            **kwargs,
        )
        self._clear_fit_state()
        return self

    def experimental(self, store=True, **kwargs):
        """Calculate the full space-time cloud with stored spatial anisotropy."""
        if "anisotropy" in kwargs and kwargs["anisotropy"] is not None:
            self.set_spacetime_anisotropy(**kwargs["anisotropy"])
        elif "anisotropy" not in kwargs and self.spacetime_anisotropy_ != {
            "anis1": 1.0,
            "anis2": 1.0,
            "azimuth": 0.0,
            "dip": 0.0,
            "plunge": 0.0,
        }:
            kwargs["anisotropy"] = dict(self.spacetime_anisotropy_)
        return super().experimental(store=store, **kwargs)

    def calc_spacetime_variogram(self, spatial_lag, temporal_lag, params=None):
        """Evaluate the fitted product-sum space-time semivariogram.

        Parameters
        ----------
        spatial_lag, temporal_lag : array-like
            Broadcastable spatial and temporal lag arrays.
        params : array-like, optional
            ``(a, b, p, spatial_range, temporal_range)``. If omitted, use
            ``spacetime_params_`` from :meth:`fit_spacetime_product_sum` or
            :meth:`set_spacetime_params`.
        """
        if params is None:
            params = self.spacetime_params_
        if params is None:
            raise RuntimeError(
                "call fit_spacetime_product_sum() or set_spacetime_params() first"
            )
        if self.spacetime_spatial_vtype_ is None:
            raise RuntimeError("space-time marginal model types are not configured")

        a, b, p, spatial_range, temporal_range = np.asarray(
            params, dtype=float
        ).reshape(-1)
        gs = calc_vgm(
            self.spacetime_spatial_vtype_,
            spatial_lag,
            rng=spatial_range,
        )
        gt = calc_vgm(
            self.spacetime_temporal_vtype_,
            temporal_lag,
            rng=temporal_range,
        )
        return a * gs + b * gt + p * gs * gt

    def calc_spacetime_variogram_between(
        self,
        coord0,
        coord1,
        time0,
        time1,
        *,
        pairwise=False,
        params=None,
    ):
        """Evaluate the space-time variogram between coordinates.

        Spatial lag vectors are rotated and scaled using the anisotropy stored
        by :meth:`set_spacetime_anisotropy`. The resulting distance is the
        equivalent lag along the major axis, in the original coordinate units.
        """
        lag = calc_lag_vectors(coord0, coord1, pairwise=pairwise)
        spatial_lag = calc_anisotropic_lag(lag, **self.spacetime_anisotropy_)
        time0 = np.asarray(time0, dtype=float)
        time1 = np.asarray(time1, dtype=float)
        if pairwise:
            temporal_lag = np.abs(time1.reshape(1, -1) - time0.reshape(-1, 1))
        else:
            temporal_lag = np.abs(time1 - time0)
        result = self.calc_spacetime_variogram(
            spatial_lag,
            temporal_lag,
            params=params,
        )
        return result if pairwise else np.asarray(result).squeeze()

    def set_spacetime_anisotropy(
        self,
        *,
        anis1=1.0,
        anis2=1.0,
        azimuth=0.0,
        dip=0.0,
        plunge=0.0,
    ):
        """Set spatial anisotropy for product-sum fitting and evaluation.

        ``anis1`` and ``anis2`` are minor/major range ratios, matching the
        Fortran engine. Angles use the engine convention: azimuth clockwise
        from north and dip positive downward.
        """
        if anis1 <= 0.0 or anis2 <= 0.0:
            raise ValueError("anis1 and anis2 must be positive")
        self.spacetime_anisotropy_ = {
            "anis1": float(anis1),
            "anis2": float(anis2),
            "azimuth": float(azimuth),
            "dip": float(dip),
            "plunge": float(plunge),
        }
        return self

    def set_spacetime_params(
        self,
        params,
        *,
        spatial_vtype=None,
        temporal_vtype=None,
    ):
        """Manually set product-sum space-time parameters.

        ``params`` is ``(a, b, p, spatial_range, temporal_range)``. Valid
        covariance conversion requires ``p <= 0``, ``a + p > 0`` and
        ``b + p > 0``.
        """
        params = np.asarray(params, dtype=float).reshape(-1)
        if len(params) != 5:
            raise ValueError(
                "params must be (a, b, p, spatial_range, temporal_range)"
            )
        a, b, p, spatial_range, temporal_range = params
        if a <= 0.0 or b <= 0.0:
            raise ValueError("a and b must be positive")
        if p > 0.0 or a + p <= 0.0 or b + p <= 0.0:
            raise ValueError("require p <= 0, a + p > 0 and b + p > 0")
        if spatial_range <= 0.0 or temporal_range <= 0.0:
            raise ValueError("spatial and temporal ranges must be positive")

        if spatial_vtype is not None:
            self.spacetime_spatial_vtype_ = resolve_model(spatial_vtype)
        if temporal_vtype is not None:
            self.spacetime_temporal_vtype_ = resolve_model(temporal_vtype)
        if self.spacetime_spatial_vtype_ is None:
            self.spacetime_spatial_vtype_ = "sph"
        if self.spacetime_temporal_vtype_ is None:
            self.spacetime_temporal_vtype_ = "gau"

        self.spacetime_params_ = params.copy()
        return self

    def fit_spacetime_product_sum(
        self,
        avgvgm=None,
        *,
        spatial_vtype="sph",
        temporal_vtype="gau",
        starts=None,
        bounds=None,
        spatial_col=None,
        temporal_col=("time_lag", "mean"),
        variogram_col=("variogram", "mean"),
        count_col=("variogram", "count"),
        weight_cap_quantile=0.90,
        min_marginal_sill=1.0e-4,
        options=None,
        avg_kwargs=None,
    ):
        """Fit a constrained product-sum model to averaged space-time bins.

        The fitted form is

        ``a*g_s(h_s) + b*g_t(h_t) + p*g_s(h_s)*g_t(h_t)``,

        where both marginal variograms have unit sill. Multiple starting
        points are fitted with SLSQP; the successful result with the smallest
        weighted objective is stored.

        Returns
        -------
        VariogramModel
            ``self`` with ``spacetime_params_``, ``spacetime_fit_result_`` and
            ``spacetime_fit_results_`` populated.
        """
        if avgvgm is None:
            if self._avg is None:
                avgvgm = self.calc_average(**(avg_kwargs or {}))
            else:
                avgvgm = self._avg
        if not isinstance(avgvgm, pd.DataFrame) or len(avgvgm) == 0:
            raise ValueError("avgvgm must be a non-empty pandas DataFrame")

        if spatial_col is None:
            anisotropic_col = ("anisotropic_distance", "mean")
            spatial_col = (
                anisotropic_col
                if anisotropic_col in avgvgm.columns
                else ("distance", "mean")
            )
        for column in (spatial_col, temporal_col, variogram_col, count_col):
            if column not in avgvgm.columns:
                raise KeyError(f"column {column!r} is not present in avgvgm")

        self.spacetime_spatial_vtype_ = resolve_model(spatial_vtype)
        self.spacetime_temporal_vtype_ = resolve_model(temporal_vtype)
        hs = np.asarray(avgvgm[spatial_col], dtype=float)
        ht = np.asarray(avgvgm[temporal_col], dtype=float)
        gamma = np.asarray(avgvgm[variogram_col], dtype=float)
        count = np.asarray(avgvgm[count_col], dtype=float)
        valid = (
            np.isfinite(hs)
            & np.isfinite(ht)
            & np.isfinite(gamma)
            & np.isfinite(count)
            & (count > 0.0)
        )
        hs, ht, gamma, count = hs[valid], ht[valid], gamma[valid], count[valid]
        if len(gamma) == 0:
            raise ValueError("avgvgm contains no finite positive-count bins")

        if weight_cap_quantile is None:
            capped_count = count
        else:
            if not 0.0 < weight_cap_quantile <= 1.0:
                raise ValueError("weight_cap_quantile must be in (0, 1]")
            cap = float(np.quantile(count, weight_cap_quantile))
            capped_count = np.minimum(count, cap)
        weights = capped_count / np.max(capped_count)

        if starts is None:
            starts = [(0.1, 0.05, -0.005, np.nanmax(hs), np.nanmax(ht))]
        starts = [np.asarray(start, dtype=float).reshape(-1) for start in starts]
        if any(len(start) != 5 for start in starts):
            raise ValueError("each start must contain five product-sum parameters")
        if bounds is None:
            bounds = [
                (0.0, np.inf),
                (0.0, np.inf),
                (-np.inf, 0.0),
                (np.finfo(float).eps, np.inf),
                (np.finfo(float).eps, np.inf),
            ]

        def predict(params):
            a, b, p, spatial_range, temporal_range = params
            gs = calc_vgm(self.spacetime_spatial_vtype_, hs, rng=spatial_range)
            gt = calc_vgm(self.spacetime_temporal_vtype_, ht, rng=temporal_range)
            return a * gs + b * gt + p * gs * gt

        def objective(params):
            residual = predict(params) - gamma
            return float(np.sum(weights * residual ** 2))

        constraints = [
            {
                "type": "ineq",
                "fun": lambda x: x[0] + x[2] - min_marginal_sill,
            },
            {
                "type": "ineq",
                "fun": lambda x: x[1] + x[2] - min_marginal_sill,
            },
        ]
        fit_options = {"maxiter": 5000, "ftol": 1.0e-12}
        fit_options.update(options or {})
        results = [
            minimize(
                objective,
                x0=start,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options=fit_options,
            )
            for start in starts
        ]
        successful = [result for result in results if result.success]
        if not successful:
            messages = "; ".join(str(result.message) for result in results)
            raise RuntimeError(f"all product-sum fits failed: {messages}")

        best = min(successful, key=lambda result: result.fun)
        self.spacetime_fit_results_ = results
        self.spacetime_fit_result_ = best
        self.set_spacetime_params(best.x)
        return self

    def calc_spacetime_sum_metric_variogram(
        self,
        spatial_lag,
        temporal_lag,
        params=None,
    ):
        """Evaluate a fitted sum-metric space-time semivariogram.

        ``params`` contains ``spatial_scale, temporal_scale``, one joint sill
        per spatial structure, and the joint temporal scale ``at``.
        """
        if params is None:
            params = self.sum_metric_params_
        if params is None or self.sum_metric_spatial_model_ is None:
            raise RuntimeError("call fit_spacetime_sum_metric() first")

        params = np.asarray(params, dtype=float).reshape(-1)
        spatial_scale, temporal_scale = params[:2]
        joint_sills = params[2:-1]
        at = params[-1]
        hs, ht = np.broadcast_arrays(
            np.asarray(spatial_lag, dtype=float),
            np.asarray(temporal_lag, dtype=float),
        )

        total = (
            spatial_scale * self.sum_metric_spatial_model_.variogram(hs)
            + temporal_scale * self.sum_metric_temporal_model_.variogram(ht)
        )
        dw = calc_vgm(
            self.sum_metric_transform_,
            ht,
            psill=self.sum_metric_time_sill_,
            rng=at,
            nugget=self.sum_metric_time_nugget_,
        )
        dw = np.where(np.abs(ht) <= np.finfo(float).eps, 0.0, dw)
        for sill, comp in zip(
            joint_sills,
            self.sum_metric_spatial_model_.structure.components,
        ):
            if comp.vtype == "nug":
                continue
            hst = np.sqrt((hs / comp.a_major) ** 2 + dw ** 2)
            total = total + sill * calc_vgm(comp.vtype, hst, rng=1.0)
        return total

    def fit_spacetime_sum_metric(
        self,
        spatial_model=None,
        temporal_model=None,
        avgvgm=None,
        *,
        transform="lin",
        time_nugget=0.0,
        time_sill=1.0,
        p0=None,
        bounds=None,
        spatial_col=("distance", "mean"),
        temporal_col=("time_lag", "mean"),
        variogram_col=("variogram", "mean"),
        count_col=("variogram", "count"),
        weight_cap_quantile=0.90,
        max_nfev=20_000,
        avg_kwargs=None,
    ):
        """Fit sum-metric coupling while retaining fitted marginal shapes.

        The fit estimates a spatial marginal scale, temporal marginal scale,
        one non-negative joint sill per spatial structure, and the joint
        temporal scale ``at``. Allowing the two marginal scales to adjust avoids
        forcing separately fitted boundary marginals to explain the entire
        interior space-time lag surface.
        """
        spatial_model = self.spatial if spatial_model is None else spatial_model
        temporal_model = self.temporal if temporal_model is None else temporal_model
        if not isinstance(spatial_model, VariogramModel):
            raise TypeError("spatial_model must be a VariogramModel")
        if not isinstance(temporal_model, VariogramModel):
            raise TypeError("temporal_model must be a VariogramModel")
        if not spatial_model.structure.components or not temporal_model.structure.components:
            raise ValueError("spatial_model and temporal_model must contain structures")
        if avgvgm is None:
            if self._avg is None:
                avgvgm = self.calc_average(**(avg_kwargs or {}))
            else:
                avgvgm = self._avg
        if not isinstance(avgvgm, pd.DataFrame) or len(avgvgm) == 0:
            raise ValueError("avgvgm must be a non-empty pandas DataFrame")
        for column in (spatial_col, temporal_col, variogram_col, count_col):
            if column not in avgvgm.columns:
                raise KeyError(f"column {column!r} is not present in avgvgm")
        if time_sill <= 0.0:
            raise ValueError("time_sill must be positive")

        hs = np.asarray(avgvgm[spatial_col], dtype=float)
        ht = np.asarray(avgvgm[temporal_col], dtype=float)
        gamma = np.asarray(avgvgm[variogram_col], dtype=float)
        count = np.asarray(avgvgm[count_col], dtype=float)
        valid = (
            np.isfinite(hs)
            & np.isfinite(ht)
            & np.isfinite(gamma)
            & np.isfinite(count)
            & (count > 0.0)
        )
        hs, ht, gamma, count = hs[valid], ht[valid], gamma[valid], count[valid]
        if len(gamma) == 0:
            raise ValueError("avgvgm contains no finite positive-count bins")

        if weight_cap_quantile is None:
            capped_count = count
        else:
            if not 0.0 < weight_cap_quantile <= 1.0:
                raise ValueError("weight_cap_quantile must be in (0, 1]")
            capped_count = np.minimum(
                count,
                np.quantile(count, weight_cap_quantile),
            )
        weights = np.sqrt(capped_count / np.max(capped_count))

        nstruct = len(spatial_model.structure.components)
        max_time = max(float(np.nanmax(ht)), 1.0)
        if p0 is None:
            p0 = np.r_[1.0, 1.0, np.full(nstruct, 0.05), max_time]
        p0 = np.asarray(p0, dtype=float).reshape(-1)
        if len(p0) != nstruct + 3:
            raise ValueError(
                "p0 must contain spatial_scale, temporal_scale, one joint "
                "sill per spatial structure, and at"
            )
        if bounds is None:
            bounds = (
                np.r_[0.0, 0.0, np.zeros(nstruct), np.finfo(float).eps],
                np.r_[np.inf, np.inf, np.full(nstruct, np.inf), np.inf],
            )

        self.sum_metric_spatial_model_ = spatial_model
        self.sum_metric_temporal_model_ = temporal_model
        self.sum_metric_transform_ = resolve_model(transform)
        self.sum_metric_time_nugget_ = float(time_nugget)
        self.sum_metric_time_sill_ = float(time_sill)

        result = least_squares(
            lambda params: weights * (
                self.calc_spacetime_sum_metric_variogram(hs, ht, params)
                - gamma
            ),
            x0=p0,
            bounds=bounds,
            max_nfev=max_nfev,
        )
        if not result.success:
            raise RuntimeError(f"sum-metric fit failed: {result.message}")
        self.sum_metric_params_ = result.x.copy()
        self.sum_metric_fit_result_ = result
        return self

    def fit_product_sum(self, *args, **kwargs):
        """Alias for :meth:`fit_spacetime_product_sum`."""
        return self.fit_spacetime_product_sum(*args, **kwargs)

    def fit_sum_metric(self, *args, **kwargs):
        """Alias for :meth:`fit_spacetime_sum_metric`."""
        return self.fit_spacetime_sum_metric(*args, **kwargs)

    def __repr__(self):
        """Return a compact representation of marginal and coupling state."""
        return (
            "SpaceTimeVariogramModel("
            f"spatial={len(self.spatial)}, temporal={len(self.temporal)}, "
            f"coupling_fitted={self.sum_metric_params_ is not None or self.spacetime_params_ is not None}"
            ")"
        )

    def to_sum_metric_kriging_specs(self):
        """Return fitted sum-metric marginal, coupling, and search parameters."""
        if self.sum_metric_params_ is None:
            raise RuntimeError("call fit_spacetime_sum_metric() first")
        spatial_scale, temporal_scale = self.sum_metric_params_[:2]
        joint_sills = self.sum_metric_params_[2:-1]
        at = self.sum_metric_params_[-1]
        spatial_specs = []
        for spec in self.sum_metric_spatial_model_.to_kriging_specs():
            spec = dict(spec)
            spec.pop("append", None)
            if not spec["product"]:
                spec["nugget"] = float(spec["nugget"] * spatial_scale)
                spec["sill"] = float(spec["sill"] * spatial_scale)
            spatial_specs.append(spec)
        temporal_specs = []
        for spec in self.sum_metric_temporal_model_.to_temporal_specs():
            spec = dict(spec)
            if not spec["product"]:
                spec["nugget"] = float(spec["nugget"] * temporal_scale)
                spec["sill"] = float(spec["sill"] * temporal_scale)
            temporal_specs.append(spec)
        return {
            "model": "sum_metric",
            "transform": self.sum_metric_transform_,
            "at": float(at),
            "time_nugget": self.sum_metric_time_nugget_,
            "time_sill": self.sum_metric_time_sill_,
            "spatial_specs": spatial_specs,
            "temporal_specs": temporal_specs,
            "joint_sills": joint_sills.astype(float).tolist(),
            "time_at": float(at),
        }

    def to_spacetime_kriging_specs(
        self,
        *,
        z_scale=None,
        spatial_nugget=0.0,
        temporal_nugget=0.0,
    ):
        """Convert fitted product-sum parameters to engine-ready dictionaries."""
        if self.spacetime_params_ is None:
            raise RuntimeError(
                "call fit_spacetime_product_sum() or set_spacetime_params() first"
            )
        anisotropy = dict(self.spacetime_anisotropy_)
        if z_scale is not None:
            if z_scale <= 0.0:
                raise ValueError("z_scale must be positive")
            anisotropy["anis2"] = 1.0 / float(z_scale)
        a, b, p, spatial_range, temporal_range = self.spacetime_params_
        sill_s = a + p
        sill_t = b + p
        k_ps = -p / (sill_s * sill_t)
        time_at = (spatial_range / temporal_range) * (sill_s / sill_t)
        return {
            "model": "product_sum",
            "k_ps": float(k_ps),
            "spatial_spec": {
                "vtype": self.spacetime_spatial_vtype_,
                "nugget": float(spatial_nugget),
                "sill": float(sill_s),
                "a_major": float(spatial_range),
                "a_minor1": float(spatial_range * anisotropy["anis1"]),
                "a_minor2": float(spatial_range * anisotropy["anis2"]),
                "azimuth": anisotropy["azimuth"],
                "dip": anisotropy["dip"],
                "plunge": anisotropy["plunge"],
            },
            "temporal_spec": {
                "vtype": self.spacetime_temporal_vtype_,
                "nugget": float(temporal_nugget),
                "sill": float(sill_t),
                "at_k": float(temporal_range),
            },
            "time_at": float(time_at),
        }

