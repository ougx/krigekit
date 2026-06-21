"""The complete theoretical space-time variogram for one variable pair.

A :class:`VgmStructureST` is the space-time analogue of :class:`VgmStructure`:
it owns the *theoretical* coupling model and nothing else -- no observations,
empirical clouds, or fit state.  It mirrors the Fortran ``vgm_struct_st`` layout,
holding a spatial structure (``cs``) and a temporal structure (``ct``) plus the
coupling parameters of either a product-sum or a sum-metric model.  The analysis
wrapper :class:`krigekit.SpaceTimeVariogramModel` fits empirical clouds and
writes the result into one of these objects.
"""

import numpy as np

from .variogram_kernels import calc_vgm, resolve_model
from .variogram_structure import VgmStructure

_ISOTROPIC = {"anis1": 1.0, "anis2": 1.0, "azimuth": 0.0, "dip": 0.0, "plunge": 0.0}


class VgmStructureST:
    """Theoretical space-time variogram: spatial/temporal structures + coupling.

    The active coupling form is named by :attr:`model` (``"product_sum"`` or
    ``"sum_metric"``).  Use :meth:`set_product_sum` or :meth:`set_sum_metric`
    to populate it, then :meth:`calc_variogram` to evaluate and
    :meth:`to_kriging_specs` to emit engine-ready dictionaries.
    """

    def __init__(self, spatial=None, temporal=None, *, name=None):
        """Create an empty structure or wrap existing marginal structures."""
        self.name = None if name is None else str(name)
        self.spatial = spatial if spatial is not None else VgmStructure()
        self.temporal = temporal if temporal is not None else VgmStructure()
        if not isinstance(self.spatial, VgmStructure):
            raise TypeError("spatial must be a VgmStructure")
        if not isinstance(self.temporal, VgmStructure):
            raise TypeError("temporal must be a VgmStructure")
        self.model = None
        # product-sum coupling
        self.ps_params = None            # (a, b, p, spatial_range, temporal_range)
        self.ps_spatial_vtype = None
        self.ps_temporal_vtype = None
        self.anisotropy = dict(_ISOTROPIC)
        # sum-metric coupling
        self.sm_params = None            # (spatial_scale, temporal_scale, *joint_sills, at)
        self.transform = None
        self.time_nugget = 0.0
        self.time_sill = 1.0

    # -- container protocol -------------------------------------------------
    @property
    def cs(self):
        """Alias for the spatial structure, matching ``vgm_struct_st%cs``."""
        return self.spatial

    @property
    def ct(self):
        """Alias for the temporal structure, matching ``vgm_struct_st%ct``."""
        return self.temporal

    @property
    def ncomponent_spatial(self):
        """Number of nested spatial components."""
        return self.spatial.ncomponent

    @property
    def ncomponent_temporal(self):
        """Number of nested temporal components."""
        return self.temporal.ncomponent

    def __repr__(self):
        """Return a compact debugging representation."""
        label = "" if self.name is None else f"name={self.name!r}, "
        return (f"VgmStructureST({label}model={self.model!r}, "
                f"cs={self.ncomponent_spatial}, ct={self.ncomponent_temporal})")

    def copy(self):
        """Return an independent copy of this structure and its parameters."""
        out = VgmStructureST(self.spatial.copy(), self.temporal.copy(),
                             name=self.name)
        out.model = self.model
        out.ps_params = None if self.ps_params is None else np.array(self.ps_params, dtype=float)
        out.ps_spatial_vtype = self.ps_spatial_vtype
        out.ps_temporal_vtype = self.ps_temporal_vtype
        out.anisotropy = dict(self.anisotropy)
        out.sm_params = None if self.sm_params is None else np.array(self.sm_params, dtype=float)
        out.transform = self.transform
        out.time_nugget = self.time_nugget
        out.time_sill = self.time_sill
        return out

    # -- configuration ------------------------------------------------------
    def set_product_sum(self, params, *, spatial_vtype, temporal_vtype,
                        anisotropy=None):
        """Populate a product-sum coupling and rebuild ``cs``/``ct``.

        ``params`` is ``(a, b, p, spatial_range, temporal_range)`` with unit-sill
        marginals; a valid covariance conversion requires ``p <= 0``,
        ``a + p > 0`` and ``b + p > 0``.  ``cs``/``ct`` are rebuilt as
        single-component structures whose sills are ``a + p`` and ``b + p``.
        """
        params = np.asarray(params, dtype=float).reshape(-1)
        if len(params) != 5:
            raise ValueError("params must be (a, b, p, spatial_range, temporal_range)")
        if anisotropy is not None:
            self.anisotropy = {**_ISOTROPIC, **anisotropy}
        self.model = "product_sum"
        self.ps_params = params
        self.ps_spatial_vtype = resolve_model(spatial_vtype)
        self.ps_temporal_vtype = resolve_model(temporal_vtype)
        self._rebuild_product_sum_marginals()
        return self

    def _rebuild_product_sum_marginals(self):
        """Refresh ``cs``/``ct`` single components from product-sum parameters."""
        a, b, p, spatial_range, temporal_range = self.ps_params
        self.spatial = VgmStructure([{
            "vtype": self.ps_spatial_vtype, "sill": float(a + p),
            "a_major": float(spatial_range),
            "a_minor1": float(spatial_range * self.anisotropy["anis1"]),
            "a_minor2": float(spatial_range * self.anisotropy["anis2"]),
            "azimuth": self.anisotropy["azimuth"], "dip": self.anisotropy["dip"],
            "plunge": self.anisotropy["plunge"],
        }])
        self.temporal = VgmStructure([{
            "vtype": self.ps_temporal_vtype, "sill": float(b + p),
            "a_major": float(temporal_range),
        }])

    def set_sum_metric(self, spatial, temporal, params, *, transform,
                       time_nugget=0.0, time_sill=1.0):
        """Populate a sum-metric coupling from fitted marginal structures.

        ``spatial`` and ``temporal`` are :class:`VgmStructure` marginal shapes,
        and ``params`` is ``(spatial_scale, temporal_scale, *joint_sills, at)``
        with one joint sill per spatial component.
        """
        if not isinstance(spatial, VgmStructure) or not isinstance(temporal, VgmStructure):
            raise TypeError("spatial and temporal must be VgmStructure objects")
        params = np.asarray(params, dtype=float).reshape(-1)
        if len(params) != spatial.ncomponent + 3:
            raise ValueError(
                "params must contain spatial_scale, temporal_scale, one joint "
                "sill per spatial component, and at")
        if time_sill <= 0.0:
            raise ValueError("time_sill must be positive")
        self.model = "sum_metric"
        self.spatial = spatial
        self.temporal = temporal
        self.sm_params = params
        self.transform = resolve_model(transform)
        self.time_nugget = float(time_nugget)
        self.time_sill = float(time_sill)
        return self

    def validate(self):
        """Validate the active coupling parameters and return ``self``."""
        if self.model == "product_sum":
            a, b, p, sr, tr = self.ps_params
            if a <= 0.0 or b <= 0.0:
                raise ValueError("a and b must be positive")
            if p > 0.0 or a + p <= 0.0 or b + p <= 0.0:
                raise ValueError("require p <= 0, a + p > 0 and b + p > 0")
            if sr <= 0.0 or tr <= 0.0:
                raise ValueError("spatial and temporal ranges must be positive")
        elif self.model == "sum_metric":
            self.spatial.validate()
            self.temporal.validate()
            if self.sm_params[-1] <= 0.0:
                raise ValueError("the joint temporal scale 'at' must be positive")
        else:
            raise RuntimeError("call set_product_sum() or set_sum_metric() first")
        return self

    # -- theoretical evaluation --------------------------------------------
    def calc_variogram(self, spatial_lag, temporal_lag):
        """Evaluate the active space-time semivariogram at lag arrays."""
        if self.model == "product_sum":
            return self._calc_product_sum(spatial_lag, temporal_lag)
        if self.model == "sum_metric":
            return self._calc_sum_metric(spatial_lag, temporal_lag)
        raise RuntimeError("call set_product_sum() or set_sum_metric() first")

    def _calc_product_sum(self, spatial_lag, temporal_lag):
        """Evaluate ``a*g_s + b*g_t + p*g_s*g_t`` from unit-sill marginals."""
        a, b, p, spatial_range, temporal_range = self.ps_params
        gs = calc_vgm(self.ps_spatial_vtype, spatial_lag, rng=spatial_range)
        gt = calc_vgm(self.ps_temporal_vtype, temporal_lag, rng=temporal_range)
        return a * gs + b * gt + p * gs * gt

    def _calc_sum_metric(self, spatial_lag, temporal_lag):
        """Evaluate the sum-metric surface from marginal shapes and coupling."""
        params = np.asarray(self.sm_params, dtype=float).reshape(-1)
        spatial_scale, temporal_scale = params[:2]
        joint_sills = params[2:-1]
        at = params[-1]
        hs, ht = np.broadcast_arrays(
            np.asarray(spatial_lag, dtype=float),
            np.asarray(temporal_lag, dtype=float),
        )
        total = (spatial_scale * self.spatial.variogram(hs)
                 + temporal_scale * self.temporal.variogram(ht))
        dw = calc_vgm(self.transform, ht, psill=self.time_sill, rng=at,
                      nugget=self.time_nugget)
        dw = np.where(np.abs(ht) <= np.finfo(float).eps, 0.0, dw)
        for sill, comp in zip(joint_sills, self.spatial.components):
            if comp.vtype == "nug":
                continue
            hst = np.sqrt((hs / comp.a_major) ** 2 + dw ** 2)
            total = total + sill * calc_vgm(comp.vtype, hst, rng=1.0)
        return total

    # -- engine transfer ----------------------------------------------------
    def to_kriging_specs(self, **kwargs):
        """Return engine-ready specs for the active coupling model."""
        if self.model == "product_sum":
            return self._product_sum_specs(**kwargs)
        if self.model == "sum_metric":
            if kwargs:
                raise TypeError("sum-metric specs take no keyword arguments")
            return self._sum_metric_specs()
        raise RuntimeError("call set_product_sum() or set_sum_metric() first")

    def _product_sum_specs(self, *, z_scale=None, spatial_nugget=0.0,
                          temporal_nugget=0.0):
        """Convert product-sum parameters to engine-ready dictionaries."""
        anisotropy = dict(self.anisotropy)
        if z_scale is not None:
            if z_scale <= 0.0:
                raise ValueError("z_scale must be positive")
            anisotropy["anis2"] = 1.0 / float(z_scale)
        a, b, p, spatial_range, temporal_range = self.ps_params
        sill_s = a + p
        sill_t = b + p
        k_ps = -p / (sill_s * sill_t)
        time_at = (spatial_range / temporal_range) * (sill_s / sill_t)
        return {
            "model": "product_sum",
            "k_ps": float(k_ps),
            "spatial_spec": {
                "vtype": self.ps_spatial_vtype,
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
                "vtype": self.ps_temporal_vtype,
                "nugget": float(temporal_nugget),
                "sill": float(sill_t),
                "at_k": float(temporal_range),
            },
            "time_at": float(time_at),
        }

    def _sum_metric_specs(self):
        """Return fitted sum-metric marginal, coupling, and search parameters."""
        params = np.asarray(self.sm_params, dtype=float).reshape(-1)
        spatial_scale, temporal_scale = params[:2]
        joint_sills = params[2:-1]
        at = params[-1]
        spatial_specs = []
        for spec in self.spatial.to_kriging_specs():
            spec = dict(spec)
            spec.pop("append", None)
            if not spec["product"]:
                spec["nugget"] = float(spec["nugget"] * spatial_scale)
                spec["sill"] = float(spec["sill"] * spatial_scale)
            spatial_specs.append(spec)
        temporal_specs = []
        for spec in self.temporal.to_temporal_specs():
            spec = dict(spec)
            if not spec["product"]:
                spec["nugget"] = float(spec["nugget"] * temporal_scale)
                spec["sill"] = float(spec["sill"] * temporal_scale)
            temporal_specs.append(spec)
        return {
            "model": "sum_metric",
            "transform": self.transform,
            "at": float(at),
            "time_nugget": self.time_nugget,
            "time_sill": self.time_sill,
            "spatial_specs": spatial_specs,
            "temporal_specs": temporal_specs,
            "joint_sills": np.asarray(joint_sills, dtype=float).tolist(),
            "time_at": float(at),
        }


__all__ = ["VgmStructureST"]
