"""The complete theoretical variogram for one variable pair.

A :class:`VgmStructure` owns an ordered list of :class:`VgmComponent` objects
and nothing else -- no observations, empirical clouds, fit results, or plotting
state.  Nested *product* structures are evaluated exactly like the Fortran
engine: a component with ``product=True`` multiplies the immediately preceding
component in covariance space, and each product group is added to the total.
"""

import numpy as np

from .variogram_component import VgmComponent
from .variogram_geometry import calc_lag_vectors


class VgmStructure:
    """An ordered set of nested variogram components for one variable pair."""

    def __init__(self, components=None, name=None):
        """Create a structure from components or flat component specs."""
        self.name = None if name is None else str(name)
        self.components = []
        if components is not None:
            for comp in components:
                if isinstance(comp, VgmComponent):
                    self.components.append(comp)
                else:
                    self.components.append(VgmComponent.from_flat_dict(comp))

    # -- container protocol -------------------------------------------------
    @property
    def ncomponent(self):
        """Number of nested components (always ``len(self.components)``)."""
        return len(self.components)

    def __len__(self):
        """Return the number of nested components."""
        return len(self.components)

    def __iter__(self):
        """Iterate over the nested components in order."""
        return iter(self.components)

    def __getitem__(self, index):
        """Return the component at a zero-based position."""
        return self.components[index]

    def __repr__(self):
        """Return a compact debugging representation."""
        label = "" if self.name is None else f"name={self.name!r}, "
        return f"VgmStructure({label}ncomponent={self.ncomponent})"

    def copy(self):
        """Return an independent copy with copied components."""
        return VgmStructure([comp.copy() for comp in self.components], name=self.name)

    def clear(self):
        """Remove all components and return ``self``."""
        self.components.clear()
        return self

    def validate(self):
        """Validate every component and return ``self``."""
        for comp in self.components:
            comp.validate()
        return self

    # -- configuration ------------------------------------------------------
    def set_vgm(
        self,
        vtype,
        nugget=0.0,
        sill=1.0,
        a_major=1.0,
        a_minor1=None,
        a_minor2=None,
        azimuth=0.0,
        dip=0.0,
        plunge=0.0,
        append=True,
        product=False,
        name=None,
    ):
        """Add one nested component.

        Parameters mirror :meth:`krigekit.Kriging.set_vgm` (without ``ivar`` and
        ``jvar``).  Pass ``append=False`` to clear existing components first, or
        ``product=True`` to multiply this component with the preceding one in
        covariance space.
        """
        if not append:
            self.components.clear()
        self.components.append(VgmComponent(
            vtype=vtype,
            nugget=nugget,
            sill=sill,
            a_major=a_major,
            a_minor1=a_minor1,
            a_minor2=a_minor2,
            azimuth=azimuth,
            dip=dip,
            plunge=plunge,
            product=product,
            name=name,
        ))
        return self

    def set_structure_params(self, index=0, **params):
        """Update fields on one component, with validation.

        ``index`` is a zero-based component position.  Accepts any
        :class:`VgmComponent` field; the component is rebuilt and revalidated.
        """
        if not self.components:
            raise RuntimeError("call set_vgm() before set_structure_params()")
        if not 0 <= index < len(self.components):
            raise IndexError("component index out of range")
        allowed = set(VgmComponent.__dataclass_fields__)
        unknown = set(params) - allowed
        if unknown:
            raise TypeError(f"unknown component parameter(s): {sorted(unknown)}")
        self.components[index] = self.components[index].copy(**params)
        return self

    def set_anisotropy(self, index=None, **params):
        """Update anisotropy on one, several, or all components.

        ``index`` accepts ``None`` (all components), one zero-based integer, or
        a sequence of integers.  Keyword arguments are forwarded to
        :meth:`VgmComponent.set_anisotropy` (``a_minor1``, ``ratio_minor1`` /
        ``anis1``, ``azimuth``, ...).
        """
        if not self.components:
            raise RuntimeError("call set_vgm() before set_anisotropy()")
        if index is None:
            indices = range(len(self.components))
        elif np.isscalar(index):
            indices = [int(index)]
        else:
            indices = [int(i) for i in index]
        for i in indices:
            if not 0 <= i < len(self.components):
                raise IndexError("component index out of range")
            self.components[i].set_anisotropy(**params)
        return self

    # -- theoretical evaluation --------------------------------------------
    def _accumulate(self, per_component):
        """Sum product groups over already-evaluated per-component arrays."""
        total = None
        i, n = 0, len(self.components)
        while i < n:
            group = per_component[i]
            i += 1
            while i < n and self.components[i].product:
                group = group * per_component[i]
                i += 1
            total = group if total is None else total + group
        return total

    def covariance(self, distance):
        """Evaluate the nested/product covariance at scalar lag distance(s)."""
        distance = np.asarray(distance, dtype=float)
        if not self.components:
            return np.zeros_like(distance)
        per = [comp.calc_covariance(distance) for comp in self.components]
        return self._accumulate(per)

    @property
    def cov0(self):
        """Covariance at zero lag, including nugget and product groups."""
        return self.covariance(0.0)

    def variogram(self, distance):
        """Evaluate ``gamma(h) = C(0) - C(h)`` at scalar lag distance(s)."""
        return self.cov0 - self.covariance(distance)

    def calc_covariance(self, coord0, coord1, pairwise=False):
        """Evaluate covariance between coordinates, applying anisotropy.

        ``coord0`` and ``coord1`` have shape ``(dim,)`` or ``(n, dim)``.  By
        default matching rows are compared and a single point is broadcast;
        ``pairwise=True`` returns the full ``(n0, n1)`` matrix.
        """
        lag = calc_lag_vectors(coord0, coord1, pairwise=pairwise)
        if not self.components:
            return np.zeros(lag.shape[:-1], dtype=float)
        per = [comp.calc_covariance_lag(lag) for comp in self.components]
        total = np.asarray(self._accumulate(per))
        return total if pairwise else total.squeeze()

    def calc_variogram(self, coord0, coord1, pairwise=False):
        """Evaluate semivariogram values between coordinates with anisotropy."""
        gamma = np.asarray(self.cov0 - self.calc_covariance(
            coord0, coord1, pairwise=pairwise))
        return gamma if pairwise else gamma.squeeze()

    # -- fitting ------------------------------------------------------------
    def _with_flat_params(self, params, fit_nugget):
        """Return component copies updated from a flat ``(sill, range, ...)`` vector.

        Model type, anisotropy, ``product`` and ``name`` are preserved; only
        sill, major range, and (optionally) the first nugget are replaced.
        """
        params = np.asarray(params, dtype=float)
        fitted = [comp.copy(sill=float(params[2 * i]), a_major=float(params[2 * i + 1]))
                  for i, comp in enumerate(self.components)]
        if fit_nugget and len(params) % 2 == 1:
            fitted[0] = fitted[0].copy(nugget=float(params[-1]))
        return fitted

    def fit(self, data, *, kind="auto", p0=None, x_col=("distance", "mean"),
            y_col=("variogram", "mean"), sigma_col=None, weight_col=None,
            weights=None, bounds=None, fit_nugget=True, inplace=True, **kwargs):
        """Fit sills, major ranges, and nugget to an averaged variogram.

        Returns a :class:`krigekit.variogram_fitting.FitResult`.  Fitted values
        are written into copies of the current components, so model type,
        anisotropy, and ``name`` are preserved.  ``kind`` selects the fit; only
        the ``"isotropic"`` path (default ``"auto"`` for a distance/variogram
        table) is implemented here -- use ``VariogramModel.fit_anisotropy`` for
        directional fits.
        """
        from .variogram_fitting import FitResult, calc_fit_metrics, fit_vgm

        if not self.components:
            raise RuntimeError("call set_vgm() before fit()")
        if kind not in ("auto", "isotropic", "directional"):
            raise ValueError("kind must be 'auto', 'isotropic', or 'directional'")
        if kind == "directional":
            raise NotImplementedError(
                "directional fitting is available via VariogramModel.fit_anisotropy")

        if p0 is None:
            p0 = []
            for comp in self.components:
                p0.extend([comp.sill, comp.a_major])
            if fit_nugget:
                p0.append(self.components[0].nugget)

        params, cov = fit_vgm(
            data, x_col=x_col, y_col=y_col, sigma_col=sigma_col,
            weight_col=weight_col, weights=weights,
            models=self.to_kriging_specs(), p0=p0, bounds=bounds, **kwargs,
        )
        fitted = self._with_flat_params(params, fit_nugget)
        target = self if inplace else VgmStructure(fitted, name=self.name)
        if inplace:
            self.components = fitted

        labels = []
        for comp in fitted:
            labels.append((comp.display_name, comp.vtype, "sill"))
            labels.append((comp.display_name, comp.vtype, "range"))
        if fit_nugget and len(params) % 2 == 1:
            labels.append((fitted[0].display_name, fitted[0].vtype, "nugget"))

        return FitResult(
            target=target,
            params=np.asarray(params, dtype=float),
            cov=cov,
            metrics=calc_fit_metrics(data, target, x_col, y_col),
            nobs=int(len(data)),
            param_labels=labels,
        )

    # -- engine transfer ----------------------------------------------------
    def to_kriging_specs(self, replace=False):
        """Return flat specs accepted by ``Kriging.set_vgm``.

        With ``replace=True`` the first spec carries ``append=False`` so it
        clears any existing model for the pair; later specs append.
        """
        specs = []
        for i, comp in enumerate(self.components):
            spec = comp.to_flat_dict()
            spec["append"] = not (replace and i == 0)
            specs.append(spec)
        return specs

    def apply_to(self, kriging, ivar, jvar, replace=True):
        """Apply this structure to a :class:`krigekit.Kriging` object."""
        for spec in self.to_kriging_specs(replace=replace):
            kriging.set_vgm(ivar=ivar, jvar=jvar, **spec)
        return kriging


__all__ = ["VgmStructure"]
