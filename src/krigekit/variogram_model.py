"""Marginal variogram model, fitting, anisotropy, plotting, and transfer."""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from scipy.optimize import curve_fit

from .variogram_base import _VariogramModelBase
from .variogram_empirical import (
    avg_vgm,
    directional_vgm,
    estimate_aniso_angle,
    fit_aniso_angle as _fit_aniso_angle,
    raw_vgm,
)
from .variogram_fitting import FitResult, _fit_sigma
from .variogram_component import VgmComponent
from .variogram_structure import VgmStructure
from .variogram_geometry import (
    _engine_rotation,
    rotation_matrix_3d,
)
from .variogram_kernels import resolve_model
from .variogram_plotting import plot_vgm_map, plot_vgm_map3d


class VariogramModel(_VariogramModelBase):
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
        """Create an empty marginal model or load ``set_vgm`` specifications."""
        super().__init__()
        self.structure = VgmStructure()
        if structures is not None:
            for i, spec in enumerate(structures):
                spec = dict(spec)
                spec.setdefault("append", i > 0)
                self.set_vgm(**spec)

    @staticmethod
    def _raw_dimension(rawvgm):
        """Infer spatial dimension from ``d0``, ``d1`` and ``d2`` columns."""
        return sum(f"d{k}" in rawvgm.columns for k in range(3))

    def _common_orientation(self):
        """Return the shared ``(azimuth, dip, plunge)`` for all structures."""
        if not self.structure.components:
            raise RuntimeError("call set_vgm() before directional fitting")
        ref = self.structure.components[0]
        values = (ref.azimuth, ref.dip, ref.plunge)
        for comp in self.structure.components[1:]:
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
        if not self.structure.components:
            raise RuntimeError("call set_vgm() before fit() or pass p0")
        params = []
        for comp in self.structure.components:
            params.extend([comp.sill, comp.a_major])
        if fit_nugget:
            params.append(self.structure.components[0].nugget)
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
        if not self.structure.components:
            raise RuntimeError("call set_vgm() before set_params()")
        nstruct = len(self.structure.components)
        include_nugget = bool(fit_nugget)

        if params is not None:
            flat = np.asarray(params, dtype=float).reshape(-1)
            if len(flat) not in (2 * nstruct, 2 * nstruct + 1):
                raise ValueError(
                    "params must contain one (sill, range) pair per structure, "
                    "optionally followed by one trailing nugget"
                )
            for i, comp in enumerate(self.structure.components):
                comp.sill = float(flat[2 * i])
                comp.a_major = float(flat[2 * i + 1])
            if len(flat) == 2 * nstruct + 1:
                self.structure.components[0].nugget = float(flat[-1])
                include_nugget = True
            else:
                include_nugget = False

        if sills is not None:
            sills = np.asarray(sills, dtype=float).reshape(-1)
            if len(sills) != nstruct:
                raise ValueError("sills must have one value per structure")
            for comp, value in zip(self.structure.components, sills):
                comp.sill = float(value)

        if ranges is not None:
            ranges = np.asarray(ranges, dtype=float).reshape(-1)
            if len(ranges) != nstruct:
                raise ValueError("ranges must have one value per structure")
            if np.any(ranges <= 0.0):
                raise ValueError("ranges must be positive")
            for comp, value in zip(self.structure.components, ranges):
                comp.a_major = float(value)

        if sill is not None:
            if nstruct != 1:
                raise ValueError("use sills=... when the model has multiple structures")
            self.structure.components[0].sill = float(sill)

        new_range = a_major if a_major is not None else range_
        if new_range is not None:
            if nstruct != 1:
                raise ValueError("use ranges=... when the model has multiple structures")
            if new_range <= 0.0:
                raise ValueError("a_major/range_ must be positive")
            self.structure.components[0].a_major = float(new_range)

        for comp in self.structure.components:
            if comp.a_major <= 0.0:
                raise ValueError("ranges must be positive")

        if nugget is not None:
            self.structure.components[0].nugget = float(nugget)
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
        makeplot: bool = False,
        fit_nugget: bool = True,
        raw_kwargs=None,
        avg_kwargs=None,
        **kwargs,
    ):
        """Fit this model to an averaged variogram, returning a ``FitResult``.

        If ``avgvgm`` is omitted, the cached ``avg_variogram_`` is used when
        available, otherwise it is computed from observations supplied by
        :meth:`set_obs`.  The theoretical fit delegates to
        :meth:`VgmStructure.fit`; this wrapper adds the empirical-data
        convenience and updates the analysis fit-state caches.

        Returns
        -------
        FitResult
            ``.target`` is this model (``inplace=True``) or a new fitted model;
            ``.params``/``.cov``/``.metrics`` carry the fit outputs and
            ``.summary()`` gives a labelled table.
        """
        if avgvgm is None:
            avgvgm = self._avg if self._avg is not None else \
                self.average(raw_kwargs=raw_kwargs, **(avg_kwargs or {}))

        common = dict(p0=p0, x_col=x_col, y_col=y_col, sigma_col=sigma_col,
                      weight_col=weight_col, weights=weights, bounds=bounds,
                      fit_nugget=fit_nugget, **kwargs)
        if inplace:
            res = self.structure.fit(avgvgm, inplace=True, **common)
            target = self
        else:
            res = self.structure.fit(avgvgm, inplace=False, **common)
            target = VariogramModel()
            target.structure = res.target
            target._raw = self._raw
            target._avg = self._avg

        target._params = res.params
        target._pcov = res.cov
        target._fitted_model = target
        self._params = res.params
        self._pcov = res.cov
        self._fitted_model = target

        ax = None
        if makeplot:
            ax = target.plot(avgvgm, x_col=x_col, y_col=y_col)
        return FitResult(target=target, params=res.params, cov=res.cov,
                         metrics=res.metrics, ax=ax, nobs=res.nobs,
                         param_labels=res.param_labels)

    def fit_aniso_angle(self, rawvgm=None, n_struct=None, set_ranges=True,
                        raw_kwargs=None, **kwargs):
        """Estimate the anisotropy orientation from the empirical cloud and apply it.

        This is the **first** fitting step of the anisotropic workflow: run it
        *before* :meth:`fit_anisotropy` so the directional binning and the
        sill/range fit use the correct axes.  A 3-D cloud uses the multi-started
        model-based profile fit (:func:`krigekit.fit_aniso_angle`); a 2-D cloud
        falls back to the fast PCA azimuth (:func:`estimate_aniso_angle`).  The
        fitted ``azimuth`` / ``dip`` / ``plunge`` are written into every
        structure, and with ``set_ranges=True`` the minor ranges are seeded from
        the fitted anisotropy ratios.

        ``rawvgm`` defaults to the cached cloud, otherwise it is computed from
        observations.  ``n_struct`` defaults to the number of structures.
        Returns ``self`` so it can precede the rest of the workflow.
        """
        if not self.structure.components:
            raise RuntimeError("call set_vgm() before fit_aniso_angle()")
        if rawvgm is None:
            rawvgm = self._raw if self._raw is not None else \
                self.experimental(**(raw_kwargs or {}))
        dim = self._raw_dimension(rawvgm)
        if n_struct is None:
            n_struct = max(1, len(self.structure.components))

        if dim >= 3:
            (azimuth, dip, plunge), (anis1, anis2) = _fit_aniso_angle(
                rawvgm, n_struct=n_struct, **kwargs)
        elif dim == 2:
            (azimuth,), (anis1,) = estimate_aniso_angle(rawvgm, dim3d=False)
            dip = plunge = 0.0
            anis2 = None
        else:
            raise ValueError("fit_aniso_angle() needs a 2-D or 3-D variogram cloud")

        for comp in self.structure.components:
            comp.azimuth = float(azimuth)
            if dim >= 3:
                comp.dip = float(dip)
                comp.plunge = float(plunge)
            if set_ranges:
                comp.a_minor1 = comp.a_major * float(anis1)
                if dim >= 3 and anis2 is not None:
                    comp.a_minor2 = comp.a_major * float(anis2)

        self.aniso_angle_ = ((azimuth, dip, plunge), (anis1, anis2))
        self._store_manual_params(fit_nugget=True)
        return self

    def _default_anisotropic_fit_p0(self, include_minor2: bool, fit_nugget: bool = True):
        """Build default ``(sill, major, minor1[, minor2], ..., [nugget])`` params."""
        if not self.structure.components:
            raise RuntimeError("call set_vgm() before fit_anisotropy() or pass p0")
        params = []
        for comp in self.structure.components:
            params.extend([comp.sill, comp.a_major, comp.a_minor1])
            if include_minor2:
                params.append(comp.a_minor2)
        if fit_nugget:
            params.append(self.structure.components[0].nugget)
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
        self.structure.components = [comp.copy() for comp in fitted.structure.components]
        self._params = np.asarray(params, dtype=float)
        self._pcov = None
        self._fitted_model = self
        return self

    def _model_from_anisotropic_params(self, params, include_minor2: bool,
                                       fit_nugget: bool = True):
        """Build a model from anisotropic flat fit parameters."""
        params = np.asarray(params, dtype=float).reshape(-1)
        nper = 4 if include_minor2 else 3
        expected = nper * len(self.structure.components) + (1 if fit_nugget else 0)
        if len(params) != expected:
            raise ValueError("anisotropic parameter vector has the wrong length")

        out = VariogramModel()
        for i, template in enumerate(self.structure.components):
            offset = nper * i
            spec = template.to_flat_dict()
            spec["name"] = template.name           # preserve metadata across the fit
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
                nstruct=len(self.structure.components),
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
            self.structure.components = [comp.copy() for comp in fitted.structure.components]
            fitted = self
            self._fitted_model = self
        else:
            fitted._raw = self._raw
            fitted._avg = self._avg
            fitted._dir = self._dir
            fitted._params = np.asarray(params, dtype=float)
            fitted._pcov = cov
            fitted._fitted_model = fitted

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

        labels = []
        for comp in fitted.structure.components:
            labels.append((comp.display_name, comp.vtype, "sill"))
            labels.append((comp.display_name, comp.vtype, "a_major"))
            labels.append((comp.display_name, comp.vtype, "a_minor1"))
            if include_minor2:
                labels.append((comp.display_name, comp.vtype, "a_minor2"))
        if fit_nugget:
            c0 = fitted.structure.components[0]
            labels.append((c0.display_name, c0.vtype, "nugget"))

        return FitResult(target=fitted, params=np.asarray(params, dtype=float),
                         cov=cov, ax=ax if makeplot else None,
                         nobs=int(len(y)), param_labels=labels)

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
        name: str = None,
    ):
        """Add one nested variogram structure.

        Parameters mirror :meth:`krigekit.Kriging.set_vgm`, except ``ivar`` and
        ``jvar`` are omitted because this object represents one variable-pair
        model.  Pass ``append=False`` to clear existing structures before
        adding the new one.  Pass ``product=True`` to multiply this structure
        with the immediately preceding structure in covariance space.

        The stored ranges and rotation angles define theoretical-model
        evaluation, directional axes, fitting, plotting overlays, and transfer
        to kriging.  They do not implicitly transform a raw empirical cloud;
        pass ``anisotropy=...`` to :meth:`calc_experimental` when desired.
        This separation is necessary because nested structures may have
        different anisotropy parameters.

        Returns
        -------
        VariogramModel
            ``self``, so calls can be chained.
        """
        self.structure.set_vgm(
            vtype, nugget=nugget, sill=sill, a_major=a_major, a_minor1=a_minor1,
            a_minor2=a_minor2, azimuth=azimuth, dip=dip, plunge=plunge,
            append=append, product=product, name=name)
        self._clear_fit_state()
        return self

    def set_structure_params(self, index: int = 0, **params):
        """Manually update fields on one stored variogram structure.

        Parameters
        ----------
        index : int, optional
            Zero-based structure index.
        **params
            Any :class:`VgmComponent` field except ``append``.  Use this for
            edits that do not fit in the flat ``set_params`` vector, such as
            ``a_minor1``, ``azimuth``, ``dip`` or ``product``.
        """
        if not self.structure.components:
            raise RuntimeError("call set_vgm() before set_structure_params()")
        if not 0 <= index < len(self.structure.components):
            raise IndexError("structure index out of range")
        allowed = set(VgmComponent.__dataclass_fields__)
        unknown = set(params) - allowed
        if unknown:
            raise TypeError(f"unknown structure parameter(s): {sorted(unknown)}")

        comp = self.structure.components[index]
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
        if not self.structure.components:
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
            indices = list(range(len(self.structure.components)))
        elif np.isscalar(structures):
            indices = [int(structures)]
        else:
            indices = [int(i) for i in structures]
        for index in indices:
            if not 0 <= index < len(self.structure.components):
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
            comp = self.structure.components[index]
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

    def covariance(self, h):
        """Evaluate the nested/product covariance model at lag distance ``h``.

        Delegates to :meth:`VgmStructure.covariance`; product groups are
        evaluated exactly like the Fortran engine.
        """
        return self.structure.covariance(h)

    @property
    def cov0(self):
        """Covariance at zero lag, including nugget and product groups."""
        return self.structure.cov0

    def variogram(self, h):
        """Evaluate the semivariogram ``gamma(h) = C(0) - C(h)``."""
        return self.structure.variogram(h)

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
        return self.structure.calc_covariance(coord0, coord1, pairwise=pairwise)

    def calc_variogram(self, coord0, coord1, pairwise: bool = False):
        """Evaluate semivariogram values between coordinates with anisotropy."""
        return self.structure.calc_variogram(coord0, coord1, pairwise=pairwise)

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
                    if self.structure.components:
                        xmax = max(comp.a_major for comp in self.structure.components)
                    else:
                        xmax = 1.0
                h = np.linspace(0.0, xmax * 1.1, 200)
            ax.plot(h, self.variogram(h), **plotkws_model)
        if annotate and self.structure.components:
            ms = "Model: " + "\t".join(comp.vtype.capitalize() for comp in self.structure.components)
            ss = "\nSill : " + "\t".join(f"{comp.sill:.5g}" for comp in self.structure.components)
            rr = "\nRange: " + "\t".join(f"{comp.a_major:.6g}" for comp in self.structure.components)
            nn = f"\nNugget: {self.structure.components[0].nugget:.5g}"
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
            angle = self.structure.components[0].azimuth if self.structure.components else None
        elif angle_aniso == "estimate":
            angle = estimate_aniso_angle(rawvgm, dim3d=False)[0][0]
        else:
            angle = angle_aniso

        if ellipse_aniso == "model":
            if self.structure.components:
                comp = self.structure.components[0]
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

        - **Fence A** â€” horizontal XY plane (azimuth pattern).
        - **Fence B** (``n_fences â‰¥ 2``) â€” vertical XZ Eastâ€“West section (dip).
        - **Fence C** (``n_fences â‰¥ 3``) â€” vertical YZ Northâ€“South section.

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
            if self.structure.components:
                comp = self.structure.components[0]
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
        for i, comp in enumerate(self.structure.components):
            spec = comp.to_flat_dict()
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

    def to_temporal_specs(self):
        """Return structures accepted by ``SpaceTimeKriging.set_vgm_temporal``.

        The one-dimensional ``a_major`` value is renamed to ``at_k``.  Spatial
        anisotropy fields are intentionally omitted because temporal marginal
        structures are one-dimensional.
        """
        return [
            {
                "vtype": comp.vtype,
                "nugget": comp.nugget,
                "sill": comp.sill,
                "at_k": comp.a_major,
                "product": comp.product,
            }
            for comp in self.structure.components
        ]

    def apply_temporal_to(self, kriging, ivar: int, jvar: int):
        """Append this model to a ``SpaceTimeKriging`` temporal marginal.

        The target pair should not already contain temporal structures.  The
        space-time API currently resets spatial and temporal marginals
        together, so this helper deliberately does not offer a replace mode.
        """
        for spec in self.to_temporal_specs():
            kriging.set_vgm_temporal(ivar=ivar, jvar=jvar, **spec)
        return kriging

    def __len__(self):
        """Return the number of stored structures."""
        return len(self.structure.components)

    def __repr__(self):
        """Return a compact debugging representation."""
        return f"VariogramModel(nstruct={len(self.structure.components)})"


