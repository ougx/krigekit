"""Generic weighted fitting for marginal nested variogram models."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

from .variogram_kernels import vgmfunc


@dataclass
class FitResult:
    """Uniform result of a variogram fit at any level.

    ``target`` is the fitted object (a :class:`VgmStructure`, analysis model, or
    system).  ``params`` is the flat fitted vector, ``cov`` its covariance when
    available, ``optimizer`` the SciPy result for joint/constrained fits, and
    ``metrics`` a goodness-of-fit dict.  :meth:`summary` returns a labelled
    parameter table using each component's ``name``/``vtype``.
    """

    target: object
    params: object = None
    cov: object = None
    optimizer: object = None
    metrics: object = None
    ax: object = None

    @property
    def success(self):
        """True unless an optimizer reported failure."""
        if self.optimizer is None:
            return True
        return bool(getattr(self.optimizer, "success", True))

    def _structure(self):
        """Return the VgmStructure carried by the target, if any."""
        if hasattr(self.target, "components"):
            return self.target
        return getattr(self.target, "structure", None)

    def summary(self):
        """Return a labelled parameter table (one row per fitted value)."""
        import pandas as pd

        structure = self._structure()
        if structure is None:
            raise TypeError("summary() requires a structure-bearing fit target")
        name = getattr(structure, "name", None)
        rows = []
        comps = structure.components
        if comps:
            c0 = comps[0]
            rows.append((name, c0.display_name, c0.vtype, "nugget", c0.nugget))
        for comp in comps:
            rows.append((name, comp.display_name, comp.vtype, "sill", comp.sill))
            rows.append((name, comp.display_name, comp.vtype, "range", comp.a_major))
        return pd.DataFrame(
            rows, columns=["structure", "component", "vtype", "param", "value"])


def _normalise_model_specs(models):
    """Return structure specifications from strings, dictionaries, or a model."""
    from .variogram_model import VariogramModel

    if isinstance(models, VariogramModel):
        return models.to_kriging_specs()
    if isinstance(models, dict):
        return [dict(models)]
    return [
        {"vtype": item} if isinstance(item, str) else dict(item)
        for item in models
    ]


def _uses_model_template(models):
    """Return whether models carry structure metadata beyond model names."""
    from .variogram_model import VariogramModel

    if isinstance(models, (VariogramModel, dict)):
        return True
    return any(not isinstance(model, str) for model in models)


def _model_from_params(models, params):
    """Build a marginal model from a structure template and flat parameters."""
    from .variogram_model import VariogramModel

    specs = _normalise_model_specs(models)
    params = tuple(params)
    has_nugget = len(params) % 2 != 0
    if len(params) // 2 != len(specs):
        raise ValueError("params must contain one (sill, range) pair per model")

    result = VariogramModel()
    for index, spec in enumerate(specs):
        spec = dict(spec)
        spec["sill"] = params[2 * index]
        spec["a_major"] = params[2 * index + 1]
        spec.setdefault("append", index > 0)
        if has_nugget and index == 0:
            spec["nugget"] = params[-1]
        else:
            spec.setdefault("nugget", 0.0)
        result.set_vgm(**spec)
    return result


def _vgmfunc_from_model_specs(models, h, *params):
    """Evaluate a templated nested/product variogram."""
    return _model_from_params(models, params).variogram(h)


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
            bounds=None, return_model=False, return_metrics=False):
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
    if return_model or return_metrics:
        fitted_model = _model_from_params(models, p)
    if return_model:
        result.append(fitted_model)
    if return_metrics:
        result.append(calc_fit_metrics(avgvgm, fitted_model, x_col, y_col))
    if makeplot:
        from .variogram_plotting import plot_vgm

        ax = plot_vgm(avgvgm, x_col, y_col, models, p, ax=ax,
                      xlabel=xlabel, ylabel=ylabel)
        result.append(ax)
    return tuple(result)

def calc_fit_metrics(avgvgm, model, x_col=("distance", "mean"), y_col=("variogram", "mean")):
    """Calculate goodness-of-fit metrics for a VariogramModel against empirical data."""
    import numpy as np
    xdata = np.asarray(avgvgm[x_col], dtype=float)
    ydata = np.asarray(avgvgm[y_col], dtype=float)
    
    ypred = model.variogram(xdata)
    
    mse = np.mean((ydata - ypred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(ydata - ypred))
    
    ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
    if ss_tot > 0:
        r2 = 1.0 - (np.sum((ydata - ypred) ** 2) / ss_tot)
    else:
        r2 = np.nan
        
    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }

# ---------------------------------------------------------------------------
