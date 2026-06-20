"""Compatibility facade for krigekit's variogram analysis API.

Implementation is split by responsibility:

``variogram_kernels``
    Engine-compatible covariance kernels and components.
``variogram_geometry``
    Lag vectors, anisotropic distances, and rotations.
``variogram_empirical``
    Variogram clouds, binning, and directional analysis.
``variogram_fitting``
    Generic marginal fitting.
``variogram_plotting``
    Curve, map, polar, and 3-D plots.
``variogram_model``
    Marginal :class:`VariogramModel`.
``variogram_st``
    Composed :class:`SpaceTimeVariogramModel`.
``variogram_system``
    Multivariable :class:`VariogramSystem`.

Imports from ``krigekit.variogram`` remain supported.
"""

from .variogram_empirical import (
    avg_vgm,
    cross_vgm,
    directional_vgm,
    distance_pnt_line,
    estimate_angle_angular_profile,
    estimate_aniso_angle,
    filter_vgm,
    raw_cross_vgm,
    raw_vgm,
)
from .variogram_fitting import fit_vgm
from .variogram_component import VgmComponent
from .variogram_structure import VgmStructure
from .variogram_geometry import (
    _engine_rotation,
    _great_circle_dist,
    azimuth_dip_to_vector,
    calc_anisotropic_lag,
    calc_lag_vectors,
    rotation_matrix_3d,
)
from .variogram_kernels import (
    _ANALYTIC_TAIL,
    _MODEL_ALIASES,
    _VgmComponent,
    _covfunc,
    _vgmfunc,
    calc_cov,
    calc_vgm,
    resolve_model,
    vgm,
    vgmfunc,
)
from .variogram_model import VariogramModel
from .variogram_plotting import (
    _fill_nan_nearest,
    plot_vgm,
    plot_vgm_anisotropy3d,
    plot_vgm_map,
    plot_vgm_map3d,
    plot_vgm_map_polar,
)
from .variogram_st import SpaceTimeVariogramModel
from .variogram_system import VariogramSystem

__all__ = [
    "VgmComponent",
    "VgmStructure",
    "VariogramModel",
    "SpaceTimeVariogramModel",
    "VariogramSystem",
    "avg_vgm",
    "azimuth_dip_to_vector",
    "calc_anisotropic_lag",
    "calc_cov",
    "calc_lag_vectors",
    "calc_vgm",
    "cross_vgm",
    "directional_vgm",
    "distance_pnt_line",
    "estimate_angle_angular_profile",
    "estimate_aniso_angle",
    "filter_vgm",
    "fit_vgm",
    "plot_vgm",
    "plot_vgm_anisotropy3d",
    "plot_vgm_map",
    "plot_vgm_map3d",
    "plot_vgm_map_polar",
    "raw_cross_vgm",
    "raw_vgm",
    "resolve_model",
    "rotation_matrix_3d",
    "vgm",
    "vgmfunc",
]
