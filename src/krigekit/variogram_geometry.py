"""Lag geometry, anisotropic distance, and engine-compatible rotations."""

import numpy as np
from scipy.spatial.transform import Rotation


def calc_lag_vectors(coord0, coord1, pairwise=False):
    """Return signed lag vectors ``coord1 - coord0``.

    Coordinates may be single points or ``(n, dim)`` arrays. By default,
    matching rows are compared and a single point is broadcast. With
    ``pairwise=True`` the result has shape ``(n0, n1, dim)``.
    """
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


def calc_anisotropic_lag(
    lag,
    *,
    anis1=1.0,
    anis2=1.0,
    azimuth=0.0,
    dip=0.0,
    plunge=0.0,
):
    """Calculate engine-compatible equivalent major-axis lag distance.

    ``anis1`` and ``anis2`` are the first and second minor/major range ratios.
    Isotropic input therefore uses both ratios equal to one. The returned lag
    remains in coordinate units; dividing it by the major range gives the
    reduced lag used by the variogram kernels.
    """
    if anis1 <= 0.0 or anis2 <= 0.0:
        raise ValueError("anis1 and anis2 must be positive")
    lag = np.asarray(lag, dtype=float)
    if lag.ndim == 0 or lag.shape[-1] not in (1, 2, 3):
        raise ValueError("lag must have a final coordinate dimension of 1, 2 or 3")
    dim = lag.shape[-1]
    if dim == 1:
        return np.abs(lag[..., 0])
    flat = lag.reshape(-1, dim)
    if dim == 2:
        flat = np.column_stack([flat, np.zeros(len(flat))])
    rotated = _engine_rotation(azimuth, dip, plunge).apply(flat)
    scale = np.array([anis1, 1.0, anis2], dtype=float)
    distance = np.sqrt(np.sum((rotated / scale) ** 2, axis=-1))
    return distance.reshape(lag.shape[:-1])


def _great_circle_dist(coord1, coord2, earth_r=6371.2):
    """Great-circle distance for ``(lon, lat)`` coordinates (degrees)."""
    coord1 = np.radians(coord1)
    coord2 = np.radians(coord2)
    lon1, lat1 = coord1[:, 0], coord1[:, 1]
    lon2, lat2 = coord2[:, 0], coord2[:, 1]
    res = (np.sin(lat1) * np.sin(lat2)
           + np.cos(lat1) * np.cos(lat2) * np.cos(lon1 - lon2))
    return earth_r * np.arccos(np.clip(res, -1.0, 1.0))


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


