"""Empirical variogram clouds, binning, directions, and anisotropy estimates."""

import random
import sys

import numpy as np
import pandas as pd

from .variogram_geometry import (
    _great_circle_dist,
    calc_anisotropic_lag,
    calc_lag_vectors,
)


def _choose_data(coords, vals, nmax, times=None):
    """Optionally draw a random subset of at most *nmax* points.

    Returns ``(selected_indices, coords, vals, times)`` where the arrays are
    restricted to the selection (and the indices map back to the originals).
    """
    n = len(coords)
    if nmax is None or nmax >= n:
        return np.arange(n), coords, vals, times

    selected = np.sort(random.sample(range(n), nmax))
    times = np.asarray(times)[selected] if times is not None else None
    return selected, coords[selected], vals[selected], times


def _pair_lags(coords0, coords1, val0, val1, cutoff,
               time0, time1, time_cutoff, calc_angle,
               valB0=None, valB1=None, great_circle=False, anisotropy=None):
    """Compute lags and (cross-)semivariances between one point and a block.

    ``coords0`` holds a single point (shape ``(1, dim)``); ``coords1`` holds the
    candidate partners.  Returns a dict of the per-pair arrays already reduced
    to the pairs that pass the spatial (and optional temporal) cutoff, plus the
    boolean ``mask`` over ``coords1``.

    When a second variable (``valB0`` for the anchor point, ``valB1`` for the
    block) is supplied, the ``variogram`` column holds the **cross**-semivariance
    ``0.5 * (A_i - A_j) * (B_i - B_j)`` instead of the direct semivariance.

    With ``great_circle`` (2D ``(lon, lat)`` coordinates in degrees) the spatial
    lag ``distance`` is the great-circle distance; the signed components and
    azimuth are still derived from the planar degree differences.
    """
    dh = calc_lag_vectors(coords0, coords1)
    if great_circle:
        hlag = _great_circle_dist(np.broadcast_to(coords0, coords1.shape), coords1)
        selection_lag = hlag
    else:
        hlag = calc_anisotropic_lag(dh)
        selection_lag = hlag
        if anisotropy is not None:
            selection_lag = calc_anisotropic_lag(dh, **anisotropy)

    mask = selection_lag <= cutoff
    if time0 is not None:
        tlag = np.abs(time1 - time0)
        mask &= tlag <= time_cutoff

    out = {"mask": mask}
    if not np.any(mask):
        return out

    dim = coords1.shape[1]
    dh = dh[mask]
    out["distance"] = hlag[mask]
    if anisotropy is not None:
        out["anisotropic_distance"] = selection_lag[mask]
    dA = val1[mask] - val0
    if valB0 is None:
        out["variogram"] = 0.5 * dA ** 2
    else:
        out["variogram"] = 0.5 * dA * (valB1[mask] - valB0)
    out["dh"] = dh
    if time0 is not None:
        out["time_lag"] = tlag[mask]

    if dim >= 2:
        dx, dy = dh[:, 0], dh[:, 1]
        out["dh_hori"] = np.hypot(dx, dy)
        if calc_angle:
            # azimuth clockwise from +Y (North): atan2(dx, dy)
            out["angle_h"] = np.degrees(np.arctan2(dx, dy)) % 360.0
            if dim == 3:
                # dip below horizontal plane, positive downward (matches the
                # model ``dip`` convention and scipy 'zxy' Euler angles)
                out["angle_v"] = np.degrees(np.arctan2(-dh[:, 2], out["dh_hori"]))
    return out


def _build_cloud(dim, idx0, idx1, records, i0s, i1s):
    """Assemble the variogram-cloud DataFrame from accumulated per-block lists."""
    if not records["distance"]:
        cols = ["index0", "index1", "distance", "variogram"] + [f"d{k}" for k in range(dim)]
        return pd.DataFrame(columns=cols)

    out = {
        "index0": idx0[np.concatenate(i0s)],
        "index1": idx1[np.concatenate(i1s)],
        "distance": np.concatenate(records["distance"]),
        "variogram": np.concatenate(records["variogram"]),
    }
    dh = np.concatenate(records["dh"], axis=0)
    for k in range(dim):
        out[f"d{k}"] = dh[:, k]
    for key in ("anisotropic_distance", "time_lag", "dh_hori", "angle_h", "angle_v"):
        if records[key]:
            out[key] = np.concatenate(records[key])
    return pd.DataFrame(out)


def _empty_records():
    """Create empty list accumulators for variogram-cloud columns."""
    return {k: [] for k in
            ("distance", "anisotropic_distance", "variogram", "dh", "time_lag",
             "dh_hori", "angle_h", "angle_v")}


def _accumulate(records, pair):
    """Append one block of pair arrays into the variogram-cloud accumulators."""
    for key in records:
        if key in pair:
            records[key].append(pair[key])


def _progress(i, total, verbose):
    """Print an in-place integer progress percentage when ``verbose`` is true."""
    if verbose and total > 0 and i % max(total // 100, 1) == 0:
        sys.stdout.write(f"\rProgress: {int(i / total * 100)}%")
        sys.stdout.flush()


def _axis_angle_diff(angle, target):
    """Smallest angular difference in degrees between two undirected axes."""
    return np.abs((np.asarray(angle) - target + 90.0) % 180.0 - 90.0)


def _circular_angle_diff(angle, target):
    """Smallest angular difference in degrees between two directed bearings."""
    return np.abs((np.asarray(angle) - target + 180.0) % 360.0 - 180.0)


def _axis_dip_mask(angle_h, angle_v, target_h, target_v, h_tol, v_tol):
    """Mask for an undirected 3D variogram axis.

    The same axis can be represented as ``(azimuth, dip)`` or as the reversed
    lag vector ``(azimuth + 180, -dip)``.  Test both directed representations so
    vertical directions are not mismatched when horizontal azimuth is flipped.
    """
    angle_h = np.asarray(angle_h)
    angle_v = np.asarray(angle_v)
    same = (
        (_circular_angle_diff(angle_h, target_h) <= h_tol)
        & (np.abs(angle_v - target_v) <= v_tol)
    )
    flipped = (
        (_circular_angle_diff(angle_h, target_h + 180.0) <= h_tol)
        & (np.abs(angle_v + target_v) <= v_tol)
    )
    return same | flipped


def raw_vgm(coords, vals, cutoff=np.inf, times=None, t_cutoff=np.inf,
             calc_angle=False, maxobs=None, seed=None, verbose=True,
             great_circle=False, anisotropy=None):
    """Compute the raw empirical variogram cloud (all admissible pairs).

    Parameters
    ----------
    coords : array-like, shape (n, dim)
        Sample coordinates; ``dim`` may be 1, 2 or 3.
    vals : array-like, shape (n,)
        Sample values.
    cutoff : float, optional
        Maximum spatial lag ``|h|`` to keep (default: no cutoff).
    times : array-like, optional
        Per-sample time stamps for a space-time variogram.
    t_cutoff : float, optional
        Maximum absolute time lag to keep.
    calc_angle : bool, optional
        Also compute ``angle_h`` (and ``angle_v`` in 3D) for each pair.
    maxobs : int, optional
        Randomly subsample to at most ``maxobs`` points before pairing.
    seed : int, optional
        Random seed for the subsampling.
    verbose : bool, optional
        Print a progress indicator.
    great_circle : bool, optional
        Treat 2D coordinates as ``(lon, lat)`` in degrees and use the
        great-circle distance (km) for the ``distance`` lag and ``cutoff``.
    anisotropy : dict, optional
        Arguments accepted by :func:`calc_anisotropic_lag`. When supplied,
        ``cutoff`` is applied to the equivalent major-axis lag and the cloud
        includes an ``anisotropic_distance`` column. The physical Euclidean
        lag remains available as ``distance``.

    Returns
    -------
    pandas.DataFrame
        The variogram cloud (see module docstring for the columns).
    """
    coords = np.asarray(coords, dtype=float)
    vals = np.asarray(vals, dtype=float).reshape(-1)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 1)
    n, dim = coords.shape
    if len(vals) != n:
        raise ValueError("coords and vals must have matching lengths")
    if dim not in (1, 2, 3):
        raise ValueError("coordinates must be 1D, 2D or 3D")
    if great_circle and dim != 2:
        raise ValueError("great_circle requires 2D (lon, lat) coordinates")
    if great_circle and anisotropy is not None:
        raise ValueError("great_circle and Cartesian anisotropy cannot be combined")
    if anisotropy is not None:
        anisotropy = dict(anisotropy)
        calc_anisotropic_lag(np.zeros((1, dim)), **anisotropy)

    if seed is not None:
        random.seed(seed)
    selected, coords, vals, times = _choose_data(coords, vals, maxobs, times)
    n = len(selected)
    if verbose:
        print(f"{n} points, dim={dim}")
    has_time = times is not None

    records = _empty_records()
    i0s, i1s = [], []
    for i in range(n - 1):
        pair = _pair_lags(
            coords[i:i + 1], coords[i + 1:], vals[i], vals[i + 1:], cutoff,
            times[i] if has_time else None, times[i + 1:] if has_time else None,
            t_cutoff, calc_angle, great_circle=great_circle,
            anisotropy=anisotropy)
        mask = pair["mask"]
        if np.any(mask):
            i0s.append(np.full(np.count_nonzero(mask), i))
            i1s.append(np.arange(i + 1, n)[mask])
            _accumulate(records, pair)
        _progress(i, n - 2, verbose)

    if verbose:
        print("\rProgress: 100%")
    return _build_cloud(dim, selected, selected, records, i0s, i1s)


def raw_cross_vgm(coords, valsA, valsB, cutoff=np.inf, times=None, t_cutoff=np.inf,
                  calc_angle=False, maxobs=None, seed=None, verbose=True,
                  anisotropy=None):
    """Traditional (LMC) cross-variogram cloud on **collocated** data.

    Both variables are measured at the same ``coords`` (isotopic data).  Pairs
    are formed *within* the single point set and the cross-semivariance::

        gamma_AB(h) = 0.5 * (A_i - A_j) * (B_i - B_j)

    is stored in the ``variogram`` column.  Unlike the pseudo estimator in
    :func:`cross_vgm`, this is symmetric in A/B, its sill is the LMC cross-sill
    you can feed into cokriging, and it costs O(n**2 / 2) rather than O(nA*nB).

    The returned cloud has exactly the same columns as :func:`raw_vgm`, so it
    flows straight into :func:`avg_vgm`, :func:`estimate_aniso_angle`,
    :func:`directional_vgm` and the variogram-map plots.  Calling it with
    ``valsB is valsA`` reproduces the direct variogram of A.

    For *heterotopic* data (A and B at different locations) the true
    cross-variogram is not computable; use :func:`cross_vgm` (pseudo estimator)
    or restrict to a collocated subset before calling this function.
    """
    coords = np.asarray(coords, dtype=float)
    valsA = np.asarray(valsA, dtype=float).reshape(-1)
    valsB = np.asarray(valsB, dtype=float).reshape(-1)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 1)
    n, dim = coords.shape
    if not (len(valsA) == n and len(valsB) == n):
        raise ValueError("coords, valsA and valsB must have matching lengths")
    if dim not in (1, 2, 3):
        raise ValueError("coordinates must be 1D, 2D or 3D")
    if anisotropy is not None:
        anisotropy = dict(anisotropy)
        calc_anisotropic_lag(np.zeros((1, dim)), **anisotropy)

    if seed is not None:
        random.seed(seed)
    selected, coords, _, times = _choose_data(coords, valsA, maxobs, times)
    valsA = valsA[selected]
    valsB = valsB[selected]
    n = len(selected)
    if verbose:
        print(f"{n} collocated points, dim={dim}")
    has_time = times is not None

    records = _empty_records()
    i0s, i1s = [], []
    for i in range(n - 1):
        pair = _pair_lags(
            coords[i:i + 1], coords[i + 1:], valsA[i], valsA[i + 1:], cutoff,
            times[i] if has_time else None, times[i + 1:] if has_time else None,
            t_cutoff, calc_angle, valB0=valsB[i], valB1=valsB[i + 1:],
            anisotropy=anisotropy)
        mask = pair["mask"]
        if np.any(mask):
            i0s.append(np.full(np.count_nonzero(mask), i))
            i1s.append(np.arange(i + 1, n)[mask])
            _accumulate(records, pair)
        _progress(i, n - 2, verbose)

    if verbose:
        print("\rProgress: 100%")
    return _build_cloud(dim, selected, selected, records, i0s, i1s)


def cross_vgm(coordsA, valsA, coordsB, valsB, cutoff=np.inf, residual=True,
              timesA=None, timesB=None, t_cutoff=np.inf, calc_angle=False,
              maxobs=None, maxobsA=None, maxobsB=None, seed=None, verbose=True,
              anisotropy=None):
    """Cross-variogram cloud between two co-located variables A and B.

    Pairs are formed across the two data sets (every A vs every B).  When
    ``residual`` is True the means are removed first so the result is a
    cross-covariance-style estimator.

    .. warning::
        This is the **pseudo** cross-variogram ``0.5 * E[(A(x) - B(x+h))**2]``,
        not the linear-model-of-coregionalisation (LMC) cross-variogram
        ``0.5 * E[(A(x) - A(x+h)) * (B(x) - B(x+h))]``.  Its sill mixes
        ``var(A) + var(B) - 2*cov(A, B)``, so do **not** read an LMC cross-sill
        directly off it.  For heterotopic data (A and B at different locations)
        the true cross-variogram is not computable and this pseudo form is a
        reasonable substitute; for a cokriging cross-sill, derive
        ``cov(A, B)`` from collocated hard-data pairs instead.
    """
    coordsA = np.asarray(coordsA, dtype=float)
    coordsB = np.asarray(coordsB, dtype=float)
    valsA = np.asarray(valsA, dtype=float).reshape(-1)
    valsB = np.asarray(valsB, dtype=float).reshape(-1)
    if coordsA.ndim == 1:
        coordsA = coordsA.reshape(-1, 1)
    if coordsB.ndim == 1:
        coordsB = coordsB.reshape(-1, 1)

    nA, dA = coordsA.shape
    nB, dB = coordsB.shape
    if not (nA == len(valsA) and nB == len(valsB)):
        raise ValueError("coords and vals must have matching lengths")
    if dA != dB:
        raise ValueError("A and B must share the same dimensionality")
    dim = dA
    if dim not in (1, 2, 3):
        raise ValueError("coordinates must be 1D, 2D or 3D")
    if anisotropy is not None:
        anisotropy = dict(anisotropy)
        calc_anisotropic_lag(np.zeros((1, dim)), **anisotropy)

    if maxobs is not None:
        maxobsA = maxobsA or min(maxobs, nA)
        maxobsB = maxobsB or min(maxobs, nB)
    if maxobsA is not None and maxobsB is None:
        maxobsB = min(maxobsA, nB)

    if seed is not None:
        random.seed(seed)
    selA, coordsA, valsA, timesA = _choose_data(coordsA, valsA, maxobsA, timesA)
    selB, coordsB, valsB, timesB = _choose_data(coordsB, valsB, maxobsB, timesB)
    nA = len(selA)

    if residual:
        valsA = valsA - np.mean(valsA)
        valsB = valsB - np.mean(valsB)

    has_time = timesA is not None and timesB is not None

    records = _empty_records()
    i0s, i1s = [], []
    for i in range(nA):
        pair = _pair_lags(
            coordsA[i:i + 1], coordsB, valsA[i], valsB, cutoff,
            timesA[i] if has_time else None, timesB if has_time else None,
            t_cutoff, calc_angle, anisotropy=anisotropy)
        mask = pair["mask"]
        if np.any(mask):
            i0s.append(np.full(np.count_nonzero(mask), i))
            i1s.append(np.arange(len(selB))[mask])
            _accumulate(records, pair)
        _progress(i, nA - 1, verbose)

    if verbose:
        print("\rProgress: 100%")
    return _build_cloud(dim, selA, selB, records, i0s, i1s)


# ---------------------------------------------------------------------------
# 4. Binning / averaging
# ---------------------------------------------------------------------------
def _cressie_gamma(semivariances):
    """Robust Cressie--Hawkins variogram estimate from per-pair semivariances.

    From semivariance ``g = 0.5 (z_i - z_j)^2`` we recover ``|z_i - z_j|^0.5 =
    (2 g)^0.25``.  The robust estimator is::

        2*gamma = mean(|dz|^0.5)^4 / (0.457 + 0.494/N + 0.045/N^2)
    """
    g = np.asarray(semivariances, dtype=float)
    n = g.size
    if n == 0:
        return np.nan
    term = np.mean((2.0 * g) ** 0.25) ** 4
    return 0.5 * term / (0.457 + 0.494 / n + 0.045 / n ** 2)


def _lag_bin_indices(values, width_or_edges, name):
    """Return integer lag-bin indices and a mask of values inside the bins."""
    values = np.asarray(values, dtype=float)
    if np.ndim(width_or_edges) == 0:
        width = float(width_or_edges)
        if not np.isfinite(width) or width <= 0:
            raise ValueError(f"{name} must be a positive finite scalar")
        valid = np.isfinite(values)
        indices = np.zeros(values.shape, dtype=int)
        indices[valid] = np.floor(values[valid] / width).astype(int)
        return indices, valid

    edges = np.asarray(width_or_edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError(f"{name} edges must be a one-dimensional array of length >= 2")
    if not np.all(np.isfinite(edges)):
        raise ValueError(f"{name} edges must contain only finite values")
    if np.any(np.diff(edges) <= 0):
        raise ValueError(f"{name} edges must be strictly increasing")

    valid = np.isfinite(values) & (values >= edges[0]) & (values <= edges[-1])
    indices = np.searchsorted(edges, values, side="right") - 1
    indices[values == edges[-1]] = edges.size - 2
    return indices.astype(int), valid


def avg_vgm(rawvgm, h_col="distance", t_col=None, cutoff=None, t_cutoff=None,
            h_width=None, t_width=None, h_bins=None, t_bins=None,
            tor_hori=None, tor_vert=None, angleh=None, anglev=None,
            angleh_tor=15, anglev_tor=10, robust=False, vgm_col="variogram"):
    """Bin a variogram cloud and average it.

    Supports distance (and optional time) binning, directional and bandwidth
    filtering, and either the classic (Matheron) mean or the robust
    Cressie--Hawkins estimator (``robust=True``).  ``h_width`` and ``t_width``
    may be positive scalars for fixed-width bins or one-dimensional arrays of
    explicit bin edges.  Edge-array bins are left-closed and right-open, except
    that the final edge is included; lags outside the edge range are omitted.
    Arrays passed through the legacy ``h_bins`` and ``t_bins`` arguments use
    the same edge convention.

    Directional filtering follows variogram-axis symmetry.  If only
    ``angleh`` is supplied, azimuths ``angleh`` and ``angleh + 180`` are treated
    as the same horizontal axis.  If both ``angleh`` and ``anglev`` are
    supplied, the same 3D axis is represented by ``(angleh, anglev)`` and the
    reversed lag vector ``(angleh + 180, -anglev)``.

    Returns
    -------
    pandas.DataFrame
        Grouped variogram table whose columns are a ``(variable, statistic)``
        MultiIndex with statistics ``mean``, ``std`` and ``count``.  The
        averaged semivariogram is available as ``result[(vgm_col, "mean")]``.
    """
    df = rawvgm
    if cutoff is not None:
        df = df.loc[df[h_col] <= cutoff]
    if t_cutoff is not None and t_col is not None:
        df = df.loc[df[t_col] <= t_cutoff]
    if tor_hori is not None:
        df = df.loc[df["dh_hori"] <= tor_hori]
    if tor_vert is not None:
        if "d2" not in df.columns:
            raise ValueError(
                "tor_vert requires a 3D variogram cloud with a 'd2' column; "
                "recompute the cloud with 3D coordinates"
            )
        df = df.loc[df["d2"].abs() <= tor_vert]
    if angleh is not None and anglev is not None:
        mask = _axis_dip_mask(
            df["angle_h"], df["angle_v"],
            angleh, anglev, angleh_tor, anglev_tor,
        )
        df = df.loc[mask]
    elif angleh is not None:
        # Keep pairs whose azimuth is within +/-angleh_tor of angleh or its
        # 180-degree opposite; without a dip constraint this is an undirected
        # horizontal axis.
        diff = _axis_angle_diff(df["angle_h"], angleh)
        df = df.loc[diff <= angleh_tor]
    elif anglev is not None:
        diff = df["angle_v"] - anglev
        df = df.loc[(diff >= -anglev_tor) & (diff < anglev_tor)]

    df = df.copy()
    indices = []
    if h_col in df.columns:
        if h_bins is None and h_width is None:
            h_bins = 15
        if isinstance(h_bins, int) and h_width is None:
            if h_bins <= 0:
                raise ValueError("h_bins must be a positive integer")
            hmax = df[h_col].max()
            if not np.isfinite(hmax) or hmax <= 0:
                raise ValueError("cannot bin a variogram cloud with no positive lags")
            h_width = hmax / h_bins
        bin_spec = h_width if h_width is not None else h_bins
        hindex, valid = _lag_bin_indices(df[h_col], bin_spec, "h_width")
        df = df.loc[valid].copy()
        df["hindex"] = hindex[valid]
        indices.append("hindex")

    if t_col is not None and t_col in df.columns:
        if t_bins is None and t_width is None:
            t_bins = 15
        if isinstance(t_bins, int) and t_width is None:
            if t_bins <= 0:
                raise ValueError("t_bins must be a positive integer")
            tmax = df[t_col].max()
            if not np.isfinite(tmax) or tmax <= 0:
                raise ValueError("cannot bin a variogram cloud with no positive time lags")
            t_width = tmax / t_bins
        bin_spec = t_width if t_width is not None else t_bins
        tindex, valid = _lag_bin_indices(df[t_col], bin_spec, "t_width")
        df = df.loc[valid].copy()
        df["tindex"] = tindex[valid]
        indices.append("tindex")

    grouped = df.groupby(indices)
    out = grouped.agg(["mean", "std", "count"])
    if robust:
        out[(vgm_col, "mean")] = grouped[vgm_col].apply(_cressie_gamma).values
    return out


# ---------------------------------------------------------------------------
# 5. Anisotropy and directional analysis
# ---------------------------------------------------------------------------
def distance_pnt_line(dx, dy, azimuth):
    """Perpendicular distance from offset vectors ``(dx, dy)`` to a line through
    the origin with the given *azimuth* (degrees, clockwise from +Y)."""
    theta = np.deg2rad(azimuth)
    nx, ny = np.cos(theta), -np.sin(theta)   # unit normal to the line
    return np.abs(dx * nx + dy * ny)


def filter_vgm(dx, dy, azimuth, azimuth_anis, bandwidth=None, angle_tol=10):
    """Boolean mask selecting pairs aligned with *azimuth_anis*.

    This is a 2D/horizontal-axis helper.  A pair is kept when its azimuth is
    within ``angle_tol`` of the target axis, treating ``azimuth_anis`` and
    ``azimuth_anis + 180`` as equivalent.  If ``bandwidth`` is given, the pair
    must also fall within ``bandwidth / 2`` of the directional line.

    For 3D azimuth+dip filtering, use :func:`avg_vgm`, which flips the sign of
    dip for the reversed azimuth.
    """
    ang_diff = _axis_angle_diff(azimuth, azimuth_anis)
    mask = ang_diff < angle_tol
    if bandwidth is not None:
        mask = mask & (distance_pnt_line(dx, dy, azimuth_anis) < bandwidth / 2.0)
    return mask


def _canonical_axis_sign(vector, dim3d):
    """Fix the arbitrary sign of an eigenvector for reproducible angles.

    ``numpy.linalg.eigh`` returns eigenvectors with an arbitrary sign, which
    would flip the reported dip (and, in 2D, the azimuth by 180 degrees) from
    run to run on identical data.  An anisotropy axis is undirected, so we pick a
    deterministic orientation: the vertical component non-positive in 3D (so a
    typical down-tilted axis reports a positive, down-dip), the +Y (northing)
    component non-negative in 2D.
    """
    vector = np.asarray(vector, dtype=float)
    if dim3d:
        if vector[2] > 0:
            vector = -vector
    elif vector[1] < 0:
        vector = -vector
    return vector


def estimate_aniso_angle(rawvgm, x="d0", y="d1", dist="distance", vgm="variogram",
                         r_max=None, z="d2", dim3d=False, robust=False,
                         get_eigens=False, dx=None, dy=None, dz=None):
    """Estimate anisotropy directions from a variogram cloud via weighted PCA.

    Lag coordinates near the origin are binned, weighted inversely by the
    (normalised) variogram value -- low variogram means strong continuity --
    and an eigen-decomposition of the weighted covariance yields the principal
    axes.

    Returns
    -------
    If ``get_eigens`` is True: ``(eigvals, eigvecs)`` sorted descending.
    If ``dim3d`` is False: ``((azimuth_deg,), axis_lengths)``.
    If ``dim3d`` is True:  ``((azimuth_deg, dip_deg), axis_lengths)``.

    ``azimuth_deg`` is the compass azimuth (0 = North, 90 = East) of the major
    (maximum-continuity) axis and ``dip_deg`` its tilt below the horizontal
    plane, **positive downward** -- the same convention as the model ``dip``
    field and the Fortran engine, so the returned pair can be fed straight into
    :meth:`VariogramModel.set_vgm` as ``azimuth``/``dip``.
    """
    if r_max is None:
        r_max = np.quantile(rawvgm[dist], 0.25)
    df = rawvgm[rawvgm[dist] < r_max].copy()

    if dx is None:
        dx = abs(np.quantile(df[x], 0.9) / 20)
    if dy is None:
        dy = dx
    if dz is None and dim3d:
        dz = abs(np.quantile(df[z], 0.9) / 20)

    df["ix"] = (df[x] // dx).astype(int) * dx
    df["iy"] = (df[y] // dy).astype(int) * dy
    if dim3d:
        df["iz"] = (df[z] // dz).astype(int) * dz

    dims = ["ix", "iy", "iz"] if dim3d else ["ix", "iy"]
    binned = df.groupby(dims, as_index=False)[vgm].mean()

    g = binned[vgm].values
    gmin, gmax = np.nanmin(g), np.nanmax(g)
    if gmax == gmin:
        w = np.ones_like(g)
    else:
        w = np.maximum(1.0 - (g - gmin) / (gmax - gmin), 1e-6)
    if robust:
        w = np.clip(w, *np.quantile(w, [0.05, 0.95]))

    cov = np.cov(binned[dims].values.T, aweights=w)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    if get_eigens:
        return eigvals, eigvecs

    v = _canonical_axis_sign(eigvecs[:, 0], dim3d)
    azimuth = (90 - np.degrees(np.arctan2(v[1], v[0]))) % 360
    if dim3d:
        # tilt below horizontal, positive down (matches the model ``dip`` field)
        dip = np.degrees(-np.arctan2(v[2], np.hypot(v[0], v[1])))
        return (azimuth, dip), 2 * np.sqrt(eigvals)
    return (azimuth,), 2 * np.sqrt(eigvals)


def estimate_angle_angular_profile(rawvgm, angle="angle_h", dist="distance",
                                   r_profile=None, da=10, a_tol=2.5, close=False,
                                   vgm="variogram", dim3d=False):
    """Mean variogram as a function of azimuth, within a near-origin radius.

    Sweeps the azimuth in steps of *da* degrees (overlapping by ``a_tol``) and
    returns the per-sector mean of ``angle``, ``dist`` and ``vgm``.
    """
    if r_profile is None:
        r_profile = np.quantile(rawvgm[dist], 0.25)
    df = rawvgm[rawvgm[dist] <= r_profile].copy()

    profile = []
    for i in np.arange(0, 360, da):
        center = i + da / 2.0
        sel = (df[angle] - center).abs() < da / 2.0 + a_tol
        if sel.any():
            profile.append(df.loc[sel, [angle, dist, vgm]].mean())
    if close and profile:
        profile.append(profile[0])
    return pd.DataFrame(profile)



def directional_vgm(rawvgm, directions, h_width=None, h_bins=15, bandwidth=None,
                    angle_tol=22.5, dist="distance", vgm="variogram",
                    robust=False, dim=None):
    """Experimental directional variograms along arbitrary (2D/3D) axes.

    For each direction the pairs whose lag vector falls within ``angle_tol`` of
    that axis (and within the perpendicular ``bandwidth``) are selected and
    binned by lag distance -- the multi-dimensional analogue of the gstools
    ``vario_estimate(direction=...)`` workflow.

    Parameters
    ----------
    rawvgm : DataFrame
        Variogram cloud with ``d0..d{dim-1}`` lag components.
    directions : array-like, shape (n_dir, dim)
        Direction vectors (need not be normalised).  Use
        :func:`rotation_matrix_3d` to obtain a rotated orthonormal axis set.
    h_width : float, optional
        Fixed bin width applied to every direction.  When ``None`` (default)
        the width is computed **per direction** as ``max_proj / h_bins``,
        where ``max_proj`` is the largest projected lag along that axis.
        This gives equal resolution across all axes regardless of range, which
        matters for strongly anisotropic data where a single cutoff-derived
        width would leave the short-range axes with very few bins.
    h_bins : int
        Number of bins per direction when ``h_width`` is ``None``.
    bandwidth : float, optional
        Maximum perpendicular distance from the directional line.
    angle_tol : float, optional
        Angular tolerance (degrees) between a lag and the direction.

    Returns
    -------
    DataFrame
        Long format with columns ``direction`` (axis index), ``lag``
        (mean **projected** lag in the bin), ``variogram`` and ``count``.
    """
    directions = np.atleast_2d(np.asarray(directions, dtype=float))
    n_dir, ddim = directions.shape
    if dim is None:
        dim = ddim
    comp_cols = [f"d{k}" for k in range(dim)]
    dh = rawvgm[comp_cols].values
    h = rawvgm[dist].values
    g = rawvgm[vgm].values

    fixed_h_width = h_width  # None â†’ compute per-direction from projected lag
    cos_tol = np.cos(np.deg2rad(angle_tol))

    rows = []
    safe_h = np.where(h == 0, np.nan, h)
    for j in range(n_dir):
        u = directions[j] / np.linalg.norm(directions[j])
        proj_abs = np.abs(dh @ u)                  # projected lag along this axis
        cos_ang = proj_abs / safe_h                # |cos(angle to axis)|
        mask = cos_ang >= cos_tol
        if bandwidth is not None:
            perp = np.sqrt(np.maximum(h ** 2 - proj_abs ** 2, 0.0))
            mask &= perp <= bandwidth
        if not np.any(mask):
            continue
        hh, gg = proj_abs[mask], g[mask]          # project onto axis; calibrates per direction
        if fixed_h_width is None:
            hmax_j = np.nanmax(hh)
            if not np.isfinite(hmax_j) or hmax_j <= 0:
                continue
            this_h_width = hmax_j / h_bins
        else:
            this_h_width = fixed_h_width
        bins = (hh // this_h_width).astype(int)
        sub = pd.DataFrame({"bin": bins, "lag": hh, vgm: gg})
        grouped = sub.groupby("bin")
        agg = grouped.agg(lag=("lag", "mean"), count=(vgm, "count"))
        agg[vgm] = (grouped[vgm].apply(_cressie_gamma) if robust
                    else grouped[vgm].mean())
        agg["direction"] = j
        rows.append(agg.reset_index(drop=True))

    if not rows:
        return pd.DataFrame(columns=["direction", "lag", vgm, "count"])
    return pd.concat(rows, ignore_index=True)[["direction", "lag", vgm, "count"]]


# ---------------------------------------------------------------------------
# 6. Model fitting
# ---------------------------------------------------------------------------
