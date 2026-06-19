"""Variogram curves, maps, polar plots, and 3-D fence visualization."""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse

from .variogram_empirical import (
    _canonical_axis_sign,
    estimate_angle_angular_profile,
    estimate_aniso_angle,
)
from .variogram_fitting import (
    _normalise_model_specs,
    _uses_model_template,
    _vgmfunc_from_model_specs,
)
from .variogram_geometry import rotation_matrix_3d
from .variogram_kernels import calc_cov, resolve_model, vgmfunc


def plot_vgm(avgvgm, x_col=("distance", "mean"), y_col=("variogram", "mean"),
             models=("exponential",), parameters=(1, 100, 0),
             plot_data=True, plot_model=True, annotate=True, ax=None,
             plotkws_data=None, plotkws_model=None,
             xlabel="Lag", ylabel="Semivariogram"):
    """Plot an averaged variogram and/or a fitted model curve."""
    plotkws_data = plotkws_data or dict(color="darkgrey", marker=".", linewidth=0.5)
    plotkws_model = plotkws_model or dict(color="r", linewidth=1.5)
    has_nugget = len(parameters) % 2 != 0
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    x = avgvgm.loc[:, x_col]
    if plot_data:
        ax.plot(x, avgvgm.loc[:, y_col], **plotkws_data)
    if plot_model:
        xhat = np.linspace(0, x.max() * 1.1, 200)
        if _uses_model_template(models):
            yhat = _vgmfunc_from_model_specs(models, xhat, *parameters)
        else:
            yhat = vgmfunc(models, xhat, *parameters)
        ax.plot(xhat, yhat, **plotkws_model)
    if annotate and plot_model:
        specs = _normalise_model_specs(models)
        ms = "Model: " + "\t".join(resolve_model(m["vtype"])[:3].capitalize() for m in specs)
        ss = "\nSill : " + "\t".join(f"{parameters[i * 2]:.5g}" for i in range(len(specs)))
        rr = "\nRange: " + "\t".join(f"{parameters[i * 2 + 1]:.6g}" for i in range(len(specs)))
        nn = f"\nNugget: {parameters[-1]:.5g}" if has_nugget else ""
        ax.text(0.95, 0.05, ms + ss + rr + nn, ha="right", va="bottom",
                transform=ax.transAxes)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    return ax


def _grid_average(df, xcol, ycol, dx, dy, vgm_col, mirror=True):
    """Bin pairs onto a regular ``(x, y)`` grid and return ``(xx, yy, vv)``.

    When *mirror* is True the cloud is reflected through the origin first so the
    map is point-symmetric (as a variogram map should be).
    """
    rr = df[[xcol, ycol, vgm_col]].copy()
    if mirror:
        mir = rr.copy()
        mir[xcol] *= -1
        mir[ycol] *= -1
        rr = pd.concat([rr, mir], ignore_index=True)

    rr["ix"] = (rr[xcol] // dx).astype(int)
    rr["iy"] = (rr[ycol] // dy).astype(int)
    vmap = rr.groupby(["iy", "ix"], as_index=False)[vgm_col].mean()

    ix0, ix1 = vmap["ix"].min(), vmap["ix"].max()
    iy0, iy1 = vmap["iy"].min(), vmap["iy"].max()
    vv = np.full([iy1 - iy0 + 1, ix1 - ix0 + 1], np.nan)
    vv[vmap.iy - iy0, vmap.ix - ix0] = vmap[vgm_col]

    xx, yy = np.meshgrid(np.arange(ix0, ix1 + 2), np.arange(iy0, iy1 + 2))
    return dx * xx, dy * yy, vv


def plot_vgm_map(rawvgm, x="d0", y="d1", dist="distance", dx=None, dy=None,
                 cutoff=None, ax=None, angle_aniso=None, ellipse_aniso=None,
                 r_profile=None, angle=None, cmap="viridis_r", vmin=None,
                 vmax=None, vgm="variogram",
                 kws_cbar=None, title="Variogram Map", show_cbar=True,
                 show_title=True):
    """2D variogram map (pcolormesh) with optional anisotropy overlays."""
    kws_cbar = kws_cbar or {"label": "Variogram", "pad": 0.01}
    if dx is None:
        dx = np.quantile(rawvgm[x], 0.9) / 20
    if dy is None:
        dy = dx

    r0 = rawvgm.copy()
    if cutoff is not None:
        r0 = r0[r0[dist] <= cutoff].copy()

    xx, yy, vv = _grid_average(r0, x, y, dx, dy, vgm, mirror=True)
    if ax is None:
        nx, ny = vv.shape[1], vv.shape[0]
        _, ax = plt.subplots(figsize=(8, 8 / 1.2 / nx * ny))

    finite = vv[np.isfinite(vv)]
    cm = ax.pcolormesh(xx, yy, vv, cmap=cmap,
                       vmin=vmin if vmin is not None else np.quantile(finite, 0.05),
                       vmax=vmax if vmax is not None else np.quantile(finite, 0.95),
                       zorder=10)
    ax.axline((0, 0), (0, 1), color="w", linewidth=0.5, zorder=11)
    ax.axline((0, 0), (1, 0), color="w", linewidth=0.5, zorder=11)

    if angle_aniso is not None:
        rm = r0[dist].max() * 1.1
        xm = rm * np.sin(np.deg2rad(angle_aniso))
        ym = rm * np.cos(np.deg2rad(angle_aniso))
        ax.plot((xm, -xm), (ym, -ym), color="r", zorder=20)
        extra = f"Angle of maximum continuity: {angle_aniso:.1f}$^o$"
        title = f"{title}\n{extra}" if title else extra
    if ellipse_aniso is not None and angle_aniso is not None:
        ax.add_patch(Ellipse((0, 0), width=ellipse_aniso[0], height=ellipse_aniso[1],
                             angle=90 - angle_aniso, edgecolor="m",
                             facecolor="none", lw=2, zorder=30))
    if angle is not None and r_profile is not None:
        rr = r0.copy()
        rr[x] *= -1
        rr[y] *= -1
        rr[angle] = (rr[angle] + 180) % 360
        rr = pd.concat([rr, r0], ignore_index=True)
        prof = estimate_angle_angular_profile(rr, angle, dist, r_profile,
                                              da=10, a_tol=2.5, close=True)
        scale = prof[dist].mean() / prof[vgm].mean()
        ax.plot(scale * prof[vgm] * np.sin(np.deg2rad(prof[angle])),
                scale * prof[vgm] * np.cos(np.deg2rad(prof[angle])),
                color="m", label="Mean variogram\n(angular profile)", zorder=25)
        ax.legend(loc="upper left").set_zorder(50)
    if show_cbar:
        ax.figure.colorbar(cm, **kws_cbar)
    ax.set(aspect=1.0)
    if cutoff is not None:
        ax.set(xlim=(-cutoff, cutoff), ylim=(-cutoff, cutoff))
    if show_title:
        ax.set_title(title, fontsize="large")
    return ax


def plot_vgm_map_polar(rawvgm, angle="angle_h", dist="distance", da=None, dr=None,
                       cutoff=None, angle_aniso=None, r_profile=None,
                       cmap="viridis", vmin=None, vmax=None, vgm="variogram",
                       kws_cbar=None):
    """Variogram map in polar coordinates (azimuth vs lag distance)."""
    kws_cbar = kws_cbar or {"label": "Variogram", "pad": 0.1}
    if dr is None:
        dr = np.quantile(rawvgm[dist], 0.9) / 20
    if da is None:
        da = 10

    r0 = rawvgm[rawvgm[dist] <= cutoff].copy() if cutoff is not None else rawvgm.copy()
    r1 = r0.copy()
    r1[angle] = (r1[angle] + 180) % 360
    rr = pd.concat([r0, r1], ignore_index=True)

    rr["ia"] = (rr[angle] // da).astype(int)
    rr["ir"] = (rr[dist] // dr).astype(int)
    vmap = rr.groupby(["ir", "ia"], as_index=False)[vgm].mean()

    ia0, ia1 = vmap["ia"].min(), vmap["ia"].max()
    ir0, ir1 = vmap["ir"].min(), vmap["ir"].max()
    vv = np.full([ir1 - ir0 + 1, ia1 - ia0 + 1], np.nan)
    vv[vmap.ir - ir0, vmap.ia - ia0] = vmap[vgm]

    xx, yy = np.meshgrid(np.arange(ia0, ia1 + 2), np.arange(ir0, ir1 + 2))
    xx = np.deg2rad(da * xx)
    yy = dr * yy

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, polar=True)
    pcm = ax.pcolormesh(xx, yy, vv, cmap=cmap, shading="auto",
                        vmin=vmin, vmax=vmax, zorder=0)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.figure.colorbar(pcm, **kws_cbar)

    title = "Variogram Map (Polar, North=0$^o$)"
    if angle_aniso:
        ax.axvline(np.deg2rad(angle_aniso), color="r", zorder=20)
        ax.axvline(np.deg2rad(angle_aniso + 180), color="r", zorder=20)
        title += f"\nAngle of maximum continuity: {angle_aniso:.1f}$^o$"
    if r_profile is not None:
        prof = estimate_angle_angular_profile(rr, angle, dist, r_profile,
                                              da=10, a_tol=2.5, close=True)
        ax.plot(np.deg2rad(prof[angle]), prof[dist], color="m",
                label="Mean variogram\n(angular profile)")
        ax.legend(loc=(-0.07, -0.07))
    ax.set_title(title, fontsize="medium")
    return ax


def plot_vgm_anisotropy3d(eigenvalues, eigenvectors, ax=None):
    """Plot the three principal anisotropy axes as 3D arrows scaled by their
    eigenvalues, annotating the azimuth and dip of the major axis."""
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

    total = np.sqrt((eigenvalues ** 2).sum())
    for k, color in enumerate("rgb"):
        v = eigenvectors[:, k]
        ax.quiver(0, 0, 0, v[0], v[1], v[2], color=color,
                  length=eigenvalues[k] / total, arrow_length_ratio=0.03)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=25, azim=65)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])

    v = _canonical_axis_sign(eigenvectors[:, 0], dim3d=True)
    azimuth = (90 - np.degrees(np.arctan2(v[1], v[0]))) % 360
    # tilt below horizontal, positive down (matches the model ``dip`` field)
    dip = np.degrees(-np.arctan2(v[2], np.hypot(v[0], v[1])))
    ax.set_title(f"Anisotropy angles\nAzimuth: {azimuth:.1f}$^o$; Dip: {dip:.1f}$^o$")
    return ax


def _face_rgba(vv, cmap_obj, norm):
    """RGBA array for plot_surface facecolors; NaN bins are fully transparent."""
    rgba = cmap_obj(norm(np.where(np.isfinite(vv), vv, 0.0)))
    rgba[~np.isfinite(vv), 3] = 0.0
    return rgba


def _fill_nan_nearest(vv, support_mask=None):
    """Replace NaN cells with their nearest valid neighbour (display only).

    When ``support_mask`` is supplied, only NaN cells inside that mask are
    filled.  Cells outside the mask remain NaN, which prevents display-only
    filling from extrapolating beyond the lag-distance support.
    """
    valid = np.isfinite(vv)
    if valid.all() or not valid.any():
        return vv
    if support_mask is None:
        support_mask = np.ones_like(valid, dtype=bool)
    else:
        support_mask = np.asarray(support_mask, dtype=bool)
        if support_mask.shape != valid.shape:
            raise ValueError("support_mask shape must match vv")
    fillable = support_mask & ~valid
    if not fillable.any():
        return vv
    from scipy.ndimage import distance_transform_edt
    _, idx = distance_transform_edt(~valid, return_indices=True)
    out = vv.copy()
    out[fillable] = vv[tuple(idx[:, fillable])]
    return out


def _cell_center_radius(xx, yy, zz):
    """Euclidean radius at each surface cell centre."""
    cx = 0.25 * (xx[:-1, :-1] + xx[1:, :-1] + xx[1:, 1:] + xx[:-1, 1:])
    cy = 0.25 * (yy[:-1, :-1] + yy[1:, :-1] + yy[1:, 1:] + yy[:-1, 1:])
    cz = 0.25 * (zz[:-1, :-1] + zz[1:, :-1] + zz[1:, 1:] + zz[:-1, 1:])
    return np.sqrt(cx * cx + cy * cy + cz * cz)


def _axis_bin(u, dx, dz):
    """Return dz for axes more than 50% vertical, dx otherwise."""
    return dz if abs(float(u[2])) > 0.5 else dx


def _rotated_fence(df, u1, u2, u_normal, d1, d2, bandwidth, vgm_col="variogram"):
    """Bin pairs near the plane spanned by ``u1 Ã— u2`` into a 2-D grid.

    Parameters
    ----------
    u1, u2 : array-like, shape (3,)
        Unit vectors in the fence plane.
    u_normal : array-like, shape (3,)
        Fence normal; pairs with ``|dh @ u_normal| > bandwidth`` are excluded.
    d1, d2 : float
        Bin sizes along u1 and u2.
    bandwidth : float
        Half-thickness slab for pair selection.

    Returns
    -------
    tuple ``(X, Y, Z, vv)`` or None
        3-D mesh coordinates and binned variogram values; None if empty.
    """
    u1 = np.asarray(u1, dtype=float); u1 /= np.linalg.norm(u1)
    u2 = np.asarray(u2, dtype=float); u2 /= np.linalg.norm(u2)
    u_normal = np.asarray(u_normal, dtype=float); u_normal /= np.linalg.norm(u_normal)
    dh = df[["d0", "d1", "d2"]].values
    mask = np.abs(dh @ u_normal) <= bandwidth
    if not mask.any():
        return None
    r1 = (dh @ u1)[mask]
    r2 = (dh @ u2)[mask]
    sub = pd.DataFrame({"_r1": r1, "_r2": r2,
                        vgm_col: df[vgm_col].values[mask]})
    rr, ss, vv = _grid_average(sub, "_r1", "_r2", d1, d2, vgm_col, mirror=True)
    return (rr * u1[0] + ss * u2[0],
            rr * u1[1] + ss * u2[1],
            rr * u1[2] + ss * u2[2],
            vv)


def plot_vgm_map3d(rawvgm, x="d0", y="d1", z="d2", dist="distance",
                   dx=None, dy=None, dz=None, cutoff=None, ax=None,
                   angle_aniso=None, rotate_fences=False,
                   vgm="variogram", cmap="viridis_r", vmin=None, vmax=None,
                   bandwidth_factor=2.0, n_fences=2, fill_nan=False,
                   kws_cbar=None, title="3D Variogram Map",
                   show_cbar=True, show_title=True):
    """3D variogram map shown as orthogonal fence sections.

    Up to three mutually orthogonal fences are drawn.  By default they are
    aligned with the world coordinate axes (XY, XZ, YZ), which makes the
    anisotropy angles easy to read off the plot.  Set ``rotate_fences=True``
    to align the fences with the fitted model axes instead.

    World-axis-aligned fences (``rotate_fences=False``, default):

    * **Fence A** (always) â€” horizontal XY plane.
    * **Fence B** (``n_fences >= 2``) â€” vertical XZ (Eastâ€“West) plane.
    * **Fence C** (``n_fences >= 3``) â€” vertical YZ (Northâ€“South) plane.

    Model-axis-aligned fences (``rotate_fences=True``):

    * **Fence A** â€” minor1 Ã— minor2 plane (normal = major axis).
    * **Fence B** (``n_fences >= 2``) â€” major Ã— minor2 plane (dip section).
    * **Fence C** (``n_fences >= 3``) â€” major Ã— minor1 plane (plunge section).

    Parameters
    ----------
    dx, dy : float, optional
        Reference horizontal bin size.  Defaults to ``cutoff / 10``.  ``dy``
        is accepted for API compatibility but is ignored; use ``dx`` only.
    dz : float, optional
        Reference vertical bin size.  Defaults to the 90th-percentile of
        ``|d2|`` divided by 10, which naturally tracks the (typically finer)
        vertical sampling scale.
    angle_aniso : None, float, or tuple, optional
        Model orientation.  A scalar is interpreted as azimuth (degrees);
        a 2-tuple as ``(azimuth, dip)``; a 3-tuple as
        ``(azimuth, dip, plunge)``.  Only used when ``rotate_fences=True``
        (or to label the title).  ``None`` estimates the horizontal azimuth
        from the cloud with :func:`estimate_aniso_angle`.
    rotate_fences : bool, optional
        If ``False`` (default) fences align with the world X/Y/Z axes.
        If ``True`` fences are rotated to the model anisotropy axes defined
        by ``angle_aniso``.
    bandwidth_factor : float, optional
        Each fence selects pairs within ``bandwidth_factor`` Ã— (normal-axis
        bin size) of the fence plane.  Default 2.
    n_fences : {1, 2, 3}, optional
        Number of orthogonal fences to draw (A only, A+B, or A+B+C).
        Default 2.  Three fences can be informative but crowded.
    fill_nan : bool, optional
        If ``True``, empty bins are filled with their nearest valid neighbour
        before rendering, but only inside the cutoff/max-lag radius.  Useful
        when data are sparse and bins patchy.  Default ``False``.
    """
    kws_cbar = kws_cbar or {"label": "Variogram", "pad": 0.05}

    df = rawvgm[rawvgm[dist] <= cutoff].copy() if cutoff is not None else rawvgm.copy()

    # Horizontal bin size: tied to the search radius.
    if dx is None:
        hmax = cutoff if cutoff is not None else float(rawvgm[dist].quantile(0.9))
        dx = hmax / 10
    # Vertical bin size: tied to the actual vertical lag spread (typically finer).
    if dz is None:
        dz = max(abs(float(df[z].quantile(0.9))), dx * 0.1) / 10
    fill_radius = float(cutoff) if cutoff is not None else float(df[dist].max())

    # --- Parse model orientation ------------------------------------------
    if angle_aniso is None:
        try:
            ang = estimate_aniso_angle(df, x, y, dist, vgm, dim3d=False)[0]
        except Exception:
            ang = None
        azimuth = float(ang[0]) if ang is not None else 0.0
        dip     = float(ang[1]) if ang is not None and len(ang) > 1 else 0.0
        plunge  = 0.0
        _angles_known = ang is not None
    elif np.ndim(angle_aniso) == 0:
        azimuth, dip, plunge = float(angle_aniso), 0.0, 0.0
        _angles_known = True
    else:
        a = tuple(angle_aniso)
        azimuth = float(a[0])
        dip     = float(a[1]) if len(a) > 1 else 0.0
        plunge  = float(a[2]) if len(a) > 2 else 0.0
        _angles_known = True

    # --- Step 1: compute fence grids (no drawing yet) ---------------------
    if rotate_fences:
        # Model-axis-aligned fences â€” rotated to fitted anisotropy orientation.
        R = rotation_matrix_3d(azimuth, dip, plunge)
        u_major, u_minor1, u_minor2 = R[:, 0], R[:, 1], R[:, 2]
        d_maj  = _axis_bin(u_major,  dx, dz)
        d_min1 = _axis_bin(u_minor1, dx, dz)
        d_min2 = _axis_bin(u_minor2, dx, dz)
        # Fence A: minor1â€“minor2 plane, normal = major
        fence_A = _rotated_fence(df, u_minor1, u_minor2, u_major,
                                 d_min1, d_min2,
                                 bandwidth=d_maj * bandwidth_factor, vgm_col=vgm)
        # Fence B: majorâ€“minor2 plane (dip section), normal = minor1
        fence_B = None
        if n_fences >= 2:
            fence_B = _rotated_fence(df, u_major, u_minor2, u_minor1,
                                     d_maj, d_min2,
                                     bandwidth=d_min1 * bandwidth_factor, vgm_col=vgm)
        # Fence C: majorâ€“minor1 plane (plunge section), normal = minor2
        fence_C = None
        if n_fences >= 3:
            fence_C = _rotated_fence(df, u_major, u_minor1, u_minor2,
                                     d_maj, d_min1,
                                     bandwidth=d_min2 * bandwidth_factor, vgm_col=vgm)
    else:
        # World-axis-aligned fences â€” easy to relate to map coordinates.
        _X = np.array([1., 0., 0.])
        _Y = np.array([0., 1., 0.])
        _Z = np.array([0., 0., 1.])
        # Fence A: horizontal XY plane
        fence_A = _rotated_fence(df, _X, _Y, _Z, dx, dx,
                                 bandwidth=dz * bandwidth_factor, vgm_col=vgm)
        # Fence B: vertical XZ plane (Eastâ€“West section)
        fence_B = None
        if n_fences >= 2:
            fence_B = _rotated_fence(df, _X, _Z, _Y, dx, dz,
                                     bandwidth=dx * bandwidth_factor, vgm_col=vgm)
        # Fence C: vertical YZ plane (Northâ€“South section)
        fence_C = None
        if n_fences >= 3:
            fence_C = _rotated_fence(df, _Y, _Z, _X, dx, dz,
                                     bandwidth=dx * bandwidth_factor, vgm_col=vgm)

    # --- Step 2: colour scale from averaged values, not raw pairs ----------
    cmap_obj = plt.get_cmap(cmap)
    all_avg = np.concatenate([
        s[3][np.isfinite(s[3])].ravel()
        for s in (fence_A, fence_B, fence_C) if s is not None
    ])
    if vmin is None:
        vmin = float(np.nanmin(all_avg)) if len(all_avg) else 0.0
    if vmax is None:
        vmax = float(np.nanquantile(all_avg, 0.95)) if len(all_avg) else 1.0
    norm = Normalize(vmin, vmax)

    # --- Step 3: draw -------------------------------------------------------
    # Merge all fence faces into a single Poly3DCollection so matplotlib
    # depth-sorts all polygons together â€” fixes z-order artefacts when rotating.
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if ax is None:
        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection="3d")

    alphas = [1.0, 0.85, 0.70]
    all_verts, all_fc = [], []
    all_xx, all_yy, all_zz = [], [], []

    for surf, alpha in zip((fence_A, fence_B, fence_C), alphas):
        if surf is None:
            continue
        xx, yy, zz, vv = surf
        all_xx.append(xx.ravel())
        all_yy.append(yy.ravel())
        all_zz.append(zz.ravel())
        if fill_nan:
            support = _cell_center_radius(xx, yy, zz) <= fill_radius
            vv = _fill_nan_nearest(vv, support_mask=support)
        fc = _face_rgba(vv, cmap_obj, norm)  # (R, C, 4)
        nr, nc = vv.shape
        for i in range(nr):
            for j in range(nc):
                if not np.isfinite(vv[i, j]):
                    continue
                all_verts.append([
                    (xx[i,   j],   yy[i,   j],   zz[i,   j]),
                    (xx[i+1, j],   yy[i+1, j],   zz[i+1, j]),
                    (xx[i+1, j+1], yy[i+1, j+1], zz[i+1, j+1]),
                    (xx[i,   j+1], yy[i,   j+1], zz[i,   j+1]),
                ])
                c = fc[i, j].copy()
                c[3] = alpha
                all_fc.append(c)

    # Poly3DCollection doesn't auto-scale the axes; compute limits from the grids.
    xs = np.concatenate(all_xx) if all_xx else np.array([-dx, dx])
    ys = np.concatenate(all_yy) if all_yy else np.array([-dx, dx])
    zs = np.concatenate(all_zz) if all_zz else np.array([-dz, dz])

    if all_verts:
        coll = Poly3DCollection(all_verts, facecolors=all_fc,
                                edgecolors=all_fc, linewidths=0.3,
                                antialiased=False)
        ax.add_collection3d(coll)
        ax.set_xlim(xs.min(), xs.max())
        ax.set_ylim(ys.min(), ys.max())
        ax.set_zlim(zs.min(), zs.max())

    # --- Anisotropy axis lines projected onto each fence plane --------------
    # Each line lies ON its fence (z=0 for XY, y=0 for XZ, x=0 for YZ) so
    # you can read the azimuth directly off the horizontal fence and the dip
    # off the vertical fences.
    if not rotate_fences and _angles_known:
        R = rotation_matrix_3d(azimuth, dip, plunge)
        u_major, u_minor1, u_minor2 = R[:, 0], R[:, 1], R[:, 2]
        rm = max(float(np.abs(xs).max()), float(np.abs(ys).max()),
                 float(np.abs(zs).max()))

        def _proj_seg(u, drop_idx):
            """Project unit vector u onto the plane normal to axis drop_idx.
            Returns plot (xs, ys, zs) tuple, or None if projection is degenerate."""
            p = np.array(u, dtype=float)
            p[drop_idx] = 0.0
            n = np.linalg.norm(p)
            if n < 0.15:   # axis nearly perpendicular to this fence â€” skip
                return None
            p = p / n * rm
            return ([-p[0], p[0]], [-p[1], p[1]], [-p[2], p[2]])

        _az_lbl = f"az={azimuth:.0f}Â°, dip={dip:.0f}Â°"
        if n_fences >= 3 or abs(plunge) > 0.1:
            _az_lbl += f", pl={plunge:.0f}Â°"

        shown = set()

        def _draw_proj(u, drop_idx, color, label):
            seg = _proj_seg(u, drop_idx)
            if seg is None:
                return
            kw = dict(color=color, linewidth=1.5, zorder=200)
            if label and label not in shown:
                kw["label"] = label
                shown.add(label)
            ax.plot(*seg, **kw)

        # Project major axis onto each fence plane to show azimuth (XY) and dip (XZ/YZ).
        _draw_proj(u_major, 2, "r", f"major ({_az_lbl})")   # Fence A: XY
        if n_fences >= 2:
            _draw_proj(u_major, 1, "r", f"major ({_az_lbl})")  # Fence B: XZ
        if n_fences >= 3:
            _draw_proj(u_major, 0, "r", f"major ({_az_lbl})")  # Fence C: YZ

        if shown:
            ax.legend(fontsize=8, loc="upper right")

    if show_cbar:
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        mappable.set_array([])
        ax.figure.colorbar(mappable, ax=ax, **kws_cbar)

    ax.set_xlabel("X lag")
    ax.set_ylabel("Y lag")
    ax.set_zlabel("Z lag")
    if show_title and title:
        if rotate_fences:
            extra = f"Azimuth {azimuth:.1f}Â°  Dip {dip:.1f}Â°  Plunge {plunge:.1f}Â°"
            ax.set_title(f"{title}\n{extra}", fontsize="large")
        else:
            ax.set_title(title, fontsize="large")
    return ax


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    hh = np.linspace(0, 1.5, 151)

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, f in _vgmfunc.items():
        ax.plot(hh, f(hh), label=name)
    ax.legend()
    ax.set_title("Normalised variogram models")

    fig, ax = plt.subplots(figsize=(10, 6))
    for name in _vgmfunc:
        ax.plot(hh, calc_cov(name, hh), label=name)
    ax.legend()
    ax.set_title("Normalised covariance models")
    plt.show()
