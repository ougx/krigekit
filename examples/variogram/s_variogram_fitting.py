r"""
Nested variogram fitting for an indicator variable
==================================================

This example fits a two-structure indicator variogram to the
``lithofacies.csv`` outcrop data.  The response variable is the binary
indicator for gravel:

.. math::

   I_G(\mathbf{x}) =
   \begin{cases}
   1, & \text{gravel at } \mathbf{x} \\
   0, & \text{otherwise.}
   \end{cases}

The example demonstrates a more realistic modelling workflow:

* compute and cache an empirical variogram cloud,
* average the cloud into directional major/minor lag bins,
* fit two nested spherical structures with count weights and fixed orientation,
* manually round the fitted parameters to a cleaner practical model,
* apply the fitted model to ordinary kriging of the indicator.
"""

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from krigekit import Kriging, VariogramModel


def _find_data_dir():
    """Locate test_data when run as a script or through Sphinx-Gallery."""
    candidates = []
    if "__file__" in globals():
        candidates.append(Path(__file__).resolve().parents[2] / "test_data")
    cwd = Path.cwd().resolve()
    candidates.extend([cwd / "test_data", cwd.parent / "test_data", cwd.parent.parent / "test_data"])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate the test_data directory.")


DATA_DIR = _find_data_dir()

CAT_COL = "lithologic_code"
X_COL = "x_pixel"
Y_COL = "y_pixel0"
TARGET_CAT = "G"

NX, NY = 90, 32
AZIMUTH = 90.0
INITIAL_ANIS1 = 0.16
CAT_COLORS = {"C": "#8B5E3C", "Cl": "#4682B4", "FS": "#DAA520", "G": "#696969"}

data = pd.read_csv(DATA_DIR / "lithofacies.csv")
img = plt.imread(DATA_DIR / "lithofacies.png")
img_h, img_w = img.shape[:2]

obs_coord = data[[X_COL, Y_COL]].to_numpy(dtype=float)
obs_cats = data[CAT_COL].to_numpy()
obs_value = (obs_cats == TARGET_CAT).astype(float)
sample_var = float(np.var(obs_value, ddof=1))

gx, gy = np.meshgrid(
    np.linspace(0.0, img_w, NX),
    np.linspace(0.0, img_h, NY),
)
grid_coord = np.column_stack([gx.ravel(), gy.ravel()])

cat_labels = sorted(data[CAT_COL].unique())
patches = [mpatches.Patch(color=CAT_COLORS[c], label=c) for c in cat_labels]

# %%
# Compute and fit a nested variogram
# ----------------------------------
# The two ``set_vgm`` calls define the model template: a short-range spherical
# structure plus a longer-range spherical structure.  The azimuth is fixed from
# the visible bedding direction, then ``calc_directional_average`` filters the
# cloud into major/minor axes.  ``fit_anisotropy`` updates partial sills, major
# ranges, minor ranges, and a single trailing nugget using the flat order::
#
#     sill_short, major_short, minor_short,
#     sill_long, major_long, minor_long, nugget

model = VariogramModel()
model.set_obs(obs_coord, obs_value)
model.set_vgm(vtype="sph", nugget=0.0, sill=0.12, a_major=100.0,
              a_minor1=100.0 * INITIAL_ANIS1, azimuth=AZIMUTH)
model.set_vgm(vtype="sph", nugget=0.0, sill=0.06, a_major=800.0,
              a_minor1=800.0 * INITIAL_ANIS1, azimuth=AZIMUTH)

model.calc_experimental(
    cutoff=2000.0,
    calc_angle=True,
    verbose=False,
)
dir_avg = model.calc_directional_average(
    h_width=100.0,
    cutoff=1800.0,
    angle_tol=12.5,
    robust=False,
)

model.fit_anisotropy(
    dir_avg,
    p0=(0.12, 100.0, 16.0, 0.06, 800.0, 128.0, 0.02),
    bounds=([0.0, 50.0, 5.0, 0.0, 400.0, 20.0, 0.0],
            [0.5, 1000.0, 400.0, 0.5, 2000.0, 1000.0, 0.2]),
    weight_col="count",
    inplace=True,
    maxfev=50000,
)

fit_params = model.params_.copy()

# A common modelling step is to round a numerical optimum to an interpretable
# practical model before using it in kriging.
model.set_anisotropic_params([
    round(fit_params[0], 3),
    round(fit_params[1] / 10.0) * 10.0,
    round(fit_params[2], 3),
    round(fit_params[3], 3),
    round(fit_params[4] / 10.0) * 10.0,
    round(fit_params[5] / 10.0) * 10.0,
    round(fit_params[6], 3),
])

structure_ratios = [
    comp.a_minor1 / comp.a_major
    for comp in model.structure.components
]
search_anis1 = max(structure_ratios)

print(
    "Optimized nested model: "
    f"short sill={fit_params[0]:.4f}, "
    f"short major={fit_params[1]:.1f}, short minor={fit_params[2]:.1f}, "
    f"long sill={fit_params[3]:.4f}, "
    f"long major={fit_params[4]:.1f}, long minor={fit_params[5]:.1f}, "
    f"nugget={fit_params[6]:.4f}"
)
print(
    "Rounded nested model:   "
    f"short sill={model.params_[0]:.4f}, "
    f"short major={model.params_[1]:.1f}, short minor={model.params_[2]:.1f}, "
    f"long sill={model.params_[3]:.4f}, "
    f"long major={model.params_[4]:.1f}, long minor={model.params_[5]:.1f}, "
    f"nugget={model.params_[6]:.4f}"
)
print(
    "Anisotropy:             "
    f"azimuth={AZIMUTH:.0f} deg, search minor/major={search_anis1:.2f}, "
    f"structure ratios="
    f"{structure_ratios[0]:.2f}, {structure_ratios[1]:.2f}"
)
# %%
# Plot observations and the fitted nested model
# ---------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout="constrained")

ax = axes[0]
ax.imshow(img, extent=[0, img_w, img_h, 0], aspect="auto")
for c in cat_labels:
    mask = obs_cats == c
    ax.scatter(
        obs_coord[mask, 0],
        obs_coord[mask, 1],
        c=CAT_COLORS[c],
        s=30,
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
    )
ax.set_title("Lithofacies observations")
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels, down)")
ax.legend(handles=patches, loc="lower right", fontsize=8, ncol=4, framealpha=0.85)

ax = axes[1]
model.plot_map(
    ax=ax,
    cutoff=1800.0,
    dx=100.0,
    dy=100.0,
    show_cbar=False,
    title="Variogram map with fitted anisotropy",
)

ax = axes[2]
axis_vectors = {
    "major": np.array([np.sin(np.deg2rad(AZIMUTH)), np.cos(np.deg2rad(AZIMUTH))]),
    "minor1": np.array([np.cos(np.deg2rad(AZIMUTH)), -np.sin(np.deg2rad(AZIMUTH))]),
}
axis_colors = {"major": "crimson", "minor1": "steelblue"}
for axis_name, axis_vec in axis_vectors.items():
    sub = dir_avg.loc[dir_avg["axis"] == axis_name]
    ax.plot(
        sub["lag"],
        sub["variogram"],
        marker="o",
        linestyle="none",
        markersize=5.0,
        color=axis_colors[axis_name],
        alpha=0.65,
        label=f"{axis_name} bins",
    )
    h = np.linspace(0.0, float(sub["lag"].max()) * 1.1, 200)
    coord0 = np.zeros((len(h), 2))
    coord1 = axis_vec * h[:, None]
    ax.plot(
        h,
        model.calc_variogram(coord0, coord1),
        color=axis_colors[axis_name],
        lw=2.0,
        label=f"{axis_name} model",
    )
ax.axhline(sample_var, color="0.45", ls="--", lw=1.0, label="Indicator variance")
ax.set_xlabel("Lag")
ax.set_ylabel("Semivariogram")
ax.set_title(f"Directional variograms for indicator I({TARGET_CAT})")
ax.legend(fontsize=8)

plt.show()

# %%
# Krige the indicator with the nested model
# -----------------------------------------
# Ordinary kriging of a binary indicator gives a smooth indicator estimate.
# It is often interpreted as a probability-like surface, though ordinary
# kriging can produce small values outside ``[0, 1]``.

k = Kriging()
k.set_obs(ivar=1, coord=obs_coord, value=obs_value)
k.set_grid(coord=grid_coord)
model.apply_to(k, ivar=1, jvar=1)
k.set_search(ivar=1, anis1=search_anis1, azimuth=AZIMUTH, nmax=20)
k.solve()
out = k.get_result_df()

estimate = out["estimate"].to_numpy().reshape(NY, NX)
variance = out["variance"].to_numpy().reshape(NY, NX)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")

ax = axes[0]
im = ax.imshow(
    estimate,
    origin="upper",
    extent=[0, img_w, img_h, 0],
    cmap="viridis",
    vmin=0.0,
    vmax=1.0,
    aspect="auto",
)
ax.scatter(obs_coord[:, 0], obs_coord[:, 1], c=obs_value, cmap="gray_r",
           vmin=0.0, vmax=1.0, s=12, edgecolors="white", linewidths=0.4)
plt.colorbar(im, ax=ax, label=f"Estimate of I({TARGET_CAT})", shrink=0.86)
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels, down)")
ax.set_title("Indicator kriging estimate")

ax = axes[1]
im = ax.imshow(
    variance,
    origin="upper",
    extent=[0, img_w, img_h, 0],
    cmap="Reds",
    vmin=0.0,
    vmax=max(sample_var, float(np.nanmax(variance))),
    aspect="auto",
)
ax.scatter(obs_coord[:, 0], obs_coord[:, 1], c="k", s=10, alpha=0.55)
plt.colorbar(im, ax=ax, label="Variance", shrink=0.86)
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels, down)")
ax.set_title("Kriging variance")

plt.show()
