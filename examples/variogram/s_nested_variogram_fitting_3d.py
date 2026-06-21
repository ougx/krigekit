r"""
Estimating 3-D orientation, then fitting a nested variogram
===========================================================

:meth:`~krigekit.VariogramModel.fit_anisotropy` fits sills and anisotropic
ranges for a **fixed** orientation -- it bins along axes derived from the
supplied ``azimuth`` / ``dip`` / ``plunge``.  When the orientation is unknown,
estimate it first and then fit the shape.  This is the robust, two-step
workflow for a nested 3-D model:

1. **Estimate orientation** with :func:`~krigekit.estimate_aniso_angle`, a
   weighted-PCA of the near-origin lag cloud.  It returns all three angles
   ``(azimuth, dip, plunge)`` and the ``(anis1, anis2)`` minor/major axis
   ratios.
2. **Fit the model** with :meth:`~krigekit.VariogramModel.fit_anisotropy` at the
   estimated orientation.

A single realisation's empirical cloud is too noisy for the PCA, so several
realisations are **pooled** first (the same denoising trick as the 2-D nested
example).

What is and isn't well determined here is itself the lesson:

* The **orientation** (major axis especially) is recovered well -- reported as
  **axis-direction alignment** with the truth.  ``plunge`` -- the twist of the
  minor axes about the major axis -- is the least certain angle.
* The **total** sill and the fitted curves match the data, but the **split**
  of that sill and range between the two nested structures is only weakly
  identifiable: two spherical components with similar shapes trade off, so the
  per-component numbers need not match the truth even when the overall model
  does.  This non-uniqueness is intrinsic to nested fitting, not a solver bug.

The synthetic field has a known nested model, simulated by Cholesky
factorisation of the model covariance.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from krigekit import Kriging, VariogramModel
from krigekit.variogram import estimate_aniso_angle, rotation_matrix_3d

# %%
# Ground-truth nested model
# -------------------------
# Two strongly anisotropic spherical structures share one triaxial orientation.

# Ranges are kept well above the grid spacing (5) so they are resolvable, and
# the two structures are well separated so the nested fit can split them.
TRUE_ANGLES = dict(azimuth=35.0, dip=22.0, plunge=12.0)
TRUE_SHORT = dict(vtype="sph", sill=0.4, a_major=18.0, a_minor1=11.0, a_minor2=6.0)
TRUE_LONG = dict(vtype="sph", sill=0.6, a_major=45.0, a_minor1=27.0, a_minor2=14.0)

truth = VariogramModel()
truth.set_vgm(nugget=0.0, name="short spherical", **TRUE_SHORT, **TRUE_ANGLES)
truth.set_vgm(name="long spherical", **TRUE_LONG, **TRUE_ANGLES)

# %%
# Simulate and pool several realisations
# --------------------------------------
# The field is drawn on a regular grid by Cholesky factorisation of the model
# covariance.  Each realisation gives a noisy empirical cloud; pooling them
# denoises the near-origin structure enough for the orientation estimate.

axis = np.arange(0.0, 60.0, 5.0)
gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
grid = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

chol = np.linalg.cholesky(
    truth.calc_covariance(grid, grid, pairwise=True) + 1e-8 * np.eye(len(grid)))
rng = np.random.default_rng(3)

CUTOFF = 48.0          # beyond the long range so both structures are resolved
N_REAL = 4
clouds = []
for _ in range(N_REAL):
    value = chol @ rng.standard_normal(len(grid))
    sampler = VariogramModel()
    sampler.set_obs(grid, value)
    sampler.set_vgm("sph", sill=1.0, a_major=30.0, a_minor1=20.0, a_minor2=12.0)
    cloud = sampler.calc_experimental(cutoff=CUTOFF, calc_angle=True, verbose=False)
    clouds.append(cloud)
pooled = pd.concat(clouds, ignore_index=True)
print(f"pooled {len(pooled):,} pairs from {N_REAL} realisations")

# %%
# Step 1 -- estimate the orientation and anisotropy ratios
# -------------------------------------------------------
# ``estimate_aniso_angle`` returns ``(azimuth, dip, plunge)`` and the minor/major
# ratios ``(anis1, anis2)`` from a weighted PCA of the pooled cloud.

# Estimate orientation from near-origin lags (where the anisotropy is sharpest);
# a larger radius would reach past the short structure and blur the axes.
(azimuth, dip, plunge), (anis1, anis2) = estimate_aniso_angle(
    pooled, dim3d=True, r_max=14.0)

print(f"estimated orientation: azimuth={azimuth:.1f}  dip={dip:.1f}  "
      f"plunge={plunge:.1f}")
print(f"estimated ratios:      anis1={anis1:.2f}  anis2={anis2:.2f}")

true_axes = rotation_matrix_3d(
    TRUE_ANGLES["azimuth"], TRUE_ANGLES["dip"], TRUE_ANGLES["plunge"])
est_axes = rotation_matrix_3d(azimuth, dip, plunge)
print("axis-direction alignment with truth (1.0 = identical):")
for k, axis_name in enumerate(("major", "minor1", "minor2")):
    print(f"  {axis_name:6s}: {abs(float(true_axes[:, k] @ est_axes[:, k])):.3f}")

# %%
# Step 2 -- fit the nested ranges at the estimated orientation
# -----------------------------------------------------------
# Build the nested template with the estimated angles, seed the minor ranges
# from ``anis1`` / ``anis2``, then fit sills and the three ranges per component
# with the orientation held fixed.

model = VariogramModel()
p0, lower, upper = [], [], []
for name, major0, lo, hi in (("short spherical", 16.0, 8.0, 30.0),
                             ("long spherical", 42.0, 30.0, 70.0)):
    model.set_vgm("sph", sill=0.5, a_major=major0,
                  a_minor1=0.6 * major0, a_minor2=0.35 * major0,
                  azimuth=azimuth, dip=dip, plunge=plunge, name=name,
                  append=name != "short spherical")
    # seed minor ranges from generic ratios (the PCA anis1/anis2 are biased high)
    p0 += [0.5, major0, 0.6 * major0, 0.35 * major0]
    lower += [0.02, lo, 1.0, 0.5]
    upper += [2.0, hi, hi, hi]

directional = model.calc_directional_average(
    pooled, h_bins=16, cutoff=CUTOFF, angle_tol=22.0, include_minor2=True)
fit = model.fit_anisotropy(directional, p0=p0, bounds=(lower, upper),
                           include_minor2=True, fit_nugget=False,
                           weight_col="count", inplace=True, maxfev=50000)

print("\nNested ranges (fitted vs true -- per-component split is non-unique):")
for comp, spec in zip(model.structure.components, (TRUE_SHORT, TRUE_LONG)):
    print(f"  {comp.display_name:16s} sill={comp.sill:4.2f} ({spec['sill']:.2f})"
          f"  major/minor1/minor2={comp.a_major:5.1f}/{comp.a_minor1:5.1f}"
          f"/{comp.a_minor2:5.1f}  (true {spec['a_major']:.0f}/{spec['a_minor1']:.0f}"
          f"/{spec['a_minor2']:.0f})")
total_sill = sum(comp.sill for comp in model.structure.components)
true_total = TRUE_SHORT["sill"] + TRUE_LONG["sill"]
print(f"  total sill = {total_sill:.2f}  (true {true_total:.2f})   <- well determined")

# %%
# Empirical vs fitted model along the estimated axes
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(8.0, 5.0))
colors = {"major": "crimson", "minor1": "steelblue", "minor2": "seagreen"}
names = ["major", "minor1", "minor2"]
for k, axis_name in enumerate(names):
    sub = directional[directional["direction"] == k]
    ax.plot(sub["lag"], sub["variogram"], "o", ms=4, alpha=0.5,
            color=colors[axis_name], label=f"{axis_name} bins")
    h = np.linspace(0.0, CUTOFF, 200)
    curve = model.calc_variogram(np.zeros((h.size, 3)), est_axes[:, k] * h[:, None])
    ax.plot(h, curve, color=colors[axis_name], lw=2.0, label=f"{axis_name} model")

ax.set(xlabel="Lag distance", ylabel="Semivariogram",
       title="3-D orientation estimated, then nested ranges fitted")
ax.legend(ncol=3, fontsize=8)
fig.tight_layout()
plt.show()

# %%
# Transfer the fitted model to a kriging system
# ---------------------------------------------

k = Kriging(ndim=3, nvar=1, verbose=0)
k.set_obs(ivar=1, coord=grid, value=value, nmax=24)
model.apply_to(k, ivar=1, jvar=1)
print(f"\nTransferred {k._nvgm_struct[0, 0]} nested structures to the kriging engine")
