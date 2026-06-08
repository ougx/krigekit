Variogram models
================

Parameters
----------

Variograms are set with ``set_vgm()`` using keyword arguments:

.. code-block:: python

   k.set_vgm(
       ivar=1, jvar=1,
       vtype="sph",
       nugget=0.05,
       sill=0.45,
       a_major=500.0,
       a_minor1=200.0,
       a_minor2=200.0,   # 3-D only; defaults to a_minor1
       azimuth=45.0,
       dip=0.0,
       plunge=0.0,
   )

.. list-table::
   :header-rows: 1
   :widths: 12 12 76

   * - Parameter
     - Default
     - Description
   * - ``vtype``
     - *(required)*
     - Model type code (see table below)
   * - ``nugget``
     - ``0.0``
     - Nugget effect (discontinuity at origin)
   * - ``sill``
     - ``1.0``
     - Partial sill — variance contributed by this structure
   * - ``a_major``
     - ``1.0``
     - Range along the major (longest) axis; see per-model meaning below
   * - ``a_minor1``
     - ``a_major``
     - Range along the first minor axis (defaults to isotropic)
   * - ``a_minor2``
     - ``a_minor1``
     - Range along the vertical axis (3-D only)
   * - ``azimuth``
     - ``0.0``
     - Azimuth of major axis, degrees clockwise from North
   * - ``dip``
     - ``0.0``
     - Dip angle, degrees positive downward
   * - ``plunge``
     - ``0.0``
     - Plunge angle, degrees
   * - ``append``
     - ``True``
     - ``True`` appends a nested structure; ``False`` replaces the current model
   * - ``product``
     - ``False``
     - ``True`` multiplies with the preceding structure (non-additive nesting)

The dimensionless lag is :math:`r = h / a_\text{major}` (after anisotropy
scaling).  The covariance is :math:`C(h) = \text{sill} \times \rho(r)` where
:math:`\rho(r)` is the correlation function listed in the table below.

Supported model types
---------------------

.. list-table::
   :header-rows: 1
   :widths: 6 14 80

   * - Code
     - Name
     - Correlation :math:`\rho(r)`
   * - ``nug``
     - Pure nugget
     - :math:`0` for :math:`r > 0`; 1 at origin
   * - ``sph``
     - Spherical
     - :math:`1 - \tfrac{3}{2}r + \tfrac{1}{2}r^3` for :math:`r < 1`, else :math:`0`
   * - ``exp``
     - Exponential
     - :math:`\exp(-3r)`
   * - ``gau``
     - Gaussian
     - :math:`\exp(-3.0625\,r^2)`
   * - ``hol``
     - Hole effect
     - :math:`\cos(\pi r)`
   * - ``pow``
     - Power
     - :math:`1 - r^{1.5}` for :math:`r < 1`, else :math:`0`
   * - ``bsq``
     - Bi-square
     - :math:`(1 - r^2)^2` for :math:`r < 1`, else :math:`0`
   * - ``cir``
     - Circular
     - :math:`1 - \tfrac{2}{\pi}\!\left(r\sqrt{1-r^2} + \arcsin r\right)` for :math:`r < 1`, else :math:`0`
   * - ``lin``
     - Linear
     - :math:`1 - r` for :math:`r < 1`, else :math:`0`
   * - ``cyc``
     - GP periodic
     - :math:`\exp\!\left(-2\sin^2(\pi r)\right)`
   * - ``dco``
     - Damped cosine
     - :math:`\exp(-3r)\cos(\pi r)`

For ``sph``, ``exp``, and ``gau`` the **practical range** convention is used:
:math:`\rho(1) \approx 0` (spherical reaches exactly 0; exponential and
Gaussian reach :math:`\approx 5\%`), so ``a_major`` is the distance at which
spatial correlation is effectively zero.

.. plot::
   :include-source: false

   import numpy as np
   import matplotlib.pyplot as plt

   r = np.linspace(0, 1.5, 400)

   models = [
       ("sph", np.where(r < 1, 1 - 1.5*r + 0.5*r**3, 0)),
       ("exp", np.exp(-3*r)),
       ("gau", np.exp(-3.0625*r**2)),
       ("lin", np.where(r < 1, 1 - r, 0)),
       ("hol", np.cos(np.pi*r)),
       ("dco", np.exp(-3*r)*np.cos(np.pi*r)),
       ("cyc", np.exp(-2*np.sin(np.pi*r)**2)),
   ]

   fig, ax = plt.subplots(figsize=(7.5, 3.4))
   colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
             "#9467bd", "#8c564b", "#e377c2"]
   for (name, rho), col in zip(models, colors):
       ax.plot(r, rho, lw=1.8, label=name, color=col)
   ax.axhline(0, color="k", lw=0.6, ls=":")
   ax.axvline(1, color="gray", lw=0.6, ls=":", alpha=0.6)
   ax.set_xlabel("Normalised lag  r = h / a_major")
   ax.set_ylabel("ρ(r) = C(h) / sill")
   ax.set_xlim(0, 1.5)
   ax.set_ylim(-0.25, 1.05)
   ax.legend(ncol=4, fontsize=9, loc="upper right")
   ax.set_title("Correlation functions  (grey dotted line at r = 1 = practical range)")
   plt.tight_layout()
   plt.show()

.. plot::
   :include-source: false

   import numpy as np
   import matplotlib.pyplot as plt

   r  = np.linspace(0, 1.5, 400)
   rc = np.clip(r, 0, 1)

   models = {
       "sph": np.where(r < 1, 1 - 1.5*r + 0.5*r**3, 0),
       "exp": np.exp(-3*r),
       "gau": np.exp(-3.0625*r**2),
       "hol": np.cos(np.pi*r),
       "pow": np.where(r < 1, 1 - r**1.5, 0),
       "bsq": np.where(r < 1, (1 - r**2)**2, 0),
       "cir": np.where(r < 1, 1 - 2/np.pi*(rc*np.sqrt(1 - rc**2) + np.arcsin(rc)), 0),
       "lin": np.where(r < 1, 1 - r, 0),
       "cyc": np.exp(-2*np.sin(np.pi*r)**2),
       "dco": np.exp(-3*r)*np.cos(np.pi*r),
   }

   fig, axes = plt.subplots(2, 5, figsize=(12, 4.8), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.45, "wspace": 0.12})
   for ax, (name, rho) in zip(axes.flat, models.items()):
       ax.plot(r, rho, lw=1.8, color="steelblue")
       ax.axhline(0, color="k", lw=0.6, ls="--")
       ax.axvline(1, color="gray", lw=0.6, ls=":", alpha=0.7)
       ax.set_title(f"``{name}``", fontsize=10)
       ax.set_xlim(0, 1.5)
       ax.set_ylim(-0.25, 1.05)
   for ax in axes[1]:
       ax.set_xlabel("r = h / a", fontsize=8.5)
   for ax in axes[:, 0]:
       ax.set_ylabel("ρ(r)", fontsize=8.5)
   fig.suptitle("Correlation functions ρ(r) — dotted line at r = 1 (practical range)",
                fontsize=11)
   plt.show()

Model notes
-----------

Hole effect (``hol``)
~~~~~~~~~~~~~~~~~~~~~

.. math::

   C(h) = \text{sill} \cdot \cos\!\left(\frac{\pi h}{a}\right)

A pure cosine with period :math:`2a`.  Correlation is zero at :math:`h = a/2`,
reaches its most negative value (:math:`-\text{sill}`) at :math:`h = a`, and
returns to :math:`+\text{sill}` at :math:`h = 2a`.  Valid in 1-D and 2-D; use
with caution in 3-D (not guaranteed positive-definite).  The oscillation never
damps, so the kriging matrix can be indefinite — the SSYTRF fallback solver
handles this.

Damped cosine (``dco``)
~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   C(h) = \text{sill} \cdot \exp\!\left(\frac{-3h}{a}\right) \cos\!\left(\frac{\pi h}{a}\right)

An oscillating covariance that decays exponentially with distance.
**``a_major`` controls both the decay length and the oscillation period
simultaneously** — the first zero-crossing is at :math:`h = a/2` and the
first negative lobe peaks at :math:`h = a`.  Valid and positive-definite in
all dimensions.  Suitable when the cyclic pattern weakens over long distances,
e.g. annual signals in a climate record with increasing measurement gaps.

GP periodic / ExpSineSquared (``cyc``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   C(h) = \text{sill} \cdot \exp\!\left(-2\sin^2\!\left(\frac{\pi h}{a}\right)\right)

**``a_major`` is the period** — correlation returns to ``sill`` at every
integer multiple of ``a_major``.  The model is always positive (correlation
:math:`\geq \exp(-2) \approx 0.14` at the half-period :math:`h = a/2`) and is
valid in all dimensions.

This is identical to the scikit-learn
`ExpSineSquared <https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.ExpSineSquared.html>`_
kernel with ``periodicity = a_major`` and ``length_scale = 1``.  The
length-scale (smoothness within each cycle) is fixed; only the period is a
free parameter.

**Choosing between** ``cyc`` **and** ``dco``:

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * -
     - ``cyc``
     - ``dco``
   * - Cyclicity
     - Strictly periodic — repeats forever
     - Quasi-periodic — damps with distance
   * - ``a_major`` meaning
     - Period of oscillation
     - Decay length ≈ oscillation half-period
   * - Correlation at :math:`h = a/2`
     - :math:`\exp(-2) \approx 0.14`
     - :math:`0` (first zero-crossing)
   * - Correlation at :math:`h = a`
     - :math:`1` (full repeat)
     - :math:`-\exp(-3) \approx -0.05`
   * - Min correlation
     - :math:`\exp(-2) > 0` (always positive)
     - Negative — can cause Cholesky failure
   * - Good for
     - Annual cycles, tidal data, repeating spatial patterns
     - Damped oscillations, waves losing energy with distance

Single-structure model
----------------------

.. code-block:: python

   k.set_vgm(ivar=1, jvar=1,
             vtype="sph", nugget=0.05, sill=0.95, a_major=500.0)

Nested (multi-structure) model
-------------------------------

Call ``set_vgm`` multiple times.  Each call **appends** one structure
(``append=True`` is the default):

.. code-block:: python

   k.set_vgm(ivar=1, jvar=1, vtype="nug", nugget=0.05, sill=0.0,  a_major=1.0)
   k.set_vgm(ivar=1, jvar=1, vtype="sph", nugget=0.0,  sill=0.45, a_major=500.0, a_minor1=200.0, azimuth=45.0)
   k.set_vgm(ivar=1, jvar=1, vtype="exp", nugget=0.0,  sill=0.50, a_major=800.0)
   # total sill = 0.05 + 0.45 + 0.50 = 1.0

The variogram :math:`\gamma(h) = \text{sill} - C(h)` for each structure
stacks additively:

.. plot::
   :include-source: false

   import numpy as np
   import matplotlib.pyplot as plt

   h = np.linspace(0, 1200, 400)

   def sph(r): return np.where(r < 1, 1 - 1.5*r + 0.5*r**3, 0.0)
   def expc(r): return np.exp(-3.0*r)

   sill_nug, sill_sph, sill_exp = 0.05, 0.45, 0.50
   a_sph, a_exp = 500.0, 800.0

   gam_nug = sill_nug * np.where(h > 0, 1.0, 0.0)
   gam_sph = sill_sph * (1.0 - sph(h / a_sph))
   gam_exp = sill_exp * (1.0 - expc(h / a_exp))
   gam_tot = gam_nug + gam_sph + gam_exp

   fig, ax = plt.subplots(figsize=(7.5, 3.6))
   ax.stackplot(h, gam_nug, gam_sph, gam_exp,
                labels=["nug (0.05)", "sph  a=500 (0.45)", "exp  a=800 (0.50)"],
                alpha=0.55, colors=["#a6cee3", "#1f78b4", "#b2df8a"])
   ax.plot(h, gam_tot, "k-", lw=1.8, label="total (1.00)")
   ax.set_xlabel("Lag  h  (m)")
   ax.set_ylabel("Variogram  γ(h)")
   ax.set_xlim(0, 1200)
   ax.set_ylim(0, 1.1)
   ax.legend(fontsize=9, loc="lower right")
   ax.set_title("Three-structure nested variogram — contribution of each structure")
   plt.tight_layout()
   plt.show()

Periodic + background trend
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A common pattern for time-series data with an annual cycle and a long-range
trend:

.. code-block:: python

   k.set_vgm(ivar=1, jvar=1, vtype="nug", nugget=0.1, sill=0.0,  a_major=1.0)
   k.set_vgm(ivar=1, jvar=1, vtype="cyc", nugget=0.0, sill=0.6,  a_major=1.0)   # period = 1 year
   k.set_vgm(ivar=1, jvar=1, vtype="exp", nugget=0.0, sill=0.3,  a_major=5.0)   # long-range decay

Product variogram (non-additive nesting)
-----------------------------------------

Setting ``product=True`` on a ``set_vgm`` call **multiplies** the new
structure with the immediately preceding one instead of adding it.  The
`Schur product <https://en.wikipedia.org/wiki/Hadamard_product_(matrices)>`_
of two positive-definite covariance functions is always positive-definite, so
validity is guaranteed regardless of the parameter values chosen.

The primary use case is **independent control over the decay envelope and the
oscillation period** — something ``dco`` cannot provide because it ties both
to the same ``a_major``:

.. code-block:: python

   # dco: decay length AND period both governed by a_major
   k.set_vgm(1, 1, vtype="dco", sill=1.0, a_major=1.0)
   # C(h) = exp(-3h) cos(πh)  — first zero at h = 0.5, coupled to range

   # Product exp × hol: independent ranges
   k.set_vgm(1, 1, vtype="exp", sill=1.0, a_major=5.0)                 # slow decay envelope
   k.set_vgm(1, 1, vtype="hol", sill=1.0, a_major=1.0, product=True)  # oscillation half-period = 1
   # C(h) = exp(-3h/5) cos(πh)  — same period as dco(a=1), but envelope decays 5× more slowly

.. plot::
   :include-source: false

   import numpy as np
   import matplotlib.pyplot as plt

   h = np.linspace(0, 4.0, 400)

   dco_1     = np.exp(-3.0*h)       * np.cos(np.pi*h)   # dco a=1
   prod_slow = np.exp(-3.0*h / 5.0) * np.cos(np.pi*h)   # exp(a=5) × hol(a=1)
   prod_fast = np.exp(-3.0*h / 0.5) * np.cos(np.pi*h)   # exp(a=0.5) × hol(a=1)

   fig, ax = plt.subplots(figsize=(7.5, 3.6))
   ax.plot(h, dco_1,     lw=1.8, color="#1f77b4",
           label=r"dco($a$=1): $e^{-3h}\cos(\pi h)$ — tied decay and period")
   ax.plot(h, prod_slow, lw=1.8, color="#ff7f0e",
           label=r"exp($a$=5)$\times$hol($a$=1): $e^{-3h/5}\cos(\pi h)$ — slow decay")
   ax.plot(h, prod_fast, lw=1.8, color="#2ca02c",
           label=r"exp($a$=0.5)$\times$hol($a$=1): $e^{-6h}\cos(\pi h)$ — fast decay")
   ax.axhline(0, color="k", lw=0.6, ls=":")
   ax.set_xlabel("Lag  h  (in units of a_hol = 1)")
   ax.set_ylabel("C(h) / sill")
   ax.set_xlim(0, 4)
   ax.set_ylim(-0.65, 1.05)
   ax.legend(fontsize=8.5)
   ax.set_title("Product variogram: independent a_exp (decay) and a_hol (period)")
   plt.tight_layout()
   plt.show()

**Rules for product structures:**

- ``product=True`` on structure *k* multiplies with structure *k−1*.
- Both structures may have **different** ``a_major`` values (independent
  scales) but must share the same orientation (``azimuth``, ``dip``,
  ``plunge``).
- Chaining three or more consecutive ``product=True`` structures multiplies
  left-to-right: A × B × C.
- To add a second independent product group, place a non-product structure
  between the two groups.

Anisotropic model
-----------------

The rotation convention follows standard geostatistical practice:

- **azimuth**: clockwise from North (Y-axis), in the XY plane
- **dip**: tilt of the major axis below horizontal (positive downward)
- **plunge**: rotation of the semi-axes around the major axis
- **a_major**: range along the major (longest) axis
- **a_minor1**: range perpendicular to major in the horizontal plane
- **a_minor2**: range in the vertical direction (3-D)

At azimuth=0, dip=0, plunge=0 the major axis points North (Y direction).

.. code-block:: python

   k.set_vgm(ivar=1, jvar=1,
             vtype="sph",
             nugget=0.0, sill=1.0,
             a_major=1000.0, a_minor1=400.0,  # 2-D anisotropy
             azimuth=45.0)                     # major axis points NE

For 3-D add ``a_minor2`` (vertical range) and ``dip`` / ``plunge``.

.. plot::
   :include-source: false

   import numpy as np
   import matplotlib.pyplot as plt

   x = np.linspace(-1.5, 1.5, 200)
   y = np.linspace(-1.5, 1.5, 200)
   X, Y = np.meshgrid(x, y)

   fig, axes = plt.subplots(1, 2, figsize=(10, 4.4),
                             gridspec_kw={"wspace": 0.4})

   for ax, (theta_deg, subtitle) in zip(axes, [
       (0,  "azimuth=0°  (major axis → North)"),
       (45, "azimuth=45°  (major axis → NE)"),
   ]):
       theta  = np.radians(theta_deg)
       a_maj, a_min = 1.0, 0.35

       # azimuth = clockwise from North (Y), so major direction = (sin θ, cos θ)
       r1 = X*np.sin(theta) + Y*np.cos(theta)    # distance along major axis
       r2 = X*np.cos(theta) - Y*np.sin(theta)    # distance along minor axis
       r  = np.sqrt((r1 / a_maj)**2 + (r2 / a_min)**2)
       rc = np.clip(r, 0, 1)
       C  = np.where(r < 1, 1 - 1.5*rc + 0.5*rc**3, 0.0)

       cs = ax.contourf(X, Y, C, levels=11, cmap="RdYlGn", vmin=0, vmax=1)
       ax.contour(X, Y, C, levels=[0.25, 0.5, 0.75], colors="k", linewidths=0.8)
       plt.colorbar(cs, ax=ax, label="ρ(h)", shrink=0.88)

       dx_maj = np.sin(theta) * a_maj
       dy_maj = np.cos(theta) * a_maj
       dx_min = np.cos(theta) * a_min
       dy_min = -np.sin(theta) * a_min

       ax.annotate("", xy=(dx_maj, dy_maj), xytext=(0, 0),
                   arrowprops=dict(arrowstyle="->", color="k", lw=2.0))
       ax.text(dx_maj + np.sin(theta)*0.2, dy_maj + np.cos(theta)*0.2,
               f"a_major\n={a_maj:.1f}", fontsize=7.5, ha="center", va="center")

       ax.annotate("", xy=(dx_min, dy_min), xytext=(0, 0),
                   arrowprops=dict(arrowstyle="->", color="navy", lw=2.0))
       ax.text(dx_min + np.cos(theta)*0.16, dy_min - np.sin(theta)*0.16,
               f"a_minor\n={a_min:.2f}", fontsize=7.5, ha="center", va="center",
               color="navy")

       ax.set_xlim(-1.5, 1.5)
       ax.set_ylim(-1.5, 1.5)
       ax.set_aspect("equal")
       ax.set_xlabel("Easting  (X)")
       ax.set_ylabel("Northing  (Y)")
       ax.set_title(subtitle, fontsize=10)
       ax.axhline(0, color="gray", lw=0.5, ls=":")
       ax.axvline(0, color="gray", lw=0.5, ls=":")

   fig.suptitle("Spherical covariance — geometric anisotropy  (a_major=1.0, a_minor=0.35)",
                fontsize=11)
   plt.show()

Replacing a variogram on a reused object
-----------------------------------------

When reusing a ``Kriging`` object with a **different** variogram, pass
``append=False`` on the first ``set_vgm`` call to clear the previous model:

.. code-block:: python

   # first run
   k.set_obs(...)
   k.set_vgm(ivar=1, jvar=1, vtype="sph", sill=1.0, a_major=500.0)
   k.set_grid(...)
   k.set_search(ivar=1)
   k.solve()

   # second run — different variogram
   k.set_vgm(ivar=1, jvar=1, vtype="exp", sill=1.0, a_major=800.0, append=False)
   k.solve()

Without ``append=False`` the second run would accumulate structures from the
first run, silently doubling (or tripling) the total sill.

Linear Model of Coregionalisation (LMC)
-----------------------------------------

For co-kriging with variables 1 and 2, every nested structure *k* must
satisfy the LMC constraint:

.. math::

   b_{12,k}^2 \leq b_{11,k} \times b_{22,k}

where *b* denotes the partial sill for each variable pair.  Violating this
makes the co-kriging matrix indefinite and will produce negative variances.

The correlation coefficient per structure is:

.. math::

   \rho_k = \frac{b_{12,k}}{\sqrt{b_{11,k} \times b_{22,k}}}

A safe starting point is ``b12 = 0.8 * sqrt(b11 * b22)`` (ρ = 0.8).

**Example LMC:**

.. code-block:: python

   rho = 0.8
   b11, b22 = 0.7, 0.3
   b12 = rho * (b11 * b22) ** 0.5

   k.set_vgm(ivar=1, jvar=1, vtype="sph", nugget=0.0, sill=b11, a_major=1000.0, a_minor1=500.0)
   k.set_vgm(ivar=2, jvar=2, vtype="sph", nugget=0.0, sill=b22, a_major=1000.0, a_minor1=500.0)
   k.set_vgm(ivar=1, jvar=2, vtype="sph", nugget=0.0, sill=b12, a_major=1000.0, a_minor1=500.0)
