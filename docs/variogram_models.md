# Variogram models

## Parameters

Variograms are set with `set_vgm()`.  All parameters are keyword arguments:

| Parameter | Default | Description |
|---|---|---|
| `vtype` | *(required)* | Model type code (see table below) |
| `nugget` | `0.0` | Nugget effect (discontinuity at origin) |
| `sill` | `1.0` | Partial sill — variance contributed by this structure |
| `a_major` | `1.0` | Range along the major (longest) axis; see per-model meaning below |
| `a_minor1` | `a_major` | Range along the first minor axis (defaults to isotropic) |
| `a_minor2` | `a_minor1` | Range along the vertical axis (3-D only) |
| `azimuth` | `0.0` | Azimuth of major axis, degrees clockwise from North |
| `dip` | `0.0` | Dip angle, degrees positive downward |
| `plunge` | `0.0` | Plunge angle, degrees |
| `append` | `True` | `True` appends a nested structure; `False` replaces the current model |

The dimensionless lag is $r = h / a_\text{major}$ (after anisotropy scaling).
The covariance is $C(h) = \text{sill} \times \rho(r)$ where $\rho(r)$ is the
correlation function listed in the table below.

## Supported model types

| Code | Name | Correlation $\rho(r)$ |
|---|---|---|
| `nug` | Pure nugget | $0$ for $r > 0$; 1 at origin |
| `sph` | Spherical | $1 - \tfrac{3}{2}r + \tfrac{1}{2}r^3$ for $r < 1$, else $0$ |
| `exp` | Exponential | $\exp(-3r)$ |
| `gau` | Gaussian | $\exp(-3.0625 r^2)$ |
| `hol` | Hole effect | $\cos(\pi r)$ |
| `pow` | Power | $1 - r^{1.5}$ for $r < 1$, else $0$ |
| `bsq` | Bi-square | $(1 - r^2)^2$ for $r < 1$, else $0$ |
| `cir` | Circular | $1 - \tfrac{2}{\pi}\!\left(r\sqrt{1-r^2} + \arcsin r\right)$ for $r < 1$, else $0$ |
| `lin` | Linear | $1 - r$ for $r < 1$, else $0$ |
| `cyc` | GP periodic | $\exp\!\left(-2\sin^2(\pi r)\right)$ |
| `dco` | Damped cosine | $\exp(-3r)\cos(\pi r)$ |

For `sph`, `exp`, and `gau` the **practical range** convention is used:
$\rho(1) \approx 0$ (spherical reaches exactly 0; exponential and Gaussian
reach $\approx 5\%$), so `a_major` is the distance at which spatial
correlation is effectively zero.

## Model notes

### Hole effect (`hol`)

$$C(h) = \text{sill} \cdot \cos\!\left(\frac{\pi h}{a}\right)$$

A pure cosine with period $2a$.  Correlation is zero at $h = a/2$, reaches
its most negative value ($-\text{sill}$) at $h = a$, and returns to
$+\text{sill}$ at $h = 2a$.  Valid in 1-D and 2-D; use with caution in 3-D
(not guaranteed positive-definite).  The oscillation never damps, so the
kriging matrix can be indefinite — the SSYTRF fallback solver handles this.

### Damped cosine (`dco`)

$$C(h) = \text{sill} \cdot \exp\!\left(\frac{-3h}{a}\right) \cos\!\left(\frac{\pi h}{a}\right)$$

An oscillating covariance that decays exponentially with distance.
**`a_major` controls both the decay length and the oscillation period
simultaneously** — the first zero-crossing is at $h = a/2$ and the first
negative lobe peaks at $h = a$.  Valid and positive-definite in all
dimensions.  Suitable when the cyclic pattern weakens over long distances,
e.g. annual signals in a climate record with increasing measurement gaps.

### GP periodic / ExpSineSquared (`cyc`)

$$C(h) = \text{sill} \cdot \exp\!\left(-2\sin^2\!\left(\frac{\pi h}{a}\right)\right)$$

**`a_major` is the period** — correlation returns to `sill` at every integer
multiple of `a_major`.  The model is always positive (correlation $\geq
\exp(-2) \approx 0.14$ at the half-period $h = a/2$) and is valid in all
dimensions.

This is identical to the scikit-learn
[`ExpSineSquared`](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.ExpSineSquared.html)
kernel with `periodicity = a_major` and `length_scale = 1`.  The
length-scale (smoothness within each cycle) is fixed; only the period is a
free parameter.

**Choosing between `cyc` and `dco`:**

| | `cyc` | `dco` |
|---|---|---|
| Cyclicity | Strictly periodic — repeats forever | Quasi-periodic — damps with distance |
| `a_major` meaning | Period of oscillation | Decay length ≈ oscillation half-period |
| Correlation at $h = a/2$ | $\exp(-2) \approx 0.14$ | $0$ (first zero-crossing) |
| Correlation at $h = a$ | $1$ (full repeat) | $-\exp(-3) \approx -0.05$ |
| Min correlation | $\exp(-2) > 0$ (always positive) | Negative — can cause Cholesky failure |
| Good for | Annual cycles, tidal data, repeating spatial patterns | Damped oscillations, waves losing energy with distance |

## Single-structure model

```python
k.set_vgm(ivar=1, jvar=1,
          vtype="sph", nugget=0.05, sill=0.95, a_major=500.0)
```

## Nested (multi-structure) model

Call `set_vgm` multiple times.  Each call **appends** one structure
(`append=True` is the default):

```python
k.set_vgm(ivar=1, jvar=1, vtype="nug", nugget=0.05, sill=0.0,  a_major=1.0)
k.set_vgm(ivar=1, jvar=1, vtype="sph", nugget=0.0,  sill=0.45, a_major=300.0)
k.set_vgm(ivar=1, jvar=1, vtype="exp", nugget=0.0,  sill=0.50, a_major=800.0)
# total sill = 0.05 + 0.45 + 0.50 = 1.0
```

### Periodic + background trend

A common pattern for time-series data with an annual cycle and a long-range
trend:

```python
k.set_vgm(ivar=1, jvar=1, vtype="nug", nugget=0.1, sill=0.0,  a_major=1.0)
k.set_vgm(ivar=1, jvar=1, vtype="cyc", nugget=0.0, sill=0.6,  a_major=1.0)   # period = 1 year
k.set_vgm(ivar=1, jvar=1, vtype="exp", nugget=0.0, sill=0.3,  a_major=5.0)   # long-range decay
```

## Anisotropic model

Specify different ranges along each axis and rotate with `azimuth` / `dip`:

```python
k.set_vgm(ivar=1, jvar=1,
          vtype="sph",
          nugget=0.0, sill=1.0,
          a_major=1000.0, a_minor1=400.0,  # 2-D anisotropy
          azimuth=45.0)                     # major axis points NE
```

For 3-D add `a_minor2` (vertical range) and `dip` / `plunge`.

## Replacing a variogram on a reused object

When reusing a `Kriging` object with a **different** variogram, pass
`append=False` on the first `set_vgm` call to clear the previous model:

```python
# first run
k.set_obs(...)
k.set_vgm(ivar=1, jvar=1, vtype="sph", sill=1.0, a_major=500.0)
k.set_grid(...)
k.set_search(ivar=1)
k.solve()

# second run — different variogram
k.set_vgm(ivar=1, jvar=1, vtype="exp", sill=1.0, a_major=800.0, append=False)
k.solve()
```

Without `append=False` the second run would accumulate structures from the
first run, silently doubling (or tripling) the total sill.

## Linear Model of Coregionalisation (LMC)

For co-kriging with variables 1 and 2, every nested structure *k* must
satisfy the LMC constraint:

$$b_{12,k}^2 \leq b_{11,k} \times b_{22,k}$$

where *b* denotes the partial sill for each variable pair.  Violating this
makes the co-kriging matrix indefinite and will produce negative variances.

The correlation coefficient per structure is:

$$\rho_k = \frac{b_{12,k}}{\sqrt{b_{11,k} \times b_{22,k}}}$$

A safe starting point is `b12 = 0.8 * sqrt(b11 * b22)` (ρ = 0.8).

**Example LMC:**

```python
k.set_vgm(ivar=1, jvar=1, vtype="sph", sill=1.00, a_major=500)  # b11
k.set_vgm(ivar=1, jvar=2, vtype="sph", sill=0.80, a_major=500)  # b12 — ρ = 0.8
k.set_vgm(ivar=2, jvar=2, vtype="sph", sill=1.00, a_major=500)  # b22
# check: 0.80² = 0.64 ≤ 1.00 × 1.00 = 1.00  ✓
```
