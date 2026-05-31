# Variogram models

## Parameters

Variograms are set with `set_vgm()`.  All parameters are keyword arguments:

| Parameter | Default | Description |
|---|---|---|
| `vtype` | *(required)* | Model type code (see table below) |
| `nugget` | `0.0` | Nugget effect (discontinuity at origin) |
| `sill` | `1.0` | Partial sill — variance contributed by this structure |
| `a_major` | `1.0` | Range along the major (longest) axis |
| `a_minor1` | `a_major` | Range along the first minor axis (defaults to isotropic) |
| `a_minor2` | `a_minor1` | Range along the vertical axis (3-D only) |
| `azimuth` | `0.0` | Azimuth of major axis, degrees clockwise from North |
| `dip` | `0.0` | Dip angle, degrees positive downward |
| `plunge` | `0.0` | Plunge angle, degrees |
| `append` | `True` | `True` appends a nested structure; `False` replaces the current model |

## Supported model types

| Code | Name | Covariance C(h) |
|---|---|---|
| `sph` | Spherical | `1 - 1.5(h/a) + 0.5(h/a)³` for h < a, else 0 |
| `exp` | Exponential | `exp(-3h/a)` |
| `gau` | Gaussian | `exp(-3(h/a)²)` |
| `pow` | Power | `h^α` (α < 2) |
| `lin` | Linear | `1 - h/a` |
| `hol` | Hole effect | `sin(πh/a) / (πh/a)` |
| `bsq` | Bessel-squared spherical | — |
| `cir` | Circular | — |
| `nug` | Pure nugget | 1 at h = 0, else 0 |

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
