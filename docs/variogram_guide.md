# Variogram Guide

## Parameters

Variograms are set with `set_vgm()` using keyword arguments:

```python
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
```

| Parameter | Default | Description |
|---|---|---|
| `vtype` | *(required)* | Model type code (see table below) |
| `nugget` | `0.0` | Nugget effect |
| `sill` | `1.0` | Partial sill of this structure |
| `a_major` | `1.0` | Range along the major (longest) axis |
| `a_minor1` | `a_major` | Range perpendicular to major in the horizontal plane |
| `a_minor2` | `a_minor1` | Range in the vertical direction (3-D only) |
| `azimuth` | `0.0` | Clockwise from North (degrees) |
| `dip` | `0.0` | Tilt below horizontal, positive downward (degrees) |
| `plunge` | `0.0` | Rotation of semi-axes around the major axis (degrees) |
| `append` | `True` | `True` adds a nested structure; `False` replaces the current model |

## Supported model types

| Code | Name | C(h) expression |
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

## Nested (multi-structure) model

Call `set_vgm` once per structure; each call **appends** by default:

```python
k.set_vgm(ivar=1, jvar=1, vtype="nug", nugget=0.05, sill=0.0,  a_major=1.0)
k.set_vgm(ivar=1, jvar=1, vtype="sph", nugget=0.0,  sill=0.45, a_major=500.0, a_minor1=200.0, azimuth=45.0)
k.set_vgm(ivar=1, jvar=1, vtype="exp", nugget=0.0,  sill=0.50, a_major=800.0)
# total sill = 0.05 + 0.45 + 0.50 = 1.0
```

## Anisotropy

The rotation convention follows standard geostatistical practice:

- **azimuth**: clockwise from North (Y-axis), in the XY plane
- **dip**: tilt of the major axis below horizontal (positive downward)
- **plunge**: rotation of the semi-axes around the major axis
- **a_major**: range along the major (longest) axis
- **a_minor1**: range perpendicular to major in the horizontal plane
- **a_minor2**: range in the vertical direction (3-D)

At azimuth=0, dip=0, plunge=0 the major axis points North (Y direction).

## Linear Model of Coregionalisation (LMC)

For co-kriging with variables 1 and 2, every nested structure must satisfy:

```
b₁₂² ≤ b₁₁ × b₂₂
```

Violating this makes the co-kriging matrix indefinite and produces negative variances.

The correlation coefficient per structure is:

```
ρ = b₁₂ / sqrt(b₁₁ × b₂₂)    ∈ [-1, +1]
```

A useful parameterisation — choose ρ first, then derive b₁₂:

```python
rho = 0.8
b11, b22 = 0.7, 0.3
b12 = rho * (b11 * b22) ** 0.5

k.set_vgm(ivar=1, jvar=1, vtype="sph", nugget=0.0, sill=b11, a_major=1000.0, a_minor1=500.0)
k.set_vgm(ivar=2, jvar=2, vtype="sph", nugget=0.0, sill=b22, a_major=1000.0, a_minor1=500.0)
k.set_vgm(ivar=1, jvar=2, vtype="sph", nugget=0.0, sill=b12, a_major=1000.0, a_minor1=500.0)
# check: b12² = 0.336 ≤ b11 × b22 = 0.21  ✓
```
