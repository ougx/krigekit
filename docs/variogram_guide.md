# Variogram Guide

## Specification string format

```
"vtype  nugget  sill  a_major  a_minor1  a_minor2  azimuth  dip  plunge"
```

Call `set_vgm` once per nested structure.  Example composite model:

```python
k.set_vgm(1, 1, "sph 0.05 0.45 500 200 200 45 0 0")   # short-range anisotropic sph
k.set_vgm(1, 1, "exp 0.00 0.50 800 800 800  0 0 0")   # long-range isotropic exp
```

## Supported model types

| Code | Name | C(h) expression |
|------|------|-----------------|
| `sph` | Spherical | `1 - 1.5(h/a) + 0.5(h/a)³` for h<a, else 0 |
| `exp` | Exponential | `exp(-3h/a)` |
| `gau` | Gaussian | `exp(-3(h/a)²)` |
| `pow` | Power | `h^α` (α<2) |
| `lin` | Linear | `1 - h/a` |
| `hol` | Hole effect | `sin(πh/a) / (πh/a)` |
| `bsq` | Besselised spherical | — |
| `cir` | Circular | — |

## Linear Model of Coregionalisation (LMC)

For co-kriging with variables 1 and 2, each nested structure k must satisfy:

```
b₁₂ₖ² ≤ b₁₁ₖ × b₂₂ₖ
```

Violating this makes the covariance matrix indefinite and the kriging system
will fail or produce negative variances.

The correlation coefficient per structure is:

```
ρₖ = b₁₂ₖ / sqrt(b₁₁ₖ × b₂₂ₖ)    ∈ [-1, +1]
```

A useful parameterisation is to choose ρ and then compute b₁₂:

```python
rho = 0.8
b11 = 0.7; b22 = 0.3
b12 = rho * (b11 * b22) ** 0.5

k.set_vgm(1, 1, f"sph 0 {b11} 1000 500 500 0 0 0")
k.set_vgm(2, 2, f"sph 0 {b22} 1000 500 500 0 0 0")
k.set_vgm(1, 2, f"sph 0 {b12} 1000 500 500 0 0 0")
```

## Anisotropy

The rotation convention follows standard geostatistical practice:

- **azimuth**: clockwise from North (Y-axis), in the XY plane
- **dip**: tilt of the major axis below horizontal (positive downward)
- **plunge**: rotation of the semi-axes around the major axis
- **a_major**: range along the major (longest) axis
- **a_minor1**: range perpendicular to major in the horizontal plane
- **a_minor2**: range in the vertical direction (3D)

At azimuth=0, dip=0, plunge=0: major axis points North (Y direction).
