# pykriging

A Python wrapper for a high-performance Fortran kriging and Sequential
Gaussian Simulation (SGSIM) engine.  The Fortran core is parallelised with
OpenMP and supports:

- **Ordinary and simple kriging** (point and block)
- **Co-kriging** (multiple variables, Linear Model of Coregionalisation)
- **Universal kriging** (external drift / KED)
- **Sequential Gaussian Simulation** (SGSIM) with reproducible paths and samples
- **Space-time kriging** — sum-metric and product-sum ST covariance models
- **Spatially Varying Anisotropy (SVA)** — different variogram per estimation block
- **Cross-validation** (leave-one-out)
- **Anisotropic search** and **per-block variogram scaling**

---

## Requirements

| Component | Minimum version |
|-----------|----------------|
| Python    | 3.10            |
| NumPy     | 1.24            |
| gfortran **or** Intel ifx/ifort | any recent |

---

## Installation

### 1 — Clone the repository

```bash
git clone https://github.com/your-username/pykriging.git
cd pykriging
```

### 2 — Compile the Fortran library

**Linux / macOS (gfortran)**

```bash
python build_lib.py
# or with explicit compiler:
python build_lib.py --compiler gfortran
# debug build (adds -g -fcheck=all):
python build_lib.py --opt debug
```

**Windows (Intel ifx)**

```bat
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
python build_lib.py --compiler ifx
```

The script compiles all Fortran sources in `src/libkriging/` in dependency order and
places the resulting `libkriging.so` (or `kriging.dll`) inside `src/pykriging/`.

### 3 — Install the Python package

```bash
pip install -e .           # editable install (recommended for development)
# or:
pip install .              # regular install
```

### 4 — Run the tests

```bash
pip install -e ".[dev]"
pytest
```

---

## Quick start

```python
import numpy as np
from pykriging import ordinary_kriging, Kriging

# --- Convenience function: one-shot ordinary kriging ---
obs_coord  = np.array([[0,0],[1,0],[0,1],[1,1],[0.5,0.5]], dtype=float)
obs_value  = np.array([1.0, 2.0, 3.0, 4.0, 2.5])
grid_coord = np.mgrid[0:1.1:0.25, 0:1.1:0.25].reshape(2,-1).T

est, var = ordinary_kriging(
    obs_coord, obs_value, grid_coord,
    vgm_spec=dict(vtype="sph", nugget=0.0, sill=1.0, a_major=1.0, a_minor1=0.8),
    nmax=5,
)
print(est.shape, var.shape)   # (25,) (25,)

# --- Full class interface ---
# Important: call must be used in this order, i.e. set_obs(), set_grid(), set_vgm(), set_search(), solve()
k = Kriging(ndim=2, nvar=1)
k.set_obs(ivar=1, coord=obs_coord, value=obs_value, nmax=5)
k.set_grid(coord=grid_coord)
k.set_vgm(ivar=1, jvar=1, vtype="sph", nugget=0.0, sill=1.0, a_major=1.0, a_minor1=0.8)
k.set_search(ivar=1)
k.solve()
est, var = k.get_results()
```

---

## Array convention

All coordinate arrays use **(nobs, ndim)** shape — rows are points, columns
are spatial dimensions.  This matches NumPy, pandas, and scikit-learn.
The wrapper transposes to Fortran's (ndim, nobs) column-major layout internally.

```python
obs_coord  = np.array([[x1,y1], [x2,y2], ...])  # shape (nobs, ndim)
grid_coord = np.array([[gx1,gy1], [gx2,gy2], ...])  # shape (ngrid, ndim)
drift      = np.array([[d1a,d1b], [d2a,d2b], ...])  # shape (nobs, ndrift)
```

---

## Variogram parameters

Variograms are specified via keyword arguments to `set_vgm`.

> **Call order**: `set_grid()` (or `set_grid_block()` / `set_grid_cv()`) must be called
> **before** `set_vgm()`.  The Fortran engine allocates the variogram storage when
> the grid is set, so the number of blocks must be known first.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vtype` | *(required)* | Model type: `sph` `exp` `gau` `pow` `lin` `hol` `bsq` `cir` `nug` |
| `nugget` | `0.0` | Nugget effect |
| `sill` | `1.0` | Partial sill (variance contributed by this structure) |
| `a_major` | `1.0` | Range along the major axis |
| `a_minor1` | `a_major` | Range along the minor horizontal axis (defaults to isotropic) |
| `a_minor2` | `a_minor1` | Range along the vertical axis (3D only) |
| `azimuth` | `0.0` | Azimuth of major axis (degrees, clockwise from North) |
| `dip` | `0.0` | Dip angle (degrees, positive downward) |
| `plunge` | `0.0` | Plunge angle (degrees) |

Call `set_vgm` multiple times to build a composite (nested) model:

```python
k.set_obs(...)
k.set_grid(...)                                                           # must come first
k.set_vgm(1, 1, vtype="sph", nugget=100.0, sill=400.0, a_major=1000, a_minor1=500)  # structure 1
k.set_vgm(1, 1, vtype="exp", nugget=0.0,   sill=500.0, a_major=500,  a_minor1=300)  # structure 2
```

---

## Co-kriging

```python
from pykriging import cokriging

est, var = cokriging(
    obs_coords=[coord_primary, coord_secondary],
    obs_values=[value_primary, value_secondary],
    grid_coord=grid,
    vgm_spec={
        (1, 1): dict(vtype="sph", nugget=0, sill=1.0, a_major=1000, a_minor1=500),  # primary auto-vgm
        (2, 2): dict(vtype="sph", nugget=0, sill=1.0, a_major=1000, a_minor1=500),  # secondary auto-vgm
        (1, 2): dict(vtype="sph", nugget=0, sill=0.8, a_major=1000, a_minor1=500),  # cross-vgm (b12²≤b11·b22)
    },
    nmax=20,
)
```

---

## SGSIM

```python
from pykriging import sequential_gaussian_simulation

# Returns shape (nsim, ngrid)
sims = sequential_gaussian_simulation(
    obs_coord, obs_value, grid_coord,
    vgm_spec=dict(vtype="sph", nugget=0.0, sill=1.0, a_major=1000, a_minor1=500),
    nsim=100,
    nmax=20,
    seed=42,
)
ensemble_mean = sims.mean(axis=0)
```

---

## Space-Time Kriging

`SpaceTimeKriging` (and its convenience wrappers `spacetime_kriging` /
`spacetime_cokriging`) handles datasets where each observation has both a
3-D spatial location **and** a time stamp.  The engine supports two joint
space-time covariance models:

| Model | Description |
|-------|-------------|
| `sum_metric` | Weighted sum of a pure-spatial, a pure-temporal, and a joint component. Requires `joint_sills`. |
| `product_sum` | Product of spatial and temporal marginals plus a sum correction. Requires `k_ps`. |

### Coordinate convention

| Array | Shape | Description |
|-------|-------|-------------|
| `coord` (obs/grid) | **(n, 3)** | Rows are points; columns are x, y, z |
| `time` (obs/grid) | **(n,)** | Any consistent unit (decimal years, days, …) |

### Workflow

```python
from pykriging import SpaceTimeKriging

k = SpaceTimeKriging(nvar=1)

# 1 — choose ST covariance model (must come before set_vgm)
k.set_st_model(model="sum_metric", transform="bounded", at=5.0)

# 2 — load observations: coord shape (nobs, 3), time shape (nobs,)
k.set_obs(ivar=1, coord=obs_coord, value=obs_value, time=obs_time,
          nmax=30, maxdist=5000.0, maxtlag=20.0)

# 3 — spatial variogram (call multiple times for nested structures)
k.set_vgm(ivar=1, jvar=1, vtype="sph",
          nugget=0.0, sill=0.8, a_major=1000, a_minor1=500, a_minor2=200)

# 4 — temporal variogram (one call per nested structure)
k.set_vgm_temporal(ivar=1, jvar=1, vtype="exp",
                   nugget=0.0, sill=0.6, at_k=10.0)

# 5 — joint sills (sum-metric only; one float per spatial nested structure)
k.set_vgm_joint_sills(ivar=1, jvar=1, 0.4)

# 6 — estimation targets: coord shape (ngrid, 3), time shape (ngrid,)
k.set_grid(coord=grid_coord, time=grid_time)

# 7 — build KD-tree
k.set_search(ivar=1)

# 8 — run and retrieve
k.solve()
estimate, variance = k.get_results()   # shapes (ngrid,), (ngrid,)
del k
```

### `set_st_model` parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `"sum_metric"` | `"sum_metric"` or `"product_sum"` |
| `transform` | `"linear"` | How temporal lag enters the joint distance: `"linear"` → `|dt|/at`; `"bounded"` → `1 − exp(−|dt|/at)`; `"power"` → `(|dt|/at)^alpha` |
| `at` | `1.0` | Joint temporal scale (same units as the time arrays) |
| `alpha` | `1.0` | Power exponent (only used when `transform="power"`) |
| `k_ps` | `0.0` | Product-sum coefficient k (only used when `model="product_sum"`) |

### `set_vgm_temporal` parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vtype` | *(required)* | Variogram type (same types as spatial: `sph`, `exp`, `gau`, …) |
| `nugget` | `0.0` | Nugget contribution |
| `sill` | `1.0` | Partial sill |
| `at_k` | `1.0` | Temporal practical range (same time units as observations) |

### One-shot convenience function

```python
from pykriging import spacetime_kriging

est, var = spacetime_kriging(
    obs_coord   = obs_coord,        # (nobs, 3)
    obs_value   = obs_value,        # (nobs,)
    obs_time    = obs_time,         # (nobs,)
    grid_coord  = grid_coord,       # (ngrid, 3)
    grid_time   = grid_time,        # (ngrid,)
    spatial_spec  = dict(vtype="sph", nugget=0.0, sill=0.8,
                         a_major=1000, a_minor1=500, a_minor2=200),
    temporal_spec = dict(vtype="exp", nugget=0.0, sill=0.6, at_k=10.0),
    joint_sills   = [0.4],          # one per spatial nested structure
    model="sum_metric", transform="bounded", at=5.0,
    nmax=30, maxdist=5000.0, maxtlag=20.0,
)
```

### Space-time co-kriging

```python
from pykriging import spacetime_cokriging

est, var = spacetime_cokriging(
    obs_coords  = [coord1, coord2],
    obs_values  = [value1, value2],
    obs_times   = [time1,  time2],
    grid_coord  = grid_coord,
    grid_time   = grid_time,
    spatial_specs  = {
        (1,1): dict(vtype="sph", nugget=0.0, sill=1.0, a_major=1000),
        (2,2): dict(vtype="sph", nugget=0.0, sill=1.0, a_major=1000),
        (1,2): dict(vtype="sph", nugget=0.0, sill=0.7, a_major=1000),
    },
    temporal_specs = {
        (1,1): dict(vtype="exp", nugget=0.0, sill=0.8, at_k=10.0),
        (2,2): dict(vtype="exp", nugget=0.0, sill=0.8, at_k=10.0),
        (1,2): dict(vtype="exp", nugget=0.0, sill=0.6, at_k=10.0),
    },
    joint_sills = {(1,1): [0.4], (2,2): [0.4], (1,2): [0.3]},
    model="sum_metric", transform="bounded", at=5.0,
    nmax=20,
)
```

---

## Spatially Varying Anisotropy (SVA)

When the spatial structure of the data changes across the domain (e.g.
channelised geological units, anisotropy that rotates with depth), you can
assign a **different variogram to each estimation block**.

### `varying_vgm` flag

Pass `varying_vgm=True` to the constructor.  The Fortran engine then allocates
one variogram slot per block instead of a single global model.  The
OMP-parallel block loop automatically uses each block's own variogram — no
shared state is modified inside the parallel region.

```python
from pykriging import Kriging

k = Kriging(ndim=2, nvar=1, varying_vgm=True)
k.set_obs(ivar=1, coord=obs_coord, value=obs_value, nmax=20)
k.set_grid(coord=grid_coord)          # allocates one variogram slot per block

# Assign a variogram to every block individually:
for ib in range(1, nblocks + 1):
    azimuth = local_azimuth[ib - 1]   # block-specific rotation, for example
    k.set_vgm_block(ib=ib, ivar=1, jvar=1, vtype="sph",
                    nugget=0.0, sill=1.0, a_major=800, a_minor1=400,
                    azimuth=azimuth)

k.set_search(ivar=1)
k.solve()
est, var = k.get_results()
```

`set_vgm_block` accepts the same keyword arguments as `set_vgm` plus the block
index `ib` (1-based).  Call it multiple times for the same `ib` to build a
nested model for that block.

### `set_vgm` with a single model applied to all blocks

If you want a single spec assigned to *all* blocks (e.g. as a starting point
before overriding specific blocks), use plain `set_vgm` without `ib` — it fills
every block slot:

```python
k = Kriging(ndim=2, nvar=1, varying_vgm=True)
k.set_obs(...)
k.set_grid(...)
k.set_vgm(ivar=1, jvar=1, vtype="sph", nugget=0.0, sill=1.0, a_major=1000)  # all blocks
# then override individual blocks:
k.set_vgm_block(ib=5, ivar=1, jvar=1, vtype="exp", nugget=0.1, sill=0.9, a_major=400)
```

### `Kriging.__init__` parameters related to SVA

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `varying_vgm` | `bool` | `False` | Enable per-block variogram storage.  Must be `True` to use `set_vgm_block`. |

---

## Weight File (Factor File)

Kriging weights — the solution of the per-block linear system — can be saved to a
**factor file** and reloaded in later runs, skipping the expensive matrix assembly
and factorisation step entirely. This is exactly the pilot point workflow.

**Typical use cases**

- The grid, observation locations, and variogram are fixed, but estimates need to be recomputed
  with updated observation values.
- You want to inspect or archive the exact weights used for each block.

### Step 1 — Write weights

Set `store_weight=True` and provide a `weight_file` path.
The solve loop assembles and solves the kriging system for every block as normal,
writes the weights to the file, **and** computes estimates.

> **Note**: `store_weight=True` forces **sequential** execution — OpenMP is disabled
> to guarantee that blocks are written to the file in a consistent order.
> Expect roughly single-thread performance during the write run.

```python
k = Kriging(ndim=2, nvar=1,
            store_weight=True, weight_file="weights.fac")
k.set_obs(ivar=1, coord=obs_coord, value=obs_value, nmax=20)
k.set_grid(coord=grid_coord)
k.set_vgm(ivar=1, jvar=1, vtype="sph", sill=1.0, a_major=1000)
k.set_search(ivar=1)
k.solve()                          # solves + writes weights + computes estimates
est, var = k.get_results()
```

### Step 2 — Reuse weights

Set `use_old_weight=True` with the same `weight_file`.
The solve loop reads pre-computed weights from the file and calls `estimate_block`
directly — no matrix assembly or factorisation.

> **Note**: `use_old_weight=True` forces **sequential** execution — OpenMP is disabled
> to guarantee that blocks are read from the file in a consistent order.
> Expect roughly single-thread performance during the read run.

```python
k = Kriging(ndim=2, nvar=1,
            use_old_weight=True, weight_file="weights.fac")
k.set_obs(ivar=1, coord=obs_coord, value=obs_value, nmax=20)
k.set_grid(coord=grid_coord)
k.set_vgm(ivar=1, jvar=1, vtype="sph", sill=1.0, a_major=1000)
k.set_search(ivar=1)
k.solve()                          # loads weights + computes estimates (fast)
est, var = k.get_results()
```

### Rules and constraints

| Constraint | Detail |
|---|---|
| `store_weight` and `use_old_weight` are **mutually exclusive** | Passing both raises an error at initialisation |
| `weight_file` is **required** for both modes | An empty path raises an error |
| Grid, observations, variogram, and `nmax` must be **identical** between runs | The file stores neighbour indices; mismatched setups produce wrong results silently |
| `store_weight` **disables OpenMP** | Blocks must be written sequentially to preserve file order |

---

## Parallel execution

**Single large job** — control OpenMP threads from Python before importing:

```python
import os
os.environ["OMP_NUM_THREADS"] = "8"
from pykriging import Kriging
```

**Multiple independent jobs** — use `multiprocessing` to avoid shared Fortran state:

```python
import os, multiprocessing as mp
from pykriging import ordinary_kriging

def run(args):
    os.environ["OMP_NUM_THREADS"] = "4"
    coord, value, grid, spec = args
    return ordinary_kriging(coord, value, grid, spec)

with mp.Pool(4) as pool:
    results = pool.map(run, jobs)
```

---

## Repository structure

```
pykriging/
├── src/                 Source codes
│   ├── libkriging       Core kriging engine/library
│   ├── ppsgs            Pilot point based SGSIM tool
│   └── pykriging        Python wrapper
├── tests/               pytest test suite
├── test_data/           CSV files used by the test suite
├── docs/                Extended documentation (optional)
├── build_lib.py         Compile script (gfortran / ifx / ifort)
├── pyproject.toml       pip package configuration
├── LICENSE              MIT
└── README.md
```

---

## Contributing

1. Fork the repository on GitHub.
2. Create a feature branch: `git checkout -b feature/my-feature`.
3. Make your changes and add tests.
4. Run `pytest` to ensure all tests pass.
5. Open a pull request.

---

## License

MIT — see [LICENSE](LICENSE).
