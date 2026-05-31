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

Result arrays use these shapes:

| Method | Shape | Notes |
|--------|-------|-------|
| `get_results()` for kriging or one SGSIM realization | `(nblock,)` | Primary variable only |
| `get_results()` for co-kriging (nvar>1, nsim=1) | `(nblock, nvar)` | All variables; `out[ib, kvar]` |
| `get_results()` for multiple SGSIM realizations | `(nblock, nvar, nsim)` | Primary variable only; `out[ib, ivar, isim]` |
| `get_estimate_all()` | `(nblock, nvar, nsim)` | All variables and realizations; `out[ib, ivar, isim]` |
| `get_variance_all()` | `(nblock, nvar, nvar)` | Full conditional covariance matrix; `out[ib, :, :]` is the nvar×nvar matrix at block ib; diagonal `out[:, k, k]` is variable k's kriging variance |

Internally the Fortran core stores multivariable results as
`estimate(isim, ivar, iblock)` and conditional covariance as
`est_var(ivar, jvar, iblock)`. The Python getters return the transpose dimension
order.
Use `get_results(squeeze=False)` to keep the leading simulation dimension when
`nsim == 1`, and `copy=True` when a C-contiguous copy is preferred for downstream
NumPy/Pandas workflows.

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

# 3 — estimation targets: coord shape (ngrid, 3), time shape (ngrid,)
k.set_grid(coord=grid_coord, time=grid_time)

# 4 — spatial variogram (call multiple times for nested structures)
k.set_vgm(ivar=1, jvar=1, vtype="sph",
          nugget=0.0, sill=0.8, a_major=1000, a_minor1=500, a_minor2=200)

# 5 — temporal variogram (one call per nested structure)
k.set_vgm_temporal(ivar=1, jvar=1, vtype="exp",
                   nugget=0.0, sill=0.6, at_k=10.0)

# 6 — joint sills (sum-metric only; one float per spatial nested structure)
k.set_vgm_joint_sills(ivar=1, jvar=1, 0.4)

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

## In-Memory Weight Store

Setting `store_weight=True` saves every block's kriging weights in memory during
`solve()`.  The stored weights and neighbour indices are then accessible via
`get_weights()` — no file required.

**Typical use cases**

- Inspect which observations contributed most to each estimate (audit / QC).
- Reconstruct estimates in Python from updated observation values without
  re-running the Fortran solver.
- Archive or visualise the spatial influence of each data point.

### Basic usage

```python
k = Kriging(ndim=2, nvar=1, store_weight=True)
k.set_obs(ivar=1, coord=obs_coord, value=obs_value, nmax=20)
k.set_grid(coord=grid_coord)
k.set_vgm(ivar=1, jvar=1, vtype="sph", sill=1.0, a_major=1000)
k.set_search(ivar=1)
k.solve()               # solves + stores weights in memory
est, var = k.get_results()

w = k.get_weights()     # retrieve stored weights
```

### `get_weights()` return value

Returns a dict with three NumPy arrays indexed `[block, group, neighbour]`:

| Key | Shape | dtype | Description |
|-----|-------|-------|-------------|
| `nnear` | `(nblock, ngroups)` | `int32` | Active neighbour count per block and group |
| `inear` | `(nblock, ngroups, nmax)` | `int32` | **1-based** obs/block indices; `0` = unused |
| `weight` | `(nblock, nvar, ngroups, nmax)` | `float64` | Kriging weights; `0.0` = unused |

**Groups** correspond to the neighbour-group layout:

| `group` index | Contents |
|---|---|
| `0 … nvar-1` | Real observation neighbours, variable `group+1` |
| `nvar … ngroups-1` | Previously-simulated block neighbours (SGSIM only) |

For ordinary kriging with one variable `ngroups == 1`, so `group=0` is the only
group and its weights always sum to 1.

### Example: reconstructing estimates from weights

```python
w = k.get_weights()
nblock, ngroups, nmax = w["weight"].shape

est_manual = np.zeros(nblock)
for ib in range(nblock):
    nn  = w["nnear"][ib, 0]               # number of active obs neighbours
    idx = w["inear"][ib, 0, :nn] - 1      # 1-based → 0-based Python index
    wts = w["weight"][ib, 0, :nn]
    est_manual[ib] = np.dot(wts, obs_value[idx])
```

### Optionally persist to a factor file

Add `weight_file` to also write the completed weight store to disk after `solve()`.
The file can be reloaded on a subsequent run with `use_old_weight=True`, skipping
the matrix assembly and factorisation step entirely (fast re-estimation with
different data values on the same grid).

```python
# --- Run 1: solve, store in memory, and write factor file ---
k = Kriging(ndim=2, nvar=1, store_weight=True, weight_file="weights.fac")
k.set_obs(ivar=1, coord=obs_coord, value=obs_value, nmax=20)
k.set_grid(coord=grid_coord)
k.set_vgm(ivar=1, jvar=1, vtype="sph", sill=1.0, a_major=1000)
k.set_search(ivar=1)
k.solve()
w = k.get_weights()     # available in memory immediately
est1, _ = k.get_results()

# --- Run 2: read weights from file, skip linear solve (fast) ---
k2 = Kriging(ndim=2, nvar=1, use_old_weight=True, weight_file="weights.fac")
k2.set_obs(ivar=1, coord=obs_coord, value=new_obs_value, nmax=20)
k2.set_grid(coord=grid_coord)
k2.set_sim(sample=sample)   # set sample for SGSIM; skip this for kriging
k2.solve()                  # reads weights, calls estimate_block only
est2, _ = k2.get_results()
```

### Constraints

| Constraint | Detail |
|---|---|
| `store_weight` and `use_old_weight` are **mutually exclusive** | Passing both raises an error |
| `weight_file` is optional for `store_weight` | Omit it for memory-only storage |
| `weight_file` is **required** for `use_old_weight` | An empty path raises an error |
| Grid, observations, variogram, and `nmax` must be **identical** between runs | The file stores neighbour indices; mismatched setups produce wrong results silently |
| `store_weight` is now **OMP-compatible** | OpenMP is fully active during `solve()` |

---

## Parallel execution

The Fortran block loop (kriging or SGSIM) is parallelised with OpenMP.
Thread count can be controlled at two levels.

### Per-call: `solve(nthread=N)`

Pass `nthread` to `solve()` to cap OpenMP threads **for that call only**.
The runtime's previous setting is saved and restored automatically.

| `nthread` | Effect |
|-----------|--------|
| `0` (default) | Leave the OMP runtime setting unchanged |
| `1` | Force single-threaded execution |
| `N > 1` | Use at most N threads |

```python
k = Kriging(ndim=2, nvar=1)
k.set_obs(...)
k.set_grid(...)
k.set_vgm(...)
k.set_search(ivar=1)

k.solve(nthread=4)   # use 4 threads for this solve
k.solve(nthread=1)   # serial — fully reproducible, no OMP scheduling variation
k.solve()            # use whatever OMP_NUM_THREADS is set to (default)
```

`nthread=1` is useful when you need **bitwise-reproducible results**: with
multiple threads and dynamic scheduling the factorization-cache hit/miss pattern
can vary between runs, producing sub-machine-epsilon differences for near-zero
estimates (see performance notes below).

### Global: `OMP_NUM_THREADS` environment variable

Set the environment variable **before importing pykriging** to apply a global
default that affects all subsequent `solve()` calls:

```python
import os
os.environ["OMP_NUM_THREADS"] = "8"
from pykriging import Kriging
```

### Multiple independent jobs with `multiprocessing`

Avoid shared Fortran state by running jobs in separate processes:

```python
import os, multiprocessing as mp
from pykriging import ordinary_kriging

def run(args):
    coord, value, grid, spec = args
    return ordinary_kriging(coord, value, grid, spec)

# Each worker process inherits OMP_NUM_THREADS from the parent.
# Set it before spawning if you want to cap threads per worker:
os.environ["OMP_NUM_THREADS"] = "4"
with mp.Pool(4) as pool:
    results = pool.map(run, jobs)
```

---

## Repository structure

```
pykriging/
├── src/                 Source codes
│   ├── libkriging       Core kriging engine/library
│   ├── sparks           Pilot point based Kriging and SGSIM CLI
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

## Performance notes

**Factorization cache** — The Fortran core caches the Cholesky factorization of
the kriging matrix on a per-thread basis.  When consecutive estimation blocks
share the same neighbour set (common on regular grids), the cached factorization
is reused — only the right-hand side is reassembled.  This can give a significant
speedup on dense grids where neighbours change slowly across the domain.

**Weight store and OpenMP** — `store_weight=True` writes to disjoint per-block
array slices and is fully OpenMP-compatible.  The block loop runs in parallel at
full thread count; the factor file is written as a single sequential pass *after*
the parallel loop completes.

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
