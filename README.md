# pykriging

A Python wrapper for a high-performance Fortran kriging and Sequential
Gaussian Simulation (SGSIM) engine.  The Fortran core is parallelised with
OpenMP and supports:

- **Ordinary and simple kriging** (point and block)
- **Co-kriging** (multiple variables, Linear Model of Coregionalisation)
- **Universal kriging** (external drift / KED)
- **Sequential Gaussian Simulation** (SGSIM) with reproducible paths and samples
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

The script compiles all Fortran sources in `fortran/` in dependency order and
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
    variogram_spec="sph 0.0 1.0 0.8 1.0 0.8 0.0 0.0 0.0",
    nmax=5,
)
print(est.shape, var.shape)   # (25,) (25,)

# --- Full class interface ---
k = Kriging(ndim=2, nvar=1)
k.set_obs(ivar=1, coord=obs_coord, value=obs_value, nmax=5)
k.set_vgm(ivar=1, jvar=1, spec="sph 0.0 1.0 0.8 1.0 0.8 0.0 0.0 0.0")
k.set_grid(coord=grid_coord)
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

## Variogram specification string

Variograms are described as a space-separated string:

```
"vtype  nugget  sill  a_minor1  a_major  a_minor2  azimuth  dip  plunge"
```

| Field | Description |
|-------|-------------|
| `vtype` | Model type: `sph` `exp` `gau` `pow` `lin` `hol` `bsq` `cir` |
| `nugget` | Nugget effect |
| `sill` | Partial sill (variance contributed by this structure) |
| `a_minor1` | Range along the minor horizontal axis |
| `a_major` | Range along the major axis |
| `a_minor2` | Range along the vertical axis (3D only) |
| `azimuth` | Azimuth of major axis (degrees, clockwise from North) |
| `dip` | Dip angle (degrees, positive downward) |
| `plunge` | Plunge angle (degrees) |

Call `set_vgm` multiple times to build a composite (nested) model:

```python
k.set_vgm(1, 1, "sph 100.0 400.0 500 1000 500 0 0 0")   # structure 1
k.set_vgm(1, 1, "exp   0.0 500.0 300  500 300 0 0 0")   # structure 2
```

---

## Co-kriging

```python
from pykriging import cokriging

est, var = cokriging(
    obs_coords=[coord_primary, coord_secondary],
    obs_values=[value_primary, value_secondary],
    grid_coord=grid,
    variogram_specs={
        (1, 1): "sph 0 1.0 500 1000 500 0 0 0",   # primary auto-variogram
        (2, 2): "sph 0 1.0 500 1000 500 0 0 0",   # secondary auto-variogram
        (1, 2): "sph 0 0.8 500 1000 500 0 0 0",   # cross-variogram (b12²≤b11·b22)
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
    variogram_spec="sph 0.0 1.0 500 1000 500 0 0 0",
    nsim=100,
    nmax=20,
    seed=42,
)
ensemble_mean = sims.mean(axis=0)
```

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
├── src/pykriging/       Python package (wrapper + compiled library)
│   ├── __init__.py
│   └── _kriging.py      ctypes wrapper around kriging_capi
├── fortran/             Fortran sources
│   ├── common.f90
│   ├── utils.F90
│   ├── rotation.f90
│   ├── variogram.f90
│   ├── kriging.F90      Core kriging engine
│   └── kriging_capi.f90 ISO C Binding wrapper
├── tests/               pytest test suite
│   ├── conftest.py      Shared fixtures and data helpers
│   ├── test_ordinary_kriging.py
│   ├── test_cokriging.py
│   ├── test_sgsim.py
│   └── test_api.py
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
