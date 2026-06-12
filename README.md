# krigekit

A Python wrapper for a high-performance Fortran kriging engine parallelised with
OpenMP.

| Capability | Notes |
|---|---|
| Ordinary and simple kriging | Point and block support |
| Co-kriging | Linear Model of Coregionalisation |
| Universal kriging / KED | External drift variables |
| Sequential Gaussian Simulation | Reproducible paths, multi-realisation |
| Space-time kriging | Sum-metric and product-sum ST models |
| Multiple Indicator Kriging / SIS | Categorical variables, three cross-variogram strategies |
| Spatially Varying Anisotropy | Per-block variogram |
| Cross-validation | Leave-one-out |
| Kriging weight reuse | Store and replay weights |

**[Full documentation →](docs/)**

---

## Installation

**conda / mamba (recommended)**

```bash
mamba env create -f environment.yml
mamba activate krigekit
```

**pip**

```bash
pip install -e ".[dev]"   # after compiling the Fortran library
```

Compile the Fortran library first:

```bash
python build_lib.py                    # Linux/macOS (gfortran)
python build_lib.py --compiler ifx     # Windows (Intel ifx)
```

See [docs/installation.md](docs/installation.md) for full details including
debug builds, `--no-openmp`, and docs dependencies.

---

## Quick start

```python
import numpy as np
from krigekit import Kriging

obs_coord  = np.array([[0,0],[1,0],[0,1],[1,1],[0.5,0.5]], dtype=float)
obs_value  = np.array([1.0, 2.0, 3.0, 4.0, 2.5])
grid_coord = np.mgrid[0:1.1:0.25, 0:1.1:0.25].reshape(2,-1).T

k = Kriging()
k.set_obs(ivar=1, coord=obs_coord, value=obs_value)
k.set_grid(coord=grid_coord)
k.set_vgm(ivar=1, jvar=1, vtype="sph", sill=1.0, a_major=1.0)
k.set_search()
k.solve()
df = k.get_result_df()
del k
```

For the full class API, co-kriging, SGSIM, space-time kriging, indicator
simulation, and more, see the [user guide](docs/user_guide/) and
[gallery examples](examples/).

---

## Repository structure

```
krigekit/
├── src/
│   ├── libkriging/      Fortran kriging engine
│   ├── sparks/          Pilot-point kriging/SGSIM CLI
│   └── krigekit/       Python ctypes wrapper
├── examples/            Sphinx-Gallery example scripts
├── tests/               pytest test suite
├── test_data/           CSV/image data used by tests and examples
├── docs/                Sphinx documentation source
├── build_lib.py         Fortran compile script
├── environment.yml      conda/mamba environment
└── pyproject.toml       pip package configuration
```

---

## Contributing

1. Fork the repository and create a feature branch.
2. Add tests for any new behaviour.
3. Run `pytest` to confirm all tests pass.
4. Open a pull request.

---

## License

MIT — see [LICENSE](LICENSE).
