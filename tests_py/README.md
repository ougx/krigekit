# Manual Debugging Tests

A collection of standalone Python scripts for debugging and validating the
krigekit library outside of the pytest suite. Each script can be run
directly from this directory without any additional setup beyond a compiled
`kriging.dll` / `libkriging.so` and the package on `sys.path`.

## Prerequisites

All scripts assume the following layout relative to this folder:

```
krigekit/
├── src/
│   └── krigekit/
│       ├── kriging.dll        # compiled Fortran library (Windows)
│       └── _kriging.py        # low-level ctypes wrapper
├── test_data/
│   ├── pc2d.csv
│   └── grid2d.csv
└── tests/                     # ← this folder
    ├── test_loadDLL.py
    ├── test_ok.py
    ├── test_ok_wrapper.py
    ├── test_sgsim_ok.py
    └── test_kriging_sva.py
```

Build the library before running any test:

```bash
# Linux / macOS
python build_lib.py

# Windows (Intel oneAPI)
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
python build_lib.py --compiler ifx
```

---

## Scripts

### `test_loadDLL.py` — DLL load check (Windows)

Verifies that `kriging.dll` can be loaded by ctypes and that the expected
entry point `krige_initialize` is exported. If the symbol is not found, it
falls back to running `dumpbin /exports` to list all available exports.

**Run:**
```bash
python test_loadDLL.py
```

**Expected output:**
```
Loading: <absolute path>\kriging.dll
✓ DLL loaded successfully
✓ Found initialize function
```

Use this as the first check when the DLL fails to import. Common failures:

| Symptom | Likely cause |
|---|---|
| `OSError: ... not found` | DLL or a dependency (e.g. OpenMP runtime) not on `PATH` |
| `✗ Initialize function not found` | Fortran name mangling mismatch; check `BIND(C, name=...)` in the source |

---

### `test_ok.py` — Ordinary kriging, low-level `_kriging` interface

End-to-end ordinary kriging test using the ctypes `Kriging` class directly.
Reads the 2-D porosity dataset (`pc2d.csv`), estimates on a regular grid
(`grid2d.csv`), and prints the shape and value range of the estimate and
kriging variance arrays.

**Run:**
```bash
python test_ok.py
```

**Expected output (values approximate):**
```
1. Creating Kriging...   OK handle=<int>
2. set_obs...            OK
3. set_vgm...            OK
4. set_grid...           OK
5. set_search...         OK
6. solve...              OK
7. get_results...
   est: shape=(N,), range=[...]
   var: shape=(N,), range=[0.000, ...]
DONE!
```

Variogram used: spherical, nugget=0, sill=0.12, isotropic range=5000 m.
`nmax=62` (all observations).

---

### `test_ok_wrapper.py` — Ordinary kriging, high-level `krigekit` wrapper

Same dataset and variogram as `test_ok.py`, but exercised through the
one-shot `ordinary_kriging()` convenience function. Prints a Pearson
correlation table between the computed estimates and the reference column
`grid2d.csv::estimate`.

**Run:**
```bash
python test_ok_wrapper.py
```

**Expected output:**
```
          est       ref
est  1.000000  0.99...
ref  0.99...   1.000000
```

A correlation above 0.99 indicates a correct result. Use this to catch
regressions in the Python wrapper layer independently of the Fortran core.

---

### `test_sgsim_ok.py` — Sequential Gaussian Simulation

Runs a single SGSIM realisation (`nsim=1`) on 20 randomly generated
observations over a 100×100 domain, estimating at 20 random grid nodes.
Writes per-block matrix debug files (`matA_*.csv`, `data_*.csv`) to the
working directory because `write_mat=True`.

**Run:**
```bash
python test_sgsim_ok.py
```

**Expected output:**
```
Starting Kriging loop
...
Kriging completed.
Done-------
[array of 20 simulated values]
```

The random seed is fixed (`rng = np.random.default_rng(42)`) so the output
is reproducible. Variogram: spherical, sill=1.0, isotropic range=50 m.

> **Note:** `write_mat=True` produces one `matA_<ib>.csv` and one
> `data_<ib>.csv` per block in the working directory. Delete these files
> after debugging to keep the folder clean.

---

### `test_kriging_sva.py` — Spatially Varying Anisotropy (SVA) kriging

Validates the `t_kriging_sva` Fortran subclass and its `KrigingSVA` Python
wrapper across three scenarios. All scenarios use synthetic data generated
with a fixed seed (`np.random.default_rng(0)`) — no CSV files needed.

**Run:**
```bash
python test_kriging_sva.py
```

**Expected output:**
```
============================================================
Test 1: Uniform SVA must match ordinary kriging
============================================================
  est max |SVA - OK| = <1e-4  (tol=1e-04)
  var max |SVA - OK| = <1e-4  (tol=1e-04)
  PASSED

============================================================
Test 2: Split variogram — variance differs between halves
============================================================
  est range (left) : [...]
  est range (right): [...]
  mean var (left,  short-range sph): ...
  mean var (right, long-range  exp): ...
  variance ratio left/right = ...  (expected > 1.0)
  PASSED

============================================================
Test 3: Error paths
============================================================
  3a: set_vgm_block before allocate_sva ...  PASSED
  3b: allocate_sva before set_grid ...       PASSED
  3c: set_vgm_block with ib out of range ... PASSED
  3d: solve with missing block variograms .. PASSED

============================================================
Results: 3 passed, 0 failed
============================================================
```

**What each test checks:**

| Test | What it proves |
|---|---|
| 1 — Uniform SVA | The pointer-redirection mechanism (`self%vgm => vgm_sva(:,:,ib)`) does not corrupt covariance assembly; results must be numerically identical to ordinary kriging within 1e-4. |
| 2 — Split variogram | Per-block variogram parameters are actually used: short-range model on the left half produces higher kriging variance than the long-range model on the right half. |
| 3 — Error paths | The guard clauses in `allocate_sva`, `set_vgm_block`, and `solve_sva` raise the expected errors when the API is called out of order or with invalid arguments. |

> **Note:** `KrigingSVA` must be added to `_kriging.py` as a Python wrapper
> around the Fortran `t_kriging_sva` type before this test can run. The
> wrapper follows the same pattern as `Kriging`, with three additional
> methods: `allocate_sva()`, `set_vgm_block(ib, ivar, jvar, spec)`, and
> `set_vgm_block_all(ivar, jvar, spec)`.

---

## Debugging tips

- Run `test_loadDLL.py` first on Windows; skip it on Linux/macOS.
- Run `test_ok.py` before `test_ok_wrapper.py` to isolate Fortran-layer
  issues from Python-wrapper issues.
- If `test_ok.py` passes but `test_ok_wrapper.py` fails, the bug is in
  `krigekit/__init__.py` or the argument marshalling in `_kriging.py`.
- For SGSIM failures, inspect the `matA_*.csv` files to check whether the
  kriging matrix is singular or poorly conditioned for specific blocks.
- For SVA failures, use this decision tree:
  - **Test 1 fails (uniform SVA ≠ OK):** the pointer redirection
    `self%vgm => vgm_sva(:,:,ib)` is not taking effect — check that
    `solve_sva` overrides the base `solve` correctly and that the DLL was
    recompiled after adding `kriging_sva.F90`.
  - **Test 2 fails (no variance contrast):** `vgm_sva` is being allocated
    but the per-block pointer is not actually being used inside
    `calc_covariance` — add a `verbose` print of `vgm%cov0` per block to
    confirm the values differ.
  - **Test 3 fails (errors not raised):** the guard clauses in the Fortran
    `error stop` paths are not being propagated back to Python — check the
    ctypes error-handling layer in `_kriging.py`.
