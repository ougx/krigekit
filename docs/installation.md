# Installation

## Requirements

| Component | Minimum version |
|---|---|
| Python | 3.10 |
| NumPy | 1.24 |
| Fortran compiler | gfortran, Intel ifx, or Intel ifort |

## Step 1 — Clone the repository

```bash
git clone https://github.com/your-username/pykriging.git
cd pykriging
```

## Step 2 — Compile the Fortran library

The shared library (`libkriging.so` on Linux/macOS, `kriging.dll` on Windows)
must be compiled before use.  Run `build_lib.py` from the project root:

**Linux / macOS (gfortran)**

```bash
python build_lib.py
# explicit compiler:
python build_lib.py --compiler gfortran
# debug build:
python build_lib.py --opt debug
```

**Windows (Intel ifx)**

```bat
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
python build_lib.py --compiler ifx
```

**Windows (gfortran via MSYS2 / rtools)**

```bash
python build_lib.py --compiler gfortran
```

The script compiles all Fortran sources in `src/libkriging/` in dependency
order and places the compiled library inside `src/pykriging/`.

:::{note}
Pass `--no-openmp` to disable OpenMP if your compiler or environment does not
support it.  Single-threaded performance is unaffected for most workloads.
:::

## Step 3 — Install the Python package

```bash
pip install -e .        # editable install (recommended for development)
# or:
pip install .
```

## Step 4 — Verify

```bash
pip install -e ".[dev]"
pytest
```

All tests should pass.  If the shared library is missing, you will get an
`OSError` when importing `pykriging`.

## Docs dependencies (optional)

To build this documentation locally:

```bash
pip install -e ".[docs]"
cd docs
sphinx-build . _build/html
# open _build/html/index.html
```
