# Installation

## Install from PyPI

The recommended installation uses the precompiled binary wheel from PyPI:

```bash
python -m pip install krigekit
```

The wheel includes the compiled Fortran kriging engine. A Fortran compiler,
manual DLL installation, and a source checkout are not required.

Precompiled wheels are available for supported Python versions on:

| Platform | Architecture |
|---|---|
| Linux | x86_64 |
| macOS | arm64 (Apple Silicon) and x86_64 |
| Windows | x86_64 |

Precompiled wheels are currently published for Python 3.10, 3.11, and 3.12.
pip installs the required NumPy, pandas, SciPy, matplotlib, and scikit-learn
dependencies automatically.

:::{tip}
Upgrade pip first if it does not select a binary wheel:

```bash
python -m pip install --upgrade pip
python -m pip install krigekit
```
:::

### Verify the installation

```bash
python -c "import krigekit; print(krigekit.__version__)"
```

You can also run a small import check:

```python
from krigekit import Kriging, SpaceTimeKriging, VariogramModel

k = Kriging(ndim=2, nvar=1)
del k
```

## Install in a conda environment

Create an environment and install the precompiled PyPI wheel with pip:

```bash
conda create -n krigekit python=3.12
conda activate krigekit
python -m pip install krigekit
```

`mamba` can be substituted for `conda`.

## Build from source

Building from source is intended for contributors, custom compiler settings,
or platforms without a compatible wheel. It requires a Fortran compiler such
as gfortran, Intel ifx, or Intel ifort.

```bash
git clone https://github.com/ougx/krigekit.git
cd krigekit
```

Compile the shared library:

**Linux or macOS with gfortran**

```bash
python build_lib.py --compiler gfortran
```

**Windows with Intel ifx**

```bat
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
python build_lib.py --compiler ifx
```

**Windows with MinGW gfortran**

```bash
python build_lib.py --compiler gfortran
```

The build script places `libkriging.so`, `libkriging.dylib`, or `kriging.dll`
inside `src/krigekit/`. Then install the local package:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

Useful build options:

```bash
python build_lib.py --opt debug
python build_lib.py --no-openmp
python build_lib.py --hcache 0
```

## Build the documentation

From a source checkout:

```bash
python -m pip install -e ".[docs]"
python -m sphinx -b html docs docs/_build/html
```
