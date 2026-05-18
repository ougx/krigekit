#!/usr/bin/env python3
"""
build_lib.py
============
Compile the Fortran sources into a shared library and install it into the
pykriging package directory so it can be found at import time.

Usage
-----
    python build_lib.py                   # auto-detect compiler
    python build_lib.py --compiler gfortran
    python build_lib.py --compiler ifx
    python build_lib.py --compiler ifort
    python build_lib.py --opt debug       # no optimisation, add -g
    python build_lib.py --no-openmp       # Disable OpenMP parallelization

The compiled library is placed in:
    src/pykriging/libkriging.so   (Linux / macOS)
    src/pykriging/kriging.dll     (Windows)
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Source files in dependency order (each module compiled before its users)
# ---------------------------------------------------------------------------
SOURCES = [
    "common.f90",
    "utils.F90",
    "progress_bar.F90",
    "rotation.f90",
    "variogram.f90",
    "kdtree2_maxidx.f90",
    "gaussian_quadrature.f90",
    "solver.f90",
    "sposv.f",
    "kriging.F90",
    "kriging_capi.f90",
]

# ---------------------------------------------------------------------------
# Compiler flag sets
# ---------------------------------------------------------------------------
# Intel compilers (ifx/ifort) use different flag syntax on Windows vs Linux/macOS:
#   Windows : /O2  /fPIC  /real-size:64  /Qopenmp  /dll
#   Linux   : -O2  -fPIC  -real-size:64  -qopenmp  -shared
_ON_WINDOWS = sys.platform == "win32"

def _intel_flags(opt_win, opt_linux, debug_win, debug_linux, shared_win, shared_linux):
    """Return platform-correct Intel release/debug/shared/implib flag lists."""
    if _ON_WINDOWS:
        return {
            "release": opt_win,
            "debug":   debug_win,
            "shared":  shared_win,
            "implib":  [],
        }
    else:
        return {
            "release": opt_linux,
            "debug":   debug_linux,
            "shared":  shared_linux,
            "implib":  [],
        }

FLAGS = {
    "gfortran": {
        "release": ["-O2", "-fPIC", "-fdefault-real-8", "-fopenmp", "-cpp", "-fbacktrace", "-ffree-line-length-none"],
        "debug": ["-O0", "-g", "-fPIC", "-fdefault-real-8", "-fopenmp", "-Wall", "-fcheck=all", "-fbacktrace", "-cpp", "-ffree-line-length-none"],
        "shared": ["-shared"],
        "implib": [],
    },
    "ifx": _intel_flags(
        opt_win   = ["/O2", "/real-size:64", "/Qopenmp", "/libs:dll", "/heap-arrays:0", "/traceback", "/fpp"],
        opt_linux = ["-O2", "-real-size:64", "-qopenmp", "-traceback", "-fpp"],
        debug_win = ["/Od", "/debug:full", "/real-size:64", "/Qopenmp", "/libs:dll", "/heap-arrays:0", "/traceback", "/warn:all", "/fpp", "/check:all"],
        debug_linux=["-O0", "-g", "-real-size:64", "-qopenmp", "-traceback", "-fpp", "-warn all", "-check all"],
        shared_win = ["/dll"],
        shared_linux = ["-shared", "-fPIC"]
    ),
    "ifort": _intel_flags(
        # Classic ifort matches ifx flag syntax exactly on Windows/Linux
        opt_win   = ["/O2", "/real-size:64", "/Qopenmp", "/libs:dll", "/heap-arrays:0", "/traceback", "/fpp"],
        opt_linux = ["-O2", "-real-size:64", "-qopenmp", "-traceback", "-fpp"],
        debug_win = ["/Od", "/debug:full", "/real-size:64", "/Qopenmp", "/libs:dll", "/heap-arrays:0", "/traceback", "/warn:all", "/fpp", "/check:all"],
        debug_linux=["-O0", "-g", "-real-size:64", "-qopenmp", "-traceback", "-fpp", "-warn all", "-check all"],
        shared_win = ["/dll"],
        shared_linux = ["-shared", "-fPIC"]
    ),
}


def detect_compiler():
    for compiler in ("gfortran", "ifx", "ifort"):
        if shutil.which(compiler):
            return compiler
    raise RuntimeError(
        "No Fortran compiler found. Install gfortran (Linux/macOS) or "
        "Intel oneAPI (ifx/ifort) and ensure it is on PATH."
    )


def output_name(compiler: str) -> str:
    if sys.platform == "win32":
        return "kriging.dll"
    elif sys.platform == "darwin":
        return "libkriging.dylib"
    else:
        return "libkriging.so"


def build(compiler: str, arg: argparse.ArgumentParser, fortran_dir: Path, out_dir: Path):
    flag_set = FLAGS.get(compiler)
    if flag_set is None:
        raise ValueError(f"Unknown compiler {compiler!r}. Choose: gfortran, ifx, ifort")
    if arg.no_openmp:
        openmp_flags = {
            "gfortran": ["fopenmp"],
            "ifx": ["qopenmp", "Qopenmp"],
            "ifort": ["qopenmp", "Qopenmp"],
        }
        for flag in flag_set[arg.opt]:
            if flag[1:] in openmp_flags.get(compiler, []):
                flag_set[arg.opt].remove(flag)

    out_name = output_name(compiler)
    out_path = out_dir / out_name
    sources   = [str(fortran_dir / s) for s in SOURCES]

    # Extra Windows linker flag for gfortran to produce an import library
    extra = []
    # if sys.platform == "win32" and compiler == "gfortran":
    #     extra = [f"-Wl,--out-implib,{out_dir / 'kriging.lib'}"]

    cmd = (
        [compiler]
        + flag_set[arg.opt]
        + flag_set["shared"]
        + sources
        + ["-o", str(out_path)]
        + extra
        + (flag_set["implib"] if sys.platform == "win32" and compiler != "gfortran" else [])
    )

    print("Compiling with:")
    print(" ", " ".join(cmd))
    print()

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"\nCompilation failed (exit code {result.returncode})")
        sys.exit(result.returncode)

    print(f"\nSuccess: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Build the pykriging Fortran library.")
    parser.add_argument("--compiler", default=None,
                        help="Fortran compiler: gfortran, ifx, ifort (default: auto-detect)")
    parser.add_argument("--opt", default="release", choices=["release", "debug"],
                        help="Optimisation level (default: release)")
    parser.add_argument("--no-openmp", action="store_true",
                        help="Disable OpenMP parallelization")
    args = parser.parse_args()

    compiler = args.compiler or detect_compiler()
    print(f"Compiler: {compiler}")
    print(f"Mode:     {args.opt}")

    root       = Path(__file__).parent
    fortran_dir = root / "src" / "libkriging"
    out_dir     = root / "src" / "pykriging"

    if not fortran_dir.exists():
        raise FileNotFoundError(f"Fortran source directory not found: {fortran_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    build(compiler, args, fortran_dir, out_dir)


if __name__ == "__main__":
    main()
