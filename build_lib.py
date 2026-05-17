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
    "rotation.f90",
    "variogram.f90",
    "kdtree2_maxidx.f90",
    "gaussian_quadrature.f90",
    "solver.f90",
    "ppsgs.f90",
    "kriging.F90",
    "kriging_capi.f90",
]

# ---------------------------------------------------------------------------
# Compiler flag sets
# ---------------------------------------------------------------------------
FLAGS = {
    "gfortran": {
        "release": ["-O2", "-fPIC", "-fdefault-real-8", "-fopenmp"],
        "debug":   ["-O0", "-g", "-fPIC", "-fdefault-real-8", "-fopenmp",
                    "-Wall", "-fcheck=all", "-fbacktrace"],
        "shared":  ["-shared"],
        "implib":  [],   # gfortran uses -Wl,--out-implib on Windows
    },
    "ifx": {
        "release": ["-O2", "-fPIC", "-r8", "-qopenmp"],
        "debug":   ["-O0", "-g", "-fPIC", "-r8", "-qopenmp", "-warn", "all"],
        "shared":  ["-shared"],
        "implib":  ["-link", "/dll", "/implib:kriging.lib"],
    },
    "ifort": {
        "release": ["-O2", "-fPIC", "-r8", "-qopenmp"],
        "debug":   ["-O0", "-g", "-fPIC", "-r8", "-qopenmp", "-warn", "all"],
        "shared":  ["-shared"],
        "implib":  ["-link", "/dll", "/implib:kriging.lib"],
    },
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


def build(compiler: str, opt: str, fortran_dir: Path, out_dir: Path):
    flag_set = FLAGS.get(compiler)
    if flag_set is None:
        raise ValueError(f"Unknown compiler {compiler!r}. Choose: gfortran, ifx, ifort")

    out_name = output_name(compiler)
    out_path = out_dir / out_name
    sources   = [str(fortran_dir / s) for s in SOURCES]

    # Extra Windows linker flag for gfortran to produce an import library
    extra = []
    if sys.platform == "win32" and compiler == "gfortran":
        extra = [f"-Wl,--out-implib,{out_dir / 'kriging.lib'}"]

    cmd = (
        [compiler]
        + flag_set[opt]
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
    args = parser.parse_args()

    compiler = args.compiler or detect_compiler()
    print(f"Compiler: {compiler}")
    print(f"Mode:     {args.opt}")

    root       = Path(__file__).parent
    fortran_dir = root / "fortran"
    out_dir     = root / "src" / "pykriging"

    if not fortran_dir.exists():
        raise FileNotFoundError(f"Fortran source directory not found: {fortran_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    build(compiler, args.opt, fortran_dir, out_dir)


if __name__ == "__main__":
    main()
