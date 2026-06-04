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
    "kriging_err.f90",           # must precede variogram (uses kriging_error)
    "utils.F90",
    "vgmfunc.f90",
    "progress_bar.F90",
    "rotation.f90",
    "variogram.f90",
    "variogram_st.f90",          # ST variogram models (sum-metric, product-sum)
    "kdtree2_maxidx.f90",
    "gaussian_quadrature.f90",
    "lapack.f",
    "solver.f90",
    "kriging.F90",
    "kriging_capi_common.F90",
    "kriging_capi.F90",          # uppercase F — preprocessed by cpp
    "kriging_st.F90",            # t_kriging_st — space-time kriging type
    "kriging_st_capi.f90",       # C API for ST types
]

# ---------------------------------------------------------------------------
# Windows export definition file
#
# src/pykriging/kriging.def is maintained by hand — edit it whenever a
# new bind(C) entry point is added to kriging_capi.F90 or
# kriging_st_capi.f90.  This script never writes or overwrites it.
#
# On Windows:
#   gfortran  — passed as a positional linker input (MinGW ld reads .def
#               files natively)
#   ifx/ifort — passed as  -link /def:<path>  to the MSVC link.exe
# On Linux/macOS the file is not referenced; -shared exports everything.
# ---------------------------------------------------------------------------
_DEF_FILE = Path(__file__).parent / "src" / "pykriging" / "kriging.def"

# ---------------------------------------------------------------------------
# Compiler flag sets
# ---------------------------------------------------------------------------
# Intel compilers (ifx/ifort) use different flag syntax on Windows vs Linux/macOS:
#   Windows : /O2  /real-size:64  /Qopenmp  /dll
#   Linux   : -O2  -real-size:64  -qopenmp  -shared
_ON_WINDOWS = sys.platform == "win32"

def _intel_flags(opt_win, opt_linux, debug_win, debug_linux, shared_win, shared_linux):
    """Return platform-correct Intel release/debug/shared flag lists."""
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
        "release": ["-O2", "-fdefault-real-8", "-fopenmp", "-cpp",
                    "-fbacktrace", "-ffree-line-length-none"],
        "debug":   ["-O0", "-g", "-fdefault-real-8", "-fopenmp", "-Wall",
                    "-fcheck=all", "-fbacktrace", "-cpp", "-DDEBUG",
                    "-ffree-line-length-none"],
        "shared":  ["-shared", "-fPIC"],
        "implib":  [],
    },
    "ifx": _intel_flags(
        opt_win    = ["/O2", "/real-size:64", "/Qopenmp", "/heap-arrays:0",
                      "/traceback", "/fpp"],
        opt_linux  = ["-O2", "-real-size:64", "-qopenmp", "-traceback", "-fpp"],
        debug_win  = ["/Od", "/debug:full", "/real-size:64", "/Qopenmp",
                      "/heap-arrays:0", "/traceback", "/warn:all", "/DDEBUG",
                      "/fpp", "/check:all"],
        debug_linux= ["-O0", "-g", "-real-size:64", "-qopenmp", "-traceback",
                      "-fpp", "-warn all", "-DDEBUG", "-check all"],
        shared_win = ["/dll", "/libs:static"],
        shared_linux = ["-shared", "-fPIC"],
    ),
}
FLAGS["ifort"] = FLAGS["ifx"]


def _module_flags(compiler: str, mod_dir: str) -> list:
    """Return the flags that set the Fortran module output and search directory.

    gfortran  : -J <dir>  -I <dir>   (two tokens each)
    ifx/ifort : /module:<dir>  /I<dir>   (Windows, single token)
               -module <dir>  -I<dir>    (Linux/macOS)

    mod_dir should be a build directory so generated .mod files stay out of
    the source tree.
    """
    if compiler == "gfortran":
        return ["-J", mod_dir, "-I", mod_dir]
    elif compiler in ("ifx", "ifort"):
        if _ON_WINDOWS:
            return [f"/module:{mod_dir}", f"/I{mod_dir}"]
        else:
            return ["-module", mod_dir, f"-I{mod_dir}"]
    else:
        return ["-J", mod_dir, "-I", mod_dir]


def detect_compiler():
    for compiler in ("ifx", "gfortran", "ifort"):
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


def _clean_mod_files(mod_dir: Path) -> None:
    """Delete stale .mod files from the build directory before recompiling."""
    for f in mod_dir.glob("*.mod"):
        os.remove(f)
        print("Deleted: ", f)


def build(compiler: str, arg: argparse.Namespace, fortran_dir: Path,
          out_dir: Path, mod_dir: Path) -> Path:
    flag_set = FLAGS.get(compiler)
    if flag_set is None:
        raise ValueError(
            f"Unknown compiler {compiler!r}. Choose: gfortran, ifx, ifort"
        )

    _clean_mod_files(mod_dir)

    if arg.no_openmp:
        # Strip the OpenMP flag from the chosen optimisation level in-place.
        openmp_flags = {
            "gfortran": ["fopenmp"],
            "ifx":      ["qopenmp", "Qopenmp"],
            "ifort":    ["qopenmp", "Qopenmp"],
        }
        flag_set[arg.opt] = [
            f for f in flag_set[arg.opt]
            if f.lstrip("-/") not in openmp_flags.get(compiler, [])
        ]

    out_name = output_name(compiler)
    out_path = out_dir / out_name
    sources  = [str(fortran_dir / s) for s in SOURCES]

    # ------------------------------------------------------------------
    # Windows symbol exports — both compilers use kriging.def.
    #
    # Linux/macOS: -shared exports every public symbol by default.
    #
    # Windows gfortran: pass kriging.def as a positional linker input;
    #   MinGW ld reads .def files natively.  Also link statically so the
    #   DLL carries no gfortran/libgcc runtime dependency.
    #
    # Windows Intel (ifx/ifort): pass /def:<path> via the -link
    #   passthrough to MSVC link.exe.
    # ------------------------------------------------------------------
    extra: list[str] = []
    if sys.platform == "win32":
        if not _DEF_FILE.exists():
            raise FileNotFoundError(
                f"Export definition file not found: {_DEF_FILE}\n"
                "Edit src/pykriging/kriging.def and add the missing symbols."
            )
        if compiler == "gfortran":
            extra = [
                str(_DEF_FILE),
                "-static",
                "-static-libgcc",
                "-static-libgfortran",
            ]
        elif compiler in ("ifx", "ifort"):
            extra = ["-link", f"/def:{_DEF_FILE}"]

    cmd = (
        [compiler]
        + flag_set[arg.opt]
        + flag_set["shared"]
        + _module_flags(compiler, str(mod_dir))
        + sources
        + ["-o", str(out_path)]
        + extra
        + flag_set["implib"]
    )

    print("Compiling with:")
    print(" ", " ".join(cmd))
    print()

    result = subprocess.run(
        cmd,
        capture_output=False,
        cwd=out_dir if sys.platform == "win32" else None,
    )
    if result.returncode != 0:
        print(f"\nCompilation failed (exit code {result.returncode})")
        sys.exit(result.returncode)

    print(f"\nSuccess: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Build the pykriging Fortran library."
    )
    parser.add_argument(
        "--compiler", default=None,
        help="Fortran compiler: gfortran, ifx, ifort (default: auto-detect)",
    )
    parser.add_argument(
        "--opt", default="release", choices=["release", "debug"],
        help="Optimisation level (default: release)",
    )
    parser.add_argument(
        "--no-openmp", action="store_true",
        help="Disable OpenMP parallelization",
    )
    args = parser.parse_args()

    compiler = args.compiler or detect_compiler()
    print(f"Compiler: {compiler}")
    print(f"Mode:     {args.opt}")

    root        = Path(__file__).parent
    fortran_dir = root / "src" / "libkriging"
    out_dir     = root / "src" / "pykriging"
    mod_dir     = root / "build" / "libkriging"

    if not fortran_dir.exists():
        raise FileNotFoundError(
            f"Fortran source directory not found: {fortran_dir}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    mod_dir.mkdir(parents=True, exist_ok=True)
    build(compiler, args, fortran_dir, out_dir, mod_dir)


if __name__ == "__main__":
    main()
