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
    "kriging_err.f90",         # must precede variogram (variogram uses kriging_error)
    "utils.F90",
    "progress_bar.F90",
    "rotation.f90",
    "variogram.f90",
    "variogram_st.f90",        # ST variogram models (sum-metric, product-sum)
    "kdtree2_maxidx.f90",
    "gaussian_quadrature.f90",
    "lapack.f",
    "solver.f90",
    "kriging.F90",
    "kriging_capi.f90",
    "kriging_st.F90",          # t_kriging_st — space-time kriging type
    "kriging_st_capi.f90",     # C API for ST types
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
        "release": ["-O2", "-fdefault-real-8", "-fopenmp", "-cpp", "-fbacktrace", "-ffree-line-length-none"],
        "debug": ["-O0", "-g", "-fdefault-real-8", "-fopenmp", "-Wall", "-fcheck=all", "-fbacktrace", "-cpp", "-DDEBUG", "-ffree-line-length-none"],
        "shared": ["-shared", "-fPIC"],
        "implib": [],
    },
    "ifx": _intel_flags(
        opt_win   = ["/O2", "/real-size:64", "/Qopenmp", "/heap-arrays:0", "/traceback", "/fpp"],
        opt_linux = ["-O2", "-real-size:64", "-qopenmp", "-traceback", "-fpp"],
        debug_win = ["/Od", "/debug:full", "/real-size:64", "/Qopenmp", "/heap-arrays:0", "/traceback", "/warn:all", "/DDEBUG", "/fpp", "/check:all"],
        debug_linux=["-O0", "-g", "-real-size:64", "-qopenmp", "-traceback", "-fpp", "-warn all", "/DDEBUG", "-check all"],
        shared_win = ["/dll", "/libs:dll"],
        shared_linux = ["-shared", "-fPIC"]
    ),
}
FLAGS["ifort"] = FLAGS["ifx"]

def _module_flags(compiler: str, mod_dir: str) -> list:
    """Return the flags that set the Fortran module output and search directory.

    gfortran  : -J <dir>  -I <dir>   (two tokens each, space-separated)
    ifx/ifort : /module:<dir>  /I<dir>   (Windows, single token, no space)
               -module <dir>  -I<dir>    (Linux/macOS)

    mod_dir should be a build directory (e.g. build/libkriging) so that
    generated .mod files stay out of the source tree.
    """
    if compiler == "gfortran":
        return ["-J", mod_dir, "-I", mod_dir]
    elif compiler in ("ifx", "ifort"):
        if _ON_WINDOWS:
            # Single-token form so subprocess quoting handles spaces in path
            return [f"/module:{mod_dir}", f"/I{mod_dir}"]
        else:
            return ["-module", mod_dir, f"-I{mod_dir}"]
    else:
        return ["-J", mod_dir, "-I", mod_dir]


def _module_flags(compiler: str, mod_dir: str) -> list:
    """Return the flags that set the Fortran module output and search directory.

    gfortran  : -J <dir>  -I <dir>   (two tokens each, space-separated)
    ifx/ifort : /module:<dir>  /I<dir>   (Windows, single token, no space)
               -module <dir>  -I<dir>    (Linux/macOS)

    mod_dir should be a build directory (e.g. build/libkriging) so that
    generated .mod files stay out of the source tree.
    """
    if compiler == "gfortran":
        return ["-J", mod_dir, "-I", mod_dir]
    elif compiler in ("ifx", "ifort"):
        if _ON_WINDOWS:
            # Single-token form so subprocess quoting handles spaces in path
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



# All bind(C) entry points exposed by kriging_capi.f90.
# This list is used to generate the .def file on Windows so that
# ifx/ifort /dll actually exports these symbols (Linux -shared exports
# everything by default; Windows DLLs require an explicit exports list).
_CAPI_EXPORTS = [
    "krige_create",
    "krige_destroy",
    "krige_initialize",
    "krige_set_obs",
    "krige_set_obs_drift",
    "krige_set_vgm",
    "krige_set_vgm_block",
    "krige_set_grid",
    "krige_set_grid_block",
    "krige_set_grid_cv",
    "krige_set_grid_drift",
    "krige_set_sim",
    "krige_set_search",
    "krige_prepare",
    "krige_get_max_threads",
    "krige_get_num_threads",
    "krige_solve",
    "krige_get_nblocks",
    "krige_get_nsim",
    "krige_get_estimate",
    "krige_get_variance",
    # Space-time kriging
    "krige_st_create",
    "krige_st_destroy",
    "krige_st_initialize",
    "krige_st_set_st_model",
    "krige_st_set_obs",
    "krige_st_set_obs_drift",
    "krige_st_set_vgm",
    "krige_st_set_vgm_temporal",
    "krige_st_set_vgm_joint_sills",
    "krige_st_set_grid",
    "krige_st_set_grid_block",
    "krige_st_set_grid_cv",
    "krige_st_set_grid_drift",
    "krige_st_set_sim",
    "krige_st_set_search",
    "krige_st_solve",
    "krige_st_get_nblocks",
    "krige_st_get_nsim",
    "krige_st_get_estimate",
    "krige_st_get_variance",
]


def _write_def_file(path: Path) -> None:
    """Write a MSVC-style module definition file listing all C API exports."""
    with open(path, "w") as f:
        f.write("EXPORTS\n")
        for sym in _CAPI_EXPORTS:
            f.write(f"    {sym}\n")
    print(f"Generated: {path}")



def _clean_mod_files(mod_dir: Path) -> None:
    """Delete stale .mod files from the build directory before recompiling."""
    for f in mod_dir.glob("*.mod"):
        os.remove(f)
        print("Deleted: ", f)


def build(compiler: str, arg: argparse.ArgumentParser, fortran_dir: Path,
          out_dir: Path, mod_dir: Path):
    flag_set = FLAGS.get(compiler)
    if flag_set is None:
        raise ValueError(f"Unknown compiler {compiler!r}. Choose: gfortran, ifx, ifort")
    _clean_mod_files(mod_dir)
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

    # On Windows, Intel ifx/ifort with /dll does NOT automatically export all
    # symbols the way Linux -shared does.  We must supply a .def file that
    # explicitly lists every C-callable entry point; without it ctypes raises
    # "function 'krige_create' not found".
    extra = []
    if sys.platform == "win32" and compiler in ("ifx", "ifort"):
        # /def: must be passed as a linker flag, not a compiler flag.
        # ifx forwards everything after -link directly to the MSVC linker.
        # Use a relative path (just the filename) so no spaces appear in the
        # linker response file ifx doesn't quote paths when writing it.
        _write_def_file(out_dir / "kriging.def")
        extra = ["-link", "/def:kriging.def"]

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

    result = subprocess.run(cmd, capture_output=False,
                            cwd=out_dir if sys.platform == "win32" else None)
    if result.returncode != 0:
        print(f"\nCompilation failed (exit code {result.returncode})")
        sys.exit(result.returncode)

    else:
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

    root        = Path(__file__).parent
    fortran_dir = root / "src" / "libkriging"
    out_dir     = root / "src" / "pykriging"
    mod_dir     = root / "build" / "libkriging"

    if not fortran_dir.exists():
        raise FileNotFoundError(f"Fortran source directory not found: {fortran_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    mod_dir.mkdir(parents=True, exist_ok=True)
    build(compiler, args, fortran_dir, out_dir, mod_dir)


if __name__ == "__main__":
    main()
