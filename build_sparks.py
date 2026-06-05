#!/usr/bin/env python3
"""
build_sparks.py
============
Compile the SPARKS Fortran sources into an executable.

Usage
-----
python build_sparks.py               # auto-detect compiler
python build_sparks.py --compiler gfortran
python build_sparks.py --compiler ifx
python build_sparks.py --compiler ifort
python build_sparks.py --opt debug  # no optimisation, add -g
python build_sparks.py --no-openmp  # Disable OpenMP parallelization

The compiled library is placed in:
    bin/sparks (Linux / macOS)
    bin/sparks.exe (Windows)
"""
import re
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Source files in dependency order (each module compiled before its users)
# ---------------------------------------------------------------------------
SOURCES = [
    "common.f90",
    "kriging_err.f90",
    "utils.F90",
    "vgmfunc.f90",
    "progress_bar.F90",
    "rotation.f90",
    "variogram.f90",
    "kdtree2_maxidx.f90",
    "gaussian_quadrature.f90",
    "lapack.f",
    "solver.f90",
    "kriging_base.F90",
    "kriging.F90",
    "../sparks/f90getopt.F90",
    "../sparks/io.f90",
    "../sparks/sparks.f90",
]

# ---------------------------------------------------------------------------
# Compiler flag sets
# ---------------------------------------------------------------------------
_ON_WINDOWS = sys.platform == "win32"

def _intel_flags(opt_win, opt_linux, debug_win, debug_linux, shared_win, shared_linux):
    """Return platform-correct Intel release/debug/shared/implib flag lists."""
    if _ON_WINDOWS:
        return {
            "release": opt_win,
            "debug": debug_win,
            "shared": shared_win,
            "implib": ["/link", "/FORCE:MULTIPLE", "/OPT:REF", "/OPT:ICF"],
        }
    else:
        return {
            "release": opt_linux,
            "debug": debug_linux,
            "shared": shared_linux,
            "implib": [],
        }

FLAGS = {
    "gfortran": {
        "release": ["-O2", "-fdefault-real-8", "-fopenmp", "-cpp", "-fbacktrace", "-ffree-line-length-none"],
        "debug": ["-O0", "-g", "-fdefault-real-8", "-fopenmp", "-Wall", "-fcheck=all", "-fbacktrace", "-cpp", "-ffree-line-length-none"],
        "shared": [],
        "implib": [],
    },
    "ifx": _intel_flags(
        opt_win = ["/O2", "/real-size:64", "/Qopenmp", "/heap-arrays:0", "/traceback", "/fpp"],
        opt_linux = ["-O2", "-real-size:64", "-qopenmp", "-traceback", "-fpp"],
        debug_win = ["/Od", "/debug:full", "/real-size:64", "/Qopenmp", "/heap-arrays:0", "/traceback", "/warn:all", "/fpp", "/check:all"],
        debug_linux=["-O0", "-g", "-real-size:64", "-qopenmp", "-traceback", "-fpp", "-warn all", "-check all"],
        shared_win = ['/nologo', '/MT'],
        shared_linux = ['-nologo', '-static', '-MT']
    ),
}
FLAGS["ifort"] = FLAGS["ifx"]

def detect_compiler():
    for compiler in ("ifx", "gfortran", "ifort"):
        if shutil.which(compiler):
            return compiler
    raise RuntimeError(
        "No Fortran compiler found. Install gfortran (Linux/macOS) or "
        "Intel oneAPI (ifx/ifort) and ensure it is on PATH."
    )

def output_name(compiler: str) -> str:
    return "sparks"


def get_compiler_version(compiler: str) -> str:
    try:
        result = subprocess.run(
            [compiler, "--version"],
            capture_output=True, text=True
        )
        first_line = (result.stdout or result.stderr).splitlines()[0].strip()
        # Extract just the version number: first token matching x.y.z or x.y.z.w
        match = re.search(r'\d+\.\d+[\.\d]*', first_line)
        return match.group(0) if match else first_line[:40]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"

def get_git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_define_flag(compiler: str, name: str, value: str) -> str:
    prefix = "/" if (_ON_WINDOWS and compiler in ("ifx", "ifort")) else "-"
    return f'{prefix}D{name}="{value}"'


def _module_flags(compiler: str, mod_dir: str) -> list:
    """Return the flags that direct .mod file output and search to mod_dir.

    gfortran  : -J <dir>  -I <dir>
    ifx/ifort : /module:<dir>  /I<dir>   (Windows)
               -module <dir>  -I<dir>    (Linux/macOS)

    mod_dir should be a build directory (e.g. build/sparks) so that generated
    .mod files stay out of the source tree.
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


def build(compiler: str, arg: argparse.ArgumentParser, fortran_dir: Path,
          out_dir: Path, mod_dir: Path):
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
    sources = [str(fortran_dir / s) for s in SOURCES]

    git_hash     = get_git_hash()
    compiler_ver = get_compiler_version(compiler)
    defines = [
        get_define_flag(compiler, "GIT_HASH",   git_hash),
        get_define_flag(compiler, "FC_NAME",    compiler),
        get_define_flag(compiler, "FC_VERSION", compiler_ver),
    ]

    cmd = (
        [compiler]
        + flag_set[arg.opt]
        + flag_set["shared"]
        + _module_flags(compiler, str(mod_dir))
        + defines
        + sources
        + ["-o", str(out_path)]
        + flag_set["implib"]
    )

    print("Compiling with:")
    print("  ", " ".join(cmd))
    print()

    result = subprocess.run(cmd, capture_output=False, cwd=out_dir if sys.platform == "win32" else None)
    if result.returncode != 0:
        print(f"\nCompilation failed (exit code {result.returncode})")
        sys.exit(result.returncode)

    print(f"\nSuccess: {out_path}")
    return out_path

def main():
    parser = argparse.ArgumentParser(description="Build the SPARKS executable.")
    parser.add_argument("--compiler", default=None, help="Fortran compiler: gfortran, ifx, ifort (default: auto-detect)")
    parser.add_argument("--opt", default="release", choices=["release", "debug"], help="Optimisation level (default: release)")
    parser.add_argument("--no-openmp", action="store_true", help="Disable OpenMP parallelization")
    args = parser.parse_args()

    compiler = args.compiler or detect_compiler()
    print(f"Compiler: {compiler}")
    print(f"Mode: {args.opt}")

    root        = Path(__file__).parent
    fortran_dir = root / "src" / "libkriging"
    out_dir     = root / "bin"
    mod_dir     = root / "build" / "sparks"

    if not fortran_dir.exists():
        raise FileNotFoundError(f"Fortran source directory not found: {fortran_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    mod_dir.mkdir(parents=True, exist_ok=True)
    build(compiler, args, fortran_dir, out_dir, mod_dir)

if __name__ == "__main__":
    main()
