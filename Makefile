# ===========================================================================
# Makefile — pyKriging
# Builds:  libkriging  (shared library → src/pykriging/kriging.dll / .so)
#          sparks       (CLI executable → src/sparks/sparks[.exe])
#
# Requires GNU Make >= 4 (rtools44, msys2, or brew).
# Run from the project root directory.
#
# Usage
# -----
#   make                          # auto-detect compiler, release mode
#   make FC=gfortran              # force gfortran
#   make FC=ifx                   # force Intel ifx
#   make FC=ifort                 # force Intel ifort (classic)
#   make OPT=debug                # debug build
#   make libkriging               # shared library only
#   make sparks                   # executable only
#   make clean                    # remove all compiled outputs
#   make info                     # print detected settings
#
# Supported compilers:  gfortran   ifx   ifort
# ===========================================================================

# ---------------------------------------------------------------------------
# Platform — detect Windows via COMSPEC (always set, even in Git Bash/MSYS2)
# or via OS (set in cmd / PowerShell).
# ---------------------------------------------------------------------------
ifdef COMSPEC
  WINDOWS := 1
else ifeq ($(OS),Windows_NT)
  WINDOWS := 1
endif

ifeq ($(WINDOWS),1)
  DLL_FILE  := src/pykriging/kriging.dll
  EXE_FILE  := bin/sparks
  OBJEXT    := obj
  MKDIR     := mkdir -p # depending if you have cygwin or msys
else
  DLL_FILE  := src/pykriging/libkriging.so
  EXE_FILE  := bin/sparks
  OBJEXT    := o
  MKDIR     := mkdir -p
endif

# Default goal must come before the first target rule.
.DEFAULT_GOAL := all

# ---------------------------------------------------------------------------
# Compiler — auto-detect when the user has not set FC
# (GNU Make's built-in default is 'f77' so we check $(origin FC))
# ---------------------------------------------------------------------------
ifeq ($(filter command_line environment,$(origin FC)),)
  # Better Auto-detection: If we are on Windows, check if the shell is Bash or CMD
  ifeq ($(WINDOWS),1)
      ifneq ($(findstring sh,$(SHELL)),)
          # Git Bash / MSYS2: Find the path, then use 'cygpath -d' to force a 100% space-free short path
          FIND_FC_CMD = (which ifx || which gfortran || which ifort) 2>/dev/null | xargs -I {} cygpath -d "{}" 2>/dev/null
      else
          # Windows CMD: 'where' syntax
          FIND_FC_CMD = where ifx gfortran ifort 2>nul
      endif
  else
      # Linux / macOS
      FIND_FC_CMD = which ifx 2>/dev/null || which gfortran 2>/dev/null || which ifort 2>/dev/null
  endif

  # Using 'subst' handles any leftover escaped characters cleanly
  _FC_FOUND := $(subst \,/,$(firstword $(shell $(FIND_FC_CMD))))
  FC := $(notdir $(basename $(_FC_FOUND)))
#   $(info [DEBUG] Current FIND_FC_CMD is: $(FIND_FC_CMD))
#   $(info [DEBUG] Discovered compiler path: $(_FC_FOUND))
endif
ifeq ($(FC),)
  $(error No Fortran compiler found. Set FC=gfortran, FC=ifx, or FC=ifort)
endif

# ---------------------------------------------------------------------------
# Build mode
# ---------------------------------------------------------------------------
OPT ?= release
# Set OPENMP=0 to disable OpenMP parallelisation (e.g. make OPENMP=0)
OPENMP ?= 1

# ---------------------------------------------------------------------------
# Object-file directories (per-target to keep libkriging and sparks separate)
# Must be defined before the compiler-flags block so that LIB_MODF / SPK_MODF
# can reference them with := (immediate expansion).
# ---------------------------------------------------------------------------
LIB_BDIR := build/libkriging
SPK_BDIR := build/sparks

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------
# 1. Initialize the variable as empty first
OMP_FLAGS :=

# 2. Conditionals MUST start on their own fresh lines
ifeq ($(OPENMP),1)
    ifeq ($(FC),gfortran)
        OMP_FLAGS := -fopenmp
    else ifneq ($(filter $(FC),ifx ifort),)
        ifeq ($(WINDOWS),1)
            OMP_FLAGS := /Qopenmp
        else
            OMP_FLAGS := -qopenmp
        endif
    endif
endif

ifeq ($(FC),gfortran)
  FFLAGS         := -fdefault-real-8 -cpp -fbacktrace -ffree-line-length-none $(OMP_FLAGS)
  FFLAGS_release := -O2 $(FFLAGS)
  FFLAGS_debug   := -O0 -g -Wall -fcheck=all $(FFLAGS) -DDEBUG
  LIB_SHARED     := -shared -fPIC
  # -J <dir>: write .mod files; -I <dir>: search for .mod files
  LIB_MODF       := -J $(LIB_BDIR) -I $(LIB_BDIR)
  SPK_MODF       := -J $(SPK_BDIR) -I $(SPK_BDIR)
  DLL_EXTRA      :=

else ifneq ($(filter $(FC),ifx ifort),)
  ifeq ($(WINDOWS),1)
    # MSYS2/Git Bash auto-converts arguments starting with '/' to Windows paths
    # (e.g. /O2 → C:/Users/hydro/O2).  Setting these env vars disables that so
    # Intel's /flag-style options reach ifx/ifort unchanged.
    export MSYS2_ARG_CONV_EXCL := *
    export MSYS_NO_PATHCONV    := 1
    FFLAGS         := /real-size:64 /traceback /fpp /nologo $(OMP_FLAGS) /heap-arrays:10
    FFLAGS_release := /O2 $(FFLAGS)
    FFLAGS_debug   := /Od /debug:full /warn:all /check:all $(FFLAGS) /DDEBUG
    # /libs:static — embed Intel Fortran core runtime into the DLL so that
    # ifcore.dll / libcaf_ifx.dll are NOT required at runtime.  Without this,
    # the Intel runtime tries to dynamically load libcaf_ifx.dll (the Coarray
    # runtime) on the first Fortran call, which fails when the Intel oneAPI
    # directory is not on the Python process PATH (error 493).
    # libiomp5md.dll (OpenMP) is still linked dynamically and must be on PATH.
    LIB_SHARED     := /dll /libs:static
    # /module:<dir>: write .mod;  /I<dir>: search (no space before path)
    LIB_MODF       := /module:$(LIB_BDIR) /I$(LIB_BDIR)
    SPK_MODF       := /module:$(SPK_BDIR) /I$(SPK_BDIR)
    DLL_EXTRA      := -link /def:src/pykriging/kriging.def
  else
    FFLAGS         := -real-size:64 -traceback -fpp -nologo $(OMP_FLAGS) -heap-arrays:10
    FFLAGS_release := -O2 $(FFLAGS)
    FFLAGS_debug   := -O0 -g -warn all -check all $(FFLAGS) -DDEBUG
    LIB_SHARED     := -shared -fPIC
    LIB_MODF       := -module $(LIB_BDIR) -I$(LIB_BDIR)
    SPK_MODF       := -module $(SPK_BDIR) -I$(SPK_BDIR)
    DLL_EXTRA      :=
  endif

else
  $(error Unsupported compiler '$(FC)'. Use FC=gfortran, FC=ifx, or FC=ifort)
endif

FFLAGS := $(FFLAGS_$(OPT))

# ---------------------------------------------------------------------------
# Source lists — in USE-before-USE dependency order
# ---------------------------------------------------------------------------

# Core modules shared by both libkriging and sparks
_CORE_SRCS := \
  src/libkriging/common.f90          \
  src/libkriging/kriging_err.f90     \
  src/libkriging/utils.F90           \
  src/libkriging/progress_bar.F90    \
  src/libkriging/rotation.f90        \
  src/libkriging/variogram.f90       \
  src/libkriging/kdtree2_maxidx.f90  \
  src/libkriging/gaussian_quadrature.f90 \
  src/libkriging/lapack.f            \
  src/libkriging/solver.f90          \
  src/libkriging/kriging.F90

# libkriging adds variogram_st and both C-API layers on top of core
LIB_SRCS := \
  $(_CORE_SRCS) \
  src/libkriging/variogram_st.f90    \
  src/libkriging/kriging_capi.f90    \
  src/libkriging/kriging_st.F90      \
  src/libkriging/kriging_st_capi.f90

# sparks adds its own three sources on top of core
SPK_SRCS := \
  $(_CORE_SRCS) \
  src/sparks/f90getopt.F90           \
  src/sparks/io.f90                  \
  src/sparks/sparks.f90

# ---------------------------------------------------------------------------
# Object-file lists — flat in each build dir (basename only, no subdir)
# ---------------------------------------------------------------------------
_src2obj = $(addprefix $(1)/,$(addsuffix .$(OBJEXT),$(notdir $(basename $(2)))))

LIB_OBJS := $(call _src2obj,$(LIB_BDIR),$(LIB_SRCS))
SPK_OBJS := $(call _src2obj,$(SPK_BDIR),$(SPK_SRCS))

# ---------------------------------------------------------------------------
# .def file — Windows Intel only; lists every C-API export symbol
# ---------------------------------------------------------------------------
DEF_FILE  := src/pykriging/kriging.def
_DEF_SYMS := \
  krige_create krige_destroy krige_initialize \
  krige_set_obs krige_set_obs_drift krige_set_vgm krige_set_vgm_block krige_to_str \
  krige_set_grid krige_set_grid_block krige_set_grid_cv krige_set_grid_drift \
  krige_set_sim krige_set_search krige_prepare \
  krige_get_max_threads krige_get_num_threads \
  krige_solve krige_get_nblocks krige_get_nsim \
  krige_get_estimate krige_get_variance \
  krige_st_create krige_st_destroy krige_st_initialize \
  krige_st_set_st_model \
  krige_st_set_obs krige_st_set_obs_drift krige_st_set_vgm \
  krige_st_set_vgm_temporal krige_st_set_vgm_joint_sills \
  krige_st_set_grid krige_st_set_grid_block krige_st_set_grid_cv \
  krige_st_set_grid_drift krige_st_set_sim krige_st_set_search \
  krige_st_solve krige_st_get_nblocks krige_st_get_nsim \
  krige_st_get_estimate krige_st_get_variance

# ---------------------------------------------------------------------------
# Top-level targets
# ---------------------------------------------------------------------------
.PHONY: all libkriging sparks clean info

all: libkriging sparks

libkriging: $(DLL_FILE)

sparks: $(EXE_FILE)

# ---------------------------------------------------------------------------
# libkriging shared library
# Compile each source to its own object file, then link.
# Per-file compilation ensures .mod files are complete before dependent files
# read them — avoiding the "Unexpected EOF" stale-mod problem with gfortran.
# ---------------------------------------------------------------------------
$(DLL_FILE): $(LIB_OBJS)
ifeq ($(WINDOWS),1)
ifneq ($(filter $(FC),ifx ifort),)
	$(FC) $(FFLAGS) $(LIB_SHARED) $(LIB_OBJS) -o $@ $(DLL_EXTRA)
else
	$(FC) $(FFLAGS) $(LIB_SHARED) $(LIB_OBJS) -o $@
endif
else
	$(FC) $(FFLAGS) $(LIB_SHARED) $(LIB_OBJS) -o $@
endif
	@echo ""
	@echo "Built: $@"
	@echo ""
	@echo ""

# Pattern rules for object files — libkriging sources
$(LIB_BDIR)/%.$(OBJEXT): src/libkriging/%.f90 | $(LIB_BDIR)
	$(FC) $(FFLAGS) -c $< -o $@ $(LIB_MODF)

$(LIB_BDIR)/%.$(OBJEXT): src/libkriging/%.F90 | $(LIB_BDIR)
	$(FC) $(FFLAGS) -c $< -o $@ $(LIB_MODF)

$(LIB_BDIR)/%.$(OBJEXT): src/libkriging/%.f | $(LIB_BDIR)
	$(FC) $(FFLAGS) -c $< -o $@ $(LIB_MODF)

# ---------------------------------------------------------------------------
# sparks executable
# Core sources are recompiled (standalone; writes mods to build/sparks/).
# ---------------------------------------------------------------------------
$(EXE_FILE): $(SPK_OBJS)
	$(FC) $(FFLAGS) $(SPK_OBJS) -o $@
	@echo ""
	@echo "Built: $@"

# Pattern rules for object files — libkriging core sources (for sparks)
$(SPK_BDIR)/%.$(OBJEXT): src/libkriging/%.f90 | $(SPK_BDIR)
	$(FC) $(FFLAGS) -c $< -o $@ $(SPK_MODF)

$(SPK_BDIR)/%.$(OBJEXT): src/libkriging/%.F90 | $(SPK_BDIR)
	$(FC) $(FFLAGS) -c $< -o $@ $(SPK_MODF)

$(SPK_BDIR)/%.$(OBJEXT): src/libkriging/%.f | $(SPK_BDIR)
	$(FC) $(FFLAGS) -c $< -o $@ $(SPK_MODF)

# Pattern rules for sparks-specific sources
$(SPK_BDIR)/%.$(OBJEXT): src/sparks/%.f90 | $(SPK_BDIR)
	$(FC) $(FFLAGS) -c $< -o $@ $(SPK_MODF)

$(SPK_BDIR)/%.$(OBJEXT): src/sparks/%.F90 | $(SPK_BDIR)
	$(FC) $(FFLAGS) -c $< -o $@ $(SPK_MODF)

# ---------------------------------------------------------------------------
# Build directories (order-only prerequisites)
# ---------------------------------------------------------------------------
$(LIB_BDIR) $(SPK_BDIR):
	$(MKDIR) $@

#python -c "import os; os.makedirs('$@', exist_ok=True)"

# ---------------------------------------------------------------------------
# .def file
# ---------------------------------------------------------------------------
$(DEF_FILE):
	python -c "\
syms = '$(_DEF_SYMS)'.split(); \
open('$@', 'w').write('EXPORTS\n' + ''.join('    ' + s + '\n' for s in syms))"

# ---------------------------------------------------------------------------
# Windows Intel: .def file is a prerequisite of the DLL
# ---------------------------------------------------------------------------
ifeq ($(WINDOWS),1)
ifneq ($(filter $(FC),ifx ifort),)
$(DLL_FILE): $(DEF_FILE)
endif
endif

# ---------------------------------------------------------------------------
# clean
# ---------------------------------------------------------------------------
clean:
	-rm -f $(DLL_FILE) $(EXE_FILE) $(DEF_FILE)
	-rm -rf build
	-rm -f *.mod *.obj *.o

# ---------------------------------------------------------------------------
# info — print build settings
# ---------------------------------------------------------------------------
info:
	@echo 'Compiler :' $(FC)
	@echo 'Path     :' $(_FC_FOUND)
	@echo 'Mode     :' $(OPT)
	@echo 'OpenMP   :' $(OPENMP)
	@echo 'OMP_FLAGS:' $(OMP_FLAGS)
	@echo 'Platform :' $(if $(WINDOWS),Windows,Linux/macOS)
	@echo 'DLL      :' $(DLL_FILE)
	@echo 'EXE      :' $(EXE_FILE)
	@echo 'FFLAGS   :' $(FFLAGS)
	@echo 'LIB_MODF :' $(LIB_MODF)
