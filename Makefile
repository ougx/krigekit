# ===========================================================================
# Makefile — pyKriging
# Builds:  libkriging  (shared library → src/pykriging/kriging.dll / .so)
#          sparks       (CLI executable → bin/sparks[.exe])
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
# Platform — detect Windows and properly configure the SHELL
# ---------------------------------------------------------------------------
ifdef COMSPEC
  WINDOWS := 1
  # If we are in pure CMD/PowerShell (no MSYSTEM), force Make to use cmd.exe
  # This prevents the "sh: Command not found" crash during $(shell ...) functions.
  ifndef MSYSTEM
    SHELL := cmd.exe
    .SHELLFLAGS := /C
  endif
else ifeq ($(OS),Windows_NT)
  WINDOWS := 1
  ifndef MSYSTEM
    SHELL := cmd.exe
    .SHELLFLAGS := /C
  endif
endif

ifeq ($(WINDOWS),1)
  DLL_FILE  := src/pykriging/kriging.dll
  EXE_FILE  := bin/sparks.exe
  OBJEXT    := obj
  # mkdir -p fails in pure CMD. Since Python is required for this library, we use it safely.
  MKDIR     := python -c "import os, sys; [os.makedirs(d, exist_ok=True) for d in sys.argv[1:]]"
else
  DLL_FILE  := src/pykriging/libkriging.so
  EXE_FILE  := bin/sparks
  OBJEXT    := o
  MKDIR     := mkdir -p
endif

# Default goal must come before the first target rule.
.DEFAULT_GOAL := all

# Fortran modules make parallel compilation fragile unless every .mod
# dependency is modeled explicitly.  Keep this Makefile serial so a normal
# `make` works reliably on Windows with GNU Make and gfortran from %PATH%.
.NOTPARALLEL:

# ---------------------------------------------------------------------------
# Compiler — auto-detect unless the user passed FC=... on the make command line.
# Some Windows developer shells set environment variables such as FC=Microsoft;
# ignore those so plain `make` still finds gfortran from PATH.
# ---------------------------------------------------------------------------
ifneq ($(origin FC),command line)
  ifeq ($(WINDOWS),1)
      ifneq ($(findstring sh,$(SHELL)),)
          # Unix-like shell on Windows (Git Bash, MSYS2, MinGW bash)
          FIND_FC_CMD = which gfortran 2>/dev/null || which ifx 2>/dev/null || which ifort 2>/dev/null
      else
          # Pure Windows CMD
          FIND_FC_CMD = where gfortran 2>nul || where ifx 2>nul || where ifort 2>nul
      endif
  else
      # Linux / macOS
      FIND_FC_CMD = which gfortran 2>/dev/null || which ifx 2>/dev/null || which ifort 2>/dev/null
  endif

  # Capture the first valid path found and strip it down to just the executable name
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
# Object-file directories
# ---------------------------------------------------------------------------
LIB_BDIR := build/libkriging
SPK_BDIR := build/sparks

# ---------------------------------------------------------------------------
# Windows: .def file lists every C-API export symbol
# ---------------------------------------------------------------------------
DEF_FILE  :=
_DEF_SYMS := \
  krige_create krige_destroy krige_initialize \
  krige_set_obs krige_set_obs_drift krige_set_vgm krige_set_vgm_block krige_to_str \
  krige_set_grid krige_set_grid_block krige_set_grid_cv krige_set_grid_drift \
  krige_set_sim krige_set_search krige_prepare \
  krige_get_max_threads krige_get_num_threads \
  krige_solve krige_get_nblocks krige_get_nsim \
  krige_get_estimate krige_get_estimate_all krige_get_variance krige_get_variance_all krige_get_last_error \
  krige_free_weight_store krige_get_weight_nnear krige_get_weight_inear krige_get_weight_data \
  krige_get_weight_var krige_set_weights \
  krige_st_create krige_st_destroy krige_st_initialize \
  krige_st_set_st_model krige_update_obs_value \
  krige_st_set_obs krige_st_set_obs_drift krige_st_set_vgm \
  krige_st_set_vgm_temporal krige_st_set_vgm_joint_sills \
  krige_st_set_grid krige_st_set_grid_block krige_st_set_grid_cv \
  krige_st_set_grid_drift krige_st_set_sim krige_st_set_search \
  krige_st_solve krige_st_get_nblocks krige_st_get_nsim \
  krige_st_get_estimate krige_st_get_variance

ifeq ($(WINDOWS),1)
  DEF_FILE := src/pykriging/kriging.def
endif

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------
OMP_FLAGS :=

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
  LIB_MODF       := -J $(LIB_BDIR) -I $(LIB_BDIR)
  SPK_MODF       := -J $(SPK_BDIR) -I $(SPK_BDIR)

  ifeq ($(WINDOWS),1)
    DLL_EXTRA := $(DEF_FILE) -static -static-libgcc -static-libgfortran
  else
    DLL_EXTRA :=
  endif

else ifneq ($(filter $(FC),ifx ifort),)
  ifeq ($(WINDOWS),1)
    export MSYS2_ARG_CONV_EXCL := *
    export MSYS_NO_PATHCONV    := 1
    FFLAGS         := /real-size:64 /traceback /fpp /nologo $(OMP_FLAGS) /heap-arrays:10
    FFLAGS_release := /O2 $(FFLAGS)
    FFLAGS_debug   := /Od /debug:full /warn:all /check:all $(FFLAGS) /DDEBUG
    LIB_SHARED     := /dll /libs:static
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
# Source lists
# ---------------------------------------------------------------------------
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

LIB_SRCS := \
  $(_CORE_SRCS) \
  src/libkriging/variogram_st.f90    \
  src/libkriging/kriging_capi.F90    \
  src/libkriging/kriging_st.F90      \
  src/libkriging/kriging_st_capi.f90

SPK_SRCS := \
  $(_CORE_SRCS) \
  src/sparks/f90getopt.F90           \
  src/sparks/io.f90                  \
  src/sparks/sparks.f90

# ---------------------------------------------------------------------------
# Object-file lists
# ---------------------------------------------------------------------------
_src2obj = $(addprefix $(1)/,$(addsuffix .$(OBJEXT),$(notdir $(basename $(2)))))

LIB_OBJS := $(call _src2obj,$(LIB_BDIR),$(LIB_SRCS))
SPK_OBJS := $(call _src2obj,$(SPK_BDIR),$(SPK_SRCS))

# ---------------------------------------------------------------------------
# Top-level targets
# ---------------------------------------------------------------------------
.PHONY: all libkriging sparks clean info

all: libkriging sparks

libkriging: $(DLL_FILE)

sparks: $(EXE_FILE)

# ---------------------------------------------------------------------------
# libkriging shared library
# ---------------------------------------------------------------------------
$(DLL_FILE): $(DEF_FILE) $(LIB_OBJS)
	$(FC) $(FFLAGS) $(LIB_SHARED) $(LIB_OBJS) -o $@ $(DLL_EXTRA)
	@echo ""
	@echo "Built: $@"
	@echo ""
	@echo ""

$(LIB_BDIR)/%.$(OBJEXT): src/libkriging/%.f90 | $(LIB_BDIR)
	$(FC) $(FFLAGS) -c $< -o $@ $(LIB_MODF)

$(LIB_BDIR)/%.$(OBJEXT): src/libkriging/%.F90 | $(LIB_BDIR)
	$(FC) $(FFLAGS) -c $< -o $@ $(LIB_MODF)

$(LIB_BDIR)/%.$(OBJEXT): src/libkriging/%.f | $(LIB_BDIR)
	$(FC) $(FFLAGS) -c $< -o $@ $(LIB_MODF)

# ---------------------------------------------------------------------------
# sparks executable
# ---------------------------------------------------------------------------
$(EXE_FILE): $(SPK_OBJS) | bin
	$(FC) $(FFLAGS) $(SPK_OBJS) -o $@
	@echo ""
	@echo "Built: $@"

$(SPK_BDIR)/%.$(OBJEXT): src/libkriging/%.f90 | $(SPK_BDIR)
	$(FC) $(FFLAGS) -c $< -o $@ $(SPK_MODF)

$(SPK_BDIR)/%.$(OBJEXT): src/libkriging/%.F90 | $(SPK_BDIR)
	$(FC) $(FFLAGS) -c $< -o $@ $(SPK_MODF)

$(SPK_BDIR)/%.$(OBJEXT): src/libkriging/%.f | $(SPK_BDIR)
	$(FC) $(FFLAGS) -c $< -o $@ $(SPK_MODF)

$(SPK_BDIR)/%.$(OBJEXT): src/sparks/%.f90 | $(SPK_BDIR)
	$(FC) $(FFLAGS) -c $< -o $@ $(SPK_MODF)

$(SPK_BDIR)/%.$(OBJEXT): src/sparks/%.F90 | $(SPK_BDIR)
	$(FC) $(FFLAGS) -c $< -o $@ $(SPK_MODF)

# ---------------------------------------------------------------------------
# Build directories (order-only prerequisites)
# ---------------------------------------------------------------------------
$(LIB_BDIR) $(SPK_BDIR) bin:
	$(MKDIR) $@

# ---------------------------------------------------------------------------
# .def file
# ---------------------------------------------------------------------------
$(DEF_FILE):
	python -c "syms = '$(_DEF_SYMS)'.split(); open(r'$@', 'w').write('EXPORTS\n' + ''.join('    ' + s + '\n' for s in syms))"

# ---------------------------------------------------------------------------
# clean (Uses standard CMD commands if run in pure Windows environment)
# ---------------------------------------------------------------------------
clean:
ifeq ($(WINDOWS),1)
ifneq ($(findstring cmd.exe,$(SHELL)),)
	-del /Q /F $(subst /,\,$(DLL_FILE)) $(subst /,\,$(EXE_FILE)) $(subst /,\,$(DEF_FILE)) 2>nul
	-rmdir /S /Q build 2>nul
	-del /Q /F *.mod *.obj *.o 2>nul
else
	-rm -f $(DLL_FILE) $(EXE_FILE) $(DEF_FILE)
	-rm -rf build
	-rm -f *.mod *.obj *.o
endif
else
	-rm -f $(DLL_FILE) $(EXE_FILE) $(DEF_FILE)
	-rm -rf build
	-rm -f *.mod *.obj *.o
endif

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
