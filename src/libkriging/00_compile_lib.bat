rem call ..\compiler_setting.bat & set real8=-real-size:64
rem -fPIC is required for shared libraries.
rem -fdefault-real-8 must match your module compilation.
rem -fopenmp is needed because solve() uses OpenMP internally — omitting it will link but crash at runtime when the OpenMP runtime symbols are missing.
rem
rem Preprocessor feature flags (set before calling this script to override):
rem   HCACHE          — multi-slot factor cache size (default 64; 0 = disabled)
rem   USE_COV_TABLE   — use lookup-table covariance evaluation (default 1 = on;
rem                     set to 0 to use analytic path for debugging)

del *.exe *.obj *.o *.mod *.pdb
rem -fopenmp
if "%HCACHE%"=="" set HCACHE=64
if "%PYKRIGING_DISABLE_HCACHE%"=="1" set HCACHE=0
set CACHE_FLAGS=-DPYKRIGING_HCACHE_SLOTS=%HCACHE%
if "%HCACHE%"=="0" set CACHE_FLAGS=%CACHE_FLAGS% -DPYKRIGING_DISABLE_HCACHE
if "%USE_COV_TABLE%"=="" set USE_COV_TABLE=1
set COV_FLAGS=
if "%USE_COV_TABLE%"=="1" set COV_FLAGS=-DUSE_COV_TABLE
gfortran -cpp -fbacktrace -ffree-line-length-none -O2 -fdefault-real-8 %CACHE_FLAGS% %COV_FLAGS% -fPIC  -shared ^
   common.f90 ^
   kriging_err.f90 ^
   utils.F90 ^
   progress_bar.F90 ^
   rotation.f90 ^
   kdtree2_maxidx.f90 ^
   gaussian_quadrature.f90 ^
   lapack.f ^
   solver.f90 ^
   kriging_base.F90 ^
   variogram.f90 ^
   variogram_st.f90 ^
   kriging.F90 ^
   kriging_capi_common.F90 ^
   kriging_capi.F90 ^
   kriging_st.F90 ^
   kriging_st_capi.f90 ^
   -o ..\pykriging\kriging.dll


pause
