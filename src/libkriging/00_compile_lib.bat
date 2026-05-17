rem call ..\compiler_setting.bat & set real8=-real-size:64
rem -fPIC is required for shared libraries.
rem -fdefault-real-8 must match your module compilation.
rem -fopenmp is needed because solve() uses OpenMP internally — omitting it will link but crash at runtime when the OpenMP runtime symbols are missing.


del *.exe *.obj *.o *.mod *.pdb

gfortran -cpp -fbacktrace -ffree-line-length-none -O2 -static -s -fdefault-real-8 %real8% -fPIC -fopenmp -shared ^
   kdtree2_maxidx.f90 common.f90 sposv.f rotation.f90 utils.f90 variogram.f90 gaussian_quadrature.f90 progress_bar.F90 solver.f90 kriging.F90 kriging_capi.f90 ^
   -o ..\pykriging\kriging.dll


pause