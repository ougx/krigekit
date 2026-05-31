!==============================================================================
! kriging_capi.f90
!
! ISO C Binding wrapper for the t_kriging Fortran module.
! Exposes every public method as a C-callable function that takes an opaque
! integer(c_intptr_t) handle instead of the derived type.
!
! Design convention:
!   - All Fortran optional arguments are handled on the Python side.
!     Python always supplies concrete values (using defaults when the user
!     does not specify), so no sentinel / has_* flag logic is needed here.
!   - Drift arrays are set through dedicated subroutines (set_obs_drift,
!     set_grid_drift) instead of optional arguments inside set_obs / set_grid.
!   - Boolean Fortran arguments are passed as integer(c_int): 0=.false., 1=.true.
!   - Strings are passed as null-terminated C character arrays and converted
!     with c2fstr before being forwarded to Fortran.
!   - Array dimensions are passed explicitly alongside every array pointer so
!     Fortran can declare explicit-shape dummies (required by C binding).
!     Only genuinely necessary size parameters are included:
!       * pointweight in krige_set_grid_block uses assumed-size (*) so npw
!         is not needed — Fortran derives it via sum(nblockpnt).
!       * randpath and sample in krige_set_sim share nblocks (both equal the
!         number of blocks), so n_rp and n_s collapse to a single parameter.
!
! Compile as a shared library (Linux):
!   gfortran -O2 -fPIC -fdefault-real-8 -fopenmp -shared \
!     common.f90 utils.F90 rotation.f90 variogram.f90 \
!     kriging.F90 kriging_capi.f90 \
!     -o libkriging.so
!
! Compile as a DLL (Windows / ifx):
!   ifx -O2 -fPIC -qopenmp -r8 -shared \
!     common.f90 utils.F90 rotation.f90 variogram.f90 \
!     kriging.F90 kriging_capi.f90 \
!     -o kriging.dll -link /dll /implib:kriging.lib
! A Fortran procedure interface is interoperable with a C function prototype
!  under the condition that any dummy argument with the VALUE attribute is
!  interoperable with the corresponding formal parameter of the prototype,
!  while any dummy argument without the VALUE attribute corresponds to a formal
!  parameter of the prototype that is of a pointer type. Fortran Programming Language
! This is the key sentence. In C, all scalar arguments are passed by value by default.
!  So when Python (or any C caller) calls a Fortran bind(C) subroutine and passes an
!  integer scalar, it pushes the integer value directly onto the call stack or into
!  a register. Without VALUE, Fortran expects a pointer to the integer. It dereferences
!  what it received — which is the integer value itself treated as an address —
!  and reads garbage memory or crashes.
!==============================================================================
module kriging_capi
  use, intrinsic :: ieee_arithmetic
  use iso_c_binding
  use kriging, only: t_kriging
  use kriging_err, only: kriging_clear_error, kriging_ierr, kriging_copy_error, &
                         kriging_error, kriging_failed
  implicit none
  private

  ! Registry-backed handles avoid passing raw addresses of non-C-interoperable
  ! Fortran objects through ctypes.  Python receives a 1-based slot index and
  ! every C entry point resolves that index back to the live Fortran pointer.
  type kriging_handle_slot
    type(t_kriging), pointer :: obj => null()
  end type kriging_handle_slot

  type(kriging_handle_slot), allocatable, save :: kriging_registry(:)

contains

  !=============================================================================
  ! Lifecycle: create / destroy
  !=============================================================================

  !-- Allocate a new t_kriging object on the heap and return its registry slot
  !   as an opaque 64-bit integer handle.  Python stores this handle and passes
  !   it back on every subsequent call.
  integer(c_int) function krige_create(handle) bind(C, name='krige_create') result(ierr)
    integer(c_intptr_t), intent(out) :: handle
    type(t_kriging), pointer :: obj
    integer :: stat
    call kriging_clear_error()
    handle = 0_c_intptr_t
    allocate(obj, stat=stat)
    if (stat /= 0) then
      call kriging_error('krige_create', 'Failed to allocate t_kriging object')
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    call store_obj(obj, handle)
    ierr = int(kriging_ierr(), c_int)
  end function krige_create

  !-- Finalize and deallocate the object; zero the handle so stale use is
  !   caught early.
  integer(c_int) function krige_destroy(handle) bind(C, name='krige_destroy') result(ierr)
    integer(c_intptr_t), intent(inout) :: handle
    type(t_kriging), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    call obj%finalize()
    deallocate(obj)
    call release_obj(handle)
    handle = 0_c_intptr_t
    ierr = int(kriging_ierr(), c_int)
  end function krige_destroy

  !=============================================================================
  ! krige_initialize
  !
  ! All parameters are required; Python supplies defaults for anything the
  ! user did not explicitly set.
  !
  ! Parameters
  !   ndim               : spatial dimensions (2 or 3)
  !   nvar               : number of variables (1=kriging, >1=cokriging)
  !   ndrift             : number of external drift functions (0=none)
  !   unbias             : 1=ordinary kriging; 0=simple kriging
  !   nsim               : 0=kriging only; >0=number of SGSIM realisations
  !   anisotropic_search : 0/1 use anisotropic search ellipse
  !   weight_correction  : 0/1 clip negative weights and re-normalise
  !   use_old_weight     : 0/1 read weights from weight_file
  !   store_weight       : 0/1 write weights to weight_file
  !   cross_validation   : 0/1 leave-one-out cross-validation mode
  !   write_mat          : 0/1 write the matrix for debugging
  !   neglect_error      : 0/1 set NaN instead of stopping on singular matrix
  !   varying_vgm        : 0/1 use a different variogram per estimation block
  !   verbose            : 0/1 print progress messages
  !   weight_file        : null-terminated path (empty string when not used)
  !   bounds             : [lower, upper] clipping bounds for the estimate
  !   seed               : random seed for SGSIM (0 = use clock)
  !=============================================================================
  integer(c_int) function krige_initialize(handle, &
      ndim, nvar, ndrift, unbias, nsim, &
      anisotropic_search, weight_correction, use_old_weight, &
      store_weight, cross_validation, write_mat, neglect_error, varying_vgm, verbose, &
      weight_file, bounds, seed) &
      bind(C, name='krige_initialize') result(ierr)

    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ndim, nvar, ndrift, unbias, nsim, seed
    integer(c_int),      intent(in), value :: anisotropic_search, weight_correction
    integer(c_int),      intent(in), value :: use_old_weight, store_weight
    integer(c_int),      intent(in), value :: cross_validation, write_mat, neglect_error
    integer(c_int),      intent(in), value :: varying_vgm, verbose
    character(kind=c_char), intent(in) :: weight_file(*)
    real(c_double),      intent(in) :: bounds(2)

    type(t_kriging), pointer :: obj
    !-- Local copy avoids an implicit array temporary for the 'bounds' argument
    !   (Intel warning 406: "array temporary created for argument #N").
    !   real(c_double) and the default real kind are both 8-byte with the
    !   compiler flags used (/real-size:64 / -fdefault-real-8), so the
    !   assignment is lossless.
    real :: fbounds(2)
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    fbounds = real(bounds)

    call obj%initialize( &
      ndim               = int(ndim), &
      nvar               = int(nvar), &
      ndrift             = int(ndrift), &
      unbias             = int(unbias), &
      nsim               = int(nsim), &
      anisotropic_search = l(anisotropic_search), &
      weight_correction  = l(weight_correction), &
      use_old_weight     = l(use_old_weight), &
      store_weight       = l(store_weight), &
      cross_validation   = l(cross_validation), &
      write_mat          = l(write_mat), &
      neglect_error      = l(neglect_error), &
      varying_vgm        = l(varying_vgm), &
      verbose            = l(verbose), &
      weight_file        = c2fstr(weight_file), &
      bounds             = fbounds, &
      seed               = int(seed))
    ierr = int(kriging_ierr(), c_int)
  end function krige_initialize

  !=============================================================================
  ! krige_set_obs
  !
  ! Sets coordinates, values, and measurement variance for one variable.
  ! Drift is set separately via krige_set_obs_drift.
  !
  ! Parameters
  !   ivar     : variable index, 1-based
  !   nobs     : number of observations
  !   ndim_c   : number of spatial dimensions
  !   coord    : coordinates [ndim_c, nobs], Fortran (column-major) order
  !   value    : observed values [nobs]
  !   variance : per-observation measurement error variance [nobs];
  !              pass zeros when measurement error is unknown
  !   nmax     : maximum number of neighbours; pass huge(0) to use all
  !   maxdist  : maximum search distance; pass huge(0.0) for unlimited
  !   sk_mean  : global mean for simple kriging (unbias=0)
  !=============================================================================
  integer(c_int) function krige_set_obs(handle, ivar, nobs, ndim_c, &
      coord, value, variance, nmax, maxdist, sk_mean) &
      bind(C, name='krige_set_obs') result(ierr)

    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ivar, nobs, ndim_c
    real(c_double),      intent(in) :: coord(ndim_c, nobs)
    real(c_double),      intent(in) :: value(nobs)
    real(c_double),      intent(in) :: variance(nobs)
    integer(c_int),      intent(in), value :: nmax
    real(c_double),      intent(in), value :: maxdist
    real(c_double),      intent(in), value :: sk_mean

    type(t_kriging), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if

    call obj%set_obs(int(ivar), real(coord), real(value), &
      variance = real(variance), &
      nmax     = int(nmax), &
      maxdist  = real(maxdist), &
      sk_mean  = real(sk_mean))
    ierr = int(kriging_ierr(), c_int)
  end function krige_set_obs

  !=============================================================================
  ! krige_set_obs_drift
  !
  ! Sets external drift values at observation locations for variable ivar.
  ! Must be called after krige_set_obs for the same ivar, and only when
  ! ndrift > 0 was passed to krige_initialize.
  !
  ! Parameters
  !   ivar     : variable index, 1-based
  !   ndrift_c : number of drift functions (= ndrift)
  !   nobs     : number of observations
  !   drift    : drift values [ndrift_c, nobs], Fortran order
  !=============================================================================
  integer(c_int) function krige_set_obs_drift(handle, ivar, ndrift_c, nobs, drift) &
      bind(C, name='krige_set_obs_drift') result(ierr)

    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ivar, ndrift_c, nobs
    real(c_double),      intent(in) :: drift(ndrift_c, nobs)

    type(t_kriging), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    call obj%set_obs_drift(int(ivar), real(drift))
    ierr = int(kriging_ierr(), c_int)
  end function krige_set_obs_drift

  !=============================================================================
  ! krige_update_obs_value
  !
  ! Replace observation values for variable ivar in-place.
  ! Coordinates and the kd-tree are unchanged; use with use_old_weight to
  ! re-estimate with new data without recomputing search neighbourhoods or
  ! the LHS factorization.
  !
  ! Parameters
  !   ivar  : variable index, 1-based
  !   nobs  : number of observations (must equal the count set by set_obs)
  !   value : new observed values [nobs]
  !=============================================================================
  integer(c_int) function krige_update_obs_value(handle, ivar, nobs, value) &
      bind(C, name='krige_update_obs_value') result(ierr)

    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ivar, nobs
    real(c_double),      intent(in) :: value(nobs)

    type(t_kriging), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    call obj%update_obs_value(int(ivar), real(value))
    ierr = int(kriging_ierr(), c_int)
  end function krige_update_obs_value

  !=============================================================================
  ! krige_reset_vgm
  !
  ! Clear all nested variogram structures for the (ivar, jvar) pair.
  ! After this call, krige_set_vgm builds a fresh model for the pair.
  !=============================================================================
  integer(c_int) function krige_reset_vgm(handle, ivar, jvar) &
      bind(C, name='krige_reset_vgm') result(ierr)

    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ivar, jvar

    type(t_kriging), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    call obj%reset_vgm(int(ivar), int(jvar))
    ierr = int(kriging_ierr(), c_int)
  end function krige_reset_vgm

  !=============================================================================
  ! krige_set_vgm
  !
  ! Add one nested variogram structure for the (ivar, jvar) pair.
  ! Call multiple times to build composite (nested) models.
  ! For cokriging the LMC constraint b12^2 <= b11*b22 must hold per structure.
  !
  ! Parameters
  !   ivar, jvar : variable indices, 1-based
  !   vtype      : null-terminated variogram type: sph exp gau pow lin hol bsq cir nug
  !   nugget     : nugget contribution of this structure
  !   sill       : partial sill
  !   a_major    : range along principal direction
  !   a_minor1   : range along first minor direction
  !   a_minor2   : range along second minor direction
  !   azimuth, dip, plunge : rotation angles in degrees
  !=============================================================================
  integer(c_int) function krige_set_vgm(handle, ivar, jvar, vtype, &
                            nugget, sill, a_major, a_minor1, a_minor2, &
                            azimuth, dip, plunge) &
      bind(C, name='krige_set_vgm') result(ierr)

    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ivar, jvar
    character(kind=c_char), intent(in) :: vtype(*)
    real(c_double), intent(in), value :: nugget, sill, a_major, a_minor1, a_minor2
    real(c_double), intent(in), value :: azimuth, dip, plunge

    type(t_kriging), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    call obj%set_vgm(int(ivar), int(jvar), c2fstr(vtype), &
                     real(nugget), real(sill), real(a_major), &
                     real(a_minor1), real(a_minor2), &
                     real(azimuth), real(dip), real(plunge))
    ierr = int(kriging_ierr(), c_int)
  end function krige_set_vgm

  !=============================================================================
  ! krige_set_vgm_block
  !
  ! Add one nested variogram structure for block ib and variable pair
  ! (ivar, jvar).  Requires varying_vgm=1 to have been passed to
  ! krige_initialize and set_grid to have been called before set_vgm.
  ! Call multiple times per block to build composite (nested) models.
  !
  ! Parameters
  !   ivar, jvar : variable indices, 1-based
  !   ib         : block index, 1-based
  !   vtype      : null-terminated variogram type: sph exp gau pow lin hol bsq cir nug
  !   nugget     : nugget contribution of this structure
  !   sill       : partial sill
  !   a_major    : range along principal direction
  !   a_minor1   : range along first minor direction
  !   a_minor2   : range along second minor direction
  !   azimuth, dip, plunge : rotation angles in degrees
  !=============================================================================
  integer(c_int) function krige_set_vgm_block(handle, ivar, jvar, ib, vtype, &
                                  nugget, sill, a_major, a_minor1, a_minor2, &
                                  azimuth, dip, plunge) &
      bind(C, name='krige_set_vgm_block') result(ierr)

    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ivar, jvar, ib
    character(kind=c_char), intent(in) :: vtype(*)
    real(c_double), intent(in), value :: nugget, sill, a_major, a_minor1, a_minor2
    real(c_double), intent(in), value :: azimuth, dip, plunge

    type(t_kriging), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    call obj%set_vgm(int(ivar), int(jvar), &
                     vtype   = c2fstr(vtype), &
                     nugget  = real(nugget), &
                     sill    = real(sill), &
                     a_major = real(a_major), &
                     a_minor1= real(a_minor1), &
                     a_minor2= real(a_minor2), &
                     azimuth = real(azimuth), &
                     dip     = real(dip), &
                     plunge  = real(plunge), &
                     ib      = int(ib))
    ierr = int(kriging_ierr(), c_int)
  end function krige_set_vgm_block

  !=============================================================================
  ! krige_set_grid
  !
  ! Sets the estimation grid for point kriging (block_type = 0).
  ! For block kriging use krige_set_grid_block.
  ! For cross-validation use krige_set_grid_cv.
  ! Drift is set separately via krige_set_grid_drift.
  !
  ! Parameters
  !   ngrid       : number of grid nodes
  !   ndim_c      : number of spatial dimensions
  !   coord       : grid coordinates [ndim_c, ngrid], Fortran order
  !   rangescale  : per-block variogram range scaling [ngrid];
  !                 pass 1.0 for every element when not needed
  !   localnugget : additional per-block nugget [ngrid];
  !                 pass 0.0 for every element when not needed
  !=============================================================================
  integer(c_int) function krige_set_grid(handle, ngrid, ndim_c, coord, &
      rangescale, localnugget) &
      bind(C, name='krige_set_grid') result(ierr)

    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ngrid, ndim_c
    real(c_double),      intent(in) :: coord(ndim_c, ngrid)
    real(c_double),      intent(in) :: rangescale(ngrid)
    real(c_double),      intent(in) :: localnugget(ngrid)

    type(t_kriging), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    call obj%set_grid(coord       = real(coord), &
                      rangescale  = real(rangescale), &
                      localnugget = real(localnugget))
    ierr = int(kriging_ierr(), c_int)
  end function krige_set_grid

  !=============================================================================
  ! krige_set_grid_block
  !
  ! Sets the estimation grid for block kriging (block_type > 0 or -4).
  ! Drift is set separately via krige_set_grid_drift.
  !
  ! Parameters
  !   block_type  : -4=Gaussian quadrature; >0=user-supplied sub-nodes
  !   ngrid       : total number of sub-nodes across all blocks
  !   ndim_c      : number of spatial dimensions
  !   coord       : sub-node coordinates [ndim_c, ngrid], Fortran order
  !   nblock      : number of blocks
  !   nblockpnt   : number of sub-nodes per block [nblock]
  !   pointweight : weight of each sub-node [sum(nblockpnt)]
  !   rangescale  : per-block range scaling [nblock]
  !   localnugget : per-block additional nugget [nblock]
  !
  ! Note: pointweight length is sum(nblockpnt); Fortran derives it via
  ! size(pointweight) so no separate npw argument is needed.
  !=============================================================================
  integer(c_int) function krige_set_grid_block(handle, block_type, &
      ngrid, ndim_c, coord, &
      nblock, nblockpnt, pointweight, blocksize, &
      rangescale, localnugget) &
      bind(C, name='krige_set_grid_block') result(ierr)

    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: block_type
    integer(c_int),      intent(in), value :: ngrid, ndim_c
    real(c_double),      intent(in) :: coord(ndim_c, ngrid)
    integer(c_int),      intent(in), value :: nblock
    integer(c_int),      intent(in) :: nblockpnt(nblock)
    real(c_double),      intent(in) :: pointweight(*)   ! length = sum(nblockpnt)
    real(c_double),      intent(in) :: blocksize(ndim_c, nblock)     ! length = 3
    real(c_double),      intent(in) :: rangescale(nblock)
    real(c_double),      intent(in) :: localnugget(nblock)

    integer :: npw
    type(t_kriging), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    npw = sum(nblockpnt)   ! derive length instead of receiving it as an argument
    call obj%set_grid(coord       = real(coord), &
                      block_type  = int(block_type), &
                      nblockpnt   = int(nblockpnt), &
                      pointweight = real(pointweight(1:npw)), &
                      blocksize   = real(blocksize), &
                      rangescale  = real(rangescale), &
                      localnugget = real(localnugget))
    ierr = int(kriging_ierr(), c_int)
  end function krige_set_grid_block

  !=============================================================================
  ! krige_set_grid_cv
  !
  ! Sets up the grid for cross-validation mode.  No coord is needed; Fortran
  ! derives the grid from the observation coordinates automatically.
  ! Call instead of krige_set_grid when cross_validation=1.
  !=============================================================================
  integer(c_int) function krige_set_grid_cv(handle) bind(C, name='krige_set_grid_cv') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    type(t_kriging), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    call obj%set_grid()
    ierr = int(kriging_ierr(), c_int)
  end function krige_set_grid_cv

  !=============================================================================
  ! krige_set_grid_drift
  !
  ! Sets external drift values at block locations.
  ! Must be called after krige_set_grid (or krige_set_grid_block / _cv), and
  ! only when ndrift > 0 was passed to krige_initialize.
  !
  ! Parameters
  !   ndrift_c : number of drift functions (= ndrift)
  !   nblocks  : number of blocks (= block%n, not grid%n)
  !   drift    : drift values [ndrift_c, nblocks], Fortran order
  !=============================================================================
  integer(c_int) function krige_set_grid_drift(handle, ndrift_c, nblocks, drift) &
      bind(C, name='krige_set_grid_drift') result(ierr)

    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ndrift_c, nblocks
    real(c_double),      intent(in) :: drift(ndrift_c, nblocks)

    type(t_kriging), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    call obj%set_grid_drift(real(drift))
    ierr = int(kriging_ierr(), c_int)
  end function krige_set_grid_drift

  !=============================================================================
  ! krige_set_sim
  !
  ! Configures Sequential Gaussian Simulation parameters.
  ! Call after krige_set_grid and before krige_set_search.
  ! Only needed when nsim > 0.
  ! Python always generates randpath and sample before calling, so both are
  ! always provided (no optional dispatching needed).
  !
  ! Parameters
  !   nblocks  : number of blocks (= length of randpath = second dim of sample)
  !   randpath : random visiting order for the block loop [nblocks]
  !   nsim_c   : number of simulations (= nsim)
  !   nvar_c   : number of variables (= nvar)
  !   sample   : pre-drawn standard-normal samples [nsim_c, nblocks]
  !
  ! Note: randpath length and sample second dimension are both nblocks, so a
  ! single parameter covers both — no separate n_rp / n_s needed.
  !=============================================================================
  integer(c_int) function krige_set_sim(handle, nblocks, randpath, nsim_c, nvar_c, sample) &
      bind(C, name='krige_set_sim') result(ierr)

    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: nblocks
    integer(c_int),      intent(in) :: randpath(nblocks)
    integer(c_int),      intent(in), value :: nsim_c
    integer(c_int),      intent(in), value :: nvar_c
    real(c_double),      intent(in) :: sample(nsim_c, nvar_c, nblocks)

    type(t_kriging), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    call obj%set_sim(randpath = int(randpath), sample = real(sample))
    ierr = int(kriging_ierr(), c_int)
  end function krige_set_sim

  !=============================================================================
  ! krige_set_search
  !
  ! Builds the KD-tree and configures the search ellipse for variable ivar.
  ! Call once per variable after krige_set_obs (and krige_set_sim for SGSIM).
  !
  ! Parameters
  !   ivar    : variable index, 1-based
  !   anis1   : horizontal anisotropy ratio (minor/major). 1.0 = isotropic.
  !   anis2   : vertical anisotropy ratio (vertical/major). 1.0 = isotropic.
  !   azimuth : azimuth of major axis (degrees, clockwise from North)
  !   dip     : dip angle (degrees, positive downward)
  !   plunge  : plunge angle (degrees)
  !=============================================================================
  integer(c_int) function krige_set_search(handle, ivar, anis1, anis2, azimuth, dip, plunge) &
      bind(C, name='krige_set_search') result(ierr)

    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ivar
    real(c_double),      intent(in), value :: anis1, anis2, azimuth, dip, plunge

    type(t_kriging), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    call obj%set_search(int(ivar), real(anis1), real(anis2), &
      real(azimuth), real(dip), real(plunge))
    ierr = int(kriging_ierr(), c_int)
  end function krige_set_search

  !=============================================================================
  ! krige_prepare
  !
  ! Prepare the kriging or SGSIM block loop.
  !=============================================================================
  integer(c_int) function krige_prepare(handle) bind(C, name='krige_prepare') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    type(t_kriging), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    call obj%prepare()
    ierr = int(kriging_ierr(), c_int)
  end function krige_prepare

  !=============================================================================
  ! krige_solve
  !
  ! Runs the kriging or SGSIM block loop.
  ! nthread: max OMP threads for this call (0 = use the OMP runtime default).
  ! After this returns, results are available via the getters below.
  !=============================================================================
  integer(c_int) function krige_solve(handle, nthread) bind(C, name='krige_solve') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: nthread
    type(t_kriging), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    if (nthread > 0) then
      call obj%solve(nthread = int(nthread))
    else
      call obj%solve()
    end if
    ierr = int(kriging_ierr(), c_int)
  end function krige_solve

  !=============================================================================
  ! Result getters
  !=============================================================================

  !-- Number of blocks = size of the estimate and variance arrays.
  integer(c_int) function krige_get_nblocks(handle, n) bind(C, name='krige_get_nblocks') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(out) :: n
    type(t_kriging), pointer :: obj
    call kriging_clear_error()
    n = 0_c_int
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    n = int(obj%block%n, c_int)
    ierr = int(kriging_ierr(), c_int)
  end function krige_get_nblocks

  !-- Number of simulations (returns 1 for plain kriging).
  integer(c_int) function krige_get_nsim(handle, n) bind(C, name='krige_get_nsim') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(out) :: n
    type(t_kriging), pointer :: obj
    call kriging_clear_error()
    n = 0_c_int
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    n = int(obj%nsim, c_int)
    ierr = int(kriging_ierr(), c_int)
  end function krige_get_nsim

  !-- Copy primary estimate(1:nblocks, 1:nsim_c, 1) into the caller-allocated out array.
  integer(c_int) function krige_get_estimate(handle, nsim_c, nblocks, out) &
      bind(C, name='krige_get_estimate') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value  :: nsim_c, nblocks
    real(c_double),      intent(out) :: out(nblocks,nsim_c)
    type(t_kriging), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    out = real(transpose(obj%block%value(1:nsim_c, 1:nblocks, 1)), c_double)
    ierr = int(kriging_ierr(), c_int)
  end function krige_get_estimate

  !-- Copy estimate into out(nblocks, nvar_c, nsim_c) — Python convention.
  !
  !   block%value is stored as (nsim, nvar, nblock) in Fortran column-major order.
  !   Python expects (nblock, nvar, nsim) so that the block index is first, matching
  !   the coord[nobs, ndim] convention used throughout the API.
  !
  !   The loop iterates over blocks and uses the 2-D TRANSPOSE intrinsic to swap
  !   the (nsim, nvar) slice into (nvar, nsim) for each block.
  !     out[ib, kvar, isim] = estimate of variable kvar+1 at block ib+1
  !                           in realization isim+1.
  integer(c_int) function krige_get_estimate_all(handle, nblocks, nvar_c, nsim_c, out) &
      bind(C, name='krige_get_estimate_all') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: nblocks, nvar_c, nsim_c
    real(c_double),      intent(out) :: out(nblocks, nvar_c, nsim_c)
    type(t_kriging), pointer :: obj
    integer :: ib
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    if (.not. allocated(obj%block%value)) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    !-- For each block, transpose the (nsim, nvar) slice → (nvar, nsim).
    do ib = 1, nblocks
      out(ib, 1:nvar_c, 1:nsim_c) = &
        real(transpose(obj%block%value(1:nsim_c, 1:nvar_c, ib)), c_double)
    end do
    ierr = int(kriging_ierr(), c_int)
  end function krige_get_estimate_all

  !-- Copy variance(1:nblocks) into the caller-allocated out array.
  integer(c_int) function krige_get_variance(handle, nblocks, out) &
      bind(C, name='krige_get_variance') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value  :: nblocks
    real(c_double),      intent(out) :: out(nblocks)
    type(t_kriging), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    if (allocated(obj%block%variance)) then
      out = real(obj%block%variance(1, 1, 1:nblocks), c_double)
    else
      out = IEEE_VALUE(0.0_c_double, IEEE_QUIET_NAN)
    end if
    ierr = int(kriging_ierr(), c_int)
  end function krige_get_variance

  !-- Copy block%variance into out(nblocks, nvar_c, nvar_c).
  !
  !   block%variance is stored in Fortran as (nvar, nvar, nblock).
  !   out is (nblock, nvar, nvar) — block index first so that Python sees
  !   out[ib, iv, jv] = covariance between variable iv+1 and jv+1 at block ib+1.
  !   The loop transposes the leading (nvar,nvar) pair against the block dimension;
  !   a plain whole-array assignment would do a flat copy and produce garbage.
  integer(c_int) function krige_get_variance_all(handle, nblocks, nvar_c, out) &
      bind(C, name='krige_get_variance_all') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: nblocks, nvar_c
    real(c_double),      intent(out) :: out(nblocks, nvar_c, nvar_c)
    type(t_kriging), pointer :: obj
    integer :: ib
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    if (allocated(obj%block%variance)) then
      do ib = 1, nblocks
        out(ib, 1:nvar_c, 1:nvar_c) = &
          real(obj%block%variance(1:nvar_c, 1:nvar_c, ib), c_double)
      end do
    else
      out = IEEE_VALUE(0.0_c_double, IEEE_QUIET_NAN)
    end if
    ierr = int(kriging_ierr(), c_int)
  end function krige_get_variance_all

  !=============================================================================
  ! Weight-store API
  !=============================================================================

  !-- Free the in-memory weight store.
  integer(c_int) function krige_free_weight_store(handle) &
      bind(C, name='krige_free_weight_store') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    type(t_kriging), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then; ierr = int(kriging_ierr(), c_int); return; end if
    call obj%free_weight_store()
    ierr = int(kriging_ierr(), c_int)
  end function krige_free_weight_store

  !-- Copy nnear[ngroups, nblock] into the caller-allocated integer buffer.
  integer(c_int) function krige_get_weight_nnear(handle, ngroups_c, nblock_c, out) &
      bind(C, name='krige_get_weight_nnear') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ngroups_c, nblock_c
    integer(c_int),      intent(out)       :: out(ngroups_c, nblock_c)
    type(t_kriging), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then; ierr = int(kriging_ierr(), c_int); return; end if
    if (.not. allocated(obj%wstore)) then
      call kriging_error('krige_get_weight_nnear', 'Weight store not allocated')
      ierr = int(kriging_ierr(), c_int); return
    end if
    out = int(obj%wstore%nnear(1:ngroups_c, 1:nblock_c), c_int)
    ierr = int(kriging_ierr(), c_int)
  end function krige_get_weight_nnear

  !-- Copy inear[nmax, ngroups, nblock] into the caller-allocated integer buffer.
  integer(c_int) function krige_get_weight_inear(handle, nmax_c, ngroups_c, nblock_c, out) &
      bind(C, name='krige_get_weight_inear') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: nmax_c, ngroups_c, nblock_c
    integer(c_int),      intent(out)       :: out(nmax_c, ngroups_c, nblock_c)
    type(t_kriging), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then; ierr = int(kriging_ierr(), c_int); return; end if
    if (.not. allocated(obj%wstore)) then
      call kriging_error('krige_get_weight_inear', 'Weight store not allocated')
      ierr = int(kriging_ierr(), c_int); return
    end if
    out = int(obj%wstore%inear(1:nmax_c, 1:ngroups_c, 1:nblock_c), c_int)
    ierr = int(kriging_ierr(), c_int)
  end function krige_get_weight_inear

  !-- Copy weight[nmax, ngroups, nblock] into the caller-allocated double buffer.
  integer(c_int) function krige_get_weight_data(handle, nmax_c, ngroups_c, nvar_c, nblock_c, out) &
      bind(C, name='krige_get_weight_data') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: nmax_c, ngroups_c, nvar_c, nblock_c
    real(c_double),      intent(out)       :: out(nmax_c, ngroups_c, nvar_c, nblock_c)
    type(t_kriging), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then; ierr = int(kriging_ierr(), c_int); return; end if
    if (.not. allocated(obj%wstore)) then
      call kriging_error('krige_get_weight_data', 'Weight store not allocated')
      ierr = int(kriging_ierr(), c_int); return
    end if
    out = real(obj%wstore%weight(1:nmax_c, 1:ngroups_c, 1:nvar_c, 1:nblock_c), c_double)
    ierr = int(kriging_ierr(), c_int)
  end function krige_get_weight_data

  !-- Copy var[nvar, nvar, nblock] from wstore into the caller-allocated double buffer.
  integer(c_int) function krige_get_weight_var(handle, nvar_c, nblock_c, out) &
      bind(C, name='krige_get_weight_var') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: nvar_c, nblock_c
    real(c_double),      intent(out)       :: out(nvar_c, nvar_c, nblock_c)
    type(t_kriging), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then; ierr = int(kriging_ierr(), c_int); return; end if
    if (.not. allocated(obj%wstore)) then
      call kriging_error('krige_get_weight_var', 'Weight store not allocated')
      ierr = int(kriging_ierr(), c_int); return
    end if
    if (.not. allocated(obj%wstore%var)) then
      call kriging_error('krige_get_weight_var', 'Variance not stored (recompile with updated library)')
      ierr = int(kriging_ierr(), c_int); return
    end if
    out = real(obj%wstore%var(1:nvar_c, 1:nvar_c, 1:nblock_c), c_double)
    ierr = int(kriging_ierr(), c_int)
  end function krige_get_weight_var

  !-- Set nnear, inear, weight, and variance from caller-supplied arrays.
  !   Allocates (or re-allocates) wstore and sets use_old_weight=.true. so that
  !   solve() applies the supplied weights directly without re-solving the system.
  integer(c_int) function krige_set_weights(handle, nmax_c, ngroups_c, nvar_c, nblock_c, &
                                             nnear_in, inear_in, weight_in, order_in, var_in) &
      bind(C, name='krige_set_weights') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: nmax_c, ngroups_c, nvar_c, nblock_c
    integer(c_int),      intent(in)        :: nnear_in (ngroups_c, nblock_c)
    integer(c_int),      intent(in)        :: inear_in (nmax_c, ngroups_c, nblock_c)
    real(c_double),      intent(in)        :: weight_in(nmax_c, ngroups_c, nvar_c, nblock_c)
    integer(c_int),      intent(in)        :: order_in (nblock_c)
    real(c_double),      intent(in)        :: var_in   (nvar_c, nvar_c, nblock_c)
    type(t_kriging), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then; ierr = int(kriging_ierr(), c_int); return; end if
    if (.not. allocated(obj%wstore)) call obj%alloc_weight_store()
    if (kriging_failed()) then; ierr = int(kriging_ierr(), c_int); return; end if
    call obj%set_weights(int(nnear_in), int(inear_in), real(weight_in), int(order_in), real(var_in))
    obj%use_old_weight = .true.
    ierr = int(kriging_ierr(), c_int)
  end function krige_set_weights

  integer(c_int) function krige_get_last_error(buffer, nbuf) &
      bind(C, name='krige_get_last_error') result(ierr)
    character(kind=c_char), intent(out) :: buffer(*)
    integer(c_int), intent(in), value :: nbuf
    call kriging_copy_error(buffer, int(nbuf))
    ierr = int(kriging_ierr(), c_int)
  end function krige_get_last_error

  !-- Return a string representation of the kriging object.
  integer(c_intptr_t) function krige_to_str(handle) result(ptr) bind(C, name='krige_to_str')
    integer(c_intptr_t), intent(in), value :: handle
    type(t_kriging), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ptr = 0_c_intptr_t
      return
    end if
    call obj%update_info()
    ptr = transfer(c_loc(obj%krige_info(1)), ptr)
  end function krige_to_str

  !=============================================================================
  ! Internal helpers (private to this module)
  !=============================================================================

  !-- Recover a typed Fortran pointer from the opaque handle.  Invalid or stale
  !   handles record a normal kriging error instead of dereferencing garbage.
  subroutine get_obj(handle, obj)
    integer(c_intptr_t), intent(in), value :: handle
    type(t_kriging),     pointer    :: obj
    integer :: idx
    nullify(obj)
    if (handle == 0_c_intptr_t) then
      call kriging_error('kriging_capi', 'Null kriging object handle')
      return
    end if
    idx = int(handle)
    if (.not. allocated(kriging_registry) .or. idx < 1 .or. idx > size(kriging_registry)) then
      call kriging_error('kriging_capi', 'Invalid kriging object handle')
      return
    end if
    if (associated(kriging_registry(idx)%obj)) obj => kriging_registry(idx)%obj
    if (.not. associated(obj)) &
      call kriging_error('kriging_capi', 'Invalid kriging object handle')
  end subroutine get_obj

  subroutine store_obj(obj, handle)
    ! Store a newly allocated object in the first free slot.  Growing the
    ! registry keeps old slot numbers stable for existing Python objects.
    type(t_kriging), pointer, intent(in) :: obj
    integer(c_intptr_t), intent(out) :: handle
    integer :: i

    if (.not. allocated(kriging_registry)) allocate(kriging_registry(16))

    do i = 1, size(kriging_registry)
      if (.not. associated(kriging_registry(i)%obj)) then
        kriging_registry(i)%obj => obj
        handle = int(i, c_intptr_t)
        return
      end if
    end do

    call grow_registry()
    do i = 1, size(kriging_registry)
      if (.not. associated(kriging_registry(i)%obj)) then
        kriging_registry(i)%obj => obj
        handle = int(i, c_intptr_t)
        return
      end if
    end do

    handle = 0_c_intptr_t
    call kriging_error('krige_create', 'Failed to allocate a kriging handle slot')
  end subroutine store_obj

  subroutine release_obj(handle)
    ! Destroy leaves the slot reusable but does not compact the registry because
    ! compacting would change handles held by other Python objects.
    integer(c_intptr_t), intent(in), value :: handle
    integer :: idx
    idx = int(handle)
    if (allocated(kriging_registry) .and. idx >= 1 .and. idx <= size(kriging_registry)) &
      nullify(kriging_registry(idx)%obj)
  end subroutine release_obj

  subroutine grow_registry()
    ! Double the slot array and preserve pointer associations manually; ordinary
    ! assignment would copy the slot values but move_alloc makes ownership clear.
    type(kriging_handle_slot), allocatable :: tmp(:)
    integer :: i, old_n, new_n

    old_n = size(kriging_registry)
    new_n = max(1, old_n * 2)
    allocate(tmp(new_n))
    do i = 1, old_n
      if (associated(kriging_registry(i)%obj)) tmp(i)%obj => kriging_registry(i)%obj
    end do
    call move_alloc(tmp, kriging_registry)
  end subroutine grow_registry

  !-- Convert a null-terminated C string to a Fortran character(len=1024).
  function c2fstr(cstr) result(fstr)
    character(kind=c_char), intent(in) :: cstr(*)
    character(len=1024) :: fstr
    integer :: i
    fstr = ''
    do i = 1, 1024
      if (cstr(i) == c_null_char) exit
      fstr(i:i) = cstr(i)
    end do
  end function c2fstr

  !-- Convert integer(c_int) flag (0/1) to Fortran logical.
  !   Only 1 maps to .true.; 0 (and any other value) maps to .false.
  elemental function l(v) result(r)
    integer(c_int), intent(in), value :: v
    logical :: r
    r = (v == 1_c_int)
  end function l

  !=============================================================================
  ! krige_get_max_threads / krige_get_num_threads
  !
  ! Query the OpenMP thread count from Python so callers can verify that
  ! parallelism is active without needing to inspect environment variables.
  !
  ! When the library is compiled WITHOUT OpenMP (--no-openmp), both routines
  ! return 1 so Python code can treat the result uniformly.
  !=============================================================================
#ifdef _OPENMP
  subroutine krige_get_max_threads(n) bind(C, name='krige_get_max_threads')
    use omp_lib
    integer(c_int), intent(out) :: n
    n = int(omp_get_max_threads(), c_int)
  end subroutine krige_get_max_threads

  subroutine krige_get_num_threads(n) bind(C, name='krige_get_num_threads')
    use omp_lib
    integer(c_int), intent(out) :: n
    !$OMP PARALLEL
    !$OMP SINGLE
    n = int(omp_get_num_threads(), c_int)
    !$OMP END SINGLE
    !$OMP END PARALLEL
  end subroutine krige_get_num_threads
#else
  subroutine krige_get_max_threads(n) bind(C, name='krige_get_max_threads')
    integer(c_int), intent(out) :: n
    n = 1_c_int   ! OpenMP not compiled in
  end subroutine krige_get_max_threads

  subroutine krige_get_num_threads(n) bind(C, name='krige_get_num_threads')
    integer(c_int), intent(out) :: n
    n = 1_c_int   ! OpenMP not compiled in
  end subroutine krige_get_num_threads
#endif

end module kriging_capi
