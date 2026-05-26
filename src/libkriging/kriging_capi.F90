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
  use iso_c_binding
  use kriging, only: t_kriging
  implicit none
  private

contains

  !=============================================================================
  ! Lifecycle: create / destroy
  !=============================================================================

  !-- Allocate a new t_kriging object on the heap and return its address
  !   as an opaque 64-bit integer handle.  Python stores this handle and
  !   passes it back on every subsequent call.
  subroutine krige_create(handle) bind(C, name='krige_create')
    integer(c_intptr_t), intent(out) :: handle
    type(t_kriging), pointer :: obj
    allocate(obj)
    handle = transfer(c_loc(obj), handle)
  end subroutine krige_create

  !-- Finalize and deallocate the object; zero the handle so stale use is
  !   caught early.
  subroutine krige_destroy(handle) bind(C, name='krige_destroy')
    integer(c_intptr_t), intent(inout) :: handle
    type(t_kriging), pointer :: obj
    call get_obj(handle, obj)
    call obj%finalize()
    deallocate(obj)
    handle = 0_c_intptr_t
  end subroutine krige_destroy

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
  !   sk_mean            : global mean for simple kriging (unbias=0)
  !   seed               : random seed for SGSIM (0 = use clock)
  !=============================================================================
  subroutine krige_initialize(handle, &
      ndim, nvar, ndrift, unbias, nsim, &
      anisotropic_search, weight_correction, use_old_weight, &
      store_weight, cross_validation, write_mat, neglect_error, varying_vgm, verbose, &
      weight_file, bounds, sk_mean, seed) &
      bind(C, name='krige_initialize')

    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ndim, nvar, ndrift, unbias, nsim, seed
    integer(c_int),      intent(in), value :: anisotropic_search, weight_correction
    integer(c_int),      intent(in), value :: use_old_weight, store_weight
    integer(c_int),      intent(in), value :: cross_validation, write_mat, neglect_error
    integer(c_int),      intent(in), value :: varying_vgm, verbose
    character(kind=c_char), intent(in) :: weight_file(*)
    real(c_double),      intent(in) :: bounds(2)
    real(c_double),      intent(in), value :: sk_mean

    type(t_kriging), pointer :: obj
    !-- Local copy avoids an implicit array temporary for the 'bounds' argument
    !   (Intel warning 406: "array temporary created for argument #N").
    !   real(c_double) and the default real kind are both 8-byte with the
    !   compiler flags used (/real-size:64 / -fdefault-real-8), so the
    !   assignment is lossless.
    real :: fbounds(2)
    call get_obj(handle, obj)
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
      sk_mean            = real(sk_mean), &
      seed               = int(seed))
  end subroutine krige_initialize

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
  !=============================================================================
  subroutine krige_set_obs(handle, ivar, nobs, ndim_c, &
      coord, value, variance, nmax, maxdist) &
      bind(C, name='krige_set_obs')

    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ivar, nobs, ndim_c
    real(c_double),      intent(in) :: coord(ndim_c, nobs)
    real(c_double),      intent(in) :: value(nobs)
    real(c_double),      intent(in) :: variance(nobs)
    integer(c_int),      intent(in), value :: nmax
    real(c_double),      intent(in), value :: maxdist

    type(t_kriging), pointer :: obj
    call get_obj(handle, obj)

    call obj%set_obs(int(ivar), real(coord), real(value), &
      variance = real(variance), &
      nmax     = int(nmax), &
      maxdist  = real(maxdist))
  end subroutine krige_set_obs

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
  subroutine krige_set_obs_drift(handle, ivar, ndrift_c, nobs, drift) &
      bind(C, name='krige_set_obs_drift')

    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ivar, ndrift_c, nobs
    real(c_double),      intent(in) :: drift(ndrift_c, nobs)

    type(t_kriging), pointer :: obj
    call get_obj(handle, obj)
    call obj%set_obs_drift(int(ivar), real(drift))
  end subroutine krige_set_obs_drift

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
  subroutine krige_set_vgm(handle, ivar, jvar, vtype, &
                            nugget, sill, a_major, a_minor1, a_minor2, &
                            azimuth, dip, plunge) &
      bind(C, name='krige_set_vgm')

    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ivar, jvar
    character(kind=c_char), intent(in) :: vtype(*)
    real(c_double), intent(in), value :: nugget, sill, a_major, a_minor1, a_minor2
    real(c_double), intent(in), value :: azimuth, dip, plunge

    type(t_kriging), pointer :: obj
    call get_obj(handle, obj)
    call obj%set_vgm(int(ivar), int(jvar), c2fstr(vtype), &
                     real(nugget), real(sill), real(a_major), &
                     real(a_minor1), real(a_minor2), &
                     real(azimuth), real(dip), real(plunge))
  end subroutine krige_set_vgm

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
  subroutine krige_set_vgm_block(handle, ivar, jvar, ib, vtype, &
                                  nugget, sill, a_major, a_minor1, a_minor2, &
                                  azimuth, dip, plunge) &
      bind(C, name='krige_set_vgm_block')

    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ivar, jvar, ib
    character(kind=c_char), intent(in) :: vtype(*)
    real(c_double), intent(in), value :: nugget, sill, a_major, a_minor1, a_minor2
    real(c_double), intent(in), value :: azimuth, dip, plunge

    type(t_kriging), pointer :: obj
    call get_obj(handle, obj)
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
  end subroutine krige_set_vgm_block

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
  subroutine krige_set_grid(handle, ngrid, ndim_c, coord, &
      rangescale, localnugget) &
      bind(C, name='krige_set_grid')

    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ngrid, ndim_c
    real(c_double),      intent(in) :: coord(ndim_c, ngrid)
    real(c_double),      intent(in) :: rangescale(ngrid)
    real(c_double),      intent(in) :: localnugget(ngrid)

    type(t_kriging), pointer :: obj
    call get_obj(handle, obj)
    call obj%set_grid(coord       = real(coord), &
                      rangescale  = real(rangescale), &
                      localnugget = real(localnugget))
  end subroutine krige_set_grid

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
  subroutine krige_set_grid_block(handle, block_type, &
      ngrid, ndim_c, coord, &
      nblock, nblockpnt, pointweight, &
      rangescale, localnugget) &
      bind(C, name='krige_set_grid_block')

    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: block_type
    integer(c_int),      intent(in), value :: ngrid, ndim_c
    real(c_double),      intent(in) :: coord(ndim_c, ngrid)
    integer(c_int),      intent(in), value :: nblock
    integer(c_int),      intent(in) :: nblockpnt(nblock)
    real(c_double),      intent(in) :: pointweight(*)   ! length = sum(nblockpnt)
    real(c_double),      intent(in) :: rangescale(nblock)
    real(c_double),      intent(in) :: localnugget(nblock)

    integer :: npw
    type(t_kriging), pointer :: obj
    call get_obj(handle, obj)
    npw = sum(nblockpnt)   ! derive length instead of receiving it as an argument
    call obj%set_grid(coord       = real(coord), &
                      block_type  = int(block_type), &
                      nblockpnt   = int(nblockpnt), &
                      pointweight = real(pointweight(1:npw)), &
                      rangescale  = real(rangescale), &
                      localnugget = real(localnugget))
  end subroutine krige_set_grid_block

  !=============================================================================
  ! krige_set_grid_cv
  !
  ! Sets up the grid for cross-validation mode.  No coord is needed; Fortran
  ! derives the grid from the observation coordinates automatically.
  ! Call instead of krige_set_grid when cross_validation=1.
  !=============================================================================
  subroutine krige_set_grid_cv(handle) bind(C, name='krige_set_grid_cv')
    integer(c_intptr_t), intent(in), value :: handle
    type(t_kriging), pointer :: obj
    call get_obj(handle, obj)
    call obj%set_grid()
  end subroutine krige_set_grid_cv

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
  subroutine krige_set_grid_drift(handle, ndrift_c, nblocks, drift) &
      bind(C, name='krige_set_grid_drift')

    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ndrift_c, nblocks
    real(c_double),      intent(in) :: drift(ndrift_c, nblocks)

    type(t_kriging), pointer :: obj
    call get_obj(handle, obj)
    call obj%set_grid_drift(real(drift))
  end subroutine krige_set_grid_drift

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
  !   sample   : pre-drawn standard-normal samples [nsim_c, nblocks]
  !
  ! Note: randpath length and sample second dimension are both nblocks, so a
  ! single parameter covers both — no separate n_rp / n_s needed.
  !=============================================================================
  subroutine krige_set_sim(handle, nblocks, randpath, nsim_c, sample) &
      bind(C, name='krige_set_sim')

    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: nblocks
    integer(c_int),      intent(in) :: randpath(nblocks)
    integer(c_int),      intent(in), value :: nsim_c
    real(c_double),      intent(in) :: sample(nsim_c, nblocks)

    type(t_kriging), pointer :: obj
    call get_obj(handle, obj)
    call obj%set_sim(randpath = int(randpath), sample = real(sample))
  end subroutine krige_set_sim

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
  subroutine krige_set_search(handle, ivar, anis1, anis2, azimuth, dip, plunge) &
      bind(C, name='krige_set_search')

    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ivar
    real(c_double),      intent(in), value :: anis1, anis2, azimuth, dip, plunge

    type(t_kriging), pointer :: obj
    call get_obj(handle, obj)
    call obj%set_search(int(ivar), real(anis1), real(anis2), &
      real(azimuth), real(dip), real(plunge))
  end subroutine krige_set_search

  !=============================================================================
  ! krige_prepare
  !
  ! Prepare the kriging or SGSIM block loop.
  !=============================================================================
  subroutine krige_prepare(handle) bind(C, name='krige_prepare')
    integer(c_intptr_t), intent(in), value :: handle
    type(t_kriging), pointer :: obj
    call get_obj(handle, obj)
    call obj%prepare()
  end subroutine krige_prepare

  !=============================================================================
  ! krige_solve
  !
  ! Runs the kriging or SGSIM block loop.
  ! After this returns, results are available via the getters below.
  !=============================================================================
  subroutine krige_solve(handle) bind(C, name='krige_solve')
    integer(c_intptr_t), intent(in), value :: handle
    type(t_kriging), pointer :: obj
    call get_obj(handle, obj)
    call obj%solve()
  end subroutine krige_solve

  !=============================================================================
  ! Result getters
  !=============================================================================

  !-- Number of blocks = size of the estimate and variance arrays.
  subroutine krige_get_nblocks(handle, n) bind(C, name='krige_get_nblocks')
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(out) :: n
    type(t_kriging), pointer :: obj
    call get_obj(handle, obj)
    n = int(obj%block%n, c_int)
  end subroutine krige_get_nblocks

  !-- Number of simulations (returns 1 for plain kriging).
  subroutine krige_get_nsim(handle, n) bind(C, name='krige_get_nsim')
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(out) :: n
    type(t_kriging), pointer :: obj
    call get_obj(handle, obj)
    n = max(int(obj%nsim, c_int), 1_c_int)
  end subroutine krige_get_nsim

  !-- Copy estimate(1:nsim_c, 1:nblocks) into the caller-allocated out array.
  subroutine krige_get_estimate(handle, nsim_c, nblocks, out) &
      bind(C, name='krige_get_estimate')
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value  :: nsim_c, nblocks
    real(c_double),      intent(out) :: out(nsim_c, nblocks)
    type(t_kriging), pointer :: obj
    call get_obj(handle, obj)
    out = real(obj%block%estimate(1:nsim_c, 1:nblocks), c_double)
  end subroutine krige_get_estimate

  !-- Copy variance(1:nblocks) into the caller-allocated out array.
  subroutine krige_get_variance(handle, nblocks, out) &
      bind(C, name='krige_get_variance')
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value  :: nblocks
    real(c_double),      intent(out) :: out(nblocks)
    type(t_kriging), pointer :: obj
    call get_obj(handle, obj)
    out = real(obj%block%variance(1:nblocks), c_double)
  end subroutine krige_get_variance

  !-- Return a string representation of the kriging object.
  function krige_to_str(self) result(ptr) bind(C, name='krige_to_str')
    type(t_kriging), intent(in) :: self
    type(c_ptr) :: ptr
    character(len=:), allocatable :: info
    call self%update_info()
    ptr = c_loc(self%krige_info(1))
  end function

  !=============================================================================
  ! Internal helpers (private to this module)
  !=============================================================================

  !-- Recover a typed Fortran pointer from the opaque handle.
  subroutine get_obj(handle, obj)
    integer(c_intptr_t), intent(in), value :: handle
    type(t_kriging),     pointer    :: obj
    type(c_ptr) :: cptr
    cptr = transfer(handle, cptr)
    call c_f_pointer(cptr, obj)
  end subroutine get_obj

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