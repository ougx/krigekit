!==============================================================================
! kriging_st_capi.f90
!
! ISO C Binding wrapper for the t_kriging_st Fortran module (space-time).
! Contains only t_kriging_st-specific entry points; all shared base operations
! (prepare, solve, get_estimate, weight store, etc.) live in
! kriging_capi_common.F90 under the krige_* C-name prefix.
!
! ST-specific entry points (krige_st_ prefix)
! -------------------------------------------
!   Lifecycle  : krige_st_create, krige_st_destroy
!   Init       : krige_st_initialize  (always ndim=3, no varying_vgm/pf_cache)
!   Obs        : krige_st_set_obs     (coord includes time as last row)
!   ST model   : krige_st_set_st_model, krige_st_set_vgm_temporal,
!                krige_st_set_vgm_joint_sills
!   Variogram  : krige_st_reset_vgm, krige_st_set_vgm  (same shape as spatial
!                but dispatched to t_kriging_st%set_vgm)
!   Grid       : krige_st_set_grid, krige_st_set_grid_block,
!                krige_st_set_grid_drift  (broadcasts ivar=-1)
!   Sim        : krige_st_set_sim   (2-D sample array; reshaped to 3-D)
!   Search     : krige_st_set_search (adds time_at)
!   Gradient   : krige_st_set_grad  (coord includes time as last row)
!   Threads    : krige_st_get_max_threads, krige_st_get_num_threads
!==============================================================================
module kriging_st_capi
  use, intrinsic :: ieee_arithmetic
  use iso_c_binding
  use kriging_st,           only: t_kriging_st
  use kriging_base,         only: t_kriging_base
  use kriging_capi_common,  only: get_obj_base, store_obj_base, release_obj_base, c2fstr, l
  use kriging_err,          only: kriging_clear_error, kriging_ierr, kriging_error, kriging_failed
  use variogram_st,         only: ST_MODEL_SUM_METRIC, ST_MODEL_PRODUCT_SUM
  use vgm_func,             only: vtype_from_str
  implicit none
  private

contains

  !=============================================================================
  ! Lifecycle
  !=============================================================================

  integer(c_int) function krige_st_create(handle) bind(C, name='krige_st_create') result(ierr)
    integer(c_intptr_t), intent(out) :: handle
    type(t_kriging_st), pointer :: obj
    class(t_kriging_base), pointer :: base
    integer :: stat
    call kriging_clear_error()
    handle = 0_c_intptr_t
    allocate(obj, stat=stat)
    if (stat /= 0) then
      call kriging_error('krige_st_create', 'Failed to allocate t_kriging_st object')
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    base => obj
    call store_obj_base(base, handle)
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_create

  integer(c_int) function krige_st_destroy(handle) bind(C, name='krige_st_destroy') result(ierr)
    integer(c_intptr_t), intent(inout) :: handle
    type(t_kriging_st), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    call obj%finalize()
    deallocate(obj)
    call release_obj_base(handle)
    handle = 0_c_intptr_t
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_destroy

  !=============================================================================
  ! krige_st_initialize — ST-specific: always ndim=3; no varying_vgm/std_ck/pf_cache
  !=============================================================================
  integer(c_int) function krige_st_initialize(handle, &
      nvar, ndrift, unbias, nsim, &
      anisotropic_search, weight_correction, use_old_weight, &
      store_weight, cross_validation, write_mat, neglect_error, verbose, &
      weight_file, bounds, seed) &
      bind(C, name='krige_st_initialize') result(ierr)

    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: nvar, ndrift, unbias, nsim, seed
    integer(c_int),      intent(in), value :: anisotropic_search, weight_correction
    integer(c_int),      intent(in), value :: use_old_weight, store_weight
    integer(c_int),      intent(in), value :: cross_validation, write_mat, neglect_error, verbose
    character(kind=c_char), intent(in)     :: weight_file(*)
    real(c_double),      intent(in)        :: bounds(2)

    type(t_kriging_st), pointer :: obj
    real :: fbounds(2)
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    fbounds = real(bounds)

    call obj%initialize( &
      ndim               = 3, &
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
      verbose            = l(verbose), &
      weight_file        = c2fstr(weight_file), &
      bounds             = fbounds, &
      seed               = int(seed))
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_initialize

  !=============================================================================
  ! krige_st_set_st_model — global ST model parameters
  !=============================================================================
  integer(c_int) function krige_st_set_st_model(handle, model, transform, at, &
      time_nugget, time_sill, k_ps) &
      bind(C, name='krige_st_set_st_model') result(ierr)
    integer(c_intptr_t),    intent(in), value :: handle
    character(kind=c_char), intent(in)        :: model(*), transform(*)
    real(c_double),         intent(in), value :: at, time_nugget, time_sill, k_ps
    type(t_kriging_st), pointer :: obj
    integer :: imodel, itransform
    character(len=1024) :: mstr
    call kriging_clear_error()
    mstr = c2fstr(model)
    if (trim(mstr) == 'sum_metric') then
      imodel = ST_MODEL_SUM_METRIC
    else if (trim(mstr) == 'product_sum') then
      imodel = ST_MODEL_PRODUCT_SUM
    else
      call kriging_error('krige_st_set_st_model', &
        'model must be "sum_metric" or "product_sum", got: '//trim(mstr))
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    itransform = vtype_from_str(c2fstr(transform))
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    call obj%set_st_model(imodel, itransform, real(at), real(k_ps), &
                          real(time_nugget), real(time_sill))
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_set_st_model

  !=============================================================================
  ! krige_st_set_obs — coord(ndim+1, nobs): rows 1:ndim spatial, ndim+1 = time
  !=============================================================================
  integer(c_int) function krige_st_set_obs(handle, ivar, nobs, ndim, &
      coord, value, variance, nmax, maxdist, sk_mean) &
      bind(C, name='krige_st_set_obs') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ivar, nobs, ndim, nmax
    real(c_double),      intent(in)        :: coord(ndim+1, nobs)
    real(c_double),      intent(in)        :: value(nobs)
    real(c_double),      intent(in)        :: variance(nobs)
    real(c_double),      intent(in), value :: maxdist, sk_mean
    type(t_kriging_st), pointer :: obj
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
  end function krige_st_set_obs

  !=============================================================================
  ! krige_st_set_obs_drift
  !   Kept here because krige_set_obs_drift (common) is identical in body
  !   but this alias exists for API symmetry with the other krige_st_* names.
  !   TODO: Consider removing and redirecting Python to krige_set_obs_drift.
  !=============================================================================

  !=============================================================================
  ! krige_st_set_grad — coord1/coord2 include time as the last row
  !=============================================================================
  integer(c_int) function krige_st_set_grad(handle, ivar, ngrad, ndim, coord1, coord2, &
      grad_val, variance, ndrift_c, drift_ext) &
      bind(C, name='krige_st_set_grad') result(ierr)

    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ivar, ngrad, ndim, ndrift_c
    real(c_double),      intent(in) :: coord1(ndim+1, max(ngrad,1))
    real(c_double),      intent(in) :: coord2(ndim+1, max(ngrad,1))
    real(c_double),      intent(in) :: grad_val(max(ngrad,1))
    real(c_double),      intent(in) :: variance(max(ngrad,1))
    real(c_double),      intent(in) :: drift_ext(max(ndrift_c,1), max(ngrad,1))

    type(t_kriging_st), pointer :: obj
    real :: c1(ndim+1, max(ngrad,1)), c2(ndim+1, max(ngrad,1))
    real :: gv(max(ngrad,1)), gvar(max(ngrad,1))

    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if

    if (ngrad == 0) then
      call obj%reset_grad(int(ivar))
    else
      c1   = real(coord1)
      c2   = real(coord2)
      gv   = real(grad_val)
      gvar = real(variance)
      if (ndrift_c > 0) then
        call obj%set_grad(int(ivar), c1, c2, gv, variance=gvar, &
                          drift_ext=real(drift_ext))
      else
        call obj%set_grad(int(ivar), c1, c2, gv, variance=gvar)
      end if
    end if
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_set_grad

  !=============================================================================
  ! Variogram (set_vgm / reset_vgm are not on t_kriging_base)
  !=============================================================================

  integer(c_int) function krige_st_reset_vgm(handle, ivar, jvar) &
      bind(C, name='krige_st_reset_vgm') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ivar, jvar
    type(t_kriging_st), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    call obj%reset_vgm(int(ivar), int(jvar))
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_reset_vgm

  integer(c_int) function krige_st_set_vgm(handle, ivar, jvar, vtype, &
      nugget, sill, a_major, a_minor1, a_minor2, &
      azimuth, dip, plunge) &
      bind(C, name='krige_st_set_vgm') result(ierr)
    integer(c_intptr_t),    intent(in), value :: handle
    integer(c_int),         intent(in), value :: ivar, jvar
    character(kind=c_char), intent(in)        :: vtype(*)
    real(c_double),         intent(in), value :: nugget, sill, a_major, a_minor1, a_minor2
    real(c_double),         intent(in), value :: azimuth, dip, plunge
    type(t_kriging_st), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    call obj%set_vgm(int(ivar), int(jvar), &
                     vtype    = c2fstr(vtype), &
                     nugget   = real(nugget), &
                     sill     = real(sill), &
                     a_major  = real(a_major), &
                     a_minor1 = real(a_minor1), &
                     a_minor2 = real(a_minor2), &
                     azimuth  = real(azimuth), &
                     dip      = real(dip), &
                     plunge   = real(plunge))
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_set_vgm

  integer(c_int) function krige_st_set_vgm_temporal(handle, ivar, jvar, vtype, &
      nugget, sill, at_k) &
      bind(C, name='krige_st_set_vgm_temporal') result(ierr)
    integer(c_intptr_t),    intent(in), value :: handle
    integer(c_int),         intent(in), value :: ivar, jvar
    character(kind=c_char), intent(in)        :: vtype(*)
    real(c_double),         intent(in), value :: nugget, sill, at_k
    type(t_kriging_st), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    call obj%set_vgm_temporal(int(ivar), int(jvar), &
                               vtype  = c2fstr(vtype), &
                               nugget = real(nugget), &
                               sill   = real(sill), &
                               at_k   = real(at_k))
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_set_vgm_temporal

  integer(c_int) function krige_st_set_vgm_joint_sills(handle, ivar, jvar, n, sills) &
      bind(C, name='krige_st_set_vgm_joint_sills') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ivar, jvar, n
    real(c_double),      intent(in)        :: sills(n)
    type(t_kriging_st), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    call obj%set_vgm_joint_sills(int(ivar), int(jvar), real(sills), int(n))
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_set_vgm_joint_sills

  !=============================================================================
  ! Grid setup (ST-specific: coord(3,n) + separate time(n) array)
  !=============================================================================

  integer(c_int) function krige_st_set_grid(handle, ngrid, coord, time, rangescale, localnugget) &
      bind(C, name='krige_st_set_grid') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ngrid
    real(c_double),      intent(in)        :: coord(3, ngrid)
    real(c_double),      intent(in)        :: time(ngrid)
    real(c_double),      intent(in)        :: rangescale(ngrid)
    real(c_double),      intent(in)        :: localnugget(ngrid)
    type(t_kriging_st), pointer :: obj
    real, allocatable :: coord4(:,:)
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    allocate(coord4(4, ngrid))
    coord4(1:3, :) = real(coord)
    coord4(4, :)   = real(time)
    call obj%set_grid_point(coord4, rangescale=real(rangescale), &
      localnugget=real(localnugget), value_fill=0.0, variance_fill=0.0)
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_set_grid

  integer(c_int) function krige_st_set_grid_block(handle, nblocks, coord, time, &
      nblockpnt, npnts_total, blockcoord, blocktime, pointweight, &
      rangescale, localnugget) &
      bind(C, name='krige_st_set_grid_block') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: nblocks, npnts_total
    real(c_double),      intent(in)        :: coord(3, nblocks)
    real(c_double),      intent(in)        :: time(nblocks)
    integer(c_int),      intent(in)        :: nblockpnt(nblocks)
    real(c_double),      intent(in)        :: blockcoord(3, npnts_total)
    real(c_double),      intent(in)        :: blocktime(npnts_total)
    real(c_double),      intent(in)        :: pointweight(npnts_total)
    real(c_double),      intent(in)        :: rangescale(nblocks)
    real(c_double),      intent(in)        :: localnugget(nblocks)
    type(t_kriging_st), pointer :: obj
    real, allocatable :: coord4(:,:), blockcoord4(:,:)
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    allocate(coord4(4, nblocks))
    coord4(1:3, :) = real(coord)
    coord4(4, :)   = real(time)
    allocate(blockcoord4(4, npnts_total))
    blockcoord4(1:3, :) = real(blockcoord)
    blockcoord4(4, :)   = real(blocktime)
    call obj%set_grid_block( &
      coord4, int(nblockpnt), blockcoord4, real(pointweight), &
      localnugget=real(localnugget), rangescale=real(rangescale), &
      value_fill=0.0, variance_fill=0.0)
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_set_grid_block

  !-- Grid drift: always broadcasts to all variables (ivar=-1).
  !   (Spatial krige_set_grid_drift accepts an explicit ivar.)
  integer(c_int) function krige_st_set_grid_drift(handle, ndrift_c, nblocks, drift) &
      bind(C, name='krige_st_set_grid_drift') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ndrift_c, nblocks
    real(c_double),      intent(in)        :: drift(ndrift_c, nblocks)
    type(t_kriging_st), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    call obj%set_grid_drift(real(drift), ivar=-1)
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_set_grid_drift

  !=============================================================================
  ! SGSIM — ST sample array is 2-D (nsim, nblocks); reshaped to 3-D internally.
  !
  ! randpath_ptr  NULL  → Fortran generates a random permutation.
  ! sample_ptr    NULL  → Fortran generates N(0,1) samples.
  !               non-NULL → caller-supplied (nsim, nblocks); reshaped to
  !                          (nsim, nvar, nblocks) with zeros for extra vars.
  !=============================================================================
  integer(c_int) function krige_st_set_sim(handle, nblocks, randpath_ptr, nsim_c, sample_ptr) &
      bind(C, name='krige_st_set_sim') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: nblocks, nsim_c
    type(c_ptr),         intent(in), value :: randpath_ptr
    type(c_ptr),         intent(in), value :: sample_ptr
    type(t_kriging_st), pointer :: obj
    integer(c_int), pointer :: rp_f(:)
    real(c_double), pointer :: sample_f(:,:)
    real, allocatable :: sample3(:,:,:)

    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if

    if (c_associated(sample_ptr)) then
      call c_f_pointer(sample_ptr, sample_f, [nsim_c, nblocks])
      allocate(sample3(nsim_c, obj%nvar, nblocks))
      sample3 = 0.0
      if (obj%nvar >= 1) sample3(:, 1, :) = real(sample_f)
    end if

    if (c_associated(randpath_ptr) .and. c_associated(sample_ptr)) then
      call c_f_pointer(randpath_ptr, rp_f, [nblocks])
      call obj%set_sim(randpath=int(rp_f), sample=sample3)
    else if (c_associated(randpath_ptr)) then
      call c_f_pointer(randpath_ptr, rp_f, [nblocks])
      call obj%set_sim(randpath=int(rp_f))
    else if (c_associated(sample_ptr)) then
      call obj%set_sim(sample=sample3)
    else
      call obj%set_sim()
    end if
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_set_sim

  !=============================================================================
  ! krige_st_set_search — adds time_at (spatial omits it)
  !=============================================================================
  integer(c_int) function krige_st_set_search(handle, ivar, &
      time_at, anis1, anis2, azimuth, dip, plunge, sector_search) &
      bind(C, name='krige_st_set_search') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ivar
    real(c_double),      intent(in), value :: time_at
    real(c_double),      intent(in), value :: anis1, anis2, azimuth, dip, plunge
    integer(c_int),      intent(in), value :: sector_search
    type(t_kriging_st), pointer :: obj
    real :: f_time_at, f_anis1, f_anis2, f_azimuth, f_dip, f_plunge
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    f_anis1   = real(anis1)
    f_anis2   = real(anis2)
    f_azimuth = real(azimuth)
    f_dip     = real(dip)
    f_plunge  = real(plunge)
    !-- Pre-set time_at on obs BEFORE calling set_search.  set_search reads
    !   obs%time_at when present(time_at) fails (gfortran CLASS polymorphism
    !   does not set the presence flag for optional arguments passed through
    !   a non-polymorphic concrete value), so this ensures the correct value
    !   is used for both the KD-tree build and the search.
    obj%obs(int(ivar))%time_at = real(time_at)
    call obj%set_search(int(ivar), &
      anis1=f_anis1, anis2=f_anis2, &
      azimuth=f_azimuth, dip=f_dip, plunge=f_plunge, &
      sector_search=l(sector_search))
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_set_search

  !=============================================================================
  ! krige_st_get_max_threads / krige_st_get_num_threads
  !=============================================================================
#ifdef _OPENMP
  subroutine krige_st_get_max_threads(n) bind(C, name='krige_st_get_max_threads')
    use omp_lib
    integer(c_int), intent(out) :: n
    n = int(omp_get_max_threads(), c_int)
  end subroutine krige_st_get_max_threads

  subroutine krige_st_get_num_threads(n) bind(C, name='krige_st_get_num_threads')
    use omp_lib
    integer(c_int), intent(out) :: n
    !$OMP PARALLEL
    !$OMP SINGLE
    n = int(omp_get_num_threads(), c_int)
    !$OMP END SINGLE
    !$OMP END PARALLEL
  end subroutine krige_st_get_num_threads
#else
  subroutine krige_st_get_max_threads(n) bind(C, name='krige_st_get_max_threads')
    integer(c_int), intent(out) :: n
    n = 1_c_int
  end subroutine krige_st_get_max_threads

  subroutine krige_st_get_num_threads(n) bind(C, name='krige_st_get_num_threads')
    integer(c_int), intent(out) :: n
    n = 1_c_int
  end subroutine krige_st_get_num_threads
#endif

  !=============================================================================
  ! Internal helpers (private to this module)
  !=============================================================================

  subroutine get_obj(handle, obj)
    integer(c_intptr_t),  intent(in), value :: handle
    type(t_kriging_st),   pointer           :: obj
    class(t_kriging_base), pointer :: base
    nullify(obj)
    call get_obj_base(handle, base)
    if (kriging_failed()) return
    select type(base)
    type is (t_kriging_st)
      obj => base
    class default
      call kriging_error('kriging_st_capi', 'Handle is not a space-time kriging object')
    end select
  end subroutine get_obj

end module kriging_st_capi
