!==============================================================================
! kriging_st_capi.f90
!
! ISO C Binding wrapper for the t_kriging_st Fortran module.
! Mirrors the structure of kriging_capi.F90; all entry points are prefixed
! with krige_st_ to avoid name collisions with the base library.
!
! Shares the unified handle registry and utilities from kriging_capi_common.
!
! All arrays are passed as explicit-shape dummies with a preceding size
! parameter so ctypes can pass raw pointers without hidden descriptors.
!
! Entry points vs. the base spatial C API:
!   krige_st_set_obs          — coord(4,nobs): rows 1:3 spatial, row 4 time
!   krige_st_update_obs_value — replace values in-place for old-weight reuse
!   krige_st_set_grid         — C API accepts coord(3,n) + time(n)
!   krige_st_set_grid_block   — same 3D coord + time API for block centres
!   krige_st_set_grad         — coord1/coord2(4,ngrad), including time
!   krige_st_set_st_model        — sets global ST model parameters (strings for model/transform)
!   krige_st_set_vgm             — spatial nested structure (mirrors krige_set_vgm)
!   krige_st_set_vgm_temporal    — temporal nested structure: vtype + nugget/sill/at_k
!   krige_st_set_vgm_joint_sills — joint sills for sum-metric
!   krige_st_set_search          — sets search-specific spatial/time anisotropy
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
  ! krige_st_initialize
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
  !   model:     null-terminated "sum_metric" or "product_sum"
  !   transform: null-terminated vgmfunc type (sph, exp, gau, lin, …)
  !   at:        joint temporal scale
  !   time_nugget/time_sill: variogram-like distance scale for f_time
  !   k_ps:      product-sum k (model="product_sum")
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
  ! krige_st_set_obs
  !   coord   : [4, nobs]  rows 1:3 = spatial (x,y,z), row 4 = native time
  !   sk_mean : global mean for simple kriging (unbias=0)
  !=============================================================================
  integer(c_int) function krige_st_set_obs(handle, ivar, nobs, &
      coord, value, variance, nmax, maxdist, sk_mean) &
      bind(C, name='krige_st_set_obs') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ivar, nobs, nmax
    real(c_double),      intent(in)        :: coord(4, nobs)
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
  ! krige_st_update_obs_value
  !=============================================================================
  integer(c_int) function krige_st_update_obs_value(handle, ivar, nobs, value) &
      bind(C, name='krige_st_update_obs_value') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ivar, nobs
    real(c_double),      intent(in)        :: value(nobs)
    type(t_kriging_st), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    call obj%update_obs_value(int(ivar), real(value))
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_update_obs_value

  !=============================================================================
  ! krige_st_set_obs_drift
  !=============================================================================
  integer(c_int) function krige_st_set_obs_drift(handle, ivar, ndrift_c, nobs, drift) &
      bind(C, name='krige_st_set_obs_drift') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ivar, ndrift_c, nobs
    real(c_double),      intent(in)        :: drift(ndrift_c, nobs)
    type(t_kriging_st), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    call obj%set_obs_drift(int(ivar), real(drift))
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_set_obs_drift

  !=============================================================================
  ! krige_st_set_grad
  !
  ! Register ST gradient observation pairs.  coord1/coord2 are full ST endpoint
  ! coordinates: rows 1:3 = spatial, row 4 = native time.
  !=============================================================================
  integer(c_int) function krige_st_set_grad(handle, ivar, ngrad, coord1, coord2, &
      grad_val, variance, ndrift_c, drift_ext) &
      bind(C, name='krige_st_set_grad') result(ierr)

    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ivar, ngrad, ndrift_c
    real(c_double),      intent(in) :: coord1(4, max(ngrad,1))
    real(c_double),      intent(in) :: coord2(4, max(ngrad,1))
    real(c_double),      intent(in) :: grad_val(max(ngrad,1))
    real(c_double),      intent(in) :: variance(max(ngrad,1))
    real(c_double),      intent(in) :: drift_ext(max(ndrift_c,1), max(ngrad,1))

    type(t_kriging_st), pointer :: obj
    real :: c1(4, max(ngrad,1)), c2(4, max(ngrad,1))
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
  ! krige_st_reset_vgm — clear variogram structures for (ivar, jvar)
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

  !=============================================================================
  ! krige_st_set_vgm — add one spatial nested structure to vgm(ivar,jvar)%cs
  !   vtype   : null-terminated type: sph exp gau pow lin hol bsq cir nug
  !   nugget  : nugget contribution
  !   sill    : partial sill
  !   a_major : range along principal direction
  !   a_minor1, a_minor2 : ranges along minor directions
  !   azimuth, dip, plunge : rotation angles in degrees
  !=============================================================================
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

  !=============================================================================
  ! krige_st_set_vgm_temporal — add one temporal nested structure
  !   vtype  : null-terminated type: sph exp gau pow lin hol bsq cir nug
  !   nugget : nugget contribution
  !   sill   : partial sill
  !   at_k   : temporal practical range (same time units as observations)
  !=============================================================================
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

  !=============================================================================
  ! krige_st_set_vgm_joint_sills — joint sills for sum-metric model
  !   sills : [n]  one per spatial nested structure of cs
  !=============================================================================
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
  ! krige_st_set_grid — point estimation targets with times
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

  !=============================================================================
  ! krige_st_set_grid_block — block estimation targets with integration nodes
  !=============================================================================
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

  !=============================================================================
  ! krige_st_set_grid_cv — cross-validation mode
  !=============================================================================
  integer(c_int) function krige_st_set_grid_cv(handle) bind(C, name='krige_st_set_grid_cv') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    type(t_kriging_st), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    call obj%set_grid_cv()
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_set_grid_cv

  !=============================================================================
  ! krige_st_set_grid_drift — broadcasts drift to all target variables.
  !   (ST typically uses a single variable; use set_grid_drift_var for
  !   per-variable control when nvar > 1.)
  !=============================================================================
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
    ! ivar < 0 → broadcast to all variables (the common ST case)
    call obj%set_grid_drift(real(drift), ivar=-1)
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_set_grid_drift

  !=============================================================================
  ! krige_st_set_sim — SGSIM random path and samples
  !=============================================================================
  integer(c_int) function krige_st_set_sim(handle, nblocks, randpath, nsim_c, sample) &
      bind(C, name='krige_st_set_sim') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: nblocks, nsim_c
    integer(c_int),      intent(in)        :: randpath(nblocks)
    real(c_double),      intent(in)        :: sample(nsim_c, nblocks)
    type(t_kriging_st), pointer :: obj
    real, allocatable :: sample3(:,:,:)
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    allocate(sample3(nsim_c, obj%nvar, nblocks))
    sample3 = 0.0
    if (obj%nvar >= 1) sample3(:, 1, :) = real(sample)
    call obj%set_sim(randpath=int(randpath), sample=sample3)
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_set_sim

  !=============================================================================
  ! krige_st_set_search — build ST KD-tree for variable ivar
  !=============================================================================
  integer(c_int) function krige_st_set_search(handle, ivar, time_vtype, &
      time_nugget, time_sill, time_at, anis1, anis2, azimuth, dip, plunge) &
      bind(C, name='krige_st_set_search') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ivar
    character(kind=c_char), intent(in)     :: time_vtype(*)
    real(c_double),      intent(in), value :: time_nugget, time_sill, time_at
    real(c_double),      intent(in), value :: anis1, anis2, azimuth, dip, plunge
    type(t_kriging_st), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    call obj%set_search(int(ivar), &
      time_vtype=c2fstr(time_vtype), &
      time_nugget=real(time_nugget), time_sill=real(time_sill), &
      time_at=real(time_at), &
      anis1=real(anis1), anis2=real(anis2), &
      azimuth=real(azimuth), dip=real(dip), plunge=real(plunge))
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_set_search

  !=============================================================================
  ! krige_st_prepare — validate, build tables, pre-load weights
  !=============================================================================
  integer(c_int) function krige_st_prepare(handle) bind(C, name='krige_st_prepare') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    type(t_kriging_st), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    call obj%prepare()
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_prepare

  !=============================================================================
  ! krige_st_solve
  !   nthread: max OMP threads (0 = use OMP runtime default).
  !   ncache : hcache slots for this call (-1 = keep object default; 0 disables).
  !=============================================================================
  integer(c_int) function krige_st_solve(handle, nthread, ncache) bind(C, name='krige_st_solve') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: nthread
    integer(c_int),      intent(in), value :: ncache
    type(t_kriging_st), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    call obj%solve(nthread = int(nthread), ncache = int(ncache))
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_solve

  !=============================================================================
  ! Result getters
  !=============================================================================

  integer(c_int) function krige_st_get_nblocks(handle, n) bind(C, name='krige_st_get_nblocks') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(out)       :: n
    type(t_kriging_st), pointer :: obj
    call kriging_clear_error()
    n = 0_c_int
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    n = int(obj%block%n, c_int)
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_get_nblocks

  integer(c_int) function krige_st_get_nsim(handle, n) bind(C, name='krige_st_get_nsim') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(out)       :: n
    type(t_kriging_st), pointer :: obj
    call kriging_clear_error()
    n = 0_c_int
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    n = max(int(obj%nsim, c_int), 1_c_int)
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_get_nsim

  !-- Copy block%coord(1:nlag_c, 1:nblocks) into the caller-allocated array.
  !   For ST, nlag_c = ndim+1 = 4 (rows 1:3 = spatial, row 4 = time).
  integer(c_int) function krige_st_get_block_coord(handle, nlag_c, nblocks, out) &
      bind(C, name='krige_st_get_block_coord') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: nlag_c, nblocks
    real(c_double),      intent(out)       :: out(nlag_c, nblocks)
    type(t_kriging_st), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then; ierr = int(kriging_ierr(), c_int); return; end if
    out = real(obj%block%coord(1:nlag_c, 1:nblocks), c_double)
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_get_block_coord

  !-- Copy primary estimate: out(nsim_c, nblocks) for nvar=1.
  integer(c_int) function krige_st_get_estimate(handle, nsim_c, nblocks, out) &
      bind(C, name='krige_st_get_estimate') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: nsim_c, nblocks
    real(c_double),      intent(out)       :: out(nsim_c, nblocks)
    type(t_kriging_st), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    if (allocated(obj%block%value)) then
      out = real(obj%block%value(1:nsim_c, 1, 1:nblocks), c_double)
    else
      out = IEEE_VALUE(0.0_c_double, IEEE_QUIET_NAN)
    end if
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_get_estimate

  !-- Copy estimate into out(nblocks, nvar_c, nsim_c) — Python convention.
  integer(c_int) function krige_st_get_estimate_all(handle, nblocks, nvar_c, nsim_c, out) &
      bind(C, name='krige_st_get_estimate_all') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: nblocks, nvar_c, nsim_c
    real(c_double),      intent(out)       :: out(nblocks, nvar_c, nsim_c)
    type(t_kriging_st), pointer :: obj
    integer :: ib
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    if (.not. allocated(obj%block%value)) then
      out = IEEE_VALUE(0.0_c_double, IEEE_QUIET_NAN)
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    do ib = 1, nblocks
      out(ib, 1:nvar_c, 1:nsim_c) = &
        real(transpose(obj%block%value(1:nsim_c, 1:nvar_c, ib)), c_double)
    end do
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_get_estimate_all

  !-- Copy variance(1:nblocks) into the caller-allocated out array (nvar=1).
  integer(c_int) function krige_st_get_variance(handle, nblocks, out) &
      bind(C, name='krige_st_get_variance') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: nblocks
    real(c_double),      intent(out)       :: out(nblocks)
    type(t_kriging_st), pointer :: obj
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
  end function krige_st_get_variance

  !-- Copy block%variance into out(nblocks, nvar_c, nvar_c) — Python convention.
  integer(c_int) function krige_st_get_variance_all(handle, nblocks, nvar_c, out) &
      bind(C, name='krige_st_get_variance_all') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: nblocks, nvar_c
    real(c_double),      intent(out)       :: out(nblocks, nvar_c, nvar_c)
    type(t_kriging_st), pointer :: obj
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
  end function krige_st_get_variance_all

  !=============================================================================
  ! krige_st_to_str — build and return a C pointer to the null-terminated info string.
  !=============================================================================
  integer(c_intptr_t) function krige_st_to_str(handle) result(ptr) bind(C, name='krige_st_to_str')
    integer(c_intptr_t), intent(in), value :: handle
    type(t_kriging_st), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ptr = 0_c_intptr_t
      return
    end if
    call obj%update_info()
    ptr = transfer(c_loc(obj%krige_info(1)), ptr)
  end function krige_st_to_str

  !=============================================================================
  ! Persistent factorization cache API
  !=============================================================================

  !-- Query whether a valid persistent factor exists and return its dimensions.
  integer(c_int) function krige_st_get_factor_info(handle, npp_out, p_out, valid_out) &
      bind(C, name='krige_st_get_factor_info') result(ierr)
    integer(c_intptr_t), intent(in),  value :: handle
    integer(c_int),      intent(out)        :: npp_out, p_out, valid_out
    type(t_kriging_st), pointer :: obj
    integer  :: npp, p
    logical  :: valid
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then; ierr = int(kriging_ierr(), c_int); return; end if
    call obj%get_persistent_factor_info(npp, p, valid)
    npp_out   = int(npp,   c_int)
    p_out     = int(p,     c_int)
    valid_out = merge(1_c_int, 0_c_int, valid)
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_get_factor_info

  !-- Copy the three persistent factor matrices into caller-allocated arrays.
  integer(c_int) function krige_st_get_factor_matrices(handle, npp, p, &
                                                        L_out, kinv_out, schur_out) &
      bind(C, name='krige_st_get_factor_matrices') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: npp, p
    real(c_double),      intent(out)       :: L_out   (npp, npp)
    real(c_double),      intent(out)       :: kinv_out(npp, max(1, int(p)))
    real(c_double),      intent(out)       :: schur_out(max(1, int(p)), max(1, int(p)))
    type(t_kriging_st), pointer :: obj
    real, allocatable :: L_f(:,:), kinv_f(:,:), schur_f(:,:)
    integer :: pg
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then; ierr = int(kriging_ierr(), c_int); return; end if
    if (.not. obj%pf%valid) then
      ierr = int(kriging_ierr(), c_int); return
    end if
    pg = max(1, int(p))
    allocate(L_f(int(npp), int(npp)), kinv_f(int(npp), pg), schur_f(pg, pg))
    call obj%get_persistent_factor_matrices(int(npp), int(p), L_f, kinv_f, schur_f)
    L_out    = real(L_f,     c_double)
    kinv_out = real(kinv_f,  c_double)
    schur_out= real(schur_f, c_double)
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_get_factor_matrices

  !-- Copy the raw persistent linear system into caller-allocated arrays.
  integer(c_int) function krige_st_get_factor_system(handle, npp, p, nvar, &
                                                     matA_out, rhsB_out) &
      bind(C, name='krige_st_get_factor_system') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: npp, p, nvar
    real(c_double),      intent(out)       :: matA_out(npp + p, npp + p)
    real(c_double),      intent(out)       :: rhsB_out(nvar, npp + p)
    type(t_kriging_st), pointer :: obj
    real, allocatable :: matA_f(:,:), rhsB_f(:,:)
    integer :: matsize
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then; ierr = int(kriging_ierr(), c_int); return; end if
    if (.not. obj%pf%valid) then
      ierr = int(kriging_ierr(), c_int); return
    end if
    matsize = int(npp) + int(p)
    allocate(matA_f(matsize, matsize), rhsB_f(int(nvar), matsize))
    call obj%get_persistent_factor_system(int(npp), int(p), int(nvar), matA_f, rhsB_f)
    matA_out = real(matA_f, c_double)
    rhsB_out = real(rhsB_f, c_double)
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_get_factor_system

  !=============================================================================
  ! Weight-store API
  !=============================================================================

  !-- Free the in-memory weight store.
  integer(c_int) function krige_st_free_weight_store(handle) &
      bind(C, name='krige_st_free_weight_store') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    type(t_kriging_st), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then; ierr = int(kriging_ierr(), c_int); return; end if
    call obj%free_weight_store()
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_free_weight_store

  !-- Query weight-store dimensions: nmax, ngroups, nblock.
  integer(c_int) function krige_st_get_weight_dims(handle, nm_out, ng_out, nb_out) &
      bind(C, name='krige_st_get_weight_dims') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(out)       :: nm_out, ng_out, nb_out
    type(t_kriging_st), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then; ierr = int(kriging_ierr(), c_int); return; end if
    if (.not. allocated(obj%wstore)) then
      call kriging_error('krige_st_get_weight_dims', 'Weight store not allocated')
      ierr = int(kriging_ierr(), c_int); return
    end if
    nm_out = int(obj%wstore%nmax,    c_int)
    ng_out = int(obj%wstore%ngroups, c_int)
    nb_out = int(obj%wstore%nblock,  c_int)
    ierr   = int(kriging_ierr(), c_int)
  end function krige_st_get_weight_dims

  !-- Copy nnear[ngroups, nblock] into the caller-allocated integer buffer.
  integer(c_int) function krige_st_get_weight_nnear(handle, ngroups_c, nblock_c, out) &
      bind(C, name='krige_st_get_weight_nnear') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ngroups_c, nblock_c
    integer(c_int),      intent(out)       :: out(ngroups_c, nblock_c)
    type(t_kriging_st), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then; ierr = int(kriging_ierr(), c_int); return; end if
    if (.not. allocated(obj%wstore)) then
      call kriging_error('krige_st_get_weight_nnear', 'Weight store not allocated')
      ierr = int(kriging_ierr(), c_int); return
    end if
    out = int(obj%wstore%nnear(1:ngroups_c, 1:nblock_c), c_int)
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_get_weight_nnear

  !-- Copy inear[nmax, ngroups, nblock] into the caller-allocated integer buffer.
  integer(c_int) function krige_st_get_weight_inear(handle, nmax_c, ngroups_c, nblock_c, out) &
      bind(C, name='krige_st_get_weight_inear') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: nmax_c, ngroups_c, nblock_c
    integer(c_int),      intent(out)       :: out(nmax_c, ngroups_c, nblock_c)
    type(t_kriging_st), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then; ierr = int(kriging_ierr(), c_int); return; end if
    if (.not. allocated(obj%wstore)) then
      call kriging_error('krige_st_get_weight_inear', 'Weight store not allocated')
      ierr = int(kriging_ierr(), c_int); return
    end if
    out = int(obj%wstore%inear(1:nmax_c, 1:ngroups_c, 1:nblock_c), c_int)
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_get_weight_inear

  !-- Copy weight[nmax, ngroups, nvar, nblock] into the caller-allocated double buffer.
  integer(c_int) function krige_st_get_weight_data(handle, nmax_c, ngroups_c, nvar_c, nblock_c, out) &
      bind(C, name='krige_st_get_weight_data') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: nmax_c, ngroups_c, nvar_c, nblock_c
    real(c_double),      intent(out)       :: out(nmax_c, ngroups_c, nvar_c, nblock_c)
    type(t_kriging_st), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then; ierr = int(kriging_ierr(), c_int); return; end if
    if (.not. allocated(obj%wstore)) then
      call kriging_error('krige_st_get_weight_data', 'Weight store not allocated')
      ierr = int(kriging_ierr(), c_int); return
    end if
    out = real(obj%wstore%weight(1:nmax_c, 1:ngroups_c, 1:nvar_c, 1:nblock_c), c_double)
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_get_weight_data

  !-- Copy var[nvar, nvar, nblock] from wstore into the caller-allocated double buffer.
  integer(c_int) function krige_st_get_weight_var(handle, nvar_c, nblock_c, out) &
      bind(C, name='krige_st_get_weight_var') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: nvar_c, nblock_c
    real(c_double),      intent(out)       :: out(nvar_c, nvar_c, nblock_c)
    type(t_kriging_st), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then; ierr = int(kriging_ierr(), c_int); return; end if
    if (.not. allocated(obj%wstore)) then
      call kriging_error('krige_st_get_weight_var', 'Weight store not allocated')
      ierr = int(kriging_ierr(), c_int); return
    end if
    if (.not. allocated(obj%wstore%var)) then
      call kriging_error('krige_st_get_weight_var', 'Variance not stored')
      ierr = int(kriging_ierr(), c_int); return
    end if
    out = real(obj%wstore%var(1:nvar_c, 1:nvar_c, 1:nblock_c), c_double)
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_get_weight_var

  !-- Set nnear, inear, weight, and variance from caller-supplied arrays.
  integer(c_int) function krige_st_set_weights(handle, nmax_c, ngroups_c, nvar_c, nblock_c, &
                                                nnear_in, inear_in, weight_in, order_in, var_in) &
      bind(C, name='krige_st_set_weights') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: nmax_c, ngroups_c, nvar_c, nblock_c
    integer(c_int),      intent(in)        :: nnear_in (ngroups_c, nblock_c)
    integer(c_int),      intent(in)        :: inear_in (nmax_c, ngroups_c, nblock_c)
    real(c_double),      intent(in)        :: weight_in(nmax_c, ngroups_c, nvar_c, nblock_c)
    integer(c_int),      intent(in)        :: order_in (nblock_c)
    real(c_double),      intent(in)        :: var_in   (nvar_c, nvar_c, nblock_c)
    type(t_kriging_st), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then; ierr = int(kriging_ierr(), c_int); return; end if
    if (.not. allocated(obj%wstore)) call obj%alloc_weight_store()
    if (kriging_failed()) then; ierr = int(kriging_ierr(), c_int); return; end if
    call obj%set_weights(int(nnear_in), int(inear_in), real(weight_in), int(order_in), real(var_in))
    obj%use_old_weight = .true.
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_set_weights

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

  !-- Recover a typed t_kriging_st pointer from the unified registry.
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
