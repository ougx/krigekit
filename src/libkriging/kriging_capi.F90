!==============================================================================
! kriging_capi.F90
!
! ISO C Binding wrapper for the t_kriging Fortran module (spatial kriging).
! Exposes type-specific entry points as C-callable functions taking an opaque
! integer(c_intptr_t) handle.
!
! Shared base operations (prepare, solve, get_estimate, weight store, etc.)
! live in kriging_capi_common.F90 and use the krige_* C-name prefix directly.
! Only functions that require a typed t_kriging pointer (because they access
! t_kriging-specific fields or call methods not on t_kriging_base) remain here.
!
! Type-specific entry points
! --------------------------
!   Lifecycle  : krige_create, krige_destroy
!   Init       : krige_initialize  (has ndim, varying_vgm, std_ck, pf_cache)
!   Obs        : krige_set_obs     (spatial coord only)
!   Variogram  : krige_reset_vgm, krige_set_vgm, krige_set_vgm_block
!   Grid       : krige_set_grid, krige_set_grid_block, krige_set_grid_drift
!   Sim        : krige_set_sim     (3-D sample array)
!   Search     : krige_set_search  (no time_at)
!   Gradient   : krige_set_grad
!   Threads    : krige_get_max_threads, krige_get_num_threads
!==============================================================================
module kriging_capi
  use, intrinsic :: ieee_arithmetic
  use iso_c_binding
  use kriging,              only: t_kriging
  use kriging_indicator,    only: t_kriging_indicator
  use kriging_base,         only: t_kriging_base
  use kriging_capi_common,  only: get_obj_base, store_obj_base, release_obj_base, c2fstr, l
  use kriging_err,          only: kriging_clear_error, kriging_ierr, &
                                  kriging_error, kriging_failed
  implicit none
  private

contains

  !=============================================================================
  ! Lifecycle: create / destroy
  !=============================================================================

  integer(c_int) function krige_create(handle) bind(C, name='krige_create') result(ierr)
    integer(c_intptr_t), intent(out) :: handle
    class(t_kriging), pointer :: obj
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

  integer(c_int) function krige_destroy(handle) bind(C, name='krige_destroy') result(ierr)
    integer(c_intptr_t), intent(inout) :: handle
    class(t_kriging), pointer :: obj
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
  ! krige_ind_create — allocate a t_kriging_indicator object and return handle.
  !
  ! All subsequent setup calls (krige_initialize, krige_set_obs, krige_set_vgm,
  ! krige_set_grid, krige_set_sim, krige_set_search, krige_prepare, krige_solve,
  ! krige_get_estimate, krige_get_variance, krige_destroy) work transparently on
  ! this handle via polymorphic dispatch through class(t_kriging).
  !=============================================================================
  integer(c_int) function krige_ind_create(handle) &
      bind(C, name='krige_ind_create') result(ierr)
    integer(c_intptr_t), intent(out) :: handle
    type(t_kriging_indicator), pointer :: ind
    class(t_kriging), pointer :: obj
    integer :: stat
    call kriging_clear_error()
    handle = 0_c_intptr_t
    allocate(ind, stat=stat)
    if (stat /= 0) then
      call kriging_error('krige_ind_create', 'Failed to allocate t_kriging_indicator object')
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    obj => ind
    call store_obj(obj, handle)
    ierr = int(kriging_ierr(), c_int)
  end function krige_ind_create

  !=============================================================================
  ! krige_ind_set_ncat — store the number of indicator categories in a
  ! t_kriging_indicator object.  Must be called after krige_ind_create and
  ! krige_initialize, and before krige_set_obs / krige_solve.
  !
  ! ncat must satisfy 1 <= ncat <= nvar.  When ncat < nvar, the last
  ! nvar-ncat variables are treated as secondary continuous co-variates:
  ! they contribute to the kriging weights but are excluded from the
  ! CDF draw (sim_draw_indicator) and probability normalisation
  ! (post_solve_indicator).
  !=============================================================================
  integer(c_int) function krige_ind_set_ncat(handle, ncat) &
      bind(C, name='krige_ind_set_ncat') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ncat
    class(t_kriging), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then; ierr = int(kriging_ierr(), c_int); return; end if
    select type(obj)
    type is (t_kriging_indicator)
      if (ncat < 1 .or. ncat > obj%nvar) then
        call kriging_error('krige_ind_set_ncat', &
          'ncat must satisfy 1 <= ncat <= nvar')
      else
        obj%ncat = ncat
      end if
    class default
      call kriging_error('krige_ind_set_ncat', &
        'Handle does not point to a t_kriging_indicator object')
    end select
    ierr = int(kriging_ierr(), c_int)
  end function krige_ind_set_ncat

  !=============================================================================
  ! krige_initialize
  !
  ! Spatial-specific: has ndim, varying_vgm, std_ck, pf_cache parameters that
  ! the ST initializer does not expose (ST always uses ndim=3 and separate
  ! defaults for those flags).
  !=============================================================================
  integer(c_int) function krige_initialize(handle, &
      ndim, nvar, ndrift, unbias, nsim, &
      anisotropic_search, weight_correction, use_old_weight, &
      store_weight, cross_validation, write_mat, neglect_error, varying_vgm, std_ck, verbose, &
      pf_cache, &
      weight_file, bounds, seed) &
      bind(C, name='krige_initialize') result(ierr)

    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ndim, nvar, ndrift, unbias, nsim, seed
    integer(c_int),      intent(in), value :: anisotropic_search, weight_correction
    integer(c_int),      intent(in), value :: use_old_weight, store_weight
    integer(c_int),      intent(in), value :: cross_validation, write_mat, neglect_error
    integer(c_int),      intent(in), value :: varying_vgm, std_ck, verbose
    integer(c_int),      intent(in), value :: pf_cache
    character(kind=c_char), intent(in) :: weight_file(*)
    real(c_double),      intent(in) :: bounds(2)

    class(t_kriging), pointer :: obj
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
      std_ck             = l(std_ck), &
      verbose            = l(verbose), &
      pf_cache           = l(pf_cache), &
      weight_file        = c2fstr(weight_file), &
      bounds             = fbounds, &
      seed               = int(seed))
    ierr = int(kriging_ierr(), c_int)
  end function krige_initialize

  !=============================================================================
  ! krige_set_obs — spatial coord only (ndim_c columns, no time column)
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

    class(t_kriging), pointer :: obj
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
  ! Variogram (spatial — set_vgm / reset_vgm are not on t_kriging_base)
  !=============================================================================

  integer(c_int) function krige_reset_vgm(handle, ivar, jvar) &
      bind(C, name='krige_reset_vgm') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ivar, jvar
    class(t_kriging), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    call obj%reset_vgm(int(ivar), int(jvar))
    ierr = int(kriging_ierr(), c_int)
  end function krige_reset_vgm

  integer(c_int) function krige_set_vgm(handle, ivar, jvar, vtype, &
                            nugget, sill, a_major, a_minor1, a_minor2, &
                            azimuth, dip, plunge) &
      bind(C, name='krige_set_vgm') result(ierr)

    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ivar, jvar
    character(kind=c_char), intent(in) :: vtype(*)
    real(c_double), intent(in), value :: nugget, sill, a_major, a_minor1, a_minor2
    real(c_double), intent(in), value :: azimuth, dip, plunge

    class(t_kriging), pointer :: obj
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

  !-- Per-block variogram (requires varying_vgm=1 and set_grid called first).
  integer(c_int) function krige_set_vgm_block(handle, ivar, jvar, ib, vtype, &
                                  nugget, sill, a_major, a_minor1, a_minor2, &
                                  azimuth, dip, plunge) &
      bind(C, name='krige_set_vgm_block') result(ierr)

    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ivar, jvar, ib
    character(kind=c_char), intent(in) :: vtype(*)
    real(c_double), intent(in), value :: nugget, sill, a_major, a_minor1, a_minor2
    real(c_double), intent(in), value :: azimuth, dip, plunge

    class(t_kriging), pointer :: obj
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
                     plunge   = real(plunge), &
                     ib       = int(ib))
    ierr = int(kriging_ierr(), c_int)
  end function krige_set_vgm_block

  !=============================================================================
  ! Grid setup (spatial-specific coordinate formats)
  !=============================================================================

  integer(c_int) function krige_set_grid(handle, ngrid, ndim_c, coord, &
      rangescale, localnugget) &
      bind(C, name='krige_set_grid') result(ierr)

    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ngrid, ndim_c
    real(c_double),      intent(in) :: coord(ndim_c, ngrid)
    real(c_double),      intent(in) :: rangescale(ngrid)
    real(c_double),      intent(in) :: localnugget(ngrid)

    class(t_kriging), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    call obj%set_grid_point(coord       = real(coord), &
                            rangescale  = real(rangescale), &
                            localnugget = real(localnugget))
    ierr = int(kriging_ierr(), c_int)
  end function krige_set_grid

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
    real(c_double),      intent(in) :: blocksize(ndim_c, nblock)
    real(c_double),      intent(in) :: rangescale(nblock)
    real(c_double),      intent(in) :: localnugget(nblock)

    integer :: npw
    class(t_kriging), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    if (block_type == -4_c_int) then
      call obj%set_grid_gq(coord       = real(coord), &
                           blocksize   = real(blocksize), &
                           rangescale  = real(rangescale), &
                           localnugget = real(localnugget))
    else
      npw = sum(nblockpnt)
      call obj%set_grid_user_block(coord       = real(coord), &
                                   nblockpnt   = int(nblockpnt), &
                                   pointweight = real(pointweight(1:npw)), &
                                   rangescale  = real(rangescale), &
                                   localnugget = real(localnugget), &
                                   block_type  = int(block_type))
    end if
    ierr = int(kriging_ierr(), c_int)
  end function krige_set_grid_block

  !-- Grid drift: ivar selects the target variable; ivar<0 broadcasts to all.
  !   (ST version krige_st_set_grid_drift always broadcasts with ivar=-1.)
  integer(c_int) function krige_set_grid_drift(handle, ivar, ndrift_c, nblocks, drift) &
      bind(C, name='krige_set_grid_drift') result(ierr)

    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ivar, ndrift_c, nblocks
    real(c_double),      intent(in) :: drift(ndrift_c, nblocks)

    class(t_kriging), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    call obj%set_grid_drift(real(drift), int(ivar))
    ierr = int(kriging_ierr(), c_int)
  end function krige_set_grid_drift

  !=============================================================================
  ! SGSIM — spatial sample array is 3-D (nsim, nvar, nblocks).
  !
  ! randpath_ptr  NULL  → Fortran generates a random permutation.
  !               non-NULL → caller-supplied 1-based permutation of 1..nblocks.
  !
  ! sample_ptr    NULL  → Fortran generates samples (distribution determined by
  !                       the dynamic type: N(0,1) for t_kriging, U(0,1) for
  !                       t_kriging_indicator via its set_sim override).
  !               non-NULL → caller-supplied (nsim, nvar, nblocks) array.
  !
  ! nblocks is only used to shape the non-NULL pointer(s); pass 0 when both
  ! are NULL.
  !=============================================================================
  integer(c_int) function krige_set_sim(handle, nblocks, randpath_ptr, nsim_c, nvar_c, sample_ptr) &
      bind(C, name='krige_set_sim') result(ierr)

    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: nblocks, nsim_c, nvar_c
    type(c_ptr),         intent(in), value :: randpath_ptr
    type(c_ptr),         intent(in), value :: sample_ptr

    class(t_kriging), pointer :: obj
    integer(c_int),   pointer :: rp_f(:)
    real(c_double),   pointer :: sample_f(:,:,:)

    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if

    if (c_associated(randpath_ptr) .and. c_associated(sample_ptr)) then
      call c_f_pointer(randpath_ptr, rp_f,     [nblocks])
      call c_f_pointer(sample_ptr,   sample_f, [nsim_c, nvar_c, nblocks])
      call obj%set_sim(randpath = int(rp_f), sample = real(sample_f))
    else if (c_associated(randpath_ptr)) then
      call c_f_pointer(randpath_ptr, rp_f, [nblocks])
      call obj%set_sim(randpath = int(rp_f))
    else if (c_associated(sample_ptr)) then
      call c_f_pointer(sample_ptr, sample_f, [nsim_c, nvar_c, nblocks])
      call obj%set_sim(sample = real(sample_f))
    else
      call obj%set_sim()
    end if
    ierr = int(kriging_ierr(), c_int)
  end function krige_set_sim

  !=============================================================================
  ! Search — spatial (no time_at parameter)
  !=============================================================================
  integer(c_int) function krige_set_search(handle, ivar, anis1, anis2, &
      azimuth, dip, plunge, sector_search) &
      bind(C, name='krige_set_search') result(ierr)

    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ivar
    real(c_double),      intent(in), value :: anis1, anis2, azimuth, dip, plunge
    integer(c_int),      intent(in), value :: sector_search

    class(t_kriging), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    call obj%set_search(int(ivar), real(anis1), real(anis2), &
      real(azimuth), real(dip), real(plunge), sector_search = l(sector_search))
    ierr = int(kriging_ierr(), c_int)
  end function krige_set_search

  !=============================================================================
  ! Gradient constraints (spatial — coord dim is ndim_c, not ndim+1)
  !=============================================================================
  integer(c_int) function krige_set_grad(handle, ivar, ngrad, ndim_c, coord1, coord2, &
                                          grad_val, variance, ndrift_c, drift_ext) &
      bind(C, name='krige_set_grad') result(ierr)

    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ivar, ngrad, ndim_c, ndrift_c
    real(c_double),      intent(in) :: coord1(ndim_c, max(ngrad,1))
    real(c_double),      intent(in) :: coord2(ndim_c, max(ngrad,1))
    real(c_double),      intent(in) :: grad_val(max(ngrad,1))
    real(c_double),      intent(in) :: variance(max(ngrad,1))
    real(c_double),      intent(in) :: drift_ext(max(ndrift_c,1), max(ngrad,1))

    class(t_kriging), pointer :: obj
    real :: c1(ndim_c, max(ngrad,1)), c2(ndim_c, max(ngrad,1))
    real :: gv(max(ngrad,1)), gvar(max(ngrad,1))
    c1   = real(coord1);   c2   = real(coord2)
    gv   = real(grad_val); gvar = real(variance)

    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if

    if (ngrad == 0) then
      call obj%reset_grad(int(ivar))
    else if (ndrift_c > 0) then
      call obj%set_grad(int(ivar), c1, c2, gv, variance=gvar, &
                        drift_ext = real(drift_ext))
    else
      call obj%set_grad(int(ivar), c1, c2, gv, variance=gvar)
    end if
    ierr = int(kriging_ierr(), c_int)
  end function krige_set_grad

  !=============================================================================
  ! krige_get_max_threads / krige_get_num_threads
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
    n = 1_c_int
  end subroutine krige_get_max_threads

  subroutine krige_get_num_threads(n) bind(C, name='krige_get_num_threads')
    integer(c_int), intent(out) :: n
    n = 1_c_int
  end subroutine krige_get_num_threads
#endif

  !=============================================================================
  ! Internal helpers (private to this module)
  !=============================================================================

  subroutine get_obj(handle, obj)
    integer(c_intptr_t), intent(in), value :: handle
    class(t_kriging),    pointer           :: obj
    class(t_kriging_base), pointer :: base
    nullify(obj)
    call get_obj_base(handle, base)
    if (kriging_failed()) return
    select type(base)
    class is (t_kriging)
      obj => base
    class default
      call kriging_error('kriging_capi', 'Handle is not a spatial kriging object')
    end select
  end subroutine get_obj

  subroutine store_obj(obj, handle)
    class(t_kriging), pointer, intent(in) :: obj
    integer(c_intptr_t),       intent(out) :: handle
    class(t_kriging_base), pointer :: base
    base => obj
    call store_obj_base(base, handle)
  end subroutine store_obj

  subroutine release_obj(handle)
    integer(c_intptr_t), intent(in), value :: handle
    call release_obj_base(handle)
  end subroutine release_obj

end module kriging_capi
