!==============================================================================
! kriging_capi_common.F90
!
! Shared C-binding infrastructure and shared API functions for both
! kriging_capi.F90 (spatial, t_kriging) and kriging_st_capi.f90 (ST,
! t_kriging_st).
!
! Design principle
! ----------------
! Functions here operate exclusively on t_kriging_base fields and methods:
!   - get_obj_base returns class(t_kriging_base) — no downcast needed.
!   - Every method called is either non_overridable or deferred on the base
!     type, so polymorphic dispatch works correctly for both concrete types.
!
! This mirrors the Fortran inheritance hierarchy:
!   kriging_capi_common  ↔  t_kriging_base
!   kriging_capi         ↔  t_kriging
!   kriging_st_capi      ↔  t_kriging_st
!
! Shared utilities
! ----------------
!   c2fstr, l()           — type-conversion helpers
!   krige_get_last_error  — single error-queue C symbol
!   krige_solver_stats    — SSYTRF/Cholesky counter query
!
! Shared API (base-only operations; C name has no krige_st_ prefix)
! ------------------------------------------------------------------
!   Obs      : krige_update_obs_value, krige_set_obs_drift
!   Grid     : krige_set_grid_cv
!   Lifecycle: krige_prepare, krige_solve
!   Results  : krige_get_nblocks, krige_get_nsim, krige_get_block_coord,
!              krige_get_estimate, krige_get_estimate_all,
!              krige_get_variance,  krige_get_variance_all,
!              krige_to_str
!   PF cache : krige_get_factor_info, krige_get_factor_matrices,
!              krige_get_factor_system
!   Weights  : krige_free_weight_store, krige_get_weight_dims,
!              krige_get_weight_nnear,  krige_get_weight_inear,
!              krige_get_weight_data,   krige_get_weight_var,
!              krige_set_weights
!
! Array convention (Fortran → Python)
! ------------------------------------
! All multi-dimensional result arrays are declared with block index FIRST
! so Python receives data in [block, ...] order.  When the internal Fortran
! storage has block last (e.g. block%value(nsim, nvar, nblock)), the slice
! is transposed before writing to the output buffer:
!   out(nblocks, nsim_c) = transpose( block%value(1:nsim_c, 1, 1:nblocks) )
! Python allocates _fempty((nb, ns)) and accesses estimate[:, 0] for sim 1.
!==============================================================================
module kriging_capi_common
  use, intrinsic :: ieee_arithmetic
  use iso_c_binding
  use kriging_base, only: t_kriging_base
  use kriging_err,  only: kriging_copy_error, kriging_ierr, kriging_error, &
                          kriging_clear_error, kriging_failed
  implicit none
  private

  !-- Polymorphic slot: stores either t_kriging or t_kriging_st pointer.
  type :: handle_slot
    class(t_kriging_base), pointer :: obj => null()
  end type handle_slot

  type(handle_slot), allocatable, save :: registry(:)

  public :: get_obj_base
  public :: store_obj_base
  public :: release_obj_base
  public :: c2fstr
  public :: l
  public :: krige_solver_stats

contains

  !=============================================================================
  ! krige_get_last_error — single C symbol for both kriging types.
  !=============================================================================
  integer(c_int) function krige_get_last_error(buffer, nbuf) &
      bind(C, name='krige_get_last_error') result(ierr)
    character(kind=c_char), intent(out) :: buffer(*)
    integer(c_int),         intent(in), value :: nbuf
    call kriging_copy_error(buffer, int(nbuf))
    ierr = int(kriging_ierr(), c_int)
  end function krige_get_last_error

  !=============================================================================
  ! krige_solver_stats — retrieve solver counters from last solve().
  !
  ! Works for any handle (spatial or space-time) because the counters live on
  ! t_kriging_base; no typed downcast is needed.
  !
  ! out(1) = n_fail         : blocks where both Cholesky and SSYTRF failed
  !                           (solution set to NaN; only possible when
  !                            neglect_error=.true.)
  ! out(2) = n_chol_fact    : fresh Cholesky factorizations (O(n³), cache miss)
  ! out(3) = n_chol_reuse   : blocks solved via cached Cholesky (O(n²))
  ! out(4) = n_ssytrf_fact  : SSYTRF (Bunch-Kaufman LDLᵀ) factorizations;
  !                           O(n³) each, once per unique neighbourhood
  ! out(5) = n_ssytrf_reuse : blocks solved by cached SSYTRF via SSYTRS (O(n²))
  !=============================================================================
  subroutine krige_solver_stats(handle, out) &
      bind(C, name='krige_solver_stats')
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(out)       :: out(5)
    class(t_kriging_base), pointer :: obj
    call get_obj_base(handle, obj)
    if (.not. associated(obj)) then
      out = 0_c_int
      return
    end if
    out(1) = int(obj%n_fail,         c_int)
    out(2) = int(obj%n_chol_fact,    c_int)
    out(3) = int(obj%n_chol_reuse,   c_int)
    out(4) = int(obj%n_ssytrf_fact,  c_int)
    out(5) = int(obj%n_ssytrf_reuse, c_int)
  end subroutine krige_solver_stats

  !=============================================================================
  ! Observation operations
  !=============================================================================

  !-- Replace observation values for variable ivar in-place (e.g. use_old_weight).
  integer(c_int) function krige_update_obs_value(handle, ivar, nobs, value) &
      bind(C, name='krige_update_obs_value') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ivar, nobs
    real(c_double),      intent(in)        :: value(nobs)
    class(t_kriging_base), pointer :: obj
    call kriging_clear_error()
    call get_obj_base(handle, obj)
    if (.not. associated(obj)) then; ierr = int(kriging_ierr(), c_int); return; end if
    call obj%update_obs_value(int(ivar), real(value))
    ierr = int(kriging_ierr(), c_int)
  end function krige_update_obs_value

  !-- Set external drift values at observation locations for variable ivar.
  integer(c_int) function krige_set_obs_drift(handle, ivar, ndrift_c, nobs, drift) &
      bind(C, name='krige_set_obs_drift') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ivar, ndrift_c, nobs
    real(c_double),      intent(in)        :: drift(ndrift_c, nobs)
    class(t_kriging_base), pointer :: obj
    call kriging_clear_error()
    call get_obj_base(handle, obj)
    if (.not. associated(obj)) then; ierr = int(kriging_ierr(), c_int); return; end if
    call obj%set_obs_drift(int(ivar), real(drift))
    ierr = int(kriging_ierr(), c_int)
  end function krige_set_obs_drift

  !=============================================================================
  ! Grid operations
  !=============================================================================

  !-- Cross-validation grid (no coord needed; derived from observations).
  integer(c_int) function krige_set_grid_cv(handle) &
      bind(C, name='krige_set_grid_cv') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    class(t_kriging_base), pointer :: obj
    call kriging_clear_error()
    call get_obj_base(handle, obj)
    if (.not. associated(obj)) then; ierr = int(kriging_ierr(), c_int); return; end if
    call obj%set_grid_cv()
    ierr = int(kriging_ierr(), c_int)
  end function krige_set_grid_cv

  !=============================================================================
  ! Solve lifecycle
  !=============================================================================

  !-- Validate, build tables, pre-load weights.
  integer(c_int) function krige_prepare(handle) &
      bind(C, name='krige_prepare') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    class(t_kriging_base), pointer :: obj
    call kriging_clear_error()
    call get_obj_base(handle, obj)
    if (.not. associated(obj)) then; ierr = int(kriging_ierr(), c_int); return; end if
    call obj%prepare()
    ierr = int(kriging_ierr(), c_int)
  end function krige_prepare

  !-- Run the kriging block loop.
  !   nthread : max OMP threads (0 = OMP runtime default).
  !   ncache  : hcache slots (-1 = object default; 0 = disabled).
  !
  !   Note: nthread is always passed as a concrete value so present(nthread)
  !   is .true. inside solve().  Intel ifx can mishandle absent optional
  !   arguments inside !$OMP PARALLEL regions; passing 0 preserves semantics.
  integer(c_int) function krige_solve(handle, nthread, ncache) &
      bind(C, name='krige_solve') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: nthread, ncache
    class(t_kriging_base), pointer :: obj
    call kriging_clear_error()
    call get_obj_base(handle, obj)
    if (.not. associated(obj)) then; ierr = int(kriging_ierr(), c_int); return; end if
    call obj%solve(nthread = int(nthread), ncache = int(ncache))
    ierr = int(kriging_ierr(), c_int)
  end function krige_solve

  !=============================================================================
  ! Result getters
  !=============================================================================

  !-- Number of blocks (= size of estimate / variance arrays).
  integer(c_int) function krige_get_nblocks(handle, n) &
      bind(C, name='krige_get_nblocks') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(out)       :: n
    class(t_kriging_base), pointer :: obj
    call kriging_clear_error()
    n = 0_c_int
    call get_obj_base(handle, obj)
    if (.not. associated(obj)) then; ierr = int(kriging_ierr(), c_int); return; end if
    n = int(obj%block%n, c_int)
    ierr = int(kriging_ierr(), c_int)
  end function krige_get_nblocks

  !-- Number of simulations (0 for plain kriging, >0 for SGSIM).
  integer(c_int) function krige_get_nsim(handle, n) &
      bind(C, name='krige_get_nsim') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(out)       :: n
    class(t_kriging_base), pointer :: obj
    call kriging_clear_error()
    n = 0_c_int
    call get_obj_base(handle, obj)
    if (.not. associated(obj)) then; ierr = int(kriging_ierr(), c_int); return; end if
    n = int(obj%nsim, c_int)
    ierr = int(kriging_ierr(), c_int)
  end function krige_get_nsim

  !-- Copy block%coord(1:ndim_c, 1:nblocks) into the caller-allocated array.
  !   out is filled as out(ndim_c, nblocks) in Fortran column-major order so
  !   that Python can read it as a (ndim_c, nblocks) Fortran-order array and
  !   transpose to (nblocks, ndim_c) for standard Python row-major convention.
  integer(c_int) function krige_get_block_coord(handle, ndim_c, nblocks, out) &
      bind(C, name='krige_get_block_coord') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ndim_c, nblocks
    real(c_double),      intent(out)       :: out(ndim_c, nblocks)
    class(t_kriging_base), pointer :: obj
    call kriging_clear_error()
    call get_obj_base(handle, obj)
    if (.not. associated(obj)) then; ierr = int(kriging_ierr(), c_int); return; end if
    out = obj%block%coord(1:ndim_c, 1:nblocks)
    ierr = int(kriging_ierr(), c_int)
  end function krige_get_block_coord

  !-- Primary estimate for variable 1.
  !
  !   Array convention (block index first):
  !     out(nblocks, nsim_c) — Python allocates _fempty((nb, ns))
  !     out[block, sim] via Python's F-order interpretation.
  !     For plain kriging (ns=1): Python does estimate[:, 0] to get shape (nb,).
  !
  !   Internal storage is block%value(nsim, 1, nblock).  transpose() converts
  !   the (nsim, nblock) slice to (nblock, nsim) before writing to out.
  integer(c_int) function krige_get_estimate(handle, nsim_c, nblocks, out) &
      bind(C, name='krige_get_estimate') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: nsim_c, nblocks
    real(c_double),      intent(out)       :: out(nblocks, nsim_c)
    class(t_kriging_base), pointer :: obj
    call kriging_clear_error()
    call get_obj_base(handle, obj)
    if (.not. associated(obj)) then; ierr = int(kriging_ierr(), c_int); return; end if
    if (allocated(obj%block%value)) then
      out = transpose(obj%block%value(1:nsim_c, 1, 1:nblocks))
    else
      out = IEEE_VALUE(0.0_c_double, IEEE_QUIET_NAN)
    end if
    ierr = int(kriging_ierr(), c_int)
  end function krige_get_estimate

  !-- All-variable estimate: out(nblocks, nvar_c, nsim_c).
  !
  !   Internal storage is block%value(nsim, nvar, nblock).  For each block,
  !   transpose() converts the (nsim, nvar) slice to (nvar, nsim) before
  !   writing into out(ib, 1:nvar_c, 1:nsim_c).
  integer(c_int) function krige_get_estimate_all(handle, nblocks, nvar_c, nsim_c, out) &
      bind(C, name='krige_get_estimate_all') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: nblocks, nvar_c, nsim_c
    real(c_double),      intent(out)       :: out(nblocks, nvar_c, nsim_c)
    class(t_kriging_base), pointer :: obj
    integer :: ib
    call kriging_clear_error()
    call get_obj_base(handle, obj)
    if (.not. associated(obj)) then; ierr = int(kriging_ierr(), c_int); return; end if
    if (.not. allocated(obj%block%value)) then
      out = IEEE_VALUE(0.0_c_double, IEEE_QUIET_NAN)
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    do ib = 1, nblocks
      out(ib, 1:nvar_c, 1:nsim_c) = transpose(obj%block%value(1:nsim_c, 1:nvar_c, ib))
    end do
    ierr = int(kriging_ierr(), c_int)
  end function krige_get_estimate_all

  !-- Kriging variance for variable 1 (shape: nblocks).
  integer(c_int) function krige_get_variance(handle, nblocks, out) &
      bind(C, name='krige_get_variance') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: nblocks
    real(c_double),      intent(out)       :: out(nblocks)
    class(t_kriging_base), pointer :: obj
    call kriging_clear_error()
    call get_obj_base(handle, obj)
    if (.not. associated(obj)) then; ierr = int(kriging_ierr(), c_int); return; end if
    if (allocated(obj%block%variance)) then
      out = obj%block%variance(1, 1, 1:nblocks)
    else
      out = IEEE_VALUE(0.0_c_double, IEEE_QUIET_NAN)
    end if
    ierr = int(kriging_ierr(), c_int)
  end function krige_get_variance

  !-- All-variable covariance: out(nblocks, nvar_c, nvar_c).
  !
  !   Internal storage is block%variance(nvar, nvar, nblock).  The loop
  !   extracts variance(:,:,ib) — a symmetric (nvar,nvar) slice — and places
  !   it in out(ib,:,:) so block is the first Python index.  Symmetry means
  !   the copy is equivalent to a per-block transpose.
  integer(c_int) function krige_get_variance_all(handle, nblocks, nvar_c, out) &
      bind(C, name='krige_get_variance_all') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: nblocks, nvar_c
    real(c_double),      intent(out)       :: out(nblocks, nvar_c, nvar_c)
    class(t_kriging_base), pointer :: obj
    integer :: ib
    call kriging_clear_error()
    call get_obj_base(handle, obj)
    if (.not. associated(obj)) then; ierr = int(kriging_ierr(), c_int); return; end if
    if (allocated(obj%block%variance)) then
      do ib = 1, nblocks
        out(ib, 1:nvar_c, 1:nvar_c) = obj%block%variance(1:nvar_c, 1:nvar_c, ib)
      end do
    else
      out = IEEE_VALUE(0.0_c_double, IEEE_QUIET_NAN)
    end if
    ierr = int(kriging_ierr(), c_int)
  end function krige_get_variance_all

  !-- Return a C pointer to the null-terminated info string.
  !   update_info (non_overridable) refreshes the string; krige_info is on base.
  integer(c_intptr_t) function krige_to_str(handle) result(ptr) &
      bind(C, name='krige_to_str')
    integer(c_intptr_t), intent(in), value :: handle
    class(t_kriging_base), pointer :: obj
    call kriging_clear_error()
    call get_obj_base(handle, obj)
    if (.not. associated(obj)) then
      ptr = 0_c_intptr_t
      return
    end if
    call obj%update_info()
    ptr = transfer(c_loc(obj%krige_info(1)), ptr)
  end function krige_to_str

  !=============================================================================
  ! Persistent factorization cache API
  !=============================================================================

  !-- Query whether a valid persistent factor exists and return its dimensions.
  !   valid_out: 1 = valid, 0 = not yet computed or invalidated.
  integer(c_int) function krige_get_factor_info(handle, npp_out, p_out, valid_out) &
      bind(C, name='krige_get_factor_info') result(ierr)
    integer(c_intptr_t), intent(in),  value :: handle
    integer(c_int),      intent(out)        :: npp_out, p_out, valid_out
    class(t_kriging_base), pointer :: obj
    integer  :: npp, p
    logical  :: valid
    call kriging_clear_error()
    call get_obj_base(handle, obj)
    if (.not. associated(obj)) then; ierr = int(kriging_ierr(), c_int); return; end if
    call obj%get_persistent_factor_info(npp, p, valid)
    npp_out   = int(npp,   c_int)
    p_out     = int(p,     c_int)
    valid_out = merge(1_c_int, 0_c_int, valid)
    ierr = int(kriging_ierr(), c_int)
  end function krige_get_factor_info

  !-- Copy the three persistent factor matrices into caller-allocated arrays.
  !   npp and p must match what krige_get_factor_info returned (and valid=1).
  integer(c_int) function krige_get_factor_matrices(handle, npp, p, &
                                                     L_out, kinv_out, schur_out) &
      bind(C, name='krige_get_factor_matrices') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: npp, p
    real(c_double),      intent(out)       :: L_out(npp, npp)
    real(c_double),      intent(out)       :: kinv_out(npp, max(1, int(p)))
    real(c_double),      intent(out)       :: schur_out(max(1, int(p)), max(1, int(p)))
    class(t_kriging_base), pointer :: obj
    real, allocatable :: L_f(:,:), kinv_f(:,:), schur_f(:,:)
    integer :: pg, npp_q, p_q
    logical :: valid
    call kriging_clear_error()
    call get_obj_base(handle, obj)
    if (.not. associated(obj)) then; ierr = int(kriging_ierr(), c_int); return; end if
    call obj%get_persistent_factor_info(npp_q, p_q, valid)
    if (.not. valid) then; ierr = int(kriging_ierr(), c_int); return; end if
    pg = max(1, int(p))
    allocate(L_f(int(npp), int(npp)), kinv_f(int(npp), pg), schur_f(pg, pg))
    call obj%get_persistent_factor_matrices(int(npp), int(p), L_f, kinv_f, schur_f)
    L_out     = real(L_f,     c_double)
    kinv_out  = real(kinv_f,  c_double)
    schur_out = real(schur_f, c_double)
    ierr = int(kriging_ierr(), c_int)
  end function krige_get_factor_matrices

  !-- Copy the raw persistent linear system into caller-allocated arrays.
  integer(c_int) function krige_get_factor_system(handle, npp, p, nvar, &
                                                   matA_out, rhsB_out) &
      bind(C, name='krige_get_factor_system') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: npp, p, nvar
    real(c_double),      intent(out)       :: matA_out(npp + p, npp + p)
    real(c_double),      intent(out)       :: rhsB_out(nvar, npp + p)
    class(t_kriging_base), pointer :: obj
    real, allocatable :: matA_f(:,:), rhsB_f(:,:)
    integer :: matsize, npp_q, p_q
    logical :: valid
    call kriging_clear_error()
    call get_obj_base(handle, obj)
    if (.not. associated(obj)) then; ierr = int(kriging_ierr(), c_int); return; end if
    call obj%get_persistent_factor_info(npp_q, p_q, valid)
    if (.not. valid) then; ierr = int(kriging_ierr(), c_int); return; end if
    matsize = int(npp) + int(p)
    allocate(matA_f(matsize, matsize), rhsB_f(int(nvar), matsize))
    call obj%get_persistent_factor_system(int(npp), int(p), int(nvar), matA_f, rhsB_f)
    matA_out = real(matA_f, c_double)
    rhsB_out = real(rhsB_f, c_double)
    ierr = int(kriging_ierr(), c_int)
  end function krige_get_factor_system

  !=============================================================================
  ! Weight-store API
  !=============================================================================

  !-- Free the in-memory weight store.
  integer(c_int) function krige_free_weight_store(handle) &
      bind(C, name='krige_free_weight_store') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    class(t_kriging_base), pointer :: obj
    call kriging_clear_error()
    call get_obj_base(handle, obj)
    if (.not. associated(obj)) then; ierr = int(kriging_ierr(), c_int); return; end if
    call obj%free_weight_store()
    ierr = int(kriging_ierr(), c_int)
  end function krige_free_weight_store

  !-- Query weight-store dimensions: nmax, ngroups, nblock.
  integer(c_int) function krige_get_weight_dims(handle, nm_out, ng_out, nb_out) &
      bind(C, name='krige_get_weight_dims') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(out)       :: nm_out, ng_out, nb_out
    class(t_kriging_base), pointer :: obj
    call kriging_clear_error()
    call get_obj_base(handle, obj)
    if (.not. associated(obj)) then; ierr = int(kriging_ierr(), c_int); return; end if
    if (.not. allocated(obj%wstore)) then
      call kriging_error('krige_get_weight_dims', 'Weight store not allocated')
      ierr = int(kriging_ierr(), c_int); return
    end if
    nm_out = int(obj%wstore%nmax,    c_int)
    ng_out = int(obj%wstore%ngroups, c_int)
    nb_out = int(obj%wstore%nblock,  c_int)
    ierr   = int(kriging_ierr(), c_int)
  end function krige_get_weight_dims

  !-- Copy nnear[ngroups, nblock] into the caller-allocated integer buffer.
  integer(c_int) function krige_get_weight_nnear(handle, ngroups_c, nblock_c, out) &
      bind(C, name='krige_get_weight_nnear') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ngroups_c, nblock_c
    integer(c_int),      intent(out)       :: out(ngroups_c, nblock_c)
    class(t_kriging_base), pointer :: obj
    call kriging_clear_error()
    call get_obj_base(handle, obj)
    if (.not. associated(obj)) then; ierr = int(kriging_ierr(), c_int); return; end if
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
    class(t_kriging_base), pointer :: obj
    call kriging_clear_error()
    call get_obj_base(handle, obj)
    if (.not. associated(obj)) then; ierr = int(kriging_ierr(), c_int); return; end if
    if (.not. allocated(obj%wstore)) then
      call kriging_error('krige_get_weight_inear', 'Weight store not allocated')
      ierr = int(kriging_ierr(), c_int); return
    end if
    out = int(obj%wstore%inear(1:nmax_c, 1:ngroups_c, 1:nblock_c), c_int)
    ierr = int(kriging_ierr(), c_int)
  end function krige_get_weight_inear

  !-- Copy weight[nmax, ngroups, nvar, nblock] into the caller-allocated double buffer.
  integer(c_int) function krige_get_weight_data(handle, nmax_c, ngroups_c, nvar_c, nblock_c, out) &
      bind(C, name='krige_get_weight_data') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: nmax_c, ngroups_c, nvar_c, nblock_c
    real(c_double),      intent(out)       :: out(nmax_c, ngroups_c, nvar_c, nblock_c)
    class(t_kriging_base), pointer :: obj
    call kriging_clear_error()
    call get_obj_base(handle, obj)
    if (.not. associated(obj)) then; ierr = int(kriging_ierr(), c_int); return; end if
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
    class(t_kriging_base), pointer :: obj
    call kriging_clear_error()
    call get_obj_base(handle, obj)
    if (.not. associated(obj)) then; ierr = int(kriging_ierr(), c_int); return; end if
    if (.not. allocated(obj%wstore)) then
      call kriging_error('krige_get_weight_var', 'Weight store not allocated')
      ierr = int(kriging_ierr(), c_int); return
    end if
    if (.not. allocated(obj%wstore%var)) then
      call kriging_error('krige_get_weight_var', &
        'Variance not stored (recompile with updated library)')
      ierr = int(kriging_ierr(), c_int); return
    end if
    out = real(obj%wstore%var(1:nvar_c, 1:nvar_c, 1:nblock_c), c_double)
    ierr = int(kriging_ierr(), c_int)
  end function krige_get_weight_var

  !-- Set nnear, inear, weight, and variance from caller-supplied arrays.
  !   Allocates (or re-allocates) wstore and sets use_old_weight=.true.
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
    class(t_kriging_base), pointer :: obj
    call kriging_clear_error()
    call get_obj_base(handle, obj)
    if (.not. associated(obj)) then; ierr = int(kriging_ierr(), c_int); return; end if
    if (.not. allocated(obj%wstore)) call obj%alloc_weight_store()
    if (kriging_failed()) then; ierr = int(kriging_ierr(), c_int); return; end if
    call obj%set_weights(int(nnear_in), int(inear_in), real(weight_in), int(order_in), real(var_in))
    obj%use_old_weight = .true.
    ierr = int(kriging_ierr(), c_int)
  end function krige_set_weights

  !=============================================================================
  ! Registry operations
  !=============================================================================

  !-- Resolve handle → class(t_kriging_base) pointer.  Bad/stale handles set a
  !   kriging error instead of dereferencing garbage.
  subroutine get_obj_base(handle, obj)
    integer(c_intptr_t),            intent(in),  value :: handle
    class(t_kriging_base), pointer, intent(out)        :: obj
    integer :: idx
    nullify(obj)
    if (handle == 0_c_intptr_t) then
      call kriging_error('kriging_capi', 'Null kriging object handle')
      return
    end if
    idx = int(handle)
    if (.not. allocated(registry) .or. idx < 1 .or. idx > size(registry)) then
      call kriging_error('kriging_capi', 'Invalid kriging object handle')
      return
    end if
    if (associated(registry(idx)%obj)) obj => registry(idx)%obj
    if (.not. associated(obj)) &
      call kriging_error('kriging_capi', 'Invalid kriging object handle')
  end subroutine get_obj_base


  !-- Store a newly allocated concrete object in the first free registry slot.
  !   Growing keeps old slot numbers stable so existing Python handles remain valid.
  subroutine store_obj_base(obj, handle)
    class(t_kriging_base), pointer, intent(in)  :: obj
    integer(c_intptr_t),            intent(out) :: handle
    integer :: i
    if (.not. allocated(registry)) allocate(registry(16))
    do i = 1, size(registry)
      if (.not. associated(registry(i)%obj)) then
        registry(i)%obj => obj
        handle = int(i, c_intptr_t)
        return
      end if
    end do
    call grow_registry_()
    do i = 1, size(registry)
      if (.not. associated(registry(i)%obj)) then
        registry(i)%obj => obj
        handle = int(i, c_intptr_t)
        return
      end if
    end do
    handle = 0_c_intptr_t
    call kriging_error('krige_create', 'Failed to allocate a kriging handle slot')
  end subroutine store_obj_base


  !-- Nullify the slot so it can be reused.  Other slots are unchanged.
  subroutine release_obj_base(handle)
    integer(c_intptr_t), intent(in), value :: handle
    integer :: idx
    idx = int(handle)
    if (allocated(registry) .and. idx >= 1 .and. idx <= size(registry)) &
      nullify(registry(idx)%obj)
  end subroutine release_obj_base


  !-- Double the registry, preserving all pointer associations.
  subroutine grow_registry_()
    type(handle_slot), allocatable :: tmp(:)
    integer :: i, old_n, new_n
    old_n = size(registry)
    new_n = max(1, old_n * 2)
    allocate(tmp(new_n))
    do i = 1, old_n
      if (associated(registry(i)%obj)) tmp(i)%obj => registry(i)%obj
    end do
    call move_alloc(tmp, registry)
  end subroutine grow_registry_

  !=============================================================================
  ! Shared utilities
  !=============================================================================

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
  elemental function l(v) result(r)
    integer(c_int), intent(in), value :: v
    logical :: r
    r = (v == 1_c_int)
  end function l

end module kriging_capi_common
