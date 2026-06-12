!==============================================================================
! Module: kriging_base
!
! Shared infrastructure and abstract base type for spatial and space-time kriging.
! (Formerly split across kriging_shared.F90 and kriging_base.F90.)
!
! Defines:
!   t_factor_cache    — Cholesky factorization cache (intra- and inter-solve)
!   t_kriging_ctx     — unified per-thread working context
!   t_kriging_base    — abstract base type for t_kriging and t_kriging_st
!   utility subroutines
!   solve_base        — non-overridable template method
!   initialize_ctx, assign_weight_ctx, fcache_matches, fcache_save_key
!==============================================================================

!============================================================================
! t_data — base type for any spatially located dataset
!
! All three spatial classes (t_obsgrid, t_grid, t_blockgrid) extend t_data.
! Using a common base keeps covariance assembly generic: assemble_rhs and
! assemble_lhs accept class(t_data) pointers and handle obs, grid or block
! nodes without branching on the concrete type.
!============================================================================

!============================================================================
! t_obsgrid — observation dataset with search infrastructure
!
! Extends t_data with the k-d tree (tree) used for nearest-neighbour search.
! rotmat holds the 3×3 anisotropy rotation+scale matrix used to project
! coordinates into the anisotropic distance metric before tree queries.
! maxdist is stored as the SQUARED distance to allow direct comparison with
! the squared distances returned by KDTREE2 without a sqrt per neighbour.
!============================================================================


!============================================================================
! Neighbour-group layout convention
!
! Group arrays (nnear, inear, weight) use a sequential layout:
!   Groups 1:nvar               = real observations, variable ig     (always present)
!   Groups nvar+1:ngroups_base  = previously simulated blocks, ig-nvar (SGSIM only)
!   Groups ngroups_base+1:ngroups = gradient pair groups, variable ig-ngroups_base
!
! Matrix layout (ctx%istart)
! --------------------------
! Although the group arrays are sequential, the kriging matrix rows/cols are
! arranged so that each variable's obs and sim groups are adjacent:
!   obs_1, sim_1, obs_2, sim_2, ...
!
! ctx%istart(ig) holds the 1-based starting row/col in the matrix for group ig.
! It is computed in assemble_linear_system and used by assemble_lhs / assemble_rhs
! instead of a running accumulation, so both routines loop over groups in any
! order and index the matrix directly.
!
! Group type checks:
!   kvar > nvar           — simulated-block group (ivar = kvar - nvar)
!   kvar > ngroups_base   — gradient pair group   (givar = kvar - ngroups_base)
! Variable index:     obs kvar → ivar = kvar;  sim kvar → ivar = kvar - nvar
!                     grad kvar → givar = kvar - ngroups_base
!============================================================================



module kriging_base
  use, intrinsic   :: ieee_arithmetic
  use, intrinsic   :: iso_fortran_env, only: int64
  use iso_c_binding, only: c_char, c_null_char
  use common,        only: version
  use kriging_err,   only: kriging_error, kriging_failed
  use progress_bar,  only: progress
  use utils,         only: set_seq, random_seed_initialize, r8vec_normal_01, r8vec_uniform_01, yesno
  use gaussian_quadrature
  use kdtree2_module
  implicit none
  private

  ! --- shared infrastructure ---
  public :: t_factor_cache
  public :: t_factor_hash_cache
  public :: t_kriging_ctx
  public :: t_data
  public :: t_obsgrid
  public :: t_grid
  public :: t_blockgrid
  public :: t_weight_store
  public :: t_grad
  public :: kriging_check_index
  public :: kriging_check_pair_index
  public :: isort
  public :: kriging_close_unit
  public :: kriging_clip_positive_normalize
  public :: kriging_conditional_variance
  public :: kriging_identity_rotmat3
  public :: kriging_mirror_lower_to_upper
  public :: kriging_normalize_nmax
  ! --- base type and helpers ---
  public :: t_kriging_base
  public :: initialize_ctx
  public :: assign_weight_ctx
  public :: fcache_matches
  public :: fcache_save_key
  public :: fhcache_hash_key
  public :: group_ivar
  public :: filter_by_maxlag

  !============================================================================
  ! t_factor_cache — Cholesky factorization cache
  !
  ! Used in two roles:
  !   ctx%cache  (t_kriging_ctx): intra-solve, one slot per thread.
  !   self%pf    (t_kriging_base): inter-solve, shared across threads (pf_cache).
  !============================================================================
  type :: t_factor_cache
    logical  :: valid        = .false.
    logical  :: hit          = .false.
    logical  :: system_valid = .false.
    !-- When Cholesky fails, the solver falls back to SSYTRF (Bunch-Kaufman LDLᵀ).
    !   used_ssysv flags which factorization is live in this slot.
    !   Afac / ipiv are lazily allocated only when SSYTRF is actually needed,
    !   so the common Cholesky path carries zero extra memory overhead.
    logical  :: used_ssysv   = .false.
    real,    allocatable :: Afac(:,:)   ! LDLᵀ of full augmented [K F;Fᵀ 0] — (m,m) lazy
    integer, allocatable :: ipiv(:)     ! Bunch-Kaufman pivot array            — (m)   lazy
    integer  :: npp         = 0
    integer  :: p           = 0
    real     :: rangescale  = 1.0
    real     :: localnugget = 0.0
    integer, allocatable :: nnear(:)        ! [ngroups_base]
    integer, allocatable :: inear(:,:)      ! [mmax, ngroups_base]
    real,    allocatable :: matA(:,:)       ! Assembled LHS system        [nppmax+p, nppmax+p]
    real,    allocatable :: rhsB(:,:)       ! Assembled RHS system        [nvar, nppmax+p]
    real,    allocatable :: L(:,:)          ! Cholesky factor             [nppmax, nppmax]
    real,    allocatable :: kinv_drift(:,:) ! K^{-1} F                   [nppmax, max(1,p)]
    real,    allocatable :: schur(:,:)      ! Schur complement F'K^{-1}F [max(1,p), max(1,p)]
  contains
    procedure :: alloc    => fcache_alloc
    procedure :: copy_to  => fcache_copy_to
    procedure :: copy_all => fcache_copy_all
  end type t_factor_cache

  !============================================================================
  ! t_factor_hash_cache — fixed-size per-thread multi-entry factor cache
  !
  ! This is intentionally bounded: memory scales with factor_cache_size and
  ! OpenMP thread count, not with the number of grid blocks.  Each entry stores
  ! one prepared factorization keyed by the exact neighbour indices and local
  ! block modifiers.  Hash matches are always verified with fcache_matches(), so
  ! collisions cannot produce an incorrect factorization reuse.
  !============================================================================
  type :: t_factor_entry
    logical :: valid = .false.
    integer :: hash = 0
    integer :: last_used = 0
    type(t_factor_cache) :: fac
  end type t_factor_entry

  type :: t_factor_hash_cache
    integer :: nslot = 0
    integer :: nbucket = 0
    integer :: clock = 0
    type(t_factor_entry), allocatable :: slot(:)
    integer, allocatable :: bucket(:)
    integer, allocatable :: next(:)
  contains
    procedure :: alloc  => fhcache_alloc
    procedure :: free   => fhcache_free
  end type t_factor_hash_cache

  !============================================================================
  ! t_kriging_ctx — unified per-thread working context
  !
  ! Single concrete type for both t_kriging (spatial) and t_kriging_st (ST).
  !============================================================================
  type :: t_kriging_ctx
    ! --- per-block state ---
    integer  :: iblock      = 0
    integer  :: npp         = 0
    integer  :: matsize     = 0
    real     :: rangescale  = 1.0
    real     :: localnugget = 0.0
    ! --- constant per solve ---
    integer  :: ngroups_base = 0
    integer  :: p            = 0    ! ndrift + naug (augmented rows)
    ! --- arrays ---
    integer, allocatable :: nnear (:)      ! [ngroups]
    integer, allocatable :: inear (:,:)    ! [mmax, ngroups]
    integer, allocatable :: istart(:)      ! [ngroups]
    real,    allocatable :: sqdist(:,:)    ! [mmax, ngroups]
    real,    allocatable :: weight(:,:,:)  ! [mmax, ngroups, nvar]
    real,    allocatable :: x     (:,:)    ! [nvar, matsize]
    real,    allocatable :: matA  (:,:)    ! [matsize, matsize]
    real,    allocatable :: rhsB  (:,:)    ! [nvar, matsize]
    type(kdtree2_result), allocatable :: results(:) ! [ncand_limit] candidate query buffer
    type(t_factor_cache) :: cache
    type(t_factor_hash_cache) :: hcache
  contains
    procedure :: free_core => kriging_ctx_free_core
  end type t_kriging_ctx

  !============================================================================
  ! t_data — base type for any coordinate-indexed dataset.
  ! coord has nlag rows.  For spatial kriging nlag == ndim (1..3).
  ! For ST kriging nlag == ndim+1, with native time in coord(nlag,:).
  !============================================================================
  type :: t_data
    integer              :: n = 0
    real, allocatable    :: coord(:,:)
    real, allocatable    :: drift(:,:,:)
    real, allocatable    :: value(:,:,:)
    real, allocatable    :: variance(:,:,:)
  end type t_data

  type, extends(t_data) :: t_obsgrid
    integer              :: nmax = 0
    real                 :: maxdist = huge(0.0)
    real                 :: sk_mean = 0.0
    integer              :: time_vtype_id = 0
    real                 :: time_nugget = 0.0
    real                 :: time_sill = 1.0
    real                 :: time_at = 1.0
    real                 :: rotmat(3,3)
    logical              :: need_search = .false.
    logical              :: anisotropic_search = .false.
    logical              :: sector_search = .false.
    logical              :: set_search = .false.
    type(kdtree2), pointer :: tree => null()
  end type t_obsgrid

  !============================================================================
  ! t_grid — integration / sub-block nodes.
  !============================================================================
  type, extends(t_data) :: t_grid
    real, allocatable    :: weight(:)
  end type t_grid

  !============================================================================
  ! t_blockgrid — estimation targets and per-block output storage.
  ! value    is [max(1,nsim), nvar, nblock].
  ! variance is [nvar, nvar, nblock].
  !============================================================================
  type, extends(t_data) :: t_blockgrid
    integer              :: block_type = 0
    integer, allocatable :: order(:)
    integer, allocatable :: nblockpnt(:)
    integer, allocatable :: iblockpnt(:)
    real, allocatable    :: rangescale(:)
    real, allocatable    :: localnugget(:)
    real, allocatable    :: sample(:,:,:)
  end type t_blockgrid

  !============================================================================
  ! t_grad — gradient constraint pair group
  !
  ! Each entry represents one finite-difference pair:
  !   coord    [nlag, n] — positive-side fictitious point xs1
  !   coord2   [nlag, n] — negative-side fictitious point xs2
  !   value    [1, 1, n] — known delta Z = Z(xs1) - Z(xs2)
  !   variance [1, 1, n] — gradient observation variance (0 = exact)
  !   drift    [ndrift+naug, 1, n] — drift-function differences f(xs1)-f(xs2)
  !
  ! Indexed as self%grad(ivar): one group per kriging variable.
  !============================================================================
  type, extends(t_data) :: t_grad
    real, allocatable :: coord2(:,:)   ! [nlag, n] negative-side fictitious point
  end type t_grad

  !============================================================================
  ! t_weight_store — in-memory storage of per-block kriging weights
  !
  ! Allocated on demand by alloc_weight_store() before solve(), then filled
  ! block-by-block during solve().  Retrieve results via the CAPI get functions.
  !
  ! Array layout (Fortran column-major):
  !   nnear (ngroups, nblock)         — neighbour count per group per block
  !   inear (nmax,   ngroups, nblock) — neighbour obs/block indices (0 = unused)
  !   weight(nmax,   ngroups, nvar, nblock) — kriging weights per variable
  !   var   (nvar,   nvar,    nblock) — conditional covariance matrix
  !
  ! nmax is the maximum nmax across obs variables, extended for gradient groups.
  !============================================================================
  type :: t_weight_store
    integer              :: nblock  = 0
    integer              :: ngroups = 0
    integer              :: nmax    = 0
    integer, allocatable :: order (:)         ! [nblock]
    integer, allocatable :: nnear (:,   :)    ! [ngroups, nblock]
    integer, allocatable :: inear (:,:, :)    ! [nmax,    ngroups, nblock]
    real,    allocatable :: weight(:,:, :,:)  ! [nmax,    ngroups, nvar, nblock]
    real,    allocatable :: var   (:,:, :)    ! [nvar,    nvar,          nblock]
  contains
    procedure :: stored => wstore_check_stored
  end type t_weight_store

  !============================================================================
  ! t_kriging_base — abstract base type for t_kriging and t_kriging_st
  !
  ! Common scalar fields.  Type-specific state (obs, grid, block, vgm, ...)
  ! lives in the derived types.  solve() is a non-overridable template method.
  !============================================================================
  type, abstract :: t_kriging_base
    ! --- string representation (C-compatible for CAPI) ---
    character(kind=c_char), pointer :: krige_info(:) => null()
    ! --- options ---
    logical :: anisotropic_search = .false.
    logical :: weight_correction  = .false.
    logical :: use_old_weight     = .false.
    logical :: store_weight       = .false.
    logical :: cross_validation   = .false.
    logical :: write_mat          = .false.
    logical :: verbose            = .false.
    logical :: neglect_error      = .false.
    logical :: varying_vgm        = .false.
    logical :: std_ck             = .true.
    logical :: pf_cache           = .true.
#ifdef KRIGEKIT_HCACHE_SLOTS
    integer :: factor_cache_size  = KRIGEKIT_HCACHE_SLOTS
#else
    integer :: factor_cache_size  = 64      ! per-thread multi-entry factor cache; <=0 disables
#endif
    character(len=1024) :: weight_file = ""
    integer :: ifile              = 0
    integer :: seed               = 0    ! 0 = no explicit seed; >0 for reproducibility
    real    :: bounds(2)          = [-huge(0.0), huge(0.0)]
    ! --- dimensions ---
    integer :: ndim         = 2   ! spatial dimension (always 1..3)
    integer :: nlag         = 2   ! coord/lag vector length: ndim (spatial) or ndim+1 (ST)
    integer :: nvar         = 1
    integer :: ndrift       = 0
    integer :: unbias       = 1
    integer :: nsim         = 0
    integer :: ngroups      = 0
    integer :: ngroups_base = 0
    integer :: naug         = 0
    integer :: ngrad        = 0
    integer :: nppmax       = 0
    integer :: matsize_max  = 0
    integer :: mmax         = 0
    ! --- solver state ---
    logical :: solved         = .false. ! .true. after a successful solve(); reset at the start of each call
    ! --- solver statistics (reset at start of each solve()) ---
    integer :: n_fail         = 0  ! blocks solve failed. Solve continue to next block when neglect_error=.true.
    integer :: n_chol_fact    = 0  ! blocks solved by Cholesky (fresh factorize)
    integer :: n_chol_reuse   = 0  ! blocks solved by Cholesky (cache hit)
    integer :: n_ssytrf_fact  = 0  ! SSYTRF factorizations performed (O(n³), one per unique neighbourhood)
    integer :: n_ssytrf_reuse = 0  ! blocks solved by a cached SSYTRF (O(n²) SSYTRS)
    ! --- ctx init helpers (allocated and populated by pre_solve) ---
    integer, allocatable :: obs_nmax(:)   ! [nvar]  ! total neighbours for each variable; obs%nmax could be the sector size if sector_search == .true.
    integer, allocatable :: grad_n(:)     ! [nvar]; filled from grad(:) when present
    type(t_obsgrid),   pointer :: obs(:) => null()
    type(t_grid),      pointer :: grid  => null()
    type(t_blockgrid), pointer :: block => null()
    type(t_factor_cache) :: pf
    type(t_weight_store), allocatable :: wstore
    type(t_grad), pointer :: grad(:) => null()
  contains
    procedure, non_overridable :: solve           => solve_base
    procedure, non_overridable :: assemble_linear_system => assemble_linear_system_base
    procedure                  :: initialize      => initialize_base   ! concrete; overridable
    procedure                  :: init_defaults   => init_defaults_base ! no-op; type overrides for specific defaults
    procedure, non_overridable :: set_obs         => set_obs_base
    procedure                  :: update_obs_value => update_obs_value_base
    procedure, non_overridable :: set_obs_drift   => set_obs_drift_base
    procedure, non_overridable :: reset_obs       => reset_obs_common
    procedure, non_overridable :: reset_grid  => reset_grid_common
    procedure, non_overridable :: reset_block => reset_block_common
    procedure, non_overridable :: set_grid_point      => set_grid_point_base
    procedure, non_overridable :: set_grid_gq         => set_grid_gq_base
    procedure, non_overridable :: set_grid_user_block => set_grid_user_block_base
    procedure, non_overridable :: set_grid_block      => set_grid_block_base
    procedure, non_overridable :: set_grid_cv         => set_grid_cv_base
    procedure, non_overridable :: set_grid_drift      => set_grid_drift_base
    procedure :: set_sim             => set_sim_base
    procedure, non_overridable :: finalize_common
    procedure, non_overridable :: finish_grid_setup_common
    procedure, non_overridable :: set_grid_point_common
    procedure, non_overridable :: set_grid_nodes_common
    procedure, non_overridable :: set_grid_centroid_common
    procedure, non_overridable :: alloc_block_storage_common
    procedure(kriging_if),              deferred    :: init
    procedure(kriging_if),              deferred    :: prepare
    procedure(kriging_ctx_if),              deferred    :: assemble_rhs
    procedure(kriging_ctx_if),              deferred    :: assemble_lhs
    procedure(kriging_if),              deferred    :: finalize
    procedure, non_overridable :: pre_solve => pre_solve_base
    procedure, non_overridable :: solve_linear_system => solve_linear_system_base
    procedure(kriging_ctx_if),    deferred    :: calc_variance
    procedure(kriging_ivar_ctx_if),  deferred    :: search_neighbors
    procedure, non_overridable :: estimate_block => estimate_block_base
    procedure, non_overridable :: write_matrix => write_matrix_base
    procedure, non_overridable :: reorder_sgsim => reorder_sgsim_base
    procedure :: post_grid_setup => post_grid_setup_base
    procedure :: post_solve      => post_solve_base   ! no-op; subclasses override for post-solve processing (e.g. indicator normalization)
    procedure :: sim_draw        => sim_draw_base     ! Gaussian perturbation; subclasses override for alternative draws (e.g. indicator CDF draw)
    procedure, non_overridable :: fill_ctx_sizing_common
    procedure, non_overridable :: validate_pre_solve_common
    procedure, non_overridable :: prepare_common
    procedure, non_overridable :: set_grad
    procedure, non_overridable :: reset_grad
    procedure, non_overridable :: alloc_weight_store
    procedure, non_overridable :: free_weight_store
    procedure, non_overridable :: save_block_weights
    procedure, non_overridable :: load_block_weights
    procedure, non_overridable :: write_weight
    procedure, non_overridable :: write_weight_store
    procedure, non_overridable :: read_weight
    procedure, non_overridable :: read_weight_to_store
    procedure, non_overridable :: set_weights
    !-- Persistent factor cache accessors
    procedure, non_overridable :: get_persistent_factor_info     => get_persistent_factor_info_base
    procedure, non_overridable :: get_persistent_factor_matrices => get_persistent_factor_matrices_base
    procedure, non_overridable :: get_persistent_factor_system   => get_persistent_factor_system_base
    !-- String representation (update_info is non_overridable; to_str and tostr_vgm are overridable)
    procedure, non_overridable :: update_info  => update_info_base
    procedure                  :: to_str       => to_str_base
    procedure(tostr_vgm_if),     deferred      :: tostr_vgm
  end type t_kriging_base

  !============================================================================
  ! Abstract interfaces for t_kriging_base deferred procedures
  !============================================================================
  abstract interface

    subroutine kriging_if(self)
      import :: t_kriging_base
      class(t_kriging_base), intent(inout) :: self
    end subroutine kriging_if

    subroutine kriging_ctx_if(self, ctx)
      import :: t_kriging_base, t_kriging_ctx
      class(t_kriging_base), intent(inout) :: self
      type(t_kriging_ctx),   intent(inout) :: ctx
    end subroutine

    subroutine kriging_ivar_ctx_if(self, ivar, ctx)
      import :: t_kriging_base, t_kriging_ctx
      class(t_kriging_base), intent(inout) :: self
      integer,               intent(in)    :: ivar
      type(t_kriging_ctx),   intent(inout) :: ctx
    end subroutine kriging_ivar_ctx_if

    !-- Each subclass provides the variogram section of to_str.
    function tostr_vgm_if(self) result(s)
      import :: t_kriging_base
      class(t_kriging_base), intent(in) :: self
      character(len=:), allocatable :: s
    end function tostr_vgm_if

  end interface

contains

  !============================================================================
  ! initialize_base -- shared initialization template.
  !
  ! Call order:
  !   1. apply type-specific defaults
  !   2. apply user-supplied optional overrides
  !   3. compute common dimensions and validate common options
  !   4. allocate base obs/grid/block data structures
  !   5. call the type-specific init() hook
  !============================================================================
  subroutine initialize_base(self, ndim, nvar, ndrift, unbias, nsim, &
      anisotropic_search, weight_correction, use_old_weight, store_weight, &
      cross_validation, write_mat, neglect_error, varying_vgm, std_ck, &
      verbose, pf_cache, weight_file, bounds, seed)
    class(t_kriging_base), intent(inout)   :: self
    integer, intent(in), optional          :: ndim, nvar, ndrift, unbias, nsim, seed
    logical, intent(in), optional          :: anisotropic_search, weight_correction, &
                                              use_old_weight, store_weight, &
                                              cross_validation, write_mat, &
                                              neglect_error, varying_vgm, std_ck, &
                                              verbose, pf_cache
    character(*), intent(in), optional     :: weight_file
    real,    intent(in), optional          :: bounds(2)

    character(len=*), parameter :: subname = "t_kriging_base%initialize"

    !-- Step 1: type-specific defaults (ST overrides ndim, seed, neglect_error, std_ck)
    call self%init_defaults()

    !-- Step 2: apply user-supplied overrides
    if (present(ndim))               self%ndim               = ndim
    if (present(nvar))               self%nvar               = nvar
    if (present(ndrift))             self%ndrift             = ndrift
    if (present(unbias))             self%unbias             = unbias
    if (present(nsim))               self%nsim               = nsim
    if (present(anisotropic_search)) self%anisotropic_search = anisotropic_search
    if (present(weight_correction))  self%weight_correction  = weight_correction
    if (present(use_old_weight))     self%use_old_weight     = use_old_weight
    if (present(store_weight))       self%store_weight       = store_weight
    if (present(cross_validation))   self%cross_validation   = cross_validation
    if (present(write_mat))          self%write_mat          = write_mat
    if (present(neglect_error))      self%neglect_error      = neglect_error
    if (present(varying_vgm))        self%varying_vgm        = varying_vgm
    if (present(std_ck))             self%std_ck             = std_ck
    if (present(verbose))            self%verbose            = verbose
    if (present(pf_cache))           self%pf_cache           = pf_cache
    if (present(weight_file))        self%weight_file        = weight_file
    if (present(bounds))             self%bounds             = bounds
    if (present(seed))               self%seed               = seed

    !-- Step 3: common post-override calculations

    !-- Seed RNG
    if (self%seed /= 0) then
      if (self%verbose .and. self%nsim > 0) &
        print "(A,I0)", " Random seed is set to ", self%seed
      call random_seed_initialize(self%seed)
    end if

    !-- Neighbour-group layout:
    !     Groups 1:nvar              = real obs per variable
    !     Groups nvar+1:ngroups_base = previously simulated blocks (SGSIM only)
    !     Groups ngroups_base+1:...  = gradient pairs (added by prepare)
    self%ngroups_base = merge(self%nvar * 2, self%nvar, self%nsim > 0)
    self%ngroups      = self%ngroups_base

    !-- Unbiasedness constraint rows: naug = unbias*nvar (std_ck) or unbias (non-std)
    self%naug = merge(self%unbias * self%nvar, self%unbias, self%std_ck)

    !-- Common validation
    if (self%store_weight .and. self%use_old_weight) then
      call kriging_error(subname, 'store_weight and use_old_weight are mutually exclusive')
      return
    end if
    if (self%pf_cache .and. self%use_old_weight) then
      call kriging_error(subname, 'pf_cache and use_old_weight are mutually exclusive')
      return
    end if
    if (self%cross_validation .and. self%nsim > 0) then
      call kriging_error(subname, 'cross_validation and nsim>0 are mutually exclusive')
      return
    end if

    !-- Step 4: allocate base obs/grid/block data structures.
    if (associated(self%obs))   deallocate(self%obs)
    if (associated(self%grid))  deallocate(self%grid)
    if (associated(self%block)) deallocate(self%block)
    allocate(self%obs(self%nvar))
    allocate(self%grid)
    allocate(self%block)

    !-- Default nlag = ndim (spatial); ST's init() overrides to ndim+1.
    self%nlag = self%ndim

    !-- Step 5: type-specific allocations and additional validation
    call self%init()
  end subroutine initialize_base



  !============================================================================
  ! set_obs_base -- shared observation setup.
  !============================================================================
  subroutine set_obs_base(self, ivar, coord, value, variance, nmax, maxdist, sk_mean)
    use utils, only: check_duplicate_coordinates_base
    class(t_kriging_base), intent(inout) :: self
    integer, intent(in)                 :: ivar
    real,    intent(in)                 :: coord(:,:), value(:)
    real,    intent(in), optional       :: variance(:), maxdist, sk_mean
    integer, intent(in), optional       :: nmax
    integer :: i
    logical :: has_duplicates
    character(1024) :: msg
    character(len=*), parameter :: subname = "t_kriging_base%set_obs"

    if (.not. associated(self%obs)) then
      call kriging_error(subname, 'Call initialize() before set_obs.')
      return
    end if
    if (.not. kriging_check_index(subname, 'ivar', ivar, 1, self%nvar)) return
    if (size(coord, 1) /= self%nlag) then
      call kriging_error(subname, 'nlag /= size(coord, 1) for obs')
      return
    end if
    if (size(coord, 2) /= size(value)) then
      call kriging_error(subname, 'coord column count != nobs')
      return
    end if
    if (present(variance)) then
      if (size(variance) /= size(value)) then
        call kriging_error(subname, 'variance length != nobs')
        return
      end if
    end if
    call self%reset_obs(ivar)

    associate(obs => self%obs(ivar))
      obs%n = size(value)
      call check_duplicate_coordinates_base(self%nlag, obs%n, coord, has_duplicates, msg)
      if (has_duplicates) then
        call kriging_error(subname, msg)
        return
      end if
      allocate(obs%coord, source=coord)

      allocate(obs%value(1, 1, obs%n))
      obs%value(1, 1, :) = value

      allocate(obs%variance(1, 1, obs%n))
      if (present(variance)) then
        do i = 1, obs%n
          obs%variance(1, 1, i) = variance(i)
        end do
      else
        obs%variance = 0.0
      end if

      if (self%ndrift + self%naug > 0) then
        allocate(obs%drift(self%ndrift + self%naug, 1, obs%n))
        obs%drift = 0.0
        if (self%naug > 0) then
          if (self%std_ck) then
            obs%drift(self%ndrift + ivar, 1, :) = 1.0
          else
            obs%drift(self%ndrift + 1, 1, :) = 1.0
          end if
        end if
      end if

      obs%nmax          = huge(0)
      obs%maxdist       = huge(0.0)
      obs%rotmat        = kriging_identity_rotmat3()
      obs%time_vtype_id = 0
      obs%time_nugget   = 0.0
      obs%time_sill     = 1.0
      obs%time_at       = 1.0
      if (present(nmax))    obs%nmax    = nmax
      if (present(maxdist)) obs%maxdist = maxdist**2
      if (present(sk_mean)) obs%sk_mean = sk_mean
    end associate

    self%pf%valid = .false.
  end subroutine set_obs_base


  !============================================================================
  ! update_obs_value_base -- shared observation value update.
  !
  ! If a complete in-memory weight store already exists, switch to weight reuse.
  ! The stored weights depend only on geometry/search/variogram setup, so this
  ! is valid for both spatial and ST when only observation values change.
  !============================================================================
  subroutine update_obs_value_base(self, ivar, value)
    class(t_kriging_base), intent(inout) :: self
    integer, intent(in)                 :: ivar
    real,    intent(in)                 :: value(:)
    character(len=*), parameter :: subname = "t_kriging_base%update_obs_value"

    if (.not. associated(self%obs)) then
      call kriging_error(subname, 'Call initialize() before update_obs_value.')
      return
    end if
    if (.not. kriging_check_index(subname, 'ivar', ivar, 1, self%nvar)) return
    if (self%obs(ivar)%n == 0) then
      call kriging_error(subname, 'Call set_obs() before update_obs_value.')
      return
    end if
    if (size(value) /= self%obs(ivar)%n) then
      call kriging_error(subname, 'size(value) /= obs%n')
      return
    end if

    self%obs(ivar)%value(1, 1, :) = value
    if (allocated(self%wstore)) then
      if (self%wstore%stored() .and. associated(self%block)) then
        if (allocated(self%block%value)) then
          if (.not. all(ieee_is_nan(self%block%value))) then
            self%use_old_weight = .true.
            self%store_weight   = .false.
          end if
        end if
      end if
    end if
  end subroutine update_obs_value_base


  !============================================================================
  ! set_obs_drift_base -- shared observation external drift setup.
  !============================================================================
  subroutine set_obs_drift_base(self, ivar, drift)
    class(t_kriging_base), intent(inout) :: self
    integer, intent(in)                 :: ivar
    real,    intent(in)                 :: drift(:,:)
    character(len=*), parameter :: subname = "t_kriging_base%set_obs_drift"

    if (.not. associated(self%obs)) then
      call kriging_error(subname, 'Call initialize() before set_obs_drift.')
      return
    end if
    if (.not. kriging_check_index(subname, 'ivar', ivar, 1, self%nvar)) return
    if (self%obs(ivar)%n == 0) then
      call kriging_error(subname, 'Observation needs to be set before adding drift.')
      return
    end if
    if (self%ndrift == 0) then
      call kriging_error(subname, 'Observation drift is specified but ndrift==0')
      return
    end if
    if (size(drift, 1) /= self%ndrift) then
      call kriging_error(subname, 'size(drift, 1) /= ndrift')
      return
    end if
    if (size(drift, 2) /= self%obs(ivar)%n) then
      call kriging_error(subname, 'size(drift, 2) /= nobs')
      return
    end if
    if (.not. allocated(self%obs(ivar)%drift)) then
      call kriging_error(subname, 'obs%drift not allocated; call set_obs() before set_obs_drift().')
      return
    end if

    self%obs(ivar)%drift(1:self%ndrift, 1, :) = drift
  end subroutine set_obs_drift_base


  !============================================================================
  ! reset_obs_common -- shared observation reset and KD-tree cleanup.
  !============================================================================
  subroutine reset_obs_common(self, ivar)
    class(t_kriging_base), intent(inout) :: self
    integer, intent(in)                 :: ivar

    if (.not. associated(self%obs)) return
    associate(obs => self%obs(ivar))
      if (associated(obs%tree)) then
        call kdtree2_destroy(obs%tree)
        nullify(obs%tree)
      end if
      if (allocated(obs%coord))    deallocate(obs%coord)
      if (allocated(obs%drift))    deallocate(obs%drift)
      if (allocated(obs%value))    deallocate(obs%value)
      if (allocated(obs%variance)) deallocate(obs%variance)
      obs%n                  = 0
      obs%nmax               = 0
      obs%maxdist            = huge(0.0)
      obs%sk_mean            = 0.0
      obs%time_vtype_id      = 0
      obs%time_nugget        = 0.0
      obs%time_sill          = 1.0
      obs%time_at            = 1.0
      obs%rotmat             = kriging_identity_rotmat3()
      obs%need_search        = .false.
      obs%anisotropic_search = .false.
      obs%set_search         = .false.
    end associate
  end subroutine reset_obs_common


  !============================================================================
  ! reset_grid_common -- clear shared integration-node storage.
  !============================================================================
  subroutine reset_grid_common(self)
    class(t_kriging_base), intent(inout) :: self
    if (.not. associated(self%grid)) return
    associate(g => self%grid)
      g%n = 0
      if (allocated(g%coord))    deallocate(g%coord)
      if (allocated(g%drift))    deallocate(g%drift)
      if (allocated(g%value))    deallocate(g%value)
      if (allocated(g%variance)) deallocate(g%variance)
      if (allocated(g%weight))   deallocate(g%weight)
    end associate
  end subroutine reset_grid_common


  !============================================================================
  ! reset_block_common -- clear shared target-block/result storage.
  !============================================================================
  subroutine reset_block_common(self)
    class(t_kriging_base), intent(inout) :: self
    if (.not. associated(self%block)) return
    associate(b => self%block)
      b%n          = 0
      b%block_type = 0
      if (allocated(b%coord))       deallocate(b%coord)
      if (allocated(b%drift))       deallocate(b%drift)
      if (allocated(b%value))       deallocate(b%value)
      if (allocated(b%variance))    deallocate(b%variance)
      if (allocated(b%order))       deallocate(b%order)
      if (allocated(b%nblockpnt))   deallocate(b%nblockpnt)
      if (allocated(b%iblockpnt))   deallocate(b%iblockpnt)
      if (allocated(b%rangescale))  deallocate(b%rangescale)
      if (allocated(b%localnugget)) deallocate(b%localnugget)
      if (allocated(b%sample))      deallocate(b%sample)
    end associate
  end subroutine reset_block_common


  !============================================================================
  ! finalize_common -- shared cleanup for base-owned data structures.
  !============================================================================
  subroutine finalize_common(self)
    class(t_kriging_base), intent(inout) :: self
    integer :: ivar

    if (associated(self%obs)) then
      do ivar = 1, self%nvar
        call self%reset_obs(ivar)
      end do
      deallocate(self%obs)
    end if
    if (associated(self%grid)) then
      call self%reset_grid()
      deallocate(self%grid)
    end if
    if (associated(self%block)) then
      call self%reset_block()
      deallocate(self%block)
    end if

    if (allocated(self%obs_nmax)) deallocate(self%obs_nmax)
    if (allocated(self%grad_n))   deallocate(self%grad_n)
    if (allocated(self%wstore))   deallocate(self%wstore)
    if (associated(self%grad))    deallocate(self%grad)
    if (allocated(self%pf%nnear))      deallocate(self%pf%nnear)
    if (allocated(self%pf%inear))      deallocate(self%pf%inear)
    if (allocated(self%pf%matA))       deallocate(self%pf%matA)
    if (allocated(self%pf%rhsB))       deallocate(self%pf%rhsB)
    if (allocated(self%pf%L))          deallocate(self%pf%L)
    if (allocated(self%pf%kinv_drift)) deallocate(self%pf%kinv_drift)
    if (allocated(self%pf%schur))      deallocate(self%pf%schur)
    if (allocated(self%pf%Afac))       deallocate(self%pf%Afac)
    if (allocated(self%pf%ipiv))       deallocate(self%pf%ipiv)
    self%pf%valid = .false.

    call kriging_close_unit(self%ifile)
    if (associated(self%krige_info)) deallocate(self%krige_info)
  end subroutine finalize_common


  !============================================================================
  ! alloc_block_storage_common -- allocate per-block outputs and defaults.
  !============================================================================
  subroutine alloc_block_storage_common(self, value_fill, variance_fill)
    class(t_kriging_base), intent(inout) :: self
    real, intent(in), optional :: value_fill, variance_fill
    integer :: ivar

    associate(b => self%block)
      allocate(b%order      (b%n))
      allocate(b%localnugget(b%n))
      allocate(b%rangescale (b%n))
      allocate(b%value      (max(self%nsim, 1), self%nvar, b%n))
      allocate(b%variance   (self%nvar, self%nvar, b%n))

      call set_seq(b%order, b%n)
      b%localnugget = 0.0
      b%rangescale  = 1.0
      if (present(value_fill)) then
        b%value = value_fill
      else
        b%value = 0.0
      end if
      if (present(variance_fill)) then
        b%variance = variance_fill
      else
        b%variance = 0.0
      end if

      if (self%ndrift + self%naug > 0) then
        allocate(b%drift(self%ndrift + self%naug, self%nvar, b%n))
        b%drift = 0.0
        if (self%naug > 0) then
          if (self%std_ck) then
            do ivar = 1, self%nvar
              b%drift(self%ndrift + ivar, ivar, :) = 1.0
            end do
          else
            b%drift(self%ndrift + 1, :, :) = 1.0
          end if
        end if
      end if
    end associate
  end subroutine alloc_block_storage_common


  !============================================================================
  ! primary_obs_ready -- setup-time guard used by shared grid entry points.
  ! Returns .false. after registering a kriging error.
  !============================================================================
  logical function primary_obs_ready(self, subname) result(ok)
    class(t_kriging_base), intent(in) :: self
    character(len=*),      intent(in) :: subname

    ok = .false.
    if (.not. associated(self%obs)) then
      call kriging_error(subname, 'call initialize() before setting the grid.')
      return
    end if
    if (self%obs(1)%n == 0) then
      call kriging_error(subname, 'Observation needs to be set first.')
      return
    end if
    ok = .true.
  end function primary_obs_ready


  !============================================================================
  ! set_grid_point_base -- shared point kriging target setup.
  !============================================================================
  subroutine set_grid_point_base(self, coord, rangescale, localnugget, value_fill, variance_fill)
    class(t_kriging_base), intent(inout) :: self
    real, intent(in)                    :: coord(:,:)
    real, intent(in), optional          :: rangescale(:), localnugget(:)
    real, intent(in), optional          :: value_fill, variance_fill

    real :: fill_value, fill_variance
    character(len=*), parameter :: subname = "t_kriging_base%set_grid_point"

    if (.not. primary_obs_ready(self, subname)) return
    if (size(coord, 1) /= self%nlag) then
      call kriging_error(subname, 'nlag /= size(coord, 1) for self%grid')
      return
    end if

    fill_value    = IEEE_VALUE(0.0, IEEE_QUIET_NAN)
    fill_variance = IEEE_VALUE(0.0, IEEE_QUIET_NAN)
    if (present(value_fill))    fill_value    = value_fill
    if (present(variance_fill)) fill_variance = variance_fill

    call self%set_grid_point_common(coord, localnugget, fill_value, fill_variance)
    if (kriging_failed()) return
    call self%finish_grid_setup_common(rangescale, localnugget, .false., fill_value, fill_variance)
  end subroutine set_grid_point_base


  !============================================================================
  ! set_grid_gq_base -- shared Gaussian-quadrature block target setup.
  !============================================================================
  subroutine set_grid_gq_base(self, coord, blocksize, rangescale, localnugget, value_fill, variance_fill)
    class(t_kriging_base), intent(inout) :: self
    real, intent(in)                    :: coord(:,:), blocksize(:,:)
    real, intent(in), optional          :: rangescale(:), localnugget(:)
    real, intent(in), optional          :: value_fill, variance_fill

    integer :: ngrid, nb, iblock, igrid, igq
    integer, allocatable :: gq_nblockpnt(:)
    real :: fill_value, fill_variance
    real, allocatable :: gq_coord(:,:), gq_weight(:)
    character(len=*), parameter :: subname = "t_kriging_base%set_grid_gq"

    if (.not. primary_obs_ready(self, subname)) return
    if (self%ndim > 3 .or. self%nlag /= self%ndim) then
      call kriging_error(subname, &
        'Gaussian quadrature block setup requires pure spatial kriging (ndim <= 3, no time).')
      return
    end if
    if (size(coord, 1) /= self%nlag) then
      call kriging_error(subname, 'nlag /= size(coord, 1) for self%grid')
      return
    end if
    ngrid = size(coord, 2)
    if (size(blocksize, 1) /= self%nlag) then
      call kriging_error(subname, 'size(blocksize, 1) /= nlag')
      return
    end if
    if (size(blocksize, 2) /= ngrid) then
      call kriging_error(subname, 'size(blocksize, 2) /= nblock')
      return
    end if

    fill_value    = IEEE_VALUE(0.0, IEEE_QUIET_NAN)
    fill_variance = IEEE_VALUE(0.0, IEEE_QUIET_NAN)
    if (present(value_fill))    fill_value    = value_fill
    if (present(variance_fill)) fill_variance = variance_fill

    nb = 4**self%ndim
    allocate(gq_coord(self%ndim, ngrid * nb))
    allocate(gq_weight(ngrid * nb))
    allocate(gq_nblockpnt(ngrid)); gq_nblockpnt = nb
    igrid = 0
    do iblock = 1, ngrid
      call set_gaussian_quadrature(self%ndim, blocksize(:, iblock))
      do igq = 1, nb
        gq_coord(:, igrid + igq) = coord(:, iblock) + gqdelxyz(:, igq)
      end do
      gq_weight(igrid+1:igrid+nb) = gqweight
      igrid = igrid + nb
    end do

    call self%set_grid_nodes_common(coord, gq_nblockpnt, gq_coord, gq_weight, &
      localnugget, fill_value, fill_variance, -4)
    if (kriging_failed()) return
    call self%finish_grid_setup_common(rangescale, localnugget, .false., fill_value, fill_variance)
  end subroutine set_grid_gq_base


  !============================================================================
  ! set_grid_user_block_base -- shared centroid-from-integration-nodes setup.
  !============================================================================
  subroutine set_grid_user_block_base(self, coord, nblockpnt, pointweight, rangescale, &
                                      localnugget, block_type, value_fill, variance_fill)
    class(t_kriging_base), intent(inout) :: self
    real,    intent(in)                 :: coord(:,:)
    integer, intent(in)                 :: nblockpnt(:)
    real,    intent(in), optional       :: pointweight(:), rangescale(:), localnugget(:)
    integer, intent(in), optional       :: block_type
    real,    intent(in), optional       :: value_fill, variance_fill

    integer :: block_type_local
    real :: fill_value, fill_variance
    character(len=*), parameter :: subname = "t_kriging_base%set_grid_user_block"

    if (.not. primary_obs_ready(self, subname)) return
    if (size(coord, 1) /= self%nlag) then
      call kriging_error(subname, 'nlag /= size(coord, 1) for self%grid')
      return
    end if
    block_type_local = 1
    if (present(block_type)) block_type_local = block_type
    if (block_type_local <= 0) then
      call kriging_error(subname, 'block_type must be >0 for user block grids.')
      return
    end if

    fill_value    = IEEE_VALUE(0.0, IEEE_QUIET_NAN)
    fill_variance = IEEE_VALUE(0.0, IEEE_QUIET_NAN)
    if (present(value_fill))    fill_value    = value_fill
    if (present(variance_fill)) fill_variance = variance_fill

    call self%set_grid_centroid_common(coord, nblockpnt, pointweight, &
      localnugget, fill_value, fill_variance, block_type_local)
    if (kriging_failed()) return
    call self%finish_grid_setup_common(rangescale, localnugget, .false., fill_value, fill_variance)
  end subroutine set_grid_user_block_base


  !============================================================================
  ! set_grid_block_base -- shared explicit centre + integration-node setup.
  !============================================================================
  subroutine set_grid_block_base(self, coord, nblockpnt, blockcoord, pointweight, &
                                 localnugget, rangescale, block_type, value_fill, variance_fill)
    class(t_kriging_base), intent(inout) :: self
    real,    intent(in)                 :: coord(:,:), blockcoord(:,:), pointweight(:)
    integer, intent(in)                 :: nblockpnt(:)
    real,    intent(in), optional       :: localnugget(:), rangescale(:)
    integer, intent(in), optional       :: block_type
    real,    intent(in), optional       :: value_fill, variance_fill

    integer :: block_type_local
    real :: fill_value, fill_variance
    character(len=*), parameter :: subname = "t_kriging_base%set_grid_block"

    if (.not. primary_obs_ready(self, subname)) return
    if (size(coord, 1) /= self%nlag) then
      call kriging_error(subname, 'nlag /= size(coord, 1) for block centres')
      return
    end if
    if (size(blockcoord, 1) /= self%nlag) then
      call kriging_error(subname, 'nlag /= size(blockcoord, 1) for block nodes')
      return
    end if

    block_type_local = 1
    if (present(block_type)) block_type_local = block_type
    if (block_type_local <= 0) then
      call kriging_error(subname, 'block_type must be >0 for explicit block grids.')
      return
    end if

    fill_value    = IEEE_VALUE(0.0, IEEE_QUIET_NAN)
    fill_variance = IEEE_VALUE(0.0, IEEE_QUIET_NAN)
    if (present(value_fill))    fill_value    = value_fill
    if (present(variance_fill)) fill_variance = variance_fill

    call self%set_grid_nodes_common(coord, nblockpnt, blockcoord, pointweight, &
      localnugget, fill_value, fill_variance, block_type_local)
    if (kriging_failed()) return
    call self%finish_grid_setup_common(rangescale, localnugget, .false., fill_value, fill_variance)
  end subroutine set_grid_block_base


  !============================================================================
  ! set_grid_cv_base -- shared cross-validation target setup.
  !============================================================================
  subroutine set_grid_cv_base(self)
    class(t_kriging_base), intent(inout) :: self
    real :: fill_nan
    character(len=*), parameter :: subname = "t_kriging_base%set_grid_cv"

    if (.not. primary_obs_ready(self, subname)) return

    self%cross_validation = .true.
    fill_nan = IEEE_VALUE(0.0, IEEE_QUIET_NAN)
    call self%set_grid_point_common(self%obs(1)%coord, value_fill=fill_nan, variance_fill=fill_nan)
    if (kriging_failed()) return
    if (self%obs(1)%nmax > 0) self%obs(1)%nmax = self%obs(1)%nmax + 1
    call self%finish_grid_setup_common(copy_cv_drift=.true., value_fill=fill_nan, variance_fill=fill_nan)
  end subroutine set_grid_cv_base


  !============================================================================
  ! set_grid_drift_base -- shared external drift setup for target blocks.
  !============================================================================
  subroutine set_grid_drift_base(self, drift, ivar)
    class(t_kriging_base), intent(inout) :: self
    real,    intent(in)                 :: drift(:,:)
    integer, intent(in), optional       :: ivar

    integer :: iv, ivar_local
    character(len=*), parameter :: subname = "t_kriging_base%set_grid_drift"

    if (.not. associated(self%block)) then
      call kriging_error(subname, 'Call initialize() before set_grid_drift.')
      return
    end if
    if (self%block%n == 0) then
      call kriging_error(subname, 'Grid needs to be set before adding drift.')
      return
    end if
    if (self%ndrift == 0) then
      call kriging_error(subname, 'grid/block drift is specified but ndrift==0')
      return
    end if
    if (size(drift, 1) /= self%ndrift) then
      call kriging_error(subname, 'size(drift, 1) /= ndrift')
      return
    end if
    if (size(drift, 2) /= self%block%n) then
      call kriging_error(subname, 'size(drift, 2) /= block%n; one drift value per block, not per grid node')
      return
    end if
    if (.not. allocated(self%block%drift)) then
      call kriging_error(subname, 'block%drift not allocated; call set_grid() before set_grid_drift().')
      return
    end if

    ivar_local = -1
    if (present(ivar)) ivar_local = ivar

    if (ivar_local >= 1 .and. ivar_local <= self%nvar) then
      self%block%drift(1:self%ndrift, ivar_local, :) = drift
    else if (ivar_local < 0) then
      do iv = 1, self%nvar
        self%block%drift(1:self%ndrift, iv, :) = drift
      end do
    else
      call kriging_error(subname, 'ivar must be 1..nvar or < 0 (broadcast)')
    end if
  end subroutine set_grid_drift_base


  !============================================================================
  ! set_sim_base -- shared SGSIM random path, samples, and search-coordinate setup.
  !============================================================================
  subroutine set_sim_base(self, randpath, sample)
    class(t_kriging_base), intent(inout) :: self
    integer, intent(in), optional       :: randpath(:)
    real,    intent(in), optional       :: sample(:,:,:)

    real, allocatable :: temp_coord(:,:), draw(:)
    integer :: iblock, ivar, ifile, isim, nb
    character(len=*), parameter :: subname = "t_kriging_base%set_sim"

    if (self%nsim == 0) return
    if (.not. associated(self%obs)) then
      call kriging_error(subname, 'call initialize() before set_sim().')
      return
    end if
    if (.not. associated(self%block) .or. self%block%n == 0) then
      call kriging_error(subname, 'Grid needs to be set first.')
      return
    end if
    if (any(self%obs(1:self%nvar)%n == 0)) then
      call kriging_error(subname, 'Observations need to be set first.')
      return
    end if
    if (any(self%obs(1:self%nvar)%nmax > int(huge(0)*0.001))) then
      call kriging_error(subname, '`nmax` must be set for all observations for SGSIM.')
      return
    end if

    nb = self%block%n
    if (present(randpath)) then
      if (size(randpath) /= nb) then
        call kriging_error(subname, 'size(randpath) /= nblock')
        return
      end if
      self%block%order = randpath
    else
      call set_seq(self%block%order, nb, .true.)
      open(newunit=ifile, file='sgs_path.dat', status='replace')
      write(ifile, '(A,x,I0)') 'SGSIM_Path', nb
      write(ifile, '((1I0))') self%block%order
      close(ifile)
    end if

    if (allocated(self%block%sample)) deallocate(self%block%sample)
    allocate(self%block%sample(self%nsim, self%nvar, nb))
    if (present(sample)) then
      if (size(sample, 1) /= self%nsim) then
        call kriging_error(subname, 'size(sample, 1) /= nsim')
        return
      end if
      if (size(sample, 2) /= self%nvar) then
        call kriging_error(subname, 'size(sample, 2) /= nvar')
        return
      end if
      if (size(sample, 3) /= nb) then
        call kriging_error(subname, 'size(sample, 3) /= nblock')
        return
      end if
      self%block%sample = sample
    else
      allocate(draw(nb))
      do isim = 1, self%nsim
        do ivar = 1, self%nvar
          call r8vec_normal_01(nb, draw)
          self%block%sample(isim, ivar, :) = draw
        end do
      end do
      open(newunit=ifile, file='sgs_sample.dat', status='replace')
      write(ifile, '(A,x,2I10)') 'SGSIM_Sample', self%nsim, self%nvar, nb
      do iblock = 1, nb
        write(ifile, '(*(G0.7,x))') self%block%sample(:,:,iblock)
      end do
      close(ifile)
    end if

    self%block%coord       = self%block%coord      (:, self%block%order)
    self%block%iblockpnt   = self%block%iblockpnt (   self%block%order)
    self%block%nblockpnt   = self%block%nblockpnt (   self%block%order)
    self%block%rangescale  = self%block%rangescale(   self%block%order)
    self%block%localnugget = self%block%localnugget(  self%block%order)
    if (allocated(self%block%drift)) self%block%drift = self%block%drift(:, :, self%block%order)

    do ivar = 1, self%nvar
      associate(obs => self%obs(ivar))
        allocate(temp_coord(self%nlag, obs%n + nb))
        temp_coord(:, 1:obs%n)  = obs%coord
        temp_coord(:, obs%n+1:) = self%block%coord
        call move_alloc(temp_coord, obs%coord)
        obs%set_search = .false.
      end associate
    end do
  end subroutine set_sim_base


  !============================================================================
  ! finish_grid_setup_common -- shared post-processing after concrete grid setup.
  !============================================================================
  subroutine finish_grid_setup_common(self, rangescale, localnugget, copy_cv_drift, &
                                      value_fill, variance_fill)
    class(t_kriging_base), intent(inout) :: self
    real,    intent(in), optional       :: rangescale(:), localnugget(:)
    logical, intent(in), optional       :: copy_cv_drift
    real,    intent(in), optional       :: value_fill, variance_fill

    logical :: copy_drift
    character(len=*), parameter :: subname = "t_kriging_base%finish_grid_setup"

    copy_drift = .false.
    if (present(copy_cv_drift)) copy_drift = copy_cv_drift

    if (.not. associated(self%block) .or. self%block%n <= 0) then
      call kriging_error(subname, 'grid/block setup produced no blocks.')
      return
    end if
    if (present(rangescale)) then
      if (size(rangescale) /= self%block%n) then
        call kriging_error(subname, 'rangescale length != nblock')
        return
      end if
      self%block%rangescale = rangescale
    else
      self%block%rangescale = 1.0
    end if
    if (present(localnugget)) then
      if (size(localnugget) /= self%block%n) then
        call kriging_error(subname, 'localnugget length != nblock')
        return
      end if
      self%block%localnugget = localnugget
    else
      self%block%localnugget = 0.0
    end if

    if (copy_drift) then
      if (self%ndrift > 0) then
        if (.not. allocated(self%obs(1)%drift)) then
          call kriging_error(subname, 'Observation drift is not set while ndrift > 0.')
          return
        end if
        self%block%drift(1:self%ndrift, 1:self%nvar, :) = &
          spread(self%obs(1)%drift(1:self%ndrift, 1, :), dim=2, ncopies=self%nvar)
      end if
    end if

    call set_seq(self%block%order, self%block%n)
    if (present(value_fill))    self%block%value    = value_fill
    if (present(variance_fill)) self%block%variance = variance_fill

    call self%post_grid_setup()
  end subroutine finish_grid_setup_common


  !============================================================================
  ! set_grid_point_common -- shared point-grid setup.
  !============================================================================
  subroutine set_grid_point_common(self, coord, localnugget, value_fill, variance_fill)
    class(t_kriging_base), intent(inout) :: self
    real, intent(in)                    :: coord(:,:)
    real, intent(in), optional          :: localnugget(:)
    real, intent(in), optional          :: value_fill, variance_fill
    integer :: ngrid, igrid

    if (.not. associated(self%grid) .or. .not. associated(self%block)) then
      call kriging_error('set_grid_point_common', 'call initialize() before set_grid()')
      return
    end if
    if (size(coord, 1) /= self%nlag) then
      call kriging_error('set_grid_point_common', 'size(coord,1) /= nlag')
      return
    end if
    ngrid = size(coord, 2)
    if (present(localnugget)) then
      if (size(localnugget) /= ngrid) then
        call kriging_error('set_grid_point_common', 'localnugget length != ngrid')
        return
      end if
    end if

    call self%reset_grid()
    call self%reset_block()

    self%block%n          = ngrid
    self%block%block_type = 0
    self%grid%n           = ngrid
    allocate(self%block%coord, source = coord)
    allocate(self%grid%coord,  source = coord)
    allocate(self%block%nblockpnt(ngrid)); self%block%nblockpnt = 1
    allocate(self%block%iblockpnt, source = [(igrid, igrid = 1, ngrid)])
    allocate(self%grid%weight(ngrid)); self%grid%weight = 1.0

    call self%alloc_block_storage_common(value_fill, variance_fill)
    if (present(localnugget)) self%block%localnugget = localnugget
  end subroutine set_grid_point_common


  !============================================================================
  ! set_grid_nodes_common -- shared setup for block centres + integration nodes.
  !============================================================================
  subroutine set_grid_nodes_common(self, coord, nblockpnt, gridcoord, pointweight, &
                                   localnugget, value_fill, variance_fill, block_type)
    class(t_kriging_base), intent(inout) :: self
    real,    intent(in)                  :: coord(:,:)
    integer, intent(in)                  :: nblockpnt(:)
    real,    intent(in)                  :: gridcoord(:,:), pointweight(:)
    real,    intent(in), optional        :: localnugget(:)
    real,    intent(in), optional        :: value_fill, variance_fill
    integer, intent(in), optional        :: block_type
    integer :: iblock, igrid, nblocks, ngrid

    if (.not. associated(self%grid) .or. .not. associated(self%block)) then
      call kriging_error('set_grid_nodes_common', 'call initialize() before set_grid_block()')
      return
    end if
    if (size(coord, 1) /= self%nlag) then
      call kriging_error('set_grid_nodes_common', 'size(coord,1) /= nlag')
      return
    end if
    if (size(gridcoord, 1) /= self%nlag) then
      call kriging_error('set_grid_nodes_common', 'size(gridcoord,1) /= nlag')
      return
    end if
    nblocks = size(coord, 2)
    if (any(nblockpnt <= 0)) then
      call kriging_error('set_grid_nodes_common', 'nblockpnt entries must be positive')
      return
    end if
    ngrid   = sum(nblockpnt)
    if (size(nblockpnt) /= nblocks) then
      call kriging_error('set_grid_nodes_common', 'nblockpnt length != nblocks')
      return
    end if
    if (size(gridcoord, 2) /= ngrid) then
      call kriging_error('set_grid_nodes_common', 'gridcoord columns != sum(nblockpnt)')
      return
    end if
    if (size(pointweight) /= ngrid) then
      call kriging_error('set_grid_nodes_common', 'pointweight length != sum(nblockpnt)')
      return
    end if
    if (present(localnugget)) then
      if (size(localnugget) /= nblocks) then
        call kriging_error('set_grid_nodes_common', 'localnugget length != nblocks')
        return
      end if
    end if

    call self%reset_grid()
    call self%reset_block()

    self%block%n = nblocks
    if (present(block_type)) then
      self%block%block_type = block_type
    else
      self%block%block_type = 1
    end if
    self%grid%n = ngrid
    allocate(self%block%coord,     source = coord)
    allocate(self%block%nblockpnt, source = nblockpnt)
    allocate(self%block%iblockpnt(nblocks))
    igrid = 1
    do iblock = 1, nblocks
      self%block%iblockpnt(iblock) = igrid
      igrid = igrid + nblockpnt(iblock)
    end do
    allocate(self%grid%coord,  source = gridcoord)
    allocate(self%grid%weight, source = pointweight)

    call self%alloc_block_storage_common(value_fill, variance_fill)
    if (present(localnugget)) self%block%localnugget = localnugget
  end subroutine set_grid_nodes_common


  !============================================================================
  ! set_grid_centroid_common -- setup blocks from integration nodes only.
  !============================================================================
  subroutine set_grid_centroid_common(self, gridcoord, nblockpnt, pointweight, &
                                      localnugget, value_fill, variance_fill, block_type)
    class(t_kriging_base), intent(inout) :: self
    real,    intent(in)                  :: gridcoord(:,:)
    integer, intent(in)                  :: nblockpnt(:)
    real,    intent(in), optional        :: pointweight(:)
    real,    intent(in), optional        :: localnugget(:)
    real,    intent(in), optional        :: value_fill, variance_fill
    integer, intent(in), optional        :: block_type
    integer :: iblock, idim, igrid, nn, nblocks, ngrid
    real    :: denom
    real, allocatable :: blockcoord(:,:), gridweight(:)

    if (size(gridcoord, 1) /= self%nlag) then
      call kriging_error('set_grid_centroid_common', 'size(gridcoord,1) /= nlag')
      return
    end if
    nblocks = size(nblockpnt)
    if (any(nblockpnt <= 0)) then
      call kriging_error('set_grid_centroid_common', 'nblockpnt entries must be positive')
      return
    end if
    ngrid   = sum(nblockpnt)
    if (size(gridcoord, 2) /= ngrid) then
      call kriging_error('set_grid_centroid_common', 'gridcoord columns != sum(nblockpnt)')
      return
    end if
    if (present(pointweight)) then
      if (size(pointweight) /= ngrid) then
        call kriging_error('set_grid_centroid_common', 'pointweight length != sum(nblockpnt)')
        return
      end if
    end if

    allocate(gridweight(ngrid))
    if (present(pointweight)) then
      gridweight = pointweight
    else
      igrid = 0
      do iblock = 1, nblocks
        nn = nblockpnt(iblock)
        gridweight(igrid+1:igrid+nn) = 1.0 / nn
        igrid = igrid + nn
      end do
    end if

    allocate(blockcoord(self%nlag, nblocks))
    igrid = 0
    do iblock = 1, nblocks
      nn = nblockpnt(iblock)
      denom = sum(gridweight(igrid+1:igrid+nn))
      if (denom <= 0.0) then
        call kriging_error('set_grid_centroid_common', 'sum(pointweight) must be positive for every block')
        return
      end if
      do idim = 1, self%nlag
        blockcoord(idim, iblock) = &
          sum(gridcoord(idim, igrid+1:igrid+nn) * gridweight(igrid+1:igrid+nn)) / denom
      end do
      igrid = igrid + nn
    end do

    call self%set_grid_nodes_common(blockcoord, nblockpnt, gridcoord, gridweight, &
      localnugget, value_fill, variance_fill, block_type)
  end subroutine set_grid_centroid_common

  ! t_factor_cache methods
  !============================================================================

  subroutine fcache_alloc(self, npp, p, nvar, ngroups, mmax, with_system)
    class(t_factor_cache), intent(inout) :: self
    integer, intent(in) :: npp, p, nvar, ngroups, mmax
    logical, intent(in), optional :: with_system
    integer :: matsize, pg
    logical :: alloc_system
    matsize = npp + p
    pg = max(1, p)
    alloc_system = .true.
    if (present(with_system)) alloc_system = with_system
    allocate(self%nnear(ngroups))
    allocate(self%inear(mmax, ngroups))
    if (alloc_system) then
      allocate(self%matA(matsize, matsize))
      allocate(self%rhsB(nvar, matsize))
    end if
    allocate(self%L(npp, npp))
    allocate(self%kinv_drift(npp, pg))
    allocate(self%schur(pg, pg))
    self%nnear = 0
    self%inear = 0
    if (allocated(self%matA)) self%matA = 0.0
    if (allocated(self%rhsB)) self%rhsB = 0.0
    self%system_valid = .false.
  end subroutine fcache_alloc


  !----------------------------------------------------------------------------
  ! fcache_copy_to — copy the active factor slices from self into dst
  !
  ! Only the live portion [1:npp, 1:pg] is transferred; both caches must be
  ! pre-allocated to at least those dimensions.  Does NOT update the key or
  ! valid flag of dst — call save_key() separately after copy_to().
  !----------------------------------------------------------------------------
  subroutine fcache_copy_to(self, dst, npp, p)
    class(t_factor_cache), intent(in)    :: self
    type(t_factor_cache),  intent(inout) :: dst
    integer, intent(in) :: npp, p
    integer :: pg, m
    pg = max(1, p)
    dst%used_ssysv = self%used_ssysv
    if (self%used_ssysv) then
      ! SSYSV path: copy LDLᵀ factors. Allocate destination lazily —
      ! hcache slots and ctx%cache only pay this memory when SSYTRF fires.
      m = npp + p
      if (.not. allocated(dst%Afac) .or. size(dst%Afac, 1) < m) then
        if (allocated(dst%Afac)) deallocate(dst%Afac, dst%ipiv)
        allocate(dst%Afac(m, m))
        allocate(dst%ipiv(m))
      end if
      dst%Afac(1:m, 1:m) = self%Afac(1:m, 1:m)
      dst%ipiv(1:m)       = self%ipiv(1:m)
    else
      ! Cholesky path: copy block factors.
      dst%L         (1:npp, 1:npp) = self%L         (1:npp, 1:npp)
      dst%kinv_drift(1:npp, 1:pg ) = self%kinv_drift(1:npp, 1:pg )
      dst%schur     (1:pg,  1:pg ) = self%schur     (1:pg,  1:pg )
    end if
  end subroutine fcache_copy_to

  !----------------------------------------------------------------------------
  ! fcache_copy_all — full cache copy: matrices + key metadata
  !
  ! Copies everything from src into self so that self becomes an exact replica
  ! ready for cache matching, without needing a live ctx or krige argument.
  ! Used to pre-warm a per-thread ctx%cache from self%pf before the block loop,
  ! eliminating the need to enter the pf CRITICAL section for matching blocks.
  !
  ! Both caches must be pre-allocated to at least src%npp / src%p dimensions.
  ! If src%valid is .false. this is a no-op.
  subroutine fcache_copy_all(self, src)
    class(t_factor_cache), intent(inout) :: self
    type(t_factor_cache),  intent(in)    :: src
    integer :: matsize, ngroups, nvar
    if (.not. src%valid) return
    matsize = src%npp + src%p
    !-- Inline copy_to body (L, kinv_drift, schur) to avoid any implicit
    !   TYPE→CLASS conversion when src is type(t_factor_cache), which can
    !   trigger null-descriptor crashes in Intel ifort/ifx.
    self%used_ssysv = src%used_ssysv
    if (src%used_ssysv) then
      ! SSYSV path: copy LDLᵀ factors into self (lazy allocation).
      associate(m => src%npp + src%p)
      if (.not. allocated(self%Afac) .or. size(self%Afac, 1) < m) then
        if (allocated(self%Afac)) deallocate(self%Afac, self%ipiv)
        allocate(self%Afac(m, m))
        allocate(self%ipiv(m))
      end if
      self%Afac(1:m, 1:m) = src%Afac(1:m, 1:m)
      self%ipiv(1:m)       = src%ipiv(1:m)
      end associate
    else
      ! Cholesky path: copy block factors.
      associate(pg => max(1, src%p))
      self%L         (1:src%npp, 1:src%npp) = src%L         (1:src%npp, 1:src%npp)
      self%kinv_drift(1:src%npp, 1:pg     ) = src%kinv_drift(1:src%npp, 1:pg     )
      self%schur     (1:pg,      1:pg     ) = src%schur     (1:pg,      1:pg     )
      end associate
    end if
    self%npp         = src%npp
    self%p           = src%p
    self%rangescale  = src%rangescale
    self%localnugget = src%localnugget
    ngroups = size(src%nnear)
    self%nnear(1:ngroups) = src%nnear(1:ngroups)
    self%inear             = src%inear
    !-- Copy assembled LHS/RHS only when both caches carry system snapshots.
    !   The persistent cache stores these for get_factor(); hcache slots do not,
    !   and use copy_to() for factor-only transfer.
    self%system_valid = .false.
    if (src%system_valid .and. allocated(self%matA) .and. allocated(src%matA) .and. &
        allocated(self%rhsB) .and. allocated(src%rhsB)) then
      nvar = size(src%rhsB, 1)
      self%matA(1:matsize, 1:matsize) = src%matA(1:matsize, 1:matsize)
      self%rhsB(1:nvar,    1:matsize) = src%rhsB(1:nvar,    1:matsize)
      self%system_valid = .true.
    end if
    self%valid = .true.
  end subroutine fcache_copy_all


  !============================================================================
  ! t_factor_hash_cache methods
  !============================================================================

  subroutine fhcache_alloc(self, nslot, npp, p, nvar, ngroups, mmax)
    class(t_factor_hash_cache), intent(inout) :: self
    integer, intent(in) :: nslot, npp, p, nvar, ngroups, mmax
    integer :: i

    call self%free()
    self%nslot = max(0, nslot)
    self%clock = 0
    if (self%nslot <= 0) then
      self%nbucket = 0
      return
    end if
    self%nbucket = max(1, (self%nslot + 3) / 4)

    allocate(self%slot(self%nslot))
    allocate(self%bucket(self%nbucket))
    allocate(self%next(self%nslot))
    self%bucket = 0
    self%next = 0
    do i = 1, self%nslot
      call self%slot(i)%fac%alloc(npp, p, nvar, ngroups, mmax, with_system=.false.)
      self%slot(i)%valid = .false.
      self%slot(i)%hash = 0
      self%slot(i)%last_used = 0
    end do
  end subroutine fhcache_alloc


  subroutine fhcache_free(self)
    class(t_factor_hash_cache), intent(inout) :: self
    integer :: i

    if (allocated(self%slot)) then
      do i = 1, size(self%slot)
        if (allocated(self%slot(i)%fac%nnear))      deallocate(self%slot(i)%fac%nnear)
        if (allocated(self%slot(i)%fac%inear))      deallocate(self%slot(i)%fac%inear)
        if (allocated(self%slot(i)%fac%matA))       deallocate(self%slot(i)%fac%matA)
        if (allocated(self%slot(i)%fac%rhsB))       deallocate(self%slot(i)%fac%rhsB)
        if (allocated(self%slot(i)%fac%L))          deallocate(self%slot(i)%fac%L)
        if (allocated(self%slot(i)%fac%kinv_drift)) deallocate(self%slot(i)%fac%kinv_drift)
        if (allocated(self%slot(i)%fac%schur))      deallocate(self%slot(i)%fac%schur)
      end do
      deallocate(self%slot)
    end if
    if (allocated(self%bucket)) deallocate(self%bucket)
    if (allocated(self%next)) deallocate(self%next)
    self%nslot = 0
    self%nbucket = 0
    self%clock = 0
  end subroutine fhcache_free


  integer function fhcache_hash_key(ctx) result(h)
    type(t_kriging_ctx), intent(in) :: ctx
    integer :: kvar, i

    h = 216613626
    call mix_int(h, ctx%npp)
    call mix_int(h, ctx%p)
    call mix_int(h, nint(ctx%rangescale  * 1000000.0))
    call mix_int(h, nint(ctx%localnugget * 1000000.0))

    do kvar = 1, ctx%ngroups_base
      call mix_int(h, ctx%nnear(kvar))
      do i = 1, ctx%nnear(kvar)
        call mix_int(h, ctx%inear(i,kvar))
      end do
    end do

  contains

    subroutine mix_int(hh, x)
      integer, intent(inout) :: hh
      integer, intent(in)    :: x
      hh = ieor(hh, x)
      hh = hh * 16777619
      if (hh == 0) hh = 216613626
    end subroutine mix_int

  end function fhcache_hash_key


  integer function fhcache_bucket_index(cache, h) result(ibucket)
    type(t_factor_hash_cache), intent(in) :: cache
    integer, intent(in) :: h

    if (cache%nbucket <= 0) then
      ibucket = 0
    else
      ibucket = 1 + modulo(h, cache%nbucket)
    end if
  end function fhcache_bucket_index


  subroutine fhcache_unlink_slot(cache, islot)
    type(t_factor_hash_cache), intent(inout) :: cache
    integer, intent(in) :: islot
    integer :: ibucket, iprev, icur

    if (islot < 1 .or. islot > cache%nslot) return
    if (.not. allocated(cache%bucket)) return
    if (.not. allocated(cache%next)) return
    if (.not. cache%slot(islot)%valid) return

    ibucket = fhcache_bucket_index(cache, cache%slot(islot)%hash)
    if (ibucket <= 0) return

    iprev = 0
    icur = cache%bucket(ibucket)
    do while (icur > 0)
      if (icur < 1 .or. icur > cache%nslot) exit
      if (icur == islot) then
        if (iprev == 0) then
          cache%bucket(ibucket) = cache%next(icur)
        else
          cache%next(iprev) = cache%next(icur)
        end if
        cache%next(icur) = 0
        return
      end if
      iprev = icur
      icur = cache%next(icur)
    end do
  end subroutine fhcache_unlink_slot


  !-- fhcache_lookup / fhcache_insert are standalone procedures (not bound to
  !   t_factor_hash_cache) so that ctx%hcache, ctx%cache, and ctx itself are
  !   never passed as separate dummy arguments that alias the same storage.
  !   Fortran §15.5.2.13 forbids multiple dummies associated with overlapping
  !   actual arguments when any dummy is defined.

  logical function fhcache_lookup(ctx, krige) result(found)
    type(t_kriging_ctx),   intent(inout) :: ctx    ! single object — no aliasing
    class(t_kriging_base), intent(in)    :: krige
    integer :: h, ibucket, i

    found = .false.
    if (ctx%hcache%nslot <= 0) return
    if (.not. allocated(ctx%hcache%slot)) return
    if (.not. allocated(ctx%hcache%bucket)) return
    if (.not. allocated(ctx%hcache%next)) return
    if (krige%varying_vgm) return

    h = fhcache_hash_key(ctx)
    ibucket = fhcache_bucket_index(ctx%hcache, h)
    if (ibucket <= 0) return

    i = ctx%hcache%bucket(ibucket)
    do while (i > 0)
      if (i < 1 .or. i > ctx%hcache%nslot) exit
      if (ctx%hcache%slot(i)%valid .and. ctx%hcache%slot(i)%hash == h) then
        if (fcache_matches(ctx%hcache%slot(i)%fac, ctx, krige%varying_vgm)) then
          call ctx%hcache%slot(i)%fac%copy_to(ctx%cache, &
                                              ctx%hcache%slot(i)%fac%npp, &
                                              ctx%hcache%slot(i)%fac%p)
          call fcache_save_key(ctx%cache, ctx)
          ctx%hcache%clock = ctx%hcache%clock + 1
          ctx%hcache%slot(i)%last_used = ctx%hcache%clock
          found = .true.
          return
        end if
      end if
      i = ctx%hcache%next(i)
    end do
  end function fhcache_lookup


  subroutine fhcache_insert(ctx, krige)
    type(t_kriging_ctx),   intent(inout) :: ctx    ! single object — no aliasing
    class(t_kriging_base), intent(in)    :: krige
    integer :: h, ibucket, i, victim, oldest

    if (ctx%hcache%nslot <= 0) return
    if (.not. allocated(ctx%hcache%slot)) return
    if (.not. allocated(ctx%hcache%bucket)) return
    if (.not. allocated(ctx%hcache%next)) return
    if (.not. ctx%cache%valid) return
    if (krige%varying_vgm) return

    h = fhcache_hash_key(ctx)
    ibucket = fhcache_bucket_index(ctx%hcache, h)
    if (ibucket <= 0) return

    ! If the same key already exists, refresh it in place.
    i = ctx%hcache%bucket(ibucket)
    do while (i > 0)
      if (i < 1 .or. i > ctx%hcache%nslot) exit
      if (ctx%hcache%slot(i)%valid .and. ctx%hcache%slot(i)%hash == h) then
        if (fcache_matches(ctx%hcache%slot(i)%fac, ctx, krige%varying_vgm)) then
          call ctx%cache%copy_to(ctx%hcache%slot(i)%fac, ctx%cache%npp, ctx%cache%p)
          call fcache_save_key(ctx%hcache%slot(i)%fac, ctx)
          ctx%hcache%clock = ctx%hcache%clock + 1
          ctx%hcache%slot(i)%last_used = ctx%hcache%clock
          return
        end if
      end if
      i = ctx%hcache%next(i)
    end do

    victim = 1
    oldest = huge(oldest)
    do i = 1, ctx%hcache%nslot
      if (.not. ctx%hcache%slot(i)%valid) then
        victim = i
        exit
      end if
      if (ctx%hcache%slot(i)%last_used < oldest) then
        oldest = ctx%hcache%slot(i)%last_used
        victim = i
      end if
    end do

    if (ctx%hcache%slot(victim)%valid) call fhcache_unlink_slot(ctx%hcache, victim)

    call ctx%cache%copy_to(ctx%hcache%slot(victim)%fac, ctx%cache%npp, ctx%cache%p)
    call fcache_save_key(ctx%hcache%slot(victim)%fac, ctx)
    ctx%hcache%clock = ctx%hcache%clock + 1
    ctx%hcache%slot(victim)%valid = .true.
    ctx%hcache%slot(victim)%hash = h
    ctx%hcache%slot(victim)%last_used = ctx%hcache%clock
    ctx%hcache%next(victim) = ctx%hcache%bucket(ibucket)
    ctx%hcache%bucket(ibucket) = victim
  end subroutine fhcache_insert


  !============================================================================
  ! t_kriging_ctx methods
  !============================================================================

  subroutine kriging_ctx_free_core(self)
    class(t_kriging_ctx), intent(inout) :: self
    if (allocated(self%nnear )) deallocate(self%nnear)
    if (allocated(self%inear )) deallocate(self%inear)
    if (allocated(self%istart)) deallocate(self%istart)
    if (allocated(self%sqdist)) deallocate(self%sqdist)
    if (allocated(self%weight)) deallocate(self%weight)
    if (allocated(self%x     )) deallocate(self%x)
    if (allocated(self%matA  )) deallocate(self%matA)
    if (allocated(self%rhsB  )) deallocate(self%rhsB)
    if (allocated(self%results)) deallocate(self%results)
    if (allocated(self%cache%nnear))      deallocate(self%cache%nnear)
    if (allocated(self%cache%inear))      deallocate(self%cache%inear)
    if (allocated(self%cache%matA))       deallocate(self%cache%matA)
    if (allocated(self%cache%rhsB))       deallocate(self%cache%rhsB)
    if (allocated(self%cache%L))          deallocate(self%cache%L)
    if (allocated(self%cache%kinv_drift)) deallocate(self%cache%kinv_drift)
    if (allocated(self%cache%schur))      deallocate(self%cache%schur)
    if (allocated(self%cache%Afac))       deallocate(self%cache%Afac)
    if (allocated(self%cache%ipiv))       deallocate(self%cache%ipiv)
    call self%hcache%free()
    self%iblock       = 0
    self%npp          = 0
    self%matsize      = 0
    self%ngroups_base = 0
    self%p            = 0
    self%cache%valid  = .false.
    self%cache%hit    = .false.
    self%cache%system_valid = .false.
  end subroutine kriging_ctx_free_core


  !============================================================================
  ! Utility subroutines
  !============================================================================

  pure function kriging_identity_rotmat3() result(rotmat)
    real :: rotmat(3,3)
    rotmat = reshape([1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0], [3,3])
  end function kriging_identity_rotmat3


  subroutine kriging_normalize_nmax(nmax, ntotal)
    integer, intent(inout) :: nmax
    integer, intent(in)    :: ntotal
    nmax = min(nmax, ntotal)
    if (nmax <= 0) nmax = ntotal
  end subroutine kriging_normalize_nmax


  subroutine kriging_close_unit(unit)
    integer, intent(inout) :: unit
    if (unit /= 0) then
      close(unit)
      unit = 0
    end if
  end subroutine kriging_close_unit


  subroutine kriging_mirror_lower_to_upper(mat, nrow, ncol)
    real,    intent(inout) :: mat(:,:)
    integer, intent(in)    :: nrow, ncol
    integer :: irow, icol
    do irow = 1, nrow
      do icol = irow + 1, ncol
        mat(irow, icol) = mat(icol, irow)
      end do
    end do
  end subroutine kriging_mirror_lower_to_upper


  subroutine kriging_clip_positive_normalize(weight)
    real, intent(inout) :: weight(:)
    real :: total
    weight = merge(weight, 0.0, weight > 0.0)
    total = sum(weight)
    if (total > 0.0) weight = weight / total
  end subroutine kriging_clip_positive_normalize


  pure function kriging_conditional_variance(base_cov, weight, rhs) result(var)
    real, intent(in) :: base_cov
    real, intent(in) :: weight(:), rhs(:)
    real :: var
    var = max(base_cov - dot_product(weight, rhs), 0.0)
  end function kriging_conditional_variance


  logical function kriging_check_index(subname, argname, idx, lo, hi) result(ok)
    character(len=*), intent(in) :: subname, argname
    integer,          intent(in) :: idx, lo, hi
    character(len=256) :: msg
    ok = idx >= lo .and. idx <= hi
    if (.not. ok) then
      write(msg, '(A,I0,A,I0,A,I0)') trim(argname)//' out of range: ', idx, ' not in ', lo, '..', hi
      call kriging_error(subname, trim(msg))
    end if
  end function kriging_check_index


  logical function kriging_check_pair_index(subname, i, j, lo, hi) result(ok)
    character(len=*), intent(in) :: subname
    integer,          intent(in) :: i, j, lo, hi
    ok = kriging_check_index(subname, 'ivar', i, lo, hi)
    if (.not. ok) return
    ok = kriging_check_index(subname, 'jvar', j, lo, hi)
  end function kriging_check_pair_index


  !============================================================================
  ! solve
  !
  ! Main loop: kriging or SGSIM for every block.
  !
  ! Loop structure
  ! --------------
  !   prepare()           — one-time validation and sizing
  !   [OMP parallel]      — one ctx per thread
  !     for ib = 1..nblock:
  !       search_neighbors()      — find nearest data points
  !       assemble_linear_system()— build C and c0
  !       solve_linear_system()   — invert C, compute weights, kriging variance
  !       assign_weight()         — split x into per-variable weight arrays
  !       [write_weight()]        — optional: write to factor file
  !       estimate_block()        — weighted average + SGSIM draw
  !       [write_matrix()]        — optional: dump matrices for debugging
  !   [end OMP]
  !   if SGSIM: reorder estimate and coord back to original block order
  !
  ! OMP guard
  ! ---------
  ! SGSIM (nsim>0) must run sequentially because estimate_block() for block ib
  ! reads block%value for blocks 1..ib-1 (already simulated).  Parallelising
  ! over blocks would race on the shared estimate array.  The IF clause on the
  ! OMP PARALLEL directive disables OMP when nsim>0.
  ! Factor-file weights are pre-loaded into wstore by prepare() so the block
  ! loop uses load_block_weights (in-memory, thread-safe) for both the file path
  ! and the set_weights() in-memory path.  Debug matrix output is serialised with
  ! a small critical section so the kriging work can still run in parallel.
  !============================================================================
  subroutine solve_base(self, nthread, ncache)
    use omp_lib
    class(t_kriging_base) :: self
    integer, intent(in), optional :: nthread
    integer, intent(in), optional :: ncache
    type(t_kriging_ctx), allocatable :: ctx
    integer :: ib, nb, prev_nthread, nthread_local, prev_ncache
    character(len=*), parameter :: subname = "t_kriging_base%solve_base"

    prev_nthread  = 0
    nthread_local = 0
    prev_ncache   = self%factor_cache_size
    if (present(ncache) .and. ncache >= 0) self%factor_cache_size = ncache
#ifdef _OPENMP
    if (present(nthread) .and. nthread > 0) then
      prev_nthread = omp_get_max_threads()
      call omp_set_num_threads(nthread)
      nthread_local = nthread
    else
      nthread_local = omp_get_max_threads()
    end if
#endif

    ! Reset solver state and statistics so every solve() call gets a clean slate.
    self%solved         = .false.
    self%n_fail         = 0
    self%n_chol_fact    = 0
    self%n_chol_reuse   = 0
    self%n_ssytrf_fact  = 0
    self%n_ssytrf_reuse = 0

    call self%pre_solve()
    if (kriging_failed()) goto 900

    nb = self%block%n

    if (self%verbose) print '(A)', 'Starting kriging loop'
#ifdef __INTEL_COMPILER
    if (self%verbose) open(unit=6, carriagecontrol='fortran')
#endif

    !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(ctx) IF(self%nsim==0 .and. nthread_local>1)
    allocate(ctx)
    call initialize_ctx(ctx, self)
    !-- Pre-warm this thread's intra-solve cache from the persistent factor so
    !   that matching blocks get a ctx%cache hit in assemble_linear_system
    !   without ever entering the pf CRITICAL section.  self%pf is read-only
    !   here (set in a prior solve()); ctx%cache is thread-private.
    !-- Pre-warm only when ctx%cache was allocated (use_old_weight=.false. path).
    if (self%pf_cache .and. self%pf%valid .and. allocated(ctx%cache%L)) &
        call ctx%cache%copy_all(self%pf)

    !$OMP DO SCHEDULE(DYNAMIC,1)
    do ib = 1, nb
#ifdef _OPENMP
      if (self%verbose .and. omp_get_thread_num() == omp_get_num_threads()-1) &
        call progress(ib, nb)
#else
      if (self%verbose) call progress(ib, nb)
#endif
      ctx%iblock = ib
      if (self%use_old_weight) then
        call self%load_block_weights(ctx)
      else
        call self%assemble_linear_system(ctx)
        if (kriging_failed()) cycle
        if (ctx%npp > 1) call self%solve_linear_system(ctx)
        if (kriging_failed()) cycle
        call assign_weight_ctx(ctx, self)
      end if
      if (self%store_weight) call self%save_block_weights(ctx)
      if (kriging_failed()) cycle
      call self%estimate_block(ctx)
      if (self%write_mat) then
        !$OMP CRITICAL(write_matrix_io)
        call self%write_matrix(ctx)
        !$OMP END CRITICAL(write_matrix_io)
      end if
    end do
    !$OMP END DO

    if (self%pf_cache .and. ctx%cache%valid .and. ctx%cache%system_valid) then
      !$OMP CRITICAL(pf_save)
      if (.not. self%pf%valid) then
        call ctx%cache%copy_to(self%pf, ctx%cache%npp, ctx%cache%p)
        call fcache_save_key(self%pf, ctx)
        self%pf%matA(1:ctx%matsize, 1:ctx%matsize) = &
            ctx%matA(1:ctx%matsize, 1:ctx%matsize)
        self%pf%rhsB(1:self%nvar, 1:ctx%matsize) = &
            ctx%rhsB(1:self%nvar, 1:ctx%matsize)
        self%pf%system_valid = .true.
      end if
      !$OMP END CRITICAL(pf_save)
    end if
    deallocate(ctx)
    !$OMP END PARALLEL

    if (self%store_weight .and. trim(self%weight_file) /= "") &
      call self%write_weight_store()
    if (kriging_failed()) goto 900

#ifdef __INTEL_COMPILER
    if (self%verbose) close(6)
#else
    if (self%verbose) print *, ""
#endif
    if (self%verbose) print '(A)', 'Kriging completed.'

    if (self%verbose) then
      print '(A)', "Solver Stats"
      print '(A,I0)', 'Solver failure       = ', self%n_fail
      print '(A,I0)', 'Cholesky factorize   = ', self%n_chol_fact
      print '(A,I0)', 'Cholesky reuse       = ', self%n_chol_reuse
      print '(A,I0)', 'SSYTRF factorize     = ', self%n_ssytrf_fact
      print '(A,I0)', 'SSYTRF reuse (O(n²)) = ', self%n_ssytrf_reuse
    end if

    if (self%nsim > 0) call self%reorder_sgsim()
    call self%post_solve()
    self%solved = .true.

900 continue
    self%factor_cache_size = prev_ncache
#ifdef _OPENMP
    if (prev_nthread > 0) call omp_set_num_threads(prev_nthread)
#endif
#ifdef DEBUG
    print *, subname, " Finished."
#endif
  end subroutine solve_base


  subroutine write_matrix_base(self, ctx)
    class(t_kriging_base), intent(inout) :: self
    type(t_kriging_ctx),   intent(inout) :: ctx

    integer              :: kvar, ifile, ii, jvar, isim, is, iv, matsize
    integer              :: idx
    character(len=20)    :: sig, idxstr
    character(len=4)     :: cname(4) = ['x   ', 'y   ', 'z   ', 't   ']
    class(t_data), pointer :: dat

    associate( &
      ib        => ctx%iblock, &
      nnear     => ctx%nnear, &
      k1        => ctx%istart, &
      inear     => ctx%inear, &
      x         => ctx%x, &
      matA      => ctx%matA, &
      rhsB      => ctx%rhsB, &
      npp       => ctx%npp, &
      irandpath => self%block%order)

      matsize = sum(nnear) + self%ndrift + self%naug
      do isim = 1, max(self%nsim, 1)
        do jvar = 1, self%nvar
          write(idxstr, "(I0,'_var',I0,'_sim',I0)") irandpath(ib), jvar, isim
          !-- Neighbour data table
          open(newunit=ifile, file='data_'//trim(idxstr)//'.csv', status='replace')
          write(ifile, '(99(A,:,","))') 'source', 'index', &
            (trim(cname(ii)), ii=1, self%nlag), 'value', 'weight'
          do kvar = 1, self%ngroups
            if (nnear(kvar) == 0) cycle
            if (kvar > self%nvar) then
              is = isim; iv = jvar
              dat => self%block
              write(sig, "('GRID',I0)") kvar - self%nvar
            else
              is = 1; iv = 1
              dat => self%obs(kvar)
              write(sig, "('OBS',I0)") kvar
            end if
            do ii = 1, nnear(kvar)
              idx = inear(ii, kvar)
              write(ifile, "(A,',',I0,*(:,',',G0.8))") &
                trim(sig), idx, dat%coord(:, idx), dat%value(is, iv, idx), &
                x(jvar, k1(kvar) + ii)
            end do
          end do
          close(ifile)
        end do
      end do

      if (npp <= 1) return

      write(idxstr, "(I0)") irandpath(ib)
      !-- Kriging matrix
      open(newunit=ifile, file='matA_'//trim(idxstr)//'.csv', status='replace')
      do ii = 1, matsize
        write(ifile, "(*(G0.8,:,','))") matA(:matsize, ii)
      end do
      close(ifile)

      !-- Right-hand side
      open(newunit=ifile, file='rhsB_'//trim(idxstr)//'.csv', status='replace')
      do ii = 1, matsize
        write(ifile, "(*(G0.8,:,','))") rhsB(:, ii)
      end do
      close(ifile)
    end associate
  end subroutine write_matrix_base


  subroutine reorder_sgsim_base(self)
    class(t_kriging_base), intent(inout) :: self
    integer :: ib, nb
    real, allocatable :: temp_coord(:,:), temp_value(:,:,:), temp_variance(:,:,:)

    nb = self%block%n
    allocate(temp_coord,    source = self%block%coord)
    allocate(temp_value,    source = self%block%value)
    allocate(temp_variance, source = self%block%variance)
    do ib = 1, nb
      self%block%coord(:, self%block%order(ib))       = temp_coord(:, ib)
      self%block%value(:, :, self%block%order(ib))    = temp_value(:, :, ib)
      self%block%variance(:, :, self%block%order(ib)) = temp_variance(:, :, ib)
    end do
  end subroutine reorder_sgsim_base


  subroutine post_grid_setup_base(self)
    class(t_kriging_base), intent(inout) :: self
  end subroutine post_grid_setup_base


  !============================================================================
  ! post_solve_base — no-op default called once after solve() completes.
  !
  ! Subclasses override to apply post-solve corrections to block%value, e.g.:
  !   - indicator kriging: normalize K probability estimates to sum to 1
  !   - order relation correction across thresholds
  !============================================================================
  subroutine post_solve_base(self)
    class(t_kriging_base), intent(inout) :: self
    character(len=*), parameter :: subname = "t_kriging_base%post_solve_base"

#ifdef DEBUG
    print *, subname, " Finished."
#endif
  end subroutine post_solve_base


  !============================================================================
  ! sim_draw_base — default simulation draw: Gaussian perturbation.
  !
  ! Called from estimate_block_base once per (block, isim) after sim-block
  ! conditioning has been added to val.  On entry val holds the conditional
  ! mean (kriging estimate + previously-simulated-block contributions); on
  ! exit val holds the final simulated value for this realisation.
  !
  ! Subclasses override for alternative draws, e.g.:
  !   - indicator simulation: apply order correction then draw from CDF
  ! The pre-computed L_chol (lower Cholesky of conditional variance) is
  ! provided so subclasses that do not need it can simply ignore it.
  !============================================================================
  subroutine sim_draw_base(self, ctx, val, L_chol, isim)
    class(t_kriging_base), intent(inout) :: self
    type(t_kriging_ctx),   intent(in)    :: ctx
    real,                  intent(inout) :: val(:)      ! [nvar] conditional mean in, simulated value out
    real,                  intent(in)    :: L_chol(:,:) ! [nvar, nvar] lower Cholesky of conditional variance
    integer,               intent(in)    :: isim
    val = val + matmul(L_chol, self%block%sample(isim, :, ctx%iblock))
  end subroutine sim_draw_base


  !============================================================================
  ! init_defaults_base — no-op default; derived types override for type-specific
  ! default values that differ from t_kriging_base field defaults.
  !============================================================================
  subroutine init_defaults_base(self)
    class(t_kriging_base), intent(inout) :: self
  end subroutine init_defaults_base


  !============================================================================
  ! initialize_ctx — allocate and zero all t_kriging_ctx arrays
  !============================================================================
  subroutine initialize_ctx(ctx, krige)
    type(t_kriging_ctx),   intent(out) :: ctx
    class(t_kriging_base), intent(in)  :: krige
    integer :: ivar, kvar, kgrad
    ! Per-thread upper bound for the multi-entry factor cache.
#ifdef KRIGEKIT_DISABLE_HCACHE
    integer(int64), parameter :: MAX_HCACHE_BYTES = 0_int64
#else
    integer(int64), parameter :: MAX_HCACHE_BYTES = 64_int64 * 1024_int64 * 1024_int64 ! 64 MB/thread
#endif
    integer(int64) :: bytes_per_real, p_cache, slot_reals, slot_bytes, slot_limit
    integer :: safe_nslot

    associate( &
      npp     => krige%nppmax, &
      matsize => krige%matsize_max, &
      ng      => krige%ngroups, &
      nb      => krige%ngroups_base, &
      nv      => krige%nvar, &
      mmax    => krige%mmax)

      bytes_per_real = max(1_int64, int(storage_size(0.0) / 8, int64))
      p_cache = int(max(1, krige%ndrift + krige%naug), int64)
      slot_reals = int(npp, int64) * int(npp, int64) + &          ! L
                   int(npp, int64) * p_cache + &                  ! kinv_drift
                   p_cache * p_cache                              ! schur
      slot_bytes = max(1_int64, slot_reals * bytes_per_real)
      slot_limit = MAX_HCACHE_BYTES / slot_bytes
      safe_nslot = max(0, min(krige%factor_cache_size, &
                              int(min(slot_limit, int(huge(safe_nslot), int64)))))


      if (.not. krige%use_old_weight) then
        allocate(ctx%istart(ng));         ctx%istart = 0
        allocate(ctx%matA  (matsize, matsize))
        allocate(ctx%rhsB  (nv,      matsize))
        call ctx%cache%alloc(npp, krige%ndrift + krige%naug, nv, nb, mmax)
        if (safe_nslot > 0) &
          call ctx%hcache%alloc(safe_nslot, npp, krige%ndrift + krige%naug, nv, nb, mmax)
        block
          integer :: ncand_limit, ivar
          ncand_limit = 0
          do ivar = 1, nv
            if (krige%obs(ivar)%sector_search) then
              ncand_limit = max(ncand_limit, &
                max(krige%obs(ivar)%n + merge(krige%block%n, 0, associated(krige%block) .and. krige%nsim > 0), &
                    (2**krige%ndim) * krige%obs(ivar)%nmax * 4))
            else
              ncand_limit = max(ncand_limit, krige%obs(ivar)%nmax)
            end if
          end do
          if (ncand_limit > 0) allocate(ctx%results(ncand_limit))
        end block
      end if
      allocate(ctx%nnear (ng))
      allocate(ctx%inear (mmax, ng))
      allocate(ctx%sqdist(mmax, ng));    ctx%sqdist = 0.0
      allocate(ctx%weight(mmax, ng, nv))
      allocate(ctx%x     (nv, matsize)); ctx%x      = 0.0
      ctx%weight       = 0.0
      ctx%ngroups_base = nb
      ctx%p            = krige%ndrift + krige%naug

      call set_seq(ctx%inear(1:mmax, 1), mmax)
      do ivar = 1, nv
        ctx%nnear(ivar)    = krige%obs_nmax(ivar)
        ctx%inear(:, ivar) = ctx%inear(:, 1)
      end do
      do kvar = nv + 1, nb
        ctx%nnear(kvar) = 0
      end do
      if (ng > nb) then
        do ivar = 1, nv
          kgrad = nb + ivar
          ctx%nnear(kgrad) = krige%grad_n(ivar)
          if (ctx%nnear(kgrad) > 0) &
            call set_seq(ctx%inear(1:ctx%nnear(kgrad), kgrad), ctx%nnear(kgrad))
        end do
      end if
    end associate
  end subroutine initialize_ctx


  !============================================================================
  ! assign_weight_ctx — split x into per-variable weight arrays
  !============================================================================
  subroutine assign_weight_ctx(ctx, krige)
    type(t_kriging_ctx),   intent(inout) :: ctx
    class(t_kriging_base), intent(in)    :: krige
    integer :: kvar, ivar
    do kvar = 1, krige%ngroups
      if (ctx%nnear(kvar) == 0) cycle
      do ivar = 1, krige%nvar
        ctx%weight(1:ctx%nnear(kvar), kvar, ivar) = &
          ctx%x(ivar, ctx%istart(kvar)+1 : ctx%istart(kvar)+ctx%nnear(kvar))
      end do
    end do
  end subroutine assign_weight_ctx


  !============================================================================
  ! fcache_matches — true if stored key matches current ctx state
  !============================================================================
  logical function fcache_matches(cache, ctx, varying_vgm) result(ok)
    type(t_factor_cache), intent(in) :: cache
    type(t_kriging_ctx),  intent(in) :: ctx
    logical,              intent(in) :: varying_vgm
    integer :: kvar
    ok = .false.
    if (.not. cache%valid)                         return
    if (varying_vgm)                               return
    if (cache%npp         /= ctx%npp)              return
    if (cache%rangescale  /= ctx%rangescale)       return
    if (cache%localnugget /= ctx%localnugget)      return
    do kvar = 1, ctx%ngroups_base
      if (cache%nnear(kvar) /= ctx%nnear(kvar))   return
      if (ctx%nnear(kvar) > 0) then
        if (any(cache%inear(1:ctx%nnear(kvar), kvar) /= &
                 ctx%inear(1:ctx%nnear(kvar), kvar))) return
      end if
    end do
    ok = .true.
  end function fcache_matches


  !============================================================================
  ! fcache_save_key — snapshot ctx state into cache key
  !============================================================================
  subroutine fcache_save_key(cache, ctx)
    type(t_factor_cache), intent(inout) :: cache
    type(t_kriging_ctx),  intent(in)    :: ctx
    integer :: kvar
    cache%npp         = ctx%npp
    cache%p           = ctx%p
    cache%rangescale  = ctx%rangescale
    cache%localnugget = ctx%localnugget
    cache%nnear(1:ctx%ngroups_base) = ctx%nnear(1:ctx%ngroups_base)
    do kvar = 1, ctx%ngroups_base
      if (ctx%nnear(kvar) > 0) &
        cache%inear(1:ctx%nnear(kvar), kvar) = ctx%inear(1:ctx%nnear(kvar), kvar)
    end do
    cache%valid = .true.
    cache%system_valid = .false.
  end subroutine fcache_save_key

  !============================================================================
  ! filter_by_maxlag — compact (idx, d) arrays in-place, keeping only entries
  ! where d(i) <= dmax.  n is updated to the surviving count.
  !============================================================================
  subroutine filter_by_maxlag(idx, d, n, dmax)
    integer, intent(inout) :: idx(:), n
    real,    intent(inout) :: d(:)
    real,    intent(in)    :: dmax
    integer :: ii, kk
    kk = 0
    do ii = 1, n
      if (d(ii) <= dmax) then
        kk = kk + 1
        idx(kk) = idx(ii)
        d(kk)   = d(ii)
      end if
    end do
    n = kk
  end subroutine filter_by_maxlag


  !============================================================================
  ! group_ivar — map group index ig (1:ngroups) to real variable index (1:nvar).
  ! Obs groups (ig = 1:nvar) → ig; sim groups (ig = nvar+1:2*nvar) → ig - nvar.
  ! Declared elemental so it can be applied to arrays of group indices.
  !============================================================================
  pure elemental integer function group_ivar(ig, nvar) result(res)
    integer, intent(in) :: ig, nvar
    res = merge(ig, ig - nvar, ig <= nvar)
  end function group_ivar


  !============================================================================
  ! fill_ctx_sizing_common — populate obs_nmax, grad_n (zeros), and mmax.
  !
  ! Called at the start of every pre_solve implementation.  Sets grad_n = 0
  ! (the default); pre_solve_base fills actual counts when grad is associated.
  !============================================================================
  subroutine fill_ctx_sizing_common(self)
    class(t_kriging_base), intent(inout) :: self
    integer :: ivar
    if (.not. allocated(self%grad_n))   allocate(self%grad_n(self%nvar))
    do ivar = 1, self%nvar
      self%grad_n(ivar)   = 0
    end do
  end subroutine fill_ctx_sizing_common


  !============================================================================
  ! validate_pre_solve_common — pre-flight validation shared by prepare / prepare_st.
  !
  ! Checks (in order):
  !   1. self%obs associated (initialize called)
  !   2. self%block associated and block%n > 0 (set_grid called)
  !   3. obs(ivar)%n > 0 for every ivar (set_obs called for each variable)
  !   4. obs(ivar)%set_search for every ivar (set_search called)
  !   5. drift arrays allocated when ndrift > 0
  !============================================================================
  subroutine validate_pre_solve_common(self, subname)
    class(t_kriging_base), intent(in) :: self
    character(len=*),      intent(in) :: subname
    integer :: ivar

    if (.not. associated(self%obs)) then
      call kriging_error(subname, 'Call initialize() before solve().')
      return
    end if
    if (.not. associated(self%block) .or. self%block%n == 0) then
      call kriging_error(subname, 'Call set_grid() before solve().')
      return
    end if
    do ivar = 1, self%nvar
      if (self%obs(ivar)%n == 0) then
        call kriging_error(subname, 'Call set_obs() for every variable before solve().')
        return
      end if
      if (.not. self%obs(ivar)%set_search) then
        call kriging_error(subname, 'set_search() needs to be called before solve().')
        return
      end if
    end do
    if (self%ndrift > 0) then
      if (.not. allocated(self%block%drift)) then
        call kriging_error(subname, 'Grid drift is not set while ndrift > 0.')
        return
      end if
      do ivar = 1, self%nvar
        if (.not. allocated(self%obs(ivar)%drift)) then
          call kriging_error(subname, 'Observation drift is not set while ndrift > 0.')
          return
        end if
      end do
    end if
  end subroutine validate_pre_solve_common

  !============================================================================
  ! set_grad
  !
  ! Register gradient observation pairs for variable ivar.  Each pair
  ! (coord(:,k), coord2(:,k)) straddles a boundary; the constraint is
  ! Z(xs1_k) - Z(xs2_k) = value(k).  Mirrors set_obs in style.
  !
  ! Call with size(coord,2) == 0 to clear the grad group for this ivar.
  !============================================================================
  subroutine set_grad(self, ivar, coord, coord2, value, variance, drift_ext)
    class(t_kriging_base), intent(inout) :: self
    integer, intent(in) :: ivar
    real,    intent(in) :: coord(:,:), coord2(:,:), value(:)
    real, intent(in), optional :: variance(:)
    real, intent(in), optional :: drift_ext(:,:)   ! [ndrift, n]
    character(len=*), parameter :: subname = 't_kriging_base%set_grad'
    integer :: ngrad_in

    if (.not. associated(self%grad)) then
      call kriging_error(subname, 'Call initialize() before set_grad.')
      return
    end if
    if (.not. kriging_check_index(subname, 'ivar', ivar, 1, self%nvar)) return
    call self%reset_grad(ivar)

    if (size(coord, 1) /= self%nlag .or. size(coord2, 1) /= self%nlag) then
      call kriging_error(subname, 'size(coord,1) /= nlag or size(coord2,1) /= nlag')
      return
    end if
    if (size(value) /= size(coord, 2) .or. size(value) /= size(coord2, 2)) then
      call kriging_error(subname, 'size(value) /= size(coord,2) or size(value) /= size(coord2,2)')
      return
    end if
    if (present(variance)) then
      if (size(variance) /= size(value)) then
        call kriging_error(subname, 'size(variance) /= size(value)'); return
      end if
    end if
    if (present(drift_ext)) then
      if (size(drift_ext, 1) /= self%ndrift .or. size(drift_ext, 2) /= size(coord, 2)) then
        call kriging_error(subname, 'size(drift_ext,1) /= ndrift or size(drift_ext,2) /= ngrad')
        return
      end if
    end if

    ngrad_in = size(value)
    if (ngrad_in == 0) return

    associate(g => self%grad(ivar))
      g%n = ngrad_in
      allocate(g%coord,  source = coord)
      allocate(g%coord2, source = coord2)

      allocate(g%value(1, 1, ngrad_in))
      g%value(1, 1, :) = value

      allocate(g%variance(1, 1, ngrad_in))
      if (present(variance)) then
        g%variance(1, 1, :) = variance
      else
        g%variance = 0.0
      end if

      if (self%ndrift + self%naug > 0) then
        allocate(g%drift(self%ndrift + self%naug, 1, ngrad_in))
        g%drift = 0.0
        if (self%ndrift > 0) then
          if (.not. present(drift_ext)) then
            call kriging_error(subname, 'drift_ext must be provided when ndrift > 0')
            call self%reset_grad(ivar)
            return
          end if
          g%drift(1:self%ndrift, 1, :) = drift_ext
        end if
      end if
    end associate

    self%pf%valid = .false.
  end subroutine set_grad


  !============================================================================
  ! reset_grad — clear gradient pair group(s)
  !
  ! reset_grad(ivar) — clear the grad group for variable ivar only.
  ! reset_grad()     — clear all grad groups (no argument).
  !============================================================================
  subroutine reset_grad(self, ivar)
    class(t_kriging_base), intent(inout) :: self
    integer, intent(in), optional :: ivar
    integer :: iv, iv1, iv2

    if (associated(self%grad)) then
      if (present(ivar)) then
        iv1 = ivar; iv2 = ivar
      else
        iv1 = 1; iv2 = self%nvar
      end if
      do iv = iv1, iv2
        associate(g => self%grad(iv))
          g%n = 0
          if (allocated(g%coord))    deallocate(g%coord)
          if (allocated(g%coord2))   deallocate(g%coord2)
          if (allocated(g%value))    deallocate(g%value)
          if (allocated(g%variance)) deallocate(g%variance)
          if (allocated(g%drift))    deallocate(g%drift)
        end associate
      end do
    end if
    self%pf%valid = .false.
  end subroutine reset_grad


  !============================================================================
  ! alloc_weight_store
  !
  ! Allocate the in-memory weight store sized for the current problem.
  ! Must be called after set_grid() and set_search() (so block%n and obs%nmax
  ! are set) and before solve().  Calling again replaces the store.
  ! nmax is extended to cover gradient groups when grad is associated.
  !============================================================================
  subroutine alloc_weight_store(self)
    class(t_kriging_base), intent(inout) :: self
    integer :: nb, ng, nm, nv, q
    character(len=*), parameter :: subname = "t_kriging_base%alloc_weight_store"

    if (.not. associated(self%block) .or. self%block%n == 0) then
      call kriging_error(subname, 'call set_grid() before alloc_weight_store()')
      return
    end if
    if (self%ngroups == 0) then
      call kriging_error(subname, 'call initialize() before alloc_weight_store()')
      return
    end if
    nv = self%nvar
    nb = self%block%n
    ng = self%ngroups

    if (.not. allocated(self%obs_nmax)) allocate(self%obs_nmax(self%nvar))
    do q = 1, self%nvar
      if (self%obs(q)%sector_search) then
        self%obs_nmax(q) = (2**self%ndim) * self%obs(q)%nmax
      else
        self%obs_nmax(q) = self%obs(q)%nmax
      end if
    end do
    self%mmax = maxval(self%obs_nmax)

    nm = self%mmax
    if (nm <= 0) then
      call kriging_error(subname, 'call set_obs() before alloc_weight_store() so nmax is set')
      return
    end if

    if (allocated(self%wstore)) deallocate(self%wstore)
    allocate(self%wstore)
    self%wstore%nblock  = nb
    self%wstore%ngroups = ng
    self%wstore%nmax    = nm
    allocate(self%wstore%order (nb));              call set_seq(self%wstore%order, nb)
    allocate(self%wstore%nnear (ng, nb));          self%wstore%nnear  = 0
    allocate(self%wstore%inear (nm, ng, nb));      self%wstore%inear  = 0
    allocate(self%wstore%weight(nm, ng, nv, nb));  self%wstore%weight = 0.0
    allocate(self%wstore%var   (nv, nv, nb));      self%wstore%var    = 0.0
  end subroutine alloc_weight_store


  !============================================================================
  ! free_weight_store
  !============================================================================
  subroutine free_weight_store(self)
    class(t_kriging_base), intent(inout) :: self
    if (allocated(self%wstore)) deallocate(self%wstore)
  end subroutine free_weight_store


  !============================================================================
  ! save_block_weights
  !
  ! Copy ctx%nnear / ctx%inear / ctx%weight for the current block into wstore.
  ! Writes to disjoint ib-slices — safe under OpenMP without a critical section.
  !============================================================================
  subroutine save_block_weights(self, ctx)
    class(t_kriging_base), intent(inout) :: self
    type(t_kriging_ctx),   intent(inout) :: ctx
    integer :: ivar, kvar, nn

    associate(ib => ctx%iblock, ws => self%wstore)
      do kvar = 1, self%ngroups
        nn = ctx%nnear(kvar)
        ws%nnear(kvar, ib) = nn
        if (nn > 0) then
          ws%inear (1:nn, kvar, ib) = ctx%inear (1:nn, kvar)
          do ivar = 1, self%nvar
            ws%weight(1:nn, kvar, ivar, ib) = ctx%weight(1:nn, kvar, ivar)
          end do
        end if
      end do
      ws%var(1:self%nvar, 1:self%nvar, ib) = self%block%variance(1:self%nvar, 1:self%nvar, ib)
    end associate
  end subroutine save_block_weights


  !============================================================================
  ! load_block_weights
  !
  ! Copy the in-memory weight store into ctx for the current block.
  ! Inverse of save_block_weights.
  !============================================================================
  subroutine load_block_weights(self, ctx)
    class(t_kriging_base), intent(inout) :: self
    type(t_kriging_ctx),   intent(inout) :: ctx
    integer :: ivar, kvar, nn

    associate(ib => ctx%iblock, ws => self%wstore)
      self%block%order(ib) = ws%order(ib)
      do kvar = 1, self%ngroups
        nn = ws%nnear(kvar, ib)
        ctx%nnear(kvar) = nn
        if (nn > 0) then
          ctx%inear (1:nn, kvar) = ws%inear (1:nn, kvar, ib)
          do ivar = 1, self%nvar
            ctx%weight(1:nn, kvar, ivar) = ws%weight(1:nn, kvar, ivar, ib)
          end do
        end if
      end do
      ctx%npp = sum(ctx%nnear(1:self%ngroups))
      self%block%variance(1:self%nvar, 1:self%nvar, ib) = ws%var(1:self%nvar, 1:self%nvar, ib)
    end associate
  end subroutine load_block_weights


  !============================================================================
  ! write_weight
  !
  ! Write one block's weights from ctx to the open factor file self%ifile.
  ! Format: same three-line-per-block layout as write_weight_store.
  !============================================================================
  subroutine write_weight(self, ctx)
    class(t_kriging_base), intent(inout) :: self
    type(t_kriging_ctx),   intent(inout) :: ctx
    integer :: ii, ivar, jvar

    associate(ib => ctx%iblock, order => self%block%order)
      write(self%ifile, *) order(ib), &
        ctx%nnear(1:self%ngroups), &
        ((self%block%variance(ivar, jvar, ib), ivar=1, self%nvar), jvar=1, self%nvar)
      write(self%ifile, '(*(:2x,I0))') (ctx%inear(1:ctx%nnear(ii), ii), ii=1, self%ngroups)
      do ivar = 1, self%nvar
        write(self%ifile, '(*(:2x,F0.10))') (ctx%weight(1:ctx%nnear(ii), ii, ivar), ii=1, self%ngroups)
      end do
    end associate
  end subroutine write_weight


  !============================================================================
  ! write_weight_store
  !
  ! Write the complete in-memory weight store to weight_file after solve().
  ! Format (three lines per block):
  !   Header: nblock nvar nmax(1:nvar)
  !   Line 1: original_block_index  variance(1:nvar,1:nvar)  nnear(1:ngroups)
  !   Line 2: inear indices (all groups concatenated)
  !   Line 3: kriging weights (all groups concatenated), one row per variable
  !============================================================================
  subroutine write_weight_store(self)
    class(t_kriging_base), intent(inout) :: self
    integer :: ib, ii, ivar, jvar, ifile

    associate(ws => self%wstore)
      open(newunit=ifile, file=trim(self%weight_file), status='replace')
      write(ifile, *) self%block%n, self%nvar, (self%obs_nmax(ivar), ivar=1, self%nvar) ! sizes
      do ib = 1, self%block%n
        write(ifile, "(I0,*(:2x,I0))") self%block%order(ib), ws%nnear(1:self%ngroups, ib)
        write(ifile, '(*(:2x,I0))') (ws%inear(1:ws%nnear(ii,ib), ii, ib), ii=1, self%ngroups)
        write(ifile, "(*(:2x,G0.10))") &
          ((self%block%variance(ivar, jvar, ib), ivar=1, self%nvar), jvar=1, self%nvar)
        do ivar = 1, self%nvar
          write(ifile, '(*(:2x,F0.10))') &
            (ws%weight(1:ws%nnear(ii,ib), ii, ivar, ib), ii=1, self%ngroups)
        end do
      end do
      close(ifile)
    end associate
  end subroutine write_weight_store


  !============================================================================
  ! read_weight
  !
  ! Read one block's weights from the open factor file self%ifile into ctx.
  ! Used when use_old_weight=.true. (streaming file-reuse mode).
  !============================================================================
  subroutine read_weight(self, ctx)
    class(t_kriging_base), intent(inout) :: self
    type(t_kriging_ctx),   intent(inout) :: ctx
    integer :: ii, ivar, jvar

    associate(ib => ctx%iblock, order => self%block%order)
      read(self%ifile, *) order(ib), ctx%nnear(1:self%ngroups)
      read(self%ifile, *) (ctx%inear(1:ctx%nnear(ii), ii), ii=1, self%ngroups)
      read(self%ifile, *) ((self%block%variance(ivar, jvar, ib), ivar=1, self%nvar), jvar=1, self%nvar)
      do ivar = 1, self%nvar
        read(self%ifile, *) (ctx%weight(1:ctx%nnear(ii), ii, ivar), ii=1, self%ngroups)
      end do
      ctx%npp = sum(ctx%nnear)
    end associate
  end subroutine read_weight


  !============================================================================
  ! read_weight_to_store
  !
  ! Read one block from an already-open factor file directly into wstore.
  ! Called block-by-block in prepare() so the file is fully loaded before
  ! the OpenMP block loop starts.
  !============================================================================
  subroutine read_weight_to_store(self, ifile, ib)
    class(t_kriging_base), intent(inout) :: self
    integer, intent(in) :: ifile, ib
    integer :: ii, ivar, jvar
    character(len=*), parameter :: subname = "t_kriging_base%read_weight_to_store"

    associate(ws => self%wstore)
      read(ifile, *, err=10, end=10) self%wstore%order(ib), ws%nnear(1:self%ngroups, ib)
      read(ifile, *, err=10, end=10) &
        (ws%inear(1:ws%nnear(ii,ib), ii, ib), ii=1, self%ngroups)
      read(ifile, *, err=10, end=10) &
        ((self%block%variance(ivar, jvar, ib), ivar=1, self%nvar), jvar=1, self%nvar)
      ws%var(1:self%nvar, 1:self%nvar, ib) = &
        self%block%variance(1:self%nvar, 1:self%nvar, ib)
      do ivar = 1, self%nvar
        read(ifile, *, err=10, end=10) &
          (ws%weight(1:ws%nnear(ii,ib), ii, ivar, ib), ii=1, self%ngroups)
      end do
    end associate
    return
10  call kriging_error(subname, 'Unexpected end-of-file or read error in factor file.')
  end subroutine read_weight_to_store


  !============================================================================
  ! set_weights
  !
  ! Set the in-memory weight store from caller-supplied arrays (nnear, inear,
  ! weight, and optionally order and variance).
  ! wstore must already be allocated (call alloc_weight_store first).
  !============================================================================
  subroutine set_weights(self, nnear_in, inear_in, weight_in, order_in, var_in)
    class(t_kriging_base), intent(inout) :: self
    integer, intent(in)           :: nnear_in (:, :)       ! [ngroups, nblock]
    integer, intent(in)           :: inear_in (:, :, :)    ! [nmax, ngroups, nblock]
    real,    intent(in)           :: weight_in(:, :, :, :) ! [nmax, ngroups, nvar, nblock]
    integer, intent(in), optional :: order_in (:)          ! [nblock]
    real,    intent(in), optional :: var_in   (:, :, :)    ! [nvar, nvar, nblock]
    integer :: nb, ng, nm, nv
    character(len=*), parameter :: subname = "t_kriging_base%set_weights"

    if (.not. allocated(self%wstore)) then
      call kriging_error(subname, 'call alloc_weight_store() before set_weights()')
      return
    end if
    nv = self%nvar
    nb = self%block%n
    ng = self%ngroups
    nm = self%mmax
    if (nm <= 0) then
      call kriging_error(subname, 'call set_obs() before set_weights() so nmax is set')
      return
    end if
    if (size(nnear_in, 1) /= ng .or. size(nnear_in, 2) /= nb) then
      call kriging_error(subname, 'nnear shape mismatch'); return
    end if
    if (size(inear_in, 1) /= nm .or. size(inear_in, 2) /= ng .or. size(inear_in, 3) /= nb) then
      call kriging_error(subname, 'inear shape mismatch'); return
    end if
    if (size(weight_in,1) /= nm .or. size(weight_in,2) /= ng .or. &
        size(weight_in,3) /= nv .or. size(weight_in,4) /= nb) then
      call kriging_error(subname, 'weight shape mismatch'); return
    end if
    self%wstore%nnear  = nnear_in
    self%wstore%inear  = inear_in
    self%wstore%weight = weight_in
    if (present(order_in)) then
      if (size(order_in) /= nb) then
        call kriging_error(subname, 'order shape mismatch'); return
      end if
      self%wstore%order = order_in
    end if
    if (present(var_in)) then
      if (size(var_in,1) /= nv .or. size(var_in,2) /= nv .or. size(var_in,3) /= nb) then
        call kriging_error(subname, 'var shape mismatch'); return
      end if
      self%wstore%var = var_in
    end if
  end subroutine set_weights


  !============================================================================
  ! prepare_common — shared sizing, validation, and weight-store setup
  !
  ! Called from each concrete type's prepare() before any type-specific work.
  ! Handles (in order):
  !   1. Pre-solve validation (skipped when use_old_weight=.true.)
  !   2. naug recomputation (unbias * nvar or unbias)
  !   3. nppmax / ngrad / matsize_max
  !   4. ngroups adjustment for gradient pairs
  !   5. Weight-store pre-load (use_old_weight) or allocation (store_weight)
  !============================================================================
  subroutine prepare_common(self, subname)
    class(t_kriging_base), intent(inout) :: self
    character(len=*),      intent(in)    :: subname
    integer :: ivar, ib, ios, pf_matsize
    integer :: hdr_nblock, hdr_nvar, nmax(self%nvar)
    logical :: need_pf_realloc

    if (.not. self%use_old_weight) then
      call self%validate_pre_solve_common(subname)
      if (kriging_failed()) return
    end if

    !-- Calculate self%obs_nmax and self%mmax once, taking sector_search into account
    if (.not. allocated(self%obs_nmax)) allocate(self%obs_nmax(self%nvar))
    do ivar = 1, self%nvar
      if (self%obs(ivar)%sector_search) then
        self%obs_nmax(ivar) = (2**self%ndim) * self%obs(ivar)%nmax
      else
        self%obs_nmax(ivar) = self%obs(ivar)%nmax
      end if
    end do
    self%mmax = maxval(self%obs_nmax)

    !-- naug may be recomputed if unbias/nvar/std_ck changed since initialize.
    self%naug = merge(self%unbias * self%nvar, self%unbias, self%std_ck)

    !-- Total neighbours (nppmax) includes grad pairs; track ngrad for ngroups.
    self%nppmax = 0
    self%ngrad  = 0
    do ivar = 1, self%nvar
      self%nppmax = self%nppmax + self%obs_nmax(ivar)
      if (associated(self%grad)) then
        self%nppmax = self%nppmax + self%grad(ivar)%n
        self%ngrad  = self%ngrad  + self%grad(ivar)%n
      end if
    end do
    self%matsize_max = self%nppmax + self%ndrift + self%naug

    !-- ngroups: expand to include grad-group slots only when grad is present.
    if (self%ngrad > 0) then
      self%ngroups = self%ngroups_base + self%nvar
    else
      self%ngroups = self%ngroups_base
    end if

    !-- Pre-allocate the persistent factor cache to worst-case dimensions so
    !   no reallocation occurs inside the !$OMP CRITICAL section during solve.
    !   Reallocate when the cached system shape changes.
    if (self%pf_cache) then
      if (allocated(self%pf%L)) then
        pf_matsize = self%nppmax + self%ndrift + self%naug
        need_pf_realloc = size(self%pf%L, 1) /= self%nppmax
        if (.not. allocated(self%pf%matA)) then
          need_pf_realloc = .true.
        else
          need_pf_realloc = need_pf_realloc .or. size(self%pf%matA, 1) /= pf_matsize
        end if
        if (.not. allocated(self%pf%rhsB)) then
          need_pf_realloc = .true.
        else
          need_pf_realloc = need_pf_realloc .or. size(self%pf%rhsB, 1) /= self%nvar
        end if
        if (need_pf_realloc) then
          if (allocated(self%pf%matA))       deallocate(self%pf%matA)
          if (allocated(self%pf%rhsB))       deallocate(self%pf%rhsB)
          if (allocated(self%pf%L))          deallocate(self%pf%L)
          if (allocated(self%pf%kinv_drift)) deallocate(self%pf%kinv_drift)
          if (allocated(self%pf%schur))      deallocate(self%pf%schur)
          if (allocated(self%pf%Afac))       deallocate(self%pf%Afac)
          if (allocated(self%pf%ipiv))       deallocate(self%pf%ipiv)
          if (allocated(self%pf%nnear))      deallocate(self%pf%nnear)
          if (allocated(self%pf%inear))      deallocate(self%pf%inear)
          self%pf%valid = .false.
        end if
      end if
      if (.not. allocated(self%pf%L)) then
        call self%pf%alloc(self%nppmax, self%ndrift + self%naug, self%nvar, &
                           self%ngroups_base, self%mmax)
      end if
    end if

    !-- Weight reuse: pre-load all blocks from the factor file into wstore so the
    !   parallel block loop can use load_block_weights without any file I/O.
    if (self%use_old_weight) then
      if (trim(self%weight_file) /= "") then
        open(newunit=self%ifile, file=trim(self%weight_file), status='old')
        read(self%ifile, *, iostat=ios) hdr_nblock, hdr_nvar, nmax
        if (ios /= 0) then
          call kriging_error(subname, 'Failed to read weight_file header.')
          call kriging_close_unit(self%ifile); return
        end if
        if (hdr_nblock /= self%block%n .or. hdr_nvar /= self%nvar .or. &
            any(nmax /= self%obs_nmax)) then
          call kriging_error(subname, 'weight_file dimensions do not match this kriging object.')
          call kriging_close_unit(self%ifile); return
        end if
        call self%alloc_weight_store()
        if (kriging_failed()) then; call kriging_close_unit(self%ifile); return; end if
        do ib = 1, self%block%n
          call self%read_weight_to_store(self%ifile, ib)
          if (kriging_failed()) then; call kriging_close_unit(self%ifile); return; end if
        end do
        call kriging_close_unit(self%ifile)
      else
        if (.not. allocated(self%wstore)) then
          call kriging_error(subname, &
            'use_old_weight with no weight_file requires set_weights() to be called first.')
          return
        end if
        if (.not. self%wstore%stored()) then
          call kriging_error(subname, &
            'in-memory weight store is empty. Call set_weights() first.')
          return
        end if
      end if
    end if

    !-- store_weight: auto-allocate the in-memory weight store so every block
    !   can write its weights without per-block file I/O in the hot loop.
    if (self%store_weight) then
      call self%alloc_weight_store()
      self%wstore%order = self%block%order
      if (kriging_failed()) return
    end if
  end subroutine prepare_common


  !============================================================================
  ! pre_solve_base — non-overridable template for the pre-solve phase.
  !
  ! Calls the type-specific prepare() (validates vgm, calls prepare_common),
  ! then fills ctx sizing helpers.  Gradient pair counts are set when
  ! self%grad is associated.
  !============================================================================
  subroutine pre_solve_base(self)
    class(t_kriging_base), intent(inout) :: self
    integer :: ivar
    character(len=*), parameter :: subname = "t_kriging_base%pre_solve_base"

    call self%prepare()
    if (kriging_failed()) return
    call self%fill_ctx_sizing_common()
    if (associated(self%grad)) then
      do ivar = 1, self%nvar
        self%grad_n(ivar) = self%grad(ivar)%n
      end do
      self%mmax = max(self%mmax, maxval(self%grad%n))
    end if
  end subroutine pre_solve_base


    !============================================================================
  ! assemble_linear_system
  !
  ! Orchestrate neighbour search, exact-match detection, cache check, and
  ! matrix/RHS assembly for the current block.
  !
  ! On a factorization cache hit (same neighbour set as the previous block):
  !   assemble_rhs only — the LHS factorization is reused in solve_linear_system.
  !
  ! On a cache miss:
  !   assemble_lhs — fills matA (covariance, drift, unbiasedness columns)
  !   assemble_rhs — fills rhsB (target covariances, drift, unbiasedness RHS)
  !
  ! Exact matches are not special-cased here.  The kriging system always runs;
  ! when an obs sits exactly at the block centre, the solver naturally produces
  ! lambda(j*) close to 1 and calc_variance gives variance close to 0.
  !============================================================================
  subroutine assemble_linear_system_base(self, ctx)
    class(t_kriging_base), intent(inout) :: self
    type(t_kriging_ctx)  , intent(inout) :: ctx

    integer          :: ivar, istart
    character(len=*), parameter :: subname = "t_kriging_base%assemble_linear_system"

    associate(nvar => self%nvar, npp => ctx%npp)

      ctx%cache%hit   = .false.
      ctx%rangescale  = self%block%rangescale (ctx%iblock)
      ctx%localnugget = self%block%localnugget(ctx%iblock)

      !-- Find neighbours for each variable
      do ivar = 1, nvar
        call self%search_neighbors(ivar, ctx)
      end do

      !-- Calculate the starting index of each group in the matrix.
      !   obs+sim groups: obs1, sim1, obs2, sim2 ... (interleaved by variable)
      !   grad groups: follow all obs+sim groups, one slot per variable
      istart = 0
      do ivar = 1, nvar
        ctx%istart(ivar) = istart
        istart = istart + ctx%nnear(ivar)
        if (self%nsim > 0) then
          ctx%istart(ivar+self%nvar) = istart
          istart = istart + ctx%nnear(ivar+self%nvar)
        end if
      end do
      if (self%ngrad > 0) then
        do ivar = 1, nvar
          ctx%istart(self%ngroups_base + ivar) = istart
          istart = istart + self%grad(ivar)%n
        end do
      end if

      npp = sum(ctx%nnear)

      if (sum(ctx%nnear(1:self%ngroups_base)) == 0) then
        if (self%neglect_error) then
          ctx%nnear = 0
          ctx%npp = 0
          ctx%matsize = 0
        else
          call kriging_error(subname, 'not enough neighbors for kriging at block', iblock=ctx%iblock)
        end if
        return
      end if

      do ivar = 1, nvar
        if (ctx%nnear(ivar) > 0) call isort(ctx%inear(1:ctx%nnear(ivar), ivar), ctx%nnear(ivar))
      end do

      ctx%matsize = npp + self%ndrift + self%naug

      if (fcache_matches(ctx%cache, ctx, self%varying_vgm)) then
        ctx%cache%hit = .true.
        call self%assemble_rhs(ctx)
        return
      end if

      if (fhcache_lookup(ctx, self)) then
        ctx%cache%hit = .true.
        call self%assemble_rhs(ctx)
        return
      end if

      ctx%cache%valid = .false.
      ctx%cache%system_valid = .false.
      call self%assemble_lhs(ctx)
      call self%assemble_rhs(ctx)
    end associate
#ifdef DEBUG
    print *, subname, " Finished.", ctx%iblock
#endif
  end subroutine assemble_linear_system_base


  !============================================================================
  ! solve_linear_system_base
  !
  ! Solve the assembled kriging system K * lambda = c0 and compute the
  ! type-specific kriging variance via calc_variance().
  !============================================================================
  subroutine solve_linear_system_base(self, ctx)
    use solver, only: kriging_setup, kriging_solve_prepared, ssytrf_setup, ssytrs_solve
    implicit none
    class(t_kriging_base), intent(inout) :: self
    type(t_kriging_ctx),   intent(inout) :: ctx

    integer :: info, p, q, npp_obs, m
    character(len=*), parameter :: subname = "t_kriging_base%solve_linear_system"

    associate( &
      iblock => ctx%iblock, &
      matA   => ctx%matA, &
      rhsB   => ctx%rhsB, &
      npp    => ctx%npp, &
      x      => ctx%x)

      q = self%nvar
      p = self%ndrift + self%naug

      !--------------------------------------------------------------------
      ! Cache hit: reuse previously stored factorization.
      ! Dispatch to the right solver depending on which factorization is
      ! cached — Cholesky (common) or Bunch-Kaufman LDLᵀ (SSYTRF fallback).
      !--------------------------------------------------------------------
      if (ctx%cache%hit) then
        if (ctx%cache%used_ssysv) then
          call ssytrs_solve(npp, p, q, ctx%cache%Afac, ctx%cache%ipiv, &
                            rhsB, x, info)
          !$OMP ATOMIC
          self%n_ssytrf_reuse = self%n_ssytrf_reuse + 1
        else
          call kriging_solve_prepared(npp, p, q, ctx%cache%L, ctx%cache%kinv_drift, &
                                      ctx%cache%schur, rhsB, x, info)
          !$OMP ATOMIC
          self%n_chol_reuse = self%n_chol_reuse + 1
        end if
      else
        !------------------------------------------------------------------
        ! Cache miss: try Cholesky first (fast, cacheable, SPD path).
        !------------------------------------------------------------------
        call kriging_setup(npp, p, matA, ctx%cache%L, ctx%cache%kinv_drift, &
                           ctx%cache%schur, info)
        if (info == 0) then
          ctx%cache%used_ssysv = .false.
          call fcache_save_key(ctx%cache, ctx)
          !-- Mark that ctx%matA/rhsB still match ctx%cache's factors.
          ctx%cache%system_valid = .true.
          call kriging_solve_prepared(npp, p, q, ctx%cache%L, ctx%cache%kinv_drift, &
                                      ctx%cache%schur, rhsB, x, info)
          if (info == 0) then
            call fhcache_insert(ctx, self)
            !$OMP ATOMIC
            self%n_chol_fact = self%n_chol_fact + 1
          end if
        end if
      end if

      !--------------------------------------------------------------------
      ! Fallback: Cholesky (or cached solve) failed.
      ! Run SSYTRF on the full augmented system and cache the LDLᵀ factors
      ! so every subsequent block with the same neighbourhood calls the
      ! cheap SSYTRS (O(n²)) rather than refactorizing (O(n³)).
      ! This is critical when n is large and the neighbourhood is reused
      ! for many estimation blocks (e.g. global neighbourhood, n > 1000).
      !--------------------------------------------------------------------
      if (info /= 0) then
        m = npp + p
        !-- Lazy allocation: Afac/ipiv are not pre-allocated in fcache_alloc
        !   so the common Cholesky path pays zero extra memory.
        if (.not. allocated(ctx%cache%Afac) .or. size(ctx%cache%Afac, 1) < m) then
          if (allocated(ctx%cache%Afac)) deallocate(ctx%cache%Afac, ctx%cache%ipiv)
          allocate(ctx%cache%Afac(m, m))
          allocate(ctx%cache%ipiv(m))
        end if
        call ssytrf_setup(npp, p, matA, ctx%cache%Afac, ctx%cache%ipiv, info)
        if (info == 0) then
          ctx%cache%used_ssysv   = .true.
          ctx%cache%valid        = .true.
          ctx%cache%system_valid = .true.
          call fcache_save_key(ctx%cache, ctx)
          call ssytrs_solve(npp, p, q, ctx%cache%Afac, ctx%cache%ipiv, &
                            rhsB, x, info)
          if (info == 0) then
            call fhcache_insert(ctx, self)
            !$OMP ATOMIC
            self%n_ssytrf_fact = self%n_ssytrf_fact + 1
            if (self%verbose) then
              !$OMP CRITICAL(solver_print)
              print '(A,I0,A,I0,A)', &
                "  [solver] Cholesky failed — SSYTRF factorized (block ", iblock, &
                ", n=", npp, "); subsequent matching blocks use cached LDLᵀ"
              !$OMP END CRITICAL(solver_print)
            end if
          end if
        else
          ctx%cache%valid        = .false.
          ctx%cache%system_valid = .false.
          if (self%verbose) print '(A,I0)', &
            "  Both Cholesky and SSYTRF failed for block ", iblock
        end if
      end if

      if (info /= 0) then
        ! !$OMP CRITICAL(write_matrix_io)
        ! call self%write_matrix(ctx)
        ! !$OMP END CRITICAL(write_matrix_io)
        if (self%neglect_error) then
          x = IEEE_VALUE(0.0, IEEE_QUIET_NAN)
          ctx%nnear = 1
          npp = self%nvar
          !$OMP ATOMIC
          self%n_fail = self%n_fail + 1
        else
          call kriging_error(subname, 'Singular matrix', iblock=ctx%iblock)
          return
        end if
      end if

      !-- Optional: clip negative obs/sim weights and renormalise.
      !   Grad pair weights are left unchanged.
      !   Disabled for cokriging because secondary weights can legitimately sum to 0.
      if (self%weight_correction .and. self%nvar == 1) then
        npp_obs = sum(ctx%nnear(1:self%ngroups_base))
        call kriging_clip_positive_normalize(x(1, 1:npp_obs))
      end if

      call self%calc_variance(ctx)
    end associate
#ifdef DEBUG
    print *, subname, " Finished. ", ctx%iblock
#endif
  end subroutine solve_linear_system_base


  !============================================================================
  ! estimate_block_base
  !
  ! Compute the kriging estimate or SGSIM realisations for the current block.
  ! The covariance-specific variance has already been computed by calc_variance().
  !============================================================================
  subroutine estimate_block_base(self, ctx)
    implicit none
    class(t_kriging_base), intent(inout) :: self
    type(t_kriging_ctx),   intent(inout) :: ctx

    integer :: ivar, jvar, kvar, givar, kgrad, k, isim, info
    real    :: total_weight(self%nvar)
    real    :: target_mean(self%nvar)
    real    :: L_chol(self%nvar, self%nvar)
    real    :: avg
    logical :: ck_isa

    associate( &
      var    => self%block%variance(:, :, ctx%iblock), &
      val    => self%block%value(:, :, ctx%iblock), &
      iblock => ctx%iblock, &
      nnear  => ctx%nnear, &
      inear  => ctx%inear, &
      weight => ctx%weight)

      if (sum(nnear(1:self%ngroups_base)) == 0) then
        val = IEEE_VALUE(0.0, IEEE_QUIET_NAN)
        var = IEEE_VALUE(0.0, IEEE_QUIET_NAN)
        return
      end if

      !-- Isaaks & Srivastava correction is used only for non-standard cokriging.
      ck_isa = (.not. self%std_ck) .and. self%nsim == 0 &
               .and. self%unbias /= 0 .and. self%nvar > 1
      if (ck_isa) then
        do ivar = 1, self%nvar
          target_mean(ivar) = 0.0
          do k = 1, nnear(ivar)
            target_mean(ivar) = target_mean(ivar) + self%obs(ivar)%value(1, 1, inear(k, ivar))
          end do
          if (nnear(ivar) > 0) target_mean(ivar) = target_mean(ivar) / nnear(ivar)
        end do
      end if

      do jvar = 1, self%nvar
        total_weight = 0.0
        avg = 0.0
        do ivar = 1, self%nvar
          total_weight(ivar) = total_weight(ivar) + sum(weight(1:nnear(ivar), ivar, jvar))
          avg = avg + dot_product(self%obs(ivar)%value(1, 1, inear(1:nnear(ivar), ivar)), &
                                  weight(1:nnear(ivar), ivar, jvar))
          if (ck_isa .and. ivar /= jvar .and. nnear(ivar) > 0) &
            avg = avg + (target_mean(jvar) - target_mean(ivar)) * total_weight(ivar)
        end do

        if (self%ngrad > 0) then
          do givar = 1, self%nvar
            kgrad = self%ngroups_base + givar
            if (nnear(kgrad) == 0) cycle
            do k = 1, nnear(kgrad)
              avg = avg + weight(k, kgrad, jvar) * self%grad(givar)%value(1, 1, k)
            end do
          end do
        end if

        if (self%unbias == 0 .and. self%obs(jvar)%sk_mean /= 0.0) &
          avg = avg + (1.0 - sum(total_weight)) * self%obs(jvar)%sk_mean

        val(:, jvar) = avg
      end do

      if (self%nsim > 0) then
        L_chol = var
        call spotrf('L', self%nvar, L_chol, self%nvar, info)
        if (info /= 0) then
          L_chol = 0.0
          do ivar = 1, self%nvar
            L_chol(ivar, ivar) = sqrt(max(var(ivar, ivar), 0.0))
          end do
        end if

        do isim = 1, self%nsim
          do jvar = 1, self%nvar
            do ivar = 1, self%nvar
              kvar = self%nvar + ivar
              val(isim, jvar) = val(isim, jvar) + &
                dot_product(self%block%value(isim, ivar, inear(1:nnear(kvar), kvar)), &
                            weight(1:nnear(kvar), kvar, jvar))
            end do
          end do

          call self%sim_draw(ctx, val(isim, :), L_chol, isim)
        end do
      end if

      where (val < self%bounds(1)) val = self%bounds(1)
      where (val > self%bounds(2)) val = self%bounds(2)
    end associate
#ifdef DEBUG
    print *, "t_kriging_base%estimate_block Finished.", ctx%iblock
#endif
  end subroutine estimate_block_base


  !============================================================================
  ! isort — in-place insertion sort on the first n elements of integer array a.
  ! Optimal for the small n typical of kriging neighbourhoods (nmax = 20-50).
  !============================================================================
  pure subroutine isort(a, n)
    integer, intent(inout) :: a(:)
    integer, intent(in)    :: n
    integer :: i, j, tmp
    do i = 2, n
      tmp = a(i)
      j   = i - 1
      do while (j >= 1)
        if (a(j) <= tmp) exit
        a(j+1) = a(j)
        j = j - 1
      end do
      a(j+1) = tmp
    end do
  end subroutine isort


  !============================================================================
  ! wstore_check_stored — true when wstore has been populated with weights
  !============================================================================
  function wstore_check_stored(self) result(stored)
    class(t_weight_store), intent(in) :: self
    logical :: stored
    stored = .false.
    if (self%nblock == 0) return
    if (allocated(self%nnear) .and. allocated(self%inear) .and. allocated(self%weight)) &
      stored = any(self%nnear /= 0)
  end function wstore_check_stored


  !=============================================================================
  ! get_persistent_factor_info_base — return scalar metadata from self%pf.
  !=============================================================================
  subroutine get_persistent_factor_info_base(self, npp_out, p_out, valid_out)
    class(t_kriging_base), intent(in) :: self
    integer, intent(out) :: npp_out, p_out
    logical, intent(out) :: valid_out
    npp_out   = self%pf%npp
    p_out     = self%pf%p
    valid_out = self%pf%valid
  end subroutine get_persistent_factor_info_base


  !=============================================================================
  ! get_persistent_factor_matrices_base — copy the three Cholesky matrices.
  !
  ! Caller must have called get_persistent_factor_info first to size outputs.
  ! No-op when pf%valid is .false.
  !=============================================================================
  subroutine get_persistent_factor_matrices_base(self, npp, p, L_out, kinv_out, schur_out)
    class(t_kriging_base), intent(in) :: self
    integer, intent(in)  :: npp, p
    real,    intent(out) :: L_out   (npp, npp)
    real,    intent(out) :: kinv_out(npp, max(1,p))
    real,    intent(out) :: schur_out(max(1,p), max(1,p))
    integer :: pg
    pg = max(1, p)
    if (.not. self%pf%valid) return
    L_out    (1:npp, 1:npp) = self%pf%L         (1:npp, 1:npp)
    kinv_out (1:npp, 1:pg ) = self%pf%kinv_drift(1:npp, 1:pg )
    schur_out(1:pg,  1:pg ) = self%pf%schur     (1:pg,  1:pg )
  end subroutine get_persistent_factor_matrices_base


  !=============================================================================
  ! get_persistent_factor_system_base - copy the raw assembled linear system.
  !
  ! These are the LHS/RHS arrays captured before kriging_setup factorizes the
  ! LHS matrix. Caller must size outputs from get_persistent_factor_info.
  ! No-op when pf%valid is .false.
  !=============================================================================
  subroutine get_persistent_factor_system_base(self, npp, p, nvar, matA_out, rhsB_out)
    class(t_kriging_base), intent(in) :: self
    integer, intent(in)  :: npp, p, nvar
    real,    intent(out) :: matA_out(npp + p, npp + p)
    real,    intent(out) :: rhsB_out(nvar, npp + p)
    integer :: matsize
    matsize = npp + p
    if (.not. self%pf%valid) return
    matA_out(1:matsize, 1:matsize) = self%pf%matA(1:matsize, 1:matsize)
    rhsB_out(1:nvar,    1:matsize) = self%pf%rhsB(1:nvar,    1:matsize)
  end subroutine get_persistent_factor_system_base


  !=============================================================================
  ! to_str_base — build a multi-line info string from base-class fields.
  !
  ! Prints options, dimensions, and per-variable obs stats, then appends the
  ! variogram section supplied by the subclass via tostr_vgm().
  ! Call after set_search() so obs%nmax, obs%need_search, etc. are populated.
  !=============================================================================
  function to_str_base(self) result(res_str)
    class(t_kriging_base), intent(in) :: self
    character(len=:), allocatable :: res_str
    character(len=256) :: buf
    integer :: ivar
    character(len=1), parameter :: NL = new_line('A')

    res_str = NL
    write(buf, "(A)"  ) "==================== Configuration ===================="    ; res_str = res_str // trim(buf) // NL
    write(buf, "(A,A)")  " Version                : ", version                       ; res_str = res_str // trim(buf) // NL
    write(buf, "(A,I0)") " Dimension              : ", self%ndim                     ; res_str = res_str // trim(buf) // NL
    write(buf, "(A,I0)") " Number of Variables    : ", self%nvar                     ; res_str = res_str // trim(buf) // NL
    write(buf, "(A,I0)") " Number of Simulations  : ", self%nsim                     ; res_str = res_str // trim(buf) // NL
    write(buf, "(A,I0)") " Number of Drifts       : ", self%ndrift                   ; res_str = res_str // trim(buf) // NL
    if (associated(self%block)) then
      write(buf, "(A,I0)") " Number of Blocks       : ", self%block%n                ; res_str = res_str // trim(buf) // NL
    end if
    write(buf, "(A,A )") " Ordinary Kriging       : ", yesno(self%unbias == 1)       ; res_str = res_str // trim(buf) // NL
    write(buf, "(A,A )") " Standard CoKriging     : ", yesno(self%std_ck)            ; res_str = res_str // trim(buf) // NL
    write(buf, "(A,A )") " LOO-Cross Validation   : ", yesno(self%cross_validation)  ; res_str = res_str // trim(buf) // NL
    write(buf, "(A,A )") " Weight Correction      : ", yesno(self%weight_correction) ; res_str = res_str // trim(buf) // NL
    write(buf, "(A,A )") " Use Old Weights        : ", yesno(self%use_old_weight)    ; res_str = res_str // trim(buf) // NL
    write(buf, "(A,A )") " Write Matrix for Debug : ", yesno(self%write_mat)         ; res_str = res_str // trim(buf) // NL
    write(buf, "(A,A )") " Write Weight File      : ", yesno(self%store_weight)      ; res_str = res_str // trim(buf) // NL
    if (self%store_weight .or. self%use_old_weight) then
      write(buf, "(A,A )") " Weight File            : ", trim(self%weight_file)       ; res_str = res_str // trim(buf) // NL
    end if
    write(buf, "(A,G0)") " Lower Bound            : ", self%bounds(1)                ; res_str = res_str // trim(buf) // NL
    write(buf, "(A,G0)") " Upper Bound            : ", self%bounds(2)                ; res_str = res_str // trim(buf) // NL

    if (associated(self%obs)) then
      do ivar = 1, self%nvar
        write(buf, "(A,I0,A)") "Variable ", ivar, ":"                                ; res_str = res_str // trim(buf) // NL
        write(buf, "(A,I0)") " Number of data         : ", self%obs(ivar)%n          ; res_str = res_str // trim(buf) // NL
        write(buf, "(A,I0)") " Maximum neighbors      : ", self%obs(ivar)%nmax       ; res_str = res_str // trim(buf) // NL
        write(buf, "(A,G0)") " Maxdist                : ", sqrt(self%obs(ivar)%maxdist) ; res_str = res_str // trim(buf) // NL
        write(buf, "(A,G0)") " Simple Kriging Mean    : ", self%obs(ivar)%sk_mean    ; res_str = res_str // trim(buf) // NL
        write(buf, "(A,A )") " Required Search        : ", yesno(self%obs(ivar)%need_search) ; res_str = res_str // trim(buf) // NL
        write(buf, "(A,A )") " Anisotropic Search     : ", yesno(self%obs(ivar)%anisotropic_search) ; res_str = res_str // trim(buf) // NL
      end do
    end if

    res_str = res_str // self%tostr_vgm()

    write(buf, "(A)") "================== End Configuration =================="     ; res_str = res_str // trim(buf) // NL
  end function to_str_base


  !=============================================================================
  ! update_info_base — convert to_str() output into a null-terminated C string.
  !
  ! The CAPI returns c_loc(obj%krige_info(1)) so Python/C can print the text
  ! without copying.  The buffer is reallocated on every call.
  !=============================================================================
  subroutine update_info_base(self)
    class(t_kriging_base), intent(inout) :: self
    character(len=:), allocatable :: s
    integer :: n, i
    s = self%to_str()
    n = len_trim(s)
    if (associated(self%krige_info)) deallocate(self%krige_info)
    allocate(self%krige_info(n + 1))
    do i = 1, n
      self%krige_info(i) = s(i:i)
    end do
    self%krige_info(n + 1) = c_null_char
  end subroutine update_info_base


end module kriging_base
