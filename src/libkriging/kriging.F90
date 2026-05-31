!==============================================================================
! Module: kriging
!
! Purpose
! -------
! Implements the complete kriging and sequential Gaussian simulation (SGSIM)
! workflow as a Fortran 2003 object-oriented module.  The central type is
! t_kriging, which holds all data structures (observations, grid, variograms,
! solver workspace) and exposes a clean procedural API:
!
!   call k%initialize(...)    ! set options
!   call k%set_obs(...)       ! load observations
!   call k%set_vgm(...)       ! define variogram model(s)
!   call k%set_grid(...)      ! define estimation grid / blocks
!   call k%set_sim(...)       ! (SGSIM only) set random path and samples
!   call k%set_search(...)    ! build k-d trees for neighbour search
!   call k%solve()            ! run kriging or SGSIM for all blocks
  !   ! read results from k%block%value and k%block%variance
!   call k%finalize()         ! release memory
!
! Parallelism
! -----------
! The block loop inside solve() is parallelised with OpenMP.  Each thread
! owns a private t_kriging_ctx (context) object that holds its own copy of
! the working arrays (matrix, RHS, neighbour indices, weights).  The shared
! state (obs, grid, block, vgm) is read-only during the parallel region.
! SGSIM disables OMP because each block conditions on previously simulated
! values written into the shared block%value array.
!
! Key design choices
! ------------------
! * Variogram array vgm(1:nvar, 1:nvar): square matrix of vgm_struct.
!   Simulated-block neighbours use the same variogram as the corresponding real
!   observation variable, resolved via group_ivar() — no separate vgm slots.
! * block%order(ib): maps the sequential (possibly randomised) loop index to
!   the original block index.  In normal kriging order(ib)=ib; in SGSIM it
!   holds the random path permutation.
! * The factor-file (weight_file) allows pre-computed kriging weights to be
!   stored and reloaded, enabling fast ensemble generation without rebuilding
!   the linear system on every realization.
!==============================================================================
module kriging
  use, INTRINSIC    :: ieee_arithmetic
  use iso_fortran_env, only: input_unit, error_unit, output_unit
  use iso_c_binding
  use common
  use kriging_err
  use utils, only: set_seq, r8vec_normal_01, yesno, random_seed_initialize
  use progress_bar, only: progress
  use rotation
  use variogram
  use kdtree2_module
  use gaussian_quadrature
  implicit none

  !============================================================================
  ! t_data — base type for any spatially located dataset
  !
  ! All three spatial classes (t_obsgrid, t_grid, t_blockgrid) extend t_data.
  ! Using a common base keeps covariance assembly generic: assemble_rhs and
  ! assemble_lhs accept class(t_data) pointers and handle obs, grid or block
  ! nodes without branching on the concrete type.
  !============================================================================
  type t_data
    integer              :: n = 0            ! number of spatial nodes
    real, allocatable    :: coord(:,:)       ! coordinates            [ndim, n]
    real, allocatable    :: drift(:,:)       ! drift function values  [ndrift, n]
    real, allocatable    :: value(:,:,:)     ! obs values [1,1,n] or kriging/SGSIM result     [nsim, nvar, n]
    real, allocatable    :: variance(:,:,:)  ! conditional covariance [nvar, nvar, n] or [1, 1, n] for observation error
  end type t_data

  !============================================================================
  ! t_grid — integration / sub-block nodes
  !
  ! For point kriging (block_type=0): one node per block, weight=1.
  ! For block kriging (block_type=-4 or >0): multiple nodes per block,
  ! weights set by Gaussian quadrature or supplied by the user.
  ! The weight array is used in assemble_rhs to average the RHS covariance
  ! over the block volume: c0(i) = sum_k weight(k) * C(obs_i, grid_k).
  !============================================================================
  type, extends(t_data) :: t_grid
    real, allocatable    :: weight(:)    ! integration weights    [sum(nblockpnt)]
  end type t_grid

  !============================================================================
  ! t_blockgrid — estimation targets (one entry per kriging solve)
  !
  ! Each "block" corresponds to one kriging solve.  For point kriging a block
  ! is a single node; for block kriging it is a volume discretised by the
  ! corresponding t_grid nodes.
  !
  ! Fields specific to SGSIM
  ! ------------------------
  ! order(ib)    : random path permutation.  The solve loop visits block
  !                order(1), order(2), ..., order(n) so that results are
  !                written to value(:, :, order(ib)) in the original grid order.
  ! sample(s,ib) : pre-drawn i.i.d. N(0,1) value for realization s at block ib.
  !                The simulation draw is: z_sim = z_est + sqrt(var) * sample.
  !
  ! rangescale(ib)   : multiplies all variogram ranges at block ib.  Useful
  !                    when pilot-point density varies spatially; a scale < 1
  !                    tightens the search ellipsoid in data-sparse areas.
  ! localnugget(ib)  : extra nugget added to the diagonal of the kriging matrix
  !                    at block ib.  Represents unresolved sub-block variability
  !                    or stabilises near-singular systems.
  !============================================================================
  type, extends(t_data) :: t_blockgrid
    integer              :: block_type = 0     ! 0=point, -4=GQ, >0=user nodes
    integer, allocatable :: order(:)           ! visit order              [n]
    integer, allocatable :: nblockpnt(:)       ! nodes per block          [n]
    integer, allocatable :: iblockpnt(:)       ! start index in grid      [n]
    real, allocatable    :: rangescale(:)      ! variogram range scaler   [n]
    real, allocatable    :: localnugget(:)     ! extra diagonal nugget    [n]
    real, allocatable    :: sample(:,:,:)      ! N(0,1) draws for SGSIM  [nsim, nvar, n]
  end type t_blockgrid

  !============================================================================
  ! t_obsgrid — observation dataset with search infrastructure
  !
  ! Extends t_data with the k-d tree (tree) used for nearest-neighbour search.
  ! rotmat holds the 3×3 anisotropy rotation+scale matrix used to project
  ! coordinates into the anisotropic distance metric before tree queries.
  ! maxdist is stored as the SQUARED distance to allow direct comparison with
  ! the squared distances returned by KDTREE2 without a sqrt per neighbour.
  !============================================================================
  type, extends(t_data) :: t_obsgrid
    integer              :: nmax = 0           ! max neighbours used per block
    real                 :: maxdist = verylarge ! max search radius
    real                 :: sk_mean = 0.0       ! global mean for SK
    real                 :: rotmat(3,3)         ! anisotropy rotation matrix
    logical              :: need_search = .false.      ! .true. if nmax < n
    logical              :: anisotropic_search = .false. ! search in rotated coords
    logical              :: set_search = .false. ! track if search has been set
    type(kdtree2), pointer :: tree => null()   ! k-d tree for fast NN search
  end type t_obsgrid

  !============================================================================
  ! Neighbour-group layout convention
  !
  ! Group arrays (nnear, inear, weight) use a sequential layout:
  !   Groups 1:nvar         = real observations, variable ig        (always present)
  !   Groups nvar+1:ngroups = previously simulated blocks, variable ig-nvar  (SGSIM only)
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
  ! Group type checks:  kvar > nvar  — .true. when kvar is a simulated-block group
  ! Variable index:     obs kvar → ivar = kvar;  sim kvar → ivar = kvar - nvar
  !============================================================================

  !============================================================================
  ! t_weight_store — in-memory storage of per-block kriging weights
  !
  ! Allocated on demand by alloc_weight_store() before solve(), then filled
  ! block-by-block during solve().  Retrieve results via the CAPI get functions.
  !
  ! Array layout (Fortran column-major):
  !   nnear (ngroups, nblock)         — neighbour count per group per block
  !   inear (nmax,   ngroups, nblock) — neighbour obs/block indices (0 = unused)
  !   weight(nmax,   ngroups, nblock) — primary-variable kriging weights
  !   x(q, matsize_max, nblock)       — full solved RHS matrix for joint SGSIM
  !
  ! nmax is the maximum nmax across all obs variables (set by set_search).
  !============================================================================
  type :: t_weight_store
    integer              :: nblock  = 0
    integer              :: ngroups = 0
    integer              :: nmax    = 0
    integer, allocatable :: order (:)        ! [nblock]
    integer, allocatable :: nnear (:,   :)   ! [ngroups, nblock]
    integer, allocatable :: inear (:,:, :)   ! [nmax,    ngroups, nblock]
    real,    allocatable :: weight(:,:, :,:) ! [nmax,    ngroups, nvar, nblock]
    real,    allocatable :: var   (:,:, :)   ! [nvar,    nvar,          nblock] — conditional covariance
  contains
    ! add subroutine check if weights have been stored

    procedure :: stored     => wstore_check_stored
  end type t_weight_store

  !============================================================================
  ! t_kriging — main kriging object
  !
  ! Holds all problem-level state and provides the full API.  Thread-local
  ! working arrays are NOT stored here; they live in t_kriging_ctx so that
  ! multiple threads can work on different blocks simultaneously.
  !============================================================================
  type :: t_kriging
    !-- Boolean flags controlling solver behaviour
    logical              :: anisotropic_search = .false. ! use rotated coords for NN search
    logical              :: weight_correction  = .false. ! clip negative weights to 0 and renorm
    logical              :: use_old_weight     = .false. ! read weights from factor file
    logical              :: store_weight       = .false. ! save weights in memory (and optionally to weight_file)
    logical              :: cross_validation   = .false. ! leave-one-out cross-validation mode
    logical              :: write_mat          = .false. ! dump matrices to CSV for debugging
    logical              :: verbose            = .false. ! print progress to stdout
    logical              :: neglect_error      = .false. ! set NaN instead of stopping on singular
    logical              :: varying_vgm        = .false. ! use different vgm per block

    !-- File path for factor file (weight storage/reload)
    character(len=1024)  :: weight_file = ""
    integer              :: ifile = 0             ! Fortran unit for weight file

    !-- Problem dimensions
    integer              :: ndim    = 2           ! spatial dimension (1, 2, or 3)
    integer              :: nvar    = 1           ! number of co-kriging variables, default is 1 for ordinary/simple kriging
    integer              :: ngroups = 0           ! total neighbour groups (nvar obs + nvar sim when nsim>0)
    integer              :: ndrift  = 0           ! number of drift functions
    integer              :: unbias  = 1           ! 1=ordinary kriging, 0=simple kriging
    integer              :: nsim    = 0           ! simulations per block (0=kriging only)

    !-- Scratch / bookkeeping
    integer              :: nppmax = 0            ! max total neighbours across all variables
    integer              :: matsize_max = 0       ! nppmax + ndrift + unbias

    !-- Bounds for simulated/estimated values
    real                 :: bounds(2) = [-verylarge, verylarge]

    character(kind=c_char), pointer  :: krige_info(:) => null() ! kriging info string
    !-- Optional in-memory weight store (allocated by alloc_weight_store before solve)
    type(t_weight_store), allocatable :: wstore
    !-- Pointers to the three spatial objects and the variogram matrix
    type(t_obsgrid)  , pointer :: obs(:)     => null() ! observations  [1:nvar]
    type(t_grid)     , pointer :: grid       => null() ! integration nodes
    type(t_blockgrid), pointer :: block      => null() ! estimation targets
    type(vgm_struct) , pointer :: vgm(:,:,:) => null() ! variogram models [1:nvar, 1:nvar, 1]; last dim = nblock for spatially varying vgm
  contains
    procedure :: initialize
    procedure :: set_obs
    procedure :: update_obs_value
    procedure :: set_obs_drift
    procedure :: set_vgm
    procedure :: reset_vgm
    procedure :: set_grid
    procedure :: set_grid_drift
    procedure :: set_sim
    procedure :: set_search
    procedure :: search_neighbors
    procedure :: assemble_rhs
    procedure :: assemble_lhs
    procedure :: assemble_linear_system
    procedure :: solve_linear_system
    procedure :: calc_variance
    procedure :: estimate_block
    procedure :: prepare
    procedure :: solve
    procedure :: write_weight
    procedure :: write_weight_store
    procedure :: read_weight
    procedure :: validate_vgm
    procedure :: reset_obs
    procedure :: reset_grid
    procedure :: reset_block
    procedure :: alloc_weight_store
    procedure :: free_weight_store
    procedure :: save_block_weights
    procedure :: load_block_weights
    procedure :: set_weights
    procedure :: read_weight_to_store
    procedure :: finalize
    procedure :: to_str
    procedure :: update_info
  end type

  !============================================================================
  ! t_kriging_ctx — per-thread working context
  !
  ! In the OpenMP parallel block loop, each thread allocates its own ctx.
  ! This avoids false sharing and races on the working arrays.
  !
  ! Index convention for nnear / inear / weight
  ! -------------------------------------------
  ! These arrays are indexed 1:ngroups (see group layout convention above).
  !   Groups 1:nvar        = real observation neighbours for each variable.
  !   Groups nvar+1:ngroups = previously simulated block neighbours (SGSIM only).
  !============================================================================
  type :: t_kriging_ctx
    integer              :: iblock           ! current block index
    integer              :: npp              ! total neighbours = sum(nnear(1:ngroups))
    integer              :: matsize          ! npp + ndrift + unbias (actual, this block)
    integer, allocatable :: nnear(:)         ! neighbour count per variable [ngroups]
    integer, allocatable :: istart(:)        ! the starting index in the matrix for each variable; 0-based
    integer, allocatable :: inear(:,:)       ! neighbour indices            [nmax, ngroups]
    real,    allocatable :: weight(:,:,:)    ! kriging weights              [nmax, ngroups, nvar]
    real,    allocatable :: x(:,:)           ! raw solver output (weights + multipliers) [1, matsize]
    real,    allocatable :: matA(:,:)        ! covariance matrix C          [matsize, matsize]
    real,    allocatable :: rhsB(:,:)        ! right-hand-side c0           [1, matsize]
    ! ------------------------------------------------------------------
    ! Single-slot factorization cache.
    !
    ! The covariance matrix K depends only on the neighbour set (inear),
    ! the variogram model, rangescale, and localnugget — not on the block
    ! location.  When consecutive blocks processed by this thread share the
    ! same neighbour set, the Cholesky factorization of K can be reused:
    ! only the RHS c0 (which does depend on block location) is rebuilt.
    !
    ! factor_cache_hit   : set by factor_cache_matches; tells solve_linear_system
    !                      to skip kriging_setup and call kriging_solve_prepared
    !                      directly with the cached factors.
    ! factor_cache_valid : .true. once a successful kriging_setup has been stored.
    !                      Guards against using uninitialised factor arrays.
    ! factor_cache_rangescale  : rangescale value at the block whose factors are cached.
    ! factor_cache_localnugget : localnugget value at the block whose factors are cached.
    ! factor_cache_nnear : neighbour counts per group at the cached block    [1:ngroups].
    ! factor_cache_inear : sorted neighbour indices at the cached block      [nmax, 1:ngroups].
    !                      Stored sorted so the comparison is order-independent
    !                      (the KD-tree ranks by distance, which varies across blocks).
    ! factor_L           : Cholesky factor of K                              [nppmax, nppmax].
    ! factor_kinv_drift  : K^{-1} F (drift columns solved against K)         [nppmax, ndrift+unbias].
    ! factor_schur       : Cholesky factor of the Schur complement F^T K^-1 F [ndrift+unbias, ndrift+unbias].
    ! ------------------------------------------------------------------
    logical              :: factor_cache_hit   = .false.
    logical              :: factor_cache_valid = .false.
    real                 :: factor_cache_rangescale  = 1.0
    real                 :: factor_cache_localnugget = 0.0
    integer, allocatable :: factor_cache_nnear(:)
    integer, allocatable :: factor_cache_inear(:,:)
    real,    allocatable :: factor_L(:,:)
    real,    allocatable :: factor_kinv_drift(:,:)
    real,    allocatable :: factor_schur(:,:)
  contains
    procedure :: initialize  => initialize_kriging_ctx
    procedure :: factor_cache_matches   ! .true. if cached factors are valid for current block
    procedure :: save_factor_cache_key  ! snapshot current neighbour set and block scalars into cache
    procedure :: assign_weight   ! split x into per-variable weight arrays
    procedure :: write_matrix    ! dump matA, rhsB, data to CSV for debugging
  end type t_kriging_ctx

contains

  !============================================================================
  ! initialize
  !
  ! Sets all options and allocates the top-level pointer arrays.
  ! Must be called before any other method.
  !
  ! All arguments are optional; unset fields keep their type-default values.
  ! The only field that MUST be supplied is ndim (number of spatial dimensions).
  !
  ! Allocation layout after initialize():
  !   obs  (1:nvar)              one t_obsgrid per variable
  !   grid                       single t_grid  (populated by set_grid)
  !   block                      single t_blockgrid (populated by set_grid)
  !   vgm  (1:nvar, 1:nvar)      one vgm_struct per variable pair
  !   ngroups                    = nvar (kriging) or 2*nvar (SGSIM)
  !============================================================================
  subroutine initialize(self, ndim, nvar, ndrift, unbias, nsim, anisotropic_search,  &
                        weight_correction, use_old_weight, store_weight, cross_validation, &
                        write_mat, neglect_error, varying_vgm, verbose, &
                        weight_file, bounds, seed)
    class(t_kriging)                         :: self
    integer, intent(in), optional            :: ndim
    integer, intent(in), optional            :: nvar, ndrift, unbias, nsim, seed
    real,    intent(in), optional            :: bounds(2)
    logical, intent(in), optional            :: anisotropic_search, weight_correction, &
                                                use_old_weight, write_mat, store_weight, &
                                                verbose, cross_validation, neglect_error, varying_vgm
    character(len=*), intent(in), optional   :: weight_file
    character(len=*), parameter              :: subname = "t_kriging%initialize"


    !-- Transfer optional arguments to self
    if (present(ndim))               self%ndim               = ndim
    if (present(nvar))               self%nvar               = nvar
    if (present(ndrift))             self%ndrift             = ndrift
    if (present(unbias))             self%unbias             = unbias
    if (present(nsim))               self%nsim               = nsim
    if (present(anisotropic_search)) self%anisotropic_search = anisotropic_search
    if (present(weight_correction))  self%weight_correction  = weight_correction
    if (present(use_old_weight))     self%use_old_weight     = use_old_weight
    if (present(write_mat))          self%write_mat          = write_mat
    if (present(store_weight))       self%store_weight       = store_weight
    if (present(weight_file))        self%weight_file        = weight_file
    if (present(bounds))             self%bounds             = bounds
    if (present(cross_validation))   self%cross_validation   = cross_validation
    if (present(varying_vgm))        self%varying_vgm        = varying_vgm
    if (present(verbose))            self%verbose            = verbose
    if (present(neglect_error))      self%neglect_error      = neglect_error

    !-- Initialise random seed before allocation so the first draw is correct
    if (present(seed)) then
      if (self%verbose .and. self%nsim > 0) &
        print "(A,I0)", " Random seed is set to ", seed
      call random_seed_initialize(seed)
    end if

    !-- Build neighbour-group descriptors.
    !   Groups 1:nvar         = real obs for each variable (always present).
    !   Groups nvar+1:2*nvar  = previously simulated blocks (SGSIM only).
    if (self%nsim > 0) then
      self%ngroups = 2 * self%nvar
    else
      self%ngroups = self%nvar
    end if
    allocate(self%obs  (self%nvar))
    allocate(self%grid)
    allocate(self%block)
    !-- vgm is always indexed 1:nvar; vgm_real_idx() maps simulated-block ivar<=0
    !   to the correct positive index, removing the need for vgm(0,:) copies.
    if (.not. self%varying_vgm) &
      allocate(self%vgm(1:self%nvar, 1:self%nvar, 1))

    !-- Sanity checks: mutually exclusive flag combinations
    !   use_old_weight + weight_file=""  → in-memory wstore path (set_weights() must be
    !   called before solve() to populate the store); no error here.
    !   use_old_weight + weight_file!="" → read from the factor file.
    if (self%store_weight .and. self%use_old_weight) then
      call kriging_error(subname, 'store_weight and use_old_weight are mutually exclusive')
      return
    end if
    if (self%cross_validation .and. self%nsim > 0) then
      call kriging_error(subname, 'nsim>0 and cross_validation are mutually exclusive')
      return
    end if
  end subroutine initialize


  !============================================================================
  ! set_grid
  !
  ! Defines the estimation targets (blocks) and the associated integration
  ! points (grid nodes used to evaluate block-averaged covariances).
  !
  ! Three block types are supported, selected via block_type:
  !
  !   block_type = 0  (default) — point kriging
  !     One grid node per block.  block%coord = grid%coord = coord.
  !     nblockpnt = 1 everywhere, weight = 1.
  !
  !   block_type = -4 — block kriging with Gaussian quadrature discretisation
  !     Each block is discretised into 4^ndim integration points whose
  !     positions and weights are generated by Gaussian quadrature over the
  !     block volume defined by blocksize.
  !     block%coord holds the block centres (= coord); grid%coord holds all
  !     the integration points in block order.
  !
  !   block_type > 0 — block kriging with user-supplied integration nodes
  !     coord holds all integration points (total ngrid = sum(nblockpnt)).
  !     nblockpnt(:) gives the count per block; pointweight(:) gives the
  !     weights (default: equal weights within each block).
  !     block%coord is computed as the weight-averaged centroid of each block.
  !
  ! Cross-validation special case
  ! --------------------------------
  ! When cross_validation=.true., there is no separate grid; the blocks
  ! are the observation locations themselves.  nmax is incremented by 1 so
  ! the search returns nmax neighbours even after excluding the target node.
  !============================================================================
  subroutine set_grid(self, coord, block_type, blocksize, nblockpnt, pointweight, rangescale, localnugget)
    class(t_kriging)                       :: self
    integer, intent(in), optional          :: block_type
    real,    intent(in), optional          :: coord(:,:)      ! grid or block-centre coords [ndim, n]
    real,    intent(in), optional          :: blocksize(:,:)  ! block dimensions for GQ     [ndim, n]
    integer, intent(in), optional          :: nblockpnt(:)    ! nodes per block              [nblocks]
    real,    intent(in), optional          :: pointweight(:)  ! integration weights          [sum(nblockpnt)]
    real,    intent(in), optional          :: rangescale(:)   ! variogram range scaler       [nblocks]
    real,    intent(in), optional          :: localnugget(:)  ! per-block extra nugget       [nblocks]

    integer :: ngrid, nn, nb, iblock, igrid, idim, igq
    character(len=*), parameter :: subname = "t_kriging%set_grid"


    call self%reset_grid()
    call self%reset_block()

    if (self%obs(1)%n == 0) then
      call kriging_error(subname, 'Observation needs to be set first.')
      return
    end if

    associate(ndim => self%ndim, ndrift => self%ndrift)
      if (present(block_type)) self%block%block_type = block_type

      !------------------------------------------------------------------------
      ! Cross-validation: grid = observations; no coord argument needed
      !------------------------------------------------------------------------
      if (self%cross_validation) then
        ngrid = self%obs(1)%n
        self%block%n = ngrid
        allocate(self%block%coord(ndim, ngrid));  self%block%coord = self%obs(1)%coord
        allocate(self%grid%coord (ndim, ngrid));  self%grid%coord  = self%obs(1)%coord
        allocate(self%block%nblockpnt(ngrid));    self%block%nblockpnt = 1
        allocate(self%block%iblockpnt, source = [(igrid, igrid = 1, ngrid)])
        allocate(self%grid%weight(ngrid));         self%grid%weight = 1.0
        !-- +1 so the search still finds nmax neighbours after excluding self
        if (self%obs(1)%nmax>0) self%obs(1)%nmax = self%obs(1)%nmax + 1
        if (ndrift > 0) then
          allocate(self%block%drift(ndrift, ngrid))
          self%block%drift = self%obs(1)%drift
        end if

      else
        !----------------------------------------------------------------------
        ! Normal path: coord must be provided
        !----------------------------------------------------------------------
        if (.not. present(coord)) then
          call kriging_error(subname, 'coord needs to be provided.')
          return
        end if

        !-- validate ndim
        if (ndim /= size(coord, 1)) then
          call kriging_error(subname, 'ndim /= size(coord, 1) for self%grid')
          return
        end if
        ngrid = size(coord, 2)

        !-- block_type = 0: point kriging; one grid node per block
        if (self%block%block_type == 0) then
          self%block%n = ngrid
          allocate(self%block%coord, source = coord)
          allocate(self%grid%coord,  source = coord)
          allocate(self%block%nblockpnt(ngrid));  self%block%nblockpnt = 1
          allocate(self%block%iblockpnt, source = [(igrid, igrid = 1, ngrid)])
          allocate(self%grid%weight(ngrid));       self%grid%weight = 1.0

        !-- block_type = -4: GQ discretisation; blocksize required
        else if (self%block%block_type == -4) then
          if (.not. present(blocksize)) then
            call kriging_error(subname, 'blocksize needs to be provided when block_type=-4.')
            return
          end if
          if (size(blocksize, 1) /= ndim) then
            call kriging_error(subname, 'size(blocksize, 1) /= ndim when block_type=-4.')
            return
          end if
          if (size(blocksize, 2) /= ngrid) then
            call kriging_error(subname, 'size(blocksize, 2) /= nblock when block_type=-4.')
            return
          end if
          nb = 4**ndim        ! integration points per block (4-point Gauss per dimension)
          self%block%n = ngrid
          self%grid%n  = ngrid * nb
          allocate(self%block%coord, source = coord)
          allocate(self%grid%coord(ndim, self%grid%n))
          allocate(self%block%nblockpnt(ngrid));   self%block%nblockpnt = nb
          allocate(self%block%iblockpnt, source = [((igrid-1)*nb+1, igrid = 1, ngrid)])
          allocate(self%grid%weight(self%grid%n))
          igrid = 0
          do iblock = 1, self%block%n
            !-- Generate 4-point Gaussian quadrature nodes for this block.
            !   coord(:,iblock) is the block centre; gqdelxyz holds offsets
            !   from that centre, already scaled by blocksize(:,iblock).
            call set_gaussian_quadrature(ndim, blocksize(:, iblock))
            do igq = 1, nb
              self%grid%coord(:, igrid + igq) = coord(:, iblock) + gqdelxyz(:, igq)
            end do
            self%grid%weight(igrid+1:igrid+nb) = gqweight
            igrid = igrid + nb
          end do

        !-- block_type > 0: user-supplied integration nodes
        else
          self%grid%n   = ngrid
          self%block%n  = size(nblockpnt)
          allocate(self%grid%coord,    source = coord)
          allocate(self%block%nblockpnt, source = nblockpnt)
          allocate(self%block%iblockpnt(self%block%n))

          !-- Compute starting index of each block's nodes in the grid array
          igrid = 0
          do iblock = 1, self%block%n
            self%block%iblockpnt(iblock) = igrid + 1
            igrid = igrid + nblockpnt(iblock)
          end do

          !-- Integration weights: user-supplied or equal within each block
          if (present(pointweight)) then
            allocate(self%grid%weight, source = pointweight)
          else
            allocate(self%grid%weight(self%grid%n))
            igrid = 0
            do iblock = 1, self%block%n
              nb = nblockpnt(iblock)
              self%grid%weight(igrid+1:igrid+nb) = 1.0 / nb
              igrid = igrid + nb
            end do
          end if

          !-- Block centroid = weight-averaged grid node coordinate
          allocate(self%block%coord(ndim, self%block%n))
          igrid = 0
          do iblock = 1, self%block%n
            nn = nblockpnt(iblock)
            do idim = 1, ndim
              self%block%coord(idim, iblock) = &
                sum(self%grid%coord(idim, igrid+1:igrid+nn) * &
                    self%grid%weight(      igrid+1:igrid+nn)) &
                / sum(self%grid%weight(igrid+1:igrid+nn))
            end do
            igrid = igrid + nn
          end do
        end if
      end if ! cross_validation

      !-- Allocate block-level arrays common to all block types
      allocate(self%block%order      (self%block%n))
      allocate(self%block%localnugget(self%block%n))
      allocate(self%block%rangescale (self%block%n))
      allocate(self%block%value      (max(self%nsim, 1), self%nvar, self%block%n))
      allocate(self%block%variance   (self%nvar, self%nvar, self%block%n))

      !-- Default sequential visit order (overridden by set_sim for SGSIM)
      call set_seq(self%block%order, self%block%n)

      !-- Initialise outputs to NaN so unfilled blocks are detectable
      self%block%variance = IEEE_VALUE(0.0, IEEE_QUIET_NAN)
      self%block%value    = IEEE_VALUE(0.0, IEEE_QUIET_NAN)

      !-- Range scale and local nugget: user-supplied or defaults
      if (present(rangescale)  .and. .not. self%cross_validation) then
        self%block%rangescale  = rangescale
      else
        self%block%rangescale  = 1.0
      end if
      if (present(localnugget) .and. .not. self%cross_validation) then
        self%block%localnugget = localnugget
      else
        self%block%localnugget = 0.0
      end if
    end associate
    !-- vgm is always indexed 1:nvar
    if (self%varying_vgm) then
      if (associated(self%vgm)) deallocate(self%vgm)
      allocate(self%vgm(1:self%nvar, 1:self%nvar, self%block%n))
    else if (.not. associated(self%vgm)) then
      allocate(self%vgm(1:self%nvar, 1:self%nvar, 1))
    end if
  end subroutine set_grid


  !============================================================================
  ! set_grid_drift
  !
  ! Attach one drift-function value per block to block%drift.
  ! Called after set_grid() when ndrift > 0.  The drift array has shape
  ! (ndrift, nblock) — one scalar per drift function per block centre,
  ! NOT per integration node.
  !============================================================================
  subroutine set_grid_drift(self, drift)
    class(t_kriging)   :: self
    real, intent(in)   :: drift(:,:)   ! drift values [ndrift, nblock]
    character(len=*), parameter :: subname = "t_kriging%set_grid_drift"

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
    allocate(self%block%drift, source = drift)
  end subroutine set_grid_drift


  !============================================================================
  ! set_vgm
  !
  ! Add one nested variogram structure to the model for the variable pair
  ! (ivar, jvar).  Call once per nested structure (e.g. nugget + spherical
  ! requires two calls).  Only the upper triangle (jvar >= ivar) needs to be
  ! specified; the lower triangle is filled symmetrically.
  !
  !   vtype    : sph, exp, gau, pow, cir, hol, lin, or nug
  !   nugget   : nugget contribution of this structure
  !   sill     : partial sill
  !   a_major  : range along principal direction
  !   a_minor1 : range along first minor direction  (default: a_major)
  !   a_minor2 : range along second minor direction (default: a_minor1)
  !   azimuth, dip, plunge : rotation angles in degrees (default: 0)
  !   ib       : block index (default: all blocks); if ib is not present,
  !               the structure is applied to all blocks
  !============================================================================
  subroutine set_vgm(self, ivar, jvar, vtype, nugget, sill, a_major, a_minor1, a_minor2, azimuth, dip, plunge, ib)
    class(t_kriging), intent(inout)    :: self
    integer,          intent(in)       :: ivar, jvar
    integer, optional,intent(in)       :: ib
    character(*), optional, intent(in) :: vtype
    real,         optional, intent(in) :: nugget, sill, a_major, a_minor1, a_minor2
    real,         optional, intent(in) :: azimuth, dip, plunge
    ! local
    character(len=3) :: vtype_
    real             :: nugget_, sill_, a_major_, a_minor1_, a_minor2_, azimuth_, dip_, plunge_
    integer          :: ib_, mb, ib0
    character(len=*), parameter :: subname = "t_kriging%set_vgm"
    if (.not. associated(self%block)) then
      call kriging_error(subname, 'Call initialize() before set_vgm.')
      return
    end if
    if (self%varying_vgm .and. self%block%n==0) then
      call kriging_error(subname, 'Grid needs to be set before adding variogram under varying_vgm mode.')
      return
    end if
    if (.not. associated(self%vgm)) then
      call kriging_error(subname, 'Variogram storage is not allocated. Call initialize() first.')
      return
    end if
    vtype_    = 'sph'    ; if (present(vtype   )) vtype_ = vtype
    nugget_   = 0.0      ; if (present(nugget  )) nugget_ = nugget
    sill_     = 1.0      ; if (present(sill    )) sill_ = sill
    a_major_  = 1.0      ; if (present(a_major )) a_major_ = a_major
    a_minor1_ = a_major_ ; if (present(a_minor1)) a_minor1_ = a_minor1
    a_minor2_ = a_minor1_; if (present(a_minor2)) a_minor2_ = a_minor2
    azimuth_  = 0.0      ; if (present(azimuth )) azimuth_ = azimuth
    dip_      = 0.0      ; if (present(dip     )) dip_ = dip
    plunge_   = 0.0      ; if (present(plunge  )) plunge_ = plunge

    if (present(ib)) then
      ib0 = ib
      mb = ib
    else
      ib0 = 1
      ! -- If block index is not present, the structure is applied to all blocks
      if (self%varying_vgm) then
        mb = self%block%n
      else
        mb = 1
      end if
    end if

    do ib_ = ib0, mb
      if (jvar == ivar) then
        call self%vgm(jvar, ivar, ib_)%add_args(trim(vtype_), nugget_, sill_, a_major_, a_minor1_, a_minor2_, azimuth_, dip_, plunge_)
        if (kriging_failed()) return
      else if (jvar > ivar) then
        !-- Fill both triangle entries (cross-variogram is symmetric)
        call self%vgm(jvar, ivar, ib_)%add_args(trim(vtype_), nugget_, sill_, a_major_, a_minor1_, a_minor2_, azimuth_, dip_, plunge_)
        if (kriging_failed()) return
        call self%vgm(ivar, jvar, ib_)%add_args(trim(vtype_), nugget_, sill_, a_major_, a_minor1_, a_minor2_, azimuth_, dip_, plunge_)
        if (kriging_failed()) return
      else
        call kriging_error(subname, 'jvar must be >= ivar to set the upper triangle of the variogram matrix')
        return
      end if
    end do
  end subroutine set_vgm


  !============================================================================
  ! reset_vgm
  !
  ! Clear all nested structures for the (ivar, jvar) pair across every block.
  ! After this call, set_vgm may be used to build a fresh model for the pair.
  !============================================================================
  subroutine reset_vgm(self, ivar, jvar)
    class(t_kriging), intent(inout) :: self
    integer,          intent(in)    :: ivar, jvar
    integer :: ib_, mb
    if (.not. associated(self%vgm)) return
    mb = merge(self%block%n, 1, self%varying_vgm)
    do ib_ = 1, mb
      call self%vgm(jvar, ivar, ib_)%reset()
      if (ivar /= jvar) call self%vgm(ivar, jvar, ib_)%reset()
    end do
  end subroutine reset_vgm


  !============================================================================
  ! set_obs
  !
  ! Load observations for variable ivar.
  !
  ! coord  : (ndim, n) — spatial coordinates of observations
  ! value  : (n)       — observed values
  ! variance: (n)      — per-observation error variance (optional, default 0).
  !           Added to the diagonal of the kriging matrix: C_ii += variance_i.
  !           Use for heteroscedastic measurement error.
  ! nmax   : maximum number of neighbours per kriging solve (optional; if
  !           omitted all observations are used as neighbours).
  ! maxdist: maximum search radius.  Stored as maxdist^2 internally for
  !           efficient comparison with squared distances from KDTREE2.
  ! sk_mean: global mean for simgle kriging (unbias=0).  Optional, default 0.
  !============================================================================
  subroutine set_obs(self, ivar, coord, value, variance, nmax, maxdist, sk_mean)
    use rotation,       only: rotate
    use kdtree2_module, only: kdtree2_create
    class(t_kriging)              :: self
    integer, intent(in)           :: ivar
    integer, intent(in), optional :: nmax
    real,    intent(in)           :: coord(:,:), value(:)
    real,    intent(in), optional :: variance(:), maxdist, sk_mean
    character(len=*), parameter   :: subname = "t_kriging%set_obs"
    ! local
    integer                       :: i
    call self%reset_obs(ivar)

    associate(ndim => self%ndim, obs => self%obs(ivar))
      !-- Infer or validate ndim from coord
      if (ndim == 0) then
        ndim = size(coord, 1)
      else
        if (ndim /= size(coord, 1)) then
          call kriging_error(subname, 'ndim /= size(coord, 1) for grid')
          return
        end if
      end if
      obs%n = size(coord, 2)

      !-- nmax: cap at obs%n in set_search; here just record the user request
      if (present(nmax)) then
        obs%nmax = nmax
      else
        obs%nmax = huge(obs%n)   ! effectively "use all"
      end if

      !-- maxdist stored squared; KDTREE2 returns squared distances
      if (present(maxdist)) obs%maxdist = maxdist**2

      !-- global mean for simple kriging
      if (present(sk_mean)) obs%sk_mean = sk_mean

      !-- Observation error variance: default to 0 (exact observations)
      allocate(obs%variance(1,1,obs%n))
      if (present(variance)) then
        do i = 1, obs%n
          obs%variance(1,1,i) = variance(i)
        end do
      else
        obs%variance = 0.0
      end if

      allocate(obs%value(1,1,obs%n))
      obs%value(1,1,:) = value
      allocate(obs%coord, source = coord)

      !-- Identity rotation matrix (updated by set_search for anisotropic search)
      obs%rotmat = reshape([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], [3, 3])
    end associate
  end subroutine set_obs

  ! ===========================================================================
  ! update_obs_value
  !
  ! Update observation values for variable ivar; useful for weight reuse.
  ! assume that set_obs has already been called for ivar
  ! if weights have already been computed, switch to use_old_weight mode.
  ! ===========================================================================
  subroutine update_obs_value(self, ivar, value)
    class(t_kriging)              :: self
    integer, intent(in)           :: ivar
    real,    intent(in)           :: value(:)
    character(len=*), parameter   :: subname = "t_kriging%update_obs_value"
    if (self%obs(ivar)%n == 0) then
      call kriging_error(subname, 'Call set_obs() before update_obs_value.')
      return
    end if
    if (size(value) /= self%obs(ivar)%n) then
      call kriging_error(subname, 'size(value) /= self%obs(ivar)%n')
      return
    end if
    self%obs(ivar)%value(1,1,:) = value
    ! turn on weight reuse if weights have already been computed
    if (allocated(self%wstore)) then
      if (self%wstore%stored() .and. .not. all(ieee_is_nan(self%block%value))) then
        self%use_old_weight = .true.
        self%store_weight   = .false.
      end if
    end if
  end subroutine update_obs_value


  !============================================================================
  ! set_obs_drift
  !
  ! Attach drift-function values to observation set ivar.
  ! Must be called after set_obs(ivar).  drift has shape (ndrift, nobs).
  !============================================================================
  subroutine set_obs_drift(self, ivar, drift)
    class(t_kriging)   :: self
    integer, intent(in) :: ivar
    real,    intent(in) :: drift(:,:)   ! [ndrift, nobs]
    character(len=*), parameter :: subname = "t_kriging%set_obs_drift"

    if (.not. associated(self%obs)) then
      call kriging_error(subname, 'Call initialize() before set_obs_drift.')
      return
    end if
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
    allocate(self%obs(ivar)%drift, source = drift)
  end subroutine set_obs_drift


  !============================================================================
  ! set_sim
  !
  ! Prepare the SGSIM random path and pre-drawn N(0,1) samples.
  ! Must be called after set_grid() and set_obs() but before set_search().
  !
  ! Random path
  ! -----------
  ! block%order is filled with a random permutation of 1..nblock.  Each
  ! block is visited once in this order during solve().  If randpath is
  ! supplied, that permutation is used directly; otherwise a new one is
  ! drawn from the current random state (seeded in initialize if seed was
  ! passed).
  !
  ! Standard-normal samples
  ! -----------------------
  ! block%sample(isim, ivar, ib) holds the pre-drawn N(0,1) value for simulation
  ! isim variable ivar at block ib. The simulation draw at block ib for realization isim is:
  !   z_sim = z_kriging_estimate + sqrt(kriging_variance) * sample(isim, ivar, ib)
  ! Using pre-drawn samples ensures reproducibility and allows Workflow 2
  ! (nsim > 1) to reuse the same sample matrix for multiple realizations.
  !
  ! Coordinate extension for SGSIM neighbour search
  ! ------------------------------------------------
  ! obs(1)%coord is extended to size (ndim, nobs + nblock) by appending the
  ! block centres.  This allows the KDTREE2 search to find previously
  ! simulated blocks as neighbours using a single tree query.  The max_idx
  ! filter in kdtree2_n_nearest_maxidx ensures only blocks with index < ib
  ! (already simulated) are returned.
  !============================================================================
  subroutine set_sim(self, randpath, sample)
    class(t_kriging)              :: self
    real,    intent(in), optional :: sample(:,:,:) ! pre-drawn N(0,1) [nsim, nvar, nblock]
    integer, intent(in), optional :: randpath(:)   ! user-supplied visit order [nblock]

    real,    allocatable :: temp(:,:), samp(:,:,:)
    integer              :: iblock, ivar, ifile, isim
    character(len=*), parameter :: subname = "t_kriging%set_sim"

    if (self%nsim > 0) then
      if (self%block%n == 0) then
        call kriging_error(subname, 'Grid needs to be set first.')
        return
      end if
      if (any(self%obs%n == 0)) then
        call kriging_error(subname, 'Observations need to be set first.')
        return
      end if
      associate(ndim => self%ndim, obs => self%obs(1))
        !-- Random visit path
        if (present(randpath)) then
          self%block%order = randpath
        else
          call set_seq(self%block%order, self%block%n, .TRUE.)   ! .TRUE. = shuffle
          open(newunit=ifile, file='sgs_path.dat', status='replace')
          write(ifile, '(A,x,I0)') 'SGSIM_Path', self%block%n
          write(ifile, '((1I0))') self%block%order
          close(ifile)
        end if

        !-- Standard-normal samples for the simulation draws
        if (allocated(self%block%sample)) deallocate(self%block%sample)
        allocate(self%block%sample(self%nsim, self%nvar, self%block%n))
        if (present(sample)) then
          ! check sample size
          if (size(sample, 1) /= self%nsim) then
            call kriging_error(subname, 'size(sample, 1) /= nsim')
            return
          end if
          if (size(sample, 2) /= self%nvar) then
            call kriging_error(subname, 'size(sample, 2) /= nvar')
            return
          end if
          if (size(sample, 3) /= self%block%n) then
            call kriging_error(subname, 'size(sample, 3) /= nblock')
            return
          end if
          self%block%sample = sample
        else
          allocate(samp(self%block%n,1,1))
          do isim = 1, self%nsim
            do ivar = 1, self%nvar
              call r8vec_normal_01(self%block%n, samp(:,1,1))
              do iblock = 1, self%block%n
                self%block%sample(isim, ivar, :) = samp(iblock,1,1)
              end do
            end do
          end do
          !-- Write samples to file for reproducibility
          open(newunit=ifile, file='sgs_sample.dat', status='replace')
          write(ifile, '(A,x,2I10)') 'SGSIM_Sample', self%nsim, self%nvar, self%block%n
          do iblock = 1, self%block%n
            write(ifile, '(*(G0.7,x))') self%block%sample(:,:,iblock)
          end do
          close(ifile)
        end if

        !-- Reorder block arrays into random-path order so the solve loop
        !   processes them sequentially without random-access overhead
        self%block%coord      = self%block%coord     (:, self%block%order)
        self%block%iblockpnt  = self%block%iblockpnt (   self%block%order)
        self%block%nblockpnt  = self%block%nblockpnt (   self%block%order)
        self%block%rangescale = self%block%rangescale(   self%block%order)
        self%block%localnugget= self%block%localnugget(  self%block%order)
        if (self%ndrift > 0) self%block%drift = self%block%drift(:, self%block%order)

        !-- Extend obs(k)%coord for ALL variables to include all block centres so
        !   the k-d tree can return previously simulated blocks of any variable as
        !   neighbours.  After this call, obs(k)%coord has size (ndim, nobs_k + nblock).
        !   obs(k)%n is NOT changed; it still equals the original observation count.
        !   During search_neighbors, the max_idx filter restricts results to
        !   indices <= nobs_k + ib - 1 (i.e. only already-simulated blocks).
        do ivar = 1, self%nvar
          associate(obsk => self%obs(ivar))
            allocate(temp(ndim, obsk%n + self%block%n))
            temp(:, 1:obsk%n) = obsk%coord
            temp(:, obsk%n+1:) = self%block%coord
            call move_alloc(temp, obsk%coord)
          end associate
        end do
      end associate
    end if
  end subroutine set_sim


  !============================================================================
  ! set_search
  !
  ! Build the KDTREE2 nearest-neighbour tree for variable ivar.
  ! Must be called after set_obs (and after set_sim for ivar=1 in SGSIM).
  !
  ! Anisotropic search
  ! ------------------
  ! If anisotropic_search=.true. and the variogram has anisotropy (anis1 or
  ! anis2 /= 1), obs%coord is projected into the anisotropically scaled
  ! coordinate system before tree construction.  Distances in this system
  ! correspond to the anisotropic variogram metric, so neighbours with the
  ! highest spatial correlation are returned rather than nearest Euclidean
  ! neighbours.
  !
  ! If all observations fit within nmax, need_search=.false. and no tree is
  ! built; distances are computed directly in search_neighbors.
  !============================================================================
  subroutine set_search(self, ivar, anis1, anis2, azimuth, dip, plunge)
    use rotation,       only: calc_rotmat, sub_rotate
    use kdtree2_module, only: kdtree2_create
    class(t_kriging)   :: self
    integer, intent(in) :: ivar
    real,    intent(in) :: anis1, anis2, azimuth, dip, plunge
    character(len=*), parameter :: subname = "t_kriging%set_search"

    real, allocatable :: rcoord(:,:)   ! rotated coordinates for anisotropic tree
    if (self%obs(ivar)%n == 0) then
      call kriging_error(subname, 'set_obs() needs to be called before set_search().')
      return
    end if
    if (self%nsim > 0) then
      if (self%block%n == 0) then
        call kriging_error(subname, 'set_grid() needs to be called before set_search().')
        return
      end if
      if (size(self%obs(ivar)%coord, 2) == self%obs(ivar)%n) then
        call kriging_error(subname, 'set_sim() needs to be called before set_search().')
        return
      end if
    end if
    associate( &
      ndim               => self%ndim, &
      obs                => self%obs(ivar), &
      need_search        => self%obs(ivar)%need_search, &
      anisotropic_search => self%obs(ivar)%anisotropic_search)

      !-- Activate anisotropic search only when there is meaningful anisotropy
      anisotropic_search = (abs(anis1 - 1.0) > EPSLON .or. abs(anis2 - 1.0) > EPSLON) &
                           .and. self%anisotropic_search

      !-- Determine effective nmax, accounting for SGSIM's extended obs array.
      !   For all variables in SGSIM/joint co-sim, the tree is built on the
      !   extended coord array (obs + block centres); nmax spans both.
      if (self%nsim > 0) then
        obs%nmax = min(obs%nmax, obs%n + self%block%n)
        if (obs%nmax <= 0) obs%nmax = obs%n + self%block%n
        need_search = obs%n + self%block%n > obs%nmax
      else
        obs%nmax = min(obs%nmax, obs%n)
        if (obs%nmax <= 0) obs%nmax = obs%n
        need_search = obs%n > obs%nmax
      end if

      !-- Build k-d tree only when a subset search is needed
      if (need_search) then
        if (anisotropic_search) then
          !-- compute 3×3 rotation+scale matrix from variogram angles
          obs%rotmat = calc_rotmat(azimuth, dip, plunge, anis1, anis2)
          !-- Project coordinates into anisotropically scaled space before indexing
          allocate(rcoord, mold = obs%coord)
          call sub_rotate(obs%rotmat, ndim, size(obs%coord, 2), obs%coord, rcoord)
          obs%tree => kdtree2_create(rcoord, sort = .false., rearrange = .true.)
          if (kriging_failed()) return
        else
          obs%tree => kdtree2_create(obs%coord, sort = .false., rearrange = .true.)
          if (kriging_failed()) return
        end if
      end if
    end associate
    self%obs(ivar)%set_search = .true.
  end subroutine set_search


  !============================================================================
  ! initialize_kriging_ctx
  !
  ! Allocate per-thread working arrays for a kriging context.
  ! Called once per thread at the start of the parallel region in solve().
  !
  ! Array sizes are set to the worst-case maxima (nppmax, matsize_max) so
  ! that no reallocation is needed during the block loop.
  !============================================================================
  subroutine initialize_kriging_ctx(self, krige)
    class(t_kriging_ctx) :: self
    class(t_kriging)     :: krige

    integer :: ivar, kvar, mmax, pmax
    logical :: need_rhs
    mmax = maxval(krige%obs%nmax)   ! max neighbours across all variables
    pmax = krige%ndrift + krige%unbias

    associate(npp => krige%nppmax, matsize => krige%matsize_max, &
              ng  => krige%ngroups, nv => krige%nvar)

      if (.not. krige%use_old_weight) then
        allocate(self%istart(ng))
        allocate(self%matA(matsize, matsize))
        allocate(self%rhsB(nv, matsize))
        allocate(self%factor_cache_nnear(ng))
        allocate(self%factor_cache_inear(mmax, ng))
        allocate(self%factor_L(npp, npp))
        allocate(self%factor_kinv_drift(npp, max(1, pmax)))
        allocate(self%factor_schur(max(1, pmax), max(1, pmax)))
        self%factor_cache_nnear = 0
        self%factor_cache_inear = 0
      end if
      allocate(self%nnear (ng))
      allocate(self%inear (mmax, ng))
      allocate(self%weight(mmax, ng, nv))
      allocate(self%x     (nv, matsize))
      self%weight = 0.0
      self%x      = 0.0

      !-- Obs groups (1:nvar): start with all obs as candidate neighbours.
      !   Sim groups (nvar+1:ngroups): start empty; filled by search_neighbors.
      call set_seq(self%inear(1:mmax, 1), mmax)
      do ivar = 1, nv
        self%nnear(ivar)    = krige%obs(ivar)%nmax
        self%inear(:, ivar) = self%inear(:, 1)
      end do
      do kvar = nv + 1, ng   ! sim groups only
        self%nnear(kvar) = 0
      end do
    end associate
  end subroutine initialize_kriging_ctx


  !============================================================================
  ! factor_cache_matches
  !
  ! Returns .true. when the stored Cholesky factorization can be reused for
  ! the current block.  The factorization of K depends on:
  !   - the neighbour set (inear, nnear per variable)
  !   - the variogram model (skipped entirely when varying_vgm=.true.)
  !   - rangescale and localnugget (both affect K values)
  !
  ! inear is compared after sorting (see search_neighbors), so the result is
  ! order-independent: two blocks that share the same set of neighbours but
  ! receive them in different distance-rank order from the KD-tree still match.
  !
  ! varying_vgm is short-circuited at the top: each block has its own
  ! variogram model, so K can never be reused across blocks.
  !============================================================================
  logical function factor_cache_matches(self, krige)
    class(t_kriging_ctx) :: self
    class(t_kriging)     :: krige

    integer :: kvar

    factor_cache_matches = .false.

    if (.not. self%factor_cache_valid) return
    if (krige%varying_vgm) return
    if (self%factor_cache_rangescale /= krige%block%rangescale(self%iblock)) return
    if (self%factor_cache_localnugget /= krige%block%localnugget(self%iblock)) return

    do kvar = 1, krige%ngroups
      if (self%factor_cache_nnear(kvar) /= self%nnear(kvar)) return
      if (self%nnear(kvar) > 0) then
        if (any(self%factor_cache_inear(1:self%nnear(kvar), kvar) /= &
                self%inear(1:self%nnear(kvar), kvar))) return
      end if
    end do

    factor_cache_matches = .true.
  end function factor_cache_matches


  !============================================================================
  ! save_factor_cache_key
  !
  ! Snapshot the current block's neighbour set and block-level scalars into
  ! the cache fields so that factor_cache_matches can detect a hit on the
  ! next block.  Called by solve_linear_system immediately after a successful
  ! kriging_setup (Cholesky factorization), so the cached key always
  ! corresponds to the factors stored in factor_L / factor_kinv_drift /
  ! factor_schur.
  !============================================================================
  subroutine save_factor_cache_key(self, krige)
    class(t_kriging_ctx) :: self
    class(t_kriging)     :: krige

    integer :: kvar

    self%factor_cache_rangescale  = krige%block%rangescale (self%iblock)
    self%factor_cache_localnugget = krige%block%localnugget(self%iblock)
    self%factor_cache_nnear(1:krige%ngroups) = self%nnear(1:krige%ngroups)

    do kvar = 1, krige%ngroups
      if (self%nnear(kvar) > 0) then
        self%factor_cache_inear(1:self%nnear(kvar), kvar) = &
        self%inear(1:self%nnear(kvar), kvar)
      end if
    end do
    self%factor_cache_valid = .true.
  end subroutine save_factor_cache_key


  !============================================================================
  ! prepare
  !
  ! Pre-solve validation and bookkeeping called at the start of solve().
  !
  ! Computes nppmax (max total neighbours summed across all variables) and
  ! matsize_max (nppmax + ndrift + unbias) which dimension the thread-private
  ! working arrays in t_kriging_ctx.
  !
  ! SGSIM: copies vgm(1,1) into vgm(0,0) and vgm(0,ivar) / vgm(ivar,0) so
  ! that the covariance between a target block and its previously simulated
  ! neighbours (index 0) uses the primary variogram model.
  !
  ! Weight file: opens for reading when use_old_weight=.true.
  ! store_weight=.true.: auto-allocates wstore; file is written after solve().
  !============================================================================
  subroutine prepare(self)
    class(t_kriging) :: self
    ! local
    integer          :: ivar, jvar, ib, mb
    integer          :: hdr_nblock, hdr_nvar, ios
    integer          :: nmax(self%nvar)
    character(len=*), parameter :: subname = "t_kriging%prepare"


    if (.not. self%use_old_weight) then
      !-- Validate that all required arrays have been provided
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

    !-- vgm(0,:) propagation removed: vgm_real_idx() maps every ivar<=0 to a
    !   valid positive index, so no copies are needed.
      call self%validate_vgm()
      if (kriging_failed()) return
      do ivar = 1, self%nvar
        if (.not. self%obs(ivar)%set_search) then
          call kriging_error(subname, 'set_search() needs to be called before solve().')
          return
        end if
      end do
    end if

    associate(npp => self%nppmax, matsize => self%matsize_max, ifile => self%ifile)

      !-- Total neighbours and matrix size for worst-case allocation
      npp = 0
      do ivar = 1, self%nvar
        npp = npp + self%obs(ivar)%nmax
      end do
      matsize = npp + self%ndrift + self%unbias

      !-- Weight reuse: pre-load all blocks from the factor file into wstore so the
      !   parallel block loop can use load_block_weights without any file I/O.
      if (self%use_old_weight) then
        if (trim(self%weight_file) /= "") then
          open(newunit=ifile, file=trim(self%weight_file), status='old')
          read(ifile, *, iostat=ios) hdr_nblock, hdr_nvar, nmax
          if (ios /= 0) then
            call kriging_error(subname, 'Failed to read weight_file header.')
            close(ifile); return
          end if
          if (hdr_nblock /= self%block%n .or. hdr_nvar /= self%nvar .or. any(nmax /= self%obs%nmax)) then
            call kriging_error(subname, 'weight_file dimensions do not match this kriging object.')
            close(ifile); return
          end if
          call self%alloc_weight_store()
          if (kriging_failed()) then; close(ifile); return; end if
          do ib = 1, self%block%n
            call self%read_weight_to_store(ifile, ib)
            if (kriging_failed()) then; close(ifile); return; end if
          end do
          close(ifile)
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

    end associate
  end subroutine prepare


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
  subroutine solve(self, nthread)
    use omp_lib
    class(t_kriging)          :: self
    integer, intent(in), optional :: nthread  ! max OMP threads for this call (0 or absent = OMP default)
    type(t_kriging_ctx), allocatable :: ctx
    integer                   :: ib, prev_nthread, nthread_local
    real, allocatable          :: temp(:,:)
    character(len=*), parameter :: subname = "t_kriging%solve"

    !-- Temporarily override the OMP thread count if nthread was supplied.
    !   Save and restore so that callers sharing a runtime are not affected.
    prev_nthread = 0
    nthread_local = 0
#ifdef _OPENMP
    if (present(nthread) .and. nthread > 0) then
      prev_nthread = omp_get_max_threads()
      if (nthread>0) then
        call omp_set_num_threads(nthread)
        nthread_local = nthread
      end if
    else
      nthread_local = omp_get_max_threads()
    end if
#endif

    call self%prepare()
    if (kriging_failed()) return

    associate(nb => self%block%n, verbose => self%verbose)
      if (verbose) print*, "Starting Kriging loop"
#ifdef __INTEL_COMPILER
      if (verbose) open(unit=6, carriagecontrol='fortran')
#endif

      !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(ctx) IF(self%nsim==0 .and. nthread_local>1)
      allocate(ctx)                        ! each thread gets its own ctx
      call ctx%initialize(self)

      !$OMP DO SCHEDULE(DYNAMIC, 1)
      do ib = 1, nb
        if (kriging_failed()) cycle
        !-- Progress bar: only the last thread prints to avoid interleaved output
#ifdef _OPENMP
        if (verbose .and. omp_get_thread_num() == omp_get_num_threads()-1) call progress(ib, nb)
#else
        if (verbose) call progress(ib, nb)
#endif
        ctx%iblock = ib

        if (self%use_old_weight) then
          !-- load weights from memory.
          call self%load_block_weights(ctx)
        else
          call self%assemble_linear_system(ctx)
          if (kriging_failed()) cycle
          !-- Skip the matrix solve when only one neighbour exists (trivial case)
          if (ctx%npp > 1) call self%solve_linear_system(ctx)
          if (kriging_failed()) cycle
          call ctx%assign_weight(self)
        end if

        if (self%store_weight) call self%save_block_weights(ctx)
        call self%estimate_block(ctx)
        if (self%write_mat) then
          ! Files are named per block, but the Fortran runtime still shares
          ! newunit/open/write bookkeeping.  Serialize only the debug output.
          !$OMP CRITICAL(write_matrix_io)
          call ctx%write_matrix(self)
          !$OMP END CRITICAL(write_matrix_io)
        end if
      end do
      !$OMP END DO
      deallocate(ctx)     ! explicit per-thread cleanup; avoids crash on runtime auto-finalization of PRIVATE allocatables
      !$OMP END PARALLEL

      !-- Persist the in-memory weight store to disk when a path was given.
      if (self%store_weight .and. trim(self%weight_file) /= "") then
        call self%write_weight_store()
      end if

      if (kriging_failed()) return

#ifdef __INTEL_COMPILER
      if (verbose) close(6)
#else
      if (verbose) print *, ""   ! newline after progress bar
#endif
      if (verbose) print*, "Kriging completed."

      !-- SGSIM post-processing: blocks were processed in random path order;
      !   reorder coord and estimate back to the original block indices so
      !   downstream code can use block%value(isim, ivar, ib) at the correct location.
      if (self%nsim > 0) then
        allocate(temp(self%ndim, self%block%n))
        temp = self%block%coord
        block
          real, allocatable :: temp_est(:,:,:)
          allocate(temp_est, source = self%block%value)
          do ib = 1, self%block%n
            self%block%coord(:, self%block%order(ib)) = temp(:, ib)
            self%block%value(:, :, self%block%order(ib)) = temp_est(:, :, ib)
          end do
        end block
        block
          real, allocatable :: temp_var(:,:,:)
          allocate(temp_var, source = self%block%variance)
          do ib = 1, self%block%n
            self%block%variance(:, :, self%block%order(ib)) = temp_var(:, :, ib)
          end do
        end block
      end if
    end associate

    !-- Restore the original OMP thread count.
#ifdef _OPENMP
    if (prev_nthread > 0) call omp_set_num_threads(prev_nthread)
#endif
  end subroutine solve


  !============================================================================
  ! search_neighbors
  !
  ! Find the nearest observations (and, for SGSIM, previously simulated blocks)
  ! to the current block centre.  Results are stored in ctx%inear and ctx%nnear.
  !
  ! SGSIM path (ivar=1 .and. nsim>0)
  ! ----------------------------------
  ! obs(1)%coord holds nobs + nblock entries (the extension done in set_sim).
  ! kdtree2_n_nearest_maxidx returns at most nmax neighbours whose index is
  ! strictly less than nobs + iblock — i.e. only original observations
  ! (index <= nobs) and previously simulated blocks (nobs < index < nobs+iblock).
  ! The results are then partitioned into:
  !   inear(:, 1)  — original observation indices (1..nobs)
  !   inear(:, 0)  — simulated block indices, shifted to 1..iblock-1
  !
  ! When all obs + prior simulated blocks fit within nmax, the k-d tree
  ! query is skipped and distances are computed directly (rotated_dists).
  !
  ! After distance filtering: any neighbour beyond maxdist is dropped.
  !============================================================================
  subroutine search_neighbors(self, ivar, ctx)
    class(t_kriging)     :: self
    class(t_kriging_ctx) :: ctx
    integer, intent(in)  :: ivar

    integer                     :: i, k, ig_sim
    real                        :: newloc(self%ndim, 1)
    real                        :: dist (self%obs(ivar)%nmax)  ! squared distances, obs group
    real                        :: distb(self%obs(ivar)%nmax)  ! squared distances, sim group
    logical, allocatable        :: is_obs(:)
    type(kdtree2_result)        :: results(self%obs(ivar)%nmax)
    character(len=*), parameter :: subname = "t_kriging%search_neighbors"

    !-- Obs group index = ivar (groups 1:nvar map directly to obs variables 1:nvar).
    !-- Sim group index (only dereferenced when nsim > 0, so always in range):

    associate( &
      iblock  => ctx%iblock, &
      ndim    => self%ndim, &
      nsim    => self%nsim, &
      nobs    => self%obs(ivar)%n, &
      nmax    => self%obs(ivar)%nmax, &
      obsloc  => self%obs(ivar)%coord, &
      xloc    => self%block%coord(:, ctx%iblock:ctx%iblock), &
      inear   => ctx%inear(:, ivar), &
      nnear   => ctx%nnear(ivar), &
      maxdist => self%obs(ivar)%maxdist, &
      rotmat  => self%obs(ivar)%rotmat)

      !-- Project target location if anisotropic search is active
      if (self%obs(ivar)%anisotropic_search) then
        call sub_rotate(rotmat, ndim, 1, xloc, newloc)
      else
        newloc = xloc
      end if

      !------------------------------------------------------------------------
      ! SGSIM / joint co-sim neighbour search: obs + prior simulated blocks
      !------------------------------------------------------------------------
      if (nsim > 0) then
        ig_sim = self%nvar + ivar
        associate( &
          inearb => ctx%inear(:, ig_sim), &
          nnearb => ctx%nnear(ig_sim))

          if (nmax < nobs + iblock - 1) then
            !-- k-d tree query with max_idx filter: only returns entries < nobs+iblock
            call kdtree2_n_nearest_maxidx(self%obs(ivar)%tree, newloc(:,1), nmax, results, nobs+iblock-1)
            if (kriging_failed()) return
            allocate(is_obs, source = results%idx <= nobs)
            nnear              = count(is_obs)
            nnearb             = nmax - nnear
            inear (1:nnear)    = pack(results%idx, is_obs)
            inearb(1:nnearb)   = pack(results%idx, .not. is_obs) - nobs  ! shift to 1-based block index
            dist  (1:nnear)    = pack(results%dis, is_obs)
            distb (1:nnearb)   = pack(results%dis, .not. is_obs)
          else
            !-- All obs + prior blocks fit within nmax: compute distances directly
            nnear  = nobs
            nnearb = iblock - 1
            call set_seq(inear(1:nnear), nnear)
            if (nnearb > 0) call set_seq(inearb(1:nnearb), nnearb)
            dist (1:nnear)  = rotated_dists(rotmat, ndim, newloc(:,1), obsloc(:, 1:nnear))
            distb(1:nnearb) = rotated_dists(rotmat, ndim, newloc(:,1), self%block%coord(:, 1:nnearb))
          end if

        end associate  ! inearb, nnearb, distb

      !------------------------------------------------------------------------
      ! Standard kriging / cokriging neighbour search
      !------------------------------------------------------------------------
      else
        if (nmax < nobs) then
          call kdtree2_n_nearest(self%obs(ivar)%tree, newloc(:,1), nmax, results)
          if (kriging_failed()) return
          nnear          = nmax
          inear(1:nnear) = results%idx
          dist (1:nnear) = results%dis
        else
          !-- All observations fit: compute distances directly
          nnear = nobs
          call set_seq(inear(1:nnear), nnear)
          dist(1:nnear) = rotated_dists(rotmat, ndim, newloc(:,1), obsloc(:, 1:nnear))
        end if

        !-- Cross-validation: exclude the target observation from its own neighbourhood
        if (self%cross_validation) then
          do i = 1, nnear
            if (inear(i) == iblock) then
              nnear = nnear - 1
              inear(i:nnear) = inear(i+1:nnear+1)
              dist (i:nnear) = dist (i+1:nnear+1)
              exit
            end if
          end do
        end if
      end if

      !-- Drop any neighbour beyond the maximum search distance
      k = 0
      do i = 1, nnear
        if (dist(i) <= maxdist) then
          k = k + 1
          inear(k) = inear(i)
        end if
      end do
      nnear = k
      if (self%nsim > 0) then
        k = 0
        do i = 1, ctx%nnear(ig_sim)
          if (distb(i) <= maxdist) then
            k = k + 1
            ctx%inear(k, ig_sim) = ctx%inear(i, ig_sim)
          end if
        end do
        ctx%nnear(ig_sim) = k
      end if
    end associate
#ifdef DEBUG
    print *, subname, " Finished.", ivar, ctx%iblock
#endif
  end subroutine search_neighbors


  !============================================================================
  ! group_ivar
  !
  ! Map any group index ig (1:ngroups) to the real variable index (1:nvar).
  ! Obs groups (ig = 1:nvar) map to ig; sim groups (ig = nvar+1:2*nvar) map
  ! to ig - nvar.
  ! Declared elemental so it can be applied to arrays of group indices.
  !============================================================================
  pure elemental integer function group_ivar(ig, nvar)
    integer, intent(in) :: ig, nvar
    if (ig <= nvar) then
      group_ivar = ig
    else
      group_ivar = ig - nvar
    end if
  end function group_ivar


  !============================================================================
  ! assemble_rhs
  !
  ! Fill the right-hand-side covariance vector(s) rhsB for the current block.
  ! For joint co-sim (nvar > 1, nsim > 0), nvar columns are built — one per
  ! target variable ivar.  Otherwise size(rhsB,1) = 1 and the loop runs once.
  !
  ! For each target variable ivar and each neighbour group kvar:
  !   rhsB(ivar, irow+i) = sum_k weight(k) * C_ivar(obs_i, grid_k)
  ! where the sum is over block integration nodes (for block kriging;
  ! reduces to a single evaluation for point kriging with nblockpnt=1).
  !============================================================================
  subroutine assemble_rhs(self, ctx)
    class(t_kriging)     :: self
    class(t_kriging_ctx) :: ctx
    ! local
    integer                :: ivar, kvar, i, k, nn, jvar, ivgm
    real                   :: lag(3), tmp, rs
    class(t_data), pointer :: obs

    associate( &
      iblock  => ctx%iblock, &
      rhsB    => ctx%rhsB, &
      nnear   => ctx%nnear, &
      inear   => ctx%inear, &
      istart  => ctx%istart, &
      npp     => ctx%npp, &
      matsize => ctx%matsize, &
      ndrift  => self%ndrift, &
      ndim    => self%ndim)

      rs   = self%block%rangescale(ctx%iblock)

      lag  = 0.0
      ivgm = merge(iblock, 1, self%varying_vgm)

      do ivar = 1, self%nvar                  ! target variable; set the columns
        do kvar = 1, self%ngroups             ! neighbour group; set the rows
          nn = nnear(kvar)
          if (nn == 0) cycle
          !-- Data source and variogram index (old layout: 1:nvar=obs, nvar+1:end=sim)
          if (kvar > self%nvar) then
            jvar = kvar - self%nvar
            obs => self%block
          else
            jvar = kvar
            obs => self%obs(kvar)
          end if
          associate(vgm => self%vgm(jvar, ivar, ivgm))
            do i = 1, nn
              tmp = 0.0
              do k = self%block%iblockpnt(ctx%iblock), self%block%iblockpnt(ctx%iblock)+self%block%nblockpnt(iblock)-1
                lag(1:ndim) = (obs%coord(:, inear(i, kvar)) - self%grid%coord(:, k)) / rs
                tmp = tmp + vgm%cov_lag(lag) * self%grid%weight(k)
              end do
              rhsB(ivar, istart(kvar)+i) = tmp
            end do
          end associate
        end do
        if (ndrift > 0) rhsB(ivar, npp+1:npp+ndrift) = self%block%drift(:, iblock) ! TODO: need separate drift for each variable
        if (self%unbias == 1) rhsB(ivar, matsize) = 1.0
      end do
    end associate
  end subroutine assemble_rhs


  !============================================================================
  ! assemble_lhs
  !
  ! Fill the left-hand-side covariance matrix matA for the current block.
  ! Called by assemble_linear_system on a factorization cache miss.
  !
  ! Matrix layout (ordinary kriging, two variables, npp = n1 + n2)
  ! ---------------------------------------------------------------
  !   [ C₁₁  C₁₂  F1 ] [ λ₁ ]   [ c₀₁ ] rows/cols 1:n1
  !   [ C₂₁  C₂₂  F2 ] [ λ₂ ] = [ c₀₂ ] rows/cols n1+1:npp
  !   [ F1ᵀ  F2ᵀ   0 ] [ μ  ]   [ f₀  ] npp+1:matsize
  !
  ! For each row group kvar and column group lvar (lvar >= kvar, upper triangle):
  !   Diagonal block (kvar == lvar): C(0) + obs_variance + localnugget
  !   Off-diagonal: C(lag) between obs i of kvar and obs j of lvar
  ! The lower triangle is mirrored afterward.  Drift and unbiasedness
  ! columns are appended explicitly.
  !============================================================================
  subroutine assemble_lhs(self, ctx)
    class(t_kriging)     :: self
    class(t_kriging_ctx) :: ctx

    integer                :: kvar, lvar, i, j, jstart, irow1, icol1
    integer                :: ivgm, ivar, jvar
    real                   :: lag(3), ln, rs
    class(t_data), pointer :: obs1, obs2

    associate( &
      matA    => ctx%matA, &
      inear   => ctx%inear, &
      nnear   => ctx%nnear, &
      istart  => ctx%istart, &
      npp     => ctx%npp, &
      matsize => ctx%matsize, &
      ndrift  => self%ndrift, &
      ndim    => self%ndim)

      lag  = 0.0
      ivgm = merge(ctx%iblock, 1, self%varying_vgm)
      rs   = self%block%rangescale(ctx%iblock)
      ln   = self%block%localnugget(ctx%iblock)

      rowloop: do kvar = 1, self%ngroups
        if (nnear(kvar) == 0) cycle

        !-- Row-group data source and variogram index (old layout: 1:nvar=obs, nvar+1:end=sim)
        if (kvar > self%nvar) then
          ivar = kvar - self%nvar
          obs1 => self%block
        else
          ivar = kvar
          obs1 => self%obs(kvar)
        end if

        !-- Fill upper triangle in matrix position: lvar groups whose istart >= kvar's istart.
        !   The loop iterates all groups; the istart(lvar) < istart(kvar) guard skips the lower triangle.
        !   The lower triangle is filled by the mirror pass at the end.
        columnloop: do lvar = 1, self%ngroups
          if (nnear(lvar) == 0) cycle
          if (istart(lvar) < istart(kvar)) cycle   ! below the diagonal in matrix position — skip

          if (lvar > self%nvar) then
            jvar = lvar - self%nvar
            obs2 => self%block
          else
            jvar = lvar
            obs2 => self%obs(lvar)
          end if

          associate(vgm => self%vgm(jvar, ivar, ivgm))
            do i = 1, nnear(kvar)
              if (lvar == kvar) then
                !-- Diagonal element: C(0) + obs variance + local nugget.
                !   Simulated blocks carry zero observation error.
                if (kvar > self%nvar) then
                  matA(istart(lvar)+i, istart(kvar)+i) = vgm%cov0 + ln
                else
                  matA(istart(lvar)+i, istart(kvar)+i) = vgm%cov0 + obs1%variance(1,1,inear(i,kvar)) + ln
                end if
                jstart = i + 1   ! upper half of diagonal block only
              else
                jstart = 1
              end if
              !-- Off-diagonal: C(lag) between neighbour i of kvar and neighbour j of lvar
              do j = jstart, nnear(lvar)
                lag(1:ndim) = (obs1%coord(:, inear(i,kvar)) - obs2%coord(:, inear(j,lvar))) / rs
                matA(istart(lvar)+j, istart(kvar)+i) = vgm%cov_lag(lag)
              end do
            end do
          end associate
        end do columnloop

        !-- Drift columns start at npp+1 (fixed position regardless of kvar)
        if (ndrift > 0) then
          matA(npp+1:npp+ndrift, istart(kvar)+1:istart(kvar)+nnear(kvar)) = obs1%drift(:, inear(1:nnear(kvar), kvar))
        end if
      end do rowloop

      !-- Unbiasedness constraint column: sum(weights) = 1
      if (self%unbias == 1) matA(matsize, 1:npp) = 1.0

      !-- Mirror lower triangle from upper (C is symmetric)
      do irow1 = 1, npp
        do icol1 = irow1+1, matsize
          matA(irow1, icol1) = matA(icol1, irow1)
        end do
      end do

      !-- Zero the constraint-row diagonal block (no constraint-constraint covariance)
      matA(npp+1:matsize, npp+1:matsize) = 0.0
    end associate
  end subroutine assemble_lhs


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
  ! Exact-match detection is handled in estimate_block (Phase 1), not here.
  ! The kriging system always runs; when an obs sits exactly at the block
  ! centre the solver naturally produces λ(j*)→1 with variance→0, and
  ! estimate_block then overrides those approximate values with exact ones.
  !============================================================================
  subroutine assemble_linear_system(self, ctx)
    class(t_kriging)     :: self
    class(t_kriging_ctx) :: ctx

    integer          :: ivar, istart=0
    character(len=*), parameter :: subname = "t_kriging%assemble_linear_system"

    associate(nvar => self%nvar, npp => ctx%npp)

      ctx%factor_cache_hit = .false.

      !-- Find neighbours for each variable
      do ivar = 1, nvar
        call self%search_neighbors(ivar, ctx)
      end do

      !-- calcualte the starting index of each group in the matrix
      !   sim groups follow their respective obs groups immediately: obs1, sim1, obs2, sim2 ...
      istart = 0
      do ivar = 1, nvar
        ctx%istart(ivar) = istart
        istart = istart + ctx%nnear(ivar)
        if (self%nsim > 0) then
          ctx%istart(ivar+self%nvar) = istart
          istart = istart + ctx%nnear(ivar+self%nvar)
        end if
      end do

      npp = sum(ctx%nnear)

      !-- Degenerate: no neighbours found at all
      if (npp == 0) then
        call kriging_error(subname, 'not enough neighbors for kriging at block', iblock=ctx%iblock)
        return
      end if

      !-- Sort obs-group neighbour indices into canonical order for the factorization
      !   cache.  The KD-tree ranks by distance; two blocks sharing the same neighbour
      !   set but at different locations receive them in different rank order.  Sorting
      !   by observation index makes factor_cache_matches order-independent.
      !   Sim groups are not sorted: their nnear changes at virtually every SGSIM step
      !   so the cache never hits on them anyway.
      do ivar = 1, nvar
        if (ctx%nnear(ivar) > 0) call isort(ctx%inear(1:ctx%nnear(ivar), ivar), ctx%nnear(ivar))
      end do

      ctx%matsize = npp + self%unbias + self%ndrift

      if (ctx%factor_cache_matches(self)) then
        ctx%factor_cache_hit = .true.
        call self%assemble_rhs(ctx)
        return
      end if

      ctx%factor_cache_valid = .false.
      call self%assemble_lhs(ctx)
      call self%assemble_rhs(ctx)
    end associate
#ifdef DEBUG
    print *, subname, " Finished.", ctx%iblock
#endif
  end subroutine assemble_linear_system


  !============================================================================
  ! solve_linear_system
  !
  ! Solve the assembled kriging system K * lambda = c0 and compute the
  ! kriging variance.
  !
  ! Solver strategy
  ! ---------------
  ! 1. kriging_solve (spotrf + spotrs Cholesky): fast, exploits positive-definite
  !    structure.  Preferred path.
  ! 2. ssysv_fallback (SSYSV symmetric indefinite): more robust; used when
  !    Cholesky fails (e.g. due to numerical near-singularity).
  ! 3. If both fail: either stop with an error (neglect_error=.false.) or set
  !    the result to NaN and continue (neglect_error=.true.).
  !
  ! Weight correction
  ! -----------------
  ! If weight_correction=.true., negative weights are clipped to 0 and the
  ! remaining weights are renormalised to sum to 1.  This removes the
  ! screen effect where distant observations can receive negative weights
  ! that partially cancel closer ones.
  !
  ! The conditional covariance Sigma is computed by calc_variance
  ! and stored in self%block%variance(:, :, iblock).
  !============================================================================
  subroutine solve_linear_system(self, ctx)
    use solver
    implicit none
    class(t_kriging)     :: self
    class(t_kriging_ctx) :: ctx

    integer            :: info, p, q
    character(len=*), parameter :: subname = "t_kriging%solve_linear_system"

    associate( &
      iblock    => ctx%iblock, &
      matA      => ctx%matA, &
      rhsB      => ctx%rhsB, &
      matsize   => ctx%matsize, &
      npp       => ctx%npp, &
      x         => ctx%x, &
      unbias    => self%unbias, &
      ndrift    => self%ndrift)

      p = unbias + ndrift
      q = self%nvar

      !-- Primary solver: Cholesky setup plus per-block RHS solve.
      if (ctx%factor_cache_hit) then
        call kriging_solve_prepared(npp, p, q, ctx%factor_L, ctx%factor_kinv_drift, &
                                    ctx%factor_schur, rhsB, x, info)
      else
        call kriging_setup(npp, p, matA, ctx%factor_L, ctx%factor_kinv_drift, &
                           ctx%factor_schur, info)
        if (info == 0) then
          call ctx%save_factor_cache_key(self)
          call kriging_solve_prepared(npp, p, q, ctx%factor_L, ctx%factor_kinv_drift, &
                                      ctx%factor_schur, rhsB, x, info)
        end if
      end if

      !-- Fallback: symmetric indefinite solver (SSYSV)
      if (info /= 0) then
        ctx%factor_cache_valid = .false.
        call ssysv_fallback(npp, p, q, matA, rhsB, x, info)
        if (self%verbose) &
          print*, "Cholesky fails. Fallback SSYSV is used for block", iblock
      end if

      !-- Both solvers failed
      if (info /= 0) then
        ! Singular-matrix diagnostics use the same debug writer as write_mat.
        !$OMP CRITICAL(write_matrix_io)
        call ctx%write_matrix(self)
        !$OMP END CRITICAL(write_matrix_io)
        if (self%neglect_error) then
          x = IEEE_VALUE(0.0, IEEE_QUIET_NAN)
          ctx%nnear = 1
          npp = self%nvar
        else
          call kriging_error(subname, 'Singular matrix', iblock=ctx%iblock)
          return
        end if
      end if

      !-- Optional: clip negative weights and renormalise
      if (self%weight_correction .and. q == 1) then
        x(1, 1:npp) = merge(x(1, 1:npp), 0.0, x(1, 1:npp) > 0)
        x(1, 1:npp) = x(1, 1:npp) / sum(x(1, 1:npp))
      end if

      call self%calc_variance(ctx)
    end associate
#ifdef DEBUG
    print *, subname, " Finished. ", ctx%iblock
#endif
  end subroutine solve_linear_system


  !============================================================================
  ! calc_variance
  !
  ! Compute the conditional covariance matrix for the current block and store
  ! it in self%block%variance(1:nvar, 1:nvar, ctx%iblock).
  !
  ! For each variable pair (ivar, jvar):
  !   Sigma(ivar, jvar) = C_ij(x0, x0) - lambda_ivar^T * c0_jvar
  !
  ! where:
  !   C_ij(x0, x0)  prior cross-covariance at x0; for point kriging this is
  !                  vgm(ivar, jvar)%cov0; for block kriging it is the
  !                  integration-node-weighted sum over all node pairs.
  !   lambda_ivar   kriging weights for variable ivar = ctx%x(ivar, 1:matsize)
  !   c0_jvar       RHS covariance vector for variable jvar = ctx%rhsB(jvar, :)
  !
  ! The extended vectors (weights + Lagrange multipliers, length matsize) are
  ! used throughout, which makes the formula correct for ordinary and drift
  ! kriging as well as simple kriging.
  !
  ! After computing the raw Sigma, the diagonal is clamped to >= 0 and the
  ! off-diagonal is symmetrised to suppress numerical round-off.
  !============================================================================
  subroutine calc_variance(self, ctx)
    class(t_kriging)     :: self
    class(t_kriging_ctx) :: ctx

    integer :: i, j, k1, ivgm, ivar, jvar
    real    :: lag(3), base_cov, cov_kl

    lag  = 0.0
    ivgm = merge(ctx%iblock, 1, self%varying_vgm)

    associate( &
      ndim      => self%ndim, &
      iblock    => ctx%iblock, &
      matsize   => ctx%matsize, &
      x         => ctx%x, &
      rhsB      => ctx%rhsB, &
      weight    => self%grid%weight, &
      coord     => self%grid%coord, &
      nblockpnt => self%block%nblockpnt(ctx%iblock), &
      var       => self%block%variance(:, :, ctx%iblock))

      do ivar = 1, self%nvar
        do jvar = ivar, self%nvar
          associate( vgm => self%vgm(ivar, jvar, ivgm) )
            if (nblockpnt == 1) then
              !-- Point kriging: C_ij(x0, x0) = covariance at zero lag
              base_cov = vgm%cov0
            else
              !-- Block kriging: weighted average of C_ij(s_p, s_q) over all integration node pairs.
              !   Upper triangle with ×2 avoids double-loop; diagonal uses cov0 (p == q, lag = 0).
              base_cov = 0.0
              k1 = self%block%iblockpnt(iblock) - 1
              do i = 1, nblockpnt
                base_cov = base_cov + vgm%cov0 * weight(k1+i) * weight(k1+i)
                do j = i+1, nblockpnt
                  lag(1:ndim) = coord(:, k1+i) - coord(:, k1+j)
                  base_cov = base_cov + vgm%cov_lag(lag) * &
                    weight(k1+i) * weight(k1+j) * 2.0
                end do
              end do
            end if
          end associate
          var(ivar, jvar) = &
            base_cov - dot_product(x(ivar, 1:matsize), rhsB(jvar, 1:matsize))
          if (ivar /= jvar) var(jvar, ivar) = var(ivar, jvar) ! symmetrise
        end do
      end do

      !-- Clamp diagonal to >= 0 (negative values arise only from numerical noise).
      !-- Symmetrise off-diagonal: both (C_ij - x_i^T c0_j) and (C_ji - x_j^T c0_i)
      !   are theoretically equal by symmetry of K; averaging suppresses residual asymmetry.
      do ivar = 1, self%nvar
        var(ivar, ivar) = max(var(ivar, ivar), 0.0)
        do jvar = ivar + 1, self%nvar
          var(jvar, ivar) = max(var(jvar, ivar), 0.0)
          var(ivar, jvar) = var(jvar, ivar)
        end do
      end do

    end associate
  end subroutine calc_variance


  !============================================================================
  ! assign_weight
  !
  ! Split the flat solution vector x(1, 1:npp) into per-variable weight arrays.
  ! x is ordered as [lambda(group_1), ..., lambda(group_ngroups)], each block of size
  ! nnear(ivar).  The Lagrange multiplier(s) at positions npp+1:matsize are
  ! not needed after this point.
  !============================================================================
  subroutine assign_weight(self, krige)
    class(t_kriging_ctx) :: self
    class(t_kriging)     :: krige

    integer :: kvar, ivar, k1
    character(len=*), parameter :: subname = "t_kriging%assign_weight"

    !-- weight(:, kvar) holds the primary-variable (ivar=1) weights per group.
    !   For joint co-sim the full x(ivar,:) is used directly in estimate_block.
    do kvar = 1, krige%ngroups
      if (self%nnear(kvar) == 0) cycle
      do ivar = 1, krige%nvar
        self%weight(1:self%nnear(kvar), kvar, ivar) = &
          self%x(ivar, self%istart(kvar)+1:self%istart(kvar)+self%nnear(kvar))
      end do
    end do
#ifdef DEBUG
    print *, subname, " Finished. ", self%iblock
#endif
  end subroutine assign_weight


  !============================================================================
  ! estimate_block
  !
  ! Compute the kriging estimate or SGSIM realisations for the current block.
  !
  ! Phase 1 — conditional mean (obs-only part, constant across realisations)
  !   For each target variable ivar (1:nvar):
  !     mu_obs(ivar) = sum_{kvar=1,nvar obs groups} lambda(ivar, j) * z_obs(kvar, j)
  !                  [+ ISAAKS correction, co-kriging nsim==0 only]
  !                  [+ SK mean correction, simple kriging only]
  !   For kriging (nsim == 0) the estimate IS mu_obs; return after Phase 1.
  !   Exact matches (obs at block centre) need no special handling: the solver
  !   produces lambda(j*)→1 naturally and calc_conditional_variance gives
  !   variance→0 (clamped ≥0 on the diagonal).
  !
  ! Phase 2 — SGSIM (nsim > 0), per-realization loop
  !   mu(ivar) = mu_obs(ivar)
  !            + sum_{kvar=nvar+1,ngroups sim groups} lambda(ivar, j) * z_sim(isim, jb)
  !   Stochastic draw (unified for all nvar via spotrf):
  !     Sigma = block%variance(:,:,iblock)  [nvar x nvar conditional covariance]
  !     L     = chol(Sigma)  via spotrf('L')  [lower triangular Cholesky factor]
  !     z(isim) = mu + L * epsilon,  epsilon ~ N(0, I_nvar)
  !     epsilon(1) from pre-drawn sample (reproducibility); epsilon(2:nvar) freshly drawn.
  !     For nvar==1 this reduces to z = mu + sqrt(Sigma) * epsilon, same as before.
  !
  ! ISAAKS & SRIVASTAVA correction (co-kriging, nsim==0 only)
  !   When nvar>1 and unbias==1, adjusts secondary-variable weight sums so
  !   the co-kriging estimate is unbiased across variables with differing
  !   local means.
  !============================================================================
  subroutine estimate_block(self, ctx)
    implicit none
    class(t_kriging)     :: self
    class(t_kriging_ctx) :: ctx

    integer           :: ivar, jvar, kvar, k, k1, j, jb, nn, isim, real_ivar, info
    real              :: total_weight(self%nvar)
    real              :: target_mean(self%nvar)
    real              :: L_chol(self%nvar, self%nvar)
    real              :: avg
    real              :: weight(self%nvar), perturbation(self%nvar)
    logical           :: ck_correction

    associate( &
      var    => self%block%variance(:, :, ctx%iblock), &  ! nvar x nvar
      val    => self%block%value(:, :, ctx%iblock), &  ! nsim x nvar
      iblock => ctx%iblock, &
      nnear  => ctx%nnear, &
      inear  => ctx%inear, &
      weight => ctx%weight)

      !----------------------------------------------------------------------
      ! Phase 1: obs-only conditional mean mu_obs(ivar).
      !----------------------------------------------------------------------
      ! Pre-computed variable means for ISAAKS correction (co-kriging, nsim==0).
      if (self%nsim == 0 .and. self%unbias /= 0 .and. self%nvar > 1) then
        ck_correction = .true.
        target_mean = 0.0
        do ivar = 1, self%nvar
          do k = 1, nnear(ivar)
            target_mean(ivar) = target_mean(ivar) + self%obs(ivar)%value(1,1,inear(k, ivar))
          end do
          !-- Guard against division by zero when a variable has no neighbours
          !   (e.g. secondary maxdist is too small).  total_weight(ivar) will
          !   also be zero in that case, so the correction term below is 0
          !   regardless of the value chosen here.
          if (nnear(ivar) > 0) target_mean(ivar) = target_mean(ivar) / nnear(ivar)
        end do
      else
        ck_correction = .false.
      end if

      do jvar = 1, self%nvar  ! estiamte target variable
        total_weight = 0.0
        avg = 0.0
        do ivar = 1, self%nvar  ! source variable [observations only]
          total_weight(ivar) = total_weight(ivar) + sum(weight(1:nnear(ivar), ivar, jvar))
          avg = avg + dot_product(self%obs(ivar)%value(1,1,inear(1:nnear(ivar), ivar)), weight(1:nnear(ivar), ivar, jvar))
          if (ck_correction .and. ivar/=jvar .and. nnear(ivar) > 0) avg = avg + (target_mean(jvar) - target_mean(ivar)) * total_weight(ivar)
        end do

        !-- Simple kriging mean correction.
        if (self%unbias == 0 .and. self%obs(jvar)%sk_mean /= 0.0) &
          avg = avg + (1.0 - sum(total_weight)) * self%obs(jvar)%sk_mean

        !-- assign average
        val(:, jvar) = avg
      end do

      if (self%nsim > 0) then

        !----------------------------------------------------------------------
        ! Phase 2: SGSIM — add per-realization stochastic perturbation.
        !
        ! For nvar > 1: Cholesky-factor Sigma = block%variance(:,:,iblock) once.
        ! The factor L is the same for all realisations (Sigma does not change).
        !----------------------------------------------------------------------
        !-- Cholesky-factor the conditional covariance Sigma = L * Lᵀ via LAPACK spotrf.
        !   spotrf('L') overwrites the lower triangle of L_chol with L; the upper
        !   triangle is untouched but is never read by the draw loop below.
        !   Works for nvar==1 too: L(1,1) = sqrt(Sigma(1,1)).

        L_chol = var
        call spotrf('L', self%nvar, L_chol, self%nvar, info)
        if (info /= 0) then
          !-- Not positive definite (near-singular variance matrix).
          !   Fall back to uncorrelated diagonal draws: L = diag(sqrt(diag(Sigma))).
          L_chol = 0.0
          do ivar = 1, self%nvar
            L_chol(ivar, ivar) = sqrt(max(var(ivar, ivar), 0.0))
          end do
        end if


        ! Add sim-neighbour conditioning: for each target variable jvar, loop over
        ! all source variables ivar; sim group kvar = nvar+ivar holds the previously
        ! simulated values of variable ivar at nearby blocks.
        do isim = 1, self%nsim
          do jvar = 1, self%nvar  ! target variable being estimated
            do ivar = 1, self%nvar  ! source variable from already-simulated blocks
              kvar = self%nvar + ivar
              val(isim,jvar) = val(isim,jvar) + &
                dot_product(self%block%value(isim,jvar,inear(1:nnear(kvar), kvar)), weight(1:nnear(kvar), kvar, jvar))
            end do
          end do

          !-- Add stochastic perturbation.
          perturbation = matmul(L_chol, self%block%sample  (isim,:,ctx%iblock))
          val(isim, :) = val(isim, :) + perturbation
        end do
      end if

      !-- Clamp to bounds
      where(val < self%bounds(1)) val = self%bounds(1)
      where(val > self%bounds(2)) val = self%bounds(2)
    end associate
#ifdef DEBUG
    print *, "t_kriging%estimate_block Finished.", ctx%iblock
#endif
  end subroutine estimate_block


!============================================================================
! update_info
!
! Returns the full kriging configuration as a single, multi-line string.
! Call after set_search() so all fields are populated.
!============================================================================
function to_str(self) result(res_str)
    implicit none
    class(t_kriging), intent(inout) :: self
    character(len=256) :: buffer
    character(len=:), allocatable :: res_str
    integer :: ivar, jvar
    character(len=1), parameter :: NL = new_line('A')

    res_str = NL
    write(buffer, "(A)"  ) "==================== Configuration ===================="    ; res_str = res_str // trim(buffer) // NL
    write(buffer, "(A,A)")  " Version                : ", version                       ; res_str = res_str // trim(buffer) // NL
    write(buffer, "(A,I0)") " Dimension              : ", self%ndim                     ; res_str = res_str // trim(buffer) // NL
    write(buffer, "(A,I0)") " Number of Variables    : ", self%nvar                     ; res_str = res_str // trim(buffer) // NL
    write(buffer, "(A,I0)") " Number of Simulations  : ", self%nsim                     ; res_str = res_str // trim(buffer) // NL
    write(buffer, "(A,I0)") " Number of Drifts       : ", self%ndrift                   ; res_str = res_str // trim(buffer) // NL
    write(buffer, "(A,I0)") " Number of Blocks       : ", self%block%n                  ; res_str = res_str // trim(buffer) // NL
    write(buffer, "(A,A )") " Ordinary Kriging       : ", yesno(self%unbias == 1)       ; res_str = res_str // trim(buffer) // NL
    write(buffer, "(A,A )") " LOO-Cross Validation   : ", yesno(self%cross_validation)  ; res_str = res_str // trim(buffer) // NL
    write(buffer, "(A,A )") " Weight Correction      : ", yesno(self%weight_correction) ; res_str = res_str // trim(buffer) // NL
    write(buffer, "(A,A )") " Use Old Weights        : ", yesno(self%use_old_weight)    ; res_str = res_str // trim(buffer) // NL
    write(buffer, "(A,A )") " Write Matrix for Debug : ", yesno(self%write_mat)         ; res_str = res_str // trim(buffer) // NL
    write(buffer, "(A,A )") " Write Weight File      : ", yesno(self%store_weight)      ; res_str = res_str // trim(buffer) // NL

    if (self%store_weight .or. self%use_old_weight) then
        write(buffer, "(A,A )") " Weight File : ", trim(self%weight_file)
        res_str = res_str // trim(buffer) // NL
    end if

    if (self%unbias == 0) then
    end if

    write(buffer, "(A,G0)") " Lower Bound            : ", self%bounds(1); res_str = res_str // trim(buffer) // NL
    write(buffer, "(A,G0)") " Upper Bound            : ", self%bounds(2); res_str = res_str // trim(buffer) // NL

    do ivar = 1, self%nvar
        write(buffer, "(A,I0,A)") "Variable ", ivar, ":"
        res_str = res_str // trim(buffer) // NL

        write(buffer, "(A,I0)") " Number of data         : ", self%obs(ivar)%n
        res_str = res_str // trim(buffer) // NL

        write(buffer, "(A,I0)") " Maximum neighbors      : ", self%obs(ivar)%nmax
        res_str = res_str // trim(buffer) // NL

        write(buffer, "(A,G0)") " Maxdist                : ", sqrt(self%obs(ivar)%maxdist)
        res_str = res_str // trim(buffer) // NL

        write(buffer, "(A,G0)") " Simple Kriging Mean    : ", self%obs(ivar)%sk_mean
        res_str = res_str // trim(buffer) // NL

        write(buffer, "(A,G0)") " Required Search        : ", yesno(self%obs(ivar)%need_search)
        res_str = res_str // trim(buffer) // NL

        write(buffer, "(A,G0)") " Anisotropic Search     : ", yesno(self%obs(ivar)%anisotropic_search)
        res_str = res_str // trim(buffer) // NL
    end do

    write(buffer, "(A)") " Variogram Models"
    res_str = res_str // trim(buffer) // NL

    do ivar = 1, self%nvar
        do jvar = 1, self%nvar
            if (ivar == jvar) then
                write(buffer, "(A,I0,A,I0,A)") " Model for Variable ", ivar, self%vgm(jvar, ivar, 1)%tostr()
            else
                write(buffer, "(A,I0,A,I0,A)") " Model between Variable ", ivar, " and ", jvar, self%vgm(jvar, ivar, 1)%tostr()
            end if
            res_str = res_str // trim(buffer) // NL
        end do
    end do

    write(buffer, "(A )") "================== End Configuration =================="
    res_str = res_str // trim(buffer) // NL
end function to_str

subroutine update_info(self)
    class(t_kriging) :: self
    character(len=:), allocatable :: res_str
    integer :: n, i
    res_str = self%to_str()
    n = len_trim(res_str)
    if (associated(self%krige_info)) deallocate(self%krige_info)
    allocate(self%krige_info(n + 1))
    do i = 1, n
      self%krige_info(i) = res_str(i:i)
    end do
    self%krige_info(n+1) = c_null_char
end subroutine update_info



  !============================================================================
  ! write_matrix
  !
  ! Dump the kriging matrix, RHS, and neighbour data to CSV files for debugging.
  ! Activated when write_mat=.true.
  !
  ! Output files (named by original block index order(ib)):
  !   data_<ib>.csv  — neighbour coordinates, values, distances, weights
  !   matA_<ib>.csv  — full kriging matrix [matsize x matsize]
  !   rhsB_<ib>.csv  — right-hand-side vector [matsize]
  !
  ! The caller serializes this routine under OpenMP.  The file names are unique,
  ! but concurrent OPEN/WRITE/CLOSE through the Fortran runtime has caused
  ! access violations with the Windows DLL runtime.
  !============================================================================
  subroutine write_matrix(self, krige)
    class(t_kriging_ctx) :: self
    class(t_kriging)     :: krige

    integer              :: kvar, ifile, ii, jvar, isim, is, iv
    integer              :: idx
    character(len=20)    :: sig, idxstr
    character(len=1)     :: cname(3) = ['x', 'y', 'z']
    class(t_data), pointer :: dat
    associate( &
      ndim      => krige%ndim, &
      ib        => self%iblock, &
      nnear     => self%nnear, &
      k1        => self%istart, &
      inear     => self%inear, &
      weight    => self%x, &
      matA      => self%matA, &
      rhsB      => self%rhsB, &
      x         => self%x, &
      npp       => self%npp, &
      irandpath => krige%block%order, &
      matsize   => self%matsize)

      do isim=1, max(krige%nsim,1)
        do jvar = 1, krige%nvar
          write(idxstr, "(I0,'_var',I0,'_sim',I0)") irandpath(ib), jvar, isim
          !-- Neighbour data table
          open(newunit=ifile, file='data_'//trim(idxstr)//'.csv', status='replace')
          write(ifile, '(99(A,:,","))') 'source','index', cname(1:ndim), 'value', 'weight'
          do kvar = 1, krige%ngroups
            if (nnear(kvar) == 0) cycle
            if (kvar > krige%nvar) then
              is = isim
              iv = jvar
              dat => krige%block
              write(sig, "('GRID',I0)") kvar-krige%nvar
            else
              is = 1
              iv = 1
              dat => krige%obs(kvar)
              write(sig, "('OBS',I0)") kvar
            end if
            do ii = 1, nnear(kvar)
              idx = inear(ii, kvar)
              write(ifile, "(A,',',I0,*(:,',',G0.8))") &
                trim(sig), idx, dat%coord(:,idx), dat%value(is,iv,idx), x(k1(kvar)+ii,jvar)
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
  end subroutine write_matrix


  !============================================================================
  ! write_weight / read_weight
  !
  ! Serialise / deserialise per-block kriging weights to the factor file.
  !
  ! Factor file format (three lines per block, plus joint SGSIM solution rows):
  !   Header: nblock nvar -2 nmax(1:nvar)
  !   Line 1: order(ib)  variance(ib,1:nvar,1:nvar)  nnear(1:ngroups)
  !   Line 2: inear(1:nnear(1),1) ...  (neighbour indices)
  !   Line 3: weight(1:nnear(1),1) ... (primary-variable kriging weights)
  !   Joint SGSIM only: nvar extra lines containing x(kvar,1:matsize).
  !
  ! read_weight      is used when use_old_weight=.true. (factor-file reuse mode).
  ! write_weight     kept for direct use; no longer called by solve().
  ! write_weight_store writes the completed wstore to weight_file after solve().
  !============================================================================
  subroutine write_weight(self, ctx)
    class(t_kriging)     :: self
    class(t_kriging_ctx) :: ctx
    integer              :: ii, ivar, jvar
    associate( &
      ib    => ctx%iblock, &
      order => self%block%order)
      write(self%ifile, *) order(ib), &
      ctx%nnear(1:self%ngroups), &
        ((self%block%variance(ivar, jvar, ib), ivar=1, self%nvar), jvar=1, self%nvar)
      write(self%ifile, '(*(:2x,I0))')   (ctx%inear(1:ctx%nnear(ii), ii),  ii = 1, self%ngroups)
      do ivar = 1, self%nvar
        write(self%ifile, '(*(:2x,F0.10))')(ctx%weight(1:ctx%nnear(ii), ii, ivar), ii = 1, self%ngroups)
      end do
    end associate
  end subroutine write_weight


  !============================================================================
  ! write_weight_store
  !
  ! Write the complete in-memory weight store to weight_file in one pass after
  ! solve() completes.  Format is identical to the per-block write_weight
  ! output, so read_weight (use_old_weight path) can re-load it unchanged.
  !
  ! Three lines per block, with nvar extra x rows for joint SGSIM:
  !   Line 1: original_block_index  variance(1:nvar,1:nvar)  nnear(1:ngroups)
  !   Line 2: inear indices, flat (all groups concatenated)
  !   Line 3: primary-variable kriging weights, flat
  !============================================================================
  subroutine write_weight_store(self)
    class(t_kriging) :: self
    integer          :: ib, ii, ivar, jvar, ifile, matsize

    associate(ws => self%wstore)
      open(newunit=ifile, file=trim(self%weight_file), status='replace')
      write(ifile, *) self%block%n, self%nvar, (self%obs(ivar)%nmax, ivar=1, self%nvar)
      do ib = 1, self%block%n
        write(ifile, "(I0,*(:2x,I0))") self%block%order(ib), ws%nnear(1:self%ngroups, ib)
        write(ifile, '(*(:2x,I0))') (ws%inear (1:ws%nnear(ii,ib), ii, ib), ii = 1, self%ngroups)
        write(ifile, "(*(:2x,G0.10))") ((self%block%variance(ivar, jvar, ib), ivar=1, self%nvar), jvar=1, self%nvar)
        do ivar = 1, self%nvar
          write(ifile, '(*(:2x,F0.10))') &
            (ws%weight(1:ws%nnear(ii,ib), ii, ivar, ib), ii = 1, self%ngroups)
        end do
      end do
      close(ifile)
    end associate
  end subroutine write_weight_store


  subroutine read_weight(self, ctx)
    class(t_kriging)     :: self
    class(t_kriging_ctx) :: ctx
    integer              :: ii, ivar, jvar, kvar
    associate( &
      ib    => ctx%iblock, &
      order => self%block%order)
      read(self%ifile, *) order(ib), ctx%nnear(1:self%ngroups)
      read(self%ifile, *) (ctx%inear(1:ctx%nnear(ii), ii),  ii = 1, self%ngroups)
      read(self%ifile, *) ((self%block%variance(ivar, jvar, ib), ivar=1, self%nvar), jvar=1, self%nvar)
      do ivar = 1, self%nvar
        read(self%ifile, *) (ctx%weight(1:ctx%nnear(ii), ii, ivar), ii = 1, self%ngroups)
      end do
      ctx%npp = sum(ctx%nnear)
      ctx%matsize = ctx%npp + self%unbias + self%ndrift
    end associate
  end subroutine read_weight


  !============================================================================
  ! read_weight_to_store
  !
  ! Read one block's weights from an already-open factor file directly into
  ! wstore (instead of into a per-thread ctx).  Called block-by-block in
  ! prepare() so the entire file is loaded before the OpenMP block loop starts.
  !
  ! ifile  — Fortran unit of the open factor file (sequential, text)
  ! ib     — 1-based block index to fill in wstore
  !============================================================================
  subroutine read_weight_to_store(self, ifile, ib)
    class(t_kriging) :: self
    integer, intent(in) :: ifile, ib
    integer :: ii, ivar, jvar
    character(len=*), parameter :: subname = "t_kriging%read_weight_to_store"
    associate(ws => self%wstore)
      read(ifile, *, err=10, end=10) self%wstore%order(ib), ws%nnear(1:self%ngroups, ib)
      read(ifile, *, err=10, end=10) &
        (ws%inear(1:ws%nnear(ii,ib), ii, ib), ii = 1, self%ngroups)
      read(ifile, *, err=10, end=10) &
        ((self%block%variance(ivar, jvar, ib), ivar=1, self%nvar), jvar=1, self%nvar)
      ws%var(1:self%nvar, 1:self%nvar, ib) = &
        self%block%variance(1:self%nvar, 1:self%nvar, ib)
      do ivar = 1, self%nvar
        read(ifile, *, err=10, end=10) &
          (ws%weight(1:ws%nnear(ii,ib), ii, ivar, ib), ii = 1, self%ngroups)
      end do
    end associate
    return
10  call kriging_error(subname, 'Unexpected end-of-file or read error in factor file.')
  end subroutine read_weight_to_store


  !============================================================================
  ! validate_vgm  (private helper)
  !
  ! Verify that every block/variable-pair has at least one structure defined.
  ! Stops with a descriptive error if any block is missing its variogram.
  !============================================================================
  subroutine validate_vgm(self)
    class(t_kriging), intent(in) :: self
    integer                  :: ib, ivar, jvar, mb
    character(len=256)       :: msg
    character(*), parameter  :: subname = 't_kriging_sva%validate_vgm'
    mb = merge(self%block%n, 1, self%varying_vgm)
    do ib = 1, mb
      do ivar = 1, self%nvar
        do jvar = 1, self%nvar
          if (self%vgm(jvar, ivar, ib)%nstruct == 0) then
            write(msg, '(A,I0,A,I0,A,I0,A)') &
              't_kriging_sva: variogram not set for block ', ib, &
              ', ivar=', ivar, ', jvar=', jvar, &
              '. Call set_vgm_block() or set_vgm_block_all().'
            call kriging_error(subname, trim(msg))
            return
          end if
          if (.not. self%vgm(jvar, ivar, ib)%is_valid()) then
            write(msg, '(A,I0,A,I0,A,I0,A)') &
              't_kriging_sva: variogram is not valid for block ', ib, &
              ', ivar=', ivar, ', jvar=', jvar, &
              '. Check your variogram parameters.'
            call kriging_error(subname, trim(msg))
            return
          end if
        end do
      end do
    end do
  end subroutine validate_vgm


  !============================================================================
  ! reset_data
  !
  ! Deallocate all allocatable fields of a t_data instance and reset n to 0.
  ! Called by reset_obs and reset_block (which extend t_data).
  !============================================================================
  subroutine reset_data(d)
    class(t_data), intent(inout) :: d
    if (allocated(d%coord))    deallocate(d%coord)
    if (allocated(d%drift))    deallocate(d%drift)
    if (allocated(d%value))    deallocate(d%value)
    if (allocated(d%variance)) deallocate(d%variance)
    d%n = 0
  end subroutine reset_data


  !============================================================================
  ! reset_obs
  !
  ! Deallocate all fields of a t_obsgrid, destroy the k-d tree if present,
  ! and reset scalar members to their defaults.  Call before re-loading
  ! observations into an already-initialised t_kriging.
  !============================================================================
  subroutine reset_obs(self, ivar)
    class(t_kriging), intent(inout) :: self
    integer,          intent(in)    :: ivar
    associate(obs => self%obs(ivar))
      call reset_data(obs)
      if (associated(obs%tree)) then
        call kdtree2_destroy(obs%tree)
        nullify(obs%tree)
      end if
      obs%nmax              = 0
      obs%maxdist           = verylarge
      obs%rotmat            = reshape([1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0], [3,3])
      obs%need_search       = .false.
      obs%anisotropic_search= .false.
    end associate
  end subroutine reset_obs


  !============================================================================
  ! reset_grid
  !
  ! Deallocate all fields of the t_grid (integration nodes).  Call before
  ! re-loading the estimation grid into an already-initialised t_kriging.
  !============================================================================
  subroutine reset_grid(self)
    class(t_kriging), intent(inout) :: self
    associate(g => self%grid)
      call reset_data(g)
      if (allocated(g%weight)) deallocate(g%weight)
    end associate
  end subroutine reset_grid


  !============================================================================
  ! reset_block
  !
  ! Deallocate all fields of the t_blockgrid (estimation targets) and reset
  ! block_type to 0 (point kriging).  Call before re-loading the block grid
  ! into an already-initialised t_kriging.
  !============================================================================
  subroutine reset_block(self)
    class(t_kriging), intent(inout) :: self
    associate(b => self%block)
      call reset_data(b)
      if (allocated(b%order))         deallocate(b%order)
      if (allocated(b%nblockpnt))     deallocate(b%nblockpnt)
      if (allocated(b%iblockpnt))     deallocate(b%iblockpnt)
      if (allocated(b%rangescale))    deallocate(b%rangescale)
      if (allocated(b%localnugget))   deallocate(b%localnugget)
      if (allocated(b%sample))        deallocate(b%sample)
      b%block_type = 0
    end associate
  end subroutine reset_block


  !============================================================================
  ! finalize
  !
  ! Release all allocated memory.  Call after all results have been read from
  ! block%value and block%variance.
  !============================================================================
  subroutine finalize(self)
    class(t_kriging) :: self
    if (associated(self%obs))        deallocate(self%obs)
    if (associated(self%grid))       deallocate(self%grid)
    if (associated(self%block))      deallocate(self%block)
    if (associated(self%vgm))        deallocate(self%vgm)
    if (associated(self%krige_info)) deallocate(self%krige_info)
    if (allocated(self%wstore))      deallocate(self%wstore)
  end subroutine finalize


  !============================================================================
  ! alloc_weight_store
  !
  ! Allocate the in-memory weight store sized for the current problem.
  ! Must be called after set_grid() and set_search() (so that block%n and
  ! obs%nmax are set), and before solve().  Calling again replaces the store.
  !============================================================================
  subroutine alloc_weight_store(self)
    class(t_kriging) :: self
    integer :: nb, ng, nm, nv, q
    character(len=*), parameter :: subname = "t_kriging%alloc_weight_store"

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
    nm = maxval(self%obs(1:self%nvar)%nmax)
    if (nm <= 0) then
      call kriging_error(subname, 'call set_obs() before alloc_weight_store() so nmax is set')
      return
    end if

    if (allocated(self%wstore)) deallocate(self%wstore)
    allocate(self%wstore)
    self%wstore%nblock  = nb
    self%wstore%ngroups = ng
    self%wstore%nmax    = nm
    allocate(self%wstore%order (nb));               call set_seq(self%wstore%order, nb)
    allocate(self%wstore%nnear (ng, nb));           self%wstore%nnear  = 0
    allocate(self%wstore%inear (nm, ng, nb));       self%wstore%inear  = 0
    allocate(self%wstore%weight(nm, ng, nv, nb));   self%wstore%weight = 0.0
    allocate(self%wstore%var   (nv, nv, nb));       self%wstore%var    = 0.0
  end subroutine alloc_weight_store


  !============================================================================
  ! set_weights
  !
  !-- Set the in-memory weight store using user supplied nnear, inear, weight,
  !   order(optional) and variance(optional), usually from a previous run.
  !============================================================================
  subroutine set_weights(self, nnear_in, inear_in, weight_in, order_in, var_in)
    class(t_kriging) :: self
    integer, intent(in)            :: nnear_in (:, :)          ! nnear(ngroups, nblock)
    integer, intent(in)            :: inear_in (:, :, :)       ! inear(nmax, ngroups, nblock)
    real   , intent(in)            :: weight_in(:, :, :, :)    ! weight(nmax, ngroups, nvar, nblock)
    integer, intent(in), optional  :: order_in (:)             ! order_in(nblock); used for simulation
    real   , intent(in), optional  :: var_in   (:, :, :)       ! var(nvar, nvar, nblock); used for simulation
    integer :: nb, ng, nm, nv, q
    character(len=*), parameter :: subname = "t_kriging%set_weights"

    if (.not. allocated(self%wstore)) then
      call kriging_error(subname, 'call set_grid() before set_weights()')
      return
    end if
    nv = self%nvar
    nb = self%block%n
    ng = self%ngroups
    nm = maxval(self%obs(1:self%nvar)%nmax)
    if (nm <= 0) then
      call kriging_error(subname, 'call set_obs() before alloc_weight_store() so nmax is set')
      return
    end if
    ! check incomming array shape matches in-memory store
    if (size(nnear_in, 1) /= ng .or. size(nnear_in, 2) /= nb) then
      call kriging_error(subname, 'nnear shape mismatch')
      return
    end if
    if (size(inear_in, 1) /= nm .or. size(inear_in, 2) /= ng .or. size(inear_in, 3) /= nb) then
      call kriging_error(subname, 'inear shape mismatch')
      return
    end if
    if (size(weight_in, 1) /= nm .or. size(weight_in, 2) /= ng .or. size(weight_in, 3) /= nv .or. size(weight_in, 4) /= nb) then
      call kriging_error(subname, 'weight shape mismatch')
      return
    end if
    if (present(var_in)) then
    end if
    self%wstore%nnear = nnear_in
    self%wstore%inear = inear_in
    self%wstore%weight = weight_in
    if (present(order_in)) then
      if (size(order_in) /=  nb) then
        call kriging_error(subname, 'order shape mismatch')
        return
      end if
      self%wstore%order = order_in
    end if
    if (present(var_in)) then
      if (size(var_in, 1) /= nv .or. size(var_in, 2) /= nv .or. size(var_in, 3) /= nb) then
        call kriging_error(subname, 'var shape mismatch')
        return
      end if
      self%wstore%var = var_in
    end if
  end subroutine set_weights


  !============================================================================
  ! free_weight_store
  !
  ! Release the in-memory weight store.
  !============================================================================
  subroutine free_weight_store(self)
    class(t_kriging) :: self
    if (allocated(self%wstore)) deallocate(self%wstore)
  end subroutine free_weight_store


  !============================================================================
  ! save_block_weights
  !
  ! Copy ctx%nnear / ctx%inear / ctx%weight for the current block into the
  ! weight store.  Called from solve() after assign_weight.
  ! Writes to disjoint ib-slices — safe under OpenMP without a critical section.
  !============================================================================
  subroutine save_block_weights(self, ctx)
    class(t_kriging)     :: self
    class(t_kriging_ctx) :: ctx
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
  ! Inverse of save_block_weights — called from solve() when wstore was
  ! pre-populated by krige_set_weights (Python set_weights API).
  !============================================================================
  subroutine load_block_weights(self, ctx)
    class(t_kriging)     :: self
    class(t_kriging_ctx) :: ctx
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
      ctx%npp     = sum(ctx%nnear(1:self%ngroups))
      ctx%matsize = ctx%npp + self%unbias + self%ndrift
      self%block%variance(1:self%nvar, 1:self%nvar, ib) = ws%var(1:self%nvar, 1:self%nvar, ib)
    end associate
  end subroutine load_block_weights


  !============================================================================
  ! isort
  !
  ! In-place insertion sort on the first n elements of integer array a.
  ! Insertion sort is optimal for the small n typical of kriging neighbourhoods
  ! (nmax = 20-50): O(n) for nearly-sorted input, no allocation, cache-friendly.
  !============================================================================
  pure subroutine isort(a, n)
    integer, intent(inout) :: a(:)
    integer, intent(in)    :: n
    integer :: i, j, tmp
    do i = 2, n
      tmp = a(i)
      j   = i - 1
      do while (j >= 1 .and. a(j) > tmp)
        a(j+1) = a(j)
        j = j - 1
      end do
      a(j+1) = tmp
    end do
  end subroutine isort

  ! ===========================================================================
  ! wstore_check_stored
  !
  ! Check that the weights have been stored.
  !============================================================================
  function wstore_check_stored(self) result(stored)
    class(t_weight_store) :: self
    logical :: stored
    stored = .false.
    if (self%nblock == 0) return
    if (allocated(self%nnear) .and. allocated(self%inear) .and. allocated(self%weight)) then
      stored = any(self%nnear /= 0)
    end if
  end function wstore_check_stored

end module kriging
