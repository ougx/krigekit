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
!   ! read results from k%block%estimate and k%block%variance
!   call k%finalize()         ! release memory
!
! Parallelism
! -----------
! The block loop inside solve() is parallelised with OpenMP.  Each thread
! owns a private t_kriging_ctx (context) object that holds its own copy of
! the working arrays (matrix, RHS, neighbour indices, weights).  The shared
! state (obs, grid, block, vgm) is read-only during the parallel region.
! SGSIM disables OMP because each block conditions on previously simulated
! values written into the shared block%estimate array.
!
! Key design choices
! ------------------
! * Variogram array vgm(ivar0:nvar, ivar0:nvar): square matrix of vgm_struct.
!   For kriging ivar0=1; for SGSIM ivar0=0 so that previously simulated
!   blocks (index 0) can share the primary variogram via vgm(0,0)=vgm(1,1).
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
  ! Using a common base keeps the covariance assembly generic: calc_covariance
  ! accepts class(t_data) pointers and can handle obs, grid or block nodes
  ! without branching on the concrete type.
  !============================================================================
  type t_data
    integer              :: n = 0        ! number of spatial nodes
    real, allocatable    :: coord(:,:)   ! coordinates            [ndim, n]
    real, allocatable    :: drift(:,:)   ! drift function values  [ndrift, n]
    real, allocatable    :: value(:)     ! variable values        [n]
    real, allocatable    :: variance(:)  ! per-node variance: obs error or kriging variance [n]
  end type t_data

  !============================================================================
  ! t_grid — integration / sub-block nodes
  !
  ! For point kriging (block_type=0): one node per block, weight=1.
  ! For block kriging (block_type=-4 or >0): multiple nodes per block,
  ! weights set by Gaussian quadrature or supplied by the user.
  ! The weight array is used in calc_covariance to average the RHS covariance
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
  !                written to estimate(:, order(ib)) in the original grid order.
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
    real, allocatable    :: estimate(:,:)      ! kriging or SGSIM result  [nsim, n]
    integer, allocatable :: order(:)           ! visit order              [n]
    integer, allocatable :: nblockpnt(:)       ! nodes per block          [n]
    integer, allocatable :: iblockpnt(:)       ! start index in grid      [n]
    real, allocatable    :: rangescale(:)      ! variogram range scaler   [n]
    real, allocatable    :: localnugget(:)     ! extra diagonal nugget    [n]
    real, allocatable    :: sample(:,:)        ! N(0,1) draws for SGSIM  [nsim, n]
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
    real                 :: rotmat(3,3)         ! anisotropy rotation matrix
    type(kdtree2), pointer :: tree => null()   ! k-d tree for fast NN search
    logical              :: need_search = .false.      ! .true. if nmax < n
    logical              :: anisotropic_search = .false. ! search in rotated coords
  end type t_obsgrid

  !============================================================================
  ! t_kriging — main kriging object
  !
  ! Holds all problem-level state and provides the full API.  Thread-local
  ! working arrays are NOT stored here; they live in t_kriging_ctx so that
  ! multiple threads can work on different blocks simultaneously.
  !
  ! ivar0 index convention
  ! ----------------------
  ! obs and vgm are indexed from ivar0 to nvar.
  ! ivar0 = 1 for ordinary/simple kriging.
  ! ivar0 = 0 for SGSIM: obs(0)%coord holds the extended coordinate array
  !   (original obs + all block centres) so that the k-d tree can return
  !   previously simulated blocks as neighbours.  vgm(0,0) is a copy of
  !   vgm(1,1) so the same covariance function is used for simulated-block
  !   conditioning.
  !============================================================================
  type :: t_kriging
    !-- Boolean flags controlling solver behaviour
    logical              :: anisotropic_search = .false. ! use rotated coords for NN search
    logical              :: weight_correction  = .false. ! clip negative weights to 0 and renorm
    logical              :: use_old_weight     = .false. ! read weights from factor file
    logical              :: store_weight       = .false. ! write weights to factor file
    logical              :: cross_validation   = .false. ! leave-one-out cross-validation mode
    logical              :: write_mat          = .false. ! dump matrices to CSV for debugging
    logical              :: verbose            = .false. ! print progress to stdout
    logical              :: neglect_error      = .false. ! set NaN instead of stopping on singular
    logical              :: varying_vgm        = .false. ! use different vgm per block

    !-- File path for factor file (weight storage/reload)
    character(len=1024)  :: weight_file = ""
    integer              :: ifile = 0             ! Fortran unit for weight file

    !-- Problem dimensions
    integer              :: ndim   = 2            ! spatial dimension (1, 2, or 3)
    integer              :: nvar   = 1            ! number of co-kriging variables
    integer              :: ivar0  = 1            ! first obs/vgm index (0 for SGSIM)
    integer              :: ndrift = 0            ! number of drift functions
    integer              :: unbias = 1            ! 1=ordinary kriging, 0=simple kriging
    integer              :: nsim   = 0            ! simulations per block (0=kriging only)

    !-- Scratch / bookkeeping
    integer              :: iblock = 0            ! current block index (used in serial SGSIM)
    integer              :: nppmax = 0            ! max total neighbours across all variables
    integer              :: matsize_max = 0       ! nppmax + ndrift + unbias

    !-- Bounds for simulated/estimated values
    real                 :: bounds(2) = [-verylarge, verylarge]

    !-- Simple kriging mean (used when unbias=0)
    real                 :: sk_mean = 0.0

    character(kind=c_char), pointer  :: krige_info(:) ! kriging info string
    !-- Pointers to the three spatial objects and the variogram matrix
    type(t_obsgrid)  , pointer :: obs(:)          ! observations  [ivar0:nvar]
    type(t_grid)     , pointer :: grid            ! integration nodes
    type(t_blockgrid), pointer :: block           ! estimation targets
    type(vgm_struct) , pointer :: vgm(:,:,:)      ! variogram models [ivar0:nvar, ivar0:nvar, 1] last dimension can be nblock for spatial varying vgm
  contains
    procedure :: initialize
    procedure :: set_obs
    procedure :: set_obs_drift
    procedure :: set_vgm
    procedure :: set_grid
    procedure :: set_grid_drift
    procedure :: set_sim
    procedure :: set_search
    procedure :: search_neighbors
    procedure :: calc_covariance
    procedure :: assemble_linear_system
    procedure :: solve_linear_system
    procedure :: estimate_block
    procedure :: prepare
    procedure :: solve
    procedure :: write_weight
    procedure :: read_weight
    procedure :: validate_vgm
    procedure :: reset_obs
    procedure :: reset_grid
    procedure :: reset_block
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
  ! These arrays are indexed 0:nvar.  Index 0 is reserved for SGSIM:
  !   nnear(0)     : number of previously simulated blocks in the neighbourhood
  !   inear(:, 0)  : their indices into block%estimate
  !   weight(:, 0) : their kriging weights
  ! Indices 1:nvar hold the corresponding quantities for each observation variable.
  !============================================================================
  type :: t_kriging_ctx
    integer              :: iblock           ! current block index
    integer              :: npp              ! total neighbours = sum(nnear(ivar0:nvar))
    integer              :: matsize          ! npp + ndrift + unbias (actual, this block)
    integer, allocatable :: nnear(:)         ! neighbour count per variable [0:nvar]
    integer, allocatable :: inear(:,:)       ! neighbour indices            [nmax, 0:nvar]
    real,    allocatable :: weight(:,:)      ! kriging weights              [nmax, 0:nvar]
    real,    allocatable :: sqdist(:,:)      ! squared distances to neighbours [nmax, 0:nvar]
    real,    allocatable :: x(:,:)           ! raw solver output (weights + multipliers) [1, matsize]
    real,    allocatable :: matA(:,:)        ! covariance matrix C          [matsize, matsize]
    real,    allocatable :: rhsB(:,:)        ! right-hand-side c0           [1, matsize]
    type(vgm_struct), pointer :: vgm(:,:) => null() ! thread-private variogram slice
  contains
    procedure :: initialize  => initialize_kriging_ctx
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
  !   obs  (ivar0:nvar)          one t_obsgrid per variable
  !   grid                       single t_grid  (populated by set_grid)
  !   block                      single t_blockgrid (populated by set_grid)
  !   vgm  (ivar0:nvar, ivar0:nvar)  one vgm_struct per variable pair
  !
  ! SGSIM path: if nsim>0, ivar0 is set to 0 so obs(0) can be used to hold
  ! the extended coordinate array (obs + block centres) for SGSIM neighbour
  ! search, and vgm(0,:) / vgm(:,0) get copies of the primary variogram.
  !============================================================================
  subroutine initialize(self, ndim, nvar, ndrift, unbias, nsim, anisotropic_search,  &
                        weight_correction, use_old_weight, store_weight, cross_validation, &
                        write_mat, neglect_error, varying_vgm, verbose, &
                        weight_file, bounds, sk_mean, seed)
    class(t_kriging)                         :: self
    integer, intent(in), optional            :: ndim
    integer, intent(in), optional            :: nvar, ndrift, unbias, nsim, seed
    real,    intent(in), optional            :: bounds(2), sk_mean
    logical, intent(in), optional            :: anisotropic_search, weight_correction, &
                                                use_old_weight, write_mat, store_weight, &
                                                verbose, cross_validation, neglect_error
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
    if (present(sk_mean))            self%sk_mean            = sk_mean
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

    !-- SGSIM: extend index range to 0 so obs(0) / vgm(0,:) are available
    if (self%nsim > 0) self%ivar0 = 0

    allocate(self%obs  (self%nvar))
    allocate(self%grid)
    allocate(self%block)

    !-- Sanity checks: mutually exclusive flag combinations
    if (self%use_old_weight .and. self%weight_file == "") &
      call kriging_error(subname, 'use_old_weight requires weight_file to be specified')
    if (self%store_weight .and. self%weight_file == "") &
      call kriging_error(subname, 'store_weight requires weight_file to be specified')
    if (self%store_weight .and. self%use_old_weight) &
      call kriging_error(subname, 'store_weight and use_old_weight are mutually exclusive')
    if (self%cross_validation .and. self%nsim > 0) &
      call kriging_error(subname, 'nsim>0 and cross_validation are mutually exclusive')
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

    integer :: ngrid, nn, nb, iblock, igrid, idim
    character(len=*), parameter :: subname = "t_kriging%set_grid"


    call self%reset_grid()
    call self%reset_block()

    if (self%obs(1)%n == 0) &
      call kriging_error(subname, 'Observation needs to be set first.')

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
        if (.not. present(coord)) &
          call kriging_error(subname, 'coord needs to be provided.')

        !-- validate ndim
        if (ndim /= size(coord, 1)) &
          call kriging_error(subname, 'ndim /= size(coord, 1) for self%grid')
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
          if (.not. present(blocksize)) &
            call kriging_error(subname, 'blocksize needs to be provided when block_type=-4.')
          nb = 4**ndim        ! integration points per block (4-point Gauss per dimension)
          self%block%n = ngrid
          allocate(self%block%coord, source = coord)
          allocate(self%grid%coord(ndim, ngrid * nb))
          allocate(self%block%nblockpnt(ngrid));   self%block%nblockpnt = nb
          allocate(self%block%iblockpnt, source = [((igrid-1)*nb+1, igrid = 1, ngrid)])
          allocate(self%grid%weight(nb * ngrid))
          igrid = 0
          do iblock = 1, self%block%n
            !-- TODO: replace equal weights with true Gaussian quadrature weights
            self%grid%weight(igrid+1:igrid+nb) = 1.0 / nb
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
      allocate(self%block%estimate   (max(self%nsim, 1), self%block%n))
      allocate(self%block%variance   (self%block%n))

      !-- Default sequential visit order (overridden by set_sim for SGSIM)
      call set_seq(self%block%order, self%block%n)

      !-- Initialise outputs to NaN so unfilled blocks are detectable
      self%block%variance = IEEE_VALUE(0.0, IEEE_QUIET_NAN)
      self%block%estimate = IEEE_VALUE(0.0, IEEE_QUIET_NAN)

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
    if (self%varying_vgm) then
      allocate(self%vgm(self%ivar0:self%nvar, self%ivar0:self%nvar, self%block%n))
    else
      allocate(self%vgm(self%ivar0:self%nvar, self%ivar0:self%nvar, 1))
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

    if (.not. associated(self%block)) &
      call kriging_error(subname, 'Call initialize() before set_grid_drift.')
    if (self%block%n == 0) &
      call kriging_error(subname, 'Grid needs to be set before adding drift.')
    if (self%ndrift == 0) &
      call kriging_error(subname, 'grid/block drift is specified but ndrift==0')
    if (size(drift, 1) /= self%ndrift) &
      call kriging_error(subname, 'size(drift, 1) /= ndrift')
    if (size(drift, 2) /= self%block%n) &
      call kriging_error(subname, 'size(drift, 2) /= block%n; one drift value per block, not per grid node')
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
    if (.not. associated(self%block)) &
      call kriging_error(subname, 'Call initialize() before set_vgm.')
    if (self%block%n == 0) &
      call kriging_error(subname, 'Grid needs to be set before adding variogram.')

    vtype_    = 'sph'    ; if (present(vtype   )) vtype_ = 'sph'
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
      mb= size(self%vgm, 3)
    end if

    do ib_ = ib0, mb
      if (jvar == ivar) then
        call self%vgm(jvar, ivar, ib_)%add_args(trim(vtype_), nugget_, sill_, a_major_, a_minor1_, a_minor2_, azimuth_, dip_, plunge_)
      else if (jvar > ivar) then
        !-- Fill both triangle entries (cross-variogram is symmetric)
        call self%vgm(jvar, ivar, ib_)%add_args(trim(vtype_), nugget_, sill_, a_major_, a_minor1_, a_minor2_, azimuth_, dip_, plunge_)
        call self%vgm(ivar, jvar, ib_)%add_args(trim(vtype_), nugget_, sill_, a_major_, a_minor1_, a_minor2_, azimuth_, dip_, plunge_)
      else
        call kriging_error(subname, 'jvar must be >= ivar to set the upper triangle of the variogram matrix')
      end if
    end do
  end subroutine set_vgm


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
  !============================================================================
  subroutine set_obs(self, ivar, coord, value, variance, nmax, maxdist)
    use rotation,       only: rotate
    use kdtree2_module, only: kdtree2_create
    class(t_kriging)              :: self
    integer, intent(in)           :: ivar
    integer, intent(in), optional :: nmax
    real,    intent(in)           :: coord(:,:), value(:)
    real,    intent(in), optional :: variance(:), maxdist
    character(len=*), parameter   :: subname = "t_kriging%set_obs"

    call self%reset_obs(ivar)
    associate(ndim => self%ndim, obs => self%obs(ivar))
      !-- Infer or validate ndim from coord
      if (ndim == 0) then
        ndim = size(coord, 1)
      else
        if (ndim /= size(coord, 1)) &
          call kriging_error(subname, 'ndim /= size(coord, 1) for grid')
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

      !-- Observation error variance: default to 0 (exact observations)
      if (present(variance)) then
        allocate(obs%variance, source = variance)
      else
        allocate(obs%variance(obs%n))
        obs%variance = 0.0
      end if

      allocate(obs%value, source = value)
      allocate(obs%coord, source = coord)

      !-- Identity rotation matrix (updated by set_search for anisotropic search)
      obs%rotmat = reshape([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], [3, 3])
    end associate
  end subroutine set_obs


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

    if (.not. associated(self%obs)) &
      call kriging_error(subname, 'Call initialize() before set_obs_drift.')
    if (self%obs(ivar)%n == 0) &
      call kriging_error(subname, 'Observation needs to be set before adding drift.')
    if (self%ndrift == 0) &
      call kriging_error(subname, 'Observation drift is specified but ndrift==0')
    if (size(drift, 1) /= self%ndrift) &
      call kriging_error(subname, 'size(drift, 1) /= ndrift')
    if (size(drift, 2) /= self%obs(ivar)%n) &
      call kriging_error(subname, 'size(drift, 2) /= nobs')
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
  ! block%sample(isim, ib) holds the pre-drawn N(0,1) value for simulation
  ! isim at block ib.  The simulation draw at block ib for realization isim is:
  !   z_sim = z_kriging_estimate + sqrt(kriging_variance) * sample(isim, ib)
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
    real,    intent(in), optional :: sample(:,:)   ! pre-drawn N(0,1) [nsim, nblock]
    integer, intent(in), optional :: randpath(:)   ! user-supplied visit order [nblock]

    real,    allocatable :: temp(:,:), samp(:)
    integer              :: iblock, ifile, isim
    character(len=*), parameter :: subname = "t_kriging%set_sim"

    if (self%nsim > 0) then
      if (self%block%n == 0) call kriging_error(subname, 'Grid needs to be set first.')
      if (any(self%obs%n == 0)) call kriging_error(subname, 'Observations need to be set first.')
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
        allocate(self%block%sample(self%nsim, self%block%n))
        if (present(sample)) then
          self%block%sample = sample
        else
          allocate(samp(self%block%n))
          do isim = 1, self%nsim
            call r8vec_normal_01(self%block%n, samp)
            self%block%sample(isim, :) = samp
          end do
          !-- Write samples to file for reproducibility
          open(newunit=ifile, file='sgs_sample.dat', status='replace')
          write(ifile, '(A,x,2I10)') 'SGSIM_Sample', self%nsim, self%block%n
          do iblock = 1, self%block%n
            write(ifile, '(*(G0.7,x))') self%block%sample(:, iblock)
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

        !-- Extend obs(1)%coord to include all block centres so the k-d tree
        !   can return previously simulated blocks as neighbours.
        !   After this call, obs%coord has size (ndim, nobs + nblock).
        !   During search_neighbors, only entries with index <= nobs + ib - 1
        !   are eligible (the max_idx filter enforces this).
        allocate(temp(ndim, obs%n + self%block%n))
        temp(:, 1:obs%n)         = obs%coord
        temp(:, obs%n+1:)        = self%block%coord
        call move_alloc(temp, obs%coord)
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

    real, allocatable :: rcoord(:,:)   ! rotated coordinates for anisotropic tree

    associate( &
      ndim               => self%ndim, &
      obs                => self%obs(ivar), &
      need_search        => self%obs(ivar)%need_search, &
      anisotropic_search => self%obs(ivar)%anisotropic_search)

      !-- Precompute 3×3 rotation+scale matrix from variogram angles
      obs%rotmat = calc_rotmat(azimuth, dip, plunge, anis1, anis2)

      !-- Activate anisotropic search only when there is meaningful anisotropy
      anisotropic_search = (abs(anis1 - 1.0) > EPSLON .or. abs(anis2 - 1.0) > EPSLON) &
                           .and. self%anisotropic_search

      !-- Determine effective nmax, accounting for SGSIM's extended obs array
      if (ivar == 1 .and. self%nsim > 0) then
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
          !-- Project coordinates into anisotropically scaled space before indexing
          allocate(rcoord, mold = obs%coord)
          call sub_rotate(obs%rotmat, ndim, size(obs%coord, 2), obs%coord, rcoord)
          obs%tree => kdtree2_create(rcoord, sort = .false., rearrange = .true.)
        else
          obs%tree => kdtree2_create(obs%coord, sort = .false., rearrange = .true.)
        end if
      end if
    end associate
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

    integer :: ivar, mmax
    mmax = maxval(krige%obs%nmax)   ! max neighbours across all variables

    associate(npp => krige%nppmax, matsize => krige%matsize_max)
      if (.not. krige%use_old_weight) then
        allocate(self%sqdist(mmax,    0:krige%nvar))
        allocate(self%matA  (matsize, matsize))
        allocate(self%rhsB  (1,       matsize))
        self%sqdist = 0.0
      end if
      allocate(self%nnear (     0:krige%nvar))
      allocate(self%inear (mmax,0:krige%nvar))
      allocate(self%weight(mmax,0:krige%nvar))
      allocate(self%x     (1,      matsize))
      self%weight = 0.0
      self%x      = 0.0

      !-- Default: start as if all observations are neighbours (no search needed)
      self%nnear(0) = 0
      call set_seq(self%inear(1:mmax, 0), mmax)
      do ivar = 1, krige%nvar
        self%nnear(ivar)  = krige%obs(ivar)%nmax
        self%inear(:,ivar) = self%inear(:, 0)
      end do
    end associate
    self%vgm => krige%vgm(:,:,1)
  end subroutine initialize_kriging_ctx


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
  ! Weight file: opens for reading (use_old_weight) or writing (store_weight).
  !============================================================================
  subroutine prepare(self)
    class(t_kriging) :: self
    ! local
    integer          :: ivar, jvar, ib, mb
    character(len=*), parameter :: subname = "t_kriging%prepare"


    !-- Validate that all required arrays have been provided
    if (self%ndrift > 0) then
      if (.not. allocated(self%block%drift)) &
        call kriging_error(subname, 'Grid drift is not set while ndrift > 0.')
      do ivar = 1, self%nvar
        if (.not. allocated(self%obs(ivar)%drift)) &
          call kriging_error(subname, 'Observation drift is not set while ndrift > 0.')
      end do
    end if

    call self%validate_vgm()
    if (self%nsim > 0 .and. size(self%obs(1)%coord, 2) == self%obs(1)%n) &
      call kriging_error(subname, 'set_sim() needs to be called before solve().')

    associate(npp => self%nppmax, matsize => self%matsize_max, ifile => self%ifile)

      !-- Total neighbours and matrix size for worst-case allocation
      npp = 0
      do ivar = 1, self%nvar
        npp = npp + self%obs(ivar)%nmax
      end do
      matsize = npp + self%ndrift + self%unbias

      !-- Open weight file
      if (self%use_old_weight) then
        open(newunit=ifile, file=trim(self%weight_file), status='old')
        read(ifile, *)   ! skip header line
      else if (self%store_weight) then
        open(newunit=ifile, file=trim(self%weight_file), status='replace')
        write(ifile, *) self%block%n, self%nvar, (self%obs(ivar)%nmax, ivar=1, self%nvar)
      end if

      !-- SGSIM: propagate primary variogram to index-0 slots so that
      !   covariances between observation and previously simulated blocks
      !   use the same model as covariances between observations
      if (self%nsim > 0) then
        mb = merge(self%block%n, 1, self%varying_vgm)
        do ib = 1, mb
          self%vgm(0, 0, ib) = self%vgm(1, 1, ib)
          do ivar = 1, self%nvar
            self%vgm(0, ivar, ib) = self%vgm(1, ivar, ib)
            self%vgm(ivar, 0, ib) = self%vgm(ivar, 1, ib)
          end do
        end do
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
  ! reads block%estimate for blocks 1..ib-1 (already simulated).  Parallelising
  ! over blocks would race on the shared estimate array.  The IF clause on the
  ! OMP PARALLEL directive disables OMP when nsim>0 or store_weight=.true.
  ! (store_weight must be sequential to preserve factor-file line order).
  !============================================================================
  subroutine solve(self)
    use omp_lib
    class(t_kriging)          :: self
    type(t_kriging_ctx), allocatable :: ctx
    integer                   :: ib
    real, allocatable          :: temp(:,:)
    character(len=*), parameter :: subname = "t_kriging%solve"

    call self%prepare()

    associate(nb => self%block%n, verbose => self%verbose)
      if (verbose) print*, "Starting Kriging loop"
#ifdef __INTEL_COMPILER
      if (verbose) open(unit=6, carriagecontrol='fortran')
#endif

      !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(ctx) IF(self%nsim==0 .and. .not. (self%store_weight .or. self%use_old_weight))
      allocate(ctx)                        ! each thread gets its own ctx
      call ctx%initialize(self)

      !$OMP DO SCHEDULE(DYNAMIC, 1)
      do ib = 1, nb
        !-- Progress bar: only the last thread prints to avoid interleaved output
#ifdef _OPENMP
        if (verbose .and. omp_get_thread_num() == omp_get_num_threads()-1) call progress(ib, nb)
#else
        if (verbose) call progress(ib, nb)
#endif
        ctx%iblock = ib

        !-- Point ctx%vgm at this block's variogram slice (thread-private).
        if (self%varying_vgm) ctx%vgm => self%vgm(:,:,ib)
        if (self%use_old_weight) then
          !-- Factor-file path: read pre-computed weights, skip the solve
          call self%read_weight(ctx)
        else
          call self%assemble_linear_system(ctx)
          !-- Skip the matrix solve when only one neighbour exists (trivial case)
          if (ctx%npp > 1) call self%solve_linear_system(ctx)
          call ctx%assign_weight(self)
        end if

        if (self%store_weight) call self%write_weight(ctx)
        call self%estimate_block(ctx)
        if (self%write_mat) call ctx%write_matrix(self)
      end do
      !$OMP END DO
      deallocate(ctx)     ! explicit per-thread cleanup; avoids crash on runtime auto-finalization of PRIVATE allocatables
      !$OMP END PARALLEL

#ifdef __INTEL_COMPILER
      if (verbose) close(6)
#else
      if (verbose) print *, ""   ! newline after progress bar
#endif
      if (verbose) print*, "Kriging completed."

      !-- SGSIM post-processing: blocks were processed in random path order;
      !   reorder coord and estimate back to the original block indices so
      !   downstream code can use block%estimate(isim, ib) at the correct location.
      if (self%nsim > 0) then
        allocate(temp(self%nsim + self%ndim, self%block%n))
        temp(1:self%ndim, :)       = self%block%coord
        temp(self%ndim+1:, :)      = self%block%estimate
        do ib = 1, self%block%n
          self%block%coord   (:, self%block%order(ib)) = temp(1:self%ndim,    ib)
          self%block%estimate(:, self%block%order(ib)) = temp(self%ndim+1:,   ib)
        end do
      end if
    end associate
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

    integer                     :: i, k
    real                        :: newloc(self%ndim, 1)
    logical, allocatable        :: is_obs(:)
    type(kdtree2_result)        :: results(self%obs(ivar)%nmax)
    character(len=*), parameter :: subname = "t_kriging%search_neighbors"

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
      dist    => ctx%sqdist(:, ivar), &
      maxdist => self%obs(ivar)%maxdist, &
      inearb  => ctx%inear(:, 0), &      ! previously simulated blocks
      nnearb  => ctx%nnear(0), &
      distb   => ctx%sqdist(:, 0), &
      rotmat  => self%obs(ivar)%rotmat)

      !-- Project target location if anisotropic search is active
      if (self%obs(ivar)%anisotropic_search) then
        call sub_rotate(rotmat, ndim, 1, xloc, newloc)
      else
        newloc = xloc
      end if

      !------------------------------------------------------------------------
      ! SGSIM neighbour search: includes both original obs and prior simulated blocks
      !------------------------------------------------------------------------
      if (nsim > 0 .and. ivar == 1) then
        if (nmax < nobs + iblock - 1) then
          !-- k-d tree query with max_idx filter: only returns entries < nobs+iblock
          call kdtree2_n_nearest_maxidx(self%obs(ivar)%tree, newloc(:,1), nmax, results, nobs+iblock-1)
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
          dist (1:nnear)  = rotated_dists(rotmat, ndim, newloc(:,1), obsloc(:, 1:nnear))
          distb(1:nnearb) = rotated_dists(rotmat, ndim, newloc(:,1), self%block%coord(:, 1:nnearb))
        end if

      !------------------------------------------------------------------------
      ! Standard kriging / cokriging neighbour search
      !------------------------------------------------------------------------
      else
        if (nmax < nobs) then
          call kdtree2_n_nearest(self%obs(ivar)%tree, newloc(:,1), nmax, results)
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
          dist (k) = dist (i)
        end if
      end do
      nnear = k
    end associate
#ifdef DEBUG
    print *, subname, " Finished.", ivar, ctx%iblock
#endif
  end subroutine search_neighbors


  !============================================================================
  ! calc_covariance
  !
  ! Fill one block of the kriging matrix (matA) or the right-hand-side (rhsB).
  !
  ! Called by assemble_linear_system in two modes:
  !
  !   jvar == -1  (RHS mode)
  !     Fills rhsB(1, ir0+1:ir0+nnear(ivar)) with the covariance between
  !     each neighbour of variable ivar and the target block x0.  For block
  !     kriging, this is the weighted average over all integration nodes:
  !       c0(i) = sum_k weight(k) * C(obs_i, grid_k)
  !     For point kriging (nblockpnt=1) this reduces to a single covariance.
  !
  !   jvar >= 0  (LHS mode)
  !     Fills the (ivar, jvar) sub-block of matA with covariances between
  !     all neighbours of ivar and all neighbours of jvar.
  !     For the diagonal block (ivar==jvar), the diagonal entry is C(0) plus
  !     the observation error variance (obs%variance) and the local nugget,
  !     implementing heteroscedastic measurement error.
  !     Off-diagonal entries use lag-based covariance cov_lag(lag).
  !     By symmetry only the upper triangle (jvar>=ivar) is computed;
  !     assemble_linear_system mirrors the lower triangle afterward.
  !
  ! Range scaler
  ! ------------
  ! All lag vectors are divided by rangescale(iblock) before passing to the
  ! variogram function.  A scale < 1 effectively tightens the variogram (makes
  ! it shorter-range), which can represent data-sparse areas where spatial
  ! continuity is locally reduced.
  !============================================================================
  subroutine calc_covariance(self, ctx, ir0, ic0, ivar, jvar)
    class(t_kriging)     :: self
    class(t_kriging_ctx) :: ctx
    integer, intent(in)  :: ivar, jvar   ! variable indices; jvar=-1 for RHS
    integer, intent(in)  :: ir0, ic0     ! row / column offsets into matA / rhsB

    integer              :: i, j, k, k1, istart
    real                 :: lag(3), tmp
    class(t_data), pointer :: obs1, obs2
    character(len=*), parameter :: subname = "t_kriging%calc_covariance"

    lag = 0.0
    associate( &
      ndim  => self%ndim, &
      nnear => ctx%nnear(ivar), &
      inear => ctx%inear(1:ctx%nnear(ivar), ivar), &
      rs    => self%block%rangescale (ctx%iblock), &  ! variogram range scaler
      ln    => self%block%localnugget(ctx%iblock))    ! local nugget

      !-- Resolve obs1: variable ivar (or block for SGSIM conditioning)
      if (ivar == 0) then
        obs1 => self%block
      else
        obs1 => self%obs(ivar)
      end if

      !------------------------------------------------------------------------
      ! RHS mode: covariance between each neighbour and the block to estimate
      !------------------------------------------------------------------------
      if (jvar == -1) then
        associate(vgm => ctx%vgm(1, ivar))
          do i = 1, nnear
            tmp = 0.0
            k1 = self%block%iblockpnt(ctx%iblock) - 1
            !-- Average covariance over block integration nodes (block kriging)
            do k = 1, self%block%nblockpnt(ctx%iblock)
              lag(1:ndim) = (obs1%coord(:, inear(i)) - self%grid%coord(:, k1+k)) / rs
              tmp = tmp + vgm%cov_lag(lag) * self%grid%weight(k1+k)
            end do
            ctx%rhsB(1, ir0+i) = tmp
          end do
        end associate

      !------------------------------------------------------------------------
      ! LHS mode: covariance matrix block between variable ivar and jvar
      !------------------------------------------------------------------------
      else
        associate( &
          nnear2 => ctx%nnear(jvar), &
          inear2 => ctx%inear(1:ctx%nnear(jvar), jvar), &
          vgm    => ctx%vgm(jvar, ivar))

          if (jvar == 0) then
            obs2 => self%block
          else
            obs2 => self%obs(jvar)
          end if

          do i = 1, nnear
            if (ivar == jvar) then
              !-- Diagonal: C(0) + observation error variance + local nugget
              istart = i + 1
              ctx%matA(ic0+i, ir0+i) = vgm%cov0 + obs1%variance(inear(i)) + ln
            else
              istart = 1
            end if
            !-- Off-diagonal: C(lag) between obs i of ivar and obs j of jvar
            do j = istart, nnear2
              lag(1:ndim) = (obs1%coord(:, inear(i)) - obs2%coord(:, inear2(j))) / rs
              ctx%matA(ic0+j, ir0+i) = vgm%cov_lag(lag)
            end do
          end do
        end associate
      end if
    end associate
#ifdef DEBUG
    print *, subname, " Finished.", ctx%iblock,  ir0, ic0, ivar, jvar
#endif
  end subroutine calc_covariance


  !============================================================================
  ! assemble_linear_system
  !
  ! Build the full kriging matrix (matA) and RHS vector (rhsB) for block ib.
  !
  ! Matrix layout (ordinary kriging, two variables)
  ! ------------------------------------------------
  !   [ C11  C12  1  0 ] [ lambda1 ]   [ c01 ]
  !   [ C21  C22  0  1 ] [ lambda2 ] = [ c02 ]
  !   [ 1^T  0    0  0 ] [ mu1     ]   [ 1   ]
  !   [ 0    1^T  0  0 ] [ mu2     ]   [ 1   ]
  !
  ! For simple kriging (unbias=0): the last two rows/columns are omitted.
  ! For drift functions (ndrift>0): additional rows/columns are inserted after
  ! the covariance block and before the unbiasedness row.
  !
  ! Exact-match detection
  ! ---------------------
  ! If any observation exactly coincides with the block centre (distance <= EPSLON),
  ! the system is bypassed: the weight is set to 1 for that observation and 0
  ! elsewhere, and the kriging variance is set to the observation error variance
  ! plus the local nugget (not zero, because measurement error is unresolved).
  !
  ! Symmetry
  ! --------
  ! calc_covariance fills only the upper triangle (jvar >= ivar).
  ! After all blocks are filled, the lower triangle is mirrored from the upper.
  !============================================================================
  subroutine assemble_linear_system(self, ctx)
    class(t_kriging)     :: self
    class(t_kriging_ctx) :: ctx

    integer          :: ivar, jvar, irow1, irow2, icol1, icol2
    character(len=*), parameter :: subname = "t_kriging%assemble_linear_system"

    associate(nvar => self%nvar, dist => ctx%sqdist, npp => ctx%npp)

      !-- Find neighbours for each variable
      do ivar = 1, nvar
        call self%search_neighbors(ivar, ctx)

        !-- Exact match: an observation coincides with the block centre
        if (ivar == 1 .and. minval(dist(1:ctx%nnear(ivar), ivar)) <= EPSLON) then
          npp = 1
          ctx%x = 0.0;  ctx%x(:, 1) = 1.0
          ctx%weight = 0.0;  ctx%weight(1, 1) = 1.0
          ctx%inear(1, ivar) = ctx%inear(minloc(dist(1:ctx%nnear(ivar), ivar), dim=1), ivar)
          ctx%nnear(ivar) = 1;  ctx%nnear(0) = 0
          self%block%variance(ctx%iblock) = self%obs(1)%variance(ctx%inear(1, ivar)) &
                                           + self%block%localnugget(ctx%iblock)
          do jvar = 2, nvar; ctx%nnear(jvar) = 0; end do
          if (self%verbose) print*, "Exact match detected at block ", ctx%iblock
          return
        end if
      end do

      npp = sum(ctx%nnear)

      !-- Degenerate: no neighbours found at all
      if (ctx%nnear(0) + ctx%nnear(1) == 0) then
        call kriging_error(subname, 'not enough neighbors for kriging at block', iblock=ctx%iblock)
      end if

      !-- Assemble matrix blocks
      associate( &
        iblock  => ctx%iblock, &
        matA    => ctx%matA, &
        rhsB    => ctx%rhsB, &
        inear   => ctx%inear, &
        nnear   => ctx%nnear, &
        matsize => ctx%matsize, &
        ndrift  => self%ndrift)

        matsize = npp + self%unbias + ndrift

        !-- Loop over row-variable blocks
        irow1 = 0
        rowloop: do ivar = self%ivar0, nvar
          if (nnear(ivar) == 0) cycle
          irow2 = irow1 + nnear(ivar)
          icol1 = 0

          !-- RHS: covariance between ivar neighbours and the block
          call self%calc_covariance(ctx, irow1, icol1, ivar, -1)

          !-- Upper triangle of the covariance matrix
          columnloop: do jvar = self%ivar0, nvar
            if (nnear(jvar) == 0) cycle
            icol2 = icol1 + nnear(jvar)
            if (jvar >= ivar .and. nnear(jvar) > 0) then
              call self%calc_covariance(ctx, irow1, icol1, ivar, jvar)
            end if
            icol1 = icol2
          end do columnloop

          !-- Drift columns: D(ivar) = obs(ivar)%drift at neighbour locations
          if (ndrift > 0) then
            icol2 = icol1 + ndrift
            matA(icol1+1:icol2, irow1+1:irow2) = &
              self%obs(ivar)%drift(:, inear(1:nnear(ivar), ivar))
          end if
          irow1 = irow2
        end do rowloop

        !-- Drift row in the RHS: drift value at the block centre
        if (ndrift > 0) rhsB(1, npp+1:npp+ndrift) = self%block%drift(:, iblock)

        !-- Unbiasedness constraint row/column: sum(weights) = 1
        if (self%unbias == 1) then
          matA(matsize, 1:ctx%npp) = 1.0
          rhsB(1, matsize)         = 1.0
        end if

        !-- Mirror lower triangle from upper triangle (C is symmetric)
        do irow1 = 1, npp
          do icol1 = irow1+1, matsize
            matA(irow1, icol1) = matA(icol1, irow1)
          end do
        end do

        !-- Zero out the constraint-row diagonal block (no constraint-constraint cov)
        matA(npp+1:matsize, npp+1:matsize) = 0.0
      end associate
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
  ! Kriging variance
  ! ----------------
  !   sigma^2_k = C(0) - lambda^T * c0 - mu
  ! For point kriging (nblockpnt=1): C(0) = cov0.
  ! For block kriging: C(0) is the within-block variance, computed as the
  ! weighted sum of covariances among all integration node pairs.
  !============================================================================
  subroutine solve_linear_system(self, ctx)
    use solver
    implicit none
    class(t_kriging)     :: self
    class(t_kriging_ctx) :: ctx

    integer            :: info, i, j, k1
    real               :: lag(3)
    character(len=*), parameter :: subname = "t_kriging%solve_linear_system"

    lag = 0.0

    associate( &
      ndim      => self%ndim, &
      iblock    => ctx%iblock, &
      matA      => ctx%matA, &
      rhsB      => ctx%rhsB, &
      matsize   => ctx%matsize, &
      npp       => ctx%npp, &
      x         => ctx%x, &
      unbias    => self%unbias, &
      ndrift    => self%ndrift)

      !-- Primary solver: packed Cholesky (SSPSV)
      call kriging_solve(npp, unbias + ndrift, 1, matA, rhsB, x, info)

      !-- Fallback: symmetric indefinite solver (SSYSV)
      if (info /= 0) then
        call ssysv_fallback(npp, unbias + ndrift, 1, matA, rhsB, x, info)
        if (self%verbose) &
          print*, "Cholesky fails. Fallback SSYSV is used for block", iblock
      end if

      !-- Both solvers failed
      if (info /= 0) then
        call ctx%write_matrix(self)
        if (self%neglect_error) then
          x = IEEE_VALUE(0.0, IEEE_QUIET_NAN)
        else
          call kriging_error(subname, 'Singular matrix', iblock=ctx%iblock)
        end if
      end if

      !-- Optional: clip negative weights and renormalise
      if (self%weight_correction) then
        x(1, 1:npp) = merge(x(1, 1:npp), 0.0, x(1, 1:npp) > 0)
        x(1, 1:npp) = x(1, 1:npp) / sum(x(1, 1:npp))
      end if

      !-- Kriging variance: sigma^2 = C(0) - lambda^T * c0
      associate( &
        vgm        => ctx%vgm(1, 1), &
        var        => self%block%variance(iblock), &
        weight     => self%grid%weight, &
        coord      => self%grid%coord, &
        nblockpnt  => self%block%nblockpnt(iblock))

        if (nblockpnt == 1) then
          !-- Point kriging: within-block variance = C(0)
          var = vgm%cov0
        else
          !-- Block kriging: within-block variance = weighted sum of C(node_i, node_j)
          var = 0.0
          k1 = self%block%iblockpnt(iblock) - 1
          do i = 1, nblockpnt
            var = var + vgm%cov0 * weight(k1+i) * weight(k1+i)
            do j = i+1, nblockpnt
              lag(1:ndim) = coord(:, k1+i) - coord(:, k1+j)
              var = var + vgm%cov_lag(lag) * weight(k1+i) * weight(k1+j) * 2.0
            end do
          end do
        end if

        !-- Clamp to avoid tiny negative values from floating-point errors
        var = max(var - dot_product(x(1, 1:matsize), rhsB(1, 1:matsize)), 0.0)
      end associate
    end associate
#ifdef DEBUG
    print *, subname, " Finished. ", ctx%iblock
#endif
  end subroutine solve_linear_system


  !============================================================================
  ! assign_weight
  !
  ! Split the flat solution vector x(1, 1:npp) into per-variable weight arrays.
  ! x is ordered as [lambda(ivar0), ..., lambda(nvar)], each block of size
  ! nnear(ivar).  The Lagrange multiplier(s) at positions npp+1:matsize are
  ! not needed after this point.
  !============================================================================
  subroutine assign_weight(self, krige)
    class(t_kriging_ctx) :: self
    class(t_kriging)     :: krige

    integer :: ivar, k1
    character(len=*), parameter :: subname = "t_kriging%assign_weight"

    k1 = 0
    do ivar = krige%ivar0, krige%nvar
      if (self%nnear(ivar) == 0) cycle
      self%weight(1:self%nnear(ivar), ivar) = self%x(1, k1+1:k1+self%nnear(ivar))
      k1 = k1 + self%nnear(ivar)
    end do
#ifdef DEBUG
    print *, subname, " Finished. ", self%iblock
#endif
  end subroutine assign_weight


  !============================================================================
  ! estimate_block
  !
  ! Compute the kriging estimate (or SGSIM realisation) for the current block.
  !
  ! Kriging estimate
  ! ----------------
  !   z*(x0) = sum_{ivar} sum_i lambda(i,ivar) * z(x_i, ivar)
  !
  ! For simple kriging (unbias=0) with a known mean:
  !   z*(x0) += (1 - sum(lambda)) * sk_mean
  !
  ! For co-kriging (nvar>1) with unbias=1, the ISAAKS and SRIVASTAVA (1989)
  ! correction is applied to secondary variable weights to ensure global
  ! unbiasedness when the secondary variable has a different local mean.
  !
  ! SGSIM draw
  ! ----------
  !   z_sim = z_est + sqrt(kriging_variance) * sample(isim, iblock)
  ! sample was pre-drawn in set_sim.  This produces a realisation that is
  ! conditionally unbiased (mean = z_est) with variance = kriging_variance.
  !
  ! Bounds
  ! ------
  ! Result is clamped to [bounds(1), bounds(2)] across all nsim realisations.
  !============================================================================
  subroutine estimate_block(self, ctx)
    implicit none
    class(t_kriging)      :: self
    class(t_kriging_ctx)  :: ctx

    integer           :: ivar, k, nx, nnearb
    real, allocatable :: v(:), w(:)
    real              :: avg(max(1, self%nsim)), total_weight(self%ivar0:self%nvar)
    character(len=*), parameter :: subname = "t_kriging%estimate_block"

    nx = max(1, self%nsim)
    associate( &
      var    => self%block%variance(   ctx%iblock), &
      val    => self%block%estimate(:, ctx%iblock), &
      nnear  => ctx%nnear, &
      inear  => ctx%inear, &
      weight => ctx%weight)

      val = 0.0
      avg = 0.0

      !-- SGSIM: add weighted contributions from previously simulated blocks
      if (self%nsim > 0) then
        do k = 1, nnear(0)
          val = val + self%block%estimate(:, inear(k,0)) * weight(k, 0)
          avg = avg + self%block%estimate(:, inear(k,0))
        end do
        total_weight(0) = sum(weight(1:nnear(0), 0))
        nnearb = nnear(0)
      else
        nnearb = 0
      end if

      !-- Co-kriging: compute local mean of primary for the ISAAKS correction
      if (self%nvar > 1) then
        avg = avg + self%obs(1)%value(inear(1:nnear(1), 1))
        avg = avg / (nnearb + nnear(1))
      end if

      !-- Add weighted observation contributions for each variable
      do ivar = 1, self%nvar
        if (nnear(ivar) == 0) then
          total_weight(ivar) = 0.0
          cycle
        end if
        v = self%obs(ivar)%value(inear(1:nnear(ivar), ivar))
        w =                      weight(1:nnear(ivar), ivar)
        val = val + dot_product(w, v)
        total_weight(ivar) = sum(w)
        !-- ISAAKS & SRIVASTAVA co-kriging correction (secondary variable, OK only)
        if (self%unbias /= 0 .and. ivar > 1) &
          val = val + total_weight(ivar) * (avg - sum(v)/nnear(ivar))
      end do

      !-- Simple kriging mean correction
      if (self%unbias == 0 .and. self%sk_mean /= 0.0) &
        val = val + (1.0 - sum(total_weight)) * self%sk_mean

      !-- SGSIM stochastic perturbation
      if (self%nsim > 0) &
        val = val + sqrt(var) * self%block%sample(:, ctx%iblock)

      !-- Clamp to bounds
      where(val < self%bounds(1)) val = self%bounds(1)
      where(val > self%bounds(2)) val = self%bounds(2)
    end associate
#ifdef DEBUG
    print *, subname, " Finished. ", ctx%iblock
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
        write(buffer, "(A,G0)") " Simple Kriging Mean    : ", self%sk_mean
        res_str = res_str // trim(buffer) // NL
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
  !============================================================================
  subroutine write_matrix(self, krige)
    class(t_kriging_ctx) :: self
    class(t_kriging)     :: krige

    integer              :: ivar, ifile, ii, k1
    integer, allocatable :: idx(:)
    real, allocatable    :: v(:), w(:), xyz(:,:)
    character(len=20)    :: sig, idxstr
    character(len=6)     :: cname(3) = ['x_orig', 'y_orig', 'z_orig']

    associate( &
      ndim      => krige%ndim, &
      ib        => self%iblock, &
      nnear     => self%nnear, &
      inear     => self%inear, &
      dist      => self%sqdist, &
      weight    => self%x, &
      matA      => self%matA, &
      rhsB      => self%rhsB, &
      npp       => self%npp, &
      irandpath => krige%block%order, &
      matsize   => self%matsize)

      write(idxstr, "(I0)") irandpath(ib)

      !-- Neighbour data table
      open(newunit=ifile, file='data_'//trim(idxstr)//'.csv', status='replace')
      write(ifile, '(99(A,:,","))') 'source','index', cname(1:ndim), 'value', 'distance', 'weight'
      k1 = 0
      do ivar = krige%ivar0, krige%nvar
        if (nnear(ivar) == 0) cycle
        w  = weight(1, k1+1:k1+nnear(ivar))
        k1 = k1 + nnear(ivar)
        if (ivar == 0) then
          sig = "GRID"
          idx = krige%block%order(inear(1:nnear(0), 0))
          xyz = krige%block%coord(1:ndim, inear(1:nnear(0), 0))
          v   = krige%block%estimate(1, inear(1:nnear(0), 0))
        else
          write(sig, "('OBS',I0)") ivar
          idx = inear(1:nnear(ivar), ivar)
          xyz = krige%obs(ivar)%coord(1:ndim, inear(1:nnear(ivar), ivar))
          v   = krige%obs(ivar)%value(inear(1:nnear(ivar), ivar))
        end if
        do ii = 1, nnear(ivar)
          write(ifile, "(A,',',I0,*(:,',',G0.8))") &
            trim(sig), idx(ii), xyz(:,ii), v(ii), sqrt(dist(ii, ivar)), w(ii)
        end do
      end do
      close(ifile)

      if (npp <= 1) return

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
  ! Factor file format (three lines per block):
  !   Line 1: order(ib)  kriging_variance  nnear(0:nvar)
  !   Line 2: inear(1:nnear(0),0)  inear(1:nnear(1),1) ...  (neighbour indices)
  !   Line 3: weight(1:nnear(0),0) weight(1:nnear(1),1) ... (kriging weights)
  !
  ! read_weight is used when use_old_weight=.true. (factor-file reuse mode).
  ! write_weight is used when store_weight=.true.  (factor-file creation mode).
  !============================================================================
  subroutine write_weight(self, ctx)
    class(t_kriging)     :: self
    class(t_kriging_ctx) :: ctx
    integer              :: ii
    associate( &
      ib    => ctx%iblock, &
      order => self%block%order, &
      var   => self%block%variance(ctx%iblock))
      write(self%ifile, '(I0,x,G0.12,99(x,I0))') order(ib), var, ctx%nnear(0:self%nvar)
      write(self%ifile, '(*(:2x,I0))')   (ctx%inear(1:ctx%nnear(ii), ii),  ii = 0, self%nvar)
      write(self%ifile, '(*(:2x,F0.10))')(ctx%weight(1:ctx%nnear(ii), ii), ii = 0, self%nvar)
    end associate
  end subroutine write_weight


  subroutine read_weight(self, ctx)
    class(t_kriging)     :: self
    class(t_kriging_ctx) :: ctx
    integer              :: ii
    associate( &
      ib    => ctx%iblock, &
      order => self%block%order, &
      var   => self%block%variance(ctx%iblock))
      read(self%ifile, *) order(ib), var, ctx%nnear(0:self%nvar)
      read(self%ifile, *) (ctx%inear(1:ctx%nnear(ii), ii),  ii = 0, self%nvar)
      read(self%ifile, *) (ctx%weight(1:ctx%nnear(ii), ii), ii = 0, self%nvar)
    end associate
  end subroutine read_weight


  !============================================================================
  ! validate_vgm  (private helper)
  !
  ! Verify that every block/variable-pair has at least one structure defined.
  ! Stops with a descriptive error if any block is missing its variogram.
  !============================================================================
  subroutine validate_vgm(self)
    class(t_kriging), intent(in) :: self
    integer                  :: ib, iv, jv, mb
    character(len=256)       :: msg
    character(*), parameter  :: subname = 't_kriging_sva%validate_vgm'
    mb = merge(self%block%n, 1, self%varying_vgm)
    do ib = 1, mb
      do iv = self%ivar0, self%nvar
        do jv = self%ivar0, self%nvar
          if (self%vgm(jv, iv, ib)%nstruct == 0) then
            write(msg, '(A,I0,A,I0,A,I0,A)') &
              't_kriging_sva: variogram not set for block ', ib, &
              ', ivar=', iv, ', jvar=', jv, &
              '. Call set_vgm_block() or set_vgm_block_all().'
            call kriging_error(subname, trim(msg))
            if (.not. self%vgm(jv, iv, ib)%is_valid()) then
              write(msg, '(A,I0,A,I0,A,I0,A)') &
                't_kriging_sva: variogram is not valid for block ', ib, &
                ', ivar=', iv, ', jvar=', jv, &
                '. Check your variogram parameters.'
              call kriging_error(subname, trim(msg))
            end if
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
      if (allocated(b%estimate))    deallocate(b%estimate)
      if (allocated(b%order))       deallocate(b%order)
      if (allocated(b%nblockpnt))   deallocate(b%nblockpnt)
      if (allocated(b%iblockpnt))   deallocate(b%iblockpnt)
      if (allocated(b%rangescale))  deallocate(b%rangescale)
      if (allocated(b%localnugget)) deallocate(b%localnugget)
      if (allocated(b%sample))      deallocate(b%sample)
      b%block_type = 0
    end associate
  end subroutine reset_block


  !============================================================================
  ! finalize
  !
  ! Release all allocated memory.  Call after all results have been read from
  ! block%estimate and block%variance.
  !============================================================================
  subroutine finalize(self)
    class(t_kriging) :: self
    deallocate(self%obs)
    deallocate(self%grid)
    deallocate(self%block)
    deallocate(self%vgm)
  end subroutine finalize

end module kriging