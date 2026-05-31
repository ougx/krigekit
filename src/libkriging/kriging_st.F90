!==============================================================================
! Module: kriging_st
!
! Space-time kriging and co-kriging for 3D spatial + 1D temporal data.
!
! Design
! ------
! t_kriging_st is a fresh type (not extending t_kriging) that handles 4D
! observations (x, y, z, t).  The spatial part uses the existing kdtree2,
! rotation, solver, and variogram_st infrastructure unchanged.
!
! Key differences from t_kriging:
!   - t_obsgrid_st  adds time(:) and maxtlag to the observation dataset.
!   - t_grid_st     adds time(:) to integration nodes.
!   - t_blockgrid_st adds time(:) to estimation targets.
!   - vgm(:,:) is type(vgm_struct_st) — the ST covariance model.
!   - calc_covariance_st computes dt and calls vgm%cov_lag_st(lag_s, dt).
!   - search_neighbors_st applies a temporal window filter (maxtlag) after
!     the spatial KD-tree search.
!   - SGSIM: primary variable only.  obs(1)%time is extended alongside
!     obs(1)%coord to include prediction block times for SGSIM conditioning.
!
! ndim is always 3 (three spatial dimensions).  Time is carried separately.
!==============================================================================
module kriging_st

  use, intrinsic :: ieee_arithmetic
  use iso_fortran_env, only: output_unit
  use common,           only: EPSLON
  use kriging_err,      only: kriging_error, kriging_failed
  use utils,            only: set_seq, r8vec_normal_01, random_seed_initialize
  use progress_bar,     only: progress
  use rotation,         only: calc_rotmat, sub_rotate, rotated_dists
  use variogram_st,     only: vgm_struct_st, ST_MODEL_SUM_METRIC, ST_MODEL_PRODUCT_SUM, &
                               ST_TRANSFORM_LINEAR, ST_TRANSFORM_BOUNDED, ST_TRANSFORM_POWER
  use kdtree2_module
  use solver,           only: kriging_solve, ssysv_fallback
  implicit none
  private

  public :: t_kriging_st

  real, parameter :: verylarge = huge(0.0)

  !=============================================================================
  ! t_obsgrid_st — observations with time and search infrastructure
  !=============================================================================
  type :: t_obsgrid_st
    integer              :: n        = 0
    real, allocatable             :: coord(:,:)   ! [3, n]
    real, allocatable             :: time(:)      ! [n]   observation times
    real, allocatable             :: drift(:,:)   ! [ndrift, n]
    real, allocatable             :: value(:)     ! [n]
    real, allocatable             :: variance(:)  ! [n]  measurement error variance
    integer              :: nmax     = 0
    real                 :: maxdist  = verylarge   ! stored as maxdist^2
    real                 :: maxtlag  = verylarge   ! physical time units
    real                 :: rotmat(3,3) = reshape([1.,0.,0.,0.,1.,0.,0.,0.,1.],[3,3])
    type(kdtree2), pointer :: tree => null()
    logical              :: need_search        = .false.
    logical              :: anisotropic_search = .false.
  end type t_obsgrid_st

  !=============================================================================
  ! t_grid_st — integration nodes (point or block kriging)
  !=============================================================================
  type :: t_grid_st
    integer              :: n = 0
    real, allocatable    :: coord(:,:)   ! [3, n]
    real, allocatable    :: time(:)      ! [n]
    real, allocatable    :: drift(:,:)   ! [ndrift, n]
    real, allocatable    :: weight(:)    ! integration weights [n]
  end type t_grid_st

  !=============================================================================
  ! t_blockgrid_st — estimation targets
  !=============================================================================
  type :: t_blockgrid_st
    integer              :: n          = 0
    integer              :: block_type = 0    ! 0=point, -4=GQ, >0=user nodes
    real, allocatable             :: coord(:,:)        ! [3, n]
    real, allocatable             :: time(:)           ! [n]
    real, allocatable    :: drift(:,:)        ! [ndrift, n]
    real, allocatable    :: estimate(:,:)     ! [max(1,nsim), n]
    real, allocatable    :: variance(:)       ! [n]
    integer, allocatable :: order(:)          ! SGSIM random path [n]
    integer, allocatable :: nblockpnt(:)      ! nodes per block [n]
    integer, allocatable :: iblockpnt(:)      ! start index in grid [n]
    real, allocatable    :: rangescale(:)     ! variogram range scaler [n]
    real, allocatable    :: localnugget(:)    ! extra diagonal nugget [n]
    real, allocatable    :: sample(:,:)       ! N(0,1) draws [nsim, n]
  end type t_blockgrid_st

  !=============================================================================
  ! Neighbour-group layout (same convention as kriging.F90):
  !   Groups 1:nvar         = real observations, variable ig        (always)
  !   Groups nvar+1:ngroups = previously simulated blocks, variable ig-nvar  (SGSIM only)
  !   group_ivar(ig, nvar)  -> real variable index 1:nvar
  !   ig > nvar             -> .true. for a simulated-block group
  !=============================================================================

  !=============================================================================
  ! t_kriging_st_ctx — per-thread working context (allocated per OMP thread)
  !=============================================================================
  type :: t_kriging_st_ctx
    integer              :: iblock  = 0
    integer              :: npp     = 0     ! total neighbours = sum(nnear)
    integer              :: matsize = 0     ! npp + ndrift + unbias
    integer, allocatable :: nnear(:)        ! [1:ngroups]
    integer, allocatable :: inear(:,:)      ! [nmax, 1:ngroups]
    real,    allocatable :: weight(:,:)     ! [nmax, 1:ngroups]
    real,    allocatable :: sqdist(:,:)     ! squared spatial distances [nmax, 1:ngroups]
    real,    allocatable :: x(:,:)          ! solver output [1, matsize]
    real,    allocatable :: matA(:,:)       ! covariance matrix [matsize, matsize]
    real,    allocatable :: rhsB(:,:)       ! right-hand side [1, matsize]
  contains
    procedure :: initialize  => ctx_initialize
    procedure :: assign_weight => ctx_assign_weight
  end type t_kriging_st_ctx

  !=============================================================================
  ! t_kriging_st — main space-time kriging object
  !=============================================================================
  type :: t_kriging_st
    !-- Options
    logical              :: anisotropic_search = .false.
    logical              :: weight_correction  = .false.
    logical              :: use_old_weight     = .false.
    logical              :: store_weight       = .false.
    logical              :: cross_validation   = .false.
    logical              :: write_mat          = .false.
    logical              :: verbose            = .false.
    logical              :: neglect_error      = .true.
    character(len=1024)  :: weight_file = ""
    integer              :: ifile = 0
    real                 :: bounds(2) = [-verylarge, verylarge]
    real                 :: sk_mean   = 0.0

    !-- Dimensions
    integer              :: ndim   = 3    ! always 3 (spatial)
    integer              :: nvar    = 1
    integer              :: ngroups = 0   ! nvar (kriging) or 2*nvar (SGSIM)
    integer              :: ndrift  = 0
    integer              :: unbias = 1    ! 1=OK, 0=SK
    integer              :: nsim   = 0
    integer              :: seed   = 12345

    !-- ST model global parameters (set by set_st_model; copied into every vgm entry)
    integer              :: st_model     = ST_MODEL_SUM_METRIC
    integer              :: st_transform = ST_TRANSFORM_LINEAR
    real                 :: st_at        = 1.0
    real                 :: st_alpha     = 1.0

    !-- Bookkeeping
    integer              :: nppmax      = 0
    integer              :: matsize_max = 0

    !-- Data
    type(t_obsgrid_st),  allocatable :: obs(:)    ! [1:nvar]
    type(t_grid_st)                  :: grid
    type(t_blockgrid_st)             :: block
    type(vgm_struct_st), allocatable :: vgm(:,:)  ! [1:nvar, 1:nvar]

  contains
    procedure :: initialize         => initialize_st
    procedure :: set_st_model       => set_st_model_st
    procedure :: set_obs            => set_obs_st
    procedure :: set_obs_drift      => set_obs_drift_st
    procedure :: set_vgm            => set_vgm_st
    procedure :: set_vgm_temporal   => set_vgm_temporal_st
    procedure :: set_vgm_joint_sills => set_vgm_joint_sills_st
    procedure :: set_grid           => set_grid_st
    procedure :: set_grid_block     => set_grid_block_st
    procedure :: set_grid_cv        => set_grid_cv_st
    procedure :: set_grid_drift     => set_grid_drift_st
    procedure :: set_sim            => set_sim_st
    procedure :: set_search         => set_search_st
    procedure :: search_neighbors   => search_neighbors_st
    procedure :: calc_covariance    => calc_covariance_st
    procedure :: assemble_system    => assemble_system_st
    procedure :: solve_system       => solve_system_st
    procedure :: estimate_block     => estimate_block_st
    procedure :: prepare            => prepare_st
    procedure :: solve              => solve_st
    procedure :: finalize           => finalize_st
  end type t_kriging_st

contains

  !=============================================================================
  ! ctx_initialize — allocate per-thread working arrays
  !=============================================================================
  subroutine ctx_initialize(self, krige)
    class(t_kriging_st_ctx), intent(inout) :: self
    class(t_kriging_st),     intent(in)    :: krige
    integer :: mmax, iv
    mmax = maxval(krige%obs%nmax)
    associate(npp => krige%nppmax, matsize => krige%matsize_max, ng => krige%ngroups)
      allocate(self%sqdist(mmax,    ng));  self%sqdist = 0.0
      allocate(self%matA  (matsize, matsize))
      allocate(self%rhsB  (1,       matsize))
      allocate(self%nnear (         ng))
      allocate(self%inear (mmax,    ng))
      allocate(self%weight(mmax,    ng));  self%weight = 0.0
      allocate(self%x     (1,       matsize));       self%x      = 0.0
      call set_seq(self%inear(1:mmax, 1), mmax)
      do iv = 1, krige%nvar
        self%nnear(iv)   = krige%obs(iv)%nmax
        self%inear(:,iv) = self%inear(:,1)
      end do
      do iv = krige%nvar + 1, ng
        self%nnear(iv) = 0
      end do
    end associate
  end subroutine ctx_initialize

  subroutine ctx_assign_weight(self, krige)
    class(t_kriging_st_ctx), intent(inout) :: self
    class(t_kriging_st),     intent(in)    :: krige
    integer :: ivar, k1
    k1 = 0
    do ivar = 1, krige%ngroups
      if (self%nnear(ivar) == 0) cycle
      self%weight(1:self%nnear(ivar), ivar) = self%x(1, k1+1:k1+self%nnear(ivar))
      k1 = k1 + self%nnear(ivar)
    end do
  end subroutine ctx_assign_weight


  !=============================================================================
  ! group_ivar — map group index ig (1:ngroups) to real variable index (1:nvar).
  ! Obs groups (ig = 1:nvar) -> ig; sim groups (ig = nvar+1:2*nvar) -> ig-nvar.
  !=============================================================================
  pure elemental integer function group_ivar(ig, nvar)
    integer, intent(in) :: ig, nvar
    group_ivar = mod(ig - 1, nvar) + 1
  end function group_ivar

  !=============================================================================
  ! initialize_st — allocate obs, vgm arrays and set options
  !=============================================================================
  subroutine initialize_st(self, nvar, ndrift, unbias, nsim, &
      anisotropic_search, weight_correction, use_old_weight, store_weight, &
      cross_validation, write_mat, neglect_error, verbose, &
      weight_file, bounds, sk_mean, seed)
    class(t_kriging_st)                      :: self
    integer, intent(in), optional            :: nvar, ndrift, unbias, nsim, seed
    logical, intent(in), optional            :: anisotropic_search, weight_correction, &
                                                use_old_weight, store_weight, &
                                                cross_validation, write_mat, &
                                                neglect_error, verbose
    character(*), intent(in), optional       :: weight_file
    real,    intent(in), optional            :: bounds(2), sk_mean

    if (present(nvar))               self%nvar               = nvar
    if (present(ndrift))             self%ndrift             = ndrift
    if (present(unbias))             self%unbias             = unbias
    if (present(nsim))               self%nsim               = nsim
    if (present(seed))               self%seed               = seed
    if (present(anisotropic_search)) self%anisotropic_search = anisotropic_search
    if (present(weight_correction))  self%weight_correction  = weight_correction
    if (present(use_old_weight))     self%use_old_weight     = use_old_weight
    if (present(store_weight))       self%store_weight       = store_weight
    if (present(cross_validation))   self%cross_validation   = cross_validation
    if (present(write_mat))          self%write_mat          = write_mat
    if (present(neglect_error))      self%neglect_error      = neglect_error
    if (present(verbose))            self%verbose            = verbose
    if (present(weight_file))        self%weight_file        = weight_file
    if (present(bounds))             self%bounds             = bounds
    if (present(sk_mean))            self%sk_mean            = sk_mean

    self%ngroups = merge(2 * self%nvar, self%nvar, self%nsim > 0)

    call random_seed_initialize(self%seed)

    if (allocated(self%obs)) deallocate(self%obs)
    if (allocated(self%vgm)) deallocate(self%vgm)
    allocate(self%obs(1:self%nvar))
    allocate(self%vgm(1:self%nvar, 1:self%nvar))
  end subroutine initialize_st


  !=============================================================================
  ! set_st_model_st — set global ST model params and propagate to all vgm entries
  !=============================================================================
  subroutine set_st_model_st(self, model, transform, at, alpha, k_ps)
    class(t_kriging_st), intent(inout) :: self
    integer, intent(in)                :: model, transform
    real,    intent(in)                :: at
    real,    intent(in), optional      :: alpha, k_ps
    integer :: i, j

    self%st_model     = model
    self%st_transform = transform
    self%st_at        = at
    if (present(alpha)) self%st_alpha = alpha

    do j = 1, self%nvar
      do i = 1, self%nvar
        self%vgm(i,j)%model     = model
        self%vgm(i,j)%transform = transform
        self%vgm(i,j)%at        = at
        if (present(alpha)) self%vgm(i,j)%alpha = alpha
        if (present(k_ps))  self%vgm(i,j)%k_ps  = k_ps
      end do
    end do
  end subroutine set_st_model_st


  !=============================================================================
  ! set_obs_st — load observations for variable ivar
  !   coord  : [3, nobs] spatial coordinates
  !   time   : [nobs]    observation times (any consistent unit, e.g. years)
  !   value  : [nobs]
  !   variance : [nobs]  measurement error variance (optional)
  !   nmax   : max neighbours (optional)
  !   maxdist: max spatial search radius, physical (optional)
  !   maxtlag: max temporal lag (optional)
  !=============================================================================
  subroutine set_obs_st(self, ivar, coord, value, time, variance, nmax, maxdist, maxtlag)
    class(t_kriging_st), intent(inout) :: self
    integer,             intent(in)    :: ivar
    real,                intent(in)    :: coord(:,:), value(:), time(:)
    real,                intent(in), optional :: variance(:), maxdist, maxtlag
    integer,             intent(in), optional :: nmax

    associate(obs => self%obs(ivar))
      obs%n = size(value)
      if (size(coord, 1) /= 3) then
        call kriging_error('set_obs_st', 'coord must have 3 rows (x,y,z)')
        return
      end if
      if (size(coord, 2) /= obs%n) then
        call kriging_error('set_obs_st', 'coord column count != nobs')
        return
      end if
      if (size(time) /= obs%n) then
        call kriging_error('set_obs_st', 'time length != nobs')
        return
      end if

      if (allocated(obs%coord))    deallocate(obs%coord)
      if (allocated(obs%time))     deallocate(obs%time)
      if (allocated(obs%value))    deallocate(obs%value)
      if (allocated(obs%variance)) deallocate(obs%variance)

      allocate(obs%coord,    source = coord)
      allocate(obs%time,     source = time)
      allocate(obs%value,    source = value)
      if (present(variance)) then
        allocate(obs%variance, source = variance)
      else
        allocate(obs%variance(obs%n)); obs%variance = 0.0
      end if

      obs%nmax    = merge(nmax,          huge(0),      present(nmax))
      obs%maxdist = merge(maxdist**2,    verylarge,    present(maxdist))
      obs%maxtlag = merge(maxtlag,       verylarge,    present(maxtlag))
      obs%rotmat  = reshape([1.,0.,0., 0.,1.,0., 0.,0.,1.], [3,3])
    end associate
  end subroutine set_obs_st


  !=============================================================================
  ! set_obs_drift_st — attach external drift values at observation locations
  !=============================================================================
  subroutine set_obs_drift_st(self, ivar, drift)
    class(t_kriging_st), intent(inout) :: self
    integer,             intent(in)    :: ivar
    real,                intent(in)    :: drift(:,:)   ! [ndrift, nobs]
    if (self%obs(ivar)%n == 0) then
      call kriging_error('set_obs_drift_st', 'call set_obs first')
      return
    end if
    if (size(drift,1) /= self%ndrift) then
      call kriging_error('set_obs_drift_st', 'size(drift,1) /= ndrift')
      return
    end if
    if (allocated(self%obs(ivar)%drift)) deallocate(self%obs(ivar)%drift)
    allocate(self%obs(ivar)%drift, source = drift)
  end subroutine set_obs_drift_st


  !=============================================================================
  ! set_vgm_st — add one nested SPATIAL structure to vgm(ivar,jvar)%cs
  !   spec: "vtype nugget sill a_major a_minor1 a_minor2 azimuth dip plunge"
  !=============================================================================
  subroutine set_vgm_st(self, ivar, jvar, spec)
    class(t_kriging_st), intent(inout) :: self
    integer,             intent(in)    :: ivar, jvar
    character(*),        intent(in)    :: spec
    if (jvar == ivar) then
      call self%vgm(ivar, jvar)%add_spatial(spec)
      if (kriging_failed()) return
    else if (jvar > ivar) then
      call self%vgm(ivar, jvar)%add_spatial(spec)
      if (kriging_failed()) return
      call self%vgm(jvar, ivar)%add_spatial(spec)
      if (kriging_failed()) return
    else
      call kriging_error('set_vgm_st', 'jvar must be >= ivar')
      return
    end if
  end subroutine set_vgm_st


  !=============================================================================
  ! set_vgm_temporal_st — add one nested TEMPORAL structure to vgm(ivar,jvar)%ct
  !   spec: "vtype nugget sill at_k"    (simplified 4-param format)
  !=============================================================================
  subroutine set_vgm_temporal_st(self, ivar, jvar, spec)
    class(t_kriging_st), intent(inout) :: self
    integer,             intent(in)    :: ivar, jvar
    character(*),        intent(in)    :: spec
    if (jvar == ivar) then
      call self%vgm(ivar, jvar)%add_temporal(spec)
      if (kriging_failed()) return
    else if (jvar > ivar) then
      call self%vgm(ivar, jvar)%add_temporal(spec)
      if (kriging_failed()) return
      call self%vgm(jvar, ivar)%add_temporal(spec)
      if (kriging_failed()) return
    else
      call kriging_error('set_vgm_temporal_st', 'jvar must be >= ivar')
      return
    end if
  end subroutine set_vgm_temporal_st


  !=============================================================================
  ! set_vgm_joint_sills_st — set joint sills for sum-metric vgm(ivar,jvar)
  !   sills: one per nested structure of cs (length must equal cs%nstruct)
  !=============================================================================
  subroutine set_vgm_joint_sills_st(self, ivar, jvar, sills, n)
    class(t_kriging_st), intent(inout) :: self
    integer,             intent(in)    :: ivar, jvar, n
    real,                intent(in)    :: sills(n)
    if (jvar == ivar) then
      call self%vgm(ivar, jvar)%set_joint_sills(sills, n)
      if (kriging_failed()) return
    else if (jvar > ivar) then
      call self%vgm(ivar, jvar)%set_joint_sills(sills, n)
      if (kriging_failed()) return
      call self%vgm(jvar, ivar)%set_joint_sills(sills, n)
      if (kriging_failed()) return
    else
      call kriging_error('set_vgm_joint_sills_st', 'jvar must be >= ivar')
      return
    end if
  end subroutine set_vgm_joint_sills_st


  !=============================================================================
  ! set_grid_st — set point estimation targets
  !   coord : [3, ngrid]  spatial coordinates
  !   time  : [ngrid]     prediction times
  !=============================================================================
  subroutine set_grid_st(self, coord, time, rangescale, localnugget)
    class(t_kriging_st), intent(inout) :: self
    real,                intent(in)    :: coord(:,:), time(:)
    real,                intent(in), optional :: rangescale(:), localnugget(:)
    integer :: ng

    ng = size(coord, 2)
    if (size(coord,1) /= 3) then
      call kriging_error('set_grid_st', 'coord must have 3 rows')
      return
    end if
    if (size(time) /= ng) then
      call kriging_error('set_grid_st', 'time length != ngrid')
      return
    end if

    associate(b => self%block, g => self%grid)
      b%n          = ng
      b%block_type = 0   ! point kriging
      if (allocated(b%coord))       deallocate(b%coord)
      if (allocated(b%time))        deallocate(b%time)
      if (allocated(b%estimate))    deallocate(b%estimate)
      if (allocated(b%variance))    deallocate(b%variance)
      if (allocated(b%order))       deallocate(b%order)
      if (allocated(b%nblockpnt))   deallocate(b%nblockpnt)
      if (allocated(b%iblockpnt))   deallocate(b%iblockpnt)
      if (allocated(b%rangescale))  deallocate(b%rangescale)
      if (allocated(b%localnugget)) deallocate(b%localnugget)

      allocate(b%coord,     source = coord)
      allocate(b%time,      source = time)
      allocate(b%estimate(max(1, self%nsim), ng)); b%estimate = 0.0
      allocate(b%variance(ng));                    b%variance = 0.0
      allocate(b%order(ng));      call set_seq(b%order, ng)
      allocate(b%nblockpnt(ng));  b%nblockpnt = 1
      allocate(b%iblockpnt(ng));  call set_seq(b%iblockpnt, ng)
      allocate(b%rangescale(ng))
      allocate(b%localnugget(ng))
      if (present(rangescale))  then; b%rangescale  = rangescale;  else; b%rangescale  = 1.0; end if
      if (present(localnugget)) then; b%localnugget = localnugget; else; b%localnugget = 0.0; end if

      !-- Integration grid for point kriging: one node per block, weight=1
      g%n = ng
      if (allocated(g%coord))  deallocate(g%coord)
      if (allocated(g%time))   deallocate(g%time)
      if (allocated(g%weight)) deallocate(g%weight)
      allocate(g%coord,  source = coord)
      allocate(g%time,   source = time)
      allocate(g%weight(ng)); g%weight = 1.0
    end associate
  end subroutine set_grid_st


  !=============================================================================
  ! set_grid_block_st — set block estimation targets with integration nodes
  !   coord      : [3, nblocks]    block centre coordinates
  !   time       : [nblocks]       block centre times
  !   nblockpnt  : [nblocks]       number of integration nodes per block
  !   blockcoord : [3, sum(nblockpnt)]  integration node coordinates
  !   blocktime  : [sum(nblockpnt)] integration node times
  !   pointweight: [sum(nblockpnt)] integration weights
  !=============================================================================
  subroutine set_grid_block_st(self, coord, time, nblockpnt, blockcoord, blocktime, &
                                pointweight, rangescale, localnugget)
    class(t_kriging_st), intent(inout) :: self
    real,                intent(in)    :: coord(:,:), time(:)
    integer,             intent(in)    :: nblockpnt(:)
    real,                intent(in)    :: blockcoord(:,:), blocktime(:), pointweight(:)
    real,                intent(in), optional :: rangescale(:), localnugget(:)

    integer :: nb, ig, ib

    nb = size(coord, 2)
    if (size(coord,1) /= 3) then
      call kriging_error('set_grid_block_st', 'coord must have 3 rows')
      return
    end if
    if (size(time) /= nb) then
      call kriging_error('set_grid_block_st', 'time length != nblocks')
      return
    end if
    if (size(nblockpnt) /= nb) then
      call kriging_error('set_grid_block_st', 'nblockpnt length != nblocks')
      return
    end if
    if (size(blockcoord,2) /= sum(nblockpnt)) then
      call kriging_error('set_grid_block_st', 'blockcoord columns != sum(nblockpnt)')
      return
    end if

    associate(b => self%block, g => self%grid)
      b%n          = nb
      b%block_type = 1   ! user-supplied integration nodes
      if (allocated(b%coord))       deallocate(b%coord)
      if (allocated(b%time))        deallocate(b%time)
      if (allocated(b%estimate))    deallocate(b%estimate)
      if (allocated(b%variance))    deallocate(b%variance)
      if (allocated(b%order))       deallocate(b%order)
      if (allocated(b%nblockpnt))   deallocate(b%nblockpnt)
      if (allocated(b%iblockpnt))   deallocate(b%iblockpnt)
      if (allocated(b%rangescale))  deallocate(b%rangescale)
      if (allocated(b%localnugget)) deallocate(b%localnugget)

      allocate(b%coord,     source = coord)
      allocate(b%time,      source = time)
      allocate(b%estimate(max(1,self%nsim), nb)); b%estimate = 0.0
      allocate(b%variance(nb));                   b%variance = 0.0
      allocate(b%order(nb));       call set_seq(b%order, nb)
      allocate(b%nblockpnt,        source = nblockpnt)
      allocate(b%iblockpnt(nb))
      allocate(b%rangescale(nb))
      allocate(b%localnugget(nb))
      ig = 1
      do ib = 1, nb
        b%iblockpnt(ib) = ig
        ig = ig + nblockpnt(ib)
      end do
      if (present(rangescale))  then; b%rangescale  = rangescale;  else; b%rangescale  = 1.0; end if
      if (present(localnugget)) then; b%localnugget = localnugget; else; b%localnugget = 0.0; end if

      g%n = sum(nblockpnt)
      if (allocated(g%coord))  deallocate(g%coord)
      if (allocated(g%time))   deallocate(g%time)
      if (allocated(g%weight)) deallocate(g%weight)
      allocate(g%coord,  source = blockcoord)
      allocate(g%time,   source = blocktime)
      allocate(g%weight, source = pointweight)
    end associate
  end subroutine set_grid_block_st


  !=============================================================================
  ! set_grid_cv_st — cross-validation: predict at observation locations
  !=============================================================================
  subroutine set_grid_cv_st(self)
    class(t_kriging_st), intent(inout) :: self
    if (self%obs(1)%n == 0) then
      call kriging_error('set_grid_cv_st', 'call set_obs first')
      return
    end if
    call self%set_grid(self%obs(1)%coord, self%obs(1)%time)
    self%cross_validation = .true.
  end subroutine set_grid_cv_st


  !=============================================================================
  ! set_grid_drift_st — attach drift values at estimation grid points
  !=============================================================================
  subroutine set_grid_drift_st(self, drift)
    class(t_kriging_st), intent(inout) :: self
    real,                intent(in)    :: drift(:,:)   ! [ndrift, nblocks]
    if (self%block%n == 0) then
      call kriging_error('set_grid_drift_st', 'call set_grid first')
      return
    end if
    if (size(drift,1) /= self%ndrift) then
      call kriging_error('set_grid_drift_st', 'size(drift,1) /= ndrift')
      return
    end if
    if (allocated(self%block%drift)) deallocate(self%block%drift)
    allocate(self%block%drift, source = drift)
  end subroutine set_grid_drift_st


  !=============================================================================
  ! set_sim_st — prepare SGSIM random path and extend obs(1) with block coords/times
  !=============================================================================
  subroutine set_sim_st(self, randpath, sample)
    class(t_kriging_st),          intent(inout) :: self
    integer, intent(in), optional               :: randpath(:)
    real,    intent(in), optional               :: sample(:,:)   ! [nsim, nblock]

    real,    allocatable :: temp_r(:,:)
    real,    allocatable :: temp_t(:)
    integer              :: ifile, isim, ib

    if (self%nsim == 0) return
    if (self%block%n == 0) then
      call kriging_error('set_sim_st', 'call set_grid first')
      return
    end if
    if (self%obs(1)%n == 0) then
      call kriging_error('set_sim_st', 'call set_obs(1) first')
      return
    end if

    associate(nb => self%block%n, obs => self%obs(1))
      !-- Random visit path (order already allocated by set_grid_st)
      if (present(randpath)) then
        self%block%order = randpath
      else
        call set_seq(self%block%order, nb, .true.)
      end if

      !-- N(0,1) samples
      if (allocated(self%block%sample)) deallocate(self%block%sample)
      allocate(self%block%sample(self%nsim, nb))
      if (present(sample)) then
        self%block%sample = sample
      else
        do isim = 1, self%nsim
          call r8vec_normal_01(nb, self%block%sample(isim,:))
        end do
      end if

      !-- Reorder block arrays into random-path order
      self%block%coord      = self%block%coord     (:, self%block%order)
      self%block%time       = self%block%time       (   self%block%order)
      self%block%iblockpnt  = self%block%iblockpnt (   self%block%order)
      self%block%nblockpnt  = self%block%nblockpnt (   self%block%order)
      self%block%rangescale = self%block%rangescale(   self%block%order)
      self%block%localnugget= self%block%localnugget(  self%block%order)
      if (self%ndrift > 0 .and. allocated(self%block%drift)) &
        self%block%drift = self%block%drift(:, self%block%order)

      !-- Extend obs(1)%coord and obs(1)%time to include block centres
      allocate(temp_r(3, obs%n + nb))
      temp_r(:, 1:obs%n) = obs%coord
      temp_r(:, obs%n+1:) = self%block%coord
      call move_alloc(temp_r, obs%coord)

      allocate(temp_t(obs%n + nb))
      temp_t(1:obs%n) = obs%time
      temp_t(obs%n+1:) = self%block%time
      call move_alloc(temp_t, obs%time)

      !-- Also extend variance array with zeros for block slots
      allocate(temp_t(obs%n + nb))
      temp_t(1:obs%n)  = obs%variance
      temp_t(obs%n+1:) = 0.0
      call move_alloc(temp_t, obs%variance)
    end associate
  end subroutine set_sim_st


  !=============================================================================
  ! set_search_st — build the spatial KD-tree for variable ivar
  !=============================================================================
  subroutine set_search_st(self, ivar, anis1, anis2, azimuth, dip, plunge)
    class(t_kriging_st), intent(inout) :: self
    integer,             intent(in)    :: ivar
    real,    intent(in), optional      :: anis1, anis2, azimuth, dip, plunge

    real    :: a1, a2, az, dp, pl
    real, allocatable :: rcoord(:,:)

    a1 = merge(anis1,   1.0, present(anis1))
    a2 = merge(anis2,   1.0, present(anis2))
    az = merge(azimuth, 0.0, present(azimuth))
    dp = merge(dip,     0.0, present(dip))
    pl = merge(plunge,  0.0, present(plunge))

    associate(obs => self%obs(ivar))
      obs%rotmat = calc_rotmat(az, dp, pl, a1, a2)
      obs%anisotropic_search = &
        (abs(a1-1.0) > EPSLON .or. abs(a2-1.0) > EPSLON) .and. self%anisotropic_search

      !-- For SGSIM ivar=1: obs%n is the ORIGINAL count; extended array includes block centres
      if (ivar == 1 .and. self%nsim > 0) then
        !-- original obs count = size - nblock
        obs%nmax = min(obs%nmax, size(obs%coord,2))
        if (obs%nmax <= 0) obs%nmax = size(obs%coord,2)
        obs%need_search = size(obs%coord,2) > obs%nmax
      else
        obs%nmax = min(obs%nmax, obs%n)
        if (obs%nmax <= 0) obs%nmax = obs%n
        obs%need_search = obs%n > obs%nmax
      end if

      if (obs%need_search) then
        if (associated(obs%tree)) call kdtree2_destroy(obs%tree)
        if (obs%anisotropic_search) then
          allocate(rcoord, mold=obs%coord)
          call sub_rotate(obs%rotmat, 3, size(obs%coord,2), obs%coord, rcoord)
          obs%tree => kdtree2_create(rcoord, sort=.false., rearrange=.true.)
          if (kriging_failed()) return
        else
          obs%tree => kdtree2_create(obs%coord, sort=.false., rearrange=.true.)
          if (kriging_failed()) return
        end if
      end if
    end associate
  end subroutine set_search_st


  !=============================================================================
  ! search_neighbors_st — spatial KD-tree search + temporal window filter
  !=============================================================================
  subroutine search_neighbors_st(self, ivar, ctx)
    class(t_kriging_st),     intent(inout) :: self
    class(t_kriging_st_ctx), intent(inout) :: ctx
    integer,                 intent(in)    :: ivar

    integer                  :: i, k, nobs
    real                     :: newloc(3,1), block_t
    logical, allocatable     :: is_obs(:)
    type(kdtree2_result), allocatable :: results(:)

    associate( &
      iblock  => ctx%iblock, &
      nmax    => self%obs(ivar)%nmax, &
      obsloc  => self%obs(ivar)%coord, &
      obstime => self%obs(ivar)%time, &
      xloc    => self%block%coord(:, ctx%iblock:ctx%iblock), &
      inear   => ctx%inear(:, ivar), &
      nnear   => ctx%nnear(ivar), &
      dist    => ctx%sqdist(:, ivar), &
      maxdist => self%obs(ivar)%maxdist, &
      maxtlag => self%obs(ivar)%maxtlag, &
      rotmat  => self%obs(ivar)%rotmat)

      block_t = self%block%time(iblock)
      nobs    = self%obs(ivar)%n    ! original obs count (not extended)

      if (self%obs(ivar)%anisotropic_search) then
        call sub_rotate(rotmat, 3, 1, xloc, newloc)
      else
        newloc = xloc
      end if

      !----------------------------------------------------------------------
      ! SGSIM path: search over original obs + previously simulated blocks
      !----------------------------------------------------------------------
      if (self%nsim > 0 .and. ivar == 1) then
        associate( &
          inearb => ctx%inear(:, self%nvar + ivar), &
          nnearb => ctx%nnear(self%nvar + ivar), &
          distb  => ctx%sqdist(:, self%nvar + ivar))

          allocate(results(nmax))
          if (nmax < nobs + iblock - 1) then
            call kdtree2_n_nearest_maxidx(self%obs(ivar)%tree, newloc(:,1), nmax, &
                                          results, nobs + iblock - 1)
            if (kriging_failed()) return
            allocate(is_obs, source = results%idx <= nobs)
            nnear  = count(is_obs)
            nnearb = nmax - nnear
            inear (1:nnear)  = pack(results%idx,  is_obs)
            inearb(1:nnearb) = pack(results%idx, .not. is_obs) - nobs
            dist  (1:nnear)  = pack(results%dis,  is_obs)
            distb (1:nnearb) = pack(results%dis, .not. is_obs)
          else
            nnear  = nobs
            nnearb = iblock - 1
            dist (1:nnear)  = rotated_dists(rotmat, 3, newloc(:,1), obsloc(:, 1:nnear))
            distb(1:nnearb) = rotated_dists(rotmat, 3, newloc(:,1), &
                                            self%block%coord(:, 1:nnearb))
          end if

          !-- Spatial + temporal filter on obs neighbours
          k = 0
          do i = 1, nnear
            if (dist(i) <= maxdist .and. &
                abs(obstime(inear(i)) - block_t) <= maxtlag) then
              k = k + 1
              inear(k) = inear(i)
              dist(k)  = dist(i)
            end if
          end do
          nnear = k

          !-- Temporal filter on simulated-block neighbours
          k = 0
          do i = 1, nnearb
            if (distb(i) <= maxdist .and. &
                abs(self%block%time(inearb(i)) - block_t) <= maxtlag) then
              k = k + 1
              inearb(k) = inearb(i)
              distb(k)  = distb(i)
            end if
          end do
          nnearb = k

        end associate  ! inearb, nnearb, distb

      !----------------------------------------------------------------------
      ! Standard kriging / cokriging search
      !----------------------------------------------------------------------
      else
        allocate(results(nmax))
        if (self%obs(ivar)%need_search) then
          call kdtree2_n_nearest(self%obs(ivar)%tree, newloc(:,1), nmax, results)
          if (kriging_failed()) return
          nnear          = nmax
          inear(1:nnear) = results%idx
          dist (1:nnear) = results%dis
        else
          nnear = nobs
          call set_seq(inear(1:nnear), nnear)
          dist(1:nnear) = rotated_dists(rotmat, 3, newloc(:,1), obsloc(:,1:nnear))
        end if

        !-- Cross-validation: exclude target from its own neighbourhood
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

        !-- Spatial + temporal filter
        k = 0
        do i = 1, nnear
          if (dist(i) <= maxdist .and. &
              abs(obstime(inear(i)) - block_t) <= maxtlag) then
            k = k + 1
            inear(k) = inear(i)
            dist(k)  = dist(i)
          end if
        end do
        nnear = k
      end if

    end associate
  end subroutine search_neighbors_st


  !=============================================================================
  ! calc_covariance_st — fill matA block or rhsB for variable pair (ivar, jvar)
  !   jvar == -1 : RHS mode (obs-to-block covariances)
  !   jvar >= 0  : LHS mode (obs-to-obs covariance sub-block)
  !=============================================================================
  subroutine calc_covariance_st(self, ctx, ir0, ic0, ivar, jvar)
    class(t_kriging_st),     intent(inout) :: self
    class(t_kriging_st_ctx), intent(inout) :: ctx
    integer,                 intent(in)    :: ivar, jvar, ir0, ic0

    integer :: i, j, k, k1, istart
    real    :: lag_s(3), dt, tmp

    lag_s = 0.0

    associate( &
      nnear  => ctx%nnear(ivar), &
      inear  => ctx%inear(1:ctx%nnear(ivar), ivar), &
      rs     => self%block%rangescale(ctx%iblock), &
      ln     => self%block%localnugget(ctx%iblock))

      !------------------------------------------------------------------------
      ! RHS mode: C(obs_i, block_x0)
      ! obs1 coord/time: ivar==0 → block (SGSIM prior), else obs(ivar)
      !------------------------------------------------------------------------
      if (jvar == -1) then
        associate(vgm => self%vgm(1, group_ivar(ivar, self%nvar)))
          do i = 1, nnear
            tmp = 0.0
            k1  = self%block%iblockpnt(ctx%iblock) - 1
            do k = 1, self%block%nblockpnt(ctx%iblock)
              if (ivar > self%nvar) then
                lag_s = (self%block%coord(:, inear(i)) - self%grid%coord(:, k1+k)) / rs
                dt    =  self%block%time(inear(i))     - self%block%time(ctx%iblock)
              else
                lag_s = (self%obs(ivar)%coord(:, inear(i)) - self%grid%coord(:, k1+k)) / rs
                dt    =  self%obs(ivar)%time(inear(i))     - self%block%time(ctx%iblock)
              end if
              tmp = tmp + vgm%cov_lag_st(lag_s, dt) * self%grid%weight(k1+k)
            end do
            ctx%rhsB(1, ir0+i) = tmp
          end do
        end associate

      !------------------------------------------------------------------------
      ! LHS mode: C(obs_i of ivar, obs_j of jvar)
      !------------------------------------------------------------------------
      else
        associate( &
          nnear2 => ctx%nnear(jvar), &
          inear2 => ctx%inear(1:ctx%nnear(jvar), jvar), &
          vgm    => self%vgm(group_ivar(jvar, self%nvar), group_ivar(ivar, self%nvar)))

          do i = 1, nnear
            if (ivar == jvar) then
              istart = i + 1
              !-- Diagonal: C(0) + obs error + local nugget.
              !   Simulated blocks are treated as hard data (variance = 0).
              if (ivar > self%nvar) then
                ctx%matA(ic0+i, ir0+i) = vgm%cov0_val + ln
              else
                ctx%matA(ic0+i, ir0+i) = &
                  vgm%cov0_val + self%obs(ivar)%variance(inear(i)) + ln
              end if
            else
              istart = 1
            end if
            do j = istart, nnear2
              if (ivar > self%nvar) then
                lag_s = self%block%coord(:, inear(i))
                dt    = self%block%time(inear(i))
              else
                lag_s = self%obs(ivar)%coord(:, inear(i))
                dt    = self%obs(ivar)%time(inear(i))
              end if
              if (jvar > self%nvar) then
                lag_s = (lag_s - self%block%coord(:, inear2(j))) / rs
                dt    =  dt    - self%block%time(inear2(j))
              else
                lag_s = (lag_s - self%obs(group_ivar(jvar, self%nvar))%coord(:, inear2(j))) / rs
                dt    =  dt    - self%obs(group_ivar(jvar, self%nvar))%time(inear2(j))
              end if
              ctx%matA(ic0+j, ir0+i) = vgm%cov_lag_st(lag_s, dt)
            end do
          end do
        end associate
      end if
    end associate
  end subroutine calc_covariance_st


  !=============================================================================
  ! assemble_system_st — build the full kriging matrix and RHS for block ctx%iblock
  !=============================================================================
  subroutine assemble_system_st(self, ctx)
    class(t_kriging_st),     intent(inout) :: self
    class(t_kriging_st_ctx), intent(inout) :: ctx

    integer :: ivar, jvar, irow1, irow2, icol1, icol2

    associate(nvar => self%nvar, npp => ctx%npp)

      !-- Spatial + temporal neighbour search for each variable
      do ivar = 1, nvar
        call self%search_neighbors(ivar, ctx)
        !-- Exact spatial match for primary variable
        if (ivar == 1 .and. ctx%nnear(ivar) > 0) then
          if (minval(ctx%sqdist(1:ctx%nnear(ivar), ivar)) <= EPSLON) then
            npp = 1
            ctx%x = 0.0;  ctx%x(:,1) = 1.0
            ctx%weight = 0.0;  ctx%weight(1,1) = 1.0
            ctx%inear(1,ivar) = ctx%inear( &
              minloc(ctx%sqdist(1:ctx%nnear(ivar),ivar), dim=1), ivar)
            ctx%nnear(ivar) = 1
            ctx%nnear(self%nvar+1:self%ngroups) = 0
            self%block%variance(ctx%iblock) = &
              self%obs(1)%variance(ctx%inear(1,ivar)) + self%block%localnugget(ctx%iblock)
            do jvar = 2, nvar; ctx%nnear(jvar) = 0; end do
            return
          end if
        end if
      end do

      npp = sum(ctx%nnear)
      if (npp == 0) then
        if (self%neglect_error) then
          !-- No temporal neighbours: return NaN estimate and prior variance
          self%block%estimate(:, ctx%iblock) = IEEE_VALUE(0.0, IEEE_QUIET_NAN)
          self%block%variance(ctx%iblock)    = self%vgm(1,1)%cov0_val
          return
        else
          call kriging_error('assemble_system_st', 'No neighbours found', iblock=ctx%iblock)
          return
        end if
      end if

      associate( &
        matA    => ctx%matA, &
        rhsB    => ctx%rhsB, &
        nnear   => ctx%nnear, &
        matsize => ctx%matsize, &
        ndrift  => self%ndrift)

        matsize = npp + self%unbias + ndrift

        irow1 = 0
        do ivar = 1, self%ngroups
          if (nnear(ivar) == 0) cycle
          irow2 = irow1 + nnear(ivar)
          icol1 = 0

          !-- RHS
          call self%calc_covariance(ctx, irow1, icol1, ivar, -1)

          !-- Upper triangle LHS
          do jvar = 1, self%ngroups
            if (nnear(jvar) == 0) cycle
            icol2 = icol1 + nnear(jvar)
            if (jvar >= ivar) call self%calc_covariance(ctx, irow1, icol1, ivar, jvar)
            icol1 = icol2
          end do

          !-- Drift columns (obs groups only; simulated blocks carry no drift)
          if (ndrift > 0 .and. ivar <= self%nvar) then
            icol2 = icol1 + ndrift
            matA(icol1+1:icol2, irow1+1:irow2) = &
              self%obs(ivar)%drift(:, ctx%inear(1:nnear(ivar), ivar))
          end if
          irow1 = irow2
        end do

        if (ndrift > 0) rhsB(1, npp+1:npp+ndrift) = self%block%drift(:, ctx%iblock)
        if (self%unbias == 1) then
          matA(matsize, 1:npp) = 1.0
          rhsB(1, matsize)     = 1.0
        end if

        !-- Mirror lower triangle
        do irow1 = 1, npp
          do icol1 = irow1+1, matsize
            matA(irow1, icol1) = matA(icol1, irow1)
          end do
        end do
        matA(npp+1:matsize, npp+1:matsize) = 0.0
      end associate
    end associate
  end subroutine assemble_system_st


  !=============================================================================
  ! solve_system_st — solve the assembled kriging system; compute kriging variance
  !=============================================================================
  subroutine solve_system_st(self, ctx)
    class(t_kriging_st),     intent(inout) :: self
    class(t_kriging_st_ctx), intent(inout) :: ctx

    integer :: info, i, j, k1
    real    :: lag_s(3)

    associate( &
      iblock   => ctx%iblock, &
      matA     => ctx%matA, &
      rhsB     => ctx%rhsB, &
      matsize  => ctx%matsize, &
      npp      => ctx%npp, &
      x        => ctx%x, &
      var      => self%block%variance(ctx%iblock), &
      vgm11    => self%vgm(1,1))

      call kriging_solve(npp, self%unbias + self%ndrift, 1, matA, rhsB, x, info)
      if (info /= 0) then
        call ssysv_fallback(npp, self%unbias + self%ndrift, 1, matA, rhsB, x, info)
        if (self%verbose) print '(A,I0)', &
          '  Cholesky failed; using SSYSV for block ', iblock
      end if
      if (info /= 0) then
        if (self%neglect_error) then
          x = IEEE_VALUE(0.0, IEEE_QUIET_NAN)
        else
          call kriging_error('solve_system_st', 'Singular matrix', iblock=iblock)
          return
        end if
      end if

      if (self%weight_correction) then
        x(1, 1:npp) = merge(x(1,1:npp), 0.0, x(1,1:npp) > 0.0)
        x(1, 1:npp) = x(1,1:npp) / sum(x(1,1:npp))
      end if

      !-- Kriging variance: C(0) - lambda^T * c0
      if (self%block%nblockpnt(iblock) == 1) then
        var = vgm11%cov0_val
      else
        !-- Block kriging: within-block variance via pairwise ST covariances
        !   All integration nodes share the block's time → dt = 0
        var = 0.0
        k1 = self%block%iblockpnt(iblock) - 1
        associate(w => self%grid%weight, coord => self%grid%coord, &
                  nb => self%block%nblockpnt(iblock))
          do i = 1, nb
            var = var + vgm11%cov0_val * w(k1+i) * w(k1+i)
            do j = i+1, nb
              lag_s = coord(:, k1+i) - coord(:, k1+j)
              var   = var + vgm11%cov_lag_st(lag_s, 0.0) * w(k1+i) * w(k1+j) * 2.0
            end do
          end do
        end associate
      end if
      var = max(var - dot_product(x(1,1:matsize), rhsB(1,1:matsize)), 0.0)
    end associate
  end subroutine solve_system_st


  !=============================================================================
  ! estimate_block_st — weighted estimate and optional SGSIM draw
  !=============================================================================
  subroutine estimate_block_st(self, ctx)
    class(t_kriging_st),     intent(inout) :: self
    class(t_kriging_st_ctx), intent(inout) :: ctx

    integer           :: ivar, k, nx, nnearb, ig_sim1
    real, allocatable :: v(:), w(:)
    real              :: avg(max(1, self%nsim)), total_weight(self%ngroups)

    nx      = max(1, self%nsim)
    ig_sim1 = self%nvar + 1   ! sim group index for variable 1 (valid only when nsim > 0)
    associate( &
      var    => self%block%variance(   ctx%iblock), &
      val    => self%block%estimate(:, ctx%iblock), &
      nnear  => ctx%nnear, &
      inear  => ctx%inear, &
      weight => ctx%weight)

      val          = 0.0
      avg          = 0.0
      total_weight = 0.0

      !-- SGSIM: previously simulated block contributions
      if (self%nsim > 0) then
        do k = 1, nnear(ig_sim1)
          val = val + self%block%estimate(:, inear(k, ig_sim1)) * weight(k, ig_sim1)
          avg = avg + self%block%estimate(:, inear(k, ig_sim1))
        end do
        total_weight(ig_sim1) = sum(weight(1:nnear(ig_sim1), ig_sim1))
        nnearb = nnear(ig_sim1)
      else
        nnearb = 0
      end if

      !-- Co-kriging: local mean of primary for Isaaks-Srivastava correction
      if (self%nvar > 1) then
        avg = avg + self%obs(1)%value(inear(1:nnear(1), 1))
        avg = avg / max(1, nnearb + nnear(1))
      end if

      !-- Observation contributions
      do ivar = 1, self%nvar
        if (nnear(ivar) == 0) then; total_weight(ivar) = 0.0; cycle; end if
        v = self%obs(ivar)%value(inear(1:nnear(ivar), ivar))
        w =                        weight(1:nnear(ivar), ivar)
        val = val + dot_product(w, v)
        total_weight(ivar) = sum(w)
        if (self%unbias /= 0 .and. ivar > 1) &
          val = val + total_weight(ivar) * (avg - sum(v)/nnear(ivar))
      end do

      if (self%unbias == 0 .and. self%sk_mean /= 0.0) &
        val = val + (1.0 - sum(total_weight)) * self%sk_mean
      if (self%nsim > 0) &
        val = val + sqrt(max(var, 0.0)) * self%block%sample(:, ctx%iblock)

      where (val < self%bounds(1)) val = self%bounds(1)
      where (val > self%bounds(2)) val = self%bounds(2)
    end associate
  end subroutine estimate_block_st


  !=============================================================================
  ! prepare_st — validate and size working arrays; propagate vgm for SGSIM
  !=============================================================================
  subroutine prepare_st(self)
    class(t_kriging_st), intent(inout) :: self
    integer :: ivar, jvar

    !-- Validate variograms and compute cov0_val for every entry
    do ivar = 1, self%nvar
      do jvar = 1, self%nvar
        call self%vgm(jvar, ivar)%compute_cov0()
        if (kriging_failed()) return
        if (.not. self%vgm(jvar, ivar)%is_valid_st(ivar, jvar)) then
          call kriging_error('prepare_st', 'Invalid ST variogram')
          return
        end if
      end do
    end do

    !-- vgm is always 1:nvar; group_ivar() maps sim-group indices to the correct
    !   variogram entry at evaluation time — no slot-0 copies needed.

    associate(npp => self%nppmax, matsize => self%matsize_max)
      npp = 0
      do ivar = 1, self%nvar
        npp = npp + self%obs(ivar)%nmax
      end do
      matsize = npp + self%ndrift + self%unbias
    end associate

    if (self%use_old_weight) then
      open(newunit=self%ifile, file=trim(self%weight_file), status='old')
      read(self%ifile, *)
    else if (self%store_weight) then
      open(newunit=self%ifile, file=trim(self%weight_file), status='replace')
      write(self%ifile, *) self%block%n, self%nvar, (self%obs(ivar)%nmax, ivar=1,self%nvar)
    end if
  end subroutine prepare_st


  !=============================================================================
  ! solve_st — main loop: kriging or SGSIM for every block
  !=============================================================================
  subroutine solve_st(self)
    use omp_lib
    class(t_kriging_st), intent(inout) :: self
    type(t_kriging_st_ctx), allocatable :: ctx
    integer :: ib, nb
    real, allocatable :: temp(:,:), temp_t(:), temp_v(:)

    call self%prepare()
    if (kriging_failed()) return
    nb = self%block%n

    if (self%verbose) print '(A)', 'Starting ST kriging loop'

    !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(ctx) IF(self%nsim==0 .and. .not. self%store_weight)
    allocate(ctx)
    call ctx%initialize(self)

    !$OMP DO SCHEDULE(DYNAMIC,1)
    do ib = 1, nb
      if (kriging_failed()) cycle
      ctx%iblock = ib
#ifdef _OPENMP
      if (self%verbose .and. omp_get_thread_num() == omp_get_num_threads()-1) &
        call progress(ib, nb)
#else
      if (self%verbose) call progress(ib, nb)
#endif
      call self%assemble_system(ctx)
      if (kriging_failed()) cycle
      if (ctx%npp > 1) call self%solve_system(ctx)
      if (kriging_failed()) cycle
      call ctx%assign_weight(self)
      call self%estimate_block(ctx)
    end do
    !$OMP END DO

    deallocate(ctx)
    !$OMP END PARALLEL

    if (kriging_failed()) return

    if (self%verbose) print '(A)', '  ST kriging completed.'

    !-- SGSIM post-processing: reorder estimate, coord, time, variance back
    if (self%nsim > 0) then
      nb = self%block%n
      allocate(temp  (self%nsim, nb))
      allocate(temp_t(nb))
      allocate(temp_v(nb))
      do ib = 1, nb
        temp  (:, self%block%order(ib)) = self%block%estimate(:, ib)
        temp_t(   self%block%order(ib)) = self%block%time(ib)
        temp_v(   self%block%order(ib)) = self%block%variance(ib)
      end do
      self%block%estimate = temp
      self%block%time     = temp_t
      self%block%variance = temp_v
      !-- Also reorder coord
      deallocate(temp)
      allocate(temp(3, nb))
      do ib = 1, nb
        temp(:, self%block%order(ib)) = self%block%coord(:, ib)
      end do
      self%block%coord = temp
    end if
  end subroutine solve_st


  !=============================================================================
  ! finalize_st — release all allocated memory and KD-tree pointers
  !=============================================================================
  subroutine finalize_st(self)
    class(t_kriging_st), intent(inout) :: self
    integer :: ivar

    if (allocated(self%obs)) then
      do ivar = 1, self%nvar
        if (associated(self%obs(ivar)%tree)) then
          call kdtree2_destroy(self%obs(ivar)%tree)
          self%obs(ivar)%tree => null()
        end if
        if (allocated(self%obs(ivar)%coord))    deallocate(self%obs(ivar)%coord)
        if (allocated(self%obs(ivar)%time))     deallocate(self%obs(ivar)%time)
        if (allocated(self%obs(ivar)%value))    deallocate(self%obs(ivar)%value)
        if (allocated(self%obs(ivar)%variance)) deallocate(self%obs(ivar)%variance)
        if (allocated(self%obs(ivar)%drift))    deallocate(self%obs(ivar)%drift)
      end do
      deallocate(self%obs)
    end if

    if (allocated(self%grid%coord))  deallocate(self%grid%coord)
    if (allocated(self%grid%time))   deallocate(self%grid%time)
    if (allocated(self%grid%weight)) deallocate(self%grid%weight)
    if (allocated(self%grid%drift))  deallocate(self%grid%drift)

    if (allocated(self%block%coord))       deallocate(self%block%coord)
    if (allocated(self%block%time))        deallocate(self%block%time)
    if (allocated(self%block%estimate))    deallocate(self%block%estimate)
    if (allocated(self%block%variance))    deallocate(self%block%variance)
    if (allocated(self%block%order))       deallocate(self%block%order)
    if (allocated(self%block%nblockpnt))   deallocate(self%block%nblockpnt)
    if (allocated(self%block%iblockpnt))   deallocate(self%block%iblockpnt)
    if (allocated(self%block%rangescale))  deallocate(self%block%rangescale)
    if (allocated(self%block%localnugget)) deallocate(self%block%localnugget)
    if (allocated(self%block%sample))      deallocate(self%block%sample)
    if (allocated(self%block%drift))       deallocate(self%block%drift)

    if (allocated(self%vgm)) deallocate(self%vgm)

    if (self%ifile /= 0) then; close(self%ifile); self%ifile = 0; end if
  end subroutine finalize_st

end module kriging_st
