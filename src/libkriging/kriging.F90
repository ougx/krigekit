 module kriging
  use, INTRINSIC    :: ieee_arithmetic
  use iso_fortran_env, only: input_unit, error_unit, output_unit

  use common
  use utils, only: set_seq, r8vec_normal_01
  use rotation
  use variogram
  use kdtree2_module
  use gaussian_quadrature
  implicit none

  ! base type for data
  type t_data
    integer               :: n=0           ! number of data nodes
    real   , allocatable  :: coord(:,:)    ! coordinates [ndim, n]
    real   , allocatable  :: drift(:,:)    ! drift values [ndrift, n]
    real   , allocatable  :: value(:)      ! values [n]
    real   , allocatable  :: variance(:)   ! variance of estimates or measurements [n]
  end type t_data

  ! observation points
  type, extends(t_data) :: t_grid
    real   , allocatable  :: weight(:)     ! weights of points[sum(nblockpnt)] within each block
  end type t_grid

  ! blocks
  type, extends(t_data) :: t_blockgrid
    integer               :: block_type = 0
    real   , allocatable  :: estimate(:,:) ! values [nsim, n]
    integer, allocatable  :: order(:)      ! the order of blocks; order(iblock) = iblock if not SGSIM
    integer, allocatable  :: nblockpnt(:)  ! number of nodes for each block  [n]
    integer, allocatable  :: iblockpnt(:)  ! starting index of nodes for each block  [n]
    real   , allocatable  :: rangescale(:) ! scaler of variogram range; used to account for data sparsity
    real   , allocatable  :: localnugget(:)! additional nugget at each block; account for spatial data uncertainty
    real   , allocatable  :: sample(:,:)   ! iid sample for simulation
  end type t_blockgrid

  type, extends(t_data) :: t_obsgrid
    integer               :: nmax = 0      ! maximum number of observations used for Kriging
    real                  :: maxdist = verylarge  ! maximum distance of nearest neighbor; when anisotropic search is used, this is the anisotropic distance
    real                  :: rotmat(3,3)   ! rotation matrix for anisotropic search
    type(kdtree2),pointer :: tree          ! kdtree for the data
    logical               :: need_search = .false.
    logical               :: anisotropic_search = .false.
  end type t_obsgrid

  type :: t_kriging
    logical               :: anisotropic_search = .false.
    logical               :: weight_correction = .false.
    logical               :: use_old_weight = .false.
    logical               :: store_weight = .false.
    logical               :: cross_validation = .false.
    character(len=1024)   :: weight_file = ""
    integer               :: ifile  = 0           ! file unit for weight file
    integer               :: ndim   = 2
    integer               :: nvar   = 1
    integer               :: ivar0 = 1            ! index of the first variable used for Kriging; 0 for SGSIM
    integer               :: ndrift = 0
    integer               :: unbias = 1
    integer               :: nsim   = 0
    integer               :: iblock = 0           ! index of current block
    integer               :: nppmax = 0           ! maximum number of observations used for Kriging
    integer               :: matsize_max = 0      ! maximum size of matrix = nppmax + ndrift + unbias
    real                  :: bounds(2) = [-verylarge, verylarge]
    real                  :: sk_mean = 0.0        ! simple kriging mean
    type(t_obsgrid)  , allocatable :: obs(:)
    type(t_grid)     , allocatable :: grid
    type(t_blockgrid), allocatable :: block
    type(vgm_struct) , allocatable :: vgm(:,:)
  contains
    procedure             :: initialize
    procedure             :: set_obs
    procedure             :: set_obs_drift
    procedure             :: set_vgm
    procedure             :: set_grid
    procedure             :: set_grid_drift
    procedure             :: set_sim
    procedure             :: set_search
    procedure             :: search_neighbors
    procedure             :: calc_covariance
    procedure             :: assemble_linear_system
    procedure             :: solve_linear_system
    procedure             :: estimate_block
    procedure             :: prepare
    procedure             :: solve
    procedure             :: write_weight
    procedure             :: read_weight
    procedure             :: finalize
  end type

  ! ============== for parallel kriging ================
  ! holds per-block working state, one instance per thread
  type :: t_kriging_ctx   ! kriging context struct; local to each thread
    integer              :: iblock
    integer              :: npp
    integer              :: matsize
    integer, allocatable :: nnear(:)       ! nnear(0:nvar): neighbours found per variable, index starting from 0 for neighbour blocks for SGSIM
    integer, allocatable :: inear(:,:)     ! inear(nmax, 0:nvar): neighbour indices, index starting from 0 for neighbour blocks for SGSIM
    real   , allocatable :: weight(:,:)    ! weight(nmax, 0:nvar): neighbour weights, index starting from 0 for neighbour blocks for SGSIM
    real,    allocatable :: sqdist(:)      ! squared dist(nmax): used to determine if a point is within maxdist
    real   , allocatable :: x(:,:)         ! weights
    real   , allocatable :: matA(:,:)      ! matrix
    real   , allocatable :: rhsB(:,:)      ! right hand side
  contains
    procedure             :: initialize => initialize_kriging_ctx
    procedure             :: assign_weight  ! assign the weight from x to each variable
  end type t_kriging_ctx

  contains

  subroutine initialize(self, ndim_, nvar, ndrift, unbias, nsim, anisotropic_search,  &
                        weight_correction, use_old_weight, store_weight, cross_validation, &
                        verbose_, weight_file, bounds, sk_mean)
    class(t_kriging)      :: self
    integer, intent(in), optional   :: ndim_           ! number of dimensions; must be defined
    integer, intent(in), optional   :: nvar, ndrift, unbias, nsim
    real,    intent(in), optional   :: bounds(2)
    real,    intent(in), optional   :: sk_mean
    logical, intent(in), optional   :: anisotropic_search, weight_correction, use_old_weight, store_weight, verbose_, cross_validation
    character(len=*), intent(in), optional :: weight_file
    if (present(ndim_)) then; ndim = ndim_; else; ndim = 0; end if
    if (present(nvar))               self%nvar               = nvar
    if (present(ndrift))             self%ndrift             = ndrift
    if (present(unbias))             self%unbias             = unbias
    if (present(nsim))               self%nsim               = nsim
    if (present(anisotropic_search)) self%anisotropic_search = anisotropic_search
    if (present(weight_correction))  self%weight_correction  = weight_correction
    if (present(use_old_weight))     self%use_old_weight     = use_old_weight
    if (present(store_weight))       self%store_weight       = store_weight
    if (present(weight_file))        self%weight_file        = weight_file
    if (present(bounds))             self%bounds             = bounds
    if (present(sk_mean))            self%sk_mean            = sk_mean
    if (present(cross_validation))   self%cross_validation   = cross_validation
    if (present(verbose_))           verbose                 = verbose_
    allocate(self%obs(nvar))
    allocate(self%grid)
    allocate(self%block)
    if (self%nsim>0) self%ivar0 = 0
    allocate(self%vgm(self%ivar0:nvar, self%ivar0:nvar))
    call init_nan()
    ! sanity check
    if (self%use_old_weight .and. self%weight_file=="") stop "ERROR: use_old_weight requires weight_file to be specified"
    if (self%store_weight .and. self%weight_file=="") stop "ERROR: store_weight requires weight_file to be specified"
    if (self%store_weight .and. self%use_old_weight) stop "ERROR: store_weight and use_old_weight are mutually exclusive"
    if (self%cross_validation .and. self%nsim>0) stop "ERROR: nsim>0 and cross_validation are mutually exclusive"
  end subroutine initialize

  ! set grid and block:
  !   case 1; block_type= 0: each block contain one node; default
  !   case 2; block_type=-4: each block contain more than one node;
  !           coordinates of the nodes are generated by Gaussian quadrature (dx,dy,dz are specified); the weight is based on gaussian quadrature
  !   case 3; block_type> 0: each block contain more than one node; coordinates of the nodes are specified, weights will be equally distributed or specified
  ! Kriging will always be performed for each block.
  subroutine set_grid(self, coord, block_type, blocksize, nblockpnt, pointweight, rangescale, localnugget)
    ! set grid coordinates
    class(t_kriging)      :: self
    integer, intent(in), optional :: block_type
    real   , intent(in), optional :: coord(:,:)        ! coordinates; doesnot needed for cross validation
    real   , intent(in), optional :: blocksize(:,:)    ! block size (dx, dy, dz) used for Gaussian quadrature [ndim, n]    ! TODO: check coord has the same ndim
    integer, intent(in), optional :: nblockpnt(:)      ! number of nodes for each block  [n]
    real   , intent(in), optional :: pointweight(:)    ! block blockweights [sum(nblockpnt)]
    real   , intent(in), optional :: rangescale(:)     ! scaler of variogram ranges at each block; used to account for data sparsity
    real   , intent(in), optional :: localnugget(:)    ! additional nugget at each block added to the global nugget
    ! local
    integer               :: ngrid, nn, nb
    integer               :: iblock, igrid, idim, isim

    if (self%obs(1)%n==0) error stop 'Observation needs to be set first.'
    if (present(block_type)) self%block%block_type = block_type

    if (self%cross_validation) then
      ngrid = self%obs(1)%n
      self%block%n = ngrid
      allocate(self%block%coord, source=self%obs(1)%coord)
      allocate(self%grid%coord , source=self%obs(1)%coord)
      allocate(self%block%nblockpnt(ngrid)); self%block%nblockpnt=1
      allocate(self%block%iblockpnt, source=[(igrid, igrid=1,ngrid)])
      allocate(self%grid%weight(ngrid)); self%grid%weight=1.0
      self%obs(1)%nmax = self%obs(1)%nmax + 1 ! search will include itself but will be excluded from the search results
      if (self%ndrift>0) allocate(self%block%drift, source=self%obs(1)%drift)
    else
      if (.not. present(coord)) error stop 'coord needs to be provided.'
      if (ndim == 0) then
        ndim = size(coord, 1)
      else
        if (ndim /= size(coord, 1)) error stop 'ndim /= size(coord, 1) for grid'   ! TODO: check coord has the same ndim
      end if
      ngrid = size(coord, 2)

      ! set up the block
      if (self%block%block_type == 0) then
        self%block%n = ngrid
        allocate(self%block%coord, source=coord)
        allocate(self%grid%coord, source=coord)
        allocate(self%block%nblockpnt(ngrid)); self%block%nblockpnt=1
        allocate(self%block%iblockpnt, source=[(igrid, igrid=1,ngrid)])
        allocate(self%grid%weight(ngrid)); self%grid%weight=1.0
      else if (self%block%block_type == -4) then
        nb = (4**ndim)
        self%block%n = ngrid
        allocate(self%block%coord, source=coord)
        allocate(self%grid%coord(ndim, ngrid*nb))
        allocate(self%block%nblockpnt(ngrid)); self%block%nblockpnt=4**ndim
        allocate(self%block%iblockpnt, source=[((igrid-1)*(4**ndim)+1, igrid=1,ngrid)])
        allocate(self%grid%weight((4**ndim)*ngrid))
        igrid = 0
        do iblock=1,self%block%n
          self%grid%weight(igrid+1:igrid+nb) = 1.0/4**ndim    ! TODO: set up gaussian quadrature weights
          igrid = igrid + nb
        end do  ! iblock
      else
        self%grid%n = ngrid
        self%block%n = size(nblockpnt)
        allocate(self%grid%coord, source=coord)
        allocate(self%block%nblockpnt, source=nblockpnt)  ! TODO: check if nblockpnt is correct
        allocate(self%block%iblockpnt(self%block%n))
        igrid = 0
        do iblock=1,self%block%n
          self%block%iblockpnt(iblock) = igrid + 1
          igrid = igrid + nblockpnt(iblock)
        end do  ! iblock
        if (present(pointweight)) then
          allocate(self%grid%weight, source=pointweight) ! TODO: check if pointweight is correct
        else
          allocate(self%grid%weight(self%grid%n))
          igrid = 0
          do iblock=1,self%block%n
            nb = nblockpnt(iblock)
            self%grid%weight(igrid+1:igrid+nb) = 1.0/nb
            igrid = igrid + nb
          end do  ! iblock
        end if
        ! calculate the block coordinates
        allocate(self%block%coord(ndim, self%block%n))
        igrid = 0
        do iblock=1,self%block%n
          nn = nblockpnt(iblock)
          do idim=1,ndim
            self%block%coord(idim, iblock) = sum(self%grid%coord(idim,igrid+1:igrid+nn)*self%grid%weight(igrid+1:igrid+nn))
          end do
          igrid = igrid + nn
        end do  ! iblock
      end if
    end if
    allocate(self%block%order(self%block%n))
    allocate(self%block%localnugget(self%block%n))
    allocate(self%block%rangescale(self%block%n))
    allocate(self%block%estimate(max(self%nsim,1),self%block%n))
    allocate(self%block%variance(self%block%n))

    call set_seq(self%block%order, self%block%n)
    self%block%variance = 0.0
    self%block%estimate = IEEE_VALUE(0.0, IEEE_QUIET_NAN)
    if (present(rangescale) .and. .not. self%cross_validation) then
      self%block%rangescale=rangescale
    else
      self%block%rangescale=1.0
    end if
    if (present(localnugget) .and. .not. self%cross_validation) then
      self%block%localnugget=localnugget
    else
      self%block%localnugget=0.0
    end if
  end subroutine set_grid


  subroutine set_grid_drift(self, drift)
    class(t_kriging)      :: self
    real   , intent(in)   :: drift(:,:)        ! drifts
    if (.not. allocated(self%block)) error stop 'Call initialize() before set_grid_drift.'
    if (self%block%n==0) error stop 'Grid needs to be set before adding drift.'
    if (self%ndrift==0) error stop 'grid/block drift is specified but ndrift==0'
    if (size(drift, 1) /= self%ndrift) error stop 'size(drift, 1) /= ndrift'
    if (size(drift, 2) /= self%block%n) error stop 'size(drift, 2) /= block%n; one drift value per block, not per grid node'
    allocate(self%block%drift, source=drift)
  end subroutine set_grid_drift


  subroutine set_vgm(self, ivar, jvar, spec)
    class(t_kriging)  :: self  ! TODO: check ivar and jvar
    character(*), intent(in)      :: spec
    integer     , intent(in)      :: ivar, jvar
    ! spec is a string of the form: vtype, nugget, sill, a_major, a_minor1, a_minor2, azimuth, dip, plunge
    call self%vgm(jvar, ivar)%add(spec=spec)
    if (jvar/=ivar) call self%vgm(ivar, jvar)%add(spec=spec)
  end subroutine set_vgm


  subroutine set_obs(self, ivar, coord, value, variance, nmax, maxdist)
    ! reading OBS file
    use rotation, only             : rotate
    use kdtree2_module, only       : kdtree2_create
    class(t_kriging)              :: self
    integer, intent(in)           :: ivar
    integer, intent(in), optional :: nmax
    real, intent(in)              :: coord(:,:), value(:)
    real, intent(in), optional    :: variance(:), maxdist

    ! local
    integer               :: i, nobs
    real   , allocatable  :: temp(:,:)

    if (ndim == 0) then
      ndim = size(coord, 1)
    else
      if (ndim /= size(coord, 1)) error stop 'ndim /= size(coord, 1) for grid'   ! TODO: check coord has the same ndim
    end if
    associate(obs=>self%obs(ivar))
      nobs = size(coord, 2)
      obs%n = nobs
      if (present(nmax)) then
        obs%nmax = nmax
      else
        obs%nmax = huge(obs%n)
      end if
      if (present(maxdist)) obs%maxdist = maxdist**2  ! store maxdist^2 instead of maxdist for comparison to kdtree reults
      if (present(variance)) then
        allocate(obs%variance, source=variance)
      else
        allocate(obs%variance(nobs))
        obs%variance = 0.0
      end if
      allocate(obs%value, source=value)
      allocate(obs%coord, source=coord)
      obs%rotmat = reshape([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], [3,3])
    end associate
  end subroutine set_obs


  subroutine set_obs_drift(self, ivar, drift)
    class(t_kriging)      :: self
    integer, intent(in)   :: ivar              ! index of the variable
    real   , intent(in)   :: drift(:,:)        ! drifts
    if (.not. allocated(self%obs)) error stop 'Call initialize() before set_obs_drift.'
    if (self%obs(ivar)%n==0) error stop 'Observation needs to be set before adding drift.'
    if (self%ndrift==0) then
      error stop 'Observation drift is specified but ndrift==0'
    end if
    if (size(drift, 1) /= self%ndrift) error stop 'size(drift, 1) /= ndrift'
    if (size(drift, 2) /= self%obs(ivar)%n) error stop 'size(drift, 2) /= nobs'
    allocate(self%obs(ivar)%drift, source=drift)
  end subroutine set_obs_drift


  ! set up for SGSIM, execute only once before set_search
  subroutine set_sim(self, randpath, sample)
    ! reading OBS file
    class(t_kriging)              :: self
    real   , intent(in), optional :: sample(:,:)       ! sample from standard normal distribution
    integer, intent(in), optional :: randpath(:)       ! random path of each block if nsim>0 [n]; if not sepcified, it will be generated.
    ! local
    real   , allocatable          :: temp(:,:)
    integer                       :: iblock, ifile, isim

    ! check if Grid is set
    if (self%block%n==0) error stop 'Grid needs to be set first.'
    if (any(self%obs%n==0)) error stop 'Observations need to be set first.'
    if (self%nsim>0) then
      associate(obs=>self%obs(1))
        if (present(randpath)) then
          self%block%order = randpath
        else
          call set_seq(self%block%order, self%block%n, .TRUE.)
          open(newunit=ifile, file='sgs_path.dat', status='replace')
          write(ifile, '(A,x,I0)') 'SGSIM_Path', self%block%n    ! TODO
          write(ifile, '((1I0))') self%block%order
          close(ifile)
        end if

        allocate(self%block%sample(self%nsim, self%block%n))
        if (present(sample)) then
          self%block%sample = sample
        else
          do isim=1, self%nsim
            call r8vec_normal_01(self%block%n, self%block%sample(isim, :))
          end do
          open(newunit=ifile, file='sgs_sample.dat', status='replace')
          write(ifile, '(A,x,2I10)') 'SGSIM_Sample', self%nsim, self%block%n    ! TODO
          do iblock=1,self%block%n
            write(ifile, '(*(G0.7,x))') self%block%sample(:, iblock)
          end do
          close(ifile)
        end if
        ! reorder the coordinates based on random path
        self%block%coord     = self%block%coord (:, self%block%order)
        self%block%iblockpnt = self%block%iblockpnt(self%block%order)
        self%block%nblockpnt = self%block%nblockpnt(self%block%order)
        self%block%rangescale = self%block%rangescale(self%block%order)
        self%block%localnugget = self%block%localnugget(self%block%order)
        if (self%ndrift>0) self%block%drift=self%block%drift(:, self%block%order)

        ! extend obs coordinates to include all blocks for SGSIM
        allocate(temp(ndim, obs%n+self%block%n))
        temp(:,1:obs%n) = obs%coord
        temp(:,obs%n+1:) = self%block%coord
        call move_alloc(temp, obs%coord)
      end associate
    end if
  end subroutine set_sim

  ! set up search
  subroutine set_search(self,ivar,anis1,anis2,azimuth,dip,plunge)
    use rotation, only             : calc_rotmat, sub_rotate
    use kdtree2_module, only       : kdtree2_create
    class(t_kriging)              :: self
    integer, intent(in)           :: ivar
    real, intent(in)              :: anis1,anis2,azimuth,dip,plunge

    ! local
    real   , allocatable          :: rcoord(:,:)  ! rotated coordinates

    associate(obs=>self%obs(ivar), need_search=>self%obs(ivar)%need_search, anisotropic_search=>self%obs(ivar)%anisotropic_search)
      anisotropic_search = (abs(anis1-1.0)>EPSLON .or. abs(anis2-1.0)>EPSLON) .and. self%anisotropic_search

      if (ivar==1 .and. self%nsim>0) then
        obs%nmax = min(obs%nmax, obs%n+self%block%n)
        need_search = obs%n+self%block%n>obs%nmax
      else
        obs%nmax = min(obs%nmax, obs%n)
        need_search = obs%n>obs%nmax
      end if

      if (need_search) then
        if (anisotropic_search) then
          allocate(rcoord, mold=obs%coord)
          obs%rotmat = calc_rotmat(azimuth,dip,plunge,anis1,anis2)
          call sub_rotate(obs%rotmat, ndim, size(obs%coord,2), obs%coord, rcoord)
          obs%tree => kdtree2_create(rcoord, sort=.false., rearrange=.true.)
        else
          obs%tree => kdtree2_create(obs%coord, sort=.false., rearrange=.true.)
        end if
      end if
    end associate
  end subroutine set_search

  ! initialize the kriging context for thread private variables
  subroutine initialize_kriging_ctx(self, krige)
    class(t_kriging_ctx)     :: self
    class(t_kriging)         :: krige

    integer                  :: ivar, mmax
    mmax = maxval(krige%obs%nmax)
    associate(npp => krige%nppmax, matsize => krige%matsize_max)
      if (.not. krige%use_old_weight) then
        allocate(self%sqdist(mmax))
        allocate(self%matA(matsize, matsize))
        allocate(self%rhsB(1, matsize))
        self%sqdist = 0.0
      end if
      allocate(self%nnear (     0:krige%nvar))
      allocate(self%inear (mmax,0:krige%nvar))
      allocate(self%weight(mmax,0:krige%nvar))
      allocate(self%x     (1, matsize))
      ! initialize values
      self%nnear(0) = 0 ! neighbor blocks
      call set_seq(self%inear(1:krige%obs(1)%nmax, 0), krige%obs(1)%nmax)
      do ivar = 1, krige%nvar
        self%nnear(ivar) = krige%obs(ivar)%nmax
        call set_seq(self%inear(1:mmax, ivar), mmax)
      end do
    end associate
  end subroutine initialize_kriging_ctx


  subroutine prepare(self)
    class(t_kriging)      :: self
    integer               :: ivar, jvar

    ! check if everything is set
    if (self%ndrift>0) then
      if (.not. allocated(self%block%drift)) error stop 'Grid drift is not set while ndrift > 0.'
      do ivar=1, self%nvar
        if (.not. allocated(self%obs(ivar)%drift)) error stop 'Observation drift is not set while ndrift > 0.'
      end do
    end if
    do ivar=self%ivar0, self%nvar
      do jvar=self%ivar0, self%nvar
        if (self%vgm(jvar, ivar)%nstruct==0) error stop 'Variogram is not set.'
      end do
    end do

    associate(npp=>self%nppmax, matsize=>self%matsize_max, ifile=>self%ifile)
      npp = 0
      do ivar = 1, self%nvar
        npp = npp + self%obs(ivar)%nmax
      end do
      matsize = npp + self%ndrift + self%unbias
      if (self%use_old_weight) then
        open(newunit=ifile, file=trim(self%weight_file), status='old')
        read(ifile,*) ! skip header
      else
        if (self%store_weight) then
          open(newunit=ifile, file=trim(self%weight_file), status='replace')
          write(ifile,*) self%block%n,self%nvar,self%obs%nmax
        end if
      end if
      if (self%nsim>0) then
        self%vgm(0, 0) = self%vgm(1, 1)
        do ivar = 1, self%nvar
          self%vgm(0, ivar) = self%vgm(1, ivar)
          self%vgm(ivar, 0) = self%vgm(ivar, 1)
        end do
      end if
    end associate
  end subroutine prepare

  ! solve for all blocks
  subroutine solve(self)
    class(t_kriging)      :: self
    ! local
    type(t_kriging_ctx), allocatable :: ctx   ! one per thread
    integer               :: ib, ivar, ifile
    real, allocatable     :: temp(:,:)

    ! allocation
    associate(nb=>self%block%n)
      call self%prepare()

      ! start the block loop
      if (verbose) print*, "Starting Kriging loop"

      !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(ctx)
      allocate(ctx)               ! Required OpenMP 4.0+ for allocatable private variable
      call ctx%initialize(self)
      !$OMP DO SCHEDULE(DYNAMIC, 1)
      do ib = 1, nb
        ctx%iblock = ib
        if (verbose) call progress(real(ib)/real(nb))
        if (self%use_old_weight) then
          call self%read_weight(ctx)
        else
          ! assemble linear system
          call self%assemble_linear_system(ctx)
          ! solve linear system
          if (ctx%npp>1) call self%solve_linear_system(ctx) ! use one observation when self%npp=1
        end if
        if (self%store_weight) then
          call self%write_weight(ctx)
        else
          call self%estimate_block(ctx)
        end if
      end do
      !$OMP END DO
      !$OMP END PARALLEL
      if (verbose) print*,""
      if (self%nsim>0) then
        allocate(temp, source=self%block%estimate)
        do ib = 1, self%block%n
          self%block%estimate(:, self%block%order(ib)) = temp(:, ib)
        end do
      end if
    end associate
  end subroutine solve


  subroutine search_neighbors(self, ivar, ctx)
    class(t_kriging)              :: self
    class(t_kriging_ctx)          :: ctx
    integer, intent(in)           :: ivar

    ! local
    integer                       :: i
    real                          :: newloc(ndim,1)    ! rotated coordinates of the new location to be estimated
    logical, allocatable          :: mask(:) !, mask1(:)
    logical, allocatable          :: is_obs(:)
    type(kdtree2_result), allocatable :: results(:) ! nearest neighbor search results

    associate(&
      iblock =>ctx%iblock, &
      nsim   =>self%nsim, &
      nobs   =>self%obs(ivar)%n, &
      nmax   =>self%obs(ivar)%nmax, &
      obsloc =>self%obs(ivar)%tree%the_data, &
      xloc   =>self%block%coord(:, ctx%iblock:ctx%iblock), &
      inear  =>ctx%inear(:,ivar), & ! obs
      nnear  =>ctx%nnear(ivar), &   ! obs
      inearb =>ctx%inear(:,0), &    ! block
      nnearb =>ctx%nnear(0), &      ! block
      dist   =>ctx%sqdist, &
      maxdist=>self%obs(ivar)%maxdist, &
      rotmat =>self%obs(ivar)%rotmat)
      !

      if (self%obs(ivar)%anisotropic_search) then
        call sub_rotate(rotmat, ndim, 1, xloc, newloc)
      else
        newloc = xloc
      end if

      if (nsim>0 .and. ivar==1) then
        if (nmax<nobs+iblock) then
          call kdtree2_n_nearest_maxidx(self%obs(ivar)%tree, newloc(:,1), nmax, results, nobs+iblock-1)
          allocate(is_obs, source=results%idx<=nobs)
          nnear            = count(is_obs)
          nnearb           = nmax - nnear
          inear (1:nnear)  = pack(results%idx, is_obs)
          inearb(1:nnearb) = pack(results%idx, .not. is_obs)
          dist(1:nnear)    = pack(results%dis, is_obs)    ! assume no co-located blocks, so estimated blocks will never exactly match the block to be estimated
        else
          ! no search
          nnear  = nobs
          nnearb = iblock
          ! inear no need to touch until search is needed
          dist(1:nnear) = [(sum((newloc(:,1)-obsloc(:,i))**2), i=1, nnear)]
        end if
      else
        if (nmax<nobs) then
          call kdtree2_n_nearest       (self%obs(ivar)%tree, newloc(:,1), nmax, results)
          nnear          = nmax
          inear(1:nnear) = results%idx
          dist(1:nnear) = results%dis
        else
          ! no search
          nnear = nobs                                                         ! reset for corss validation
          if (self%cross_validation) call set_seq(inear(1:nnear), nnear)       ! reset for corss validation
          dist(1:nnear) = [(sum((newloc(:,1)-obsloc(:,i))**2), i=1, nnear)]
        end if
        if (self%cross_validation) then
          ! exclude self from neighbors for cross validation
          do i = 1, nnear
            if (inear(i)==iblock) then
              nnear = nnear - 1
              inear(i:nnear) = inear(i+1:nnear+1)
              exit
            end if
          end do
        end if
      end if

      ! finally check maximum distance
      mask = dist(1:nnear)<=maxdist
      if (any(.not. mask)) then
        nnear = count(mask)
        inear(1:nnear) = pack(inear, mask)
        dist (1:nnear) = pack(dist , mask)
      end if
    end associate
  end subroutine search_neighbors


  subroutine calc_covariance(self, ctx, ir0, ic0, ivar, jvar)
    class(t_kriging)              :: self
    class(t_kriging_ctx)          :: ctx
    integer, intent(in)           :: ivar, jvar ! index of variables
    integer, intent(in)           :: ir0, ic0   ! starting row and column

    ! local
    integer                       :: i, j, k, k1, istart, n, n2
    real                          :: lag(3)=0.0, tmp

    associate( &
      obs=>self%obs(ivar), &
      nnear=>ctx%nnear(ivar), &
      inear=>ctx%inear(1:ctx%nnear(ivar), ivar), &
      rs=>self%block%rangescale(ctx%iblock), &
      ln=>self%block%localnugget(ctx%iblock))
      if (jvar==-1) then
        ! setting rhsB: calculating covariance between data and block to be eastimated
        associate(vgm=>self%vgm(1, ivar))
          do i = 1, nnear
            tmp = 0
            k1 = self%block%iblockpnt(ctx%iblock)-1
            do k = 1, self%block%nblockpnt(ctx%iblock)
              lag(1:ndim) = obs%coord(:,inear(i)) - self%grid%coord(:, k1+k)
              tmp = tmp + vgm%cov_lag(lag/rs) * self%grid%weight(k1+k)
            end do
            ctx%rhsB(1, ir0+i) = tmp
          end do
        end associate
      else
        ! setting matA: calculating covariance between data points
        associate(obs2=>self%obs(jvar), nnear2=>ctx%nnear(jvar), inear2=>ctx%inear(1:ctx%nnear(jvar), jvar), vgm=>self%vgm(jvar, ivar))
          do i = 1, nnear
            if (ivar==jvar) then
              istart = i + 1
              ctx%matA(ic0+i, ir0+i) = vgm%cov0 + obs%variance(inear(i)) + ln
            else
              istart = 1
            end if
            do j = istart, nnear2
              lag(1:ndim) = obs%coord(:,inear(i)) - obs2%coord(:,inear2(j))
              ctx%matA(ic0+j, ir0+i) = vgm%cov_lag(lag/rs)
            end do
          end do
        end associate
      end if
    end associate
  end subroutine calc_covariance


  subroutine assemble_linear_system(self, ctx)
    class(t_kriging)      :: self
    class(t_kriging_ctx)  :: ctx
    ! local
    integer               :: nvar, ivar, jvar, ivar0
    integer               :: irow1, irow2, icol1, icol2

    ! search for neighbor
    associate(nvar=>self%nvar, dist=>ctx%sqdist, npp=>ctx%npp)
    npp = 0
    ! exact-match detection
    do ivar = 1, nvar
      call self%search_neighbors(ivar, ctx)
      npp = npp + ctx%nnear(ivar)
      if (ivar==1 .and. minval(dist(1:ctx%nnear(ivar)))<=EPSLON) then
        npp = 1    ! signal exact match
        ctx%x=0.0
        ctx%x(:,1)=1.0
        ctx%nnear(ivar) = 1
        ctx%inear(1,ivar) = ctx%inear(minloc(dist(1:ctx%nnear(ivar)), dim=1),ivar)
        ctx%nnear(0) = 0
        self%block%variance(ctx%iblock) = self%obs(1)%variance(ctx%inear(1,ivar)) + self%block%localnugget(ctx%iblock)
        do jvar = 2, nvar
          ctx%nnear(jvar) = 0
        end do
        return
      end if
    end do
    end associate

    ! check if enough neighbors
    if (ctx%nnear(0) + ctx%nnear(1) == 0) then
      write(*,*) 'not enough neighbors for kriging at block ', ctx%iblock
      error stop
    end if

    ! set up matrix
    associate( &
    iblock=>ctx%iblock, &
    matA=>ctx%matA, &
    rhsB=>ctx%rhsB, &
    inear=>ctx%inear, &
    nnear=>ctx%nnear, &
    matsize=>ctx%matsize, &
    npp=>ctx%npp, &
    ndrift=>self%ndrift)
      matsize = npp + self%unbias + ndrift
      irow1 = 0
      do ivar = self%ivar0, nvar
        irow2 = irow1 + nnear(ivar)
        icol1 = 0
        call self%calc_covariance(ctx,irow1,icol1,ivar, -1)
        do jvar = self%ivar0, nvar
          icol2 = icol1 + nnear(jvar)
          if (jvar>=ivar .and. nnear(jvar)>0) then
            call self%calc_covariance(ctx,irow1,icol1,ivar,jvar)
          end if
          icol1 = icol2
        end do
        if (ndrift>0) then
          icol2 = icol1+ndrift
          matA(icol1+1:icol2, irow1+1:irow2) = self%obs(ivar)%drift(:, inear(1:nnear(ivar), ivar))
        end if
        irow1 = irow2
      end do
      if (ndrift>0) rhsB(1, npp+1:npp+ndrift) = self%block%drift(:, iblock)

      if (self%unbias==1) then
        matA(matsize, 1:ctx%npp) = 1.0
        rhsB(1, matsize) = 1.0
      end if
      ! assign values to lower triangle from upper triangle (assuming symmetric matrix)
      do irow1 = 1, npp
        do icol1 = irow1+1, npp
          matA(irow1, icol1) = matA(icol1, irow1)
        end do
      end do
      matA(npp+1:matsize, npp+1:matsize) = 0.0
    end associate
  end subroutine assemble_linear_system


  subroutine solve_linear_system(self, ctx)
    use solver
    implicit none
    class(t_kriging)      :: self
    class(t_kriging_ctx)  :: ctx

    !-------------------------
    ! Inputs
    integer               :: info
    integer               :: i, j, k1, ivar
    real                  :: lag(3)
    logical, allocatable  :: is_grid(:)

    associate( &
    iblock=>ctx%iblock, &
    matA=>ctx%matA, &
    rhsB=>ctx%rhsB, &
    matsize=>ctx%matsize, &
    npp=>ctx%npp, &
    x=>ctx%x, &
    unbias=>self%unbias, &
    ndrift=>self%ndrift)

      call kriging_solve( &
        npp, unbias+ndrift, &
        matA(1:npp,1:npp), &
        matA(1:npp,npp+1:matsize), &
        rhsB(1,1:npp), &
        rhsB(1,npp+1:matsize), &
        x(1,1:npp), x(1,npp+1:matsize), &
        info)

      if (info /= 0) then
        call gaussian_elimination(matsize, matA, rhsB(1,:), x(1,:), info)
      end if

      if (info /= 0) then
        write(*,*) 'Singluar matrix at block ', iblock
        error stop
      end if

      if (self%weight_correction) then
        x(1,1:npp) = merge(x(1,1:npp), 0.0, x(1,1:npp)>0)
        x(1,1:npp) = x(1,1:npp) / sum(x(1,1:npp))
      end if

      ! calculate kriging variance
      associate(vgm=>self%vgm(1, 1), &
        var=>self%block%variance(iblock), &       ! write to shared array for different threads
        weight=>self%grid%weight, &
        coord=>self%grid%coord, &
        nblockpnt=>self%block%nblockpnt(iblock))
        if (nblockpnt==1) then
          var = vgm%cov0
        else
          var = 0.0
          k1 = self%block%iblockpnt(iblock)-1
          do i = 1, nblockpnt
            var = var + vgm%cov0 * weight(k1+i) * weight(k1+i)
            do j = i+1, nblockpnt
              lag(1:ndim) = coord(:, k1+i) - coord(:, k1+j)
              var = var + vgm%cov_lag(lag) * weight(k1+i) * weight(k1+j) * 2.0
            end do
          end do
        end if
        var = max(var - dot_product(x(1,1:matsize), rhsB(1,1:matsize)), 0.0)
      end associate
      call ctx%assign_weight()
    end associate
  end subroutine solve_linear_system


  subroutine assign_weight(self)
    class(t_kriging_ctx)  :: self

    ! local
    integer               :: ivar, k1
    ! assign the weights to respective variables
    k1 = 0
    do ivar = lbound(self%nnear, 1), ubound(self%nnear, 1)
      if (self%nnear(ivar)==0) cycle
      self%weight(1:self%nnear(ivar), ivar) = self%x(1, k1+1:k1+self%nnear(ivar))
      k1 = k1 + self%nnear(ivar)
    end do
  end subroutine assign_weight

  ! calculate weighted average
  subroutine estimate_block(self, ctx)
    implicit none
    class(t_kriging)      :: self
    class(t_kriging_ctx)  :: ctx
    integer               :: ivar, k, knear, ival, nx, nnearb
    real, allocatable     :: v(:)           ! store observation values
    real, allocatable     :: w(:)           ! store weights
    real                  :: avg(max(1, self%nsim)), total_weight(self%ivar0:self%nvar)

    nx = max(1, self%nsim)
    associate(&
      sam=>self%block%sample  (:, ctx%iblock), &
      var=>self%block%variance(   ctx%iblock), &
      val=>self%block%estimate(:, ctx%iblock), &
      nnear=>ctx%nnear, &
      inear=>ctx%inear, &
      weight=>ctx%weight)
      val = 0
      avg = 0.0
      if (self%nsim>0) then
        do k = 1, nnear(0)
          val = val + self%block%estimate(:, inear(k,0)) *weight(k,0)
          avg = avg + self%block%estimate(:, inear(k,0))
        end do
        total_weight(0) = sum(weight(1:nnear(0), 0))
        nnearb = nnear(0)
      else
        nnearb = 0
      end if
      if (self%nvar>1) then
        ! calculate the average of the primary variable
        avg = avg + self%obs(1)%value(inear(1:nnear(1), 1))
        avg = avg / (nnearb + nnear(1))
      end if
      do ivar = 1, self%nvar
        if (nnear(ivar)==0) then
          total_weight(ivar) = 0.0
          cycle
        end if
        v = self%obs(ivar)%value(inear(1:nnear(ivar), ivar))
        w =                     weight(1:nnear(ivar), ivar)
        val = val + dot_product(w,v)
        total_weight(ivar) = sum(w)
        if (self%unbias/=0 .and. ivar>1) &
          val = val + total_weight(ivar) * (avg - sum(v)/nnear(ivar)) ! ISAAKS and SRIVASTAVA, An Introduction to Applied Geostatistics, pp410
      end do
      if (self%unbias==0 .and. self%sk_mean/=0.0) val = val + (1.0 - sum(total_weight)) * self%sk_mean
      if (self%nsim>0) then
        val = val + sqrt(var) * sam
      end if
      val = merge(val, self%bounds(1), val<self%bounds(1))
      val = merge(val, self%bounds(2), val>self%bounds(2))
    end associate
  end subroutine estimate_block

  subroutine write_weight(self, ctx)
    class(t_kriging)      :: self
    class(t_kriging_ctx)  :: ctx
    ! local
    integer               :: ii
    associate(&
    ib=>ctx%iblock, &
    order=>self%block%order, &
    var=>self%block%variance(ctx%iblock))
      write(self%ifile, '(I0,x,G0.12,99(x,I0))') order(ib), var, ctx%nnear(0:self%nvar)
      write(self%ifile, '(*(:2x,I0))') (ctx%inear(1:ctx%nnear(ii), ii),ii=0,self%nvar)
      write(self%ifile, '(*(:2x,F0.10))') (ctx%weight(1:ctx%nnear(ii), ii),ii=0,self%nvar)
    end associate
  end subroutine write_weight

  subroutine read_weight(self, ctx)
    class(t_kriging)      :: self
    class(t_kriging_ctx)  :: ctx
    ! local
    integer               :: ii
    associate(&
    ib=>ctx%iblock, &
    order=>self%block%order, &
    var=>self%block%variance(ctx%iblock))
      read(self%ifile, *) order(ib), var, ctx%nnear(0:self%nvar)
      read(self%ifile, *) (ctx%inear(1:ctx%nnear(ii), ii),ii=0,self%nvar)
      read(self%ifile, *) (ctx%weight(1:ctx%nnear(ii), ii),ii=0,self%nvar)
    end associate
  end subroutine read_weight


  subroutine finalize(self)
    class(t_kriging)      :: self
    deallocate(self%obs)
    deallocate(self%grid)
    deallocate(self%block)
    deallocate(self%vgm)
  end subroutine finalize


end module kriging

