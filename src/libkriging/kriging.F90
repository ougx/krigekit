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

#include "cov_dispatch.fh"
module kriging
  use iso_fortran_env, only: input_unit, error_unit, output_unit
  use iso_c_binding
  use common
  use kriging_err
  use utils, only: set_seq, yesno
  use progress_bar, only: progress
  use rotation
  use variogram
  use kdtree2_module
  use kriging_base
  implicit none

!============================================================================
! t_kriging — main kriging object
! t_kriging_ctx is now a unified concrete type defined in kriging_base.
!============================================================================
  type, extends(t_kriging_base) :: t_kriging
    !-- Spatial-specific state (all common options/dims are on t_kriging_base)
    type(vgm_struct), pointer :: vgm(:,:,:) => null()
  contains
    procedure :: init
    procedure :: post_grid_setup          ! set up varying variogram after grid setup
    procedure :: set_grid
    procedure :: set_vgm
    procedure :: reset_vgm
    procedure :: set_search
    procedure :: prepare
    procedure :: validate_vgm
    procedure :: search_neighbors
    procedure :: assemble_lhs
    procedure :: assemble_rhs
    ! --- implementations of t_kriging_base deferred procedures ---
    procedure :: calc_variance
    procedure :: tostr_vgm                ! satisfies base deferred (variogram section)
    procedure :: finalize
  end type


contains

  !============================================================================
  ! init -- t_kriging implementation of t_kriging_base%init.
  ! Called from initialize_base after all shared calculations and validation.
  ! Responsibility: allocate gradient and variogram data structures.
  !============================================================================
  subroutine init(self)
    class(t_kriging), intent(inout) :: self

    if (associated(self%grad)) deallocate(self%grad)
    if (associated(self%vgm))  deallocate(self%vgm)

    allocate(self%grad(self%nvar))
    if (.not. self%varying_vgm) then
      allocate(self%vgm(1:self%nvar, 1:self%nvar, 1))
      self%vgm%ndim = self%ndim
    end if
  end subroutine init


  subroutine post_grid_setup(self)
    class(t_kriging), intent(inout) :: self

    if (self%varying_vgm) then
      if (associated(self%vgm)) deallocate(self%vgm)
      allocate(self%vgm(1:self%nvar, 1:self%nvar, self%block%n))
    else if (.not. associated(self%vgm)) then
      allocate(self%vgm(1:self%nvar, 1:self%nvar, 1))
    end if
    self%vgm%ndim = self%ndim
  end subroutine post_grid_setup


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

    integer :: block_type_local
    character(len=*), parameter :: subname = "t_kriging%set_grid"

    if (self%cross_validation) then
      call self%set_grid_cv()
      return
    end if

    if (.not. present(coord)) then
      call kriging_error(subname, 'coord needs to be provided.')
      return
    end if

    block_type_local = 0
    if (present(block_type)) block_type_local = block_type

    select case (block_type_local)
    case (0)
      call self%set_grid_point(coord, rangescale, localnugget)
    case (-4)
      if (.not. present(blocksize)) then
        call kriging_error(subname, 'blocksize needs to be provided when block_type=-4.')
        return
      end if
      call self%set_grid_gq(coord, blocksize, rangescale, localnugget)
    case default
      if (.not. present(nblockpnt)) then
        call kriging_error(subname, 'nblockpnt needs to be provided when block_type>0.')
        return
      end if
      call self%set_grid_user_block(coord, nblockpnt, pointweight, rangescale, localnugget, block_type_local)
    end select
  end subroutine set_grid


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
    if (.not. kriging_check_pair_index(subname, ivar, jvar, 1, self%nvar)) return
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
    self%pf%valid = .false.   ! variogram changed → persistent factor stale
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
    character(len=*), parameter :: subname = "t_kriging%reset_vgm"
    if (.not. associated(self%vgm)) return
    if (.not. kriging_check_pair_index(subname, ivar, jvar, 1, self%nvar)) return
    mb = merge(self%block%n, 1, self%varying_vgm)
    do ib_ = 1, mb
      call self%vgm(jvar, ivar, ib_)%reset_model()
      if (ivar /= jvar) call self%vgm(ivar, jvar, ib_)%reset_model()
    end do
  end subroutine reset_vgm


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
  subroutine set_search(self, ivar, anis1, anis2, azimuth, dip, plunge, sector_search)
    use rotation,       only: calc_rotmat, sub_rotate
    use kdtree2_module, only: kdtree2_create
    class(t_kriging)   :: self
    integer, intent(in) :: ivar
    real,    intent(in) :: anis1, anis2, azimuth, dip, plunge
    logical, intent(in), optional :: sector_search
    character(len=*), parameter :: subname = "t_kriging%set_search"

    real, allocatable :: rcoord(:,:)   ! rotated coordinates for anisotropic tree
    if (.not. associated(self%obs)) then
      call kriging_error(subname, 'Call initialize() before set_search.')
      return
    end if
    if (.not. kriging_check_index(subname, 'ivar', ivar, 1, self%nvar)) return
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

      if (present(sector_search)) obs%sector_search = sector_search

      !-- Activate anisotropic search only when there is meaningful anisotropy
      anisotropic_search = (abs(anis1 - 1.0) > EPSLON .or. abs(anis2 - 1.0) > EPSLON) &
                           .and. self%anisotropic_search

      !-- Determine effective nmax, accounting for SGSIM's extended obs array.
      !   For all variables in SGSIM/joint co-sim, the tree is built on the
      !   extended coord array (obs + block centres); nmax spans both.
      if (self%nsim > 0) then
        call kriging_normalize_nmax(obs%nmax, obs%n + self%block%n)
        need_search = obs%n + self%block%n > obs%nmax
      else
        call kriging_normalize_nmax(obs%nmax, obs%n)
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
          obs%tree => kdtree2_create(rcoord, sort = obs%sector_search, rearrange = .true.)
          if (kriging_failed()) return
        else
          obs%tree => kdtree2_create(obs%coord, sort = obs%sector_search, rearrange = .true.)
          if (kriging_failed()) return
        end if
      end if
    end associate
    self%obs(ivar)%set_search = .true.
  end subroutine set_search


  !============================================================================
  ! prepare
  !
  ! Pre-solve validation and bookkeeping called at the start of solve().
  ! Sets vgm%ndim on every entry, validates that every block/variable pair has
  ! at least one variogram structure (unless use_old_weight), then delegates
  ! sizing and weight-file bookkeeping to prepare_common.
  !============================================================================
  subroutine prepare(self)
    class(t_kriging), intent(inout) :: self
    character(len=*), parameter :: subname = "t_kriging%prepare"
    integer :: iv, jv, ib
    if (.not. self%use_old_weight) then
      call self%validate_vgm()
      if (kriging_failed()) return
    end if
    self%vgm%ndim     = self%ndim
    !!!! table is skipped for varying-vgm because storing one table per grid block
    !      is too expensive for large number of blocks
    !-- Build piecewise lookup tables now that ndim and all structures are final.
    !   build_table tries Level B first (1 aniso_h + 1 lookup, all nstruct),
    !   falling back to Level A (one table per component) when needed.
    if (.not. self%varying_vgm) then
      do ib = 1, merge(self%block%n, 1, self%varying_vgm)
        do iv = 1, self%nvar
          do jv = iv, self%nvar
            call self%vgm(iv, jv, ib)%build_table( &
              h_bounds=[0.0, 0.1, 0.5, 3.5], dh=[1e-5, 1e-4, 1e-3])
            if (kriging_failed()) return
          end do
        end do
      end do
    end if
    call self%prepare_common(subname)
  end subroutine prepare


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


  ! initialize_kriging_ctx, assign_weight, and fcache_* routines have moved to
  ! kriging_base (initialize_ctx, assign_weight_ctx, fcache_matches, fcache_save_key)
  ! and kriging_base (fcache_alloc, copy_to, copy_all on t_factor_cache).


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
    class(t_kriging),    intent(inout) :: self
    type(t_kriging_ctx), intent(inout) :: ctx
    integer,             intent(in)    :: ivar

    integer                     :: i, ig_sim
    real                        :: newloc(self%ndim, 1)
    logical, allocatable        :: is_obs(:)
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
      dist    => ctx%sqdist(:, ivar), &
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
          nnearb => ctx%nnear(ig_sim), &
          distb  => ctx%sqdist(:, self%nvar + ivar))

          if (self%obs(ivar)%sector_search) then
            !-- Sector search for SGSIM: same quadrant/octant logic as standard kriging
            !   but applied to the combined obs+prior-block pool.
            !   obsloc contains both obs (1:nobs) and block centres (nobs+1:end),
            !   so sector assignment works identically for both; we just split
            !   accepted candidates by idx <= nobs afterward.
            nnear  = 0
            nnearb = 0
            if (nobs + iblock - 1 > 0) then
              block
                integer :: ncand
                ncand = min(nobs + iblock - 1, (2**ndim) * nmax * 4)
                call kdtree2_n_nearest_maxidx(self%obs(ivar)%tree, newloc(:,1), ncand, ctx%results, nobs+iblock-1)
                if (kriging_failed()) return
                call apply_sector_filter( &
                  ctx%results, ncand, ndim, nmax, nobs, &
                  newloc(:,1), obsloc, rotmat, self%obs(ivar)%anisotropic_search, &
                  maxdist, 0, &
                  inear, dist, nnear, inearb, distb, nnearb)
              end block
            end if

          else if (nmax < nobs + iblock - 1) then
            !-- k-d tree query with max_idx filter: only returns entries < nobs+iblock
            call kdtree2_n_nearest_maxidx(self%obs(ivar)%tree, newloc(:,1), nmax, ctx%results, nobs+iblock-1)
            if (kriging_failed()) return
            allocate(is_obs, source = ctx%results(1:nmax)%idx <= nobs)
            nnear              = count(is_obs)
            nnearb             = nmax - nnear
            inear (1:nnear)    = pack(ctx%results(1:nmax)%idx, is_obs)
            inearb(1:nnearb)   = pack(ctx%results(1:nmax)%idx, .not. is_obs) - nobs  ! shift to 1-based block index
            dist  (1:nnear)    = pack(ctx%results(1:nmax)%dis, is_obs)
            distb (1:nnearb)   = pack(ctx%results(1:nmax)%dis, .not. is_obs)
          else
            !-- All obs + prior blocks fit within nmax: compute distances directly
            nnear  = nobs
            nnearb = iblock - 1
            call set_seq(inear(1:nnear), nnear)
            if (nnearb > 0) call set_seq(inearb(1:nnearb), nnearb)
            dist (1:nnear)  = rotated_dists(rotmat, ndim, newloc(:,1), obsloc(:, 1:nnear))
            distb(1:nnearb) = rotated_dists(rotmat, ndim, newloc(:,1), self%block%coord(:, 1:nnearb))
          end if
          call filter_by_maxlag(ctx%inear(:, ig_sim), distb, ctx%nnear(ig_sim), maxdist)
        end associate  ! inearb, nnearb, distb

      !------------------------------------------------------------------------
      ! Standard kriging / cokriging neighbour search
      !------------------------------------------------------------------------
      else
        if (self%obs(ivar)%sector_search .and. self%obs(ivar)%need_search) then
          block
            integer :: ncand, cv_skip, dummy_nnearb
            integer :: dummy_inearb(1)
            real    :: dummy_distb(1)
            ncand = min(nobs, (2**ndim) * nmax * 4)
            call kdtree2_n_nearest(self%obs(ivar)%tree, newloc(:,1), ncand, ctx%results)
            if (kriging_failed()) return
            nnear   = 0
            cv_skip = merge(iblock, 0, self%cross_validation)
            call apply_sector_filter( &
              ctx%results, ncand, ndim, nmax, huge(0), &
              newloc(:,1), obsloc, rotmat, self%obs(ivar)%anisotropic_search, &
              maxdist, cv_skip, &
              inear, dist, nnear, dummy_inearb, dummy_distb, dummy_nnearb)
          end block
        else if (nmax < nobs) then
          call kdtree2_n_nearest(self%obs(ivar)%tree, newloc(:,1), nmax, ctx%results)
          if (kriging_failed()) return
          nnear          = nmax
          inear(1:nnear) = ctx%results(1:nmax)%idx
          dist (1:nnear) = ctx%results(1:nmax)%dis
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
      call filter_by_maxlag(inear, dist, nnear, maxdist)
    end associate
#ifdef DEBUG
    print *, subname, " Finished.", ivar, ctx%iblock
#endif
  end subroutine search_neighbors


  !============================================================================
  ! assemble_lhs
  !
  ! Fill the left-hand-side covariance matrix matA for the current block.
  ! Called by assemble_linear_system on a factorization cache miss.
  !
  ! Matrix layout: nvar=2, ndrift=1 external drift, naug=2 unbiasedness, ng grad pairs
  ! npp = n1 + n2 + ng,   p = ndrift + naug = 3
  !
  !           ←────── npp ──────→  ←──── p = ndrift+naug ────→
  !            n1     n2     ng      d₁      u₁      u₂
  !         ┌───────────────────────────────────────────┐
  !   n1    │  C₁₁    C₁₂   Cg₁ᵀ  │  fo₁₁   1ᵀ     0ᵀ   │  [ w₁ ]   [ c₀₁ ]
  !   n2    │  C₂₁    C₂₂   Cg₂ᵀ  │  fo₁₂   0ᵀ     1ᵀ   │  [ w₂ ] = [ c₀₂ ]
  !   ng    │  Cg₁    Cg₂   Cgg   │  fg₁    0      0    │  [ θg ]   [ cg₀ ]
  !         │─────────────────────│─────────────────────│──────────────│
  !   d₁    │  fo₁₁ᵀ fo₁₂ᵀ  fg₁ᵀ  │   0     0      0    │  [ β₁ ]   [ f₀₁ ]
  !   u₁    │   1ᵀ    0ᵀ    0ᵀ    │   0     0      0    │  [ μ₁ ]   [  1  ]
  !   u₂    │   0ᵀ    1ᵀ    0ᵀ    │   0     0      0    │  [ μ₂ ]   [  0  ]
  !         └───────────────────────────────────────────┘
  !
  ! Notation
  ! --------
  ! C₁₁, C₁₂   : obs-obs covariance blocks between variables
  ! Cg₁, Cg₂   : obs-grad cross blocks — C(obs_i, xs1_t) - C(obs_i, xs2_t)
  ! Cgg        : grad-grad block — C(xs1_t,xs1_s) - C(xs1_t,xs2_s)
  !                                               - C(xs2_t,xs1_s) + C(xs2_t,xs2_s)
  !               diagonal: 2C(0) - 2C(xs1-xs2) + grad_variance
  ! fo₁₁, fo₁₂ : external drift values at obs var1 / var2 locations
  ! fg₁        : external drift differences d(xs1_t) - d(xs2_t) at grad pair t
  !               (zero for constant drift, only nonzero when ndrift > 0)
  ! u₁, u₂     : unbiasedness rows — indicator 1 for own variable, 0 otherwise;
  !               all-zero for grad pairs (constant f=1 → difference = 0)
  ! β₁         : external drift coefficient(s); μ₁,μ₂ Lagrange multipliers
  ! θg         : grad pair kriging weights (solved alongside obs weights w₁,w₂)
  ! f₀₁        : external drift value at estimation point (for estimating var1)
  !
  ! Assembly strategy
  ! -----------------
  ! Obs groups (kvar <= ngroups_base): filled by rowloop/columnloop (upper triangle).
  ! Grad groups (kvar > ngroups_base): filled separately — obs-grad cross (lower
  !   triangle), grad-grad (both triangles), grad-drift (both triangles).
  ! Full mirror pass at the end copies lower → upper for the obs rows.
  ! Drift and unbiasedness columns are appended explicitly for both obs and grad.
  !============================================================================
  subroutine assemble_lhs(self, ctx)
    class(t_kriging)   , intent(inout) :: self
    type(t_kriging_ctx), intent(inout) :: ctx

    integer                :: kvar, lvar, i, j, jstart
    integer                :: ivgm, ivar, jvar, givar
    real                   :: lag(self%ndim), ln, rs, cov_g, c11, c12, c21, c22
    class(t_data), pointer :: obs1, obs2

    associate( &
      matA      => ctx%matA, &
      inear     => ctx%inear, &
      nnear     => ctx%nnear, &
      istart    => ctx%istart, &
      npp       => ctx%npp, &
      matsize   => ctx%matsize, &
      ndrift    => self%ndrift, &
      naug      => self%naug, &
      ndim      => self%ndim)

      matA(1:matsize, 1:matsize) = 0.0

      ivgm = merge(ctx%iblock, 1, self%varying_vgm)
      rs   = self%block%rangescale(ctx%iblock)
      ln   = self%block%localnugget(ctx%iblock)

      rowloop: do kvar = 1, self%ngroups_base
        if (nnear(kvar) == 0) cycle

        !-- Row-group data source and variogram index (1:nvar=obs, nvar+1:ngroups_base=sim)
        if (kvar > self%nvar) then
          ivar = kvar - self%nvar
          obs1 => self%block
        else
          ivar = kvar
          obs1 => self%obs(kvar)
        end if

        !-- Fill upper triangle in matrix position: lvar groups whose istart >= kvar's istart.
        !   The loop iterates obs+sim groups; the istart(lvar) < istart(kvar) guard skips lower.
        !   The lower triangle is filled by the mirror pass at the end.
        columnloop: do lvar = 1, self%ngroups_base
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
                lag = (obs1%coord(:, inear(i,kvar)) - obs2%coord(:, inear(j,lvar))) / rs
                matA(istart(lvar)+j, istart(kvar)+i) = COV(vgm, lag)
              end do
            end do
          end associate
        end do columnloop

        !-- Drift/unbiasedness rows at npp+1:matsize for obs+sim columns.
        !   obs%drift(:,1,:): rows 1:ndrift = external drift; ndrift+1:end = unbiasedness indicators.
        if (ndrift + naug > 0) then
          matA(npp+1:matsize, istart(kvar)+1:istart(kvar)+nnear(kvar)) = &
            obs1%drift(:, 1, inear(1:nnear(kvar), kvar))
        end if

        !-- Gradient pair augmentation: one group per variable, appended after obs+sim columns.
        !   Only attempted when prepare() expanded ngroups > ngroups_base.
        gradidentloop: do lvar = self%ngroups_base+1, self%ngroups
          if (nnear(lvar) == 0) cycle
          givar = lvar - self%ngroups_base

          !-- grad-obs cross block: upper triangle (rows kgrad, cols obs+sim).
          associate(vgm => self%vgm(givar, ivar, ivgm))
            do i = 1, nnear(kvar)
              do j = 1, nnear(lvar)
                lag = (self%grad(givar)%coord(:,j)  - obs1%coord(:,inear(i,kvar))) / rs
                cov_g = COV(vgm, lag)
                lag = (self%grad(givar)%coord2(:,j) - obs1%coord(:,inear(i,kvar))) / rs
                matA(istart(lvar)+j, istart(kvar)+i) = cov_g - COV(vgm, lag)
              end do
            end do
          end associate
        end do gradidentloop

      end do rowloop

      !-- grad-grad covariance (diagonal and both off-diagonal triangles explicit).
      do ivar = 1, self%nvar
        kvar = self%ngroups_base + ivar
        if (self%grad(ivar)%n == 0) cycle
        associate(vgm => self%vgm(ivar, ivar, ivgm), grad=>self%grad(ivar), is=>istart(kvar))
          do i = 1, grad%n
            !-- Diagonal: 2*C(0) - 2*C(xs1-xs2) + grad variance
            lag = (grad%coord(:,i) - grad%coord2(:,i)) / rs
            matA(is+i, is+i) = 2.0 * vgm%cov0 - 2.0 * COV(vgm, lag) + grad%variance(1,1,i)
            do j = i+1, grad%n
              lag = (grad%coord (:,i) - grad%coord (:,j)) / rs; c11 = COV(vgm, lag)
              lag = (grad%coord (:,i) - grad%coord2(:,j)) / rs; c12 = COV(vgm, lag)
              lag = (grad%coord2(:,i) - grad%coord (:,j)) / rs; c21 = COV(vgm, lag)
              lag = (grad%coord2(:,i) - grad%coord2(:,j)) / rs; c22 = COV(vgm, lag)
              matA(is+j, is+i) = c11 - c12 - c21 + c22
            end do
          end do
          !-- grad-drift columns at npp+1:matsize (both triangles).
          if (ndrift + naug > 0) then
            do i = 1, grad%n
              matA(npp+1:matsize, is+i) = self%grad(ivar)%drift(:, 1, i)
            end do
          end if
        end associate
      end do

      !-- Mirror lower triangle to upper: copies matA(col, row) → matA(row, col) for col > row.
      !   Covers: obs-obs and grad-grad upper triangle, obs-grad cross, grad-drift.
      call kriging_mirror_lower_to_upper(matA, npp, matsize)

    end associate
  end subroutine assemble_lhs


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
    class(t_kriging)   , intent(inout) :: self
    type(t_kriging_ctx), intent(inout) :: ctx
    ! local
    integer                :: ivar, givar, kvar, kgrad, i, k, nn, jvar, ivgm, igrad
    real                   :: lag(self%ndim), tmp, tmp2, rs
    class(t_data), pointer :: obs

    associate( &
      iblock    => ctx%iblock, &
      rhsB      => ctx%rhsB, &
      nnear     => ctx%nnear, &
      inear     => ctx%inear, &
      istart    => ctx%istart, &
      npp       => ctx%npp, &
      matsize   => ctx%matsize, &
      ndrift    => self%ndrift, &
      naug      => self%naug, &
      ndim      => self%ndim)

      rs   = self%block%rangescale(ctx%iblock)


      ivgm = merge(iblock, 1, self%varying_vgm)

      do ivar = 1, self%nvar                  ! target variable; set the columns
        do kvar = 1, self%ngroups_base        ! neighbour group; set the rows
          nn = nnear(kvar)
          if (nn == 0) cycle
          !-- Data source and variogram index (1:nvar=obs, nvar+1:ngroups_base=sim)
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
                lag = (obs%coord(:, inear(i, kvar)) - self%grid%coord(:, k)) / rs
                tmp = tmp + COV(vgm, lag) * self%grid%weight(k)
              end do
              rhsB(ivar, istart(kvar)+i) = tmp
            end do
          end associate
        end do

        !-- Gradient RHS: C(xs1,x0) - C(xs2,x0) for each grad group ivar.
        !   vgm(givar, ivar): cross-covariance between gradient variable and estimation target.
        do givar = 1, self%nvar
          if (self%grad(givar)%n == 0) cycle
          kgrad = self%ngroups_base + givar
          associate(vgm => self%vgm(givar, ivar, ivgm))
            do igrad = 1, self%grad(givar)%n
              tmp  = 0.0
              tmp2 = 0.0
              do k = self%block%iblockpnt(iblock), self%block%iblockpnt(iblock)+self%block%nblockpnt(iblock)-1
                lag = (self%grad(givar)%coord(:,igrad) - self%grid%coord(:,k)) / rs
                tmp  = tmp  + COV(vgm, lag) * self%grid%weight(k)
                lag = (self%grad(givar)%coord2(:,igrad) - self%grid%coord(:,k)) / rs
                tmp2 = tmp2 + COV(vgm, lag) * self%grid%weight(k)
              end do
              rhsB(ivar, istart(kgrad)+igrad) = tmp - tmp2
            end do
          end associate
        end do

        !-- Augmented RHS (external drift + unbiasedness) at npp+1:matsize.
        !   block%drift(:, ivar, :) carries both rows, varying correctly per target variable.
        if (ndrift + naug > 0) &
          rhsB(ivar, npp+1:matsize) = self%block%drift(:, ivar, iblock)
      end do
    end associate
  end subroutine assemble_rhs





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
    class(t_kriging)   , intent(inout) :: self
    type(t_kriging_ctx), intent(inout) :: ctx

    integer :: i, j, k1, ivgm, ivar, jvar
    real    :: lag(self%ndim), base_cov

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
              !-- Point kriging: C_ij(x0, x0) = covariance at zero lag.
              base_cov = vgm%cov0
            else
              !-- Block kriging: weighted average of C_ij(s_p, s_q) over all integration node pairs.
              !   Upper triangle with x2 avoids double-loop; diagonal uses cov0 (p == q, lag = 0).
              base_cov = 0.0
              k1 = self%block%iblockpnt(iblock) - 1
              do i = 1, nblockpnt
                base_cov = base_cov + vgm%cov0 * weight(k1+i) * weight(k1+i)
                do j = i+1, nblockpnt
                  lag = coord(:, k1+i) - coord(:, k1+j)
                  base_cov = base_cov + COV(vgm, lag) * &
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
  ! tostr_vgm — variogram section of the info string (satisfies base deferred).
  !
  ! Called by to_str_base() on t_kriging_base; appended after the common fields.
  !============================================================================
  function tostr_vgm(self) result(res_str)
    class(t_kriging), intent(in) :: self
    character(len=:), allocatable :: res_str
    character(len=256) :: buf
    integer :: ivar, jvar
    character(len=1), parameter :: NL = new_line('A')
    res_str = ''
    write(buf, "(A)") " Variogram Models"
    res_str = res_str // trim(buf) // NL

    do ivar = 1, self%nvar
        do jvar = 1, self%nvar
            if (ivar == jvar) then
                write(buf, "(A,I0,A,I0,A)") " Model for Variable ", ivar, self%vgm(jvar, ivar, 1)%tostr()
            else
                write(buf, "(A,I0,A,I0,A)") " Model between Variable ", ivar, " and ", jvar, self%vgm(jvar, ivar, 1)%tostr()
            end if
            res_str = res_str // trim(buf) // NL
        end do
    end do
  end function tostr_vgm




  !============================================================================
  ! finalize
  !
  ! Release all allocated memory.  Call after all results have been read from
  ! block%value and block%variance.
  !============================================================================
  subroutine finalize(self)
    class(t_kriging), intent(inout) :: self
    call self%finalize_common()      ! frees krige_info (on base) and all shared data
    if (associated(self%vgm)) deallocate(self%vgm)
  end subroutine finalize


  !=============================================================================
  ! apply_sector_filter — private helper for sector (quadrant/octant) search
  !
  ! Parameters
  !   results     : kdtree2 query results, already sorted by ascending distance
  !   ncand       : number of valid entries in results
  !   ndim        : 2 or 3
  !   nmax        : maximum neighbours per sector
  !   nobs_split  : threshold index; idx <= nobs_split → obs (inear), else → block (inearb).
  !                 Pass huge(0) when there are no simulated blocks in the tree.
  !   newloc      : target location, already rotated if aniso (length ndim)
  !   obsloc      : combined obs + block coordinate array (ndim x *)
  !   rotmat      : rotation matrix handle (for aniso)
  !   aniso       : apply rotation before computing sector offsets
  !   maxdist     : squared-distance cutoff; candidates beyond this are skipped
  !   skip_idx    : index to skip unconditionally (cross-validation); 0 = none
  !   inear, dist, nnear   : accepted observation neighbours (output)
  !   inearb, distb, nnearb: accepted prior-block neighbours (output; unused for nsim==0)
  !=============================================================================
  subroutine apply_sector_filter(results, ncand, ndim, nmax, nobs_split, &
                                  newloc, obsloc, rotmat, aniso, maxdist, skip_idx, &
                                  inear, dist, nnear, &
                                  inearb, distb, nnearb)
    type(kdtree2_result), intent(in)  :: results(*)
    integer,              intent(in)  :: ncand, ndim, nmax, nobs_split, skip_idx
    real,                 intent(in)  :: newloc(ndim), maxdist
    real,                 intent(in)  :: obsloc(ndim, *)
    real,                 intent(in)  :: rotmat(3, 3)
    logical,              intent(in)  :: aniso
    integer,              intent(out) :: inear(*), nnear
    real,                 intent(out) :: dist(*)
    integer,              intent(out) :: inearb(*), nnearb
    real,                 intent(out) :: distb(*)

    integer :: i, idx, sector, ix, iy, iz
    integer :: sector_count(8)
    real    :: pt_rotated(ndim, 1), dx, dy, dz

    nnear  = 0
    nnearb = 0
    sector_count = 0

    do i = 1, ncand
      idx = results(i)%idx
      if (idx == skip_idx) cycle
      if (results(i)%dis > maxdist) cycle

      if (aniso) then
        call sub_rotate(rotmat, ndim, 1, obsloc(:, idx:idx), pt_rotated)
      else
        pt_rotated(:, 1) = obsloc(:, idx)
      end if

      dx = pt_rotated(1, 1) - newloc(1)
      dy = pt_rotated(2, 1) - newloc(2)
      dz = 0.0
      if (ndim == 3) dz = pt_rotated(3, 1) - newloc(3)

      ix = merge(1, 0, dx < 0.0)
      iy = merge(1, 0, dy < 0.0)
      if (ndim == 3) then
        iz     = merge(1, 0, dz < 0.0)
        sector = 1 + ix + 2*iy + 4*iz
      else
        sector = 1 + ix + 2*iy
      end if

      if (sector_count(sector) < nmax) then
        sector_count(sector) = sector_count(sector) + 1
        if (idx <= nobs_split) then
          nnear         = nnear + 1
          inear(nnear)  = idx
          dist(nnear)   = results(i)%dis
        else
          nnearb         = nnearb + 1
          inearb(nnearb) = idx - nobs_split   ! shift to 1-based block index
          distb(nnearb)  = results(i)%dis
        end if
      end if
    end do
  end subroutine apply_sector_filter


  ! get_persistent_factor_info and get_persistent_factor_matrices have moved
  ! to t_kriging_base (kriging_base.F90) — they only access self%pf.


end module kriging
