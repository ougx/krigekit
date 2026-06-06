!==============================================================================
! Module: kriging_st
!
! Space-time kriging and co-kriging for 1D/2D/3D spatial data plus time.
!
! Design
! ------
! t_kriging_st handles spatial+temporal observations.  Time is stored as
! coord(nlag,:) in all obs/grid/block types.  ndim=spatial dimension (1..3),
! nlag=ndim+1 (spatial + 1 for time).  Coord arrays have nlag rows.
!
! Neighbor search uses an nlag-dimensional KD-tree built in set_search.  Time is
! mapped to a search coordinate by linear scaling: t_kd = t * at.  This keeps
! the KD-tree metric consistent with the sum-metric ST distance
! h_ST = sqrt(h_S^2 + (at*dt)^2), because |t1*at - t2*at| = at*|t1-t2|.
! The raw native time is always stored in coord(nlag,:); covariance evaluation
! extracts coord(1:ndim) for the spatial lag and coord(nlag) for dt.
!
! Key differences from t_kriging:
!   - vgm(:,:,:) is type(vgm_struct_st) — the ST covariance model.
!   - assemble_lhs/rhs compute lag from coord(1:ndim) and dt from coord(nlag).
!   - set_search builds an nlag-dimensional KD-tree with search-specific time transform.
!   - SGSIM extends obs(:)%coord with all nlag rows
!     to include prediction block coords for SGSIM conditioning.
!==============================================================================
module kriging_st

  use iso_fortran_env, only: output_unit
  use common,           only: EPSLON
  use kriging_err,      only: kriging_error, kriging_failed
  use utils,            only: set_seq

  use rotation,         only: calc_rotmat, sub_rotate, rotated_dists
  use variogram_st,     only: vgm_struct_st, ST_MODEL_SUM_METRIC, ST_MODEL_PRODUCT_SUM, &
                               ST_TRANSFORM_LINEAR, ST_TRANSFORM_BOUNDED, ST_TRANSFORM_POWER
  use vgm_func,         only: VGM_NUG, VGM_HOL, VGM_LIN, vtype_from_str
  use kdtree2_module
  use kriging_base
  implicit none



  !=============================================================================
  ! t_kriging_st — main space-time kriging object
  ! t_kriging_ctx is now a unified concrete type defined in kriging_base.
  !=============================================================================
  type, extends(t_kriging_base) :: t_kriging_st
    !-- ST-specific options (all common options/dims are on t_kriging_base)

    !-- ST model global parameters (set by set_st_model; copied into every vgm entry)
    integer :: st_model         = ST_MODEL_SUM_METRIC
    integer :: st_time_vtype_id = ST_TRANSFORM_LINEAR
    real    :: st_time_nugget   = 0.0
    real    :: st_time_sill     = 1.0
    real    :: st_at            = 1.0

    !-- Data
    type(vgm_struct_st), pointer :: vgm(:,:,:) => null()

  contains
    procedure :: init_defaults
    procedure :: init
    procedure :: set_st_model
    procedure :: set_vgm
    procedure :: set_vgm_temporal
    procedure :: set_vgm_joint_sills
    procedure :: reset_vgm
    procedure :: set_search
    procedure :: prepare
    procedure :: search_neighbors
    procedure :: assemble_lhs
    procedure :: assemble_rhs
    ! --- implementations of t_kriging_base deferred procedures ---
    procedure :: calc_variance
    procedure :: tostr_vgm                ! satisfies base deferred (variogram section)
    procedure :: finalize
  end type t_kriging_st

contains

  !=============================================================================
  ! time_search_coord
  !
  ! Convert native time to the one-dimensional KD-tree search coordinate.
  ! Linear scaling t * at maps the time axis to km-equivalent units so that
  ! the L2 distance in the (x,y,z, t*at) search space equals h_ST for the
  ! sum-metric model.  Monotone and unbounded — no saturation artifacts.
  !=============================================================================
  pure elemental function time_search_coord(at, time) result(ht)
    real, intent(in) :: at, time
    real :: ht

    ht = time * at
  end function time_search_coord

  ! ctx_initialize and ctx_assign_weight replaced by the unified
  ! initialize_ctx and assign_weight_ctx from kriging_base.


  !============================================================================
  ! init_defaults — override t_kriging_base defaults for ST kriging.
  ! Called by initialize_base before user overrides are applied.
  !============================================================================
  subroutine init_defaults(self)
    class(t_kriging_st), intent(inout) :: self
    self%neglect_error = .true.   ! ST default: skip singular blocks gracefully
    self%std_ck        = .true.   ! ST always uses standard cokriging layout
  end subroutine init_defaults


  !============================================================================
  ! init -- t_kriging_st implementation of t_kriging_base%init.
  ! Called from initialize_base after all shared calculations and validation.
  ! Responsibility: enforce ST-specific constraints and allocate data structures.
  !============================================================================
  subroutine init(self)
    class(t_kriging_st), intent(inout) :: self
    if (associated(self%grad)) deallocate(self%grad)
    if (associated(self%vgm)) deallocate(self%vgm)
    self%varying_vgm = .false.    ! ST does not use varying anisotropy
    allocate(self%grad(self%nvar))
    allocate(self%vgm(1:self%nvar, 1:self%nvar, 1))
    !-- Store spatial dimension on each vgm entry for use in cov_lag wrapper.
    self%vgm%ndim = self%ndim
    !-- nlag = spatial ndim + 1 (time occupies the last coord row).
    !   ndim stays as the pure spatial dimension for KD-tree search and rotation.
    self%nlag = min(4, self%ndim + 1)
  end subroutine init


  !=============================================================================
  ! set_st_model — set global ST model params and propagate to all vgm entries
  !=============================================================================
  subroutine set_st_model(self, model, transform, at, k_ps, time_nugget, time_sill)
    class(t_kriging_st), intent(inout) :: self
    integer, intent(in)                :: model, transform
    real,    intent(in)                :: at
    real,    intent(in), optional      :: k_ps, time_nugget, time_sill
    integer :: i, j
    real :: nugget_, sill_

    if (.not. associated(self%vgm)) then
      call kriging_error('set_st_model', 'call initialize() before set_st_model()')
      return
    end if
    if (transform < VGM_NUG .or. transform > VGM_LIN .or. transform == VGM_HOL) then
      call kriging_error('set_st_model', &
        'transform must be a monotone vgmfunc id: nug, sph, exp, gau, pow, bsq, cir, or lin.')
      return
    end if
    if (at <= EPSLON) then
      call kriging_error('set_st_model', 'at must be positive.')
      return
    end if

    nugget_ = self%st_time_nugget
    sill_   = self%st_time_sill
    if (present(time_nugget)) nugget_ = time_nugget
    if (present(time_sill))   sill_   = time_sill
    if (nugget_ < 0.0 .or. sill_ < 0.0) then
      call kriging_error('set_st_model', 'time_nugget and time_sill must be non-negative.')
      return
    end if

    self%st_model         = model
    self%st_time_vtype_id = transform
    self%st_time_nugget   = nugget_
    self%st_time_sill     = sill_
    self%st_at            = at

    do j = 1, self%nvar
      do i = 1, self%nvar
        self%vgm(i,j,1)%model         = model
        self%vgm(i,j,1)%time_vtype_id = transform
        self%vgm(i,j,1)%time_nugget   = nugget_
        self%vgm(i,j,1)%time_sill     = sill_
        self%vgm(i,j,1)%at            = at
        if (present(k_ps))  self%vgm(i,j,1)%k_ps  = k_ps
      end do
    end do
  end subroutine set_st_model


  !=============================================================================
  ! set_vgm
  !
  ! Add one nested SPATIAL structure to vgm(ivar,jvar)%cs.
  ! Call once per nested spatial structure; combine with set_vgm_temporal
  ! for the full joint ST model.  Only the upper triangle (jvar >= ivar)
  ! needs to be specified; the lower triangle is filled symmetrically.
  !
  ! Unlike the spatial kriging set_vgm, there is no per-block index:
  ! the ST variogram is stationary (no varying_vgm mode).
  !=============================================================================
  subroutine set_vgm(self, ivar, jvar, vtype, nugget, sill, a_major, a_minor1, a_minor2, azimuth, dip, plunge)
    class(t_kriging_st), intent(inout) :: self
    integer,             intent(in)    :: ivar, jvar
    character(*), optional, intent(in) :: vtype
    real,         optional, intent(in) :: nugget, sill, a_major, a_minor1, a_minor2
    real,         optional, intent(in) :: azimuth, dip, plunge
    ! local
    character(len=3) :: vtype_
    real             :: nugget_, sill_, a_major_, a_minor1_, a_minor2_, azimuth_, dip_, plunge_
    if (.not. associated(self%vgm)) then
      call kriging_error('set_vgm', 'call initialize() before set_vgm()')
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
    if (.not. kriging_check_pair_index('set_vgm', ivar, jvar, 1, self%nvar)) return
    if (jvar == ivar) then
      call self%vgm(ivar, jvar, 1)%add_spatial(trim(vtype_), nugget_, sill_, a_major_, a_minor1_, a_minor2_, azimuth_, dip_, plunge_)
      if (kriging_failed()) return
    else if (jvar > ivar) then
      call self%vgm(ivar, jvar, 1)%add_spatial(trim(vtype_), nugget_, sill_, a_major_, a_minor1_, a_minor2_, azimuth_, dip_, plunge_)
      if (kriging_failed()) return
      call self%vgm(jvar, ivar, 1)%add_spatial(trim(vtype_), nugget_, sill_, a_major_, a_minor1_, a_minor2_, azimuth_, dip_, plunge_)
      if (kriging_failed()) return
    else
      call kriging_error('set_vgm', 'jvar must be >= ivar')
      return
    end if
  end subroutine set_vgm


  !=============================================================================
  ! set_vgm_temporal — add one nested TEMPORAL structure to vgm(ivar,jvar)%ct
  !   spec: "vtype nugget sill at_k"    (simplified 4-param format)
  !=============================================================================
  subroutine set_vgm_temporal(self, ivar, jvar, vtype,nugget,sill,at_k)
    class(t_kriging_st), intent(inout) :: self
    integer,             intent(in)    :: ivar, jvar
    character(*), optional, intent(in) :: vtype
    real,         optional, intent(in) :: nugget, sill, at_k
    if (.not. associated(self%vgm)) then
      call kriging_error('set_vgm_temporal', 'call initialize() before set_vgm_temporal()')
      return
    end if
    if (.not. kriging_check_pair_index('set_vgm_temporal', ivar, jvar, 1, self%nvar)) return
    if (jvar == ivar) then
      call self%vgm(ivar, jvar, 1)%add_temporal(vtype,nugget,sill,at_k)
      if (kriging_failed()) return
    else if (jvar > ivar) then
      call self%vgm(ivar, jvar, 1)%add_temporal(vtype,nugget,sill,at_k)
      if (kriging_failed()) return
      call self%vgm(jvar, ivar, 1)%add_temporal(vtype,nugget,sill,at_k)
      if (kriging_failed()) return
    else
      call kriging_error('set_vgm_temporal', 'jvar must be >= ivar')
      return
    end if
  end subroutine set_vgm_temporal


  !=============================================================================
  ! set_vgm_joint_sills — set joint sills for sum-metric vgm(ivar,jvar)
  !   sills: one per nested structure of cs (length must equal cs%nstruct)
  !=============================================================================
  subroutine set_vgm_joint_sills(self, ivar, jvar, sills, n)
    class(t_kriging_st), intent(inout) :: self
    integer,             intent(in)    :: ivar, jvar, n
    real,                intent(in)    :: sills(n)
    if (.not. associated(self%vgm)) then
      call kriging_error('set_vgm_joint_sills', 'call initialize() before set_vgm_joint_sills()')
      return
    end if
    if (.not. kriging_check_pair_index('set_vgm_joint_sills', ivar, jvar, 1, self%nvar)) return
    if (jvar == ivar) then
      call self%vgm(ivar, jvar, 1)%set_joint_sills(sills, n)
      if (kriging_failed()) return
    else if (jvar > ivar) then
      call self%vgm(ivar, jvar, 1)%set_joint_sills(sills, n)
      if (kriging_failed()) return
      call self%vgm(jvar, ivar, 1)%set_joint_sills(sills, n)
      if (kriging_failed()) return
    else
      call kriging_error('set_vgm_joint_sills', 'jvar must be >= ivar')
      return
    end if
  end subroutine set_vgm_joint_sills


  !=============================================================================
  ! reset_vgm
  !
  ! Clear all nested spatial, temporal, and joint ST structures for the
  ! (ivar, jvar) pair.  After this call, set_vgm / set_vgm_temporal /
  ! set_vgm_joint_sills may be used to build a fresh model for the pair.
  !=============================================================================
  subroutine reset_vgm(self, ivar, jvar)
    class(t_kriging_st), intent(inout) :: self
    integer,             intent(in)    :: ivar, jvar
    character(len=*), parameter :: subname = 't_kriging_st%reset_vgm'

    if (.not. associated(self%vgm)) return
    if (.not. kriging_check_pair_index(subname, ivar, jvar, 1, self%nvar)) return

    call self%vgm(jvar, ivar, 1)%reset_model()
    if (ivar /= jvar) call self%vgm(ivar, jvar, 1)%reset_model()
  end subroutine reset_vgm


  !=============================================================================
  ! set_search — build nlag-dimensional KD-tree for variable ivar
  !
  ! Spatial search anisotropy is independent of variogram anisotropy.
  ! The time axis is mapped to km-equivalent units via t_kd = t * time_at,
  ! which is consistent with the sum-metric distance h_ST = sqrt(h_S^2 + (at*dt)^2).
  ! maxdist on set_obs therefore acts as a radius in km-equivalent space.
  !
  ! If all observations fit within nmax, need_search=.false. and no tree is
  ! built; distances are computed directly in search_neighbors.
  !=============================================================================
  subroutine set_search(self, ivar, anis1, anis2, azimuth, dip, plunge, time_at, sector_search)
    class(t_kriging_st), intent(inout) :: self
    integer,             intent(in)    :: ivar
    real,    intent(in), optional      :: anis1, anis2, azimuth, dip, plunge
    real,    intent(in), optional      :: time_at
    logical, intent(in), optional      :: sector_search

    real    :: a1, a2, az, dp, pl, ta
    real, allocatable :: tcoord(:,:)

    if (.not. associated(self%obs)) then
      call kriging_error('set_search', 'call initialize() before set_search()')
      return
    end if
    if (.not. kriging_check_index('set_search', 'ivar', ivar, 1, self%nvar)) return
    if (self%obs(ivar)%n == 0) then
      call kriging_error('set_search', 'set_obs() needs to be called before set_search().')
      return
    end if
    if (self%nsim > 0) then
      if (self%block%n == 0) then
        call kriging_error('set_search', 'set_grid() needs to be called before set_search().')
        return
      end if
      if (ivar == 1 .and. size(self%obs(ivar)%coord, 2) == self%obs(ivar)%n) then
        call kriging_error('set_search', 'set_sim() needs to be called before set_search().')
        return
      end if
    end if

    a1 = 1.0
    a2 = 1.0
    az = 0.0
    dp = 0.0
    pl = 0.0
    ! Use any pre-stored time_at as the default (CAPI pre-sets it before calling us
    ! to work around gfortran not setting present() through CLASS polymorphic dispatch).
    ta = self%obs(ivar)%time_at
    if (present(anis1))   a1 = anis1
    if (present(anis2))   a2 = anis2
    if (present(azimuth)) az = azimuth
    if (present(dip))     dp = dip
    if (present(plunge))  pl = plunge
    if (present(time_at)) ta = time_at
    if (ta <= EPSLON) then
      call kriging_error('set_search', 'time_at must be positive.')
      return
    end if

    associate(obs => self%obs(ivar))
      if (present(sector_search)) obs%sector_search = sector_search
      obs%rotmat  = calc_rotmat(az, dp, pl, a1, a2)
      obs%time_at = ta
      obs%anisotropic_search = (abs(a1-1.0) > EPSLON .or. abs(a2-1.0) > EPSLON) &
                                .and. self%anisotropic_search

      !-- For SGSIM ivar=1: extended array includes block centres
      if (ivar == 1 .and. self%nsim > 0) then
        call kriging_normalize_nmax(obs%nmax, size(obs%coord,2))
        obs%need_search = size(obs%coord,2) > obs%nmax
      else
        call kriging_normalize_nmax(obs%nmax, obs%n)
        obs%need_search = obs%n > obs%nmax
      end if

      if (obs%need_search) then
        if (associated(obs%tree)) call kdtree2_destroy(obs%tree)
        !-- Build nlag-dimensional transformed coords.
        allocate(tcoord(self%nlag, size(obs%coord,2)))
        if (obs%anisotropic_search) then
          call sub_rotate(obs%rotmat, self%ndim, size(obs%coord,2), &
                          obs%coord(1:self%ndim,:), tcoord(1:self%ndim,:))
        else
          tcoord(1:self%ndim,:) = obs%coord(1:self%ndim,:)
        end if
        tcoord(self%nlag,:) = time_search_coord(ta, obs%coord(self%nlag,:))
        obs%tree => kdtree2_create(tcoord, sort=obs%sector_search, rearrange=.true.)
        if (kriging_failed()) return
      end if
      obs%set_search = .true.
    end associate
  end subroutine set_search


  !=============================================================================
  ! prepare
  !
  ! Pre-solve validation and bookkeeping called at the start of solve().
  ! Computes cov0 for every vgm entry and checks structural validity via
  ! is_valid_st() unless use_old_weight is enabled, then delegates sizing and
  ! weight-file bookkeeping to prepare_common.
  !=============================================================================
  subroutine prepare(self)
    class(t_kriging_st), intent(inout) :: self
    integer :: ivar, jvar
    character(len=*), parameter :: subname = 't_kriging_st%prepare'

    if (.not. associated(self%vgm)) then
      call kriging_error(subname, 'Call initialize() before solve().')
      return
    end if

    !-- Validate variograms and compute cov0 for every entry.
    !   vgm is always 1:nvar; group_ivar() maps sim-group indices to the correct
    !   variogram entry at evaluation time — no slot-0 copies needed.
    if (.not. self%use_old_weight) then
      do ivar = 1, self%nvar
        do jvar = 1, self%nvar
          call self%vgm(jvar, ivar, 1)%compute_cov0()
          if (kriging_failed()) return
          if (.not. self%vgm(jvar, ivar, 1)%is_valid_st(ivar, jvar)) then
            call kriging_error(subname, 'Invalid ST variogram')
            return
          end if
          !-- Build piecewise lookup tables now that ndim and all structures are final.
          !   build_struct_table requires all non-nugget structures to share the same
          !   anisotropy matrix (Level B: 1 aniso_h + 1 lookup regardless of nstruct).
          !   If structures have incompatible matrices, fall through to build_all_tables.
          call self%vgm(jvar, ivar,1)%build_struct_table( &
            h_bounds=[0.0, 0.1, 0.5, 3.5], dh=[1e-5, 1e-4, 1e-3])
        end do
      end do
    end if

    call self%prepare_common(subname)
  end subroutine prepare


  !=============================================================================
  ! search_neighbors
  !
  ! Find the nearest observations (and, for SGSIM, previously simulated blocks)
  ! to the current block centre.  Results are stored in ctx%inear and ctx%nnear.
  ! Searches an nlag-dimensional space [spatial', h_time] using the KD-tree built
  ! in set_search.
  !
  ! SGSIM path (ivar=1 .and. nsim>0)
  ! ----------------------------------
  ! obs(1)%coord holds nobs + nblock entries (extended by set_sim).
  ! Results are partitioned into original obs (index <= nobs) and previously
  ! simulated blocks (index > nobs, shifted to 1-based block index).
  ! Temporal distance is computed in the transformed search-time coordinate.
  !
  ! After distance filtering: any neighbour beyond maxdist is dropped.
  !=============================================================================
  subroutine search_neighbors(self, ivar, ctx)
    class(t_kriging_st),     intent(inout) :: self
    type(t_kriging_ctx),     intent(inout) :: ctx
    integer,                 intent(in)    :: ivar

    integer                  :: i, nobs, igsim
    real                     :: newloc(self%nlag,1), block_t, block_ht
    real                     :: ta
    logical, allocatable     :: is_obs(:)
    character(len=*), parameter :: subname = "t_kriging_st%search_neighbors"

    associate( &
      iblock  => ctx%iblock, &
      nmax    => self%obs(ivar)%nmax, &
      obsloc  => self%obs(ivar)%coord, &
      xloc    => self%block%coord(:, ctx%iblock:ctx%iblock), &
      inear   => ctx%inear(:, ivar), &
      nnear   => ctx%nnear(ivar), &
      dist    => ctx%sqdist(:, ivar), &
      maxdist => self%obs(ivar)%maxdist, &
      rotmat  => self%obs(ivar)%rotmat)

      block_t  = self%block%coord(self%nlag, iblock)
      ta       = self%obs(ivar)%time_at
      block_ht = time_search_coord(ta, block_t)
      nobs    = self%obs(ivar)%n    ! original obs count (not extended)

      !-- Build nlag-dimensional transformed query point
      if (self%obs(ivar)%anisotropic_search) then
        call sub_rotate(rotmat, self%ndim, 1, xloc(1:self%ndim,:), newloc(1:self%ndim,:))
      else
        newloc(1:self%ndim,1) = xloc(1:self%ndim,1)
      end if
      newloc(self%nlag,1) = block_ht

      !----------------------------------------------------------------------
      ! SGSIM path: search over original obs + previously simulated blocks
      !----------------------------------------------------------------------
      if (self%nsim > 0) then
        igsim = self%nvar + ivar
        associate( &
          inearb => ctx%inear(:, igsim), &
          nnearb => ctx%nnear(igsim), &
          distb  => ctx%sqdist(:, igsim))

          if (nmax < nobs + iblock - 1) then
            call kdtree2_n_nearest_maxidx(self%obs(ivar)%tree, newloc(:,1), nmax, &
                                          ctx%results, nobs + iblock - 1)
            if (kriging_failed()) return
            allocate(is_obs, source = ctx%results(1:nmax)%idx <= nobs)
            nnear  = count(is_obs)
            nnearb = nmax - nnear
            inear (1:nnear)  = pack(ctx%results(1:nmax)%idx,  is_obs)
            inearb(1:nnearb) = pack(ctx%results(1:nmax)%idx, .not. is_obs) - nobs
            dist  (1:nnear)  = pack(ctx%results(1:nmax)%dis,  is_obs)
            distb (1:nnearb) = pack(ctx%results(1:nmax)%dis, .not. is_obs)
          else
            nnear  = nobs
            nnearb = iblock - 1
            call set_seq(inear(1:nnear), nnear)
            if (nnearb > 0) call set_seq(inearb(1:nnearb), nnearb)
            dist(1:nnear) = rotated_dists(rotmat, self%ndim, xloc(1:self%ndim,1), &
                              obsloc(1:self%ndim, 1:nnear)) + &
                              (time_search_coord(ta, obsloc(self%nlag,1:nnear)) - block_ht)**2
            distb(1:nnearb) = rotated_dists(rotmat, self%ndim, &
                                xloc(1:self%ndim,1), &
                                self%block%coord(1:self%ndim, 1:nnearb)) + &
                                (time_search_coord(ta, self%block%coord(self%nlag,1:nnearb)) - block_ht)**2
          end if

          !-- Apply maxdist filter (nlag-dimensional distance already in dist/distb)
          call filter_by_maxlag(inearb, distb, nnearb, maxdist)

        end associate  ! inearb, nnearb, distb

      !----------------------------------------------------------------------
      ! Standard kriging / cokriging search
      !----------------------------------------------------------------------
      else
        if (self%obs(ivar)%sector_search) then
          block_sector: block
            use kdtree2_module, only: kdtree2_sort_results
            integer :: ncand, sector, ix, iy, iz, idx
            integer :: sector_count(8)
            real :: pt_rotated(self%ndim, 1)
            real :: dx, dy, dz

            if (self%obs(ivar)%need_search) then
              ncand = min(nobs, (2**self%ndim) * nmax * 4)
              call kdtree2_n_nearest(self%obs(ivar)%tree, newloc(:,1), ncand, ctx%results)
              if (kriging_failed()) return
              call kdtree2_sort_results(ncand, ctx%results)
            else
              ncand = nobs
              ctx%results(1:ncand)%idx = [(i, i=1, ncand)]
              ctx%results(1:ncand)%dis = rotated_dists(rotmat, self%ndim, xloc(1:self%ndim, 1), &
                              obsloc(1:self%ndim, 1:ncand)) + &
                              (time_search_coord(ta, obsloc(self%nlag, 1:ncand)) - block_ht)**2
              call kdtree2_sort_results(ncand, ctx%results)
            end if

            nnear = 0
            sector_count = 0
            do i = 1, ncand
              idx = ctx%results(i)%idx
              if (self%cross_validation .and. idx == iblock) cycle
              if (ctx%results(i)%dis > maxdist) cycle

              if (self%obs(ivar)%anisotropic_search) then
                call sub_rotate(rotmat, self%ndim, 1, obsloc(1:self%ndim, idx:idx), pt_rotated)
              else
                pt_rotated = obsloc(1:self%ndim, idx:idx)
              end if

              dx = pt_rotated(1, 1) - newloc(1, 1)
              dy = pt_rotated(2, 1) - newloc(2, 1)
              if (self%ndim == 3) then
                dz = pt_rotated(3, 1) - newloc(3, 1)
              else
                dz = 0.0
              end if

              if (dx >= 0.0) then
                ix = 0
              else
                ix = 1
              end if
              if (dy >= 0.0) then
                iy = 0
              else
                iy = 1
              end if
              if (self%ndim == 3) then
                if (dz >= 0.0) then
                  iz = 0
                else
                  iz = 1
                end if
                sector = 1 + ix + 2 * iy + 4 * iz
              else
                sector = 1 + ix + 2 * iy
              end if

              if (sector_count(sector) < nmax) then
                sector_count(sector) = sector_count(sector) + 1
                nnear = nnear + 1
                inear(nnear) = idx
                dist(nnear)  = ctx%results(i)%dis
              end if
            end do
          end block block_sector
        else
          if (self%obs(ivar)%need_search) then
            call kdtree2_n_nearest(self%obs(ivar)%tree, newloc(:,1), nmax, ctx%results)
            if (kriging_failed()) return
            nnear          = nmax
            inear(1:nnear) = ctx%results(1:nmax)%idx
            dist (1:nnear) = ctx%results(1:nmax)%dis
          else
            nnear = nobs
            call set_seq(inear(1:nnear), nnear)
            dist(1:nnear) = rotated_dists(rotmat, self%ndim, xloc(1:self%ndim,1), &
                              obsloc(1:self%ndim,1:nnear)) + &
                              (time_search_coord(ta, obsloc(self%nlag,1:nnear)) - block_ht)**2
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
        end if
      end if

      !-- maxdist filter on nlag-dimensional distance
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
  ! Matrix layout contains observation groups, optional SGSIM simulated-block
  ! groups, and optional time-aware gradient pair groups.  ST gradient pair
  ! endpoints are full nlag coordinates, so covariance with another time slice
  ! is penalized by the temporal component of vgm_struct_st%cov_tab_st().
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
  ! Obs/sim groups (kvar <= ngroups_base) are filled by rowloop/columnloop.
  ! Grad groups are appended after ngroups_base, one slot per variable.
  ! Drift and unbiasedness columns are appended explicitly for both obs/sim
  ! and grad groups.  The mirror pass at the end copies the assembled lower
  ! triangle to upper.
  !============================================================================
  subroutine assemble_lhs(self, ctx)
    class(t_kriging_st), intent(inout) :: self
    type(t_kriging_ctx), intent(inout) :: ctx

    integer                :: kvar, lvar, i, j, jstart
    integer                :: ivgm, ivar, jvar, givar
    real                   :: lag(self%nlag), ln, cov_g, c11, c12, c21, c22
    class(t_data), pointer :: obs1, obs2

    associate( &
      matA      => ctx%matA, &
      inear     => ctx%inear, &
      nnear     => ctx%nnear, &
      istart    => ctx%istart, &
      npp       => ctx%npp, &
      matsize   => ctx%matsize, &
      ndrift    => self%ndrift, &
      naug      => self%naug)

      matA(1:matsize, 1:matsize) = 0.0

      ivgm = 1
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
                lag = (obs1%coord(:, inear(i,kvar)) - obs2%coord(:, inear(j,lvar)))
                matA(istart(lvar)+j, istart(kvar)+i) = vgm%cov_tab_st(lag)
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
        gradidentloop: do lvar = self%ngroups_base+1, self%ngroups
          if (nnear(lvar) == 0) cycle
          givar = lvar - self%ngroups_base

          !-- grad-obs cross block: C(xs1, obs) - C(xs2, obs).
          !   Grad endpoints and obs coords are full nlag vectors, including time.
          associate(vgm => self%vgm(givar, ivar, ivgm))
            do i = 1, nnear(kvar)
              do j = 1, nnear(lvar)
                lag   = self%grad(givar)%coord(:,j) - obs1%coord(:, inear(i,kvar))
                cov_g = vgm%cov_tab_st(lag)
                lag   = self%grad(givar)%coord2(:,j) - obs1%coord(:, inear(i,kvar))
                matA(istart(lvar)+j, istart(kvar)+i) = cov_g - vgm%cov_tab_st(lag)
              end do
            end do
          end associate
        end do gradidentloop

      end do rowloop

      !-- grad-grad covariance (diagonal and both off-diagonal triangles explicit).
      do ivar = 1, self%nvar
        kvar = self%ngroups_base + ivar
        if (nnear(kvar) == 0) cycle
        associate(vgm => self%vgm(ivar, ivar, ivgm), grad=>self%grad(ivar), is=>istart(kvar))
          do i = 1, grad%n
            !-- Diagonal: 2*C(0) - 2*C(xs1-xs2) + grad variance
            lag = grad%coord(:,i) - grad%coord2(:,i)
            matA(is+i, is+i) = &
              2.0 * vgm%cov0 - 2.0 * vgm%cov_tab_st(lag) + grad%variance(1,1,i)
            do j = i+1, grad%n
              lag = grad%coord (:,i) - grad%coord (:,j); c11 = vgm%cov_tab_st(lag)
              lag = grad%coord (:,i) - grad%coord2(:,j); c12 = vgm%cov_tab_st(lag)
              lag = grad%coord2(:,i) - grad%coord (:,j); c21 = vgm%cov_tab_st(lag)
              lag = grad%coord2(:,i) - grad%coord2(:,j); c22 = vgm%cov_tab_st(lag)
              matA(is+j, is+i) = c11 - c12 - c21 + c22
            end do
          end do
          !-- grad-drift columns at npp+1:matsize (both triangles).
          if (ndrift + naug > 0) then
            do i = 1, grad%n
              matA(npp+1:matsize, is+i) = grad%drift(:, 1, i)
            end do
          end if
        end associate
      end do

      !-- Mirror lower triangle to upper: copies matA(col, row) to matA(row, col) for col > row.
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
    class(t_kriging_st), intent(inout) :: self
    type(t_kriging_ctx), intent(inout) :: ctx
    ! local
    integer                :: ivar, givar, kvar, kgrad, i, k, nn, jvar, ivgm, igrad
    real                   :: lag(self%nlag), tmp, tmp2
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
      naug      => self%naug)

      ivgm = 1

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
                lag = (obs%coord(:, inear(i, kvar)) - self%grid%coord(:, k))
                tmp = tmp + vgm%cov_tab_st(lag) * self%grid%weight(k)
              end do
              rhsB(ivar, istart(kvar)+i) = tmp
            end do
          end associate
        end do

        !-- Gradient RHS: C(xs1,x0) - C(xs2,x0) for each grad group.
        !   Gradient endpoints and grid coords are full nlag vectors.
        do givar = 1, self%nvar
          if (self%grad(givar)%n == 0) cycle
          kgrad = self%ngroups_base + givar
          associate(vgm => self%vgm(givar, ivar, ivgm))
            do igrad = 1, self%grad(givar)%n
              tmp  = 0.0
              tmp2 = 0.0
              do k = self%block%iblockpnt(iblock), self%block%iblockpnt(iblock)+self%block%nblockpnt(iblock)-1
                lag = self%grad(givar)%coord(:,igrad) - self%grid%coord(:,k)
                tmp = tmp + vgm%cov_tab_st(lag) * self%grid%weight(k)
                lag = self%grad(givar)%coord2(:,igrad) - self%grid%coord(:,k)
                tmp2 = tmp2 + vgm%cov_tab_st(lag) * self%grid%weight(k)
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
    class(t_kriging_st), intent(inout) :: self
    type(t_kriging_ctx), intent(inout) :: ctx

    integer :: i, j, k1, ivar, jvar
    real    :: lag(self%nlag), base_cov

    lag = 0.0

    associate( &
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
          associate(vgm => self%vgm(ivar, jvar, 1))
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
                  lag = coord(1:self%nlag, k1+i) - coord(1:self%nlag, k1+j)
                  base_cov = base_cov + vgm%cov_tab_st(lag) * &
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
  ! Outputs:
  !   1. Global ST model parameters (model type, temporal transform, at, …)
  !   2. Per variable-pair: cs (spatial), ct (temporal), sill_st (sum-metric)
  !============================================================================
  function tostr_vgm(self) result(s)
    class(t_kriging_st), intent(in) :: self
    character(len=:), allocatable :: s

    character(len=256) :: buf
    character(len=32)  :: model_name, trans_name
    integer :: ivar, jvar, k
    character(len=1), parameter :: NL = new_line('A')

    associate(m => self%st_model, tv => self%st_time_vtype_id)

      !-- model type label
      select case(m)
      case(ST_MODEL_SUM_METRIC);  model_name = 'Sum-Metric'
      case(ST_MODEL_PRODUCT_SUM); model_name = 'Product-Sum'
      case default;               write(model_name,'(A,I0,A)') 'Unknown(', m, ')'
      end select

      !-- temporal transform label (VGM_LIN=8, VGM_EXP=2, VGM_POW=5)
      select case(tv)
      case(ST_TRANSFORM_LINEAR);  trans_name = 'Linear'
      case(ST_TRANSFORM_BOUNDED); trans_name = 'Bounded (exp)'
      case(ST_TRANSFORM_POWER);   trans_name = 'Power'
      case default;               write(trans_name,'(A,I0,A)') 'Unknown(', tv, ')'
      end select
    end associate

    s = ''
    s = s // " Space-Time Variogram" // NL
    write(buf,'(A,A)')       "  Model type      : ", trim(model_name)       ; s = s // trim(buf) // NL
    write(buf,'(A,A,3(A,G12.5))') &
      "  Transform f(dt) : ", trim(trans_name), &
      "  at=", self%st_at, "  nugget=", self%st_time_nugget, "  sill=", self%st_time_sill
    s = s // trim(buf) // NL
    if (self%st_model == ST_MODEL_PRODUCT_SUM) then
      if (associated(self%vgm)) then
        write(buf,'(A,G12.5)') "  k_ps            : ", self%vgm(1,1,1)%k_ps ; s = s // trim(buf) // NL
      end if
    end if

    if (.not. associated(self%vgm)) return

    do ivar = 1, self%nvar
      do jvar = 1, self%nvar
        s = s // NL
        if (ivar == jvar) then
          write(buf,'(A,I0)') " Variable ", ivar
        else
          write(buf,'(A,I0,A,I0)') " Variable ", ivar, " <-> Variable ", jvar
        end if
        s = s // trim(buf) // NL

        associate(v => self%vgm(jvar, ivar, 1))
          !-- cov0
          write(buf,'(A,G12.5)') "  cov0            : ", v%cov0 ; s = s // trim(buf) // NL

          !-- spatial component
          if (v%cs%nstruct > 0) then
            s = s // "  Spatial  (cs)   :" // v%cs%tostr() // NL
          else
            s = s // "  Spatial  (cs)   :  (not set)" // NL
          end if

          !-- temporal component
          if (v%ct%nstruct > 0) then
            s = s // "  Temporal (ct)   :" // v%ct%tostr() // NL
          else
            s = s // "  Temporal (ct)   :  (not set)" // NL
          end if

          !-- joint sills (sum-metric only)
          if (self%st_model == ST_MODEL_SUM_METRIC .and. allocated(v%sill_st)) then
            write(buf,'(A,I0,A)') "  Joint sills (", size(v%sill_st), " struct):"
            s = s // trim(buf)
            do k = 1, size(v%sill_st)
              write(buf,'(1X,G13.6)') v%sill_st(k)
              s = s // trim(buf)
            end do
            s = s // NL
          end if

        end associate
      end do
    end do
  end function tostr_vgm
  !=============================================================================
  ! finalize
  !
  ! Release all allocated memory.  Call after all results have been read from
  ! block%value and block%variance.
  !=============================================================================
  subroutine finalize(self)
    class(t_kriging_st), intent(inout) :: self
    call self%finalize_common()
    if (associated(self%vgm)) deallocate(self%vgm)
  end subroutine finalize

end module kriging_st
