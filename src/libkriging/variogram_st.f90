!==============================================================================
! Module: variogram_st
!
! Space-time variogram models for 3D spatial + 1D temporal cokriging.
!
! Design
! ------
! vgm_struct_st wraps two vgm_struct objects (cs for space, ct for time)
! plus optional joint sills for the sum-metric model.  It does NOT extend
! vgm_struct because the public cov_lag interface has a different signature
! (it needs an explicit dt argument that cannot be hidden in a 3-vector).
!
! Two ST covariance models are supported:
!
!   Sum-metric (model=ST_MODEL_SUM_METRIC, can be reduced to metric covariance model with sill=0 for Cs and Ct):
!     C(hs,dt) = Cs(hs) + Ct(dt) + sum_k{ sill_st(k) * shape_k(h_st) }
!     where h_st = sqrt( h_s_k^2 + f(dt)^2 )
!     h_s_k is the dimensionless spatial lag for structure k of cs,
!     and Cst inherits the functional form (shape + anisotropy) from cs.
!
!   Product-sum (model=ST_MODEL_PRODUCT_SUM, can be reduced to separable model with sill=0 for Cs and Ct):
!     C(hs,dt) = k_ps * Cs(hs) * Ct(dt) + Cs(hs) + Ct(dt)
!
! Temporal transform f(dt) maps physical time lag into the dimensionless
! temporal distance used by the ST joint component:
!   f(0)  = 0
!   f(dt) = nugget + sill * (1 - corefunc_fn(vtype, |dt| / at)) for dt /= 0
!
! The 'at' parameter is the joint space-time temporal scale.
! Each nested structure in ct has its own temporal range (a_major) set
! via add_temporal("vtype nugget sill at_k").
!
! Spec formats
! ------------
! Spatial (same as base variogram):
!   "vtype nugget sill a_major a_minor1 a_minor2 azimuth dip plunge"
!
! Temporal (simplified 1D):
!   "vtype nugget sill at_k"
!   Internally expanded to full 9-param spec with isotropic ranges = at_k.
!
! LMC validity for cokriging:
!   Each nested spatial structure k: sill_12_k^2 <= sill_11_k * sill_22_k
!   Each nested temporal structure k: sill_12t_k^2 <= sill_11t_k * sill_22t_k
!   Joint sills (sum-metric): sill_st_12^2 <= sill_st_11 * sill_st_22
!==============================================================================
module variogram_st
  use kriging_err, only: kriging_error, kriging_failed
  use variogram
  use vgm_func, only: corefunc_fn, VGM_NUG, VGM_SPH, VGM_EXP, VGM_GAU, &
                      VGM_HOL, VGM_POW, VGM_BSQ, VGM_CIR, VGM_LIN
  implicit none
  private

  public :: vgm_struct_st
  public :: ST_MODEL_SUM_METRIC, ST_MODEL_PRODUCT_SUM
  public :: ST_TRANSFORM_LINEAR, ST_TRANSFORM_BOUNDED, ST_TRANSFORM_POWER
  public :: f_time_vgm_st

  !-- Model type constants
  integer, parameter :: ST_MODEL_SUM_METRIC  = 0
  integer, parameter :: ST_MODEL_PRODUCT_SUM = 1

  !-- Temporal transform constants
  integer, parameter :: ST_TRANSFORM_LINEAR  = VGM_LIN
  integer, parameter :: ST_TRANSFORM_BOUNDED = VGM_EXP
  integer, parameter :: ST_TRANSFORM_POWER   = VGM_POW

  !=============================================================================
  ! vgm_struct_st
  !
  ! One space-time variogram model for a variable pair (ivar, jvar).
  ! cs  — spatial sub-variogram (shape also borrowed for Cst joint component)
  ! ct  — temporal sub-variogram (independent shape and ranges)
  ! sill_st(:) — joint sills [cs%nstruct]; allocated only for sum-metric model
  !
  ! model, transform, and at are set globally via t_kriging_st%set_st_model
  ! and copied into every vgm(:,:) entry so that cov_lag_st is self-contained.
  !=============================================================================
  type :: vgm_struct_st
    type(vgm_struct)  :: cs             ! spatial variogram
    type(vgm_struct)  :: ct             ! temporal variogram (always ndim=1)
    real, allocatable :: sill_st(:)     ! joint sills [cs%nstruct], sum-metric only
    integer           :: ndim      = 0
    integer           :: model     = ST_MODEL_SUM_METRIC
    integer           :: time_vtype_id = ST_TRANSFORM_LINEAR
    real              :: time_nugget   = 0.0   ! jump in f(dt) for dt /= 0
    real              :: time_sill     = 1.0   ! upper scale for f(dt)
    real              :: at            = 1.0   ! joint temporal scale (time units)
    real              :: k_ps      = 0.0   ! product-sum coefficient k
    real              :: cov0      = 0.0   ! C(0,0) used for kriging matrix diagonal
  contains
    procedure :: f_time           => f_time_bound_vgm_st
    procedure :: cov_lag          => cov_lag_vgm_st       ! analytic, ndim+1 vector
    procedure :: cov_tab          => cov_tab_vgm_st       ! table,    ndim+1 vector
    procedure :: add_spatial      => add_spatial_vgm_st
    procedure :: add_temporal     => add_temporal_vgm_st
    procedure :: set_joint_sills  => set_joint_sills_vgm_st
    procedure :: compute_cov0     => compute_cov0_vgm_st
    procedure :: is_valid_st      => is_valid_vgm_st
    procedure :: build_table      => build_table_vgm_st         ! entry to build table for different components
    procedure :: reset            => reset_vgm_st
    procedure :: reset_model      => reset_vgm_st   ! alias for spatial naming compat
  end type vgm_struct_st

contains

  !=============================================================================
  ! f_time — transform physical |dt| to dimensionless dw for ST joint distance
  !=============================================================================
  pure function f_time_bound_vgm_st(self, dt) result(dw)
    class(vgm_struct_st), intent(in) :: self
    real,                 intent(in) :: dt       ! physical time lag (any sign)
    real :: dw

    dw = f_time_vgm_st(self%time_vtype_id, self%time_nugget, self%time_sill, &
                       self%at, dt)
  end function f_time_bound_vgm_st


  !=============================================================================
  ! f_time_vgm_st -- shared temporal transform formula.
  !
  ! Used by vgm_struct_st%f_time() and by ST neighbour search when the search
  ! temporal anisotropy intentionally differs from the variogram anisotropy.
  !=============================================================================
  pure elemental function f_time_vgm_st(vtype_id, nugget, sill, at, dt) result(dw)
    integer, intent(in) :: vtype_id
    real,    intent(in) :: nugget
    real,    intent(in) :: sill
    real,    intent(in) :: at
    real,    intent(in) :: dt       ! physical time lag or coordinate magnitude
    real :: dw, adt

    if (abs(dt) <= epsilon(1.0)) then
      dw = 0.0
      return
    end if

    adt = abs(dt) / at              ! normalise by temporal search/model scale
    dw = nugget + sill * (1.0 - corefunc_fn(vtype_id, adt))
    dw = max(dw, 0.0)
  end function f_time_vgm_st


  !=============================================================================
  ! cov_lag_st — evaluate the ST covariance C(lag_s, dt)
  !
  ! lag_s(3) : spatial lag vector (already in the coordinate units of obs%coord)
  ! dt       : temporal lag in the same time units used when loading observations
  !=============================================================================
  !=============================================================================
  ! cov_lag — analytic, unified wrapper: takes lag(ndim+1) vector, splits at
  !           ndim+1, zero-pads spatial to 3D, then calls cov_lag_st.
  !           Matches the signature of vgm_struct%cov_lag so call sites are
  !           identical for spatial and ST covariance evaluation.
  !=============================================================================
  function cov_lag_vgm_st(self, lags) result(res)
    class(vgm_struct_st), intent(in) :: self
    real,                 intent(in) :: lags(self%ndim+1)
    real :: res
    ! local
    real :: lag_s(3)
    real :: dt
    real :: dw, h_s, h_st, cs_val, ct_val
    real :: lag_t(3)
    integer :: k

    !-- Temporal lag as a 1D spatial vector so we can reuse vgm_struct%cov_lag.
    !   ct is isotropic in time: a_major = a_minor1 = a_minor2 = at_k.
    !   cov_lag([|dt|, 0, 0]) = ct_sill * corefunc(|dt| / at_k).
    lag_t = [abs(lags(self%ndim+1)), 0.0, 0.0]
    lag_s = lags(1:self%ndim)
    dt    = lags(self%ndim+1)
    select case (self%model)

      !------------------------------------------------------------------------
      case (ST_MODEL_SUM_METRIC)
      !   C = Cs(hs) + Ct(dt) + sum_k{ sill_st_k * shape_k(sqrt(h_sk^2 + dw^2)) }
      !------------------------------------------------------------------------
        cs_val = self%cs%cov_lag(lag_s)
        ct_val = self%ct%cov_lag(lag_t)
        res    = cs_val + ct_val

        if (allocated(self%sill_st)) then
          dw = self%f_time(dt)     ! dimensionless joint temporal distance
          do k = 1, self%cs%nstruct
            associate(comp => self%cs%structs(k))
              !-- dimensionless spatial distance for structure k
              h_s  = comp%aniso%h_iso(lag_s(1), lag_s(2), lag_s(3))
              h_st = sqrt(h_s**2 + dw**2)
              !-- joint contribution: reuses cs vtype_id, different sill
              res  = res + self%sill_st(k) * corefunc_fn(comp%vtype_id, h_st)
            end associate
          end do
        end if

      !------------------------------------------------------------------------
      case (ST_MODEL_PRODUCT_SUM)
      !   C = k_ps * Cs(hs) * Ct(dt) + Cs(hs) + Ct(dt)
      !------------------------------------------------------------------------
        cs_val = self%cs%cov_lag(lag_s)
        ct_val = self%ct%cov_lag(lag_t)
        res    = self%k_ps * cs_val * ct_val + cs_val + ct_val

      case default
        res = 0.0

    end select
  end function cov_lag_vgm_st


  !=============================================================================
  ! cov_tab_st — table-accelerated twin of cov_lag_st.
  !
  ! Uses cs%cov_tab / ct%cov_tab when tables have been built; falls back to
  ! analytic automatically (vgm_struct%cov_tab handles the dispatch).
  ! For the sum-metric joint term, normalises the per-component table value
  ! by comp%sill to recover corefunc(h_st), or falls back to corefunc_fn
  ! when the table is absent or sill is zero.
  !=============================================================================
  function cov_tab_vgm_st(self, lags) result(res)
    class(vgm_struct_st), intent(in) :: self
    real,                 intent(in) :: lags(self%ndim+1)
    real :: res
    ! local
    real :: lag_s(3)
    real :: dt
    real :: dw, h_s, h_st, cs_val, ct_val
    real :: lag_t(3)
    integer :: k

    !-- Temporal lag as a 1D spatial vector (ct%ndim=1).
    lag_t = [abs(lags(self%ndim+1)), 0.0, 0.0]
    lag_s = lags(1:self%ndim)
    dt    = lags(self%ndim+1)

    select case (self%model)

      !------------------------------------------------------------------------
      case (ST_MODEL_SUM_METRIC)
      !------------------------------------------------------------------------
        cs_val = self%cs%cov_tab(lag_s)        ! Level B or A spatial table
        ct_val = self%ct%cov_tab(lag_t)        ! 1D temporal table
        res    = cs_val + ct_val

        if (allocated(self%sill_st)) then
          dw = self%f_time(dt)
          do k = 1, self%cs%nstruct
            associate(comp => self%cs%structs(k))
              h_s  = comp%aniso%h_iso(lag_s(1), lag_s(2), lag_s(3))
              h_st = sqrt(h_s**2 + dw**2)
              !-- Use the per-component table to evaluate corefunc(h_st).
              !   comp%cov_tab_h returns sill*corefunc(h), so divide by sill
              !   to isolate the shape.  Fall back to corefunc_fn if table is
              !   absent or sill is numerically zero.
              if (comp%tab_ready .and. comp%sill > 0.0) then
                res = res + self%sill_st(k) * comp%cov_tab_h(h_st) / comp%sill
              else
                res = res + self%sill_st(k) * corefunc_fn(comp%vtype_id, h_st)
              end if
            end associate
          end do
        end if

      !------------------------------------------------------------------------
      case (ST_MODEL_PRODUCT_SUM)
      !------------------------------------------------------------------------
        cs_val = self%cs%cov_tab(lag_s)
        ct_val = self%ct%cov_tab(lag_t)
        res    = self%k_ps * cs_val * ct_val + cs_val + ct_val

      case default
        res = 0.0

    end select
  end function cov_tab_vgm_st


  !=============================================================================
  ! add_spatial — parse a full 9-param spatial spec and add to cs
  !   spec: "vtype nugget sill a_major a_minor1 a_minor2 azimuth dip plunge"
  !=============================================================================
  subroutine add_spatial_vgm_st(self, vtype, nugget, sill, a_major, a_minor1, a_minor2, azimuth, dip, plunge)
    class(vgm_struct_st), intent(inout) :: self
    character(*),         intent(in) :: vtype
    real,                 intent(in) :: nugget, sill, a_major, a_minor1, a_minor2, azimuth, dip, plunge


    self%cs%ndim = self%ndim
    call self%cs%add_args(trim(vtype), nugget, sill, &
                          a_major, a_minor1, a_minor2, azimuth, dip, plunge)
  end subroutine add_spatial_vgm_st


  !=============================================================================
  ! add_temporal — parse a simplified 4-param temporal spec and add to ct
  !   spec: "vtype nugget sill at_k"
  !   at_k: temporal practical range for this nested structure (physical time units)
  !   Expanded to isotropic geometry: a_major=a_minor1=a_minor2=at_k, angles=0
  !=============================================================================
  subroutine add_temporal_vgm_st(self, vtype, nugget, sill, at_k)
    class(vgm_struct_st), intent(inout) :: self
    character(*), optional, intent(in) :: vtype
    real,         optional, intent(in) :: nugget, sill, at_k

    character(24) :: vtype_
    real          :: nugget_, sill_, at_k_

    self%ct%ndim = 1
    vtype_  = 'sph' ; if (present(vtype )) vtype_  = vtype
    nugget_ = 0.0   ; if (present(nugget)) nugget_ = nugget
    sill_   = 1.0   ; if (present(sill  )) sill_   = sill
    at_k_   = 1.0   ; if (present(at_k  )) at_k_   = at_k
    call self%ct%add_args(trim(vtype_), nugget_, sill_, &
                          at_k_, at_k_, at_k_, 0.0, 0.0, 0.0)
  end subroutine add_temporal_vgm_st


  !=============================================================================
  ! set_joint_sills — supply the joint sill array for the sum-metric model.
  !   sills(n): partial sills for the joint Cst component, one per cs structure.
  !   Must be called AFTER all spatial structures have been added via add_spatial.
  !=============================================================================
  subroutine set_joint_sills_vgm_st(self, sills, n)
    class(vgm_struct_st), intent(inout) :: self
    integer,              intent(in)    :: n
    real,                 intent(in)    :: sills(n)

    if (n /= self%cs%nstruct) then
      call kriging_error("set_joint_sills_vgm_st", 'vgm_struct_st%set_joint_sills: length of sills must equal cs%nstruct')
      return
    end if
    if (allocated(self%sill_st)) deallocate(self%sill_st)
    allocate(self%sill_st(n))
    self%sill_st = sills
  end subroutine set_joint_sills_vgm_st


  !=============================================================================
  ! compute_cov0 — compute C(0,0) for the diagonal of the kriging matrix.
  !   Must be called after all add_spatial, add_temporal, set_joint_sills calls.
  !=============================================================================
  subroutine compute_cov0_vgm_st(self)
    class(vgm_struct_st), intent(inout) :: self
    integer :: k

    select case (self%model)

      case (ST_MODEL_SUM_METRIC)
        !-- C(0,0) = Cs(0) + Ct(0) + sum_k{ sill_st_k * corefunc_k(0) }
        !   corefunc(0) = 1 for all shapes except variog_nug (which = 0)
        self%cov0 = self%cs%cov0 + self%ct%cov0
        if (allocated(self%sill_st)) then
          do k = 1, self%cs%nstruct
            associate(comp => self%cs%structs(k))
              !-- Skip nugget structures: corefunc_nug is identically 0,
              !   so the joint component at h_st=0 would also be 0.
              if (comp%vtype_id /= VGM_NUG) &
                self%cov0 = self%cov0 + self%sill_st(k)
            end associate
          end do
        end if

      case (ST_MODEL_PRODUCT_SUM)
        !-- C(0,0) = k_ps * Cs(0) * Ct(0) + Cs(0) + Ct(0)
        self%cov0 = self%k_ps * self%cs%cov0 * self%ct%cov0 &
                  + self%cs%cov0 + self%ct%cov0

    end select
  end subroutine compute_cov0_vgm_st


  !=============================================================================
  ! is_valid_st — heuristic validation of the ST variogram model.
  !   ivar, jvar: variable indices (1-based), used to tailor cross-var warnings.
  !=============================================================================
  function is_valid_vgm_st(self, ivar, jvar) result(ok)
    class(vgm_struct_st), intent(in) :: self
    integer,              intent(in) :: ivar, jvar
    logical :: ok
    integer :: k
    character(256) :: msg

    ok = self%cs%is_valid() .and. self%ct%is_valid()

    if (self%model == ST_MODEL_SUM_METRIC) then
      !-- Joint sills must be set and non-negative
      if (.not. allocated(self%sill_st)) then
        write(msg,'(A,I0,A,I0,A)') &
          'WARNING vgm_struct_st(',ivar,',',jvar,'): sill_st not set for sum-metric model'
        ok = .false.
      else
        do k = 1, size(self%sill_st)
          if (self%sill_st(k) < 0.0) then
            write(msg,'(A,I0,A,I0,A,I0)') &
              'WARNING vgm_struct_st(',ivar,',',jvar,'): negative joint sill at structure ', k
            ok = .false.
          end if
        end do
      end if
    end if

    if (self%at <= 0.0) then
      write(msg,'(A,I0,A,I0,A)') &
        'WARNING vgm_struct_st(',ivar,',',jvar,'): at must be positive'
      ok = .false.
    end if

    if (self%cov0 <= 0.0) then
      write(msg,'(A,I0,A,I0,A)') &
        'WARNING vgm_struct_st(',ivar,',',jvar,'): cov0 not computed (call compute_cov0)'
      ok = .false.
    end if
    if (.not. ok) call kriging_error('is_valid_vgm_st', trim(msg))
  end function is_valid_vgm_st


  !=============================================================================
  ! build_table — public entry point: delegates to cs%build_table then ct%build_table.
  !
  ! Each sub-call tries Level B (composite struct table) first and falls back to
  ! Level A (per-component tables) internally, so callers never need to know
  ! which level was used.  For ct this is always Level B because all temporal
  ! structures share the same 1D isotropic anisotropy matrix.
  !=============================================================================
  subroutine build_table_vgm_st(self, n_tab, hmax_factor, h_bounds, dh)
    class(vgm_struct_st), intent(inout) :: self
    integer, intent(in), optional :: n_tab
    real,    intent(in), optional :: hmax_factor, h_bounds(:), dh(:)

    call self%cs%build_table(n_tab=n_tab, hmax_factor=hmax_factor, &
                              h_bounds=h_bounds, dh=dh)
    if (kriging_failed()) return
    call self%ct%build_table(n_tab=n_tab, hmax_factor=hmax_factor, &
                              h_bounds=h_bounds, dh=dh)
  end subroutine build_table_vgm_st


  !=============================================================================
  ! reset_vgm_st — clear the ST variogram model back to an empty state
  !=============================================================================
  subroutine reset_vgm_st(self)
    class(vgm_struct_st), intent(inout) :: self
    call self%cs%reset()
    call self%ct%reset()
    if (allocated(self%sill_st)) deallocate(self%sill_st)
    self%cov0 = 0.0
  end subroutine reset_vgm_st


end module variogram_st
