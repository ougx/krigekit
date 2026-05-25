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
! Temporal transform f(dt) — controls how dt maps to dimensionless dw used
! in the ST joint distance:
!   linear  (transform=0): dw = |dt| / at
!   bounded (transform=1): dw = 1 - exp(-|dt| / at)   [dw in [0,1)]
!   power   (transform=2): dw = (|dt| / at)^alpha
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

  use variogram
  implicit none
  private

  public :: vgm_struct_st
  public :: ST_MODEL_SUM_METRIC, ST_MODEL_PRODUCT_SUM
  public :: ST_TRANSFORM_LINEAR, ST_TRANSFORM_BOUNDED, ST_TRANSFORM_POWER

  !-- Model type constants
  integer, parameter :: ST_MODEL_SUM_METRIC  = 0
  integer, parameter :: ST_MODEL_PRODUCT_SUM = 1

  !-- Temporal transform constants
  integer, parameter :: ST_TRANSFORM_LINEAR  = 0
  integer, parameter :: ST_TRANSFORM_BOUNDED = 1
  integer, parameter :: ST_TRANSFORM_POWER   = 2

  !=============================================================================
  ! vgm_struct_st
  !
  ! One space-time variogram model for a variable pair (ivar, jvar).
  ! cs  — spatial sub-variogram (shape also borrowed for Cst joint component)
  ! ct  — temporal sub-variogram (independent shape and ranges)
  ! sill_st(:) — joint sills [cs%nstruct]; allocated only for sum-metric model
  !
  ! model, transform, at, alpha are set globally via t_kriging_st%set_st_model
  ! and copied into every vgm(:,:) entry so that cov_lag_st is self-contained.
  !=============================================================================
  type :: vgm_struct_st
    type(vgm_struct)  :: cs             ! spatial variogram
    type(vgm_struct)  :: ct             ! temporal variogram
    real, allocatable :: sill_st(:)     ! joint sills [cs%nstruct], sum-metric only
    integer           :: model     = ST_MODEL_SUM_METRIC
    integer           :: transform = ST_TRANSFORM_LINEAR
    real              :: at        = 1.0   ! joint temporal scale (time units)
    real              :: alpha     = 1.0   ! power exponent for power transform
    real              :: k_ps      = 0.0   ! product-sum coefficient k
    real              :: cov0_val  = 0.0   ! C(0,0) used for kriging matrix diagonal
  contains
    procedure :: f_time           => f_time_vgm_st
    procedure :: cov_lag_st       => cov_lag_vgm_st
    procedure :: add_spatial      => add_spatial_vgm_st
    procedure :: add_temporal     => add_temporal_vgm_st
    procedure :: set_joint_sills  => set_joint_sills_vgm_st
    procedure :: compute_cov0     => compute_cov0_vgm_st
    procedure :: is_valid_st      => is_valid_vgm_st
  end type vgm_struct_st

contains

  !=============================================================================
  ! f_time — transform physical |dt| to dimensionless dw for ST joint distance
  !=============================================================================
  pure function f_time_vgm_st(this, dt) result(dw)
    class(vgm_struct_st), intent(in) :: this
    real,                 intent(in) :: dt       ! physical time lag (any sign)
    real :: dw, adt

    adt = abs(dt) / this%at          ! normalise by joint temporal scale
    select case (this%transform)
      case (ST_TRANSFORM_LINEAR);  dw = adt
      case (ST_TRANSFORM_BOUNDED); dw = 1.0 - exp(-adt)
      case (ST_TRANSFORM_POWER);   dw = adt ** this%alpha
      case default;                dw = adt
    end select
  end function f_time_vgm_st


  !=============================================================================
  ! cov_lag_st — evaluate the ST covariance C(lag_s, dt)
  !
  ! lag_s(3) : spatial lag vector (already in the coordinate units of obs%coord)
  ! dt       : temporal lag in the same time units used when loading observations
  !=============================================================================
  function cov_lag_vgm_st(this, lag_s, dt) result(res)
    class(vgm_struct_st), intent(in) :: this
    real,                 intent(in) :: lag_s(3)
    real,                 intent(in) :: dt
    real :: res

    real :: dw, h_s, h_st, cs_val, ct_val
    real :: lag_t(3)
    integer :: k

    !-- Temporal lag as a 1D spatial vector so we can reuse vgm_struct%cov_lag.
    !   ct is isotropic in time: a_major = a_minor1 = a_minor2 = at_k.
    !   cov_lag([|dt|, 0, 0]) = ct_sill * corefunc(|dt| / at_k).
    lag_t = [abs(dt), 0.0, 0.0]

    select case (this%model)

      !------------------------------------------------------------------------
      case (ST_MODEL_SUM_METRIC)
      !   C = Cs(hs) + Ct(dt) + sum_k{ sill_st_k * shape_k(sqrt(h_sk^2 + dw^2)) }
      !------------------------------------------------------------------------
        cs_val = this%cs%cov_lag(lag_s)
        ct_val = this%ct%cov_lag(lag_t)
        res    = cs_val + ct_val

        if (allocated(this%sill_st)) then
          dw = this%f_time(dt)     ! dimensionless joint temporal distance
          do k = 1, this%cs%nstruct
            associate(comp => this%cs%structs(k))
              if (.not. allocated(comp%shape)) cycle
              !-- dimensionless spatial distance for structure k
              h_s  = comp%aniso%h_iso(lag_s(1), lag_s(2), lag_s(3))
              h_st = sqrt(h_s**2 + dw**2)
              !-- joint contribution: same shape as cs, sill from sill_st
              res  = res + this%sill_st(k) * comp%shape%corefunc(h_st)
            end associate
          end do
        end if

      !------------------------------------------------------------------------
      case (ST_MODEL_PRODUCT_SUM)
      !   C = k_ps * Cs(hs) * Ct(dt) + Cs(hs) + Ct(dt)
      !------------------------------------------------------------------------
        cs_val = this%cs%cov_lag(lag_s)
        ct_val = this%ct%cov_lag(lag_t)
        res    = this%k_ps * cs_val * ct_val + cs_val + ct_val

      case default
        res = 0.0

    end select
  end function cov_lag_vgm_st


  !=============================================================================
  ! add_spatial — parse a full 9-param spatial spec and add to cs
  !   spec: "vtype nugget sill a_major a_minor1 a_minor2 azimuth dip plunge"
  !=============================================================================
  subroutine add_spatial_vgm_st(this, spec)
    class(vgm_struct_st), intent(inout) :: this
    character(*),         intent(in)    :: spec

    character(24) :: vtype
    real          :: nugget, sill, a_major, a_minor1, a_minor2, azimuth, dip, plunge

    read(spec, *) vtype, nugget, sill, a_major, a_minor1, a_minor2, azimuth, dip, plunge
    call this%cs%add_args(trim(vtype), nugget, sill, &
                          a_major, a_minor1, a_minor2, azimuth, dip, plunge)
  end subroutine add_spatial_vgm_st


  !=============================================================================
  ! add_temporal — parse a simplified 4-param temporal spec and add to ct
  !   spec: "vtype nugget sill at_k"
  !   at_k: temporal practical range for this nested structure (physical time units)
  !   Expanded to isotropic geometry: a_major=a_minor1=a_minor2=at_k, angles=0
  !=============================================================================
  subroutine add_temporal_vgm_st(this, spec)
    class(vgm_struct_st), intent(inout) :: this
    character(*),         intent(in)    :: spec

    character(24) :: vtype
    real          :: nugget, sill, at_k

    read(spec, *) vtype, nugget, sill, at_k
    call this%ct%add_args(trim(vtype), nugget, sill, &
                          at_k, at_k, at_k, 0.0, 0.0, 0.0)
  end subroutine add_temporal_vgm_st


  !=============================================================================
  ! set_joint_sills — supply the joint sill array for the sum-metric model.
  !   sills(n): partial sills for the joint Cst component, one per cs structure.
  !   Must be called AFTER all spatial structures have been added via add_spatial.
  !=============================================================================
  subroutine set_joint_sills_vgm_st(this, sills, n)
    class(vgm_struct_st), intent(inout) :: this
    integer,              intent(in)    :: n
    real,                 intent(in)    :: sills(n)

    if (n /= this%cs%nstruct) &
      error stop 'vgm_struct_st%set_joint_sills: length of sills must equal cs%nstruct'
    if (allocated(this%sill_st)) deallocate(this%sill_st)
    allocate(this%sill_st(n))
    this%sill_st = sills
  end subroutine set_joint_sills_vgm_st


  !=============================================================================
  ! compute_cov0 — compute C(0,0) for the diagonal of the kriging matrix.
  !   Must be called after all add_spatial, add_temporal, set_joint_sills calls.
  !=============================================================================
  subroutine compute_cov0_vgm_st(this)
    class(vgm_struct_st), intent(inout) :: this
    integer :: k

    select case (this%model)

      case (ST_MODEL_SUM_METRIC)
        !-- C(0,0) = Cs(0) + Ct(0) + sum_k{ sill_st_k * corefunc_k(0) }
        !   corefunc(0) = 1 for all shapes except variog_nug (which = 0)
        this%cov0_val = this%cs%cov0 + this%ct%cov0
        if (allocated(this%sill_st)) then
          do k = 1, this%cs%nstruct
            associate(comp => this%cs%structs(k))
              if (.not. allocated(comp%shape)) cycle
              !-- Skip nugget structures: corefunc_nug is identically 0,
              !   so the joint component at h_st=0 would also be 0.
              !   Detect nugget by vtype string to avoid needing variog_nug public.
              if (comp%shape%vtype /= 'nug') &
                this%cov0_val = this%cov0_val + this%sill_st(k)
            end associate
          end do
        end if

      case (ST_MODEL_PRODUCT_SUM)
        !-- C(0,0) = k_ps * Cs(0) * Ct(0) + Cs(0) + Ct(0)
        this%cov0_val = this%k_ps * this%cs%cov0 * this%ct%cov0 &
                      + this%cs%cov0 + this%ct%cov0

    end select
  end subroutine compute_cov0_vgm_st


  !=============================================================================
  ! is_valid_st — heuristic validation of the ST variogram model.
  !   ivar, jvar: variable indices (1-based), used to tailor cross-var warnings.
  !=============================================================================
  function is_valid_vgm_st(this, ivar, jvar) result(ok)
    class(vgm_struct_st), intent(in) :: this
    integer,              intent(in) :: ivar, jvar
    logical :: ok
    integer :: k

    ok = this%cs%is_valid() .and. this%ct%is_valid()

    if (this%model == ST_MODEL_SUM_METRIC) then
      !-- Joint sills must be set and non-negative
      if (.not. allocated(this%sill_st)) then
        write(*,'(A,I0,A,I0,A)') &
          'WARNING vgm_struct_st(',ivar,',',jvar,'): sill_st not set for sum-metric model'
        ok = .false.
      else
        do k = 1, size(this%sill_st)
          if (this%sill_st(k) < 0.0) then
            write(*,'(A,I0,A,I0,A,I0)') &
              'WARNING vgm_struct_st(',ivar,',',jvar,'): negative joint sill at structure ', k
            ok = .false.
          end if
        end do
      end if
    end if

    if (this%at <= 0.0) then
      write(*,'(A,I0,A,I0,A)') &
        'WARNING vgm_struct_st(',ivar,',',jvar,'): at must be positive'
      ok = .false.
    end if

    if (this%cov0_val <= 0.0) then
      write(*,'(A,I0,A,I0,A)') &
        'WARNING vgm_struct_st(',ivar,',',jvar,'): cov0_val not computed (call compute_cov0)'
      ok = .false.
    end if
  end function is_valid_vgm_st

end module variogram_st
