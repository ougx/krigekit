!==============================================================================
! Module: variogram
!
! Variogram / covariance model library for 3D universal cokriging.
!
! Design principles:
!
!  1. Per-structure anisotropy.
!     Each nested structure carries its own vgm_aniso descriptor (azimuth,
!     dip, plunge, range ratios).  The affine transformation matrix is
!     computed once on construction and cached.  Different structures in the
!     same composite model can therefore have independent geometries.
!
!  2. Clean separation of concerns.
!     variog        : shape of the correlation function (sph, exp, ...)
!     vgm_component : one structure = shape + sill + anisotropy + table
!     vgm_struct    : composite = nugget + array of vgm_component + public API
!
!  3. Lookup table lives on vgm_component, not on the composite.
!     Tables are built per structure, so the same fitted variog_sph object
!     can be reused in cross-variogram models without recomputation.
!
!  4. No raw pointers, no ptr_vgm wrapper type in the public API.
!
!  5. GSLIB rotation convention throughout.
!     Azimuth measured clockwise from North in the horizontal plane,
!     dip positive downward from horizontal, plunge = rake of the major axis.
!==============================================================================
module variogram

  use common,          only: pi, DEG2RAD, EPSLON
  use kriging_err,     only: kriging_error
  implicit none
  private

  !-- public surface
  public :: variog                                       ! abstract base
  public :: variog_sph, variog_exp, variog_hol           ! concrete models
  public :: variog_gau, variog_pow, variog_bsq
  public :: variog_cir, variog_lin
  public :: vgm_aniso                                    ! anisotropy descriptor
  public :: vgm_component                                ! one structure
  public :: vgm_struct                                   ! composite model
  public :: build_rotmat                                 ! utility

  integer, parameter, public :: maxvgm  = 99

  !=============================================================================
  ! Abstract base type: pure correlation function shape, no geometry.
  ! Declared before the abstract interface so the interface can import it.
  !=============================================================================
  type, abstract :: variog
    character(3) :: vtype  = '   '
    real         :: nugget = 0.0   ! per-structure nugget (usually 0)
  contains
    procedure(corefunc_if), deferred :: corefunc   ! C(rdist), rdist = h/range
    procedure                        :: tostr
  end type variog

  ! Abstract interface declared after the type so 'variog' is already visible.
  ! Forward reference to corefunc_if in the type body is permitted for
  ! deferred bindings (Fortran 2003 S4.5.4).
  abstract interface
    elemental function corefunc_if(this, rdist) result(res)
      import :: variog
      class(variog), intent(in) :: this
      real,      intent(in) :: rdist   ! dimensionless lag h / a_major
      real                  :: res     ! correlation in [0, 1]
    end function
  end interface

  !=============================================================================
  ! Concrete variogram shapes.
  ! Each type carries only its own extra parameters; range and sill live on
  ! vgm_component, keeping the shape types lightweight and reusable.
  !=============================================================================
  type, extends(variog) :: variog_nug
  contains
    procedure :: corefunc => corefunc_nug
  end type


  type, extends(variog) :: variog_sph
  contains
    procedure :: corefunc => corefunc_sph
  end type

  type, extends(variog) :: variog_exp
  contains
    procedure :: corefunc => corefunc_exp
  end type

  type, extends(variog) :: variog_hol
    ! Not positive-definite in 3D; use only for 1D/2D or in combination
    ! with a p.d. structure that dominates at short lags.
  contains
    procedure :: corefunc => corefunc_hol
  end type

  type, extends(variog) :: variog_gau
  contains
    procedure :: corefunc => corefunc_gau
  end type

  type, extends(variog) :: variog_pow
    real :: alpha = 1.5   ! 0 < alpha < 2; generalised covariance IRF-k
  contains
    procedure :: corefunc => corefunc_pow
  end type

  type, extends(variog) :: variog_bsq
    ! Bi-square (compact support): (1 - rdist^2)^2 for rdist < 1.
  contains
    procedure :: corefunc => corefunc_bsq
  end type

  type, extends(variog) :: variog_cir
  contains
    procedure :: corefunc => corefunc_cir
  end type

  type, extends(variog) :: variog_lin
  contains
    procedure :: corefunc => corefunc_lin
  end type

  !=============================================================================
  ! Anisotropy descriptor — one per structure.
  !
  ! Stores interpretable parameters (angles + range ratios) AND the
  ! pre-computed 3x3 affine transform matrix that maps a real-space lag vector
  ! to a dimensionless isotropic lag magnitude:
  !
  !   h_iso = || mat * lag_vec ||
  !
  ! where mat = diag(1/a_major, 1/a_minor1, 1/a_minor2) * R(az, dip, plunge)
  ! and the rotation R follows the GSLIB convention documented in build_aniso_mat.
  !=============================================================================
  type :: vgm_aniso
    real :: azimuth  = 0.0    ! clockwise from North in horizontal plane
    real :: dip      = 0.0    ! downward tilt of the major axis (degrees)
    real :: plunge   = 0.0    ! rotation of semi-axes around major axis
    real :: a_major  = 1.0    ! range along major axis
    real :: a_minor1 = 1.0    ! range along first semi-axis
    real :: a_minor2 = 1.0    ! range along second semi-axis (3D)

    !-- cached affine transform (set by build())
    real :: mat(3,3) = reshape([1,0,0, 0,1,0, 0,0,1], [3,3])
    logical  :: ready    = .false.
  contains
    procedure :: build => build_aniso_mat   ! compute mat from angles + ranges
    procedure :: h_iso => aniso_h           ! lag vector -> scalar isotropic h
  end type vgm_aniso

  !=============================================================================
  ! One nested structure: shape + partial sill + anisotropy + lookup table.
  !=============================================================================
  type :: vgm_component
    real                   :: sill  = 1.0
    type(vgm_aniso)            :: aniso
    class(variog), allocatable :: shape

    real, allocatable :: tab(:)      ! covariance values C(h_i), i=0..n_tab
    real, allocatable :: tab_h(:)    ! breakpoints h_i,        i=0..n_tab
    !-- Geometric spacing: h_i = h1 * ratio^i, i >= 1; h_0 = 0 (nugget point)
    !   Index formula (O(1), no search): i = floor(log(h/h1) / log_ratio) + 1
    real    :: tab_hmax     = 0.0
    real    :: tab_h1       = 0.0    ! first positive breakpoint
    real    :: tab_log_h1   = 0.0    ! log(h1)
    real    :: tab_log_ratio= 0.0    ! log(ratio)
    integer :: tab_n        = 0      ! number of intervals (breakpoints = n+1)
    logical :: tab_ready    = .false.
  contains
    procedure :: build_table => comp_build_table
    procedure :: cov_h       => comp_cov_h        ! analytic, isotropic scalar h
    procedure :: cov_lag     => comp_cov_lag       ! analytic, dx [,dy [,dz]]
    procedure :: cov_tab     => comp_cov_tab       ! table,    dx [,dy [,dz]]
    procedure :: tostr       => comp_tostr
    final     :: comp_finalise
  end type vgm_component

  !=============================================================================
  ! Composite variogram model: nugget + array of vgm_component.
  !=============================================================================
  type :: vgm_struct
    integer  :: nstruct = 0
    real     :: cov0    = 0.0
    type(vgm_component) :: structs(maxvgm)
  contains
    procedure :: reset         => struct_reset
    procedure :: add_args      => struct_add_args
    procedure :: add_comp      => struct_add_comp
    generic   :: add           => add_args, add_comp
    procedure :: build_all_tables
    procedure :: cov_h         => struct_cov_h     ! isotropic scalar h
    procedure :: cov_lag       => struct_cov_lag   ! analytic, dx [,dy [,dz]]
    procedure :: cov_tab       => struct_cov_tab   ! table, dx [,dy [,dz]]
    procedure :: tostr         => struct_tostr
    procedure :: is_valid      => struct_is_valid
  end type vgm_struct

contains

  !=============================================================================
  ! vgm_aniso
  !=============================================================================

  !-- Build the affine transformation matrix from angles and ranges.
  !
  !   GSLIB convention:
  !     azimuth  : clockwise from +Y (North), in XY plane, degrees
  !     dip      : downward rotation around the rotated X axis, degrees
  !     plunge   : rotation around the Z axis after dip, degrees
  !
  !   The rotation matrix is composed as R = Rz(plunge) * Rx(dip) * Rz(azimuth)
  !   and the full transform is mat = diag(1/a) * R so that:
  !     h_iso = || mat * lag ||
  subroutine build_aniso_mat(this)
    use rotation, only: calc_rotmat
    class(vgm_aniso), intent(inout) :: this

    this%mat   = calc_rotmat(&
      this%azimuth, this%dip, this%plunge, &
      this%a_minor1/this%a_major, this%a_minor2/this%a_major)/this%a_major
    this%ready = .true.
  end subroutine build_aniso_mat

  !-- Transform a 3D lag vector to a dimensionless isotropic distance.
  elemental function aniso_h(this, dx, dy, dz) result(h)
    class(vgm_aniso), intent(in) :: this
    real,             intent(in) :: dx, dy, dz
    real                     :: h
    real :: rx, ry, rz
    rx = this%mat(1,1)*dx + this%mat(1,2)*dy + this%mat(1,3)*dz
    ry = this%mat(2,1)*dx + this%mat(2,2)*dy + this%mat(2,3)*dz
    rz = this%mat(3,1)*dx + this%mat(3,2)*dy + this%mat(3,3)*dz
    h  = sqrt(rx*rx + ry*ry + rz*rz)
  end function aniso_h

  !-- Public convenience wrapper.
  subroutine build_rotmat(aniso)
    type(vgm_aniso), intent(inout) :: aniso
    call aniso%build()
  end subroutine build_rotmat

  !=============================================================================
  ! vgm_component
  !=============================================================================

  !-- Build the lookup table with geometrically increasing intervals.
  !
  !   Breakpoints:  h_0 = 0,  h_i = h_min * ratio^(i-1)  for i >= 1
  !   where h_min = hmax * h_min_frac  (default 1e-4 of hmax).
  !   The number of points n is auto-computed so that h_{n} >= hmax exactly.
  !
  !   This gives dense resolution near h=0 (where the variogram changes
  !   fastest) and coarser spacing near hmax (nearly flat).
  !
  !   Guidance on ratio choice (max relative interpolation error):
  !     ratio = 1.005 -> ~1850 pts, err < 3e-4
  !     ratio = 1.01  -> ~930  pts, err < 1e-3  (recommended default)
  !     ratio = 1.02  -> ~470  pts, err < 5e-3
  !
  !   hmax: dimensionless upper limit (rdist = h / a_major).
  !         Use 1.05 for compact-support models (sph, lin, cir, bsq),
  !         3.5 for infinite-support models (exp, gau).
  !
  !   Index lookup is O(1) via: i = floor(log(h/h_min) / log(ratio)) + 1
  subroutine comp_build_table(this, hmax, ratio)
    class(vgm_component), intent(inout) :: this
    real, intent(in) :: hmax
    real, intent(in) :: ratio          ! geometric growth factor, e.g. 1.01

    integer :: i, n_tab
    real    :: h_min, log_r

    if (.not. allocated(this%shape)) then
      call kriging_error('vgm_component%build_table', 'shape not allocated')
      return
    end if
    if (ratio <= 1.0) then
      call kriging_error('vgm_component%build_table', 'ratio must be > 1.0')
      return
    end if

    !-- h_min: smallest positive breakpoint (1e-4 of hmax)
    h_min = hmax * 1.0e-4
    log_r = log(ratio)

    !-- auto-compute n_tab to cover [h_min, hmax]
    n_tab = int(log(hmax / h_min) / log_r) + 2

    if (allocated(this%tab))   deallocate(this%tab)
    if (allocated(this%tab_h)) deallocate(this%tab_h)
    allocate(this%tab(0:n_tab), this%tab_h(0:n_tab))

    !-- h=0 entry: sill + nugget
    this%tab_h(0) = 0.0
    this%tab(0) = this%sill + this%shape%nugget

    !-- positive breakpoints
    do i = 1, n_tab
      this%tab_h(i) = h_min * ratio**(i-1)
      this%tab(i)   = this%sill * this%shape%corefunc(this%tab_h(i))
    end do
    !-- clamp last breakpoint exactly to hmax (avoids fp drift)
    this%tab_h(n_tab) = hmax
    this%tab(n_tab)   = this%sill * this%shape%corefunc(hmax)

    this%tab_n         = n_tab
    this%tab_hmax      = hmax
    this%tab_h1        = h_min
    this%tab_log_h1    = log(h_min)
    this%tab_log_ratio = log_r
    this%tab_ready = .true.
  end subroutine comp_build_table

  !-- Analytic evaluation at a pre-transformed isotropic lag h.
  function comp_cov_h(this, h) result(res)
    class(vgm_component), intent(in) :: this
    real,             intent(in) :: h
    real                         :: res
    real, parameter :: eps = tiny(1.0) * 1.0e3
    if (h > eps) then
      res = this%sill * this%shape%corefunc(h)
    else
      res = this%sill + this%shape%nugget
    end if
  end function comp_cov_h

  !-- Analytic evaluation at a lag vector
  function comp_cov_lag(this, dx, dy, dz) result(res)
    class(vgm_component), intent(in) :: this
    real,               intent(in) :: dx, dy, dz
    real                         :: res
    res = this%cov_h(this%aniso%h_iso(dx, dy, dz))
  end function comp_cov_lag

  !-- Fast table path.
  !   Index into the geometric table in O(1) via the log formula, then
  !   linearly interpolate between the two surrounding breakpoints.
  function comp_cov_tab(this, dx, dy, dz) result(res)
    class(vgm_component), intent(in) :: this
    real,               intent(in) :: dx, dy, dz
    real                         :: res
    real    :: h, frac
    integer :: i

    h = this%aniso%h_iso(dx, dy, dz)

    if (.not. this%tab_ready) then
      res = this%cov_h(h)
      return
    end if

    !-- Beyond table range: covariance is zero
    if (h >= this%tab_hmax) then
      res = 0.0
      return
    end if

    !-- Below first positive breakpoint: use h=0 entry (nugget+sill)
    if (h < this%tab_h1) then
      res = this%tab(0)
      return
    end if

    !-- O(1) index: i such that tab_h(i) <= h < tab_h(i+1)
    i = int( (log(h) - this%tab_log_h1) / this%tab_log_ratio ) + 1
    i = max(1, min(i, this%tab_n - 1))

    !-- Linear interpolation between breakpoints tab_h(i) and tab_h(i+1)
    frac = (h - this%tab_h(i)) / (this%tab_h(i+1) - this%tab_h(i))
    res  = this%tab(i) + frac * (this%tab(i+1) - this%tab(i))
  end function comp_cov_tab

  function comp_tostr(this) result(s)
    class(vgm_component), intent(in) :: this
    character(:), allocatable         :: s
    character(256) :: buf
    if (allocated(this%shape)) then
      write(buf,'(A3,"  sill=",G13.6, &
               &"  az=",F7.2,"  dip=",F7.2,"  pl=",F7.2, &
               &"  a=",3(G13.6,1X))') &
        this%shape%vtype, this%sill,                          &
        this%aniso%azimuth, this%aniso%dip, this%aniso%plunge, &
        this%aniso%a_major, this%aniso%a_minor1, this%aniso%a_minor2
    else
      buf = '(unset component)'
    end if
    s = trim(buf)
  end function comp_tostr

  subroutine comp_finalise(this)
    type(vgm_component), intent(inout) :: this
    if (allocated(this%tab))   deallocate(this%tab)
    if (allocated(this%tab_h)) deallocate(this%tab_h)
    if (allocated(this%shape)) deallocate(this%shape)
  end subroutine comp_finalise

  !=============================================================================
  ! vgm_struct
  !=============================================================================

  subroutine struct_reset(this)
    class(vgm_struct), intent(inout) :: this
    this%nstruct = 0
    this%cov0    = 0.0
  end subroutine struct_reset

  subroutine struct_add_comp(this, comp)
    class(vgm_struct),   intent(inout) :: this
    type(vgm_component), intent(in)    :: comp
    if (this%nstruct >= maxvgm) then
      call kriging_error('vgm_struct%add', 'exceeded maxvgm nested structures')
      return
    end if
    if (.not. allocated(comp%shape)) then
      call kriging_error('vgm_struct%add', 'component shape not allocated')
      return
    end if
    if (.not. comp%aniso%ready) then
      call kriging_error('vgm_struct%add', 'aniso matrix not built; call aniso%build()')
      return
    end if
    this%nstruct = this%nstruct + 1
    this%structs(this%nstruct) = comp
    this%cov0 = this%cov0 + comp%sill + comp%shape%nugget
  end subroutine struct_add_comp

  subroutine struct_add_args(this, vtype, nugget, sill, &
                              a_major, a_minor1, a_minor2, azimuth, dip, plunge)
    class(vgm_struct), intent(inout) :: this
    character(*),      intent(in)    :: vtype
    real,              intent(in)    :: nugget, sill
    real,              intent(in)    :: a_major, a_minor1, a_minor2
    real,              intent(in)    :: azimuth, dip, plunge
    if (this%nstruct >= maxvgm) then
      call kriging_error('vgm_struct%add', 'exceeded maxvgm nested structures')
      return
    end if
    this%nstruct = this%nstruct + 1
    associate (cc => this%structs(this%nstruct))
      if (allocated(cc%shape)) deallocate(cc%shape)
      cc%sill            = sill
      cc%aniso%azimuth   = azimuth
      cc%aniso%dip       = dip
      cc%aniso%plunge    = plunge
      cc%aniso%a_major   = a_major
      cc%aniso%a_minor1  = a_minor1
      cc%aniso%a_minor2  = a_minor2
      call cc%aniso%build()
      select case (trim(vtype))
        case('nug'); allocate(variog_nug :: cc%shape)
        case('sph'); allocate(variog_sph :: cc%shape)
        case('exp'); allocate(variog_exp :: cc%shape)
        case('hol'); allocate(variog_hol :: cc%shape)
        case('gau'); allocate(variog_gau :: cc%shape)
        case('pow'); allocate(variog_pow :: cc%shape)
        case('bsq'); allocate(variog_bsq :: cc%shape)
        case('cir'); allocate(variog_cir :: cc%shape)
        case('lin'); allocate(variog_lin :: cc%shape)
        case default
          this%nstruct = this%nstruct - 1
          call kriging_error('vgm_struct%add', 'unknown variogram type: '//trim(vtype))
          return
      end select
      cc%shape%vtype = trim(vtype)
      this%cov0 = this%cov0 + sill + nugget
    end associate
  end subroutine struct_add_args

  !-- Build tables for all structures.
  !-- Build tables for all structures.
  !   ratio:       geometric growth factor (e.g. 1.01).  n is auto-computed.
  !   hmax_factor: table covers [0, hmax_factor] in dimensionless rdist.
  !                Use ~1.05 for compact-support, ~3.5 for infinite-support.
  subroutine build_all_tables(this, ratio, hmax_factor)
    class(vgm_struct), intent(inout) :: this
    real,              intent(in)    :: ratio
    real, optional, intent(in)   :: hmax_factor
    real :: factor
    integer  :: iv
    factor = 3.5
    if (present(hmax_factor)) factor = hmax_factor
    do iv = 1, this%nstruct
      call this%structs(iv)%build_table(hmax=factor, ratio=ratio)
    end do
  end subroutine build_all_tables

  !-- Composite covariance at a scalar isotropic lag.
  !   Only correct when all structures are isotropic (all rotmat = I).
  !   Use cov_lag or cov_tab for the general anisotropic case.
  function struct_cov_h(this, h) result(res)
    class(vgm_struct), intent(in) :: this
    real,          intent(in) :: h
    real                      :: res
    integer :: iv
    res = 0.0
    do iv = 1, this%nstruct
      res = res + this%structs(iv)%cov_h(h)
    end do
  end function struct_cov_h

  !-- Composite covariance at a 3D lag vector — primary production interface.
  !   Each structure applies its own independent anisotropy transform.
  function struct_cov_lag(this, lag) result(res)
    class(vgm_struct), intent(in) :: this
    real,          intent(in) :: lag(3)
    real                      :: res
    integer :: iv
    res = 0.0
    do iv = 1, this%nstruct
      res = res + this%structs(iv)%cov_lag(lag(1), lag(2), lag(3))
    end do
  end function struct_cov_lag

  !-- Fast composite covariance via per-structure tables.
  !   Use this inside the cokriging assembly loop.
  function struct_cov_tab(this, lag) result(res)
    class(vgm_struct), intent(in) :: this
    real,          intent(in) :: lag(3)
    real                      :: res
    integer :: iv
    res = 0.0
    do iv = 1, this%nstruct
      res = res + this%structs(iv)%cov_tab(lag(1), lag(2), lag(3))
    end do
  end function struct_cov_tab

  function struct_tostr(this) result(s)
    class(vgm_struct), intent(in) :: this
    character(:), allocatable      :: s
    character(64) :: buf
    integer :: iv
    write(buf,'("  Number of structures = ",I0)') this%nstruct
    s = trim(buf)
    do iv = 1, this%nstruct
      s = s // new_line('a') // '    ' // this%structs(iv)%tostr()
    end do
  end function struct_tostr

  !-- Heuristic positive-definiteness check.
  function struct_is_valid(this) result(ok)
    class(vgm_struct), intent(in) :: this
    logical :: ok
    integer :: iv
    ok = .true.


    do iv = 1, this%nstruct
      associate(c => this%structs(iv))
        if (.not. allocated(c%shape)) then
          write(*,'(A,I0,A)') &
            'WARNING vgm_struct: structure ', iv, ': no shape allocated'
          ok = .false.
        end if
        if (c%shape%nugget < 0.0) then
          write(*,'(A,I0,A)') &
            'WARNING vgm_struct: structure ', iv, ': negative nugget'
          ok = .false.
        end if
        if (c%sill < 0.0) then
          write(*,'(A,I0,A)') &
            'WARNING vgm_struct: structure ', iv, ': negative sill'
          ok = .false.
        end if
        if (.not. c%aniso%ready) then
          write(*,'(A,I0,A)') &
            'WARNING vgm_struct: structure ', iv, &
            ': aniso matrix not built (call aniso%build())'
          ok = .false.
        end if
        if (c%aniso%a_major  <= 0.0)  then
          write(*,'(A,I0,A)') &
            'WARNING vgm_struct: structure ', iv, ': non-positive major range'
          ok = .false.
        end if
        if (c%aniso%a_minor1 <= 0.0)  then
          write(*,'(A,I0,A)') &
            'WARNING vgm_struct: structure ', iv, ': non-positive minor range 1'
          ok = .false.
        end if
        if (c%aniso%a_minor2 <= 0.0) then
          write(*,'(A,I0,A)') &
            'WARNING vgm_struct: structure ', iv, ': non-positive minor range 2'
          ok = .false.
        end if
        !-- Type-guard: warn if hole-effect used in 3D context
        if (allocated(c%shape)) then
          select type(sh => c%shape)
          type is (variog_hol)
            write(*,'(A,I0,A)') &
              'WARNING vgm_struct: structure ', iv, &
              ': hole-effect is not p.d. in 3D'
          end select
        end if
      end associate
    end do
  end function struct_is_valid

  !=============================================================================
  ! Core correlation functions
  ! All elemental; rdist = h / a_major (dimensionless, post-aniso transform).
  !=============================================================================

  elemental function corefunc_nug(this, rdist) result(res)
    class(variog_nug), intent(in) :: this
    real,          intent(in) :: rdist
    real                      :: res
    res = 0.0
  end function

  elemental function corefunc_sph(this, rdist) result(res)
    class(variog_sph), intent(in) :: this
    real,          intent(in) :: rdist
    real                      :: res
    if (rdist < 1.0) then
      res = 1.0 - 1.5*rdist + 0.5*rdist**3
    else
      res = 0.0
    end if
  end function

  elemental function corefunc_exp(this, rdist) result(res)
    class(variog_exp), intent(in) :: this
    real,          intent(in) :: rdist
    real                      :: res
    res = exp(-3.0 * rdist)           ! C(a_major) ~= 0.05
  end function

  elemental function corefunc_hol(this, rdist) result(res)
    class(variog_hol), intent(in) :: this
    real,          intent(in) :: rdist
    real                      :: res
    res = cos(pi * rdist)                ! first zero at rdist = 0.5
  end function

  elemental function corefunc_gau(this, rdist) result(res)
    class(variog_gau), intent(in) :: this
    real,          intent(in) :: rdist
    real                      :: res
    res = exp(-3.0625 * rdist**2)     ! -49/16; C(a_major) ~= 0.047
  end function

  elemental function corefunc_pow(this, rdist) result(res)
    class(variog_pow), intent(in) :: this
    real,          intent(in) :: rdist
    real                      :: res
    res = merge(1.0 - rdist**this%alpha, 0.0, rdist < 1.0) ! generalised covariance K(h) = -h^a
  end function

  elemental function corefunc_bsq(this, rdist) result(res)
    class(variog_bsq), intent(in) :: this
    real,          intent(in) :: rdist
    real                      :: res
    res = merge((1.0 - rdist**2)**2, 0.0, rdist < 1.0)
  end function

  elemental function corefunc_cir(this, rdist) result(res)
    class(variog_cir), intent(in) :: this
    real,          intent(in) :: rdist
    real                      :: res
    res = merge(1.0 - (2.0*rdist*sqrt(1.0 - rdist**2) + 2.0*asin(rdist)) / pi, 0.0, rdist < 1.0)
  end function

  elemental function corefunc_lin(this, rdist) result(res)
    class(variog_lin), intent(in) :: this
    real,          intent(in) :: rdist
    real                      :: res
    res = merge(1.0 - rdist, 0.0, rdist < 1.0)
  end function

  !=============================================================================
  ! variog base: tostr
  !=============================================================================
  function tostr(this) result(s)
    class(variog), intent(in)  :: this
    character(:), allocatable  :: s
    character(64) :: buf
    write(buf,'(A3,"  nugget=",G13.6)') this%vtype, this%nugget
    s = trim(buf)
  end function tostr

end module variogram


!==============================================================================
! Usage example
!==============================================================================
!
!   use variogram
!
!   type(vgm_struct)    :: vg
!   type(vgm_component) :: c1, c2
!
!   !-- Structure 1: isotropic spherical, sill = 0.70
!   allocate(variog_sph :: c1%shape)
!   c1%shape%vtype   = 'sph'
!   c1%sill          = 0.70
!   c1%aniso%a_major  = 1200.0    ! isotropic: all three ranges equal
!   c1%aniso%a_minor1 = 1200.0
!   c1%aniso%a_minor2 = 1200.0
!   call c1%aniso%build()            ! computes the affine transform matrix
!
!   !-- Structure 2: anisotropic exponential, sill = 0.25
!   !   Major axis: 5000 m azimuth=30 deg, dip=10 deg
!   !   Semi-axes:  2000 m (horizontal), 300 m (vertical)
!   allocate(variog_exp :: c2%shape)
!   c2%shape%vtype   = 'exp'
!   c2%sill          = 0.25
!   c2%aniso%a_major  = 5000.0
!   c2%aniso%a_minor1 = 2000.0
!   c2%aniso%a_minor2 =  300.0
!   c2%aniso%azimuth  =  30.0
!   c2%aniso%dip      =  10.0
!   c2%aniso%plunge   =   0.0
!   call c2%aniso%build()
!
!   !-- Assemble composite
!   vg%nugget = 0.05
!   call vg%add(c1)
!   call vg%add(c2)
!
!   !-- Build tables (3.5 x a_major per structure, 10000 points each)
!   call vg%build_all_tables(ratio=1.01, hmax_factor=3.5)
!
!   if (.not. vg%is_valid()) error stop 'invalid variogram model'
!
!   !-- In the cokriging assembly loop (lag = x_i - x_j as a 3-vector):
!   C(i,j) = vg%cov_tab(lag)        ! fast: table + per-structure aniso
!   C(i,j) = vg%cov_lag(lag)        ! exact: analytic + per-structure aniso
!   C(i,i) = C(i,i) + vg%nugget     ! nugget on diagonal only
!
!   !-- Cross-variogram (linear model of coregionalization):
!   !   Build a separate vg12 with the same ranges/aniso as vg but
!   !   c1%sill = b12_1, c2%sill = b12_2.
!   !   LMC constraint per structure k: b12_k^2 <= b11_k * b22_k
!
!==============================================================================
