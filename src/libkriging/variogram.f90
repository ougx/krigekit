!==============================================================================
! Module: variogram
!
! Variogram / covariance model library for 3D/2D universal cokriging.
!
! Design principles:
!
!  1. No polymorphism in the hot path.
!     The covariance shape is stored as an integer tag (vtype_id) in
!     vgm_component.  corefunc_fn(vtype_id, rdist) is a pure
!     elemental function using a select case — the compiler inlines it,
!     builds a jump table for the switch, and can SIMD-vectorise the call.
!     There are no abstract types, no vtable lookups, no heap allocations
!     for shape objects.
!
!  2. Per-structure anisotropy.
!     vgm_aniso holds the cached affine transform matrix (built once).
!     ndim = 2 skips the z-row in aniso_h (4 mul+1 sqrt vs 9 mul+1 sqrt).
!
!  3. Two-level piecewise-uniform lookup table.
!
!     Level A — build_all_tables():  one table per vgm_component.
!       Works for any combination of anisotropies.
!       cov_tab cost: nstruct × (1 aniso_h + 1 table lookup).
!
!     Level B — build_struct_table():  one composite table for the whole struct.
!       REQUIRES all non-nugget structures to share the same mat matrix.
!       cov_tab cost: 1 aniso_h + 1 table lookup (independent of nstruct).
!       cov_tab() auto-dispatches to Level B when struct_tab_ready = .true.
!
!     Both levels use piecewise-uniform zones (no log in the hot path):
!       h_bounds = [0.0, 0.1, 0.5, 3.5]   → 3 zones + model-specific tail
!       dh       = [1e-5, 1e-4, 1e-3]
!       entries: 10000+4000+3000 = 17001, ~66 KB, max err < 5e-7
!==============================================================================
module variogram

  use common,          only: pi, DEG2RAD, EPSLON
  use kriging_err,     only: kriging_error, kriging_failed, kriging_clear_error
  use vgm_func,        only: VGM_NUG, VGM_HOL, corefunc_fn, &
                             corefunc_has_analytic_tail, vtype_from_str
  implicit none
  private

  !-- Public API
  public :: vgm_aniso                                    ! anisotropy descriptor
  public :: vgm_component                                ! one structure
  public :: vgm_struct                                   ! composite model
  public :: build_rotmat                                 ! utility

  integer, parameter, public :: maxvgm  = 99

  !-- Tolerance for anisotropy compatibility check in build_struct_table
  real, parameter :: MAT_TOL = 1.0e-6
  !-- Tolerance (degrees) for rotation-angle check when product members have different ranges
  real, parameter :: ANG_TOL = 1.0e-4



  !=============================================================================
  ! vgm_aniso — anisotropy descriptor.
  ! ndim = 1: range only
  ! ndim = 2: azimuth only; 4 mul + 1 sqrt per aniso_h call.
  ! ndim = 3: full GSLIB rotation; 9 mul + 1 sqrt per aniso_h call.
  !=============================================================================
  type :: vgm_aniso
    integer :: ndim  = 3
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
  ! vgm_component — one nested structure.
  !
  ! Fields replacing the old class(variog), allocatable :: shape:
  !   vtype_id  : integer tag (VGM_NUG, VGM_SPH, ...) — used in corefunc_fn
  !   vtype     : 3-character string ('nug','sph',...) — for display only
  !   nugget    : per-structure nugget (usually 0; adds to C(0) but not C(h>0))
  !
  ! The lookup table stores tab(h) = sill * corefunc_fn(vtype_id, h).
  !=============================================================================
  type :: vgm_component
    real           :: sill      = 1.0
    real           :: nugget    = 0.0
    integer        :: vtype_id  = VGM_NUG
    character(3)   :: vtype     = 'nug'
    type(vgm_aniso) :: aniso

    !-- Product-group flag: when .true. this component multiplies the previous one.
    !   False (default) means additive nesting (the standard geostatistics convention).
    logical :: is_product = .false.

    !-- Piecewise-uniform lookup table
    real,    allocatable :: tab(:)
    real,    allocatable :: tab_zone_bounds(:)  ! [0:nzones]
    real,    allocatable :: tab_zone_inv_dh(:)  ! [nzones]
    integer, allocatable :: tab_zone_offset(:)  ! [nzones]
    integer, allocatable :: tab_zone_n(:)        ! [nzones]
    real    :: tab_hmax   = 0.0
    integer :: tab_n      = 0
    integer :: tab_nzones = 0
    logical :: tab_ready  = .false.
    logical :: tab_analytic_tail = .false.

  contains
    procedure :: build_table => comp_build_table
    procedure :: cov_h       => comp_cov_h
    procedure :: cov_lag     => comp_cov_lag
    procedure :: cov_tab     => comp_cov_tab
    procedure :: cov_tab_h   => comp_cov_tab_h
    procedure :: tostr       => comp_tostr
    final     :: comp_finalise
  end type vgm_component


  !=============================================================================
  ! vgm_struct — composite model.
  !=============================================================================
  type :: vgm_struct
    integer :: ndim    = 0
    integer :: nstruct = 0
    real    :: cov0    = 0.0
    type(vgm_component) :: structs(maxvgm)

    !-- Level-B composite struct table
    type(vgm_aniso)      :: struct_aniso
    real,    allocatable :: struct_tab(:)
    real,    allocatable :: struct_zone_bounds(:)
    real,    allocatable :: struct_zone_inv_dh(:)
    integer, allocatable :: struct_zone_offset(:)
    integer, allocatable :: struct_zone_n(:)
    real    :: struct_hmax    = 0.0
    integer :: struct_n       = 0
    integer :: struct_nzones  = 0
    logical :: struct_tab_ready = .false.
    logical :: struct_analytic_tail = .false.

  contains
    procedure :: reset         => struct_reset
    procedure :: reset_model   => struct_reset
    procedure :: add_args      => struct_add_args
    procedure :: add_comp      => struct_add_comp
    generic   :: add           => add_args, add_comp
    procedure :: build_table   => struct_build_table  ! entry to build table, branch to build_all_tables or build_composite_table
    procedure :: build_all_tables                     ! Level A (per-component)
    procedure :: build_composite_table                ! Level B (whole-struct composite)
    procedure :: cov_h         => struct_cov_h        ! isotropic scalar h
    procedure :: cov_lag       => struct_cov_lag      ! analytic, dx [,dy [,dz]]
    procedure :: cov_tab       => struct_cov_tab      ! table, dx [,dy [,dz]]
    procedure :: tostr         => struct_tostr
    procedure :: is_valid      => struct_is_valid
    procedure :: set_ndim      => struct_set_ndim
    procedure, private :: cov_struct_tab_h
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
      this%mat = calc_rotmat( &
        this%azimuth, this%dip, this%plunge, &
        this%a_minor1/this%a_major, this%a_minor2/this%a_major) / this%a_major
    this%ready = .true.
  end subroutine build_aniso_mat

  elemental function aniso_h(this, dx, dy, dz) result(h)
    class(vgm_aniso), intent(in) :: this
    real,             intent(in) :: dx, dy, dz
    real :: h, rx, ry, rz
    if (this%ndim == 1) then
      h  = abs(dx * this%mat(1,1))
    else if (this%ndim == 2) then
      rx = this%mat(1,1)*dx + this%mat(1,2)*dy
      ry = this%mat(2,1)*dx + this%mat(2,2)*dy
      h  = sqrt(rx*rx + ry*ry)
    else
      rx = this%mat(1,1)*dx + this%mat(1,2)*dy + this%mat(1,3)*dz
      ry = this%mat(2,1)*dx + this%mat(2,2)*dy + this%mat(2,3)*dz
      rz = this%mat(3,1)*dx + this%mat(3,2)*dy + this%mat(3,3)*dz
      h  = sqrt(rx*rx + ry*ry + rz*rz)
    end if
  end function aniso_h

  !-- Public convenience wrapper.
  subroutine build_rotmat(aniso)
    type(vgm_aniso), intent(inout) :: aniso
    call aniso%build()
  end subroutine build_rotmat


  !=============================================================================
  ! Private zone-building helper
  !=============================================================================

  subroutine alloc_zones(h_bounds, dh, hmax, &
      nzones, ntab_total, n_zone, zone_bounds, zone_inv_dh, zone_offset, zone_n)
    real,    intent(in)  :: h_bounds(:), hmax
    real,    intent(in)  :: dh(:)
    integer, intent(out) :: nzones, ntab_total
    integer, allocatable, intent(out) :: n_zone(:)
    real,    allocatable, intent(out) :: zone_bounds(:)
    real,    allocatable, intent(out) :: zone_inv_dh(:)
    integer, allocatable, intent(out) :: zone_offset(:)
    integer, allocatable, intent(out) :: zone_n(:)

    integer :: k, offset

    nzones = size(dh)
    allocate(n_zone(nzones))
    do k = 1, nzones
      n_zone(k) = max(1, nint((h_bounds(k+1) - h_bounds(k)) / dh(k)))
    end do
    ntab_total = sum(n_zone)

    allocate(zone_bounds(0:nzones))
    allocate(zone_inv_dh(nzones))
    allocate(zone_offset(nzones))
    allocate(zone_n(nzones))

    zone_bounds = h_bounds
    offset = 0
    do k = 1, nzones
      zone_offset(k)  = offset
      zone_n(k)       = n_zone(k)
      zone_inv_dh(k)  = real(n_zone(k)) / (h_bounds(k+1) - h_bounds(k))
      offset          = offset + n_zone(k)
    end do
  end subroutine alloc_zones


  !=============================================================================
  ! vgm_component
  !=============================================================================

  subroutine comp_build_table(this, hmax, n_tab, h_bounds, dh)
    class(vgm_component), intent(inout) :: this
    real,    intent(in)           :: hmax
    integer, intent(in), optional :: n_tab
    real,    intent(in), optional :: h_bounds(:), dh(:)

    integer              :: i, k, offset, ntab_total, nzones_local
    integer, allocatable :: n_zone(:)
    real,    allocatable :: z_bounds(:), z_inv_dh(:), h_b(:), d_h(:)
    integer, allocatable :: z_offset(:), z_n(:)
    real :: step_k, h_lo, rdist

    if (hmax <= 0.0) then
      call kriging_error('vgm_component%comp_build_table', 'hmax must be positive')
      return
    end if

    if (allocated(this%tab))             deallocate(this%tab)
    if (allocated(this%tab_zone_bounds)) deallocate(this%tab_zone_bounds)
    if (allocated(this%tab_zone_inv_dh)) deallocate(this%tab_zone_inv_dh)
    if (allocated(this%tab_zone_offset)) deallocate(this%tab_zone_offset)
    if (allocated(this%tab_zone_n))      deallocate(this%tab_zone_n)

    if (present(h_bounds) .and. present(dh)) then
      if (size(h_bounds) /= size(dh)+1) then
        call kriging_error ('vgm_component%comp_build_table', 'size(h_bounds) must be size(dh)+1')
        return
      end if
      allocate(h_b, source=h_bounds)
      allocate(d_h, source=dh)
    else
      allocate(h_b(2));  h_b = [0.0, hmax]
      allocate(d_h(1));  d_h(1) = hmax / real(merge(n_tab, 10000, present(n_tab)))
    end if

    call alloc_zones(h_b, d_h, hmax, &
      nzones_local, ntab_total, n_zone, z_bounds, z_inv_dh, z_offset, z_n)

    allocate(this%tab(0:ntab_total))
    this%tab(0) = this%sill + this%nugget   ! C(0)

    offset = 0
    do k = 1, nzones_local
      h_lo   = z_bounds(k-1)
      step_k = 1.0 / z_inv_dh(k)
      do i = 1, n_zone(k)
        rdist = h_lo + real(i) * step_k
        this%tab(offset+i) = this%sill * corefunc_fn(this%vtype_id, rdist)
      end do
      offset = offset + n_zone(k)
    end do

    this%tab_nzones = nzones_local
    this%tab_n      = ntab_total
    this%tab_hmax   = hmax
    this%tab_analytic_tail = corefunc_has_analytic_tail(this%vtype_id)
    call move_alloc(z_bounds, this%tab_zone_bounds)
    call move_alloc(z_inv_dh, this%tab_zone_inv_dh)
    call move_alloc(z_offset, this%tab_zone_offset)
    call move_alloc(z_n,      this%tab_zone_n)
    this%tab_ready  = .true.
  end subroutine comp_build_table


  function comp_cov_h(this, h) result(res)
    class(vgm_component), intent(in) :: this
    real,                 intent(in) :: h
    real :: res
    real, parameter :: eps = tiny(1.0) * 1.0e3
    if (h > eps) then
      res = this%sill * corefunc_fn(this%vtype_id, h)
    else
      res = this%sill + this%nugget
    end if
  end function comp_cov_h

  function comp_cov_lag(this, dx, dy, dz) result(res)
    class(vgm_component), intent(in) :: this
    real,                 intent(in) :: dx, dy, dz
    real :: res
    res = this%cov_h(this%aniso%h_iso(dx, dy, dz))
  end function comp_cov_lag

  function comp_cov_tab(this, dx, dy, dz) result(res)
    class(vgm_component), intent(in) :: this
    real,                 intent(in) :: dx, dy, dz
    real :: res
    res = this%cov_tab_h(this%aniso%h_iso(dx, dy, dz))
  end function comp_cov_tab

  function comp_cov_tab_h(this, h) result(res)
    class(vgm_component), intent(in) :: this
    real,                 intent(in) :: h
    real :: res, fi
    integer :: k, i
    if (.not. this%tab_ready) then; res = this%cov_h(h); return; end if
    if (h >= this%tab_hmax) then
      if (this%tab_analytic_tail) then
        res = this%cov_h(h)
      else
        res = 0.0
      end if
      return
    end if
    if (h <= 0.0)             then; res = this%tab(0);    return; end if
    do k = 1, this%tab_nzones
      if (h < this%tab_zone_bounds(k)) then
        fi = (h - this%tab_zone_bounds(k-1)) * this%tab_zone_inv_dh(k)
        i  = this%tab_zone_offset(k) + min(int(fi), this%tab_zone_n(k)-1)
        fi = fi - real(int(fi))
        res = this%tab(i) + fi*(this%tab(i+1) - this%tab(i))
        return
      end if
    end do
    if (this%tab_analytic_tail) then
      res = this%cov_h(h)
    else
      res = 0.0
    end if
  end function comp_cov_tab_h

  function comp_tostr(this) result(s)
    class(vgm_component), intent(in) :: this
    character(:), allocatable :: s
    character(256) :: buf
    write(buf,'(A5,"  sill=",G13.6,"  nug=",G13.6, &
             &"  az=",F7.2,"  dip=",F7.2,"  pl=",F7.2, &
             &"  a=",3(G13.6,1X))') &
      merge('x '//this%vtype, '  '//this%vtype, this%is_product), &
      this%sill, this%nugget, &
      this%aniso%azimuth, this%aniso%dip, this%aniso%plunge, &
      this%aniso%a_major, this%aniso%a_minor1, this%aniso%a_minor2
    s = trim(buf)
  end function comp_tostr

  subroutine comp_finalise(this)
    type(vgm_component), intent(inout) :: this
    if (allocated(this%tab))             deallocate(this%tab)
    if (allocated(this%tab_zone_bounds)) deallocate(this%tab_zone_bounds)
    if (allocated(this%tab_zone_inv_dh)) deallocate(this%tab_zone_inv_dh)
    if (allocated(this%tab_zone_offset)) deallocate(this%tab_zone_offset)
    if (allocated(this%tab_zone_n))      deallocate(this%tab_zone_n)
  end subroutine comp_finalise


  !=============================================================================
  ! vgm_struct
  !=============================================================================

  subroutine struct_reset(this)
    class(vgm_struct), intent(inout) :: this
    this%nstruct = 0
    this%cov0    = 0.0
    this%struct_tab_ready = .false.
  end subroutine struct_reset

  subroutine struct_add_comp(this, comp)
    class(vgm_struct),   intent(inout) :: this
    type(vgm_component), intent(in)    :: comp
    if (this%nstruct >= maxvgm) then
      call kriging_error('vgm_struct%struct_add_comp', 'exceeded maxvgm nested structures')
      return
    end if
    if (comp%vtype_id < 0) then
      call kriging_error('vgm_struct%struct_add_comp', 'invalid vgm_id')
      return
    end if
    if (.not. comp%aniso%ready) then
      call kriging_error('vgm_struct%struct_add_comp', 'aniso matrix not built; call aniso%build()')
      return
    end if
    if (comp%is_product .and. this%nstruct == 0) then
      call kriging_error('vgm_struct%struct_add_comp', &
        'first structure cannot be a product member (is_product must be .false.)')
      return
    end if
    this%nstruct = this%nstruct + 1
    this%structs(this%nstruct) = comp
    call struct_recompute_cov0(this)
  end subroutine struct_add_comp

  subroutine struct_add_args(this, vtype, nugget, sill, &
                              a_major, a_minor1, a_minor2, azimuth, dip, plunge, product)
    class(vgm_struct), intent(inout) :: this
    character(*),      intent(in)    :: vtype
    real,              intent(in)    :: nugget, sill
    real,              intent(in)    :: a_major, a_minor1, a_minor2
    real,              intent(in)    :: azimuth, dip, plunge
    logical, optional, intent(in)    :: product   ! .true. = multiply with preceding structure
    integer :: id
    if (this%nstruct >= maxvgm) then
      call kriging_error('vgm_struct%add', 'exceeded maxvgm nested structures')
      return
    end if
    id = vtype_from_str(vtype)
    if (id < 0) then
      call kriging_error('vgm_struct%struct_add_comp', 'Unknown variogram type: '//trim(vtype))
      return
    end if
    if (present(product) .and. product .and. this%nstruct == 0) then
      call kriging_error('vgm_struct%add', &
        'first structure cannot be a product member (product=.true. requires a preceding structure)')
      return
    end if
    this%nstruct = this%nstruct + 1
    associate(cc => this%structs(this%nstruct))
      cc%sill            = sill
      cc%nugget          = nugget
      cc%vtype_id        = id
      cc%vtype           = vtype(1:min(3,len_trim(vtype)))
      cc%aniso%ndim      = this%ndim
      cc%aniso%azimuth   = azimuth;  cc%aniso%dip      = dip
      cc%aniso%plunge    = plunge
      cc%aniso%a_major   = a_major;  cc%aniso%a_minor1 = a_minor1
      cc%aniso%a_minor2  = a_minor2
      cc%is_product      = .false.
      if (present(product)) cc%is_product = product
      call cc%aniso%build()
    end associate
    call struct_recompute_cov0(this)
  end subroutine struct_add_args

  subroutine struct_set_ndim(self, ndim)
    class(vgm_struct), intent(inout) :: self
    integer,           intent(in)    :: ndim
    integer :: iv
    self%ndim = ndim
    do iv = 1, self%nstruct
      self%structs(iv)%aniso%ndim = ndim
    end do
  end subroutine struct_set_ndim

  !-- Public entry point: try Level B first; fall back to Level A if the
  !   non-nugget structures have incompatible anisotropy matrices.
  subroutine struct_build_table(this, n_tab, hmax_factor, h_bounds, dh)
    class(vgm_struct), intent(inout) :: this
    integer, intent(in), optional :: n_tab
    real,    intent(in), optional :: hmax_factor, h_bounds(:), dh(:)

    integer :: iv, ref_idx
    !-- Find first non-nugget structure; use its mat as the reference.
    ref_idx = 0
    do iv = 1, this%nstruct
      if (this%structs(iv)%vtype_id == VGM_NUG) cycle
      if (ref_idx == 0) then
        ref_idx = iv
        this%struct_aniso = this%structs(iv)%aniso
      else
        associate(cc => this%structs(iv), ref => this%struct_aniso)
          if (cc%is_product) then
            ! Product members may have a different a_major (independent decay / period).
            ! Only rotation must match — the composite table loop adjusts rdist per component.
            if (abs(cc%aniso%azimuth - ref%azimuth) > ANG_TOL .or. &
                abs(cc%aniso%dip     - ref%dip    ) > ANG_TOL .or. &
                abs(cc%aniso%plunge  - ref%plunge  ) > ANG_TOL) then
              call this%build_all_tables(n_tab, hmax_factor, h_bounds, dh)
              return
            end if
          else
            if (maxval(abs(cc%aniso%mat - ref%mat)) > MAT_TOL) then
              call this%build_all_tables(n_tab, hmax_factor, h_bounds, dh)
              return
            end if
          end if
        end associate
      end if
    end do

    if (ref_idx == 0) then
      ! write(*,'(A)') 'WARNING build_table: all structures are nuggets. No table built.'
    else
      call this%build_composite_table(n_tab=n_tab, hmax_factor=hmax_factor, &
                                    h_bounds=h_bounds, dh=dh)
    end if
  end subroutine struct_build_table


  !-- Level A: build per-component tables.
  subroutine build_all_tables(this, n_tab, hmax_factor, h_bounds, dh)
    class(vgm_struct), intent(inout) :: this
    integer, intent(in), optional :: n_tab
    real,    intent(in), optional :: hmax_factor, h_bounds(:), dh(:)
    real :: factor
    integer :: iv
    factor = 3.5;  if (present(hmax_factor)) factor = hmax_factor
    do iv = 1, this%nstruct
      if (present(h_bounds) .and. present(dh)) then
        call this%structs(iv)%build_table( &
          hmax=h_bounds(size(h_bounds)), h_bounds=h_bounds, dh=dh)
      else
        call this%structs(iv)%build_table(hmax=factor, n_tab=n_tab)
      end if
    end do
  end subroutine build_all_tables


  !-- Level B: build composite struct-level table.
  !
  !   Stores: tab(0)   = cov0  (computed by struct_recompute_cov0)
  !           tab(i>0) = sum of product-groups evaluated at rdist_i
  !
  !   Product groups: consecutive components with is_product=.true. are
  !   multiplied together before being added to the sum.  Each component's
  !   rdist is scaled by a_major_ref / comp%a_major so that components in
  !   the same product group can have independent ranges (e.g. exp × hol
  !   with separate decay length and oscillation period).
  !
  !   REQUIRES: all non-nugget non-product structures share the same mat;
  !   product members must share the same rotation angles as the reference.
  subroutine build_composite_table(this, n_tab, hmax_factor, h_bounds, dh)
    class(vgm_struct), intent(inout) :: this
    integer, intent(in), optional :: n_tab
    real,    intent(in), optional :: hmax_factor, h_bounds(:), dh(:)

    integer              :: iv, k, i, ref_idx, ntab_total, nzones_local
    integer, allocatable :: n_zone(:)
    real,    allocatable :: z_bounds(:), z_inv_dh(:), h_b(:), d_h(:)
    integer, allocatable :: z_offset(:), z_n(:)
    real :: factor, hmax, h_lo, step_k, rdist, rdist_k, a_major_ref, val, group_val

    factor = 3.5;  if (present(hmax_factor)) factor = hmax_factor

    !-- Zone setup
    if (present(h_bounds) .and. present(dh)) then
      allocate(h_b, source=h_bounds)
      allocate(d_h, source=dh)
      hmax = h_bounds(size(h_bounds))
    else
      hmax = factor
      allocate(h_b(2));  h_b = [0.0, hmax]
      allocate(d_h(1))
      if (present(n_tab)) then
          d_h(1) = hmax / real(n_tab)
      else
          d_h(1) = hmax / 10000.0
      end if
    end if

    if (allocated(this%struct_tab))         deallocate(this%struct_tab)
    if (allocated(this%struct_zone_bounds)) deallocate(this%struct_zone_bounds)
    if (allocated(this%struct_zone_inv_dh)) deallocate(this%struct_zone_inv_dh)
    if (allocated(this%struct_zone_offset)) deallocate(this%struct_zone_offset)
    if (allocated(this%struct_zone_n))      deallocate(this%struct_zone_n)

    call alloc_zones(h_b, d_h, hmax, &
      nzones_local, ntab_total, n_zone, z_bounds, z_inv_dh, z_offset, z_n)

    !-- Reference a_major for per-component rdist scaling.
    ref_idx = 0
    do iv = 1, this%nstruct
      if (this%structs(iv)%vtype_id /= VGM_NUG) then; ref_idx = iv; exit; end if
    end do
    a_major_ref = merge(this%struct_aniso%a_major, 1.0, ref_idx > 0)

    allocate(this%struct_tab(0:ntab_total))
    this%struct_tab(0) = this%cov0

    do k = 1, nzones_local
      h_lo   = z_bounds(k-1)
      step_k = 1.0 / z_inv_dh(k)
      do i = 1, n_zone(k)
        rdist = h_lo + real(i) * step_k   ! dimensionless lag relative to a_major_ref
        val   = 0.0
        iv    = 1
        do while (iv <= this%nstruct)
          associate(cc => this%structs(iv))
            if (cc%vtype_id == VGM_NUG) then
              iv = iv + 1
              cycle
            end if
            rdist_k   = rdist * a_major_ref / cc%aniso%a_major
            group_val = cc%sill * corefunc_fn(cc%vtype_id, rdist_k)
          end associate
          ! Absorb any subsequent product members into this group.
          do while (iv < this%nstruct)
            if (.not. this%structs(iv+1)%is_product) exit
            iv = iv + 1
            associate(cc => this%structs(iv))
              rdist_k   = rdist * a_major_ref / cc%aniso%a_major
              group_val = group_val * cc%sill * corefunc_fn(cc%vtype_id, rdist_k)
            end associate
          end do
          val = val + group_val
          iv  = iv  + 1
        end do
        this%struct_tab(z_offset(k)+i) = val
      end do
    end do

    this%struct_nzones = nzones_local
    this%struct_n      = ntab_total
    this%struct_hmax   = hmax
    this%struct_analytic_tail = .false.
    iv = 1
    do while (iv <= this%nstruct)
      if (corefunc_has_analytic_tail(this%structs(iv)%vtype_id)) then
        ! A product group has an analytic tail when ALL members have one.
        if (this%structs(iv)%vtype_id /= VGM_NUG) then
          block
            logical :: group_tail
            group_tail = corefunc_has_analytic_tail(this%structs(iv)%vtype_id)
            do while (iv < this%nstruct .and. this%structs(iv+1)%is_product)
              iv = iv + 1
              group_tail = group_tail .and. corefunc_has_analytic_tail(this%structs(iv)%vtype_id)
            end do
            this%struct_analytic_tail = this%struct_analytic_tail .or. group_tail
          end block
        end if
      end if
      iv = iv + 1
    end do
    call move_alloc(z_bounds, this%struct_zone_bounds)
    call move_alloc(z_inv_dh, this%struct_zone_inv_dh)
    call move_alloc(z_offset, this%struct_zone_offset)
    call move_alloc(z_n,      this%struct_zone_n)
    this%struct_tab_ready = .true.
  end subroutine build_composite_table


  function cov_struct_tab_h(this, h) result(res)
    class(vgm_struct), intent(in) :: this
    real,              intent(in) :: h
    real :: res, fi
    integer :: k, i
    if (h >= this%struct_hmax) then
      if (this%struct_analytic_tail) then
        res = this%cov_h(h)
      else
        res = 0.0
      end if
      return
    end if
    if (h <= 0.0)              then; res = this%struct_tab(0);   return; end if
    do k = 1, this%struct_nzones
      if (h < this%struct_zone_bounds(k)) then
        fi = (h - this%struct_zone_bounds(k-1)) * this%struct_zone_inv_dh(k)
        i  = this%struct_zone_offset(k) + min(int(fi), this%struct_zone_n(k)-1)
        fi = fi - real(int(fi))
        res = this%struct_tab(i) + fi*(this%struct_tab(i+1) - this%struct_tab(i))
        return
      end if
    end do
    if (this%struct_analytic_tail) then
      res = this%cov_h(h)
    else
      res = 0.0
    end if
  end function cov_struct_tab_h


  function struct_cov_h(this, h) result(res)
    class(vgm_struct), intent(in) :: this
    real,              intent(in) :: h
    real :: res, group_val
    integer :: iv
    res = 0.0
    iv  = 1
    do while (iv <= this%nstruct)
      group_val = this%structs(iv)%cov_h(h)
      do while (iv < this%nstruct .and. this%structs(iv+1)%is_product)
        iv        = iv + 1
        group_val = group_val * this%structs(iv)%cov_h(h)
      end do
      res = res + group_val
      iv  = iv  + 1
    end do
  end function struct_cov_h

  function struct_cov_lag(this, lag) result(res)
    class(vgm_struct), intent(in) :: this
    real,              intent(in) :: lag(3)
    real :: res, group_val
    integer :: iv
    res = 0.0
    iv  = 1
    do while (iv <= this%nstruct)
      group_val = this%structs(iv)%cov_lag(lag(1), lag(2), lag(3))
      do while (iv < this%nstruct .and. this%structs(iv+1)%is_product)
        iv        = iv + 1
        group_val = group_val * this%structs(iv)%cov_lag(lag(1), lag(2), lag(3))
      end do
      res = res + group_val
      iv  = iv  + 1
    end do
  end function struct_cov_lag

  !-- Dispatches to Level B (struct_tab) if ready, else Level A (per-component).
  function struct_cov_tab(this, lag) result(res)
    class(vgm_struct), intent(in) :: this
    real,              intent(in) :: lag(3)
    real :: res, group_val
    integer :: iv
    if (this%struct_tab_ready) then
      res = this%cov_struct_tab_h( &
        this%struct_aniso%h_iso(lag(1), lag(2), lag(3)))
    else
      res = 0.0
      iv  = 1
      do while (iv <= this%nstruct)
        group_val = this%structs(iv)%cov_tab(lag(1), lag(2), lag(3))
        do while (iv < this%nstruct .and. this%structs(iv+1)%is_product)
          iv        = iv + 1
          group_val = group_val * this%structs(iv)%cov_tab(lag(1), lag(2), lag(3))
        end do
        res = res + group_val
        iv  = iv  + 1
      end do
    end if
  end function struct_cov_tab

  function struct_tostr(this) result(s)
    class(vgm_struct), intent(in) :: this
    character(:), allocatable :: s
    character(64) :: buf
    integer :: iv
    write(buf,'("  ndim=",I0,"  nstruct=",I0,"  struct_tab=",L1)') &
      this%ndim, this%nstruct, this%struct_tab_ready
    s = trim(buf)
    do iv = 1, this%nstruct
      s = s // new_line('a') // '    ' // this%structs(iv)%tostr()
    end do
  end function struct_tostr

  !-- Heuristic positive-definiteness check.
  function struct_is_valid(this, allow_neg_sill) result(ok)
    class(vgm_struct), intent(in)           :: this
    logical,           intent(in), optional :: allow_neg_sill
    ! allow_neg_sill: when .true., skip the sill >= 0 check.
    ! Cross-variograms (ivar /= jvar) may have negative partial sills in a
    ! valid LMC, so callers should pass allow_neg_sill=.true. for those pairs.
    logical :: ok, neg_sill_ok
    integer :: iv
    character(256) :: msg
    neg_sill_ok = .false.
    if (present(allow_neg_sill)) neg_sill_ok = allow_neg_sill
    ok = .true.
    if (this%ndim < 1 .or. this%ndim > 3) then
      write(*,'(A,I0)') 'WARNING: ndim must be 1, 2 or 3, got ', this%ndim; ok = .false.
    end if
    do iv = 1, this%nstruct
      associate(c => this%structs(iv))

        if (c%nugget < 0.0) then
          write(*,'(A,I0,A)') &
            'WARNING vgm_struct: structure ', iv, ': negative nugget'
          ok = .false.
        end if
        if (c%sill < 0.0 .and. .not. neg_sill_ok) then
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
        if (c%vtype_id == VGM_HOL .and. this%ndim==3) then
            write(*,'(A,I0,A)') &
              'WARNING vgm_struct: structure ', iv, &
              ': hole-effect is not p.d. in 3D'
            ok = .false.
        end if
      end associate
      if (.not. ok) then
        call kriging_error('is_valid_vgm_st', trim(msg))
        return
      end if
    end do
  end function struct_is_valid

  !-- Recompute cov0 = C(0) from all structures, respecting product groups.
  !   Called after each struct_add_comp / struct_add_args.
  subroutine struct_recompute_cov0(this)
    class(vgm_struct), intent(inout) :: this
    real    :: group_c0, total
    integer :: iv
    total = 0.0
    iv    = 1
    do while (iv <= this%nstruct)
      group_c0 = this%structs(iv)%sill + this%structs(iv)%nugget
      do while (iv < this%nstruct .and. this%structs(iv+1)%is_product)
        iv       = iv + 1
        group_c0 = group_c0 * (this%structs(iv)%sill + this%structs(iv)%nugget)
      end do
      total = total + group_c0
      iv    = iv + 1
    end do
    this%cov0 = total
  end subroutine struct_recompute_cov0

end module variogram


!==============================================================================
! Usage examples
!==============================================================================
!
!   use variogram
!   type(vgm_struct) :: vg
!
!   !=== Standard model setup ===
!   vg%ndim = 3
!   call vg%add(spec='nug 0.1 0.0 1 1 1 0 0 0')            ! nugget
!   call vg%add(spec='exp 0.0 0.9 5000 2000 300 30 10 0')  ! exponential
!
!   !=== Level B composite table (1 aniso_h + 1 lookup in assembly loop) ===
!   call vg%build_struct_table(h_bounds=[0.,0.1,0.5,3.5], dh=[1e-5,1e-4,1e-3])
!
!   !=== Level A per-component tables (any anisotropy, fallback) ===
!   call vg%build_all_tables(h_bounds=[0.,0.1,0.5,3.5], dh=[1e-5,1e-4,1e-3])
!
!   !=== 2D model ===
!   vg%ndim = 2                                ! set BEFORE first add()
!   call vg%add(spec='sph 0 0.8 1000 500 500 45 0 0')
!   call vg%build_struct_table()               ! n_tab=10000, hmax=3.5
!
!   !=== Assembly loop ===
!   C(i,j) = vg%cov_tab(lag)                  ! fast: table + ndim aniso
!   C(i,j) = vg%cov_lag(lag)                  ! exact analytic (for validation)
!
!   !=== Call corefunc_fn directly (variogram fitting, etc.) ===
!   C = corefunc_fn(VGM_EXP, rdist)           ! elemental, vectorisable
!   C_arr = corefunc_fn(VGM_SPH, h_array)     ! works on whole arrays
!
!==============================================================================
