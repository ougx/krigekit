!==============================================================================
! kriging_capi_common.F90
!
! Shared infrastructure for kriging_capi.F90 and kriging_st_capi.f90:
!
!   Unified polymorphic handle registry
!   ------------------------------------
!   Both spatial (t_kriging) and space-time (t_kriging_st) objects are stored
!   as class(t_kriging_base) pointers in a single registry array.  Each CAPI
!   module provides its own thin get_obj wrapper that calls get_obj_base and
!   then does a select type downcast to recover the concrete typed pointer.
!
!   Shared utilities
!   ----------------
!   c2fstr, l() — identical helpers used by both CAPIs.
!   krige_get_last_error — single C symbol; both CAPIs share the error queue.
!==============================================================================
module kriging_capi_common
  use iso_c_binding
  use kriging_base, only: t_kriging_base
  use kriging_err,  only: kriging_copy_error, kriging_ierr, kriging_error, kriging_failed
  implicit none
  private

  !-- Polymorphic slot: stores either t_kriging or t_kriging_st pointer.
  type :: handle_slot
    class(t_kriging_base), pointer :: obj => null()
  end type handle_slot

  type(handle_slot), allocatable, save :: registry(:)

  public :: get_obj_base
  public :: store_obj_base
  public :: release_obj_base
  public :: c2fstr
  public :: l

contains

  !=============================================================================
  ! krige_get_last_error — single C symbol for both kriging types.
  !=============================================================================
  integer(c_int) function krige_get_last_error(buffer, nbuf) &
      bind(C, name='krige_get_last_error') result(ierr)
    character(kind=c_char), intent(out) :: buffer(*)
    integer(c_int),         intent(in), value :: nbuf
    call kriging_copy_error(buffer, int(nbuf))
    ierr = int(kriging_ierr(), c_int)
  end function krige_get_last_error

  !=============================================================================
  ! Registry operations
  !=============================================================================

  !-- Resolve handle → class(t_kriging_base) pointer.  Bad/stale handles set a
  !   kriging error instead of dereferencing garbage.
  subroutine get_obj_base(handle, obj)
    integer(c_intptr_t),            intent(in),  value :: handle
    class(t_kriging_base), pointer, intent(out)        :: obj
    integer :: idx
    nullify(obj)
    if (handle == 0_c_intptr_t) then
      call kriging_error('kriging_capi', 'Null kriging object handle')
      return
    end if
    idx = int(handle)
    if (.not. allocated(registry) .or. idx < 1 .or. idx > size(registry)) then
      call kriging_error('kriging_capi', 'Invalid kriging object handle')
      return
    end if
    if (associated(registry(idx)%obj)) obj => registry(idx)%obj
    if (.not. associated(obj)) &
      call kriging_error('kriging_capi', 'Invalid kriging object handle')
  end subroutine get_obj_base


  !-- Store a newly allocated concrete object in the first free registry slot.
  !   Growing keeps old slot numbers stable so existing Python handles remain valid.
  subroutine store_obj_base(obj, handle)
    class(t_kriging_base), pointer, intent(in)  :: obj
    integer(c_intptr_t),            intent(out) :: handle
    integer :: i
    if (.not. allocated(registry)) allocate(registry(16))
    do i = 1, size(registry)
      if (.not. associated(registry(i)%obj)) then
        registry(i)%obj => obj
        handle = int(i, c_intptr_t)
        return
      end if
    end do
    call grow_registry_()
    do i = 1, size(registry)
      if (.not. associated(registry(i)%obj)) then
        registry(i)%obj => obj
        handle = int(i, c_intptr_t)
        return
      end if
    end do
    handle = 0_c_intptr_t
    call kriging_error('krige_create', 'Failed to allocate a kriging handle slot')
  end subroutine store_obj_base


  !-- Nullify the slot so it can be reused.  Other slots are unchanged.
  subroutine release_obj_base(handle)
    integer(c_intptr_t), intent(in), value :: handle
    integer :: idx
    idx = int(handle)
    if (allocated(registry) .and. idx >= 1 .and. idx <= size(registry)) &
      nullify(registry(idx)%obj)
  end subroutine release_obj_base


  !-- Double the registry, preserving all pointer associations.
  subroutine grow_registry_()
    type(handle_slot), allocatable :: tmp(:)
    integer :: i, old_n, new_n
    old_n = size(registry)
    new_n = max(1, old_n * 2)
    allocate(tmp(new_n))
    do i = 1, old_n
      if (associated(registry(i)%obj)) tmp(i)%obj => registry(i)%obj
    end do
    call move_alloc(tmp, registry)
  end subroutine grow_registry_

  !=============================================================================
  ! Shared utilities
  !=============================================================================

  !-- Convert a null-terminated C string to a Fortran character(len=1024).
  function c2fstr(cstr) result(fstr)
    character(kind=c_char), intent(in) :: cstr(*)
    character(len=1024) :: fstr
    integer :: i
    fstr = ''
    do i = 1, 1024
      if (cstr(i) == c_null_char) exit
      fstr(i:i) = cstr(i)
    end do
  end function c2fstr

  !-- Convert integer(c_int) flag (0/1) to Fortran logical.
  !   Only 1 maps to .true.; 0 and any other value map to .false.
  elemental function l(v) result(r)
    integer(c_int), intent(in), value :: v
    logical :: r
    r = (v == 1_c_int)
  end function l

end module kriging_capi_common
