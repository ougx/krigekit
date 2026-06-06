!============================================================================
! kriging_error -- non-fatal error state for DLL callers
!
! Code inside kriging.dll must not call STOP or ERROR STOP: doing so terminates
! the host process, including python.exe when the library is loaded with ctypes.
! This module records the first error from a call chain so the C API can return
! a non-zero ierr and Python can raise an exception with the stored message.
!============================================================================

module kriging_err
  use iso_c_binding,  only: c_char, c_null_char
  implicit none
  private

  integer, parameter :: KRIGING_OK = 0
  integer, parameter :: KRIGING_ERROR_CODE = 1
  integer, parameter :: KRIGING_MSG_LEN = 2048

  ! One shared error slot is enough because every C API entry clears it before
  ! calling into t_kriging.  Nested Fortran calls then preserve the first error
  ! so the caller sees the root cause rather than a later cascade.
  integer :: last_ierr = KRIGING_OK
  character(len=KRIGING_MSG_LEN) :: last_message = ''

  public :: KRIGING_OK, KRIGING_ERROR_CODE
  public :: kriging_error, kriging_clear_error, kriging_ierr
  public :: kriging_failed, kriging_last_error, kriging_copy_error

  interface kriging_error
    module procedure kriging_error_plain
    module procedure kriging_error_block
  end interface kriging_error

contains

  subroutine kriging_clear_error()
    ! Called at the top of each C API entry point.
    last_ierr = KRIGING_OK
    last_message = ''
  end subroutine kriging_clear_error

  integer function kriging_ierr()
    kriging_ierr = last_ierr
  end function kriging_ierr

  logical function kriging_failed()
    kriging_failed = (last_ierr /= KRIGING_OK)
  end function kriging_failed

  function kriging_last_error() result(msg)
    character(len=:), allocatable :: msg
    msg = trim(last_message)
  end function kriging_last_error

  subroutine kriging_copy_error(cbuf, nbuf)
    ! Copy the Fortran message into a null-terminated C buffer for ctypes.
    ! The buffer is always terminated when nbuf > 0, even after truncation.
    character(kind=c_char), intent(out) :: cbuf(*)
    integer, intent(in) :: nbuf
    integer :: i, n

    if (nbuf <= 0) return

    n = min(len_trim(last_message), nbuf - 1)
    do i = 1, n
      cbuf(i) = last_message(i:i)
    end do
    cbuf(n + 1) = c_null_char
    do i = n + 2, nbuf
      cbuf(i) = c_null_char
    end do
  end subroutine kriging_copy_error

  subroutine set_error(context, msg, iblock)
    character(len=*), intent(in) :: context
    character(len=*), intent(in) :: msg
    integer, intent(in), optional :: iblock

    character(len=64) :: blkstr
    character(len=KRIGING_MSG_LEN) :: formatted
    character(len=1), parameter :: NL=new_line('A')

    if (present(iblock)) then
      write(blkstr, '(I0)') iblock
      formatted = NL//'KRIGING ERROR: Location: ' // trim(context) // &
                  '; Block: ' // trim(blkstr) // &
                  NL//'Description: ' // trim(msg)
    else
      formatted = NL//'KRIGING ERROR: Location: ' // trim(context) // &
                  NL//'Description: ' // trim(msg)
    end if

    ! Preserve the first failure in a call chain; later failures are usually
    ! consequences of continuing up the stack after that original error.  The
    ! critical section keeps simultaneous OpenMP blocks from racing while they
    ! record that first failure.
    !$OMP CRITICAL(kriging_error_state)
    if (last_ierr == KRIGING_OK) then
      last_ierr = KRIGING_ERROR_CODE
      ! Do not print here.  C/Python wrappers and SPARKS decide how to present
      ! the stored message so callers do not get duplicate diagnostics.
      last_message = formatted
    end if
    !$OMP END CRITICAL(kriging_error_state)
  end subroutine set_error

  subroutine kriging_error_plain(context, msg)
    character(len=*), intent(in) :: context
    character(len=*), intent(in) :: msg
    call set_error(context, msg)
  end subroutine kriging_error_plain

  subroutine kriging_error_block(context, msg, iblock)
    character(len=*), intent(in) :: context
    character(len=*), intent(in) :: msg
    integer, intent(in) :: iblock
    call set_error(context, msg, iblock)
  end subroutine kriging_error_block

end module kriging_err
