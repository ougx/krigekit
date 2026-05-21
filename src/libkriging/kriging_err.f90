!============================================================================
! kriging_error — universal error handler
!
! Every guard clause in this module calls kriging_error instead of using
! error stop directly.  This provides a single point of control for:
!
!   1. Formatting: the subroutine name prefix (subname) and the specific
!      message are printed together with a clear ERROR banner so the user
!      can distinguish a kriging error from a generic Fortran runtime abort.
!
!   2. Extensibility: replacing error stop with a non-fatal path (e.g.
!      writing to a log file, setting a global status flag, or calling a
!      C-level exception handler for Python integration) requires changing
!      only this subroutine, not every call site.
!
!   3. OMP safety: error stop inside an OMP parallel region is implementation-
!      defined.  kriging_error is called before the parallel region or in the
!      sequential sections of the solve loop; it is not thread-safe and should
!      not be called from within a parallelised block.
!
! Parameters
!   context : calling subroutine name prefix, e.g. "t_kriging%set_obs: "
!   msg     : specific error description
!   iblock  : optional block index; printed when >= 0 to help locate the
!             failing block in the SGSIM / kriging loop
!
! Usage
!   call kriging_error(subname, "coord must be provided.")
!   call kriging_error(subname, "Singular matrix at block", iblock=ctx%iblock)
!============================================================================

module kriging_err
  use iso_fortran_env, only: input_unit, error_unit, output_unit
  interface kriging_error
    module procedure kriging_error_plain
    module procedure kriging_error_block
  end interface kriging_error

contains


  !============================================================================
  ! kriging_error_plain — error handler without block index
  !============================================================================
  subroutine kriging_error_plain(context, msg)
    character(len=*), intent(in) :: context   ! subroutine name prefix
    character(len=*), intent(in) :: msg       ! specific error message

    write(error_unit, '(A)')       ''
    write(error_unit, '(A)')       '  *** KRIGING ERROR ***'
    write(error_unit, '(A,A)')     '  Location    : ', trim(context)
    write(error_unit, '(A,A)')     '  Description : ', trim(msg)
    write(error_unit, '(A)')       ''
    error stop 1
  end subroutine kriging_error_plain


  !============================================================================
  ! kriging_error_block — error handler with optional block index
  !============================================================================
  subroutine kriging_error_block(context, msg, iblock)
    character(len=*), intent(in) :: context
    character(len=*), intent(in) :: msg
    integer,          intent(in) :: iblock

    character(len=32) :: blkstr
    write(blkstr, '(I0)') iblock
    write(error_unit, '(A)')   ''
    write(error_unit, '(A)')   '  *** KRIGING ERROR ***'
    write(error_unit, '(A,A)') '  Location    : ', trim(context)
    write(error_unit, '(A,A)') '  Block       : ', trim(blkstr)
    write(error_unit, '(A,A)') '  Description : ', trim(msg)
    write(error_unit, '(A)')   ''
    error stop 1
  end subroutine kriging_error_block

end module kriging_err
