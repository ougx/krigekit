module common
  use, INTRINSIC    :: ieee_arithmetic
  ! use, intrinsic    :: iso_fortran_env
  logical           :: verbose_c = .false.
  integer           :: ndim_c = 0
  real, parameter   :: pi = 4.0*atan(1.0)
  real, parameter   :: verysmall = tiny(1.0e0) * 1000
  real, parameter   :: verylarge = huge(1.0e0) * 1e-3
  real, parameter   :: zero = 0.0e0
  real, parameter   :: one  = 1.0e0
  real, parameter   :: DEG2RAD = pi/180.0, EPSLON = 1.e-10
  real, protected   :: nan
contains

  subroutine init_nan()
     nan = IEEE_VALUE(0.0, IEEE_QUIET_NAN)
  end subroutine init_nan

  ! elemental function isnan(x) result(res) ! intrinsic  function
  !   real, intent(in) :: x
  !   logical          :: res
  !   res = x == nan
  ! end function
end module common