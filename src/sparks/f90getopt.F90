! This program is free software: you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.
!
! This program is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.
!
! You should have received a copy of the GNU General Public License
! along with this program.  If not, see <http://www.gnu.org/licenses/>.

module f90getopt
!
! ================== Prologue =====================================================================================================
!
! Purpose:
!    getopt()- and getopt_long()-like functionality (similar to the
!    -functions) for Fortran 2003.
!
!
! History:
!    Version   Programmer         Date       Description
!    -------   ----------         ---------- -----------
!    0.8.0     Mark Gates         2014/04/27 Original code from Gates
!    0.9.0     Hani Ibrahim       2014/04/28 Removed non-standard CLI-functions and added standard F2K CLI-functions
!    1.0.0     Hani Ibrahim       2017/01/07 Parse "=" with long options, error messages to stderr not stdout
!    1.0.1     Hani Ibrahim       2017/09/10 longopt bug fixed
!    1.0.2     Hani Ibrahim       2017/09/29 Readme.md error fixed
!    1.0.3     Hani Ibrahim       2018/07/09 Several errors in Readme.md fixed
!    1.0.4     Hani Ibrahim       2022/03/31 Portable declaration of stdin/out/err fixed, refactoring, documentation
!    1.1.0     Hani Ibrahim       2022/04/10 Utility function "isnum()" added
!
! User routines:
!    getopt, isnum
!
! Global variables/types
!    option_s, optarg
!
! Special requirements:
!    Fortran 2003 compliant compiler
!
! ------------------ Use Module / Include files -----------------------------------------------------------------------------------
!
! ------------------ Implicit -----------------------------------------------------------------------------------------------------
  IMPLICIT NONE
! ------------------ Local declarations -------------------------------------------------------------------------------------------
  PUBLIC  :: getopt, option_s, optarg, isnum, reset_opt
  PRIVATE ! all other are private (hidden)
! ------------------ Constant declarations ----------------------------------------------------------------------------------------

  ! Portable declaration of stderr
#ifdef f2003
  use, intrinsic :: iso_fortran_env, only: stderr => error_unit
#else
#define stderr 0
#endif
  integer, parameter       :: shortlen = 2    ! maximum number of characters for short options
  character(len=2048)      :: optarg        ! Option's value
  character(len=shortlen)  :: optopt        ! Option's character
  integer                  :: optind = 1      ! Index of the next argument to process
  logical                  :: opterr = .true. ! Errors are printed by default. Set opterr=.false. to suppress them

  type option_s
    character(len=80)        :: name     ! Name of the option
    character(len=shortlen)  :: short    ! Option's short character equal to optopt
    integer                  :: narg     ! Option's number of arguments following this option, if it is zero then no arguments
    character(len=2048)      :: description   ! Description of the option
  end type option_s

  ! grpind is index of next option within group; always >= 2
  integer, private:: grpind = 2

contains

  ! ----------------------------------------

  character function substr(str, i, j)
    ! Return str(i:j) if 1 <= i <= j <= len(str),
    ! else return empty string.
    !
    ! This is needed because Fortran standard allows but doesn't *require* short-circuited
    ! logical AND and OR operators. So this sometimes fails:
    !     if ( i < len(str) .and. str(i+1:i+1) == ':' ) then
    ! but this works:
    !     if ( substr(str, i+1, i+1) == ':' ) then

    ! arguments
    character(len=*), intent(in) :: str
    integer, intent(in)          :: i, j

    if (1 <= i .and. i <= j .and. j <= len(str)) then
      substr = str(i:j)
    else
      substr = ''
    end if
  end function substr

  ! ----------------------------------------

  character(len=shortlen) function getopt(opts)
    ! Returns short option character & value (if applicable) of all arguments one by one

    ! arguments
    type(option_s), intent(in) :: opts(:)

    ! local variables
    character(len=512)                     :: arg

    optarg = ''
    if (optind > command_argument_count()) then
      getopt = char(0)
    end if

    call get_command_argument(optind, arg)
    if (arg(1:2) == '--') then
      getopt = process_long(opts, arg)
    elseif (arg(1:1) == '-') then
      getopt = process_short(opts, arg)
    else
      optarg = trim(arg)
      getopt = char(0)
    end if
  end function getopt

  subroutine reset_opt()
    optind = 1
  end subroutine
  ! ----------------------------------------

  character(len=shortlen) function process_long(opts, arg)
    ! Process long options

    ! arguments
    type(option_s), intent(in) :: opts(:)
    character(len=*), intent(in) :: arg

    ! local variables
    integer                      :: i = 0
    integer                      :: j = 0
    integer                      :: len_arg = 0             ! length of arg
    integer                      :: has_equalsign = 0       ! arg contains equal sign?
    character(len=2048)          :: carg

    len_arg = len_trim(arg)

    ! search for equal sign in arg and set flag "has_equalsign" and
    ! length of arg (till equal sign)
    do j = 1, len_arg
      if (arg(j:j) == "=") then
        has_equalsign = -1
        len_arg = j - 1
        exit
      end if
    end do

    ! search for matching long option

    if (has_equalsign /= 0) then
      if (arg(len_arg + 2:) == '') then ! no value (len_arg+2 value after "="
        write (stderr, '(a,a,a)') "ERROR: Option '", trim(arg), "' requires a value"
        process_long = char(0) ! Option not valid
        return
      end if
      optarg = optarg(len_arg + 2:)
    else
      optarg = ''
    end if

    do i = 1, size(opts)
      if (arg(3:len_arg) == opts(i)%name) then
        optopt = opts(i)%short
        process_long = optopt
        optind = optind + 1
        do j = 1, opts(i)%narg + has_equalsign
          if (optind <= command_argument_count()) then
            call get_command_argument(optind, carg)
            optarg = trim(optarg)//' '//trim(carg)
            optind = optind + 1
          elseif (opterr) then
            write (stderr, '(a,a,a)') "ERROR: Option '", trim(arg), "' requires values"
            process_long = char(0) ! Option not valid
          end if
        end do
        return
      end if
    end do
    ! else not found
    process_long = char(0)
    optopt = '?'
    if (opterr) then
      write (stderr, '(a,a,a)') "ERROR: Unrecognized option '", arg(1:len_arg), "'"
    end if
    return
  end function process_long

  ! ----------------------------------------

  character(len=shortlen) function process_short(opts, arg)
    ! Process short options

    ! arguments
    character(len=*), intent(in) :: arg
    type(option_s), intent(in), optional :: opts(:)

    ! local variables
    integer                      :: i, j, arglen, iopt
    integer                      :: narg = 0
    character(len=2048)          :: carg

    arglen = len_trim(arg)
    optopt = trim(arg(grpind:grpind + 1))
    process_short = optopt

    ! check if opts is used
    iopt = 0
    do j = 1, size(opts)
      if (optopt == opts(j)%short) then
        iopt = j
        exit
      end if
    end do

    if (iopt == 0) then
      ! unrecognised option
      process_short = '?'
      if (opterr) then
        write (stderr, '(a,a,a)') "ERROR: Unrecognized option '-", optopt, "'"
      end if

    else
      narg = opts(iopt)%narg
      if (optopt == ':') narg = 1
    end if

    if (narg > 0) then

      optarg = ''

      do j = 1, narg
        if (optind <= command_argument_count()) then
          optind = optind + 1
          call get_command_argument(optind, carg)
          optarg = trim(optarg)//' '//trim(carg)
        elseif (opterr) then
          write (*, '(a,a,a)') "ERROR: Option '-", optopt, "' requires values"
          process_short = char(0) ! Option not valid
          return
        end if
      end do
    end if
    grpind = 2
    optind = optind + 1
  end function process_short

  ! ----------------------------------------
  ! Utility function(s)
  ! ----------------------------------------

  integer function isnum(txtval)
    ! Verify whether a character string represents a numerical value
    !
    ! Can be used to check "optarg" for numbers. Can distinguish
    ! integer, real/double and character strings:
    !
    ! isnum = 0 => txtval is a string
    ! isnum = 1 => txtval is a integer
    ! isnum > 1 => txtval is a real/double

    character(len=*), intent(in) :: txtval

    ! Declaration local constants
    integer, parameter :: CINT = 1 ! when txtval contains integer
    integer, parameter :: CREAL = 2 ! when txtval contains real
    integer, parameter :: CREXP = 3 ! when txtval contains real (exponential)

    ! Declaration local variables
    integer :: num       ! numerical indicator variable, if > 0 (0 >= num >= CREXP)
    logical :: isint     ! integer indicator, if .true.
    logical :: isexp     ! real with exponent indicator, if .true.
    logical :: issign    ! sign (+/-) indicator, if .true.
    logical :: issignexp ! sign (+/-) indicator for exponents, if .true.
    logical :: isblank   ! indicator for blanks between characters
    integer :: i         ! control variable (index), max. len(txtvar)

    ! Initialize local variables
    num = 0
    isint = .false.
    isexp = .false.
    issign = .false.
    issignexp = .false.
    isblank = .false.
    i = 0

    ! loop over characters
    do
      if (i >= len(txtval)) then
        ! last check
        if (.not. isint) exit
        if (num >= CREXP .and. (.not. isexp)) exit
        isnum = num
        return
      end if

      i = i + 1

      select case (txtval(i:i))
        ! process blanks
      case (' ')
        if (num == 0 .and. (.not. isblank)) then ! preceding or trailing blanks
          continue
        else if (num /= 0) then ! blank after sign or digit
          isblank = .true.
        end if
        ! process digits
      case ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
        if (num == 0) num = CINT ! first digit
        if (num < CREXP) then ! no exponent number
          isint = .true.
        else ! exponent number
          isexp = .true.
        end if
        if (isblank) exit ! if blanks are in the middle => string
        ! process signs
      case ('+', '-')
        if (num == 0) then ! sign of number
          if (issign) exit ! second sign without digit => string
          issign = .true.
          num = CINT
        else ! sign of exponent
          if (num < CREXP) exit
          if (issignexp) exit
          issignexp = .true.
        end if
        ! process decimal point
      case ('.')
        if (num /= CINT .and. i /= 1) exit
        num = CREAL
        ! process exponent
      case ('e', 'E', 'd', 'D')
        if (num >= CREXP) exit
        if (.not. isint) exit
        num = CREXP
      case default ! any other character means the string is non-numeric
        exit
      end select
    end do

    isnum = 0 ! if this point is reached, the string is non-numeric
    return
  end function isnum

end module f90getopt
