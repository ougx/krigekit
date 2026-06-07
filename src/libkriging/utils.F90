module utils
  use common, only: pi
  use kriging_err, only: kriging_error
  implicit none
contains

!@(#) M_random::scramble(3f): return integer array of random values 1 to N.
! https://fortranwiki.org/fortran/show/scramble
subroutine set_seq(array, number_of_values, random)
  integer, intent(out) :: array(:)
  integer, intent(in), optional :: number_of_values
  logical, intent(in), optional :: random
  ! local
  integer :: i, n, m, k, l, j, temp
  real    :: u
  if (present(number_of_values)) then
    n = number_of_values
  else
    n = size(array)
  end if
  do i = 1, n
    array(i) = i
  end do
  if(present(random)) then
    if (random) then
      m=n
      n=1
      do k=1,2
          do l=1,m
              call random_number(u)
              j = n + FLOOR((m+1-n)*u)
              ! switch values
              temp=array(j)
              array(j)=array(l)
              array(l)=temp
          enddo
      enddo
    end if
  end if
end subroutine


subroutine check_duplicate_coordinates_base(ndim, n, coord, has_duplicates, msg)
  integer, intent(in)           :: ndim, n
  real,    intent(in)           :: coord(ndim, n)
  logical, intent(out)          :: has_duplicates
  character(len=*), intent(out) :: msg

  integer, allocatable :: perm(:)
  integer :: i, j, k, d, ileft, iright, val_idx
  logical :: match, is_less

  has_duplicates = .false.
  if (n <= 1) return

  allocate(perm(n))
  do i = 1, n
    perm(i) = i
  end do

  ileft = n / 2 + 1
  iright = n

  do
    if (ileft > 1) then
      ileft = ileft - 1
      val_idx = perm(ileft)
    else
      val_idx = perm(iright)
      perm(iright) = perm(1)
      iright = iright - 1
      if (iright == 1) then
        perm(1) = val_idx
        exit
      end if
    end if

    i = ileft
    j = 2 * ileft
    do while (j <= iright)
      if (j < iright) then
        is_less = .false.
        do d = 1, ndim
          if (coord(d, perm(j)) < coord(d, perm(j+1))) then
            is_less = .true.
            exit
          else if (coord(d, perm(j)) > coord(d, perm(j+1))) then
            exit
          end if
        end do
        if (is_less) j = j + 1
      end if

      is_less = .false.
      do d = 1, ndim
        if (coord(d, val_idx) < coord(d, perm(j))) then
          is_less = .true.
          exit
        else if (coord(d, val_idx) > coord(d, perm(j))) then
          exit
        end if
      end do

      if (is_less) then
        perm(i) = perm(j)
        i = j
        j = j + j
      else
        j = iright + 1
      end if
    end do
    perm(i) = val_idx
  end do

  do i = 1, n - 1
    match = .true.
    do k = 1, ndim
      if (coord(k, perm(i)) /= coord(k, perm(i+1))) then
        match = .false.
        exit
      end if
    end do

    if (match) then
      has_duplicates = .true.
      write(msg, '(A, I0, A, I0, A)') "ERROR: Duplicate coordinate found! Station ", &
            perm(i), " and Station ", perm(i+1), " share identical coordinates."
      exit
    end if
  end do

  deallocate(perm)
end subroutine check_duplicate_coordinates_base


subroutine random_seed_initialize (key)
  !*****************************************************************************
  !
  !! random_seed_initialize() initializes the FORTRAN90 random number generator.
  !
  !  Discussion:
  !
  !    This is the stupidest, most awkward procedure I have seen!
  !
  !  Modified:
  !
  !    27 October 2021
  !
  !  Author:
  !
  !    John Burkardt
  !
  !  Input:
  !
  !    integer KEY: an initial seed for the random number generator.
  !
  implicit none

  integer key
  integer, allocatable :: iseed(:)
  integer seed_size

  if (key<=0) key = huge(seed_size) / 17
  call random_seed ( size = seed_size )
  allocate ( iseed(seed_size) )
  iseed(1:seed_size) = key
  call random_seed ( put = iseed )
  deallocate ( iseed )

  return
end subroutine random_seed_initialize


function avgval(x, dim) result(avg)
  real, intent(in) :: x(:,:)
  integer,  intent(in), optional :: dim
  real, allocatable :: avg(:)
  ! local
  integer    :: idim
  if (present(dim)) then
    idim = dim
    avg = sum(x, dim=idim) / real(size(x,idim))
  else
    avg = [sum(x) / real(size(x))]
  end if
end function avgval


subroutine r8vec_normal_01 ( n, x )

  !*****************************************************************************80
  !
  !! r8vec_normal_01() returns a unit pseudonormal R8VEC.
  !
  !  Discussion:
  !
  !    An R8VEC is an array of double precision real values.
  !
  !    The standard normal probability distribution function (PDF) has
  !    mean 0 and standard deviation 1.
  !
  !  Licensing:
  !
  !    This code is distributed under the MIT license.
  !
  !  Modified:
  !
  !    18 May 2014
  !
  !  Author:
  !
  !    John Burkardt
  !
  !  Input:
  !
  !    integer N, the number of values desired.
  !
  !  Output:
  !
  !    real ( kind = rk ) X(N), a sample of the standard normal PDF.
  !
  !  Local:
  !
  !    real ( kind = rk ) R(N+1), is used to store some uniform
  !    random values.  Its dimension is N+1, but really it is only needed
  !    to be the smallest even number greater than or equal to N.
  !
  !    integer X_LO_INDEX, X_HI_INDEX, records the range
  !    of entries of X that we need to compute
  !

    integer n
    integer m
    real  r(n+1)
    real  x(n)
    integer x_hi_index
    integer x_lo_index
  !
  !  Record the range of X we need to fill in.
  !

    if (n < 1) then
      call kriging_error('r8vec_normal_01', 'N must be larger than 0')
      return
    end if
    x_lo_index = 1
    x_hi_index = n
  !
  !  If we need just one new value, do that here to avoid null arrays.
  !
    if ( x_hi_index - x_lo_index + 1 == 1 ) then

      call random_number ( harvest = r(1:2) )

      x(x_hi_index) = &
        sqrt ( - 2.0e+00 * log ( r(1) ) ) * cos ( 2.0e+00 * pi * r(2) )
  !
  !  If we require an even number of values, that's easy.
  !
    else if ( mod ( x_hi_index - x_lo_index, 2 ) == 1 ) then

      m = ( x_hi_index - x_lo_index + 1 ) / 2

      call random_number ( harvest = r(1:2*m) )

      x(x_lo_index:x_hi_index-1:2) = &
        sqrt ( - 2.0e+00 * log ( r(1:2*m-1:2) ) ) &
        * cos ( 2.0e+00 * pi * r(2:2*m:2) )

      x(x_lo_index+1:x_hi_index:2) = &
        sqrt ( - 2.0e+00 * log ( r(1:2*m-1:2) ) ) &
        * sin ( 2.0e+00 * pi * r(2:2*m:2) )
  !
  !  If we require an odd number of values, we generate an even number,
  !  and handle the last pair specially, storing one in X(N), and
  !  saving the other for later.
  !
    else

      x_hi_index = x_hi_index - 1

      m = ( x_hi_index - x_lo_index + 1 ) / 2 + 1

      call random_number ( harvest = r(1:2*m) )

      x(x_lo_index:x_hi_index-1:2) = &
        sqrt ( - 2.0e+00 * log ( r(1:2*m-3:2) ) ) &
        * cos ( 2.0e+00 * pi * r(2:2*m-2:2) )

      x(x_lo_index+1:x_hi_index:2) = &
        sqrt ( - 2.0e+00 * log ( r(1:2*m-3:2) ) ) &
        * sin ( 2.0e+00 * pi * r(2:2*m-2:2) )

      x(n) = sqrt ( - 2.0e+00 * log ( r(2*m-1) ) ) &
        * cos ( 2.0e+00 * pi * r(2*m) )

    end if

    return
  end subroutine

  !============================================================================
  ! r8vec_uniform_01 — fill X(1:N) with U(0,1) pseudorandom values.
  !
  ! Uses Fortran's intrinsic random_number (range [0,1)).  Exact zero is
  ! clamped to epsilon so the result is safe for inverse-CDF operations
  ! (e.g. finding which threshold interval contains the draw).
  !============================================================================
  subroutine r8vec_uniform_01(n, x)
    integer, intent(in)  :: n
    real,    intent(out) :: x(n)

    if (n < 1) then
      call kriging_error('r8vec_uniform_01', 'N must be larger than 0')
      return
    end if

    call random_number(harvest = x)
    where (x <= 0.0) x = epsilon(0.0)   ! clamp exact zero; random_number never returns 1.0
  end subroutine r8vec_uniform_01

  !=============================================================
! Randomly select a subset from an integer array
!
! INPUT
!   input           Input array
!   nselect         Number of samples to select
!   allow_repeat    Optional:
!                     .true.  -> sampling WITH replacement
!                     .false. -> sampling WITHOUT replacement
!                                 (default)
!
! OUTPUT
!   subset          Randomly selected subset
!
! NOTES
!   - Uses Fisher-Yates partial shuffle for sampling
!     without replacement.
!   - Uses direct random indexing for sampling
!     with replacement.
!=============================================================

  subroutine random_subset(input, nselect, subset, allow_repeat)

    implicit none

    integer, intent(in)              :: input(:)
    integer, intent(in)              :: nselect
    integer, intent(out)             :: subset(nselect)

    logical, optional, intent(in)    :: allow_repeat

    logical                          :: repeat_ok

    integer                          :: temp(size(input))
    integer                          :: i, j, n
    integer                          :: itmp

    real                             :: r

    n = size(input)

    !---------------------------------------------
    ! Optional argument handling
    repeat_ok = .false.
    if (present(allow_repeat)) repeat_ok = allow_repeat

    !---------------------------------------------
    ! Sampling WITH replacement
    if (repeat_ok) then

      do i = 1, nselect

        call random_number(r)

        j = 1 + int(r * n)

        subset(i) = input(j)

      end do

      return

    end if

    !---------------------------------------------
    ! Sampling WITHOUT replacement

    if (nselect > n) then
      call kriging_error('random_subset', 'nselect > size(input) without replacement')
      return
    end if

    temp = input

    ! Partial Fisher-Yates shuffle
    do i = 1, nselect

      call random_number(r)

      j = i + int(r * real(n - i + 1))

      ! swap
      itmp    = temp(i)
      temp(i) = temp(j)
      temp(j) = itmp

    end do

    subset = temp(1:nselect)

  end subroutine random_subset

  function yesno(condition) result(res)
    character*3  :: res
    logical      :: condition
    if (condition) then
      res = "Yes"
    else
      res = "No"
    end if
  end function yesno

  pure function to_lower(str) result(lower_str)
    character(len=*), intent(in) :: str
    character(len=len(str))      :: lower_str
    integer                      :: i
    integer                      :: code

    do i = 1, len(str)
      code = iachar(str(i:i))
      ! Check if the character is an uppercase letter (A-Z)
      if (code >= iachar('A') .and. code <= iachar('Z')) then
        ! Convert to lowercase by offsetting the ASCII value
        lower_str(i:i) = achar(code + (iachar('a') - iachar('A')))
      else
        lower_str(i:i) = str(i:i)
      endif
    end do
  end function to_lower

  ! --- Main Function: Starts_with check ---
  pure function starts_with_ignore_case(str, prefix) result(is_match)
    character(len=*), intent(in) :: str
    character(len=*), intent(in) :: prefix
    logical                      :: is_match

    character(len=len(str))      :: clean_str
    character(len=len(str))      :: lower_str
    character(len=len(prefix))   :: lower_prefix
    integer                      :: prefix_len

    ! 1. Clear leading spaces and establish lengths
    clean_str = adjustl(str)
    prefix_len = len(prefix)

    ! 2. Guard clause: if the prefix is longer than the cleaned text, it's a mismatch
    if (prefix_len > len_trim(clean_str)) then
      is_match = .false.
      return
    end if

    ! 3. Normalize both inputs using the to_lower function
    lower_str    = to_lower(clean_str)
    lower_prefix = to_lower(prefix)

    ! 4. Check if the start of the lowercase string matches the lowercase prefix
    if (lower_str(1:prefix_len) == lower_prefix) then
      is_match = .true.
    else
      is_match = .false.
    end if

  end function starts_with_ignore_case
end module

