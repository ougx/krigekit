module utils
  use common, only: pi
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
      print*, 'N must be larger than 0'
      stop 1
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
      error stop "random_subset: nselect > size(input) without replacement"
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

end module


