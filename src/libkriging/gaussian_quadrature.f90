module gaussian_quadrature
real, parameter :: frac(4)=[0.4305681558,0.1699905218,-0.1699905218,-0.4305681558] ! see Palmer, 1993
real, parameter :: wgq4(4)=[0.1739274226,0.3260725774, 0.3260725774, 0.1739274226] ! see Palmer, 1993

real, allocatable            :: gqweight(:)
real, allocatable            :: gqdelxyz(:,:)
integer                      :: ngq
contains
subroutine set_gaussian_quadrature(ndim, blocksize)
  integer :: ndim
  real    :: blocksize(ndim)
  integer :: i, j, k, ii(3, 64)

  ngq = 4**ndim
  allocate(gqweight(ngq), gqdelxyz(ndim, ngq))

  do k=1, 4
    do j=1, 4
      do i=1, 4
        ii(1,i+(j-1)*4+(k-1)*16) = i
        ii(2,i+(j-1)*4+(k-1)*16) = j
        ii(3,i+(j-1)*4+(k-1)*16) = k
      end do
    end do
  end do
  gqweight = 1.0
  do i=1, ngq
    do k = 1, ndim
      gqdelxyz(k, i) = frac(ii(k,i)) * blocksize(k)
      gqweight(i) = gqweight(i) *  wgq4(ii(k,i))
    end do
  end do

  ! gqweight = 1.0 / (sum(gqdelxyz**2, dim=1))**0.5
  ! gqweight = gqweight / sum(gqweight)
  ! call print_weights
  ! call print_gqdelx(ndim)
end subroutine set_gaussian_quadrature


subroutine print_weights
  print "(F10.7)", gqweight
end subroutine

subroutine print_gqdelx(ndim)
  integer :: ndim
  character*1   :: sdim
  write(sdim, "(I1)") ndim
  print "("//sdim//"F10.5)", gqdelxyz
end subroutine

end module