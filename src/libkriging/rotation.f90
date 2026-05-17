!!---------------------------------------------------------------------
!! Module: rotation
!! Purpose:
!!   Construct and apply 3D anisotropic rotation matrices.
!!
!! Coordinate system:
!!   - ang1: azimuth (clockwise from north, Z-axis rotation)
!!   - ang2: dip (rotation about X-axis)
!!   - ang3: twist about principal axis (Y-axis rotation in this setup)
!!
!! Anisotropy:
!!   - anis1: scaling in X direction
!!   - anis2: scaling in Z direction
!!
!! Author: Michael Ou
!! Date  : Oct 2025
!!---------------------------------------------------------------------

module rotation
use common, only: DEG2RAD, EPSLON
implicit none

private

public :: calc_rotmat, rotate, rotated_dist, rotated_dists, print_rotmat, sub_rotate

contains


!-----------------------------------------------------------------------
!> Compute anisotropic rotation matrix (3x3)
!!
!! Constructs a transformation matrix that:
!!   1) Applies anisotropic scaling
!!   2) Applies rotations in order: Y (ang3), X (ang2), Z (ang1)
!!
!! Angles are in degrees.
!!
!! @param ang1   Azimuth (clockwise from North, Z-rotation)
!! @param ang2   Dip (positive up, X-rotation)
!! @param ang3   Rotation about principal axis (Y-rotation)
!! @param anis1  First anisotropy ratio (X scaling)
!! @param anis2  Second anisotropy ratio (Z scaling)
!!
!! @return rotmat 3x3 transformation matrix
!-----------------------------------------------------------------------
function calc_rotmat(ang1,ang2,ang3,anis1,anis2) result(rotmat)
implicit none
real                    :: rotmat(3,3)
real, intent(in)        :: ang1,ang2,ang3,anis1,anis2
real                    :: alpha, sina, cosa, tmp(3,3)
logical                 :: scaling = .false.

! Initialize as identity with anisotropic scaling
  rotmat = 0.0
  rotmat(1,1) = 1.0 / anis1
  rotmat(2,2) = 1.0
  rotmat(3,3) = 1.0 / anis2

! rotate about Y axis
if (ABS(ang3)>EPSLON) then
    alpha = -ang3 * DEG2RAD
    sina = sin(alpha)
    cosa = cos(alpha)
    tmp(:,1) = [ cosa, 0.0, sina]
    tmp(:,2) = [ 0.0 , 1.0, 0.0 ]
    tmp(:,3) = [-sina, 0.0, cosa]
    rotmat = MATMUL(rotmat, tmp)
end if

! rotate about X axis
if (ABS(ang2)>EPSLON) then
    alpha = -ang2 * DEG2RAD
    sina = sin(alpha)
    cosa = cos(alpha)
    tmp(:,1) = [1.0,  0.0 , 0.0 ]
    tmp(:,2) = [0.0,  cosa, sina]
    tmp(:,3) = [0.0, -sina, cosa]
    rotmat = MATMUL(rotmat, tmp)
end if

! rotate about Z axis
if (ABS(ang1)>EPSLON) then
    alpha = ang1 * DEG2RAD
    sina = sin(alpha)
    cosa = cos(alpha)
    tmp(:,1) = [ cosa, sina, 0.0]
    tmp(:,2) = [-sina, cosa, 0.0]
    tmp(:,3) = [ 0.0 , 0.0 , 1.0]
    rotmat = MATMUL(rotmat, tmp)
end if
end function


!-----------------------------------------------------------------------
!> Print rotation matrix to output unit
!!
!! @param rotmat 3x3 matrix
!! @param iout   (optional) output unit (default = 6)
!-----------------------------------------------------------------------
subroutine print_rotmat(rotmat, iout)
implicit none
integer, optional       :: iout
real                    :: rotmat(3,3)
integer                 :: iiout
if (present(iout)) then
    iiout=iout
else
    iiout=6
end if
write(iiout, *) ''
write(iiout, *) 'Rotation matrix:'
write(iiout, '(3F15.10)') rotmat
write(iiout, *) ''
end subroutine


!-----------------------------------------------------------------------
!> Rotate a set of points (function for backward compatibility, use sub_rotate for efficiency)
!!
!! Applies rotation matrix to coordinates, optionally shifting origin.
!!
!! @param rotmat 3x3 rotation matrix
!! @param coord1 Input coordinates (ndim x npnt)
!! @param origin Optional origin shift
!!
!! @return coord2 Rotated coordinates
!-----------------------------------------------------------------------
function rotate(rotmat, ndim, npnt, coord1, origin) result(coord2)
implicit none
real                    :: rotmat(3,3)
integer                 :: npnt, ndim
real                    :: coord1(ndim, npnt)
real                    :: coord2(ndim, npnt)
real, optional          :: origin(ndim)
! local
integer                 :: idim, j
real                    :: tmp(ndim, npnt)

call sub_rotate(rotmat, ndim, npnt, coord1, coord2, origin)
end function


!-----------------------------------------------------------------------
!> Rotate a set of points
!!
!! Applies rotation matrix to coordinates, optionally shifting origin.
!!
!! @param rotmat 3x3 rotation matrix
!! @param coord1 Input coordinates (ndim x npnt)
!! @param origin Optional origin shift
!!
!! @return coord2 Rotated coordinates
!-----------------------------------------------------------------------
subroutine sub_rotate(rotmat, ndim, npnt, coord1, coord2, origin)
implicit none
real                    :: rotmat(3,3)
integer                 :: npnt, ndim
real                    :: coord1(ndim, npnt)
real                    :: coord2(ndim, npnt)
real, optional          :: origin(ndim)
! local
integer                 :: idim, j
real                    :: tmp(ndim, npnt)

if (present(origin)) then
  do j = 1, npnt
    tmp(:, j) = coord1(:, j) - origin
  end do
  call mat_mul(rotmat, ndim, tmp   , coord2)
else
  call mat_mul(rotmat, ndim, coord1, coord2)
end if
end subroutine


!-----------------------------------------------------------------------
!> Distance between two points in rotated space
!-----------------------------------------------------------------------
function rotated_dist (rotmat, ndim, coord1, coord2) result(res)
implicit none
real                    :: rotmat(3,3)
integer                 :: ndim
real                    :: coord1(ndim), coord2(ndim), res
res = sqrt(sum(rotate(rotmat, ndim, 1, coord1, coord2) ** 2))
end function


!-----------------------------------------------------------------------
!> Distances from a reference point to multiple points in rotated space
!-----------------------------------------------------------------------
function rotated_dists(rotmat, ndim, coord0, coords) result(res)
implicit none
! Arguments
integer, intent(in)          :: ndim
real, intent(in)             :: rotmat(3,3)
real, intent(in)             :: coord0(ndim)
real, intent(in)             :: coords(:,:)
real                         :: res(size(coords,2))

! Locals
integer                      :: npnt, j
real                         :: diffs(ndim, size(coords,2))
real                         :: tmp  (ndim, size(coords,2))

npnt = size(coords, 2)

do j = 1, npnt
  diffs(:, j) = coords(:, j) - coord0
end do

! Rotate differences
call mat_mul(rotmat, ndim, diffs, tmp)

! Euclidean distances in rotated space
res = sqrt(sum(tmp**2, dim=1))
end function rotated_dists


!-----------------------------------------------------------------------
!> replace MATMUL; more efficient for small 3x3 matrix
!-----------------------------------------------------------------------
subroutine mat_mul(rotmat, ndim, x, res)
real, intent(in)             :: rotmat(3,3)
integer, intent(in)          :: ndim
real, intent(in)             :: x(:, :)
real, intent(out)            :: res(ndim, size(x, 2))

! local
integer                      :: i

select case (ndim)
case (1)
res(1,:) = rotmat(1,1) * x(1, :)

case (2)
res(1,:) = rotmat(1,1)*x(1,:) + rotmat(1,2)*x(2,:)
res(2,:) = rotmat(2,1)*x(1,:) + rotmat(2,2)*x(2,:)

case (3)
res(1,:) = rotmat(1,1)*x(1,:) + rotmat(1,2)*x(2,:) + rotmat(1,3)*x(3,:)
res(2,:) = rotmat(2,1)*x(1,:) + rotmat(2,2)*x(2,:) + rotmat(2,3)*x(3,:)
res(3,:) = rotmat(3,1)*x(1,:) + rotmat(3,2)*x(2,:) + rotmat(3,3)*x(3,:)

end select
end subroutine

end module