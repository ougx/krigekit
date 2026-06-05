module vgm_func
  use common, only: pi
  use utils, only: starts_with_ignore_case
  use kriging_err, only: kriging_error
  !-- Variogram type identifiers (integer tags replacing polymorphic types)
  integer, parameter :: VGM_NUG = 0   ! nugget
  integer, parameter :: VGM_SPH = 1   ! spherical
  integer, parameter :: VGM_EXP = 2   ! exponential
  integer, parameter :: VGM_GAU = 3   ! gaussian
  integer, parameter :: VGM_HOL = 4   ! hole-effect
  integer, parameter :: VGM_POW = 5   ! power (generalised covariance)
  integer, parameter :: VGM_BSQ = 6   ! bi-square
  integer, parameter :: VGM_CIR = 7   ! circular
  integer, parameter :: VGM_LIN = 8   ! linear
contains
  !=============================================================================
  ! corefunc_fn — pure elemental correlation function dispatcher.
  !
  !   rdist     : dimensionless isotropic lag = h / a_major  (in [0, hmax])
  !   returns   : correlation C(rdist); most models are in [0, 1],
  !               while hole-effect oscillates.
  !
  ! The compiler inlines this and may build a jump table for the select case.
  ! Being pure elemental enables SIMD vectorisation over arrays of rdist.
  !=============================================================================
  pure elemental function corefunc_fn(vtype_id, rdist) result(res)
    integer, intent(in) :: vtype_id
    real,    intent(in) :: rdist
    real :: res
    real, parameter :: POW_ALPHA = 1.5  ! fixed VGM_POW exponent
    select case (vtype_id)
      case (VGM_NUG)
        res = 0.0
      case (VGM_SPH)
        if (rdist < 1.0) then; res = 1.0 - 1.5*rdist + 0.5*rdist**3
        else;                  res = 0.0
        end if
      case (VGM_EXP)
        res = exp(-3.0 * rdist)
      case (VGM_GAU)
        res = exp(-3.0625 * rdist**2)
      case (VGM_HOL)
        res = cos(pi * rdist)
      case (VGM_POW)
        res = merge(1.0 - rdist**POW_ALPHA, 0.0, rdist < 1.0)
      case (VGM_BSQ)
        res = merge((1.0 - rdist**2)**2, 0.0, rdist < 1.0)
      case (VGM_CIR)
        res = merge( &
          1.0 - (2.0*rdist*sqrt(1.0-rdist**2) + 2.0*asin(rdist)) / pi, &
          0.0, rdist < 1.0)
      case (VGM_LIN)
        res = merge(1.0 - rdist, 0.0, rdist < 1.0)
      case default
        res = 0.0
    end select
  end function corefunc_fn

  !-- True for covariance shapes that remain non-zero or can recover past
  !   hr = 1.0.  Tabular covariance must not use a zero tail for these models.
  pure elemental function corefunc_has_analytic_tail(vtype_id) result(res)
    integer, intent(in) :: vtype_id
    logical :: res
    select case (vtype_id)
      case (VGM_EXP, VGM_GAU, VGM_HOL)
        res = .true.
      case default
        res = .false.
    end select
  end function corefunc_has_analytic_tail

  !-- Map 3-char spec string to vtype_id integer constant.
  function vtype_from_str(vtype) result(id)
    character(*), intent(in) :: vtype
    integer :: id
    id = -1
    if (starts_with_ignore_case(vtype, 'nug')) id = VGM_NUG
    if (starts_with_ignore_case(vtype, 'sph')) id = VGM_SPH
    if (starts_with_ignore_case(vtype, 'exp')) id = VGM_EXP
    if (starts_with_ignore_case(vtype, 'gau')) id = VGM_GAU
    if (starts_with_ignore_case(vtype, 'hol')) id = VGM_HOL
    if (starts_with_ignore_case(vtype, 'pow')) id = VGM_POW
    if (starts_with_ignore_case(vtype, 'bsq')) id = VGM_BSQ
    if (starts_with_ignore_case(vtype, 'cir')) id = VGM_CIR
    if (starts_with_ignore_case(vtype, 'lin')) id = VGM_LIN
    if (id==-1) call kriging_error('vtype_from_str', 'unknown vtype: '//vtype)
  end function vtype_from_str

end module vgm_func
