!==============================================================================
! Module: normal_score
!
! Normal-score (Gaussian anamorphosis) transform for sequential Gaussian
! simulation.  Implemented in the engine — rather than in any one client
! wrapper — so that every C-API client (Python, C/C++, R, Julia, MATLAB, ...)
! shares one consistent transform and back-transform, and seeded realisations
! are reproducible across clients.
!
! Workflow
! --------
!   call ns%build(zdata [, wt, zmin, zmax, ltail, utail, ltpar, utpar])
!   call ns%forward(zin, yout)   ! data units -> normal scores  (e.g. obs)
!   ... simulate in Gaussian space ...
!   call ns%back(yin, zout)      ! normal scores -> data units  (e.g. sims)
!
! The forward map is the standard rank-based normal-score transform with a
! weighted empirical CDF and tie averaging (declustering weights optional).
! The back-transform interpolates linearly in cumulative probability between
! tabulated quantiles, with GSLIB-style tail extrapolation (linear / power /
! hyperbolic) outside the data range bounded by zmin/zmax.
!
! All values use the engine's default real kind.
!==============================================================================
module normal_score
  use kriging_err, only: kriging_error
  implicit none
  private

  !-- Tail-extrapolation model codes (GSLIB convention).
  integer, parameter, public :: NS_TAIL_LINEAR = 1
  integer, parameter, public :: NS_TAIL_POWER  = 2
  integer, parameter, public :: NS_TAIL_HYPERB = 4

  real, parameter :: SQRT2 = 1.41421356237309504880
  real, parameter :: P_EPS = 1.0e-7     ! clamp on CDF to keep Ginv finite

  type, public :: t_nscore
    logical           :: ready = .false.
    integer           :: m     = 0      ! number of distinct quantile nodes
    real, allocatable :: z(:)           ! sorted distinct data values
    real, allocatable :: p(:)           ! midpoint cumulative probabilities
    real              :: zmin  = 0.0
    real              :: zmax  = 0.0
    integer           :: ltail = NS_TAIL_LINEAR
    integer           :: utail = NS_TAIL_LINEAR
    real              :: ltpar = 1.0
    real              :: utpar = 1.0
  contains
    procedure :: build   => nscore_build
    procedure :: forward => nscore_forward
    procedure :: back    => nscore_back
    procedure :: free    => nscore_free
  end type t_nscore

contains

  !=============================================================================
  ! nscore_build — build the transform table from data (optional decluster wt).
  !=============================================================================
  subroutine nscore_build(self, zdata, wt, zmin, zmax, ltail, utail, ltpar, utpar)
    class(t_nscore),   intent(inout)        :: self
    real,              intent(in)           :: zdata(:)
    real,              intent(in), optional :: wt(:)
    real,              intent(in), optional :: zmin, zmax, ltpar, utpar
    integer,           intent(in), optional :: ltail, utail
    character(len=*), parameter :: subname = "normal_score%build"

    integer               :: n, i, k
    integer,  allocatable :: idx(:)
    real,     allocatable :: zs(:), ws(:), pp(:), zz(:)
    real                  :: total, wsum, dmin, dmax

    call self%free()
    n = size(zdata)
    if (n < 2) then
      call kriging_error(subname, 'normal-score transform needs at least 2 observations')
      return
    end if

    allocate(zs(n), ws(n), idx(n))
    zs = zdata
    if (present(wt)) then
      if (size(wt) /= n) then
        call kriging_error(subname, 'size(wt) /= size(zdata)')
        return
      end if
      ws = wt
      if (any(ws < 0.0)) then
        call kriging_error(subname, 'declustering weights must be non-negative')
        return
      end if
    else
      ws = 1.0
    end if

    !-- sort indices by zs (ascending)
    call argsort(zs, idx)
    zs = zs(idx)
    ws = ws(idx)
    total = sum(ws)
    if (total <= 0.0) then
      call kriging_error(subname, 'sum of declustering weights must be positive')
      return
    end if

    !-- collapse ties; midpoint cumulative probability for each distinct value
    allocate(pp(n), zz(n))
    wsum = 0.0
    k    = 0
    i    = 1
    do while (i <= n)
      block
        integer  :: j
        real     :: gw
        j  = i
        gw = ws(i)
        do while (j < n)
          if (zs(j+1) /= zs(i)) exit
          j  = j + 1
          gw = gw + ws(j)
        end do
        k      = k + 1
        zz(k)  = zs(i)
        pp(k)  = (wsum + 0.5 * gw) / total
        wsum   = wsum + gw
        i      = j + 1
      end block
    end do

    self%m = k
    allocate(self%z(k), self%p(k))
    self%z = zz(1:k)
    self%p = pp(1:k)

    dmin = self%z(1)
    dmax = self%z(k)
    self%zmin = merge(zmin, dmin, present(zmin))
    self%zmax = merge(zmax, dmax, present(zmax))
    if (self%zmin > dmin) self%zmin = dmin       ! must bracket the data
    if (self%zmax < dmax) self%zmax = dmax
    self%ltail = merge(ltail, NS_TAIL_LINEAR, present(ltail))
    self%utail = merge(utail, NS_TAIL_LINEAR, present(utail))
    self%ltpar = merge(ltpar, 1.0, present(ltpar))
    self%utpar = merge(utpar, 1.0, present(utpar))
    if (self%ltpar <= 0.0) self%ltpar = 1.0
    if (self%utpar <= 0.0) self%utpar = 1.0

    self%ready = .true.
  end subroutine nscore_build

  !=============================================================================
  ! nscore_forward — data values -> normal scores (rank-based, ties averaged).
  !=============================================================================
  subroutine nscore_forward(self, zin, yout)
    class(t_nscore), intent(in)  :: self
    real,            intent(in)  :: zin(:)
    real,            intent(out) :: yout(:)
    integer :: i, k
    real    :: zv, pv, frac
    if (.not. self%ready) then
      call kriging_error('normal_score%forward', 'transform table not built')
      return
    end if
    do i = 1, size(zin)
      zv = zin(i)
      if (zv <= self%z(1)) then
        pv = self%p(1)
      else if (zv >= self%z(self%m)) then
        pv = self%p(self%m)
      else
        k    = bracket(self%z, zv)                  ! z(k) <= zv < z(k+1)
        frac = (zv - self%z(k)) / (self%z(k+1) - self%z(k))
        pv   = self%p(k) + frac * (self%p(k+1) - self%p(k))
      end if
      yout(i) = gauss_inv(min(max(pv, P_EPS), 1.0 - P_EPS))
    end do
  end subroutine nscore_forward

  !=============================================================================
  ! nscore_back — normal scores -> data values, with tail extrapolation.
  !=============================================================================
  subroutine nscore_back(self, yin, zout)
    class(t_nscore), intent(in)  :: self
    real,            intent(in)  :: yin(:)
    real,            intent(out) :: zout(:)
    integer :: i, k
    real    :: pv, frac, zv
    if (.not. self%ready) then
      call kriging_error('normal_score%back', 'transform table not built')
      return
    end if
    do i = 1, size(yin)
      pv = gauss_cdf(yin(i))
      if (pv <= self%p(1)) then
        zv = tail_lower(self, pv)
      else if (pv >= self%p(self%m)) then
        zv = tail_upper(self, pv)
      else
        k    = bracket(self%p, pv)                  ! p(k) <= pv < p(k+1)
        frac = (pv - self%p(k)) / (self%p(k+1) - self%p(k))
        zv   = self%z(k) + frac * (self%z(k+1) - self%z(k))
      end if
      zout(i) = min(max(zv, self%zmin), self%zmax)
    end do
  end subroutine nscore_back

  subroutine nscore_free(self)
    class(t_nscore), intent(inout) :: self
    if (allocated(self%z)) deallocate(self%z)
    if (allocated(self%p)) deallocate(self%p)
    self%m     = 0
    self%ready = .false.
  end subroutine nscore_free

  !=============================================================================
  ! Lower / upper tail extrapolation (between zmin..z(1) and z(m)..zmax).
  !=============================================================================
  pure function tail_lower(self, pv) result(zv)
    type(t_nscore), intent(in) :: self
    real,           intent(in) :: pv
    real :: zv, t
    t = pv / self%p(1)                              ! in [0, 1]
    if (t < 0.0) t = 0.0
    select case (self%ltail)
    case (NS_TAIL_POWER)
      zv = self%zmin + (t ** self%ltpar) * (self%z(1) - self%zmin)
    case default                                    ! linear
      zv = self%zmin + t * (self%z(1) - self%zmin)
    end select
  end function tail_lower

  pure function tail_upper(self, pv) result(zv)
    type(t_nscore), intent(in) :: self
    real,           intent(in) :: pv
    real :: zv, t, lambda, denom
    select case (self%utail)
    case (NS_TAIL_HYPERB)
      if (self%z(self%m) > 0.0) then
        lambda = (1.0 - self%p(self%m)) * self%z(self%m) ** self%utpar
        denom  = max(1.0 - pv, P_EPS)
        zv     = (lambda / denom) ** (1.0 / self%utpar)
      else                                          ! hyperbolic needs z>0; fall back
        t  = (pv - self%p(self%m)) / (1.0 - self%p(self%m))
        zv = self%z(self%m) + t * (self%zmax - self%z(self%m))
      end if
    case (NS_TAIL_POWER)
      t  = (pv - self%p(self%m)) / (1.0 - self%p(self%m))
      if (t < 0.0) t = 0.0
      zv = self%z(self%m) + (t ** self%utpar) * (self%zmax - self%z(self%m))
    case default                                    ! linear
      t  = (pv - self%p(self%m)) / (1.0 - self%p(self%m))
      if (t < 0.0) t = 0.0
      zv = self%z(self%m) + t * (self%zmax - self%z(self%m))
    end select
  end function tail_upper

  !=============================================================================
  ! Standard-normal CDF and inverse CDF.
  !=============================================================================
  pure function gauss_cdf(y) result(p)
    real, intent(in) :: y
    real :: p
    p = 0.5 * erfc(-y / SQRT2)
  end function gauss_cdf

  !-- Acklam's rational approximation to the inverse normal CDF, refined with
  !   one Halley step using the (intrinsic) error function.
  pure function gauss_inv(p) result(x)
    real, intent(in) :: p
    real :: x, q, r, e, u
    real, parameter :: a(6) = [ &
      -3.969683028665376e+01,  2.209460984245205e+02, &
      -2.759285104469687e+02,  1.383577518672690e+02, &
      -3.066479806614716e+01,  2.506628277459239e+00 ]
    real, parameter :: b(5) = [ &
      -5.447609879822406e+01,  1.615858368580409e+02, &
      -1.556989798598866e+02,  6.680131188771972e+01, &
      -1.328068155288572e+01 ]
    real, parameter :: c(6) = [ &
      -7.784894002430293e-03, -3.223964580411365e-01, &
      -2.400758277161838e+00, -2.549732539343734e+00, &
       4.374664141464968e+00,  2.938163982698783e+00 ]
    real, parameter :: d(4) = [ &
       7.784695709041462e-03,  3.224671290700398e-01, &
       2.445134137142996e+00,  3.754408661907416e+00 ]
    real, parameter :: plow = 0.02425, phigh = 1.0 - 0.02425
    if (p < plow) then
      q = sqrt(-2.0 * log(p))
      x = (((((c(1)*q + c(2))*q + c(3))*q + c(4))*q + c(5))*q + c(6)) / &
          ((((d(1)*q + d(2))*q + d(3))*q + d(4))*q + 1.0)
    else if (p <= phigh) then
      q = p - 0.5
      r = q * q
      x = (((((a(1)*r + a(2))*r + a(3))*r + a(4))*r + a(5))*r + a(6))*q / &
          (((((b(1)*r + b(2))*r + b(3))*r + b(4))*r + b(5))*r + 1.0)
    else
      q = sqrt(-2.0 * log(1.0 - p))
      x = -(((((c(1)*q + c(2))*q + c(3))*q + c(4))*q + c(5))*q + c(6)) / &
           ((((d(1)*q + d(2))*q + d(3))*q + d(4))*q + 1.0)
    end if
    !-- one Halley refinement step
    e = 0.5 * erfc(-x / SQRT2) - p
    u = e * 2.5066282746310002 * exp(0.5 * x * x)   ! e * sqrt(2*pi) * exp(x^2/2)
    x = x - u / (1.0 + 0.5 * x * u)
  end function gauss_inv

  !=============================================================================
  ! Small utilities: index sort and bracket search.
  !=============================================================================
  !-- Heapsort-based index sort (ascending).
  pure subroutine argsort(a, idx)
    real,    intent(in)  :: a(:)
    integer, intent(out) :: idx(:)
    integer :: n, i, ir, j, l, iidx
    real    :: q
    n = size(a)
    do i = 1, n
      idx(i) = i
    end do
    if (n < 2) return
    l  = n / 2 + 1
    ir = n
    do
      if (l > 1) then
        l    = l - 1
        iidx = idx(l)
        q    = a(iidx)
      else
        iidx     = idx(ir)
        q        = a(iidx)
        idx(ir)  = idx(1)
        ir       = ir - 1
        if (ir == 1) then
          idx(1) = iidx
          return
        end if
      end if
      i = l
      j = l + l
      do while (j <= ir)
        if (j < ir) then
          if (a(idx(j)) < a(idx(j+1))) j = j + 1
        end if
        if (q < a(idx(j))) then
          idx(i) = idx(j)
          i      = j
          j      = j + j
        else
          j = ir + 1
        end if
      end do
      idx(i) = iidx
    end do
  end subroutine argsort

  !-- Largest k with arr(k) <= v, assuming arr ascending and arr(1) <= v < arr(n).
  pure function bracket(arr, v) result(k)
    real, intent(in) :: arr(:)
    real, intent(in) :: v
    integer :: k, lo, hi, mid
    lo = 1
    hi = size(arr)
    do while (hi - lo > 1)
      mid = (lo + hi) / 2
      if (arr(mid) <= v) then
        lo = mid
      else
        hi = mid
      end if
    end do
    k = lo
  end function bracket

end module normal_score
