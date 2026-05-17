! This is the block Cholesky / Schur complement solver for universal kriging (UK)

! Schur complement
!   | A   B | | x | = | u |
!   | C   D | | y |   | v |
!  x =  A⁻¹ (u - B y)

! Solving sytem below:
!   | K   F | | w | = | k |
!   | Fᵀ  0 | | λ |   | f |
! Define:
!    A = K      (n×n)
!    B = F      (n×p)
! From the first row:
!    A w + B λ = k
! Solve for w:
!    w = A⁻¹ (k − Bλ)
! Substitute into second row:
!    Bᵀ w = f
!    Bᵀ A⁻¹ (k − Bλ) = f
! Expand:
!    Bᵀ A⁻¹ k − Bᵀ A⁻¹ B λ = f
! Rearrange:
!    (Bᵀ A⁻¹ B) λ = Bᵀ A⁻¹ k − f
! Define:
!    S = Bᵀ A⁻¹ B   (p×p)
!    rhs = Bᵀ A⁻¹ k − f
! So:
!    S λ = rhs
! Once λ is known:
!    w = A⁻¹ (k − Bλ)
! We never form A⁻¹. Instead:
!    A = L Lᵀ
!    A Y = B   →   Y = A⁻¹ B
!    A y = k   →   y = A⁻¹ k
! Then:
!    S = Bᵀ Y
!    rhs = Bᵀ y − f
! Solve small system:
!    S λ = rhs
! Then:
!    w = y - Y λ
! Summary (compact form)
!    (1) A = L Lᵀ
!    (2) Y = A⁻¹ B
!    (3) y = A⁻¹ k
!    (4) S = Bᵀ Y
!    (5) λ = S⁻¹ (Bᵀ y − f)
!    (6) w = y − Y λ

! Solving sytem below:
!   | K   F | | w | = | k |
!   | Fᵀ  0 | | λ |   | f |
! Where:
!   K (n x n): covariance matrix (SPD)
!   F (n x p): drift/design matrix (p drift terms)
!     e.g. [1, x, y, z, ...]
!   k (n): covariance vector to estimation point
!   f (p): drift evaluated at estimation point
!   λ (p): Lagrange multipliers
! Step 1. Factor (Cholesky decomposition):
!   K = L Lᵀ
! Step 2. forward Solve:
!   A = C⁻¹ F     (n x p)
!   y = C⁻¹ k     (n)
! Step 3: Build small system:
!   S = Fᵀ A      (p x p)
!   rhs = Fᵀ y - f
! Step 4: Solve:
!   S λ = rhs
! Step 5: Solve:
!   w = y - A λ
!   variance = sill - dot_product(k, y) - mu

! Special case: ordinary kriging
!   B = 1 (vector of ones)
!   p = 1
! Then:
!   S = 1ᵀ A⁻¹ 1   (scalar)
! and:
!   λ = (1ᵀ y − 1) / (1ᵀ z)
!   z = A⁻¹ 1

! Implmentation:
!   call dpotrf   ! factor
!   call dpotrs   ! solve
module solver
character(1), parameter :: uplo='U'
contains
  ! ----- Setup phase (once per neighborhood) -----
  subroutine uk_setup(K, F, L, A, S, n, p, info)
    implicit none

    integer, intent(in)  :: n, p
    real   , intent(in)  :: K(n,n)
    real   , intent(in)  :: F(n,p)

    real   , intent(out) :: L(n,n)
    real   , intent(out) :: A(n,p)
    real   , intent(out) :: S(p,p)

    integer, intent(out) :: info

    !---------------------------------------
    ! Cholesky: K = L L^T
    L = K
    call spotrf(uplo, n, L, n, info)
    if (info /= 0) return ! stop "Cholesky failed"

    !---------------------------------------
    ! A = K^{-1} F  (solve for all columns at once)
    A = F
    call spotrs(uplo, n, p, L, n, A, n, info)
    if (info /= 0) return ! stop "Cholesky solve failed"

    !---------------------------------------
    ! S = F^T A
    S = matmul(transpose(F), A)
  end subroutine


  ! ----- Solve phase (per estimation point) -----
  subroutine uk_solve(L, A, S, k, f, w, lambda, n, p, info)
    implicit none

    integer, intent(in) :: n, p
    real, intent(in) :: L(n,n)
    real, intent(in) :: A(n,p), S(p,p)
    real, intent(in) :: k(n)
    real, intent(in) :: f(p)

    real, intent(out) :: w(n)
    real, intent(out) :: lambda(p)
    integer, intent(out) :: info

    real :: y(n)
    real :: rhs(p)

    !---------------------------------------
    ! y = K^{-1} k
    y = k
    call spotrs(uplo, n, 1, L, n, y, n, info)
    if (info /= 0) return ! stop "Cholesky solve failed"

    !---------------------------------------
    ! rhs = F^T y - f
    rhs = matmul(transpose(A), k)   ! equivalent to F^T y; works because: A = K⁻¹F  ⇒  Fᵀy = (K⁻¹F)ᵀ k = Aᵀ k
    rhs = rhs - f

    !---------------------------------------
    ! Solve S λ = rhs  (small system)
    call sposv(uplo, p, 1, S, p, rhs, p, info)
    if (info /= 0) return ! stop "Small system solve failed"

    lambda = rhs

    !---------------------------------------
    ! w = y - A λ
    w = y - matmul(A, lambda)
  end subroutine

  !
  !-----------------------------------------------------------------------
  ! Universal Kriging via Block-Cholesky / Schur Complement
  !
  ! Solves:
  !     [ K   F ] [ w ] = [ k ]
  !     [ Fᵀ  0 ] [ λ ]   [ f ]
  !
  ! where:
  !   K (n×n) : covariance matrix (SPD)
  !   F (n×p) : drift/design matrix
  !   k (n)   : covariance between data and estimation point
  !   f (p)   : drift evaluated at estimation point
  !
  ! Outputs:
  !   w (n)   : kriging weights
  !   λ (p)   : Lagrange multipliers
  !
  ! Method:
  !   1. Factor K = L Lᵀ (Cholesky)
  !   2. Solve Y = K⁻¹ F
  !   3. Solve y = K⁻¹ k
  !   4. Form Schur complement S = Fᵀ Y
  !   5. Solve S λ = Fᵀ y − f
  !   6. Compute w = y − Y λ
  !
  !-----------------------------------------------------------------------
  subroutine kriging_solve(n, p, cov_mat, drift_mat, cov_vec, drift_vec, &
                         weights, lambda, info)

    implicit none

    integer, intent(in) :: n, p

    real   , intent(in)  :: cov_mat(n,n)
    real   , intent(in)  :: drift_mat(n,p)
    real   , intent(in)  :: cov_vec(n)
    real   , intent(in)  :: drift_vec(p)

    real   , intent(out) :: weights(n)
    real   , intent(out) :: lambda(p)
    integer, intent(out) :: info

    ! -------- local ----------
    real   :: L(n,n)
    real   :: kinv_cov_vec(n)
    real   :: kinv_drift(n,p)
    real   :: schur_mat(p,p)
    real   :: rhs(p)

    !--------------------------------------------------
    ! Cholesky factorization
    L = cov_mat
    call spotrf(uplo, n, L, n, info)
    if (info /= 0) return ! stop "Cholesky factoring failed"

    !--------------------------------------------------
    ! SIMPLE KRIGING
    if (p==0) then
      kinv_cov_vec = cov_vec
      call spotrs(uplo, n, 1, L, n, kinv_cov_vec, n, info)
      if (info /= 0) return ! stop "Simple kriging solve failed"
      weights = kinv_cov_vec
      lambda = 0.0
      return
    end if

    !--------------------------------------------------
    ! UNIVERSAL KRIGING

    ! kinv_drift = K⁻¹ F
    kinv_drift = drift_mat
    call spotrs(uplo, n, p, L, n, kinv_drift, n, info)
    if (info /= 0) return ! stop "Drift solve failed"

    ! kinv_cov_vec = K⁻¹ k
    kinv_cov_vec = cov_vec
    call spotrs(uplo, n, 1, L, n, kinv_cov_vec, n, info)
    if (info /= 0) return ! stop "rhs solve failed"

    ! Schur complement: Fᵀ K⁻¹ F
    schur_mat = matmul(transpose(drift_mat), kinv_drift)

    ! rhs: Fᵀ K⁻¹ k − f
    rhs = matmul(transpose(drift_mat), kinv_cov_vec) - drift_vec

    ! solve small system
    call sposv(uplo, p, 1, schur_mat, p, rhs, p, info)
    if (info /= 0) return ! stop "small system solve failed"

    lambda = rhs

    ! final weights
    weights = kinv_cov_vec - matmul(kinv_drift, lambda)
  end subroutine


  subroutine gaussian_elimination(n, matA, rhsB, x, info)
    implicit none

    integer, intent(in) :: n
    real, intent(in) :: matA(n,n)
    real, intent(in) :: rhsB(n)

    real, intent(out) :: x(n)
    integer, intent(out) :: info

    ! locals
    real :: Acopy(n,n)
    integer :: k, piv, i
    real :: y(n)
    real :: rhs(n)
    real :: maxv, factor, tmp

    info = 0
    Acopy = matA
    rhs = rhsB

    ! forward elimination
    do k = 1, n-1
      piv = k
      maxv = abs(Acopy(k,k))
      do i = k+1, n
        if (abs(Acopy(i,k)) > maxv) then
          maxv = abs(Acopy(i,k))
          piv = i
        end if
      end do
      if (piv /= k) then
        Acopy([k,piv], :) = Acopy([piv,k], :)
        rhs([k,piv]) = rhs([piv,k])
      end if
      if (abs(Acopy(k,k)) < 1.0e-10) then
        ! print *, 'Matrix is singular or nearly singular at pivot ', k
        info = k
        return
      end if
      do i = k+1, n
        factor = Acopy(i,k) / Acopy(k,k)
        Acopy(i,k:n) = Acopy(i,k:n) - factor * Acopy(k,k:n)
        rhs(i) = rhs(i) - factor * rhs(k)
      end do
    end do

    ! back substitution
    do i = n, 1, -1
      tmp = rhs(i)
      if (i < n) tmp = tmp - sum(Acopy(i, i+1:n) * x(i+1:n))
      x(i) = tmp / Acopy(i,i)
    end do
  end subroutine
end module