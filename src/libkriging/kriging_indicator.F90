!==============================================================================
! kriging_indicator.F90
!
! t_kriging_indicator — extends t_kriging for Multiple Indicator Kriging (MIK)
! and Sequential Indicator Simulation (SIS).
!
! Each variable (ivar = 1..nvar = K) represents a binary indicator for one
! category (or one threshold class for continuous data).  The K indicators
! are kriged jointly; after solving:
!
!   Estimation (nsim=0)
!     block%value(1, 1:K, ib) = K probability estimates.
!     post_solve_indicator clips each to [0,1] and normalises the K values
!     to sum to 1, producing a valid probability simplex.
!
!   Simulation (nsim>0, SIS)
!     block%sample(isim, 1, ib) = U(0,1) draw (set by prepare_indicator;
!       replaces the N(0,1) draws written by set_sim_base).
!     sim_draw_indicator converts the K conditional expectations into a
!     binary one-hot draw by walking the CDF using that U(0,1) variate.
!     The binary result is stored back into block%value, so that subsequent
!     blocks condition on binary indicators — not on probabilities.
!==============================================================================
module kriging_indicator
  use kriging,      only: t_kriging
  use kriging_base, only: t_kriging_ctx
  use utils,        only: r8vec_uniform_01
  use kriging_err,  only: kriging_error, kriging_failed
  implicit none
  private

  public :: t_kriging_indicator

  type, extends(t_kriging) :: t_kriging_indicator
    ! ncat: number of indicator categories (first ncat variables in val(:)).
    ! Default 0 means "use nvar" (pure MIS, nvar == ncat).
    ! Set ncat < nvar to co-krige with nvar-ncat secondary continuous variables;
    ! sim_draw and post_solve then only act on val(1:ncat).
    integer :: ncat = 0
  contains
    procedure :: set_sim    => set_sim_indicator
    procedure :: prepare    => prepare_indicator
    procedure :: sim_draw   => sim_draw_indicator
    procedure :: post_solve => post_solve_indicator
  end type t_kriging_indicator

contains

  !============================================================================
  ! set_sim_indicator
  !
  ! Overrides set_sim_base to supply U(0,1) draws for CDF-inversion in
  ! sim_draw_indicator, rather than the N(0,1) draws used by plain SGSIM.
  !
  ! When no sample is provided: delegates all setup (randpath, allocation) to
  ! the parent set_sim_base — which fills block%sample with N(0,1) — then
  ! immediately overwrites every element with a U(0,1) variate.
  ! When sample is provided: delegates entirely to the parent (caller is
  ! responsible for supplying U(0,1) values).
  !============================================================================
  subroutine set_sim_indicator(self, randpath, sample)
    class(t_kriging_indicator), intent(inout) :: self
    integer, intent(in), optional :: randpath(:)
    real,    intent(in), optional :: sample(:,:,:)
    integer :: isim, ivar

    call self%t_kriging%set_sim(randpath=randpath, sample=sample)
    if (kriging_failed()) return
    if (present(sample)) return   ! caller-supplied — already stored, done

    do isim = 1, self%nsim
      do ivar = 1, self%nvar
        call r8vec_uniform_01(self%block%n, self%block%sample(isim, ivar, :))
      end do
    end do
  end subroutine set_sim_indicator


  !============================================================================
  ! prepare_indicator
  !
  ! Calls parent prepare (variogram tables, KD-trees, search validation), then
  ! validates that block%sample holds U(0,1) draws for sim_draw_indicator.
  !
  ! set_search already enforces that set_sim was called first, so block%sample
  ! is guaranteed to be allocated here.  We only need to check the [0,1] range.
  !============================================================================
  subroutine prepare_indicator(self)
    class(t_kriging_indicator), intent(inout) :: self

    call self%t_kriging%prepare()
    if (self%nsim == 0) return

    if (minval(self%block%sample) < 0.0 .or. maxval(self%block%sample) > 1.0) then
      call kriging_error('prepare_indicator', &
        'SIS sample values must be in [0, 1]; pass U(0,1) draws to set_sim().')
    end if
  end subroutine prepare_indicator


  !============================================================================
  ! sim_draw_indicator
  !
  ! Replaces the default Gaussian perturbation with a categorical CDF draw.
  !
  ! On entry, val(1:K) holds the conditional mean (kriging estimate for each of
  ! the K indicator variables, with previously-simulated-block contributions
  ! already added by estimate_block_base).  L_chol is passed in by the base
  ! class but is not used here.
  !
  ! Algorithm:
  !   1. Clip each val(k) to [0,1] and normalise the K values to sum to 1.
  !   2. Draw u ~ U(0,1) from block%sample(isim, 1, iblock).
  !      (pre-populated by prepare_indicator; only the first-variable slice is
  !       used; a single U draw per block per realisation is sufficient.)
  !   3. Walk the CDF and record which category the draw falls in.
  !   4. Encode the drawn category as a binary one-hot vector in val.
  !
  ! The one-hot result is stored back into block%value by estimate_block_base,
  ! so future blocks condition on binary indicators, not probabilities.
  !============================================================================
  subroutine sim_draw_indicator(self, ctx, val, L_chol, isim)
    class(t_kriging_indicator), intent(inout) :: self
    type(t_kriging_ctx),        intent(in)    :: ctx
    real,                       intent(inout) :: val(:)      ! [nvar=K]
    real,                       intent(in)    :: L_chol(:,:) ! [nvar,nvar] — unused
    integer,                    intent(in)    :: isim

    real    :: prob(size(val)), cumsum, u
    integer :: k, drawn, kcat

    ! kcat: number of indicator categories to draw from.
    ! When ncat == 0 (default, pure MIS) use nvar; otherwise use ncat so that
    ! secondary variables in val(ncat+1:nvar) are ignored during the CDF draw.
    kcat = merge(self%ncat, self%nvar, self%ncat > 0 .and. self%ncat <= self%nvar)

    prob(:kcat) = max(0.0, min(1.0, val(:kcat)))
    if (sum(prob(:kcat)) > 0.0) then
      prob(:kcat) = prob(:kcat) / sum(prob(:kcat))
    else
      prob(:kcat) = 1.0 / real(kcat)
    end if

    u = self%block%sample(isim, 1, ctx%iblock)

    drawn  = kcat
    cumsum = 0.0
    do k = 1, kcat
      cumsum = cumsum + prob(k)
      if (u <= cumsum) then
        drawn = k
        exit
      end if
    end do

    val           = 0.0
    val(drawn)    = 1.0
  end subroutine sim_draw_indicator


  !============================================================================
  ! post_solve_indicator
  !
  ! Called once by solve_base after all blocks are estimated (or simulated).
  ! For estimation (nsim == 0): clips block%value(1, :, ib) to [0,1] and
  ! normalises the K probabilities at each block to sum to 1.
  ! For simulation (nsim > 0): no-op; binary one-hot values from sim_draw are
  ! already valid.
  !============================================================================
  subroutine post_solve_indicator(self)
    class(t_kriging_indicator), intent(inout) :: self
    real    :: s
    integer :: ib, kcat

    if (self%nsim > 0) return
    if (.not. associated(self%block) .or. self%block%n == 0) return

    kcat = merge(self%ncat, self%nvar, self%ncat > 0 .and. self%ncat <= self%nvar)

    do ib = 1, self%block%n
      self%block%value(1, 1:kcat, ib) = max(0.0, min(1.0, self%block%value(1, 1:kcat, ib)))
      s = sum(self%block%value(1, 1:kcat, ib))
      if (s > 0.0) self%block%value(1, 1:kcat, ib) = self%block%value(1, 1:kcat, ib) / s
    end do
  end subroutine post_solve_indicator

end module kriging_indicator
