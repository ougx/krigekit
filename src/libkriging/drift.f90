module linedrift_mod
    use, intrinsic :: iso_fortran_env, only: real64
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    implicit none

    private
    public :: dp
    public :: integrate_log_r_line
    public :: d_integrate_log_r_dx0
    public :: d_integrate_log_r_dy0

    integer, parameter :: dp = real64
    real(dp), parameter :: EPS = 1.0e-12_dp

contains

    elemental function safe_u_log_abs(u) result(val)
        !! Computes u * log(abs(u)), with limiting value 0 at u = 0.
        real(dp), intent(in) :: u
        real(dp) :: val

        if (abs(u) < EPS) then
            val = 0.0_dp
        else
            val = u * log(abs(u))
        end if
    end function safe_u_log_abs


    elemental subroutine line_segment_geometry( &
        x1, y1, x2, y2, x0, y0, &
        L, tx, ty, p, d, bx, by, u1, u2, ok)

        real(dp), intent(in) :: x1, y1, x2, y2
        real(dp), intent(in) :: x0, y0

        real(dp), intent(out) :: L
        real(dp), intent(out) :: tx, ty
        real(dp), intent(out) :: p
        real(dp), intent(out) :: d
        real(dp), intent(out) :: bx, by
        real(dp), intent(out) :: u1, u2
        logical, intent(out) :: ok

        real(dp) :: dx, dy
        real(dp) :: ax, ay
        real(dp) :: a2, d2

        dx = x2 - x1
        dy = y2 - y1

        L = sqrt(dx*dx + dy*dy)

        if (L < EPS) then
            ok = .false.
            tx = 0.0_dp
            ty = 0.0_dp
            p  = 0.0_dp
            d  = 0.0_dp
            bx = 0.0_dp
            by = 0.0_dp
            u1 = 0.0_dp
            u2 = 0.0_dp
            return
        end if

        ok = .true.

        tx = dx / L
        ty = dy / L

        ax = x1 - x0
        ay = y1 - y0

        p = ax * tx + ay * ty

        a2 = ax*ax + ay*ay
        d2 = a2 - p*p

        ! Protect against tiny negative values caused by roundoff.
        if (d2 < 0.0_dp .and. abs(d2) < EPS) then
            d2 = 0.0_dp
        end if

        d = sqrt(max(d2, 0.0_dp))

        bx = ax - p * tx
        by = ay - p * ty

        u1 = p
        u2 = p + L

    end subroutine line_segment_geometry


    elemental function integrate_log_r_line(x1, y1, x2, y2, x0, y0) result(I)
        !! Computes:
        !!
        !!     I = integral_segment ln(r) ds
        !!
        !! where r is distance from observation point (x0, y0)
        !! to a source point on the line segment.
        real(dp), intent(in) :: x1, y1, x2, y2
        real(dp), intent(in) :: x0, y0

        real(dp) :: I
        real(dp) :: L, tx, ty
        real(dp) :: p, d, bx, by
        real(dp) :: u1, u2
        real(dp) :: F1, F2
        logical :: ok

        call line_segment_geometry( &
            x1, y1, x2, y2, x0, y0, &
            L, tx, ty, p, d, bx, by, u1, u2, ok)

        if (.not. ok) then
            I = ieee_value(I, ieee_quiet_nan)
            return
        end if

        if (d > EPS) then

            F1 = 0.5_dp * u1 * log(u1*u1 + d*d) &
               - u1 &
               + d * atan2(u1, d)

            F2 = 0.5_dp * u2 * log(u2*u2 + d*d) &
               - u2 &
               + d * atan2(u2, d)

            I = F2 - F1

        else

            ! Collinear case:
            !
            !     integral ln(abs(u)) du = u ln(abs(u)) - u
            !
            ! with u ln(abs(u)) -> 0 as u -> 0.
            F1 = safe_u_log_abs(u1) - u1
            F2 = safe_u_log_abs(u2) - u2

            I = F2 - F1

        end if

    end function integrate_log_r_line


    elemental function d_integrate_log_r_dx0(x1, y1, x2, y2, x0, y0) result(dIdx0)
        !! Computes:
        !!
        !!     d/dx0 [ integral_segment ln(r) ds ]
        real(dp), intent(in) :: x1, y1, x2, y2
        real(dp), intent(in) :: x0, y0

        real(dp) :: dIdx0
        real(dp) :: L, tx, ty
        real(dp) :: p, d, bx, by
        real(dp) :: u1, u2
        real(dp) :: d2
        real(dp) :: term_log, term_atan
        logical :: ok

        call line_segment_geometry( &
            x1, y1, x2, y2, x0, y0, &
            L, tx, ty, p, d, bx, by, u1, u2, ok)

        if (.not. ok) then
            dIdx0 = ieee_value(dIdx0, ieee_quiet_nan)
            return
        end if

        if (d > EPS) then

            d2 = d * d

            term_log = -0.5_dp * tx * log((u2*u2 + d2) / (u1*u1 + d2))

            term_atan = -(bx / d) * (atan2(u2, d) - atan2(u1, d))

            dIdx0 = term_log + term_atan

        else

            ! Collinear case:
            !
            ! dI/dx0 = -tx * [log(abs(u2)) - log(abs(u1))]
            !
            ! Singular if observation point lies on the segment or endpoint.
            if (abs(u1) < EPS .or. abs(u2) < EPS) then
                dIdx0 = ieee_value(dIdx0, ieee_quiet_nan)
            else
                dIdx0 = -tx * (log(abs(u2)) - log(abs(u1)))
            end if

        end if

    end function d_integrate_log_r_dx0


    elemental function d_integrate_log_r_dy0(x1, y1, x2, y2, x0, y0) result(dIdy0)
        !! Computes:
        !!
        !!     d/dy0 [ integral_segment ln(r) ds ]
        real(dp), intent(in) :: x1, y1, x2, y2
        real(dp), intent(in) :: x0, y0

        real(dp) :: dIdy0
        real(dp) :: L, tx, ty
        real(dp) :: p, d, bx, by
        real(dp) :: u1, u2
        real(dp) :: d2
        real(dp) :: term_log, term_atan
        logical :: ok

        call line_segment_geometry( &
            x1, y1, x2, y2, x0, y0, &
            L, tx, ty, p, d, bx, by, u1, u2, ok)

        if (.not. ok) then
            dIdy0 = ieee_value(dIdy0, ieee_quiet_nan)
            return
        end if

        if (d > EPS) then

            d2 = d * d

            term_log = -0.5_dp * ty * log((u2*u2 + d2) / (u1*u1 + d2))

            term_atan = -(by / d) * (atan2(u2, d) - atan2(u1, d))

            dIdy0 = term_log + term_atan

        else

            ! Collinear case:
            !
            ! dI/dy0 = -ty * [log(abs(u2)) - log(abs(u1))]
            !
            ! Singular if observation point lies on the segment or endpoint.
            if (abs(u1) < EPS .or. abs(u2) < EPS) then
                dIdy0 = ieee_value(dIdy0, ieee_quiet_nan)
            else
                dIdy0 = -ty * (log(abs(u2)) - log(abs(u1)))
            end if

        end if

    end function d_integrate_log_r_dy0


    elemental subroutine d_integrate_log_r_dxy0( &
        x1, y1, x2, y2, x0, y0, dIdx0, dIdy0)
    
        !! Computes:
        !!
        !!     dIdx0 = d/dx0 [ integral_segment ln(r) ds ]
        !!     dIdy0 = d/dy0 [ integral_segment ln(r) ds ]
        !!
        !! where r is distance from observation point (x0, y0)
        !! to a source point on the line segment.
    
        real(dp), intent(in)  :: x1, y1, x2, y2
        real(dp), intent(in)  :: x0, y0
        real(dp), intent(out) :: dIdx0, dIdy0
    
        real(dp) :: L, tx, ty
        real(dp) :: p, d, bx, by
        real(dp) :: u1, u2
        real(dp) :: d2
        real(dp) :: log_term
        real(dp) :: atan_term
        logical :: ok
    
        call line_segment_geometry( &
            x1, y1, x2, y2, x0, y0, &
            L, tx, ty, p, d, bx, by, u1, u2, ok)
    
        if (.not. ok) then
            dIdx0 = ieee_value(dIdx0, ieee_quiet_nan)
            dIdy0 = ieee_value(dIdy0, ieee_quiet_nan)
            return
        end if
    
        if (d > EPS) then
    
            d2 = d * d
    
            log_term = log((u2*u2 + d2) / (u1*u1 + d2))
    
            atan_term = atan2(u2, d) - atan2(u1, d)
    
            dIdx0 = -0.5_dp * tx * log_term - (bx / d) * atan_term
            dIdy0 = -0.5_dp * ty * log_term - (by / d) * atan_term
    
        else
    
            ! Collinear case:
            !
            ! dI/dx0 = -tx * [log(abs(u2)) - log(abs(u1))]
            ! dI/dy0 = -ty * [log(abs(u2)) - log(abs(u1))]
            !
            ! This is singular if the observation point lies exactly at
            ! an endpoint, or if either u1 or u2 is zero.
    
            if (abs(u1) < EPS .or. abs(u2) < EPS) then
                dIdx0 = ieee_value(dIdx0, ieee_quiet_nan)
                dIdy0 = ieee_value(dIdy0, ieee_quiet_nan)
            else
                log_term = log(abs(u2)) - log(abs(u1))
    
                dIdx0 = -tx * log_term
                dIdy0 = -ty * log_term
            end if
    
        end if
    
    end subroutine d_integrate_log_r_dxy0
end module linedrift_mod