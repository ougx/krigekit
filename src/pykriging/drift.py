def _line_segment_geometry(x1, y1, x2, y2, x0, y0):
    dx = x2 - x1
    dy = y2 - y1
    L = np.hypot(dx, dy)

    if L == 0:
        raise ValueError("The two endpoints are identical.")

    tx = dx / L
    ty = dy / L

    ax = x1 - x0
    ay = y1 - y0

    p = ax * tx + ay * ty
    a2 = ax**2 + ay**2
    d2 = np.maximum(a2 - p**2, 0.0)
    d = np.sqrt(d2)

    bx = ax - p * tx
    by = ay - p * ty

    u1 = p
    u2 = p + L

    return L, tx, ty, p, d, bx, by, u1, u2
    
    
def integrate_log_r_line(x1, y1, x2, y2, x0=0.0, y0=0.0):"""
    Integrates ln(r) over a straight line segment from (x1, y1) to (x2, y2),
    where r is the distance from an arbitrary reference point P0(x0, y0).
    
    Parameters:
    -----------
    x1, y1 : float - Coordinates of the starting point (P1)
    x2, y2 : float - Coordinates of the ending point (P2)
    x0, y0 : float - Coordinates of the reference point (P0)
    
    Returns:
    --------
    float - The value of the line integral
    """
    L, tx, ty, p, d, bx, by, u1, u2 = _line_segment_geometry(
        x1, y1, x2, y2, x0, y0
    )

    eps = 1e-12

    def F_regular(u, d):
        return 0.5 * u * np.log(u*u + d*d) - u + d * np.arctan2(u, d)

    def F_collinear(u):
        return np.where(
            np.abs(u) < eps,
            0.0,
            u * np.log(np.abs(u)) - u
        )

    I_regular = F_regular(u2, d) - F_regular(u1, d)
    I_collinear = F_collinear(u2) - F_collinear(u1)

    return np.where(d > eps, I_regular, I_collinear)
    
def d_integrate_log_r_dx0(x1, y1, x2, y2, x0, y0):
    L, tx, ty, p, d, bx, by, u1, u2 = _line_segment_geometry(
        x1, y1, x2, y2, x0, y0
    )

    eps = 1e-12

    d2 = d * d
    u1sq = u1 * u1
    u2sq = u2 * u2

    regular = (
        -0.5 * tx * np.log((u2sq + d2) / (u1sq + d2))
        - np.where(d > eps, bx / d, 0.0)
        * (np.arctan2(u2, d) - np.arctan2(u1, d))
    )

    collinear = -tx * (
        np.log(np.abs(u2)) - np.log(np.abs(u1))
    )

    return np.where(d > eps, regular, collinear)

def d_integrate_log_r_dy0(x1, y1, x2, y2, x0, y0):
    L, tx, ty, p, d, bx, by, u1, u2 = _line_segment_geometry(
        x1, y1, x2, y2, x0, y0
    )

    eps = 1e-12

    d2 = d * d
    u1sq = u1 * u1
    u2sq = u2 * u2

    regular = (
        -0.5 * ty * np.log((u2sq + d2) / (u1sq + d2))
        - np.where(d > eps, by / d, 0.0)
        * (np.arctan2(u2, d) - np.arctan2(u1, d))
    )

    collinear = -ty * (
        np.log(np.abs(u2)) - np.log(np.abs(u1))
    )

    return np.where(d > eps, regular, collinear)