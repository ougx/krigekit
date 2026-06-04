"""
Multi-event universal kriging (MEUK).

Reference
---------
Tonkin, M.J., Kennel, J., Huber, W., Lambie, J.M. (2016).
Multi-event universal kriging (MEUK).
Advances in Water Resources, 87, 92-105.

Mathematical framework
----------------------
Each sampling event k has n_k observations.  The random field is:

    Z^(k)(s) = sum_a alpha_a^(k) v_a^(k)(s)   [local drift]
             + sum_b beta_b  w_b^(k)(s)         [global drift]
             + eps^(k)(s)                        [residual]

with:
  * a single covariance function C(h) shared across all events;
  * local drift columns V^(k) whose coefficients alpha^(k) vary per event;
  * global drift columns W^(k) whose coefficients beta are constant across
    events;
  * cross-event residual covariance Cov[eps^(k), eps^(l)] = 0 for k != l.

The MEUK kriging system (Tonkin et al. 2016, Eq. 13) for prediction at x_0
belonging to event k_1 has a bordered block-diagonal structure.  This module
solves it via the block-reduction in Eq. 16, which requires only:

  * one (n_k + q) x (n_k + q) factorisation per event  (precomputed once);
  * one r x r global solve per prediction point.

where q = number of local drift columns and r = number of global drift columns.

Usage
-----
>>> import numpy as np
>>> from pykriging.meuk import MEUK
>>>
>>> m = MEUK(ndim=2)
>>> m.set_variogram('sph', nugget=0.01, sill=0.49, a_major=100.)
>>>
>>> # add events: local drift = intercept, global drift = pumping response
>>> m.add_event(0, coords_e1, values_e1,
...             local_drift=np.ones(n1),
...             global_drift=pump_e1)
>>> m.add_event(1, coords_e2, values_e2,
...             local_drift=np.ones(n2),
...             global_drift=pump_e2)
>>>
>>> m.precompute()   # factorise per-event systems once
>>>
>>> z_hat, sigma2 = m.predict(
...     target_event=0,
...     pred_coords=grid_coords,
...     pred_local_drift=np.ones(n_grid),
...     pred_global_drift=pump_grid_e1,
... )
"""

import numpy as np
from scipy import linalg


class MEUK:
    """
    Multi-Event Universal Kriging (Tonkin et al. 2016).

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions (1, 2, or 3).
    """

    def __init__(self, ndim=2):
        if ndim not in (1, 2, 3):
            raise ValueError("ndim must be 1, 2, or 3.")
        self.ndim = ndim
        self._vgm: list[dict] = []
        self._nugget: float = 0.0
        self._events: dict = {}
        self._pre: dict = {}
        self._A_global_lu = None
        self._r: int = 0
        self._precomputed: bool = False

    # ------------------------------------------------------------------ #
    # Variogram                                                            #
    # ------------------------------------------------------------------ #

    def set_variogram(
        self,
        vtype: str,
        nugget: float = 0.0,
        sill: float = 1.0,
        a_major: float = 1.0,
        a_minor: float | None = None,
        a_vert: float | None = None,
        azimuth: float = 0.0,
        dip: float = 0.0,
        plunge: float = 0.0,
        append: bool = False,
    ):
        """Set (or append) a variogram structure.

        Parameters match ``Kriging.set_vgm()`` conventions so that parameters
        fitted with the existing pyKriging API can be passed directly.

        Parameters
        ----------
        vtype : str
            Model type: ``'sph'``, ``'exp'``, ``'gau'``, ``'lin'``, ``'nug'``.
        nugget : float
            Nugget sill (discontinuity at h=0).  Ignored when *append* is True.
        sill : float
            Partial sill of this structure.
        a_major : float
            Major-axis range.
        a_minor : float or None
            Minor-axis range.  Defaults to *a_major* (isotropic).
        a_vert : float or None
            Vertical range (3-D only).  Defaults to *a_minor*.
        azimuth, dip, plunge : float
            Anisotropy angles in degrees (same convention as pyKriging).
        append : bool
            If True, add this structure to a nested variogram.
            If False (default), replace any existing variogram.
        """
        if not append:
            self._vgm = []
            self._nugget = float(nugget)

        a_min = float(a_minor) if a_minor is not None else float(a_major)
        a_vrt = float(a_vert) if a_vert is not None else a_min

        self._vgm.append(
            {
                "type": vtype,
                "sill": float(sill),
                "a_major": float(a_major),
                "a_minor": a_min,
                "a_vert": a_vrt,
                "rot": self._build_rotation(azimuth, dip, plunge),
            }
        )

    def _build_rotation(self, azimuth, dip, plunge):
        az = np.radians(azimuth)
        dp = np.radians(dip)
        pl = np.radians(plunge)
        if self.ndim == 1:
            return np.eye(1)
        if self.ndim == 2:
            ca, sa = np.cos(az), np.sin(az)
            return np.array([[ca, sa], [-sa, ca]])
        # 3-D: azimuth (rotation about z), dip (about new x), plunge (about new y)
        caz, saz = np.cos(az), np.sin(az)
        cdp, sdp = np.cos(dp), np.sin(dp)
        cpl, spl = np.cos(pl), np.sin(pl)
        Raz = np.array([[caz, saz, 0], [-saz, caz, 0], [0, 0, 1]])
        Rdp = np.array([[1, 0, 0], [0, cdp, sdp], [0, -sdp, cdp]])
        Rpl = np.array([[cpl, 0, spl], [0, 1, 0], [-spl, 0, cpl]])
        return Raz @ Rdp @ Rpl

    def _normed_lag(self, d: np.ndarray, s: dict) -> float:
        """Anisotropically normalised lag for one variogram structure."""
        dt = s["rot"] @ d
        if self.ndim == 1:
            return abs(dt[0]) / s["a_major"]
        if self.ndim == 2:
            return float(np.sqrt((dt[0] / s["a_major"]) ** 2 + (dt[1] / s["a_minor"]) ** 2))
        return float(
            np.sqrt(
                (dt[0] / s["a_major"]) ** 2
                + (dt[1] / s["a_minor"]) ** 2
                + (dt[2] / s["a_vert"]) ** 2
            )
        )

    def _cov_at_zero(self) -> float:
        """C(0) = nugget + sum of partial sills."""
        return self._nugget + sum(s["sill"] for s in self._vgm)

    def _cov_h(self, d: np.ndarray) -> float:
        """Covariance for spatial difference vector *d*."""
        if np.all(d == 0.0):
            return self._cov_at_zero()
        c = 0.0
        for s in self._vgm:
            h = self._normed_lag(d, s)
            sill = s["sill"]
            vt = s["type"]
            if vt == "sph":
                c += sill * (1.0 - 1.5 * h + 0.5 * h**3) if h < 1.0 else 0.0
            elif vt == "exp":
                c += sill * np.exp(-3.0 * h)
            elif vt == "gau":
                c += sill * np.exp(-3.0 * h**2)
            elif vt == "lin":
                c += max(0.0, sill * (1.0 - h))
            elif vt == "nug":
                pass  # nugget contributes only at h=0, handled by _cov_at_zero
            else:
                raise ValueError(f"Unsupported variogram type: {vt!r}")
        return c

    def _build_cov_matrix(
        self,
        coords1: np.ndarray,
        coords2: np.ndarray | None = None,
        obs_var: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Build covariance matrix.

        When *coords2* is None the square obs-obs matrix is returned with
        C(0) + obs_var on the diagonal.
        """
        square = coords2 is None
        if square:
            coords2 = coords1
        n1, n2 = len(coords1), len(coords2)
        C = np.empty((n1, n2))
        C0 = self._cov_at_zero()
        for i in range(n1):
            for j in range(n2):
                if square and i == j:
                    C[i, j] = C0 + (obs_var[i] if obs_var is not None else 0.0)
                else:
                    C[i, j] = self._cov_h(coords1[i] - coords2[j])
        if square:
            C = 0.5 * (C + C.T)
        return C

    # ------------------------------------------------------------------ #
    # Events                                                               #
    # ------------------------------------------------------------------ #

    def add_event(
        self,
        event_id,
        coords: np.ndarray,
        values: np.ndarray,
        local_drift: np.ndarray,
        global_drift: np.ndarray | None = None,
        obs_variance: np.ndarray | None = None,
    ):
        """Add a sampling event.

        Parameters
        ----------
        event_id : hashable
            Unique identifier for this event (int, str, datetime, …).
        coords : array-like, shape (n, ndim)
            Observation locations.
        values : array-like, shape (n,)
            Observed values.
        local_drift : array-like, shape (n,) or (n, q)
            Local drift columns.  Each column corresponds to a covariate whose
            regression coefficient is estimated *separately* for each event.
            Include a column of ones here if each event should have its own mean.
        global_drift : array-like, shape (n,) or (n, r), optional
            Global drift columns.  Each column corresponds to a covariate whose
            regression coefficient is *shared* across all events.  If None,
            no global drift is used (pure local model = independent UK per event).
        obs_variance : array-like, shape (n,), optional
            Measurement-error variance added to the diagonal of C^(k).
        """
        coords = np.asarray(coords, dtype=float)
        if coords.ndim == 1:
            coords = coords[:, None]
        values = np.asarray(values, dtype=float).ravel()
        n = len(values)

        V = np.asarray(local_drift, dtype=float)
        if V.ndim == 1:
            V = V[:, None]

        if global_drift is None:
            W = np.empty((n, 0), dtype=float)
        else:
            W = np.asarray(global_drift, dtype=float)
            if W.ndim == 1:
                W = W[:, None]

        obs_var = np.asarray(obs_variance, dtype=float).ravel() if obs_variance is not None else None

        self._events[event_id] = dict(coords=coords, values=values, V=V, W=W, obs_var=obs_var)
        self._precomputed = False

    # ------------------------------------------------------------------ #
    # Precomputation (Tonkin et al. 2016, Eq. 13-16 setup)               #
    # ------------------------------------------------------------------ #

    def precompute(self):
        """Factorise per-event systems and build the global matrix A_global.

        Must be called after all events are added and before ``predict()``.
        Re-call if events or variogram parameters change.

        What is precomputed
        -------------------
        For each event k:

            Sigma^(k)  = [[C^(k),  V^(k)],    shape (n_k+q) x (n_k+q)
                          [V^(k)', 0      ]]

            X^(k)      = [[W^(k)],             shape (n_k+q) x r
                          [0_q×r ]]

            G^(k)      = Sigma^(k)^{-1} X^(k)  (n_k+q) x r
            A_k        = X^(k)' G^(k)           r x r

        Global:

            A_global   = sum_k A_k              r x r  (factorised for fast solves)
        """
        if not self._vgm:
            raise RuntimeError("Set a variogram with set_variogram() before calling precompute().")
        if not self._events:
            raise RuntimeError("Add at least one event with add_event() before calling precompute().")

        r = None
        q = None
        self._pre = {}

        for eid, ev in self._events.items():
            n_k = len(ev["values"])
            q_k = ev["V"].shape[1]
            r_k = ev["W"].shape[1]

            if q is None:
                q = q_k
            elif q != q_k:
                raise ValueError(
                    f"Event {eid!r}: local_drift has {q_k} columns; expected {q}."
                )
            if r is None:
                r = r_k
            elif r != r_k:
                raise ValueError(
                    f"Event {eid!r}: global_drift has {r_k} columns; expected {r}."
                )

            # Build Sigma^(k) = [[C, V], [V', 0]]
            C_k = self._build_cov_matrix(ev["coords"], obs_var=ev["obs_var"])
            sz = n_k + q_k
            Sigma_k = np.zeros((sz, sz))
            Sigma_k[:n_k, :n_k] = C_k
            Sigma_k[:n_k, n_k:] = ev["V"]
            Sigma_k[n_k:, :n_k] = ev["V"].T

            # LU factorisation of the symmetric indefinite Sigma^(k)
            lu, piv = linalg.lu_factor(Sigma_k)

            if r_k > 0:
                # X^(k) = [[W^(k)], [0_q×r]]
                X_k = np.zeros((sz, r_k))
                X_k[:n_k, :] = ev["W"]

                # G^(k) = Sigma^{-1}(k) X^(k)
                G_k = linalg.lu_solve((lu, piv), X_k)

                # A_k = X^(k)' G^(k)  (r x r)
                A_k = X_k.T @ G_k
            else:
                G_k = np.empty((sz, 0))
                A_k = np.empty((0, 0))

            self._pre[eid] = dict(lu=lu, piv=piv, G=G_k, A=A_k, n=n_k, q=q_k)

        self._r = r if r is not None else 0
        self._q = q if q is not None else 0

        # Global matrix A_global = sum_k A_k  and factorise once
        if self._r > 0:
            A_global = sum(p["A"] for p in self._pre.values())
            self._A_global_lu = linalg.lu_factor(A_global)
        else:
            self._A_global_lu = None

        self._precomputed = True

    # ------------------------------------------------------------------ #
    # Prediction (Tonkin et al. 2016, Eq. 16-17)                         #
    # ------------------------------------------------------------------ #

    def predict(
        self,
        target_event,
        pred_coords: np.ndarray,
        pred_local_drift: np.ndarray,
        pred_global_drift: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict at grid locations for one target event.

        Parameters
        ----------
        target_event : hashable
            Which event to predict.  Must match an *event_id* passed to
            ``add_event()``.
        pred_coords : array-like, shape (m, ndim)
            Prediction locations.
        pred_local_drift : array-like, shape (m,) or (m, q)
            Local drift values at the prediction points *for the target event*.
        pred_global_drift : array-like, shape (m,) or (m, r), optional
            Global drift values at the prediction points.

        Returns
        -------
        z_hat : ndarray, shape (m,)
            Kriging predictions.
        sigma2 : ndarray, shape (m,)
            Kriging variances (floored at zero).

        Algorithm (per prediction point x_0 in event k_1)
        --------------------------------------------------
        1.  c_0   = covariances from event k_1 observations to x_0
        2.  u_t   = [c_0, v_0]          (n_k1 + q vector)
        3.  alpha = Sigma^{-1}(k_1) u_t (using precomputed LU)
        4.  b     = -w_0 + G_t' u_t     (r vector; G_t' u_t = X^(k_1)' alpha)
        5.  theta = A_global^{-1} b      (r vector; precomputed A_global)
        6.  rho_t = alpha - G_t @ theta  (n_k1 + q vector)
        7.  lam_t = rho_t[:n_k1]         (kriging weights for target event)
        8.  lam_k = -G_k[:n_k] @ theta  (weights for non-target events k != k_1)
        9.  z_hat = lam_t @ z_t + sum_{k!=k_1} lam_k @ z_k
        10. sigma2 = C(0) - u_t @ rho_t - w_0 @ theta
        """
        if not self._precomputed:
            raise RuntimeError("Call precompute() before predict().")
        if target_event not in self._events:
            raise KeyError(f"Event {target_event!r} not found.")

        ev_t = self._events[target_event]
        pre_t = self._pre[target_event]
        n_t, q, r = pre_t["n"], pre_t["q"], self._r

        pred_coords = np.asarray(pred_coords, dtype=float)
        if pred_coords.ndim == 1:
            pred_coords = pred_coords[:, None]
        m0 = len(pred_coords)

        pred_local_drift = np.asarray(pred_local_drift, dtype=float)
        if pred_local_drift.ndim == 1:
            pred_local_drift = pred_local_drift[:, None]

        if pred_global_drift is None:
            pred_global_drift = np.empty((m0, 0), dtype=float)
        else:
            pred_global_drift = np.asarray(pred_global_drift, dtype=float)
            if pred_global_drift.ndim == 1:
                pred_global_drift = pred_global_drift[:, None]

        C0 = self._cov_at_zero()
        z_hat = np.empty(m0)
        sigma2 = np.empty(m0)

        for i0 in range(m0):
            x0 = pred_coords[i0]
            v0 = pred_local_drift[i0]    # (q,)
            w0 = pred_global_drift[i0]   # (r,)

            # Step 1-2: RHS for target event
            c0 = np.array([self._cov_h(ev_t["coords"][j] - x0) for j in range(n_t)])
            u_t = np.concatenate([c0, v0])

            # Step 3: solve per-event system  Sigma^(k_1) alpha = u_t
            alpha = linalg.lu_solve((pre_t["lu"], pre_t["piv"]), u_t)

            if r > 0:
                # Step 4: b = -w_0 + G_t' u_t
                # G_t' u_t = X^(k_1)' Sigma^{-1}(k_1) u_t = X^(k_1)' alpha
                b = -w0 + pre_t["G"].T @ u_t

                # Step 5: theta = A_global^{-1} b
                theta = linalg.lu_solve(self._A_global_lu, b)
            else:
                theta = np.empty(0)

            # Step 6-7: adjusted weights for target event
            rho_t = alpha - pre_t["G"] @ theta   # (n_t + q,)
            lam_t = rho_t[:n_t]

            # Step 8-9: prediction = target event + contribution from others
            z_pred = lam_t @ ev_t["values"]
            if r > 0:
                for eid, ev in self._events.items():
                    if eid == target_event:
                        continue
                    pre_k = self._pre[eid]
                    # For k != k_1: u^(k) = 0, so alpha^(k) = 0
                    # rho^(k) = 0 - G^(k) theta  →  lam^(k) = -G^(k)[:n_k] @ theta
                    lam_k = -pre_k["G"][: pre_k["n"], :] @ theta
                    z_pred += lam_k @ ev["values"]

            z_hat[i0] = z_pred

            # Step 10: kriging variance  C(0) - u_t' rho_t - w_0' theta
            sigma2[i0] = C0 - u_t @ rho_t - (w0 @ theta if r > 0 else 0.0)

        return z_hat, np.maximum(sigma2, 0.0)

    # ------------------------------------------------------------------ #
    # Diagnostics                                                          #
    # ------------------------------------------------------------------ #

    def get_beta(self, target_event, at_coords, pred_local_drift, pred_global_drift=None):
        """
        Return the effective GLS trend-coefficient estimates at given locations.

        This exposes the MEUK-estimated local coefficients alpha^(k_1) and
        global coefficients beta (= theta in the Lagrange-multiplier sense) as
        they would be used for each prediction point.

        Returns
        -------
        dict with keys:
            'theta' : ndarray, shape (m, r) – global coefficient Lagrange multipliers
            'alpha_local' : ndarray, shape (m, q) – local coefficient Lagrange multipliers
        """
        if not self._precomputed:
            raise RuntimeError("Call precompute() first.")

        ev_t = self._events[target_event]
        pre_t = self._pre[target_event]
        n_t, q, r = pre_t["n"], pre_t["q"], self._r

        pred_coords = np.asarray(at_coords, dtype=float)
        if pred_coords.ndim == 1:
            pred_coords = pred_coords[:, None]
        m0 = len(pred_coords)
        pred_local_drift = np.asarray(pred_local_drift, dtype=float)
        if pred_local_drift.ndim == 1:
            pred_local_drift = pred_local_drift[:, None]
        if pred_global_drift is None:
            pred_global_drift = np.empty((m0, 0), dtype=float)
        else:
            pred_global_drift = np.asarray(pred_global_drift, dtype=float)
            if pred_global_drift.ndim == 1:
                pred_global_drift = pred_global_drift[:, None]

        thetas = np.empty((m0, r))
        alphas = np.empty((m0, q))

        for i0 in range(m0):
            x0 = pred_coords[i0]
            v0 = pred_local_drift[i0]
            w0 = pred_global_drift[i0]

            c0 = np.array([self._cov_h(ev_t["coords"][j] - x0) for j in range(n_t)])
            u_t = np.concatenate([c0, v0])
            alpha = linalg.lu_solve((pre_t["lu"], pre_t["piv"]), u_t)

            if r > 0:
                b = -w0 + pre_t["G"].T @ u_t
                theta = linalg.lu_solve(self._A_global_lu, b)
            else:
                theta = np.empty(0)

            rho_t = alpha - pre_t["G"] @ theta
            thetas[i0] = theta
            alphas[i0] = rho_t[n_t:]   # Lagrange multipliers for local drift

        return {"theta": thetas, "alpha_local": alphas}
