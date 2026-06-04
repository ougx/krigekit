"""
Multi-event universal kriging (MEUK) — Fortran-backed implementation.

This module provides :class:`MEUKFortran`, which delivers the same public API
as :class:`~pykriging.meuk.MEUK` but delegates all numerical work to the
Fortran cokriging engine via the augmented-drift formulation.

How it maps to a standard cokriging call
-----------------------------------------
Each sampling event becomes one co-kriging variable (``ivar``).  The Fortran
engine is configured with::

    Kriging(ndim=..., nvar=m, ndrift=q_total+r, unbias=0)

where

* ``m``       = number of events
* ``q_total`` = Σ q_k  (sum of local drift columns across all events)
* ``r``       = number of global drift columns
* ``unbias=0`` suppresses the automatic intercept row so that the
  user-supplied drift IS the complete design matrix Y.

The observation drift for event k has shape ``(n_k, q_total+r)``::

    cols 0 … Σ_{j<k} q_j − 1          : zeros  (other events' local drifts)
    cols Σ_{j≤k} q_j − q_k … − 1      : V^(k)  (event k's local drift)
    cols q_total … q_total+r−1         : W^(k)  (global drift, same for all)

Because the Fortran assembles the constraint rows by writing each variable's
drift block into the *same* shared rows of matA (assemble_lhs, line 1910-12),
every column of the augmented drift matrix becomes one *shared* Lagrange
constraint summed across all events — which is exactly MEUK's Eq. 11 design
matrix (Tonkin et al. 2016).

Reference
---------
Tonkin et al. (2016). Multi-event universal kriging (MEUK).
Advances in Water Resources, 87, 92–105.
"""

from __future__ import annotations

import hashlib
import numpy as np
from collections import OrderedDict
from typing import Optional

from pykriging.kriging import Kriging


class MEUKFortran:
    """
    MEUK implemented via the pyKriging Fortran cokriging engine.

    The public interface mirrors :class:`~pykriging.meuk.MEUK` so both
    backends can be used interchangeably.

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions (1, 2, or 3).

    Examples
    --------
    >>> m = MEUKFortran(ndim=2)
    >>> m.set_variogram('sph', nugget=0.01, sill=0.49, a_major=200.)
    >>> m.add_event(0, coords_e1, vals_e1,
    ...             local_drift=np.ones(n1), global_drift=pump_e1)
    >>> m.add_event(1, coords_e2, vals_e2,
    ...             local_drift=np.ones(n2), global_drift=pump_e2)
    >>> m.build()
    >>> z_hat, sigma2 = m.predict(
    ...     target_event=0,
    ...     pred_coords=grid_xy,
    ...     pred_local_drift=np.ones(ngrid),
    ...     pred_global_drift=pump_grid_e0,
    ... )
    """

    def __init__(self, ndim: int = 2) -> None:
        if ndim not in (1, 2, 3):
            raise ValueError("ndim must be 1, 2, or 3.")
        self.ndim = ndim

        # Variogram: list of (args, kwargs) from each set_variogram call
        self._vgm_calls: list[dict] = []

        # Events: ordered so ivar = position + 1
        self._events: OrderedDict = OrderedDict()

        # Derived layout (populated by build())
        self._event_order: list  = []   # [event_id, …] in insertion order
        self._q_k: list[int]     = []   # local drift width per event
        self._col_off: list[int] = []   # column offset of each event's local block
        self._q_total: int       = 0
        self._r: int             = 0
        self._ndrift: int        = 0
        self._n_total: int       = 0

        # The underlying Fortran kriging object (created by build())
        self._kriging: Optional[Kriging] = None

        # Result cache: {_CacheKey: (z_hat, sigma2)}
        # Keyed per-event so predict_all populates entries that predict() can reuse.
        self._cache: dict = {}

    # ------------------------------------------------------------------ #
    # Variogram                                                            #
    # ------------------------------------------------------------------ #

    def set_variogram(
        self,
        vtype: str,
        nugget: float = 0.0,
        sill: float = 1.0,
        a_major: float = 1.0,
        a_minor: Optional[float] = None,
        a_vert: Optional[float] = None,
        azimuth: float = 0.0,
        dip: float = 0.0,
        plunge: float = 0.0,
        append: bool = False,
    ) -> None:
        """Set (or append) a variogram structure — identical signature to
        :meth:`~pykriging.meuk.MEUK.set_variogram` and to
        :meth:`~pykriging.kriging.Kriging.set_vgm`.

        Parameters
        ----------
        vtype : str
            Model type: ``'sph'``, ``'exp'``, ``'gau'``, ``'lin'``, ``'nug'``.
        nugget : float
            Nugget sill (applied to the first structure only).
        sill : float
            Partial sill of this structure.
        a_major : float
            Major-axis range.
        a_minor : float or None
            Minor-axis range. Defaults to *a_major* (isotropic).
        a_vert : float or None
            Vertical range (3-D). Defaults to *a_minor*.
        azimuth, dip, plunge : float
            Anisotropy angles in degrees (same convention as :class:`Kriging`).
        append : bool
            ``True`` adds a nested structure; ``False`` (default) replaces.
        """
        if not append:
            self._vgm_calls = []
        self._vgm_calls.append(dict(
            vtype=vtype, nugget=nugget, sill=sill,
            a_major=a_major,
            a_minor=a_minor if a_minor is not None else a_major,
            a_vert=a_vert   if a_vert  is not None else (a_minor or a_major),
            azimuth=azimuth, dip=dip, plunge=plunge,
            append=append,
        ))
        self._kriging = None   # invalidate
        self._cache.clear()

    # ------------------------------------------------------------------ #
    # Events                                                               #
    # ------------------------------------------------------------------ #

    def add_event(
        self,
        event_id,
        coords: np.ndarray,
        values: np.ndarray,
        local_drift: np.ndarray,
        global_drift: Optional[np.ndarray] = None,
        obs_variance: Optional[np.ndarray] = None,
        nmax: Optional[int] = None,
        maxdist: Optional[float] = None,
    ) -> None:
        """Add a sampling event.

        Parameters
        ----------
        event_id : hashable
            Unique identifier (int, str, datetime, …).
        coords : array-like, shape (n, ndim)
            Observation locations.
        values : array-like, shape (n,)
            Observed values.
        local_drift : array-like, shape (n,) or (n, q)
            Local drift columns whose regression coefficients may vary between
            events.  Include a column of ones for an event-specific mean.
        global_drift : array-like, shape (n,) or (n, r), optional
            Global drift columns whose coefficient is *shared* across events.
        obs_variance : array-like, shape (n,), optional
            Per-observation measurement-error variance added to the diagonal
            of C^(k).
        nmax : int, optional
            Maximum neighbours used per prediction point. Default: all
            observations across *all* events.
        maxdist : float, optional
            Maximum search distance. Default: unlimited.
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

        obs_var = (np.asarray(obs_variance, dtype=float).ravel()
                   if obs_variance is not None else None)

        if event_id in self._events:
            raise KeyError(f"Event {event_id!r} already added.")

        self._events[event_id] = dict(
            coords=coords, values=values,
            V=V, W=W, obs_var=obs_var,
            nmax=nmax, maxdist=maxdist,
        )
        self._kriging = None   # invalidate
        self._cache.clear()

    # ------------------------------------------------------------------ #
    # Build (precompute equivalent)                                        #
    # ------------------------------------------------------------------ #

    def build(self) -> None:
        """Construct the underlying :class:`Kriging` object.

        Must be called after all :meth:`add_event` calls and before
        :meth:`predict` / :meth:`predict_all`.  Re-call if events or
        variogram parameters change.

        What happens inside
        -------------------
        1. Validate consistency (same r across events).
        2. Compute the augmented drift layout:
           ``ndrift = q_total + r``, ``unbias = 0``.
        3. Create ``Kriging(ndim, nvar=m, ndrift=ndrift, unbias=0)``.
        4. Register the auto-variogram for every event and zero
           cross-variograms for every off-diagonal pair.
        5. Pass observations + augmented drift to the Fortran engine.
        6. Build neighbour k-d trees (``set_search``).
        """
        if not self._vgm_calls:
            raise RuntimeError("Call set_variogram() before build().")
        if not self._events:
            raise RuntimeError("Add at least one event with add_event() before build().")

        self._cache.clear()   # new model → all cached predictions are stale

        # ── Layout ────────────────────────────────────────────────────
        event_order = list(self._events.keys())
        m = len(event_order)

        q_k   = [self._events[eid]['V'].shape[1] for eid in event_order]
        r_all = [self._events[eid]['W'].shape[1] for eid in event_order]
        if len(set(r_all)) != 1:
            raise ValueError("All events must have the same number of global "
                             "drift columns.")
        r       = r_all[0]
        q_total = sum(q_k)
        ndrift  = q_total + r
        n_total = sum(self._events[eid]['values'].size for eid in event_order)

        # Column offsets: where each event's local block starts
        col_off = [0] * m
        for k in range(1, m):
            col_off[k] = col_off[k - 1] + q_k[k - 1]

        # ── Kriging object ─────────────────────────────────────────────
        kg = Kriging(ndim=self.ndim, nvar=m, ndrift=ndrift, unbias=0)

        # Auto-variogram for every event (same model)
        for ivar in range(1, m + 1):
            for call in self._vgm_calls:
                kg.set_vgm(
                    ivar=ivar, jvar=ivar,
                    vtype=call['vtype'],
                    nugget=call['nugget'], sill=call['sill'],
                    a_major=call['a_major'],
                    a_minor1=call['a_minor'],
                    a_minor2=call['a_vert'],
                    azimuth=call['azimuth'],
                    dip=call['dip'],
                    plunge=call['plunge'],
                    append=call['append'],
                )

        # Zero cross-variograms for every off-diagonal pair
        for i in range(1, m + 1):
            for j in range(1, m + 1):
                if i != j:
                    kg.set_vgm(ivar=i, jvar=j,
                               vtype='nug', nugget=0.0, sill=0.0, a_major=1.0)

        # ── Observations + augmented drift ─────────────────────────────
        for k, eid in enumerate(event_order):
            ev   = self._events[eid]
            ivar = k + 1
            n_k  = ev['values'].size

            # nmax: default is total obs across all events
            nmax_k = ev['nmax'] if ev['nmax'] is not None else n_total

            kg.set_obs(
                ivar=ivar,
                coord=ev['coords'],
                value=ev['values'],
                variance=ev['obs_var'],
                nmax=nmax_k,
                maxdist=ev['maxdist'],
            )

            aug = self._aug_drift(k, q_k, col_off, q_total, r,
                                  ev['coords'][:, 0] if self.ndim == 1 else ev['coords'],
                                  ev['V'], ev['W'])
            kg.set_obs_drift(ivar=ivar, drift=aug)

        # ── Search trees ───────────────────────────────────────────────
        for ivar in range(1, m + 1):
            kg.set_search(ivar=ivar)

        # Store
        self._kriging    = kg
        self._event_order = event_order
        self._q_k        = q_k
        self._col_off    = col_off
        self._q_total    = q_total
        self._r          = r
        self._ndrift     = ndrift
        self._n_total    = n_total

    # ------------------------------------------------------------------ #
    # Augmented drift helpers                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _aug_drift(
        k_idx: int,
        q_k: list[int],
        col_off: list[int],
        q_total: int,
        r: int,
        coords,           # unused for now; kept for future coordinate-based drift
        V: np.ndarray,    # (n, q_k) local drift
        W: np.ndarray,    # (n, r)   global drift
    ) -> np.ndarray:
        """
        Build the ``(n, ndrift)`` augmented drift matrix for event *k_idx*.

        Layout::

            cols 0 … col_off[k]-1              : zeros (other events' local)
            cols col_off[k] … col_off[k]+q_k-1 : V^(k)
            cols col_off[k]+q_k … q_total-1    : zeros (other events' local)
            cols q_total … q_total+r-1          : W^(k)
        """
        n      = V.shape[0]
        ndrift = q_total + r
        D      = np.zeros((n, ndrift), dtype=float)
        c0     = col_off[k_idx]
        qk     = q_k[k_idx]
        D[:, c0 : c0 + qk] = V          # local block for this event
        if r > 0:
            D[:, q_total:]  = W          # global block (shared)
        return D

    def _aug_grid_drift(
        self,
        k_idx: int,
        V_g: np.ndarray,   # (ngrid, q_k) local drift at grid for event k
        W_g: np.ndarray,   # (ngrid, r)   global drift at grid
    ) -> np.ndarray:
        """Augmented grid drift for event *k_idx*."""
        return self._aug_drift(
            k_idx, self._q_k, self._col_off,
            self._q_total, self._r,
            None, V_g, W_g,
        )

    # ------------------------------------------------------------------ #
    # Cache helpers                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _digest(a: np.ndarray | None) -> bytes:
        """Compact MD5 fingerprint of a NumPy array (or None)."""
        if a is None:
            return b'\x00'
        h = hashlib.md5(usedforsecurity=False)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
        return h.digest()

    def _cache_key(
        self,
        event_id,
        pred_coords: np.ndarray,
        pred_local_drift: np.ndarray,
        pred_global_drift: np.ndarray | None,
    ) -> tuple:
        """
        Cache key for one event's prediction.

        Keyed on the *event identity* and the three input arrays that
        determine that event's result.  Non-target events' local drifts
        are deliberately excluded: as proven from the Fortran RHS assembly
        (``assemble_rhs``, line 1807), each target variable's RHS is
        built exclusively from ``block%drift(:, ivar, :)``, so the drift
        values set for other variables have no effect on this event's result.
        """
        return (
            event_id,
            self._digest(pred_coords),
            self._digest(pred_local_drift),
            self._digest(pred_global_drift),
        )

    def clear_cache(self) -> None:
        """Manually discard all cached prediction results."""
        self._cache.clear()

    # ------------------------------------------------------------------ #
    # Prediction                                                           #
    # ------------------------------------------------------------------ #

    def predict(
        self,
        target_event,
        pred_coords: np.ndarray,
        pred_local_drift: np.ndarray,
        pred_global_drift: Optional[np.ndarray] = None,
        nthread: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict at grid locations for one target event.

        This mirrors :meth:`~pykriging.meuk.MEUK.predict` exactly.
        Internally it calls :meth:`predict_all` with zero local drift for
        every non-target event (which does not affect the target event's
        result, because the Fortran RHS for each target variable is
        independent of the other variables' RHS vectors).

        Parameters
        ----------
        target_event : hashable
            Which event to predict (must match an *event_id* from
            :meth:`add_event`).
        pred_coords : array-like, shape (m_0, ndim)
            Prediction locations.
        pred_local_drift : array-like, shape (m_0,) or (m_0, q)
            Local drift values at prediction points *for the target event*.
        pred_global_drift : array-like, shape (m_0,) or (m_0, r), optional
            Global drift values at prediction points.
        nthread : int
            Number of OpenMP threads (0 = all available).

        Returns
        -------
        z_hat : ndarray, shape (m_0,)
        sigma2 : ndarray, shape (m_0,)  — kriging variance (≥ 0)
        """
        if target_event not in self._events:
            raise KeyError(f"Event {target_event!r} not found.")

        pred_coords = np.asarray(pred_coords, dtype=float)
        if pred_coords.ndim == 1:
            pred_coords = pred_coords[:, None]
        ngrid = len(pred_coords)

        V_g = np.asarray(pred_local_drift, dtype=float)
        if V_g.ndim == 1:
            V_g = V_g[:, None]

        W_g: np.ndarray | None = None
        if pred_global_drift is not None:
            W_g = np.asarray(pred_global_drift, dtype=float)
            if W_g.ndim == 1:
                W_g = W_g[:, None]

        # ── Cache lookup ──────────────────────────────────────────────
        key = self._cache_key(target_event, pred_coords, V_g, W_g)
        if key in self._cache:
            return self._cache[key]

        # ── Cache miss: run a solve for all variables ─────────────────
        # Non-target events receive zero local drift and the same global drift
        # as the target.  Neither affects target_event's result: the Fortran
        # RHS for each ivar is independent (assemble_rhs line 1807).
        per_event_V: dict = {}
        per_event_W: dict = {}
        for k, eid in enumerate(self._event_order):
            per_event_V[eid] = V_g if eid == target_event else np.zeros(
                (ngrid, self._q_k[k]), dtype=float)
            per_event_W[eid] = W_g   # same W for all; irrelevant for non-target

        all_results = self._solve(pred_coords, per_event_V, per_event_W, nthread)

        # Cache only the target event — other events used dummy local drift
        # so their predictions would be incorrect.
        self._cache[key] = all_results[target_event]
        return self._cache[key]

    def predict_all(
        self,
        pred_coords: np.ndarray,
        pred_local_drifts: dict,
        pred_global_drifts=None,
        nthread: int = 0,
    ) -> dict:
        """Predict all events simultaneously in a single Fortran solve call.

        Parameters
        ----------
        pred_coords : array-like, shape (m_0, ndim)
            Prediction locations (same for all events).
        pred_local_drifts : dict  {event_id: array-like (m_0, q_k)}
            Local drift values at prediction points for *each* event.
            Every event in the model must have an entry.
        pred_global_drifts : optional — one of:
            * ``None``                       — no global drift (r = 0)
            * ndarray, shape (m_0, r)        — *same* global drift for every event
            * dict {event_id: ndarray}       — *per-event* global drift values

            Use the dict form when the global covariate values differ between
            events (e.g., pumping rates Q_k vary while the coefficient 1/T is
            shared).  The cache key for each event is its own global drift
            array, so a subsequent ``predict()`` call with the same per-event
            drift will hit the cache directly.
        nthread : int
            Number of OpenMP threads.

        Returns
        -------
        dict  {event_id: (z_hat, sigma2)}
            *z_hat* and *sigma2* each have shape (m_0,).
        """
        if self._kriging is None:
            self.build()

        pred_coords = np.asarray(pred_coords, dtype=float)
        if pred_coords.ndim == 1:
            pred_coords = pred_coords[:, None]
        ngrid = len(pred_coords)

        # ── Normalise pred_global_drifts to a per-event dict ──────────
        W_per_event: dict = self._normalise_global_drifts(
            pred_global_drifts, ngrid)

        # ── Validate local drifts; check cache per event ───────────────
        coerced_V: dict = {}
        cached_results: dict = {}
        missing: list = []

        for k, eid in enumerate(self._event_order):
            if eid not in pred_local_drifts:
                raise KeyError(
                    f"pred_local_drifts is missing event {eid!r}. "
                    f"For single-event prediction use predict()."
                )
            V_g = np.asarray(pred_local_drifts[eid], dtype=float)
            if V_g.ndim == 1:
                V_g = V_g[:, None]
            if V_g.shape != (ngrid, self._q_k[k]):
                raise ValueError(
                    f"Event {eid!r}: pred_local_drifts shape {V_g.shape} "
                    f"expected ({ngrid}, {self._q_k[k]})."
                )
            coerced_V[eid] = V_g

            key = self._cache_key(eid, pred_coords, V_g, W_per_event[eid])
            if key in self._cache:
                cached_results[eid] = self._cache[key]
            else:
                missing.append(eid)

        # All events already cached — skip solve entirely
        if not missing:
            return cached_results

        # ── At least one event is missing — run one Fortran solve ──────
        all_results = self._solve(pred_coords, coerced_V, W_per_event, nthread)

        # Populate cache for every event in this call
        for eid in self._event_order:
            key = self._cache_key(eid, pred_coords, coerced_V[eid], W_per_event[eid])
            self._cache[key] = all_results[eid]

        # Merge with any pre-cached results and return
        all_results.update(cached_results)
        return all_results

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _normalise_global_drifts(self, pred_global_drifts, ngrid: int) -> dict:
        """
        Return a per-event dict ``{eid: W_g_or_None}`` from any of the three
        accepted forms of *pred_global_drifts*.
        """
        if pred_global_drifts is None:
            return {eid: None for eid in self._event_order}

        if isinstance(pred_global_drifts, dict):
            result: dict = {}
            for eid in self._event_order:
                if eid not in pred_global_drifts:
                    raise KeyError(
                        f"pred_global_drifts is missing event {eid!r}."
                    )
                w = np.asarray(pred_global_drifts[eid], dtype=float)
                if w.ndim == 1:
                    w = w[:, None]
                result[eid] = w
            return result

        # Shared ndarray — broadcast to all events
        w = np.asarray(pred_global_drifts, dtype=float)
        if w.ndim == 1:
            w = w[:, None]
        return {eid: w for eid in self._event_order}

    def _solve(
        self,
        pred_coords: np.ndarray,
        pred_local_drifts: dict,   # {eid: (ngrid, q_k)}
        pred_global_drifts: dict,  # {eid: (ngrid, r) | None}  — already normalised
        nthread: int,
    ) -> dict:
        """Configure the prediction grid, run the Fortran solve, return results."""
        kg    = self._kriging
        ngrid = len(pred_coords)
        empty_W = np.empty((ngrid, 0), dtype=float)

        # Set prediction grid
        kg.set_grid(coord=pred_coords)

        # Set augmented grid drift per event (ivar-specific RHS in Fortran)
        for k, eid in enumerate(self._event_order):
            ivar  = k + 1
            V_g   = pred_local_drifts[eid]                          # (ngrid, q_k)
            _w    = pred_global_drifts.get(eid)
            W_g   = empty_W if _w is None else _w                   # (ngrid, r)
            aug_g = self._aug_grid_drift(k, V_g, W_g)
            kg.set_grid_drift(drift=aug_g, ivar=ivar)

        kg.solve(nthread=nthread)

        # Extract results
        est_all, var_all = kg.get_results()
        # est_all: (ngrid, nvar) when nvar>1 and squeeze=True
        # var_all: (ngrid, nvar, nvar)

        results: dict = {}
        for k, eid in enumerate(self._event_order):
            z_hat  = est_all[:, k] if est_all.ndim == 2 else est_all
            sigma2 = var_all[:, k, k] if var_all.ndim == 3 else var_all
            results[eid] = (z_hat, np.maximum(sigma2, 0.0))

        return results

    # ------------------------------------------------------------------ #
    # Accessors                                                            #
    # ------------------------------------------------------------------ #

    @property
    def kriging(self) -> Kriging:
        """The underlying :class:`~pykriging.kriging.Kriging` object.

        Built on first access if :meth:`build` has not yet been called.
        """
        if self._kriging is None:
            self.build()
        return self._kriging

    def event_ids(self) -> list:
        """Ordered list of event identifiers."""
        return list(self._event_order)

    def ndrift(self) -> int:
        """Total number of drift columns in the augmented design matrix."""
        if self._kriging is None:
            self.build()
        return self._ndrift
