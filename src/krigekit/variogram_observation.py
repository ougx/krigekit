"""One variable's observation and neighbour-search configuration.

``ObservationSet`` carries everything :meth:`krigekit.Kriging.set_obs` and
``set_obs_drift`` need, plus the per-variable search settings transferred by
``VariogramSystem.apply``.  It holds data only -- the Python side does not
perform neighbour selection; that stays in the Fortran engine.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class ObservationSet:
    """Observation data and search configuration for one variable."""

    coord: np.ndarray | None = None
    value: np.ndarray | None = None
    times: np.ndarray | None = None
    variance: np.ndarray | None = None
    nmax: int | None = None
    maxdist: float | None = None
    search_anis1: float = 1.0
    search_anis2: float = 1.0
    search_azimuth: float = 0.0
    search_dip: float = 0.0
    search_plunge: float = 0.0
    search_time_at: float | None = None
    sector_search: bool = False
    sk_mean: float | None = None
    drift: np.ndarray | None = None

    # -- derived sizes ------------------------------------------------------
    @property
    def configured(self):
        """True once valid coordinates and values are present."""
        return self.coord is not None and self.value is not None

    @property
    def nobs(self):
        """Number of observation rows (zero before :meth:`set`)."""
        return 0 if self.value is None else len(self.value)

    @property
    def ndim(self):
        """Spatial dimension (zero before :meth:`set`)."""
        return 0 if self.coord is None else self.coord.shape[1]

    @property
    def ndrift(self):
        """Number of external-drift columns (zero when drift is absent)."""
        return 0 if self.drift is None else self.drift.shape[1]

    # -- configuration ------------------------------------------------------
    def set(self, coord, value, times=None, variance=None, sk_mean=None,
            drift=None):
        """Set observation data, validating shapes.

        ``coord``/``value`` are required.  ``times``, ``variance`` and ``drift``
        are per-observation and replace any previous values (``None`` clears
        them).  ``sk_mean`` is updated only when provided, so it survives a
        data-only re-:meth:`set`.
        """
        coord = np.asarray(coord, dtype=float)
        value = np.asarray(value, dtype=float).reshape(-1)
        if coord.ndim == 1:
            coord = coord.reshape(-1, 1)
        if coord.ndim != 2:
            raise ValueError("coord must have shape (n, dim)")
        if coord.shape[0] != len(value):
            raise ValueError("coord and value must have matching lengths")
        if coord.shape[1] not in (1, 2, 3):
            raise ValueError("coordinates must be 1D, 2D or 3D")
        if times is not None:
            times = np.asarray(times, dtype=float).reshape(-1)
            if len(times) != len(value):
                raise ValueError("times and value must have matching lengths")
        if variance is not None:
            variance = np.asarray(variance, dtype=float).reshape(-1)
            if len(variance) != len(value):
                raise ValueError("variance and value must have matching lengths")
        if drift is not None:
            drift = np.asarray(drift, dtype=float)
            if drift.ndim == 1:
                drift = drift.reshape(-1, 1)
            if drift.ndim != 2 or drift.shape[0] != len(value):
                raise ValueError("drift must have shape (n,) or (n, ndrift)")

        self.coord = coord
        self.value = value
        self.times = times
        self.variance = variance
        self.drift = drift
        if sk_mean is not None:
            self.sk_mean = float(sk_mean)
        return self

    def set_search(self, nmax=None, maxdist=None, anis1=1.0, anis2=1.0,
                   azimuth=0.0, dip=0.0, plunge=0.0, sector_search=False,
                   time_at=None):
        """Configure neighbour search (stored for transfer to the engine)."""
        if anis1 <= 0.0 or anis2 <= 0.0:
            raise ValueError("search anis1 and anis2 must be positive")
        if time_at is not None and time_at <= 0.0:
            raise ValueError("search time_at must be positive")
        if nmax is not None:
            self.nmax = int(nmax)
        if maxdist is not None:
            self.maxdist = float(maxdist)
        self.search_anis1 = float(anis1)
        self.search_anis2 = float(anis2)
        self.search_azimuth = float(azimuth)
        self.search_dip = float(dip)
        self.search_plunge = float(plunge)
        self.sector_search = bool(sector_search)
        self.search_time_at = None if time_at is None else float(time_at)
        return self

    def clear(self):
        """Reset all data and search configuration to defaults."""
        self.coord = self.value = self.times = self.variance = None
        self.nmax = self.maxdist = self.sk_mean = self.drift = None
        self.search_anis1 = self.search_anis2 = 1.0
        self.search_azimuth = self.search_dip = self.search_plunge = 0.0
        self.search_time_at = None
        self.sector_search = False
        return self

    # -- validation and queries --------------------------------------------
    def validate(self, ndim=None):
        """Validate configured data; optionally require a spatial dimension."""
        if not self.configured:
            raise RuntimeError("observation has no coordinates/values; call set()")
        if self.coord.shape[0] != len(self.value):
            raise ValueError("coord and value lengths disagree")
        if ndim is not None and self.ndim != ndim:
            raise ValueError(f"expected {ndim}D coordinates, got {self.ndim}D")
        return self

    def is_collocated_with(self, other):
        """Return true when two variables share coordinates and times."""
        if not self.configured or not other.configured:
            return False
        if (self.coord.shape != other.coord.shape
                or not np.allclose(self.coord, other.coord)):
            return False
        ti, tj = self.times, other.times
        if ti is None or tj is None:
            return ti is None and tj is None
        return ti.shape == tj.shape and np.allclose(ti, tj)

    # -- engine transfer ----------------------------------------------------
    def apply_to(self, kriging, ivar):
        """Transfer observations (and drift) to a kriging object for ``ivar``."""
        self.validate()
        kwargs = {}
        if self.variance is not None:
            kwargs["variance"] = self.variance
        if self.sk_mean is not None:
            kwargs["sk_mean"] = self.sk_mean
        kriging.set_obs(ivar, self.coord, self.value, **kwargs)
        if self.drift is not None:
            kriging.set_obs_drift(ivar, self.drift)
        return kriging


__all__ = ["ObservationSet"]
