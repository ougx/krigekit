"""Shared observation, empirical-cloud, and averaged-bin workflow."""

import numpy as np

from .variogram_empirical import avg_vgm, raw_vgm


class _VariogramModelBase:
    """Private base for marginal and space-time variogram workflow objects."""

    def __init__(self):
        """Initialize observation data and cached empirical results."""
        self.obs_coord = None
        self.obs_value = None
        self.obs_time = None
        self._raw = None
        self._avg = None
        self._dir = None
        self._params = None
        self._pcov = None
        self._fitted_model = None

    @property
    def raw_variogram_(self):
        """Raw empirical variogram cloud cached by :meth:`calc_experimental`."""
        return self._raw

    @raw_variogram_.setter
    def raw_variogram_(self, value):
        """Set the cached raw empirical variogram cloud."""
        self._raw = value

    @property
    def avg_variogram_(self):
        """Averaged variogram table cached by :meth:`calc_average`."""
        return self._avg

    @avg_variogram_.setter
    def avg_variogram_(self, value):
        """Set the cached averaged variogram table."""
        self._avg = value

    @property
    def directional_variogram_(self):
        """Directional averaged variogram cached by :meth:`calc_directional_average`."""
        return self._dir

    @directional_variogram_.setter
    def directional_variogram_(self, value):
        """Set the cached directional averaged variogram table."""
        self._dir = value

    @property
    def params_(self):
        """Fitted flat parameter vector, or ``None`` before :meth:`fit`."""
        return self._params

    @property
    def pcov_(self):
        """Approximate fitted parameter covariance, or ``None`` before fitting."""
        return self._pcov

    @property
    def fitted_model_(self):
        """Most recent fitted model returned by :meth:`fit`."""
        return self._fitted_model

    def _clear_fit_state(self):
        """Clear cached fit outputs after model inputs or template change."""
        self._params = None
        self._pcov = None
        self._fitted_model = None

    def set_obs(self, coord, value, times=None):
        """Store observations for empirical variogram calculation.

        Parameters
        ----------
        coord : array-like, shape (n, dim)
            Observation coordinates.  One-dimensional coordinate vectors are
            reshaped to ``(n, 1)``.
        value : array-like, shape (n,)
            Observation values for the direct variogram.
        times : array-like, optional
            Optional per-observation time stamps for space-time variograms.

        Returns
        -------
        VariogramModel
            ``self``, so calls can be chained.
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

        self.obs_coord = coord
        self.obs_value = value
        self.obs_time = times
        self._raw = None
        self._avg = None
        self._dir = None
        self._clear_fit_state()
        return self

    def experimental(self, store: bool = True, **kwargs):
        """Compute the raw empirical variogram cloud from stored observations.

        Extra keyword arguments are forwarded to :func:`raw_vgm`.  If
        ``times`` is not supplied explicitly, the time vector passed to
        :meth:`set_obs` is used.

        Model structures configured by ``set_vgm`` do not implicitly transform
        the empirical cloud.  Pass ``anisotropy=dict(anis1=..., anis2=...,
        azimuth=..., dip=..., plunge=...)`` explicitly when pair selection
        should use equivalent major-axis distance.  The cloud then contains
        both physical ``distance`` and ``anisotropic_distance`` columns.
        """
        if self.obs_coord is None or self.obs_value is None:
            raise RuntimeError("call set_obs() before experimental()")
        kwargs.setdefault("times", self.obs_time)
        cloud = raw_vgm(self.obs_coord, self.obs_value, **kwargs)
        if store:
            self._raw = cloud
            self._avg = None
            self._dir = None
        return cloud

    def calc_experimental(self, store: bool = True, **kwargs):
        """Compute the raw empirical variogram cloud from stored observations.

        This is the preferred verb-style alias for :meth:`experimental`.
        """
        return self.experimental(store=store, **kwargs)

    def calc_empirical(self, store: bool = True, **kwargs):
        """Alias for :meth:`calc_experimental`."""
        return self.calc_experimental(store=store, **kwargs)

    def average(self, rawvgm=None, store: bool = True, raw_kwargs=None, **kwargs):
        """Average a variogram cloud into lag bins.

        Parameters
        ----------
        rawvgm : pandas.DataFrame, optional
            Existing raw variogram cloud.  If omitted, the cached cloud from a
            previous :meth:`experimental` call is used; if no cache exists, a
            new cloud is computed from stored observations.
        store : bool, optional
            Store the averaged table as ``avg_variogram_``.
        raw_kwargs : dict, optional
            Keyword arguments passed to :meth:`experimental` if a new raw cloud
            must be computed.
        **kwargs
            Forwarded to :func:`avg_vgm`.
        """
        if rawvgm is None:
            if self._raw is None:
                rawvgm = self.experimental(**(raw_kwargs or {}))
            else:
                rawvgm = self._raw
        avg = avg_vgm(rawvgm, **kwargs)
        if store:
            self._avg = avg
        return avg

    def calc_average(self, rawvgm=None, store: bool = True, raw_kwargs=None, **kwargs):
        """Preferred verb-style alias for :meth:`average`."""
        return self.average(
            rawvgm=rawvgm,
            store=store,
            raw_kwargs=raw_kwargs,
            **kwargs,
        )

