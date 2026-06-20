"""One flat, engine-compatible theoretical variogram component."""

from dataclasses import asdict, dataclass, fields, replace

import numpy as np

from .variogram_geometry import calc_anisotropic_lag
from .variogram_kernels import calc_cov, resolve_model


@dataclass
class VgmComponent:
    """One nested variogram component.

    The public representation deliberately matches the flat arguments accepted
    by :meth:`krigekit.Kriging.set_vgm`.  The Fortran engine groups the
    anisotropy fields into its internal ``vgm_aniso`` type after transfer.
    """

    vtype: str
    nugget: float = 0.0
    sill: float = 1.0
    a_major: float = 1.0
    a_minor1: float | None = None
    a_minor2: float | None = None
    azimuth: float = 0.0
    dip: float = 0.0
    plunge: float = 0.0
    product: bool = False

    def __post_init__(self):
        """Normalize aliases/default ranges and validate the component."""
        self.vtype = resolve_model(self.vtype)
        self.nugget = float(self.nugget)
        self.sill = float(self.sill)
        self.a_major = float(self.a_major)
        if self.a_minor1 is None:
            self.a_minor1 = self.a_major
        if self.a_minor2 is None:
            self.a_minor2 = self.a_minor1
        self.a_minor1 = float(self.a_minor1)
        self.a_minor2 = float(self.a_minor2)
        self.azimuth = float(self.azimuth)
        self.dip = float(self.dip)
        self.plunge = float(self.plunge)
        self.product = bool(self.product)
        self.validate()

    def validate(self):
        """Validate finite parameters and strictly positive ranges."""
        numeric = {
            "nugget": self.nugget,
            "sill": self.sill,
            "a_major": self.a_major,
            "a_minor1": self.a_minor1,
            "a_minor2": self.a_minor2,
            "azimuth": self.azimuth,
            "dip": self.dip,
            "plunge": self.plunge,
        }
        invalid = [name for name, value in numeric.items() if not np.isfinite(value)]
        if invalid:
            raise ValueError(
                "variogram component parameters must be finite; invalid: "
                + ", ".join(invalid)
            )
        if self.a_major <= 0.0 or self.a_minor1 <= 0.0 or self.a_minor2 <= 0.0:
            raise ValueError("a_major, a_minor1 and a_minor2 must be positive")
        return self

    def copy(self, **changes):
        """Return a validated copy, optionally replacing selected fields."""
        return replace(self, **changes)

    def set_anisotropy(
        self,
        *,
        a_major=None,
        a_minor1=None,
        a_minor2=None,
        ratio_minor1=None,
        ratio_minor2=None,
        anis1=None,
        anis2=None,
        azimuth=None,
        dip=None,
        plunge=None,
    ):
        """Update ranges and angles in place.

        ``anis1`` and ``anis2`` are aliases for the minor/major ratios used by
        the kriging search API.
        """
        if anis1 is not None:
            if ratio_minor1 is not None:
                raise ValueError("pass either ratio_minor1 or anis1, not both")
            ratio_minor1 = anis1
        if anis2 is not None:
            if ratio_minor2 is not None:
                raise ValueError("pass either ratio_minor2 or anis2, not both")
            ratio_minor2 = anis2
        if a_minor1 is not None and ratio_minor1 is not None:
            raise ValueError("pass either a_minor1 or ratio_minor1, not both")
        if a_minor2 is not None and ratio_minor2 is not None:
            raise ValueError("pass either a_minor2 or ratio_minor2, not both")

        major = self.a_major if a_major is None else float(a_major)
        minor1 = self.a_minor1 if a_minor1 is None else float(a_minor1)
        minor2 = self.a_minor2 if a_minor2 is None else float(a_minor2)
        if ratio_minor1 is not None:
            minor1 = major * float(ratio_minor1)
        if ratio_minor2 is not None:
            minor2 = major * float(ratio_minor2)

        candidate = replace(
            self,
            a_major=major,
            a_minor1=minor1,
            a_minor2=minor2,
            azimuth=self.azimuth if azimuth is None else float(azimuth),
            dip=self.dip if dip is None else float(dip),
            plunge=self.plunge if plunge is None else float(plunge),
        )
        self.a_major = candidate.a_major
        self.a_minor1 = candidate.a_minor1
        self.a_minor2 = candidate.a_minor2
        self.azimuth = candidate.azimuth
        self.dip = candidate.dip
        self.plunge = candidate.plunge
        return self

    def anisotropy_dict(self):
        """Return ranges, ratios, and angles used by geometry helpers."""
        return {
            "a_major": self.a_major,
            "a_minor1": self.a_minor1,
            "a_minor2": self.a_minor2,
            "anis1": self.a_minor1 / self.a_major,
            "anis2": self.a_minor2 / self.a_major,
            "azimuth": self.azimuth,
            "dip": self.dip,
            "plunge": self.plunge,
        }

    def calc_anisotropic_distance(self, lag):
        """Return equivalent major-axis distance for lag vector(s)."""
        return calc_anisotropic_lag(
            lag,
            anis1=self.a_minor1 / self.a_major,
            anis2=self.a_minor2 / self.a_major,
            azimuth=self.azimuth,
            dip=self.dip,
            plunge=self.plunge,
        )

    def calc_covariance(self, distance):
        """Evaluate this component's covariance at scalar lag distance(s)."""
        distance = np.asarray(distance, dtype=float)
        base = calc_cov(
            self.vtype,
            distance,
            psill=self.sill,
            rng=self.a_major,
        )
        return np.where(distance <= 0.0, self.sill + self.nugget, base)

    def calc_covariance_lag(self, lag):
        """Evaluate covariance for coordinate lag vector(s), with anisotropy."""
        return self.calc_covariance(self.calc_anisotropic_distance(lag))

    @property
    def cov0(self):
        """Covariance at zero lag, including this component's nugget."""
        return self.sill + self.nugget

    def calc_variogram(self, distance):
        """Evaluate ``gamma(h) = C(0) - C(h)`` at scalar lag distance(s)."""
        return self.cov0 - self.calc_covariance(distance)

    def calc_variogram_lag(self, lag):
        """Evaluate the semivariogram for coordinate lag vector(s)."""
        return self.cov0 - self.calc_covariance_lag(lag)

    def to_flat_dict(self):
        """Return the stable flat representation used for engine transfer."""
        return asdict(self)

    @classmethod
    def from_flat_dict(cls, spec):
        """Construct a component from a strict flat engine-style mapping."""
        spec = dict(spec)
        allowed = {field.name for field in fields(cls)}
        unknown = set(spec) - allowed
        if unknown:
            raise TypeError(f"unknown VgmComponent field(s): {sorted(unknown)}")
        return cls(**spec)


__all__ = ["VgmComponent"]
