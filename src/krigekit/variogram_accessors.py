"""Internal 1-based accessors for the multivariable variogram system.

These wrap the system's per-pair storage so that the public surface
(``system.vgm[ivar, jvar]``) reads like a symmetric matrix while keeping the
mapping mechanics out of :class:`VariogramSystem`.  The accessor never imports
the system; the system constructs it and supplies index-validation/growth and
storage callbacks.

A pair is **materialized** once it has a stored object and **configured** only
once that object has at least one component.  Internal algorithms must select
work by ``configured_pairs()``, never by materialization, so that a harmless
read such as ``system.vgm[1, 2]`` cannot turn an empty pair into an LMC input.
"""

from .variogram_structure import VgmStructure


class _VgmAccessor:
    """Symmetric, 1-based accessor for per-pair :class:`VgmStructure` objects.

    Parameters are callbacks owned by the system:

    ``key(ivar, jvar)``
        Validate the 1-based indices (raising the teaching error on ``0`` /
        negatives / non-integers), grow or bound ``nvar``, and return the
        canonical upper-triangle key ``(a, b)`` with ``a <= b``.
    ``ensure(key)`` / ``peek(key)``
        Return the structure for a canonical key, creating it or returning
        ``None`` respectively.
    ``assign(key, structure)`` / ``drop(key)``
        Replace or remove the stored structure.
    ``materialized()``
        Return ``[(key, structure), ...]`` for every materialized pair.
    """

    def __init__(self, *, key, ensure, peek, assign, drop, materialized):
        self._key = key
        self._ensure = ensure
        self._peek = peek
        self._assign = assign
        self._drop = drop
        self._materialized = materialized

    @staticmethod
    def _split(pair):
        """Return ``(ivar, jvar)`` from a subscript, or raise a clear error."""
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise TypeError(
                "index the variogram matrix with a pair, e.g. vgm[1, 2] "
                "(auto-variogram: vgm[1, 1])"
            )
        return pair

    def __getitem__(self, pair):
        """Return the (lazily created) structure for ``vgm[ivar, jvar]``."""
        ivar, jvar = self._split(pair)
        return self._ensure(self._key(ivar, jvar))

    def __setitem__(self, pair, structure):
        """Assign a :class:`VgmStructure` to a pair, preserving symmetry."""
        if not isinstance(structure, VgmStructure):
            raise TypeError("vgm[i, j] must be assigned a VgmStructure")
        ivar, jvar = self._split(pair)
        self._assign(self._key(ivar, jvar), structure)

    def get(self, ivar, jvar=None, create=True):
        """Return the pair structure; with ``create=False`` return ``None``
        for an unmaterialized pair instead of creating one."""
        key = self._key(ivar, jvar)
        return self._ensure(key) if create else self._peek(key)

    def clear(self, ivar, jvar=None):
        """Remove a materialized pair entry."""
        self._drop(self._key(ivar, jvar))

    def configured_pairs(self):
        """Return sorted canonical keys whose structure has components."""
        return sorted(k for k, s in self._materialized() if s.ncomponent)

    def configured_items(self):
        """Return sorted ``(key, structure)`` for configured pairs only."""
        return [(k, s) for k, s in sorted(self._materialized()) if s.ncomponent]

    def items(self):
        """Alias for :meth:`configured_items` (configured pairs only)."""
        return self.configured_items()

    def materialized_items(self):
        """Return sorted ``(key, structure)`` including empty materialized pairs."""
        return sorted(self._materialized())

    @property
    def nconfigured(self):
        """Number of configured pairs."""
        return sum(1 for _, s in self._materialized() if s.ncomponent)

    @property
    def nmaterialized(self):
        """Number of materialized pairs, configured or not."""
        return len(self._materialized())

    def __repr__(self):
        """Return a compact debugging representation."""
        return (f"_VgmAccessor(nconfigured={self.nconfigured}, "
                f"nmaterialized={self.nmaterialized})")


__all__ = ["_VgmAccessor"]
