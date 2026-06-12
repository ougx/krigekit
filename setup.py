from setuptools import setup
from setuptools.dist import Distribution

# Patch has_ext_modules at the *class* level so every Distribution instance
# setuptools creates (including its internal one read from pyproject.toml)
# returns True.
#
# Why this is needed:
#   krigekit ships a pre-compiled Fortran shared library (libkriging.so /
#   .dylib / .dll) as package data.  setuptools sees no C extension modules,
#   so it normally produces a purelib wheel.  auditwheel (Linux) and
#   delocate (macOS) both require the shared library to be in the *platlib*
#   section of the wheel, not purelib.
#
# Why distclass=BinaryDistribution alone is not enough:
#   In setuptools 61+, build_meta creates its own Distribution from
#   pyproject.toml and may not use the distclass from setup().  Patching
#   the base class ensures the override is visible to every code path.
Distribution.has_ext_modules = lambda self: True

setup()
