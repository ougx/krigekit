from setuptools import setup

# Force setuptools to produce a platform-specific wheel.
# Without this, the wheel is tagged py3-none-any and the compiled
# Fortran library (kriging.dll / libkriging.so / .dylib) gets bundled
# but pip will happily install it on the wrong OS.
try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            super().finalize_options()
            self.root_is_pure = False

    setup(cmdclass={"bdist_wheel": bdist_wheel})
except ImportError:
    setup()
