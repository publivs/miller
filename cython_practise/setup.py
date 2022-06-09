from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = 'Example_Cython',
    ext_modules = cythonize("run_cython_1.pyx")
)