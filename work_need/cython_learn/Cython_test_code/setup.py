from distutils.core import setup
from Cython.Build import cythonize
setup(
    name="Example Cython",
    ext_modules=cythonize(["test_cython_code.pyx"])
)
