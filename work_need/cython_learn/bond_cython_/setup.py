from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
ext_mod = cythonize(["time_libs.pyx"],
                    language='c++',
                    language_level = '3')

setup(
    name = 'time_lib',
    ext_modules=ext_mod ,
    cmdclass= {'build_ext':build_ext}
)
# python setup.py build_ext --inplace