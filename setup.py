# This setup.py file is used to compile pyx files
# python setup.py build_ext --inplace

from setuptools import setup, Extension
from Cython.Build import cythonize
import Cython.Compiler.Options
import numpy

Cython.Compiler.Options.annotate = True

ext_modules = [
    Extension("caffe_core_parallel", ["./src/caffe_core_parallel.pyx"],
              extra_compile_args=['-fopenmp'],
              extra_link_args=['-fopenmp']),
    Extension("caffe_core", ["./src/caffe_core.pyx"])
]


setup(
    ext_modules=cythonize(ext_modules,
                          compiler_directives={'profile': True}),
    include_dirs=[numpy.get_include()]
)
