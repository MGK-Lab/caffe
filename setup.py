# This setup.py file is used to compile (Cythonize) caffe.pyx file
# python3 setup.py build_ext --inplace

from setuptools import setup
from Cython.Build import cythonize
import Cython.Compiler.Options
import numpy

Cython.Compiler.Options.annotate = True

setup(
    ext_modules=cythonize("./src/caffe_core.pyx",
                          compiler_directives={'profile': True}),
    include_dirs=[numpy.get_include()]
)
