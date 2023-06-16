# This setup.py file is used to compile pyx files
# python setup.py build_ext --inplace

from setuptools import setup, Extension
from Cython.Build import cythonize
import Cython.Compiler.Options
import numpy
import subprocess


Cython.Compiler.Options.annotate = True

ext_modules = [
    Extension("caffe_core", ["./src/caffe_core.pyx"])
]


setup(
    ext_modules=cythonize(ext_modules,
                          compiler_directives={'profile': True}),
    include_dirs=[numpy.get_include()]
)

# Run the g++ command to compile the C++ file
process = subprocess.Popen(
    ['g++', '-fopenmp', '-fPIC', "./src/caffe_core_parallel.cpp", '-shared',
     '-o', 'caffe_core_parallel.so'],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = process.communicate()
if process.returncode != 0:
    print("Compilation failed. Error message:")
    print(stderr.decode('utf-8'))
else:
    print("caffe_core_parallel.cpp compilation was successful. The library is caffe_core_parallel.so")
