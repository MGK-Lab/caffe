# This setup.py file is used to compile pyx files
# python setup.py 

import subprocess

# Compile the parallel C++ file
process = subprocess.Popen(
    [
        'g++',
        '-fopenmp',       # enable OpenMP
        '-fPIC',          # generate position-independent code
        './src/caffe_core_parallel.cpp',  # source file
        '-shared',        # build shared library
        '-o', 'caffe_core_parallel.so'    # output file
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

stdout, stderr = process.communicate()

if process.returncode != 0:
    print("Compilation failed. Error message:")
    print(stderr.decode('utf-8'))
else:
    print("caffe_core_parallel.cpp compilation was successful. The library is caffe_core_parallel.so")
