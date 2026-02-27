# ------------------------------------------------------------------------------
# Dynamic CA-ffe
# Copyright (C) 2022–2026 Maziar Gholami Korzani
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
# ------------------------------------------------------------------------------

# run "python setup.py" for compiling C++ files  

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
