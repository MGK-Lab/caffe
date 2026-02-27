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

# run "python setup.py build_ext --inplace" for compiling C++ files  

from setuptools import setup, Extension
import platform
import sys
import os

system = platform.system()

extra_compile_args = ["-O3"]
extra_link_args = []

if system == "Linux":
    extra_compile_args += ["-fopenmp", "-fPIC"]
    extra_link_args += [
        "-fopenmp",
        "-static-libgcc",
        "-static-libstdc++"
        # do NOT try to static-link libgomp
    ]
    library_name = "caffe_core_parallel.so"

elif system == "Darwin":  # macOS
    extra_compile_args += ["-Xpreprocessor", "-fopenmp"]
    extra_link_args += ["-lomp"]
    library_name = "caffe_core_parallel.dylib"

elif system == "Windows":
    extra_compile_args += ["-fopenmp"]
    extra_link_args += ["-static-libgcc", "-static-libstdc++"]
    library_name = "caffe_core_parallel.dll"

else:
    sys.exit(f"Unsupported OS: {system}")

# Use setuptools Extension just to compile, then rename output manually
module = Extension(
    "_dummy_module",  # dummy name; we will ignore the generated file
    sources=["src/caffe_core_parallel.cpp"],
    language="c++",
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

setup(
    name="caffe_core_parallel_build",
    version="1.0.0",
    ext_modules=[module],
    script_args=["build_ext", "--inplace"]
)

# Rename generated file to plain shared library
import glob
import shutil

if system == "Linux":
    ext = "*.so"
elif system == "Darwin":
    ext = "*.dylib"
else:
    ext = "*.pyd"

built_files = glob.glob(ext)
for f in built_files:
    shutil.move(f, library_name)
    print(f"Renamed {f} → {library_name}")