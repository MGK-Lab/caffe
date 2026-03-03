#!/bin/bash

# create build folder if it doesn't exist
mkdir -p ./build

# compile and link into build folder
g++ -O3 -fopenmp -shared -fPIC src/caffe_core_parallel.cpp \
    -o build/caffe_core_parallel.so