# copy this file to your conda folder, like /miniconda3/envs/caffe/etc/conda/activate.d
# change addresses below to match your folders 
# the first one includes addresses of caffe and caffe/src for your python environment
# the second one includes the address of the static library for C++ for the parallel version
#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/home/maziar/projects/caffe:/home/maziar/projects/caffe/src
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/maziar/projects/caffe
