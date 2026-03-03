:: create build folder if it doesn't exist
if not exist .\build mkdir .\build

:: compile and link into ../build folder
cl /O2 /openmp /LD src\caffe_core_parallel.cpp /Fo:.\build\caffe_core_parallel.obj /Fe:.\build\caffe_core_parallel.dll