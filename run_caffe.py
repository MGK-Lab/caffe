# python3 run_caffe.py

from src import caffe

if __name__ == "__main__":

    # user input data:
    input_DEM_file = './tests/dem_s1.tif'
    hf = 0.09                   # First CAffe model parameter selected by user
    increment_constant = 0.0005  # Second CAffe model parameter selected by user
    EV_threshold = 0.00002
    result_path = './tests/'
    result_name = "hf" + str(hf)+"_IC_" + str(increment_constant)

    # run model
    caffe.run_caffe(input_DEM_file, increment_constant, hf,
                    result_path, result_name, EV_threshold)
