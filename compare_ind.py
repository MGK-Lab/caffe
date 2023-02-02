import src.indicators as ind
import numpy as np
import src.util as ut

caffe_wd_file = './tests/caffe_test_out_mwd.tif'
reference_wd_file = './tests/caffe_test_out_wd.tif'

caffe_wd, mask, bounds = ut.DEMRead(caffe_wd_file)
reference_wd, mask, bounds = ut.DEMRead(reference_wd_file)

pm = ind.PerformanceIndicators(caffe_wd, reference_wd)
pm.print_indicators()