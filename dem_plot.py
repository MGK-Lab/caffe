import src.visual as vs
import matplotlib.pyplot as plt

if __name__ == "__main__":

    file = './tests/caffe_test_out_wd.tif'
    vs.PlotDEM3d(file, 5)
    # plt.gca().set_box_aspect([1, 1, 1])
    # ut.PlotDEM2d(file)

    plt.show()
