import src.util as ut
import matplotlib.pyplot as plt

if __name__ == "__main__":

    file = './tests/caffe_test.tif'
    ut.PlotDEM3d(file, 1)
    # plt.gca().set_box_aspect([1, 1, 1])
    # ut.PlotDEM2d(file)

    plt.show()
