import numpy as np
import sys
sys.path.append("./src")
import visual as vs  # NOQA

if __name__ == "__main__":

    mypath = './maz/wd'
    vs.AnimateDEMs(mypath, 'water_depth.gif', 5, 1)
