from .caffe import caffe
from .swmm import swmm
import numpy as np
import sys
import os
from colorama import Fore, Back, Style
from copy import deepcopy


class csc:
    def __init__(self):
        print("\n .....Initiating 2D-1D modelling using coupled SWMM & CA-ffé.....")
        self.g = 9.81
        self.rel_diff = np.zeros(2)

    def LoadCaffe(self, DEM_file, hf, increment_constant, EV_threshold):
        self.caffe = caffe(DEM_file)
        self.caffe.setConstants(hf, increment_constant, EV_threshold)
        self.caffe_output_name = self.caffe.outputs_name

    def LoadSwmm(self, SWMM_inp):
        self.swmm = swmm(SWMM_inp)
        self.swmm.LoadNodes()

    def NodeElvCoordChecks(self):
        # domain size check
        c_b= self.caffe.bounds
        s_b= self.swmm.bounds
        if (c_b[0]<=s_b[0] and c_b[1]>=s_b[1] and c_b[2]>=s_b[2] and c_b[3]<=s_b[3]):
            print('All SWMM junctions are bounded by the provided DEM')
        else:
            sys.exit(
                "The SWMM junctions coordinates are out of the boundry of the provided DEM")

        self.swmm_node_info = self.swmm.nodes_info.to_numpy()
        self.swmm_node_info = np.column_stack(
            (self.swmm_node_info[:, 0], self.swmm_node_info[:, 1],
             self.swmm_node_info[:, 2] + self.swmm_node_info[:, 3], self.swmm_node_info[:, 4]))

        self.swmm_node_info[:, 0] = np.int_(
            (self.swmm_node_info[:, 0]-self.rel_diff[0]) / self.caffe.length)
        self.swmm_node_info[:, 1] = np.int_(
            (self.swmm_node_info[:, 1]-self.rel_diff[1]) / self.caffe.length)

        err = False
        i = 0
        for r in self.swmm_node_info:
            if ((r[0] < 0 or r[0] > self.caffe.DEMshape[0]
                    or r[1] < 0 or r[1] > self.caffe.DEMshape[1]) and r[3] == False):
                print(self.swmm.node_list[i])
                err = True
            i += 1
        if err:
            sys.exit(
                "The above SWMM junctions coordinates are out of the range in the provided DEM")

        err = False
        i = 0
        for r in self.swmm_node_info:
            if ((abs(r[2] - self.caffe.DEM[np.int_(r[0]), np.int_(r[1])])
                    > 0.01 * self.caffe.DEM[np.int_(r[0]), np.int_(r[1])]) and r[3] == False):
                print(self.swmm.node_list[i], " diff = ", self.caffe.DEM[np.int_(r[0]), np.int_(r[1])]-r[2])
                err = True
            i += 1
        if err:
            sys.exit(
                "The above SWMM junctions surface elevation have > 1% difference with the provided DEM")

        print("-----Nodes elevation and coordinates are compatible in both models (outfalls excluded)")

    def InteractionInterval(self, sec):
        self.swmm.InteractionInterval(sec)
        self.IntTimeStep = sec

    def RunOne_SWMMtoCaffe(self):
        floodvolume = 0
        for step in self.swmm.sim:
            floodvolume += self.swmm.getNodesFlooding()
        floodvolume = np.column_stack((self.swmm_node_info[:, 0:2],
                                       np.transpose(floodvolume)*self.IntTimeStep))
        self.swmm.CloseSimulation()

        self.caffe.ExcessVolumeMapArray(floodvolume)
        self.caffe.RunSimulation()
        self.caffe.CloseSimulation()

    def RunMulti_SWMMtoCaffe(self):
        origin_name = self.caffe.outputs_name

        for step in self.swmm.sim:
            tmp = self.swmm.getNodesFlooding()
            if (np.sum(tmp) > 0):
                floodvolume = np.column_stack((self.swmm_node_info[:, 0:2],
                                               np.transpose(tmp)*self.IntTimeStep))
                self.caffe.ExcessVolumeMapArray(floodvolume)
                self.caffe.outputs_name = origin_name + \
                    "_" + str(self.swmm.sim.current_time)
                self.caffe.RunSimulation()
                self.caffe.CloseSimulation()

        self.swmm.CloseSimulation()

    def Run_Caffe_BD_SWMM(self):

        origin_name = self.caffe.outputs_name
        First_Step = True
        last_wd = np.array([[], []])
        self.exchangeamount = np.array([])
        self.nodeinfo = np.array([[], []])
        self.time = np.array([])

        print(Fore.RED + "Starts running SWMM" + Style.RESET_ALL + "\n\n")

        for step in self.swmm.sim:
            print(Fore.GREEN + "SWMM @ "
                  + str(self.swmm.sim.current_time) + Style.RESET_ALL + "\n")
            self.time = np.append(self.time, self.swmm.sim.current_time)
            temp = np.array([0, 0], dtype=np.double)

            nodeinfo = np.column_stack((self.swmm.getNodesHead(
            ), self.swmm.getNodesTotalInflow(), self.swmm.getNodesTotalOutflow()))

            if First_Step:
                First_Step = False
                self.nodeinfo = nodeinfo
            else:
                self.caffe.Reset_WL_EVM()
                self.caffe.LoadInitialExcessWaterDepthArray(last_wd)
                self.nodeinfo = np.dstack((self.nodeinfo, nodeinfo))

            flooded_nodes_flowrates = np.transpose(
                self.swmm.getNodesFlooding()) * self.IntTimeStep
            tot_flood = np.sum(flooded_nodes_flowrates)
            temp[0] = tot_flood

            if tot_flood > 0:
                print(Fore.RED + "SWMM surcharged "
                      + str(tot_flood) + Style.RESET_ALL + "\n")
                floodvolume = np.column_stack((self.swmm_node_info[:, 0:2],
                                               flooded_nodes_flowrates
                                               / self.caffe.cell_area))
                self.caffe.ExcessVolumeMapArray(floodvolume, True)

            nonflooded_nodes = self.swmm_node_info[np.logical_not(
                    flooded_nodes_flowrates > 0), 0:2]

            # to run caffe without open boundries
            caffecopy = deepcopy(self.caffe)

            self.caffe.OpenBCMapArray(nonflooded_nodes)

            if (tot_flood > 0 or np.sum(self.caffe.excess_volume_map) > 0):
                sys.stdout = open(os.devnull, 'w')
                caffecopy.RunSimulation()
                # caffecopy.ReportScreen()
                actual_wd = caffecopy.water_depths
                sys.stdout = sys.__stdout__
                del caffecopy

                self.caffe.RunSimulation()
                self.caffe.ReportScreen()

                last_wd = self.caffe.water_depths
                if (np.sum(last_wd) > 0):
                    j = 0
                    k = 0
                    inflow = np.zeros(self.swmm_node_info.shape[0])
                    for i in flooded_nodes_flowrates:
                        if i > 0:
                            inflow[j] = 0
                        else:
                            x = int(self.caffe.OBC_cells[k, 0])
                            y = int(self.caffe.OBC_cells[k, 1])
                            inflow[j] = self.MaxFlowrate(
                                j, actual_wd[x, y], last_wd[x, y])
                            last_wd[x, y] -= inflow[j]
                            k += 1
                        j += 1
                    self.swmm.setNodesInflow(
                        inflow * self.caffe.cell_area / self.IntTimeStep)
                    print(Fore.BLUE + "\nCA-ffé drained "
                          + str(np.sum(inflow) * self.caffe.cell_area) + Style.RESET_ALL + "\n")
                    temp[1] = np.sum(inflow) * self.caffe.cell_area

                #For reporting purpose, WD changed. BTW, it will be reseted in the next step
                if np.sum(last_wd) > 0:
                    self.caffe.water_depths = last_wd
                    self.caffe.outputs_name = origin_name + \
                        "_" + str(self.swmm.sim.current_time)
                    self.caffe.ReportFile()

            else:
                last_wd = np.zeros_like(self.caffe.DEM, dtype=np.double)

            self.exchangeamount = np.append(
                self.exchangeamount, temp, axis=0)

        self.exchangeamount = self.exchangeamount.reshape(
            [int(self.exchangeamount.size/2), 2])

        self.swmm.CloseSimulation()

    def ManholeProp(self, coef, length):
        coef = np.asarray(coef)
        length = np.asarray(length)
        if coef.size == 1 or coef.size == self.swmm_node_info.shape[0]:
            if coef.size == 1:
                coef = np.repeat(coef, self.swmm_node_info.shape[0])
        else:
            sys.exit("Weir discharge coefficient array size does not match ",
                     "junction numbers")

        if length.size == 1 or length.size == self.swmm_node_info.shape[0]:
            if length.size == 1:
                length = np.repeat(length, self.swmm_node_info.shape[0])
        else:
            sys.exit("Weir crest length array size does not match junction numbers")

        self.WeirDisCoef = coef
        self.WeirCrest = length

    def MaxFlowrate(self, counter, waterdepth, calcq_height):
        q = self.WeirDisCoef[counter] * self.WeirCrest[counter] * \
            waterdepth * (self.g * waterdepth * 2)**0.5
        if (q / self.caffe.cell_area * self.IntTimeStep) < calcq_height:
            return q
        else:
            return calcq_height
