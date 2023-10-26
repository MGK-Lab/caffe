from caffe import caffe
from cswmm import swmm
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from colorama import Fore, Back, Style
from copy import deepcopy
import time
import util
import csv


class csc:
    def __init__(self):
        print("\n .....Initiating 2D-1D modelling using coupled SWMM & CA-ffé.....")
        print("\n", time.ctime(), "\n")
        self.t = time.time()

        self.g = 9.81
        self.rel_diff = np.zeros(2)
        self.elv_dif = 0.01
        self.plot_Node_DEM = False
        self.failed_node_check = np.array([], dtype=int)
        self.weir_approach = False
        self.volume_conversion = 0.001
        self.recrusive_run = False
        self.active_nodes = np.array([])
        self.report_max = True
        self.report_interval = True
        self.project_name = ''

    def ActiveNodes(self, filename):
        with open(filename, 'r') as file:
            names = file.read().splitlines()
        self.active_nodes = np.array(names)

    def LoadCaffe(self, DEM_file, hf, increment_constant, EV_threshold):
        self.caffe = caffe(DEM_file)
        self.caffe.setConstants(hf, increment_constant, EV_threshold)

    def LoadSwmm(self, SWMM_inp):
        self.swmm = swmm(SWMM_inp)
        self.swmm.LoadNodes()
        print("\nSWMM system unit is: ", self.swmm.sim.system_units)
        print("SWMM flow unit is:   ", self.swmm.sim.flow_units, "\n")

    def NodeElvCoordChecks(self):
        # domain size check is conducted here
        c_b = self.caffe.bounds
        s_b = self.swmm.bounds
        if (c_b[0] <= s_b[0] and c_b[1] >= s_b[1] and c_b[2] >= s_b[2] and c_b[3] <= s_b[3]):
            print('All SWMM junctions are bounded by the provided DEM')
        else:
            sys.exit(
                "The SWMM junctions coordinates are out of the boundry of the provided DEM")

        # convert node locations to DEM numpy system
        # coulumns are: 1 and 2 coordinates, 3 elevation, 4 outfall and 5 active
        self.swmm_node_info = self.swmm.nodes_info.to_numpy()
        self.swmm_node_info = np.column_stack(
            (self.swmm_node_info[:, 0],
             self.swmm_node_info[:, 1],
             self.swmm_node_info[:, 2] + self.swmm_node_info[:, 3],
             self.swmm_node_info[:, 4]))
        self.rel_diff = [self.caffe.bounds[0], self.caffe.bounds[1]]
        self.swmm_node_info[:, 0] = np.int_(
            (self.swmm_node_info[:, 0]-self.rel_diff[0]) / self.caffe.length)
        self.swmm_node_info[:, 1] = np.int_(
            (self.rel_diff[1]-self.swmm_node_info[:, 1]) / self.caffe.length)

        if self.active_nodes.size > 0:
            active_nodes = np.isin(self.swmm.node_list, self.active_nodes)
        else:
            active_nodes = np.ones((len(self.swmm.node_list)), dtype=bool)

        self.swmm_node_info = np.column_stack(
            (self.swmm_node_info, active_nodes))

        # when a DEM loaded coordinate system is reverse (Rasterio)
        self.swmm_node_info[:, [0, 1]] = self.swmm_node_info[:, [1, 0]]
        # saving nodes on a raster
        tmp = np.zeros_like(self.caffe.DEM, dtype=np.double)
        for i in self.swmm_node_info:
            tmp[i[0], i[1]] = 1

        name = self.caffe.outputs_path + self.caffe.outputs_name + '_swmm_nodes_raster.tif'
        util.ArrayToRaster(tmp, name, self.caffe.dem_file)

        # plot SWMM nodes on DEM
        if self.plot_Node_DEM:
            temp = deepcopy(self.caffe.DEM)
            temp[self.caffe.ClosedBC == True] = np.max(temp)
            plt.imshow(temp, cmap='gray')
            plt.scatter(self.swmm_node_info[:, 1], self.swmm_node_info[:, 0])
            plt.show()

        # err = False
        # i = 0
        # for r in self.swmm_node_info:
        #     if ((r[0] < 0 or r[0] > self.caffe.DEMshape[0]
        #             or r[1] < 0 or r[1] > self.caffe.DEMshape[1]) and r[3] == False):
        #        print(self.swmm.node_list[i])
        #        err = True
        #     i += 1
        # if err:
        #     sys.exit(
        #         "The above SWMM junctions coordinates are out of the range in the provided DEM")

        # Elevation check is conducted here
        err = False
        i = 0
        for r in self.swmm_node_info:
            if ((abs(r[2] - self.caffe.DEM[np.int_(r[0]), np.int_(r[1])])
                    > self.elv_dif * self.caffe.DEM[np.int_(r[0]), np.int_(r[1])]) and r[3] == False):
                if not (self.recrusive_run):
                    print(
                        self.swmm.node_list[i],
                        " diff = ", self.caffe.DEM
                        [np.int_(r[0]),
                         np.int_(r[1])] - r[2])
                self.failed_node_check = np.append(
                    self.failed_node_check, np.int_(i))
                err = True
            i += 1

        if (err and not (self.recrusive_run)):
            print("The above SWMM junctions surface elevation have >",
                  self.elv_dif*100, "% difference with the provided DEM.\n")
            temp = deepcopy(self.caffe.DEM)
            temp[self.caffe.ClosedBC == True] = np.max(temp)
            plt.imshow(temp, cmap='gray')
            plt.scatter(self.swmm_node_info[self.failed_node_check, 1],
                        self.swmm_node_info[self.failed_node_check, 0])
            plt.show()

        while (err and not (self.recrusive_run)):
            answer = input("Do you want to continue? (yes/no)")
            if answer == "yes":
                pass
                break
            elif answer == "no":
                sys.exit()
            else:
                print("Invalid answer, please try again.")

        if not (err):
            print(
                "-----Nodes elevation and coordinates are compatible in both models (outfalls excluded)")

    def InteractionInterval(self, sec):
        self.swmm.InteractionInterval(sec)
        self.IntTimeStep = sec

    def RunOne_SWMMtoCaffe(self):
        if not (self.recrusive_run):
            print("\nFor one-time one-way coupling of SWMM to Caffe apprach, SWMM model should generate a report for all nodes.")
            continue_choice = input("Do you want to continue? (yes/no): ")
            if continue_choice.lower() not in ["yes", "y"]:
                raise Exception("Program terminated to revise SWMM input file")

        self.swmm.sim.execute()
        floodvolume = self.swmm.Output_getNodesFlooding()
        # it is multiplied by DEM length as the caffe excess volume will get coordinates
        # not cell. swmm_node_info is already converted to cell location in NodeElvCoordChecks function
        floodvolume = np.column_stack(
            (self.swmm_node_info[:, 0: 2] * self.caffe.length, np.transpose(
                floodvolume * self.volume_conversion)))

        self.caffe.ExcessVolumeArray(floodvolume)
        self.caffe.RunSimulation()
        self.caffe.CloseSimulation()
        print("\n .....finished one-directional coupled SWMM & CA-ffé - one timestep.....")
        print("\n", time.ctime(), "\n")
        print("\n Duration: ", time.time()-self.t, "\n")

    def RunMulti_SWMMtoCaffe(self):
        old_mwd = np.zeros_like(self.caffe.DEM, dtype=np.double)
        self.exchange_amount = []

        for step in self.swmm.sim:
            print(Fore.GREEN + "SWMM @ "
                  + str(self.swmm.sim.current_time) + Style.RESET_ALL + "\n")
            floodvolume = self.swmm.getNodesFlooding()*self.IntTimeStep*self.volume_conversion
            total_surcharged = np.sum(floodvolume)
            self.exchange_amount.append(
                [self.swmm.sim.current_time, total_surcharged])

            if (total_surcharged > 0):
                print(Fore.RED + "SWMM surcharged "
                      + str(total_surcharged) + Style.RESET_ALL + "\n")

                floodvolume = np.column_stack(
                    (self.swmm_node_info[:, 0:2] * self.caffe.length, np.transpose(floodvolume)))
                self.caffe.ExcessVolumeArray(floodvolume, False)
                name = self.caffe.outputs_name + \
                    "_" + str(self.swmm.sim.current_time)
                self.caffe.RunSimulation()
                # to avoid loosing friction headloss
                # self.caffe.max_water_depths = np.maximum(
                #     self.caffe.max_water_depths, old_mwd)
                # old_mwd = self.caffe.max_water_depths
                self.caffe.CloseSimulation(name)

        # save all exchanges between models in a csv file including the exchange time
        name = self.caffe.outputs_path + self.caffe.outputs_name + '_exchange_report.csv'
        with open(name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for entry in self.exchange_amount:
                formatted_entry = [entry[0].strftime(
                    '%Y-%m-%d %H:%M:%S')] + entry[1:]
                writer.writerow(formatted_entry)

        self.swmm.CloseSimulation()
        print(
            "\n .....finished one-directional coupled SWMM & CA-ffé - multi timestep.....")
        print("\n", time.ctime(), "\n")
        print("\n Duration: ", time.time()-self.t, "\n")

    def Run_Caffe_BD_SWMM(self):
        max_mwd = np.array([[], []])
        max_wd = np.array([[], []])
        max_wl = np.array([[], []])
        First_Step = True
        self.exchange_amount = []
        total_surcharged = 0
        total_drained = 0
        self.caffe.waterdepth_excess = True

        print(Fore.RED + "Starts running SWMM" + Style.RESET_ALL + "\n\n")

        for step in self.swmm.sim:
            print(Fore.GREEN + "SWMM @ "
                  + str(self.swmm.sim.current_time) + Style.RESET_ALL + "\n")

            if First_Step:
                First_Step = False
            else:
                self.caffe.Reset_WL_EVM()
                self.caffe.ExcessWaterDepthRaster(self.caffe.water_depths)

            # convert flood flowrate at each node to volume
            flooded_nodes_flowrates = np.transpose(
                self.swmm.getNodesFlooding()) * self.IntTimeStep * self.volume_conversion
            total_surcharged = np.sum(flooded_nodes_flowrates)

            if total_surcharged > 0:
                print(Fore.RED + "SWMM surcharged "
                      + str(total_surcharged) + Style.RESET_ALL + "\n")
                # convert node cells to node coordinates (local) for feeding ExcessVolumeArray
                # since it reads the last water depth to initialise, the flood
                # volumes needs to be converted to water depths
                floodvolume = np.column_stack(
                    (self.swmm_node_info[:, 0: 2] * self.caffe.length,
                     flooded_nodes_flowrates / self.caffe.cell_area))
                self.caffe.ExcessVolumeArray(floodvolume, True)

            # extract active non-flooded nodes for setting as open BCs
            # convert node cells to node coordinates (local) for feeding caffe.OpenBCArray
            nonflooded_nodes = np.where(
                flooded_nodes_flowrates == 0, True, False)
            nonflooded_nodes = nonflooded_nodes * self.swmm_node_info[:, 4]
            nonflooded_nodes_coords = self.swmm_node_info[nonflooded_nodes >
                                                          0, 0:2] * self.caffe.length

            # to run caffe without open boundries for weir equation calculations
            if self.weir_approach:
                caffecopy = deepcopy(self.caffe)

            self.caffe.OpenBCArray(nonflooded_nodes_coords)

            if (total_surcharged > 0 or np.sum(self.caffe.excess_volume_map) > 0):

                if self.weir_approach:
                    sys.stdout = open(os.devnull, 'w')
                    caffecopy.RunSimulation()
                    # caffecopy.ReportScreen()
                    actual_wd = deepcopy(caffecopy.water_depths)
                    sys.stdout = sys.__stdout__
                    del caffecopy

                self.caffe.RunSimulation()
                self.caffe.ReportScreen()

                if (np.sum(self.caffe.water_depths) > 0):
                    j = 0
                    k = 0
                    inflow = np.zeros(self.swmm_node_info.shape[0])
                    inflow_raster = np.zeros_like(
                        self.caffe.water_depths, dtype=np.double)

                    for i in nonflooded_nodes:
                        if i == 1:
                            x = int(self.caffe.OBC_cells[k, 0])
                            y = int(self.caffe.OBC_cells[k, 1])

                            if self.weir_approach:
                                inflow[j] = self.MaxFlowrate(
                                    j, actual_wd[x, y],
                                    self.caffe.water_depths[x, y])
                            else:
                                inflow[j] = deepcopy(
                                    self.caffe.water_depths[x, y])

                            inflow_raster[x, y] = inflow[j]
                            k += 1
                        j += 1

                    self.swmm.setNodesInflow(
                        inflow * self.caffe.cell_area / self.IntTimeStep /
                        self.volume_conversion)
                    total_drained = np.sum(
                        inflow) * self.caffe.cell_area
                    print(Fore.BLUE + "\nCA-ffé drained "
                          + str(total_drained) + Style.RESET_ALL + "\n")

                # For reporting purpose, WD changed. BTW, it will be reseted in the next step
                self.caffe.water_depths -= inflow_raster
                self.caffe.max_water_depths -= inflow_raster
                self.caffe.water_levels -= inflow_raster
                name = self.caffe.outputs_name + \
                    "_" + str(self.swmm.sim.current_time)
                if self.report_interval == True:
                    self.caffe.ReportFile(name)
                if self.report_max == True:
                    if np.any(max_mwd) == False:
                        max_mwd = deepcopy(self.caffe.max_water_depths)
                        max_wd = deepcopy(self.caffe.water_depths)
                        max_wl = deepcopy(self.caffe.water_levels)
                    else:
                        max_mwd = np.maximum(
                            max_mwd, self.caffe.max_water_depths)
                        max_wd = np.maximum(max_wd, self.caffe.water_depths)
                        max_wl = np.maximum(max_wl, self.caffe.water_levels)

            self.exchange_amount.append(
                [self.swmm.sim.current_time, total_surcharged, total_drained])

        self.swmm.CloseSimulation()

        if self.report_max == True:
            self.caffe.Reset_WL_EVM()
            self.caffe.max_water_depths = deepcopy(max_mwd)
            self.caffe.water_depths = deepcopy(max_wd)
            self.caffe.water_levels = deepcopy(max_wl)
            self.caffe.ReportFile(self.project_name + '_max_all')

        # save all exchanges between models in a csv file including the exchange time
        name = self.caffe.outputs_path + self.caffe.outputs_name + '_exchange_report.csv'
        with open(name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for entry in self.exchange_amount:
                formatted_entry = [entry[0].strftime(
                    '%Y-%m-%d %H:%M:%S')] + entry[1:]
                writer.writerow(formatted_entry)

        print("\n .....finished bi-directional coupled SWMM & CA-ffé.....")
        print("\n", time.ctime(), "\n")
        print("\n Duration: ", time.time()-self.t, "\n")

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

    def Run_Caffe_BD_SWMM_ROG(self):
        max_mwd = np.array([[], []])
        max_wd = np.array([[], []])
        max_wl = np.array([[], []])
        self.exchange_amount = []
        total_surcharged = 0
        total_drained = 0
        self.caffe.waterdepth_excess = True
        initial_interval = 60

        # for fist time running caffe; a SWMM run will be done for initial_interval
        #
        First_Step = True
        rain_step = 1
        self.InteractionInterval(initial_interval)

        for step in self.swmm.sim:

            if First_Step:
                First_Step = False
                self.InteractionInterval(
                    self.caffe.rain['TimeDiff'].iloc[rain_step] -
                    initial_interval)
                nonflooded_nodes = self.swmm_node_info[:, 4]
                nonflooded_nodes_coords = self.swmm_node_info[nonflooded_nodes
                                                              > 0, 0:2] * self.caffe.length
            else:
                print(
                    Fore.GREEN + "SWMM @ " + str(self.swmm.sim.current_time) +
                    Style.RESET_ALL + "\n")

                # convert flood flowrate at each node to volume
                flooded_nodes_flowrates = np.transpose(
                    self.swmm.getNodesFlooding()) * self.IntTimeStep * self.volume_conversion
                total_surcharged = np.sum(flooded_nodes_flowrates)

                # extract active non-flooded nodes for setting as open BCs
                # convert node cells to node coordinates (local) for feeding caffe.OpenBCArray
                nonflooded_nodes = np.where(
                    flooded_nodes_flowrates == 0, True, False)
                nonflooded_nodes = nonflooded_nodes * self.swmm_node_info[:, 4]
                nonflooded_nodes_coords = self.swmm_node_info[nonflooded_nodes >
                                                              0, 0:2] * self.caffe.length

                self.caffe.Reset_WL_EVM()

                if total_surcharged > 0:
                    print(Fore.RED + "SWMM surcharged "
                          + str(total_surcharged) + Style.RESET_ALL + "\n")
                    # convert node cells to node coordinates (local) for feeding ExcessVolumeArray
                    # since it reads the last water depth to initialise, the flood
                    # volumes needs to be converted to water depths
                    floodvolume = np.column_stack(
                        (self.swmm_node_info[:, 0: 2] * self.caffe.length,
                         flooded_nodes_flowrates / self.caffe.cell_area))
                    self.caffe.ExcessVolumeArray(floodvolume, True)

                self.InteractionInterval(
                    self.caffe.rain['TimeDiff'].iloc[rain_step])

            print(
                Fore.GREEN + "CA-ffé @ ", self.caffe.rain['Time'].iloc
                [rain_step], Style.RESET_ALL + "\n\n")

            self.caffe.OpenBCArray(nonflooded_nodes_coords)
            name = self.caffe.StepSimulationROG(rain_step)

            if (np.sum(self.caffe.water_depths) > 0):
                j = 0
                k = 0
                inflow = np.zeros(self.swmm_node_info.shape[0])
                inflow_raster = np.zeros_like(
                    self.caffe.water_depths, dtype=np.double)
                for i in nonflooded_nodes:
                    if i == 1:
                        x = int(self.caffe.OBC_cells[k, 0])
                        y = int(self.caffe.OBC_cells[k, 1])

                        inflow[j] = deepcopy(self.caffe.water_depths[x, y])

                        inflow_raster[x, y] = inflow[j]
                        k += 1
                    j += 1

                self.swmm.setNodesInflow(
                    inflow * self.caffe.cell_area / self.IntTimeStep / self.volume_conversion)
                total_drained = np.sum(
                    inflow) * self.caffe.cell_area
                print(Fore.BLUE + "\nCA-ffé drained "
                      + str(total_drained) + Style.RESET_ALL + "\n")

                # For reporting purpose, WD changed. BTW, it will be reseted in the next step
                self.caffe.water_depths -= inflow_raster
                self.caffe.max_water_depths -= inflow_raster
                self.caffe.water_levels -= inflow_raster
                name = name + "_swmm_" + str(self.swmm.sim.current_time)
                if self.report_interval == True:
                    self.caffe.ReportFile(name)
                if self.report_max == True:
                    if np.any(max_mwd) == False:
                        max_mwd = deepcopy(self.caffe.max_water_depths)
                        max_wd = deepcopy(self.caffe.water_depths)
                        max_wl = deepcopy(self.caffe.water_levels)
                    else:
                        max_mwd = np.maximum(
                            max_mwd, self.caffe.max_water_depths)
                        max_wd = np.maximum(max_wd, self.caffe.water_depths)
                        max_wl = np.maximum(max_wl, self.caffe.water_levels)

            self.exchange_amount.append(
                [self.swmm.sim.current_time, total_surcharged, total_drained])

            rain_step += 1

        #     # to run caffe without open boundries for weir equation calculations
        #     if self.weir_approach:
        #         caffecopy = deepcopy(self.caffe)

        #     self.caffe.OpenBCArray(nonflooded_nodes_coords)

        #     if (total_surcharged > 0 or np.sum(self.caffe.excess_volume_map) > 0):

        #         if self.weir_approach:
        #             sys.stdout = open(os.devnull, 'w')
        #             caffecopy.RunSimulation()
        #             # caffecopy.ReportScreen()
        #             actual_wd = caffecopy.water_depths
        #             sys.stdout = sys.__stdout__
        #             del caffecopy

        #         self.caffe.RunSimulation()
        #         self.caffe.ReportScreen()

        #     else:
        #         last_wd = np.zeros_like(self.caffe.DEM, dtype=np.double)

        self.swmm.CloseSimulation()

        if self.report_max == True:
            self.caffe.Reset_WL_EVM()
            self.caffe.max_water_depths = deepcopy(max_mwd)
            self.caffe.water_depths = deepcopy(max_wd)
            self.caffe.water_levels = deepcopy(max_wl)
            self.caffe.ReportFile(self.project_name + '_max_all')

        # save all exchanges between models in a csv file including the exchange time
        name = self.caffe.outputs_path + self.caffe.outputs_name + '_exchange_report.csv'
        with open(name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for entry in self.exchange_amount:
                formatted_entry = [entry[0].strftime(
                    '%Y-%m-%d %H:%M:%S')] + entry[1:]
                writer.writerow(formatted_entry)

        print("\n .....finished bi-directional coupled SWMM & CA-ffé.....")
        print("\n", time.ctime(), "\n")
        print("\n Duration: ", time.time()-self.t, "\n")
