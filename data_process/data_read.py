#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
#from structured_array import str_array
import sys
from data_process.structured_array import str_array


class data_3d:
    # parameters
    # parameterized constructor

    def __init__(self, xpt, ypt, zpt, data_directory, filename, data_type):
        self.xpt, self.ypt, self.zpt = xpt, ypt, zpt
        self.dir, self.fname, self.data_type = data_directory, filename, data_type

    def point_3d(self):
        # print(data_directory, filename)
        print('********* Reading coordinates from csv data starts from ', self.dir,' *************\n')
        if self.data_type == '3d':
            # out_3d = []
            x = pd.read_csv(os.path.join(self.dir, self.fname), usecols=['Points:0']).values[:, 0]
            y = pd.read_csv(os.path.join(self.dir, self.fname), usecols=['Points:1']).values[:, 0]
            z = pd.read_csv(os.path.join(self.dir, self.fname), usecols=['Points:2']).values[:, 0]
            print('********* Reading coordinates from csv data ends *************\n')
            if len(x) != self.xpt*self.ypt*self.zpt:
                print(" the 3d points", len(x), "does not match with points ", self.xpt*self.ypt*self.zpt,
                      "defined in the input file")
                raise ValueError("The number of elements defined in the input file is wrong please check the "
                                 "configuration")
            points = np.zeros((3, len(x)))
            points[0, :], points[1, :], points[2, :] = x, y, z
            print('********* Start converting point data to 3d array *************\n')
            self.point3d = str_array(self.xpt, self.ypt, self.zpt, points, 3)
            print('********* Converting point data to 3d array ends *************\n')
        return self.point3d

    def vel_3d(self):
        print('********* Reading velocity from csv data starts *************\n')
        if self.data_type == '3d':
            u = pd.read_csv(os.path.join(self.dir, self.fname), usecols=['VELOC:0']).values[:, 0]
            v = pd.read_csv(os.path.join(self.dir, self.fname), usecols=['VELOC:1']).values[:, 0]
            w = pd.read_csv(os.path.join(self.dir, self.fname), usecols=['VELOC:2']).values[:, 0]
            V = np.sqrt(u ** 2 + v ** 2 + w ** 2)
            print('********* Reading velocity from csv data ends *************\n')
            vel = np.zeros((4, len(u)))
            vel[0, :], vel[1, :], vel[2, :], vel[3, :] = u, v, w, V
            print('********* Start converting velocity data to 3d array *************\n')
            self.vel3d = str_array(self.xpt, self.ypt, self.zpt, vel, 4)
            print('********* Converting velocity data to 3d array ends *************\n')
        return self.vel3d

    def lev_3d(self):
        print('********* Reading level from csv data starts *************\n')
        if self.data_type == '3d':
            lev = pd.read_csv(os.path.join(self.dir, self.fname), usecols=['LEVEL']).values[:, 0]
            mask = pd.read_csv(os.path.join(self.dir, self.fname), usecols=['vtkValidPointMask']).values[:, 0]
            lev[mask == 0] = -1
            print('********* Reading level from csv data ends *************\n')
            print('********* Start converting levels to 3d array *************\n')
            self.lev3d = str_array(self.xpt, self.ypt, self.zpt, lev, 0)
            self.mask3d = str_array(self.xpt, self.ypt, self.zpt, mask, 0)
            print('********* Converting to levels 3d array ends *************\n')
        return self.lev3d, self.mask3d

    def rho_3d(self):
        print('********* Reading density from csv data starts *************\n')
        if self.data_type == '3d':
            rho = pd.read_csv(os.path.join(self.dir, self.fname), usecols=['DENSI']).values[:, 0]
            print('********* Reading density from csv data ends *************\n')
            print('********* Start converting rho to 3d array *************\n')
            self.rho3d = str_array(self.xpt, self.ypt, self.zpt, rho, 0)
            print('********* Converting rho to 3d array ends *************\n')
        return self.rho3d

    def pressure_3d(self):
        print('********* Reading pressure from csv data starts *************\n')
        if self.data_type == '3d':
            pressure = pd.read_csv(os.path.join(self.dir, self.fname), usecols=['PRESS']).values[:, 0]
            print('********* Reading pressure from csv data ends *************\n')
            print('********* Start converting pressure to 3d array *************\n')
            self.pressure3d = str_array(self.xpt, self.ypt, self.zpt, pressure, 0)
            print('********* Converting pressure to 3d array ends *************\n')
        return self.pressure3d

    def nusgs_3d(self):
        print('********* Reading nusgs from csv data starts *************\n')
        if self.data_type == '3d':
            nusgs = pd.read_csv(os.path.join(self.dir, self.fname), usecols=['VISCO']).values[:, 0]
            print('********* Reading nusgs from csv data ends *************\n')
            print('********* Start converting nusgs to 3d array *************\n')
            self.nusgs3d = str_array(self.xpt, self.ypt, self.zpt, nusgs, 0)
            print('********* Converting nusgs to 3d array ends *************\n')
        return self.nusgs3d

    def tau_3d(self):
        print('********* Reading wall shear stress from csv data starts *************\n')
        if self.data_type == '3d':
            # out_3d = []
            tau1 = pd.read_csv(os.path.join(self.dir, self.fname), usecols=['TANGE:0']).values[:, 0]
            tau2 = pd.read_csv(os.path.join(self.dir, self.fname), usecols=['TANGE:1']).values[:, 0]
            tau3 = pd.read_csv(os.path.join(self.dir, self.fname), usecols=['TANGE:2']).values[:, 0]
            tau = np.sqrt((tau1 * 1e3) ** 2 + (tau2 * 1e3) ** 2 + (tau3 * 1e3) ** 2)
            print('********* Reading wall shear stress from csv data ends *************\n')
            tau = np.zeros((4, len(tau1)))
            tau[0, :], tau[1, :], tau[2, :], tau[3, :] = tau1, tau2, tau3, tau
            print('********* Start converting tau to 3d array *************\n')
            self.tau3d = str_array(self.xpt, self.ypt, self.zpt, tau, 4)
            print('********* Converting tau to 3d array ends *************\n')
        return self.tau3d
