#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# parameters

import csv
import os
import sys
import numpy as np
import csv

sys.dont_write_bytecode = True


def t_input(i, directory, time_dir):
    # data_directory = os.path.join(os.getcwd(), directory)
    # sys.path.append(data_directory)
    fname = 'time'
    fe = '_' #'_'
    filename = fname + fe + str(i) + '.csv'
    input_path = os.path.join(directory, filename)
    #print(input_path)
    # plane = np.genfromtxt(os.path.join(data_directory, filename), delimiter=';')[:,:-1]

    # plane = np.loadtxt(os.path.join(data_directory, filename), delimiter=",", skiprows=1, dtype=str)
    t = 0
    with open(input_path, 'r') as f:
        reader = csv.reader(f)
        #print(reader)
        for row in reader:
            #print('row[0]', row)
            if t > 0:
                time = float(row[0])
            t = t + 1
    return time
