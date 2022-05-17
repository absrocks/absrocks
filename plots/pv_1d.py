# import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import integrate
import pandas as pd
# from labellines import labelLine, labelLines
# from scipy import interpolate
import matplotlib.patches as patches


# plt.close("all")
# plt.style.use('classic')
# font = {'family': 'Arial',
##        'weight': 'normal',
#        'size': 28}
# plt.rc('font', **font)
# fig1, ax1 = plt.subplots(figsize=(11, 7), dpi=90)

# plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
# fig2, ax2 = plt.subplots(figsize=(11, 7), dpi=90)
# plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
# fig3, ax3 = plt.subplots(figsize=(11, 7), dpi=90)


# plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)


# Input Settings starts
class pv_plot:

    def __init__(self, src, case, case_type, lws, xloc, ls, ms, ds, legend, format, time_format, plt, colorke, colorpe):
        self.src, self.case = src, case
        self.lws = lws
        self.locx, self.fi = xloc, format
        self.ds, self.ms, self.ls = ds, ms, ls
        self.type, self.legend = case_type, legend
        self.tformat = time_format
        self.cke, self.cpe = colorke, colorpe

        self.fig1, self.ax1 = plt.subplots(figsize=(11, 7), dpi=90)
        plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)

    def pveta(self):
        # Figure set up
        case_path = []
        for icase in range(len(self.case)):
            case_split = self.case[icase].split('/')
            case_path.append(self.src + '/' + case_split[0] + '/' + case_split[1])
            frep = case_split[1] + '-' + case_split[2]
            print(self.src,  self.case[icase], frep)
            fname = frep + '.csv'
            t = pd.read_csv(os.path.join(case_path[icase], fname), usecols=['Time']).values[:, 0]
            eta_max = pd.read_csv(os.path.join(case_path[icase], fname), usecols=['max Z ( block=1)']).values[:, 0]
            eta_avg = pd.read_csv(os.path.join(case_path[icase], fname), usecols=['avg Z ( block=1)']).values[:, 0]
            self.ax1.plot(t+3.6, eta_avg * 1000, self.ms[icase], color=self.cke[icase+1],
                          label=self.legend[icase], lw=self.lws,
                          linestyle=self.ls[icase])

            self.ax1.set_xlim(5, 10.5)
            self.ax1.legend(loc='best', fontsize='24', frameon=False)
            self.ax1.set_ylim(80, 121)
            self.ax1.set_yticks(np.arange(80, 121, 10))

            self.ax1.set_xlabel(r'Time [s]')
            self.ax1.set_ylabel(r'$\eta \, [mm]$')

        return self.ax1
