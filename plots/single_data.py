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
class paraview_plot:

    def __init__(self, src, case, case_type, lws, time, ts, tf, xloc, ls, ms, ds, legend, format, time_format, plt):
        self.src, self.case = src, case
        self.lws, self.T = lws, time
        self.locx, self.fi = xloc, format
        self.ds, self.ms, self.ls = ds, ms, ls
        self.type, self.legend = case_type, legend
        self.ts, self.tf, self.tformat = ts, tf, time_format

    def pvfileread(self):

        # Figure set up
        case_path = []
        for icase in range(len(self.case)):
            if icase == 0:
                case_path.append(self.src + '/' + self.case[icase])
            else:
                case_path.append(self.src + '/' + self.case[icase])  # self.type + '/' +

            t = self.T[(self.T >= self.ts) & (self.T <= self.tf)]
            if icase == 0:
                pf, tf, kf = np.zeros((len(self.case), len(t))), np.zeros((len(self.case), len(t))), \
                             np.zeros((len(self.case), len(t)))
                tf_cum = np.zeros((len(self.case), len(t) - 1))
                kf_cum, pf_cum = 0 * tf_cum, 0 * tf_cum
                T = np.zeros(len(t))
            jj = 0
            for i in range(len(self.T)):
                if self.ts <= self.T[i] <= self.tf:
                    fname = 'eflux_' + str(i) + '.' + self.fi
                    kf[icase, jj] = pd.read_csv(os.path.join(case_path[icase], fname), usecols=['KF:0']).values[:, 0]
                    pf[icase, jj] = pd.read_csv(os.path.join(case_path[icase], fname), usecols=['PF:0']).values[:, 0]
                    tf[icase, jj] = pd.read_csv(os.path.join(case_path[icase], fname), usecols=['pressureFlux:0']).values[
                                    :, 0]
                    T[jj] = pd.read_csv(os.path.join(case_path[icase], fname), usecols=['Time']).values[:, 0]
                    jj = jj + 1

        return self, case_path, pf, kf, tf, t
