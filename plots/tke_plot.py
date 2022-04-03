import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import integrate
# import glob
# from labellines import labelLine, labelLines
# from scipy import interpolate
import matplotlib.patches as patches

plt.close("all")
# plt.style.use('classic')
font = {'family': 'Arial',
        'weight': 'normal',
        'size': 28}
plt.rc('font', **font)

fig1, ax1 = plt.subplots(figsize=(11, 7), dpi=90)
plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
'''
fig2, ax2 = plt.subplots(figsize=(11, 7), dpi=90)
plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
fig3, ax3 = plt.subplots(figsize=(11, 7), dpi=90)
plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)

for axis in ['top', 'bottom', 'left', 'right']:
    ax1.spines[axis].set_linewidth(2)
    ax2.spines[axis].set_linewidth(2)
    ax3.spines[axis].set_linewidth(2)
'''


# Input Settings starts
class param_axial:
    def __init__(self, src, case, case_type, lws, time, ls, ms, ds, legend, format, time_format,
                 skip_line, fname, plt, xin, paramin):
        self.src, self.case = src, case
        self.lws, self.T, self.skip = lws, time, skip_line
        self.fi, self.fname = format, fname
        self.ds, self.ms, self.ls = ds, ms, ls
        self.type, self.legend = case_type, legend
        self.tformat = time_format
        self.paramin, self.xin = paramin, xin

        self.fig1, self.ax1 = plt.subplots(figsize=(11, 7), dpi=90)
        plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
        #self.fig2, self.ax2 = plt.subplots(figsize=(11, 7), dpi=90)
        #plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
        #self.fig3, self.ax3 = plt.subplots(figsize=(11, 7), dpi=90)
        #plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)

        for axis in ['top', 'bottom', 'left', 'right']:
            self.ax1.spines[axis].set_linewidth(2)
            #self.ax2.spines[axis].set_linewidth(2)
            #self.ax3.spines[axis].set_linewidth(2)

    def param(self, tf):
        case_path = []
        print("slef-case", self.case)
        pt = np.zeros(len(self.case))
        for icase in range(len(self.case)):
            if self.case[icase] == 'nocyl':
                case_path.append(self.src + '/' + self.case[icase])
            else:
                case_path.append(self.src + '/' + self.case[icase])  # '/' + self.type +

            print(case_path[icase])

            for i in range(len(self.T)):
                if self.T[i] == tf:
                    fname = self.fname + '_' + self.tformat.format(self.T[i]) + '.' + self.fi
                    fdata = open(os.path.join(case_path[icase], fname), "r")
                    lines = fdata.readlines()
                    fdata.close()
                    #print(fdata)
                    x = np.zeros(len(lines) - self.skip)
                    para = 0 * x
                    for j in range(self.skip, len(lines)):
                        # print(j, self.skip)
                        line = lines[j]
                        line = line.strip()
                        line = line.split()
                        x[j - self.skip] = float(line[self.xin])
                        para[j - self.skip] = float(line[self.paramin])
                    break
            self.ax1.plot(x+0.8, para,
                          self.ms[icase], label=self.legend[icase], lw=self.lws,
                          linestyle=self.ls[icase], dashes=self.ds[icase])
            print(x)
            pt[icase] = max(para)

            # finput.ax1.set_xlim(4, 10.5)
            # finput.ax1.set_ylim(0, 0.2)
            # finput.ax1.set_yticks(np.arange(0, 0.36, 0.05))
            self.ax1.legend(loc='best', fontsize='24', frameon=False)
            self.ax1.set_xlabel(r'x [m]')
            self.ax1.set_ylabel(r'$k \rm [m^{2}/s^{2}]$')
        rectangle = patches.Rectangle((11.1, 0), 0.14, max(pt), alpha=0.1, facecolor="blue")  # , linewidth=1)
        self.ax1.add_patch(rectangle)
        return self.ax1  # , finput.ax2, finput.ax3


