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
                 skip_line, fname, plt, xin, paramin, cl, t_avg):
        self.src, self.case = src, case
        self.lws, self.T, self.skip = lws, time, skip_line
        self.fi, self.fname = format, fname
        self.ds, self.ms, self.ls = ds, ms, ls
        self.type, self.legend = case_type, legend
        self.tformat, self.color = time_format, cl
        self.paramin, self.xin = paramin, xin
        self.t_avg = t_avg
        self.fig1, self.ax1 = plt.subplots(figsize=(11, 7), dpi=90)
        plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
        # self.fig2, self.ax2 = plt.subplots(figsize=(11, 7), dpi=90)
        # plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
        # self.fig3, self.ax3 = plt.subplots(figsize=(11, 7), dpi=90)
        # plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)

        for axis in ['top', 'bottom', 'left', 'right']:
            self.ax1.spines[axis].set_linewidth(2)
            # self.ax2.spines[axis].set_linewidth(2)
            # self.ax3.spines[axis].set_linewidth(2)

    def time_average(self):
        #pass
        return

    def param(self, tf, cyl_d):
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
                if self.t_avg == 'on':
                    print("true")
                else:
                    if abs(self.T[i] - tf[icase]) <= 1e-6:
                        print("The axial distribution of the parameter is taken at time= ", tf[icase], "s")
                        fname = self.fname + '_' + self.tformat.format(self.T[i]) + '.' + self.fi
                        fdata = open(os.path.join(case_path[icase], fname), "r")
                        lines = fdata.readlines()
                        fdata.close()
                        # print(fdata)
                        x = np.zeros(len(lines) - self.skip)
                        para = 0 * x
                        for j in range(self.skip, len(lines)):
                            # print(j, self.skip)
                            line = lines[j]
                            line = line.strip()
                            line = line.split()
                            x[j - self.skip] = float(line[self.xin])
                            para[j - self.skip] = float(line[self.paramin])
                        # break
            self.ax1.plot(x[(x >= 11.1) & (x <= 12)], para[(x >= 11.1) & (x <= 12)] / 0.012,
                          color=self.color[icase], label=self.legend[icase], lw=self.lws,
                          linestyle=self.ls[icase + 1])  # tke / 0.02 ,  vel / 1.2 ** 2
            para_alt = para[(x >= 11.1) & (x <= 12)] / 0.02
            # print(para_alt[len(para_alt) - 1])
            # linestyle=finput.ls[icase]
            # print(x)
            pt[icase] = max(para)
            self.ax1.set_xlim(11, 12.2)
            self.ax1.set_ylim(0, 3)  # vel 1, # TKE 3
            # self.ax1.set_ylim(0, 1)
            # finput.ax1.set_yticks(np.arange(0, 0.36, 0.05))
            self.ax1.legend(loc='upper left', fontsize='24', frameon=False)  # 'upper left' TKE # vel 'lower right'
            self.ax1.set_xlabel(r'x [m]')
            # self.ax1.set_ylabel(r'$U/U_{\infty}$')
            self.ax1.set_ylabel(r'$k/k_{\infty}$')
        rectangle = patches.Rectangle((11.1, 0), cyl_d, 3, alpha=0.1, facecolor="#4d4d4d")  # , 3 tke
        self.ax1.add_patch(rectangle)
        return self.ax1  # , finput.ax2, finput.ax3

    def param_t(self, xc, ts, tf):
        case_path = []
        print("slef-case", self.case)
        for icase in range(len(self.case)):
            if self.case[icase] == 'nocyl':
                case_path.append(self.src + '/' + self.case[icase])
            else:
                case_path.append(self.src + '/' + self.case[icase])  # '/' + self.type +

            print(case_path[icase])
            t = self.T[(self.T >= ts) & (self.T <= tf)]
            para = 0 * t
            jj = 0
            for i in range(len(self.T)):
                if ts <= self.T[i] <= tf:
                    if self.t_avg == 'on':
                        print("true")
                    else:
                        fname = self.fname + '_' + self.tformat.format(self.T[i]) + '.' + self.fi
                        fdata = open(os.path.join(case_path[icase], fname), "r")
                        lines = fdata.readlines()
                        fdata.close()
                        # print(fdata)
                        x = np.zeros(len(lines) - self.skip)

                        for j in range(self.skip, len(lines)):
                            # print(j, self.skip)
                            line = lines[j]
                            line = line.strip()
                            line = line.split()
                            #x[j - self.skip] = float(line[self.xin])
                            if abs(float(line[self.xin]) - xc) <= 1e-6:
                                print("The axial distribution of the parameter is taken at x= ", float(line[self.xin]))
                                para[jj] = float(line[self.paramin])
                                break
                    jj = jj+1
            print(para)
            self.ax1.plot(t[1:len(t)], integrate.cumtrapz(para, t) / 0.02,
                          color=self.color[icase], label=self.legend[icase], lw=self.lws,
                          linestyle=self.ls[icase + 1])  # tke / 0.02 ,  /0.012
            # para_alt = para[(x >= 11.1) & (x <= 12)] / 0.02
            # print(para_alt[len(para_alt) - 1])
            # linestyle=finput.ls[icase]

            self.ax1.set_ylim(0, 12)
            #self.ax1.set_ylim(0, 6)  # vel 1, # TKE 3
            # self.ax1.set_ylim(0, 1)
            # finput.ax1.set_yticks(np.arange(0, 0.36, 0.05))
            self.ax1.legend(loc='upper left', fontsize='24', frameon=False)  # 'upper left' TKE # vel 'lower right'
            self.ax1.set_xlabel(r't [s]')
            self.ax1.set_ylabel(r'$k/k_{\infty}$')

        return self.ax1
