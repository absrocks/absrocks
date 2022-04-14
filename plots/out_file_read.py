import os
import numpy as np


class out_file_read:

    def __init__(self, src, case, case_type, lws, time, ts, tf, xloc, ls, ms, ds, legend, format, time_format,
                 skip_line, pfin, kfin, tfin, xlocin, fname, plt, colorke, colorpe):
        self.src, self.case = src, case
        self.lws, self.T, self.skip = lws, time, skip_line
        self.locx, self.fi, self.fname = xloc, format, fname
        self.ds, self.ms, self.ls = ds, ms, ls
        self.type, self.legend = case_type, legend
        self.ts, self.tf, self.tformat = ts, tf, time_format
        self.pfin, self.kfin, self.tfin, self.xin = pfin, kfin, tfin, xlocin
        self.cke, self.cpe = colorke, colorpe

        self.fig1, self.ax1 = plt.subplots(figsize=(11, 7), dpi=90)
        plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
        self.fig2, self.ax2 = plt.subplots(figsize=(11, 7), dpi=90)
        plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
        self.fig3, self.ax3 = plt.subplots(figsize=(11, 7), dpi=90)
        plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)

        for axis in ['top', 'bottom', 'left', 'right']:
            self.ax1.spines[axis].set_linewidth(2)
            self.ax2.spines[axis].set_linewidth(2)
            self.ax3.spines[axis].set_linewidth(2)

    def fread(self, locx, ts):
        # Figure set up

        case_path = []
        for icase in range(len(self.case)):
            if self.case[icase] == 'nocyl':
                case_path.append(self.src + '/' + self.case[icase])
            else:
                case_path.append(self.src + '/' + self.case[icase])  # '/' + self.type +

            t = self.T[(self.T >= ts) & (self.T <= self.tf)]
            if icase == 0:
                pf, tf, kf = np.zeros((len(self.case), len(t))), np.zeros((len(self.case), len(t))), \
                             np.zeros((len(self.case), len(t)))
            jj = 0
            for i in range(len(self.T)):
                if ts <= self.T[i] <= self.tf:
                    fname = self.fname + '_' + self.tformat.format(self.T[i]) + '.' + self.fi
                    fdata = open(os.path.join(case_path[icase], fname), "r")
                    lines = fdata.readlines()
                    fdata.close()
                    # print(case_path)
                    x = np.zeros(len(lines) - self.skip)
                    for j in range(self.skip, len(lines)):
                        # print(j, self.skip)
                        line = lines[j]
                        line = line.strip()
                        line = line.split()
                        x[j - self.skip] = float(line[self.xin])
                        if len(locx) == 1:
                            xx = locx[0]
                        else:
                            xx = locx[icase]
                        if x[j - self.skip] >= xx:
                            # print(x[j - self.skip])
                            pf[icase, jj], kf[icase, jj], tf[icase, jj] = float(line[self.pfin]), \
                                                                          float(line[self.kfin]), float(line[self.tfin])
                            break
                    jj = jj + 1

        return self, case_path, pf, kf, tf, t
