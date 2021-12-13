import os

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import glob
from labellines import labelLine, labelLines

plt.close("all")
# plt.style.use('classic')
font = {'family': 'Arial',
        'weight': 'normal',
        'size': 28}
plt.rc('font', **font)


def interpolation(e, t, nt):
    from scipy import interpolate

    # total energy
    ti = np.linspace(0, 1.8, nt)  # 100 represents number of points to make between T.min and T.max
    spl = interpolate.interp1d(t, e, fill_value='extrapolate', kind='linear')  # BSpline object
    energy = spl(ti)
    return energy, ti

depth_plot = 'on'
reflection_wave= 'off'
tsunami_tr = '/Volumes/Arya/PHD_2019/tsunami/'  # shortcut of tsunami directory in Transcend
tsunami_hp = '/Volumes/HP/PHD_2019/tsunami/'  # shortcut of tsunami directory in HP
parent_dir = '/media/abhishek/WD-phd/transcend/PHD_2019/tsunami/dam_break_wet_case/arnason-multi/rigid'
from time_input import t_input

ds = [(), (1, 1), (5, 5), (3, 5, 1, 5, 1, 5), (8, 4, 2, 4, 2, 4), (1, 10), (5, 10)]
case_path, cc = os.path.split(os.getcwd())
ls = ['solid', 'dotted', 'dashed', 'dashdot', 'dashdot', 'dotted', 'dashed']
case_dir = 'energy_data'
case_type = ['nocyl', 'single-cyl', 'double-cyl', 'four-cyl']  # 'no_cyl'
case_cyl = 'double_cyl_d_0.4'
case_path = []
time_dir = 'csv_data/time'
ms = ['x', '^', '*', 's', 'd', 'o', '+', '_']
#case_type = parent_dir + case_type + '/' + case_cyl
# case_path = case_path + '/' + case_type
time_directory = os.path.join(parent_dir, time_dir)
t_index = len(glob.glob1(time_directory, "*.csv"))
T = np.zeros(t_index)
for i in range(1, t_index):
    T[i] = t_input(i, time_directory, 'time')

if depth_plot == 'on':
    fig1, ax1 = plt.subplots(figsize=(15, 5), dpi=90)
    rect = patches.Rectangle((11.1, 0), 0.4, 3, linewidth=2, edgecolor='k', facecolor='none')
    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(2)
    for icase in range(len(case_type)):
        fig2, ax2 = plt.subplots(figsize=(15, 5), dpi=90)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax2.spines[axis].set_linewidth(2)
        case_path.append(parent_dir + '/' + case_dir + '/' + case_type[icase])
        tt = np.asarray([3.8, 4.6, 6, 7, 8, 9, 10, 11])     # , '7.0', '8.0', '9.1']  # 8hills
        #tt = np.asarray([10])
        dir = 'depth_data'
        # tt = np.round(np.arange(0.40, 2, 0.20), 1)
        l = 0
        #t_steps = len(glob.glob1(os.path.join(case_type, dir), "*.dat"))
        xvals = []
        #T = T[T <= 9.2]

        for i in range(len(T)):
            if l == np.size(tt):
                print('l', l)
                break
            print('index', i, (np.round(T[i], 2)), tt[l])
            if (np.round(T[i], 2)) == tt[l]:
                print('t', T[i], i)
                fname = 'eflux_' + str(T[i]) + '.dat'
                fdata = open(os.path.join(case_path[icase], fname), "r")
                lines = fdata.readlines()
                fdata.close()

                eta, x, v = np.zeros(len(lines)), np.zeros(len(lines)), np.zeros(len(lines))
                for j in range(len(lines)):
                    line = lines[j]
                    line = line.strip()
                    line = line.split()
                    eta[j] = float(line[1])
                    x[j] = float(line[4])
                    #v[j] = float(line[3])
                x, eta = x[x <= 11.1], eta[x <= 11.1]
                ax2.plot(x, eta, label='t=' + str(np.round(T[i], 1)) + 's', lw=2)
                l = l + 1
                #ax2.add_patch(rect)
        xx, etac = np.zeros((len(case_type), len(x))), np.zeros((len(case_type), len(x)))
        xx[icase, :] = x
        etac[icase,:] = eta
        xvals = [11, 11, 10.8, 10.4, 10.2, 9.92, 9.65, 9.35]  # , 34, 30, 26] # 2hills,
        #ax1.plot(x, eta,
        #         label='t=' + str(np.round(T[i], 1)) + 's', lw=2)
        labelLines(plt.gca().get_lines(), align=False, color='k', fontsize=10, xvals=xvals, zorder=2) #

        #ax2.add_patch(rect)
        #ax1.set_xlim(8, 11.5)
        #ax1.set_ylim(0.04, 0.12)

        ax2.set_xlim(8, 11.5)
        ax2.set_ylim(0.04, 0.12)
        # xdata = line.get_xdata()
        # print(xdata)
        # ax1.legend(loc='best', fontsize='15', frameon=False)
        #ax1.set_ylabel(r'$\eta\,{\rm (m)}$')
        #ax1.set_xlabel(r'X (m)')
        ax2.set_ylabel(r'$\eta\,{\rm (m)}$')
        ax2.set_xlabel(r'X (m)')
        #ax1.plot(x, etac[icase,:], label=case_type[icase], lw=2)
        #ax1.legend(loc='best', fontsize='15', frameon=False)
if reflection_wave =='on':
    cyl = [0, 1, 2, 4]
    fig1, ax1 = plt.subplots(figsize=(11, 7), dpi=90)
    xx, etac = np.zeros(len(case_type)), np.zeros(len(case_type))
    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(2)
    for icase in range(len(case_type)):
        case_path.append(parent_dir + '/' + case_dir + '/' + case_type[icase])
        #tt = np.asarray([3.8, 4.6, 6, 7, 8, 9, 10, 11])     # , '7.0', '8.0', '9.1']  # 8hills
        tt = np.asarray([10])
        l = 0
        xvals = []
        for i in range(len(T)):
            if l == np.size(tt):
                print('l', l)
                break
            #print('index', i, (np.round(T[i], 2)), tt[l])
            if (np.round(T[i], 2)) == tt[l]:
                print('t', T[i], i)
                fname = 'eflux_' + str(T[i]) + '.dat'
                fdata = open(os.path.join(case_path[icase], fname), "r")
                lines = fdata.readlines()
                fdata.close()

                eta, x = np.zeros(len(lines)), np.zeros(len(lines))
                for j in range(len(lines)):
                    line = lines[j]
                    line = line.strip()
                    line = line.split()
                    eta[j] = float(line[1])
                    x[j] = float(line[4])
                x, eta = x[x <= 11.1], eta[x <= 11.1]
                #ax2.plot(x, eta,
                #         label='t=' + str(np.round(T[i], 1)) + 's', lw=2)
                l = l + 1
                #ax2.add_patch(rect)

        etac[icase] = max(eta)
        print(etac[icase])
    if icase >0:
        print(etac)
        xmin, xmax = min(cyl[1:icase + 1]), max(cyl[1:icase + 1])
        ax1.set_xlim(0.8, 4.2)
        #ax1.set_ylim(0.04, 0.12)
        x_ticks_labels = ['1 cyl', '2 cyl', '4 cyl']
        ax1.set_ylabel(r'$H_{R}/H_{B}$')
        ax1.set_xlabel(r'Number of cylinders')
        ax1.set_xticks(cyl[1:icase + 1])
        #ax1.set_xticklabels(x_ticks_labels, rotation='horizontal', fontsize=18)
        ax1.plot(cyl[1:len(cyl)], etac[1:len(etac)]/etac[0], 'b*-', lw=2)
        #ax1.legend(loc='best', fontsize='15', frameon=False)

plt.show()



