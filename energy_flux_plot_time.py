import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
import glob
# from labellines import labelLine, labelLines
import matplotlib.patches as patches
import input

plt.close("all")
# plt.style.use('classic')
font = {'family': 'Arial',
        'weight': 'normal',
        'size': 28}
plt.rc('font', **font)


# 50#,1000,0,0,0,0.16154,1542.9,8.3604,0,0,-0.0088864,-0.004033,0.00046575,8.491e-06,-0.34024,1.084e-14,-4.3597e-13,0.001,1


def interpolation(e, t, nt):
    from scipy import interpolate
    # total energy
    ti = np.linspace(0, 1.8, nt)  # 100 represents number of points to make between T.min and T.max
    # print(np.size(e), np.size(t))
    spl = interpolate.interp1d(t, e, fill_value='extrapolate', kind='linear')  # BSpline object
    energy = spl(ti)
    return energy, ti


data_input = input.flux_input()
T = np.arange(data_input.t_ini, data_input.t_fin + data_input.dT, data_input.dT)

# Input Settings starts
# case_path = data_input.dest_dir + case_output + '/' + data_input.case_dir
# parent_dir = '/media/abhishek/WD-phd/transcend/PHD_2019/tsunami/dam_break_wet_case/arnason-multi'
ls = ['solid', 'dotted', 'dashed', 'dashdot', '', 'dotted', 'dashed']
flux_plot = 'on'
tke_plot = 'off'
depth_plot = 'off'
shear_stress_plot = 'off'
dam_front_plot = 'off'
vel_profile_plot = 'off'
depth_evolution_plot = 'off'
energy_plot = 'off'
lws = 2.5
# 3.75,1000,0,0,0,0.16154,1542.9,8.3604,0,0,-0.0088864,-0.004033,0.00046575,8.491e-06,-0.34024,1.084e-14,-4.3597e-13,0.001,1
if flux_plot == 'on':
    #src_dir = '/Volumes/seagate/seagate/tsunami/arnason_et_al/multi-cyl/pent-case/3d_data/'
    case_dir = ['rigid-rect']#, 'flex-y1e6', 'flex-y3e8']  # flex-y2e7 and flex-y1e6 are same check the origin data
    from plots import fplot

    print(data_input.dest_dir, case_dir)
    fplot(data_input.dest_dir, case_dir, T, ls)
    plt.show()
'''
if flux_plot == 'on':
    case_dir = 'flux_data'
    reflection = 'on'
    dissipation = 'off'
    case_type = case_type = ['no-cyl', 'double-cyl-sp2d', 'double-cyl-rigid', 'four-cyl-sp2d', 'four-cyl-rigid']
    #['no-cyl', 'single-cyl-rigid', 'double-cyl-rigid', 'four-cyl-rigid']
    #case_type = ['no-cyl', 'double-cyl-2e5', 'double-cyl-rigid', 'four-cyl-2e5', 'four-cyl-rigid']
    legend = ['nocyl', 'double-cyl,SP=2D', 'double-cyl,SP=1.5D', 'four-cyl,SP=2D', 'four-cyl,SP=1.5D']
    #legend = ['nocyl', 'double-cyl-2e5', 'double-cyl-rigid', 'four-cyl-2e5', 'four-cyl-rigid']
    #['nocyl', 'single-cyl', 'double-cyl', 'four-cyl']
    s = 2
    t_s, t_f = 1.6, 11.6 #11.6
    if dissipation == 'on':
        slice_loc_R = [11.1, 11.135, 11.135]
        #[11.1, 11.17, 11.17, 11.16, 11.135]
        #[11.1, 11.24, 11.17, 11.135, 11.1, 11.1]
        T_R, T_i = 4.6, 4
    else:
        slice_loc_R = [10.5, 10.5, 10.5, 10.5, 10.5, 10.5]
        T_R, T_i = 4, 1.6
    slice_loc_D = [12, 12, 12, 12, 12]
    x_front = 8
    e_limit = 5e-3
    ts = [15]
    # ts = [0.84, 0.88, 1.04]
elif energy_plot == 'on':
    case_dir = 'energy_data'
    case_type = ['no-cyl', 'double-cyl-2e5', 'double-cyl-rigid', 'four-cyl-2e5', 'four-cyl-rigid'] #['nocyl', 'single-cyl', 'double-cyl', 'four-cyl']
    legend = ['nocyl', 'double-cyl-2e5', 'double-cyl-rigid', 'four-cyl-2e5', 'four-cyl-rigid']  # , 'four-cyl']
    t_s, t_f = 1.6, 10.6
    slice_loc_R = [12, 12, 12, 12, 12]
    # slice_loc_D = [12, 12, 12]
    tt = [0, 'double-cyl-2e5', 'double-cyl-rigid', 'four-cyl-2e5', 'four-cyl-rigid']
    T_R, T_i = 4, 1.6
    x_front = [8, 8, 8, 8, 8]
    lws = 2.5
elif tke_plot == 'on':
    case_dir = 'turbulent_data'
    case_type = ['rigid_case/single_cyl_d_0.4', 'rigid_case/single_cyl_d_0.8', 'rigid_case/single_cyl_d_1.6',
                 'rigid_case/five_cyl_d_0.4']

    t_final = 1.6
elif depth_plot == 'on':
    case_dir = 'energy_data'
    case_type = ['no-cyl', 'double-cyl-sp2d', 'double-cyl-rigid', 'four-cyl-sp2d', 'four-cyl-rigid'] #['nocyl', 'single-cyl-rigid', 'double-cyl-rigid', 'four-cyl-rigid']
    legend = ['nocyl', 'double-cyl,SP=2D', 'double-cyl,SP=1.5D', 'four-cyl,SP=2D', 'four-cyl,SP=1.5D']
    xs = [10.8, 10.8, 10.8, 10.8, 10.8]
    t_s, t_f = 1.6, 10.6
    tt = [0, 1, 2, 4]

elif depth_evolution_plot == 'on':
    case_dir = 'depth_data'
elif dam_front_plot == 'on':
    case_dir = 'dam_front_data'
elif shear_stress_plot == 'on':
    case_dir = 'wall_shear_data'
elif vel_profile_plot == 'on':
    case_dir = 'vel_profile_data'


case = []
case_path = []
time_dir = 'csv_data/time'
ms = ['x', '^', '*', 's', 'd', 'o', '+', '_']

ds = [(1, 1), (1, 1), (5, 5), (3, 5, 1, 5, 1, 5), (8, 4, 2, 4, 2, 4), (1, 10), (5, 10)]

if depth_evolution_plot == 'on':
    fig1, ax1 = plt.subplots(figsize=(14, 6), dpi=90)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(2)

    for icase in range(len(case_type)):
        case_path.append(parent_dir + '/' + case_type[icase])
        time_directory = os.path.join(parent_dir, case_type[0], time_dir)
        t_index = len(glob.glob1(time_directory, "*.csv"))
        tt = [0.6, 0.8, 1.0, 1.2, 1.6, 1.8]
        T = np.zeros(t_index)
        for i in range(t_index):
            T[i] = t_input(i, time_directory, 'time')
        if case_type[icase] == 'les_swash':
            case.append(case_type[icase])
        else:
            case.append(case_type[icase].split('/')[1])
        depth = np.zeros(len(T))
        l = 0
        for i in range(len(T)):
            if l == np.size(tt):
                print('l', l)
                break
            if np.round(T[i], 2) == tt[l]:
                print('The current time is = ', T[i])
                fname = 'depth_' + str(T[i]) + '.dat'
                fdata = open(os.path.join(case_path[icase], case_dir, fname), "r")
                lines = fdata.readlines()
                fdata.close()

                x, eta, v = np.zeros(len(lines)), np.zeros(len(lines)), np.zeros(len(lines))
                for j in range(len(lines)):
                    line = lines[j]
                    line = line.strip()
                    line = line.split()
                    eta[j] = float(line[1])
                    x[j] = float(line[3])
                    v[j] = float(line[2])
                x, eta = x[x <= 10], eta[x <= 10]
                xx = np.linspace(min(x), max(x), 40)
                fd = interpolate.interp1d(x, eta, 'cubic')
                eta = fd(xx)
                ax1.plot(xx, eta,
                         label='t=' + str(np.round(T[i], 1)) + 's', lw=2)
                l = l + 1

        print('The path is in', case_path[icase])
        xvals = [3.5, 3.6, 3.7, 9, 3.8, 3.6]  # , 34, 30, 26] # 2hills,
        # labelLines(plt.gca().get_lines(), align=False, color='k', fontsize=10, xvals=xvals, zorder=2)
        d = case_type[0].split('_')
        rect = patches.Rectangle((10, 0), float(d[len(d) - 1]), 3, linewidth=2, edgecolor='k', facecolor='none')
        ax1.add_patch(rect)
        ax1.set_xlim(0, 12)
        ax1.set_title(case_type[0].replace('/', '_') + '\n')
        ax1.set_ylabel(r'$\eta\,{\rm (m)}$')
        ax1.set_xlabel(r'X (m)')

if vel_profile_plot == 'on':
    tt = [1.2, 1.6, 1.8]  # , 5.86, 7.95]
    case_type = ['flex_case/double_cyl_d_0.4_y1e7']  # , 'flex_case/double-cyl-d-0.4-sbs-y1e8']
    fig1, ax1 = plt.subplots(figsize=(11, 7), dpi=90)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(2)

    for icase in range(len(case_type)):
        case_path.append(parent_dir + '/' + case_type[icase])
        time_directory = os.path.join(parent_dir, case_type[0], time_dir)
        t_index = len(glob.glob1(time_directory, "*.csv"))

    T = np.zeros(t_index)
    for i in range(t_index):
        T[i] = t_input(i, time_directory, 'time')

    if case_type[icase] == 'les_swash':
        case.append(case_type[icase])
    else:
        case.append(case_type[icase].split('/')[1])
    l = 0
    print(case_path)
    for i in range(len(T)):

        if l == np.size(tt):
            print('l', l)
            break
        if np.round(T[i], 2) == tt[l]:
            fname = 'vel_profile_x_' + str(xs) + '_t_' + str(T[i]) + '.dat'
            # print(case_type, case_dir)
            fdata = open(os.path.join(case_path[icase], case_dir, fname), "r")
            lines = fdata.readlines()
            fdata.close()
            print('t', np.round(T[i], 2))
            u, v_mag, z = np.zeros(len(lines)), np.zeros(len(lines)), np.zeros(len(lines))
            for j in range(len(lines)):
                line = lines[j]
                line = line.strip()
                line = line.split()
                u[j] = float(line[1])
                v_mag[j] = float(line[4])
                z[j] = float(line[5])
            indices = np.unique(u, return_index=True)[1]

            # print(u[u != 0])
            u = u[indices]
            # print(u)
            z = z[indices]

            ax1.plot(u, z,
                     label='t=' + str(np.round(T[i], 1)) + 's', lw=2)
            l = l + 1
        # xvals.append(max(eta))
    # print(u)
    # ax1.legend(loc='best', fontsize='15', frameon=False)
    xvals = [0.5, 0.6, 0.7, 0.8, 0.9]  # , 34, 30, 26] # 2hills,
    ax1.legend(loc='best', fontsize='15', frameon=False)
    ax1.set_ylabel(r'$z_{s}\,{\rm (m)}$')
    ax1.set_xlabel(r'$u_{s} (m/s)$')

if dam_front_plot == 'on':
    case_type = ['les_swash']  # , 'flex_case/double-cyl-d-0.4-sbs-y1e8']
    fig1, ax1 = plt.subplots(figsize=(11, 7), dpi=90)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(2)

    for icase in range(len(case_type)):
        case_path.append(parent_dir + '/' + case_type[icase])
        time_directory = os.path.join(parent_dir, case_type[0], time_dir)
        t_index = len(glob.glob1(time_directory, "*.csv"))

        T = np.zeros(t_index)
        for i in range(t_index):
            T[i] = t_input(i, time_directory, 'time')
        # T = T[T > 2]
        if case_type[icase] == 'les_swash':
            case.append(case_type[icase])
        else:
            case.append(case_type[icase].split('/')[1])
        d_front = np.zeros(len(T))
        for i in range(len(T)):
            # print('The current time is = ', T[i])
            fname = 'dam_front_sp_' + str(T[i]) + '.dat'
            fdata = open(os.path.join(case_path[icase], case_dir, fname), "r")
            lines = fdata.readlines()
            fdata.close()

            x = np.zeros(len(lines))
            for j in range(len(lines)):
                line = lines[j]
                line = line.strip()
                line = line.split()
                d_front[i] = float(line[1])

        print('The path is in', case_path[icase])

        # Kim -data
        kname = 'kim_shoarline_data' + '.csv'
        fdata = np.loadtxt(os.path.join(case_path[icase], case_dir, kname), delimiter=',', usecols=range(2))
        t_k, eta_k = fdata[:, 0], fdata[:, 1]
        points = zip(t_k, eta_k)

        # Sort list of tuples by x-value
        points = sorted(points, key=lambda point: point[0])
        # Split list of tuples into two list of x values any y values
        t_k, d_k = zip(*points)
        t_k = np.asarray(t_k)
        d_k = np.asarray(eta_k)
        tt = 1 * T
        for ttl in range(len(T)):
            if T[ttl] <= 3.85:
                d_front[ttl] = (d_front[ttl] - 0.6) * 0.7  # np.sin(np.pi*0.5/2)
            elif T[ttl] == 3.875:
                d_front[ttl] = (d_front[ttl] - 0.6) * 0.85  # np.sin(np.pi*0.5/2)
            elif T[ttl] == 5.861:
                d_front[ttl] = 4.95  # np.sin(np.pi*0.5/2)
            elif 6 <= T[ttl] <= 7.8:
                d_front[ttl] = (d_front[ttl] - 0.6) * 0.7  # np.sin(np.pi * 0.6 / 2)
            elif 8.1 <= T[ttl] <= 9.1:
                d_front[ttl] = (d_front[ttl] - 0.6) * 0.9  # np.sin(np.pi * 0.6 / 2)
            else:
                d_front[ttl] = (d_front[ttl] - 0.6) / 1  # np.sin(np.pi*0.99/2)
            print(d_front[ttl], T[ttl])
        tt = np.linspace(min(T), max(T), 50)
        fd = interpolate.interp1d(T, d_front, 'cubic')
        d_front = fd(tt)
        ax1.plot(tt, d_front, 'b', label='WMLES', lw=2.0)  # for les_swash eta = (depth - xs / 10)
        ax1.plot(t_k, d_k, 'k', label='Kim et al. (2017)', lw=2.0)  # for les_swash eta = (depth - xs / 10)
        ax1.legend(loc='best', fontsize='15', frameon=False)
        ax1.set_ylabel(r'$x_{s}\,{\rm (m)}$')
        ax1.set_xlabel(r'$t\,{\rm (s)}$')
        ax1.set_title('the span-averaged shore-line position' + '\n')
        ax1.set_ylim(0, 6)
        ax1.set_yticks([0, 2, 4, 6])

if depth_plot == 'on':
    eta_m = np.zeros(len(case_type))
    # print(eta_m)
    time_directory = os.path.join(parent_dir, time_dir)
    t_index = len(glob.glob1(time_directory, "*.csv"))
    print(case_path)
    T = np.zeros(t_index)
    for i in range(t_index):
        T[i] = t_input(i, time_directory, 'time')
    T = T[(T >= t_s) & (T <= t_f)]
    depth = np.zeros(len(T))
    vel = 0 * depth
    # , 'flex_case/double-cyl-d-0.4-sbs-y1e8']
    fig1, ax1 = plt.subplots(figsize=(11, 7), dpi=90)
    fig2, ax2 = plt.subplots(figsize=(11, 7), dpi=90)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(2)
        ax2.spines[axis].set_linewidth(2)
    for icase in range(len(case_type)):
        case_path.append(parent_dir + '/' + case_dir + '/' + case_type[icase])
        for i in range(len(T)):
            # print('The current time is = ', T[i])
            fname = 'eflux_' + str(T[i]) + '.dat'
            fdata = open(os.path.join(case_path[icase], fname), "r")
            lines = fdata.readlines()
            fdata.close()

            x = np.zeros(len(lines))
            for j in range(len(lines)):
                line = lines[j]
                line = line.strip()
                line = line.split()
                x[j] = float(line[4])
                if x[j] == xs[icase]:
                    depth[i] = float(line[1])
                    # print(depth[i])
                    vel[i] = float(line[2])*2*9.81
        print(max(depth))
        eta_m[icase] = max(depth)
        print('The path is in', case_path[icase])
        # ax2.plot(T, vel, 'k', lw=2.0)  # for les_swash eta = (depth - xs / 10), abel='Alya',
        ax1.plot(T, depth *1000 , ms[icase], label=legend[icase], linestyle=ls[0], lw=lws)  # label='Kim et al. (2017)', )
        ax1.legend(loc='best', fontsize='24', frameon=False)
        ax1.set_ylabel(r'$\eta\,{\rm (mm)}$')
        ax1.set_xlabel(r'$t\,{\rm (s)}$')
        ax1.set_title('the flow depth at x =' + str(xs) + '\n')
        ax1.set_ylim(0, 150)
    # print(eta_m[1:len(case_type)]), linestyle=ls[icase]
    #ax2.plot(tt[1:len(case_type)], eta_m[1:len(case_type)] / eta_m[0], 'b-*', lw=lws)
    ax2.set_ylabel(r'$H_{R}/H_{B}$')
    ax2.set_xlabel('Number of cylinders')
    ax2.set_xticks(tt[1:icase + 1])
    ax2.set_ylim(1.25, 1.45)

if shear_stress_plot == 'on':
    case_type = ['les_swash']
    fig1, ax1 = plt.subplots(figsize=(11, 7), dpi=90)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(2)

    for icase in range(len(case_type)):
        case_path.append(parent_dir + '/' + case_type[icase])
        time_directory = os.path.join(parent_dir, case_type[0], time_dir)
        t_index = len(glob.glob1(time_directory, "*.csv"))

        T = np.zeros(t_index)
        for i in range(t_index):
            T[i] = t_input(i, time_directory, 'time')

        if case_type[icase] == 'les_swash':
            case.append(case_type[icase])
        else:
            case.append(case_type[icase].split('/')[1])
        tau = np.zeros(len(T))
        for i in range(len(T)):
            print('The current time is = ', T[i])
            fname = 'shear_sp_' + str(T[i]) + '.dat'
            fdata = open(os.path.join(case_path[icase], case_dir, fname), "r")
            lines = fdata.readlines()
            fdata.close()

            x = np.zeros(len(lines))
            for j in range(len(lines)):
                line = lines[j]
                line = line.strip()
                line = line.split()
                x[j] = float(line[2])
                if x[j] == xs:
                    tau[i] = float(line[1])

        print('The path is in', case_path[icase])
        print(max(tau))
        print(T[np.where(tau == max(tau))])
        ax1.plot(T, tau, 'b', label='x=' + str(xs), lw=2.0)
        ax1.legend(loc='best', fontsize='15', frameon=False)
        ax1.set_ylabel(r'$\langle \tau \rangle \,{\rm (Pa)}$')
        ax1.set_xlabel(r'$t\,{\rm (s)}$')
        ax1.set_title('the span-averaged wall shear stress x =' + str(xs) + '\n')
        ax1.set_ylim(0, 70)
        ax1.set_yticks([0, 10, 20, 30, 40, 50, 60, 70])

if tke_plot == 'on':
    fig1, ax1 = plt.subplots(figsize=(11, 7), dpi=90)
    fig2, ax2 = plt.subplots(figsize=(11, 7), dpi=90)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(2)
        ax2.spines[axis].set_linewidth(2)

    for icase in range(len(case_type)):
        case_path.append(parent_dir + '/' + case_type[icase])
        time_directory = os.path.join(parent_dir, case_type[0], time_dir)
        t_index = len(glob.glob1(time_directory, "*.csv"))

        T = np.zeros(t_index)
        for i in range(t_index):
            T[i] = t_input(i, time_directory, 'time')

        if case_type[icase] == 'no_cyl':
            case.append(case_type[icase])
        else:
            case.append(case_type[icase].split('/')[1])

        for i in range(len(T)):
            if T[i] == t_final:
                print('The kinetic energy plots are at t = ', T[i])
                fname = 'turbulent_' + str(T[i]) + '.dat'
                fdata = open(os.path.join(case_path[icase], case_dir, fname), "r")
                lines = fdata.readlines()
                fdata.close()

                tke, epsi, x = np.zeros(len(lines)), np.zeros(len(lines)), np.zeros(len(lines))
                for j in range(len(lines)):
                    line = lines[j]
                    line = line.strip()
                    line = line.split()
                    tke[j] = float(line[1])
                    # epsi[j] = float(line[2])
                    x[j] = float(line[2])

                print('The path is in', case_path[icase])
                # tke
                ax1.plot(x[(x >= 6) & (x <= 15)], tke[(x >= 6) & (x <= 15)], label=case[icase], lw=2.0,
                         dashes=ds[icase])
                ax1.legend(loc='best', fontsize='15', frameon=False)
                ax1.set_ylabel(r'$k/u_{b}^{2}$')
                ax1.set_xlabel(r'$X\,{\rm (m)}$')
                ax1.set_title('turbulent kinetic energy at  t =' + str(t_final) + '\n')
                
                ax2.plot(x[x <= 15], epsi[x <= 15], label=case[icase], lw=2.0,
                         dashes=ds[icase])
                ax2.legend(loc='best', fontsize='15', frameon=False)
                ax2.set_ylabel(r'$\epsilon $')
                ax2.set_xlabel(r'$X\,{\rm (m)}$')
                ax2.set_title('turbulent dissipation at  t =' + str(t_final) + '\n')
                
                break

if energy_plot == 'on':
    # Figure set up
    fig1, ax1 = plt.subplots(figsize=(11, 7), dpi=90)
    fig2, ax2 = plt.subplots(figsize=(11, 7), dpi=90)
    fig3, ax3 = plt.subplots(figsize=(11, 7), dpi=90)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(2)
        ax2.spines[axis].set_linewidth(2)
        ax3.spines[axis].set_linewidth(2)

    for icase in range(len(case_type)):
        case_path.append(parent_dir + '/' + case_dir + '/' + case_type[icase])
        time_directory = os.path.join(parent_dir, time_dir)
        # print(case_path, time_directory)
        case.append(case_type[icase])
        t_index = len(glob.glob1(time_directory, "*.csv"))
        T = np.zeros(t_index)
        for i in range(t_index):
            T[i] = t_input(i, time_directory, 'time')

        T = T[(T >= t_s) & (T <= t_f)]
        # print(t_s,T)
        if icase == 0:
            pf8, tf8, kf8 = np.zeros((len(case_type), len(T))), np.zeros((len(case_type), len(T))), np.zeros(
                (len(case_type), len(T)))
            pf10, tf10, kf10 = np.zeros((len(case_type), len(T))), np.zeros((len(case_type), len(T))), np.zeros(
                (len(case_type), len(T)))
            Epi8, Eki8, Epi10, Eki10 = np.zeros(len(case_type)), np.zeros(len(case_type)), \
                                       np.zeros(len(case_type)), np.zeros(len(case_type))
            Eti8, Eti10 = np.zeros(len(case_type)), np.zeros(len(case_type))
        # print(len(T))
        for i in range(len(T)):
            # print(icase)
            fname = 'eflux_' + str(T[i]) + '.dat'
            fdata = open(os.path.join(case_path[icase], fname), "r")
            lines = fdata.readlines()
            fdata.close()

            x = np.zeros(len(lines))
            for j in range(len(lines)):
                line = lines[j]
                line = line.strip()
                line = line.split()
                x[j] = float(line[4])
                if x[j] == x_front[icase]:
                    pf8[icase, i], kf8[icase, i], tf8[icase, i] = float(line[1]), float(line[2]), float(line[3])
                if x[j] == slice_loc_R[icase]:
                    pf10[icase, i], kf10[icase, i], tf10[icase, i] = float(line[1]), float(line[2]), float(line[3])
        print('The path is in', case_path[icase])
        T10 = T[T >= T_R]
        T8 = T[T >= T_i][0:len(T10)]
        # print(max(tf10[icase, :][T >= T_i]), max(tf8[icase, :][T >= T_i]))  # , T10)
        Eti8[icase] = integrate.trapz(tf8[icase, :][T >= T_i][0:len(T10)], T8)  # / (max(T8) - min(T8))
        Epi8[icase] = integrate.trapz(pf8[icase, :][T >= T_i][0:len(T10)], T8)  # / (max(T8) - min(T8))
        Eki8[icase] = integrate.trapz(kf8[icase, :][T >= T_i][0:len(T10)], T8)  # / (max(T8) - min(T8))
        Epi10[icase] = integrate.trapz(pf10[icase, :][T >= T_R], T10)  # / (max(T10) - min(T10))
        Eki10[icase] = integrate.trapz(kf10[icase, :][T >= T_R], T10)  # / (max(T10) - min(T10))
        Eti10[icase] = integrate.trapz(tf10[icase, :][T >= T_R], T10)  # / (max(T10) - min(T10))
        print(kf8[0, len(T) - 1], kf10[1:icase + 1, len(T) - 1],
              abs(3 * kf8[0, len(T) - 1] - kf10[1:icase + 1, len(T) - 1]) / (3 * kf8[0, len(T) - 1]))
        # print(abs(integrate.trapz(tf8[0, :],T)-integrate.trapz(tf10[icase, :],T))/integrate.trapz(tf8[0, :],T))
        #                                 T8) - integrate.cumtrapz(kf10[icase, :][T >= T_R][0:len(T10)],
        #  T10))))
        #if icase == 3:
        #    Epi10[icase] = Epi10[icase] * 0.9
        # print(2.5*Eki8[1:icase+1] - Eki10[1:icase+1]) #(4*Eki8 - Eki10)/4*Eti8,
    # print(len(Epi10[1:icase+1]),tt)
    if icase > 0:
        ax1.plot(tt[1:icase + 1], 1 * abs(Epi8[0] - Epi10[1:icase + 1]) / Eti8[0],
                 'b*-')  # , ms[1:icase+1])#, label=legend[1:icase+1], lw=lws,

        ax2.plot(tt[1:icase + 1], abs(3 * Eki8[icase] - Eki10[1:icase + 1]) / 3 * Eti8[0],
                 'b*-')  # , ms[1:icase+1])#, label=legend[1:icase+1], lw=lws,

        ax3.plot(tt[1:icase + 1],
                 100 * (abs(2 * Epi8[0] - Epi10[1:icase + 1]) + abs(3 * Eki8[icase] - Eki10[1:icase + 1])) / 3 * Eti8[0]
                 , 'b*-')  # 3 * kf8[0, len(T)-1] - kf10[1:icase + 1,len(T)-1])/ (3 * kf8[ 0, len(T)-1])
        # label=legend[1:icase+1], lw=lws)

        ax1.legend(loc='best', fontsize='24', frameon=False)
        ax1.set_ylabel(r'Potential Energy Reduction')
        ax1.set_xlabel(r'Number of cylinders')
        ax1.set_title('Potential Energy Reduction' + '\n')
        ymin, ymax = ax1.get_ylim()
        xmin, xmax = min(tt[1:icase + 1]), max(tt[1:icase + 1])
        x_ticks_labels = ['1 cyl', '2 cyl', '4 cyl']
        N = 4
        ax1.set_yticks(np.round(np.linspace(ymin, ymax, N), 2))
        #ax1.set_xticks(np.round(np.linspace(xmin, xmax, 4), 1))
        ax1.set_xticklabels(x_ticks_labels, rotation='horizontal', fontsize=18)

        ax2.legend(loc='best', fontsize='24', frameon=False)
        ax2.set_ylabel(r'Kinetic Energy Reduction')
        ax2.set_xlabel(r'Number of cylinders')
        ax2.set_title('Kinetic Energy Reduction' + '\n')
        ymin, ymax = ax2.get_ylim()
        # print(ymax,ymin, 2.5*abs(Eki8[0] - Eki10[1:icase+1]) / Eti8[0])
        N = 4
        ax2.set_yticks(np.round(np.linspace(ymin, ymax, N), 2))
        xmin, xmax = min(tt[1:icase + 1]), max(tt[1:icase + 1])
        ax2.set_xticks(tt[1:icase + 1])

        ax3.legend(loc='best', fontsize='24', frameon=False)
        ax3.set_ylabel(r'Total Energy Reduction(%)')
        ax3.set_xlabel(r'Number of cylinders')
        ax3.set_title('Total Energy Reduction' + '\n')
        ymin, ymax = ax3.get_ylim()
        N = 4
        ax3.set_yticks(np.round(np.linspace(ymin, ymax, N), 2))
        #ax3.set_xticks(tt[1:icase + 1])

if flux_plot == 'on':

    # Figure set up
    fig1, ax1 = plt.subplots(figsize=(11, 7), dpi=90)
    fig2, ax2 = plt.subplots(figsize=(11, 7), dpi=90)
    fig3, ax3 = plt.subplots(figsize=(11, 7), dpi=90)

    fig4, ax4 = plt.subplots(figsize=(11, 7), dpi=90)
    fig5, ax5 = plt.subplots(figsize=(11, 7), dpi=90)
    fig6, ax6 = plt.subplots(figsize=(11, 7), dpi=90)

    fig7, ax7 = plt.subplots(figsize=(11, 7), dpi=90)
    fig8, ax8 = plt.subplots(figsize=(11, 7), dpi=90)
    fig9, ax9 = plt.subplots(figsize=(11, 7), dpi=90)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(2)
        ax2.spines[axis].set_linewidth(2)
        ax3.spines[axis].set_linewidth(2)
        ax4.spines[axis].set_linewidth(2)
        ax5.spines[axis].set_linewidth(2)
        ax6.spines[axis].set_linewidth(2)
        ax7.spines[axis].set_linewidth(2)
        ax8.spines[axis].set_linewidth(2)
        ax9.spines[axis].set_linewidth(2)

    for icase in range(len(case_type)):
        case_path.append(parent_dir + '/' + case_dir + '/' + case_type[icase])
        time_directory = os.path.join(parent_dir, time_dir)
        print(case_path, time_directory)
        # if case_type[icase] == 'no_cyl':
        case.append(case_type[icase])
        #
        t_index = len(glob.glob1(time_directory, "*.csv"))
        # print(time_directory, parent_dir)
        T = np.zeros(t_index)
        for i in range(t_index):
            T[i] = t_input(i, time_directory, 'time')

        T = T[(T >= t_s) & (T <= t_f)]
        T = np.setdiff1d(T, np.asarray(ts))

        if icase == 0:
            pf8, tf8, kf8 = np.zeros((len(case_type), len(T))), np.zeros((len(case_type), len(T))), np.zeros(
                (len(case_type), len(T)))
            pf10, tf10, kf10 = np.zeros((len(case_type), len(T))), np.zeros((len(case_type), len(T))), np.zeros(
                (len(case_type), len(T)))
            pf15, tf15, kf15 = np.zeros((len(case_type), len(T))), np.zeros((len(case_type), len(T))), np.zeros(
                (len(case_type), len(T)))
            Ept8, Ekt8, Et8 = 0 * tf8, 0 * kf8, 0 * pf8
            Ept10, Ekt10, Et10 = 0 * tf10, 0 * kf10, 0 * pf10
            Ept15, Ekt15, Et15 = 0 * tf15, 0 * kf15, 0 * pf15
            Epi8, Eki8, Epi10, Eki10 = np.zeros(len(case_type)), np.zeros(len(case_type)), \
                                       np.zeros(len(case_type)), np.zeros(len(case_type))
            Eti8, Eti10 = np.zeros(len(case_type)), np.zeros(len(case_type))
        for i in range(len(T)):
            # print(T[i])
            fname = 'eflux_' + str(T[i]) + '.dat'
            fdata = open(os.path.join(case_path[icase], fname), "r")
            lines = fdata.readlines()
            fdata.close()

            x = np.zeros(len(lines))
            for j in range(len(lines)):
                line = lines[j]
                line = line.strip()
                line = line.split()
                x[j] = float(line[4])
                if x[j] == x_front:
                    pf8[icase, i], kf8[icase, i], tf8[icase, i] = float(line[1]), float(line[2]), float(line[3])
                if x[j] == slice_loc_R[icase]:
                    pf10[icase, i], kf10[icase, i], tf10[icase, i] = float(line[1]), float(line[2]), float(line[3])

                if x[j] == slice_loc_D[icase]:

                    pf15[icase, i], kf15[icase, i], tf15[icase, i] = float(line[1]), float(line[2]), float(line[3])
        print('The path is in', case_path[icase])

        if reflection == 'on':
            # print(T)
            T10 = T[T >= T_R]
            T8 = T[T >= T_i][0:len(T10)]
            # print(len(T10) - 1)
            if icase == 0:
                pf1, tf1, kf1 = np.zeros((len(case_type), len(T10) - 1)), np.zeros(
                    (len(case_type), len(T10) - 1)), np.zeros(
                    (len(case_type), len(T10) - 1))
                pf2, tf2, kf2 = np.zeros((len(case_type), len(T10) - 1)), np.zeros(
                    (len(case_type), len(T10) - 1)), np.zeros(
                    (len(case_type), len(T10) - 1))

            pf1[icase, :] = integrate.cumtrapz(pf8[icase, :][T >= T_i][0:len(T10)],
                                               T8)
            kf1[icase, :] = integrate.cumtrapz(kf8[icase, :][T >= T_i][0:len(T10)],
                                               T8)
            tf1[icase, :] = integrate.cumtrapz(tf8[icase, :][T >= T_i][0:len(T10)],
                                               T8)

            pf2[icase, :] = integrate.cumtrapz(pf10[icase, :][T >= T_R],
                                               T10)  # [tf10[icase, :] >= e_limit], T10)  # , initial=0)
            tf2[icase, :] = integrate.cumtrapz(tf10[icase, :][T >= T_R],
                                               T10)  # [tf10[icase, :] >= e_limit], T10)  # , initial=0)
            kf2[icase, :] = integrate.cumtrapz(kf10[icase, :][T >= T_R],
                                               T10)  # [tf10[icase, :] >= e_limit], T10)  # , initial=0)

            ax7.plot(T10[1:len(T10)], pf2[icase, :] / tf2[0, :], ms[icase], label=legend[icase], lw=lws,
                     linestyle=ls[icase], dashes=ds[icase])
            ax8.plot(T10[1:len(T10)], kf2[icase, :] / tf2[0, :], ms[icase], label=legend[icase], lw=lws,
                     linestyle=ls[icase], dashes=ds[icase])
            if icase > 0:
                ax9.plot(T10[1:len(T10)], tf2[icase, :] / tf2[0, :], ms[icase], label=legend[icase], lw=lws,
                         linestyle=ls[icase], dashes=ds[icase])

            ax7.legend(loc='best', fontsize='24', frameon=False)
            ax7.set_ylabel(r'$\phi^{p}_{I}$')
            ax7.set_xlabel(r'Time [s]')
            ax7.set_title('Cumulative potential Energy Flux New \n')
            ax7.set_xlim(4, 12)
            ax7.set_ylim(0.3, 0.7)
            # ax7.set_yticks(np.arange(0,0.18,0.03))

            ax8.legend(loc='best', fontsize='24', frameon=False)
            ax8.set_ylabel(r'$\phi^{k}_{I}$')
            ax8.set_xlabel(r'Time [s]')
            ax8.set_title('Cumulative kinetic Energy Flux New \n')
            ax8.set_xlim(4, 12)
            ax8.set_ylim(0.3, 0.7)
            # ax8.set_yticks(np.arange(0,0.18,0.03))

            ax9.legend(loc='best', fontsize='24', frameon=False)
            ax9.set_ylabel(r'$\phi^{t}_{I}$')
            ax9.set_xlabel(r'Time [s]')
            ax9.set_title('Cumulative total Energy Flux New \n')
            ax9.set_xlim(4, 12)
            ax9.set_ylim(0.8, 1)

            if icase > 0:
                ax1.plot(T10[1:len(T10)], (pf2[icase, :] - pf2[0, :]) / tf2[0, :], ms[icase], label=legend[icase],
                         lw=lws,
                         linestyle=ls[icase], dashes=ds[icase])
                ax2.plot(T10[1:len(T10)], (kf2[0, :] - kf2[icase, :]) / tf2[0, :], ms[icase], label=legend[icase],
                         lw=lws,
                         linestyle=ls[icase], dashes=ds[icase])

                ax3.plot(T10[1:len(T10)], abs(tf2[0, :] - tf2[icase, :]) / tf2[0, :], ms[icase], label=legend[icase],
                         lw=lws,
                         linestyle=ls[icase], dashes=ds[icase])
                ax1.legend(loc='best', fontsize='24', frameon=False)
                ax1.set_ylabel(r'$\phi^{p}_{r}$')
                ax1.set_xlabel(r'Time [s]')
                ax1.set_title('Cumulative reflected potential Energy Flux New \n')
                ax1.set_xlim(4, 12)
                ax1.set_yticks(np.arange(0, 0.26, 0.05))

                ax2.legend(loc='best', fontsize='24', frameon=False)
                ax2.set_ylabel(r'$\phi^{k}_{r}$')
                ax2.set_xlabel(r'Time [s]')
                ax2.set_title('Cumulative reflected kinetic Energy Flux New \n')
                ax2.set_xlim(4, 12)
                ax2.set_yticks(np.arange(0, 0.26, 0.05))

                ax3.legend(loc='best', fontsize='24', frameon=False)
                ax3.set_ylabel(r'$\phi^{t}_{r}$')
                ax3.set_xlabel(r'Time [s]')
                ax3.set_title('Cumulative reflected total Energy Flux New \n')
                ax3.set_xlim(4, 12)
                ax3.set_yticks(np.arange(0, 0.26, 0.05))

        elif dissipation == 'on':
            T15 = T[T >= T_R]  # T[tf10[icase, :] >= e_limit]
            T10 = T[T >= T_i][0:len(T15)]  # T[tf8[icase, :] >= e_limit]

            if icase == 0:
                pf1, tf1, kf1 = np.zeros((len(case_type), len(T15) - 1)), np.zeros(
                    (len(case_type), len(T15) - 1)), np.zeros(
                    (len(case_type), len(T15) - 1))
                pf2, tf2, kf2 = np.zeros((len(case_type), len(T15) - 1)), np.zeros(
                    (len(case_type), len(T15) - 1)), np.zeros(
                    (len(case_type), len(T15) - 1))
            pf1[icase, :] = integrate.cumtrapz(pf10[icase, :][T >= T_i][0:len(T15)],
                                               T10)  # [tf8[icase, :] >= e_limit][0:len(T10)], T8[0:len(T10)])
            # initial=0)
            kf1[icase, :] = integrate.cumtrapz(kf10[icase, :][T >= T_i][0:len(T15)],
                                               T10)  # [tf8[icase, :] >= e_limit][0:len(T10)], T8[0:len(T10)])
            # initial=0)
            tf1[icase, :] = integrate.cumtrapz(tf10[icase, :][T >= T_i][0:len(T15)],
                                               T10)  # [tf8[icase, :] >= e_limit][0:len(T10)], T8[0:len(T10)])
            # initial=0)

            pf2[icase, :] = integrate.cumtrapz(pf15[icase, :][T >= T_R],
                                               T15)  # [tf10[icase, :] >= e_limit], T10)  # , initial=0)
            tf2[icase, :] = integrate.cumtrapz(tf15[icase, :][T >= T_R],
                                               T15)  # [tf10[icase, :] >= e_limit], T10)  # , initial=0)
            kf2[icase, :] = integrate.cumtrapz(kf15[icase, :][T >= T_R], T15)

            ax7.plot(T15[1:len(T15)], pf2[icase, :] / tf2[0, :], ms[icase], label=legend[icase], lw=lws,
                     linestyle=ls[icase], dashes=ds[icase])
            ax8.plot(T15[1:len(T15)], kf2[icase, :] / tf2[0, :], ms[icase], label=legend[icase], lw=lws,
                     linestyle=ls[icase], dashes=ds[icase])
            if icase > 0:
                ax9.plot(T15[1:len(T15)], (pf2[icase, :] + kf2[icase, :]) / tf2[0, :], ms[icase], label=legend[icase], lw=lws,
                         linestyle=ls[icase], dashes=ds[icase])

            ax7.legend(loc='best', fontsize='24', frameon=False)
            ax7.set_ylabel(r'$\phi^{p}_{tr}$')
            ax7.set_xlabel(r'Time [s]')
            ax7.set_title('Cumulative transmitted potential Energy Flux New \n')
            ax7.set_xlim(4, 12)
            # ax7.set_ylim(0, 0.15)
            # ax7.set_yticks(np.arange(0,0.18,0.03))

            ax8.legend(loc='best', fontsize='24', frameon=False)
            ax8.set_ylabel(r'$\phi^{k}_{tr}$')
            ax8.set_xlabel(r'Time [s]')
            ax8.set_title('Cumulative transmitted kinetic Energy Flux New \n')
            ax8.set_xlim(4, 12)
            # ax8.set_ylim(0, 0.15)
            # ax8.set_yticks(np.arange(0,0.18,0.03))

            ax9.legend(loc='best', fontsize='24', frameon=False)
            ax9.set_ylabel(r'$\phi^{t}_{tr}$')
            ax9.set_xlabel(r'Time [s]')
            ax9.set_title('Cumulative transmitted total Energy Flux New \n')
            ax9.set_xlim(4, 12)
            # ax9.set_ylim(0, 0.15)
            # ax9.set_yticks(np.arange(0,0.18,0.03))
            if icase >=3:
                print(kf2[icase, :] , kf1[icase, :], abs(kf2[icase, :] - kf1[icase, :])/ tf1[0,:])
            if icase > 0:
                ax1.plot(T15[1:len(T15)], abs(pf2[icase, :] - pf1[icase, :]) / tf1[0, :], ms[icase],
                         label=legend[icase], lw=lws,
                         linestyle=ls[icase], dashes=ds[icase])
                ax2.plot(T15[1:len(T15)], abs(kf2[icase, :] - kf1[icase, :]) / tf1[0, :], ms[icase],
                         label=legend[icase], lw=lws,
                         linestyle=ls[icase], dashes=ds[icase])

                ax3.plot(T15[1:len(T15)],
                         (abs(pf2[icase, :] - pf1[icase, :]) + abs(kf2[icase, :] - kf1[icase, :])) / tf1[0, :],
                         ms[icase], label=legend[icase],
                         lw=lws,
                         linestyle=ls[icase], dashes=ds[icase])
                ax1.legend(loc='best', fontsize='24', frameon=False)
                ax1.set_ylabel(r'$\phi^{p}_{d}$')
                ax1.set_xlabel(r'Time [s]')
                ax1.set_title('Cumulative dissipated potential Energy Flux New \n')
                ax1.set_xlim(4, 12)
                ax1.set_ylim(0, 0.15)
                ax1.set_yticks(np.arange(0, 0.18, 0.03))

                ax2.legend(loc='best', fontsize='24', frameon=False)
                ax2.set_ylabel(r'$\phi^{k}_{d}$')
                ax2.set_xlabel(r'Time [s]')
                ax2.set_title('Cumulative dissipated kinetic Energy Flux New \n')
                ax2.set_xlim(4, 12)
                ax2.set_ylim(0, 0.2)
                ax2.set_yticks(np.arange(0, 0.21, 0.03))

                ax3.legend(loc='best', fontsize='24', frameon=False)
                ax3.set_ylabel(r'$\phi^{t}_{d}$')
                ax3.set_xlabel(r'Time [s]')
                ax3.set_title('Cumulative dissipated total Energy Flux New \n')
                ax3.set_xlim(4, 12)
                ax3.set_ylim(0, 0.30)
                ax3.set_yticks(np.arange(0, 0.31, 0.06))

        if icase > 0:
            if reflection == 'on':
                # print(T10, T8)
                ax4.plot(T10[1:len(T10)][T10[1:len(T10)] >= 4.5],
                         abs(1.4 * kf1[icase, :][T10[1:len(T10)] >= 4.5] - kf2[icase, :][T10[1:len(T10)] >= 4.5]) /
                         tf1[icase, :][T10[1:len(T10)] >= 4.5], ms[icase],
                         # - (Ekt8[0, :] - Ekt10[0, :])) / Eki8[icase]
                         label=legend[icase], lw=lws,
                         linestyle=ls[icase], dashes=ds[icase])
                ymin, ymax = ax4.get_ylim()
                N = 4
                ax4.set_yticks(np.round(np.linspace(ymin, ymax, N), 2))
                ax5.plot(T10[1:len(T10)][T10[1:len(T10)] >= 4.5],
                         abs(tf1[icase, :][T10[1:len(T10)] >= 4.5] - tf2[icase, :]
                         [T10[1:len(T10)] >= 4.5]) / tf1[icase, :][T10[1:len(T10)] >= 4.5]
                         # - (Et8[0, :] - Et10[0, :])/ Eti8[icase]
                         , ms[icase], label=legend[icase], lw=lws,
                         linestyle=ls[icase], dashes=ds[icase])

                ax6.plot(T10[1:len(T10)][T10[1:len(T10)] >= 4.5],
                         (1.2 * pf1[icase, :][T10[1:len(T10)] >= 4.5] - pf1[0, :][T10[1:len(T10)] >= 4.5]) /
                         tf1[icase, :][T10[1:len(T10)] >= 4.5],  # / Epi8[icase]
                         ms[icase], label=legend[icase], lw=lws,
                         linestyle=ls[icase], dashes=ds[icase])
                ymin, ymax = ax5.get_ylim()
                N = 4
                ax5.set_yticks(np.round(np.linspace(ymin, ymax, N), 2))
                ax4.legend(loc='best', fontsize='24', frameon=False)
                ax4.set_ylabel(r'$\phi_{r}^{k}$')
                ax4.set_xlabel(r'Time [s]')
                ax4.set_title('Reflected kinetic Energy Flux \n')
                ax4.set_xlim(4.0, 12.2)

                ax5.legend(loc='best', fontsize='24', frameon=False)
                ax5.set_ylabel(r'$\phi_{r}^{t}$')
                ax5.set_xlabel(r'Time [s]')
                ax5.set_title('Reflected total energy flux \n')
                ax5.set_xlim(4.0, 12.2)

                ax6.legend(loc='best', fontsize='24', frameon=False)
                ax6.set_ylabel(r'$\phi_{r}^{p}$')
                ax6.set_xlabel(r'Time [s]')
                ax6.set_title('Reflected potential Energy Flux \n')
                ax6.set_xlim(4.0, 12.2)

            elif dissipation == 'on':

                ax4.plot(T15[1:len(T15)][T15[1:len(T15)] >= 4.5],
                         1.5 * abs(kf1[icase, :][T15[1:len(T15)] >= 4.5] - kf2[icase, :][T15[1:len(T15)] >= 4.5]) /
                         tf1[icase, :][T15[1:len(T15)] >= 4.5], ms[icase],
                         # - (Ekt8[0, :] - EkT15[0, :])) / Eki8[icase]
                         label=legend[icase], lw=lws,
                         linestyle=ls[icase], dashes=ds[icase])

                ax5.plot(T15[1:len(T15)][T15[1:len(T15)] >= 4.5],
                         abs(tf1[icase, :][T15[1:len(T15)] >= 4.5] - tf2[icase, :]
                         [T15[1:len(T15)] >= 4.5]) / tf1[icase, :][T15[1:len(T15)] >= 4.5]
                         # - (Et8[0, :] - ET15[0, :])/ Eti8[icase]
                         , ms[icase], label=legend[icase], lw=lws,
                         linestyle=ls[icase], dashes=ds[icase])

                ax6.plot(T15[1:len(T15)][T15[1:len(T15)] >= 4.5],
                         abs(1.2 * pf1[icase, :][T15[1:len(T15)] >= 4.5] - pf2[icase, :][T15[1:len(T15)] >= 4.5]) /
                         tf1[icase, :][T15[1:len(T15)] >= 4.5],  # / Epi8[icase]
                         ms[icase], label=legend[icase], lw=lws,
                         linestyle=ls[icase], dashes=ds[icase])

                ax4.legend(loc='best', fontsize='24', frameon=False)
                ax4.set_ylabel(r'$\phi_{d}^{k}$')
                ax4.set_xlabel(r'Time [s]')
                ax4.set_title('Dissipated kinetic Energy Flux Coefficient \n')
                ax4.set_xlim(4.0, 12.2)

                ax5.legend(loc='best', fontsize='24', frameon=False)
                ax5.set_ylabel(r'$\phi_{d}^{t}$')
                ax5.set_xlabel(r'Time [s]')
                ax5.set_title('Dissipated Total Energy Flux \n')
                ax5.set_xlim(4.0, 12.2)

                ax6.legend(loc='best', fontsize='24', frameon=False)
                ax6.set_ylabel(r'$\phi_{d}^{p}$')
                ax6.set_xlabel(r'Time [s]')
                ax6.set_title('Dissipated potential Energy Flux \n')
                ax6.set_xlim(4.0, 12.2)
# print(case)


data_directory = 'interpolate_linear'
slices = ['8', '11.1', '11', '10.4', '11.5', '15', '17']
ls = ['solid', 'dotted', 'dashed', 'dashdot', 'dashdot', 'dotted', 'dashed']
ms = ['x', '^', '*', 's', 'd', 'o', '+']
ds = [(), (1, 1), (5, 5), (3, 5, 1, 5, 1, 5), (8, 4, 2, 4, 2, 4), (1, 10), (5, 10)]
t_final = 10
slice = slices[2]
dissislice = ['13', '12', '15']  # , '10.8', '10.8', '10.8']
refslice = ['10']
case = ['yang_case_0.7M']#, 'flex_case_double_cyl_d_0.4_sbs-y3e8', 'flex_case_double_cyl_d_0.4_sbs-y3e9',
    #'flex_case_single_cyl_d_0.4_y3e8',
    #'flex_case_single_cyl_d_0.4_y3e9']  # 'rigid_case_single_cyl_d_1.6',
# 'rigid_case_double_cyl_d_0.4_sbs']  # , 'flex_case_double_cyl_d_0.4_sbs-y3e8']  # ,
# 'flex_case_double_cyl_d_0.4_sbs-y3e8']  # ,]
cc = ['rd0.14'] #'rd0.4m4', 'rd0.4s', 'rd0.8s', 'rd1.6s', 'rd0.4m2']  # , 'fd0.4s']
E_Dnormt = np.zeros(len(case) - 1)
section = ['Transect I', 'Transect I', 'Transect III']
alt = 'interpolate'
cumulative = 'on'
dissipation = 'off'
reflection = 'on'
energy_flux_plot = 'on'
total_energy = 'off'
nt = 45

E_R = np.zeros((len(case), nt))

# fig, ax = plt.subplots(figsize=(11, 7), dpi=90)
interpolate = 'no'
for icase in range(len(case)):
    if icase == 0:
        if energy_flux_plot == 'on':
            fig1, ax1 = plt.subplots(figsize=(11, 7), dpi=90)
            fig2, ax2 = plt.subplots(figsize=(11, 7), dpi=90)
            fig3, ax3 = plt.subplots(figsize=(11, 7), dpi=90)

            for axis in ['top', 'bottom', 'left', 'right']:
                ax1.spines[axis].set_linewidth(2)
                ax2.spines[axis].set_linewidth(2)
                ax3.spines[axis].set_linewidth(2)

        if cumulative == 'on' and total_energy == 'off':
            fig4, ax4 = plt.subplots(figsize=(11, 7), dpi=90)
            for axis in ['top', 'bottom', 'left', 'right']:
                ax4.spines[axis].set_linewidth(2)
            fig6, ax6 = plt.subplots(figsize=(11, 7), dpi=90)
            for axis in ['top', 'bottom', 'left', 'right']:
                ax6.spines[axis].set_linewidth(2)
            fig8, ax8 = plt.subplots(figsize=(11, 7), dpi=90)
            for axis in ['top', 'bottom', 'left', 'right']:
                ax8.spines[axis].set_linewidth(2)

        if reflection == 'on':
            fig5, ax5 = plt.subplots(figsize=(11, 7), dpi=90)
            for axis in ['top', 'bottom', 'left', 'right']:
                ax5.spines[axis].set_linewidth(2)
        if dissipation == 'on':
            fig7, ax7 = plt.subplots(figsize=(11, 7), dpi=90)
            for axis in ['top', 'bottom', 'left', 'right']:
                ax7.spines[axis].set_linewidth(2)
            fig10, ax10 = plt.subplots(figsize=(11, 7), dpi=90)
            for axis in ['top', 'bottom', 'left', 'right']:
                ax10.spines[axis].set_linewidth(2)
        if total_energy == 'on':
            if energy_flux_plot == 'off':
                fig1, ax1 = plt.subplots(figsize=(11, 7), dpi=90)
                fig2, ax2 = plt.subplots(figsize=(11, 7), dpi=90)
                fig3, ax3 = plt.subplots(figsize=(11, 7), dpi=90)
                fig9, ax9 = plt.subplots(figsize=(11, 7), dpi=90)

                for axis in ['top', 'bottom', 'left', 'right']:
                    ax1.spines[axis].set_linewidth(2)
                    ax2.spines[axis].set_linewidth(2)
                    ax3.spines[axis].set_linewidth(2)
                    ax9.spines[axis].set_linewidth(2)
            if cumulative == 'on':
                fig4, ax4 = plt.subplots(figsize=(11, 7), dpi=90)
                for axis in ['top', 'bottom', 'left', 'right']:
                    ax4.spines[axis].set_linewidth(2)

    filename = 'energy_flux' + '_' + case[icase] + '_' + slice + '.txt'
    energy_data = np.loadtxt(os.path.join(os.path.dirname(os.getcwd()), data_directory, filename), delimiter=' ',
                             skiprows=1, usecols=range(4))
    energy_data = energy_data[~np.isnan(energy_data).any(axis=1)]
    pot_flux = energy_data[:, 0]  # [energy_data[:, 3] <= t_final]
    kin_flux = energy_data[:, 1]
    tot_energy_flux = energy_data[:, 2]
    time = energy_data[:, 3]
    # print(np.size(time))
    Et = integrate.simps(tot_energy_flux, time)
    #print(case[icase])

        # print(np.size(T), np.size(Ekt8[icase, :]))
        # Interpolation the data
    
Ekt8[icase, :], T = interpolation(Ekt8[icase, :], T, len(T))
Ekt10[icase, :] = interpolation(Ekt10[icase, :], T, len(T))[0]
Ekt15[icase, :] = interpolation(Ekt15[icase, :], T, len(T))[0]
# Eki8[icase] = interpolation(Eki8[icase], T, len(T))[0]
Et8[icase, :] = interpolation(Et8[icase, :], T, len(T))[0]
Et10[icase, :] = interpolation(Et10[icase, :], T, len(T))[0]
Et15[icase, :] = interpolation(Et15[icase, :], T, len(T))[0]
# Eti8[icase] = interpolation(Eti8[icase], T, len(T))[0]
Ept8[icase, :] = interpolation(Ept8[icase, :], T, len(T))[0]
Ept10[icase, :] = interpolation(Ept10[icase, :], T, len(T))[0]
Ept15[icase, :] = interpolation(Ept15[icase, :], T, len(T))[0]
# Epi8[icase] = interpolation(Epi8[icase], T, len(T))[0]

if reflection == 'on':
Ept = integrate.cumtrapz(pot_flux, time, initial=0)
Ekt = integrate.cumtrapz(kin_flux, time, initial=0)
Et = integrate.cumtrapz(tot_energy_flux, time, initial=0)
rd = 'energy_flux' + '_' + case[icase] + '_' + refslice[0] + '.txt'
energy_ref = np.loadtxt(os.path.join(os.path.dirname(os.getcwd()), data_directory, rd), delimiter=' ',
                        skiprows=1, usecols=range(4))
energy_ref = energy_ref[~np.isnan(energy_ref).any(axis=1)]
pot_ref = energy_ref[:, 0]
kin_ref = energy_ref[:, 1]
tot_energy_ref = energy_ref[:, 2]
t_ref = energy_ref[:, 3]
Epreft = integrate.cumtrapz(pot_ref, t_ref, initial=0)
Ekreft = integrate.cumtrapz(kin_ref, t_ref, initial=0)
Etreft = integrate.cumtrapz(tot_energy_ref, t_ref, initial=0)

ax5.plot(time[time <= t_final], (Epreft[t_ref <= t_final] - Ept[time <= t_final])*1000, ms[icase], color='grey',
         label=case[icase], lw=1.5,
         linestyle=ls[icase], dashes=ds[icase])
ax6.plot(time[time <= t_final], (Ekreft[t_ref <= t_final] - Ekt[time <= t_final])*1000, ms[icase], color='grey',
         label=case[icase], lw=1.5,
         linestyle=ls[icase], dashes=ds[icase])
ax8.plot(time[time <= t_final], Ekt[time <= t_final] * 1000, ms[icase], color='grey',
         label='x_'+slice, lw=1.5,
         linestyle=ls[icase], dashes=ds[icase])
ax8.plot(time[time <= t_final], Ekreft[time <= t_final] * 1000, ms[icase], color='k',
         label='x_'+refslice[0], lw=1.5,
         linestyle=ls[icase], dashes=ds[icase])
ax5.legend(loc='best', fontsize='24', frameon=False)
ax5.set_ylabel(r'Reflected Flux $\times 10^{-3} [m^{5}s^{-3}]$')
ax5.set_xlabel(r'Time [s]')
# ax5.set_ylim(-10, 500)
ax5.set_title(r'Reflected Potential Energy Flux')

ax6.legend(loc='best', fontsize='24', frameon=False)
ax6.set_ylabel(r'Reflected Flux $\times 10^{-3} [m^{5}s^{-3}]$')
ax6.set_xlabel(r'Time [s]')
ax6.set_title(r'Reflected Kinetic Energy Flux')

ax8.legend(loc='best', fontsize='24', frameon=False)
ax8.set_ylabel(r'Energy Flux $\times 10^{-3} [m^{5}s^{-3}]$')
ax8.set_xlabel(r'Time [s]')
# ax5.set_ylim(-10, 500)
ax8.set_title(r'Potential Cumulative Energy Flux')

if dissipation == 'on':
Ept = integrate.cumtrapz(pot_flux, time, initial=0)
Ekt = integrate.cumtrapz(kin_flux, time, initial=0)
Et = integrate.cumtrapz(tot_energy_flux, time, initial=0)
ld = 'energy_flux' + '_' + case[icase] + '_' + dissislice[2] + '.txt'
# print(case[icase])
energy_dis = np.loadtxt(os.path.join(os.path.dirname(os.getcwd()), data_directory, ld), delimiter=' ',
                        skiprows=1, usecols=range(4))
energy_dis = energy_dis[~np.isnan(energy_dis).any(axis=1)]
pot_dis = energy_dis[:, 0]
kin_dis = energy_dis[:, 1]
TE_dis = energy_dis[:, 2]
t_dis = energy_dis[:, 3]
TE_dis_t = integrate.cumtrapz(pot_dis, t_dis, initial=0)
kin_dis_t = integrate.cumtrapz(kin_dis, t_dis, initial=0)
pot_dis_t = integrate.cumtrapz(pot_dis, t_dis, initial=0)
ax8.plot(t_dis[t_dis <= t_final], TE_dis_t[t_dis <= t_final], 'k', lw=1.5,
         label=case[icase], linestyle=ls[icase], dashes=ds[icase])
ax8.legend(loc='best', fontsize='24', frameon=False)
ax8.set_ylabel(r'Energy Flux $[m^{5}s^{-3}]$')
ax8.set_xlabel(r'Time [s]')
# ax7.set_yscale('log')
# ax7.set_ylim(-500, 1500)
ax8.set_title(r'Cumulative Energy Flux at x' + dissislice[2])
#print(TE_dis_t[len(TE_dis_t) - 1], Et[len(Ept) - 1], case[icase])
ax7.plot(time[time <= t_final], -kin_dis_t[t_dis <= t_final] + Ekt[time <= t_final], ms[icase], color='grey',
         label=case[icase], lw=1.5,
         linestyle=ls[icase], dashes=ds[icase])
ax10.plot(time[time <= t_final], -pot_dis_t[t_dis <= t_final] + Ept[time <= t_final], ms[icase], color='grey',
          label=case[icase], lw=1.5,
          linestyle=ls[icase], dashes=ds[icase])
# ax7.plot(time[time <= t_final], -TE_dis_t[t_dis <= t_final] + Ept[time <= t_final], ms[icase], color='gray', label=case[
#            icase], markersize=5, linewidth=1.5, markerfacecolor='white',
#                 markeredgecolor='gray', markeredgewidth=2)
ax7.legend(loc='best', fontsize='24', frameon=False)
ax7.set_ylabel(r'Energy Flux $[m^{5}s^{-3}]$')
ax7.set_xlabel(r'Time [s]')
# ax7.set_yscale('log')
# ax7.set_ylim(-500, 1500)
ax7.set_title(r'Dissipated Kinetic Energy Flux')
ax10.legend(loc='best', fontsize='24', frameon=False)
ax10.set_ylabel(r'Energy Flux $[m^{5}s^{-3}]$')
ax10.set_xlabel(r'Time [s]')
# ax7.set_yscale('log')
# ax7.set_ylim(-500, 1500)
ax10.set_title(r'Dissipated Potential Energy Flux')

if cumulative == 'on' and total_energy == 'off':
tt = 0
e_int = integrate.cumtrapz(tot_energy_flux[time >= tt], time[time >= tt], initial=0)
ax4.plot(time[time >= tt], e_int*1000, 'k', lw=1.5, label=case[icase], linestyle=ls[icase], dashes=ds[icase])
ax4.legend(loc='best', fontsize='24', frameon=False)
#ax4.set_yscale('log')
ax4.set_ylabel(r'Cumulative Flux $\times 10^{-3} [m^{5}s^{-3}]$')  # unit $[m^{5}s^{-2}]$
ax4.set_xlabel(r'Time [s]')
#ax4.set_ylim(10, 1000)
ax4.set_title(r'Cumulative Total Energy Flux at x ' + slice)

if total_energy == 'on':
# print(icase)
tefile = 'total_energy' + '_' + case[icase] + '_' + data + '.txt'
te_data = np.loadtxt(tefile, delimiter=' ', skiprows=1, usecols=range(5))
te_data = te_data[~np.isnan(te_data).any(axis=1)]
pot = te_data[:, 0]
ikin = te_data[:, 1]
tot_energy = te_data[:, 2]
tkin = te_data[:, 3]
tt = te_data[:, 4]
# Total energy

ax1.plot(tt, tot_energy, lw=2, label=case[icase])
ax1.legend(loc='best', fontsize='24', frameon=False)
ax1.set_ylabel(r'Total energy $[m^{5}s^{-3}]$')
ax1.set_xlabel(r'Time [s]')
ax1.set_ylim(-10, 1000)
ax1.set_title(r'Total energy evolution')

# Potential Energy

ax2.plot(tt, pot, lw=2, label=case[icase])
ax2.legend(loc='best', fontsize='24', frameon=False)
ax2.set_ylabel(r'Potential Energy $[m^{5}s^{-3}]$')
ax2.set_xlabel(r'Time [s]')
ax2.set_ylim(-5, 1000)
ax2.set_title(r'PE evolution')

# Instantaneous Kinetic Energy

# prime Kinetic Energy

ax9.plot(tt, tkin, lw=2, label=case[icase])
ax9.legend(loc='best', fontsize='24', frameon=False)
ax9.set_ylabel(r'Kinetic Energy $[m^{5}s^{-3}]$')
ax9.set_xlabel(r'Time [s]')
ax9.set_ylim(-1, 8)
ax9.set_title(r'TKE evolution')

if cumulative == 'on':
    te_int = integrate.cumtrapz(tkin, tt, initial=0)
    ax4.plot(tt, te_int, lw=2, label=case[icase])
    ax4.legend(loc='best', fontsize='24', frameon=False)
    ax4.set_ylabel(r'Cumulative TKE $[m^{5}s^{-2}]$')
    ax4.set_xlabel(r'Time [s]')
    #ax4.set_ylim(-1, 2)
    ax4.set_title(r'Cumulative kinetic Energy')

if energy_flux_plot == 'on':
# Total Energy flux

ax1.plot(time, tot_energy_flux*1000, 'k', lw=1.5, label=case[icase], linestyle=ls[icase], dashes=ds[icase])
ax1.legend(loc='best', fontsize='24', frameon=False)
# ax1.set_yscale('log')
ax1.set_ylabel(r'Total Flux $\times 10^{-3} [m^{5}s^{-3}]$')
ax1.set_xlabel(r'Time [s]')
#ax1.set_ylim(0, 2000)
ax1.set_title(r'Total Energy Flux at x ' + slice)

# Potential flux

ax2.plot(time, pot_flux*1000, 'k', lw=1.5, label=case[icase], linestyle=ls[icase], dashes=ds[icase])
ax2.legend(loc='best', fontsize='24', frameon=False)
#ax2.set_yscale('log')
ax2.set_ylabel(r'Potential Flux $\times 10^{-3} [m^{5}s^{-3}]$')
ax2.set_xlabel(r'Time [s]')
#ax2.set_ylim(5, 150)
ax2.set_title(r'PE flux at x ' + slice)  # + case + ' cylinder')

# Kinetic flux

ax3.plot(time, kin_flux*1000, 'k', lw=1.5, label=case[icase], linestyle=ls[icase], dashes=ds[icase])
ax3.legend(loc='best', fontsize='24', frameon=False)
# ax3.set_yscale('log')
ax3.set_ylabel(r'Kinetic Flux $\times 10^{-3} [m^{5}s^{-3}]$')
ax3.set_xlabel(r'Time [s]')
#ax3.set_ylim(0, 2000)
ax3.set_title(r'KE flux at x ' + slice)

plt.tight_layout()
plt.show()
'''
