#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import tsunami
# parameters


def color_map(fname):
    data = open(fname, "r")
    lines = data.readlines()
    data.close()
    colors = []
    for i in range(len(lines)):
        line = lines[i]
        line = line.strip()
        line = line.split()
        colors.append([eval(val) for val in line[0:]])
    n_bins = 256
    cmap_name = 'coolwarm_ex'
    # Create the colormap
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    return cm, n_bins


def norm_vector(q3d, depth_2d, xpt, ypt, zpt):
    q_norm = np.zeros((xpt, ypt, zpt))
    for i in range(xpt):
        for j in range(ypt):
            for k in range(zpt):
                if depth_2d[i, j, k] == 0:
                    q_norm[i, j, k] = 0
                else:
                    q_norm[i, j, k] = q3d[i, j, k] / (9.81 * depth_2d[i, j, k])
    return q_norm


def postprocess(i):
    global lev_sl_2d


    from input import flux_input
    energy_data = 0
    data_directory, filename, xpt, ypt, zpt, energy_flux, vel_profile, turbulent_average, \
    levsmooth, wall_shear_stress, dam_front, turbulent_energy, lev_set, water_depth, slice_interpolation, \
    probe_data, force, shear_data, data_3d, vel_field_avg, side_wall, int_avg = flux_input()
    average = 'off'
    # field_average = 'off'
    g = 9.81
    energy_flux_data = 0
    total_energy_data = 0

    energy_theory = []

    # if energy_flux == 'yes':
    if data_3d == 'on':
        print(data_directory, filename)
        # cm, n_bins = color_map('cool_to_worm_extended.txt')
        # parameter = np.loadtxt(os.path.join(data_directory, filename), delimiter=',', dtype='str')[0, :]
        # plane = np.loadtxt(os.path.join(data_directory, filename), delimiter=',', skiprows=1)
        print('read start')
        # rho = plane[:, parameter.tolist().index('"DENSI"')]
        lev = pd.read_csv(os.path.join(data_directory, filename), usecols=['LEVEL']).values[:, 0]

        mask = pd.read_csv(os.path.join(data_directory, filename), usecols=['vtkValidPointMask']).values[:, 0]
        lev[mask == 0] = -1
        # pressure = plane[:, parameter.tolist().index('"PRESS"')]
        nu_sgs = pd.read_csv(os.path.join(data_directory, filename), usecols=['VISCO']).values[:, 0]
        x = pd.read_csv(os.path.join(data_directory, filename), usecols=['Points:0']).values[:, 0]
        y = pd.read_csv(os.path.join(data_directory, filename), usecols=['Points:1']).values[:, 0]
        z = pd.read_csv(os.path.join(data_directory, filename), usecols=['Points:2']).values[:, 0]
        u = pd.read_csv(os.path.join(data_directory, filename), usecols=['VELOC:0']).values[:, 0]
        v = pd.read_csv(os.path.join(data_directory, filename), usecols=['VELOC:1']).values[:, 0]
        w = pd.read_csv(os.path.join(data_directory, filename), usecols=['VELOC:2']).values[:, 0]
        try:
            tau1 = pd.read_csv(os.path.join(data_directory, filename), usecols=['TANGE:0']).values[:, 0]
            tau2 = pd.read_csv(os.path.join(data_directory, filename), usecols=['TANGE:1']).values[:, 0]
            tau3 = pd.read_csv(os.path.join(data_directory, filename), usecols=['TANGE:2']).values[:, 0]
            # tau = np.sqrt((tau1 * 1e3) ** 2 + (tau2 * 1e3) ** 2 + (tau3 * 1e3) ** 2)
        except:
            print('shear stress data is not written, check whether shear stress module is turned on in the input file')
        # uu,vv,ww = 1*u, 1*v, 1*w
        V = np.sqrt(u ** 2 + v ** 2 + w ** 2)
        print('read end')
        # v[lev < lev_set] = 0

        if average == 'on':
            if i == 0:
                u_av = np.zeros(len(u))
                v_av = np.zeros(len(v))
                w_av = np.zeros(len(w))
            else:
                u_av = plane[:, parameter.tolist().index('"AVVEL:0"')]
                v_av = plane[:, parameter.tolist().index('"AVVEL:1"')]
                w_av = plane[:, parameter.tolist().index('"AVVEL:2"')]

            # Calculate u',v',w'
            k_TE = ((u - u_av) ** 2 + (v - v_av) ** 2 + (w - w_av) ** 2) / 2
            V_av = np.sqrt(u_av ** 2 + v_av ** 2 + w_av ** 2)

        '''
        q = np.zeros((12, len(lev)))
        q[0, :], q[1, :], q[2, :], q[3, :], q[4, :], q[5, :], q[6, :], q[7, :], q[8, :], q[9, :] = lev, mask, nu_sgs, \
                                                                                                   x, y, z, u, v, w, V

        q_3d = tsunami.str_array(xpt, ypt, zpt, q, 10)
        lev_3d, mask_3d, nu_sgs_3d, x_3d, y_3d, z_3d, u_3d, v_3d, w_3d, v_mag_3d = q_3d[0, :, :, :], q_3d[1, :, :, :], \
                                                                                   q_3d[2, :, :, :], q_3d[3, :, :,
                                                                                                     :], q_3d[4, :, :,
                                                                                                         :], q_3d[5, :,
                                                                                                             :,
                                                                                                             :], q_3d[6,
                                                                                                                 :, :,
                                                                                                                 :], q_3d[
                                                                                                                     7,
                                                                                                                     :,
                                                                                                                     :,
                                                                                                                     :], \
                                                                                   q_3d[8, :, :, :], q_3d[9, :, :, :]
        c_3d = 1 * mask_3d
        c_3d[lev_3d < 0], c_3d[lev_3d >= 0] = 0, 1
        if side_wall[0] == 'on':
            y_p = side_wall[1]
            x_3d, y_3d = np.ascontiguousarray(x_3d[:, y_p:ypt - y_p, :]), np.ascontiguousarray(
                y_3d[:, y_p:ypt - y_p, :])
            z_3d, u_3d = np.ascontiguousarray(z_3d[:, y_p:ypt - y_p, :]), np.ascontiguousarray(
                u_3d[:, y_p:ypt - y_p, :])
            v_3d, w_3d = np.ascontiguousarray(v_3d[:, y_p:ypt - y_p, :]), np.ascontiguousarray(
                w_3d[:, y_p:ypt - y_p, :])
            nu_sgs_3d, lev_3d = np.ascontiguousarray(nu_sgs_3d[:, y_p:ypt - y_p, :]), \
                                np.ascontiguousarray(lev_3d[:, y_p:ypt - y_p, :])
            v_mag_3d = np.ascontiguousarray(v_mag_3d[:, y_p:ypt - y_p, :])
            c_3d = np.ascontiguousarray(c_3d[:, y_p:ypt - y_p, :])
            mask_3d = np.ascontiguousarray(mask_3d[:, y_p:ypt - y_p, :])
            ypt = 1 * ypt - 2 * y_p

        #
    # print(ypt, len(y_3d[0, :, 0]), len(x_3d[0, :, 0]))
    '''
    if vel_field_avg == 'on':
        v3d = np.zeros((3, xpt, ypt, zpt))
        v3d[0, :, :, :], v3d[1, :, :, :], v3d[2, :, :, :] = 1 * u_3d, 1 * v_3d, 1 * w_3d
        if levsmooth == 'on':
            v_3d, lev3d, z3d, x_3d, y_3d, c3d, zpt = tsunami.lev_smooth(lev_3d, z_3d, v3d, xpt, ypt, zpt, x_3d, y_3d, 3)
            lev_3d, v3d, z_3d, c_3d = tsunami.smooth_grid(z3d, lev3d, v_3d, xpt, ypt, zpt, 3)
        v_z, con_z, depth_2d = tsunami.depth_average(lev_3d, z_3d, v3d, xpt, ypt, zpt, c_3d, 3)
        v_y, con_y = tsunami.span_average(y_3d, 1 * v3d, lev_3d, xpt, ypt, zpt, c_3d, 3)

        energy_flux_data = v_y, x_3d, z_3d, v3d, y_3d, lev_3d

    if energy_flux[0] == 'on':
        # u_3d[lev_3d < lev_set], v_3d[lev_3d < lev_set], w_3d[lev_3d < lev_set], c = 0, 0, 0, 0
        v3d = np.zeros((2, xpt, ypt, zpt))
        total_energy = 'on'
        if total_energy == 'off':
            eflux_data = np.zeros((xpt, 4))
            eflux_data[:, 0], eflux_data[:, 1], eflux_data[:, 2], eflux_data[:, 3] =eflux.py \
                tsunami.eflux(xpt, ypt, zpt, x, y, z, u, V, lev, mask)
        if total_energy == 'on':
            ef_2d_z, con_z, depth_2d = tsunami.depth_average(lev_3d, z_3d, v_mag_3d, xpt, ypt, zpt, c_3d, 1)
            v_mag_3d[lev_3d < lev_set] = np.nan
            vz = tsunami.field_avg(v_mag_3d, 1, xpt, ypt, zpt, 'z')[0]
            vavg = tsunami.field_avg(vz, 1, xpt, ypt, zpt, 'y')[0]
            eflux_data = np.zeros((xpt, 4))
            ke = vavg[:, 0, 0] * vavg[:, 0, 0] / (2 * 9.81)
            depth_1d = tsunami.field_avg(depth_2d, 1, xpt, ypt, zpt, 'y')[0]
            te = depth_1d[:, 0, 0] + ke
            eflux_data[:, 0], eflux_data[:, 1], eflux_data[:, 2], eflux_data[:, 3] = depth_1d[:, 0, 0], ke, \
                                                                                     te, x_3d[:, 0, 0]
        energy_flux_data = eflux_data  # , ef_2d_z, ef_1d #, con_z, ef_2d_z, x_3d[:, 0, 0], depth_2d

    if turbulent_energy == 'on':
        # shear_data = 'off'
        v3d = np.zeros((3, xpt, ypt, zpt))
        v3d[0, :, :, :], v3d[1, :, :, :], v3d[2, :, :, :] = 1 * u_3d, 1 * v_3d, 1 * w_3d
        if levsmooth == 'on':
            from tsunami.smooth_level_set import lev_smooth
            v_3d, lev3d, z_3d, x_3d, y_3d, c3d, zpt = tsunami.lev_smooth(lev_3d, 1 * z_3d, v3d, xpt, ypt, zpt, 1 * x_3d,
                                                                         1 * y_3d, 3)
            lev_3d, v3d, z_3d, c_3d = tsunami.smooth_grid(z_3d, lev3d, v_3d, xpt, ypt, zpt, 3)

        if int_avg == 'int':
            v_y, con_y = tsunami.span_average(y_3d, 1 * v3d, lev_3d, xpt, ypt, zpt, 1 * c_3d, 3)

            v_prime = v3d - v_y
            # get spanwise average
            v_prime_y2, con_y = tsunami.span_average(y_3d, 1 * v_prime * v_prime, lev_3d, xpt, ypt, zpt, 1 * c_3d, 3)

            # get depth average

            v_prime_z2, con_z, depth_2d = tsunami.depth_average(lev_3d, z_3d, v_prime * v_prime, xpt, ypt, zpt, c_3d, 3)
            tke_z = 0.5 * (v_prime_z2[0, :, :, :] + v_prime_z2[1, :, :, :] + v_prime_z2[2, :, :, :])

            if turbulent_average == 'on':
                # get 1D-average
                v_prime_av2, con_y = span_average(y_3d, 1 * v_prime_z2, lev_3d, xpt, ypt, zpt, con_z, 3)
                k_av = 0.5 * (v_prime_av2[0, :, :, :] + v_prime_av2[1, :, :, :] + v_prime_av2[2, :, :, :])
                average_data = np.zeros((xpt, 2))
                for ipt in range(xpt):
                    average_data[ipt, 0] = max(k_av[ipt, 0, :])
                average_data[:, 1] = x_3d[:, 0, 0]

                energy_theory.append(average_data)
        if int_avg == 'field':
            u_3d[lev_3d < 0] = np.NaN
            v_3d[lev_3d < 0] = np.NaN
            w_3d[lev_3d < 0] = np.NaN
            nu_sgs_3d[lev_3d < 0] = np.NaN
            v3d[0, :, :, :], v3d[1, :, :, :], v3d[2, :, :, :] = 1 * u_3d, 1 * v_3d, 1 * w_3d
            v_y, v_prime_y2 = tsunami.field_avg(1 * v3d, 3, xpt, ypt, zpt, 'y')
            v_prime = v3d - v_y
        tke_3d = 0.5 * (v_prime[0, :, :, :] * v_prime[0, :, :, :] + v_prime[1, :, :, :] * v_prime[1, :, :, :] +
                        v_prime[2, :, :, :] * v_prime[2, :, :, :])
        r_stress = np.zeros((6, xpt, ypt, zpt))
        r_stress[0, :, :, :], r_stress[1, :, :, :], r_stress[2, :, :, :] = v_prime[0, :, :, :] * v_prime[0, :, :, :], \
                                                                           v_prime[1, :, :, :] * v_prime[1, :, :, :], \
                                                                           v_prime[2, :, :, :] * v_prime[2, :, :, :]

        r_stress[3, :, :, :], r_stress[4, :, :, :], r_stress[5, :, :, :] = v_prime[0, :, :, :] * v_prime[1, :, :, :], \
                                                                           v_prime[0, :, :, :] * v_prime[2, :, :, :], \
                                                                           v_prime[1, :, :, :] * v_prime[2, :, :, :]
        # get spanwise average

        tke_y = 0.5 * (v_prime_y2[0, :, :, :] + v_prime_y2[1, :, :, :] + v_prime_y2[2, :, :, :])

        r_stress_y = tsunami.field_avg(1 * r_stress, 6, xpt, ypt, zpt, 'y')[0]

        if shear_data == 'on':
            dupdx, dupdy, dupdz, dvpdx, dvpdy, dvpdz, dwpdx, dwpdy, dwpdz = np.zeros((xpt, ypt, zpt)), np.zeros(
                (xpt, ypt, zpt)), \
                                                                            np.zeros((xpt, ypt, zpt)), np.zeros(
                (xpt, ypt, zpt)), np.zeros((xpt, ypt, zpt)), np.zeros((xpt, ypt, zpt)), \
                                                                            np.zeros((xpt, ypt, zpt)), np.zeros(
                (xpt, ypt, zpt)), np.zeros((xpt, ypt, zpt))
            dv3dx, dv3dy, dv3dz = 0 * v3d, 0 * v3d, 0 * v3d

            # Turbulence Production
            tp = np.empty((xpt, ypt, zpt))
            for ipt in range(xpt):
                for jpt in range(ypt):
                    tp[ipt, jpt, :] = (-1 * r_stress_y[4, ipt, jpt, :] * np.gradient(v_y[0, ipt, jpt, :],
                                                                                     z_3d[ipt, jpt, :],
                                                                                     edge_order=2)) \
                                      + (-1 * r_stress_y[5, ipt, jpt, :] * np.gradient(v_y[1, ipt, jpt, :],
                                                                                       z_3d[ipt, jpt, :],
                                                                                       edge_order=2)) \
                                      + (-1 * r_stress_y[2, ipt, jpt, :] * np.gradient(v_y[2, ipt, jpt, :],
                                                                                       z_3d[ipt, jpt, :],
                                                                                       edge_order=2))

            K_y = 0.5 * (r_stress_y[0, :, :, :] + r_stress_y[1, :, :, :] + r_stress_y[2, :, :, :])
            '''
            for cc in range(3):
                for j in range(ypt):
                    for k in range(zpt):
                        dv3dx[cc, :, j, k] = np.gradient(v_y[cc, :, j, k], x_3d[:, j, k], edge_order=2)
            v_rot = 1 * v_y
            for cc in range(3):
                for i in range(xpt):
                    # for k in range(zpt):
                    #    dv3dy[cc, i, :, k] = np.gradient(v_y[cc, i, :, k], y_3d[i, :, k], edge_order=2)
                    for j in range(ypt):
                        dv3dz[cc, i, j, :] = np.gradient(1 * v_rot[cc, i, j, :], z_3d[i, j, :])  # ,
                        # edge_order=2)  # * math.sin(math.atan(0.1))


            for j in range(ypt):
                for k in range(zpt):
                    dupdx[:, j, k] = np.gradient(uprime_rot[:, j, k], x_3d[:, j, k])  # , edge_order=2)
                    dvpdx[:, j, k] = np.gradient(vprime[:, j, k], x_3d[:, j, k])  # , edge_order=2)
                    dwpdx[:, j, k] = np.gradient(wprime_rot[:, j, k], x_3d[:, j, k])  # , edge_order=2)

            for i in range(xpt):
                for k in range(zpt):
                    dupdy[i, :, k] = np.gradient(uprime_rot[i, :, k], y_3d[i, :, k])  # , edge_order=2)
                    dvpdy[i, :, k] = np.gradient(vprime[i, :, k], y_3d[i, :, k])  # , edge_order=2)
                    dwpdy[i, :, k] = np.gradient(wprime_rot[i, :, k], y_3d[i, :, k])  # , edge_order=2)

            for i in range(xpt):
                for j in range(ypt):
                    dupdz[i, j, :] = np.gradient(uprime_rot[i, j, :], z_3d[i, j, :])  # , edge_order=2)
                    dvpdz[i, j, :] = np.gradient(vprime[i, j, :], z_3d[i, j, :])  # , edge_order=2)
                    dwpdz[i, j, :] = np.gradient(wprime_rot[i, j, :], z_3d[i, j, :])  # , edge_order=2)

            
            # Turbulence dissipation
            SS = np.empty((5, xpt, ypt, zpt))
            S13 = 0.5 * (dupdz + dwpdx)
            S23 = 0.5 * (dvpdz + dwpdy)
            S33 = 0.5 * (dwpdz + dwpdz)
            S11 = 0.5 * (dupdx + dupdx)
            S21 = 0.5 * (dvpdx + dupdy)
            # S31 = 0.5 * (dwpdx + dupdz)
            SS[0, :, :, :], SS[1, :, :, :], SS[2, :, :, :] = S11 * S11, S21 * S21, S13 * S13
            SS[3, :, :, :], SS[4, :, :, :] = S23 * S23, S33 * S33
            SS_y = tsunami.field_avg(SS, 5, xpt, ypt, zpt, 1 * lev_3d, 'y')[0]
            # SS_y = tsunami.span_average(y_3d, 1 * SS, lev_3d, xpt, ypt, zpt, 1 * c_3d, 5)[0]

            epsi_3 = 2 * 1e-6 * (SS_y[2, :, :, :] + SS_y[3, :, :, :] + SS_y[4, :, :, :])
            epsi_1 = 2 * 1e-6 * (SS_y[0, :, :, :] + SS_y[1, :, :, :] + SS_y[2, :, :, :])
            # SS_y_sgs = tsunami.span_average(y_3d, nu_sgs_3d * SS, lev_3d, xpt, ypt, zpt, 1 * c_3d, 5)[0]
            SS_y_sgs = tsunami.field_avg(SS * nu_sgs_3d, 5, xpt, ypt, zpt, 1 * lev_3d, 'y')[0]
            epsi_3_sgs = 2 * (SS_y_sgs[2, :, :, :] + SS_y_sgs[3, :, :, :] + SS_y_sgs[4, :, :, :])
            epsi_1_sgs = 2 * (SS_y_sgs[0, :, :, :] + SS_y_sgs[1, :, :, :] + SS_y_sgs[2, :, :, :])
            epsi_1, epsi_3 = (epsi_1 + epsi_1_sgs), (epsi_3 + epsi_3_sgs)
            epsi = np.sqrt(epsi_1 ** 2 + epsi_3 ** 2)
            '''
            total_energy_data = x_3d, y_3d, z_3d, lev_3d, mask_3d, c_3d, \
                                tp, v_y[0, :, :, :], v_y[1, :, :, :], v_y[2, :, :, :], v3d[0, :, :, :], v3d[1, :, :, :] \
                , v3d[2, :, :, :], tke_y, tke_3d  # ,epsi
            if turbulent_average == 'on':
                up_av, con_y = span_average(y_3d, up_z, conup_z, xpt, ypt, zpt)
                vp_av, con_y = span_average(y_3d, vp_z, convp_z, xpt, ypt, zpt)
                wp_av, con_y = span_average(y_3d, wp_z, conwp_z, xpt, ypt, zpt)

                tau_avg_z = np.sqrt(epsi_xy_z ** 2 + epsi_xz_z ** 2 + epsi_yz_z ** 2)
                # print('average_case')
                tau_avg, con_y = span_average(y_3d, tau_avg_z, con_z, xpt, ypt, zpt)
                # print('tke_start')
                tke_av = 0.5 * (up_av + vp_av + wp_av)
                # print('ave_end')
                average_data = np.zeros((xpt, 3))
                average_data[:, 0], average_data[:, 1], average_data[:, 2] = tke_av[:, 0, 0], tau_avg[:, 0, 0], x_3d[:,
                                                                                                                0,
                                                                                                                0]
                energy_theory.append(average_data)
        else:
            total_energy_data = x_3d, y_3d, z_3d, lev_3d, c_3d, tke_3d, r_stress, tke_y, r_stress_y, \
                                tke_z, r_stress_z, depth_2d, con_y, con_z, v_mag_3d

    if water_depth == 'on':
        # print(int_avg)
        if int_avg == 'int':
            if levsmooth == 'on':
                v_mag_3d, lev3d, z3d, x_3d, y_3d, c_3d, zpt = tsunami.lev_smooth(lev_3d, z_3d, v_mag_3d, xpt, ypt, zpt,
                                                                                 x_3d, y_3d,
                                                                                 1)
                lev_3d, v_mag_3d, z_3d, c_3d = tsunami.smooth_grid(z3d, lev3d, v_mag_3d, xpt, ypt, zpt, 1)
            v_z, con_z, depth_2d = tsunami.depth_average(lev_3d, z_3d, v_mag_3d, xpt, ypt, zpt, c_3d, 1)
            v2d = np.zeros((2, xpt, ypt, zpt))
            v2d[0, :, :, :], v2d[1, :, :, :] = v_z, depth_2d
            vd_1d, con_y = tsunami.span_average(y_3d, 1 * v2d, lev_3d, xpt, ypt, zpt, con_z, 2)
            xx3d = x_3d[:, 0, 0][x_3d[:, 0, 0] >= 10.8]
            depth_data = np.zeros((np.shape(xx3d)[0], 3))
            index = 0
            for ipt in range(xpt):
                if x_3d[ipt, 0, 0] >= 10.8:
                    depth_data[index, 0] = max(vd_1d[0, ipt, 0, :])
                    depth_data[index, 1] = max(vd_1d[1, ipt, 0, :])
                    index = index + 1

            depth_data[:, 2] = xx3d[:]
            energy_theory.append(depth_data)
        elif int_avg == 'field':
            v_mag_3d[lev_3d < 0] = np.NaN
            z_3d[lev_3d < 0] = np.NaN
            v2d = np.zeros((2, xpt, ypt, zpt))
            v2d[0, :, :, :], v2d[1, :, :, :] = v_mag_3d, z_3d
            for ipt in range(xpt):
                for jpt in range(ypt):
                    for kpt in range(zpt):
                        v2d[1, ipt, jpt, kpt] = np.nanmax(z_3d[ipt, jpt, :])

            v2d = tsunami.field_avg(v2d, 2, xpt, ypt, zpt, 'y')[0]
            v2d[0, :, :, :] = tsunami.field_avg(v2d[0, :, :, :], 1, xpt, ypt, zpt, 'z')[0]
            xx3d = x_3d[:, 0, 0][x_3d[:, 0, 0] >= 10.8]
            depth_data = np.zeros((np.shape(xx3d)[0], 3))
            index = 0
            for ipt in range(xpt):
                if x_3d[ipt, 0, 0] >= 10.8:
                    depth_data[index, 0] = max(v2d[0, ipt, 0, :])
                    depth_data[index, 1] = max(v2d[1, ipt, 0, :])
                    index = index + 1

            depth_data[:, 2] = xx3d[:]
            # print(depth_data)
            energy_theory.append(depth_data)

    if dam_front == 'on':
        iindex = -5
        xs = np.zeros(ypt)
        for j in range(ypt):
            for i in range(xpt):
                for k in range(zpt):
                    # print(xpt - 1 - i, j, zpt - 1 - k)
                    if lev_3d[xpt - 1 - i, j, zpt - 1 - k] >= 0:
                        # print('i,xi', xpt - 1 - i, x_3d[xpt - 1 - i, j, zpt - 1 - k])
                        xs[j] = max(x_3d[xpt - 1 - i, j, :])
                        iindex = 1 * i
                        break

                if i == iindex:
                    iindex = -5
                    break

        xs_sp = integrate.simps(xs, y_3d[0, :, 0]) / max(y_3d[0, :, 0])
        energy_theory.append(xs_sp)

    if vel_profile[0] == 'on':
        v_vector_3d = np.zeros((3, xpt, ypt, zpt))
        v_vector_3d[0, :, :, :], v_vector_3d[1, :, :, :], v_vector_3d[2, :, :, :] = u_3d, v_3d, w_3d
        v_vector_3d_sp, con_y = span_average(y_3d, v_vector_3d, lev_3d, xpt, ypt, zpt, c_3d, 3)
        vs = np.zeros((zpt, 5))
        for ix in range(xpt):
            if x_3d[ix, 0, 0] == vel_profile[1]:
                vs[:, 0] = v_vector_3d_sp[0, ix, 0, :]
                vs[:, 1] = v_vector_3d_sp[1, ix, 0, :]
                vs[:, 2] = v_vector_3d_sp[2, ix, 0, :]
                vs[:, 3] = np.sqrt(v_vector_3d_sp[0, ix, 0, :] * v_vector_3d_sp[0, ix, 0, :] +
                                   v_vector_3d_sp[1, ix, 0, :] * v_vector_3d_sp[1, ix, 0, :] +
                                   v_vector_3d_sp[2, ix, 0, :] * v_vector_3d_sp[2, ix, 0, :])
        vs[:, 4] = z_3d[0, 0, :]
        energy_theory.append(vs)

    if wall_shear_stress == 'on':
        xnew = x[(x >= 0) & (x <= 5)]
        xnewpt = int(len(xnew) / (ypt * zpt))
        print('xnewpt', xnewpt, ypt)
        mask = mask[(x >= 0) & (x <= 5)]
        # print('s',np.size(mask), np.size(xnew))
        x_3d, y_3d, z_3d, u_3d, v_3d, w_3d, tau_3d, lev_3d, mask_3d = str_array(xnew, y[(x >= 0) & (x <= 5)],
                                                                                z[(x >= 0) & (x <= 5)],
                                                                                xnewpt, ypt, zpt,
                                                                                u[(x >= 0) & (x <= 5)],
                                                                                v[(x >= 0) & (x <= 5)],
                                                                                w[(x >= 0) & (x <= 5)],
                                                                                tau[(x >= 0) & (x <= 5)],
                                                                                lev[(x >= 0) & (x <= 5)],
                                                                                mask)
        c_3d = 1 * mask_3d
        c_3d[lev_3d < lev_set] = 0
        tau_data = np.zeros((xnewpt, 2))
        tau_2d = np.zeros(xnewpt)
        tau_sp, con_y = span_average(y_3d, tau_3d, c_3d, xnewpt, ypt, zpt)
        for i in range(xnewpt):
            for j in range(ypt):
                for k in range(zpt):
                    if c_3d[i, j, k] == 1:
                        tau_2d[i] = tau_sp[i, j, k]
                        break

        tau_data[:, 0], tau_data[:, 1] = tau_2d, x_3d[:, 0, 0]
        energy_theory.append(tau_data)

    return energy_flux_data, total_energy_data, energy_theory

    # return index, hh, Z


if __name__ == "__main__":
    eflux()
    '''
    #r_stress_y[0, :, :, :], r_stress_y[1, :, :, :], r_stress_y[2, :, :, :] = v_prime_y2[0, :, :, :], \
        #                                                                         v_prime_y2[1, :, :, :], \
        #                                                                         v_prime_y2[2, :, :, :]
            #s12, s21 = 0.5 * (dudy + dvdx), 0.5 * (dudy + dvdx)
            s13, s31 = 0.5 * (dudz + dwdx), 0.5 * (dudz + dwdx)
            s23, s32 = 0.5 * (dwdy + dvdz), 0.5 * (dwdy + dvdz)
            epsi_1 = 2 * 1e-2 * (s12 + s13)
            epsi_2 = 2 * 1e-2 * (s21 + s23)
            epsi_3 = 2 * 1e-2 * (s31 + s32)
    
            # span-wise average
            epsi_xy_y, con_y = span_average(y_3d, epsi_1, c_3d, xpt, ypt, zpt)
            epsi_xz_y, con_y = span_average(y_3d, epsi_2, c_3d, xpt, ypt, zpt)
            epsi_yz_y, con_y = span_average(y_3d, epsi_3, c_3d, xpt, ypt, zpt)
            # print('turbulent')
            epsi_xy_z, con_z, depth_2d = depth_average(lev_3d, z_3d, epsi_1, mask_3d, xpt, ypt, zpt, c_3d)
            epsi_xz_z, con_z, depth_2d = depth_average(lev_3d, z_3d, epsi_2, mask_3d, xpt, ypt, zpt, c_3d)
            epsi_yz_z, con_z, depth_2d = depth_average(lev_3d, z_3d, epsi_3, mask_3d, xpt, ypt, zpt, c_3d)
            # print('turbulent_end')
            

            total_energy_data = x_3d, y_3d, z_3d, epsi_1, epsi_2, epsi_3, con_z, epsi_xy_y, epsi_xz_y, epsi_yz_y, \
                                con_y, epsi_xy_z, epsi_xz_z, epsi_yz_z, lev_3d, uprime, vprime, wprime, mask_3d, c_3d
        
        # r_stress_y[3, :, :, :], r_stress_y[4, :, :, :], r_stress_y[5, :, :, :] =

        # get depth average

        # v_prime_z2, con_z, depth_2d = depth_average(lev_3d, z_3d, v_prime * v_prime, xpt, ypt, zpt, c_3d, 3)
        # tke_z = 0.5 * (v_prime_z2[0, :, :, :] + v_prime_z2[1, :, :, :] + v_prime_z2[2, :, :, :])
        # r_stress_z, con_z, depth_2d = depth_average(lev_3d, z_3d, r_stress, xpt, ypt, zpt, c_3d, 6)
            up_y, con_y = span_average(y_3d, uprime ** 2, c_3d, xpt, ypt, zpt)
            vp_y, con_y = span_average(y_3d, vprime ** 2, c_3d, xpt, ypt, zpt)
            wp_y, con_y = span_average(y_3d, wprime ** 2, c_3d, xpt, ypt, zpt)

            TKE_y = 0.5 * (up_y + vp_y + wp_y)

            R_uv_y, con_y = span_average(y_3d, R_uv, c_3d, xpt, ypt, zpt)
            R_uw_y, con_y = span_average(y_3d, R_uw, c_3d, xpt, ypt, zpt)
            R_vw_y, con_y = span_average(y_3d, R_vw, c_3d, xpt, ypt, zpt)

            R_uu_y, con_y = span_average(y_3d, R_uu, c_3d, xpt, ypt, zpt)
            R_vv_y, con_y = span_average(y_3d, R_vv, c_3d, xpt, ypt, zpt)
            R_ww_y, con_y = span_average(y_3d, R_ww, c_3d, xpt, ypt, zpt)

            # Get depth average
            up_z, conup_z, depth_2d = depth_average(lev_3d, z_3d, uprime ** 2, mask_3d, xpt, ypt, zpt, c_3d)
            vp_z, convp_z, depth_2d = depth_average(lev_3d, z_3d, vprime ** 2, mask_3d, xpt, ypt, zpt, c_3d)
            wp_z, conwp_z, depth_2d = depth_average(lev_3d, z_3d, wprime ** 2, mask_3d, xpt, ypt, zpt, c_3d)

            TKE_z = 0.5 * (up_z + vp_z + wp_z)
            R_uv_z, con_z, depth_2d = depth_average(lev_3d, z_3d, R_uv, mask_3d, xpt, ypt, zpt, c_3d)
            R_uw_z, con_z, depth_2d = depth_average(lev_3d, z_3d, R_uw, mask_3d, xpt, ypt, zpt, c_3d)
            R_vw_z, con_z, depth_2d = depth_average(lev_3d, z_3d, R_vw, mask_3d, xpt, ypt, zpt, c_3d)

            R_uu_z, con_z, depth_2d = depth_average(lev_3d, z_3d, R_uu, mask_3d, xpt, ypt, zpt, c_3d)
            R_vv_z, con_z, depth_2d = depth_average(lev_3d, z_3d, R_vv, mask_3d, xpt, ypt, zpt, c_3d)
            R_ww_z, con_z, depth_2d = depth_average(lev_3d, z_3d, R_ww, mask_3d, xpt, ypt, zpt, c_3d)

            # Get norm properties
            if norm == 'on':
                # 3d data
                # print('shape', np.shape(depth_2d), np.shape(TKE_3d))
                TKE_3d = norm_vector(TKE_3d, depth_2d, xpt, ypt, zpt)

                R_uu_3d, R_vv_3d, R_ww_3d = norm_vector(R_uu_3d, depth_2d, xpt, ypt, zpt), norm_vector(R_vv_3d,
                                                                                                       depth_2d,
                                                                                                       xpt, ypt,
                                                                                                       zpt), norm_vector(
                    R_ww_3d, depth_2d, xpt, ypt, zpt)

                R_uv_3d, R_uw_3d, R_vw_3d = norm_vector(R_uv_3d, depth_2d, xpt, ypt, zpt), norm_vector(R_uw_3d,
                                                                                                       depth_2d,
                                                                                                       xpt, ypt,
                                                                                                       zpt), norm_vector(
                    R_vw_3d, depth_2d, xpt, ypt, zpt)

                # Span-wise average data

                TKE_y = norm_vector(TKE_y, depth_2d, xpt, ypt, zpt)

                R_uu_y, R_vv_y, R_ww_y = norm_vector(R_uu_y, depth_2d, xpt, ypt, zpt), norm_vector(R_vv_y, depth_2d,
                                                                                                   xpt, ypt,
                                                                                                   zpt), norm_vector(
                    R_ww_y, depth_2d, xpt, ypt, zpt)

                R_uv_y, R_uw_y, R_vw_y = norm_vector(R_uv_y, depth_2d, xpt, ypt, zpt), norm_vector(R_uw_y, depth_2d,
                                                                                                   xpt, ypt,
                                                                                                   zpt), norm_vector(
                    R_vw_y, depth_2d, xpt, ypt, zpt)

                # depth average data

                TKE_z = norm_vector(TKE_z, depth_2d, xpt, ypt, zpt)

                R_uu_z, R_vv_z, R_ww_z = norm_vector(R_uu_z, depth_2d, xpt, ypt, zpt), norm_vector(R_vv_z, depth_2d,
                                                                                                   xpt, ypt,
                                                                                                   zpt), norm_vector(
                    R_ww_z, depth_2d, xpt, ypt, zpt)

                R_uv_z, R_uw_z, R_vw_z = norm_vector(R_uv_z, depth_2d, xpt, ypt, zpt), norm_vector(R_uw_z, depth_2d,
                                                                                                   xpt, ypt,
                                                                                                   zpt), norm_vector(
                    R_vw_z, depth_2d, xpt, ypt, zpt)
     alt_module = 'on'
     norm = 'off'
     uprime, vprime, wprime = np.zeros((xpt, ypt, zpt)), np.zeros((xpt, ypt, zpt)), np.zeros(
         (xpt, ypt, zpt))

     TKE, R_uv, R_uw, R_vw = np.zeros((xpt, ypt, zpt)), np.zeros((xpt, ypt, zpt)), np.zeros(
         (xpt, ypt, zpt)), np.zeros((xpt, ypt, zpt))
     R_uu, R_vv, R_ww = np.zeros((xpt, ypt, zpt)), np.zeros((xpt, ypt, zpt)), np.zeros(
         (xpt, ypt, zpt))

     if alt_module == 'on':

         # span average
         u_x, con_y = span_average(y_3d, u_3d, c_3d, xpt, ypt, zpt)  # span_average(y_3d, u_xy, con_z, xpt, ypt, zpt)
         v_x, con_y = span_average(y_3d, v_3d, c_3d, xpt, ypt, zpt)  # span_average(y_3d, v_xy, con_z, xpt, ypt, zpt)
         w_x, con_y = span_average(y_3d, w_3d, c_3d, xpt, ypt, zpt)  # span_average(y_3d, w_xy, con_z, xpt, ypt, zpt)

         for i in range(xpt):
             for j in range(ypt):
                 for k in range(zpt):
                     uprime[i, j, k] = (u_3d[i, j, k] - u_x[i, j, k])
                     vprime[i, j, k] = (v_3d[i, j, k] - v_x[i, j, k])
                     wprime[i, j, k] = (w_3d[i, j, k] - w_x[i, j, k])
                     TKE[i, j, k] = 0.5 * ((u_3d[i, j, k] - u_x[i, j, k]) ** 2 + (v_3d[i, j, k] - v_x[i, j, k]) ** 2
                                           + (w_3d[i, j, k] - w_x[i, j, k]) ** 2)
                     R_uv[i, j, k] = (u_3d[i, j, k] - u_x[i, j, k]) * (v_3d[i, j, k] - v_x[i, j, k])
                     R_uw[i, j, k] = (u_3d[i, j, k] - u_x[i, j, k]) * (w_3d[i, j, k] - w_x[i, j, k])
                     R_vw[i, j, k] = (v_3d[i, j, k] - u_x[i, j, k]) * (w_3d[i, j, k] - w_x[i, j, k])
                     R_uu[i, j, k] = (u_3d[i, j, k] - u_x[i, j, k]) * (u_3d[i, j, k] - u_x[i, j, k])
                     R_vv[i, j, k] = (v_3d[i, j, k] - v_x[i, j, k]) * (v_3d[i, j, k] - v_x[i, j, k])
                     R_ww[i, j, k] = (w_3d[i, j, k] - w_x[i, j, k]) * (w_3d[i, j, k] - w_x[i, j, k])

         TKE_3d, R_uv_3d, R_uw_3d, R_vw_3d, R_uu_3d, R_vv_3d, R_ww_3d = \
             1 * TKE, 1 * R_uv, 1 * R_uw, 1 * R_vw, 1 * R_uu, 1 * R_vv, 1 * R_ww

         # Smoothen level set
         # uprime, lev3d = lev_smooth(lev_3d, x_3d, y_3d, z_3d, uprime, xpt, ypt, zpt, c_3d)
         # vprime, lev3d = lev_smooth(lev_3d, x_3d, y_3d, z_3d, vprime, xpt, ypt, zpt, c_3d)
         # wprime, lev3d = lev_smooth(lev_3d, x_3d, y_3d, z_3d, wprime, xpt, ypt, zpt, c_3d)
         '''
    '''
            if alt_module == 'off':
            for ix in range(xpt):
                for jy in range(ypt):
                    u_xy[ix, jy] = integrate.simps(uu_3d[ix, jy, :], z_3d[ix, jy, :])
                    v_xy[ix, jy] = integrate.simps(vv_3d[ix, jy, :], z_3d[ix, jy, :])
                    w_xy[ix, jy] = integrate.simps(ww_3d[ix, jy, :], z_3d[ix, jy, :])
                u_x[ix] = integrate.simps(u_xy[ix, :], y_3d[ix, :, 0])
                v_x[ix] = integrate.simps(v_xy[ix, :], y_3d[ix, :, 0])
                w_x[ix] = integrate.simps(w_xy[ix, :], y_3d[ix, :, 0])

            u_sp_avg = integrate.simps(u_x, x_3d[:, 0, 0]) / water_vol
            v_sp_avg = integrate.simps(v_x, x_3d[:, 0, 0]) / water_vol
            w_sp_avg = integrate.simps(w_x, x_3d[:, 0, 0]) / water_vol
            for i in range(xpt):
                for j in range(ypt):
                    for k in range(zpt):
                        TKE[i, j, k] = 0.5 * ((u_3d[i, j, k] - u_sp_avg) ** 2 + (v_3d[i, j, k] - v_sp_avg) ** 2 + (
                                w_3d[i, j, k] - w_sp_avg) ** 2)
                        R_uv[i, j, k] = (u_3d[i, j, k] - u_sp_avg) * (v_3d[i, j, k] - v_sp_avg)
                        R_uw[i, j, k] = (u_3d[i, j, k] - u_sp_avg) * (w_3d[i, j, k] - w_sp_avg)
                        R_vw[i, j, k] = (v_3d[i, j, k] - v_sp_avg) * (w_3d[i, j, k] - w_sp_avg)
            R_uu = (u_3d - u_sp_avg) * (u_3d - u_sp_avg)
            R_vv = (v_3d - v_sp_avg) * (v_3d - v_sp_avg)
            R_ww = (w_3d - w_sp_avg) * (w_3d - w_sp_avg)
            # TKE[lev_3d < lev_set] = 0
            # R_uv[lev_3d < lev_set], R_uw[lev_3d < lev_set], R_vw[lev_3d < lev_set] = 0, 0, 0
            # R_uu[lev_3d < lev_set], R_vv[lev_3d < lev_set], R_ww[lev_3d < lev_set] = 0, 0, 0
            if depth_plot == 'on':
            fig, ax1 = plt.subplots(figsize=(12, 3.5), dpi=90, facecolor="white")
            clevels = np.linspace(vv_2d.min(), 15, n_bins)
            for axis in ['top', 'bottom', 'left', 'right']:
                ax1.spines[axis].set_linewidth(2)
            ax1.set_yticks(np.arange(0, 5, step=1))
            cf = ax1.contourf(x_3d[:, :, 0], y_3d[:, :, 0], vv_2d, clevels, cmap=cm, extend='both')
            for c in cf.collections:
                c.set_edgecolor("face")
            cbar = fig.colorbar(cf, ticks=[vv_2d.min(), 5, 10, 15], ax=ax1, extendrect=True, extendfrac=0.0,
                                fraction=0.0285, pad=0.05, label=r'Velocity Magnitude')
            plt.tight_layout()
            cbar.ax.set_yticklabels(['0', '5', '10', '15'])
            if draw_circle == 'on':
                cc, r = 10.4, 0.4
                circle = plt.Circle((cc, 2), r, color='k', fill=False)
                ax1.add_artist(circle)
            plt.ylabel(r'$Y (m)$')
            plt.xlabel(r'$X (m)$')
            plt.show()
        energy_flux_data = x_3d, y_3d, z_3d, v_z, con_z, lev_3d, mask_3d, depth_2d, v_y, con_y, v_mag_3d
        
        if total_energy == 'yes':
        # X, Y, Z, X1, Y1, Z1 = regular_grid(x, y, z, xpt, ypt, zpt)
        ke = griddata((x, y, z), TKE, (X, Y, Z), method='linear', fill_value=0.0)
        h = griddata((x, y, z), lev, (X, Y, Z), method='linear', fill_value=0.0)

        g = 9.81

        ZZ = np.reshape(Z, (len(X1), len(Y1), len(Z1)))
        hh = np.reshape(h, (len(X1), len(Y1), len(Z1)))
        KE = np.reshape(ke, (len(X1), len(Y1), len(Z1)))
        PE = np.reshape(g * Z, (len(X1), len(Y1), len(Z1)))

        kinz = np.zeros((len(X1), len(Y1)))
        pinz = np.zeros((len(X1), len(Y1)))
        kiny = np.zeros(len(X1))
        piny = np.zeros(len(X1))

        for ix in range(len(X1)):
            for iy in range(len(Y1)):
                for iz in range(len(Z1)):
                    if hh[ix, iy, iz] < 0:
                        KE[ix, iy, iz] = 0
                        PE[ix, iy, iz] = 0
                kinz[ix, iy] = integrate.simps(KE[ix, iy, :], Z1)
                pinz[ix, iy] = integrate.simps(PE[ix, iy, :], Z1)

            kiny[ix] = integrate.simps(kinz[ix, :], Y1)
            piny[ix] = integrate.simps(pinz[ix, :], Y1)
        kinx = integrate.simps(kiny, X1)
        pinx = integrate.simps(piny, X1)
        total_energy_data = kinx, pinx, kinx + pinx
        # print('total', pinx)
        plot = 'yes'
        
        if dam_theory == 'yes':
        if i == 0:
            X, Y, Z, X1, Y1, Z1 = regular_grid(x, y, z, xpt, ypt, zpt)
            u_t = np.zeros(xpt)
            H_t = np.zeros(xpt)
            ep = np.zeros((zpt, xpt))
            ek = np.zeros((zpt, xpt))
            zz = np.zeros((zpt, xpt))

            t_theo = np.linspace(0, final_time, t_steps)
            Epaux = np.zeros((len(t_theo), xpt))
            Ekaux = np.zeros((len(t_theo), xpt))
            Eptot = np.zeros(len(t_theo))
            Ektot = np.zeros(len(t_theo))

            # Dam position:
            hl = 4  # initial dam height
            x0 = d_l

            g = 9.81
            c0 = np.sqrt(hl * g)

            for it in range(len(t_theo)):
                xa = x0 - t_theo[it] * c0
                xb = x0 + 2 * t_theo[it] * c0

                for ix in range(xpt):
                    if X1[ix] < xa:
                        H_t[ix] = hl
                        u_t[ix] = 0.0

                    elif xa <= X1[ix] <= xb:
                        H_t[ix] = ((c0 - (X1[ix] - x0) / (2 * t_theo[it])) ** 2) * 4 / (9 * g)
                        u_t[ix] = (c0 + (X1[ix] - x0) / t_theo[it]) * 2 / 3

                    else:
                        H_t[ix] = 0.0
                        u_t[ix] = 0.0

                    # Build water column for integration along z

                    zz[:, ix] = np.linspace(0, H_t[ix], zpt)
                    for iz in range(zpt):
                        ep[iz, ix] = 9.81 * zz[iz, ix]
                        ek[iz, ix] = 0.5 * u_t[ix] * u_t[ix]
                    # print(ep[:, ix])
                    Epaux[it, ix] = np.nan_to_num(integrate.simps(ep[:, ix], zz[:, ix]))
                    Ekaux[it, ix] = np.nan_to_num(integrate.simps(ek[:, ix], zz[:, ix]))

                # print(Epaux[0, :])
                Eptot[it] = integrate.simps(Epaux[it, :], X1)
                Ektot[it] = integrate.simps(Ekaux[it, :], X1)
            Etot = Eptot + Ektot
            energy_theory = Ektot, Eptot, Etot, t_theo

    if energy_flux_paraview == 'yes':
        # print(data_directory, filename)
        # flux_file = flux_directory + '.csv'
        # flux_path = os.path.join(flux_directory, flux_file)
        flux_parameter = np.loadtxt(os.path.join(data_directory, filename), delimiter=',', dtype='str')[0, :]
        flux_plane = np.loadtxt(os.path.join(data_directory, filename), delimiter=',', skiprows=1)

        try:
            peflux = flux_plane[flux_parameter.tolist().index('"PE_flux"')]
            keflux = flux_plane[flux_parameter.tolist().index('"KE_flux"')]

        except:
            peflux = 0
            keflux = 0
        tot_energy_flux = peflux + keflux
        energy_flux_data = peflux, keflux, tot_energy_flux
    if force == 'on':
        force_parameter = np.loadtxt(os.path.join(data_directory, filename), delimiter=',', dtype='str')[0, :]
        force_plane = np.loadtxt(os.path.join(data_directory, filename), delimiter=',', skiprows=1)
        F_px = force_plane[force_parameter.tolist().index('"drag:0"')]
        lev = force_plane[force_parameter.tolist().index('"LEVEL"')]

        if lev < 0:
            F_px = 0

        total_energy_data = F_px
    if probe_data == 'on':
        probe_parameter = np.loadtxt(os.path.join(data_directory, filename), delimiter=',', dtype='str')[0, :]
        probe_plane = np.loadtxt(os.path.join(data_directory, filename), delimiter=',', skiprows=1)
        u_x = probe_plane[probe_parameter.tolist().index('"VELOC:0"')]
        u_y = probe_plane[probe_parameter.tolist().index('"VELOC:1"')]
        u_z = probe_plane[probe_parameter.tolist().index('"VELOC:2"')]
        lev = probe_plane[probe_parameter.tolist().index('"LEVEL"')]
        # print(lev)
        if average == 'on':
            if i == 0:
                u_avx = 0
                u_avz = 0
                u_avy = 0
            else:
                u_avx = probe_plane[probe_parameter.tolist().index('"AVVEL:0"')]
                u_avy = probe_plane[probe_parameter.tolist().index('"AVVEL:1"')]
                u_avz = probe_plane[probe_parameter.tolist().index('"AVVEL:2"')]
                u_av = np.sqrt(u_avx ** 2 + u_avy ** 2 + u_avz ** 2)
        else:
            u_avx, u_avz = 0, 0
        if lev < 0:
            u_x = 0
            u_y = 0
            u_z = 0
            u_av = 0
            lev = 0.02
        vel_mag = np.sqrt(u_x ** 2 + u_y ** 2 + u_z ** 2)
        energy_flux_data = u_x, u_y, u_z, vel_mag, vel_mag, lev

    if total_energy_paraview == 'yes':
        energy_directory = 'total_energy'
        energy_file = 'total_energy_' + str(i) + '.csv'
        energy_parameter = np.loadtxt(os.path.join(energy_directory, energy_file), delimiter=',', dtype='str')[0, :]
        energy_plane = np.loadtxt(os.path.join(energy_directory, energy_file), delimiter=',', skiprows=1)
        ike = energy_plane[energy_parameter.tolist().index('"IKE"')]
        pe = energy_plane[energy_parameter.tolist().index('"PE"')]

        try:
            tke = energy_plane[energy_parameter.tolist().index('"TKE"')]
        except:
            tke = 0

        tot_energy = ike + pe
        total_energy_data = pe, ike, tke, tot_energy
        
    water_depth
    # print(ypt, zpt)
        if max(lev) < lev_set:  #
            h_avg = 0, 0
        else:
            # Filtering the data as per the level set, LEV>=0 for water
            # znew = z[lev >= lev_set]
            # print(min(y), max(y), ypt)
            ylin = np.linspace(min(y), max(y), ypt)
            # a = np.linspace(1, 1, zpt)
            zlin = np.linspace(min(z), max(z), zpt)
            y_2d, z_2d = np.meshgrid(ylin, zlin)
            depth_2d = 1 * z_2d
            # zz = integrate.simps(z, y)
            # z_avg = zz / (max(y) - min(y))
            # print('avgz', z_avg)
            lev_str = griddata((y, z), lev, (y_2d, z_2d), method='linear', fill_value=0.0)
            # print(z_2d[:, 0])
            u_str = griddata((y, z), u, (y_2d, z_2d), method='linear', fill_value=0.0)
            v_mag = griddata((y, z), V, (y_2d, z_2d), method='nearest', fill_value=0.0)
            if average == 'on':
                k_TE_str = griddata((y, z), k_TE, (y_2d, z_2d), method='linear', fill_value=0.0)
                v_mag_av = griddata((y, z), V_av, (y_2d, z_2d), method='nearest', fill_value=0.0)
                kk_TE = np.zeros(ypt)
            vv = np.zeros(ypt)
            vv_av_max = np.zeros(ypt)
            vv_av = np.zeros(ypt)
            z_max = np.zeros(ypt)
            uu = np.zeros(ypt)
            lev_max = np.zeros(ypt)
            v_max = np.zeros(ypt)
            vt_avg, vt_max_avg = 0, 0

            for jj in range(ypt):
                for kk in range(zpt):
                    # print(kk)
                    if lev_str[kk, jj] < 0:
                        depth_2d[kk, jj] = 0
                        u_str[kk, jj] = 0
                        v_mag[kk, jj] = 0
                        if average == 'on':
                            k_TE_str[kk, jj] = 0
                            v_mag_av[kk, jj] = 0

            for jj in range(ypt):
                z_max[jj] = max(depth_2d[:, jj])
                lev_max[jj] = max(lev_str[:, jj])
                v_max[jj] = max(v_mag[:, jj])
                if average == 'on':
                    vv_av_max[jj] = max(v_mag_av[:, jj])
                if z_max[jj] > 0:
                    vv[jj] = integrate.simps(v_mag[:, jj], zlin) / z_max[jj]
                    uu[jj] = integrate.simps(u_str[:, jj], zlin) / z_max[jj]
                    if average == 'on':
                        vv_av[jj] = integrate.simps(v_mag_av[:, jj], zlin) / z_max[jj]
                        kk_TE[jj] = integrate.simps(k_TE_str[:, jj], zlin) / z_max[jj]
                else:
                    vv[jj] = 0
                    if average == 'on':
                        vv_av[jj] = 0

            z_avg = integrate.simps(z_max, y_2d[0, :]) / (max(y) - min(y))
            lev_avg = integrate.simps(lev_max, y_2d[0, :]) / (max(y) - min(y))

            if average == 'on':
                k_avg = integrate.simps(kk_TE, ylin) / (max(y) - min(y))
                vt_avg = integrate.simps(vv_av, ylin) / (max(y) - min(y))
                vt_max_avg = integrate.simps(vv_av_max, ylin) / (max(y) - min(y))
            v_avg = integrate.simps(vv, ylin) / (max(y) - min(y))
            v_max_avg = integrate.simps(v_max, ylin) / (max(y) - min(y))
            vv_max = max(v_max)
        # tot_energy_flux = pe_flux + ke_flux

        # print(ke_flux)
        energy_flux_data = v_avg, v_max_avg, z_avg, vt_max_avg
        
        
    fname = 'time'
    fe = '_'
    filename = fname + fe + str(i) + '.csv'
    input_path = os.path.join(data_directory, filename)
    # plane = np.genfromtxt(os.path.join(data_directory, filename), delimiter=';')[:,:-1]

    # plane = np.loadtxt(os.path.join(data_directory, filename), delimiter=",", skiprows=1, dtype=str)
    t = 0
    with open(input_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if t > 0:
                time = float(row[0])
            t = t + 1
    return time
    print(tot_energy_flux, vel_flux, fluid_area, area)
    
    fig, ax1 = plt.subplots(figsize=(11, 7), dpi=90)
    levels = np.linspace(VV.min(), VV.max(), 256)
    cf = ax1.contourf(YY, ZZ, VV, levels, cmap='viridis', extend='both')
    
    ax1.set_aspect('equal')
    fig.colorbar(cf, extendrect=True, extendfrac=0.0, fraction=0.0285, pad=0.05, label=r'$U/U_{\infty}$')
    
    plt.ylabel(r'$y$')
    plt.xlabel(r'$x$')
    plt.title(r'x = 20.25')
    plt.show()
    
    
    old calculation of energy flux:
                for i in range(len(Y1)):
                h_z[i] = max(hh[i, :])

                for j in range(len(Z1)):
                    if hh[i, j] < 0:
                        if j > 0:
                            print(i, j-1)
                            z_z[i] = ((ZZ[i, j - 1]) - ZZ[i, 0])
                            vel_int_z[i] = integrate.simps(VV[i, 0:j], ZZ[i, 0:j])
                            u_int_z[i] = integrate.simps(uu[i, 0:j], ZZ[i, 0:j])
                            q_int_z1[i] = integrate.simps(qq[i, 0:j] * h_z[i], ZZ[i, 0:j])
                            q_int_z2[i] = integrate.simps(qq[i, 0:j], ZZ[i, 0:j])
                            p_int_z1[i] = integrate.simps(uu[i, 0:j] * h_z[i] * h_z[i], ZZ[i, 0:j])
                            p_int_z2[i] = integrate.simps(uu[i, 0:j] * z_z[i], ZZ[i, 0:j])
                        else:
                            z_z[i] = ZZ[i, 0]
                            vel_int_z[i] = VV[i, 0]
                            u_int_z[i] = uu[i, 0]
                            q_int_z1[i] = qq[i, 0] * h_z[i]
                            q_int_z2[i] = qq[i, 0] * z_z[i]
                            p_int_z1[i] = uu[i, 0] * h_z[i] * h_z[i]
                            p_int_z2[i] = uu[i, 0] * z_z[i]
                        # print(z_z[i])
                        break
    '''
'''
alt_module = 'off'
if alt_module == 'on':
    if max(lev) < lev_set:  #
        pe_flux, ke_flux = 0, 0
    else:
        ylin = np.linspace(min(y), max(y), ypt)
        zlin = np.linspace(min(z), max(z), zpt)
        y_2d, z_2d = np.meshgrid(ylin, zlin)
        lev_str = griddata((y, z), lev, (y_2d, z_2d), method='linear', fill_value=0.0)
        v_mag = griddata((y, z), V, (y_2d, z_2d), method='linear', fill_value=0.0)
        uu = griddata((y, z), u, (y_2d, z_2d), method='linear', fill_value=0.0)
        for jj in range(ypt):
            for kk in range(zpt):
                if lev_str[kk, jj] < 0:
                    uu[kk, jj] = 0
                    v_mag[kk, jj] = 0
        kf = np.zeros((zpt, ypt))
        pf = np.zeros((zpt, ypt))
        # print(z_2d[:, 0])
        kf_int_z, pf_int_z = np.zeros(ypt), np.zeros(ypt)

        for kz in range(zpt):
            for jy in range(ypt):
                kf[kz, jy] = 0.5 * (uu[kz, jy]) * (v_mag[kz, jy]) ** 2
                pf[kz, jy] = g * z_2d[kz, jy] * uu[kz, jy]

        for jj in range(ypt):
            kf_int_z[jj] = integrate.simps(kf[:, jj], zlin)
            pf_int_z[jj] = integrate.simps(pf[:, jj], zlin)

        ke_flux = integrate.simps(kf_int_z, ylin)
        pe_flux = integrate.simps(pf_int_z, ylin)


elif alt_module == 'off':
    # print(ypt, zpt)
    if max(lev) < lev_set:  #
        pe_flux, ke_flux = 0, 0
    else:
        # Filtering the data as per the level set, LEV>=0 for water
        znew = z[lev >= lev_set]
        # print(max(znew))
        ylin = np.linspace(min(y), max(y), ypt)
        zlin = np.linspace(min(z), max(znew), zpt)
        y_2d, z_2d = np.meshgrid(ylin, zlin)
        lev_str = griddata((y, z), lev, (y_2d, z_2d), method='linear', fill_value=0.0)
        v_mag = griddata((y, z), V, (y_2d, z_2d), method='linear', fill_value=0.0)
        uu = griddata((y, z), u, (y_2d, z_2d), method='linear', fill_value=0.0)
        kf = np.zeros((zpt, ypt))
        pf = np.zeros((zpt, ypt))
        # print(z_2d[:, 0])
        kf_int_z, pf_int_z = np.zeros(ypt), np.zeros(ypt)

        for kz in range(zpt):
            for jy in range(ypt):
                kf[kz, jy] = 0.5 * (uu[kz, jy]) * (v_mag[kz, jy]) ** 2
                pf[kz, jy] = g * z_2d[kz, jy] * uu[kz, jy]

        for jj in range(ypt):
            kf_int_z[jj] = integrate.simps(kf[:, jj], zlin)
            pf_int_z[jj] = integrate.simps(pf[:, jj], zlin)

        ke_flux = integrate.simps(kf_int_z, ylin) / (max(ylin) - min(ylin))
        pe_flux = integrate.simps(pf_int_z, ylin) / (max(ylin) - min(ylin))
tot_energy_flux = pe_flux + ke_flux

# print(ke_flux)
energy_flux_data = pe_flux, ke_flux, tot_energy_flux
'''
'''
slice_x = 0
while slice_x == 0:
    X, Y, Z, X1, Y1, Z1 = regular_grid(x, y, z, xpt, ypt, zpt)
    for isx in range(len(X)):
        if abs(X[isx] - slice_loc) <= 1e-2:
            slice_x = X[isx]
            break
    else:
        xpt = xpt + 100
print("The Slice will be taken at x = ", np.round(slice_x, 2))

# print(y, lev)

if slice == 'y':
if i == 0:
    X, Y, Z, X1, Y1, Z1 = regular_grid(x, y, z, xpt, ypt, zpt)
    for j in range(len(Y)):
        if abs(Y[j] - slice_loc) <= 1e-2:
            # print(X[i])
            slice_y = Y[j]
            break
        elif abs(Y[j] - slice_loc) <= 1e-1:
            slice_y = Y[j]
            break
    print("The Slice will be taken at y = ", np.round(slice_y, 2))
    
        if vector_plot == 'yes':
        XX = np.arange(len(X1) * len(Z1), dtype=float)
        ZZ = np.arange(len(X1) * len(Z1), dtype=float)
        VV = np.arange(len(X1) * len(Z1), dtype=float)
        uu = np.arange(len(X1) * len(Z1), dtype=float)
        vv = np.arange(len(X1) * len(Z1), dtype=float)
        ww = np.arange(len(X1) * len(Z1), dtype=float)
        hh = np.arange(len(X1) * len(Z1), dtype=float)
        qq = np.arange(len(X1) * len(Z1), dtype=float)
        # interpolate "data" on new grid "X,Y,Z"
        vel = griddata((x, y, z), V, (X, Y, Z), method='linear', fill_value=0.0)
        u = griddata((x, y, z), u, (X, Y, Z), method='linear', fill_value=0.0)
        v = griddata((x, y, z), v, (X, Y, Z), method='linear', fill_value=0.0)
        w = griddata((x, y, z), w, (X, Y, Z), method='linear', fill_value=0.0)
        i = 0
        for j in range(len(Y)):
            if Y[j] == slice_y:
                XX[i] = X[j]
                ZZ[i] = Z[j]
                VV[i] = vel[j]
                hh[i] = h[j]
                uu[i] = u[j]
                ww[i] = w[j]
                vv[i] = v[j]
                qq[i] = 0.5 * u[j] * vel[j] * vel[j]
                i = i + 1

        XX = np.reshape(XX, (len(X1), len(Z1)))
        ZZ = np.reshape(ZZ, (len(X1), len(Z1)))
        VV = np.reshape(VV, (len(X1), len(Z1)))
        uu = np.reshape(uu, (len(X1), len(Z1)))
        vv = np.reshape(vv, (len(X1), len(Z1)))
        ww = np.reshape(ww, (len(X1), len(Z1)))
        hh = np.reshape(hh, (len(X1), len(Z1)))
        qq = np.reshape(qq, (len(X1), len(Z1)))

        fig, ax1 = plt.subplots(figsize=(11, 7), dpi=90)
        ax1.quiver(XX, ZZ, uu, ww)
        # levels = np.linspace(VV.min(), VV.max(), 256)
        # cf = ax1.contourf(XX, ZZ, VV, levels, cmap='viridis', extend='both')

        ax1.set_aspect('equal')
        # fig.colorbar(cf, extendrect=True, extendfrac=0.0, fraction=0.0285, pad=0.05, label=r'$U/U_{\infty}$')

        plt.ylabel(r'$z$')
        plt.xlabel(r'$x$')
        plt.title(r'y = %.1f' % (np.round(slice_y, 2)))
        plt.show()
        
            if dam_break == 'yes':
        hh = np.arange(len(X1) * len(Z1), dtype=float)
        XX = np.arange(len(X1) * len(Z1), dtype=float)
        ZZ = np.arange(len(X1) * len(Z1), dtype=float)
        # interpolate "data" on new grid "X,Y,Z"
        h = griddata((x, y, z), lev, (X, Y, Z), method='linear', fill_value=0.0)
        itr = 0
        for j in range(len(Y)):
            if Y[j] == slice_y:
                XX[itr] = X[j]
                ZZ[itr] = Z[j]
                hh[itr] = h[j]
                itr = itr + 1
        XX = np.reshape(XX, (len(X1), len(Z1)))
        ZZ = np.reshape(ZZ, (len(X1), len(Z1)))
        hh = np.reshape(hh, (len(X1), len(Z1)))

        for ix in range(len(X1)):
            H_nd = 0
            for j in range(len(Z1)):
                if hh[len(X1) - 1 - ix, j] > 0:
                    H_nd = XX[len(X1) - 1 - ix, j] / d_l
                    break
            if H_nd > 0:
                break
        energy_data = H_nd
'''
'''
alt_module = 'on'
if alt_module == 'on':
    for ix in range(xpt):
        for jy in range(ypt):
            for kz in range(zpt):
                if m3d[ix, jy, zpt - 1 - kz] == 1:
                    lindex = zpt - kz
                    # print('lindex', lindex)
                    # print('shape', np.shape(m3d))
                    # print('x,y', ix, jy)
                    zz = np.linspace(z_3d[ix, jy, 0], z_3d[ix, jy, lindex], zpt)
                    z3d = z_3d[ix, jy, :][z_3d[ix, jy, :] <= max(zz)]
                    z_bot = z_3d[ix, jy, :][m3d[ix, jy, :] == 1]
                    levz = lev_3d[ix, jy, :][z_3d[ix, jy, :] <= max(zz)]

                    q = qq_3d[ix, jy, :][z_3d[ix, jy, :] <= max(zz)]
                    flev, fq = interpolate.interp1d(z3d, levz), interpolate.interp1d(z3d, q)
                    lev_new, q_new = flev(zz), fq(zz)
                    for kk in range(zpt):
                        # print('kk', kk)
                        if lev_new[zpt - 1 - kk] >= -1e-5:
                            kindex = zpt - 1 - kk
                            break

                    depth_2d[ix, jy] = zz[kindex] - min(z_bot)
                    if depth_2d[ix, jy] == 0:
                        print('depth, kindex, q', zz[kindex], q_new[kindex], kindex, ix, jy)
                        q_2d_z[ix, jy] = 0
                    else:
                        q_new, zz = q_new[0:kindex + 1], zz[0:kindex + 1]
                        q_2d_z[ix, jy] = integrate.simps(q_new, zz) / (depth_2d[ix, jy])
                        con_2d[ix, jy] = 1
                    break

else:
    levpt = 50
    for ix in range(xpt):
        for jy in range(ypt):
            for kz in range(zpt):
                if m3d[ix, jy, zpt - 1 - kz] == 1:
                    lindex = zpt - 1 - kz
                    zz = np.linspace(z_3d[ix, jy, lindex], z_3d[ix, jy, lindex + 1], levpt)
                    zlev = z_3d[ix, jy, lindex:lindex + 2]
                    levn = lev_3d[ix, jy, lindex:lindex + 2]
                    qq = qq_3d[ix, jy, lindex:lindex + 2]
                    f = interpolate.interp1d(zlev, levn)
                    levz = f(zz)
                    fq = interpolate.interp1d(zlev, qq)
                    qqz = fq(zz)
                    for kk in range(levpt):
                        if levz[kk] <= 1e-4:
                            kindex = kk
                            # print(levz[kk], zz[kk], x_3d[ix, 0, 0], y_3d[0, jy, 0])
                            break

                    depth_2d[ix, jy] = zz[kindex]
                    print(np.size(qqz[0: kindex + 1]), np.size(zz[0: kindex + 1]))
                    qq_int[ix, jy] = integrate.simps(qqz[0: kindex + 1], zz[0: kindex + 1]) / (
                            zz[kindex + 1] - zz[0])
                    break
    for ix in range(xpt):
        for jy in range(ypt):
            if depth_2d[ix, jy] == 0:
                q_2d_z[ix, jy], con_2d[ix, jy] = 0, 0
            else:
                q_2d_z[ix, jy] = (integrate.simps(q_3d[ix, jy, :], z_3d[ix, jy, :]) / (depth_2d[ix, jy])) + \
                                 qq_int[ix, jy]
                con_2d[ix, jy] = 1
'''
