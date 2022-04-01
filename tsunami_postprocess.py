#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from datetime import datetime

# import time
import numpy as np
from pyevtk.hl import gridToVTK
from data_process import data_read
from data_process import file_out


# parameters


def Write_file(fname, data, case_path, dime=1):
    ndata = len(data)
    # print("dime:",dime)
    fdata = open(os.path.join(case_path, fname), "a+")
    for i in range(ndata):
        line = data[i]
        print(i + 1, end="\t", file=fdata)
        if dime > 1:
            for j in range(dime):
                print(format(line[j], "5.8f"), end=" \t", file=fdata)
        else:
            print(line, end=" ", file=fdata)
        print(file=fdata)


# from energy_flux import eflux
from time_input import t_input
import input

data_input = input.flux_input()
print('***************************************************************************************************************')
print('Welcome to Tsunami postprocess code')
print('The code is created by Abhishek Mukherjee')
print(datetime.now())
print('The case directory is in \n', data_input.data_directory)
print('The status of the following modules are')
print('Velocity depth average module is \n', data_input.vel_field_avg)
print('The turbulent statistics module is \n', data_input.turbulent_energy)
print('The spanwise averaged water depth module is \n', data_input.water_depth)
print('The energy flux module is \n', data_input.energy_flux[0])
print('The wall-shear stress module is \n', data_input.wall_shear_stress)
print('The dam front location module is \n', data_input.dam_front)
print('The velocity profile module is \n', data_input.vel_profile[0])
print('The side wall effects are \n', data_input.side_wall[0])
print('***************************************************************************************************************')

T = np.arange(data_input.t_ini, data_input.t_fin + data_input.dT, data_input.dT)
# import data_read
if data_input.transient_data[0] == 'on':
    ts, tf = data_input.transient_data[1], data_input.transient_data[2]
    print('The parameters will be calculated between t=', ts, 'and t=', tf)
    for i in range(len(T)):
        if ts <= T[i] <= tf:
            fin = data_input.filename + str(i) + '.csv'
            print("The source data file name is", fin)
            #timeskipdata = 3.6
            if data_input.energy_flux == 'on':
                from tsunami import eflux
                case_output = 'energy_flux'
                param = ["Kflux:0", "Kflux:1", "Kflux:2", "Pflux:0", "Pflux:1", "Pflux:2", "Tflux:0", "Tflux:1",
                         "Tflux:2", "x"]
                fout, case_path = file_out(case_output, data_input, "eflux", T, i, param, data_input.time_write_shift)
                data_obj = data_read.data_3d(data_input.xpt, data_input.ypt, data_input.zpt, data_input.data_directory,
                                             fin, data_input.data_type)
                points_3d, vel_3d = data_obj.point_3d(), data_obj.vel_3d()
                lev_3d, mask_3d = data_obj.lev_3d()
                energy_flux_data = eflux(data_input.xpt, data_input.ypt, data_input.zpt, lev_3d,
                                         mask_3d, points_3d[1, :, :, :], points_3d[2, :, :, :], vel_3d)

                # File parameters: potential flux, kinetic, flux, total flux, x-coordinates
                Write_file(fout, np.vstack((energy_flux_data, points_3d[0, :, 0, 0])).T, case_path,
                           np.shape(energy_flux_data)[0] + 1)
                print('The file writing ends at time =', T[i])
                print('energy flux data files written in ', case_path)
            print('********************************************************************************')

            if data_input.flux_dissipation == 'on':
                from tsunami import flux_dissipation
                case_output = 'flux_dissipation'
                param = ["dflux_KE", "dflux_PE", "dflux_TE", "x"]
                fout, case_path = file_out(case_output, data_input, "dis_flux", T, i, param, data_input.time_write_shift)
                try:
                    flux_dis = flux_dissipation(data_input.xpt, data_input.ypt, data_input.zpt, lev_3d, mask_3d,
                                  points_3d[1, :, :, :],points_3d[2, :, :, :], vel_3d)
                except:
                    print("*** Calling the parameters from the data read module ***")
                    data_obj = data_read.data_3d(data_input.xpt, data_input.ypt, data_input.zpt,
                                                 data_input.data_directory, fin, data_input.data_type)
                    points_3d, vel_3d = data_obj.point_3d(), data_obj.vel_3d()
                    lev_3d, mask_3d = data_obj.lev_3d()
                    flux_dis = flux_dissipation(data_input.xpt, data_input.ypt, data_input.zpt, lev_3d, mask_3d,
                                                points_3d[1, :, :, :], points_3d[2, :, :, :], vel_3d)
                # File parameters: potential flux, kinetic, flux, total flux, x-coordinates
                Write_file(fout, np.vstack((flux_dis, points_3d[0, :, 0, 0])).T, case_path, 4)
                print('The file writing ends at time =', T[i])
                print('Energy Flux dissipation data files written in ', case_path)
                print('********************************************************************************')

            if data_input.epsilon == 'on':
                from tsunami import epsilon
                case_output = 'dissipation_rate'
                param = ["epsilon", "x"]
                fout, case_path = file_out(case_output, data_input, "epsion", T, i, param, data_input.time_write_shift)
                try:
                    eps = epsilon(data_input.xpt, data_input.ypt, data_input.zpt, lev_3d, mask_3d,points_3d[0, :, :, :],
                                  points_3d[1, :, :, :],points_3d[2, :, :, :], vel_3d)
                except:
                    print("*** Calling the parameters from the data read module ***")
                    data_obj = data_read.data_3d(data_input.xpt, data_input.ypt, data_input.zpt,
                                                 data_input.data_directory, fin, data_input.data_type)
                    points_3d, vel_3d = data_obj.point_3d(), data_obj.vel_3d()
                    lev_3d, mask_3d = data_obj.lev_3d()
                    eps = epsilon(data_input.xpt, data_input.ypt, data_input.zpt, lev_3d, mask_3d,points_3d[0, :, :, :],
                                  points_3d[1, :, :, :],points_3d[2, :, :, :], vel_3d)
                # File parameters: potential flux, kinetic, flux, total flux, x-coordinates
                Write_file(fout, np.vstack((eps, points_3d[0, :, 0, 0])).T, case_path, 2)
                print('The file writing ends at time =', T[i])
                print('Energy dissipation rate data files written in ', case_path)
                print('********************************************************************************')

            if data_input.total_energy == 'on':
                from tsunami import te
                case_output = 'total_energy'
                param = ["pe", "ke", "te", "v_avg", "x"]
                fout, case_path = file_out(case_output, data_input, "te", T, i, param, data_input.time_write_shift)
                try:
                    energy_data = te(data_input.xpt, data_input.ypt, data_input.zpt, lev_3d, mask_3d,
                                     points_3d[2, :, :, :], vel_3d)
                except:
                    print("*** Calling the parameters from the data read module ***")
                    data_obj = data_read.data_3d(data_input.xpt, data_input.ypt, data_input.zpt,
                                                 data_input.data_directory,
                                                 fin, data_input.data_type)
                    points_3d, vel_3d = data_obj.point_3d(), data_obj.vel_3d()
                    lev_3d, mask_3d = data_obj.lev_3d()
                    energy_data = te(data_input.xpt, data_input.ypt, data_input.zpt, lev_3d, mask_3d,
                                     points_3d[2, :, :, :], vel_3d)
                # File parameters: potential flux, kinetic, flux, total flux, x-coordinates
                Write_file(fout, np.vstack((energy_data, points_3d[0, :, 0, 0])).T, case_path, 5)
                print('The file writing ends at time =', T[i])
                print('Total energy data files written in ', case_path)
                print('********************************************************************************')
multiple_data = 'off'
# set time steps
# dT = 0.25
# t_ini, t_fin = 0, 10


if data_input.vel_field_avg == 'on':
    if os.path.exists(case_type):
        # path, cc = os.path.split(os.getcwd())
        # path = directory + '/' + case_type
        dir = 'vts'
        vft = t_final  # 2.0
        tind = np.where(np.round(T, 2) == vft)
        case, types = os.path.split(case_type)
        print('depth average calculation will be done at t =', vft)
        # try:
        energy_flux_data, total_energy_data, energy_theory = eflux(tind[0][0])
        if not os.path.exists(case_type + '/' + dir):
            os.makedirs(case_type + '/' + dir)
        x3d, y3d, z3d, v_z, con_z, lev_3d, mask_3d, depth_2d, v_y, con_y, v_mag_3d = energy_flux_data
        gridToVTK(os.path.join(case_type, dir, 'depth_av_vel' + '_' + 't' + '_' + str(vft) + '_' + types), x3d, y3d,
                  z3d,
                  pointData={"VELOCZ": v_z, "CONZ": con_z, "VELOCY": v_y, "CONY": con_y, "LEVEL": lev_3d,
                             "Mask3D": mask_3d, "depth2d": depth_2d, "VELOC3d": v_mag_3d})  # "./depth_av_vel"
        print('depth average velocity file written in ', case_type)
        print(
            '***********************************************************************************************************')
    else:
        sys.exit("The case directory does not exist, check the input file")

if data_input.turbulent_energy == 'on':
    if os.path.exists(case_type):
        # path = directory + '/' + case_type
        dir = 'vts'
        vft_turb = t_final  # 2.0
        tind = np.where(np.round(T, 2) == vft_turb)
        case, types = os.path.split(case_type)
        if vel_field_avg == 'on' and vft_turb == vft:
            print('turbulent statistics calculation will be done at t =', vft_turb)
            print('Skip the recalculation of turbulent statistics')
            x_3d, y_3d, z_3d, lev_3d, c_3d, tke_3d, r_stress, tke_y, r_stress_y, \
            tke_z, r_stress_z, depth_2d, con_y, con_z = total_energy_data

            gridToVTK(os.path.join(case_type, dir, 'turbulent_energy' + '_' + 't' + '_' + str(vft) + '_' + types), x_3d,
                      y_3d,
                      z_3d,
                      pointData={"LEVEL": lev_3d, "CON3d": c_3d, "TKE3d": tke_3d, "RUU_3d": r_stress[0, :, :, :],
                                 "RVV_3d": r_stress[1, :, :, :], "RWW_3d": r_stress[2, :, :, :],
                                 "RUV_3d": r_stress[3, :, :, :], "RUW_3d": r_stress[4, :, :, :],
                                 "RVW_3d": r_stress[5, :, :, :], "TKEY": tke_y,
                                 "RUUY": r_stress_y[0, :, :, :], "RVVY": r_stress_y[1, :, :, :],
                                 "RWWY": r_stress_y[2, :, :, :], "RUVY": r_stress_y[3, :, :, :],
                                 "RUWY": r_stress_y[4, :, :, :], "RVWY": r_stress_y[5, :, :, :],
                                 "TKEZ": tke_z, "RUUZ": r_stress_z[0, :, :, :],
                                 "RVVZ": r_stress_z[1, :, :, :], "RWWZ": r_stress_z[2, :, :, :],
                                 "RUVZ": r_stress_z[3, :, :, :], "RUWZ": r_stress_z[4, :, :, :],
                                 "RVWZ": r_stress_z[5, :, :, :], "depth_2d": depth_2d, "CONY": con_y, "CONZ": con_z})
        elif turbulent_average == 'on':
            print('Turbulent average module is on')

            tt = [0.8, 1.2, 1.6, 1.8]  # 0.8, 1.2, 1.6,
            # path = directory + '/' + case_type
            dir = 'turbulent_data'
            if not os.path.exists(case_type + '/' + dir):
                os.makedirs(case_type + '/' + dir)
            l = 0
            case_path = case_type + '/' + dir
            case, types = os.path.split(case_type)
            for i in range(len(T)):
                if l == np.size(tt):
                    print('l', l)
                    break
                elif T[i] == tt[l]:
                    fname = 'turbulent' + '_' + str(T[i]) + '.dat'
                    energy_flux_data, total_energy_data, energy_theory = eflux(i)
                    tke1d = energy_theory[0]
                    Write_file(fname, tke1d, case_path, 2)
                    print('The current time is =', T[i])
                    l = l + 1

            print('turbulent average files written in ', case_path)
            print(
                '***********************************************************************************************************')

        elif shear_data == 'on':
            print('shear data module is on')
            # path = directory + '/' + case_type
            dir = 'vts'
            vft = t_final  # 2.0
            tind = np.where(T == vft)
            case, types = os.path.split(case_type)
            print('turbulent statistics will be calculated at t =', vft)
            # try:
            energy_flux_data, total_energy_data, energy_theory = eflux(tind[0][0])
            if not os.path.exists(case_type + '/' + dir):
                os.makedirs(case_type + '/' + dir)
            # x3d, y3d, z3d, tau_xy, tau_xz, tau_yz, con_z, tau_xy_y, tau_xz_y, tau_yz_y, \
            # con_y, tau_xy_z, tau_xz_z, tau_yz_z, lev_3d, uprime, vprime, wprime, mask_3d, c_3d = total_energy_data
            x3d, y3d, z3d, lev_3d, mask_3d, c_3d, tp, uy, vy, wy, u, v, w, \
            K_span, K_in = total_energy_data
            # print(x3d.max(),x3d.min())

            '''
            gridToVTK(os.path.join(case_type, dir, 'shear_stress' + '_' + 't' + '_' + str(vft) + '_' + types), x3d, y3d,
                      z3d,
                      pointData={"TAUX": tau_xy, "TAUY": tau_yz, "TAUZ": tau_xz, "CONZ": con_z, "LEVEL": lev_3d,
                                 "Mask3D": mask_3d, "TAUX_Y": tau_xy_y, "TAUY_Y": tau_yz_y, "TAUZ_Y": tau_xz_y,
                                 "CONY": con_y, "TAUX_Z": tau_xy_z, "TAUY_Z": tau_yz_z, "TAUZ_Z": tau_xz_z,
                                 "c_3d": c_3d, "UPRIME": uprime, "VPRIME": vprime, "WPRIME": wprime, "K":K})
            '''
            gridToVTK(os.path.join(case_type, dir, 'turbulence_stat' + '_' + 't' + '_' + str(vft) + '_' + types),
                      x3d,
                      y3d,
                      z3d,
                      pointData={"Turbul": tp, "LEVEL": lev_3d,
                                 "Mask3D": mask_3d, "c_3d": c_3d,
                                 "UY": uy, "VY": vy, "WY": wy, "U": u, "V": v, "W": w,
                                 "K_Y": K_span, "K_in": K_in})
            print('turbulent statistics file written in ', case_type)
            print(
                '***********************************************************************************************************')

        else:
            print('Only turbulent module is on')
            print('turbulent statistics calculation will be done at t =', vft_turb)
            energy_flux_data, total_energy_data, energy_theory = eflux(tind[0][0])
            if not os.path.exists(case_type + '/' + dir):
                os.makedirs(case_type + '/' + dir)
            x_3d, y_3d, z_3d, lev_3d, c_3d, tke_3d, r_stress, tke_y, r_stress_y, \
            tke_z, r_stress_z, depth_2d, con_y, con_z, v_mag_3d = total_energy_data

            gridToVTK(os.path.join(case_type, dir, 'turbulent_energy' + '_' + 't' + '_' + str(vft_turb) + '_' + types),
                      x_3d,
                      y_3d,
                      z_3d,
                      pointData={"LEVEL": lev_3d, "CON3d": c_3d, "TKE3d": tke_3d, "RUU_3d": r_stress[0, :, :, :],
                                 "RVV_3d": r_stress[1, :, :, :], "RWW_3d": r_stress[2, :, :, :],
                                 "RUV_3d": r_stress[3, :, :, :], "RUW_3d": r_stress[4, :, :, :],
                                 "RVW_3d": r_stress[5, :, :, :], "TKEY": tke_y,
                                 "RUUY": r_stress_y[0, :, :, :], "RVVY": r_stress_y[1, :, :, :],
                                 "RWWY": r_stress_y[2, :, :, :], "RUVY": r_stress_y[3, :, :, :],
                                 "RUWY": r_stress_y[4, :, :, :], "RVWY": r_stress_y[5, :, :, :],
                                 "TKEZ": tke_z, "RUUZ": r_stress_z[0, :, :, :],
                                 "RVVZ": r_stress_z[1, :, :, :], "RWWZ": r_stress_z[2, :, :, :],
                                 "RUVZ": r_stress_z[3, :, :, :], "RUWZ": r_stress_z[4, :, :, :], "V_mag": v_mag_3d,
                                 "RVWZ": r_stress_z[5, :, :, :], "depth_2d": depth_2d, "CONY": con_y, "CONZ": con_z})
            # except:
            # print('File location error!!!!! Check the directory location defined in the input file, flies are not found')

        print('turbulent statistics file written in ', case_type)
        print(
            '***********************************************************************************************************')
    else:
        sys.exit("The case directory does not exist, check the input file")

if data_input.water_depth == 'on':
    dir = 'depth_data'
    if os.path.exists(case_type):
        if not os.path.exists(case_type + '/' + dir):
            os.makedirs(case_type + '/' + dir)
        l = 0
        case_path = case_type + '/' + dir
        case, types = os.path.split(case_type)
        for i in range(len(T)):
            print('time', np.round(T[i], 2))
            if T[i] >= 0:
                fname = 'depth' + '_' + str(T[i]) + '.dat'
                energy_flux_data, total_energy_data, energy_theory = eflux(i)
                depth1d = energy_theory[0]
                Write_file(fname, depth1d, case_path, 3)
                print('The current time is =', T[i])
                l = l + 1
            if i == 0:
                cs = case_type.replace('/', '_')
                with open(os.path.join(case_path, 'depth1d' + '_' + str(
                        slice_loc) + '.txt'), 'a+') as fn:
                    fn.write("depth    velocity     time   \n")
                fn.close()
            for ipt in range(len(depth1d[:, 2])):
                if depth1d[ipt, 2] == slice_loc:
                    with open(os.path.join(case_path, 'depth1d' + '_' + str(
                            slice_loc) + '.txt'), 'a+') as fn:
                        fn.write(
                            "%.5f %.5f %.3f \n" % (float(depth1d[ipt, 0]), float(depth1d[ipt, 1]), T[i]))
                print("The output file written line for depth =", i + 1)
        print("The time is =", T[i])

        fn.close()
        print('water depth files written in ', case_path)
        print(
            '***********************************************************************************************************')
    else:
        sys.exit("The case directory does not exist, check the input file")

if data_input.vel_profile[0] == 'on':
    if os.path.exists(case_type):
        tt = [0.8, 1.2, 1.6, 1.8]
        # path = directory + '/' + case_type
        dir = 'vel_profile_data'
        if not os.path.exists(case_type + '/' + dir):
            os.makedirs(case_type + '/' + dir)
        l = 0
        case_path = case_type + '/' + dir
        case, types = os.path.split(case_type)
        for i in range(len(T)):
            if l == np.size(tt):
                print('l', l)
                break
            print('time', np.round(T[i], 2))
            if np.round(T[i], 2) == tt[l]:
                fname = 'vel_profile' + '_' + 'x_' + str(vel_profile[1]) + '_t_' + str(T[i]) + '.dat'
                energy_flux_data, total_energy_data, energy_theory = eflux(i)
                vel_z = energy_theory[0]
                # print('size', np.shape(vel_z))
                Write_file(fname, vel_z, case_path, 5)
                print('The current time is =', T[i])
                l = l + 1
        print('velocity profile data files written in ', case_path)
        print(
            '***********************************************************************************************************')
    else:
        sys.exit("The case directory does not exist, check the input file")

if data_input.wall_shear_stress == 'on':
    if os.path.exists(case_type):
        tt = np.round(np.arange(0, 10, 1), 2)
        # path = directory + '/' + case_type
        dir = 'wall_shear_data'
        if not os.path.exists(case_type + '/' + dir):
            os.makedirs(case_type + '/' + dir)
        l = 0
        case_path = case_type + '/' + dir
        case, types = os.path.split(case_type)
        for i in range(len(T)):
            if T[i] >= tt[0]:
                fname = 'shear_sp' + '_' + str(T[i]) + '.dat'
                energy_flux_data, total_energy_data, energy_theory = eflux(i)
                wall_shear = energy_theory[0]
                Write_file(fname, wall_shear, case_path, 2)
                print('The current time is =', T[i])
                l = l + 1
        print('wall shear stress files written in ', case_path)
        print(
            '***********************************************************************************************************')
    else:
        sys.exit("The case directory does not exist, check the input file")

if data_input.dam_front == 'on':
    if os.path.exists(case_type):
        tt = np.round(np.arange(0, 10, 1), 2)
        # path = directory + '/' + case_type
        dir = 'dam_front_data'
        if not os.path.exists(case_type + '/' + dir):
            os.makedirs(case_type + '/' + dir)
        l = 0
        case_path = case_type + '/' + dir
        case, types = os.path.split(case_type)
        for i in range(len(T)):
            if T[i] >= tt[0]:
                fname = 'dam_front_sp' + '_' + str(T[i]) + '.dat'
                energy_flux_data, total_energy_data, energy_theory = eflux(i)
                dam_s = energy_theory
                Write_file(fname, dam_s, case_path, 1)
                print('The current time is =', T[i])
                l = l + 1
        print('dam front location files written in ', case_path)
        print(
            '***********************************************************************************************************')
    else:
        sys.exit("The case cirectory does not exist, check the input file")

'''
if multiple_data == 'on':
    for i in range(len(T)):
        energy_flux_data, total_energy_data, energy_theory = eflux(i)

        if energy_flux == 'yes':
            path, cc = os.path.split(os.getcwd())
            if i == 0:
                cs = case_type.replace('/', '_')
                with open(os.path.join(path, 'energy_flux' + '_' + cs + mod + str(
                        slice_loc) + '.txt'), 'a+') as fn:
                    fn.write("pot_flux     ke_flux     tot_flux     time   \n")
                fn.close()

            pot_flux, kin_flux, tot_energy_flux = energy_flux_data
            # print(pot_flux, kin_flux, energy_flux_data,T[i])#, path)

            with open(os.path.join(path, 'energy_flux' + '_' + cs + mod + str(
                    slice_loc) + '.txt'), 'a+') as fn:
                fn.write("%.5f %.5f %.5f %.3f \n" % (float(pot_flux), float(kin_flux), float(tot_energy_flux), T[i]))
            print("The output file written line for energy flux=", i + 1)
            print("The time is =", T[i])

            fn.close()

        if dam_break == 'yes':
            if i == 0:
                with open('dam_break' + '_' + directory + '_' + '.txt', 'a+') as fn:
                    fn.write("H_nd   t_nd  \n")
                fn.close()

            H = eflux(i)
            # print('H',H)
            with open('dam_break' + '_' + directory + '_' + '.txt', 'a+') as fn:
                fn.write("%.4f %.4f \n" % (
                    float(H), T[i] * np.sqrt(9.8 / d_l)))
            print("The output file written line=", i + 1)
            print("The time is =", ti[i])
            fn.close()

        if total_energy == 'yes':
            if i == 0:
                with open('total_energy.txt', 'a+') as fn:
                    fn.write("TKE   TPE  TE  time\n")
                fn.close()

            Ekin, Epot, Etot = total_energy_data
            print('total energy', Ekin)
            with open('total_energy.txt', 'a+') as fn:
                fn.write("%.5f %.5f %.5f %.2f \n" % (float(Ekin), float(Epot), float(Etot), T[i]))
            fn.close()
            print("The output file written line for total energy=", i + 1)
            print("The time is =", ti[i])
            fn.close()

        if dam_theory == 'yes':
            if i == 0:
                Ekt, Ept, Et, t_theo = energy_theory
                with open('dam_theory.txt', 'a+') as ft:
                    ft.write("TKE   TPE  TE  time\n")
                    for iw in range(len(t_theo)):
                        ft.write("%.5f %.5f %.5f %.2f \n" % (float(Ekt[iw]), float(Ept[iw]), float(Et[iw]), t_theo[iw]))
                ft.close()
        if total_energy_paraview == 'yes':
            path, casename = os.path.split(os.getcwd())
            if i == 0:
                with open('total_energy' + '_' + casename + '_' + directory + '.txt', 'a+') as fn:
                    fn.write("pot     ike    tot_energy   tke     time   \n")
                fn.close()

            pe, ike, tke, tot_energy = total_energy_data

            with open('total_energy' + '_' + casename + '_' + directory + '.txt', 'a+') as fn:
                fn.write("%.5f %.5f %.5f %.5f %.3f \n" % (float(pe), float(ike), float(tot_energy), float(tke), T[i]))
            print("The output file written line for total energy paraview =", i + 1)
            print("The time is =", T[i])
        if energy_flux_paraview == 'yes':
            path, cc = os.path.split(os.getcwd())
            # casename = 'flex_single_cyl_Y2'
            if i == 0:
                with open(os.path.join(path, 'energy_flux' + '_' + casename + '_' + str(slice_loc) + '.txt'),
                          'a+') as fn:
                    fn.write("pot_flux     ke_flux     tot_flux     time   \n")
                fn.close()

            pot_flux, kin_flux, tot_energy_flux = energy_flux_data
            # print('pot_flux', pot_flux)
            with open(os.path.join(path, 'energy_flux' + '_' + casename + '_' + str(slice_loc) + '.txt'),
                      'a+') as fn:
                fn.write("%.5f %.5f %.5f %.3f \n" % (float(pot_flux), float(kin_flux), float(tot_energy_flux), T[i]))
            print("The output file written line for energy flux paraview =", i + 1)
            print("The time is =", T[i])

        if water_depth == 'yes':
            path, cc = os.path.split(os.getcwd())
            # casename = 'flex_single_cyl_Y2'
            if i == 0:
                cs = case_type.replace('/', '_')
                with open(os.path.join(path, 'water_depth' + '_' + cs + mod + str(slice_loc) + '.txt'),
                          'a+') as fn:
                    fn.write("h (m)  u_avg (m/s)    k_TE_avg    vel_avg (m/s)    time   \n")
                fn.close()
            # print(energy_flux_data)
            hh, u_avg, k_avg, v_avg = energy_flux_data
            # print('pot_flux', pot_flux)
            with open(os.path.join(path, 'water_depth' + '_' + cs + mod + str(slice_loc) + '.txt'),
                      'a+') as fn:
                fn.write(
                    "%.5f  %.5f  %.5f  %.5f  %.3f \n" % (float(hh), float(u_avg), float(k_avg), float(v_avg), T[i]))
            print("The output file written line for water depth =", i + 1)
            print("The time is =", T[i])
        if probe_data == 'on':
            path, cc = os.path.split(os.getcwd())
            # casename = 'flex_single_cyl_Y2'
            if i == 0:
                cs = case_type.replace('/', '_')
                with open(os.path.join(path, 'probe_data' + '_' + cs + mod + str(slice_loc) + '.txt'),
                          'a+') as fn:
                    fn.write("u_x(m/s)  u_z(m/s)    u_avx(m/s)    u_avz(m/s)    v_mag(m/s)    lev     time   \n")
                fn.close()
            # print(energy_flux_data)
            u_x, u_z, u_avx, u_avz, v_mag, lev = energy_flux_data
            # print('pot_flux', pot_flux)
            with open(os.path.join(path, 'probe_data' + '_' + cs + mod + str(slice_loc) + '.txt'),
                      'a+') as fn:
                fn.write("%.5f  %.5f  %.5f  %.5f  %.5f  %.5f  %.3f \n" % (
                    float(u_x), float(u_z), float(u_avx), float(u_avz), float(v_mag), float(lev), T[i]))
            print("The output file written line for water depth =", i + 1)
            print("The time is =", T[i])

        if force == 'on':
            path, cc = os.path.split(os.getcwd())
            # casename = 'flex_single_cyl_Y2'
            if i == 0:
                cs = case_type.replace('/', '_')
                with open(os.path.join(path, 'force' + '_' + cs + mod + str(slice_loc) + '.txt'),
                          'a+') as fn:
                    fn.write("F (N)  time   \n")
                fn.close()
            # print(energy_flux_data)
            F_px = total_energy_data
            # print('pot_flux', pot_flux)
            with open(os.path.join(path, 'force' + '_' + cs + mod + str(slice_loc) + '.txt'),
                      'a+') as fn:
                fn.write(
                    "%.5f  %.3f \n" % (float(F_px), T[i]))
            print("The output file written line for water depth =", i + 1)
            print("The time is =", T[i])
        if vel_profile == 'yes':
            path, cc = os.path.split(os.getcwd())
            # casename = 'flex_single_cyl_Y2'
            if i == 0:
                cs = case_type.replace('/', '_')
                with open(os.path.join(path, 'vel_profile' + '_' + cs + mod + str(slice_loc) + '.txt'),
                          'a+') as fn:
                    fn.write("v (m/s)  time   \n")
                fn.close()
            # print(energy_flux_data)
            v_max = energy_flux_data
            # print('pot_flux', pot_flux)
            with open(os.path.join(path, 'vel_profile' + '_' + cs + mod + str(slice_loc) + '.txt'),
                      'a+') as fn:
                fn.write(
                    "%.5f  %.3f \n" % (float(v_max), T[i]))
            print("The output file written line for water depth =", i + 1)
            print("The time is =", T[i])
            fn.close()
'''