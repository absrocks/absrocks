#!/usr/bin/env python3
# 1;95;0c -*- coding: utf-8 -*-


import os


# parameters

class flux_input:
    # Source directory where data is present
    src_dir = '/media/abhishek/WD-phd/transcend/PHD_2019/tsunami/dam_break_wet_case/les_swash/' \
              'csv_data'
    case_dir = '3d_data_kim'  # flex-y1e6, rigid-rect
    # destination directory
    dest_dir = '/media/abhishek/WD-phd/transcend/PHD_2019/tsunami/dam_break_wet_case/les_swash/' \
               'csv_data/3d_data_kim/output'  # shortcut o

    data_directory = os.path.join(src_dir, case_dir)
    t_ini, t_fin = 0, 7
    dT = 0.2
    time_write_shift = 3.6
    transient_data = 'off', 0, 6.6  # time interval
    data_type = '3d'
    levsmooth = 'off'
    vel_field_avg = 'off'
    energy_flux = 'off'  # specify the time interval
    side_wall = 'on', 10, 'z'  # nullify the effect of side wall, mention the number of grid points to exclude.
    # and add direction along which side walls are oriented
    water_depth = 'off'
    total_energy = 'on'
    turbulent_energy = 'off'
    turbulent_average = 'off'
    shear_data = 'off'
    vel_profile = 'off', 15  # select at which stream-wise location you want to get span-averaged velocity profile
    dam_front = 'off'
    wall_shear_stress = 'off'
    lev_set = 0
    int_avg = 'field'  # field, int
    nelx, nely, nelz = 1140, 92, 90  # 400, 106, 61  # 1000, 160, 92, # 600
    # water_vol = 6 * 3.5 * 4  # L * D * W, dry case
    probe_data = 'off'
    force = 'off'
    # flux_file = 'eflux' + fe + str(slice_loc)
    xpt, ypt, zpt = nelx + 1, nely + 1, nelz + 1
    slice_interpolation = 'off'
    flux_dissipation = 'off'
    epsilon = 'off'
    fname = '3d_data_kim'
    fe = '_'  # '0.'
    filename = fname + fe  # + str(i) + '.csv'
