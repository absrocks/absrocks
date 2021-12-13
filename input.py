#!/usr/bin/env python3
# 1;95;0c -*- coding: utf-8 -*-


import os


# parameters

class flux_input:
    # Source directory where data is present
    src_dir = '/Volumes/seagate/seagate/tsunami/arnason_et_al/multi-cyl/pent-case/3d_data/flex/SP1.5D'
    case_dir = 'flex-four-5e5'  # flex-y1e6, rigid-rect
    # destination directory
    dest_dir = '/Volumes/seagate/seagate/tsunami/arnason_et_al/multi-cyl/pent-case/output_0.005'  # shortcut o

    data_directory = os.path.join(src_dir, case_dir)
    t_ini, t_fin = 0, 7
    dT = 0.2
    time_write_shift = 3.6
    transient_data = 'on', 0, 6.6  # time interval
    data_type = '3d'
    levsmooth = 'off'
    vel_field_avg = 'off'
    energy_flux = 'on'  # specify the time interval
    side_wall = 'off', 10  # nullify the effect of side wall, mention the number of grid points to exclude.
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
    nelx, nely, nelz = 1000, 160, 92  # 400, 106, 61  # 1000, 160, 92, # 600
    # water_vol = 6 * 3.5 * 4  # L * D * W, dry case
    probe_data = 'off'
    force = 'off'
    # flux_file = 'eflux' + fe + str(slice_loc)
    xpt, ypt, zpt = nelx + 1, nely + 1, nelz + 1
    slice_interpolation = 'off'
    flux_dissipation = 'on'
    epsilon = 'on'
    fname = '3d_data'
    fe = '_'  # '0.'
    filename = fname + fe  # + str(i) + '.csv'
