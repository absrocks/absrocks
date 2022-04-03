#!/usr/bin/env python3
# 1;95;0c -*- coding: utf-8 -*-


class plots_input:
    # Source directory where data is present
    # src_dir = '/Volumes/seagate/seagate/tsunami/arnason_et_al/arnason_multi_data'
    src_dir = '/Volumes/10.0.0.104/seagate/tsunami/arnason_et_al/multi-cyl/pent-case/output'
    # '/Volumes/seagate/seagate/tsunami/arnason_et_al/multi-cyl/pent-case/output'  # Location of main source directory
    case_dir = ['nocyl']
    case_rigid = ['rigid/onecyl', 'rigid/SP1.5D/double-cyl', 'rigid/SP1.5D/four-cyl', 'rigid/SP2D/double-cyl',
                  'rigid/SP2D/four-cyl']
    # case_rigid = ['single-cyl-rigid', 'double-cyl-sp1.5d', 'four-cyl-sp1.5d', 'double-cyl-sp2d', 'four-cyl-sp2d']
    case_flex = ['flex-single-1e4', 'flex-single-5e4', 'flex-single-1e5']
    # case_flex = ['flex/SP1.5D/flex-four-5e5', 'flex/SP1.5D/flex-four-1e6',
    #             'flex/SP1.5D/flex-four-5e6']  # , 'flex/SP1.5D/flex-four-5e5']  # 'rigid/SP1.5D/four-cyl',
    # case_flex = ['double-cyl-2e5', 'double-cyl-sp1.5d', 'four-cyl-2e5', 'four-cyl-sp1.5d']
    case_type = 'flex'
    time_format = "{0:.2f}"
    t_ini, t_fin = 0, 10
    skip_line = 2
    dT = 0.2
    file_format = 'tsv'
    ts, tf = 4, 10
    ls = ['solid', 'dotted', 'dashed', 'dashdot', '', 'dotted', 'dashed']
    ms = ['x', '^', '*', 's', 'd', 'o', '+', '_']
    ds = [(1, 1), (1, 1), (5, 5), (3, 5, 1, 5, 1, 5), (8, 4, 2, 4, 2, 4), (1, 10), (5, 10)]

    flux_plot = 'off'  # , 'flux'  # type energy flux
    if flux_plot == 'on':
        fname = 'eflux'
        kfin, pfin, tfin = 1, 4, 7
        ftype = ['flux', 'cum', 'reflection', 'dissipation']
        elegends = ['nocyl', r'$E = 1 \times 10^{7} pa$', r'$E = 5 \times 10^{6} pa$', r'$E = 1 \times 10^{6} pa$',
                    r'SP2D/four-cyl', r'$E = 5 \times 10^{4} pa$', r'$E = 5 \times 10^{4} pa$',
                    r'$E = 5 \times 10^{4} pa$']
        flux_plot_type = ftype[2]  # flex, reflection, cum
        xloc, xlocin = [11, 11, 10.96, 10.85], 10  # 11.1, 11.24, 11.17, 11.17, , 11.25, 11.24, 11.24, 11.24, 11.135]
    tke_plot = 'off'
    depth_plot = 'off'
    shear_stress_plot = 'off'
    dam_front_plot = 'off'
    vel_profile_plot = 'off'
    depth_evolution_plot = 'off'
    energy_plot = 'off'
    if energy_plot == 'on':
        fname = 'te'
        kfin, pfin, tfin = 2, 1, 3
        slicex, xlocin = [8, 12], 4
        elegends = ['nocyl', r'$E = 5 \times 10^{5} pa$', r'$E = 5 \times 10^{4} pa$',
                    r'$E = 1 \times 10^{5} pa$', r'$E = 5 \times 10^{4} pa$',
                    r'$E = 5 \times 10^{4} pa$']
    lws = 2.5
    turbul_plot = 'on'
    if turbul_plot == 'on':
        fname = 'tke'
        xlocin, paramin = 2, 1  # 5, 4  # 2, 1
        elegends = [r'$E = 5 \times 10^{5} pa$', r'$E = 1 \times 10^{6} pa$',
                    r'$E = 5 \times 10^{6} pa$']
        # [r'$E = 1 \times 10^{4} pa$', r'$E = 5 \times 10^{4} pa$',
        #            r'$E = 1 \times 10^{5} pa$']
    paraview_data = 'off'
    if paraview_data == 'on':
        xloc, xlocin = 11, 10

    # transient_data = 'on', 0, 7  # time interval
