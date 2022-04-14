#!/usr/bin/env python3
# 1;95;0c -*- coding: utf-8 -*-
# class AnyObjectHandler(HandlerBase):
#    def create_artists(self, legend, orig_handle,
#                       x0, y0, width, height, fontsize, trans):
#        l1 = plt.Line2D([x0,y0+width], [0.7*height,0.7*height],
#                           linestyle=orig_handle[1], color='k')
#        l2 = plt.Line2D([x0,y0+width], [0.3*height,0.3*height],
#                           color=orig_handle[0])
#        return [l1, l2]

class plots_input:
    # Source directory where data is present
    # src_dir = '/Volumes/seagate/seagate/tsunami/arnason_et_al/arnason_multi_data'
    src_dir = '/Volumes/10.0.0.104/seagate/tsunami/arnason_et_al/multi-cyl/pent-case/output'
    # '/Volumes/seagate/seagate/tsunami/arnason_et_al/multi-cyl/pent-case/output'  # Location of main source directory
    case_dir = ['nocyl']
    case_rigid = ['rigid/onecyl', 'rigid/SP1.5D/double-cyl', 'rigid/SP1.5D/four-cyl', 'rigid/SP2D/double-cyl',
                  'rigid/SP2D/four-cyl']
    # case_rigid = ['single-cyl-rigid', 'double-cyl-sp1.5d', 'four-cyl-sp1.5d', 'double-cyl-sp2d', 'four-cyl-sp2d']
    # case_flex = ['flex-single-1e4', 'flex-single-5e4', 'flex-single-1e5']
    case_flex = ['flex/flex-single-1e5', 'flex/flex-single-5e4',
                 'flex/flex-single-1e4']  # , 'flex/SP1.5D/flex-four-5e5']  # 'rigid/SP1.5D/four-cyl',
    # case_flex = ['double-cyl-2e5', 'double-cyl-sp1.5d', 'four-cyl-2e5', 'four-cyl-sp1.5d']
    case_type = 'flex'
    time_format = "{0:.2f}"
    t_ini, t_fin = 0, 10
    skip_line = 2
    dT = 0.2
    file_format = 'tsv'
    ts, tf = 4.8, 10
    ls = ['solid', 'solid', 'dotted', 'dashed']  # 'dotted', 'dashed', 'dashdot', '', 'dotted', 'dashed']
    ms = ['x', '^', '*', 'o', 'd', 'o', '+', '_']
    ds = [(1, 1), (1, 1), (5, 5), (3, 5, 1, 5, 1, 5), (8, 4, 2, 4, 2, 4), (1, 10), (5, 10)]

    flux_plot = 'on'  # , 'flux'  # type energy flux
    if flux_plot == 'on':
        fname = 'eflux'
        kfin, pfin, tfin = 1, 4, 7
        rgb = ['null', '#a63603', '#006d2c', '#08519c']
        colorke = ['null', '#fd8d3c', '#e6550d', '#a63603']  # ['null', '#d94701', '#238b45', '#2171b5']
        colorpe = ['null', '#fd8d3c', '#e6550d', '#a63603']  # ['null', '#fd8d3c', '#74c476', '#6baed6']
        ftype = ['flux', 'cum', 'reflection', 'dissipation']
        elegends = ['nocyl', r'$E = 1 \times 10^{5} pa$', r'$E = 5 \times 10^{4} pa$', r'$E = 1 \times 10^{4} pa$',
                    r'SP2D/four-cyl', r'$E = 5 \times 10^{4} pa$', r'$E = 5 \times 10^{4} pa$',
                    r'$E = 5 \times 10^{4} pa$']
        flux_plot_type = ftype[3]  # flex, reflection, cum
        xloc, xlocin = [11.1, 11.125, 11.115, 11.105], 10  # 11.1, 11.24, 11.17, 11.17, , 11.25, 11.24, 11.24, 11.24,
        # 11.135]
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
    turbul_plot = 'off'
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
