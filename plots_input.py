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
    src_dir = '/Volumes/desktop-abhi/seagate/tsunami/arnason_et_al/multi-cyl/pent-case/output'
    # '/Volumes/seagate/seagate/tsunami/arnason_et_al/multi-cyl/pent-case/output'  # Location of main source directory
    case_dir = ['nocyl']
    case_rigid = ['rigid/onecyl', 'rigid/SP1.5D/double-cyl', 'rigid/SP1.5D/four-cyl', 'rigid/SP2D/double-cyl',
                  'rigid/SP2D/four-cyl']
    # case_rigid = ['single-cyl-rigid', 'double-cyl-sp1.5d', 'four-cyl-sp1.5d', 'double-cyl-sp2d', 'four-cyl-sp2d']
    # case_flex = ['flex/SP2D/flex-four-5e6', 'flex/SP2D/flex-four-1e6', 'flex/SP2D/flex-four-5e5']  # Four cyl
    # case_flex = ['flex/flex-single-1e5', 'flex/flex-single-5e4', 'flex/flex-single-1e4']  # Single cylinder
    case_flex = ['rigid/SP2D/double-cyl', 'flex/SP2D/flex-double-1e5', 'flex/SP2D/flex-double-5e4']  # Double cylinder
    case_type = 'flex'
    time_format = "{0:.2f}"
    t_ini, t_fin = 0, 10
    skip_line = 2
    dT = 0.2
    file_format = 'tsv'
    ts, tf = 4.2, 10
    ls = ['solid', 'solid', 'dotted', 'dashed']  # 'dotted', 'dashed', 'dashdot', '', 'dotted', 'dashed']
    ms = ['x', '^', '*', 'o', 'd', 'o', '+', '_']
    ds = [(1, 1), (1, 1), (5, 5), (3, 5, 1, 5, 1, 5), (8, 4, 2, 4, 2, 4), (1, 10), (5, 10)]
    lws = 2.5
    colorke = ['null', '#fd8d3c', '#e6550d', '#a63603']
    colorpe = ['null', '#fd8d3c', '#e6550d', '#a63603']
    flux_plot = 'on'  # , 'flux'  # type energy flux
    if flux_plot == 'on':
        fname = 'eflux'
        kfin, pfin, tfin = 1, 4, 7
        rgb = ['null', '#a63603', '#006d2c', '#08519c']
        colorke = ['null', '#fd8d3c', '#e6550d', '#a63603']  # ['null', '#d94701', '#238b45', '#2171b5']
        colorpe = ['null', '#fd8d3c', '#e6550d', '#a63603']  # ['null', '#fd8d3c', '#74c476', '#6baed6']
        ftype = ['flux', 'cum', 'reflection', 'dissipation']
        # elegends = ['nocyl', r'$E = 5 \times 10^{6} pa$', r'$E = 1 \times 10^{6} pa$', r'$E = 5 \times 10^{5} pa$']
        # elegends = ['nocyl', r'$E = 1 \times 10^{5} pa$', r'$E = 5 \times 10^{4} pa$', r'$E = 1 \times 10^{4} pa$']  # One-cyl
        elegends = ['nocyl', r'$E = 5 \times 10^{5} pa$', r'$E = 1 \times 10^{5} pa$', r'$E = 5 \times 10^{4} pa$']  # two-cyl
        #            r'SP2D/four-cyl', r'$E = 5 \times 10^{4} pa$', r'$E = 5 \times 10^{4} pa$',
        #            r'$E = 5 \times 10^{4} pa$']
        flux_plot_type = ftype[2]  # flex, reflection, cum
        xloc, xlocin = [11, 11, 10.97, 10.98], 10  # 11.1, 11.24, 11.17, 11.17, , 11.25, 11.24, 11.24, 11.24,
        # 11.135]
    tke_plot = 'off'
    depth_plot = 'off'
    shear_stress_plot = 'off'
    dam_front_plot = 'off'
    vel_profile_plot = 'off'
    depth_evolution_plot = 'off'
    energy_plot = 'off'
    t_avg = 'off'
    turbul_plot = 'off'
    eta_plot = 'off'
    if energy_plot == 'on':
        fname = 'te'
        kfin, pfin, tfin = 2, 1, 3
        slicex, xlocin = [8, 12], 4
        elegends = ['nocyl', r'$E = 5 \times 10^{6} pa$', r'$E = 5 \times 10^{4} pa$',
                    r'$E = 1 \times 10^{5} pa$', r'$E = 5 \times 10^{4} pa$',
                    r'$E = 5 \times 10^{4} pa$']
    if eta_plot == 'on':
        fname = 'te'
        kfin, pfin, tfin = 2, 1, 3
        xloc, xlocin = [11], [4, 5, 5, 5, 5]
        elegends = ['nocyl', r'$E = 5 \times 10^{6} pa$', r'$E = 1 \times 10^{6} pa$',
                    r'$E = 5 \times 10^{5} pa$', r'$E = 5 \times 10^{4} pa$',
                    r'$E = 5 \times 10^{4} pa$']

    if turbul_plot == 'on':
        fname = 'tke'
        xlocin, paramin = 2, 1
        cl = ['#762a83', 'b', '#1b7837']
        cyl_d = 0.035
        xc = 11.2 #11.9
        elegends = [r'$E = 5 \times 10^{6} pa$', r'$E = 1 \times 10^{6} pa$', r'$E = 5 \times 10^{5} pa$']  # Four-cyl
        # elegends = [r'$E = 1 \times 10^{5} pa$', r'$E = 5 \times 10^{4} pa$', r'$E = 1 \times 10^{4} pa$']  # One-cyl
        # elegends = [r'$E = 5 \times 10^{5} pa$', r'$E = 1 \times 10^{5} pa$', r'$E = 5 \times 10^{4} pa$']  # two-cyl
        # k_t = [7.8, 7.8, 7.2]  # Single-cyl
        k_t = [7, 7.4, 6.2]  # flex-four-1.5D
        # k_t = [8.6, 7, 7]  # flex-four-2D
        # k_t = [7.4, 10, 7]  # flex-double-2D
        # k_t = [7.4, 9, 7.2]  # flex-double-2D
        transient = 'on'
    paraview_data = 'off'
    if paraview_data == 'on':
        xloc, xlocin = 11, 10
    pv_eta = 'off'
    if pv_eta == 'on':
        xloc = 11
        elegends = [r'$E = 5 \times 10^{6} pa$', r'$E = 1 \times 10^{6} pa$',
                    r'$E = 5 \times 10^{5} pa$']
    v_avg_plot = 'off'
    if v_avg_plot == 'on':
        cl = ['#762a83', 'b', '#1b7837']
        fname = 'te'
        xlocin, paramin = 5, 4  # 5, 4  # 2, 1
        elegends = [r'$E = 5 \times 10^{6} pa$', r'$E = 1 \times 10^{6} pa$', r'$E = 5 \times 10^{5} pa$']  # Four-cyl
        # elegends = [r'$E = 1 \times 10^{5} pa$', r'$E = 5 \times 10^{4} pa$', r'$E = 1 \times 10^{4} pa$'] # One-cyl
        # elegends = [r'$E = 5 \times 10^{5} pa$', r'$E = 1 \times 10^{5} pa$', r'$E = 5 \times 10^{4} pa$']  # two-cyl
        v_t = [6.2, 7, 7.6]  # [5.8, 5.6, 5.4]
        cyl_d = 0.035
        # One cylinder [5.4, 6, 6.2]
        # SP2D four cylinder [6.2, 6.8, 8.8]
        # SP2D two cylinder [5.4, 5.6, 5.6]
        # SP1.5D four cylinder [6.2, 7, 7.6]
        # SP1.5D two cylinder [5.6, 5.6, 6.4]

    # transient_data = 'on', 0, 7  # time interval
