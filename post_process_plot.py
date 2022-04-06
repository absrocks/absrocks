import matplotlib.pyplot as plt
import numpy as np
import plots_input

font = {'family': 'Arial',
        'weight': 'normal',
        'size': 28}
plt.rc('font', **font)
# , flux_plot, energy_plot

data_input = plots_input.plots_input()

T = np.arange(data_input.t_ini, data_input.t_fin + data_input.dT, data_input.dT)
# Input Settings starts

if data_input.flux_plot == 'on':
    from plots import out_file_read, flux_plot

    src = data_input.src_dir + '/energy_flux'
    if data_input.case_type == 'rigid':
        data_input.case_dir.extend(data_input.case_rigid)
    elif data_input.case_type == 'flex':
        data_input.case_dir.extend(data_input.case_flex)
    # print(data_input.case_type, '******************')
    fout_obj = out_file_read(src, data_input.case_dir, data_input.case_type, data_input.lws, T,
                             data_input.ts, data_input.tf, data_input.xloc, data_input.ls, data_input.ms,
                             data_input.ds, data_input.case_dir, data_input.file_format, data_input.time_format,
                             data_input.skip_line, data_input.pfin, data_input.kfin, data_input.tfin,
                             data_input.xlocin, data_input.fname, plt, data_input.colorke, data_input.colorpe)
    plot_obj = flux_plot()
    if data_input.flux_plot_type == 'flux':
        finput, case_path, pf, kf, tf, t = fout_obj.fread(data_input.xloc)
        print('flux plot type is selected as', data_input.flux_plot_type)
        ax1, ax2, ax3 = plot_obj.eflux(finput, case_path, pf, kf, tf, t)
    elif data_input.flux_plot_type == 'cum':
        print('flux plot type is selected as', data_input.flux_plot_type)
        finput, case_path, pf, kf, tf, t = fout_obj.fread(data_input.xloc)
        ax1, ax2, ax3, pf_cum, kf_cum, tf_cum = plot_obj.cum(finput, case_path, pf, kf, tf, t, 'on')

    elif data_input.flux_plot_type == 'reflection':
        print('flux plot type is selected as', data_input.flux_plot_type)
        finput, case_path, pf, kf, tf, t = fout_obj.fread(data_input.xloc, data_input.ts)
        ax1, ax2, ax3 = plot_obj.reflection(finput, case_path, pf, kf, tf, t, data_input.elegends)

    elif data_input.flux_plot_type == 'dissipation':
        print('********** flux plot type is selected as', data_input.flux_plot_type)
        finput, case_path, pf1, kf1, tf1, t1 = fout_obj.fread(data_input.xloc, data_input.ts)
        finput, case_path, pf2, kf2, tf2, t2 = fout_obj.fread([12], 4.6)
        ax4, ax5, ax6 = plot_obj.dissipation(finput, case_path, pf1, kf1, tf1, t1, pf2, kf2, tf2, t2,
                                             data_input.elegends)

if data_input.paraview_data == 'on':

    print('paraview data is', data_input.paraview_data)
    from plots import single_data, eflux_plot

    src = data_input.src_dir + '/energy_flux/slicex' + str(data_input.xloc)
    if data_input.case_type == 'rigid':
        data_input.case_dir.extend(data_input.case_rigid)
    pv_obj = single_data.paraview_plot(src, data_input.case_dir, data_input.case_type, data_input.lws, T,
                                       data_input.ts, data_input.tf, data_input.xloc, data_input.ls, data_input.ms,
                                       data_input.ds, data_input.case_dir, data_input.file_format,
                                       data_input.time_format, plt)
    finput, case_path, pf, kf, tf, t = pv_obj.pvfileread()
    plot_obj = flux_plot()
    ax1, ax2, ax3 = plot_obj.reflection(finput, case_path, pf, kf, tf, t, data_input.elegends)

if data_input.energy_plot == 'on':
    print('******* energy plot is ', data_input.energy_plot)
    from plots import out_file_read, energy_plot

    src = data_input.src_dir + '/total_energy'
    if data_input.case_type == 'rigid':
        data_input.case_dir.extend(data_input.case_rigid)
    elif data_input.case_type == 'flex':
        data_input.case_dir.extend(data_input.case_flex)
    # print(data_input.fname)
    fout_obj = out_file_read(src, data_input.case_dir, data_input.case_type, data_input.lws, T,
                             data_input.ts, data_input.tf, [data_input.slicex[0]], data_input.ls, data_input.ms,
                             data_input.ds, data_input.case_dir, data_input.file_format, data_input.time_format,
                             data_input.skip_line, data_input.pfin, data_input.kfin, data_input.tfin,
                             data_input.xlocin, data_input.fname)
    plot_obj = energy_plot()
    finput, case_path, pe1, ke1, te1, t1 = fout_obj.fread([data_input.slicex[0]], data_input.ts)
    finput, case_path, pe2, ke2, te2, t2 = fout_obj.fread([data_input.slicex[1]], data_input.ts)
    ax1, ax2, ax3 = plot_obj.energy(finput, case_path, pe1, ke1, te1, t1, pe2, ke2, te2, data_input.elegends)

if data_input.turbul_plot == 'on':
    print('******* turbulence plot is ', data_input.turbul_plot)
    from plots import param_axial

    src = data_input.src_dir + '/turbulent_kinetic'  # '/total_energy' #'/turbulent_kinetic'
    if data_input.case_type == 'rigid':
        data_input.case_dir.extend(data_input.case_rigid)
    elif data_input.case_type == 'flex':
        data_input.case_dir.extend(data_input.case_flex)
    print(data_input.case_flex, data_input.case_type)
    fout_obj = param_axial(src, data_input.case_flex, data_input.case_type, data_input.lws, T,
                           data_input.ls, data_input.ms, data_input.ds, data_input.elegends,
                           data_input.file_format, data_input.time_format,
                           data_input.skip_line, data_input.fname, plt, data_input.xlocin,
                           data_input.paramin)

    ax1 = fout_obj.param(7)

plt.show()

'''
    from plots import eflux_plot
    case_output = 'energy_flux'
    #case_path = dest_dir + case_output + '/' + case_type
    reflection = 'off'
    dissipation = 'off'
    legend = case_type #['rigid', 'flex-E1e6', 'flex-E2e7', 'flex-E3e8']
    #['no-cyl', 'single-cyl-rigid', 'double-cyl-rigid', 'four-cyl-rigid']
    #case_type = ['no-cyl', 'double-cyl-2e5', 'double-cyl-rigid', 'four-cyl-2e5', 'four-cyl-rigid']
    #legend = ['nocyl', 'double-cyl,SP=2D', 'double-cyl,SP=1.5D', 'four-cyl,SP=2D', 'four-cyl,SP=1.5D']
    #legend = ['nocyl', 'double-cyl-2e5', 'double-cyl-rigid', 'four-cyl-2e5', 'four-cyl-rigid']
    #['nocyl', 'single-cyl', 'double-cyl', 'four-cyl']
    lws = 2
    t_s, t_f = 1, 10 #11.6
    slice_loc_R = [11.003, 11.003, 11.003, 11.003, 11.003]
    slice_loc_D = [12.602, 12.602, 12.602, 12.602, 12.602]
    x_front = 8
    e_limit = 5e-3
    ts = [15]
    # ts = [0.84, 0.88, 1.04]

    case = []
    case_path = []
    time_dir = 'csv_data/time'
    ms = ['x', '^', '*', 's', 'd', 'o', '+', '_']
    ds = [(1, 1), (1, 1), (5, 5), (3, 5, 1, 5, 1, 5), (8, 4, 2, 4, 2, 4), (1, 10), (5, 10)]


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
        case_path.append(dest_dir + case_output + '/' + case_type[icase])

        print("The case path is in:", case_path)
        # if case_type[icase] == 'no_cyl':
        case.append(case_type[icase])
        #


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
            fname = 'eflux_' + str(T[i]) + '.tsv'
            fdata = open(os.path.join(case_path[icase], fname), "r")
            lines = fdata.readlines()
            fdata.close()
            #print(case_path)
            x = np.zeros(len(lines))
            for j in range(1, len(lines)):
                line = lines[j]
                line = line.strip()
                line = line.split()
                x[j] = float(line[4])
                #print(x[j])
                if x[j] == x_front:
                    pf8[icase, i], kf8[icase, i], tf8[icase, i] = float(line[1]), float(line[2]), float(line[3])
                if x[j] == slice_loc_R[icase]:
                    pf10[icase, i], kf10[icase, i], tf10[icase, i] = float(line[1]), float(line[2]), float(line[3])

                if x[j] == slice_loc_D[icase]:

                    pf15[icase, i], kf15[icase, i], tf15[icase, i] = float(line[1]), float(line[2]), float(line[3])

        print('The path is in', case_path[icase])
        ax7.plot(T, pf8[icase, :]*1000, ms[icase], label=legend[icase], lw=lws,
                 linestyle=ls[icase], dashes=ds[icase])
        ax8.plot(T, kf8[icase, :]*1000, ms[icase], label=legend[icase], lw=lws,
                 linestyle=ls[icase], dashes=ds[icase])

        ax9.plot(T, tf8[icase, :]*1000 , ms[icase], label=legend[icase], lw=lws,
                     linestyle=ls[icase], dashes=ds[icase])

        ax7.legend(loc='best', fontsize='24', frameon=False)
        ax7.set_ylabel(r'$Pf_{8} \times 10^{-3}$')
        ax7.set_xlabel(r'Time [s]')
        ax7.set_title('Potential Energy Flux \n')
        #ax7.set_xlim(4, 12)
        #ax7.set_ylim(0.3, 0.7)
        # ax7.set_yticks(np.arange(0,0.18,0.03))

        ax8.legend(loc='best', fontsize='24', frameon=False)
        ax8.set_ylabel(r'$Kf_{8} \times 10^{-3}$')
        ax8.set_xlabel(r'Time [s]')
        ax8.set_title('Cumulative kinetic Energy Flux \n')
        #ax8.set_xlim(4, 12)
        #ax8.set_ylim(0.3, 0.7)
        # ax8.set_yticks(np.arange(0,0.18,0.03))

        ax9.legend(loc='best', fontsize='24', frameon=False)
        ax9.set_ylabel(r'$Tf_{8} \times 10^{-3}$')
        ax9.set_xlabel(r'Time [s]')
        ax9.set_title('Total Energy Flux \n')
        #ax9.set_xlim(4, 12)
        #ax9.set_ylim(0.8, 1)
        ax1.plot(T, pf10[icase, :]*1000, ms[icase], label=legend[icase], lw=lws,
                 linestyle=ls[icase], dashes=ds[icase])
        ax2.plot(T, kf10[icase, :]*1000, ms[icase], label=legend[icase], lw=lws,
                 linestyle=ls[icase], dashes=ds[icase])

        ax3.plot(T, tf10[icase, :]*1000 , ms[icase], label=legend[icase], lw=lws,
                 linestyle=ls[icase], dashes=ds[icase])

        ax4.plot(T, pf15[icase, :]*1000, ms[icase], label=legend[icase], lw=lws,
                 linestyle=ls[icase], dashes=ds[icase])
        ax5.plot(T, kf15[icase, :]*1000, ms[icase], label=legend[icase], lw=lws,
                 linestyle=ls[icase], dashes=ds[icase])

        ax6.plot(T, tf15[icase, :]*1000 , ms[icase], label=legend[icase], lw=lws,
                 linestyle=ls[icase], dashes=ds[icase])

        ax1.legend(loc='best', fontsize='24', frameon=False)
        ax1.set_ylabel(r'$Pf_{11.1} \times 10^{-3}$')
        ax1.set_xlabel(r'Time [s]')
        ax1.set_title('Potential Energy Flux \n')
        #ax7.set_xlim(4, 12)
        #ax7.set_ylim(0.3, 0.7)
        # ax7.set_yticks(np.arange(0,0.18,0.03))

        ax2.legend(loc='best', fontsize='24', frameon=False)
        ax2.set_ylabel(r'$Kf_{11.1} \times 10^{-3}$')
        ax2.set_xlabel(r'Time [s]')
        ax2.set_title('Cumulative kinetic Energy Flux \n')
        #ax8.set_xlim(4, 12)
        #ax8.set_ylim(0.3, 0.7)
        # ax8.set_yticks(np.arange(0,0.18,0.03))

        ax3.legend(loc='best', fontsize='24', frameon=False)
        ax3.set_ylabel(r'$Tf_{11.1} \times 10^{-3}$')
        ax3.set_xlabel(r'Time [s]')
        ax3.set_title('Total Energy Flux \n')

        ax4.legend(loc='best', fontsize='24', frameon=False)
        ax4.set_ylabel(r'$Pf_{12.6} \times 10^{-3}$')
        ax4.set_xlabel(r'Time [s]')
        ax4.set_title('Potential Energy Flux \n')
        #ax7.set_xlim(4, 12)
        #ax7.set_ylim(0.3, 0.7)
        # ax7.set_yticks(np.arange(0,0.18,0.03))

        ax5.legend(loc='best', fontsize='24', frameon=False)
        ax5.set_ylabel(r'$Kf_{12.6} \times 10^{-3}$')
        ax5.set_xlabel(r'Time [s]')
        ax5.set_title('Cumulative kinetic Energy Flux \n')
        #ax8.set_xlim(4, 12)
        #ax8.set_ylim(0.3, 0.7)
        # ax8.set_yticks(np.arange(0,0.18,0.03))

        ax6.legend(loc='best', fontsize='24', frameon=False)
        ax6.set_ylabel(r'$Tf_{12.6} \times 10^{-3}$')
        ax6.set_xlabel(r'Time [s]')
        ax6.set_title('Total Energy Flux \n')
        #plt.show()
        '''

'''
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
    '''
# print(case)

'''
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
