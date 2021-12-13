import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from scipy import integrate
# import glob
# from labellines import labelLine, labelLines
# from scipy import interpolate
import matplotlib.patches as patches

plt.close("all")
# plt.style.use('classic')
font = {'family': 'Arial',
        'weight': 'normal',
        'size': 28}
plt.rc('font', **font)

fig1, ax1 = plt.subplots(figsize=(11, 7), dpi=90)
plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
'''
fig2, ax2 = plt.subplots(figsize=(11, 7), dpi=90)
plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
fig3, ax3 = plt.subplots(figsize=(11, 7), dpi=90)
plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)

for axis in ['top', 'bottom', 'left', 'right']:
    ax1.spines[axis].set_linewidth(2)
    ax2.spines[axis].set_linewidth(2)
    ax3.spines[axis].set_linewidth(2)
'''


# Input Settings starts
class energy_plot:

    def energy(self, finput, case_path, pe1, ke1, te1, t, pe2, ke2, te2, legends):
        pet1 = np.zeros(len(case_path))
        ket1, tet1, pet2, ket2, tet2 = 0 * pet1, 0 * pet1, 0 * pet1, 0 * pet1, 0 * pet1

        for icase in range(len(case_path)):
            print('The path is in', case_path[icase])
            pet1[icase] = integrate.trapz(pe1[icase, :], t)
            ket1[icase] = integrate.trapz(ke1[icase, :], t)
            tet1[icase] = integrate.trapz(te1[icase, :], t)

            pet2[icase] = integrate.trapz(pe2[icase, :], t)
            ket2[icase] = integrate.trapz(ke2[icase, :], t)
            tet2[icase] = integrate.trapz(te2[icase, :], t)
        k = [0, 0.16, 0.02, 0.03]  # [0, 0.16, 0.05, 0.1]
        p = [0, 0.10, -0.01, -0.03]  # [0, 0.11, 0, 0.0]
        kp = [0, 0.26, 0.01, 0.0]
        kf = abs(ket1[1:len(case_path)] - ket2[1:len(case_path)]) / tet1[0]
        pf = abs(pet1[1:len(case_path)] - pet2[1:len(case_path)]) / tet1[0]
        tf = kf + pf
        # print(len(case_path))

        ax1.plot(legends[1:len(case_path)],
                 0 + kf,
                 finput.ms[icase], lw=finput.lws, linestyle=finput.ls[icase], dashes=finput.ds[icase])
        '''
        ax2.plot(legends[1:len(case_path)],
                 0 + pf,
                 finput.ms[icase], lw=finput.lws, linestyle=finput.ls[icase], dashes=finput.ds[icase])

        ax3.plot(legends[1:len(case_path)], 0 + tf,
                 finput.ms[icase], lw=finput.lws, linestyle=finput.ls[icase], dashes=finput.ds[icase])

        ax1.legend(loc='best', fontsize='24', frameon=False)
        ax2.legend(loc='best', fontsize='24', frameon=False)
        ax3.legend(loc='best', fontsize='24', frameon=False)  # label=finput.legend[icase],
        print(abs(ket1 - ket2) / tet1[0], '\n', abs(pet1 - pet2) / tet1[0], '\n', kp)
        ax1.set_xlabel(r'Time [s]')
        ax2.set_xlabel(r'Time [s]')
        ax3.set_xlabel(r'Time [s]')
        ax1.set_ylabel(r'$\phi^{p}_{re}$')
        ax2.set_ylabel(r'$\phi^{k}_{re}$')
        ax3.set_ylabel(r'$\phi^{t}_{re}$')
        '''
        return ax1, pf, kf  # , ax2, ax3
