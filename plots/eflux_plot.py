import numpy as np
import numpy.ma as ma
from scipy import integrate
# import glob
# from labellines import labelLine, labelLines
# from scipy import interpolate
import matplotlib.patches as patches


# Input Settings starts
class flux_plot:

    def eflux(self, finput, case_path, pf, kf, tf, t):

        for icase in range(len(case_path)):
            print('The path is in', case_path[icase])
            finput.ax1.plot(t, pf[icase, :] * 1000, finput.ms[icase], label=finput.legend[icase], lw=finput.lws,
                            linestyle=finput.ls[icase], dashes=finput.ds[icase])
            finput.ax2.plot(t, kf[icase, :] * 1000, finput.ms[icase], label=finput.legend[icase], lw=finput.lws,
                            linestyle=finput.ls[icase], dashes=finput.ds[icase])

            finput.ax3.plot(t, tf[icase, :] * 1000, finput.ms[icase], label=finput.legend[icase], lw=finput.lws,
                            linestyle=finput.ls[icase], dashes=finput.ds[icase])
            finput.ax1.legend(loc='best', fontsize='24', frameon=False)
            finput.ax2.legend(loc='best', fontsize='24', frameon=False)
            finput.ax3.legend(loc='best', fontsize='24', frameon=False)
            finput.ax1.set_xlabel(r'Time [s]')
            finput.ax2.set_xlabel(r'Time [s]')
            finput.ax3.set_xlabel(r'Time [s]')
            finput.ax1.set_ylabel(r'$Potential flux \times 10^{-3}$')
            finput.ax2.set_ylabel(r'$Kf_{8} \times 10^{-3}$')
            finput.ax3.set_ylabel(r'$Tf_{8} \times 10^{-3}$')

        return finput.ax1, finput.ax2, finput.ax3

    def cum(self, finput, case_path, pf, kf, tf, t, plot):
        pf_cum, kf_cum, tf_cum = pf * 0, kf * 0, tf * 0
        for icase in range(len(case_path)):
            print('The path is in', case_path[icase])
            pf_cum[icase, :] = integrate.cumtrapz(pf[icase, :], t, initial=pf[icase, 0])
            kf_cum[icase, :] = integrate.cumtrapz(kf[icase, :], t, initial=kf[icase, 0])
            tf_cum[icase, :] = integrate.cumtrapz(tf[icase, :], t, initial=tf[icase, 0])

            tol = 1e-4

            if icase >= 1:
                print(kf_cum[icase, :], '\n', pf_cum[icase, :])
                pf_t = abs(pf_cum[icase, :] / tf_cum[0, :])
                pf_t[np.where(pf_cum[icase, :] <= tol)] = 0.1
                kf_t = abs(kf_cum[icase, :] / tf_cum[0, :])
                kf_t[np.where(kf_cum[icase, :] <= tol)] = 0.1
                tf_t = abs(tf_cum[icase, :] / tf_cum[0, :])
                tf_t[np.where(tf_cum[icase, :] <= tol)] = 0.1
                if plot == 'on':
                    finput.ax1.plot(np.delete(t, np.where(pf_t >= 2)), np.delete(pf_t, np.where(pf_t >= 2)),
                                    finput.ms[icase], label=finput.legend[icase], lw=finput.lws,
                                    linestyle=finput.ls[icase], dashes=finput.ds[icase])
                    finput.ax2.plot(np.delete(t, np.where(kf_t >= 2)), np.delete(kf_t, np.where(kf_t >= 2)),
                                    finput.ms[icase], label=finput.legend[icase], lw=finput.lws,
                                    linestyle=finput.ls[icase], dashes=finput.ds[icase])
                    finput.ax3.plot(np.delete(t, np.where(tf_t >= 2)), np.delete(tf_t, np.where(tf_t >= 2)),
                                    finput.ms[icase], label=finput.legend[icase], lw=finput.lws,
                                    linestyle=finput.ls[icase], dashes=finput.ds[icase])

                finput.ax1.legend(loc='best', fontsize='24', frameon=False)
                finput.ax2.legend(loc='best', fontsize='24', frameon=False)
                finput.ax3.legend(loc='best', fontsize='24', frameon=False)

                finput.ax1.set_xlabel(r'Time [s]')
                finput.ax2.set_xlabel(r'Time [s]')
                finput.ax3.set_xlabel(r'Time [s]')
                finput.ax1.set_ylabel(r'$E_{p_{\tau}}/E_{I_{\tau}}$')
                finput.ax2.set_ylabel(r'$E_{k_{\tau}}/E_{I_{\tau}}$')
                finput.ax3.set_ylabel(r'$E_{t_{\tau}}/E_{I_{\tau}}$')
        return finput.ax1, finput.ax2, finput.ax3, pf_cum, kf_cum, tf_cum

    def reflection(self, finput, case_path, pf, kf, tf, t, legends):

        pf_cum = np.zeros((len(case_path), len(t) - 1))
        kf_cum, tf_cum = 0 * pf_cum, 0 * pf_cum

        for icase in range(len(case_path)):
            print('The path is in', case_path[icase])

            pf_cum[icase, :] = integrate.cumtrapz(pf[icase, :], t)
            kf_cum[icase, :] = integrate.cumtrapz(kf[icase, :], t)
            tf_cum[icase, :] = integrate.cumtrapz(tf[icase, :], t)

            rkf = abs(kf_cum[0, :] - kf_cum[icase, :]) / tf_cum[0, :]
            pkf = abs(pf_cum[0, :] - pf_cum[icase, :]) / tf_cum[0, :]
            tkf = (tf_cum[0, :] - tf_cum[icase, :]) / tf_cum[0, :]
            # print(rkf, '\n', pf_cum[icase, :], t, len(pf_cum[icase, :]))
            if icase >= 1:
                # print(kf1[icase, :][tf1[icase, :] >= tol])
                finput.ax1.plot(t[1:len(t)], rkf, finput.ms[icase], color=finput.cke[icase],
                                label=legends[icase], lw=finput.lws,
                                linestyle=finput.ls[icase], dashes=finput.ds[icase])
                finput.ax1.plot(t[1:len(t)], pkf,
                                finput.ms[icase], color=finput.cpe[icase], lw=finput.lws,
                                linestyle=finput.ls[icase], dashes=finput.ds[icase])
                finput.ax3.plot(t[1:len(t)], rkf + pkf,
                                finput.ms[icase], label=legends[icase], lw=finput.lws,
                                linestyle=finput.ls[icase], dashes=finput.ds[icase])
            # KE
            finput.ax1.set_xlim(4, 10.5)
            finput.ax1.set_ylim(0, 0.2)
            finput.ax1.set_yticks(np.arange(0, 0.36, 0.05))
            finput.ax1.text(0.5, 0.9, r'$\phi^{k}_{r}$', horizontalalignment='center',
                            verticalalignment='center', transform=finput.ax1.transAxes)
            finput.ax1.text(0.5, 0.3, r'$\phi^{p}_{r}$', horizontalalignment='center',
                           verticalalignment='center', transform=finput.ax1.transAxes)

            # PE
            finput.ax2.set_xlim(4, 10.5)
            finput.ax2.set_ylim(0, 0.2)
            finput.ax2.set_yticks(np.arange(0, 0.26, 0.05))

            # TE
            finput.ax3.set_xlim(4, 10.5)
            finput.ax3.set_ylim(0, 0.2)
            finput.ax3.set_yticks(np.arange(0, 0.41, 0.05))

            finput.ax1.legend(loc='best', fontsize='24', frameon=False)
            finput.ax2.legend(loc='best', fontsize='24', frameon=False)
            finput.ax3.legend(loc='best', fontsize='24', frameon=False)

            finput.ax1.set_xlabel(r'Time [s]')
            finput.ax2.set_xlabel(r'Time [s]')
            finput.ax3.set_xlabel(r'Time [s]')
            finput.ax1.set_ylabel('Reflected flux coefficient')  # (r'$\phi^{k}_{r}$')
            finput.ax2.set_ylabel(r'$\phi^{p}_{r}$')
            finput.ax3.set_ylabel(r'$\phi^{t}_{r}$')

        return finput.ax1, finput.ax2, finput.ax3

    def dissipation(self, finput, case_path, pf1, kf1, tf1, t1, pf2, kf2, tf2, t2, legends):
        pf_cum1 = np.zeros((len(case_path), len(t2) - 1))
        kf_cum1, tf_cum1, pf_cum2, kf_cum2, tf_cum2 = 0 * pf_cum1, 0 * pf_cum1, 0 * pf_cum1, 0 * pf_cum1, 0 * pf_cum1
        t1 = t1[0:len(t2)]
        for icase in range(len(case_path)):
            print('The path is in', case_path[icase])

            pf_cum1[icase, :] = integrate.cumtrapz(pf1[icase, :][0:len(t2)], t1)
            kf_cum1[icase, :] = integrate.cumtrapz(kf1[icase, :][0:len(t2)], t1)
            tf_cum1[icase, :] = integrate.cumtrapz(tf1[icase, :][0:len(t2)], t1)

            pf_cum2[icase, :] = integrate.cumtrapz(pf2[icase, :], t2)
            kf_cum2[icase, :] = integrate.cumtrapz(kf2[icase, :], t2)
            tf_cum2[icase, :] = integrate.cumtrapz(tf2[icase, :], t2)

            if icase > 0:
                finput.ax1.plot(t2[1:len(t2)], abs(pf_cum1[icase, :] - pf_cum2[icase, :]) / tf_cum1[0, :])
                finput.ax2.plot(t2[1:len(t2)], abs(pf_cum1[icase, :] - pf_cum2[icase, :]) / tf_cum1[0, :])
                finput.ax3.plot(t2[1:len(t2)], abs(pf_cum1[icase, :] - pf_cum2[icase, :]) / tf_cum1[0, :])
                '''
                #print(icase)
                print(pf_cum1[icase, :], '\n', len(pf_cum1[icase, :]), len(t2))
                finput.ax1.plot(t2[1:len(t2)], abs(pf_cum1[icase, :] - pf_cum2[icase, :]) / tf_cum1[0, :], finput.ms[icase],
                         label=legends[icase], lw=finput.lws,
                         linestyle=finput.ls[icase], dashes=finput.ds[icase])
                finput.ax2.plot(t2[1:len(t2)], abs(kf_cum1[icase, :] - kf_cum2[icase, :]) / tf_cum1[0, :], finput.ms[icase],
                         label=legends[icase], lw=finput.lws,
                         linestyle=finput.ls[icase], dashes=finput.ds[icase])

                ax3.plot(t2[1:len(t2)],
                         (abs(pf_cum2[icase, :] - pf_cum1[icase, :]) + abs(kf_cum1[icase, :] - kf_cum2[icase, :])) /
                         tf_cum1[0, :], label=legends[icase], lw=finput.lws,
                         linestyle=finput.ls[icase], dashes=finput.ds[icase])
        
        finput.ax1.set_xlim(4, 10.5)
        finput.ax1.set_ylim(0, 0.25)
        ax1.set_yticks(np.arange(0, 0.251, 0.05))

        ax2.set_xlim(4, 10.5)
        ax2.set_ylim(0, 0.21)
        ax2.set_yticks(np.arange(0, 0.251, 0.05))

        ax3.set_xlim(4, 10.5)
        ax3.set_yticks(np.arange(0, 0.351, 0.05))
        ax1.legend(loc='best', fontsize='24', frameon=False)
        ax2.legend(loc='best', fontsize='24', frameon=False)
        ax3.legend(loc='best', fontsize='24', frameon=False)

        ax1.set_xlabel(r'Time [s]')
        ax2.set_xlabel(r'Time [s]')
        ax3.set_xlabel(r'Time [s]')
        ax2.set_ylabel(r'$\phi^{k}_{d}$')
        ax1.set_ylabel(r'$\phi^{p}_{d}$')
        ax3.set_ylabel(r'$\phi^{t}_{d}$')
        #print(fig1)
        #plt.show()
        '''
        return ax1, ax2, ax3
