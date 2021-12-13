import numpy as np
from scipy import interpolate


def lev_smooth(lev_3d, z_3d, q_3d, xpt, ypt, zpt, x_3d, y_3d, comp):
    levpt = 100
    z3d, lev3d, x3d, y3d, c3d = np.zeros((xpt, ypt, zpt + levpt - 2)), np.zeros((xpt, ypt, zpt + levpt - 2)), \
                                np.zeros((xpt, ypt, zpt + levpt - 2)), np.zeros((xpt, ypt, zpt + levpt - 2)), \
                                np.zeros((xpt, ypt, zpt + levpt - 2))
    if comp == 1:
        q3d = np.zeros((xpt, ypt, zpt + levpt - 2))
    else:
        q3d = np.zeros((comp, xpt, ypt, zpt + levpt - 2))
    for ix in range(xpt):
        for jy in range(ypt):
            for kz in range(zpt):
                if lev_3d[ix, jy, zpt - 1 - kz] >= 0:
                    lindex = zpt - 1 - kz
                    if lindex > 250:
                        print('lindex error')
                    # print(lindex, ix)
                    zz = np.linspace(z_3d[ix, jy, lindex], z_3d[ix, jy, lindex + 1], levpt)
                    zlev = z_3d[ix, jy, lindex:lindex + 2]
                    levn = lev_3d[ix, jy, lindex:lindex + 2]
                    f = interpolate.interp1d(zlev, levn)
                    levz = f(zz)
                    if comp == 1:
                        qq = q_3d[ix, jy, lindex:lindex + 2]
                        fq = interpolate.interp1d(zlev, qq)
                        qqz = fq(zz)
                        q3d[ix, jy, :] = np.concatenate((q_3d[ix, jy, :][:lindex + 1], qqz[1: len(qqz) - 1],
                                                         q_3d[ix, jy, :][lindex + 1:]))
                    else:
                        for cc in range(comp):
                            qq = q_3d[cc, ix, jy, lindex:lindex + 2]
                            fq = interpolate.interp1d(zlev, qq)
                            qqz = fq(zz)
                            q3d[cc, ix, jy, :] = np.concatenate((q_3d[cc, ix, jy, :][:lindex + 1], qqz[1: len(qqz) - 1],
                                                                 q_3d[cc, ix, jy, :][lindex + 1:]))

                    # for kk in range(levpt):
                    #    if levz[kk] <= 1e-4:
                    #        kindex = kk
                    #        break
                    # print('l,k', lindex, kindex)
                    # print(z_3d[ix, jy, :][:lindex + 1])
                    z3d[ix, jy, :] = np.concatenate((z_3d[ix, jy, :][: lindex + 1], zz[1: len(zz) - 1],
                                                     z_3d[ix, jy, :][lindex + 1:]))
                    lev3d[ix, jy, :] = np.concatenate((lev_3d[ix, jy, :][: lindex + 1], levz[1: len(levz) - 1],
                                                       lev_3d[ix, jy, :][lindex + 1:]))

                    break

    for ix in range(xpt):
        for jy in range(ypt):
            if np.all(lev_3d[ix, jy, :] < 0):
                z3d[ix, jy, :] = np.linspace(min(z_3d[ix, jy, :]), max(z_3d[ix, jy, :]), zpt + levpt - 2)
                lev3d[ix, jy, :] = -5

    for ii in range(xpt):
        for jj in range(ypt):
            for kk in range(zpt + levpt - 2):
                x3d[ii, jj, kk] = x_3d[ii, jj, 0]
                y3d[ii, jj, kk] = y_3d[ii, jj, 0]
    c3d[lev3d >= 0] = 1

    return q3d, lev3d, z3d, x3d, y3d, c3d, zpt + levpt - 2
