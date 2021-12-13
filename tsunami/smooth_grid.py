import numpy as np
from scipy import interpolate


def smooth_grid(z3d, lev3d, q3d, xpt, ypt, zpt, comp):
    zz3d = np.zeros((xpt, ypt, zpt))
    lev_3d = np.zeros((xpt, ypt, zpt))
    if comp == 1:
        q_3d = np.zeros((xpt, ypt, zpt))
    else:
        q_3d = np.zeros((comp, xpt, ypt, zpt))
    for i in range(xpt):
        for j in range(ypt):
            zz3d[i, j, :] = np.linspace(min(z3d[i, j, :]), max(z3d[i, j, :]), zpt)
            flev = interpolate.interp1d(z3d[i, j, :], lev3d[i, j, :])
            lev_3d[i, j, :] = flev(zz3d[i, j, :])
            if comp == 1:
                fq = interpolate.interp1d(z3d[i, j, :], q3d[i, j, :])
                q_3d[i, j, :] = fq(zz3d[i, j, :])
            else:
                for cc in range(comp):
                    fq = interpolate.interp1d(z3d[i, j, :], q3d[cc, i, j, :])
                    q_3d[cc, i, j, :] = fq(zz3d[i, j, :])
    c_3d = 1 * lev_3d
    c_3d[lev_3d < 0] = 0
    c_3d[lev_3d >= 0] = 1
    return lev_3d, q_3d, zz3d, c_3d
