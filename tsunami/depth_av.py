import numpy as np
import scipy.integrate as integrate


def depth_average(lev_3d, z_3d, q_3d, xpt, ypt, zpt, c_3d, comp):
    # global kindex
    qq_3d = 1 * q_3d
    con_2d = np.zeros((xpt, ypt))
    if comp == 1:
        q_2d_z, q_2d_zz = np.zeros((xpt, ypt)), np.zeros((xpt, ypt, zpt))
    else:
        q_2d_z, q_2d_zz = np.zeros((comp, xpt, ypt)), np.zeros((comp, xpt, ypt, zpt))
    depth_2d = np.zeros((xpt, ypt))

    for ix in range(xpt):
        for jy in range(ypt):
            for kz in range(zpt):
                if lev_3d[ix, jy, zpt - 1 - kz] >= 0:
                    depth_2d[ix, jy] = z_3d[ix, jy, zpt - 1 - kz]
                    if depth_2d[ix, jy] == 0:
                        con_2d[ix, jy] = 0
                        if comp == 1:
                            q_2d_z[ix, jy] = 0
                        else:
                            for cc in range(comp):
                                q_2d_z[cc, ix, jy] = 0

                    else:
                        if comp == 1:
                            q_2d_z[ix, jy] = (integrate.simps(q_3d[ix, jy, :zpt - kz], z_3d[ix, jy, :zpt - kz])) #/ (
                              #  depth_2d[ix, jy]))
                        else:
                            for cc in range(comp):
                                q_2d_z[cc, ix, jy] = (integrate.simps(q_3d[cc, ix, jy, :zpt - kz], z_3d[ix, jy, :zpt - kz]))
                                                      #/ (depth_2d[ix, jy]))
                        con_2d[ix, jy] = 1
                    break

    con_2d_zz = 1 * c_3d
    d_2d_zz = 1 * c_3d
    for kz in range(zpt):
        if comp == 1:
            q_2d_zz[:, :, kz] = 1 * q_2d_z
        else:
            for cc in range(comp):
                q_2d_zz[cc, :, :, kz] = 1 * q_2d_z[cc, :, :]
        con_2d_zz[:, :, kz] = 1 * con_2d
        d_2d_zz[:, :, kz] = 1 * depth_2d
    return q_2d_zz, con_2d_zz, d_2d_zz
