import numpy as np
import scipy.integrate as integrate


def span_average(y_3d, q_3d, lev3d, xpt, ypt, zpt, c_3d, comp):
    con_2d, d_y = np.zeros((xpt, zpt)), np.zeros((xpt, zpt))
    if comp == 1:
        q_2d_y, q_2d_yy = np.zeros((xpt, zpt)), np.zeros((xpt, ypt, zpt))
        q_3d[c_3d == 0] = 0
    else:
        q_2d_y, q_2d_yy = np.zeros((comp, xpt, zpt)), np.zeros((comp, xpt, ypt, zpt))
        for cc in range(comp):
            q_3d[cc, :, :][c_3d == 0] = 0

    for i in range(xpt):
        for k in range(zpt):
            d_y[i, k] = integrate.simps(c_3d[i, :, k], y_3d[i, :, k])
            if np.all(c_3d[i, :, k] == 0):
                if comp == 1:
                    q_2d_y[i, k] = 0
                else:
                    for cc in range(comp):
                        q_2d_y[cc, i, k] = 0
                con_2d[i, k] = 0
            else:
                if comp == 1:
                    q_2d_y[i, k] = integrate.simps(q_3d[i, :, k], y_3d[i, :, k]) / \
                                   integrate.simps(c_3d[i, :, k], y_3d[i, :, k])
                else:
                    for cc in range(comp):
                        q_2d_y[cc, i, k] = integrate.simps(q_3d[cc, i, :, k], y_3d[i, :, k]) / \
                                           integrate.simps(c_3d[i, :, k], y_3d[i, :, k])

                con_2d[i, k] = 1

    con_2d_yy = np.zeros((xpt, ypt, zpt))

    # for ix in range(xpt):
    for jy in range(ypt):
        if comp == 1:
            q_2d_yy[:, jy, :] = 1 * q_2d_y
        else:
            for cc in range(comp):
                q_2d_yy[cc, :, jy, :] = 1 * q_2d_y[cc, :, :]

        con_2d_yy[:, jy, :] = 1 * con_2d
    return q_2d_yy, con_2d_yy
