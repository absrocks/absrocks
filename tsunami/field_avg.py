# Calculate average in Y direction:
# f_avg = sum(f(y))/ypt where ypt = number of points along Y direction

import numpy as np


def field_avg(q_3d, comp, xpt, ypt, zpt, avg):
    if avg == 'y':
        if comp == 1:
            q_2d_y, q_2d_yy = np.empty((xpt, zpt)), np.empty((xpt, ypt, zpt))
            q_2d_py, q_2d_pyy = np.empty((xpt, zpt)), np.empty((xpt, ypt, zpt))
        else:
            q_2d_y, q_2d_yy = np.empty((comp, xpt, zpt)), np.empty((comp, xpt, ypt, zpt))
            q_2d_py, q_2d_pyy = np.empty((comp, xpt, zpt)), np.empty((comp, xpt, ypt, zpt))
        # Calculate the average in Y direction
        for i in range(xpt):
            for k in range(zpt):
                if comp == 1:
                    q_2d_y[i, k] = np.nanmean(q_3d[i, :, k])
                else:
                    for cc in range(comp):
                        q_2d_y[cc, i, k] = np.nanmean(q_3d[cc, i, :, k])

        # Calculate prime2mean
        for i in range(xpt):
            for k in range(zpt):
                if comp == 1:
                    q_2d_py[i, k] = np.nanmean((q_3d[i, :, k] - q_2d_y[i, k]) ** 2)
                else:
                    for cc in range(comp):
                        q_2d_py[cc, i, k] = np.nanmean((q_3d[cc, i, :, k] - q_2d_y[cc, i, k]) ** 2)
        for jy in range(ypt):
            if comp == 1:
                q_2d_yy[:, jy, :] = 1 * q_2d_y
                q_2d_pyy[:, jy, :] = 1 * q_2d_py
            else:
                for cc in range(comp):
                    q_2d_yy[cc, :, jy, :] = 1 * q_2d_y[cc, :, :]
                    q_2d_pyy[cc, :, jy, :] = 1 * q_2d_py[cc, :, :]

    if avg == 'z':
        if comp == 1:
            q_2d_y, q_2d_yy = np.empty((xpt, ypt)), np.empty((xpt, ypt, zpt))
            q_2d_py, q_2d_pyy = np.empty((xpt, ypt)), np.empty((xpt, ypt, zpt))
        else:
            q_2d_y, q_2d_yy = np.empty((comp, xpt, ypt)), np.empty((comp, xpt, ypt, zpt))
            q_2d_py, q_2d_pyy = np.empty((comp, xpt, ypt)), np.empty((comp, xpt, ypt, zpt))
        # Calculate the average in Z direction
        for i in range(xpt):
            for j in range(ypt):
                if comp == 1:
                    q_2d_y[i, j] = np.nanmean(q_3d[i, j, :])
                else:
                    for cc in range(comp):
                        q_2d_y[cc, i, j] = np.nanmean(q_3d[cc, i, j, :])

        # Calculate prime2mean
        for i in range(xpt):
            for j in range(ypt):
                if comp == 1:
                    q_2d_py[i, j] = np.nanmean((q_3d[i, j, :] - q_2d_y[i, j]) ** 2)
                else:
                    for cc in range(comp):
                        q_2d_py[cc, i, j] = np.nanmean((q_3d[cc, i, j, :] - q_2d_y[cc, i, j]) ** 2)
        for kz in range(zpt):
            if comp == 1:
                q_2d_yy[:, :, kz] = 1 * q_2d_y
                q_2d_pyy[:, :, kz] = 1 * q_2d_py
            else:
                for cc in range(comp):
                    q_2d_yy[cc, :, :, kz] = 1 * q_2d_y[cc, :, :]
                    q_2d_pyy[cc, :, :, kz] = 1 * q_2d_py[cc, :, :]

    return q_2d_yy, q_2d_pyy
