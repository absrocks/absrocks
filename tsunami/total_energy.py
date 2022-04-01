import numpy as np
from tsunami import depth_average
from tsunami import field_avg


# from tsunami import str_array


def te(xpt, ypt, zpt, lev_3d, mask_3d, z_3d, vel_3d):
    # print("minimum of x is",x_3d.min(), min(x))
    c_3d = 1 * mask_3d
    c_3d[lev_3d < 0], c_3d[lev_3d >= 0] = 0, 1
    ef_2d_z, con_z, depth_2d = depth_average(lev_3d, z_3d, vel_3d[0, :, :, :], xpt, ypt, zpt, c_3d, 1)
    vel_3d[0, :, :, :][lev_3d < 0], vel_3d[1, :, :, :][lev_3d < 0], vel_3d[2, :, :, :][lev_3d < 0] = np.nan, np.nan, np.nan
    #V_3d = np.zeros((3, xpt, ypt, zpt))
    vel_3d_zavg = field_avg(vel_3d, 4, xpt, ypt, zpt, 'z')[0]
    vavg = field_avg(vel_3d_zavg, 4, xpt, ypt, zpt, 'y')[0]
    #energy_data = np.zeros((xpt, 4))
    ke = (vavg[0, :, 0, 0] * vavg[0, :, 0, 0] + vavg[1, :, 0, 0] * vavg[1, :, 0, 0] + vavg[2, :, 0, 0] * vavg[2, :, 0, 0]) \
         / (2 * 9.81)
    depth_1d = field_avg(depth_2d, 1, xpt, ypt, zpt, 'y')[0]
    te = depth_1d[:, 0, 0] + ke
    return depth_1d[:, 0, 0], ke, te, vavg[0, :, 0, 0]
