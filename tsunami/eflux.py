import numpy as np
from tsunami import depth_average
from tsunami import span_average
from tsunami import field_avg


def eflux(xpt, ypt, zpt, lev_3d, mask_3d, y_3d, z_3d, v3d):

    c_3d = 1 * mask_3d
    c_3d[lev_3d < 0], c_3d[lev_3d >= 0] = 0, 1
    flux3d = np.zeros((9, xpt, ypt, zpt))
    for i in range(3):
        flux3d[i, :, :, :] = 0.5 * v3d[i, :, :, :] * v3d[3, :, :, :] * v3d[3, :, :, :]  # Kinetic flux

    flux3d[3:6, :, :, :] = 9.81 * z_3d * v3d[0:3, :, :, :]  # Potential flux
    flux3d[6:9, :, :, :] = flux3d[0:3, :, :, :] + flux3d[3:6, :, :, :]
    #ef = np.zeros((3, xpt, ypt, zpt))
    # ef[0, :, :, :], ef[1, :, :, :], ef[2, :, :, :] = pf, kf, tf  # pf
    flux_2d_z, con_z, depth_2d = depth_average(lev_3d, z_3d, flux3d, xpt, ypt, zpt, c_3d, 9)
    eflux_1d, con_y = field_avg(1 * flux_2d_z, 9, xpt, ypt, zpt, 'y')
    #eflux_1d, con_y = span_average(y_3d, 1 * flux_2d_z, lev_3d, xpt, ypt, zpt, con_z, 9)
    #eflux_data = np.zeros((xpt, 4))
    # Potential flux

    return eflux_1d[:, :, 0, 0]
