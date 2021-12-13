import numpy as np
from tsunami import depth_average
from tsunami import span_average
from tsunami import field_avg


def flux_dissipation(xpt, ypt, zpt, lev_3d, mask_3d, y_3d, z_3d, v3d):
    c_3d = 1 * mask_3d
    c_3d[lev_3d < 0], c_3d[lev_3d >= 0] = 0, 1
    flux3d = np.zeros((3, xpt, ypt, zpt))

    flux3d[0, :, :, :] = 0.5 * v3d[3, :, :, :] * v3d[3, :, :, :]  # Kinetic flux

    flux3d[1, :, :, :] = 9.81 * z_3d  # Potential flux
    flux3d[2, :, :, :] = flux3d[0, :, :, :] + flux3d[1, :, :, :]

    flux_2d_z, con_z, depth_2d = depth_average(lev_3d, z_3d, flux3d, xpt, ypt, zpt, c_3d, 3)

    eflux_1d, con_y = field_avg(1 * flux_2d_z, 3, xpt, ypt, zpt, 'y')

    return eflux_1d[:, :, 0, 0]
