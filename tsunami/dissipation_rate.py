import numpy as np
from tsunami import depth_average
from tsunami import span_average
from tsunami import field_avg


def epsilon(xpt, ypt, zpt, lev_3d, mask_3d, x_3d, y_3d, z_3d, v3d):
    c_3d = 1 * mask_3d
    c_3d[lev_3d < 0], c_3d[lev_3d >= 0] = 0, 1
    gradU = np.zeros((9, xpt, ypt, zpt))  # velocity gradient
    S = np.zeros((6, xpt, ypt, zpt))  # strain tensor
    # epsilon = np.zeros((xpt, ypt, zpt))  # dissipation rate based on instantaneous velocity
    for ipt in range(xpt):
        for jpt in range(ypt):
            gradU[2, ipt, jpt, :] = np.gradient(v3d[0, ipt, jpt, :], z_3d[ipt, jpt, :], edge_order=2)
            gradU[5, ipt, jpt, :] = np.gradient(v3d[1, ipt, jpt, :], z_3d[ipt, jpt, :], edge_order=2)
            gradU[8, ipt, jpt, :] = np.gradient(v3d[2, ipt, jpt, :], z_3d[ipt, jpt, :], edge_order=2)

    for ipt in range(xpt):
        for kpt in range(zpt):
            gradU[1, ipt, :, kpt] = np.gradient(v3d[0, ipt, :, kpt], y_3d[ipt, :, kpt], edge_order=2)
            gradU[4, ipt, :, kpt] = np.gradient(v3d[1, ipt, :, kpt], y_3d[ipt, :, kpt], edge_order=2)
            gradU[7, ipt, :, kpt] = np.gradient(v3d[2, ipt, :, kpt], y_3d[ipt, :, kpt], edge_order=2)

    for jpt in range(ypt):
        for kpt in range(zpt):
            gradU[0, :, jpt, kpt] = np.gradient(v3d[0, :, jpt, kpt], x_3d[:, jpt, kpt], edge_order=2)
            gradU[3, :, jpt, kpt] = np.gradient(v3d[1, :, jpt, kpt], x_3d[:, jpt, kpt], edge_order=2)
            gradU[6, :, jpt, kpt] = np.gradient(v3d[2, :, jpt, kpt], x_3d[:, jpt, kpt], edge_order=2)
    S[0, :, :] = 1 * gradU[0, :, :, :]
    S[1, :, :] = 0.5 * (gradU[1, :, :, :] + gradU[3, :, :, :])
    S[2, :, :] = 0.5 * (gradU[2, :, :, :] + gradU[6, :, :, :])
    S[3, :, :] = 1 * gradU[4, :, :, :]
    S[4, :, :] = 0.5 * (gradU[5, :, :, :] + gradU[7, :, :, :])
    S[5, :, :] = 1 * gradU[8, :, :, :]

    epsilon = 2 * 1e-6 * (S[0, :, :] * S[0, :, :] + 2 * S[1, :, :] * S[1, :, :] +
                          2 * S[2, :, :] * S[2, :, :] + S[3, :, :] * S[3, :, :] +
                          2 * S[4, :, :] * S[4, :, :] + S[5, :, :] * S[5, :, :])
    # # dissipation rate based on instantaneous velocity

    epsi_2d_z, con_z, depth_2d = depth_average(lev_3d, z_3d, epsilon, xpt, ypt, zpt, c_3d, 1)
    epsi_1d, con_y = field_avg(1 * epsi_2d_z, 1, xpt, ypt, zpt, 'y')

    return epsi_1d[:, 0, 0]
