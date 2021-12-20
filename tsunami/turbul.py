import numpy as np

def turbul(xpt, ypt, zpt, v3d):

    P = np.zeros((6, xpt, ypt, zpt))  # turbulence production
    # epsilon = np.zeros((xpt, ypt, zpt))  # dissipation rate based on instantaneous velocity
    P[0, :, :, :] = v3d[0, :, :, :] * v3d[0, :, :, :]
    P[1, :, :, :] = v3d[0, :, :, :] * v3d[1, :, :, :]
    P[2, :, :, :] = v3d[0, :, :, :] * v3d[2, :, :, :]
    P[3, :, :, :] = v3d[1, :, :, :] * v3d[1, :, :, :]
    P[4, :, :, :] = v3d[1, :, :, :] * v3d[2, :, :, :]
    P[5, :, :, :] = v3d[2, :, :, :] * v3d[2, :, :, :]

    return P
