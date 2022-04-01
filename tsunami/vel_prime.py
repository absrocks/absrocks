import numpy as np
from tsunami import field_avg

def primel(xpt, ypt, zpt, vel_3d, lev_3d, mask_3d):

    vel_span_avg = field_avg(vel_3d, 4, xpt, ypt, zpt, 'y')[0]  # spanwise average of instantaneous velocity
    prime_vel_3d = vel_3d - vel_span_avg
    prime_vel_3d[0, :, :, :][lev_3d < 0], prime_vel_3d[1, :, :, :][lev_3d < 0], prime_vel_3d[2, :, :, :][lev_3d < 0] = np.nan, np.nan, np.nan
    k_3d = 0.5 * (prime_vel_3d[0, :, :, :]**2 + prime_vel_3d[1, :, :, :]**2 + prime_vel_3d[2, :, :, :]**2) #Turbulent kinetic energy
    k_avg_y = field_avg(k_3d, 3, xpt, ypt, zpt, 'y')[0]
    k_avg_yz = field_avg(k_3d, 3, xpt, ypt, zpt, 'z')[0]

    return k_3d, k_avg_y, k_avg_yz
