# Calculate average in Y direction:
# f_avg = sum(f(y))/ypt where ypt = number of points along Y direction

import numpy as np


def ensemble_avg(q_3d, xpt, ypt, zpt):
    q_ens_avg = np.empty((xpt, ypt, zpt))
    for ipt in range(xpt):
        for jpt in range(ypt):
            for kpt in range(zpt):
                q_ens_avg[ipt, jpt, kpt] = np.nanmean(q_3d[:, ipt, jpt, kpt])

    return q_ens_avg
