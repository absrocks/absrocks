import numpy as np
def str_array(xpt, ypt, zpt, q, comp):
    if comp > 0:
        q_3d = np.zeros((comp, xpt, ypt, zpt))
        for ic in range(comp):
            l = 0
            for kz in range(zpt):
                for jy in range(ypt):
                    for ix in range(xpt):
                        q_3d[ic, ix, jy, kz] = q[ic, l]
                        l = l + 1
    else:
        q_3d = np.zeros((xpt, ypt, zpt))
        l = 0
        for kz in range(zpt):
            for jy in range(ypt):
                for ix in range(xpt):
                    q_3d[ix, jy, kz] = q[l]
                    l = l + 1

    return q_3d
