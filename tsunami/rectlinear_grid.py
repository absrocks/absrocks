import numpy as np


def regular_grid(x, y, z, xpt, ypt, zpt):
    # generate new grid X,Y,Z
    xi, yi, zi = np.ogrid[min(x):max(x):xpt * 1j, min(y):max(y):ypt * 1j, min(z):max(z):zpt * 1j]
    X1 = xi.reshape(xi.shape[0], )
    Y1 = yi.reshape(yi.shape[1], )
    Z1 = zi.reshape(zi.shape[2], )
    ar_len = len(X1) * len(Y1) * len(Z1)
    X = np.arange(ar_len, dtype=float)
    Y = np.arange(ar_len, dtype=float)
    Z = np.arange(ar_len, dtype=float)
    l = 0
    for i in range(0, len(X1)):
        for j in range(0, len(Y1)):
            for k in range(0, len(Z1)):
                X[l] = X1[i]
                Y[l] = Y1[j]
                Z[l] = Z1[k]
                l = l + 1
    return X, Y, Z, X1, Y1, Z1
