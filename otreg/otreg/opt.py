import numpy as np
import numba as nb 

@nb.njit(fastmath=True, cache=True)
def closest_y_opt(x, Y, psi, p=2):
    """
    Find the closest y value in Y to the given x value.

    Parameters:
    x: numpy.ndarray
        Input x value.
    Y: numpy.ndarray
        Array of y values.
    psi: numpy.ndarray
        Array of psi values.
    p: int, optional
        Power parameter (default: 2).

    Returns:
    tuple
        The minimum cost value and the index of the closest y value.
    """
    m = Y.shape[0]
    min_val = np.inf
    min_index = 0

    for j in range(m):
        cost_xy = (x - Y[j]) ** p - psi[j]

        if cost_xy < min_val:
            min_val = cost_xy
            min_index = j

    return min_val, min_index


@nb.njit(cache=True, fastmath=True)
def solve_opt(X, Y, lambda_, p=2):
    """
    Solve the optimization problem.

    Parameters:
    X: numpy.ndarray
        Data matrix X.
    Y: numpy.ndarray
        Data matrix Y.
    lambda_: float
        Lambda parameter.
    p: int, optional
        Power parameter (default: 2).

    Returns:
    tuple
        The objective value, phi, psi, piRow, and piCol.
    """
    n, m = X.shape[0], Y.shape[0]
    phi = np.full(shape=n, fill_value=-np.inf)
    psi = np.full(shape=m, fill_value=lambda_)
    piRow = np.full(n, -1, dtype=np.int64)
    piCol = np.full(m, -1, dtype=np.int64)
    K = 0
    dist = np.full(n, np.inf)
    jLast = -1

    while K < n:
        x = X[K]
        if jLast == -1:
            val, j = closest_y_opt(x, Y, psi, p)
        else:
            val, j = closest_y_opt(x, Y[jLast:], psi[jLast:], p)
            j += jLast

        if val >= lambda_:
            phi[K] = lambda_
            K += 1
        elif piCol[j] == -1:
            piCol[j] = K
            piRow[K] = j
            phi[K] = val
            K += 1
            jLast = j
        else:
            phi[K] = val
            dist[K] = 0.
            dist[K - 1] = 0.
            v = 0.
            iMin = K - 1
            jMin = j
            if phi[K] > phi[K - 1]:
                lamDiff = lambda_ - phi[K]
                lamInd = K
            else:
                lamDiff = lambda_ - phi[K - 1]
                lamInd = K - 1
            resolved = False

            while not resolved:
                if jMin > 0:
                    lowEndDiff = (X[iMin] - Y[jMin - 1]) ** p - phi[iMin] - psi[jMin - 1]
                    if iMin > 0:
                        if piRow[iMin - 1] == -1:
                            lowEndDiff = np.inf
                else:
                    lowEndDiff = np.inf

                if j < m - 1:
                    hiEndDiff = (X[K] - Y[j + 1]) ** p - phi[K] - psi[j + 1] - v
                else:
                    hiEndDiff = np.inf

                if hiEndDiff <= min((lowEndDiff, lamDiff)):
                    v += hiEndDiff
                    for i in range(iMin, K):
                        phi[i] += v - dist[i]
                        psi[piRow[i]] -= v - dist[i]
                    
                    phi[K] += v
                    piRow[K] = j + 1
                    piCol[j + 1] = K
                    jLast = j + 1
                    resolved = True

                elif lowEndDiff <= min((hiEndDiff, lamDiff)):
                    if piCol[jMin - 1] == -1:
                        v += lowEndDiff

                        for i in range(iMin, K):
                            phi[i] += v - dist[i]
                            psi[piRow[i]] -= v - dist[i]

                        phi[K] += v
                        jPrime = jMin
                        piCol[jMin - 1] = iMin
                        piRow[iMin] -= 1

                        for i in range(iMin + 1, K):
                            piCol[jPrime] += 1
                            piRow[i] -= 1
                            jPrime += 1
                        
                        piRow[K] = j
                        piCol[j] += 1
                        resolved = True
                    else:
                        v += lowEndDiff
                        dist[iMin - 1] = v
                        lamDiff -= lowEndDiff
                        iMin -= 1
                        jMin -= 1

                        if lambda_ - phi[iMin] < lamDiff:
                            lamDiff = lambda_ - phi[iMin]
                            lamInd = iMin

                else:
                    v += lamDiff
                    for i in range(iMin, K):
                        phi[i] += v - dist[i]
                        psi[piRow[i]] -= v - dist[i]

                    phi[K] += v

                    if lamInd < K:
                        jPrime = piRow[lamInd]
                        piRow[lamInd] = -1

                        for i in range(lamInd + 1, K):
                            piCol[jPrime] += 1
                            piRow[i] -= 1
                            jPrime += 1
                        
                        piRow[K] = j
                        piCol[j] += 1
                    resolved = True

            K += 1

    objective = np.sum(phi) + np.sum(psi)
    return objective, phi, psi, piRow, piCol
