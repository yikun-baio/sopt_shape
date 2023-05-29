import numpy as np
import numba as nb

@nb.njit()
def Gaussian_kernel(r2, sigma2):
    """
    Compute the Gaussian kernel value.

    Parameters:
    r2: float
        Squared distance parameter.
    sigma2: float
        Squared bandwidth parameter.

    Returns:
    float
        The computed Gaussian kernel value.
    """
    return np.exp(-r2/sigma2)


@nb.njit()
def TPS_kernel_2d(r2):
    """
    Compute the 2D Thin Plate Spline (TPS) kernel value.

    Parameters:
    r2: float
        Squared distance parameter.

    Returns:
    float
        The computed 2D TPS kernel value.
    """
    return 1/2 * r2 * np.log(r2 + 1e-10)

@nb.njit()
def TPS_kernel_3d(r2):
    """
    Compute the 3D Thin Plate Spline (TPS) kernel value.

    Parameters:
    r2: float
        Squared distance parameter.

    Returns:
    float
        The computed 3D TPS kernel value.
    """
    return np.sqrt(r2)

@nb.njit(fastmath=True)
def Gaussian_kernel_matrix(c, x, sigma2):
    """
    Compute the Gaussian kernel matrix.

    Parameters:
    c: numpy.ndarray
        Centers matrix.
    x: numpy.ndarray
        Data matrix.
    sigma2: float
        Squared bandwidth parameter.

    Returns:
    numpy.ndarray
        The computed Gaussian kernel matrix.
    """
    K, d = c.shape
    n = x.shape[0]
    
    diff_matrix = np.expand_dims(x, 1) - np.expand_dims(c, 0)
    r2 = np.sum(np.square(diff_matrix), axis=2)
    Phi = Gaussian_kernel(r2, sigma2)
    return Phi


@nb.njit(fastmath=True)
def TPS_kernel_matrix(c, x, D):
    """
    Compute the Thin Plate Spline (TPS) kernel matrix.

    Parameters:
    c: numpy.ndarray
        Centers matrix.
    x: numpy.ndarray
        Data matrix.
    D: int
        Dimension parameter (2 for 2D TPS, 3 for 3D TPS).

    Returns:
    numpy.ndarray
        The computed TPS kernel matrix.
    """
    K, d = c.shape
    n = x.shape[0]
    
    diff_matrix = np.expand_dims(x, 1) - np.expand_dims(c, 0)
    r2 = np.sum(np.square(diff_matrix), axis=2)
    Phi = np.zeros((n,d))
    if D == 2:
        Phi = TPS_kernel_2d(r2)
    else:
        Phi = TPS_kernel_3d(r2)
    return Phi