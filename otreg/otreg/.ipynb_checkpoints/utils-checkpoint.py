import numpy as np
import numba as nb
from scipy import linalg

@nb.njit(['float64[:](float64[:,:])'], fastmath=True)
def calculate_vector_mean(input_array_X):
    """
    Calculates the mean of each column in the input array X.

    Parameters:
    input_array_X: numpy.ndarray
        Two-dimensional input array.

    Returns:
    mean_vector: numpy.ndarray
        The mean value of each column in the input array.
    """
    
    # Check if input array has correct dimensions
    if input_array_X.ndim != 2:
        raise ValueError("Input array must be two-dimensional.")

    _ , n_dims = input_array_X.shape

    # Initialize the mean vector with zeros
    mean_vector = np.zeros(n_dims, dtype=np.float64)

    # Calculate the mean of each column in the input array
    for dim_index in nb.prange(n_dims):
        mean_vector[dim_index] = input_array_X[:, dim_index].mean()

    return mean_vector



@nb.njit(fastmath=True)
def recover_alpha(Phi, y_prime, eps=1e-4):
    """
    Recover alpha coefficients using the given Phi matrix and y_prime vector.

    Parameters:
    Phi: numpy.ndarray
        Matrix Phi.
    y_prime: numpy.ndarray
        Vector y_prime.
    eps: float, optional
        Small constant to ensure matrix invertibility (default: 1e-4).

    Returns:
    numpy.ndarray
        The recovered alpha coefficients.
    """
    n, d = Phi.shape
    return np.linalg.inv(Phi.T.dot(Phi) + eps * np.eye(d)).dot(Phi.T.dot(y_prime))

@nb.njit(['Tuple((float64[:,:],float64))(float64[:,:],float64[:,:])'])
def recover_rotation(X, Y):
    """
    Recovers rotation matrix and scaling factor from two input arrays X and Y.

    Parameters:
    X, Y: numpy.ndarray
        Two-dimensional input arrays.

    Returns:
    corrected_rot_mat, scaling_factor: tuple
        The recovered rotation matrix and scaling factor.
    """
    # Check if input arrays have correct dimensions
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("Input arrays must be two-dimensional.")

    _, n_dims = X.shape

    # Center the input arrays
    centered_X = X - calculate_vector_mean(X)
    centered_Y = Y - calculate_vector_mean(Y)

    # Compute the dot product of the transposed centered Y and centered X
    YX_T = centered_Y.T.dot(centered_X)

    # Compute the singular value decomposition of the YX_T
    U, singular_vals, VT = np.linalg.svd(YX_T)

    # Compute the initial rotation matrix
    rot_mat = U.dot(VT)

    # Correct the rotation matrix for possible reflection
    correction_diag = np.eye(n_dims, dtype=np.float64)
    correction_diag[n_dims-1, n_dims-1] = np.linalg.det(rot_mat.T)
    corrected_rot_mat = U.dot(correction_diag).dot(VT)

    # Compute the scaling factor
    scaling_factor = np.sum(np.abs(singular_vals.T)) / np.trace(centered_Y.T.dot(centered_Y))

    return corrected_rot_mat, scaling_factor

@nb.njit(cache=True)
def cost_fn(x, y, p=2):
    """
    Compute the cost function value between x and y.

    Parameters:
    x: float
        Input x value.
    y: float
        Input y value.
    p: int, optional
        Power parameter (default: 2).

    Returns:
    float
        The computed cost function value.
    """
    return np.abs(x - y) ** p


@nb.njit(['float64[:,:](int64,int64,int64)'], fastmath=True, cache=True)
def generate_rand_projections(d, n_projections, mode=0):
    """
    Generate random projection matrices.

    Parameters:
    d: int
        Dimensionality of the projection.
    n_projections: int
        Number of projection matrices to generate.
    mode: int, optional
        Mode of generation (0 or 1) (default: 0).

    Returns:
    numpy.ndarray
        Random projection matrices.
    """
    if mode == 0:
        Gaussian_vector = np.random.normal(0, 1, size=(d, n_projections))
        projections = Gaussian_vector / np.sqrt(np.sum(np.square(Gaussian_vector), 0))
        projections = projections.T
    elif mode == 1:
        r = np.int64(n_projections / d) + 1
        projections = np.zeros((d * r, d))
        for i in range(r):
            H = np.random.randn(d, d)
            Q, R = np.linalg.qr(H)
            projections[i * d: (i + 1) * d] = Q
        projections = projections[0: n_projections]

    return projections


@nb.njit(['int64[:](int64[:],int64[:],int64[:])'], cache=True)
def recover_indices(indice_X, indice_Y, L):
    """
    Recover indices from indice_X, indice_Y, and L arrays.

    Parameters:
    indice_X: numpy.ndarray
        Array of X indices.
    indice_Y: numpy.ndarray
        Array of Y indices.
    L: numpy.ndarray
        Array L.

    Returns:
    numpy.ndarray
        The recovered indices.
    """
    n = L.shape[0]
    indice_Y_mapped = np.where(L >= 0, indice_Y[L], -1)
    mapping = np.stack((indice_X, indice_Y_mapped))
    mapping_final = mapping[1].take(mapping[0].argsort())
    return mapping_final


@nb.njit(['int64[:](int64,int64)'], fastmath=True, cache=True)
def arange(start, end):
    """
    Generate an array of integers from start (inclusive) to end (exclusive).

    Parameters:
    start: int
        Starting value.
    end: int
        Ending value.

    Returns:
    numpy.ndarray
        The generated array of integers.
    """
    n = end - start
    L = np.zeros(n, np.int64)

    for i in range(n):
        L[i] = i + start

    return L


def TPS_recover_parameter(Phi_T, X_bar, Y, epsilon):
    """
    Recover TPS parameters from Phi_T, X_bar, Y, and epsilon.

    Parameters:
    Phi_T: numpy.ndarray
        Transposed Phi array.
    X_bar: numpy.ndarray
        X_bar array.
    Y: numpy.ndarray
        Y array.
    epsilon: float
        Epsilon value.

    Returns:
    tuple
        The recovered alpha and B parameters.
    """
    n, d = X_bar.shape
    n, K = Phi_T.shape
    diag_M = np.zeros((n, K))
    np.fill_diagonal(diag_M, epsilon)
    M = Phi_T + diag_M
    Q, R0 = linalg.qr(X_bar)
    Q1, Q2 = Q[:, 0:d], Q[:, d:n]
    R = R0[0:d, :]
    alpha = Q2.dot(np.linalg.inv(Q2.T.dot(M).dot(Q2))).dot(Q2.T).dot(Y)
    B = np.linalg.inv(R).dot(Q1.T).dot(Y - M.dot(alpha))
    Yhat = Phi_T.dot(alpha) + X_bar.dot(B)
    return alpha, B
