import time
import numpy as np
import numba as nb
import torch
from tqdm import tqdm
from scipy import linalg
from typing import List, Optional, Tuple

from opt import solve_opt
from kernels import TPS_kernel_matrix, Gaussian_kernel_matrix
from utils import calculate_vector_mean, generate_rand_projections, recover_indices, recover_alpha, recover_rotation


class OTReg:
    """
    Class for Optimal Transport Registration of two point clouds.
    """

    def __init__(self, src_pts: np.ndarray, tar_pts: np.ndarray):
        """
        Constructor for the OTReg class.

        Parameters:
        src_pts (np.ndarray): Source point cloud as an NxD numpy array, where N is the number of points and D is the dimension.
        tar_pts (np.ndarray): Target point cloud as an MxD numpy array, where M is the number of points and D is the dimension.
        """
        self.src_pts = src_pts
        self.tar_pts = tar_pts
        self.n_src_pts = len(src_pts)

    def register(self, method: str = 'TPS', **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Main function to perform registration.

        Parameters:
        method (str, optional): The method to use for registration. Default is 'TPS'.
        **kwargs: Additional parameters to pass to the registration function.

        Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]: Params and list of estimated points
        """
        if method == 'TPS':
            return TPS(self.src_pts, self.tar_pts, self.n_src_pts, **kwargs)
        elif method == 'Gaussian':
            return Gaussian(self.src_pts, self.tar_pts, self.n_src_pts, **kwargs)
        else:
            raise NotImplementedError("Not implemented yet.")


def adjust_lambda(lambda_: float,
                  delta_: float,
                  epoch: int,
                  n_src_pts: int,
                  n_tar_pts: int,
                  lower_bound: float) -> Tuple[float, float]:
    """
    Adjusts the values of lambda and delta parameters.

    Parameters:
    lambda_ (float): The regularization parameter. It controls the trade-off between the data-fidelity 
                     term and the regularization term in the optimal transport problem.
    delta_ (float): The adjustment rate for the lambda parameter.
    epoch (int): The current epoch or iteration.
    n_src_pts (int): The number of source points.
    n_tar_pts (int): The number of target points.
    lower_bound (float): The minimum value that delta can take.

    Returns:
    Tuple[float, float]: The updated values of lambda and delta.
    """

    n_src_pts_adjusted = (n_tar_pts - n_src_pts) / (1 + np.log((n_tar_pts - n_src_pts + 1)/ 1) * (epoch / 500)) + n_src_pts
    mass_diff = n_tar_pts - n_src_pts_adjusted

    # Adjust lambda and delta based on the mass difference
    if mass_diff > n_src_pts_adjusted * 0.009:
        lambda_ -= delta_
    if mass_diff < -n_src_pts_adjusted * 0.003:
        lambda_ += delta_
        delta_ = lambda_ * 1/8
    if lambda_ < delta_:
        lambda_ = delta_
        delta_ *= 1/2
    if delta_ < lower_bound:
        delta_ = lower_bound

    return lambda_ , delta_


def TPS(src_pts: np.ndarray,
        tar_pts: np.ndarray,
        n_src_pts: int,
        n_projections: int = 100,
        eps: float = 5e-4,
        record_indices: List[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    with TPS kernel

    Parameters:
    src_pts (np.ndarray): The source points.
    tar_pts (np.ndarray): The target points.
    n_src_pts (int): The number of source points.
    n_projections (int, optional): The number of projections. Default is 100.
    eps (float, optional): A small value used for stability in computations. Default is 5e-4.
    record_indices (List[int], optional): List of indices at which to record estimated points.

    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]: Returns the kernel matrix (the matrix of pairwise 
    TPS kernel values between control points and source points), control point weights (weights of the control points 
    in the TPS transformation), the affine transformation matrix (rotation and translation), and a list of estimated 
    points at specified epochs.
    """
        
    n_tar_pts, dim_ = src_pts.shape
    #n_src_pts = n_tar_pts - 1
    
    ctrl_pts = src_pts.copy()
    ker_mat = TPS_kernel_matrix(ctrl_pts, src_pts, dim_)
    ext_src_pts = np.hstack((np.ones((n_tar_pts, 1)), src_pts))
    
    rot_mat = np.eye(dim_)
    b = np.zeros(3)
    
    ctrl_pt_wts = np.zeros((ctrl_pts.shape[0], dim_))
    affine = np.vstack((b, rot_mat))
    
    log_b = np.log((n_tar_pts - n_src_pts + 1) / 1)
    lambda_ = 60 * np.sum((tar_pts.mean(0) - src_pts.mean(0)) ** 2)
    delta_ = lambda_ / 8
    lower_bound = lambda_ / 10000
    
    projections = generate_rand_projections(dim_, n_projections, 1)
    est_pts = ker_mat.dot(ctrl_pt_wts) + ext_src_pts.dot(affine)
    
    domain = np.arange(n_tar_pts)
    current_lambda = lambda_
    
    est_pts_list = []
    
    for epoch, projection in tqdm(enumerate(projections), total=n_projections, ncols=70):
        est_pts_proj = np.dot(projection, est_pts.T)
        tar_pts_proj = np.dot(projection, tar_pts.T)

        # Get indices of sorted projected points
        est_pts_proj_sorted_indices = est_pts_proj.argsort()
        tar_pts_proj_sorted_indices = tar_pts_proj.argsort()
        
        est_pts_proj_sorted = est_pts_proj[est_pts_proj_sorted_indices]
        tar_pts_proj_sorted = tar_pts_proj[tar_pts_proj_sorted_indices]
        
        # Compute OT
        _, _, _, piRow, _ = solve_opt(est_pts_proj_sorted, tar_pts_proj_sorted, lambda_, 2)

        # Recover original indices and update domain
        domain = recover_indices(est_pts_proj_sorted_indices, tar_pts_proj_sorted_indices, piRow)
        domain = domain[domain >= 0]

        # Move selected estimated points
        est_pts[domain] += np.expand_dims(tar_pts_proj[domain] - est_pts_proj[domain], axis=1) * projection

        # Find optimal ctrl_pt_wts, affine
        ker_mat_selected, ext_src_pts_selected, est_pts_selected = ker_mat[domain], ext_src_pts[domain], est_pts[domain]
        ctrl_pt_wts, affine = TPS_recover_parameter(ker_mat_selected, ext_src_pts_selected, est_pts_selected, eps)

        # Update selected points
        est_pts = ker_mat.dot(ctrl_pt_wts) + ext_src_pts.dot(affine)
    
        n_src_pts = len(domain)
        
        # Adjust lambda
        lambda_, delta_ = adjust_lambda(lambda_, delta_, epoch, n_src_pts, n_tar_pts, lower_bound)

        # record estimated points in the process
        if record_indices and (epoch in record_indices or epoch == n_projections - 1):
            est_pts_list.append(est_pts.copy())
        
    return ker_mat, ctrl_pt_wts, affine, est_pts_list

        
        
def Gaussian(src_pts: np.ndarray,
                  tar_pts: np.ndarray,
                  n_src_pts: int,
                  n_proj: int = 50,
                  sigma2: float = 1e-4,
                  record_indices: List[int] = None) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    with Gaussian kernel

    Parameters:
    src_pts (np.ndarray): The source points.
    tar_pts (np.ndarray): The target points.
    n_src_pts (int): The number of source points.
    n_proj (int, optional): The number of projections. Default is 50.
    sigma2 (float, optional): Gaussian kernel parameter. Default is 1e-4.
    record_indices (List[int], optional): List of indices at which to record estimated points.

    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]: Returns the Gaussian kernel matrix 
    (the matrix of pairwise Gaussian kernel values between control points and source points), control point weights 
    (weights of the control points in the Gaussian transformation), the scaling factor (factor by which the source 
    points are scaled), the rotation matrix (used to rotate the source points), the shift vector (used to translate 
    the source points), and a list of estimated points at specified epochs.
    """
    
    n_tar_pts, dim_ = src_pts.shape
    ctrl_pts = src_pts.copy()
    ker_mat = Gaussian_kernel_matrix(ctrl_pts, src_pts, sigma2)

    # Initialize parameters
    rot_mat = np.eye(dim_)
    scale = 1.0
    beta = calculate_vector_mean(tar_pts) - calculate_vector_mean(scale * src_pts.dot(rot_mat))
    ctrl_pt_wts = np.zeros((ctrl_pts.shape[0], dim_))

    projections = generate_rand_projections(dim_, n_proj, 1)

    log_b = np.log((n_tar_pts - n_src_pts + 1) / 1)
    lambda_ = 60 * np.sum((calculate_vector_mean(tar_pts) - calculate_vector_mean(src_pts)) ** 2)
    delta_ = lambda_ / 8
    lower_bound = lambda_ / 10000

    est_pts = ker_mat.dot(ctrl_pt_wts) + scale * src_pts.dot(rot_mat) + beta

    domain_org = np.arange(n_tar_pts)
    domain = domain_org.copy()

    est_pts_list = []
    
    for epoch, projection in tqdm(enumerate(projections), total=n_proj, ncols=70):
        est_pts_proj = np.dot(projection, est_pts.T)
        tar_pts_proj = np.dot(projection, tar_pts.T)

        est_proj_sorted_indices = est_pts_proj.argsort()
        tar_proj_sorted_indices = tar_pts_proj.argsort()
        
        est_pts_proj_sorted = est_pts_proj[est_proj_sorted_indices]
        tar_pts_proj_sorted = tar_pts_proj[tar_proj_sorted_indices]

        _, _, _, piRow, _ = solve_opt(est_pts_proj_sorted, tar_pts_proj_sorted, lambda_, 2)

        domain = recover_indices(est_proj_sorted_indices, tar_proj_sorted_indices, piRow)
        domain = domain_org[domain >= 0]

        est_pts[domain] += np.expand_dims(tar_pts_proj[domain] - est_pts_proj[domain], axis=1) * projection

        est_pts_selected = est_pts[domain] - ker_mat[domain].dot(ctrl_pt_wts)
        rot_mat, scale = recover_rotation(est_pts_selected, src_pts[domain])
        beta = calculate_vector_mean(est_pts_selected) - calculate_vector_mean(src_pts[domain].dot(rot_mat) * scale)

        y_prime = est_pts[domain] - src_pts[domain].dot(rot_mat) - beta
        ctrl_pt_wts = recover_alpha(ker_mat[domain], y_prime)

        est_pts = ker_mat.dot(ctrl_pt_wts) + scale * src_pts.dot(rot_mat) + beta

        n_src_pts = len(domain)
        lambda_, delta_ = adjust_lambda(lambda_, delta_, epoch, n_src_pts, n_tar_pts, lower_bound)

        if record_indices and (epoch in record_indices or epoch == n_proj - 1):
            est_pts_list.append(est_pts.copy())

    return ker_mat, ctrl_pt_wts, scale, rot_mat, beta, est_pts_list
