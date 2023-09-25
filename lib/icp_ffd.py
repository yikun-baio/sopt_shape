import numpy as np
from scipy.special import binom
from scipy.spatial import KDTree
from scipy.linalg import svd
import torch
import matplotlib.pyplot as plt

class FFD:
    def __init__(self, n_ctrl_pts=(20, 20, 20)):
        self.n_ctrl_pts = np.array(n_ctrl_pts)
        self.arr_mu_x = np.zeros(self.n_ctrl_pts)
        self.arr_mu_y = np.zeros(self.n_ctrl_pts)
        self.arr_mu_z = np.zeros(self.n_ctrl_pts)

    def set_displacement(self, displacement, ctrl_pt):
        self.arr_mu_x[ctrl_pt] = displacement[0]
        self.arr_mu_y[ctrl_pt] = displacement[1]
        self.arr_mu_z[ctrl_pt] = displacement[2]

    def get_displacement(self):
        return self.arr_mu_x, self.arr_mu_y, self.arr_mu_z
    
    def bernstein(self, n, i, t):
        return binom(n, i) * (t ** i) * ((1 - t) ** (n - i))

    def T(self, pts):
        Yhat = np.zeros_like(pts)
        for idx, pt in enumerate(pts):
            for i in range(self.n_ctrl_pts[0]):
                for j in range(self.n_ctrl_pts[1]):
                    for k in range(self.n_ctrl_pts[2]):
                        B = (self.bernstein(self.n_ctrl_pts[0] - 1, i, pt[0]) *
                             self.bernstein(self.n_ctrl_pts[1] - 1, j, pt[1]) *
                             self.bernstein(self.n_ctrl_pts[2] - 1, k, pt[2]))
                        Yhat[idx] += B * np.array([self.arr_mu_x[i, j, k],
                                                   self.arr_mu_y[i, j, k],
                                                   self.arr_mu_z[i, j, k]])

        return Yhat

    def deform(self, pts):
        return pts + self.T(pts)

class ICP_FFD:
    def __init__(self, X, Y, n_ctrl_pts=(3, 3, 3), n_iteration=1000, eps=1e-16, record_index=None):
        self.X = X
        self.Y = Y
        self.n_ctrl_pts = n_ctrl_pts
        self.n_iteration = n_iteration
        self.eps = eps
        if record_index is None:
            record_index=np.unique(np.linspace(0,n_iteration-1,num=int(20)).astype(np.int64))
            record_index.sort()
        self.record_index=record_index
            
        # self.record_index = record_index if record_index else []
        self.params = []  


    def find_correspondences(self, X, Y):
        Y_tree = KDTree(Y)
        _, idx = Y_tree.query(X)
        corrs = Y[idx]
        return corrs

    def compute_centroids(self, X, Y):
        X_c = np.mean(X, axis=0)
        Y_c = np.mean(Y, axis=0)
        return X_c, Y_c

    def compute_T(self, X, Y):
        X_c, Y_c = self.compute_centroids(X, Y)
        H = np.dot((X - X_c).T, (Y - Y_c))
        U, _, Vt = svd(H)
        R = np.dot(Vt.T, U.T)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)
        t = Y_c.T - np.dot(R, X_c.T)
        return R, t

    def apply_T(self, X, R, t):
        return np.dot(X, R.T) + t

    def calc_err(self, X, Y):
        return np.mean(np.linalg.norm(X - Y, axis=1))

    def icp(self, X, Y):
        R_total = np.eye(3)  
        t_total = np.zeros(3)  

        for i in range(self.n_iteration):
            corrs = self.find_correspondences(X, Y)
            R, t = self.compute_T(X, corrs)
            X = self.apply_T(X, R, t)

            t_total = np.dot(R, t_total) + t
            R_total = np.dot(R, R_total)

            if i in self.record_index:
                self.params.append((R_total, t_total))
            print('i=%i'%i,end='\r')

        return X, R_total, t_total

    def ffd(self, X, Y):
        ffd = FFD(n_ctrl_pts=self.n_ctrl_pts)

        for i in range(self.n_ctrl_pts[0]):
            for j in range(self.n_ctrl_pts[1]):
                for k in range(self.n_ctrl_pts[2]):
                    ctrl_pt = np.array([(i / (self.n_ctrl_pts[0] - 1)),
                          (j / (self.n_ctrl_pts[1] - 1)),
                          (k / (self.n_ctrl_pts[2] - 1))])

                    nearest_x = X[np.argmin(np.linalg.norm(X - ctrl_pt, axis=1))]
                    nearest_y = Y[np.argmin(np.linalg.norm(Y - ctrl_pt, axis=1))]

                    displacement = nearest_y - nearest_x
                    ffd.set_displacement(displacement, (i, j, k))

        Yhat = ffd.deform(X)

        return Yhat, ffd.get_displacement()

    def run(self):
        Yhat1, R, t = self.icp(self.X, self.Y)
        Yhat2, ffd_params = self.ffd(Yhat1, self.Y)
        self.params.append((R, t, ffd_params,self.n_ctrl_pts))
        return Yhat2, self.params

def model_to_Yhat_icp(X, params):
    R, t = params

    Yhat = np.dot(X, R.T) + t

    return Yhat

def model_to_Yhat_ffd(X, params):
    R, t,ffd_params,n_ctrl_pts = params
    X = np.dot(X, R.T) + t
    arr_mu_x, arr_mu_y, arr_mu_z = ffd_params

    ffd = FFD(n_ctrl_pts=n_ctrl_pts)
    ffd.arr_mu_x, ffd.arr_mu_y, ffd.arr_mu_z = arr_mu_x, arr_mu_y, arr_mu_z
    Yhat = ffd.deform(X)
    return Yhat

