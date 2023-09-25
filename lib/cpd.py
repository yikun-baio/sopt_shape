
import numpy as np
import time
import torch
import matplotlib.pyplot as plt
from .lib_shape import kernel_matrix_gaussian

def is_psd(R):
    return np.all(np.linalg.eigvals(R)>0)

def gaussian_kernel(X, beta, C=None):
    if C is None: C = X.copy()
    diff = X[:, None, :] - C[None, :, :]
    diff = np.square(diff)
    diff = np.sum(diff, 2)
    return np.exp(-diff / (2 * beta**2))

def lr_eig(G, num_eig):
    S, Q = np.linalg.eigh(G)
    eig_idx = list(np.argsort(np.abs(S))[:,:,-1][:num_eig])
    Q = Q[:, eig_idx]
    S = S[eig_idx]
    return Q, S

def calc_stats(P, X ): # X is target
    Pt1 = np.sum(P, axis=0)
    P1 = np.sum(P, axis=1)
    PX = np.matmul(P, X)
    Np = np.sum(P1)
    return Pt1, P1, PX, Np

def Estep(Y,TY, sigma2): # X--> Y  is target, TY, souce, 
    N,D=TY.shape
    M=Y.shape[0]
    P = np.exp(-np.sum((Y[None,:, :] - TY[:, None, :])**2, axis=2) / (2 * sigma2))
    c = (2 * np.pi * sigma2) ** (D / 2) * N / M
    den = np.sum(P, axis=0, keepdims=True) + c
    P = np.divide(P, den)
    return P

def Mstep_rigid(Y,X, P, Pt1, P1, PX, Np, scale , sigma2, eps): # X-->Y, Y-->X 
    N,D=X.shape
    M=Y.shape[0]
    muX = np.divide(np.sum(PX, axis=0), Np)
    muY = np.divide(np.sum(np.dot(np.transpose(P), X), axis=0), Np)
    X_hat = Y - np.tile(muX, (M, 1)) # target
    Y_hat = X - np.tile(muY, (N, 1)) # souce
    YPY = np.dot(np.transpose(P1), np.sum(np.multiply(Y_hat, Y_hat), axis=1))
    A = np.dot(np.transpose(X_hat), np.transpose(P))
    A = np.dot(A, Y_hat)
    U, _, V = np.linalg.svd(A, full_matrices=True)
    C = np.ones((D, ))
    C[D-1] = np.linalg.det(np.dot(U, V))
    R = np.transpose(np.dot(np.dot(U, np.diag(C)), V))
    if scale:
        s = np.trace(np.dot(np.transpose(A), np.transpose(R))) / YPY
    else:
        s=1.0
    t = np.transpose(muX) - s * np.dot(np.transpose(R), np.transpose(muY))
    TY = s * np.dot(X, R) + t
    sigma2 = (np.dot(np.transpose(Pt1), np.sum(np.multiply(X_hat, X_hat), axis=1)) - 2 * s * np.trace(np.dot(A, R))) / (Np * D)
    if sigma2 <= 0:
        sigma2 = eps / 10
    return TY, sigma2, R, s, t  

def Mstep_nrigid(Y,X, P, Pt1, P1, PX, Np, G, alpha, sigma2, eps):
    N,D=X.shape
    A = np.dot(np.diag(P1), G) + alpha * sigma2 * np.eye(N)
    B = PX - np.dot(np.diag(P1), X)
    W = np.linalg.solve(A, B)
    TY = X + np.dot(G, W)
    sigma2 = (np.dot(np.transpose(Pt1), np.sum(np.multiply(Y, Y), axis=1)) - 2 * np.sum(np.multiply(TY, PX)) + np.dot(np.transpose(P1), np.sum(np.multiply(TY, TY), axis=1))) / (Np * D)
    if sigma2 <= 0:
        sigma2 = eps / 10
    return TY, sigma2, W  

def calc_err(TY, Y, clean_maxidx=6890):
    std = np.sqrt(np.sum(Y.std(0)**2))
    return np.sum(((TY[:clean_maxidx,:] - Y[:clean_maxidx,:]) / std) ** 2) / Y.shape[0]

def CPD(X, Y , n_iter_max=100, n_iter_rigid=10, record_index=None,**kwargs):
    (N, D) = X.shape # target 
    (M, _) = Y.shape # source
    TY = X.copy()
    C=X.copy()
    sigma2 = 1 
    eps = 0.001
    scale = False
    alpha = 4
    beta = 4
    # G = gaussian_kernel(X, beta)
    G=kernel_matrix_gaussian(X,C,beta**2)

    params,Yhat_list = [],[]
    if record_index is None:
        record_index=np.unique(np.linspace(0,n_iter_max-1,num=int(20)).astype(np.int64))
        record_index.sort()
    
    if n_iter_rigid is None:
        n_iter_rigid=int(n_iter_max/10)
        
    
    epoch=0
    while epoch<n_iter_rigid:
        P = Estep(Y, TY, sigma2)
        Pt1, P1, PX, Np = calc_stats(P, Y)
        TY, sigma2, R, s, t = Mstep_rigid(Y, X, P, Pt1, P1, PX, Np, scale, sigma2, eps)
        Yhat = s * np.dot(X, R) + t
        if epoch in record_index:
            params.append(('rigid', (R, s, t)))
            Yhat_list.append(TY)
        print("Rigid: ", epoch,end='\r')
        epoch+=1
        
    while epoch< n_iter_max:
        P = Estep(Y, TY, sigma2)
        Pt1, P1, PX, Np = calc_stats(P, Y)
        TY, sigma2, W = Mstep_nrigid(Y,X, P, Pt1, P1, PX, Np, G, alpha, sigma2, eps)
        if epoch in record_index:
            params.append ( ('nonrigid', W))
            Yhat_list.append(TY.copy())
        print("Non-Rigid: ", epoch,end='\r')
        epoch+=1
    
    return (params,C,beta**2),Yhat_list,record_index


def model_to_Yhat_icp(X, param):
    R, t = param
    Yhat = np.dot(X, R.T) + t
    return Yhat

