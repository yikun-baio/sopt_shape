# # -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 11:32:17 2022
@author: Yikun Bai yikun.bai@Vanderbilt.edu 
"""

import numpy as np
import torch
import os

import numba as nb
from typing import Tuple #,List
#from numba.typed import List


global p
p=2

@nb.njit(cache=True,fastmath=False,parallel=True)
def cost_matrix(X,Y):
    '''
    input: 
        X: (n,) float np array
        Y: (m,) float np array
    output:
        M: n*m matrix, M_ij=c(X_i,Y_j) where c is defined by cost_function.
    
    '''
#    XT=np.expand_dims(X,1)
    n,m=X.shape[0],Y.shape[0]
    M=np.zeros((n,m))
    for i in nb.prange(n):
        for j in nb.prange(m):
            M[i,j]=(X[i]-Y[j])**p   
    return M


@nb.njit(fastmath=True,cache=True)
def cost_matrix_d(X,Y):
    '''
    input: 
        X: (n,) float np array
        Y: (m,) float np array
    output:
        M: n*m matrix, M_ij=c(X_i,Y_j) where c is defined by cost_function.
    
    '''
    n,d=X.shape
    m,d=Y.shape
    M=np.empty((n,m))
    for i in nb.prange(n):
        M[i]=np.sum((X[i]-Y)**p,1)
    return M






@nb.njit(fastmath=True,cache=True)
def arange(start,end):
    n=end-start
    L=np.zeros(n,np.int64)
    for i in range(n):
        L[i]=i+start
    return L



@nb.njit(cache=True)
def unassign_y(L1):
    '''
    Parameters
    ----------
    L1 : n*1 list , whose entry is 0,1,2,...... 
            transporportation plan. L[i]=j denote we assign x_i to y_j, L[i]=-1, denote we destroy x_i. 
            if we ignore -1, L1 must be in increasing order 
            make sure L1 do not have -1 and is not empty, otherwise there is mistake in the main loop.  


    Returns
    -------
    i_act: integer>=0 
    j_act: integer>=0 or -1    
    j_act=max{j: j not in L1, j<L1[end]} If L1[end]=-1, there is a bug in the main loop. 
    i_act=min{i: L[i]>j_act}.
    
    Eg. input: L1=[1,3,5]
    return: 2,4
    input: L1=[2,3,4]
    return: 0,1
    input: L1=[0,1,2,3]
    return: 0,-1
    
    '''
    n=L1.shape[0]
    j_last=L1[n-1]
    i_last=L1.shape[0]-1 # this is the value of k-i_start
    for l in range(n):
        j=j_last-l
        i=i_last-l+1
        if j > L1[n-1-l]:
            return i,j
    j=j_last-n
    if j>=0:
        return 0,j
    else:       
        return 0,-1



@nb.njit(cache=True)
def unassign_y_nb(L1):
    '''
    Parameters
    ----------
    L1 : n*1 list , whose entry is 0,1,2,...... 
            transporportation plan. L[i]=j denote we assign x_i to y_j, L[i]=-1, denote we destroy x_i. 
            if we ignore -1, L1 must be in increasing order 
            make sure L1 do not have -1 and is not empty, otherwise there is mistake in the main loop.  


    Returns
    -------
    i_act: integer>=0 
    j_act: integer>=0 or -1    
    j_act=max{j: j not in L1, j<L1[end]} If L1[end]=-1, there is a bug in the main loop. 
    i_act=min{i: L[i]>j_act}.
    
    Eg. input: L1=[1,3,5]
    return: 2,4
    input: L1=[2,3,4]
    return: 0,1
    input: L1=[0,1,2,3]
    return: 0,-1
    
    '''
    
    j_last=L1[-1]
    n=L1.shape[0]
    L_range=arange(j_last-n+1,j_last+1)
    L_dif=np.where(L_range-L1>0)[0]
    if L_dif.shape[0]==0:
        return 0, L1[0]-1
    else:
        i_act=L_dif[-1]+1
        j_act=L_range[i_act-1]
    return i_act,j_act



@torch.jit.script   
def recover_indice_T(indice_X,indice_Y,L):
    '''
    input:
        indice_X: n*1 float torch tensor, whose entry is integer 0,1,2,....
        indice_Y: m*1 float torch tensor, whose entry is integer 0,1,2,.... 
        L: n*1 list, whose entry could be 0,1,2,... and -1.
        L is the original transportation plan for sorted X,Y 
        L[i]=j denote x_i->y_j and L[i]=-1 denote we destroy x_i. 
        If we ignore -1, it must be in increasing order  
    output:
        mapping_final: the transportation plan for original unsorted X,Y
        
        Eg. X=[2,1,3], indice_X=[1,0,2]
            Y=[3,1,2], indice_Y=[1,2,0]
            L=[0,1,2] which means the mapping 1->1, 2->2, 3->3
        return: 
            L=[2,1,0], which also means the mapping 2->2, 1->1,3->3.
    
    '''
    device=indice_X.device.type
    n=L.shape[0]
#    indice_Y_mapped=torch.tensor([indice_Y[i] if i>=0 else -1 for i in L],device=device)
    indice_Y_mapped=torch.where(L>=0,indice_Y[L],-1).to(device) 
    mapping=torch.stack([indice_X,indice_Y_mapped])
    mapping_final=mapping[1].take(mapping[0].argsort())
    return mapping_final



@nb.njit(['int64[:](int64[:],int64[:],int64[:])'],cache=True)
def recover_indice(indice_X,indice_Y,L):
    '''
    input:
        indice_X: n*1 float torch tensor, whose entry is integer 0,1,2,....
        indice_Y: m*1 float torch tensor, whose entry is integer 0,1,2,.... 
        L: n*1 list, whose entry could be 0,1,2,... and -1.
        L is the original transportation plan for sorted X,Y 
        L[i]=j denote x_i->y_j and L[i]=-1 denote we destroy x_i. 
        If we ignore -1, it must be in increasing order  
    output:
        mapping_final: the transportation plan for original unsorted X,Y
        
        Eg. X=[2,1,3], indice_X=[1,0,2]
            Y=[3,1,2], indice_Y=[1,2,0]
            L=[0,1,2] which means the mapping 1->1, 2->2, 3->3
        return: 
            L=[2,1,0], which also means the mapping 2->2, 1->1,3->3.
    
    '''
    n=L.shape[0]
    indice_Y_mapped=np.where(L>=0,indice_Y[L],-1)
    mapping=np.stack((indice_X,indice_Y_mapped))
    mapping_final=mapping[1].take(mapping[0].argsort())
    return mapping_final


@nb.njit(fastmath=True,cache=True)
def closest_y_M(M):
    '''
    Parameters
    ----------
    x : float number, xk
    Y : m*1 float np array, 

    Returns
    -------
    min_index : integer >=0
        argmin_j min(x,Y[j])  # you can also return 
    min_cost : float number 
        Y[min_index]

    '''
    n,m=M.shape
    argmin_Y=np.zeros(n,np.int64)
    for i in range(n):
        argmin_Y[i]=M[i,:].argmin()
    return argmin_Y


@nb.njit(['int64[:,:](int64[:],int64)'],fastmath=True,cache=True)
def array_to_matrix(L,m):
    '''
    Parameters
    ----------
    L : n*1 tensor, whose entries is 0,1,2,.... or -1
    
    m : integer >=0 
    
    Returns
    -------
    plan : n*m matrix
    plan[i,j]=1 if L[i]=j and j>=0
    otherwise, plan[i,j]=0
 

    '''
    n=L.shape[0]
    plan=np.zeros((n,m),np.int64)
    
    Ly=L[L>=0]
    Lx=arange(0,n)
    Lx=Lx[L>=0]
    for i in Lx:
        plan[i,L[i]]=1
    return plan

@nb.njit(['int64[:](int64[:,:])'],fastmath=True,cache=True)
#@nb.njit(fastmath=True)
def L_to_pi(L_lp):
    '''
    Parameters
    ----------
    L : n*1 tensor, whose entries is 0,1,2,.... or -1
    
    m : integer >=0 
    
    Returns
    -------
    plan : n*m matrix
    plan[i,j]=1 if L[i]=j and j>=0
    otherwise, plan[i,j]=0
 

    '''
    n,m=L_lp.shape
    L=np.full(n,-1,np.int64)
    for i in range(n):
        indexes=np.where(L_lp[i,:]>=0.5)[0]
        if indexes.shape[0]==1:
            L[i]=indexes[0]
        elif indexes.shape[0]>=2:
            print('error')
    return L


@nb.njit(['(float64[:])(float64[:],float64[:],int64)'],fastmath=True,cache=True)
def Gaussian_mixture(mu_list, variance_list,n):
    N=mu_list.shape[0]
    indices=np.random.randint(0,N,n)
    X=np.zeros(n)
    for i in range(n):
        X[i]=np.random.normal(mu_list[indices[i]],variance_list[indices[i]])
    return X

@nb.njit(['(float32[:])(float32[:],float32[:],int64)'],fastmath=True,cache=True)
def Gaussian_mixture_32(mu_list, variance_list,n):
    N=mu_list.shape[0]
    indices=np.random.randint(0,N,n)
    X=np.zeros(n,dtype=np.float32)
    for i in range(n):
        X[i]=np.float32(np.random.normal(mu_list[indices[i]],variance_list[indices[i]]))
    return X


def opt_pr(mu, nu, M, mass, **kwargs):
    """
    Solves the partial optimal transport problem for the quadratic cost
    and returns the OT plan

    The function considers the following problem:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F

    .. math::
        s.t. \ \gamma \mathbf{1} &\leq \mathbf{a}

             \gamma^T \mathbf{1} &\leq \mathbf{b}

             \gamma &\geq 0

             \mathbf{1}^T \gamma^T \mathbf{1} = m &\leq \min\{\|\mathbf{a}\|_1, \|\mathbf{b}\|_1\}


    where :

    - :math:`\mathbf{M}` is the metric cost matrix
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target unbalanced distributions
    - `m` is the amount of mass to be transported

    Parameters
    ----------
    a : np.ndarray (dim_a,)
        Unnormalized histogram of dimension `dim_a`
    b : np.ndarray (dim_b,)
        Unnormalized histograms of dimension `dim_b`
    M : np.ndarray (dim_a, dim_b)
        cost matrix for the quadratic cost
    m : float, optional
        amount of mass to be transported
    nb_dummies : int, optional, default:1
        number of reservoir points to be added (to avoid numerical
        instabilities, increase its value if an error is raised)
    log : bool, optional
        record log if True
    **kwargs : dict
        parameters can be directly passed to the emd solver


    .. warning::
        When dealing with a large number of points, the EMD solver may face
        some instabilities, especially when the mass associated to the dummy
        point is large. To avoid them, increase the number of dummy points
        (allows a smoother repartition of the mass over the points).


    Returns
    -------
    gamma : (dim_a, dim_b) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary returned only if `log` is `True`


    Examples
    --------
    import ot
    >>> a = [.1, .2]
    >>> b = [.1, .1]
    >>> M = [[0., 1.], [2., 3.]]
    >>> np.round(partial_wasserstein(a,b,M), 2)
    array([[0.1, 0. ],
           [0. , 0.1]])
    >>> np.round(partial_wasserstein(a,b,M,m=0.1), 2)
    array([[0.1, 0. ],
           [0. , 0. ]])

    References
    ----------
    ..  [28] Caffarelli, L. A., & McCann, R. J. (2010) Free boundaries in
        optimal transport and Monge-Ampere obstacle problems. Annals of
        mathematics, 673-730.
    ..  [29] Chapel, L., Alaya, M., Gasso, G. (2020). "Partial Optimal
        Transport with Applications on Positive-Unlabeled Learning".
        NeurIPS.

    See Also
    --------
    ot.partial.partial_wasserstein_lagrange: Partial Wasserstein with
    regularization on the marginals
    ot.partial.entropic_partial_wasserstein: Partial Wasserstein with a
    entropic regularization parameter
    """
    
    Lambda,A=1.0,1.0
    n,m=M.shape 
    mu1,nu1=np.zeros(n+1),np.zeros(m+1)
    mu1[0:n],nu1[0:m]=mu,nu
    mu1[-1],nu1[-1]=np.sum(nu)-mass,np.sum(mu)-mass
    M1=np.zeros((n+1,m+1),dtype=np.float64)
    M1[0:n,0:m]=M
    M1[:,m],M1[n,:]=Lambda,Lambda
    M1[n,m]=2*Lambda+A



    # plan1, cost1, u, v, result_code = emd_c(mu1, nu1, M1, numItermax, numThreads)
    # result_code_string = check_result(result_code)
    gamma1=ot.lp.emd(mu1,nu1,M1,**kwargs)
    gamma=gamma1[0:n,0:m]
    cost=np.sum(M*gamma)

    return cost,gamma

    
    