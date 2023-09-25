# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 11:58:08 2022
@author: Yikun Bai yikun.bai@Vanderbilt.edu 
"""
import os
import numpy as np
from typing import Tuple
import torch
from scipy.stats import ortho_group
import sys
import numba as nb
from .library import *
from .lib_ot import *


# Use numba, can not write general form 
@nb.njit(fastmath=True,cache=True)
def random_projections(d,n_projections,Type=0,dtype=np.float64,seed=-1):
    '''
    input: 
    d: int 
    n_projections: int

    output: 
    projections: d*n torch tensor

    '''
    if seed>0:
        np.random.seed(seed)
    if Type==0:
        Gaussian_vector=np.random.normal(0,1,size=(d,n_projections)).astype(dtype)
        projections=Gaussian_vector/np.sqrt(np.sum(np.square(Gaussian_vector),0))
        projections=projections.T

    elif Type==1:
        r=np.int64(n_projections/d)+1
        projections=np.zeros((d*r,d),dtype)
        for i in range(r):
            H=np.random.randn(d,d).astype(dtype)
            Q,R=np.linalg.qr(H)
            projections[i*d:(i+1)*d]=Q
        projections=projections[0:n_projections]
    return projections





@nb.njit(['Tuple((float64[:],int64[:,:]))(float64[:,:],float64[:,:],float64[:])'],parallel=True,fastmath=True,cache=True)
def opt_plans(X_sliced,Y_sliced,Lambda_list):
    N,n=X_sliced.shape
#    Dtype=type(X_sliced[0,0])
    plans=np.zeros((N,n),np.int64)
    costs=np.zeros(N,np.float64)
    for i in nb.prange(N):
        X_theta=X_sliced[i]
        Y_theta=Y_sliced[i]
        Lambda=Lambda_list[i]
        # M=cost_matrix(X_theta,Y_theta)
        obj,phi,psi,piRow,piCol=opt1d(X_theta,Y_theta,Lambda)
        cost=obj
        L=piRow
        plans[i]=L
        costs[i]=cost
    return costs,plans





@nb.njit(['(float64[:,:],float64[:,:],float64[:,:],float64[:])'],cache=True)
def X_correspondence(X,Y,projections,Lambda_list):
    N,d=projections.shape
    n=X.shape[0]
    Lx_org=arange(0,n)
    for i in range(N):
        theta=projections[i]
        X_theta=np.dot(theta,X.T)
        Y_theta=np.dot(theta,Y.T)
        X_indice=X_theta.argsort()
        Y_indice=Y_theta.argsort()
        X_s=X_theta[X_indice]
        Y_s=Y_theta[Y_indice]
        Lambda=Lambda_list[i]
        # M=cost_matrix(X_s,Y_s)
        obj,phi,psi,piRow,piCol=opt1d(X_s,Y_s,Lambda)
#        Cost,L=o(X_s,Y_s,Lambda)
        
        L=piRow
        L=recover_indice(X_indice,Y_indice,L)
        #move X
        Lx=Lx_org.copy()
        Lx=Lx[L>=0]
        if Lx.shape[0]>=1:
            Ly=L[L>=0]
#            dim=Ly.shape[0]
            X_take=X_theta[Lx]
            Y_take=Y_theta[Ly]
            X[Lx]+=np.expand_dims(Y_take-X_take,1)*theta
            


 

@nb.njit(['(float64[:,:],float64[:,:],float64[:,:])'],cache=True)
def X_correspondence_pot(X,Y,projections):
    N,d=projections.shape
    n=X.shape[0]
    for i in range(N):
        theta=projections[i]
        X_theta=np.dot(theta,X.T)
        Y_theta=np.dot(theta,Y.T)
        X_indice=X_theta.argsort()
        Y_indice=Y_theta.argsort()
        X_s=X_theta[X_indice]
        Y_s=Y_theta[Y_indice]
        # M=cost_matrix(X_s,Y_s)
        cost,L=pot(X_s,Y_s)
        L=recover_indice(X_indice,Y_indice,L)
        X_take=X_theta
        Y_take=Y_theta[L]
        X+=np.expand_dims(Y_take-X_take,1)*theta
    return X


    

@nb.njit(parallel=True,fastmath=True,cache=True)
def opt_plans(X,Y,projections,Lambda_list):
    n,d=X.shape
    n_projections=Lambda_list.shape[0]
    X_projections=projections.dot(X.T)
    Y_projections=projections.dot(Y.T)
    opt_plan_X_list=np.zeros((n_projections,n),dtype=np.int64)
    opt_cost_list=np.zeros(n_projections)
    for (epoch,(X_theta,Y_theta,Lambda)) in enumerate(zip(X_projections,Y_projections,Lambda_list)):
        X_indice=X_theta.argsort()
        Y_indice=Y_theta.argsort()
        X_s=X_theta[X_indice]
        Y_s=Y_theta[Y_indice]
        obj,phi,psi,piRow,piCol=opt1d(X_s,Y_s,Lambda)
        L1=recover_indice(X_indice,Y_indice,piRow)
        opt_cost_list[epoch]=obj
        opt_plan_X_list[epoch]=L1
    return opt_plan_X_list,X_projections,Y_projections






            


        
   
        

    

    
    
    
    


    
    
    
    
        
    

    
    




    


