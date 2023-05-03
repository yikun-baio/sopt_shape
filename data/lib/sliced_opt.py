# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 11:58:08 2022

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



def random_projections_T(d,n_projections,Type): #,device='cpu',dtype=torch.float,Type=None):
    '''
    input: 
    d: int 
    n_projections: int

    output: 
    projections: d*n torch tensor

    '''
    if Type==0:
#        torch.manual_seed(0)
        Gaussian_vector=np.random.normal(0,1,size=[d,n_projections],device=device,dtype=dtype)
        projections=Gaussian_vector/np.sqrt(np.sum(np.square(Gaussian_vector),0))
        projections=projections.T
    elif Type==1:
#        np.random.seed(0)
        r=int(n_projections/d)+1
        projections=np.concatenate([ortho_group.rvs(d) for i in range(r)],axis=1)
        projections=projections[0:n_projections]
#        projections=torch.from_numpy(projections).to(device=device).to(dtype=dtype).T
    else:
        print('Type must be None or orth')
    return projections




@nb.njit(['float64[:,:](int64,int64,int64)'],fastmath=True)
def random_projections(d,n_projections,Type=0):
    '''
    input: 
    d: int 
    n_projections: int

    output: 
    projections: (d,n) numpy array

    '''
#    np.random.seed(0)
    if Type==0:
        Gaussian_vector=np.random.normal(0,1,size=(d,n_projections)) #.astype(np.float64)
        projections=Gaussian_vector/np.sqrt(np.sum(np.square(Gaussian_vector),0))
        projections=projections.T

    elif Type==1:
        r=np.int64(n_projections/d)+1
        projections=np.zeros((d*r,d)) #,dtype=np.float64)
        for i in range(r):
            H=np.random.randn(d,d) #.astype(np.float64)
            Q,R=np.linalg.qr(H)
            projections[i*d:(i+1)*d]=Q
        projections=projections[0:n_projections]
    return projections


@nb.njit(['float32[:,:](int64,int64,int64)'],fastmath=True)
def random_projections_32(d,n_projections,Type=0):
    '''
    generate n_projection 
    
    input: 
    d: int 
    n_projections: int

    output: 
    projections: (d,n) numpy array, float32 

    '''
    np.random.seed(0)
    if Type==0:
        Gaussian_vector=np.random.normal(0,1,size=(d,n_projections)).astype(np.float32) #.astype(np.float64)
        projections=Gaussian_vector/np.sqrt(np.sum(np.square(Gaussian_vector),0))
        projections=projections.T

    elif Type==1:
        r=np.int64(n_projections/d)+1
        projections=np.zeros((r*d,d),dtype=np.float32)
        for i in range(r):
            H=np.random.randn(d,d).astype(np.float32)
            Q,R=np.linalg.qr(H)
            projections[i*d:(i+1)*d]=Q
        projections=projections[0:n_projections]
    return projections


#@nb.njit([nb.types.Tuple((nb.float64[:],nb.int64[:,:]))(nb.float64[:,:],nb.float64[:,:],nb.float64)],parallel=True,fastmath=True)
@nb.njit(['Tuple((float64[:],int64[:,:]))(float64[:,:],float64[:,:],float64[:])'],parallel=True,fastmath=True)
def allplans_s(X_sliced,Y_sliced,Lambda_list):
    """
    get all 1-D OPT distance and plans for all sliced X,Y 
    parameters: 
    ------
    X_sliced: numpy array, (N,n), float64 
    Y_sliced: numpy array, (N,n), float64 
    Lambda_list: numpy array, (N,), float64
    
    returns: 
    --------
    costs: numpy array, (N,) float64 
    plans: numpy array, (N,n) float64 
    """
    N,n=X_sliced.shape
#    Dtype=type(X_sliced[0,0])
    plans=np.zeros((N,n),np.int64)
    costs=np.zeros(N,np.float64)
    for i in nb.prange(N):
        X_theta=X_sliced[i]
        Y_theta=Y_sliced[i]
        Lambda=Lambda_list[i]
        M=cost_matrix(X_theta,Y_theta)
        obj,phi,psi,piRow,piCol=solve_opt(M,Lambda)
        L=piRow
        cost=obj
        plans[i]=L
        costs[i]=cost
    return costs,plans



@nb.njit(['(float64[:,:],float64[:,:],float64[:,:],float64[:])'])
def X_correspondence(X,Y,projections,Lambda_list):
    """
    move X based on sliced optimal partial transport between X and Y 
    
    Parameters: 
    X: numpy array (n,d) float64 
    Y: numpy array (n,d) float64
    projections: numpy array (n_projections, d)  float64
    lambda_list: numpy array (n_projections,)  float64
    """
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
        M=cost_matrix(X_s,Y_s)
        obj,phi,psi,piRow,piCol=solve_opt(M,Lambda)
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
            
@nb.njit(['(float32[:,:],float32[:,:],float32[:,:],float32[:])'])
def X_correspondence_32(X,Y,projections,Lambda_list):
    """
    move X based on sliced optimal partial transport between X and Y 
    
    Parameters: 
    X: numpy array (n,d) float32 
    Y: numpy array (n,d) float32
    projections: numpy array (n_projections, d)  float32
    lambda_list: numpy array (n_projections,)  float32
    """
    
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
        M=cost_matrix(X_s,Y_s)
        obj,phi,psi,piRow,piCol=solve_opt_32(M,Lambda)
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

 

@nb.njit(['(float64[:,:],float64[:,:],float64[:,:])'])
def X_correspondence_pot(X,Y,projections):
    
    """
    move X based on sliced partial optimal transport between X and Y 
    
    Parameters: 
    X: numpy array (n,d) float64 
    Y: numpy array (n,d) float64
    projections: numpy array (n_projections, d)  float64
    lambda_list: numpy array (n_projections,)  float64
    """
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
        cost,L=pot(X_s,Y_s)
        L=recover_indice(X_indice,Y_indice,L)
        X_take=X_theta
        Y_take=Y_theta[L]
        X+=np.expand_dims(Y_take-X_take,1)*theta
    return X

@nb.njit(['(float32[:,:],float32[:,:],float32[:,:])'])
def X_correspondence_pot_32(X,Y,projections):
    """
    move X based on sliced partial optimal transport between X and Y 
    
    Parameters: 
    X: numpy array (n,d) float64 
    Y: numpy array (n,d) float64
    projections: numpy array (n_projections, d)  float64
    lambda_list: numpy array (n_projections,)  float64
    """    
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
        cost,L=pot_32(X_s,Y_s)
        L=recover_indice(X_indice,Y_indice,L)
        X_take=X_theta
        Y_take=Y_theta[L]
        X+=np.expand_dims(Y_take-X_take,1)*theta
    return X

    
    
class sopt():    
    def __init__(self,X,Y,Lambda_list,n_projections,Type=1):
        self.X=X
        self.Y=Y
        self.device=X.device.type
        self.dtype=X.dtype
        self.n,self.d=X.shape
        self.m=Y.shape[0]
        self.n_projections=n_projections
        self.Lambda_list=Lambda_list
        self.Type=Type

    def sliced_cost(self,penulty=False):
        cost=self.refined_cost(self.X_sliced,self.Y_sliced,self.plans,penulty)
        mass=torch.sum(self.plans>=0)/self.plans.shape[0]
        return cost,mass
    
    def get_directions(self):
        projections=random_projections_32(self.d,self.n_projections,1) #,self.device,self.dtype)
        self.projections=torch.from_numpy(projections).to(self.device).to(self.dtype)
#        self.projections=random_projections_T(self.d,self.n_projections,self.device,self.dtype)

    def get_all_projections(self):
        self.X_sliced=torch.matmul(self.projections,self.X.T)
        self.Y_sliced=torch.matmul(self.projections,self.Y.T)
        
    def get_one_projection(self,i):
        self.X_sliced=torch.matmul(self.projections[i],self.X.T).unsqueeze(0)
        self.Y_sliced=torch.matmul(self.projections[i],self.Y.T).unsqueeze(0)

    def get_plans(self):
        X_sliced_s,indices_X=self.X_sliced.detach().sort()
        Y_sliced_s,indices_Y=self.Y_sliced.detach().sort()
        X_sliced_np=X_sliced_s.cpu().numpy()
        Y_sliced_np=Y_sliced_s.cpu().numpy()
#        Lambda_list_np=Lambda_list.numpy()
        self.costs,plans=allplans_s(X_sliced_np,Y_sliced_np,self.Lambda_list.numpy())
        plans=torch.from_numpy(plans).to(device=self.device,dtype=torch.int64)
        self.plans=recover_indice_M(indices_X,indices_Y,plans)
        self.costs=torch.from_numpy(self.costs)
#       self.X_frequency=torch.sum(self.plans>=0,0)
    
    def max_plan(self):
        self.get_directions()
        self.get_all_projections()
        self.get_plans()
        self.i_max=self.costs.argmax()
        self.L_max=self.plans[self.i_max]
        #self.Lx_max=torch.arange(self.n)
        #self.Lx_max=self.Lx_max[self.L_max>=0]

        

    def refined_cost(self,Xs,Ys,plans,penulty=True):
        N=Xs.shape[0]
        self.Lx=[torch.arange(self.n,device=self.device)[plans[i]>=0] for i in range(N)]
        self.mass_list=[torch.sum(plans[i]>=0) for i in range(N)]
        self.mass_list=torch.tensor(self.mass_list,dtype=torch.float64)
        self.Ly=[plans[i][plans[i]>=0] for i in range(N)]
        self.X_take=torch.cat([Xs[i][self.Lx[i]] for i in range(N)])
        self.Y_take=torch.cat([Ys[i][self.Ly[i]] for i in range(N)])        
        cost_trans=torch.sum(cost_function_T(self.X_take, self.Y_take))
  #       self.mass=[torch.sum(plans[i][plans[i]>=0]) for i in range(N)]
  #       self.mass=torch.cat(self.mass)
        penulty_value=torch.dot(self.Lambda_list,self.n-self.mass_list)
        if penulty==True:
            return (cost_trans+penulty_value)/N    
        elif penulty==False:
            return cost_trans/N



        
        
class sopt_correspondence(sopt):
    def __init__(self,X,Y,Lambda_list,N_projections=20,Type=None):
        sopt.__init__(self,X,Y,Lambda_list,N_projections,Type)
        self.Xc=self.X.clone()
        #X_correspondence(self.X.numpy(),self.Y.numpy(),self.projections.numpy())

    def correspond(self,mass=-1,b=np.float64(0)):
        if self.X.shape[0]>0:
            if mass<0:
                mass=self.n
            self.X_frequency=X_correspondence_32(self.X.numpy(),self.Y.numpy(),self.projections.numpy(),self.Lambda_list)


    def transform(self,Xs,batch_size=128):    
        #D0 = cost_matrix_T(Xs, self.Xc)
        #idx = torch.argmin(D0, axis=1)
        #transp_Xs=Xs+self.X[idx, :]  - self.Xc[idx, :]
        #     #print(transp_Xs)

        # # perform out of sample mapping
        indices = torch.arange(Xs.shape[0])
        batch_ind = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]

        transp_Xs = []

        for bi in batch_ind:
            # get the nearest neighbor in the source domain
            D0 = cost_matrix_T(Xs[bi], self.Xc)
            idx = torch.argmin(D0, axis=1)
            # define the transported points
            transp_Xs_ =Xs[bi]+self.X[idx, :]  - self.Xc[idx, :]
            #print(transp_Xs)
            transp_Xs.append(transp_Xs_)
        transp_Xs = torch.cat(transp_Xs, axis=0)
        return transp_Xs
    
        
        
class spot(sopt_correspondence):
    def __init__(self,X,Y,N_projections=20,Type=None):
        Lambda_list=torch.zeros(N_projections)
        sopt.__init__(self,X,Y,Lambda_list,N_projections,Type)
        self.Xc=self.X.clone()
        
    def correspond(self):    
        if self.X.shape[0]>0:
             X_correspondence_pot_32(self.X.numpy(),self.Y.numpy(),self.projections.numpy())      
    


        
        
   
        
        
        

        
    

    
    
    
    


    
    
    
    
        
    

    
    




    


