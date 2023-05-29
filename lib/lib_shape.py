#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 14:25:29 2022

"""
import torch 
import os
import sys
import cupy as cp

from .opt1d import solve 
from .library import *
from .lib_ot import *   
from .sliced_opt import *
from scipy import linalg


#print('hello')
import matplotlib.pyplot as plt

def get_swiss(N=100,a = 4,r_min = 0.1,r_max = 1): 
    """
    generate swiss shape data
    parameters: 
    -------
    N : int64 
    a: float or int64
    r_min: float
    r_max: float

    returns:
    ------
    X: numpy array, shape (N,2), float64 
    """

    theta = np.linspace(0, a * np.pi, N)
    r = np.linspace(r_min, r_max, N)
    X = np.stack([r * np.cos(theta),r * np.sin(theta)],1)
    return X


def rotation_matrix(theta):
    """
    generate (2,2) rotation matrix
    
    Parameter:
    ------
    theta : float
    
    Returns: 
    -------
    torch.tensor shape (2,2) float 
    """
    return torch.stack([torch.cos(theta).reshape([1]),torch.sin(theta).reshape([1]),
                        -torch.sin(theta).reshape([1]),torch.cos(theta).reshape([1])]).reshape([2,2])





def rotation_matrix_3d_x(theta_x):
    """
    generate (3,3) rotation matrix along x-axis 
    
    Parameter:
    -----
    theta: float
    
    Returns: 
    ------
    M: torch.tensor shape (3,3) float 
    """
    device=theta_x.device.type
    rotation_x=torch.zeros((3,3),dtype=torch.float64,device=device)
    rotation_x[1,1]=torch.cos(theta_x)
    rotation_x[1,2]=-torch.sin(theta_x)
    rotation_x[2,1]=torch.sin(theta_x)
    rotation_x[2,2]=torch.cos(theta_x)
    rotation_x[0,0]=1.0
    return rotation_x


def rotation_matrix_3d_y(theta_y):
    """
    generate (3,3) rotation matrix along y-axis 
    
    Parameter:
    -----
    theta: float
    
    Returns: 
    ------
    torch.tensor shape (3,3) float 
    """
        
    device=theta_y.device.type
    rotation_y=torch.zeros((3,3),dtype=torch.float64,device=device)
    rotation_y[0,0]=torch.cos(theta_y)
    rotation_y[0,2]=torch.sin(theta_y)
    rotation_y[2,0]=-torch.sin(theta_y)
    rotation_y[2,2]=torch.cos(theta_y)
    rotation_y[1,1]=1.0
    return rotation_y

def rotation_matrix_3d_z(theta_z):
    """
    generate (3,3) rotation matrix along z-axis 
    
    Parameter:
    -----
    theta: float
    
    Returns: 
    ------
    M: torch.tensor shape (3,3) float 
    """
        
    device=theta_z.device.type
    rotation_z=torch.zeros((3,3),dtype=torch.float64,device=device)
    rotation_z[0,0]=torch.cos(theta_z)
    rotation_z[0,1]=-torch.sin(theta_z)
    rotation_z[1,0]=torch.sin(theta_z)
    rotation_z[1,1]=torch.cos(theta_z)
    rotation_z[2,2]=1.0
    return rotation_z

def rotation_matrix_3d(theta,order='re'):
    
    """
    generate (3,3) rotation matrix 
    
    Parameter:
    -----
    theta: torch tensor (3,) float
    order: string "re" or "in" 
          "in" roation with respect to x-axis, then y-axis, then z-axis
          "re" rotation with rspect to z-axis, then y-axis, then x-axis 
    Returns: 
    ------
    M: torch.tensor shape (3,3) float 
    """
        
    theta_x,theta_y,theta_z=theta
    rotatioin_x=rotation_matrix_3d_x(theta_x)
    rotatioin_y=rotation_matrix_3d_y(theta_y)
    rotatioin_z=rotation_matrix_3d_z(theta_z)
    if order=='in':
        rotation_3d=torch.linalg.multi_dot((rotatioin_z,rotatioin_y,rotatioin_x))
    elif order=='re':
        rotation_3d=torch.linalg.multi_dot((rotatioin_x,rotatioin_y,rotatioin_z))
    return rotation_3d

def rotation_3d_2(theta,order='re'):
    """
    generate (3,3) rotation matrix 
    
    Parameter:
    -----
    theta: torch tensor (3,) float
    order: string "re" or "in" 
          "in" roation with respect to x-axis, then y-axis, then z-axis
          "re" rotation with rspect to z-axis, then y-axis, then x-axis 
    Returns: 
    ------
    M: torch.tensor shape (3,3) float 
    """
        
    cos_x,cos_y,cos_z=np.cos(theta)
    sin_x,sin_y,sin_z=np.sin(theta)

    if order=='re':
        M=rotation_re(cos_x,sin_x,cos_y,sin_y,cos_z,sin_z)
    elif order=='in':
        M=rotation_in(cos_x,sin_x,cos_y,sin_y,cos_z,sin_z)
    return M

def rotation_re(theta):
    """
    generate (3,3) rotation matrix along  
    
    Parameter:
    -----
    cos_x,sin_x: float,float
                cos(x), sin(x) for some angle x
    cos_y,sin_y: float, float
                cos(y), sin(y) for some angle y
    cos_z,sin_z: float, float
                cos(z), sin(z) for some angle z
    
    Returns: 
    ------
    M: torch.tensor shape (3,3) float
             rotation with rspect to z-axis, then y-axis, then x-axis
    """
    cos_x,cos_y,cos_z=np.cos(theta)
    sin_x,sin_y,sin_z=np.sin(theta)
    M=np.zeros((3,3),dtype=np.float64)
    M[0,0]=cos_y*cos_z
    M[0,1]=-cos_y*sin_z
    M[0,2]=sin_y
    M[1,0]=sin_x*sin_y*cos_z+cos_x*sin_z
    M[1,1]=-sin_x*sin_y*sin_z+cos_x*cos_z
    M[1,2]=-sin_x*cos_y
    M[2,0]=-cos_x*sin_y*cos_z+sin_x*sin_z
    M[2,1]=cos_x*sin_y*sin_z+sin_x*cos_z 
    M[2,2]=cos_x*cos_y
    return M

def rotation_in(theta):
    """
    generate (3,3) rotation matrix along  
    
    Parameter:
    -----
    cos_x,sin_x: float,float
                cos(x), sin(x) for some angle x
    cos_y,sin_y: float, float
                cos(y), sin(y) for some angle y
    cos_z,sin_z: float, float
                cos(z), sin(z) for some angle z
    
    Returns: 
    ------
    M: torch.tensor shape (3,3) float
             rotation with rspect to x-axis, then y-axis, then z-axis
    """
    cos_x,cos_y,cos_z=np.cos(theta)
    sin_x,sin_y,sin_z=np.sin(theta)
    
    M=np.zeros((3,3))
    M[0,0]=cos_y*cos_z
    M[0,1]=-cos_x*sin_z+sin_x*sin_y*cos_z
    M[0,2]=sin_x*sin_z+cos_x*sin_y*cos_z
    M[1,0]=cos_y*sin_z
    M[1,1]=cos_x*cos_z+sin_x*sin_y*sin_z
    M[1,2]=-sin_x*cos_z+cos_x*sin_y*sin_z
    M[2,0]=-sin_y
    M[2,1]=sin_x*cos_y
    M[2,2]=cos_x*cos_y
    return M

def rotation_in_T(theta):
    """
    generate (3,3) rotation matrix along  
    
    Parameter:
    -----
    cos_x,sin_x: float,float
                cos(x), sin(x) for some angle x
    cos_y,sin_y: float, float
                cos(y), sin(y) for some angle y
    cos_z,sin_z: float, float
                cos(z), sin(z) for some angle z
    
    Returns: 
    ------
    M: torch.tensor shape (3,3) float
             rotation with rspect to x-axis, then y-axis, then z-axis
    """
    cos_x,cos_y,cos_z=torch.cos(theta)
    sin_x,sin_y,sin_z=torch.sin(theta)
    
    M=torch.zeros((3,3),dtype=torch.float64)
    M[0,0]=cos_y*cos_z
    M[0,1]=-cos_x*sin_z+sin_x*sin_y*cos_z
    M[0,2]=sin_x*sin_z+cos_x*sin_y*cos_z
    M[1,0]=cos_y*sin_z
    M[1,1]=cos_x*cos_z+sin_x*sin_y*sin_z
    M[1,2]=-sin_x*cos_z+cos_x*sin_y*sin_z
    M[2,0]=-sin_y
    M[2,1]=sin_x*cos_y
    M[2,2]=cos_x*cos_y
    return M


def rotation_re_T(theta):
    """
    generate (3,3) rotation matrix along  
    
    Parameter:
    -----
    cos_x,sin_x: float,float
                cos(x), sin(x) for some angle x
    cos_y,sin_y: float, float
                cos(y), sin(y) for some angle y
    cos_z,sin_z: float, float
                cos(z), sin(z) for some angle z
    
    Returns: 
    ------
    M: torch.tensor shape (3,3) float
             rotation with rspect to z-axis, then y-axis, then x-axis
    """
    cos_x,cos_y,cos_z=torch.cos(theta)
    sin_x,sin_y,sin_z=torch.sin(theta)
    M=torch.zeros((3,3),dtype=torch.float64)
    M[0,0]=cos_y*cos_z
    M[0,1]=-cos_y*sin_z
    M[0,2]=sin_y
    M[1,0]=sin_x*sin_y*cos_z+cos_x*sin_z
    M[1,1]=-sin_x*sin_y*sin_z+cos_x*cos_z
    M[1,2]=-sin_x*cos_y
    M[2,0]=-cos_x*sin_y*cos_z+sin_x*sin_z
    M[2,1]=cos_x*sin_y*sin_z+sin_x*cos_z 
    M[2,2]=cos_x*cos_y
    return M


    

    

@nb.njit(['float64[:](float64[:,:])'],fastmath=True)
def vec_mean(X):
    """
    return X.mean(1) 
    
    Parameters:
    ----------
    X: numpy array, shape (n,d), flaot64
    
    Return:
    --------
    mean: numpy array, shape (d,), float64 
    
    
    """
    n,d=X.shape
    mean=np.zeros(d,dtype=np.float64)
    for i in nb.prange(d):
        mean[i]=X[:,i].mean()
    return mean
        

        

    
    
@nb.njit(['Tuple((float64[:,:],float64))(float64[:,:],float64[:,:])'])
def recover_rotation(X,Y):
    """
    return the optimal rotation, scaling based on the correspondence (X,Y) 
    
    Parameters:
    ----------
    X: numpy array, shape (n,d), flaot64, target
    Y: numpy array, shape (n,d), flaot64, source
    
    Return:
    --------
    rotation: numpy array, shape (d,d), float64 
    scaling: float64 
    
    """

        
    n,d=X.shape
    X_c=X-vec_mean(X)
    Y_c=Y-vec_mean(Y)
    YX=Y_c.T.dot(X_c)
    U,S,VT=np.linalg.svd(YX)
    R=U.dot(VT)
    diag=np.eye(d,dtype=np.float64)
    diag[d-1,d-1]=np.linalg.det(R.T)
    rotation=U.dot(diag).dot(VT)
    scaling=np.sum(np.abs(S.T))/np.trace(Y_c.T.dot(Y_c))
    return rotation,scaling


def recover_rotation_cuda(X,Y):
    """
    return the optimal rotation, scaling based on the correspondence (X,Y) 
    
    Parameters:
    ----------
    X: numpy array, shape (n,d), flaot64, target
    Y: numpy array, shape (n,d), flaot64, source
    
    Return:
    --------
    rotation: numpy array, shape (d,d), float64 
    scaling: float64 
    
    """

    X_c,Y_c=cp.array(X),cp.array(Y)
    n,d=X.shape
    X_center,Y_center=X_c-cp.mean(X_c,0),Y_c-cp.mean(Y_c,0)
    
    YX=Y_center.T.dot(X_center)
    U,S,VT=cp.linalg.svd(YX)
    R=U.dot(VT)
    diag=cp.eye(d)
    diag[d-1,d-1]=cp.linalg.det(R.T)
    rotation=U.dot(diag).dot(VT)
    scaling=cp.sum(cp.abs(S.T))/cp.trace(Y_c.T.dot(Y_c))
    return cp.asnumpy(rotation),cp.asnumpy(scaling)





@nb.njit(['Tuple((float64[:,:],float64[:]))(float64[:,:],float64[:,:])'],fastmath=True)
def recover_rotation_du(X,Y):
    """
    return the optimal rotation, scaling based on the correspondence (X,Y) 
    
    Parameters:
    ----------
    X: numpy array, shape (n,d), flaot64, target
    Y: numpy array, shape (n,d), flaot64, source
    
    Return:
    --------
    rotation: numpy array, shape (d,d), float64 
    scaling: numpy array, shape (d,) float64 
    
    """
    
    n,d=X.shape
    X_c=X-vec_mean(X)
    Y_c=Y-vec_mean(Y)
    YX=Y_c.T.dot(X_c)
    U,S,VT=np.linalg.svd(YX)
    R=U.dot(VT)
    diag=np.eye(d,dtype=np.float64)
    diag[d-1,d-1]=np.linalg.det(R)
    rotation=U.dot(diag).dot(VT)
    E_list=np.eye(d,dtype=np.float64)
    scaling=np.zeros(d,dtype=np.float64)
    for i in range(d):
        Ei=np.diag(E_list[i])
        num=0
        denum=0
        for j in range(d):
            num+=X_c[j].T.dot(rotation.T).dot(Ei).dot(Y_c[j])
            denum+=Y_c[j].T.dot(Ei).dot(Y_c[j])
        scaling[i]=num/denum
    return rotation,scaling










# method of spot_boneel 
@nb.njit(['Tuple((float64[:,:,:],float64[:],float64[:,:]))(float64[:,:],float64[:,:],int64,int64)'])
def spot_bonneel(S,T,n_projections=20,n_iterations=200):
    
    '''
    Parameters: 
    ------
    S: (n,d) numpy array, float64
        source data 
    T: (n,d) numpy array, float64
        target data
    n_projections: int64
        number of projections in each iteration 
    n_iterations: int64
        total number of iterations

    
    Returns: 
    -----
    rotation_list: (n_iterations,d,d) numpy array, float64
                  list of rotation matrices in all iterations
    scalar_list: (n_iterations,) numpy array, float64
                  list of scaling parameters in all interations
    beta_list: (n_iterations,d) numpy arrayy, float64 
                  list of translation parameters in all interations 
                      
    '''
        
    
    n,d=T.shape
    N1=S.shape[0]
    # initlize 
    rotation=np.eye(d) #,dtype=np.float64)
    scalar=nb.float64(1.0) #
    beta=vec_mean(T)-vec_mean(scalar*S.dot(rotation))
    #paramlist=[]
    
    rotation_list=np.zeros((n_iterations,d,d)) #.astype(np.float64)
    scalar_list=np.zeros((n_iterations)) #.astype(np.float64)
    beta_list=np.zeros((n_iterations,d)) #.astype(np.float64)
    T_hat=S.dot(rotation)*scalar+beta
    
    #Lx_hat_org=arange(0,n)
    
    for i in range(n_iterations):
#        print('i',i)

        projections=random_projections(d,n_projections,1)
        
# #        print('start1')
        T_hat=X_correspondence_pot(T_hat,T,projections)
        rotation,scalar=recover_rotation(T_hat,S)
        beta=vec_mean(T_hat)-vec_mean(scalar*S.dot(rotation))
        T_hat=S.dot(rotation)*scalar+beta

#         #move That         
        rotation_list[i]=rotation         
        scalar_list[i]=scalar
        beta_list[i]=beta

    return rotation_list,scalar_list,beta_list    




@nb.njit(['Tuple((float64[:,:,:],float64[:],float64[:,:]))(float64[:,:],float64[:,:],int64)'])
def icp_du(S,T,n_iterations):
    '''
    Parameters: 
    ------
    S: (n,d) numpy array, float64
        source data 
    T: (n,d) numpy array, float64
        target data
        
    
    n_iterations: int64
        total number of iterations

    
    Returns: 
    -----
    rotation_list: (n_iterations,d,d) numpy array, float64
                  list of rotation matrices in all iterations
    scalar_list: (n_iterations,) numpy array, float64
                  list of scaling parameters in all interations
    beta_list: (n_iterations,d) numpy arrayy, float64 
                  list of translation parameters in all interations 
                      
    '''
        
    n,d=T.shape

    # initlize 
    rotation=np.eye(d) #,dtype=np.float64)
    scalar=1.0  #nb.float64(1) #
    beta=vec_mean(T)-vec_mean(scalar*np.dot(S,rotation))

    
    
    rotation_list=np.zeros((n_iterations,d,d)) #.astype(np.float64)
    scalar_list=np.zeros((n_iterations)) #.astype(np.float64)
    beta_list=np.zeros((n_iterations,d)) #.astype(np.float64)
    T_hat=np.dot(S,rotation)*scalar+beta
    
    # #Lx_hat_org=arange(0,n)
    
    for i in range(n_iterations):
#        print('i',i)
        M=cost_matrix_d(T_hat,T)
        argmin_T=closest_y_M(M)
        T_take=T[argmin_T]
        T_hat=T_take
        rotation,scalar_d=recover_rotation_du(T_hat,S)
        scalar=np.mean(scalar_d)
        beta=vec_mean(T_hat)-vec_mean(scalar*S.dot(rotation))
        T_hat=S.dot(rotation)*scalar+beta
        
        #move Xhat         
        rotation_list[i]=rotation
        scalar_list[i]=scalar
        beta_list[i]=beta

    return rotation_list,scalar_list,beta_list  



@nb.njit(['Tuple((float64[:,:,:],float64[:],float64[:,:]))(float64[:,:],float64[:,:],int64)'])
def icp_umeyama(S,T,n_iterations):
    '''
    Parameters: 
    ------
    S: (n,d) numpy array, float64
        source data 
    T: (n,d) numpy array, float64
        target data
        
    
    n_iterations: int64
        total number of iterations

    
    Returns: 
    -----
    rotation_list: (n_iterations,d,d) numpy array, float64
                  list of rotation matrices in all iterations
    scalar_list: (n_iterations,) numpy array, float64
                  list of scaling parameters in all interations
    beta_list: (n_iterations,d) numpy arrayy, float64 
                  list of translation parameters in all interations 
                      
    '''
        
    n,d=S.shape

    # initlize 
    rotation=np.eye(d) #,dtype=np.float64)
    scalar=1.0 #nb.float64(1.0) #
    beta=vec_mean(T)-vec_mean(scalar*S.dot(rotation))
    # paramlist=[]
    rotation_list=np.zeros((n_iterations,d,d)) #.astype(np.float64)
    scalar_list=np.zeros((n_iterations)) #.astype(np.float64)
    beta_list=np.zeros((n_iterations,d)) #.astype(np.float64)
    T_hat=S.dot(rotation)*scalar+beta
    

    
    for i in range(n_iterations):
#        print('i',i)
       # print(i)
        M=cost_matrix_d(T_hat,T)
        argmin_T=closest_y_M(M)
        T_take=T[argmin_T]
        T_hat=T_take
        rotation,scalar=recover_rotation(T_hat,S)
        #scalar=np.mean(scalar_d)
        beta=vec_mean(T_hat)-vec_mean(scalar*S.dot(rotation))
        X_hat=S.dot(rotation)*scalar+beta
        
        #move That         
        rotation_list[i]=rotation
        scalar_list[i]=scalar
        beta_list[i]=beta

    return rotation_list,scalar_list,beta_list  





def save_parameter(rotation_list,scalar_list,beta_list,save_path):
    """
    convert parameter list and save as one file 
    
    parameter:
    ------------
    rotation_list: numpy array, shape (n_itetations, d,d), float
    scalar_list: numpy array, shape (n_itetations,), float
    beta_list: numpy array, shape (n_itetations, d), float
    save_path: string 
    """
    paramlist=[]
    N=len(rotation_list)
    for i in range(N):
        param={}
        param['rotation']=rotation_list[i]
        param['beta']=beta_list[i]
        param['scalar']=scalar_list[i]
        paramlist.append(param)
    torch.save(paramlist,save_path)
    #return paramlist

    
    

# visualization 

def get_noise(Y0,Y1):
    """
    get the indices of clean data and noise data of Y1 
    
    Parameters:
    ---------
    Y0: numpy array, shape (N1,d), clean data
    Y1: numpy array, shape (N1+s,d), noisy data, where s>0. Y0 \subset Y1 
    
    Returns: 
    ----------
    np.array(data_indices): numpy array, shape (N1,), int64  
    np.array(noice): numpy array, shape (s,), int64 
    """
    N=Y1.shape[0]
    data_indices=[]
    noise_indices=[]
    for j in range(N):
        yj=Y1[j]
        if yj in Y0:
            data_indices.append(j)
        else:
            noise_indices.append(j)
    return np.array(data_indices),np.array(noise_indices)

def init_image(X_data,X_noise,Y_data,Y_noise,image_path,name):
    """
    make a plot for the data and noise and save the plot 
    parameters: 
    X_data: numpy array, shape (n1,d), float 
        cliean data, target data 
    X_noise: numpy array, shape (s1,d), float 
        cliean data, target data 
    Y_data: numpy array, shape (n2,d), float 
        cliean data, source data 
    Y_noise: numpy array, shape (s2,d), float 
        cliean data, source data
    image_path: string 
    name: string 
        
    """
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')
    ax.scatter(X_data[:,0]+5,X_data[:,1],X_data[:,2]-10,alpha=.5,c='C2',s=2,marker='o')
    ax.scatter(X_noise[:,0]+5,X_noise[:,1],X_noise[:,2]-10,alpha=.5,c='C2',s=10,marker='o')
    ax.scatter(Y_data[:,0]+5,Y_data[:,1],Y_data[:,2]-10,alpha=0.5,c='C1',s=2,marker='o')
    ax.scatter(Y_noise[:,0]+5,Y_noise[:,1],Y_noise[:,2]-10,alpha=.5,c='C1',s=10,marker='o')
    ax.set_facecolor('black') 
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(True)
    ax.axis('off')
    
    # whitch_castle 
    # x+5, z-10
    ax.set_xlim([-38,38])
    ax.set_ylim([-38,38])
    ax.set_zlim([-38,38])
    ax.view_init(45,120)
    
    
    #mumble_sitting 
    # ax.set_xlim([-66,66])
    # ax.set_ylim([-66,66])
    # ax.set_zlim([-66,66])
    # ax.axis('off')
    # ax.view_init(-20,10,'y')

    
    #dragon +bunny     
    # bunny y-0.05,
    # ax.set_xlim([-.25,.25])
    # ax.set_ylim([-.25,.25])
    # ax.set_zlim([-.25,.25])
    # ax.axis('off')
    # ax.view_init( 90, -90)
    

    plt.savefig(image_path+'/'+name+'.png',dpi=200,format='png',bbox_inches='tight')
    plt.show()
    plt.close()
    
    

def normal_image(X_data,X_noise,Y_data,Y_noise,image_path,name):
    
    """
    make a plot for the data and noise and save the plot 
    truncated version 
    parameters: 
    X_data: numpy array, shape (n1,d), float 
        cliean data, target data 
    X_noise: numpy array, shape (s1,d), float 
        cliean data, target data 
    Y_data: numpy array, shape (n2,d), float 
        cliean data, source data 
    Y_noise: numpy array, shape (s2,d), float 
        cliean data, source data
    image_path: string 
    name: string 
        
    """
        
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')
    ax.scatter(X_data[:,0]+3,X_data[:,1],X_data[:,2]-15,alpha=.3,c='C2',s=5,marker='o')
    ax.scatter(X_noise[:,0]+3,X_noise[:,1],X_noise[:,2]-15,alpha=.5,c='C2',s=15,marker='o')
    ax.scatter(Y_data[:,0]+3,Y_data[:,1],Y_data[:,2]-15,alpha=.9,c='C1',s=6,marker='o')
    ax.scatter(Y_noise[:,0]+3,Y_noise[:,1],Y_noise[:,2]-15,alpha=.5,c='C1',s=15,marker='o')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.axis('off')
    ax.set_facecolor('black') 
    ax.grid(True)
    
    # castle,   
    #x+3, z-15 
    ax.set_xlim([-20,20])
    ax.set_ylim([-20,20])
    ax.set_zlim([-20,20])
    ax.view_init(45,120)
#    ax.view_init(0,10,'y')


    # #mumble_sitting, bunny  
    # y-10
    # ax.set_xlim([-36,36])
    # ax.set_ylim([-36,36])
    # ax.set_zlim([-36,36])
    # ax.view_init(-20,10,'y')
    #ax.view_init( 90, -90)
     
    #dragon, bunny 
    #dragon y-0.1
    #bunny, x+0.02, y-0.1    
    # ax.set_xlim([-.1,.1])
    # ax.set_ylim([-.1,.1])
    # ax.set_zlim([-.1,.1])

    # ax.view_init( 90, -90)
    # fig.set_facecolor('black')
    
    plt.savefig(image_path+'/'+name+'.png',dpi=200,format='png',bbox_inches='tight')
    plt.show()
    plt.close()

    

@nb.njit()
def Gaussian_kernel(r2,sigma2):
    return np.exp(-r2/sigma2)

# def Gaussian_kernel(r2,sigma2):
#     return np.exp(-r2/sigma2)


@nb.njit()
def TPS_kernel_2D(r2):
    return 1/2*r2*np.log(r2+1e-10)

@nb.njit()
def TPS_kernel_3D(r2):
    return np.sqrt(r2)



@nb.njit(fastmath=True)
def kernel_matrix_Gaussian(c,x,sigma2): #,Type='Gaussian'):
    '''
    x: (n,d) numpy array
    c: (n,d) numpy array
    type: 0 Gaussian kernel 
    type: 1 SPF kernel 
    '''
    #c1,x1=np.meshgrid(c,x)
    K,d=c.shape
    n=x.shape[0]
    r2=np.zeros((n,K))
    #Phi=np.zeros((n,d))
    for i in range(d):
        r2+=np.square(x[:,i:i+1]-c[:,i])
    if Type=='Gaussian': # Gaussian Kernel 
        Phi=Gaussian_kernel(r2,sigma2)
    if Type=='TPS': # TPS Kernel
        Phi=TPS_kernel_2D(r2)
        #ID=np.isnan(Phi)
        #Phi[ID]=0.0
    return Phi





@nb.njit(fastmath=True)
def kernel_matrix_Gaussian(c,x,sigma2):
    '''
    x: (n,d) numpy array
    c: (n,d) numpy array
    '''
   
    K,d=c.shape
    n=x.shape[0]
    #r2=np.zeros((n,K))
    diff_matrix=np.expand_dims(x,1)-np.expand_dims(c,0)
    r2=np.sum(np.square(diff_matrix),axis=2)
    Phi=np.zeros((n,d))
    Phi=Gaussian_kernel(r2,sigma2)
    return Phi

@nb.njit(fastmath=True)
def kernel_matrix_TPS(c,x,D):
    '''
    x: (n,d) numpy array
    c: (n,d) numpy array
    '''
   
    K,d=c.shape
    n=x.shape[0]
    #r2=np.zeros((n,K))
    diff_matrix=np.expand_dims(x,1)-np.expand_dims(c,0)
    r2=np.sum(np.square(diff_matrix),axis=2)
    Phi=np.zeros((n,d))
    if D==2: 
        Phi=TPS_kernel_2D(r2)
    elif D==3:
        Phi=TPS_kernel_3D(r2)
    return Phi


@nb.njit(fastmath=True)
def recover_alpha(Phi,y_prime,epsilon=1e-4):
    n,d=Phi.shape
    return np.linalg.inv(Phi.T.dot(Phi)+epsilon*np.eye(d)).dot(Phi.T.dot(y_prime))

    
def recover_alpha_cuda(Phi,y_prime,epsilon=1e-4):
    n,d=Phi.shape
    Phi_c=cp.array(Phi)
    y_prime_c=cp.array(y_prime)
    re=cp.linalg.inv(Phi_c.T.dot(Phi_c)+epsilon*cp.eye(d)).dot(Phi_c.T.dot(y_prime_c))
    return cp.asnumpy(re)


#@nb.njit()
def TPS_recover_parameter(Phi_T,X_bar,Y,epsilon):
    n,d=X_bar.shape
    n,K=Phi_T.shape
    diag_M=np.zeros((n,K))
    np.fill_diagonal(diag_M, epsilon)
    M=Phi_T+diag_M
    Q, R0 = linalg.qr(X_bar)
    Q1,Q2=Q[:,0:d],Q[:,d:n]
    R=R0[0:d,:]
    alpha=Q2.dot(np.linalg.inv(Q2.T.dot(M).dot(Q2))).dot(Q2.T).dot(Y)
    B=np.linalg.inv(R).dot(Q1.T).dot(Y-M.dot(alpha))
    return alpha,B

def TPS_recover_parameter_cuda(Phi_T,X_bar,Y,epsilon):
    n,d=X_bar.shape
    n,K=Phi_T.shape
    diag_M=np.zeros((n,K))
    np.fill_diagonal(diag_M, epsilon)
    M=Phi_T+diag_M
    Q, R0 = linalg.qr(X_bar)
    Q1,Q2=Q[:,0:d],Q[:,d:n]
    R=R0[0:d,:]
    Q1_c,Q2_c,R_c,M_c,Y_c=cp.array(Q1),cp.array(Q2),cp.array(R),cp.array(M),cp.array(Y)
    alpha=Q2_c.dot(cp.linalg.inv(Q2_c.T.dot(M_c).dot(Q2_c))).dot(Q2_c.T).dot(Y_c)
    B=np.linalg.inv(R_c).dot(Q1_c.T).dot(Y_c-M_c.dot(alpha_c))
    return cp.asnumpy(alpha),cp.asnumpy(B)


@nb.njit(fastmath=True)
def transform_TPS(X_bar,Phi,alpha,B):
    return Phi.dot(alpha)+X_bar.dot(B)


@nb.njit(fastmath=True)
def transform_ICP(x,Phi,alpha,S,R,beta):
    return Phi.dot(alpha)+X.dot(S).dot(R)+beta


@nb.njit(fastmath=True)
def cost_matrix_d(X,Y):
    '''
    input: 
        X: (n,) float np array
        Y: (m,) float np array
    output:
        M: n*m matrix, M_ij=c(X_i,Y_j) where c is defined by cost_function.
    
    '''

    X1=np.expand_dims(X,1)
    Y1=np.expand_dims(Y,0)
    M=np.sum(cost_function(X1,Y1),2)
    return M


def init_center(X,Y,K,eps):
    M=cost_matrix_d(X,Y)
    row_min=M.min(1)
    indices=np.where(row_min>eps)[0]
    n=indices.shape[0]
    if K<n:
        rand_index=indices[np.random.randint(0,n,K)]
    else: 
        rand_index=indices
    return X[rand_index]

    

def make_plot(X,Y):
    fig = plt.figure(figsize=(2*800/72,800/72))    
    ax = fig.add_subplot(projection='3d')
    x=X[:,0]
    y=X[:,1]
    z=X[:,2]
    ax.scatter3D(X[:,0], X[:,2], X[:,1], s=0.5,c='r',alpha=0.5) #=X[:,1]*np.sqrt(X[:,0]**2+X[:,2]**2), cmap='bone')
    
    #x=Y[:,0]
    #y=Y[:,1]
    #z=Y[:,2]
    ax.scatter3D(Y[:,0], Y[:,2], Y[:,1], s=0.5,c='b',alpha=0.5) #Y[:,1]*np.sqrt(Y[:,0]**2+Y[:,2]**2), cmap='bone')
    
    ax.set_xlim([-1,1]);ax.set_ylim([-1,1]);ax.set_zlim([-1,1])
    ax.view_init(10, 45)
    plt.show()




def sopt_Gaussian(X,Y,N0,n_projections=1000,sigma2=1e-4, record_index=[0,20,100,200,400,500,700,900]):
    N1,D=X.shape
    C=X.copy()
    Phi=kernel_matrix_Gaussian(C,X,sigma2) 

    # initlize 
    R=np.eye(D)    
    S=1.0 #np.eye(d)
    beta=vec_mean(Y)-vec_mean(S*X.dot(R)) 
    alpha=np.zeros((C.shape[0],D))
    
    #paramlist=[]
    projections=random_projections(D,n_projections,1)
    mass_diff=0
    #b=0
    b=np.log((N1-N0+1)/1)
    Lambda=60*np.sum((vec_mean(Y)-vec_mean(X))**2)
    Y_hat=Phi.dot(alpha)+S*X.dot(R)+beta
    # make_plot(Y_hat,Y)

    Domain_org=arange(0,N1)
    Delta=Lambda/8
    lower_bound=Lambda/10000
    L=Domain_org.copy()

    Yhat_list=list()
    for (epoch,theta) in enumerate(projections):    
        # compute correspondence 
        Y_hat_theta=np.dot(theta,Y_hat.T)
        Y_theta=np.dot(theta,Y.T)

        Y_hat_indice=Y_hat_theta.argsort()
        Y_indice=Y_theta.argsort()
        Y_hat_s=Y_hat_theta[Y_hat_indice]
        Y_s=Y_theta[Y_indice]
        obj,phi,psi,piRow,piCol=solve_opt(Y_hat_s,Y_s,Lambda)
        L=piRow.copy()
        L=recover_indice(Y_hat_indice,Y_indice,L)
        Domain=Domain_org[L>=0]

        #move selected Y_hat
        mass=Domain.shape[0]
        # if Domain.shape[0]>=1:
        Range=L[L>=0]
        Y_hat[Domain]+=np.expand_dims(Y_theta[Domain]-Y_hat_theta[Domain],1)*theta

        # find optimal R,S,beta, conditonal on alpha
        Y_prime2=Y_hat[Domain]-Phi[Domain].dot(alpha)
        R,S=recover_rotation(Y_prime2,X[Domain])
        beta=vec_mean(Y_prime2)-vec_mean(X[Domain].dot(R)*S)

        # update Y_hat by alpha, Phi 
        Y_prime=Y_hat[Domain]-X[Domain].dot(R)*S-beta
        alpha=recover_alpha(Phi[Domain],Y_prime)
        # print('error from alpha is',np.linalg.norm(Phi.dot(alpha)[Domain]-Y_prime))

        # update selected points 
        # model 1
        Y_hat=Phi.dot(alpha)+S*X.dot(R)+beta

        # update lambda 
        N=(N1-N0)*1/(1+b*(epoch/500))+N0
        mass_diff=mass-N
        if mass_diff>N*0.009:
            Lambda-=Delta 
        if mass_diff<-N*0.003:
            Lambda+=Delta
            Delta=Lambda*1/8
        if Lambda<Delta:
            Lambda=Delta
            Delta=Delta*1/2
        if Delta<lower_bound:
            Delta=lower_bound
        if epoch in record_index or epoch==n_projections-1:
            Yhat_list.append(Y_hat)
            
        # if epoch==0 or epoch%5==0 and epoch<=40:
        #     make_plot(Y_hat,Y)
        #     print(len(Yhat_list))
        #     print(Yhat_list[-1].shape)
        # elif epoch%100==0: 
        #     make_plot(Y_hat,Y)

    return Yhat_list,(Phi,alpha,S,R,beta)




def sopt_Gaussian_cuda(X,Y,N0,n_projections=1000,sigma2=1e-4, eps=1e-4, record_index=[0,20,100,200,400,500,700,900]):
    N1,D=X.shape
    C=X.copy()
    Phi=kernel_matrix_Gaussian(C,X,sigma2) 

    # initlize 
    R=np.eye(D)    
    S=1.0 #np.eye(d)
    beta=vec_mean(Y)-vec_mean(S*X.dot(R)) 
    alpha=np.zeros((C.shape[0],D))
    
    #paramlist=[]
    projections=random_projections(D,n_projections,1)
    mass_diff=0
    #b=0
    b=np.log((N1-N0+1)/1)
    Lambda=60*np.sum((vec_mean(Y)-vec_mean(X))**2)
    Y_hat=Phi.dot(alpha)+S*X.dot(R)+beta
    # make_plot(Y_hat,Y)

    Domain_org=arange(0,N1)
    Delta=Lambda/8
    lower_bound=Lambda/10000
    L=Domain_org.copy()

    Yhat_list=list()
    for (epoch,theta) in enumerate(projections):    
        # compute correspondence 
        Y_hat_theta=np.dot(theta,Y_hat.T)
        Y_theta=np.dot(theta,Y.T)

        Y_hat_indice=Y_hat_theta.argsort()
        Y_indice=Y_theta.argsort()
        Y_hat_s=Y_hat_theta[Y_hat_indice]
        Y_s=Y_theta[Y_indice]
        obj,phi,psi,piRow,piCol=solve(Y_hat_s,Y_s,Lambda)
        L=piRow.astype(np.int64)
        L=recover_indice(Y_hat_indice,Y_indice,L)
        Domain=Domain_org[L>=0]

        #move selected Y_hat
        mass=Domain.shape[0]
        # if Domain.shape[0]>=1:
        Range=L[L>=0]
        Y_hat[Domain]+=np.expand_dims(Y_theta[Domain]-Y_hat_theta[Domain],1)*theta

        # find optimal R,S,beta, conditonal on alpha
        Y_prime2=Y_hat[Domain]-Phi[Domain].dot(alpha)
        R,S=recover_rotation_cuda(Y_prime2,X[Domain])
        beta=np.mean(Y_prime2,0)-np.mean(X[Domain].dot(R)*S,0)

        # update Y_hat by alpha, Phi 
        Y_prime=Y_hat[Domain]-X[Domain].dot(R)*S-beta
        alpha=recover_alpha_cuda(Phi[Domain],Y_prime,eps)
        # print('error from alpha is',np.linalg.norm(Phi.dot(alpha)[Domain]-Y_prime))

        # update selected points 
        # model 1
        Y_hat=Phi.dot(alpha)+S*X.dot(R)+beta

        # update lambda 
        N=(N1-N0)*1/(1+b*(epoch/500))+N0
        mass_diff=mass-N
        if mass_diff>N*0.009:
            Lambda-=Delta 
        if mass_diff<-N*0.003:
            Lambda+=Delta
            Delta=Lambda*1/8
        if Lambda<Delta:
            Lambda=Delta
            Delta=Delta*1/2
        if Delta<lower_bound:
            Delta=lower_bound
        if epoch in record_index or epoch==n_projections-1:
            Yhat_list.append(Y_hat)
            
        # if epoch==0 or epoch%5==0 and epoch<=40:
        #     make_plot(Y_hat,Y)
        #     print(len(Yhat_list))
        #     print(Yhat_list[-1].shape)
        # elif epoch%100==0: 
        #     make_plot(Y_hat,Y)

    return Yhat_list,(Phi,alpha,S,R,beta)


def sopt_TPS(X,Y,N0,n_projections=300,eps=1e-4,record_index=[0,10,20,30,40,60,80,90,100,150,200,250,300]):
    N1,D=X.shape
    C=X.copy()
    Phi0=kernel_matrix_TPS(C,X,D) 
    X_bar=np.hstack((np.ones((X.shape[0],1)),X))
    # initlize 
    R=np.eye(D)    
    S=1.0
    beta=np.zeros(3) #vec_mean(Y)-vec_mean(X.dot(S).dot(R)) 
    alpha=np.zeros((C.shape[0],D))
    B=np.vstack((beta,R))

    #paramlist=[]
    projections=random_projections(D,n_projections,1)
    mass_diff=0
    b=np.log((N1-N0+1)/1)
    Lambda=60*np.sum((Y.mean(0)-X.mean(0))**2)
    Y_hat=Phi0.dot(alpha)+X_bar.dot(B)
    # make_plot(Y_hat,Y)

    Domain_org=arange(0,N1)
    Delta=Lambda/8
    lower_bound=Lambda/10000
    L=Domain_org.copy()

    Yhat_list=list()
    for (epoch,theta) in enumerate(projections):
        # compute correspondence 
        Y_hat_theta=np.dot(theta,Y_hat.T)
        Y_theta=np.dot(theta,Y.T)

        Y_hat_indice,Y_indice=Y_hat_theta.argsort(),Y_theta.argsort()
        Y_hat_s,Y_s=Y_hat_theta[Y_hat_indice],Y_theta[Y_indice]
        obj,phi,psi,piRow,piCol=solve_opt(Y_hat_s,Y_s,Lambda)
        L=piRow.astype(np.int64)
        L=recover_indice(Y_hat_indice,Y_indice,L)
        Domain=Domain_org[L>=0]

        #move selected Y_hat
        mass=Domain.shape[0]
        Range=L[L>=0]
        Y_hat[Domain]+=np.expand_dims(Y_theta[Domain]-Y_hat_theta[Domain],1)*theta

        # find optimal alpha, B
        Phi_T,X_bar_select,Y_select=Phi0[Domain],X_bar[Domain],Y_hat[Domain]
        alpha,B=TPS_recover_parameter(Phi_T,X_bar_select,Y_select,eps)


        # update selected points 
        # our model
        Y_hat=Phi0.dot(alpha)+X_bar.dot(B)

        
        # update lambda 
        N=(N1-N0)*1/(1+b*(epoch/500))+N0
        mass_diff=mass-N
        if mass_diff>N*0.009:
            Lambda-=Delta 
        if mass_diff<-N*0.003:
            Lambda+=Delta
            Delta=Lambda*1/8
        if Lambda<Delta:
            Lambda=Delta
            Delta=Delta*1/2
        if Delta<lower_bound:
            Delta=lower_bound
        
        # recode point cloud in the process
        if epoch in record_index or epoch==n_projections-1:
            Yhat_list.append(Y_hat)
            
#         if epoch%5==0 and epoch<=40:
#             make_plot(Y_hat,Y)
#         elif epoch%100==0: 
#             make_plot(Y_hat,Y)
            
        
    return Yhat_list,(Phi0,alpha,B)


def sopt_TPS_cuda(X,Y,N0,n_projections=300,eps=1e-4,record_index=[0,10,20,30,40,60,80,90,100,150,200,250,300]):
    N1,D=X.shape
    C=X.copy()
    Phi0=kernel_matrix_TPS(C,X,D) 
    X_bar=np.hstack((np.ones((X.shape[0],1)),X))
    # initlize 
    R=np.eye(D)    
    S=1.0
    beta=np.zeros(3) #vec_mean(Y)-vec_mean(X.dot(S).dot(R)) 
    alpha=np.zeros((C.shape[0],D))
    B=np.vstack((beta,R))

    #paramlist=[]
    projections=random_projections(D,n_projections,1)
    mass_diff=0
    b=np.log((N1-N0+1)/1)
    Lambda=60*np.sum((Y.mean(0)-X.mean(0))**2)
    Y_hat=Phi0.dot(alpha)+X_bar.dot(B)
    # make_plot(Y_hat,Y)

    Domain_org=np.arange(0,N1)
    Delta=Lambda/8
    lower_bound=Lambda/10000
    L=Domain_org.copy()

    Yhat_list=list()
    for (epoch,theta) in enumerate(projections):
        # compute correspondence 
        Y_hat_theta=np.dot(theta,Y_hat.T)
        Y_theta=np.dot(theta,Y.T)

        Y_hat_indice,Y_indice=Y_hat_theta.argsort(),Y_theta.argsort()
        Y_hat_s,Y_s=Y_hat_theta[Y_hat_indice],Y_theta[Y_indice]
        obj,phi,psi,piRow,piCol=solve(Y_hat_s,Y_s,Lambda)
        L=piRow.copy()
        L=recover_indice(Y_hat_indice,Y_indice,L)
        Domain=Domain_org[L>=0]

        #move selected Y_hat
        mass=Domain.shape[0]
        Range=L[L>=0]
        Y_hat[Domain]+=np.expand_dims(Y_theta[Domain]-Y_hat_theta[Domain],1)*theta

        # find optimal alpha, B
        Phi_T,X_bar_select,Y_select=Phi0[Domain],X_bar[Domain],Y_hat[Domain]
        alpha,B=TPS_recover_parameter_cuda(Phi_T,X_bar_select,Y_select,eps)


        # update selected points 
        # our model
        Y_hat=Phi0.dot(alpha)+X_bar.dot(B)

        
        # update lambda 
        N=(N1-N0)*1/(1+b*(epoch/500))+N0
        mass_diff=mass-N
        if mass_diff>N*0.009:
            Lambda-=Delta 
        if mass_diff<-N*0.003:
            Lambda+=Delta
            Delta=Lambda*1/8
        if Lambda<Delta:
            Lambda=Delta
            Delta=Delta*1/2
        if Delta<lower_bound:
            Delta=lower_bound
        
        # recode point cloud in the process
        if epoch in record_index or epoch==n_projections-1:
            Yhat_list.append(Y_hat)
            
#         if epoch%5==0 and epoch<=40:
#             make_plot(Y_hat,Y)
#         elif epoch%100==0: 
#             make_plot(Y_hat,Y)
            
        
    return Yhat_list,(Phi0,alpha,B)

