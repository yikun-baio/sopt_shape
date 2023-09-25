#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 14:25:29 2022

"""
import torch 
import os
import sys
import time 

#from .opt1d import solve 
from .library import *
from .lib_ot import *   
from .sliced_opt import *
import numba as nb
import matplotlib.pyplot as plt

@nb.njit()
def permutation_inverse(permutation):
    N=permutation.shape[0]
    Domain=np.arange(N)
    mapping=np.stack((Domain,permutation))
    mapping_final=mapping[0].take(mapping[1].argsort())
    return mapping_final

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
    return torch.stack([torch.cos(theta).reshape([1]),torch.sin(theta).reshape([1]),-torch.sin(theta).reshape([1]),torch.cos(theta).reshape([1])]).reshape([2,2])



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
        

        

# @nb.njit(cathe=True)
def recover_rotation(X,Y,rescale=True):
    """
    return the optimal rotation, scalling based on the correspondence (X,Y) 
    
    Parameters:
    ----------
    X: numpy array, shape (n,d), flaot64, target
    Y: numpy array, shape (n,d), flaot64, source
    
    Return:
    --------
    rotation: numpy array, shape (d,d), float64 
    scalling: float64 
    
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
    scalling=np.sum(np.abs(S.T))/np.trace(Y_c.T.dot(Y_c))
    return rotation,scalling


def icp_step_gpu(Y,X,rescale=False,device='cpu'):
    """
    return the optimal rotation, scalling based on the correspondence (Target,Source) 
    
    Parameters:
    ----------
    Target: numpSource array, shape (n,d), flaot64, target
    Source: numpy array, shape (n,d), flaot64, source
    
    Return:
    --------
    rotation: numpy array, shape (d,d), float64 
    scalling: float64 
    
    """
    dtype=X.dtype
    Y_tc,X_tc=torch.from_numpy(Y).to(device),torch.from_numpy(X).to(device)
    torch_dtype=X_tc.dtype
    n,d=X_tc.shape
    Y_center,X_center=Y_tc-torch.mean(Y_tc,0),X_tc-torch.mean(X_tc,0)
    
    XY=X_center.T.mm(Y_center)
    U,Sigma,VT=torch.linalg.svd(XY)
    UVT=U.mm(VT)
    diag=torch.eye(d,device=device,dtype=torch_dtype)
    diag[d-1,d-1]=torch.linalg.det(UVT.T)
    R=U.mm(diag).mm(VT)
    S=torch.sum(torch.abs(Sigma.T))/torch.trace(X_center.T.mm(X_center))
    
    if rescale==True:
        beta=torch.mean(Y_tc,0)-torch.mean(X_tc.mm(R),0)*S
    else:
        beta=torch.mean(Y_tc,0)-torch.mean(X_tc.mm(R),0)
    return R.cpu().numpy(),S.cpu().numpy(),beta.cpu().numpy()
    




@nb.njit(fastmath=True)
def rbf_recover_alpha(Phi,Y,eps=1.0):
    dtype=Y.dtype
    n,d=Phi.shape
    return np.linalg.inv(Phi.T.dot(Phi)+eps*np.eye(d).astype(dtype)).dot(Phi.T.dot(Y))

    
def rbf_recover_alpha_gpu(Phi,Y,eps=1.0,device='cuda:0'):
    n,d=Phi.shape
    # dtype=Y.dtype
    #Phi,Y=Phi,Y
    Phi_tc,Y_tc=torch.from_numpy(Phi).to(device),torch.from_numpy(Y).to(device)
    torch_dtype=Y_tc.dtype
    Phi_matrix=Phi_tc.T.mm(Phi_tc)+eps*torch.eye(d,device=device,dtype=torch_dtype)
    PhiY=Phi_tc.T.mm(Y_tc)
    alpha=torch.linalg.inv(Phi_matrix).mm(PhiY)
    return alpha.cpu().numpy()





#@nb.njit()
def tps_regression(Phi,X_bar,Y,eps=1.0,**kwargs):
    n,d=X_bar.shape
    n,K=Phi.shape
    diag_M=np.zeros((n,K))
    np.fill_diagonal(diag_M, eps)
    M=Phi+diag_M
    Q, R0 = np.linalg.qr(X_bar,'complete')
    Q1,Q2=Q[:,0:d],Q[:,d:n]
    R=R0[0:d,:]
    alpha=Q2.dot(np.linalg.inv(Q2.T.dot(M).dot(Q2))).dot(Q2.T).dot(Y)
    B=np.linalg.inv(R).dot(Q1.T).dot(Y-M.dot(alpha))
    return B,alpha

def tps_regression_gpu(Phi,X_bar,Y,eps=1.0,device='cuda:0'):
    dtype=X_bar.dtype
    Phi,X_bar,Y=torch.from_numpy(Phi).to(device),torch.from_numpy(X_bar).to(device),torch.from_numpy(Y).to(device)
    torch_dtype=Phi.dtype
    #print('torch type',torch_dtype)
    n,d=X_bar.shape
    n,K=Phi.shape
    diag_M=torch.zeros((n,K),device=device,dtype=torch_dtype) #.to(torch_dtype)
    diag_M.fill_diagonal_(eps)
    M=Phi+diag_M
    Q, R0 = torch.linalg.qr(X_bar,'complete')
    Q1,Q2=Q[:,0:d].clone(),Q[:,d:n].clone()
    R=R0[0:d,:].clone()
    Re1=torch.linalg.inv(Q2.T.mm(M).mm(Q2))
    alpha=Q2.mm(Re1).mm(Q2.T).mm(Y)
    B=torch.linalg.inv(R).mm(Q1.T).mm(Y-M.mm(alpha))
    return B.cpu().numpy(),alpha.cpu().numpy()







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
def gaussian_kernel(r2,sigma2):
    return np.exp(-r2/sigma2)


@nb.njit()
def tps_kernel_2D(r2):
    return 1/2*r2*np.log(r2+1e-10)

@nb.njit()
def tps_kernel_3D(r2):
    return np.sqrt(r2)





@nb.njit(fastmath=True)
def kernel_matrix_gaussian(x,c,sigma2):
    '''
    x: (N,d) numpy array
    c: (K,d) numpy array
    '''
   
    K,d=c.shape
    n=x.shape[0]
    #r2=np.zeros((n,K))
    diff_matrix=np.expand_dims(x,1)-np.expand_dims(c,0)
    r2=np.sum(np.square(diff_matrix),axis=2)
    # Phi=np.zeros((n,d))
    Phi=np.exp(-r2/(2*sigma2))
    return Phi

def kernel_matrix_gaussian_T(x,c,sigma2):
    '''
    x: (N,d) numpy array
    c: (K,d) numpy array
    '''
   
    K,d=c.shape
    n=x.shape[0]
    #r2=np.zeros((n,K))
    diff_matrix=torch.expand_dims(x,1)-torch.expand_dims(c,0)
    r2=torch.sum(torch.square(diff_matrix),axis=2)
    # Phi=np.zeros((n,d))
    Phi=torch.exp(-r2/(2*sigma2))
    return Phi
    

@nb.njit(fastmath=True)
def kernel_matrix_tps(x,c,D):
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
    if D==3:
        Phi=tps_kernel_3D(r2)
    else:
        Phi=tps_kernel_2D(r2)
    return Phi

@nb.njit()
def update_lambda(Lambda,Lambda_step,mass_diff_relative,Lambda_lb):
    #mass_diff_relative=mass_diff/N0
    if mass_diff_relative>0.003:
        Lambda-=Lambda_step 
    if mass_diff_relative<-0.003:
        Lambda+=Lambda_step
        Lambda_step=Lambda*1/8
    if Lambda<Lambda_step:
        Lambda=Lambda_step
        Lambda_step=Lambda_step*1/2
    if Lambda_step<Lambda_lb:
        Lambda_step=Lambda_lb
    return Lambda,Lambda_step




@nb.njit(fastmath=True)
def transform_tps(X_bar,Phi,alpha,B):
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


@nb.njit(parallel=True,fastmath=True,cache=True)
def opt_plans(X_projections,Y_projections,Lambda_list):
    n_projection,n=X_projections.shape
    opt_plan_list=np.zeros((n_projection,n),dtype=np.int64)
    opt_cost_list=np.zeros(n_projection)
    for epoch in nb.prange(n_projection): #enumerate(zip(X_projections,Y_projections,Lambda_list)):
        X_theta,Y_theta,Lambda = X_projections[epoch],Y_projections[epoch],Lambda_list[epoch]
        # X_indice,Y_indice=X_theta.argsort(),Y_theta.argsort()
        # X_s,Y_s=X_theta[X_indice],Y_theta[Y_indice]
        obj,phi,psi,piRow,piCol=opt1d(X_theta,Y_theta,Lambda)
        # L1=recover_indice(X_indice,Y_indice,piRow)
        opt_cost_list[epoch],opt_plan_list[epoch]=obj,piRow
    return opt_cost_list,opt_plan_list


def init_kernel_matrix(kernel,X):
    dtype=X.dtype
    if kernel['C'] is None:
        kernel['C']=X.copy()
        kernel['C']=kernel['C'].astype(dtype=dtype)
    if kernel['name']=='gaussian':
        if kernel['param'] is None:
            kernel['param']=0.05
        Phi=kernel_matrix_gaussian(X,kernel['C'],kernel['param'])
    elif kernel['name']=='tps': 
        if kernel['param'] is None:
            kernel['param']=2
        if np.linalg.norm(kernel['C']-X)>1e-10:
            raise TypeError("TPS requires C=X!")
        Phi=kernel_matrix_tps(X,kernel['C'],kernel['param'])
    return Phi


def barycentric_projection_ot(corr,Yhat,Y):
    mu,nu=corr['mu'],corr['nu']
    cost_M=cost_matrix_d(Yhat,Y)
    dtype=Yhat.dtype
    cost_M=cost_M.astype(np.float64) # Must be float64 number 
    if corr['name']=='ot':       
        # print('mu is',mu.sum())
        # print('nu is',nu.sum())
        gamma=ot.lp.emd(mu, nu, cost_M, numItermax=1e7,numThreads=10)
    elif corr['name']=='opt':
        gamma=ot.partial.partial_wasserstein(mu, nu, cost_M, m=corr['N0'], nb_dummies=1, numItermax=1e7,numThreads=10)
        # gamma=opt_pr(mu, nu, cost_M, mass=corr['N0']-1e-10, numItermax=1e7,numThreads=10)
    elif corr['name']=='sinkhorn':
        gamma=sinkhorn_knopp_opt_1(mu, nu, M=cost_M, Lambda=corr['Lambda'], 
                                   reg=corr['reg'], numItermax=500,verbose=False)
    elif corr['name']=='sinkhorn_pr':
        gamma=sinkhorn_knopp_opt_pr_1(mu, nu, M=cost_M, mass=corr['N0']-1e-9,
                                      reg=corr['reg'],numItermax=500,verbose=False)
    
    
    gamma=gamma.astype(dtype)# convert gamma back the to original data type 
    p1_hat=np.sum(gamma,1)
    # print('p1_hat',p1_hat)
    Domain=p1_hat>1e-10
    BaryP=gamma.dot(Y)[Domain]/np.expand_dims(p1_hat,1)[Domain]
    Yhat[Domain]=BaryP
    return Yhat,Domain

def init_ot_param(corr,X,Y,N0):
    dtype=X.dtype
    N,M=X.shape[0],Y.shape[0]
    if 'N0' not in corr or corr['N0'] is None:
        corr['N0']=N0
    if corr['name'] in ['opt','sinkhorn','sinkhorn_pr']:    
        corr['mu'],corr['nu']=np.ones(N,dtype),np.ones(M,dtype)
    if corr['name']=='ot':
        corr['mu'],corr['nu']=np.ones(N,dtype)/N,np.ones(M,dtype)/M
    elif corr['name']== 'sinkhorn':
        if corr['reg'] is None:
            corr['reg']=0.1*Y.var()
        if corr['Lambda'] is None:
            corr['Lambda']=2*np.sum((X.mean(0)-Y.mean(0))**2)
    elif corr['name'] =='sinkhorn_pr':
        if corr['reg'] is None:
            corr['reg']=0.01*Y.var()        
    elif corr['name'] =='sopt':
        if 'Lambda' not in corr or corr['Lambda'] is None:
            corr['Lambda']=4*np.sum((vec_mean(Y)-vec_mean(X))**2)
            corr['Lambda_step'],corr['Lambda_lb']=corr['Lambda']/8,corr['Lambda']/10000
    # elif corr['name']=='sot':
    #     if N!=M:
    #         raise TypeError("SOT requires N=M!")

# 
# @nb.njit(cache=True)
# def gradient_flow_sot(X,Y,mu,nu,projections):
#     """
#     can work for both float32 and float64
#     """
#     n_proj,d=projections.shape
#     N,M=X.shape[0],Y.shape[0]
#     mu_s,nu_s=np.ones(N)/N,np.ones(M)/M
#     for i in range(n_proj):
#         theta=projections[i]
#         X_theta,Y_theta=np.dot(theta,X.T),np.dot(theta,Y.T)
#         sorter_X,sorter_Y=X_theta.argsort(),Y_theta.argsort()
#         X_s,Y_s=X_theta[sorter_X],Y_theta[sorter_Y]
#         #move X
#         gamma=ot1d(mu_s,mu_s)
#         p1_hat=np.sum(gamma,1)
#         # print('p1_hat',p1_hat)
#         # Domain=p1_hat>1e-10
#         BaryP_Y=gamma.dot(Y_s)/np.expand_dims(p1_hat,1)
#         displacement_s=BaryP_Y-X_s
#         inv_sorter_X=permutation_inverse(sorter_X)
#         displacement=displacement_s[inv_sorter_X]
#         X+=np.expand_dims(displacement,1)*theta
#     Domain=arange(0,N)
#     return X,Domain


@nb.njit(cache=True)
def gradient_flow_sot(X,Y,projections):
    """
    can work for both float32 and float64
    """
    n_proj,d=projections.shape
    N,M=X.shape[0],Y.shape[0]
    mu_s,nu_s=np.ones(N)/N,np.ones(M)/M
    for i in range(n_proj):
        theta=projections[i]
        X_theta,Y_theta=np.dot(theta,X.T),np.dot(theta,Y.T)
        sorter_X,sorter_Y=X_theta.argsort(),Y_theta.argsort()
        X_s,Y_s=X_theta[sorter_X],Y_theta[sorter_Y]
        #move X
        gamma=ot1d(mu_s,nu_s)
        p1_hat=np.sum(gamma,1)
        BaryP_Y=gamma.dot(np.expand_dims(Y_s,1))/np.expand_dims(p1_hat,1)
        BaryP_Y=BaryP_Y.T[0]
        displacement_s=BaryP_Y-X_s
        inv_sorter_X=permutation_inverse(sorter_X)
        displacement=displacement_s[inv_sorter_X]
        X+=np.expand_dims(displacement,1)*theta
    Domain=arange(0,N)
    return X,Domain
    
@nb.njit(cache=True)
def gradient_flow_sopt(X,Y,projections,Lambda=1.0,Lambda_step=0.1,Lambda_lb=0.001,N0=4):
    '''
    '''
    dtype=X.dtype
    (n_proj,d),N=projections.shape,X.shape[0]
    Domain=np.full(N,False)
    for i in range(n_proj):
        theta=projections[i]
        X_theta,Y_theta=np.dot(theta,X.T),np.dot(theta,Y.T)
        sorter_X,sorter_Y=X_theta.argsort(),Y_theta.argsort()
        X_s,Y_s=X_theta[sorter_X],Y_theta[sorter_Y]
        X_s,Y_s=X_s.astype(np.float64),Y_s.astype(np.float64)
        obj,phi,psi,piRow,piCol=opt1d(X_s,Y_s,Lambda)
            
        #move X        
        displacement_s=np.zeros(N).astype(dtype)
        Domain_s=piRow>=0
        Range_s=piRow[Domain_s]
        displacement_s[Domain_s]=Y_s[Range_s]-X_s[Domain_s]
        inv_sorter_X=permutation_inverse(sorter_X)
        displacement,Domain_theta=displacement_s[inv_sorter_X],Domain_s[inv_sorter_X]
        X+=np.expand_dims(displacement,1)*theta
        
        #update domain 
        Domain+=np.logical_or(Domain,Domain_theta)
        
        # update Lambda
        mass=np.sum(Domain_s)
        mass_diff_relative=(mass-N0)/N0
        Lambda,Lambda_step=update_lambda(Lambda,Lambda_step,mass_diff_relative,Lambda_lb)
    return X,Domain,Lambda,Lambda_step
    
        
def init_parameter(X,Y,R=None,alpha=None,beta=None,dtype=np.float64):
    N,D=X.shape
    if R is None:
        R=np.eye(D).astype(dtype=dtype)
    if alpha is None:
        alpha=np.zeros((N,D)).astype(dtype=dtype)
    if beta is None:
        beta=np.mean(Y,0)-np.mean(X.dot(R),0) 
    B=np.vstack((beta,R))
    return alpha,B        
        
def init_Lambda(X,Y):
    Lambda=4*np.sum((vec_mean(Y)-vec_mean(X))**2)
    Lambda_step,Lambda_lb=Lambda/8,Lambda/10000
    return Lambda,Lambda_step,Lambda_lb


        
    
    
def init_record_idx(record_idx,n_iter_max):
    if record_idx is None:
        record_idx=np.unique(np.linspace(0,n_iter_max-1,num=int(n_iter_max/10)).astype(np.int64))
        record_idx.sort()
    return record_idx
    
def init_n_iter_rigid(n_iter_rigid,n_iter_max):
    if n_iter_rigid is None:
        n_iter_rigid=int(n_iter_max/10)
    return n_iter_rigid






def ot_registration(X,Y,N0,
             kernel={'name':'gaussian','C':None,'param':0.05,'eps':1.0},
             corr={'name':'sopt','n_proj':100},
             n_iter_max=200,n_iter_rigid=20,
             threshold=0.4,
             record_idx=None,
             dtype=np.float64,
             device='cuda:0',
             verbose=False,
             fix_mass=True,
             **kwargs):
    '''
    X: N*D float numpy array, source point cloud 
    Y: M*D float number array, target point cloud 
    kernel: Choose one of the following: 
        Gaussian kernel: {'name':'gaussian','C':X.copy(),'param':0.05,'eps':1.0}
        TPS kernel: {'name': 'tps', 'C':X.copy(),'param':2, 'eps':1.0}
        'C' is control point, 'param' for Gaussian is sigma2, for TPS is dimension D. 
        'eps' is regular term for inverse computation. 
    corr: Choose one of the following: 
         OT:     corr_ot={'name':'ot'} 
         OPT:    corr_opt={'name':'opt','N0':N0}
         sliced OT:   corr_sot={'name':'sot','n_proj':100}
         sliced OPT:  corr_sopt={'name':'sopt','N0':N0+20,'n_proj':100}
         Sinkhorn for OPT:   corr_sinkhorn={'name':'sinkhorn','reg':0.1*Y.var(),'N0':N0,'Lambda':Lambda}
         Sinkhorn for primal form OPT:   corr_sinkhorn_pr={'name':'sinkhorn_pr','reg':0.01*Y.var(),'N0':N0}
    n_iter_max: positive integer, total number of iterations
    n_iter_rigid: positive integer total number of rigid iterations 
    threshold: positive float 0.8,
    record_idx: the iterations that we record the results 
    device: device for computation of regression 'cuda:0',
    verbose: True, or False. For debug. 
    
    '''
    # initial kernel (the model)
    X,Y=X.astype(dtype),Y.astype(dtype)
    Phi=init_kernel_matrix(kernel,X)
    
    # init Lambda parameters 
    init_ot_param(corr,X,Y,N0)

    #print(corr)
    # initlize parameters 
    alpha,B=init_parameter(X,Y)

    N,M,D=X.shape[0],Y.shape[0],X.shape[1]
    X_bar=np.hstack((np.ones((N,1)).astype(dtype),X))
#    X_bar_orig,X_orig=X_bar.copy(),X.copy()
    Yhat=Phi.dot(alpha)+X_bar.dot(B)
                 
    B_list,alpha_list,Yhat_list=list(),list(),list()

    #init record_idx 
    record_idx=init_record_idx(record_idx,n_iter_max)
    # init n_iter_rigid
    n_iter_rigid=init_n_iter_rigid(n_iter_rigid,n_iter_max)

    epoch=0
    N0=corr['N0']
    mass_gap_ub=N-N0
                 
    while epoch<n_iter_max:
        if fix_mass==False and mass_gap_ub>0 and epoch>n_iter_rigid:
            mass_vary=mass_gap_ub*(np.exp(-(epoch-n_iter_rigid)/(n_iter_max-n_iter_rigid)*3))
            corr['N0']=N0+mass_vary
        # if mass_fixed==False:
        #     if epoch<n_iter_rigid:
        #         corr['N0']=N
        #     else:
        #         corr['N0']=N0
        print('current, %i/%i'%(epoch,n_iter_max),end='\r')
      
        time1=time.time()
        #print('corr_name',corr['name'])
        #print('end')
        if corr['name'] =='sot':
            #print('epoch',epoch )
           # print('n_proj is',corr['n_proj'])
            projections=random_projections(D,corr['n_proj'],1,dtype)
            Yhat,Domain=gradient_flow_sot(Yhat,Y,projections)
        elif corr['name'] =='sopt':
            projections=random_projections(D,corr['n_proj'],1,dtype)
            Yhat,Domain,corr['Lambda'],corr['Lambda_step']=gradient_flow_sopt(Yhat,Y,projections,Lambda=corr['Lambda'],Lambda_step=corr['Lambda_step'],Lambda_lb=corr['Lambda_lb'],N0=corr['N0'])
        elif corr['name'] in ['ot','opt','sinkhorn','sinkhorn_pr']:
            Yhat,Domain=barycentric_projection_ot(corr,Yhat,Y)
        #Yhat2=Yhat.copy()
        time2=time.time()


        # step 2
        if epoch%5==0:
            B_pre=B.copy()
            alpha_pre=alpha.copy() # when we do 
        rigid_error=np.linalg.norm(B-B_pre)
        #nonrigid_error=np.linalg.norm(alpha-alpha_pre)
        # print('nonrigid_error',nonrigid_error,end='\r')
        time3=time.time()
        if epoch>n_iter_rigid and rigid_error<threshold:
            if kernel['name']=='gaussian':
                Y_prime2=Yhat[Domain]-Phi[Domain].dot(alpha)
                R,S,beta=icp_step_gpu(Y_prime2,X[Domain],rescale=False,device=device)
                B=np.vstack((beta,R))
                Y_prime=Yhat[Domain]-X_bar[Domain].dot(B)
                alpha=rbf_recover_alpha_gpu(Phi[Domain],
                                            Y_prime,eps=kernel['eps'],device=device)
                B_pre=B.copy()
                alpha_pre=alpha.copy()
                #alpha2=rbf_recover_alpha(Phi[Domain],Y_prime,eps=kernel['eps'])
                #if verbose:
                   #print('rigid_error',rigid_error)
                   #print('epoch',epoch)
                #   print('RBF error',np.linalg.norm(alpha-alpha2))
                   #print('TPS error2',np.linalg.norm(alpha-alpha2))
                
            elif kernel['name']=='tps':
                B,alpha=tps_regression_gpu(Phi,X_bar,
                                           Yhat,eps=kernel['eps'],device=device)
                #B2,alpha2=tps_regression(Phi,X_bar,Yhat,eps=kernel['eps'])
                # if verbose:
                   #print('rigid_error',rigid_error)
                   #print('epoch',epoch)
                   #print('TPS error',np.linalg.norm(B-B2))
                   #print('TPS error2',np.linalg.norm(alpha-alpha2))
                B_pre=B.copy()
                alpha_pre=alpha.copy() # If we do non-rigid regression one time, update the B_pre, alpha_pre
                
        else:
            Y_prime2=Yhat[Domain]-Phi[Domain].dot(alpha)
            R,S,beta=icp_step_gpu(Y_prime2,X[Domain],rescale=False,device=device)
            B=np.vstack((beta,R))
        time4=time.time()

        Yhat=Phi.dot(alpha)+X_bar.dot(B)



        if epoch in record_idx:
            B_list.append(B.copy()),alpha_list.append(alpha.copy()),Yhat_list.append(Yhat.copy())
            if verbose:
                #print('regression time',time4-time3)
                #print('after corr')
                #make_plot(Yhat2,Y,N0)
                print('after regresion')
                make_plot(Yhat,Y,N0)
                
                #print('corr time is',time2-time1)

        epoch+=1
    model={}
    model['B_list']=B_list
    model['alpha_list']=alpha_list
    model['kernel']=kernel
    return model,Yhat_list,record_idx



def model_to_Yhat(model,X,method):
    N=X.shape[0]
    dtype_orig=X.dtype
    Yhat_list=list()

    if 'RBF' in method or 'TPS' in method: 
        B_list,alpha_list,kernel=model['B_list'],model['alpha_list'],model['kernel']
        dtype=B_list[0].dtype
        X_bar=np.hstack((np.ones((N,1),dtype),X.astype(dtype)))
        if kernel['name']=='gaussian':
            Phi=kernel_matrix_gaussian(X,kernel['C'],kernel['param'])
        elif kernel['name']=='tps':
            Phi=kernel_matrix_tps(X,kernel['C'],kernel['param'])
    

        for epoch, (B,alpha) in enumerate(zip(B_list,alpha_list)):
          Yhat=Phi.dot(alpha)+X_bar.dot(B)
          Yhat_list.append(Yhat.astype(dtype_orig))
    if method=='CPD':
        params,C,beta_sq=model
        G=kernel_matrix_gaussian(X,C,beta_sq)
        N0=X.shape[0]
        for (epoch,param) in enumerate(params):
            Yhat=model_to_Yhat_CPD(X,param,G)
            Yhat_list.append(Yhat)
    return Yhat_list

def compute_error(X,Y,N0,permutation,model=None,Yhat_list=None,method='OT-RBF',plot_func=None):
    if plot_func is None:
        plot_func=make_plot
    X0,Y0=X[0:N0],Y[0:N0]
    if permutation is None:
        permutation=np.arange(0,N0)

        
    Y0=Y0[permutation]
    std=np.sqrt(np.mean(Y0.var(0)))
    # print('standard deviation',std)
    if Yhat_list is None:
        Yhat_list=model_to_Yhat(model,X0,method)
    error_list=np.zeros(len(Yhat_list))
    
    for (i,Yhat) in enumerate(Yhat_list):
        err_sq_mat=((Yhat[0:N0]-Y0)/std)**2
        error_list[i]=np.sqrt(np.sum(err_sq_mat)/N0)
    plot_func(Yhat[0:N0],Y0,N0)
    print('last error is',error_list[-1])
    return error_list


def compute_error_ot(X,Y,N0,model=None,Yhat_list=None,method='OT-RBF',plot_func=None,permutation=None):
    if plot_func is None:
        plot_func=make_plot
    X0,Y0=X[0:N0],Y[0:N0]
    std=np.sqrt(np.mean(Y0.var(0)))
    if Yhat_list is None:
        Yhat_list=model_to_Yhat(model,X0,method)

    error_list=np.zeros(len(Yhat_list))
    
    for (i,Yhat) in enumerate(Yhat_list):
        Yhat=Yhat[0:N0]
        Yhat,Y0=Yhat/std,Y0/std
        cost_M=cost_matrix_d(Yhat,Y0)
        mu,nu=np.ones(N0)/N0,np.ones(N0)/N0
        gamma=emd(mu,nu,cost_M)
        
        error_list[i]=np.sum(cost_M*gamma)
    plot_func(Yhat,Y0,N0)
    print(gamma.sum())

    print('last error is',error_list[-1])
    return error_list
    

    
    

def model_to_Yhat_CPD(X,param,G):
    transform_type = param[0]
    if transform_type == 'rigid':
        R, s, t = param[1]
        Yhat = s * np.dot(X, R) + t
    elif transform_type == 'nonrigid':
        W = param[1]
        Yhat = X + np.dot(G, W)
    else:
        raise ValueError("Unknown transformation type.")
#     else:
#         raise ValueError("Index not in record_idx.")
        
    return Yhat



def make_plot(X,Y,N0,path=None,show=True):
    fig = plt.figure(figsize=(10,10), dpi=200)    
    ax = fig.add_subplot(projection='3d')
    ax.scatter3D(X[0:N0,0], X[0:N0,2], X[0:N0,1], s=3,c='red',alpha=0.3) #=X[:,1]*np.sqrt(X[:,0]**2+X[:,2]**2), cmap='bone')
    ax.scatter3D(Y[0:N0,0], Y[0:N0,2], Y[0:N0,1], s=3,c='g',alpha=0.3) #Y[:,1]*np.sqrt(Y[:,0]**2+Y[:,2]**2), cmap='bone')
    ax.scatter3D(X[N0:,0], X[N0:,2], X[N0:,1], s=6,c='red',alpha=0.6) #=X[:,1]*np.sqrt(X[:,0]**2+X[:,2]**2), cmap='bone')
    ax.scatter3D(Y[N0:,0], Y[N0:,2], Y[N0:,1], s=6,c='g',alpha=0.6) #Y[:,1]*np.sqrt(Y[:,0]**2+Y[:,2]**2), cmap='bone')
    
    eps=0.1
    ax.set_xlim([-1,1]);ax.set_ylim([-1,1]);ax.set_zlim([-1,1])
    ax.view_init(10, 45)
    
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.axis('off')
    ax.set_facecolor('whitesmoke') 
    if path!=None:
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    if show==True:
        plt.show()
    else:
        plt.close()
    
def visual(Yhat_list,Y,N0,path,record_idx,plot_func=None,log=False):
    if plot_func is None:
        plot_func=make_plot
    for (Yhat,index) in zip(Yhat_list,record_idx):
        file_name=path+'_'+str(index)+'.jpg'
        plot_func(Yhat,Y,N0,file_name,show=log)
    file_name=path+'_final'+'.jpg'    
    plot_func(Yhat,Y,N0,file_name,show=True)
    
def make_plot_fish(X,Y,N0,path=None,show=True):
    fig = plt.figure(figsize=(10,10), dpi=200)    
    ax = plt.axes()
    ax.scatter(X[0:N0,0], X[0:N0,1], s=50,c='red',alpha=0.6) #=X[:,1]*np.sqrt(X[:,0]**2+X[:,2]**2), cmap='bone') Noise 
    ax.scatter(Y[0:N0,0], Y[0:N0,1], s=50,c='g',alpha=0.6) #Y[:,1]*np.sqrt(Y[:,0]**2+Y[:,2]**2), cmap='bone') Noise 
    ax.scatter(X[N0:,0], X[N0:,1], s=60,c='red',alpha=1) #=X[:,1]*np.sqrt(X[:,0]**2+X[:,2]**2), cmap='bone')
    ax.scatter(Y[N0:,0], Y[N0:,1], s=60,c='g',alpha=1) #Y[:,1]*np.sqrt(Y[:,0]**2+Y[:,2]**2), cmap='bone')
    ax.set_xlim([-1.5,1.5]);ax.set_ylim([-1.5,2.0])
    # ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.axis('off')
    ax.set_facecolor('lightgrey') #('whitesmoke') 
    if path!=None:
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    if show==True:
        plt.show()
    else:
        plt.close()
        

