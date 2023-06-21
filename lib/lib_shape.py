#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 14:25:29 2022

"""
import torch 
import os
import sys

#from .opt1d import solve 
from .library import *
from .lib_ot import *   
from .sliced_opt import *

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

def recover_rotation_gpu(X,Y,device='mps',**kwargs):
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
    X,Y=X.astype(np.float32),Y.astype(np.float32)
    X_tc,Y_tc=torch.from_numpy(X).to(device),torch.from_numpy(Y).to(device)
    n,d=X_tc.shape
    X_center,Y_center=X_tc-torch.mean(X_tc,0),Y_tc-torch.mean(Y_tc,0)
    YX=Y_center.T.mm(X_center)
    U,S,VT=torch.linalg.svd(YX)
    R=U.mm(VT)
    diag=torch.eye(d,device=device)
    diag[d-1,d-1]=torch.linalg.det(R.T)
    rotation=U.mm(diag).mm(VT)
    scaling=torch.sum(torch.abs(S.T))/torch.trace(Y_tc.T.mm(Y_tc))
    return rotation.cpu().numpy().astype(np.float64),scaling.cpu().numpy().astype(np.float64)







# @nb.njit(fastmath=True)
def recover_alpha(Phi,Y,epsilon=1.0,**kwargs):
    n,d=Phi.shape
    return np.linalg.inv(Phi.T.dot(Phi)+epsilon*np.eye(d)).dot(Phi.T.dot(Y))

    
def recover_alpha_gpu(Phi,Y,epsilon=1.0,device='mps',**kwargs):
    n,d=Phi.shape
    Phi,Y=Phi.astype(np.float32),Y.astype(np.float32)
    
    Phic,Y_tc=torch.from_numpy(Phi).to(device),torch.from_numpy(Y).to(device)
    re=torch.linalg.inv(Phic.T.mm(Phic)+epsilon*torch.eye(d,device=device)).mm(Phic.T.mm(Y_tc))
    return re.cpu().numpy().astype(np.float64)





#@nb.njit()
def TPS_recover_parameter(Phi,X_bar,Y,epsilon=1.0,**kwargs):
    n,d=X_bar.shape
    n,K=Phi.shape
    diag_M=np.zeros((n,K))
    np.fill_diagonal(diag_M, epsilon)
    M=Phi+diag_M
    Q, R0 = np.linalg.qr(X_bar,'complete')
    Q1,Q2=Q[:,0:d],Q[:,d:n]
    R=R0[0:d,:]
    alpha=Q2.dot(np.linalg.inv(Q2.T.dot(M).dot(Q2))).dot(Q2.T).dot(Y)
    B=np.linalg.inv(R).dot(Q1.T).dot(Y-M.dot(alpha))
    return alpha,B

def TPS_recover_parameter_gpu(Phi,X_bar,Y,epsilon=1.0,device='mps',**kwargs):
    Phi,X_bar,Y=torch.from_numpy(Phi.astype(np.float32)).to(device),torch.from_numpy(X_bar.astype(np.float32)).to(device),torch.from_numpy(Y.astype(np.float32)).to(device)
    n,d=X_bar.shape
    n,K=Phi.shape
    diag_M=torch.zeros((n,K),device=device)
    diag_M.fill_diagonal_(epsilon)
    M=Phi+diag_M
    Q, R0 = torch.linalg.qr(X_bar,'complete')
    Q1,Q2=Q[:,0:d].clone(),Q[:,d:n].clone()
    R=R0[0:d,:].clone()
    Re1=torch.linalg.inv(Q2.T.mm(M).mm(Q2))
    alpha=Q2.mm(Re1).mm(Q2.T).mm(Y)
    B=torch.linalg.inv(R).mm(Q1.T).mm(Y-M.mm(alpha))
    return alpha.cpu().numpy().astype(np.float64),B.cpu().numpy().astype(np.float64)







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


@nb.njit()
def TPS_kernel_2D(r2):
    return 1/2*r2*np.log(r2+1e-10)

@nb.njit()
def TPS_kernel_3D(r2):
    return np.sqrt(r2)





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
    if D==3:
        Phi=TPS_kernel_3D(r2)
    else:
        Phi=TPS_kernel_2D(r2)
    return Phi

def update_lambda(Lambda,Delta,mass_diff,N0,lower_bound):
  if mass_diff>N0*0.003:
    Lambda-=Delta 
  if mass_diff<-N0*0.003:
    Lambda+=Delta
    Delta=Lambda*1/8
  if Lambda<Delta:
    Lambda=Delta
    Delta=Delta*1/2
  if Delta<lower_bound:
    Delta=lower_bound
  return Lambda,Delta




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

    

def make_plot(X,Y,path=None):
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
    if path!=None:
       plt.savefig(path)
    plt.show()

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
        obj,phi,psi,piRow,piCol=solve_opt(X_theta,Y_theta,Lambda)
        # L1=recover_indice(X_indice,Y_indice,piRow)
        opt_cost_list[epoch],opt_plan_list[epoch]=obj,piRow
    return opt_cost_list,opt_plan_list


def choose_kernel(kernel,X):
    if kernel[0]=='Gaussian':
        C,sigma2,eps=kernel[1],kernel[2],kernel[3]
        if len(C)==0:
            C=X.copy()
        Phi=kernel_matrix_Gaussian(C,X,sigma2) 
    elif kernel[0]=='TPS': 
        kernel[1]=='TPS'
        C,_,eps=kernel[1],kernel[2],kernel[3]
        if len(C)==0:
            C=X.copy()
        Phi=kernel_matrix_TPS(C,X)
    return Phi,eps


def SOPT_GD(X,Y,N0,kernel=['Gaussian',[],0.1,3.0],n_projection=100,n_iteration=2000,record_index=[],start_epoch=None,threshold=0.8,**kwargs):
    # input kernel: method name, control point, sigma2, epsilon
    if len(kernel[1])==0:
        kernel[1]=X.copy()
    C=kernel[1]
    Phi,eps=choose_kernel(kernel,X)
    Phi_tc,X_tc,Y_tc=torch.from_numpy(Phi),torch.from_numpy(X),torch.from_numpy(Y) 
    N1,D=X.shape
    # initlize 
    theta_tc=torch.zeros(D,requires_grad=True)
    R_tc=rotation_re_T(theta_tc)
    beta_tc,alpha_tc=torch.mean(Y_tc,0)-torch.mean(X_tc.mm(R_tc),0),torch.zeros((C.shape[0],D),requires_grad=True,dtype=torch.float64)
    beta_tc=beta_tc.clone().detach().requires_grad_(True)
    Yhat_tc=Phi_tc.mm(alpha_tc)+X_tc.mm(R_tc)+beta_tc #Phi.dot(alpha)+X.dot(R)+beta 

    # mass and Lambda setting 
    mass_diff=0
    Lambda=4*np.sum((np.mean(Y,0)-np.mean(X,0))**2)
    Delta,lower_bound=Lambda/8,Lambda/10000

    # record parameters 
    R_list,beta_list,alpha_list=list(),list(),list()
    if len(record_index)==0:
        record_index=np.unique(np.linspace(0,n_iteration-1,num=int(20)).astype(np.int64))
        record_index.sort()
    
    if start_epoch==None:
        start_epoch=int(n_iteration/10)

    optimizer_rigid,optimizer_nonrigid = torch.optim.Adam([theta_tc,beta_tc], lr=0.1),torch.optim.Adam([alpha_tc], lr=0.1/N1) 
    epoch=0
    while epoch<n_iteration:
        print('current, %i/%i'%(epoch,n_iteration),end='\r')
        R_pre,beta_pre=R_tc.detach().numpy().copy(),beta_tc.detach().numpy().copy(),
        projections_tc=torch.from_numpy(random_projections(D,n_projection,1))
        Yhat_projections_tc,Y_projections_tc=projections_tc.mm(Yhat_tc.T),projections_tc.mm(Y_tc.T)
        (Yhat_projections_s,_),(Y_projections_s,_)=Yhat_projections_tc.sort(),Y_projections_tc.sort()  
        Lambda_list=np.full(n_projection,Lambda)
        _,opt_plan_list=opt_plans(Yhat_projections_s.detach().numpy(),Y_projections_s.numpy(),Lambda_list)
        opt_plan_list_tc=torch.from_numpy(opt_plan_list)
        obj_cost_list=[torch.sum((Yhat_theta[L>=0]-Y_theta[L[L>=0]])**2) for (L,Yhat_theta,Y_theta) in zip(opt_plan_list_tc,Yhat_projections_s,Y_projections_s)]
        obj_cost=sum(obj_cost_list)/n_projection
        mass=np.sum(opt_plan_list>=0)/n_projection
        mass_diff=mass-N0
        Lambda,Delta=update_lambda(Lambda,Delta,mass_diff,N0,lower_bound)
        optimizer_rigid.zero_grad(),optimizer_nonrigid.zero_grad()
        obj_cost.backward()
        if epoch>=start_epoch and np.linalg.norm(R_tc.detach().numpy()-R_pre)+np.linalg.norm(beta_tc.detach().numpy()-beta_pre)<threshold:
            #print('non rigid')
            optimizer_rigid.step(),optimizer_nonrigid.step()
        else:
            optimizer_rigid.step()
        # update Yhat 
        R_tc=rotation_re_T(theta_tc)
        Yhat_tc=Phi_tc.mm(alpha_tc)+X_tc.mm(R_tc)+beta_tc
        if epoch in record_index:
            R_list.append(R_tc.detach().numpy()),beta_list.append(beta_tc.detach().numpy()),alpha_list.append(alpha_tc.detach().numpy())
        epoch+=1

    return (R_list,beta_list,alpha_list,Phi),record_index

def SOPT_RBF(X,Y,N0,kernel=['Gaussian',[],0.1,3.0],n_projection=100,n_iteration=200,record_index=[],start_epoch=None,threshold=0.8,**kwargs):
    # input kernel: method name, control point, sigma2, epsilon
    if len(kernel[1])==0:
        kernel[1]=X.copy()
    C=kernel[1]
    Phi,eps=choose_kernel(kernel,X)

    N1,D=X.shape
    # initlize 
    R,alpha=np.eye(D),np.zeros((C.shape))    
    beta=vec_mean(Y)-vec_mean(X.dot(R)) 
    mass_diff=0
    Lambda=4*np.sum((vec_mean(Y)-vec_mean(X))**2)
    Yhat=Phi.dot(alpha)+X.dot(R)+beta

    Delta,lower_bound=Lambda/8,Lambda/10000
    R_list,beta_list,alpha_list=list(),list(),list()
    if len(record_index)==0:
        record_index=np.unique(np.linspace(0,n_iteration-1,num=int(n_iteration/10)).astype(np.int64))
        record_index.sort()
    
    if start_epoch==None:
        start_epoch=int(n_iteration/10)
    epoch=0

    while epoch<n_iteration:
        print('current, %i/%i'%(epoch,n_iteration),end='\r')
        projections=random_projections(D,n_projection,1)
        #Yhat_pre=Yhat.copy()
        R_pre,beta_pre=R.copy(),beta.copy()
        domain_sum=np.full(N1,False)
        for (epoch2,theta) in enumerate(projections):
            Yhat_theta,Y_theta=np.dot(theta,Yhat.T),np.dot(theta,Y.T)
            Yhat_indice,Y_indice=Yhat_theta.argsort(),Y_theta.argsort()
            Yhat_s,Y_s=Yhat_theta[Yhat_indice],Y_theta[Y_indice]
            obj,phi,psi,piRow,piCol=solve_opt(Yhat_s,Y_s,Lambda)
            L=recover_indice(Yhat_indice,Y_indice,piRow)
            Domain,Range=L>=0,L[L>=0]
            domain_sum=np.logical_or(Domain,domain_sum) 
            Yhat[Domain]+=np.expand_dims(Y_theta[Range]-Yhat_theta[Domain],1)*theta
            mass=np.sum(Domain)
            mass_diff=mass-N0
            Lambda,Delta=update_lambda(Lambda,Delta,mass_diff,N0,lower_bound)

        # find optimal R,S,beta, conditonal on alpha
        Y_prime2=Yhat[domain_sum]-Phi[domain_sum].dot(alpha)

        # update Yhat by R,beta

        if epoch>=start_epoch and np.linalg.norm(R-R_pre)+np.linalg.norm(beta-beta_pre)<threshold:
            R,S=recover_rotation_gpu(Y_prime2,X[domain_sum],**kwargs)
            beta=vec_mean(Y_prime2)-vec_mean(X[domain_sum].dot(R)) 
            Y_prime=Yhat[domain_sum]-X[domain_sum].dot(R)-beta
            alpha=recover_alpha_gpu(Phi[domain_sum],Y_prime,eps,**kwargs)
            Yhat=Phi.dot(alpha)+X.dot(R)+beta
        else:
            R,S=recover_rotation_gpu(Y_prime2,X[domain_sum],**kwargs)
            beta=vec_mean(Y_prime2)-vec_mean(X[domain_sum].dot(R)) 
            
        Yhat=Phi.dot(alpha)+X.dot(R)+beta    
        if epoch in record_index:
            R_list.append(R),beta_list.append(beta),alpha_list.append(alpha)
        epoch+=1
    return (R_list,beta_list,alpha_list,Phi),record_index



def SOPT_TPS(X,Y,N0,eps=3.0,n_projection=100,n_iteration=200,record_index=[],start_epoch=20,threshold=0.8,**kwargs):
    X_bar,C=np.hstack((np.ones((X.shape[0],1)),X)),X.copy()
    Phi=kernel_matrix_TPS(C,X,D=2) 
    N1,D=X.shape
    # initlize 
    R=np.eye(D)
    beta,alpha=np.mean(Y,0)-np.mean(X.dot(R),0),np.zeros((C.shape[0],D))
    B=np.vstack((beta,R))
    Yhat=Phi.dot(alpha)+X_bar.dot(B) #Phi.dot(alpha)+X.dot(R)+beta 

    mass_diff=0
    Lambda=4*np.sum((vec_mean(Y)-vec_mean(X))**2)
    Delta,lower_bound=Lambda/8,Lambda/10000

    B_list,alpha_list=list(),list()  

    if len(record_index)==0:
        record_index=np.unique(np.linspace(0,n_iteration-1,num=int(n_iteration/10)).astype(np.int64))
        record_index.sort()
    
    if start_epoch==None:
        start_epoch=int(n_iteration/10)


    epoch=0
    while epoch<n_iteration:
        print('current, %i/%i'%(epoch,n_iteration),end='\r')
        B_pre=B.copy()
        projections=random_projections(D,n_projection,1)
        #Yhat_pre=Yhat.copy()
        R_pre,beta_pre,Yhat_pre=R.copy(),beta.copy(),Yhat.copy()
        domain_sum=np.full(N1,False)

        for (epoch2,theta) in enumerate(projections):
            Yhat_theta,Y_theta=np.dot(theta,Yhat.T),np.dot(theta,Y.T)
            Yhat_indice,Y_indice=Yhat_theta.argsort(),Y_theta.argsort()
            Yhat_s,Y_s=Yhat_theta[Yhat_indice],Y_theta[Y_indice]
            obj,phi,psi,piRow,piCol=solve_opt(Yhat_s,Y_s,Lambda)
            L=recover_indice(Yhat_indice,Y_indice,piRow)
            Domain,Range=L>=0,L[L>=0]
            domain_sum=np.logical_or(Domain,domain_sum) 
            Yhat[Domain]+=np.expand_dims(Y_theta[Range]-Yhat_theta[Domain],1)*theta
            mass=np.sum(Domain)
            mass_diff=mass-N0
            Lambda,Delta=update_lambda(Lambda,Delta,mass_diff,N0,lower_bound)
        if epoch>=start_epoch and np.linalg.norm(B-B_pre)<threshold:
            alpha,B=TPS_recover_parameter_gpu(Phi,X_bar,Yhat,eps,**kwargs)
        else:
            # find optimal R,S,beta, conditonal on alpha    
            Y_prime2=Yhat[domain_sum]-Phi[domain_sum].dot(alpha)
            R,S=recover_rotation_gpu(Y_prime2,X[domain_sum],**kwargs)
            beta=vec_mean(Y_prime2)-vec_mean(X[domain_sum].dot(R))
            B=np.vstack((beta,R))
        Yhat=Phi.dot(alpha)+X_bar.dot(B) #Phi.dot(alpha)+X.dot(R)+beta  
        if epoch in record_index:
            B_list.append(B),alpha_list.append(alpha)    
        epoch+=1
    return (B_list,alpha_list,Phi),record_index

def OPT_RBF(X,Y,N0,kernel=['Gaussian',[],0.1,3.0],n_iteration=200,record_index=[],start_epoch=None,threshold=0.8,**kwargs):
    if len(kernel[1])==0:
        kernel[1]=X.copy()
    C=kernel[1]
    Phi,eps=choose_kernel(kernel,X)
    N1,D=X.shape
    # initlize 
    R=np.eye(D)
    beta,alpha=np.mean(Y,0)-np.mean(X.dot(R),0),np.zeros((C.shape[0],D))
    Yhat=Phi.dot(alpha)+X.dot(R)+beta 
    epoch=0
    mu,nu=np.ones(Yhat.shape[0]),np.ones(Y.shape[0])
    R_list,beta_list,alpha_list=list(),list(),list()
    # period to record previous model: 
    period=10

    if len(record_index)==0:
        record_index=np.unique(np.linspace(0,n_iteration-1,num=int(n_iteration/10)).astype(np.int64))
        record_index.sort()
    B_list,alpha_list=list(),list()  
    if start_epoch==None:
        start_epoch=int(n_iteration/10)

    while epoch<n_iteration:
        print('current, %i/%i'%(epoch,n_iteration),end='\r')
        R_pre,beta_pre=R.copy(),beta.copy()
        M=cost_matrix_d(Yhat,Y)
        cost,gamma=opt_pr(mu, nu, M, N0, numItermax=1e7,numThreads=10)
        p1_hat=np.sum(gamma,1)
        Domain=p1_hat>1e-10
        BaryP=gamma.dot(Y)[Domain]/np.expand_dims(p1_hat,1)[Domain]
        Yhat[Domain]=BaryP  

        # find optimal R,S,beta, conditonal on alpha    
        if epoch>=start_epoch and np.linalg.norm(R-R_pre)+np.linalg.norm(beta-beta_pre)<threshold:
            Y_prime2=Yhat[Domain]-Phi[Domain].dot(alpha)
            R,S=recover_rotation_gpu(Y_prime2,X[Domain],**kwargs)
            beta=vec_mean(Y_prime2)-vec_mean(X[Domain].dot(R))
            Y_prime=Yhat[Domain]-X[Domain].dot(R)-beta
            alpha=recover_alpha_gpu(Phi[Domain],Y_prime,eps,**kwargs)
        else:
            Y_prime2=Yhat[Domain]-Phi[Domain].dot(alpha)
            R,S=recover_rotation_gpu(Y_prime2,X[Domain],**kwargs)
            beta=vec_mean(Y_prime2)-vec_mean(X[Domain].dot(R))

        Yhat=Phi.dot(alpha)+X.dot(R)+beta
        if epoch in record_index:
            R_list.append(R),beta_list.append(beta),alpha_list.append(alpha)
        epoch+=1
    return (R_list,beta_list,alpha_list,Phi),record_index

    

  

def OPT_TPS(X,Y,N0,eps=3.0,n_iteration=200,record_index=[],start_epoch=None,threshold=0.8,**kwargs):
    N1,D=X.shape
    C=X.copy()
    Phi=kernel_matrix_TPS(C,X,D=2)
    X_bar=np.hstack((np.ones((X.shape[0],1)),X))
    # initlize 
    R=np.eye(D)
    beta,alpha=np.mean(Y,0)-np.mean(X.dot(R),0),np.zeros((C.shape[0],D))
    B=np.vstack((beta,R))
    Yhat=Phi.dot(alpha)+X_bar.dot(B) #Phi.dot(alpha)+X.dot(R)+beta 
    epoch=0
    mu,nu=np.ones(Yhat.shape[0]),np.ones(Y.shape[0])
    alpha_list,B_list=list(),list()

    if len(record_index)==0:
        record_index=np.unique(np.linspace(0,n_iteration-1,num=int(n_iteration/10)).astype(np.int64))
    if start_epoch==None:
        start_epoch=int(n_iteration/10)

    while epoch<n_iteration:
        print('current, %i/%i'%(epoch,n_iteration),end='\r')
        B_pre=B.copy()
        M=cost_matrix_d(Yhat,Y)
        cost,gamma=opt_pr(mu, nu, M, N0, numItermax=1e7,numThreads=10)
        p1_hat=np.sum(gamma,1)
        Domain=p1_hat>1e-10
        BaryP=gamma.dot(Y)[Domain]/np.expand_dims(p1_hat,1)[Domain]
        Yhat[Domain]=BaryP
        if epoch>=start_epoch and np.linalg.norm(B-B_pre)<threshold:
            alpha,B=TPS_recover_parameter_gpu(Phi,X_bar,Yhat,eps,**kwargs)
        else:
            # find optimal R,S,beta, conditonal on alpha    
            Y_prime2=Yhat[Domain]-Phi[Domain].dot(alpha)
            R,S=recover_rotation_gpu(Y_prime2,X[Domain],**kwargs)
            beta=vec_mean(Y_prime2)-vec_mean(X[Domain].dot(R))
            B=np.vstack((beta,R))

        Yhat=Phi.dot(alpha)+X_bar.dot(B) #Phi.dot(alpha)+X.dot(R)+beta  

        if epoch in record_index:
            B_list.append(B),alpha_list.append(alpha)
        epoch+=1
        
    return (B_list,alpha_list,Phi),record_index


def TPS_RPM(X,Y,Lambda,eps=3.0,reg=0.001,n_iteration=200,record_index=[],**kwargs):
    N1,D=X.shape
    C=X.copy()
    Phi=kernel_matrix_TPS(C,X,D=2)
    X_bar=np.hstack((np.ones((X.shape[0],1)),X))
    # initlize 
    R=np.eye(D)
    beta,alpha=np.mean(Y,0)-np.mean(X.dot(R),0),np.zeros((C.shape[0],D))
    B=np.vstack((beta,R))
    Yhat=Phi.dot(alpha)+X_bar.dot(B) #Phi.dot(alpha)+X.dot(R)+beta 
    epoch=0
    mu,nu=np.ones(Yhat.shape[0]),np.ones(Y.shape[0])
    alpha_list,B_list=list(),list()

    if len(record_index)==0:
        record_index=np.unique(np.linspace(0,n_iteration-1,num=int(n_iteration/10)).astype(np.int64))
    

    while epoch<n_iteration:
        print('current, %i/%i'%(epoch,n_iteration),end='\r')
        B_pre=B.copy()
        M=cost_matrix_d(Yhat,Y)
        gamma=sinkhorn_knopp_opt(mu, nu, M, Lambda, reg, numItermax=1000)
        p1_hat=np.sum(gamma,1)
        Domain=p1_hat>1e-10
        BaryP=gamma.dot(Y)[Domain]/np.expand_dims(p1_hat,1)[Domain]
        Yhat[Domain]=BaryP
        if epoch>=start_epoch and np.linalg.norm(B-B_pre)<threshold:
            alpha,B=TPS_recover_parameter_gpu(Phi,X_bar,Yhat,eps,**kwargs)
        else:
            # find optimal R,S,beta, conditonal on alpha    
            Y_prime2=Yhat[Domain]-Phi[Domain].dot(alpha)
            R,S=recover_rotation_gpu(Y_prime2,X[Domain],**kwargs)
            beta=vec_mean(Y_prime2)-vec_mean(X[Domain].dot(R))
            B=np.vstack((beta,R))

        Yhat=Phi.dot(alpha)+X_bar.dot(B) #Phi.dot(alpha)+X.dot(R)+beta  

        if epoch in record_index:
            B_list.append(B),alpha_list.append(alpha)
        epoch+=1
        
    return (B_list,alpha_list,Phi),record_index

