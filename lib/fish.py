import numpy as np
import matplotlib.pyplot as plt
import ot
import torch
import os

def toNp(X):
    return X.data.cpu().numpy()

def scale_fish(fishin, k=.65):
    '''Scale the fish by a factor of k'''
    return k * fishin

def make_noise(fishin,p,k=2):
    '''
    Add p% noise to a k-dim fish
    '''
    N = len(fishin)
    noise = np.random.uniform(low=-1.5,high=1.5,size=(int(N*p/100),2))
    fishout = np.concatenate((fishin, noise))
    return noise, fishout

def reduce_fish(fishin, p):
    n = int(len(fishin) * p / 100)
    ind = np.random.choice(len(fishin), n, replace=False)
    fishout = np.delete(fishin, ind, axis = 0)
    return fishout

def shear_fish(fishin):
    theta = np.pi / 6.0
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    shear = np.array([[1, 0.5], [0, 1]])
    R = np.dot(R, shear)
    t = np.array([0.5, 1.0])
    fishout = np.dot(fishin, R) + t
    return fishout