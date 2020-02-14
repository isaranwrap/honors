#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 17:00:50 2020

@author: ishi
"""

#imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy import pi
import random

def getAngle(v1, v2):
  '''
  Input: two vectors, v1 & v2, which are np arrays of size (N, 2) --> time series of (x,y) coordinates, N time points
  Output: joint angle time series, np array of shape (N, ) --> time series of the angle between the two vectors
          // might need to reshape the array with .reshape(-1,1) to fit into multi-dimensional array of all joint angles (as I did in getJointAngles) 
  '''
  #cos = ((v1*v2).sum(axis=1))/(np.linalg.norm(v1, axis=1)*np.linalg.norm(v2, axis=1)) %%timeit 28.5 ms per loop vs 32.7 ms per loop
  cos = (v1[:,0]*v2[:,0] + v1[:,1]*v2[:,1])/(np.linalg.norm(v1, axis=1)*np.linalg.norm(v2, axis=1))
  return np.arccos(cos)

def getJointAngles(pos, origin_index1 = 4, origin_index2 = 3, num_angles=30):
  ''' 
  Input: position time series, np(or h5py) array of shape (N, 2, j) --> time series of (x,y) coords for j joints tracked
    (optional) origin_index1, origin_index2 --> which indices to define the origin from (default 3 & 4 for fly, corresponding to neck & thorax)
  Output: angle time series, np array of shape (N-nans, 1, num_angles) --> gets rid of all the rows with nans in the angles, and 
        (for the fly) there are num_angles=30 total angles which are computed (specified below)

  Angles 
  -------
  # 0 is the head
  # 1 & 2 are L & R eyes, respectively
  # 3 & 4 are L & R wings, respectvely
  # 5 is the abdomen
  # 6 - 9 are the R foreleg joints
  # 10 - 13 are the R midleg joints
  # 14 - 17 are the R hindleg joints
  # 18 - 21 are the L foreleg joints
  # 22 - 25 are the L midleg joints
  # 26 - 29 are the L hindleg joints
  '''

  # need to make thorax--->neck the base line
  # and every other angle relative to that
  origin = pos[:,:,origin_index1]
  neck = pos[:,:, origin_index2]

  N = pos.shape[0]
  #x is the line everything will be referenced to 
  x = neck - origin

  #initialize empty array for joint angles --> this will be shape of array returned
  joint_angles = np.empty((N, 1, num_angles))
  joints_list = set(np.arange(32)) - {3, 4, 30, 31} # get rid of the joints which we are using to define our origin line... doesn't make sense to
  # have an angle between them.... and then 30,31 out of index range --> add back after loop

  #populate joint_angles array
  for p in joints_list: #find angle between x-vector and joint vector
    joint_angles[:,:, p] = getAngle(x, pos[:,:,p] - origin).reshape(-1,1) 
  joint_angles[:,:,3] = getAngle(x, pos[:,:,30] - origin).reshape(-1,1) #make joint_angles 3 and 4 the wings
  joint_angles[:,:,4] = getAngle(x, pos[:,:,31] - origin).reshape(-1,1) #since they are definining origin
  #np.where(np.isnan(joint_angles))[0].shape ([1] or [2] as well) is 29... so 29 null vals 
  # --> np.isnan(joint_angles).sum() gives 26... presumably because of frame 48 which was giving hella problems earlier

  #remove nans
  nan_indices = np.where(np.isnan(joint_angles))[0] #rows which have a SINGLE nan in any of the cols
  notnans = [i for i in range(N) if i not in nan_indices] #the indices which aren't nans
  joint_angles = joint_angles[notnans,0,:]
  return joint_angles

def create_dataset(ja, lag=10, num_samples=1000, indices = None):
  #number of angles
  num_angles = ja.shape[1] # just for readability

  #initialize & shuffle indices, if need be 
  if indices == None:
      indices = np.arange(ja.shape[0]-lag) #can't have the ones where adding the lag would get index out of range... so subtract lag
      random.shuffle(indices)
      
  #initialize empty X & Y datasets
  X = np.empty((num_samples, lag, num_angles))
  Y = np.empty((num_samples, num_angles))

  for i in range(num_samples): #num_samples we want created
    indx = indices[i]
    X[i, ...] = ja[indx:indx+lag].reshape(10,30) #shape is (lag, 1, num_joints) --> (10 x 1 x 30) ... 
    Y[i, ...] = ja[indx+lag].reshape(1,30)
  return X, Y, indices