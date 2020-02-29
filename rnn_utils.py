#!/usr/bin/env python
# coding: utf-8

# ## Copied from rnn_utils.py on local comp

# In[1]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_d on Wed Jan 29 17:00:50 2020

@author: ishi
"""

#imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy import pi
import random

import os
import h5py
from sklearn import preprocessing

from scripts import dlc_rnn

def nan_helper(y):
    """Helper function to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

rightJointIndices = [16, 17, 30]#note extra 5 to left indices, arbitrarily added
leftJointIndices  = [28, 29, 31,5]#just so that it's not hopping back n forth
def getAngle(v1, v2, p=None):
  '''
  Input: two vectors, v1 & v2, which are np arrays of size (N, 2) --> time series of (x,y) coordinates, N time points
  Output: joint angle time series, np array of shape (N, ) --> time series of the angle between the two vectors
          // might need to reshape the array with .reshape(-1,1) to fit into multi-dimensional array of all joint angles (as I did in getJointAngles) 
  '''
#old --> I was using cosines.... switched over to arctan
  #cos = ((v1*v2).sum(axis=1))/(np.linalg.norm(v1, axis=1)*np.linalg.norm(v2, axis=1)) %%timeit 28.5 ms per loop vs 32.7 ms per loop
  # cos = (v1[:,0]*v2[:,0] + v1[:,1]*v2[:,1])/(np.linalg.norm(v1, axis=1)*np.linalg.norm(v2, axis=1))
  #again --> try 2, this was wrong.... LOL --> y, x = v2[:,1] - v1[:,1], v2[:,0] - v1[:,0]
  
 #yet another time I fixed it... this time adding if statements so that oscillations don't occur. 
  ang1 = np.arctan2(v1[:,1], v1[:,0])
  ang2 = np.arctan2(v2[:,1], v2[:,0])
  ang = ang2-ang1
 #this is the most current implementation:
  if p is not None:
    if p in rightJointIndices:
        ang[ang < 0] = ang[ang < 0] + 2*np.pi
        return ang
    elif p in leftJointIndices:
        ang[ang > 0] = ang[ang > 0] - 2*np.pi
        return ang
  return ang

def getJointAngles(pos, origin_index1 = 4, origin_index2 = 3, num_angles=30):
  ''' 
  Input: position time series, np(or h5py) array of shape (N, 2, j) --> time series of (x,y) coords for j joints tracked
    (optional) origin_index1, origin_index2 --> which indices to define the origin from (default 3 & 4 for fly, corresponding to neck & thorax)
  Output: angle time series, np array of shape (N, 1, num_angles) --> linearly interpolates the nan values, and 
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
        joint_angles[:,:, p] = getAngle(x, pos[:,:,p] - origin, p = p).reshape(-1,1) 
  joint_angles[:,:,3] = getAngle(x, pos[:,:,30] - origin, p=30).reshape(-1,1) #make joint_angles 3 and 4 the wings
  joint_angles[:,:,4] = getAngle(x, pos[:,:,31] - origin, p=31).reshape(-1,1) #since they are definining origin
  #np.where(np.isnan(joint_angles))[0].shape ([1] or [2] as well) is 29... so 29 null vals 
  # --> np.isnan(joint_angles).sum() gives 26... presumably because of frame 48 which was giving hella problems earlier

  #remove nans
  #nan_indices = np.where(np.isnan(joint_angles))[0] #rows which have a SINGLE nan in any of the cols
  #notnans = [i for i in range(N) if i not in nan_indices] #the indices which aren't nans
  #joint_angles = joint_angles[notnans,0,:]
    
  #linearly interpolate nans
  nans, filter_func = nan_helper(joint_angles)
  joint_angles[nans] = np.interp(filter_func(nans), filter_func(~nans), joint_angles[~nans])
  joint_angles = joint_angles.reshape((N, num_angles))
  return joint_angles
 

#Old function, extract_samples not necessary anymore since new dataset creation operates on matrices
def extract_samples(ja, lag=10, num_samples=1000, indices = None, Y_shape = ('N','num_feats'), stateful=True):
  #number of angles
  num_angles = ja.shape[1] # just for readability
  #initialize & shuffle indices, if need be 
  if indices == None:
      indices = np.arange(ja.shape[0]-lag) #can't have the ones where adding the lag would get index out of range... so subtract lag
      random.shuffle(indices)
      
  #initialize empty X & Y datasets
  X = np.empty((num_samples, lag, num_angles))
  if Y_shape == ('N', 'num_feats'):
    Y = np.empty((num_samples, num_angles))
  elif Y_shape == ('N', 'lag', 'num_feats'):
    Y = np.empty((num_samples, lag, num_angles))

  for indx in range(num_samples): #num_samples we want created
    indx = indices[i]
    X[i, ...] = ja[indx:indx+lag].reshape(lag,num_angles) #shape is (lag, 1, num_joints) --> (10 x 1 x 30) ... 
    if Y_shape == ('N', 'num_feats'):
        Y[i, ...] = ja[indx+lag].reshape(1,num_angles)
    if Y_shape == ('N', 'lag', 'num_feats'):
        Y[i, ...] = ja[indx+1:indx+lag+1].reshape(lag, num_angles)
  return X, Y, indices

def create_dataset(jaFolder, num_samples=int(1e5),
                   batch_size=32,num_angles=30, depth=1,
                   offset=int(4e4),lookback=100):
    num_batches = int(num_samples/batch_size)
    ja_slice = num_batches*lookback
    Xuncut = np.empty((batch_size, ja_slice, num_angles))
    Yuncut = np.empty((batch_size, ja_slice, num_angles))
    for indx, file in enumerate(os.listdir(jaFolder)):
        ja = h5py.File(os.path.join(jaFolder,file))['joint_angles']
        Xuncut[indx,...] = ja[:ja_slice]
        Yuncut[indx,...] = ja[depth:ja_slice+depth]

    f1, f2 = os.listdir(jaFolder)[0:2]
    ja1 = h5py.File(os.path.join(jaFolder,f1))['joint_angles']
    ja2 = h5py.File(os.path.join(jaFolder,f2))['joint_angles']
    Xuncut[30,...] = ja1[offset:offset+ja_slice]
    Xuncut[31,...] = ja2[offset:offset+ja_slice]
    Yuncut[30,...] = ja1[offset+depth:offset+ja_slice+depth]
    Yuncut[31,...] = ja2[offset+depth:offset+ja_slice+depth]
    #shape is 32 x 100000 x 30

    X = np.empty((num_samples,lookback,num_angles))
    Y = np.empty((num_samples,lookback,num_angles))
    for b in range(num_batches):
        Xbatch = Xuncut[:,b*lookback:(b+1)*lookback,:]
        Ybatch = Yuncut[:,b*lookback:(b+1)*lookback,:]
        X[b*batch_size:b*batch_size+batch_size] = Xbatch
        Y[b*batch_size:b*batch_size+batch_size] = Ybatch
    return X, Y

#generally, should use draw_skeleton() instead... since it plots w/ skeleton overlayed
def plot_fly(pos, type='pos', num_angles=30,wings=2,frame=1,skeleton=True):
    for ang in range(num_angles+wings):
        plt.scatter(pos[frame,0,ang], pos[frame,1,ang], marker='$'+str(ang)+'$', s=200)
    plt.gca().invert_yaxis()
    
def normalize_set(X):
    num_samples, lookback, num_angles = X.shape
    Xflat = X.reshape(num_samples*lookback,num_angles)

    scaler = preprocessing.StandardScaler().fit(Xflat)
    Xnorm = scaler.transform(Xflat)
    Xnorm = Xnorm.reshape((num_samples, lookback, num_angles))
    return Xnorm, scaler

def shape(*args):
    shapes = list()
    for arg in args:
        shapes.append(arg.shape)
    return shapes

def draw_line(arr, point1, point2, dtype = 'pos', **kwargs):
        xs = [arr[0, point1], arr[0, point2]]
        ys = [arr[1, point1], arr[1, point2]]
        plt.plot(xs, ys, **kwargs)
    
def draw_skeleton(arr=None, means=None, color = 'red', linewidth=0.7, input_type='ang'):
    if input_type == 'pos':
        plt.figure(figsize=(10,8))
        for p in range(6, 9, 1):
            draw_line(arr, p, p+1, color=color, linewidth=linewidth)
        for p in range(10, 13, 1):
            draw_line(arr, p, p+1, color=color, linewidth=linewidth)
        for p in range(14, 17, 1):
            draw_line(arr, p, p+1, color=color, linewidth=linewidth)
        for p in range(18, 21, 1):
            draw_line(arr, p, p+1, color=color, linewidth=linewidth)
        for p in range(22, 25, 1):
            draw_line(arr, p, p+1, color=color, linewidth=linewidth)
        for p in range(26, 29, 1):
            draw_line(arr, p, p+1, color=color, linewidth=linewidth)
        draw_line(arr, 2, 0, linewidth=linewidth)
        draw_line(arr, 1, 0, linewidth=linewidth)
        draw_line(arr, 5, 4, linewidth=linewidth)
        draw_line(arr, 4, 3, color='purple', linewidth=linewidth)
        draw_line(arr, 30, 3, linewidth=linewidth)
        draw_line(arr, 31, 3, linewidth=linewidth)
        for i in range(32):
            plt.scatter(arr[0,i], arr[1,i], marker='$' + str(i)+ '$', s=200)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        
    if input_type == 'ang':
        if means is None:
            means = np.array([[145.4873612 , 133.85781315, 133.67147231, 126.58415065,
         97.8499675 ,  49.04866306, 115.03435667, 126.57664241,
        136.6654787 , 147.54284296, 104.14832639,  98.08537269,
         97.5099163 ,  93.44100222,  92.59418426,  80.12077111,
         67.43725833,  46.79131352, 115.27751259, 126.66277787,
        136.48709333, 147.22914222, 104.19417769,  98.32634593,
         97.87074148,  94.08399361,  92.70216287,  80.31333463,
         67.70918454,  47.12081435,  18.57976667,  18.75787278],
       [ 94.9058512 ,  78.1094025 , 111.35902611,  94.62192528,
         94.47558787,  93.8548138 , 103.76169704, 110.83637648,
        109.35607269, 113.65153019, 103.97928528, 118.41667648,
        121.39174278, 135.12658296, 103.38936426, 116.27441046,
        111.59319704, 120.60877898,  85.77009241,  78.37048722,
         79.6991313 ,  75.19082435,  85.21323407,  70.76826157,
         67.4492088 ,  53.65891204,  85.60276287,  72.56368139,
         76.95799972,  67.81673   ,  81.54497037, 106.54920185]])
            x_means = means[0,:] - means[0,4]
            y_means = means[1,:] - means[1,4]
            mean_lngth = np.sqrt(x_means**2 + y_means**2)
        ja_ = np.zeros(32)
        ja_[:30] = arr[:]
        ja_[30] = arr[3]
        ja_[31] = arr[4]
        ja_[3] = 0
        ja_[4] = 0
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, polar=True)
        for p in range(6, 9, 1):
            ax.plot([ja_[p],ja_[p+1]], [mean_lngth[p],mean_lngth[p+1]], color='red', linewidth=linewidth)
        for p in range(10, 13, 1):
            ax.plot([ja_[p],ja_[p+1]], [mean_lngth[p],mean_lngth[p+1]], color='red', linewidth=linewidth)
        for p in range(14, 17, 1):
            ax.plot([ja_[p],ja_[p+1]], [mean_lngth[p],mean_lngth[p+1]], color='red', linewidth=linewidth)
        for p in range(18, 21, 1):
            ax.plot([ja_[p],ja_[p+1]], [mean_lngth[p],mean_lngth[p+1]], color='red', linewidth=linewidth)
        for p in range(22, 25, 1):
            ax.plot([ja_[p],ja_[p+1]], [mean_lngth[p],mean_lngth[p+1]], color='red', linewidth=linewidth)
        for p in range(26, 29, 1):
            ax.plot([ja_[p],ja_[p+1]], [mean_lngth[p],mean_lngth[p+1]], color='red', linewidth=linewidth)
        ax.plot([ja_[2],ja_[0]], [mean_lngth[2],mean_lngth[0]], linewidth=linewidth)
        ax.plot([ja_[1],ja_[0]], [mean_lngth[1],mean_lngth[0]], linewidth=linewidth)
        ax.plot([ja_[5],ja_[4]], [mean_lngth[5],mean_lngth[4]], linewidth=linewidth)
        ax.plot([ja_[4],ja_[3]], [mean_lngth[4],mean_lngth[3]], linewidth=linewidth)
        ax.plot([ja_[30],ja_[3]], [mean_lngth[30],mean_lngth[3]], linewidth=linewidth)
        ax.plot([ja_[31],ja_[3]], [mean_lngth[31],mean_lngth[3]], linewidth=linewidth)
        for i in range(32):
            ax.scatter(ja_[i], mean_lngth[i], marker='$' + str(i)+ '$', s=200)
        ax.set_yticklabels([])
        #ax.set_xticklabels([])
        ax.xaxis.grid(False)
        ax.yaxis.grid(False)
        #ax.spines['polar'].set_visible(False)
        #ax.set_theta_zero_location('E')
        return fig, ax
#Quick check:
#np.all(scaler.mean_ == X.mean(axis=0).mean(axis=0))
#Xnorm[:80000].mean(), Xnorm[:80000].std()
#Xnorm[80000:].mean(), Xnorm[80000:].std()

def getFeedForwardTimeSeries(h5filePath = '/home/ishan/honors_thesis/joint_angles/ja_072212_163153.h5',timeSeriesName = 'joint_angles', modelPath = '/home/ishan/honors_thesis/models/flyRNN_batchsize=32.hdf5', NN_type = '1 layer', gputouse=1):
    h5f = h5py.File(h5filePath)[timeSeriesName] #read in time series
    indx = np.random.randint(h5f.shape[0]) #pick random index to get consecutive 1000-time series from
    X_drive = np.array(h5f)[indx:indx+1024, np.newaxis, :] #subset array, reshape for RNN
    
    if NN_type == '1 layer':
        model = dlc_rnn.LSTM1Dense1(mpath=modelPath, gputouse=gputouse).model
    elif NN_type == '2 layers':
        model = dlc_rnn.LSTM2Dense1(mpath=modelPath, gputouse=gputouse).model
    
    ff = np.empty((2000,1,30))
    preds = model.predict(X_drive)
    last_pred = preds[-1][np.newaxis,...]
    ff_pred = model.predict(last_pred)
    ff[0,:,:] = ff_pred
    for i in range(1,2000):
        ff[i,:,:] = model.predict(ff[i-1:i,:,:]) 
    return ff, X_drive
