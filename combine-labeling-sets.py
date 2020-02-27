import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.io as sci
import h5py 

fly1 = h5py.File('fly1_labelingset.h5', 'r') # list(fly1.keys()) gives ['box', 'clusters', 'dataIdx', 'exptID', 'pca']
orig = h5py.File('labeling_setn=3500.h5', 'r') # list(orig.keys()) only gives ['box', 'ans'] (which are the same...) 

f1 = fly1['box']

#list(fly1.keys())

comb = np.concatenate((np.array(fly1['box']), np.array(orig['box'])))

with h5py.File('combined.h5', 'w') as f:
  f.create_dataset('box', data=comb)
