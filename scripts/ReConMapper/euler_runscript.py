import os

os.environ['DLClight'] = 'True'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    warnings.filterwarnings('ignore', category=Warning)
    import deeplabcut
    from recon import *
    from dataset.dataloaders import VideoLoader
    import tensorflow.keras as keras

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import cv2


task_id = 'test'
rc = ReCon(
    wpath='/home/kanishk/Rat-Kanishk-2019-12-05/dlc-models/iteration-1/RatDec5-trainset95shuffle1/train/snapshot-1030000',)
    # restore_model_path='/home/kanishk/ongpu/weights-improvement-1829-310.16.hdf5')
# rc.model[0].summary()

train_gen = VideoLoader(vpath='/home/kanishk/videoclips/01_118_clip2.mp4',
                        h5path='/home/kanishk/videoclips/01_118_clip2DLC_mobnet_100_RatDec5shuffle1_1030000.h5',
                        batchsize=16,
                        lag=10,
                        nsamples=1000)
#
# filepath = "/home/kanishk/weights-improvement-{epoch:02d}-{loss:.2f}.hdf5"
# checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, mode='min',
#                                              save_freq=10)
#
callbacks_list = [keras.callbacks.TerminateOnNaN(),]# checkpoint, keras.callbacks.ReduceLROnPlateau('loss'),
#                   #keras.callbacks.EarlyStopping(monitor='loss', min_delta=10, verbose=0, patience=50,
#                   #                             restore_best_weights=True)]
#
history = rc.model[0].fit_generator(generator=train_gen, use_multiprocessing=False,
                                    workers=1, epochs=10000, verbose=2,
                                    callbacks=callbacks_list)
# rc.model[0].save('/home/kanishk/ongpu/'+task_id + '_ReConModel_' + datetime.now().strftime("%m-%d-%Y_%H_%M_%S") + '.h5')
# import pickle
#
# with open('train_history.pickle', 'wb') as f:
#     pickle.dump(history.history, f)
#
# #Started on 4:14PM
