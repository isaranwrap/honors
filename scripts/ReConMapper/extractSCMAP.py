import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import matplotlib
matplotlib.use('TkAgg')
import tensorflow as tf
import tensorflow.keras as k

import numpy as np

import time
import matplotlib.pyplot as plt
from DLCmobilenet_v2 import MobileNetV2

# import keras as k

class SCMapNet(object):
    def __init__(self, wpath=None):
        self.alpha = 0.35
        self.wpath = wpath
        if self.wpath is None:
            self.wpath = '/Users/kj/Rat-Kanishk-2019-12-05/dlc-models/iteration-1/RatDec5-trainset95shuffle1/train' \
                         '/snapshot-1030000'
        pass

    def _loadDLCweights(self):
        sess = tf.compat.v1.InteractiveSession()
        loader = tf.compat.v1.train.import_meta_graph(self.wpath + '.meta')

        loader.restore(sess, self.wpath)
        dlcweights = {}
        for v in tf.compat.v1.global_variables():
            try:
                dlcweights[v.name] = v.eval()
            except tf.errors.FailedPreconditionError as E:
                pass
        sess.close()
        return dlcweights

    def _loadDLCModel(self):
        weights_DLC = self._loadDLCweights()
        k2 = [i for i in list(weights_DLC.keys())]  # if 'MobilenetV2' in i]
        mnet = MobileNetV2(input_shape=(800,800,3), include_top=False, alpha=self.alpha, backend=k.backend, layers = k.layers, models = k.models,
                           utils = k.utils)
        part_pred = k.layers.Conv2DTranspose(10, kernel_size=(3, 3), strides=(2, 2), padding='same')(mnet.output)
        locref = k.layers.Conv2DTranspose(20, kernel_size=(3, 3), strides=(2, 2), padding='same')(mnet.output)

        dlc_mnet = k.Model(inputs=mnet.input, outputs=[part_pred, locref])

        assert len(k2) == len(dlc_mnet.weights)

        if 'moving' not in ''.join([i.name for i in dlc_mnet.weights[:10]]):
            k2 = [i for i in k2 if 'moving' not in i] + [i for i in k2 if 'moving' in i]
            print('DLC weights realigned.')
        mismatch = False
        for i in range(len(dlc_mnet.weights)):
            if dlc_mnet.weights[i].shape != weights_DLC[k2[i]].shape:
                print(i, dlc_mnet.weights[i], k2[i])
                mismatch = True
                break
            else:
                pass
        if not mismatch:
            print('No mismatch found...')
            dlc_mnet.set_weights([weights_DLC[i] for i in k2])
            print('DLC WEIGHTS LOADED')
        else:
            print('Weights not restored to keras model...')

        return dlc_mnet

    def save_scmap(self):
        model = self._loadDLCModel()
        imgs = SCMAP_loader(batchsize=32)
        sp = savePreds()
        sp.imgs = imgs
        preds = model.predict_generator(imgs, verbose=1, callbacks=[sp])


class savePreds(k.callbacks.Callback):
    def __init__(self):
        pass
    def on_predict_batch_end(self, batch, logs):
        t1 = time.time()
        stride = 8.0
        locref_stdev = 7.2801
        probs, locref = 1 / (1 + np.exp(-logs['outputs'][0])), logs['outputs'][1]

        l_shape = np.shape(probs)  # batchsize times x times y times body parts
        # locref = locref*cfg.locref_stdev
        locref = np.reshape(locref, (l_shape[0], l_shape[1], l_shape[2], l_shape[3], 2))
        # turn into x times y time bs * bpts
        locref = np.transpose(locref, [1, 2, 0, 3, 4])
        probs = np.transpose(probs, [1, 2, 0, 3])

        # print(locref.get_shape().as_list())
        # print(probs.get_shape().as_list())
        l_shape = np.shape(probs)  # x times y times batch times body parts

        locref = np.reshape(locref, (l_shape[0] * l_shape[1], -1, 2))
        probs = np.reshape(probs, (l_shape[0] * l_shape[1], -1))
        maxloc = np.argmax(probs, axis=0)
        loc = np.unravel_index(maxloc,
                               (l_shape[0], l_shape[1]))  # tuple of max indices

        maxloc = np.reshape(maxloc, (1, -1))
        joints = np.reshape(np.arange(0, l_shape[2] * l_shape[3]), (1, -1))
        indices = np.transpose(np.concatenate([maxloc, joints], axis=0))

        # extract corresponding locref x and y as well as probability
        offset = locref[indices[:,0],indices[:,1],:]
        offset = offset[:,[1, 0]]
        likelihood = np.reshape(probs[indices[:,0], indices[:,1]], (-1, 1))

        pose = stride * np.transpose(loc) + stride * 0.5 + offset * locref_stdev
        pose = np.concatenate([pose, likelihood], axis=1)
        pose[:, [0, 1, 2]] = pose[:, [1, 0, 2]]
        pose = np.reshape(pose, (logs['size'], -1))

        fig, ax = plt.subplots()
        t = 0
        ax.imshow((self.imgs.__getitem__(batch)[t]+ [123.68, 116.779, 103.939]).astype('uint8'))
        ax.scatter(pose[t,::3], pose[t,1::3],s=10)
        plt.show()

        # out = np.zeros((logs['size'], 800, 800, scmap.shape[-1]), dtype='float32')
        # i_t = (np.pad(np.atleast_2d(np.arange(locrefx.shape[-2])).T, ((0, 0), (0, locrefx.shape[-2] - 1)),
        #               'edge') + 0.5) * stride
        #
        # j_t = (np.pad(np.atleast_2d(np.arange(locrefy.shape[-2])), ((0, locrefy.shape[-2] - 1), (0, 0)),
        #               'edge') + 0.5) * stride
        #
        # locrefx = locrefx + np.pad(np.atleast_3d(i_t), ((0, 0), (0, 0), (0, 9)), 'edge')
        # locrefy = locrefy + np.pad(np.atleast_3d(j_t), ((0, 0), (0, 0), (0, 9)), 'edge')
        #
        # for i in range(logs['size']):
        #     for j in range(10):
        #         out[i, locrefx[i, :, :, j].flatten().round().astype('int'), locrefy[i, :, :,
        #                                                                     j].flatten().round().astype(
        #             'int'), j] = scmap[i, :, :, j].flatten()
        # print(time.time() - t1)
        # fig, ax = plt.subplots()
        # # ax.imshow(self.imgs.__getitem__(0)[0])
        # for i in range(10):
        #     x, y = np.unravel_index(np.argmax(out[0, :, :, i]), (800, 800))[::-1]
        #     ax.scatter(y / 0.8, x / 0.8, s=5)
        #
        # print('done')

from moviepy.editor import VideoFileClip
import cv2
import time
import os
import shutil
from skimage.util import img_as_ubyte

class SCMAP_loader(k.utils.Sequence):
    def __init__(self, vpath=None, batchsize=16):
        self.vpath = '/Users/kj/Rat-Kanishk-2019-12-05/videos/01_118_clip2.mp4' if vpath is None else vpath
        self.batchsize = batchsize
        self.nsamples = len(os.listdir(self.vpath.split('.')[0]))

    def __len__(self):
        return int(self.nsamples / self.batchsize)

    def __getitem__(self, index):
        # 10 second load time for each batch
        scale = 1.0
        t1 = time.time()
        images = np.zeros((self.batchsize, int(800*scale), int(800*scale), 3), dtype='uint8')
        for i, ind1 in enumerate(range(index, index+self.batchsize)):
                images[i] = cv2.resize(img_as_ubyte(cv2.cvtColor(cv2.imread(self.vpath.split('.')[0] + '/frame%05d.png'
                                        % (ind1 + 1)), cv2.COLOR_BGR2RGB)), None, fx=scale, fy=scale,
                                       interpolation=cv2.INTER_AREA)
        return images - [123.68, 116.779, 103.939]

    def on_epoch_end(self):
        pass


sc = SCMapNet()
sc.save_scmap()
