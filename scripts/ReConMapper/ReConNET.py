import matplotlib
matplotlib.use('TkAgg')
import tensorflow as tf
import tensorflow.keras as k
import numpy as np

import os
os.environ['DLClight'] = 'True'
import deeplabcut
from pathlib import Path

import pandas as pd
#comment below line at runtime
import keras as k
import copy
import matplotlib.pyplot as plt

class ReConNet(object):
    def __init__(self, gputouse=None):
        if isinstance(gputouse, int):
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gputouse)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        self.model = self.buildmodel()


    def buildmodel(self):
        model = k.models.Sequential()
        model.add(k.layers.LSTM(64, input_shape=(None, 20)))
        model.add(k.layers.Dense(2))
        model.compile(loss='mse', optimizer='adam')
        return model

    def prep_data(self):
        pass




class DLCMultiOutputh5(object):
    def __init__(self, config=None, videos=None, shuffle=1, trainingsetindex=0, batchsize=None, videotype='avi',
                 destfolder=None, save_as_csv=False, gputouse=None):
        """Currently only one video as input."""
        videos = ['/Users/kj/Rat-Kanishk-2019-12-05/videos/01_118_clip3.mp4'] if videos is None else videos
        config = '/Users/kj/Rat-Kanishk-2019-12-05/config.yaml' if config is None else config
        self.loadh5(config, videos, shuffle, trainingsetindex, batchsize, videotype, destfolder, save_as_csv, gputouse)
        print('***RECON: All videos loaded.')



    def loadh5(self, config, videos, shuffle, trainingsetindex, batchsize, videotype, destfolder, save_as_csv, gputouse):

        self.h5s = {}
        start_path = os.getcwd()  # record cwd to return to this directory in the end

        cfg = deeplabcut.auxiliaryfunctions.read_config(config)
        trainFraction = cfg['TrainingFraction'][trainingsetindex]

        modelfolder = os.path.join(cfg["project_path"],
                                   str(deeplabcut.auxiliaryfunctions.GetModelFolder(trainFraction, shuffle, cfg)))
        path_test_config = Path(modelfolder) / 'test' / 'pose_cfg.yaml'
        try:
            dlc_cfg = deeplabcut.pose_estimation_tensorflow.load_config(str(path_test_config))
        except FileNotFoundError:
            raise FileNotFoundError(
                "It seems the model for shuffle %s and trainFraction %s does not exist." % (shuffle, trainFraction))
        try:
            Snapshots = np.array(
                [fn.split('.')[0] for fn in os.listdir(os.path.join(modelfolder, 'train')) if "index" in fn])
        except FileNotFoundError:
            raise FileNotFoundError(
                "Snapshots not found! It seems the dataset for shuffle %s has not been trained/does not exist.\n Please train it before using it to analyze videos.\n Use the function 'train_network' to train the network for shuffle %s." % (
                shuffle, shuffle))

        if cfg['snapshotindex'] == 'all':
            print(
                "Snapshotindex is set to 'all' in the config.yaml file. Running video analysis with all snapshots is very costly! Use the function 'evaluate_network' to choose the best the snapshot. For now, changing snapshot index to -1!")
            snapshotindex = -1
        else:
            snapshotindex = cfg['snapshotindex']

        increasing_indices = np.argsort([int(m.split('-')[1]) for m in Snapshots])
        Snapshots = Snapshots[increasing_indices]

        print("Using %s" % Snapshots[snapshotindex], "for model", modelfolder)

        ##################################################
        # Load and setup CNN part detector
        ##################################################

        # Check if data already was generated:
        dlc_cfg['init_weights'] = os.path.join(modelfolder, 'train', Snapshots[snapshotindex])
        trainingsiterations = (dlc_cfg['init_weights'].split(os.sep)[-1]).split('-')[-1]
        # Update number of output and batchsize
        dlc_cfg['num_outputs'] = cfg.get('num_outputs', dlc_cfg.get('num_outputs', 1))

        if batchsize == None:
            # update batchsize (based on parameters in config.yaml)
            dlc_cfg['batch_size'] = cfg['batch_size']
        else:
            dlc_cfg['batch_size'] = batchsize
            cfg['batch_size'] = batchsize

        DLCscorer, DLCscorerlegacy = deeplabcut.auxiliaryfunctions.GetScorerName(cfg, shuffle, trainFraction,
                                                                      trainingsiterations=trainingsiterations)
        if dlc_cfg['num_outputs'] <= 1:
            raise ValueError('***RECON: Please ensure num_outputs is greater than 1 in ' + str(path_test_config))

        Videos = deeplabcut.auxiliaryfunctions.Getlistofvideos(videos, videotype)
        net_loaded = False

        if len(Videos) > 0:
            # looping over videos

            for video in Videos:
                vname = Path(video).stem
                if destfolder is None:
                    destfolder = str(Path(video).parents[0])
                notanalyzed, dataname, DLCscorer = deeplabcut.auxiliaryfunctions.CheckifNotAnalyzed(destfolder, vname,
                                                                                                    DLCscorer,
                                                                                                    DLCscorerlegacy)
                if not notanalyzed:
                    h5 = pd.read_hdf(dataname, 'df_with_missing')
                    if h5.shape[1] != len(dlc_cfg['all_joints'])*3*dlc_cfg['num_outputs']:
                        if input('***RECON: ' + video + ' was already analyzed and h5 file was found. However the num_'
                                 'outputs is inconsistent in h5 file. Do you want to analyze this video again? Please '
                                 'backup current h5 file if ou choose to say yes. The file will be skipped if chosen N.'
                                 '[y/N] : ') == 'y':
                            notanalyzed = True
                            print('***RECON: Video will be reanalyzed.')
                        else:
                            print('***RECON: Video skipped.')
                            continue
                if notanalyzed:
                    if not net_loaded:
                        if 'TF_CUDNN_USE_AUTOTUNE' in os.environ:
                            del os.environ['TF_CUDNN_USE_AUTOTUNE']  # was potentially set during training

                        if gputouse is not None:  # gpu selection
                            os.environ['CUDA_VISIBLE_DEVICES'] = str(gputouse)

                        tf.reset_default_graph()
                        sess, inputs, outputs = deeplabcut.pose_estimation_tensorflow.predict.setup_pose_prediction(
                            dlc_cfg)
                        net_loaded = False

                    print('***RECON: '+ vname + 'wasn\'t analyzed, analyzing now.')
                    TFGPUinference = False

                    xyz_labs_orig = ['x', 'y', 'likelihood']
                    suffix = [str(s + 1) for s in range(dlc_cfg['num_outputs'])]
                    suffix[0] = ''  # first one has empty suffix for backwards compatibility
                    xyz_labs = [x + s for s in suffix for x in xyz_labs_orig]
                    pdindex = pd.MultiIndex.from_product([[DLCscorer],
                                                          dlc_cfg['all_joints_names'],
                                                          xyz_labs],
                                                         names=['scorer', 'bodyparts', 'coords'])

                    DLCscorer = deeplabcut.pose_estimation_tensorflow.AnalyzeVideo(video, DLCscorer, DLCscorerlegacy,
                            trainFraction, cfg, dlc_cfg, sess, inputs, outputs, pdindex, save_as_csv, destfolder,
                                                                               TFGPUinference, dynamic=(False,.5,10))
                self.h5s[dataname] = pd.read_hdf(dataname, 'df_with_missing')



        else:
            print("No video/s found. Please check your path!")
        os.chdir(str(start_path))


    def getBpart(self, dataname = None, ind=0):
        """Once data loaded, fetch the multi-output time series for a particular bodypart."""
        a = list(self.h5s.values())[0].values
        a = np.reshape(a, (a.shape[0],10,10,3))
        a = a[:,ind,:,:2]
        b = copy.deepcopy(a)
        for i in range(b.shape[0]):
            b[i] = b[i, np.random.permutation(10)]
        return np.reshape(b, (b.shape[0],1,-1)), a[:,0]



class ReCon():
    def __init__(self, wpath=None, restore_model_path=None):
        self.alpha = 0.35
        self.wpath = wpath
        if self.wpath is None:
            self.wpath = '/Users/kj/Rat-Kanishk-2019-12-05/dlc-models/iteration-0/RatDec5-trainset95shuffle1/train' \
                         '/snapshot-1030000'
        if restore_model_path is None:
            self.model = self._makeModel()
        else:
            self.model = self._make_saved_model(restore_model_path)



    def _make_saved_model(self, restore_model_path):
        mgpumodel = k.models.load_model(restore_model_path)
        return [mgpumodel]

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
        k2 = [i for i in list(weights_DLC.keys()) if 'MobilenetV2' in i]
        base_model = k.applications.mobilenet_v2.MobileNetV2(include_top=False, alpha=self.alpha)
        assert len(k2) == len(base_model.weights)
        if 'moving' not in ''.join([i.name for i in base_model.weights[:10]]):
            k2 = [i for i in k2 if 'moving' not in i] + [i for i in k2 if 'moving' in i]
            print('DLC weights realigned.')
        mismatch = False
        for i in range(len(base_model.weights)):
            if base_model.weights[i].shape != weights_DLC[k2[i]].shape:
                print(i, base_model.weights[i], k2[i])
                mismatch = True
                break
            else:
                pass
        if not mismatch:
            print('No mismatch found...')
            base_model.set_weights([weights_DLC[i] for i in k2])
        else:
            print('Weights not restored to keras model...')
        return base_model

    def _makeModel(self):
        dlcmnet = self._loadDLCModel()
        dlcmnet.trainable = False
        part_pred = k.layers.Conv2DTranspose(10, kernel_size=(3, 3), strides=(2, 2))(dlcmnet.output)
        locref = k.layers.Conv2DTranspose(20, kernel_size=(3, 3), strides=(2, 2))(dlcmnet.output)
        part_pred_sig = k.layers.Activation('sigmoid')#k.activations.sigmoid)

        locref = tf.squeeze()
        base_model = k.Model(inputs=dlcmnet.input, outputs=[part_pred, locref])
        base_model.trainable = False

        with tf.device('/device:CPU:0'):
            in_tensor = k.layers.Input(shape=(None, None, None, 3))
            td = k.layers.TimeDistributed(base_model)(in_tensor)
            if tf.test.is_gpu_available():
                y = tf.compat.v1.keras.layers.CuDNNLSTM(128, return_sequences=True)(td)
                y.trainable=True
                y = tf.compat.v1.keras.layers.CuDNNLSTM(64)(y)
                y.trainable=True
            else:
                y = k.layers.LSTM(128, return_sequences=True)(td)
                y = k.layers.LSTM(64)(y)
            d = k.layers.Dense(20)(y)
            d.trainable=True
            reconnet = k.Model(inputs=in_tensor, outputs=d)
        if tf.test.is_gpu_available():
            parallel_model = k.utils.multi_gpu_model(reconnet, gpus=3, cpu_merge=False)
        else:
            parallel_model = reconnet
        assert np.array_equal(dlcmnet.get_weights()[0], parallel_model.get_weights()[0])
        parallel_model.compile(loss='mse', optimizer='adam')
        return parallel_model, in_tensor, td, y, d

    def _testnet(self):
        reconnet = self._makeModel()[0]

        img_sequence = np.random.rand(32, 10, 128, 128, 3)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            out = sess.run((reconnet.output), feed_dict={reconnet.input: img_sequence})
        print(out.shape)
        return

from moviepy.editor import VideoFileClip
import cv2
import time
import os
import shutil
from skimage.util import img_as_ubyte

class VideoLoader(k.utils.Sequence):
    def __init__(self, vpath=None, h5path=None, batchsize=32, lag=10, nsamples=1000, tinds=None, ):
        self.vpath = '/Users/kj/Rat-Kanishk-2019-12-05/videos/01_118_clip2.mp4' if vpath is None else vpath
        self.h5path = '/Users/kj/Rat-Kanishk-2019-12-05/videos/01_118_clip2DLC_mobnet_100_RatDec5shuffle1_1030000.h5' \
            if h5path is None else h5path
        self.clip = VideoFileClip(self.vpath)
        self.h5 = pd.read_hdf(self.h5path, 'df_with_missing')

        self.lag = lag
        self.batchsize = batchsize
        self.nsamples = nsamples if nsamples < self.h5.shape[0] - lag else self.h5.shape[0] - lag - 10

        self.tinds = np.random.randint(low=self.lag, high=self.h5.shape[0], size=self.nsamples) if tinds is None \
            else np.array(tinds)
        self.iinds = np.arange(self.nsamples)
        assert len(self.tinds.shape) == 1

        t1 = time.time()
        self.images = np.zeros((self.nsamples, self.lag, 800, 800, 3), dtype='uint8')
        for i, ind1 in enumerate(self.tinds):
            for j, ind2 in enumerate(range(ind1 - self.lag, ind1)):
                self.images[i, j] = cv2.cvtColor(cv2.imread(self.vpath.split('.')[0] + '/frame%05d.png' % (ind2 + 1)),
                                                 cv2.COLOR_BGR2RGB)
        print('Time to load train images : %0.2fs' % (time.time() - t1))
        self.on_epoch_end()

    def _test_plot_training_set(self, test_dir='/home/kanishk/dataloader_test'):
        index = np.random.randint(0,high=int(self.nsamples/self.batchsize))
        indexes = self.iinds[index * self.batchsize:(index + 1) * self.batchsize]

        X = self.images[indexes]# - [123.68, 116.779, 103.939]
        y = self.h5.values[self.tinds[indexes]][:, (np.arange(self.h5.shape[1]) + 1) % 3 != 0]

        for i in range(self.batchsize):
            print(i)
            curdir = test_dir + '/batch_%.5i' % (i)
            try:
                os.mkdir(curdir)
            except FileExistsError:
                shutil.rmtree(curdir)
                os.mkdir(curdir)
            fig, ax = plt.subplots()

            fig.subplots_adjust(0,0,1,1,0,0)

            for j in range(self.lag-1):
                ax.cla()
                ax.imshow(X[i,j])
                ax.axis('off')
                plt.savefig(curdir + '/frame_%0.4i.png' % j)

            ax.cla()
            ax.imshow(X[i, -1])
            ax.axis('off')
            ax.scatter(y[i, ::2], y[i, 1::2], color='r', s=3)

            plt.savefig(curdir + '/frame_%0.4i.png'%(j+1))
            plt.close()
        return



    def __len__(self):
        return int(self.nsamples / self.batchsize)

    def __getitem__(self, index):
        # 10 second load time for each batch
        t1 = time.time()
        indexes = self.iinds[index * self.batchsize:(index + 1) * self.batchsize]
        # X = np.zeros((self.batchsize, self.lag,)+self.clip.get_frame(0).shape).astype('float32')
        # y = np.zeros((self.batchsize, int(self.h5.shape[1]*2/3))).astype('float32')
        # for i,ind1 in enumerate(indexes):
        #     # for j,ind2 in enumerate(range(ind1-self.lag, ind1)):
        #         # X[i,j] = self.clip.get_frame(ind2/self.clip.fps) - [123.68, 116.779, 103.939]
        #     X[i] = self.images[ind1] - [123.68, 116.779, 103.939]
        #     y[i] = self.h5.values[self.tinds[ind1], (np.arange(self.h5.shape[1])+1)%3 != 0]
        X = self.images[indexes] - [123.68, 116.779, 103.939]
        y = self.h5.values[self.tinds[indexes]][:, (np.arange(self.h5.shape[1]) + 1) % 3 != 0]
        # print('Time to load batch : %0.2fs'%(time.time()-t1))
        return X, y

    def on_epoch_end(self):
        np.random.shuffle(self.iinds)
