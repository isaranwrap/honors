import matplotlib
matplotlib.use('Agg')
import keras as k
import matplotlib.pyplot as plt
import tqdm
from keras_tqdm import TQDMCallback
from . import utils
import numpy as np

class PlotLosses(k.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('Epochs')
        self.ax.set_ylabel('RMSE Loss (px)')
        self.logs = []

        # plt.ion()
        # plt.show()

    def on_epoch_end(self, epoch, logs={}):

        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(np.sqrt(logs.get('loss')))
        self.val_losses.append(np.sqrt(logs.get('val_loss')))
        self.i += 1

        try:
            self.txtvar.remove()
            self.txtvar2.remove()
        except:
            pass
        self.ax.cla()
        self.ax.semilogy(self.x, self.losses, label="loss")
        self.ax.semilogy(self.x, self.val_losses, label="val_loss")
        self.ax.legend()
        self.ax.set_xlabel('Epochs')
        self.ax.set_ylabel('RMSE Loss (px)')
        self.txtvar = self.ax.text(self.x[np.argmin(self.val_losses)]+5, np.min(self.val_losses),
                                   '%i, %0.3f'%(self.x[np.argmin(self.val_losses)],np.min(self.val_losses)))
        self.txtvar2 = self.ax.text(self.x[-1], self.val_losses[-1], '%0.3f' % (self.val_losses[-1]))
        self.fig.savefig(self.task+'losscurve.png')
        # self.fig.canvas.draw()

class h5sPredictGenerator(k.utils.Sequence):

    def __init__(self, h5s, batch_size=None, egocentered=True, ratbodyonly=False, bindcenter=8, align=True, b1=2, b2=11):

        if batch_size is None or batch_size < len(h5s):
            raise ValueError('Please give an integer batch size which is greater than len(h5s)!!!')
        self.batch_size = batch_size
        self.t = 0
        self.h5s = h5s
        self.egocentered =egocentered
        self.ratbodyonly, self.bindcenter, self.align, self.b1, self.b2 = ratbodyonly, bindcenter, align, b1, b2

    def __len__(self):

        return np.max([len(h5) for h5 in self.h5s])

    def __getitem__(self, index):

        # Generate indexes of the batch

        ginds = np.setdiff1d(np.arange(self.h5s[0].shape[1]), self.bindcenter)
        nbparts = 13 if self.ratbodyonly else self.h5s[0].shape[1]
        X = np.zeros((self.batch_size, 1, nbparts*2))
        for i in range(len(self.h5s)):
            if self.t >= self.h5s[i].shape[0]:
                continue
            if not self.egocentered:
                h = self.h5s[i][self.t]
                h = h[:,:2] - h[[self.bindcenter for j in range(self.h5s[0].shape[1])],:2]
                h = h[ginds]
                if not self.align:
                    X[i, 0] = h[:nbparts].flatten()
                else:
                    dir = h[self.b1] - h[self.b2-1]
                    dir = dir/np.linalg.norm(dir)
                    rot_mat = np.array([[dir[0], dir[1]], [-dir[1], dir[0]]])
                    h = np.array(np.dot(h, rot_mat.T))
                    X[i,0] = h[:nbparts].flatten()
            else:
                X[i,0] = self.h5s[i][self.t].flatten()
        self.t += 1
        return X

        
class LSTM1Dense1():
    def __init__(self, trainX=None, trainY=None, valX=None, valY=None, mpath=None, gputouse=None, task=None,
                 batch_size=None, num_units=None):

        self.gputouse = gputouse
        if isinstance(gputouse, int):
            utils.setGPU(gputouse)
        if mpath is None:
            assert trainX.shape[1:] == valX.shape[1:]
            assert trainY.shape[1:] == valY.shape[1:]
            assert trainX.shape[0] == trainY.shape[0]
            assert valX.shape[0] == valY.shape[0]

            self.batch_size = 32 if batch_size is None else batch_size
            self.num_units = 64 if num_units is None else num_units

            self.trainX, self.trainY, self.valX, self.valY = trainX, trainY, valX, valY
            self.model = self._model(trainX, trainY)
        else:
            self.model = k.models.load_model(mpath)
            self.batch_size = self.model.input_shape[0]
            self.num_units = self.model.layers[0].units
        self.task = 'rat_1layer_%iUnits_%iBatchSize_Stateful' % (self.num_units, self.batch_size) if task is None \
            else task + '_1layer_%iUnits_%iBatchsize_Stateful' % (self.num_units, self.batch_size)
        self.histories = []

    def export_hidden_states(self, h5s, ratbodyonly=False):
        # assert isinstance(h5s, list)
        try:
            self.model2
        except:
            inp = k.layers.Input(batch_shape=(len(h5s),)+self.model.input_shape[1:])
            _, state_h, state_c = k.layers.CuDNNLSTM(self.num_units, return_state=True, return_sequences=True,
                                                         stateful=True)(inp)
            self.model2 = k.models.Model(inputs=inp, outputs=[state_h, state_c])
            self.model2.compile(loss='mse', optimizer='adam')
            self.model2.set_weights(self.model.get_weights()[:3])

        pred_gen = h5sPredictGenerator(h5s, batch_size=len(h5s), ratbodyonly=ratbodyonly, egocentered=True)

        h,c = self.model2.predict_generator(pred_gen, workers=1, verbose=1)

        return np.array([h[i::len(h5s)][:len(h5s[i])] for i in range(len(h5s))]),\
               np.array([c[i::len(h5s)][:len(h5s[i])] for i in range(len(h5s))])

    def export_predictions_old(self, h5, ratbodyonly=False):
        try:
            self.model3
        except:
            inp = k.layers.Input(batch_shape=(1,)+self.model.input_shape[1:])
            lstm1, state_h, state_c = k.layers.CuDNNLSTM(self.num_units, return_state=True, return_sequences=True,
                                                         stateful=True)(inp)
            den = k.layers.Dense(self.model.output_shape[-1])(lstm1)
            self.model3 = k.models.Model(inputs=inp, outputs=den)
            self.model3.compile(loss='mse', optimizer='adam')
            self.model3.set_weights(self.model.get_weights())
        if ratbodyonly:
            eh5 = self._eh5(h5)
            outh5 = self.model3.predict(eh5,verbose=1,batch_size=1)
        else:
            outh5 = self.model3.predict(h5, verbose=1, batch_size=1)
        return outh5

    def export_predictions(self, h5s, ratbodyonly=False):
        assert isinstance(h5s, list)

        pred_gen = h5sPredictGenerator(h5s, batch_size=self.batch_size, ratbodyonly=ratbodyonly)

        outh5 = self.model.predict_generator(pred_gen,verbose=1,workers=1)

        return np.array([outh5[i::self.batch_size][:len(h5s[i])] for i in range(len(h5s))])

    def _eh5(self, h5):
        eh5 = utils.egoh5(h5)[:,:13,:]
        eh5 = np.reshape(eh5, (eh5.shape[0], 1, eh5.shape[1]*2))
        return eh5

    def _model(self, trainX, trainY):
        model = k.models.Sequential()
        if self.gputouse is not None:
            model.add(k.layers.CuDNNLSTM(self.num_units, batch_input_shape=(self.batch_size , None, trainX.shape[-1]),
                                         return_sequences=True, stateful=True))
        else:
            model.add(k.layers.LSTM(64, batch_input_shape=(self.batch_size, None, trainX.shape[-1]),
                                    return_sequences=True, stateful=True))
        model.add(k.layers.Dense(trainY.shape[-1]))
        model.compile(loss='mse', optimizer='adam')
        return model
    
    def train(self, epochs=500, trainX=None, trainY=None, valX=None, valY = None):
        try:
            tqdm._instances.clear()
        except:
            pass
        if trainX is None or trainY is None or valX is None or valY is None:
            trainX, trainY, valX, valY = self.trainX, self.trainY, self.valX, self.valY
        plot_losses = PlotLosses()
        plot_losses.task = self.task
        history = self.model.fit(trainX, trainY, epochs=epochs, batch_size=self.batch_size ,
                    validation_data=(valX, valY), shuffle=False, verbose=0,
                    callbacks=[TQDMCallback(leave_inner=False, show_inner=False), 
                    plot_losses,
                    k.callbacks.ModelCheckpoint(self.task+'_Model_0.hdf5',monitor='val_loss', verbose=0, save_best_only=True)])
        self.histories.append(history)
        return history

    def continue_training(self, epochs=100, last_epoch = None, trainX=None, trainY=None, valX=None, valY = None):
        try:
            tqdm._instances.clear()
        except:
            pass

        last_epoch = self.histories[-1].history['epoch'][-1] if last_epoch is None else last_epoch

        if trainX is None or trainY is None or valX is None or valY is None:
            trainX, trainY, valX, valY = self.trainX, self.trainY, self.valX, self.valY

        plot_losses = PlotLosses()
        plot_losses.task = self.task+str(last_epoch)
        history = self.model.fit(trainX, trainY, epochs=last_epoch+epochs, batch_size=self.batch_size ,
        validation_data=(valX, valY), shuffle=False, verbose=0,
        callbacks=[TQDMCallback(leave_inner=False, show_inner=False),
        plot_losses,
        k.callbacks.ModelCheckpoint(self.task+'_Model_%i.hdf5'%last_epoch,monitor='val_loss', verbose=0,
                                    save_best_only=True)],
        initial_epoch=last_epoch)
        self.histories.append(history)
        return history


class LSTM1Dense1_dropout():
    def __init__(self, trainX=None, trainY=None, valX=None, valY=None, mpath=None, gputouse=None, task=None,
                 batch_size=None, num_units=None, start_weights=None):

        self.gputouse = gputouse
        if isinstance(gputouse, int):
            utils.setGPU(gputouse)
        if mpath is None:
            assert trainX.shape[1:] == valX.shape[1:]
            assert trainY.shape[1:] == valY.shape[1:]
            assert trainX.shape[0] == trainY.shape[0]
            assert valX.shape[0] == valY.shape[0]

            self.batch_size = 32 if batch_size is None else batch_size
            self.num_units = 64 if num_units is None else num_units
            self.start_weights = None if start_weights is None else start_weights
            self.trainX, self.trainY, self.valX, self.valY = trainX, trainY, valX, valY
            self.model = self._model(trainX, trainY)
        else:
            self.model = k.models.load_model(mpath)
            self.batch_size = self.model.input_shape[0]
            self.num_units = self.model.layers[0].units
        self.task = 'rat_1layer_%iUnits_%iBatchSize_Stateful' % (self.num_units, self.batch_size) if task is None \
            else task + '_1layer_%iUnits_%iBatchsize_Stateful' % (self.num_units, self.batch_size)
        self.histories = []

    def export_hidden_states(self, h5s, ratbodyonly=False):
        assert isinstance(h5s, list)
        try:
            self.model2
        except:
            inp = k.layers.Input(batch_shape=(len(h5s),) + self.model.input_shape[1:])
            _, state_h, state_c = k.layers.CuDNNLSTM(self.num_units, return_state=True, return_sequences=True,
                                                     stateful=True, dropout=0.1, recurrent_dropout=0.1)(inp)
            self.model2 = k.models.Model(inputs=inp, outputs=[state_h, state_c])
            self.model2.compile(loss='mse', optimizer='adam')
            self.model2.set_weights(self.model.get_weights()[:3])

        pred_gen = h5sPredictGenerator(h5s, batch_size=len(h5s), ratbodyonly=ratbodyonly)

        h, c = self.model2.predict_generator(pred_gen, workers=1, verbose=1)

        return np.array([h[i::len(h5s)][:len(h5s[i])] for i in range(len(h5s))]), \
               np.array([c[i::len(h5s)][:len(h5s[i])] for i in range(len(h5s))])

    def export_predictions_old(self, h5, ratbodyonly=False):
        try:
            self.model3
        except:
            inp = k.layers.Input(batch_shape=(1,) + self.model.input_shape[1:])
            lstm1, state_h, state_c = k.layers.CuDNNLSTM(self.num_units, return_state=True, return_sequences=True,
                                                         stateful=True)(inp)
            den = k.layers.Dense(self.model.output_shape[-1])(lstm1)
            self.model3 = k.models.Model(inputs=inp, outputs=den)
            self.model3.compile(loss='mse', optimizer='adam')
            self.model3.set_weights(self.model.get_weights())
        if ratbodyonly:
            eh5 = self._eh5(h5)
            outh5 = self.model3.predict(eh5, verbose=1, batch_size=1)
        else:
            outh5 = self.model3.predict(h5, verbose=1, batch_size=1)
        return outh5

    def export_predictions(self, h5s, ratbodyonly=False):
        assert isinstance(h5s, list)

        pred_gen = h5sPredictGenerator(h5s, batch_size=self.batch_size, ratbodyonly=ratbodyonly)

        outh5 = self.model.predict_generator(pred_gen, verbose=1, workers=1)

        return np.array([outh5[i::self.batch_size][:len(h5s[i])] for i in range(len(h5s))])

    def _eh5(self, h5):
        eh5 = utils.egoh5(h5)[:, :13, :]
        eh5 = np.reshape(eh5, (eh5.shape[0], 1, eh5.shape[1] * 2))
        return eh5

    def _model(self, trainX, trainY):
        model = k.models.Sequential()
        if self.gputouse is not None:
            model.add(k.layers.CuDNNLSTM(self.num_units, batch_input_shape=(self.batch_size, None, trainX.shape[-1]),
                                         return_sequences=True, stateful=True))
            model.add(k.layers.Dropout(0.1))
        else:
            model.add(k.layers.LSTM(64, batch_input_shape=(self.batch_size, None, trainX.shape[-1]),
                                    return_sequences=True, stateful=True))
            model.add(k.layers.Dropout(0.1))
        model.add(k.layers.Dense(trainY.shape[-1]))
        model.compile(loss='mse', optimizer='adam')
        model.load_weights(self.start_weights)
        return model

    def train(self, epochs=500, trainX=None, trainY=None, valX=None, valY=None):
        try:
            tqdm._instances.clear()
        except:
            pass
        if trainX is None or trainY is None or valX is None or valY is None:
            trainX, trainY, valX, valY = self.trainX, self.trainY, self.valX, self.valY
        plot_losses = PlotLosses()
        plot_losses.task = self.task
        history = self.model.fit(trainX, trainY, epochs=epochs, batch_size=self.batch_size,
                                 validation_data=(valX, valY), shuffle=False, verbose=0,
                                 callbacks=[TQDMCallback(leave_inner=False, show_inner=False),
                                            plot_losses,
                                            k.callbacks.ModelCheckpoint(self.task + '_Model_0.hdf5', monitor='val_loss',
                                                                        verbose=0, save_best_only=True)])
        self.histories.append(history)
        return history

    def continue_training(self, epochs=100, last_epoch=None, trainX=None, trainY=None, valX=None, valY=None):
        try:
            tqdm._instances.clear()
        except:
            pass

        last_epoch = self.histories[-1].history['epoch'][-1] if last_epoch is None else last_epoch

        if trainX is None or trainY is None or valX is None or valY is None:
            trainX, trainY, valX, valY = self.trainX, self.trainY, self.valX, self.valY

        plot_losses = PlotLosses()
        plot_losses.task = self.task + str(last_epoch)
        history = self.model.fit(trainX, trainY, epochs=last_epoch + epochs, batch_size=self.batch_size,
                                 validation_data=(valX, valY), shuffle=False, verbose=0,
                                 callbacks=[TQDMCallback(leave_inner=False, show_inner=False),
                                            plot_losses,
                                            k.callbacks.ModelCheckpoint(self.task + '_Model_%i.hdf5' % last_epoch,
                                                                        monitor='val_loss', verbose=0,
                                                                        save_best_only=True)],
                                 initial_epoch=last_epoch)
        self.histories.append(history)
        return history


class LSTM2Dense1():
    def __init__(self, trainX=None, trainY=None, valX=None, valY=None, mpath=None, gputouse=None, task=None,
                 batch_size=None, num_units=None):

        self.gputouse = gputouse
        if isinstance(gputouse, int):
            utils.setGPU(gputouse)
        if mpath is None:
            assert trainX.shape[1:] == valX.shape[1:]
            assert trainY.shape[1:] == valY.shape[1:]
            assert trainX.shape[0] == trainY.shape[0]
            assert valX.shape[0] == valY.shape[0]

            self.batch_size = 32 if batch_size is None else batch_size
            self.num_units = 64 if num_units is None else num_units

            self.trainX, self.trainY, self.valX, self.valY = trainX, trainY, valX, valY
            self.model = self._model(trainX, trainY)
        else:
            self.model = k.models.load_model(mpath)
            self.batch_size = self.model.input_shape[0]
            self.num_units = self.model.layers[0].units
        self.task = 'rat_2layer_%iUnits_%iBatchSize_Stateful' % (self.num_units, self.batch_size) if task is None \
            else task + '_2layer_%iUnits_%iBatchsize_Stateful' % (self.num_units, self.batch_size)
        self.histories = []

    def export_hidden_states(self, h5s, ratbodyonly=False):
        # assert isinstance(h5s, list)
        try:
            self.model2
        except:
            inp = k.layers.Input(batch_shape=(len(h5s),)+self.model.input_shape[1:])
            lstm1, state_h_1, state_c_1 = k.layers.CuDNNLSTM(self.num_units, return_state=True, return_sequences=True,
                                                             stateful=True)(inp)
            lstm2, state_h_2, state_c_2 = k.layers.CuDNNLSTM(self.num_units, return_state=True, return_sequences=True,
                                                             stateful=True)(lstm1)
            self.model2 = k.models.Model(inputs=inp, outputs=[state_h_1, state_c_1, state_h_2, state_c_2])
            self.model2.compile(loss='mse', optimizer='adam')
            self.model2.set_weights(self.model.get_weights()[:6])
        pred_gen = h5sPredictGenerator(h5s, batch_size=len(h5s), ratbodyonly=ratbodyonly, egocentered=True)

        h1, c1, h2, c2 = self.model2.predict_generator(pred_gen, verbose=1, workers=1)

        return np.array([h1[i::len(h5s)][:len(h5s[i])] for i in range(len(h5s))]),\
               np.array([c1[i::len(h5s)][:len(h5s[i])] for i in range(len(h5s))]), \
               np.array([h2[i::len(h5s)][:len(h5s[i])] for i in range(len(h5s))]), \
               np.array([c2[i::len(h5s)][:len(h5s[i])] for i in range(len(h5s))])

    def _eh5(self, h5):
        eh5 = utils.egoh5(h5)[:, :13, :]
        eh5 = np.reshape(eh5, (eh5.shape[0], 1, eh5.shape[1] * 2))
        return eh5

    def export_predictions(self, h5s, ratbodyonly=False):
        assert isinstance(h5s, list)

        pred_gen = h5sPredictGenerator(h5s, batch_size=self.batch_size, ratbodyonly=ratbodyonly)

        outh5 = self.model.predict_generator(pred_gen,verbose=1,workers=1)

        return np.array([outh5[i::self.batch_size][:len(h5s[i])] for i in range(len(h5s))])

    def _model(self, trainX, trainY):
        model = k.models.Sequential()
        if self.gputouse is not None:
            model.add(k.layers.CuDNNLSTM(self.num_units, batch_input_shape=(self.batch_size, None, trainX.shape[-1]),
                                         return_sequences=True, stateful=True))
            model.add(k.layers.CuDNNLSTM(self.num_units, return_sequences=True, stateful=True))
        else:
            model.add(k.layers.LSTM(self.num_units, batch_input_shape=(self.batch_size, None, trainX.shape[-1]),
                                    return_sequences=True, stateful=True))
            model.add(k.layers.LSTM(self.num_units, return_sequences=True, stateful=True))
        model.add(k.layers.Dense(trainY.shape[-1]))
        model.compile(loss='mse', optimizer='adam')
        return model

    def train(self, epochs=500, trainX=None, trainY=None, valX=None, valY=None):
        try:
            tqdm._instances.clear()
        except:
            pass
        if trainX is None or trainY is None or valX is None or valY is None:
            trainX, trainY, valX, valY = self.trainX, self.trainY, self.valX, self.valY
        plot_losses = PlotLosses()
        plot_losses.task = self.task
        history = self.model.fit(trainX, trainY, epochs=epochs, batch_size=self.batch_size,
                validation_data=(valX, valY), shuffle=False,
                                 verbose=0, callbacks=[TQDMCallback(leave_inner=False, show_inner=False),
                plot_losses,
                k.callbacks.ModelCheckpoint(self.task + '_Model_0.hdf5',monitor='val_loss', verbose=0, save_best_only=True)])
        self.histories.append(history)
        return history

    def continue_training(self, epochs=100, last_epoch = None, trainX=None, trainY=None, valX=None, valY = None):
        try:
            tqdm._instances.clear()
        except:
            pass

        last_epoch = self.histories[-1].history['epoch'][-1] if last_epoch is None else last_epoch

        if trainX is None or trainY is None or valX is None or valY is None:
            trainX, trainY, valX, valY = self.trainX, self.trainY, self.valX, self.valY

        plot_losses = PlotLosses()
        plot_losses.task = self.task+str(last_epoch)
        history = self.model.fit(trainX, trainY, epochs=last_epoch+epochs, batch_size=self.batch_size ,
        validation_data=(valX, valY), shuffle=False, verbose=0,
        callbacks=[TQDMCallback(leave_inner=False, show_inner=False),
        plot_losses,
        k.callbacks.ModelCheckpoint(self.task+'_Model_%i.hdf5'%last_epoch,monitor='val_loss', verbose=0,
                                    save_best_only=True)],
        initial_epoch=last_epoch)
        self.histories.append(history)
        return history


class LSTM3Dense1():
    def __init__(self, trainX=None, trainY=None, valX=None, valY=None, mpath=None, gputouse=None, task=None,
                 batch_size=None, num_units=None):

        self.gputouse = gputouse
        if isinstance(gputouse, int):
            utils.setGPU(gputouse)
        if mpath is None:
            assert trainX.shape[1:] == valX.shape[1:]
            assert trainY.shape[1:] == valY.shape[1:]
            assert trainX.shape[0] == trainY.shape[0]
            assert valX.shape[0] == valY.shape[0]

            self.batch_size = 32 if batch_size is None else batch_size
            self.num_units = 64 if num_units is None else num_units

            self.trainX, self.trainY, self.valX, self.valY = trainX, trainY, valX, valY
            self.model = self._model(trainX, trainY)
        else:
            self.model = k.models.load_model(mpath)
            self.batch_size = self.model.input_shape[0]
            self.num_units = self.model.layers[0].units
        self.task = 'rat_3layer_%iUnits_%iBatchSize_Stateful' % (self.num_units, self.batch_size) if task is None \
            else task + '_3layer_%iUnits_%iBatchsize_Stateful' % (self.num_units, self.batch_size)
        self.histories = []

    def export_hidden_states(self, h5s, ratbodyonly=False):
        # assert isinstance(h5s, list)

        try:
            self.model2
        except:
            inp = k.layers.Input(batch_shape=(len(h5s),)+self.model.input_shape[1:])
            lstm1, state_h_1, state_c_1 = k.layers.CuDNNLSTM(self.num_units, return_state=True, return_sequences=True,
                                                             stateful=True)(inp)
            lstm2, state_h_2, state_c_2 = k.layers.CuDNNLSTM(self.num_units, return_state=True, return_sequences=True,
                                                             stateful=True)(lstm1)
            lstm3, state_h_3, state_c_3 = k.layers.CuDNNLSTM(self.num_units, return_state=True, return_sequences=True,
                                                             stateful=True)(lstm2)
            self.model2 = k.models.Model(inputs=inp, outputs=[state_h_1, state_c_1, state_h_2, state_c_2, state_h_3,
                                                              state_c_3])
            self.model2.compile(loss='mse', optimizer='adam')
            self.model2.set_weights(self.model.get_weights()[:9])
        pred_gen = h5sPredictGenerator(h5s, batch_size=len(h5s), ratbodyonly=ratbodyonly)

        h1, c1, h2, c2, h3, c3 = self.model2.predict_generator(pred_gen, verbose=1, workers=1)

        return np.array([h1[i::len(h5s)][:len(h5s[i])] for i in range(len(h5s))]), \
               np.array([c1[i::len(h5s)][:len(h5s[i])] for i in range(len(h5s))]), \
               np.array([h2[i::len(h5s)][:len(h5s[i])] for i in range(len(h5s))]), \
               np.array([c2[i::len(h5s)][:len(h5s[i])] for i in range(len(h5s))]),\
               np.array([h3[i::len(h5s)][:len(h5s[i])] for i in range(len(h5s))]),\
               np.array([c3[i::len(h5s)][:len(h5s[i])] for i in range(len(h5s))])


    def _eh5(self, h5):
        eh5 = utils.egoh5(h5)[:, :13, :]
        eh5 = np.reshape(eh5, (eh5.shape[0], 1, eh5.shape[1] * 2))
        return eh5

    def export_predictions(self, h5s, ratbodyonly=False):
        assert isinstance(h5s, list)

        pred_gen = h5sPredictGenerator(h5s, batch_size=self.batch_size, ratbodyonly=ratbodyonly)

        outh5 = self.model.predict_generator(pred_gen,verbose=1,workers=1)

        return np.array([outh5[i::self.batch_size][:len(h5s[i])] for i in range(len(h5s))])


    def _model(self, trainX, trainY):
        model = k.models.Sequential()
        if self.gputouse is not None:
            model.add(k.layers.CuDNNLSTM(self.num_units, batch_input_shape=(self.batch_size, None, trainX.shape[-1]),
                                         return_sequences=True, stateful=True))
            model.add(k.layers.CuDNNLSTM(self.num_units, return_sequences=True, stateful=True))
            model.add(k.layers.CuDNNLSTM(self.num_units, return_sequences=True, stateful=True))

        else:
            model.add(k.layers.LSTM(self.num_units, batch_input_shape=(self.batch_size, None, trainX.shape[-1]),
                                    return_sequences=True, stateful=True))
            model.add(k.layers.LSTM(self.num_units, return_sequences=True, stateful=True))
            model.add(k.layers.LSTM(self.num_units, return_sequences=True, stateful=True))
        model.add(k.layers.Dense(trainY.shape[-1]))
        model.compile(loss='mse', optimizer='adam')
        return model

    def train(self, epochs=500, trainX=None, trainY=None, valX=None, valY=None):
        try:
            tqdm._instances.clear()
        except:
            pass
        if trainX is None or trainY is None or valX is None or valY is None:
            trainX, trainY, valX, valY = self.trainX, self.trainY, self.valX, self.valY
        plot_losses = PlotLosses()
        plot_losses.task = self.task
        history = self.model.fit(trainX, trainY, epochs=epochs, batch_size=self.batch_size,
                validation_data=(valX, valY), shuffle=False,
                                 verbose=0, callbacks=[TQDMCallback(leave_inner=False, show_inner=False),
                plot_losses,
                k.callbacks.ModelCheckpoint(self.task + '_Model_0.hdf5',monitor='val_loss', verbose=0, save_best_only=True)])
        self.histories.append(history)
        return history

    def continue_training(self, epochs=100, last_epoch = None, trainX=None, trainY=None, valX=None, valY = None):
        try:
            tqdm._instances.clear()
        except:
            pass

        last_epoch = self.histories[-1].history['epoch'][-1] if last_epoch is None else last_epoch

        if trainX is None or trainY is None or valX is None or valY is None:
            trainX, trainY, valX, valY = self.trainX, self.trainY, self.valX, self.valY

        plot_losses = PlotLosses()
        plot_losses.task = self.task+str(last_epoch)
        history = self.model.fit(trainX, trainY, epochs=last_epoch+epochs, batch_size=self.batch_size ,
        validation_data=(valX, valY), shuffle=False, verbose=0,
        callbacks=[TQDMCallback(leave_inner=False, show_inner=False),
        plot_losses,
        k.callbacks.ModelCheckpoint(self.task+'_Model_%i.hdf5'%last_epoch,monitor='val_loss', verbose=0,
                                    save_best_only=True)],
        initial_epoch=last_epoch)
        self.histories.append(history)
        return history


class LSTM5Dense1():
    def __init__(self, trainX=None, trainY=None, valX=None, valY=None, mpath=None, gputouse=None, task=None,
                 batch_size=None, num_units=None):

        self.gputouse = gputouse
        if isinstance(gputouse, int):
            utils.setGPU(gputouse)
        if mpath is None:
            assert trainX.shape[1:] == valX.shape[1:]
            assert trainY.shape[1:] == valY.shape[1:]
            assert trainX.shape[0] == trainY.shape[0]
            assert valX.shape[0] == valY.shape[0]

            self.batch_size = 32 if batch_size is None else batch_size
            self.num_units = 64 if num_units is None else num_units

            self.trainX, self.trainY, self.valX, self.valY = trainX, trainY, valX, valY
            self.model = self._model(trainX, trainY)
        else:
            self.model = k.models.load_model(mpath)
            self.batch_size = self.model.input_shape[0]
            self.num_units = self.model.layers[0].units
        self.task = 'rat_5layer_%iUnits_%iBatchSize_Stateful' % (self.num_units, self.batch_size) if task is None \
            else task + '_5layer_%iUnits_%iBatchsize_Stateful' % (self.num_units, self.batch_size)
        self.histories = []

    def export_hidden_states(self, h5s, ratbodyonly=False):
        # assert isinstance(h5s, list)

        try:
            self.model2
        except:
            inp = k.layers.Input(batch_shape=(len(h5s),)+self.model.input_shape[1:])
            lstm1, state_h_1, state_c_1 = k.layers.CuDNNLSTM(self.num_units, return_state=True, return_sequences=True,
                                                             stateful=True)(inp)
            lstm2, state_h_2, state_c_2 = k.layers.CuDNNLSTM(self.num_units, return_state=True, return_sequences=True,
                                                             stateful=True)(lstm1)
            lstm3, state_h_3, state_c_3 = k.layers.CuDNNLSTM(self.num_units, return_state=True, return_sequences=True,
                                                             stateful=True)(lstm2)
            lstm4, state_h_4, state_c_4 = k.layers.CuDNNLSTM(self.num_units, return_state=True, return_sequences=True,
                                                             stateful=True)(lstm3)
            lstm5, state_h_5, state_c_5 = k.layers.CuDNNLSTM(self.num_units, return_state=True, return_sequences=True,
                                                             stateful=True)(lstm4)
            self.model2 = k.models.Model(inputs=inp, outputs=[state_h_1, state_c_1, state_h_2, state_c_2, state_h_3,
                                                              state_c_3, state_h_4, state_c_4, state_h_5, state_c_5])
            self.model2.compile(loss='mse', optimizer='adam')
            self.model2.set_weights(self.model.get_weights()[:9])
        pred_gen = h5sPredictGenerator(h5s, batch_size=len(h5s), ratbodyonly=ratbodyonly)

        h1, c1, h2, c2, h3, c3, h4, c4, h5, c5 = self.model2.predict_generator(pred_gen, verbose=1, workers=1)

        return np.array([h1[i::len(h5s)][:len(h5s[i])] for i in range(len(h5s))]), \
               np.array([c1[i::len(h5s)][:len(h5s[i])] for i in range(len(h5s))]), \
               np.array([h2[i::len(h5s)][:len(h5s[i])] for i in range(len(h5s))]), \
               np.array([c2[i::len(h5s)][:len(h5s[i])] for i in range(len(h5s))]),\
               np.array([h3[i::len(h5s)][:len(h5s[i])] for i in range(len(h5s))]),\
               np.array([c3[i::len(h5s)][:len(h5s[i])] for i in range(len(h5s))]),\
               np.array([h4[i::len(h5s)][:len(h5s[i])] for i in range(len(h5s))]),\
               np.array([c4[i::len(h5s)][:len(h5s[i])] for i in range(len(h5s))]),\
               np.array([h5[i::len(h5s)][:len(h5s[i])] for i in range(len(h5s))]),\
               np.array([c5[i::len(h5s)][:len(h5s[i])] for i in range(len(h5s))])


    def _eh5(self, h5):
        eh5 = utils.egoh5(h5)[:, :13, :]
        eh5 = np.reshape(eh5, (eh5.shape[0], 1, eh5.shape[1] * 2))
        return eh5

    def export_predictions(self, h5s, ratbodyonly=False):
        assert isinstance(h5s, list)

        pred_gen = h5sPredictGenerator(h5s, batch_size=self.batch_size, ratbodyonly=ratbodyonly)

        outh5 = self.model.predict_generator(pred_gen,verbose=1,workers=1)

        return np.array([outh5[i::self.batch_size][:len(h5s[i])] for i in range(len(h5s))])


    def _model(self, trainX, trainY):
        model = k.models.Sequential()
        if self.gputouse is not None:
            model.add(k.layers.CuDNNLSTM(self.num_units, batch_input_shape=(self.batch_size, None, trainX.shape[-1]),
                                         return_sequences=True, stateful=True))
            model.add(k.layers.CuDNNLSTM(self.num_units, return_sequences=True, stateful=True))
            model.add(k.layers.CuDNNLSTM(self.num_units, return_sequences=True, stateful=True))
            model.add(k.layers.CuDNNLSTM(self.num_units, return_sequences=True, stateful=True))
            model.add(k.layers.CuDNNLSTM(self.num_units, return_sequences=True, stateful=True))

        else:
            model.add(k.layers.LSTM(self.num_units, batch_input_shape=(self.batch_size, None, trainX.shape[-1]),
                                    return_sequences=True, stateful=True))
            model.add(k.layers.LSTM(self.num_units, return_sequences=True, stateful=True))
            model.add(k.layers.LSTM(self.num_units, return_sequences=True, stateful=True))
            model.add(k.layers.LSTM(self.num_units, return_sequences=True, stateful=True))
            model.add(k.layers.LSTM(self.num_units, return_sequences=True, stateful=True))
        model.add(k.layers.Dense(trainY.shape[-1]))
        model.compile(loss='mse', optimizer='adam')
        return model

    def train(self, epochs=500, trainX=None, trainY=None, valX=None, valY=None):
        try:
            tqdm._instances.clear()
        except:
            pass
        if trainX is None or trainY is None or valX is None or valY is None:
            trainX, trainY, valX, valY = self.trainX, self.trainY, self.valX, self.valY
        plot_losses = PlotLosses()
        plot_losses.task = self.task
        history = self.model.fit(trainX, trainY, epochs=epochs, batch_size=self.batch_size,
                validation_data=(valX, valY), shuffle=False,
                                 verbose=0, callbacks=[TQDMCallback(leave_inner=False, show_inner=False),
                plot_losses,
                k.callbacks.ModelCheckpoint(self.task + '_Model_0.hdf5',monitor='val_loss', verbose=0, save_best_only=True)])
        self.histories.append(history)
        return history

    def continue_training(self, epochs=100, last_epoch = None, trainX=None, trainY=None, valX=None, valY = None):
        try:
            tqdm._instances.clear()
        except:
            pass

        last_epoch = self.histories[-1].history['epoch'][-1] if last_epoch is None else last_epoch

        if trainX is None or trainY is None or valX is None or valY is None:
            trainX, trainY, valX, valY = self.trainX, self.trainY, self.valX, self.valY

        plot_losses = PlotLosses()
        plot_losses.task = self.task+str(last_epoch)
        history = self.model.fit(trainX, trainY, epochs=last_epoch+epochs, batch_size=self.batch_size ,
        validation_data=(valX, valY), shuffle=False, verbose=0,
        callbacks=[TQDMCallback(leave_inner=False, show_inner=False),
        plot_losses,
        k.callbacks.ModelCheckpoint(self.task+'_Model_%i.hdf5'%last_epoch,monitor='val_loss', verbose=0,
                                    save_best_only=True)],
        initial_epoch=last_epoch)
        self.histories.append(history)
        return history
