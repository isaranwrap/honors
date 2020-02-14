import matplotlib
matplotlib.use('TkAgg')
import tensorflow as tf
import keras as k
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
from keras_tqdm import TQDMCallback
from . import utils


class PlotLosses(k.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig, self.ax = plt.subplots()
        
        self.logs = []
        plt.show()

    def on_epoch_end(self, epoch, logs={}):

        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        # clear_output(wait=True)
        self.ax.cla()
        self.ax.semilogy(self.x, self.losses, label="loss")
        self.ax.semilogy(self.x, self.val_losses, label="val_loss")
        self.ax.legend()
        self.ax.text(self.x[-1], self.val_losses[-1], str(self.val_losses))
        self.fig.canvas.draw()


        
class LSTM1Dense1():
    def __init__(self, trainX=None, trainY=None, valX=None, valY=None, mpath=None, gputouse=None):
        self.gputouse = gputouse
        if isinstance(gputouse, int):
            utils.setGPU(gputouse)
        if mpath is None:
            assert trainX.shape[1:] == valX.shape[1:]
            assert trainY.shape[1:] == valY.shape[1:]
            assert trainX.shape[0] == trainY.shape[0]
            assert valX.shape[0] == valY.shape[0]

            self.trainX, self.trainY, self.valX, self.valY = trainX, trainY, valX, valY
            self.model = self._model(trainX, trainY)
        else:
            pass
    def train(self):
        pass
    def test(self):
        pass
    def verifydata(self):
        pass
    def _model(self, trainX, trainY, num_units=64):
        model = k.models.Sequential()
        if self.gputouse is not None:
            model.add(k.layers.CuDNNLSTM(num_units, input_shape=(None, trainX.shape[-1])))
        else:
            model.add(k.layers.LSTM(64, input_shape=(None, trainX.shape[-1])))
        model.add(k.layers.Dense(trainY.shape[-1]))
        model.compile(loss='mse', optimizer='adam')
        return model
    
    def train(self, epochs=2500, batch_size=32, shuffle=True, trainX=None, trainY=None, valX=None, valY = None):
        try:
            tqdm._instances.clear()
        except:
            pass
        if trainX is None or trainY is None or valX is None or valY is None:
            trainX, trainY, valX, valY = self.trainX, self.trainY, self.valX, self.valY
        plot_losses = PlotLosses()
        history = self.model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size,
                    validation_data=(valX, valY), shuffle=shuffle, verbose=0,
                    callbacks=[TQDMCallback(leave_inner=False, show_inner=False), 
                              plot_losses])
        
        

