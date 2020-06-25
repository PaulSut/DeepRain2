from __future__ import print_function
from Utils.Data import Dataset, dataWrapper
import numpy as np
from Models.Unet import unet
from Models.tfModels import UNet64
from Utils.loss import SSIM
from tensorflow.keras.optimizers import *
# from keras import backend as K
import tensorflow as tf
import keras
import os
import json

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("Num GPUs Available:", len(
    tf.config.experimental.list_physical_devices('GPU')))
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)


class Trainer(object):
    """
        docstring for Trainer

        Wrapper for handling keras models.
        This Class will create a folder where it stores such things as weights
        , history and more (in the future)

        model       : function to model definition
        lossfunction: e.g. mse
        pathToData  : Full path to data (recursive paths are allowed too)
        channels    : number of input channels which are used for forecasting
        dimension   : image size (input = output dimension)
        metrics     : list of metrics e.g. ["accuracy","mae"]
        flatten     : useful for binary/cross entropy
        pathToModel : Folder where Models are stored
        load        : load model (if available)
        kwargs      : dictionary which will be passed to model definiton


        Functions:

        fit()
            >> trains the network
            The only argument is the epochs to train

    """

    def __init__(self,
                 model,
                 lossfunction,
                 pathToData,
                 batch_size=5,
                 channels=5,
                 channels_output = 1,
                 optimizer="adam",
                 dimension=(272, 224),
                 metrics=[],
                 flatten=False,
                 pathToModel="./model_data",
                 load=True,
                 kwargs={}):

        super(Trainer, self).__init__()
        self.nameOfModel = model.__name__
        self.pathToData = pathToData
        self.batch_size = batch_size
        self.channels = channels
        self.channels_output = channels_output
        self.dimension = dimension
        self.lossfunction = lossfunction
        self.flatten = flatten
        self.metrics = metrics
        self.load = load
        self.initialEpoch = 0
        self.history = None

        if type(self.pathToData) is tuple:
            self.train, self.test = self.pathToData

        else:
            self.train, self.test = dataWrapper(self.pathToData,
                                                dimension=dimension,
                                                channels=channels,
                                                channels_output = self.channels_output,
                                                batch_size=batch_size,
                                                flatten=flatten)

        if type(lossfunction) is str:
            self.nameOfModel += "_" + lossfunction

        else:
            self.nameOfModel += "_" + lossfunction.__class__.__name__

        self.pathToModel = os.path.join(pathToModel, self.nameOfModel)

        if len(kwargs) > 0:
            self.model = model((*dimension, channels), **kwargs)
        else:
            self.model = model((*dimension, channels))

        for i, d in enumerate((*dimension, channels)):
            self.nameOfModel += str(d)
            if i < len((*dimension, channels)) - 1:
                self.nameOfModel += "x"

        self.model.compile(loss=lossfunction, optimizer=optimizer, metrics=metrics)
        if self.load:
            try:
                filename = os.path.join(self.pathToModel, self.nameOfModel + ".h5")
                self.model.load_weights(os.path.join(filename))
                print("[Loaded file] ", filename)

            except Exception as e:
                print("[Load file failed] ", filename)

            try:
                historypath = os.path.join(self.pathToModel, self.nameOfModel + 'history.json')
                file = open(historypath, 'r')
                json_str = file.read()
                self.history = json.loads(json_str)
                self.initialEpoch = len(self.history["loss"])
                print(self.initialEpoch)
                print("[Loaded file] ", historypath)

            except Exception as e:
                print("[Load file failed] ", historypath)

        if not os.path.exists(pathToModel):
            os.mkdir(pathToModel)

        if not os.path.exists(self.pathToModel):
            os.mkdir(self.pathToModel)
        self.model.summary()

    def fit(self, epochs):
        filepath = os.path.join(self.pathToModel, self.nameOfModel+"-{epoch:02d}.hdf5")
        training_callbacks = [
            keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min'),
            #keras.callbacks.ReduceLROnPlateau( monitor='val_loss', factor=0.3, patience=2, verbose=1, mode='auto',
            #    cooldown=0, min_lr=0,)
            ]
        history = self.model.fit(self.train,
                                 epochs=self.initialEpoch + epochs,
                                 initial_epoch=self.initialEpoch,
                                 workers=0,
                                 use_multiprocessing=False,
                                 validation_data=self.test,
                                 verbose=1,
                                 shuffle=False,
                                 callbacks = training_callbacks
                                 )  # Shuffle needs to be False, cause of shuffle buffer
        if self.history is None:
            self.history = history.history
        else:
            for key in history.history:
                self.history[key] += history.history[key]

        for key in self.history:
            self.history[key] = list(np.array(self.history[key]).astype(float))

        with open(os.path.join(self.pathToModel, self.nameOfModel + 'history.json'), 'w') as f:
            json.dump(self.history, f)

        self.model.save_weights(os.path.join(self.pathToModel,self.nameOfModel+".h5"))
        #self.model.save_weights(os.path.join(self.pathToModel, self.nameOfModel + '_weights.h5'))
        #self.model.save(os.path.join(self.pathToModel, self.nameOfModel + '.tf'))
