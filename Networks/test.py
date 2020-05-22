#!/home/simon/anaconda3/envs/tensorflow-gpu/bin/python
from Models.tfModels import *
from trainer import Trainer
from Utils.loss import SSIM
from Models.Unet import unet
from tensorflow.keras.optimizers import *
from Models.CnnLSTM import *
from Utils.loadset import getDataSet
from Utils.Data import dataWrapper
from tensorflow.keras.optimizers import Adam
from Utils.transform import *
from keras.losses import *
import os
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

DatasetFolder = "./Data/RAW"
PathToData = os.path.join(DatasetFolder, "MonthPNGData")

# dimension = (256,192)
dimension = (64,64)
#dimension = (272, 224)
channels = 5
optimizer = Adam(lr=1e-3)


# pathToData = "/home/simon/MonthPNGData/MonthPNGData"


def provideData(flatten=False, dimension=dimension, batch_size=10, transform_input=None,
                transform_output=None, preTransformation=None):
    getDataSet(DatasetFolder, year=[2017])
    train, test = dataWrapper(PathToData,
                              dimension=dimension,
                              channels=channels,
                              batch_size=batch_size,
                              overwritecsv=True,
                              flatten=flatten,
                              onlyUseYears=[2017],
                              transform_input=transform_input,
                              transform_output=transform_output,
                              preTransformation=preTransformation)

    return train, test

    """
    t = Trainer(LSTM_Meets_Unet_Upconv,
                    SSIM(),
                    pathToData,
                    batch_size = 10,
                    optimizer=optimizer,
                    dimension = dimension,
                    channels = channels)
    
    t.fit( epochs = 3 )
    
    
    """


def small_uet():
    dimension = (64, 64,)
    dimension = (448, 448,)
    channels = 5
    optimizer = Adam(lr=5e-5)
    BATCH_SIZE = 1
    #optimizer = Adadelta()
    #optimizer = RMSprop(learning_rate=1e-3)

    slices = [256, 320, 256, 320]
    slices = [100, 548, 200, 648]
    cutOutFrame = cutOut(slices)

    PRETRAINING_TRANSFORMATIONS = [cutOutFrame]



    t = Trainer(UNet64,
                lossfunction='categorical_crossentropy',
                pathToData=provideData(batch_size=BATCH_SIZE, dimension=dimension,preTransformation=PRETRAINING_TRANSFORMATIONS, transform_input=[Normalize()],
                                       transform_output=[ToCategorical([-10000, 0, 2, 10, 256])]),
                batch_size=BATCH_SIZE,
                optimizer=optimizer,
                dimension=dimension,
                channels=channels,
                load=True,
                metrics=['categorical_crossentropy', 'mse', 'accuracy'],
                )

    t.fit(epochs=30)

def lstm_many_to_one_train():
    dimension = (64, 64,)
    #dimension = (448, 448,)
    channels = 5
    optimizer = Adam(lr=1e-3)
    BATCH_SIZE = 1
    # optimizer = Adadelta()
    # optimizer = RMSprop(learning_rate=1e-3)

    slices = [256, 320, 256, 320]
    #slices = [100, 548, 200, 648]
    cutOutFrame = cutOut(slices)

    PRETRAINING_TRANSFORMATIONS = [cutOutFrame]

    t = Trainer(medium_thin_UNet64,
                lossfunction='mse',
                pathToData=provideData(batch_size=BATCH_SIZE, dimension=dimension,
                                       preTransformation=PRETRAINING_TRANSFORMATIONS, transform_input=[Normalize()],
                                       transform_output=None),
                batch_size=BATCH_SIZE,
                optimizer=optimizer,
                dimension=dimension,
                channels=channels,
                load=False,
                metrics=['mse', 'accuracy'],
                )

    t.fit(epochs=1)


def medium_unet():
    dimension = (64, 64,)
    #dimension = (128, 128,)
    #dimension = (256, 256,)
    dimension = (448, 448,)
    channels = 5
    optimizer = Adam(lr=3e-3)
    BATCH_SIZE = 1
    #optimizer = Adadelta()
    #optimizer = RMSprop(learning_rate=1e-3)

    slices = [256, 320, 256, 320]
    #slices = [256, 384, 256, 384]
    #slices = [256, 512, 256, 512]
    slices = [100, 548, 200, 648]
    cutOutFrame = cutOut(slices)

    PRETRAINING_TRANSFORMATIONS = [cutOutFrame]



    t = Trainer(medium_thin_UNet64,
                lossfunction='categorical_crossentropy',
                pathToData=provideData(batch_size=BATCH_SIZE, dimension=dimension,preTransformation=PRETRAINING_TRANSFORMATIONS, transform_input=[Normalize()],
                                       transform_output=[ToCategorical([-10000, 0, 2, 10, 256])]),
                batch_size=BATCH_SIZE,
                optimizer=optimizer,
                dimension=dimension,
                channels=channels,
                load=True,
                metrics=['categorical_crossentropy', 'mse', 'accuracy'],
                )

    '''
    t = Trainer(medium_thin_UNet64,
            lossfunction='mse',
            pathToData=provideData(batch_size=BATCH_SIZE, dimension=dimension,
                                   preTransformation=PRETRAINING_TRANSFORMATIONS, transform_input=[Normalize()],
                                   transform_output=None),
            batch_size=BATCH_SIZE,
            optimizer=optimizer,
            dimension=dimension,
            channels=channels,
            load=True,
            metrics=['mse', 'accuracy'],
            )
            '''

    t.fit(epochs=5)



def NLL(y_true, y_hat):
    return -y_hat.log_prob(y_true)


negative_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)


def Bernoulli():
    from Utils.transform import Binarize

    t = Trainer(UNet64_Bernoulli,
                NLL,
                provideData(batch_size=50, transform=[Binarize(threshold=0)]),
                batch_size=50,
                optimizer=optimizer,
                dimension=dimension,
                channels=channels)

    t.fit(epochs=2)


def Poisson():
    dimension = (272, 224)
    dimension = (64, 64)
    from Utils.transform import Flatten
    t = Trainer(UNet64_Poisson,
                NLL,
                provideData(batch_size=200),
                batch_size=200,
                optimizer=optimizer,
                dimension=dimension,
                channels=channels,
                metrics=["mse", "accuracy"])

    t.fit(epochs=20)


# Bernoulli()
# Poisson()
def LSTM_POISSON():
    dimension = (128, 112)
    t = Trainer(LSTM_Meets_Unet_Poisson,
                NLL,
                provideData(batch_size=25, dimension=dimension),
                batch_size=25,
                optimizer=optimizer,
                dimension=dimension,
                channels=channels)

    t.fit(epochs=3)


def LSTM_Meets_Unet_MIXED():
    def NLL(y_true, y_hat):
        return -y_hat.log_prob(y_true[:, :])

    dimension = (64, 64)
    t = Trainer(LSTM_Meets_Unet_Poisson,
                NLL,
                provideData(batch_size=25, dimension=dimension),
                batch_size=25,
                optimizer=optimizer,
                dimension=dimension,
                channels=channels,
                metrics=["mse", "accuracy"])

    t.fit(epochs=2)


# LSTM_POISSON()
# Poisson()
# LSTM_Meets_Unet_MIXED()

medium_unet()
#lstm_many_to_one_train()
