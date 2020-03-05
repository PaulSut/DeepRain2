#!/home/simon/anaconda3/envs/tensorflow-gpu/bin/python
from __future__ import print_function
import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.optimizers import *

from Utils.Data import Dataset,dataWrapper
from Models.Unet import unet
import numpy as np
import os

#dtype='float16'
#K.set_floatx(dtype)
#K.set_epsilon(1e-7) 

PATHTOMODEL = "./models_h5"
MODELNAME = "UNET.h5"
MODELPATH = os.path.join(PATHTOMODEL,MODELNAME)

if not os.path.exists(PATHTOMODEL):
    os.mkdir(PATHTOMODEL)

print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)



def Unet(optimizer,loss='mse',metrics = ['accuracy'],dimension = (256,256),channels=5 ):

    model = unet(input_size = (*dimension,channels))
    model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
    model.summary()

    return model




pathToData = "/home/simon/gitprojects/DeepRain2/opticFlow/PNG_NEW/MonthPNGData/YW2017.002_200801"

channels = 5
# we should keep ratio of original images
# dimension should be multiple divisible by two (4 times)

dimension = (272,224)
batch_size = 5
epochs = 10


train,test = dataWrapper(pathToData,dimension = dimension,channels = channels,batch_size = batch_size)

model = Unet(Adam(lr = 1e-6),loss='mse',dimension=dimension,channels=channels)
try:
    model.load(MODELPATH)
except Exception as e:
    pass

history = model.fit(train,verbose=1,epochs=epochs,workers = 0,use_multiprocessing=False,validation_data=test)

model.reset_metrics()
model.save(MODELPATH)

