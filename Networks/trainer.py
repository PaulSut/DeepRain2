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

from Utils.Data import Dataset
from Models.Unet import unet
import numpy as np

#dtype='float16'
#K.set_floatx(dtype)
#K.set_epsilon(1e-7) 


print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)



def Unet(optimizer,loss='mse',metrics = ['accuracy'],dimension = (256,256),channels=5 ):

    model = unet(input_size = (*dimension,channels))
    model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
    model.summary()

    return model





channels = 5
#dimension = (256,256)
dimension = (128,128)
batch_size = 20
epochs = 10

pathToData = "/home/simon/gitprojects/DeepRain2/opticFlow/PNG_NEW/MonthPNGData/YW2017.002_200801"

model = Unet(Adam(lr = 1e-7),loss='mse',dimension=dimension,channels=channels)


data = Dataset(pathToData,batch_size=batch_size,dim=dimension,n_channels=channels)

 
history = model.fit(data,verbose=1,epochs=epochs,workers = 0,use_multiprocessing=False,validation_data=data)

print(history)


            
    
    


    
    
    
    
#print(Image.fromarray(data[0]).show())