#!/home/simon/anaconda3/envs/tensorflow-gpu/bin/python
from __future__ import print_function
import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.optimizers import *
from Utils.loss import SSIM
from Utils.Data import Dataset,dataWrapper
from Models.tfModels import UNet64
from Models.Unet import unet
import numpy as np



#dtype='float16'
#K.set_floatx(dtype)
#K.set_epsilon(1e-7) 
pathToData = "/home/simon/gitprojects/DeepRain2/opticFlow/PNG_NEW/MonthPNGData/YW2017.002_200801"
print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

def traditionalUnet():
    PATHTOMODEL = "./models_h5"
    MODELNAME = "UNET.h5"
    MODELPATH = os.path.join(PATHTOMODEL,MODELNAME)

    if not os.path.exists(PATHTOMODEL):
        os.mkdir(PATHTOMODEL)




    def Unet(optimizer,loss='mse',metrics = ['accuracy'],dimension = (256,256),channels=5 ):

        model = unet(input_size = (*dimension,channels))
        model.compile(optimizer = optimizer, loss = loss, metrics = metrics,sample_weight_mode=None)
        model.summary()

        return model





    epochs = 10
    channels = 5
    # we should keep ratio of original images
    # dimension should be multiple divisible by two (4 times)

    dimension = (272,224)
    batch_size = 5


    train,test = dataWrapper(pathToData,dimension = dimension,channels = channels,batch_size = batch_size)

    #model = Unet(Adam(lr = 1e-3),loss=SSIM(kernel_size=11),dimension=dimension,channels=channels)
    model = Unet(Adam(lr = 1e-3),loss="mse",dimension=dimension,channels=channels)
    try:
        model.load(MODELPATH)
    except Exception as e:
        pass

    history = model.fit(train,verbose=1,epochs=epochs,workers = 0,use_multiprocessing=False,validation_data=test)

    model.reset_metrics()
    model.save(MODELPATH)

def deepRainUnet():

    PATHTOMODEL = "./models_h5"
    MODELNAME = "UNET64.h5"
    MODELPATH = os.path.join(PATHTOMODEL,MODELNAME)

    if not os.path.exists(PATHTOMODEL):
        os.mkdir(PATHTOMODEL)

    epochs = 15
    channels = 5
    dimension = (128,112)
    batch_size = 30
    flatten = False
    model = UNet64((*dimension,channels),
                    lossfunction=SSIM(kernel_size=11),
                    metrics=["accuracy"],
                    flatten_output=flatten,
                    optimizer = Adam(lr = 1e-3))
    model.summary()

    train,test = dataWrapper(pathToData,dimension = dimension,channels = channels,batch_size = batch_size,flatten=flatten)
    
        
    history = model.fit(train,verbose=1,epochs=epochs,workers = 0,use_multiprocessing=False,validation_data=test)

    model.save_weights(MODELPATH)

def lossTest():
    epochs = 10
    channels = 5
    dimension = (128,112)
    batch_size = 5
    train,test = dataWrapper(pathToData,dimension = dimension,channels = channels,batch_size = batch_size,flatten=False)
    
    
    #y_1 = np.arange(0,100,dtype=np.float32).reshape(1,10,10,1) / 100
    #y_2 = np.arange(0,100,dtype=np.float32).reshape(1,10,10,1) / 100
    
    
    _,y_1 = train[0]
    _,y_2 = train[1]
    ssim = SSIM(kernel_size=11)
    print(ssim(y_1,y_2))
    print(tf.image.ssim(tf.convert_to_tensor(y_1),tf.convert_to_tensor(y_2),max_val=1.0,filter_size=3))
    
#lossTest()
deepRainUnet()
#traditionalUnet()



