#!/home/simon/anaconda3/envs/tensorflow-gpu/bin/python
"""
from __future__ import print_function
import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.optimizers import *
"""
from Utils.Data import Dataset
from Models.Unet import unet
from PIL import Image

"""
print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)



def Unet(optimizer,loss='binary_crossentropy',metrics = ['accuracy'] ):

    
        
   # build Model

    

    model = unet()
    model.compile(optimizer = optimizer, loss = loss, metrics = metrics)

    model.summary()
    return model

model = Unet(Adam(lr = 1e-4))
"""
#data = Dataset("/home/simon/gitprojects/DeepRain2/opticFlow/PNG_NEW/MonthPNGData/YW2017.002_200801",batch_size=5,dim=(256,256))
data = Dataset("/home/simon/gitprojects/DeepRain2_TEST/PNG",batch_size=5,dim=(1100,900))



for i,(x,y) in enumerate(data):
	
	print(x.shape,y.shape,"\t",i,len(data))
	
	


	
	
	
	
#print(Image.fromarray(data[0]).show())