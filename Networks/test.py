#!/home/simon/anaconda3/envs/tensorflow-gpu/bin/python
from Models.tfModels import *
from trainer import Trainer
from Utils.loss import SSIM
from Models.Unet import unet
from keras.optimizers import *
from Models.CnnLSTM import *

dimension = (256,192)
dimension = (272,224)
#dimension = (128, 112)
#dimension = (64, 56)
channels  = 5
optimizer = Adam( lr = 1e-5 )

args = {   "n_predictions":1,
           "simpleclassification":None,
           "flatten_output":False,
           "activation_hidden":"relu",
           "activation_output":"sigmoid"}

#t = Trainer(UNet64,batch_size = 30, optimizer=optimizer, lossfunction = SSIM(kernel_size = 11), kwargs=args)
#t = Trainer(unet,batch_size = 10, optimizer=optimizer, lossfunction = SSIM(kernel_size = 11))

t = Trainer(LSTM_Meets_Unet,
#t = Trainer(CnnLSTM3,
                batch_size = 10,
                optimizer=optimizer,
                dimension = dimension, 
                channels = channels,
                #lossfunction = SSIM(kernel_size=11,l=1.0))
                lossfunction = "mse")

t.fit( epochs = 3 )




#t = Trainer(unet,lossfunction = SSIM())