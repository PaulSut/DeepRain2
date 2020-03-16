#!/home/simon/anaconda3/envs/tensorflow-gpu/bin/python
from Models.tfModels import *
from trainer import Trainer
from Utils.loss import SSIM
from Models.Unet import unet
from keras.optimizers import *
from Models.CnnLSTM import CnnLSTM

#dimension = (272,224)
dimension = (128, 112)
channels  = 5
optimizer = Adam( lr = 1e-4 )

args = {   "n_predictions":1,
           "simpleclassification":None,
           "flatten_output":False,
           "activation_hidden":"relu",
           "activation_output":"sigmoid"}

#t = Trainer(UNet64,batch_size = 30, optimizer=optimizer, lossfunction = SSIM(kernel_size = 11), kwargs=args)
#t = Trainer(unet,batch_size = 10, optimizer=optimizer, lossfunction = SSIM(kernel_size = 11))
t = Trainer(CnnLSTM,batch_size = 8, optimizer=optimizer,dimension = dimension, lossfunction = SSIM(kernel_size = 7))
t.fit( epochs = 5 )
#t = Trainer(unet,lossfunction = SSIM())