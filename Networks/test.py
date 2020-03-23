#!/home/simon/anaconda3/envs/tensorflow-gpu/bin/python
from Models.tfModels import *
from trainer import Trainer
from Utils.loss import SSIM
from Models.Unet import unet
from keras.optimizers import *
from Models.CnnLSTM import *

dimension = (256,192)
dimension = (272,224)
channels  = 5
optimizer = Adam( lr = 1e-5 )
pathToData = "/home/simon/MonthPNGData/MonthPNGData"


t = Trainer(LSTM_Meets_Unet_Upconv,
                "mse",
                pathToData,
                batch_size = 10,
                optimizer=optimizer,
                dimension = dimension,
                channels = channels)

t.fit( epochs = 3 )



