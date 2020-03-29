#!/home/simon/anaconda3/envs/tensorflow-gpu/bin/python
from Utils.loadset import getDataSet
from Utils.Data import dataWrapper
from Models.tfModels import UNet64
from keras.optimizers import *
from trainer import Trainer
try:
    from Utils.connection_cfg import *
except Exception as e:
    PSWD = None
    USRN = None
import os


DatasetFolder = "./Data/RAW"
PathToData = os.path.join(DatasetFolder,"MonthPNGData")
dimension = (64,64)
channels  = 5
optimizer = Adam( lr = 1e-5 )
batch_size = 30
def classification():

    args = {"flatten_output" : True,
            "activation_output" : "softmax",
            "simpleclassification" : dimension[0]*dimension[1]}

    train, test = provideData(flatten = True)

    trainer = Trainer(UNet64,
                "categorical_crossentropy",
                pathToData=(train,test),
                batch_size = batch_size,
                flatten = True,
                optimizer=optimizer,
                dimension = dimension,
                channels = channels,
                kwargs=args)


    trainer.fit( epochs = 3 )
    

def mse():

    trainer = Trainer(UNet64,
                "mse",
                pathToData=provideData(),
                batch_size = 10,
                optimizer=optimizer,
                dimension = dimension,
                channels = channels)


    trainer.fit( epochs = 3 )



def provideData(flatten=False,batch_size=10):
    getDataSet(DatasetFolder,year=[2017],username=USRN,pswd=PSWD)
    train,test = dataWrapper(PathToData,
                            dimension=dimension,
                            channels=channels,
                            batch_size=batch_size,
                            overwritecsv=True,
                            flatten=flatten,
                            onlyUseYears=[2017])
    
    return train,test


classification()