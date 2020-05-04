#!/home/simon/anaconda3/envs/tensorflow-gpu/bin/python
from Utils.loadset import getDataSet
from Utils.Data import dataWrapper
from Models.tfModels import UNet64
from keras.optimizers import *
from trainer import Trainer
from Utils.transform import ToCategorical,cutOut
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
batch_size = 400

# map values to categorical conditions
categorical_conditions = [0,20,40,60,80,100,140,255]

transform = [ToCategorical(categorical_conditions)]


def classification():

    args = {"flatten_output" : True,
            "activation_output" : "softmax",
            "n_predictions" : len(categorical_conditions) - 1}
            #"simpleclassification" : dimension[0]*dimension[1]}


    train, test = provideData(flatten = True,transform=transform,batch_size=batch_size,preTransformation=cutOut())

    trainer = Trainer(UNet64,
                "categorical_crossentropy",
                pathToData=(train,test),
                batch_size = batch_size,
                flatten = True,
                optimizer=optimizer,
                dimension = dimension,
                channels = channels,
                metrics=["mse"],
                kwargs=args)


    trainer.fit( epochs = 15 )
    

def mse():

    trainer = Trainer(UNet64,
                "mse",
                pathToData=provideData(),
                batch_size = 10,
                optimizer=optimizer,
                dimension = dimension,
                channels = channels)


    trainer.fit( epochs = 30 )



def provideData(flatten=False,batch_size=10,transform=None,preTransformation=None):

    getDataSet(DatasetFolder,year=[2017],username=USRN,pswd=PSWD)
    train,test = dataWrapper(PathToData,
                            dimension=dimension,
                            channels=channels,
                            batch_size=batch_size,
                            overwritecsv=True,
                            flatten=flatten,
                            onlyUseYears=[2017],
                            transform=transform,
                            preTransformation=preTransformation)
    
    return train,test


classification()