#!/home/simon/anaconda3/envs/tensorflow-gpu/bin/python

from trainer import Trainer
from nn_Setup import cnnLSTM_config as cnnConf
from Utils.Data import provideData
from Models.CnnLSTM import LSTM_Meets_Unet_ZeroInflated
from predicter import Predictor
from tensorflow.keras.models import load_model
import tensorflow as tf



trainset,testset = provideData(cnnConf.DIMENSION,
                    cnnConf.BATCHSIZE,
                    cnnConf.INPUT_CHANNELS,
                    preTransformation=cnnConf.PRETRAINING_TRANSFORMATIONS,
                    transform=cnnConf.TRANSFORMATIONS)



t = Trainer(model           = LSTM_Meets_Unet_ZeroInflated,
            pathToData      = (trainset,testset),
            lossfunction    = cnnConf.LOSSFUNCTION,
            batch_size      = cnnConf.BATCHSIZE,
            channels        = cnnConf.INPUT_CHANNELS,
            optimizer       = cnnConf.OPTIMIZER,
            dimension       = cnnConf.DIMENSION,
            metrics         = cnnConf.METRICS
            )

t.fit( epochs = 100 )


#model = load_model('model_data/LSTM_Meets_Unet_ZeroInflated_function/LSTM_Meets_Unet_ZeroInflated_function64x64x5.h5',
#                    custom_objects = cnnConf.CUSTOM_OBJECTS)