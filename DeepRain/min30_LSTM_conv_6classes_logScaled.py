#!/home/simon/anaconda3/envs/DeepRain/bin/python

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import *
import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import l2
import tensorflow_probability as tfp
tfd = tfp.distributions
from Models.Utils import *
from Utils.Dataset import getData
from Utils.transform import cutOut,Normalize,LogBin,LinBin
import os
from Models.Lstm_conv import *


physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

BATCH_SIZE = 50
DIMENSION = (96,96)
CHANNELS = 5
MODELPATH = "./Models_weights"
MODELNAME = "30min_LSTM_categorical_6classes_logScaled"


def getModel(compile_=True):
    modelpath = MODELPATH
    modelname = MODELNAME

    if not os.path.exists(modelpath):
        os.mkdir(modelpath)

    modelpath = os.path.join(modelpath,modelname)

    if not os.path.exists(modelpath):
        os.mkdir(modelpath)

    
    y_transform = [cutOut([16,80,16,80]),LogBin()]
    train,test = getData(BATCH_SIZE,
                         DIMENSION,CHANNELS,
                         timeToPred=30,
                         y_transform=y_transform)

    model = CNN_LSTM_categorical((*DIMENSION,CHANNELS))
    if compile_ == False:
        return model,modelpath,train,test

    neg_log_likelihood = lambda x, rv_x: tf.math.reduce_mean(-rv_x.log_prob(x))


    model.compile(loss=neg_log_likelihood,
                  optimizer=Adam( lr= 1e-4 ))
    model.summary()


    modelpath_h5 = os.path.join(modelpath,
                                modelname+'-{epoch:03d}-{loss:03f}-{val_loss:03f}.h5')

    checkpoint = ModelCheckpoint(modelpath_h5,
                                 verbose=0,
                                 monitor='val_loss',
                                 save_best_only=True,
                                 mode='auto')


    return model,checkpoint,modelpath,train,test

def train():

    modelpath = MODELPATH
    modelname = MODELNAME
    model,checkpoint,modelpath,train,test = getModel()
    history_path = os.path.join(modelpath,modelname+"_history")
    laststate = getBestState(modelpath,history_path)
    test.setWiggle_off()


    if laststate:
        epoch = laststate["epoch"]
        model.load_weights(laststate["modelpath"])

        loss = model.evaluate(x=test, verbose=2)
        print("Restored model, loss: {:5.5f}".format(loss))

        history = model.fit(train,
                            validation_data = test,
                            shuffle         = True,
                            epochs          = 10+epoch,
                            initial_epoch   = epoch,
                            batch_size      = BATCH_SIZE,
                            callbacks       = checkpoint)

        history = mergeHist(laststate["history"],history.history)

    else:
        history = model.fit(train,
                            validation_data = test,
                            shuffle         = True,
                            epochs          = 20,
                            batch_size      = BATCH_SIZE,
                            callbacks       = checkpoint)

        history = history.history



    saveHistory(history_path,history)
    plotHistory(history,history_path,title="6 classes log scaled")


#train()
