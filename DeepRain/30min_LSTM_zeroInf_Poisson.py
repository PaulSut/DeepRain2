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
from Utils.transform import cutOut,Normalize
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
MODELNAME = "30min_LSTM_zPoisson"

modelpath = MODELPATH
modelname = MODELNAME

if not os.path.exists(modelpath):
    os.mkdir(modelpath)

modelpath = os.path.join(modelpath,modelname)

if not os.path.exists(modelpath):
    os.mkdir(modelpath)
y_transform = [cutOut([16,80,16,80])]
#x_transform = [Normalize(0.007742631458799244, 0.015872015890555563 )]
train,test = getData(BATCH_SIZE,
                     DIMENSION,CHANNELS,
                     timeToPred=30,
                     keep_sequences = True,
                     y_transform=y_transform)


neg_log_likelihood = lambda x, rv_x: tf.math.reduce_mean(-rv_x.log_prob(x))


model = CNN_LSTM_Poisson((*DIMENSION,CHANNELS))
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
                        epochs          = 65+epoch,
                        initial_epoch   = epoch,
                        batch_size      = BATCH_SIZE,
                        callbacks       = checkpoint)

    history = mergeHist(laststate["history"],history.history)

else:
    history = model.fit(train,
                        validation_data = test,
                        shuffle         = True,
                        epochs          = 5,
                        batch_size      = BATCH_SIZE,
                        callbacks       = checkpoint)

    history = history.history



saveHistory(history_path,history)
plotHistory(history,history_path,title="LSTM zPoisson")
