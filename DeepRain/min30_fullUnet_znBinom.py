#!/home/simon/anaconda3/envs/DeepRain/bin/python
from tensorflow.keras.optimizers import Adam
from Models.Unet import Unet,simpleUnet
from Models.Loss import NLL
from Models.Distributions import *
from Models.Utils import *
from tensorflow.keras.layers import *
from tensorflow.keras import Sequential, Model
from Utils.Dataset import getData
from Utils.transform import cutOut,Normalize
from tensorflow.keras.callbacks import *
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import l2
import os
import cv2 as cv

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

BATCH_SIZE = 100
DIMENSION = (96,96)
CHANNELS = 5
MODELPATH = "./Models_weights"
MODELNAME = "30min_fullUnet_znBinom"


def fullUnet_znBinom(input_shape):

    inputs,outputs = Unet(input_shape,
        output_dim = 3,
        down_channels=[64,128,256,512],
        bias_regularizer = 0.01,
        kernel_regularizer = 0.01)
    
    outputs = GaussianDropout(0.25)(outputs)
    cat = Flatten()(outputs[:,:,:,:1])
    count = Flatten()(outputs[:,:,:,1:2])
    prob = Flatten()(outputs[:,:,:,2:])
    
    cat      = Dense(256)(cat)
    count      = Dense(256)(count)
    prob      = Dense(256)(prob)
    
    
    cat = Dense(64*64,activation="linear")(cat)
    count = Dense(64*64,activation="linear")(count)
    prob = Dense(64*64,activation="linear")(prob)
    
    cat = tf.keras.layers.Reshape((64,64,1))(cat)
    count = tf.keras.layers.Reshape((64,64,1))(count)
    prob = tf.keras.layers.Reshape((64,64,1))(prob)

    
    input_dist= tf.concat([cat,count,prob],axis=-1,name="ConcatLayer")

    output_dist = tfp.layers.DistributionLambda(
        name="DistributionLayer",
        make_distribution_fn=lambda t: tfp.distributions.Independent(
        tfd.Mixture(
            cat=tfd.Categorical(tf.stack([1-tf.math.sigmoid(t[...,:1]), tf.math.sigmoid(t[...,:1])],axis=-1)),
            components=[tfd.Deterministic(loc=tf.zeros_like(t[...,:1])),
            tfp.distributions.NegativeBinomial(
            total_count=tf.math.softplus(t[..., 1:2]), 
            logits=tf.math.sigmoid(t[..., 2:]) ),])
        ,name="ZeroInflated_Binomial",reinterpreted_batch_ndims=0 ))
    

    output = output_dist(input_dist)
    model = Model(inputs=inputs, outputs=output)
    return model


def getModel(compile_ = True):

    modelpath = MODELPATH
    modelname = MODELNAME
    if not os.path.exists(modelpath):
        os.mkdir(modelpath)

    modelpath = os.path.join(modelpath,modelname)

    if not os.path.exists(modelpath):
        os.mkdir(modelpath)


    
    y_transform = [cutOut([16,80,16,80])]
    train,test = getData(BATCH_SIZE,
                         DIMENSION,CHANNELS,
                         timeToPred=40,
                         y_transform=y_transform)

                         #x_transform=x_transform)

    input_shape = (*DIMENSION,CHANNELS)

    model = fullUnet_znBinom(input_shape=input_shape)

    if compile_ == False:
        return model,modelpath,train,test
    neg_log_likelihood = lambda x, rv_x: tf.math.reduce_mean(-rv_x.log_prob(x))
    
    model.compile(loss=neg_log_likelihood,
                  optimizer=Adam( lr= 1e-3 ))
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
                            epochs          = 45+epoch,
                            initial_epoch   = epoch,
                            batch_size      = BATCH_SIZE,
                            callbacks       = checkpoint)

        history = mergeHist(laststate["history"],history.history)

    else:
        history = model.fit(train,
                            validation_data = test,
                            shuffle         = True,
                            epochs          = 30,
                            batch_size      = BATCH_SIZE,
                            callbacks       = checkpoint)

        history = history.history



    saveHistory(history_path,history)
    plotHistory(history,history_path,title="ZeroInf NegativeBinomial NLL-loss")






#train()
#eval()
