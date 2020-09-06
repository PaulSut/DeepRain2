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

BATCH_SIZE = 200
DIMENSION = (96,96)
CHANNELS = 5
MODELPATH = "./Models_weights"
MODELNAME = "10min_categorical_6classes_logScaled"


def six_class_categorical(input_shape,
                          activation_hidden="relu"):


    inputs = Input(shape=input_shape) 

    conv01 = Conv2D(20, kernel_size=(3, 3), padding="same")(inputs)       # 10 x 64x64
    conv01 = Activation(activation_hidden)(conv01)
    conv01_pool = MaxPooling2D((2, 2), strides=(2, 2))(conv01)            # 10 x 32x32


    conv02 = Conv2D(25, kernel_size=(3, 3), padding="same")(conv01_pool)  # 20 x 32x32
    conv02 = Activation(activation_hidden)(conv02)
    conv02_pool = MaxPooling2D((2, 2), strides=(2, 2))(conv02)            # 20 x 16x16


    conv03 = Conv2D(25, kernel_size=(3, 3), padding="same")(conv02_pool)  # 20 x 16x16
    conv03 = Activation(activation_hidden)(conv03)
    conv03_pool = MaxPooling2D((2, 2), strides=(2, 2))(conv03)            # 20 x 8x8


    conv04 = Conv2D(25, kernel_size=(3, 3), padding="same")(conv03_pool)  # 20 x 8x8
    conv04 = Activation(activation_hidden)(conv04)
    conv04_pool = MaxPooling2D((2, 2), strides=(2, 2))(conv04)            # 20 x 4x4


    ### UPSAMPLING:
    up04 = UpSampling2D((2, 2))(conv04_pool)    # 20 x 8x8
    up04 = concatenate([conv04, up04], axis=3)  # 20+20 x 8x8


    up03 = UpSampling2D((2, 2))(up04)           # 40 x 16x16
    up03 = concatenate([conv03, up03], axis=3)  # 20+40 x 16x16


    up02 = UpSampling2D((2, 2))(up03)           # 60 x 32x32
    up02 = concatenate([conv02, up02], axis=3)  # 20+60 x 32x32

    
    up01 = UpSampling2D((2, 2))(up02)           # 80 x 64x64
    up01 = concatenate([conv01, up01], axis=3)  # 10+80 x 64x64

    
    layer = Conv2D(1, (7, 7), activation="relu")(up01)  # 1 x 64x64


    prob = Flatten()(layer)
    
    prob      = Dense(128)(prob)
    

    prob = Dense(64*64*7,activation="linear")(prob)
    
    prob = tf.keras.layers.Reshape((64,64,1,7))(prob)

    #prob = tf.nn.softmax(prob,axis=-1)
    
    prob = tf.math.softmax(prob,axis=-1)
    #prob = tf.nn.softmax(tf.math.log(prob),axis=-1)
    

    output_dist = tfp.layers.DistributionLambda(
        name="DistributionLayer",
        make_distribution_fn=lambda t: tfp.distributions.Independent(
        tfd.Categorical(logits=tf.math.log(t[...,:,:,:]))
        #tfd.Categorical(probs = t[...,:,:,:])
        ,name="categorical",reinterpreted_batch_ndims=0 ))
    
    
    

    output = output_dist(prob)
    
    model = Model(inputs=inputs, outputs=output)
    return model


def getModel(compile_=True):
    modelpath = MODELPATH
    modelname = MODELNAME

    if not os.path.exists(modelpath):
        os.mkdir(modelpath)

    modelpath = os.path.join(modelpath,modelname)

    if not os.path.exists(modelpath):
        os.mkdir(modelpath)

    #x_transform = [Normalize(0.0033127920560417023,0.012673124326485102)]
    #x_transform = [LogBin()]
    y_transform = [cutOut([16,80,16,80]),LogBin()]
    #y_transform = [cutOut([96,160,96,160]),LogBin()]

    train,test = getData(BATCH_SIZE,
                         DIMENSION,CHANNELS,
                         timeToPred=10,
                         y_transform=y_transform)


    model = six_class_categorical((*DIMENSION,CHANNELS))
    if compile_ == False:
        return model,modelpath,train,test

    neg_log_likelihood = lambda x, rv_x: tf.math.reduce_mean(-rv_x.log_prob(x))
    def NLL(y_true, y_hat):
        return -y_hat.log_prob(y_true)

    model.compile(loss=NLL,
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
                            epochs          = 65+epoch,
                            initial_epoch   = epoch,
                            batch_size      = BATCH_SIZE,
                            callbacks       = checkpoint)

        history = mergeHist(laststate["history"],history.history)

    else:
        history = model.fit(train,
                            validation_data = test,
                            shuffle         = True,
                            epochs          = 100,
                            batch_size      = BATCH_SIZE,
                            callbacks       = checkpoint)

        history = history.history



    saveHistory(history_path,history)
    plotHistory(history,history_path,title="10min Categorical 7 classes")


#train()
