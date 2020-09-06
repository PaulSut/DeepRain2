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

BATCH_SIZE = 150
DIMENSION = (96,96)
CHANNELS = 5
MODELPATH = "./Models_weights"
MODELNAME = "zeroInf_negative_Binomial"


def ZeroInflated_negativ_Binomial(input_shape,
           n_predictions=1,
           simpleclassification=None,
           flatten_output=False,
           activation_hidden="relu",
           activation_output="relu"):


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

    
    layer = Conv2D(6, (3, 3), activation="selu")(up01)  # 1 x 64x64

    cat = Flatten()(layer[:,:,:,:2])
    count = Flatten()(layer[:,:,:,2:4])
    prob = Flatten()(layer[:,:,:,4:6])
    
    cat      = Dense(512)(cat)
    count      = Dense(512)(count)
    prob      = Dense(512)(prob)
    
    
    #cat = Dropout(0.25)(cat)
    #count = Dropout(0.25)(count)
    #prob = Dropout(0.25)(prob)
    
    cat = Dense(64*64,activation="sigmoid")(cat)
    count = Dense(64*64,activation="relu")(count)
    prob = Dense(64*64,activation="sigmoid")(prob)
    
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
                         timeToPred=30,
                         y_transform=y_transform)

                         #x_transform=x_transform)

    input_shape = (*DIMENSION,CHANNELS)

    model = ZeroInflated_negativ_Binomial(input_shape=input_shape)

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
                            epochs          = 45+epoch,
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
    plotHistory(history,history_path,title="ZeroInf NegativeBinomial NLL-loss")



def eval():
    #windowname = 'OpenCvFrame'
    #cv.namedWindow(windowname)
    #cv.moveWindow(windowname,0,00)
    modelpath = MODELPATH
    modelname = MODELNAME

    values = 256
    ones = np.ones((1,64,64,1),np.float32)
    value_array = np.repeat(ones,values,axis=-1)

    for i in range(values):
        value_array[:,:,:,i] *= i


    def probs(distribution):
        prob_array = np.zeros_like(value_array)
        for i in range(values):
            prob_array[:,:,:,i:i+1] = distribution.prob(value_array[:,:,:,i:i+1])
        print(prob_array.argmax(axis=-1).max())
        return prob_array

    model,checkpoint,modelpath,train,test = getModel()   
    predictions = []
    for nbr,(x,y) in enumerate(test):
        print("{:7d}|{:7d}".format(nbr,len(test)),end="\r")

        for i in range(BATCH_SIZE):
            pred = model(np.array([x[i,:,:,:]]))
            p = np.array(pred.mean())
            #probs(pred)
            print(y[i,:,:].max(),np.max(pred.cdf(5)) )
            
            
    np.save(os.path.join(modelpath,modelname+"_predictions"),np.array(predictions))



#train()
#eval()