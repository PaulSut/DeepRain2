
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import *
import tensorflow as tf
from tensorflow.keras.regularizers import l2
import tensorflow_probability as tfp
tfd = tfp.distributions
def convLayer(inp,
        nbrLayer,
        channel,
        activation="relu",
        bias_regularizer = 0.01,
        kernel_regularizer = 0.01):

    assert nbrLayer > 0, "In Function convLayer nbrLayer > 0 ?"
    layer = Conv2D(channel, kernel_size=(3, 3), padding="same",
        kernel_regularizer=l2(kernel_regularizer), 
        bias_regularizer=l2(bias_regularizer)) (inp)
    layer = Activation(activation)(layer)
    layer = BatchNormalization()(layer)
    
    for i in range(1,nbrLayer):
        layer = Conv2D(channel, kernel_size=(3, 3), padding="same",
            kernel_regularizer=l2(kernel_regularizer), 
            bias_regularizer=l2(bias_regularizer))  (layer)
        layer = Activation(activation)(layer)
        layer = BatchNormalization()(layer)
    return layer

def Unet(input_shape,
        down_channels=[64,128,256,512],
        downLayer=2,
        activation="selu",
        output_activation = "selu",
        output_dim = 1,
        bias_regularizer = 0.01,
        kernel_regularizer = 0.01,
        ):
    
    inputs = Input(shape=input_shape)
    
    layer = Conv2D(down_channels[0], kernel_size=(3, 3), padding="same",
        kernel_regularizer=l2(kernel_regularizer), 
        bias_regularizer=l2(bias_regularizer)) (inputs)
    layer = Activation(activation)(layer)
    layer = BatchNormalization()(layer)
    
    layer = Conv2D(down_channels[0], kernel_size=(3, 3), padding="same",
        kernel_regularizer=l2(kernel_regularizer), 
        bias_regularizer=l2(bias_regularizer)) (layer)
    layer = Activation(activation)(layer)
    firstLayer = BatchNormalization()(layer)
    
    pool  = MaxPooling2D((2, 2), strides=(2, 2))(firstLayer)
    
    layerArray = []
    
    for channel in down_channels[1:]:
        
        layer = convLayer(pool,
                        downLayer,
                        channel,
                        kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer)
       
        if channel != down_channels[-1]:
            layerArray.append(layer)
            pool  = MaxPooling2D((2, 2), strides=(2, 2))(layer)
            
    for i,channel in enumerate(reversed(down_channels[:-1])):
        
        layer = Conv2DTranspose(channel,(3, 3),strides=(2,2),padding="same",
            kernel_regularizer=l2(kernel_regularizer), 
            bias_regularizer=l2(bias_regularizer))(layer)
        layer = Activation(activation)(layer)
        layer = BatchNormalization() (layer)
        
        if len(layerArray) >= (i+1):
            layer = concatenate([layerArray[-(i+1)], layer], axis=3)
        else:
            layer = concatenate([firstLayer, layer], axis=3)
        
        layer = convLayer(layer,
            downLayer,
            channel,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
            )
        
    output = Conv2D(output_dim, kernel_size=(1, 1), padding="same",
        activation=output_activation,
        kernel_regularizer=l2(kernel_regularizer), 
        bias_regularizer=l2(bias_regularizer)) (layer)
    return inputs,output
    #return Model(inputs=inputs, outputs=output)
    


def simpleUnet(input_shape,
           n_predictions=1,
           simpleclassification=None,
           flatten_output=False,
           activation_hidden="relu",
           activation_output="relu"):


    

    inputs = Input(shape=input_shape) 

    conv01 = Conv2D(10, kernel_size=(3, 3), padding="same")(inputs)       # 10 x 64x64
    conv01 = Activation(activation_hidden)(conv01)
    conv01_pool = MaxPooling2D((2, 2), strides=(2, 2))(conv01)            # 10 x 32x32


    conv02 = Conv2D(20, kernel_size=(3, 3), padding="same")(conv01_pool)  # 20 x 32x32
    conv02 = Activation(activation_hidden)(conv02)
    conv02_pool = MaxPooling2D((2, 2), strides=(2, 2))(conv02)            # 20 x 16x16


    conv03 = Conv2D(20, kernel_size=(3, 3), padding="same")(conv02_pool)  # 20 x 16x16
    conv03 = Activation(activation_hidden)(conv03)
    conv03_pool = MaxPooling2D((2, 2), strides=(2, 2))(conv03)            # 20 x 8x8


    conv04 = Conv2D(20, kernel_size=(3, 3), padding="same")(conv03_pool)  # 20 x 8x8
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

    
    layer = Conv2D(2, (1, 1), activation="linear")(up01)  # 1 x 64x64

    layer_1 = Flatten()(layer[:,:,:,:1])
    layer_2 = Flatten()(layer[:,:,:,1:2])
    layer_1 = Dropout(0.25)(layer_1)
    layer_2 = Dropout(0.25)(layer_2)
    layer_1 = Dense(64*64,activation="relu")(layer_1)
    layer_2 = Dense(64*64,activation="relu")(layer_2)
    layer_1 = tf.keras.layers.Reshape((64,64,1))(layer_1)
    layer_2 = tf.keras.layers.Reshape((64,64,1))(layer_2)
    input_dist= tf.concat([layer_1,layer_2],axis=-1)

    output_dist = tfp.layers.DistributionLambda(name="DistributionLayer",
                                                make_distribution_fn=lambda t: tfp.distributions.Independent(
                                                tfd.Mixture(
                                                cat=tfd.Categorical(tf.stack([1-t[...,:1], t[...,:1]],axis=-1)),
                                                components=[tfd.Deterministic(loc=tf.zeros_like(t[...,:1])),
                                                tfd.Poisson(rate=tf.math.softplus(t[...,1:2]))])
                                                ,name="ZeroInflated",reinterpreted_batch_ndims=0 ))
    
    
    output = output_dist(input_dist)
    model = Model(inputs=inputs, outputs=output)
    return model