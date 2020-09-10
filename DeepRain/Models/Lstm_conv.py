
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.backend import int_shape
import tensorflow_probability as tfp
tfd = tfp.distributions

def inception_v1(inp,channel,activation="selu"):

    inception_1 = Conv2D(channel, 
                        kernel_size=(1, 1), 
                        padding="same",
                        activation=activation)(inp)
    inception_3 = Conv2D(channel, 
                        kernel_size=(3, 3), 
                        padding="same",
                        activation=activation)(inp)
    inception_5 = Conv2D(channel, 
                        kernel_size=(5, 5), 
                        padding="same",
                        activation=activation)(inp)


    return tf.keras.layers.concatenate([inception_1, 
                                        inception_3, 
                                        inception_5], axis = 3)
    

def inception_v2(inp,channel,activation="selu"):

    inception_1 = Conv2D(channel, 
                        kernel_size=(1, 1), 
                        padding="same",
                        activation=activation)(inp)

    inception_3 = Conv2D(channel, 
                        kernel_size=(3, 3), 
                        padding="same",
                        activation=activation)(inp)
    inception_3_1 = Conv2D(channel, 
                        kernel_size=(3, 3), 
                        padding="same",
                        activation=activation)(inp)
    inception_3_2 = Conv2D(channel, 
                        kernel_size=(3, 3), 
                        padding="same",
                        activation=activation)(inception_3_1)

    return tf.keras.layers.concatenate([inception_1, 
                                        inception_3, 
                                        inception_3_2], axis = 3)



def lstmLayer(inp,filters = [5,5],activation="selu",padding="same",kernel_size=(3,3)):

    shape_inp = int_shape(inp)


    lstm_shape = Reshape((shape_inp[-1],shape_inp[1],shape_inp[2],1))(inp)


    lstm_conv = ConvLSTM2D(filters=filters[0], 
                           kernel_size=kernel_size, 
                           activation=activation,
                           padding=padding, 
                           return_sequences=True,
                           data_format='channels_last')(lstm_shape)
    
    

    for i in filters[1:-1]:
        lstm_conv = ConvLSTM2D(filters=i, 
                               kernel_size=kernel_size, 
                               activation=activation,
                               padding=padding, 
                               return_sequences=True,
                               data_format='channels_last')(lstm_conv)
        

    lstm_conv = ConvLSTM2D(filters=filters[-1], 
                           kernel_size=kernel_size, 
                           activation=activation,
                           padding=padding, 
                           return_sequences=False,
                           data_format='channels_last')(lstm_conv)
    

    return lstm_conv


def CNN_LSTM(input_shape,output = (64,64)):
    inputs      = Input(shape=input_shape)
    inception_1 = inception_v2(inputs,input_shape[-1])
    lstm_conv1 = lstmLayer(inception_1,filters = [5,5,5])

    inception_2 = inception_v2(inception_1,16)
    inception_3 = inception_v2(inception_2,16)
    inception_4 = inception_v2(inception_3,16)
    #lstm_conv2 = lstmLayer(inception_4,filters = [3,5,1])

    layer = tf.concat([lstm_conv1,inception_4],axis=-1,name="ConcatLayer")
    layer = inception_v2(layer,3)
    layer = Conv2D(3,kernel_size=(7,7))(layer)
    layer = Conv2D(3,kernel_size=(7,7))(layer)


    cat = Flatten()(layer[:,:,:,:1])
    count = Flatten()(layer[:,:,:,1:2])
    prob = Flatten()(layer[:,:,:,2:])
    
    cat      = Dense(128)(cat)
    count      = Dense(128)(count)
    prob      = Dense(128)(prob)
    
    
    cat = Dense(output[0]*output[1],activation="sigmoid")(cat)
    count = Dense(output[0]*output[1],activation="selu")(count)
    prob = Dense(output[0]*output[1],activation="sigmoid")(prob)
    
    cat = tf.keras.layers.Reshape((*output,1))(cat)
    count = tf.keras.layers.Reshape((*output,1))(count)
    prob = tf.keras.layers.Reshape((*output,1))(prob)
 
    
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

def CNN_LSTM_Poisson(input_shape):
    inputs      = Input(shape=input_shape)
    inception_1 = inception_v2(inputs,input_shape[-1])

    inception_2 = inception_v2(inception_1,16)

    inception_1_2 = inception_v2(inputs,input_shape[-1])
    inception_2_2 = inception_v2(inception_1_2,16)
    

    layer = tf.concat([inception_2_2,inception_2],axis=-1,name="ConcatLayer")
    layer = inception_v2(layer,5)
    layer = Conv2D(6,kernel_size=(5,5))(layer)
    layer = Conv2D(6,kernel_size=(5,5))(layer)
    layer = Conv2D(6,kernel_size=(5,5))(layer)
    layer = Conv2D(5,kernel_size=(5,5))(layer)
    layer = lstmLayer(layer,filters = [5,3,2])

    
    cat = Flatten()(layer[:,:,:,:1])
    count = Flatten()(layer[:,:,:,1:])

    
    cat      = Dense(256)(cat)
    count      = Dense(256)(count)

    
    
    cat = Dense(64*64,activation="sigmoid")(cat)
    count = Dense(64*64,activation="selu")(count)

    
    cat = tf.keras.layers.Reshape((64,64,1))(cat)
    count = tf.keras.layers.Reshape((64,64,1))(count)
 
    
    input_dist= tf.concat([cat,count],axis=-1,name="ConcatLayer")
 

    def ZeroInflated_Poisson():
      return tfp.layers.DistributionLambda(
            name="DistributionLayer",
            make_distribution_fn=lambda t: tfp.distributions.Independent(
            tfd.Mixture(
                cat=tfd.Categorical(probs=tf.stack([1-t[...,0:1], t[...,0:1]],axis=-1)),
                components=[tfd.Deterministic(loc=tf.zeros_like(t[...,0:1])),
                tfd.Poisson(rate=tf.math.softplus(t[...,1:2]))]),
            name="ZeroInflated",reinterpreted_batch_ndims=0 ))
    output_dist = ZeroInflated_Poisson()
    output = output_dist(input_dist)
    model = Model(inputs=inputs, outputs=output)

    return model


def CONV_LSTM_SMALL(input_shape):
    inputs      = Input(shape=input_shape)

    
    inception_1 = inception_v2(inputs,input_shape[-1])
    conv_1 = Conv2D(12,kernel_size=(5,5))(inception_1)
    conv_1 = Conv2D(3,kernel_size=(3,3))(conv_1)

    inception_2 = inception_v2(conv_1,6)
    inception_2 = inception_v2(inception_2,20)
    inception_2 = inception_v2(inception_2,6)

    conv_2 = Conv2D(12,kernel_size=(5,5))(inception_2)
    conv_2 = Conv2D(6,kernel_size=(3,3))(conv_2)

    lstm_conv2 = lstmLayer(conv_2,filters = [6,10,6],padding="valid")
    inception_3 = inception_v2(lstm_conv2,32)
    inception_3 = inception_v2(inception_3,32)


    inception_1_1 = inception_v2(inception_1,16)
    conv_1_1 = Conv2D(12,kernel_size=(5,5))(inception_1_1)
    conv_1_1 = Conv2D(16,kernel_size=(3,3))(conv_1_1)

    inception_1_2 = inception_v2(conv_1_1,16)
    conv_1_2 = Conv2D(12,kernel_size=(5,5))(inception_1_2)
    conv_1_2 = Conv2D(16,kernel_size=(3,3))(conv_1_2)
    inception_1_3 = inception_v2(conv_1_2,16)
    conv_1_3 = Conv2D(12,kernel_size=(5,5))(inception_1_3)
    conv_1_3 = Conv2D(12,kernel_size=(3,3))(conv_1_3)
    conv_1_3 = tf.concat([lstm_conv2,conv_1_3],axis=-1)
    
    
    conc = tf.concat([conv_1_3,inception_3],axis=-1,name="ConcatLayer")

    conv_3 = Conv2D(160,kernel_size=(3,3))(conc)
    conv_3 = Conv2D(32,kernel_size=(3,3))(conv_3)
    inception_3 = inception_v2(conv_3,32)

    inception_4 = inception_v2(inception_3,16)
    conv_4 = Conv2D(8,kernel_size=(5,5))(inception_4)
    conv_5 = Conv2D(3,kernel_size=(3,3))(conv_4)

    cat = Flatten()(conv_5[:,:,:,:1])
    count = Flatten()(conv_5[:,:,:,1:2])
    prob = Flatten()(conv_5[:,:,:,2:])
    
    cat      = Dense(128)(cat)
    count      = Dense(128)(count)
    prob      = Dense(128)(prob)
    
    
    cat = Dense(32*32,activation="sigmoid")(cat)
    count = Dense(32*32,activation="selu")(count)
    prob = Dense(32*32,activation="sigmoid")(prob)
    
    cat = tf.keras.layers.Reshape((32,32,1))(cat)
    count = tf.keras.layers.Reshape((32,32,1))(count)
    prob = tf.keras.layers.Reshape((32,32,1))(prob)
 
    
    input_dist= tf.concat([cat,count,prob],axis=-1,name="ConcatLayer")

    #input_dist= tf.concat([cat,count,prob],axis=-1,name="ConcatLayer")
    output_dist = tfp.layers.DistributionLambda(
        name="DistributionLayer",
        make_distribution_fn=lambda t: tfp.distributions.Independent(
        tfd.Mixture(
            cat=tfd.Categorical(tf.stack([1-tf.math.sigmoid(t[...,:1]), tf.math.sigmoid(t[...,:1])],axis=-1)),
            components=[tfd.Deterministic(loc=tf.zeros_like(t[...,:1])),
            tfp.distributions.NegativeBinomial(
            total_count=tf.math.softplus(t[..., 1:2]), 
            probs=tf.math.sigmoid(t[..., 2:]) ),])
        ,name="ZeroInflated_Binomial",reinterpreted_batch_ndims=0 ))

    output = output_dist(input_dist)
    model = Model(inputs=inputs, outputs=output)

    return model
    



def LSTM_O(input_shape,output=(32,32)):
  inputs      = Input(shape=input_shape)

  inception_1_1 = inception_v2(inputs,input_shape[-1])
  inception_1_2 = inception_v2(inception_1_1,64)
  pool_1        = MaxPooling2D((2, 2), strides=(2, 2))(inception_1_1)
  inception_1_2 = inception_v2(pool_1,5)


  layer_1 = lstmLayer(pool_1,filters = [15,15,10,5],activation="relu",padding="valid")


  pool_2        = AveragePooling2D((2, 2), strides=(2, 2))(inception_1_2)
  inception_2_1 = inception_v2(pool_2,5)
  layer_2 = lstmLayer(inception_2_1,filters = [15,15,10,5],activation="relu",padding="valid")

  layer_2 = Conv2DTranspose(32,(7, 7),padding="valid")(layer_2)
  layer_2 = Conv2DTranspose(32,(7, 7),padding="valid")(layer_2)
  layer_2 = Conv2DTranspose(64,(5, 5),padding="valid")(layer_2)


  inception_3_1 = inception_v2(layer_2,16)
  inception_3_2 = inception_v2(layer_1,16)




  layer= tf.concat([inception_3_2,inception_3_1,layer_1,layer_2],axis=-1,name="ConcatLayer")
  cat = inception_v2(layer,2)
  prob = inception_v2(layer,2)
  count = inception_v2(layer,2)
  

  prob  = Flatten()(prob)
  cat   = Flatten()(cat)
  count = Flatten()(count)

  cat      = Dense(256)(cat)
  count    = Dense(256)(count)
  prob     = Dense(256)(prob)
    
  cat = Dense(output[0]*output[1],activation="tanh")(cat)
  count = Dense(output[0]*output[1],activation="selu")(count)
  prob = Dense(output[0]*output[1],activation="tanh")(prob)
  
  cat = tf.keras.layers.Reshape((*output,1))(cat)
  count = tf.keras.layers.Reshape((*output,1))(count)
  prob = tf.keras.layers.Reshape((*output,1))(prob)
 
    
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


def CNN_LSTM_categorical(input_shape):
    inputs      = Input(shape=input_shape)
    inception_1 = inception_v2(inputs,input_shape[-1])
    lstm_conv1 = lstmLayer(inception_1,filters = [5,5,5])

    inception_2 = inception_v2(inception_1,16)
    inception_3 = inception_v2(inception_2,16)
    inception_4 = inception_v2(inception_3,16)
    #lstm_conv2 = lstmLayer(inception_4,filters = [3,5,1])

    layer = tf.concat([lstm_conv1,inception_4],axis=-1,name="ConcatLayer")
    layer = inception_v2(layer,3)
    layer = Conv2D(3,kernel_size=(7,7))(layer)
    layer = Conv2D(1,kernel_size=(7,7))(layer)


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
