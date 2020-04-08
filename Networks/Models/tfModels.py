from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

def UNet64(input_shape,
           n_predictions=1,
           simpleclassification=None,
           flatten_output=False,
           activation_hidden="relu",
           activation_output="sigmoid"):

    print("OUT",type(n_predictions),n_predictions)
    print('input shape:', input_shape)
    
    inputs = Input(shape=input_shape)

    conv01 = Conv2D(10, kernel_size=(3, 3), padding="same")(inputs)       # 10 x 64x64
    conv01 = Activation(activation_hidden)(conv01)
    conv01_pool = MaxPooling2D((2, 2), strides=(2, 2))(conv01)            # 10 x 32x32
    print("0)", conv01_pool.shape, "10 x 32x32")

    conv02 = Conv2D(20, kernel_size=(3, 3), padding="same")(conv01_pool)  # 20 x 32x32
    conv02 = Activation(activation_hidden)(conv02)
    conv02_pool = MaxPooling2D((2, 2), strides=(2, 2))(conv02)            # 20 x 16x16
    print("1)", conv02_pool.shape, "20 x 16x16")

    conv03 = Conv2D(20, kernel_size=(3, 3), padding="same")(conv02_pool)  # 20 x 16x16
    conv03 = Activation(activation_hidden)(conv03)
    conv03_pool = MaxPooling2D((2, 2), strides=(2, 2))(conv03)            # 20 x 8x8
    print("2)", conv03_pool.shape, "20 x 8x8")

    conv04 = Conv2D(20, kernel_size=(3, 3), padding="same")(conv03_pool)  # 20 x 8x8
    conv04 = Activation(activation_hidden)(conv04)
    conv04_pool = MaxPooling2D((2, 2), strides=(2, 2))(conv04)            # 20 x 4x4
    print("3)", conv04_pool.shape, "20 x 4x4")

    ### UPSAMPLING:
    up04 = UpSampling2D((2, 2))(conv04_pool)    # 20 x 8x8
    up04 = concatenate([conv04, up04], axis=3)  # 20+20 x 8x8
    print("4)", up04.shape, "40 x 8x8")

    up03 = UpSampling2D((2, 2))(up04)           # 40 x 16x16
    up03 = concatenate([conv03, up03], axis=3)  # 20+40 x 16x16
    print("5)", up03.shape, "60 x 16x16")

    up02 = UpSampling2D((2, 2))(up03)           # 60 x 32x32
    up02 = concatenate([conv02, up02], axis=3)  # 20+60 x 32x32
    print("6)", up02.shape, "80 x 32x32")

    up01 = UpSampling2D((2, 2))(up02)           # 80 x 64x64
    up01 = concatenate([conv01, up01], axis=3)  # 10+80 x 64x64
    print("7)", up01.shape, "90 x 64x64")

    output = Conv2D(n_predictions, (1, 1), activation=activation_output)(up01)  # 1 x 64x64
    print("8)", output.shape, "{} x 64x64".format(n_predictions))
    if flatten_output:
        output = Flatten()(output)
        print("output flattened to {}".format(output.shape))
        if simpleclassification is not None:
            output = Dense(simpleclassification, activation='softmax')(output)
            print("9)", output.shape, "zur Klassifikation von {} Klassen (mit softmax)".format(simpleclassification))

    model = Model(inputs=inputs, outputs=output)
    return model


def UNet64_Bernoulli(input_shape,
           n_predictions=1,
           simpleclassification=None,
           flatten_output=False,
           activation_hidden="relu",
           activation_output="relu"):

    def zero_inf(out): 
        #print(out.shape)
        #tfd = tfp.distributions
        #rate = tf.squeeze(Flatten()(tf.math.exp(out[:,:,0]))) #A 
        #s = tf.math.sigmoid(Flatten()(out[:,:,2])) #B  
        #probs = tf.concat([1-s, s], axis=0) #C 
        #print(rate)
        #print(probs)
        rate = out[:,:,:,2]
        probs = out[:,:,:,1]
        return tfd.Mixture(
              cat=tfd.Categorical(probs=probs),#D
              components=[
              tfd.Deterministic(loc=tf.zeros_like(rate)), #E
              tfd.Poisson(rate=rate), #F 
            ])

    print("OUT",type(n_predictions),n_predictions)

    
    inputs = Input(shape=input_shape)

    conv01 = Conv2D(10, kernel_size=(3, 3), padding="same")(inputs)       # 10 x 64x64
    conv01 = Activation(activation_hidden)(conv01)
    conv01_pool = MaxPooling2D((2, 2), strides=(2, 2))(conv01)            # 10 x 32x32
    print("0)", conv01_pool.shape, "10 x 32x32")

    conv02 = Conv2D(20, kernel_size=(3, 3), padding="same")(conv01_pool)  # 20 x 32x32
    conv02 = Activation(activation_hidden)(conv02)
    conv02_pool = MaxPooling2D((2, 2), strides=(2, 2))(conv02)            # 20 x 16x16
    print("1)", conv02_pool.shape, "20 x 16x16")

    conv03 = Conv2D(20, kernel_size=(3, 3), padding="same")(conv02_pool)  # 20 x 16x16
    conv03 = Activation(activation_hidden)(conv03)
    conv03_pool = MaxPooling2D((2, 2), strides=(2, 2))(conv03)            # 20 x 8x8
    print("2)", conv03_pool.shape, "20 x 8x8")

    conv04 = Conv2D(20, kernel_size=(3, 3), padding="same")(conv03_pool)  # 20 x 8x8
    conv04 = Activation(activation_hidden)(conv04)
    conv04_pool = MaxPooling2D((2, 2), strides=(2, 2))(conv04)            # 20 x 4x4
    print("3)", conv04_pool.shape, "20 x 4x4")

    ### UPSAMPLING:
    up04 = UpSampling2D((2, 2))(conv04_pool)    # 20 x 8x8
    up04 = concatenate([conv04, up04], axis=3)  # 20+20 x 8x8
    print("4)", up04.shape, "40 x 8x8")

    up03 = UpSampling2D((2, 2))(up04)           # 40 x 16x16
    up03 = concatenate([conv03, up03], axis=3)  # 20+40 x 16x16
    print("5)", up03.shape, "60 x 16x16")

    up02 = UpSampling2D((2, 2))(up03)           # 60 x 32x32
    up02 = concatenate([conv02, up02], axis=3)  # 20+60 x 32x32
    print("6)", up02.shape, "80 x 32x32")

    up01 = UpSampling2D((2, 2))(up02)           # 80 x 64x64
    up01 = concatenate([conv01, up01], axis=3)  # 10+80 x 64x64
    print("7)", up01.shape, "90 x 64x64")

    output = Conv2D(1, (1, 1), activation=None)(up01)  # 1 x 64x64
    output = Flatten()(output)
    
    output = tfp.layers.IndependentBernoulli((input_shape[0],input_shape[1],n_predictions), \
                                            tfp.distributions.Bernoulli.logits)(output)

    #output = tfp.layers.DistributionLambda(zero_inf)(output)

    model = Model(inputs=inputs, outputs=output)
    return model

def UNet64_zeroInflatedPoisson(input_shape,
           n_predictions=1,
           simpleclassification=None,
           flatten_output=False,
           activation_hidden="relu",
           activation_output="relu"):


    def zeroInflatedPoisson(output):
        rate = tf.math.exp(output[0,:,:,0]) #A 
        s = tf.math.sigmoid(output[0,:,:,1])
        components = [tfd.Deterministic(loc=tf.zeros_like(rate)), #E
         tfd.Poisson(rate=rate) #F 
         ]
        mixture = tfd.Mixture(
              cat=tfd.Categorical(probs=tf.stack([1-s, s],axis=-1)),#D
              components=components)
        return mixture

    
    inputs = Input(shape=input_shape)

    conv01 = Conv2D(10, kernel_size=(3, 3), padding="same")(inputs)       # 10 x 64x64
    conv01 = Activation(activation_hidden)(conv01)
    conv01_pool = MaxPooling2D((2, 2), strides=(2, 2))(conv01)            # 10 x 32x32
    print("0)", conv01_pool.shape, "10 x 32x32")

    conv02 = Conv2D(20, kernel_size=(3, 3), padding="same")(conv01_pool)  # 20 x 32x32
    conv02 = Activation(activation_hidden)(conv02)
    conv02_pool = MaxPooling2D((2, 2), strides=(2, 2))(conv02)            # 20 x 16x16
    print("1)", conv02_pool.shape, "20 x 16x16")

    conv03 = Conv2D(20, kernel_size=(3, 3), padding="same")(conv02_pool)  # 20 x 16x16
    conv03 = Activation(activation_hidden)(conv03)
    conv03_pool = MaxPooling2D((2, 2), strides=(2, 2))(conv03)            # 20 x 8x8
    print("2)", conv03_pool.shape, "20 x 8x8")

    conv04 = Conv2D(20, kernel_size=(3, 3), padding="same")(conv03_pool)  # 20 x 8x8
    conv04 = Activation(activation_hidden)(conv04)
    conv04_pool = MaxPooling2D((2, 2), strides=(2, 2))(conv04)            # 20 x 4x4
    print("3)", conv04_pool.shape, "20 x 4x4")

    ### UPSAMPLING:
    up04 = UpSampling2D((2, 2))(conv04_pool)    # 20 x 8x8
    up04 = concatenate([conv04, up04], axis=3)  # 20+20 x 8x8
    print("4)", up04.shape, "40 x 8x8")

    up03 = UpSampling2D((2, 2))(up04)           # 40 x 16x16
    up03 = concatenate([conv03, up03], axis=3)  # 20+40 x 16x16
    print("5)", up03.shape, "60 x 16x16")

    up02 = UpSampling2D((2, 2))(up03)           # 60 x 32x32
    up02 = concatenate([conv02, up02], axis=3)  # 20+60 x 32x32
    print("6)", up02.shape, "80 x 32x32")

    up01 = UpSampling2D((2, 2))(up02)           # 80 x 64x64
    up01 = concatenate([conv01, up01], axis=3)  # 10+80 x 64x64
    print("7)", up01.shape, "90 x 64x64")

    output = Conv2D(2, (1, 1), activation=tf.exp)(up01)  # 1 x 64x64
    #output = Flatten()(output)
    
    #output = tfp.layers.IndependentPoisson(1)(output)
    output = tfp.layers.DistributionLambda(zeroInflatedPoisson)(output)
    #output = tfp.layers.IndependentPoisson(1)(output)

    model = Model(inputs=inputs, outputs=output)
    return model