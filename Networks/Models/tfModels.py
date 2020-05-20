from tensorflow.keras import Sequential, Model
from keras.backend import int_shape
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


# n_predictions = nuber of classe fpr sparse cat entropy
def UNet64(input_shape,
           n_predictions=4,
           simpleclassification=None,
           flatten_output=False,
           activation_hidden="relu",
           activation_output="sigmoid"):
    print("OUT", type(n_predictions), n_predictions)
    print('input shape:', input_shape)

    inputs = Input(shape=input_shape)

    conv01 = Conv2D(10, kernel_size=(3, 3), padding="same")(inputs)  # 10 x 64x64
    conv01 = Activation(activation_hidden)(conv01)
    conv01_pool = MaxPooling2D((2, 2), strides=(2, 2))(conv01)  # 10 x 32x32
    print("0)", conv01_pool.shape, "10 x 32x32")

    conv02 = Conv2D(20, kernel_size=(3, 3), padding="same")(conv01_pool)  # 20 x 32x32
    conv02 = Activation(activation_hidden)(conv02)
    conv02_pool = MaxPooling2D((2, 2), strides=(2, 2))(conv02)  # 20 x 16x16
    print("1)", conv02_pool.shape, "20 x 16x16")

    conv03 = Conv2D(20, kernel_size=(3, 3), padding="same")(conv02_pool)  # 20 x 16x16
    conv03 = Activation(activation_hidden)(conv03)
    conv03_pool = MaxPooling2D((2, 2), strides=(2, 2))(conv03)  # 20 x 8x8
    print("2)", conv03_pool.shape, "20 x 8x8")

    conv04 = Conv2D(20, kernel_size=(3, 3), padding="same")(conv03_pool)  # 20 x 8x8
    conv04 = Activation(activation_hidden)(conv04)
    conv04_pool = MaxPooling2D((2, 2), strides=(2, 2))(conv04)  # 20 x 4x4
    print("3)", conv04_pool.shape, "20 x 4x4")

    ### UPSAMPLING:
    up04 = UpSampling2D((2, 2))(conv04_pool)  # 20 x 8x8
    up04 = concatenate([conv04, up04], axis=3)  # 20+20 x 8x8
    print("4)", up04.shape, "40 x 8x8")

    up03 = UpSampling2D((2, 2))(up04)  # 40 x 16x16
    up03 = concatenate([conv03, up03], axis=3)  # 20+40 x 16x16
    print("5)", up03.shape, "60 x 16x16")

    up02 = UpSampling2D((2, 2))(up03)  # 60 x 32x32
    up02 = concatenate([conv02, up02], axis=3)  # 20+60 x 32x3
    print("6)", up02.shape, "80 x 32x32")

    up01 = UpSampling2D((2, 2))(up02)  # 80 x 64x64
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


def medium_UNet64(input_shape,
                  n_predictions=4,
                  simpleclassification=None,
                  flatten_output=False,
                  activation_hidden="relu",
                  activation_output="sigmoid"):
    start_neurons = 5
    print("OUT", type(n_predictions), n_predictions)
    print('input shape:', input_shape)

    inputs = Input(shape=input_shape)

    conv01 = Conv2D(start_neurons * 1, kernel_size=(3, 3), activation=activation_hidden, padding="same")(inputs)
    conv01 = Conv2D(start_neurons * 1, kernel_size=(3, 3), activation=activation_hidden, padding="same")(conv01)
    pool1 = MaxPooling2D((2, 2))(conv01)
    pool1 = Dropout(0.25)(pool1)
    print("0)", pool1.shape)

    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=activation_hidden, padding="same")(pool1)
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=activation_hidden, padding="same")(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(0.5)(pool2)
    print("1)", pool2.shape)

    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=activation_hidden, padding="same")(pool2)
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=activation_hidden, padding="same")(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(0.5)(pool3)
    print("2)", pool3.shape)

    conv4 = Conv2D(start_neurons * 8, (3, 3), activation=activation_hidden, padding="same")(pool3)
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation=activation_hidden, padding="same")(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)
    print("3)", pool4.shape)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation=activation_hidden, padding="same")(pool4)
    convm = Conv2D(start_neurons * 16, (3, 3), activation=activation_hidden, padding="same")(convm)
    print("4)", convm.shape)

    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(0.5)(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=activation_hidden, padding="same")(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=activation_hidden, padding="same")(uconv4)
    print("5)", uconv4.shape)

    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(0.5)(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=activation_hidden, padding="same")(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=activation_hidden, padding="same")(uconv3)
    print("6)", uconv3.shape)

    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(0.5)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=activation_hidden, padding="same")(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=activation_hidden, padding="same")(uconv2)
    print("7)", uconv2.shape)

    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv01])
    uconv1 = Dropout(0.5)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=activation_hidden, padding="same")(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=activation_hidden, padding="same")(uconv1)
    print("8)", uconv1.shape)

    output = Conv2D(n_predictions, (1, 1), padding="same", activation=activation_output)(uconv1)
    print("9)", output.shape)

    model = Model(inputs=inputs, outputs=output)
    return model

def lstm_many_to_one(input_shape,
                       n_predictions=1,
                       simpleclassification=None,
                       flatten_output=False,
                       activation_hidden="relu",
                       activation_output="sigmoid"):

    #start_neurons = 30

    print("OUT", type(n_predictions), n_predictions)
    print('input shape:', input_shape)


    inputs = Input(shape=input_shape + (1,))
    inputs = Input((input_shape[2], input_shape[0], input_shape[1], 1))

    print('inputs:', inputs.shape)
    #bidirectional_lstm = Bidirectional(ConvLSTM2D(64,  kernel_size=(3, 3), activation=activation_hidden, data_format= 'channels_last', padding='same', return_sequences=True), input_shape=input_shape)(inputs)
    bidirectional_lstm = ConvLSTM2D(64,  kernel_size=(3, 3), activation=activation_hidden, data_format= 'channels_last', padding='same', return_sequences=True)(inputs)
    batch1 = BatchNormalization()(bidirectional_lstm)
    output = Conv3D(n_predictions, (1, 1, 1), padding="same", activation=activation_output)(batch1)
    print('out',output.shape)
    #dense = Dense(input_shape[0]*input_shape[1])(bidirectional_lstm)
    output = tf.keras.layers.Reshape((input_shape[0], input_shape[1]))(output)

    model = Model(inputs=inputs, outputs=output)

    return model

    inputs = Input((input_shape[2], input_shape[0], input_shape[1], 1))

    cnnlstm1 = ConvLSTM2D(filters=64, kernel_size=(9, 9), activation='relu',
                          padding='same', return_sequences=True, data_format='channels_last')(inputs)

    batch1 = BatchNormalization()(cnnlstm1)

    cnnlstm2 = ConvLSTM2D(filters=32, kernel_size=(5, 5), activation='relu',
                          padding='same', return_sequences=True, data_format='channels_last')(batch1)

    batch2 = BatchNormalization()(cnnlstm2)

    cnnlstm3 = ConvLSTM2D(filters=18, kernel_size=(3, 3), activation='relu',
                          padding='same', return_sequences=True, data_format='channels_last')(batch2)

    batch3 = BatchNormalization()(cnnlstm3)

    cnnlstm4 = ConvLSTM2D(filters=8, kernel_size=(3, 3), activation='relu',
                          padding='same', return_sequences=True, data_format='channels_last')(batch3)

    batch4 = BatchNormalization()(cnnlstm4)

    cnnlstm5 = ConvLSTM2D(filters=4, kernel_size=(3, 3), activation='relu',
                          padding='same', return_sequences=False, data_format='channels_last')(batch4)

    batch5 = BatchNormalization()(cnnlstm5)

    output = Conv2D(n_predictions, (1, 1), activation='sigmoid', data_format='channels_last')(batch5)
    model = Model(inputs=inputs, outputs=output)

    return model


def lstmLayer(inp,filters = [20,20]):

    shape_inp = int_shape(inp)


    lstm_shape = Reshape((shape_inp[-1],shape_inp[1],shape_inp[2],1))(inp)


    lstm_conv = ConvLSTM2D(filters=filters[0], kernel_size=(3, 3), activation='relu',
                       padding='same', return_sequences=True,data_format='channels_last')(lstm_shape)

def medium_thin_UNet64(input_shape,
                       n_predictions=4,
                       simpleclassification=None,
                       flatten_output=False,
                       activation_hidden="relu",
                       activation_output="sigmoid"):
    start_neurons = 30
    print("OUT", type(n_predictions), n_predictions)
    print('input shape:', input_shape)

    inputs = Input(shape=input_shape)

    #inputs = Input(shape=input_shape + (1,))
    #inputs = Input((input_shape[2], input_shape[0], input_shape[1], 1))

    print('inputs:', inputs.shape)
    conv01 = Conv2D(start_neurons, kernel_size=(3, 3), padding="same")(inputs)  # 10 x 64x64
    conv01_act = Activation(activation_hidden)(conv01)
    conv01_pool = MaxPooling2D((2, 2), strides=(2, 2))(conv01_act)  # 10 x 32x32
    print("0)", conv01_pool.shape, "10 x 32x32")

    conv02 = Conv2D(start_neurons*2, kernel_size=(3, 3), padding="same")(conv01_pool)  # 20 x 32x32
    conv02_act = Activation(activation_hidden)(conv02)
    conv02_pool = MaxPooling2D((2, 2), strides=(2, 2))(conv02_act)  # 20 x 16x16
    print("1)", conv02_pool.shape, "20 x 16x16")

    conv03 = Conv2D(start_neurons*2, kernel_size=(3, 3), padding="same")(conv02_pool)  # 20 x 16x16
    conv03_act = Activation(activation_hidden)(conv03)
    conv03_pool = MaxPooling2D((2, 2), strides=(2, 2))(conv03_act)  # 20 x 8x8
    print("2)", conv03_pool.shape, "20 x 8x8")

    conv04 = Conv2D(start_neurons*2, kernel_size=(3, 3), padding="same")(conv03_pool)  # 20 x 8x8
    conv04_act = Activation(activation_hidden)(conv04)
    conv04_pool = MaxPooling2D((2, 2), strides=(2, 2))(conv04_act)  # 20 x 4x4
    print("3)", conv04_pool.shape, "20 x 4x4")

    #lstm_layer = lstmLayer(conv04_pool,filters = [20,20,20])
    #lstm_test = BatchNormalization()(lstm_layer)
    #conv04 = concatenate([conv04, lstm_test], axis=3)
    #dense01 = Dense(conv04_pool.shape[-1], activation= activation_hidden)(bidirectional_lstm)
    #dense01_reshape = tf.reshape(dense01, conv04_pool.shape)
    #conv01 = Conv2D(start_neurons, kernel_size=(3, 3), padding="same")(dense01_reshape)  # 10 x 64x64

    ### UPSAMPLING:
    up04 = UpSampling2D((2, 2))(conv04_pool)  # 20 x 8x8
    up04_con = concatenate([conv04, up04], axis=3)  # 20+20 x 8x8
    print("4)", up04.shape, "40 x 8x8")

    up03 = UpSampling2D((2, 2))(up04_con)  # 40 x 16x16
    up03_con = concatenate([conv03, up03], axis=3)  # 20+40 x 16x16
    print("5)", up03.shape, "60 x 16x16")

    up02 = UpSampling2D((2, 2))(up03_con)  # 60 x 32x32
    up02_con = concatenate([conv02, up02], axis=3)  # 20+60 x 32x32
    print("6)", up02.shape, "80 x 32x32")

    up01 = UpSampling2D((2, 2))(up02_con)  # 80 x 64x64
    up01_con = concatenate([conv01, up01], axis=3)  # 10+80 x 64x64
    print("7)", up01.shape, "90 x 64x64")

    output = Conv2D(n_predictions, (1, 1), activation=activation_output)(up01_con)  # 1 x 64x64
    #output = Dense(tfp.layers.IndependentBernoulli.params_size(output))  # 1 x 64x64
    #output = Flatten()(output)
    #output = tfp.layers.IndependentBernoulli((input_shape[0], input_shape[1], n_predictions),
    #                                         tfp.distributions.Bernoulli.logits)(output)
    #output = tfp.layers.IndependentBernoulli(output)

    print("8)", output.shape, "{} x 64x64".format(n_predictions))
    if flatten_output:
        output = Flatten()(output)
        print("output flattened to {}".format(output.shape))
        if simpleclassification is not None:
            output = Dense(simpleclassification, activation='softmax')(output)
            print("9)", output.shape, "zur Klassifikation von {} Klassen (mit softmax)".format(simpleclassification))


    model = Model(inputs=inputs, outputs=output)

    return model



def UNet643D(input_shape,
             n_predictions=4,
             simpleclassification=None,
             flatten_output=False,
             activation_hidden="relu",
             activation_output="sigmoid"):
    print("OUT", type(n_predictions), n_predictions)
    print('input shape:', input_shape)

    inputs = Input(shape=input_shape)

    conv01 = Conv3D(10, kernel_size=[3, 3, 3], padding="same")(inputs)  # 10 x 64x64
    conv01 = Activation(activation_hidden)(conv01)
    conv01_pool = MaxPooling3D([2, 2, 2], strides=[2, 2, 1])(conv01)  # 10 x 32x32
    print("0)", conv01_pool.shape, "10 x 32x32")

    conv02 = Conv3D(20, kernel_size=[3, 3, 3], padding="same")(conv01_pool)  # 20 x 32x32
    conv02 = Activation(activation_hidden)(conv02)
    conv02_pool = MaxPooling3D([2, 2, 2], strides=[2, 2, 1])(conv02)  # 20 x 16x16
    print("1)", conv02_pool.shape, "20 x 16x16")

    conv03 = Conv3D(20, kernel_size=[3, 3, 3], padding="same")(conv02_pool)  # 20 x 16x16
    conv03 = Activation(activation_hidden)(conv03)
    conv03_pool = MaxPooling3D([2, 2, 2], strides=[2, 2, 1])(conv03)  # 20 x 8x8
    conv03_pool = tf.reshape(conv03_pool, (tf.shape(conv03_pool)[0], 34, 28, 20))
    # conv03_pool = tf.transpose(conv03_pool, (0, 1, 2, 4, 3))

    print("2)", conv03_pool.shape, "20 x 8x8")

    conv04 = Conv2D(20, kernel_size=(3, 3), padding="same")(conv03_pool)  # 20 x 8x8
    conv04 = Activation(activation_hidden)(conv04)
    conv04_pool = MaxPooling2D((2, 2), strides=(2, 2))(conv04)  # 20 x 4x4
    print("3)", conv04_pool.shape, "20 x 4x4")

    ### UPSAMPLING:
    up04 = UpSampling2D((2, 2))(conv04_pool)  # 20 x 8x8
    up04 = concatenate([conv04, up04], axis=3)  # 20+20 x 8x8
    print("4)", up04.shape, "40 x 8x8")

    up03 = UpSampling2D((2, 2))(up04)  # 40 x 16x16
    up03 = tf.reshape(up03, (tf.shape(up03)[0], 68, 56, 2, 20))
    up03 = concatenate([conv03, up03], axis=4)  # 20+40 x 16x16
    print("5)", up03.shape, "60 x 16x16")

    up02 = UpSampling3D([2, 2, 2])(up03)  # 60 x 32x32
    up02 = tf.reshape(up02, (tf.shape(up02)[0], 136, 112, 3, 20))
    up02 = concatenate([conv02, up02], axis=4)  # 20+60 x 32x32
    print("6)", up02.shape, "80 x 32x32")

    up01 = UpSampling3D([2, 2, 2])(up02)  # 80 x 64x64
    up01 = tf.reshape(up01, (tf.shape(up01)[0], 272, 224, 4, 10))
    up01 = concatenate([conv01, up01], axis=4)  # 10+80 x 64x64
    print("7)", up01.shape, "90 x 64x64")

    output = Conv3D(n_predictions, (1, 1, 1), activation=activation_output)(up01)  # 1 x 64x64
    print("8)", output.shape, "{} x 64x64".format(n_predictions))
    if flatten_output:
        output = Flatten()(output)
        print("output flattened to {}".format(output.shape))
        if simpleclassification is not None:
            output = Dense(simpleclassification, activation='softmax')(output)
            print("9)", output.shape, "zur Klassifikation von {} Klassen (mit softmax)".format(simpleclassification))

    model = Model(inputs=inputs, outputs=output)
    return model


'''
    conv04 = Conv3D(20, kernel_size=[3, 3, 3], padding="same")(conv03_pool)  # 20 x 8x8
    conv04 = Activation(activation_hidden)(conv04)
    conv04_pool = MaxPooling3D((2, 2), strides=(2, 2))(conv04)  # 20 x 4x4
    print("3)", conv04_pool.shape, "20 x 4x4")

    ### UPSAMPLING:
    up04 = UpSampling3D([2, 2, 2])(conv04_pool)  # 20 x 8x8
    up04 = concatenate([conv04, up04], axis=3)  # 20+20 x 8x8
    print("4)", up04.shape, "40 x 8x8")

    up03 = UpSampling3D([2, 2, 2])(up04)  # 40 x 16x16
    up03 = concatenate([conv03, up03], axis=3)  # 20+40 x 16x16
    print("5)", up03.shape, "60 x 16x16")

    up02 = UpSampling3D([2, 2, 2])(up03)  # 60 x 32x32
    up02 = concatenate([conv02, up02], axis=3)  # 20+60 x 32x32
    print("6)", up02.shape, "80 x 32x32")

    up01 = UpSampling3D([2, 2, 2])(up02)  # 80 x 64x64
    up01 = concatenate([conv01, up01], axis=3)  # 10+80 x 64x64
    print("7)", up01.shape, "90 x 64x64")

    output = Conv3D(n_predictions, [1, 1, 1], activation=activation_output)(up01)  # 1 x 64x64
    print("8)", output.shape, "{} x 64x64".format(n_predictions))
    if flatten_output:
        output = Flatten()(output)
        print("output flattened to {}".format(output.shape))
        if simpleclassification is not None:
            output = Dense(simpleclassification, activation='softmax')(output)
            print("9)", output.shape, "zur Klassifikation von {} Klassen (mit softmax)".format(simpleclassification))

    model = Model(inputs=inputs, outputs=output)
    return model
    '''


def UNet64_Bernoulli(input_shape,
                     n_predictions=1,
                     simpleclassification=None,
                     flatten_output=False,
                     activation_hidden="relu",
                     activation_output="relu"):
    def zero_inf(out):
        # print(out.shape)
        # tfd = tfp.distributions
        # rate = tf.squeeze(Flatten()(tf.math.exp(out[:,:,0]))) #A
        # s = tf.math.sigmoid(Flatten()(out[:,:,2])) #B
        # probs = tf.concat([1-s, s], axis=0) #C
        # print(rate)
        # print(probs)
        rate = out[:, :, :, 2]
        probs = out[:, :, :, 1]
        return tfd.Mixture(
            cat=tfd.Categorical(probs=probs),  # D
            components=[
                tfd.Deterministic(loc=tf.zeros_like(rate)),  # E
                tfd.Poisson(rate=rate),  # F
            ])

    print("OUT", type(n_predictions), n_predictions)

    inputs = Input(shape=input_shape)

    conv01 = Conv2D(10, kernel_size=(3, 3), padding="same")(inputs)  # 10 x 64x64
    conv01 = Activation(activation_hidden)(conv01)
    conv01_pool = MaxPooling2D((2, 2), strides=(2, 2))(conv01)  # 10 x 32x32
    print("0)", conv01_pool.shape, "10 x 32x32")

    conv02 = Conv2D(20, kernel_size=(3, 3), padding="same")(conv01_pool)  # 20 x 32x32
    conv02 = Activation(activation_hidden)(conv02)
    conv02_pool = MaxPooling2D((2, 2), strides=(2, 2))(conv02)  # 20 x 16x16
    print("1)", conv02_pool.shape, "20 x 16x16")

    conv03 = Conv2D(20, kernel_size=(3, 3), padding="same")(conv02_pool)  # 20 x 16x16
    conv03 = Activation(activation_hidden)(conv03)
    conv03_pool = MaxPooling2D((2, 2), strides=(2, 2))(conv03)  # 20 x 8x8
    print("2)", conv03_pool.shape, "20 x 8x8")

    conv04 = Conv2D(20, kernel_size=(3, 3), padding="same")(conv03_pool)  # 20 x 8x8
    conv04 = Activation(activation_hidden)(conv04)
    conv04_pool = MaxPooling2D((2, 2), strides=(2, 2))(conv04)  # 20 x 4x4
    print("3)", conv04_pool.shape, "20 x 4x4")

    ### UPSAMPLING:
    up04 = UpSampling2D((2, 2))(conv04_pool)  # 20 x 8x8
    up04 = concatenate([conv04, up04], axis=3)  # 20+20 x 8x8
    print("4)", up04.shape, "40 x 8x8")

    up03 = UpSampling2D((2, 2))(up04)  # 40 x 16x16
    up03 = concatenate([conv03, up03], axis=3)  # 20+40 x 16x16
    print("5)", up03.shape, "60 x 16x16")

    up02 = UpSampling2D((2, 2))(up03)  # 60 x 32x32
    up02 = concatenate([conv02, up02], axis=3)  # 20+60 x 32x32
    print("6)", up02.shape, "80 x 32x32")

    up01 = UpSampling2D((2, 2))(up02)  # 80 x 64x64
    up01 = concatenate([conv01, up01], axis=3)  # 10+80 x 64x64
    print("7)", up01.shape, "90 x 64x64")

    output = Conv2D(1, (1, 1), activation=None)(up01)  # 1 x 64x64
    output = Flatten()(output)

    output = tfp.layers.IndependentBernoulli((input_shape[0], input_shape[1], n_predictions), \
                                             tfp.distributions.Bernoulli.logits)(output)

    # output = tfp.layers.DistributionLambda(zero_inf)(output)

    model = Model(inputs=inputs, outputs=output)
    return model


def UNet64_zeroInflatedPoisson(input_shape,
                               n_predictions=1,
                               simpleclassification=None,
                               flatten_output=False,
                               activation_hidden="relu",
                               activation_output="relu"):
    def zeroInflatedPoisson(output):
        rate = tf.math.exp(output[0, :, :, 0])  # A
        s = tf.math.sigmoid(output[0, :, :, 1])
        components = [tfd.Deterministic(loc=tf.zeros_like(rate)),  # E
                      tfd.Poisson(rate=rate)  # F
                      ]
        mixture = tfd.Mixture(
            cat=tfd.Categorical(probs=tf.stack([1 - s, s], axis=-1)),  # D
            components=components)
        return mixture

    inputs = Input(shape=input_shape)

    conv01 = Conv2D(10, kernel_size=(3, 3), padding="same")(inputs)  # 10 x 64x64
    conv01 = Activation(activation_hidden)(conv01)
    conv01_pool = MaxPooling2D((2, 2), strides=(2, 2))(conv01)  # 10 x 32x32
    print("0)", conv01_pool.shape, "10 x 32x32")

    conv02 = Conv2D(20, kernel_size=(3, 3), padding="same")(conv01_pool)  # 20 x 32x32
    conv02 = Activation(activation_hidden)(conv02)
    conv02_pool = MaxPooling2D((2, 2), strides=(2, 2))(conv02)  # 20 x 16x16
    print("1)", conv02_pool.shape, "20 x 16x16")

    conv03 = Conv2D(20, kernel_size=(3, 3), padding="same")(conv02_pool)  # 20 x 16x16
    conv03 = Activation(activation_hidden)(conv03)
    conv03_pool = MaxPooling2D((2, 2), strides=(2, 2))(conv03)  # 20 x 8x8
    print("2)", conv03_pool.shape, "20 x 8x8")

    conv04 = Conv2D(20, kernel_size=(3, 3), padding="same")(conv03_pool)  # 20 x 8x8
    conv04 = Activation(activation_hidden)(conv04)
    conv04_pool = MaxPooling2D((2, 2), strides=(2, 2))(conv04)  # 20 x 4x4
    print("3)", conv04_pool.shape, "20 x 4x4")

    ### UPSAMPLING:
    up04 = UpSampling2D((2, 2))(conv04_pool)  # 20 x 8x8
    up04 = concatenate([conv04, up04], axis=3)  # 20+20 x 8x8
    print("4)", up04.shape, "40 x 8x8")

    up03 = UpSampling2D((2, 2))(up04)  # 40 x 16x16
    up03 = concatenate([conv03, up03], axis=3)  # 20+40 x 16x16
    print("5)", up03.shape, "60 x 16x16")

    up02 = UpSampling2D((2, 2))(up03)  # 60 x 32x32
    up02 = concatenate([conv02, up02], axis=3)  # 20+60 x 32x32
    print("6)", up02.shape, "80 x 32x32")

    up01 = UpSampling2D((2, 2))(up02)  # 80 x 64x64
    up01 = concatenate([conv01, up01], axis=3)  # 10+80 x 64x64
    print("7)", up01.shape, "90 x 64x64")

    output = Conv2D(2, (1, 1), activation=tf.exp)(up01)  # 1 x 64x64
    # output = Flatten()(output)

    # output = tfp.layers.IndependentPoisson(1)(output)
    output = tfp.layers.DistributionLambda(zeroInflatedPoisson)(output)
    # output = tfp.layers.IndependentPoisson(1)(output)

    model = Model(inputs=inputs, outputs=output)
    return model
