from keras.models import *
from keras.layers import *


def UNet64(input_shape,
           n_predictions=1,
           simpleclassification=None,
           flatten_output=False,
           activation_hidden="relu",
           activation_output="relu"):

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
