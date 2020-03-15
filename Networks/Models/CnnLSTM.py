from keras.models import *
from keras.layers import *



def CnnLSTM(input_shape):


    inputs = Input((input_shape[2],input_shape[0],input_shape[1],1))
    
    cnnlstm1 = ConvLSTM2D(filters=40, kernel_size=(3, 3),
                       padding='same', return_sequences=True,data_format='channels_last')(inputs)

    batch1 = BatchNormalization() (cnnlstm1)

    cnnlstm2 = ConvLSTM2D(filters=40, kernel_size=(3, 3),
                       padding='same', return_sequences=True,data_format='channels_last')(batch1)

    batch2 = BatchNormalization() (cnnlstm2)   

    cnnlstm3 = ConvLSTM2D(filters=40, kernel_size=(3, 3),
                       padding='same', return_sequences=True,data_format='channels_last')(batch2)
    batch3 = BatchNormalization() (cnnlstm3)

    cnnlstm4 = ConvLSTM2D(filters=40, kernel_size=(3, 3),
                       padding='same', return_sequences=True,data_format='channels_last')(batch3)
    batch4 = BatchNormalization() (cnnlstm4)

    out = ConvLSTM2D(filters=1, kernel_size=(3, 3),
                       padding='same', return_sequences=False,data_format='channels_last')(batch4)

    output = Conv2D(1, (1, 1), activation='relu')(out) 
    #output = Conv3D(filters=1,kernel_size=(1,1,1),activation='sigmoid',padding='same')(out)
    print(output)
    model = Model(input = inputs, output = output)
    return model


    