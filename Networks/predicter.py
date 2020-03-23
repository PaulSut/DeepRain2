#!/home/simon/anaconda3/envs/tensorflow-gpu/bin/python
from Utils.Data import Dataset, dataWrapper
from Models.Unet import unet
from keras.optimizers import *
from keras.models import load_model
from Models.tfModels import UNet64
from Utils.loss import SSIM
from Models.CnnLSTM import *
import os
import cv2

print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

#PATHTOMODEL = "model_data/UNet64_SSIM"
#MODELNAME = "UNet64_SSIM.h5"


PATHTOMODEL = "model_data/CnnLSTM_mse"
PATHTOMODEL = "model_data/LSTM_Meets_Unet_mse"
MODELNAME = "CnnLSTM_mse.h5"
MODELNAME = "LSTM_Meets_Unet_mse272x224x5.h5"
MODELPATH = os.path.join(PATHTOMODEL, MODELNAME)


def Unet(optimizer, loss='mse', metrics=['accuracy'], dimension=(256, 256), channels=5):
    model = unet(input_size=(*dimension, channels))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary()

    return model


pathToData = "/home/simon/gitprojects/DeepRain2/opticFlow/PNG_NEW/MonthPNGData/YW2017.002_200801"

# we should keep ratio of original images
# dimension should be multiple divisible by two (4 times)

dimension = (256, 192)
epochs = 15
channels = 5
#dimension = (128,112)
batch_size = 10
flatten = False

train,test = dataWrapper(pathToData,dimension = dimension,channels = channels,batch_size = batch_size,flatten=flatten,shuffle=False)

#model = UNet64((*dimension,channels))
model = LSTM_Meets_Unet((*dimension,channels))
model.summary()

model.load_weights(MODELPATH, by_name=False)
for x, y in test:
    print(x.shape,y.shape)

    prediction = model.predict(x,batch_size=batch_size)
    #prediction *= 255
    if len(x.shape) == 5:
        bs,ts,row,col,ch = x.shape
        bs,row,col,ch = y.shape

        for batch in range(bs):
            x_img = None
            for t in range(ts):
                if x_img is None:
                    x_img = x[batch,t,:,:,0]
                    continue
                x_img = np.concatenate((x_img,x[batch,t,:,:,0]),axis=1)


            print(prediction[batch,:,:,0].min(),y[batch,:,:,0].min(),prediction[batch,:,:,0].max(),y[batch,:,:,0].max())
            x_img = np.concatenate((x_img,prediction[batch,:,:,0]),axis=1)
            x_img = np.concatenate((x_img,y[batch,:,:,0]),axis=1)

            i = np.where(x_img > 0)
            #x_img[i] = 255
            cv2.imshow("windowname", x_img.astype(np.uint8))
            if cv2.waitKey(25) & 0XFF == ord('q'):
                break

    else:
        bs,row,col,ts = x.shape
        bs,row,col,ch = y.shape

        for batch in range(bs):
            x_img = None
            for t in range(ts):
                #print(x[batch,:,:,t].mean(),end= "\t")
                if x_img is None:
                    x_img = x[batch,:,:,t]
                    continue
                x_img = np.concatenate((x_img,x[batch,:,:,t]),axis=1)

            print()
            print(prediction[batch,:,:,0].min(),y[batch,:,:,0].min(),prediction[batch,:,:,0].max(),y[batch,:,:,0].max())
            x_img = np.concatenate((x_img,prediction[batch,:,:,0]),axis=1)
            x_img = np.concatenate((x_img,y[batch,:,:,0]),axis=1)
            x_img *= 255
            #i = np.where(x_img > 0)
            #x_img[i] = 255
            cv2.imshow("windowname", x_img.astype(np.uint8))
            if cv2.waitKey(25) & 0XFF == ord('q'):
                break

