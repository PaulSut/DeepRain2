#!/home/simon/anaconda3/envs/tensorflow-gpu/bin/python
from Utils.Data import Dataset, dataWrapper
from Models.Unet import unet
from keras.optimizers import *
from keras.models import load_model
from Models.tfModels import UNet64
from Utils.loss import SSIM
from Models.CnnLSTM import CnnLSTM
import os
import cv2

print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

#PATHTOMODEL = "model_data/UNet64_SSIM"
#MODELNAME = "UNet64_SSIM.h5"


PATHTOMODEL = "model_data/CnnLSTM_SSIM"
MODELNAME = "CnnLSTM_SSIM.h5"
MODELPATH = os.path.join(PATHTOMODEL, MODELNAME)


def Unet(optimizer, loss='mse', metrics=['accuracy'], dimension=(256, 256), channels=5):
    model = unet(input_size=(*dimension, channels))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary()

    return model


pathToData = "/home/simon/gitprojects/DeepRain2/opticFlow/PNG_NEW/MonthPNGData/YW2017.002_200801"

# we should keep ratio of original images
# dimension should be multiple divisible by two (4 times)

#dimension = (272, 224)
epochs = 15
channels = 5
dimension = (128,112)
batch_size = 30
flatten = False

train,test = dataWrapper(pathToData,dimension = dimension,channels = channels,batch_size = batch_size,flatten=flatten,shuffle=False)

#model = UNet64((*dimension,channels))
model = CnnLSTM((*dimension,channels))
model.summary()

model.load_weights(MODELPATH, by_name=False)
for x, y in train:
    prediction = model.predict(x, batch_size=batch_size)
    #prediction = 255 * prediction
    inp = x * 255
    bs,t,col,row,ch = inp.shape
    inp = inp.reshape(bs,col,row,t)
    print("SHAPE", inp.shape)
    label = y
    print(label.shape)

    #print(inp.max(), inp.min())
    

    for i, img in enumerate(prediction):

        frame = None
        for j in range(channels):
            x_img = inp[i, :, :, j]
            if frame is None:
                frame = x_img
                continue
            frame = np.concatenate((frame, x_img), axis=1)

        print(prediction.max(),img.max(), img.min(), inp[i, :, :, :].max(),"\t",frame.shape)
        print(label[i, :, :,0].shape)
        print(img[:, :, 0].shape)
        y_ = np.concatenate((img[:, :, 0], label[i, :, :,0]), axis=1)
        print(frame.shape,y_.shape)
        frame = np.concatenate((frame, y_), axis=1)
        indizes = np.where(frame > 0)
        #frame[indizes] = 255
        #while True:
        cv2.imshow("windowname", frame.astype(np.uint8))
        if cv2.waitKey(25) & 0XFF == ord('q'):
                break

                
