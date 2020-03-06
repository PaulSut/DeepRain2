from Utils.Data import Dataset, dataWrapper
from Models.Unet import unet
from keras.optimizers import *
from keras.models import load_model
import os
import cv2

PATHTOMODEL = "/home/paul/Documents/DeepRain/DeepRain2/Networks/models_h5"
MODELNAME = "UNET.h5"
MODELPATH = os.path.join(PATHTOMODEL, MODELNAME)


def Unet(optimizer, loss='mse', metrics=['accuracy'], dimension=(256, 256), channels=5):
    model = unet(input_size=(*dimension, channels))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary()

    return model


pathToData = "/home/paul/Documents/DeepRain/DeepRain2/Networks/Dataset"

channels = 5
# we should keep ratio of original images
# dimension should be multiple divisible by two (4 times)

dimension = (272, 224)
batch_size = 10

train, test = dataWrapper(pathToData, dimension=dimension, channels=channels, batch_size=batch_size)
#model = Unet(Adam(lr=1e-6), loss='mse', dimension=dimension, channels=channels)
model = load_model(MODELPATH)

for x,y in test:
    prediction = model.predict(x,batch_size=batch_size)
    label = y
    break


for i,img in enumerate(prediction):

    while True:
        frame = np.concatenate((img,label[i]),axis=1)
        indizes = np.where(frame > 0)
        frame[indizes] = 255
        cv2.imshow("windowname", frame)
        if cv2.waitKey(25) & 0XFF == ord('q'):
            break


