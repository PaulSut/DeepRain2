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
# model = Unet(Adam(lr=1e-6), loss='mse', dimension=dimension, channels=channels)
model = load_model(MODELPATH)

for x, y in test:
    prediction = model.predict(x, batch_size=batch_size)
    prediction = 255 * prediction
    inp = x * 255
    print("SHAPE", x.shape)
    label = y

    print(inp.max(), inp.min())
    break

for i, img in enumerate(prediction):

    frame = None
    for j in range(channels):
        x_img = inp[i, :, :, j]

        if frame is None:
            frame = x_img
            continue
        frame = np.concatenate((frame, x_img), axis=1)

    print(img.max(), img.min(), inp[i, :, :, :].max())
    y_ = np.concatenate((img[:, :, 0], label[i, :, :, 0]), axis=1)

    frame = np.concatenate((frame, y_), axis=1)
    indizes = np.where(frame > 0)
    frame[indizes] = 255
    while True:
        cv2.imshow("windowname", frame.astype(np.uint8))
        if cv2.waitKey(25) & 0XFF == ord('q'):
            break
