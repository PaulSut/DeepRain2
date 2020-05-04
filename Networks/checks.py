#!/home/simon/anaconda3/envs/tensorflow-gpu/bin/python
from Utils.Data import Dataset, dataWrapper
import cv2
import numpy as np
pathToData = "/home/simon/gitprojects/DeepRain2/opticFlow/PNG_NEW/MonthPNGData/YW2017.002_200801"
dimension = (128, 112)
#dimension = (1100, 900)
channels  = 5
batch_size = 5
flatten = False
train, test = dataWrapper(pathToData, 
                                            dimension=dimension,
                                            channels=channels, 
                                            batch_size=batch_size, 
                                            flatten=flatten)



for i,T in enumerate(train):
    x,y = T
    print(i,len(train))
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


            x_img = np.concatenate((x_img,y[batch,:,:,0]),axis=1)
            i = np.where(x_img > 0)
            x_img[i] = 255
            cv2.imshow("windowname", x_img.astype(np.uint8))
            if cv2.waitKey(25) & 0XFF == ord('q'):
                    break

    else:
        bs,row,col,ts = x.shape
        bs,row,col,ch = y.shape
        for batch in range(bs):
            x_img = None
            for t in range(ts):
                if x_img is None:
                    x_img = x[batch,:,:,t]
                    continue
                x_img = np.concatenate((x_img,x[batch,:,:,0]),axis=1)


            x_img = np.concatenate((x_img,y[batch,:,:,0]),axis=1)
            i = np.where(x_img > 0)
            x_img[i] = 255
            cv2.imshow("windowname", x_img.astype(np.uint8))
            if cv2.waitKey(25) & 0XFF == ord('q'):
                    break

