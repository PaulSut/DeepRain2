#!/home/simon/anaconda3/envs/tensorflow-gpu/bin/python
from Utils.Data import Dataset, dataWrapper, getListOfFiles
from Models.Unet import *
from trainer import Trainer
from keras.optimizers import *
from keras.models import load_model
from Models.tfModels import *
from Utils.loss import SSIM, NLL
from Models.CnnLSTM import *
import pandas as pd
import Models
import os
import cv2
import re

print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)


networks = {"CnnLSTM":CnnLSTM,
            "CnnLSTM2":CnnLSTM2,
            "CnnLSTM3":CnnLSTM3,
            "LSTM_Meets_Unet_Upconv":LSTM_Meets_Unet_Upconv,
            "LSTM_Meets_Unet":LSTM_Meets_Unet,
            "UNet64":UNet64,
            "unet":unet,
            "UNet64_Poisson":UNet64_Poisson,
            "UNet64_Bernoulli":UNet64_Bernoulli
}

loss = {"SSIM":SSIM(),
        "mse":"mse"}


def nameToDim(filename):
    regex = r"([\d]+)x([\d]+)x([\d]+)"
    matches = re.search(regex, filename)
    dim = (int(matches[1]),int(matches[2]),int(matches[3]))
    return dim
    

class Predictor(object):
    """docstring for Predictor"""
    def __init__(self, modelPath,PathToData):
        super(Predictor, self).__init__()
        self.modelPath = modelPath
        self.PathToData = PathToData


        l = getListOfFiles(self.modelPath)
        fileList = []
        for elem in l:
            if "validation" in elem:
                continue
            fileList.append(elem)

        self.model_infos = {}

        for j,i in enumerate(range(0,len(fileList),2)):
            if '.h5' in fileList[i]:
                weights = fileList[i]
                history = fileList[i+1]
            else:
                weights = fileList[i+1]
                history = fileList[i]

            self.model_infos[j] = {'weights' : weights,
                              'history' : history,
                              'dim'     : nameToDim(fileList[i])}

        print("\tFound "+str(len(self.model_infos))+" models")
        for i,key in enumerate(self.model_infos):
            print("{})\n\tName:\t\t{}\n\tDimension:\t{}".\
                format(i,self.model_infos[key]['weights'].split("/")[-1],
                        self.model_infos[key]['dim']))


    def evaluateImage(self,pred,true,totalsize):


        pred_rain = np.argwhere(pred > 0)
        true_rain = np.argwhere(true > 0)

        TP = 0
        TN = 0
        FP = 0
        FN = 0

        for idx in pred_rain:
            if idx in true_rain:
                TP += 1
            else:
                FP += 1

        for idx in true_rain:
            if idx not in pred_rain:
                FN += 1
        TN = totalsize - TP - FP - FN

        return TP,TN,FP,FN

    def heatMap(self,pred,ytrue):
        pass


    def __predict(self,model,testdata):

        TP = 0
        TN = 0
        FP = 0
        FN = 0

        confusion = np.array([TP,TN,FP,FN])
        l = len(testdata)

        animFolder = os.path.join(self.validationFolder,"animation")
        if not os.path.exists(animFolder):
            os.mkdir(animFolder )

        print("\t\tTP\t\tTN\t\tFP\t\tFN")
        for i,(x, y) in enumerate(testdata):
            #pred = model.predict(x)
            pred = model(x)

            pred = pred.mean()

            b,x_s,y_s,ch = y.shape
            prediction,label = pred[0,:,:,0],y[0,:,:,0]
            #prediction *= 255
            #label *= 255

            con = np.concatenate((prediction,label),axis=1)


            confusion += self.evaluateImage(prediction,label,totalsize = y_s*x_s)
            filename = "{0:0>6}".format(i)+".png"
            

            con[:,y_s-1:y_s-1+1] = 255

            con = cv2.cvtColor(con.astype(np.uint8),cv2.COLOR_GRAY2RGB)

            diff = np.abs(prediction-label).astype(np.uint8)


            heatmap_img = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
            
            heatmap_img[np.where(diff == 0)] = [0,0,0]
            con = np.concatenate((con,heatmap_img),axis=1)
            con[:,(y_s*2)-1:(y_s*2)-1+1] = 255



            cv2.imwrite(os.path.join(animFolder,filename), con.astype(np.uint8))
            print("{:5}/{}\t{:.1E}\t\t{:.1E}\t\t{:.1E}\t\t{:.1E}".format(i,l,*confusion),end="\r")
            cv2.imshow("windowname", con)
            if cv2.waitKey(25) & 0XFF == ord('q'):
                break
        print("{:5}/{}\t{}\t\t{}\t\t{}\t\t{}".format(i,l,*confusion),end="\n")

        confusionFrame = pd.DataFrame(confusion.reshape((1,4)),columns=["TP","TN","FP","FN"])
        confusionFrame.to_csv(os.path.join(self.validationFolder ,"confusion.csv"),index=False)

    def predict(self):

        modules = dir(Models)

        for key in self.model_infos:
            weights = self.model_infos[key]['weights']
            history = self.model_infos[key]['history']
            dim     = self.model_infos[key]['dim']
            self.validationFolder    = os.path.join(os.path.dirname(weights),'validation')
            basename = os.path.basename(weights)


            if not os.path.exists(self.validationFolder ):
                os.mkdir(self.validationFolder )

            else:
                print(self.validationFolder+" exists.. skipping")
                continue
            try:
                nnname = os.path.dirname(weights).split("/")[-1]
                nnname = nnname.split("_")
                l = nnname[-1]
                nnname = "_".join(nnname[:len(nnname)-1])
                optimizer = Adam( lr = 1e-5 )
                
            except Exception as e:
                print("Couldn't create Network : ", )
            if l not in loss:
                loss_f = NLL
            else:
                loss_f = loss[l]
            print(loss_f)
            t = Trainer(networks[nnname],
                        pathToData=self.PathToData,
                        lossfunction=loss_f,
                        batch_size=1,
                        dimension=(dim[0],dim[1]),
                        channels=dim[2])
            test_data = t.test
            model = t.model
            history = t.history

            self.__predict(model,test_data)
            


PathToData = "/home/simon/MonthPNGData/MonthPNGData"

pred=Predictor(modelPath="./model_data",PathToData = PathToData)
pred.predict()
