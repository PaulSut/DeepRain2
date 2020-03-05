 
from PIL import Image
import numpy as np
import pandas as pd
import re
import keras
import os
from .transform import resizeImages


CSVFILE = "./.listOfFiles.csv"
WRKDIR = "./Data"
TRAINSETFOLDER=os.path.join(WRKDIR,"train")
VALSETFOLDER=os.path.join(WRKDIR,"val")

def dataWrapper(path,dimension,channels,batch_size,csvFile=CSVFILE,workingdir=WRKDIR,split=0.25):
    data = prepareListOfFiles(path)
    trainingsSet,validationSet = splitData(data)
    

    if not os.path.exists(TRAINSETFOLDER):
        os.mkdir(TRAINSETFOLDER)
    if not os.path.exists(VALSETFOLDER):
        os.mkdir(VALSETFOLDER)


    filename,ext = os.path.splitext(csvFile) 
    trainsetCSV = filename+"_train_"+ext
    valsetCSV = filename+"_val_"+ext


    train_dataframe = pd.DataFrame(trainingsSet,columns=["colummn"])
    train_dataframe.to_csv(os.path.join(TRAINSETFOLDER,trainsetCSV),index=False)
    val_dataframe = pd.DataFrame(validationSet,columns=["colummn"])
    val_dataframe.to_csv(os.path.join(VALSETFOLDER,valsetCSV),index=False)


    train = Dataset(TRAINSETFOLDER,
                    dim = dimension,
                    n_channels = channels,
                    batch_size = batch_size,
                    workingdir=TRAINSETFOLDER,
                    saveListOfFiles=trainsetCSV)

    val = Dataset(VALSETFOLDER,
                    dim = dimension,
                    n_channels = channels,
                    batch_size = batch_size,
                    workingdir=VALSETFOLDER,
                    saveListOfFiles=valsetCSV)


    return train,val

def splitData(data,split=0.25):
    
    dataLength = len(data)
    validation_length = int(np.floor(dataLength * split))

    validationSet = data[-validation_length:]
    trainingsSet = data[:-validation_length]

    return trainingsSet,validationSet

def dimToFolder(dim):
    savefolder = ""
    for i in dim:
        savefolder += str(i)+"x"
    savefolder = savefolder[:-1]
    return savefolder

def prepareListOfFiles(path,workingdir = WRKDIR,nameOfCsvFile=CSVFILE):
    if not os.path.exists(workingdir):
        os.mkdir(workingdir)

    if not os.path.exists(os.path.join(workingdir,nameOfCsvFile)):
        listOfFiles = getListOfFiles(path)
        listOfFiles.sort()
        dataframe = pd.DataFrame(listOfFiles,columns=["colummn"])
        dataframe.to_csv(os.path.join(workingdir,nameOfCsvFile),index=False)
        listOfFiles = dataframe
    
    listOfFiles = list(pd.read_csv(os.path.join(workingdir,nameOfCsvFile))["colummn"])

    return listOfFiles

def getListOfFiles(path):
    """
    
        stolen from :
        https://thispointer.com/python-how-to-get-list-of-files-in-directory-and-sub-directories/

    """


    directory_entries = os.listdir(path)
    files = []

    for entry in directory_entries:
        fullPath = os.path.join(path,entry)
        if os.path.isdir(fullPath):
            files = files + getListOfFiles(fullPath)
        else:
            files.append(fullPath)
    return files


class Dataset(keras.utils.Sequence):

    def __init__(self,path,
                      batch_size,
                      dim,
                      n_channels=4,
                      shuffle=True,
                      saveListOfFiles=CSVFILE,
                      workingdir=WRKDIR,
                      timeToPred = 30,
                      timeSteps = 5,
                      sequenceExist = False,
                      dtype=np.float32):


        """
        
            timeToPred    : minutes forecast, default is 30 minutes
            timeSteps     : time between images, default is 5 minutes

        """
        assert batch_size > 0, "batch_size needs to be greater than 0"
        assert timeSteps % 5 == 0, "timesteps % 5 needs to be 0"
        assert timeToPred % 5 == 0, "timeToPred % 5 needs to be 0"


        self.path = path
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.workingdir = workingdir
        self.saveListOfFiles = saveListOfFiles
        self.timeToPred = timeToPred
        self.timeSteps = timeSteps
        self.steps = int(timeToPred / timeSteps)
        self.datatype = dtype


        # index offset
        self.label_offset = self.n_channels + self.steps - 1

        if not os.path.exists(self.workingdir):
            os.mkdir(self.workingdir)

        if not os.path.exists(os.path.join(self.workingdir,saveListOfFiles)):
            self.listOfFiles = getListOfFiles(self.path)
            self.listOfFiles.sort()
            dataframe = pd.DataFrame(self.listOfFiles,columns=["colummn"])
            dataframe.to_csv(os.path.join(self.workingdir,saveListOfFiles),index=False)
            self.listOfFiles = dataframe


        self.listOfFiles = list(pd.read_csv(os.path.join(self.workingdir,saveListOfFiles))["colummn"])

        savefolder = dimToFolder(self.dim)
        
     
        self.new_listOfFiles = resizeImages(self.listOfFiles,dim,os.path.join(workingdir,savefolder),saveListOfFiles)

        if len(self.new_listOfFiles) != len(self.listOfFiles):
            print("WARNING: Length of lists does not match! ")

        self.listOfFiles = self.new_listOfFiles
        #self.listOfFiles = self.new_listOfFiles[:300]
        self.indizes = np.arange(len(self.listOfFiles))




    def __data_generation(self,index):

        X = np.empty((*self.dim,self.n_channels))
        Y = np.empty((*self.dim,1))

        for i,id in enumerate(range(index,index+self.n_channels)):
            
            img = np.array(Image.open(self.listOfFiles[id]),dtype=self.datatype)
            
            assert img.shape == self.dim, \
            "[Error] (Data generation) Image shape {} does not match dimension {}".format(img.shape,self.dim)            
            
            X[:,:,i] = img

        Y = np.array(Image.open(self.listOfFiles[index+self.label_offset]),dtype=self.datatype)
        
        return X,Y
        

    def on_epoch_end(self):
        
        self.indizes = np.arange(len(self))
        if self.shuffle == True:
            np.random.shuffle(self.indizes)

    def __len__(self):
        return int(np.floor(len(self.listOfFiles)/self.batch_size )) - self.label_offset

    def __getitem__(self,index):

        """

            index of Y = index + channels +steps

        """
        X = None
        Y = None
        X = np.empty((self.batch_size,*self.dim,self.n_channels))
        Y = np.empty((self.batch_size,*self.dim,1))

        id_list = self.indizes[index*self.batch_size:(index+1)*self.batch_size]
 
        for i, idd in enumerate(id_list):
            X[i,],Y[i,:,:,0] = self.__data_generation(idd)

        
        if np.isnan(X).any():
            print(X)
            print("NAN")
            exit(-1)

        if np.isnan(Y).any():
            print(Y)
            print("NAN")
            exit(-1)

        if np.isinf(X).any():
            print("INF")
            exit(-1)

        if np.isinf(Y).any():
            print("INF")
            exit(-1)

        X,Y = X/255,Y/255
        #print("\t{:5.2f}\t{:5.2f}\t{:5.2f}\t{:5.2f}".format(X.max(),X.min(),Y.max(),Y.min()))
        
        return X,Y

