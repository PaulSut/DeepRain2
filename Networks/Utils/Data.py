 
from PIL import Image
import numpy as np
import pandas as pd
import re
import keras
import os

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
                      saveListOfFiles="./.listOfFiles.csv",
                      workingdir="./Data",
                      timeToPred = 35,
                      timeSteps = 5,
                      sequenceExist = False):


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
        self.indizes = np.arange(len(self.listOfFiles))




    def __data_generation(self,index):

        X = np.empty((*self.dim,self.n_channels))
        Y = np.empty((*self.dim,1))

        for i,id in enumerate(range(index,index+self.n_channels)):
            
            img = np.array(Image.open(self.listOfFiles[id]))
            assert img.shape == self.dim, \
            "[Error] (Data generation) Image shape {} does not match dimension {}".format(img.shape,self.dim)            
            
            X[:,:,i] = img

        Y = np.array(Image.open(self.listOfFiles[index+self.label_offset]))
        
        return X,Y
        

    def on_epoch_end(self):
        
        self.indizes = np.arange(len(self.listOfFiles))
        if self.shuffle == True:
            np.random.shuffle(self.indizes)

    def __len__(self):
        
        return int(np.floor(len(self.listOfFiles)/self.batch_size ))

    def __getitem__(self,index):

        """

            index of Y = index + channels +steps

        """

        X = np.empty((self.batch_size,*self.dim,self.n_channels))
        Y = np.empty((self.batch_size,*self.dim))

        id_list = self.indizes[index*self.batch_size:(index+1)*self.batch_size]
 
        for i, idd in enumerate(id_list):
            X[i,],Y[i,] = self.__data_generation(idd)

        return X,Y

