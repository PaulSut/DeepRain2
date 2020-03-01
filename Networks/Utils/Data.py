 
from PIL import Image
import numpy as np
import pandas as pd
import os
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

#https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

class Dataset(keras.utils.Sequence):

    def __init__(self,path,
                      batch_size,
                      dim,
                      n_channels=7,
                      shuffle=True,
                      saveListOfFiles="./.listOfFiles.csv",
                      workingdir="./",
                      timeToPred = 30,
                      timeSteps = 5):


        """
        
            timeToPred : minutes forecast, default is 30 minutes
            timeSteps  : time between images, default is 5 minutes

        """

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


        if not os.path.exists(saveListOfFiles):
            self.listOfFiles = getListOfFiles(self.path)
            self.listOfFiles.sort()
            dataframe = pd.DataFrame(self.listOfFiles,columns=["colummn"])
            dataframe.to_csv(saveListOfFiles,index=False)
            self.listOfFiles = dataframe

        else:
            self.listOfFiles = list(pd.read_csv(saveListOfFiles)["colummn"])

        self.indizes = np.arange(len(self.listOfFiles)-self.n_channels-self.steps)



    def __len__(self):
        
        return int(np.floor(len(self.listOfFiles)/self.batch_size )) - self.n_channels - self.steps

    def __getitem__(self,index):

        """

            index of Y = index + channels +steps

        """


        Y = self.listOfFiles[index + self.n_channels + self.steps]
        X = [ self.listOfFiles[i] for i in range(index+self.n_channels) ]

        return X,Y

