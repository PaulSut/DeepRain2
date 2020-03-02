 
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
                      n_channels=7,
                      shuffle=True,
                      saveListOfFiles="./.listOfFiles.csv",
                      workingdir="./",
                      timeToPred = 20,
                      timeSteps = 5,
                      sequenceExist = True):


        """
        
            timeToPred    : minutes forecast, default is 30 minutes
            timeSteps     : time between images, default is 5 minutes
            sequenceExist : only use coherent data

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
        self.sequenceExist = sequenceExist
        self.sequenceList = []

        # index offset
        self.label_offset = self.n_channels + self.steps - 1

        if not os.path.exists(saveListOfFiles):
            self.listOfFiles = getListOfFiles(self.path)
            self.listOfFiles.sort()
            dataframe = pd.DataFrame(self.listOfFiles,columns=["colummn"])
            dataframe.to_csv(saveListOfFiles,index=False)
            self.listOfFiles = dataframe

        else:
            self.listOfFiles = list(pd.read_csv(saveListOfFiles)["colummn"])

        #self.indizes = np.arange(len(self.listOfFiles)-self.n_channels-self.steps)


        if self.sequenceExist:
            self.sortOutSequence()



    def sortOutSequence(self):

        """
            
            This function ensures that a sequence is coherent.
            Some measurements doesn't exists, so we need to sort them out.


            ..YW2017.002_200801/0801010000.png +
            ..YW2017.002_200801/0801010005.png +
            ..YW2017.002_200801/0801010010.png +
            ..YW2017.002_200801/0801010015.png +
            ..YW2017.002_200801/0801010020.png -> Channels
            ..YW2017.002_200801/0801010025.png +
            ..YW2017.002_200801/0801010030.png +
            ..YW2017.002_200801/0801010035.png +
            ..YW2017.002_200801/0801010040.png +
            ..YW2017.002_200801/0801010045.png -> steps
            ..YW2017.002_200801/0801010050.png -> label

            here: 5 channels, 30 min forecasting



            ..YW2017.002_200801/0801010050.png
            ..YW2017.002_200801/0801010055.png !!
            ..YW2017.002_200801/0801010100.png !! BIG GAP
            ..YW2017.002_200801/0801010105.png


        """

        regex = r".+\/(YW[\d\._]+)\/([\d]+).png"

        def regexPath(path):


            matches = re.search(regex, path)
            if matches:
                x,y = matches.groups()
                return x,y
            return None


        def getSequence(index,path):


            result = regexPath(path)
            if result is None:
                return False

            yearmonth,timestamp = result[0],int(result[1])
            ctr = 0
            sequence = []

            for i in range(index - self.label_offset,index - self.steps + 1):
                result = regexPath(self.listOfFiles[i])
                if result is None:
                    #print("No Sequence found for: ",path)
                    return None

                yearmonth_ch,timestamp_ch = result[0],int(result[1])
                
                if timestamp-(self.label_offset*self.timeSteps) + ctr != timestamp_ch:
                    #print("No Sequence found for: ",path)
                    return None

                sequence.append(i)
                ctr+=self.timeSteps
            #print("+ Sequence found for: ",path)
            return sequence


        exist = 0
        sum_e = 0

        for i in range(self.label_offset,len(self.listOfFiles)):
            
            path = self.listOfFiles[i]
            sequence = getSequence(i,path)
            if sequence:
                self.sequenceList.append((sequence,i))
                exist += 1
            sum_e += 1

        print("#Sequences: ",exist," | #total",sum_e," => ",exist/sum_e,"%")


    def __len__(self):
        
        self.len = int(np.floor(len(self.listOfFiles)/self.batch_size )) - self.n_channels - self.steps

    def __getitem__(self,index):

        """

            index of Y = index + channels +steps

        """

        Y = self.listOfFiles[index + self.label_offset]
        X = [ self.listOfFiles[i] for i in range(index,index+self.n_channels) ]

        return X,Y

