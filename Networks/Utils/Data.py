 
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

class DataProvider(object):
    """
        
        path: string
            Path to directory were the files are stored.
    
    """
    
    def __init__(self,path,openIMG=True,pattern=".*"):
        def get_files(pwd,pattern=""):
    
            """
                Read all files from path which match a pattern and return a list with the names
            """

            import re
            files = []
            for file in os.listdir(pwd):
                matches = re.search(pattern, file)
                if matches:
                    files.append(matches[0])
            
            files.sort()
            return files

        self.files = get_files(path,pattern)
        self.path = path
        self.openIMG = openIMG
        self.transform = []


    def binary(self):
        self.transform.append(self.binary_)


    def max_contrast(self):
        self.transform.append(self.max_contrast_)

    def scale(self,factor):
        self.factor=factor
        self.transform.append(self.scale_)

      
    def binary_(self,img,threshold=5):
        
        
        img[img > threshold ] = 255
        img[img <= threshold] = 0
        
        return img

    def scale_(self,img):
        w,h = img.shape
        img = Image.fromarray(img)
        newsize = (int(w*self.factor),int(h*self.factor) )
        img = img.resize(newsize) 
        return np.array(img)


    def transform(function):
        self.transform.append(function)

    def max_contrast_(self,img):
        
            """
                maximize contrast of images
                also deleting "edges"
            """
            print("max_contrast_")

            img[img == img[0,0]] = 0

            mi,ma = img.min(),img.max()
            if ma == 0:
                return img

            img -= mi
            img[img == ma -mi] = 0
            ma = img.max()
            if ma == 0:
                ma = 1
            img = np.array(((img / ma) * 255),dtype='uint8')
            return img
    
    def _openIMG(self,pwd):
        img = np.array(Image.open(pwd))
        if len(self.transform) > 0:
            for f in self.transform:
                img = f(img)

        return img
        
    def __getitem__(self,i):
        if self.openIMG:
            return self._openIMG(os.path.join(self.path,self.files[i]))
        else:
            return os.path.join(self.path,self.files[i])
                
    def __len__(self):
        return len(self.files)
    
    def __iter__(self):
        self.n = -1
        return self
        
    def __next__(self):
        self.n += 1
        if self.n < len(self.files):
            if self.openIMG:
                return self._openIMG(os.path.join(self.path,self.files[self.n]))
            else:
                return os.path.join(self.path,self.files[self.n])
        else:
            raise StopIteration