from PIL import Image
import numpy as np
import pandas as pd
from Utils.loadset import getDataSet
from tensorflow.python.keras.utils.data_utils import Sequence
import wget
import os
import tarfile
import re
import cv2
from .transform import ImageToPatches, transformImages, cutOut, Wiggle
from .URL import YEARS


KONSTANCE  = (800,430)
DATAFOLDER = "./Data"
RAWFOLDER  = "RAW"
CSVFILE    = ".listOfFiles.csv"

def yearExists(year):

    regex = r"YW2017\.002_("+year+").*"
    monthpng = os.path.join(RAWFOLDER,"MonthPNGData")
    pwd = os.path.join(DATAFOLDER,monthpng)
    if not os.path.exists(pwd):
        return False
    data = os.listdir(pwd)
    for d in data:
        matches = re.search(regex, d, re.DOTALL)
        if matches: 
            return True
    return False

def llist_to_list(group):
    n_list = []
    for p in group:
        n_list +=  p
    return n_list

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


def cluster(data, maxgap):
    """
    Stolen from :

    https://stackoverflow.com/questions/14783947/grouping-clustering-numbers-in-python

    """
    '''Arrange data into groups where successive elements
       differ by no more than *maxgap*

        >>> cluster([1, 6, 9, 100, 102, 105, 109, 134, 139], maxgap=10)
        [[1, 6, 9], [100, 102, 105, 109], [134, 139]]

        >>> cluster([1, 6, 9, 99, 100, 102, 105, 134, 139, 141], maxgap=10)
        [[1, 6, 9], [99, 100, 102, 105], [134, 139, 141]]

    '''
    data.sort()
    groups = [[data[0]]]
    for x in data[1:]:
        if abs(x - groups[-1][-1]) <= maxgap:
            groups[-1].append(x)
        else:
            groups.append([x])
    return groups

def getData(batch_size,
            dimension,
            channels,
            fillSteps = False,
            timeToPred = 30,
            years = [2008,2009,2010,2011,2012,2013,2014,2015,2016,2017],
            split = 0.25,
            y_transform = [],
            x_transform = [],
            area_ = None,
            sortOut = True,
            keep_sequences = False):

    # Download and extract files
    for key in years:
        if type(key) is int:
            key = str(key)
        url           = YEARS[key]
        filename      = 'YW{}.tar.gz'.format(key)
        download_path = os.path.join(DATAFOLDER,filename)
        
        extract_folder = os.path.join(DATAFOLDER,"RAW")

        if not os.path.exists(extract_folder):
            os.makedirs(extract_folder)

        if yearExists(key):
            continue

        wget.download(url,download_path)


        tfile = tarfile.open(download_path, 'r')
        tfile.extractall(path=extract_folder)

        os.remove(download_path)


    # Save path to files as csv
    path_to_csvfile = os.path.join(DATAFOLDER,".listOfFiles.csv")


    # if csv file exists, dont create it again

    if not os.path.exists(path_to_csvfile):

        datafolder = os.path.join(os.path.join(DATAFOLDER,RAWFOLDER),"MonthPNGData")
        listOfFiles = getListOfFiles(datafolder)

        data_info = []
        listOfFiles.sort()
        
        print("Creating CSV File and some additional information... \nthis may take a while")
        for i,path in enumerate(listOfFiles):
            print("Creating CSV file: {:07d}/{}".format(i,len(listOfFiles)),end="\r")
            img = Image.open(path)
            img = np.array(img)
            data_info.append([path,img.mean(),img.std(),img.max()])
        columns = ["path","mean","std","max"]

        dframe  = pd.DataFrame(np.array(data_info),columns=columns)
        dframe.to_csv (path_to_csvfile, index = False, header=True)
    

    csvfile = pd.read_csv(path_to_csvfile)
    subset = csvfile[["path","mean","std","max"]]
    data = [tuple(x) for x in subset.to_numpy()]

    
    l = len(data)

    train = data[:int(l*(1-split))] 
    test  = data[int(l*(1-split)):]

    trainFolder = os.path.join(DATAFOLDER,"train")
    testFolder  = os.path.join(DATAFOLDER,"test")

    if not os.path.exists(trainFolder):
        os.mkdir(trainFolder)
        os.mkdir(testFolder)

    
    x,y = dimension
    area = [KONSTANCE[0] - x//2,KONSTANCE[0] + (x - x//2),\
            KONSTANCE[1] - y//2,KONSTANCE[1] + (y - y//2)]

    transformation = cutOut(area)

    trainFolder = os.path.join(trainFolder,str(transformation))
    testFolder  = os.path.join(testFolder ,str(transformation))



    traincsv    = os.path.join(trainFolder,CSVFILE)
    testcsv     = os.path.join(testFolder ,CSVFILE)

    if not os.path.exists(trainFolder):
        os.mkdir(trainFolder)
        os.mkdir(testFolder)

        trainlist = transformImages(train,[transformation],trainFolder,area=area_)
        testlist  = transformImages(test ,[transformation],testFolder,area=area_)

        trainlist.to_csv(traincsv, index = False, header=True)
        testlist.to_csv(testcsv, index = False, header=True)


    train = Dataset(batch_size,
                    dimension,
                    channels,
                    traincsv,
                    timeToPred = timeToPred,
                    fillSteps  = fillSteps,
                    sortOut    = sortOut,
                    y_transform = y_transform,
                    x_transform = x_transform,
                    keep_sequences = keep_sequences,
                    area=area)

    test = Dataset(batch_size,
                    dimension,
                    channels,
                    testcsv,
                    timeToPred = timeToPred,
                    fillSteps  = fillSteps,
                    sortOut    = sortOut,
                    y_transform = y_transform,
                    x_transform = x_transform,
                    keep_sequences = keep_sequences,
                    area=area_)

    return train,test



class Dataset(Sequence):

    def __init__(self,
                 batch_size,
                 dimension,
                 channels,
                 csvfile,
                 timeToPred = 30,
                 fillSteps = False,
                 years = [2008],
                 sortOut = False,
                 y_transform = [],
                 x_transform = [],
                 keep_sequences = False,
                 area = None):

        self.batch_size   = batch_size
        self.channels     = channels
        self.dimension    = dimension
        self.years        = years
        self.csvfile      = csvfile
        self.timeSteps    = 5
        self.timeToPred   = timeToPred
        self.fillSteps    = fillSteps
        self.label_Offset = timeToPred // self.timeSteps
        self.sortOut      = sortOut
        self.x_transform  = x_transform
        self.y_transform  = y_transform
        csvfile = pd.read_csv(self.csvfile)
        subset = csvfile[["path","mean","std","max"]]
        self.data = [tuple(x) for x in subset.to_numpy()]
        self.Wiggle = Wiggle()
        self.Wiggle_off = False
        
        self.keep_sequence = keep_sequences
        



        if self.sortOut:
            sortedOut = []
            for i in range(0,len(self.data) - self.label_Offset - self.channels):
                if self.data[i][-1] > 0:
                    sortedOut.append(i)
            self.indices = sortedOut

        else:
            self.indices = [i for i in range(0,len(self.data) \
                - self.label_Offset \
                - self.channels)]

        if self.keep_sequence:
            self.indices_copy = self.indices.copy()
            groups_list = cluster(self.indices_copy,1)
            """ remove sequences < channelsize """
            self.grouped = [ x for x in groups_list if len(x) >= self.channels]
            self.indices = llist_to_list(self.grouped)
            

    def X_Processing(self,index):
        
        img = cv2.imread(self.data[index][0],0)
        if not self.Wiggle_off:
            img = self.Wiggle(img)

        for t in self.x_transform:
            img = t(img)

        return img


    def Y_Processing(self,index):
        
        img = cv2.imread(self.data[index][0],0)
        if not self.Wiggle_off:
            img = self.Wiggle(img)
        for t in self.y_transform:
            img = t(img)
        #return img.flatten()
        return img


    def __data_generation(self,index):

        start = index
        end   = start + self.channels
        X = [self.X_Processing(i) for i in range(start,end)]

        if not self.fillSteps:
            i = end + self.label_Offset
            Y = [self.Y_Processing(i)]
        else:
            Y = [self.Y_Processing(i) for i in range(end,end+self.label_Offset)]

        
        self.Wiggle.draw()
        
        return np.array(X),np.array(Y)




    def getMean(self):
        sum_mean = 0
        for idx in self.indices:
            #_,mean,_,_ = self.data[idx]
            img = cv2.imread(self.data[idx][0],0)
            mean = np.mean(img / 255.0,axis=(0,1))
            sum_mean += mean
        return sum_mean / len(self.indices)

    def getStd(self):
        sum_std = 0
        for idx in self.indices:
            #_,_,std,_ = self.data[idx]
            img = cv2.imread(self.data[idx][0],0)
            std = np.std(img / 255.0,axis=(0,1))
            sum_std += std
        return sum_std / (len(self.indices) -1)


    def setWiggle_off(self):
        self.Wiggle_off = True

    def setWiggle_on(self):
        self.Wiggle_off = False

    def on_epoch_end(self):
        self.Wiggle.draw()
        if self.keep_sequence:
            grouped = self.grouped.copy()
            np.random.shuffle(grouped)
            self.indices = llist_to_list(grouped)
        else:
            np.random.shuffle(self.indices)


    def __len__(self):
        return int(np.floor((len(self.indices) - self.label_Offset - self.channels -1) \
                / self.batch_size))


    def __getitem__(self,index):


        if index >= len(self):
            return None

        X = []
        Y = []

        start = index * self.batch_size
        end   = start + self.batch_size

        id_list = self.indices[start:end]


        for id in id_list:
            x,y = self.__data_generation(id)

            X.append(x)
            Y.append(y)

        X = np.array(X)
        Y = np.array(Y)

        X = np.transpose(X,(0,2,3,1))
        Y = np.transpose(Y,(0,2,3,1))
        
        return X/255.0 ,Y/1.0

